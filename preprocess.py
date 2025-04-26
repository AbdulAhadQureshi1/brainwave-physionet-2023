import os
import numpy as np
import wfdb
from scipy import signal
import h5py

def read_eeg_for_inference(record_path):
    """Loads EEG signal and returns it with sampling rate."""
    if not is_valid_recording(record_path + ".hea", min_duration=610):
        raise ValueError(f"Recording {record_path} does not meet the minimum duration requirement.")
    record = wfdb.rdrecord(record_path)
    return record.p_signal, record.fs

def preprocess_eeg_signal(eeg_signal, sampling_rate, target_fs=100):
    """Applies preprocessing steps to the EEG signal."""
    eeg_signal = limit_recording_duration(eeg_signal, sampling_rate, max_duration=40*60)
    eeg_signal = signal.resample(eeg_signal, int(eeg_signal.shape[0] * (target_fs / sampling_rate)), axis=0)
    eeg_signal = remove_nan_values(eeg_signal, method="interpolate")
    eeg_signal = apply_filtering(eeg_signal, target_fs)
    eeg_signal = normalize_eeg_voltages(eeg_signal)
    eeg_signal = standardize(eeg_signal, target_channels=19)
    return eeg_signal

def is_valid_recording(header_path, min_duration=180):
    """Check if the recording meets the minimum duration requirement using the header file."""
    try:
        with open(header_path, "r") as f:
            lines = f.readlines()
        first_line = lines[0].strip().split()
        num_samples = int(first_line[3])
        sampling_frequency = int(first_line[2])
        return (num_samples / sampling_frequency) >= min_duration
    except Exception as e:
        print(f"Error reading header file {header_path}: {e}")
        return False

def limit_recording_duration(eeg_signal, sampling_rate, max_duration=2400):
    max_samples = int(max_duration * sampling_rate)
    return eeg_signal[:max_samples, :] if eeg_signal.shape[0] > max_samples else eeg_signal

def remove_nan_values(eeg_signal, method="zero"):
    eeg_signal[np.isinf(eeg_signal)] = np.nan
    if method == "zero":
        return np.nan_to_num(eeg_signal, nan=0.0)
    elif method == "mean":
        channel_means = np.nanmean(eeg_signal, axis=0)
        channel_means = np.where(np.isnan(channel_means), 0, channel_means)
        return np.where(np.isnan(eeg_signal), channel_means, eeg_signal)
    elif method == "interpolate":
        cleaned_signal = np.copy(eeg_signal)
        for channel in range(eeg_signal.shape[1]):
            nan_indices = np.isnan(eeg_signal[:, channel])
            if np.all(nan_indices):
                cleaned_signal[:, channel] = 0.0
            else:
                x = np.arange(eeg_signal.shape[0])
                valid_x = x[~nan_indices]
                valid_y = eeg_signal[~nan_indices, channel]
                cleaned_signal[:, channel] = np.interp(x, valid_x, valid_y)
        return cleaned_signal
    else:
        raise ValueError(f"Unsupported method: {method}")

def apply_filtering(eeg_signal, fs, lowcut=0.5, highcut=40):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(5, [low, high], btype="band")
    return signal.lfilter(b, a, eeg_signal, axis=0)

def normalize_eeg_voltages(eeg_signal, norm_range=(-1, 1)):
    min_val, max_val = norm_range
    signal_min = eeg_signal.min(axis=0, keepdims=True)
    signal_max = eeg_signal.max(axis=0, keepdims=True)
    denom = signal_max - signal_min
    denom[denom == 0] = 1
    normalized_signal = (eeg_signal - signal_min) / denom
    return normalized_signal * (max_val - min_val) + min_val

def standardize(eeg_signal, target_channels=19):
    if eeg_signal.shape[1] > target_channels:
        return eeg_signal[:, :target_channels]
    elif eeg_signal.shape[1] < target_channels:
        padding = np.zeros((eeg_signal.shape[0], target_channels - eeg_signal.shape[1]))
        return np.hstack((eeg_signal, padding))
    return eeg_signal

def create_windows(eeg_signal, window_size, fs):
    """Segments EEG data into fixed-size windows."""
    num_time_samples = window_size * fs
    num_windows = eeg_signal.shape[0] // num_time_samples
    eeg_signal = eeg_signal[: num_windows * num_time_samples]
    eeg_windows = eeg_signal.reshape(num_windows, num_time_samples, 19)
    return eeg_windows[0]

# Example usage for inference
def preprocess_for_inference(record_path, fs=100, window_size=300):
    raw_signal, raw_fs = read_eeg_for_inference(record_path)
    processed_signal = preprocess_eeg_signal(raw_signal, raw_fs, fs)
    windows = create_windows(processed_signal, window_size, fs)

    # Save to disk as h5
    cache_path = "temp/temp.h5"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with h5py.File(cache_path, "w") as hf:
        hf.create_dataset("windows", data=windows, compression="gzip", compression_opts=4)
    
    # Read back from h5
    with h5py.File(cache_path, "r") as hf:
        compressed_wins = hf["windows"][:]
        
    os.remove(cache_path)  # Clean up temporary file

    return compressed_wins
