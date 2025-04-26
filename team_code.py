#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from preprocess import preprocess_for_inference
from scipy.stats import kurtosis
from scipy import signal
from antropy import perm_entropy, petrosian_fd
import pandas as pd

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        current_features = get_features(data_folder, patient_ids[i])
        features.append(current_features)

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)

    # Train the models.
    features = imputer.transform(features)
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Extract features.
    features = get_features(data_folder, patient_id)
        
    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)
    outcome_probability = outcome_model.predict_proba(features)
    cpc = cpc_model.predict(features)

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)
    
    print(f'outcome= {outcome}')
    print(f'outcome_probability= {outcome_probability}')
    print(f'cpc= {cpc}')

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient metadata.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)

    # Extract patient features (single vector).
    patient_features, patient_features_names = get_patient_features(patient_metadata)

    total_features = []
    full_feature_names = None  # We'll define it once inside the loop

    # Loop through each recording
    for recording_id in recording_ids:
        print(f'Extracting features from {recording_id}...')
        
        # Preprocess EEG data
        sampling_frequency = 100  # Hz
        record_path = os.path.join(data_folder, patient_id, recording_id)
        eeg_data = preprocess_for_inference(record_path, sampling_frequency, window_size=180)
        
        # Extract EEG features
        eeg_features, eeg_feature_names = get_eeg_features(eeg_data)

        # Combine EEG + Patient features
        combined_features = eeg_features + patient_features
        total_features.append(combined_features)

        # Set column names if not already set
        if full_feature_names is None:
            full_feature_names = eeg_feature_names + patient_features_names

    # Create DataFrame
    full_features_df = pd.DataFrame(total_features, columns=full_feature_names)

    # Make sure all values are numeric (convert non-numeric to NaN)
    full_features_df = full_features_df.apply(pd.to_numeric, errors='coerce')

    print(full_features_df.head())
    return full_features_df

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = [age, male, female, other, rosc, ohca, shockable_rhythm, ttm]
    patient_keys = ["Age", "Male", "Female", "Other", "ROSC", "OHCA", "Shockable Rhythm", "TTM"]
    
    return features, patient_keys

# Extract features from the EEG data.
def get_eeg_features(data):
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    vars_ = np.var(data, axis=1)
    rms = np.sqrt(np.mean(data ** 2, axis=1))
    kurt = kurtosis(data, axis=1)
    power = np.mean(data ** 2, axis=1)
    psd_approx = vars_

    # Real PFD and PE
    pfd_vals = np.array([petrosian_fd(window) for window in data])
    pe_vals = np.array([perm_entropy(window, normalize=True) for window in data])

    eeg_keys = ["mean", "std", "var", "rms", "kurtosis", "power", "psd", "pfd", "pe"]

    feature_list = []
    for i in range(data.shape[0]):
        feature_list.append([
            float(means[i]),
            float(stds[i]),
            float(vars_[i]),
            float(rms[i]),
            float(kurt[i]),
            float(power[i]),
            float(psd_approx[i]),
            float(pfd_vals[i]),
            float(pe_vals[i])
        ])
        
    print(f"feature_list shape= {len(feature_list)}")
    return feature_list[0], eeg_keys

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features
