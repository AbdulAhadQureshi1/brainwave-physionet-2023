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
from antropy import perm_entropy, petrosian_fd
import pandas as pd
import torch
from model import CombinedModel, resnet_config, transformer_config

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    os.makedirs(model_folder, exist_ok=True)
    train_dl_model(data_folder, model_folder, verbose)
    train_ml_model(data_folder, model_folder, verbose)

def train_dl_model(data_folder, model_folder, verbose):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from model import CombinedModel, resnet_config, transformer_config
    from preprocess import preprocess_for_inference
    from helper_code import find_data_folders, load_challenge_data, get_outcome, find_recording_files

    model = CombinedModel(resnet_config, transformer_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 5
    batch_size = 16

    patient_ids = find_data_folders(data_folder)
    all_inputs, all_labels = [], []

    if verbose:
        print("Preparing data for DL training...")

    for patient_id in tqdm(patient_ids, desc="DL - Patients"):
        try:
            metadata = load_challenge_data(data_folder, patient_id)
            label = get_outcome(metadata)
            recording_ids = find_recording_files(data_folder, patient_id)

            for recording_id in recording_ids:
                record_path = os.path.join(data_folder, patient_id, recording_id)
                try:
                    _, eeg_window = preprocess_for_inference(record_path, fs=100, window_size=20)
                    eeg_tensor = torch.tensor(eeg_window.T, dtype=torch.float32).unsqueeze(0)
                    all_inputs.append(eeg_tensor)
                    all_labels.append(torch.tensor([label], dtype=torch.float32))
                except:
                    continue
        except:
            continue

    if not all_inputs:
        raise ValueError("No valid EEG data found for DL model training.")

    inputs_tensor = torch.cat(all_inputs, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    dataset = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"DL Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader):.4f}")

    model_path = os.path.join(model_folder, 'dl_model.pth')
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f"DL model saved to {model_path}")

def train_ml_model(data_folder, model_folder, verbose):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    import joblib
    from tqdm import tqdm

    from helper_code import (
        find_data_folders, load_challenge_data, get_outcome, get_cpc, find_recording_files
    )
    from preprocess import preprocess_for_inference

    if verbose:
        print("Extracting features for ML training...")

    features, outcomes, cpcs = [], [], []
    patient_ids = find_data_folders(data_folder)

    for patient_id in tqdm(patient_ids, desc="ML - Patients"):
        try:
            metadata = load_challenge_data(data_folder, patient_id)
            label = get_outcome(metadata)
            cpc = get_cpc(metadata)
            patient_feats, _ = get_patient_features(metadata)
            recording_ids = find_recording_files(data_folder, patient_id)

            for recording_id in recording_ids:
                record_path = os.path.join(data_folder, patient_id, recording_id)
                try:
                    eeg_data, _ = preprocess_for_inference(record_path, fs=100, window_size=20)
                    eeg_feats, _ = get_eeg_features(eeg_data)
                    full_feats = eeg_feats + patient_feats
                    features.append(full_feats)
                    outcomes.append(label)
                    cpcs.append(cpc)
                except:
                    continue
        except:
            continue

    if not features:
        raise ValueError("No ML features extracted.")

    features = pd.DataFrame(features).apply(pd.to_numeric, errors='coerce')
    outcomes = np.array(outcomes).reshape(-1, 1)
    cpcs = np.array(cpcs).reshape(-1, 1)

    imputer = SimpleImputer().fit(features)
    features_imputed = imputer.transform(features)

    scaler = StandardScaler().fit(features_imputed)
    features_scaled = scaler.transform(features_imputed)

    clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=200, random_state=42)
    clf.fit(features_scaled, outcomes.ravel())

    reg = RandomForestRegressor(n_estimators=100, max_leaf_nodes=200, random_state=42)
    reg.fit(features_scaled, cpcs.ravel())

    ml_model_path = os.path.join(model_folder, 'ml_models.sav')
    joblib.dump({
        'imputer': imputer,
        'scaler': scaler,
        'outcome_model': clf,
        'cpc_model': reg
    }, ml_model_path, protocol=0)

    if verbose:
        print(f"ML models saved to {ml_model_path}")

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    # Check if the model folder exists
    ml_model_path = os.path.join(model_folder, 'ml_models.sav')
    ml_model = joblib.load(ml_model_path)
    if verbose:
        print(f"Loaded ML model from {ml_model_path}")

    # Load DL model
    dl_model_path = os.path.join(model_folder, 'dl_model.pth')
    dl_model = CombinedModel(resnet_config, transformer_config)
    dl_model.load_state_dict(torch.load(dl_model_path))
    
    if verbose:
        print(f"Loaded DL model from {dl_model_path}")

    return {
        'imputer': ml_model['imputer'],
        'scaler': ml_model['scaler'],
        'outcome_model': ml_model['outcome_model'],
        'cpc_model': ml_model['cpc_model'],
        'dl_model': dl_model
    }

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    scaler = models['scaler']
    dl_model = models['dl_model']

    # Extract features
    features, dl_outcome_probs = get_features(data_folder, patient_id, dl_model)
    dl_outcome_probs = np.array([prob[0] for prob in dl_outcome_probs])
    
    # Impute and scale
    features = imputer.transform(features)
    features = scaler.transform(features)

    # ML predictions
    outcome = outcome_model.predict(features)
    outcome_probability = outcome_model.predict_proba(features)
    cpc = cpc_model.predict(features)

    # Adjust CPC
    cpc = np.clip(cpc + 1, 1, 5)
    
    # Select positive class probabilities
    ml_positive_probs = outcome_probability[:, 1]

    # Confidence
    confidence_ml = abs(ml_positive_probs - 0.5)
    confidence_dl = abs(dl_outcome_probs - 0.5)

    total_confidence = confidence_ml + confidence_dl
    weight_ml = confidence_ml / total_confidence
    weight_dl = confidence_dl / total_confidence

    # Final combined probability
    final_prob = weight_ml * ml_positive_probs + weight_dl * dl_outcome_probs

    # Final binary decision
    final_outcome = (final_prob >= 0.5).astype(int)

    # Final outcome and cpc
    final_outcome, cpc = choose_final_outcome_and_cpc(final_outcome, final_prob, cpc)

    if verbose:
        print(f"DL Model Outcome: {dl_outcome_probs}")
        print(f'ML Outcome Probability: {ml_positive_probs}')
        print(f'ML Confidence: {confidence_ml}')
        print(f'DL Confidence: {confidence_dl}')
        print(f'Final Combined Probability: {final_prob}')
        print(f'Final Outcome: {final_outcome}')
        print(f'CPC: {cpc}')
        print(f'Final Patient Outcome: {final_outcome}')
        print(f'Final Patient CPC: {cpc}')

    return final_outcome, final_prob, cpc
    
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

def get_dl_outcome_prob(eeg_data_window, dl_model):
    dl_model.eval()
    with torch.no_grad():
        dl_data = torch.tensor(eeg_data_window, dtype=torch.float32)
        dl_data = dl_data.unsqueeze(0)  # Add batch dimension
        dl_output = dl_model(dl_data)
        dl_outcome_prob = torch.sigmoid(dl_output).numpy()
        return dl_outcome_prob

def load_patient_data(data_folder, patient_id):
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    return patient_metadata, recording_ids

def extract_patient_features(patient_metadata):
    patient_features, patient_features_names = get_patient_features(patient_metadata)
    return patient_features, patient_features_names

def process_single_recording(record_path, sampling_frequency, patient_features, dl_model):
    eeg_data, eeg_data_window = preprocess_for_inference(record_path, sampling_frequency, window_size=180)
    dl_outcome_prob = get_dl_outcome_prob(eeg_data_window.T, dl_model)
    eeg_features, eeg_feature_names = get_eeg_features(eeg_data)
    combined_features = eeg_features + patient_features
    return combined_features, eeg_feature_names, dl_outcome_prob

def get_features(data_folder, patient_id, dl_model):
    # Load data
    patient_metadata, recording_ids = load_patient_data(data_folder, patient_id)
    patient_features, patient_features_names = extract_patient_features(patient_metadata)

    dl_outcome_probs = []
    total_features = []
    full_feature_names = None

    sampling_frequency = 100  # Hz

    for recording_id in recording_ids:
        print(f'Extracting features from {recording_id}...')
        record_path = os.path.join(data_folder, patient_id, recording_id)

        try:
            combined_features, eeg_feature_names, dl_outcome_prob = process_single_recording(
                record_path, sampling_frequency, patient_features, dl_model
            )
        except Exception as e:
            print(f"Error processing {record_path}: {e}")
            continue

        dl_outcome_probs.append(dl_outcome_prob)
        
        if combined_features is not None:
            total_features.append(combined_features)

            if full_feature_names is None:
                full_feature_names = eeg_feature_names + patient_features_names

    full_features_df = pd.DataFrame(total_features, columns=full_feature_names)
    full_features_df = full_features_df.apply(pd.to_numeric, errors='coerce')

    return full_features_df, dl_outcome_probs

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
def get_eeg_features(eeg_windows):
    """Feature extraction with real PFD and PE"""

    eeg_feature_names = ["mean", "std", "var", "rms", "kurtosis", "power", "psd", "pfd", "pe"]

    # Ensure batch dimension
    if len(eeg_windows.shape) == 2:
        eeg_windows = eeg_windows[np.newaxis, ...]
    
    means = np.mean(eeg_windows, axis=(1, 2))
    stds = np.std(eeg_windows, axis=(1, 2))
    vars_ = np.var(eeg_windows, axis=(1, 2))
    rms = np.sqrt(np.mean(eeg_windows ** 2, axis=(1, 2)))
    kurt = kurtosis(eeg_windows.reshape(eeg_windows.shape[0], -1), axis=1)
    power = np.mean(eeg_windows ** 2, axis=(1, 2))
    psd_approx = vars_

    pfd_vals = np.array([petrosian_fd(window.reshape(-1)) for window in eeg_windows])
    pe_vals = np.array([perm_entropy(window.reshape(-1), normalize=True) for window in eeg_windows])

    # No need for a loop if just 1 window
    feature_list = [
        float(means[0]),
        float(stds[0]),
        float(vars_[0]),
        float(rms[0]),
        float(kurt[0]),
        float(power[0]),
        float(psd_approx[0]),
        float(pfd_vals[0]),
        float(pe_vals[0])
    ]
    
    return feature_list, eeg_feature_names


def choose_final_outcome_and_cpc(final_outcome, final_prob, cpc_preds):
    # Majority voting for outcome
    # counts = np.bincount(final_outcome)
    # final_patient_outcome = np.argmax(counts)
    final_patient_outcome = int(np.round(np.mean(final_outcome)))

    # Split CPCs
    if final_patient_outcome == 1:
        valid_cpcs = [1, 2]
    else:
        valid_cpcs = [3, 4, 5]

    cpc_preds = np.round(cpc_preds).astype(int)  # Ensure CPCs are integers
    matching_cpcs = [cpc for cpc in cpc_preds if cpc in valid_cpcs]

    if matching_cpcs:
        # Pick most common among matching CPCs
        final_patient_cpc = int(np.round(np.median(matching_cpcs)))
    else:
        # No matching CPCs, fallback to most common CPC overall
        final_patient_cpc = int(np.round(np.median(cpc_preds)))

    return final_patient_outcome, final_patient_cpc
