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
