import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pywt
import antropy as ant
import time
import os
import sys

# ==== Parameters ====
fs = 256  # Sampling frequency
window_sec = 5
window_size = fs * window_sec
stride = window_size // 2  # 50% overlap
label_col = 'Outcome'

# ==== Load CSV ====
try:
    path = 'mne_env/chbmit_preprocessed_data.csv'
    if not os.path.exists(path):
        print(f"❌ CSV file not found at: {path}")
        sys.exit()

    df = pd.read_csv(path)
    print(f"✅ Data loaded. Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    sys.exit()

# ==== Validate Columns ====
if label_col not in df.columns:
    print(f"❌ Label column '{label_col}' not found in CSV.")
    print("Available columns:", df.columns)
    sys.exit()

eeg_channels = [col for col in df.columns if col != label_col]
print(f"Detected {len(eeg_channels)} EEG channels.")

if len(df) < window_size:
    print(f"❌ Not enough rows ({len(df)}) for one window of size {window_size}.")
    sys.exit()

# ==== EEG Frequency Bands ====
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

def bandpower_ratios(powers):
    try:
        return {
            'theta_alpha': powers['theta'] / (powers['alpha'] + 1e-6),
            'beta_delta': powers['beta'] / (powers['delta'] + 1e-6),
            'alpha_beta': powers['alpha'] / (powers['beta'] + 1e-6),
            'gamma_beta': powers['gamma'] / (powers['beta'] + 1e-6)
        }
    except:
        return {'theta_alpha': 0, 'beta_delta': 0, 'alpha_beta': 0, 'gamma_beta': 0}

def extract_features_segment(segment):
    features = {}
    for ch in eeg_channels:
        signal = segment[ch].values
        powers = {}

        features[f'{ch}_mean'] = np.mean(signal)
        features[f'{ch}_std'] = np.std(signal)

        if np.std(signal) < 1e-8:
            features[f'{ch}_skew'] = 0
            features[f'{ch}_kurtosis'] = 0
        else:
            features[f'{ch}_skew'] = skew(signal)
            features[f'{ch}_kurtosis'] = kurtosis(signal)

        features[f'{ch}_max'] = np.max(signal)
        features[f'{ch}_min'] = np.min(signal)

        freqs, psd = welch(signal, fs=fs, nperseg=512)
        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            powers[band] = np.sum(psd[idx])
            features[f'{ch}_power_{band}'] = powers[band]

        ratios = bandpower_ratios(powers)
        for name, val in ratios.items():
            features[f'{ch}_ratio_{name}'] = val

        features[f'{ch}_spectral_entropy'] = ant.spectral_entropy(signal, sf=fs, method='welch')

        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        var_zero = np.var(signal)
        var_d1 = np.var(diff1)
        var_d2 = np.var(diff2)
        features[f'{ch}_hjorth_activity'] = var_zero
        features[f'{ch}_hjorth_mobility'] = np.sqrt(var_d1 / (var_zero + 1e-6))
        features[f'{ch}_hjorth_complexity'] = np.sqrt(var_d2 / (var_d1 + 1e-6)) / (np.sqrt(var_d1 / (var_zero + 1e-6)) + 1e-6)

        coeffs = pywt.wavedec(signal, 'db4', level=4)
        energy = [np.sum(np.square(c)) for c in coeffs]
        total_energy = np.sum(energy)
        features[f'{ch}_dwt_entropy'] = -np.sum([(e / total_energy) * np.log(e / total_energy + 1e-10) for e in energy])
        for i, e in enumerate(energy):
            features[f'{ch}_dwt_energy_l{i}'] = e

        features[f'{ch}_sampen'] = ant.sample_entropy(signal)
        features[f'{ch}_higuchi_fd'] = ant.higuchi_fd(signal)

    return features

# ==== Sliding Window ====
print("\n🚀 Starting sliding window feature extraction...")
feature_rows = []
start_time = time.time()

for i, start_idx in enumerate(range(0, len(df) - window_size + 1, stride)):
    print(f"▶️ Window {i+1} | Row: {start_idx} - {start_idx + window_size}", end='... ')
    try:
        segment = df.iloc[start_idx:start_idx + window_size]
        t0 = time.time()
        feat = extract_features_segment(segment)
        feat['label'] = segment[label_col].mode()[0]
        feature_rows.append(feat)
        print(f"Done in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"❌ Error in window {i+1}: {e}")
        continue

print(f"\n✅ All windows processed. Total time: {time.time() - start_time:.2f} seconds")
print(f"✅ Total segments extracted: {len(feature_rows)}")

# ==== Save ====
features_df = pd.DataFrame(feature_rows)
features_df.to_parquet('chbmit_features_full.parquet', index=False)
features_df.to_csv('chbmit_features_full.csv', index=False)

print(f"\n💾 Features saved: {features_df.shape}")
print(features_df.head())
