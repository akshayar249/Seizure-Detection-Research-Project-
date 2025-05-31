import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pywt
import antropy as ant

# Load your cleaned EEG dataframe
df = pd.read_parquet('eeg_dataset_cleaned.parquet')

fs = 256  # Sampling frequency
window_sec = 5
window_size = fs * window_sec  # samples per window
stride = window_size // 2      # 50% overlap windows

label_col = 'outcome'
eeg_channels = [col for col in df.columns if col != label_col]

bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

def bandpower_ratios(powers):
    ratios = {}
    try:
        ratios['theta_alpha'] = powers['theta'] / (powers['alpha'] + 1e-6)
        ratios['beta_delta'] = powers['beta'] / (powers['delta'] + 1e-6)
        ratios['alpha_beta'] = powers['alpha'] / (powers['beta'] + 1e-6)
        ratios['gamma_beta'] = powers['gamma'] / (powers['beta'] + 1e-6)
    except:
        ratios = {'theta_alpha': 0, 'beta_delta': 0, 'alpha_beta': 0, 'gamma_beta': 0}
    return ratios

def extract_features_segment(segment):
    features = {}
    for ch in eeg_channels:
        signal = segment[ch].values
        powers = {}

        # Time-domain stats
        features[f'{ch}_mean'] = np.mean(signal)
        features[f'{ch}_std'] = np.std(signal)
        features[f'{ch}_skew'] = skew(signal)
        features[f'{ch}_kurtosis'] = kurtosis(signal)
        features[f'{ch}_max'] = np.max(signal)
        features[f'{ch}_min'] = np.min(signal)

        # Frequency domain - PSD using Welch
        freqs, psd = welch(signal, fs=fs, nperseg=512)
        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            powers[band] = np.sum(psd[idx])
            features[f'{ch}_power_{band}'] = powers[band]

        # Band power ratios
        ratios = bandpower_ratios(powers)
        for name, val in ratios.items():
            features[f'{ch}_ratio_{name}'] = val

        # Spectral entropy
        features[f'{ch}_spectral_entropy'] = ant.spectral_entropy(signal, sf=fs, method='welch')

        # Hjorth parameters
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        var_zero = np.var(signal)
        var_d1 = np.var(diff1)
        var_d2 = np.var(diff2)
        features[f'{ch}_hjorth_activity'] = var_zero
        features[f'{ch}_hjorth_mobility'] = np.sqrt(var_d1 / var_zero)
        features[f'{ch}_hjorth_complexity'] = np.sqrt(var_d2 / var_d1) / (np.sqrt(var_d1 / var_zero) + 1e-6)

        # DWT-based energy and entropy
        coeffs = pywt.wavedec(signal, 'db4', level=4)
        energy = [np.sum(np.square(c)) for c in coeffs]
        total_energy = np.sum(energy)
        features[f'{ch}_dwt_entropy'] = -np.sum([(e / total_energy) * np.log(e / total_energy + 1e-10) for e in energy])
        for i, e in enumerate(energy):
            features[f'{ch}_dwt_energy_l{i}'] = e

        # Nonlinear features: Sample Entropy, Higuchi FD
        features[f'{ch}_sampen'] = ant.sample_entropy(signal)
        features[f'{ch}_higuchi_fd'] = ant.higuchi_fd(signal)

    return features

feature_rows = []
for start_idx in range(0, len(df) - window_size + 1, stride):
    segment = df.iloc[start_idx:start_idx + window_size]
    feat = extract_features_segment(segment)
    feat['label'] = segment[label_col].mode()[0]  # majority label in segment
    feature_rows.append(feat)

features_df = pd.DataFrame(feature_rows)
features_df.to_parquet('chbmit_features_full.parquet', index=False)
print(f'Extracted features shape: {features_df.shape}')
print(features_df.head())
