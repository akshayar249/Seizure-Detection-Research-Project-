import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# Load your preprocessed CHB-MIT data (replace with your filename)
df = pd.read_parquet('eeg_dataset_cleaned.parquet')

# Sampling frequency for CHB-MIT
fs = 256  
window_sec = 5
window_size = fs * window_sec  # 1280 samples per window

# Assuming 'outcome' column exists as label
label_col = 'outcome'

# Get EEG channel columns (exclude label)
eeg_channels = [col for col in df.columns if col != label_col]

def extract_features_segment(segment):
    features = {}
    for ch in eeg_channels:
        signal = segment[ch].values
        
        # Time-domain features
        features[f'{ch}_mean'] = np.mean(signal)
        features[f'{ch}_std'] = np.std(signal)
        features[f'{ch}_skew'] = skew(signal)
        features[f'{ch}_kurtosis'] = kurtosis(signal)
        features[f'{ch}_max'] = np.max(signal)
        features[f'{ch}_min'] = np.min(signal)
        
        # Frequency bands power using Welch PSD
        freqs, psd = welch(signal, fs=fs, nperseg=512)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            features[f'{ch}_power_{band}'] = np.sum(psd[idx])
    return features

# Extract features in non-overlapping 5s windows
feature_rows = []
for start_idx in range(0, len(df) - window_size + 1, window_size):
    segment = df.iloc[start_idx:start_idx+window_size]
    feat = extract_features_segment(segment)
    
    # Majority label in the window
    feat['label'] = segment[label_col].mode()[0]
    feature_rows.append(feat)

features_df = pd.DataFrame(feature_rows)
print("Feature DataFrame shape:", features_df.shape)

# Save features for next steps (e.g., feature selection)
features_df.to_parquet('chbmit_extracted_features.parquet', index=False)
