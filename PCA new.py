import mne
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load EEG from .edf file
raw = mne.io.read_raw_edf(r'mne_env/chb01_08.edf', preload=True)

# Step 2: Rename channels (optional, for consistency)
raw.rename_channels(lambda name: name.upper().replace('-', '_'))

# Step 3: Pick EEG channels only
raw.pick_types(eeg=True)

# Step 4: Bandpass filter (1–40 Hz for ICA)
raw.filter(l_freq=1., h_freq=40., fir_design='firwin')

# Step 5: ICA for artifact removal
ica = ICA(n_components=len(raw.ch_names), random_state=97, max_iter='auto')
ica.fit(raw)

# Optional: Plot ICA sources to visually inspect components
ica.plot_sources(raw)

# Step 6: Exclude bad components (adjust these after visual inspection)
ica.exclude = [0, 1]  # Replace with actual indices of artifact components

# Step 7: Apply ICA to remove artifacts
raw_clean = ica.apply(raw.copy())

# Step 8: Get cleaned EEG data
data_clean = raw_clean.get_data()  # shape: (n_channels, n_times)

# Step 9: Apply PCA (dimensionality reduction)
pca = PCA(n_components=0.99, svd_solver='full')  # Retain 99% variance
transformed = pca.fit_transform(data_clean.T)  # shape: (n_times, components)
reconstructed = pca.inverse_transform(transformed).T  # Back to (n_channels, n_times)

# Step 10: Create new Raw object with PCA-reconstructed data
raw_pca = raw_clean.copy()
raw_pca._data = reconstructed

# Step 11: Plot comparison
print("EEG after ICA (artifact removal):")
raw_clean.plot(n_channels=10, title="EEG after ICA")

print("EEG after ICA + PCA:")
raw_pca.plot(n_channels=10, title="EEG after ICA + PCA")

# Show all plots
plt.show()
