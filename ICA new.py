import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

# Step 1: Load the EEG data
raw = mne.io.read_raw_edf("mne_env/chb01_08.edf", preload=True)

# Step 2: Rename channels to a consistent format (e.g., FP1-F7 → FP1_F7)
raw.rename_channels(lambda name: name.upper().replace('-', '_'))

# Step 3: Pick EEG channels (assuming all channels are EEG)
raw.pick_types(eeg=True)

# Step 4: Filter the data (1–40 Hz recommended for ICA)
raw.filter(l_freq=1., h_freq=40., fir_design='firwin')

# Step 5: Run ICA
ica = ICA(n_components=len(raw.ch_names), random_state=97, max_iter='auto')
ica.fit(raw)

# Step 6: Plot ICA component time series (no topomap needed)
ica.plot_sources(raw)

# Step 7: Mark components you want to exclude (update based on visual inspection)
ica.exclude = [0, 1]  # Replace with actual components to exclude

# Step 8: Apply ICA to remove artifacts
raw_clean = ica.apply(raw.copy())

# Step 9: Compare before and after
print("Before ICA:")
raw.plot(n_channels=10, title="Original EEG")

print("After ICA:")
raw_clean.plot(n_channels=10, title="Cleaned EEG after ICA")

# Show all plots
plt.show()
