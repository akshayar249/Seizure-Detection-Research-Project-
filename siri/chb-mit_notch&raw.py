import mne
edf_file = "C:/Users/your_file_path/Python/Python313/chb01_01.edf"  
raw = mne.io.read_raw_edf(edf_file, preload=True)
print("Loaded EDF file")

print("Showing original raw EEG data...")
raw.plot(n_channels=10, duration=20, scalings='auto', title='Original EEG (Before Notch Filter)')

# Most U.S. powerline interference is at 60 Hz
raw_notch = raw.copy().notch_filter(freqs=[60], verbose=True)
print("Applied 60 Hz notch filter")

print("Showing EEG after Notch Filter applied...")
raw_notch.plot(n_channels=10, duration=20, scalings='auto', title='EEG After 60 Hz Notch Filter')


