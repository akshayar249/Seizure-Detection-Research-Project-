import mne
import matplotlib.pyplot as plt


edf_file =  "C:/Users/your_file_path/Python/Python313/chb01_01.edf" 
raw = mne.io.read_raw_edf(edf_file, preload=True)
print("Loaded EDF file")

print("Showing original raw EEG data...")
raw.plot(n_channels=10, duration=20, scalings='auto', title='Original EEG (Before CAR)')

raw_car = raw.copy().set_eeg_reference('average', projection=False)
print("Applied Common Average Referencing (CAR)")

print("Showing EEG after CAR applied...")
raw_car.plot(n_channels=10, duration=20, scalings='auto', title='EEG After CAR')
print(raw.ch_names)
