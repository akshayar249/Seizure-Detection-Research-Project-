import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_edf(
    'C:/Users/your_file_path/Python/Python313/chb01_01.edf', preload=True)

raw.pick_types(eeg=True)

psd_raw = raw.compute_psd(fmax=60)

raw_notch = raw.copy().notch_filter(freqs=50)
psd_notch = raw_notch.compute_psd(fmax=60)

raw_notch_car = raw_notch.copy().set_eeg_reference(ref_channels='average')
psd_notch_car = raw_notch_car.compute_psd(fmax=60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

psd_raw.plot(axes=axes[0], show=False)
axes[0].set_title('Raw EEG (No Filter)')

psd_notch.plot(axes=axes[1], show=False)
axes[1].set_title('After Notch Filter (50 Hz)')

psd_notch_car.plot(axes=axes[2], show=False)
axes[2].set_title('After Notch + CAR')

fig.suptitle('PSD Comparison: Raw vs Notch vs Notch + CAR', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("psd_comparison_raw_notch_car.png", dpi=300)  
plt.show()
