import mne
import matplotlib.pyplot as plt


edf_file = "C:/Users/your_file_path/Python313/chb01_01.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

raw_notch = raw.copy().notch_filter(freqs=[60], verbose=True)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Power Spectral Density Before vs After Notch Filtering", fontsize=16)

raw.plot_psd(fmax=100, ax=axs[0], show=False)
axs[0].set_title("Before Notch Filter")

raw_notch.plot_psd(fmax=100, ax=axs[1], show=False)
axs[1].set_title("After Notch Filter")

plt.tight_layout()
plt.show()

