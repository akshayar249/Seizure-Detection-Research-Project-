
# dataset :   noise_filters/chb17b_68.edf       
# dataset source: mit


import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np

def apply_filter(raw, option):
    if option == 1:
        raw_filtered = raw.copy().filter(1., 30.)
        label = "General EEG (1–30 Hz)"
    elif option == 2:
        raw_filtered = raw.copy().filter(8., 13.)
        label = "Alpha Rhythm (8–13 Hz)"
    elif option == 3:
        raw_filtered = raw.copy().filter(13., 30.)
        label = "Beta Activity (13–30 Hz)"
    elif option == 4:
        raw_filtered = raw.copy().filter(0.5, 45.)
        label = "Full EEG (0.5–45 Hz)"
    else:
        raise ValueError("Invalid option. Choose 1, 2, 3, or 4.")
    return raw_filtered, label

def plot_eeg(raw_filtered, start_sec=10, end_sec=14, channels=['FP1-F7', 'F7-T7', 'T7-P7']):
    sfreq = int(raw_filtered.info['sfreq'])
    start_sample = int(start_sec * sfreq)
    end_sample = int(end_sec * sfreq)

    raw_filtered.pick_channels(channels)
    data, times = raw_filtered[:, start_sample:end_sample]

    data_uV = data * 1e6
    spacing = 150

    plt.figure(figsize=(12, 6))
    for i in range(len(data_uV)):
        plt.plot(times, data_uV[i] + i * spacing, label=raw_filtered.ch_names[i])

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.title(f"EEG Plot: {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_frequency_domain(raw_filtered, label, channels=['FP1-F7', 'F7-T7', 'T7-P7']):
    # Frequency domain analysis
    data, times = raw_filtered.get_data(return_times=True)
    sfreq = raw_filtered.info['sfreq']
    
    # Create frequency-domain plot for the selected channels
    plt.figure(figsize=(12, 6))
    for i, channel_data in enumerate(data):
        if raw_filtered.ch_names[i] in channels:  # Ensure we're only plotting the selected channels
            f, Pxx = welch(channel_data, fs=sfreq, nperseg=1024)
            plt.plot(f, 10 * np.log10(Pxx), label=raw_filtered.ch_names[i])

    plt.xlim([0, 60])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    plt.title(f"Frequency Domain Plot: {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ======== MAIN FLOW ========

# Load file
raw = mne.io.read_raw_edf('noise_filters/chb17b_68.edf', preload=True)

# Choose analysis option
print("Choose EEG Analysis Mode:")
print("1. General EEG (1–30 Hz)")
print("2. Alpha Rhythm (8–13 Hz)")
print("3. Beta Activity (13–30 Hz)")
print("4. Full EEG (0.5–45 Hz)")

option = int(input("Enter option number (1–4): "))
raw_filtered, label = apply_filter(raw, option)

# Plot Time Domain
plot_eeg(raw_filtered)

# Plot Frequency Domain for 3 channels
plot_frequency_domain(raw_filtered, label)
