import mne
import matplotlib.pyplot as plt
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
    elif option == 0:
        raw_filtered = raw.copy()  # no filtering
        label = "Raw EEG (no filtering)"
    else:
        raise ValueError("Invalid option. Choose 0, 1, 2, 3, or 4.")
    return raw_filtered, label

def main():
    # Load file
    raw = mne.io.read_raw_bdf("noise_filters\\EEG_Cat_Study4_II_II_S1.bdf", preload=True)

    # Choose filtering option
    print("Choose EEG Analysis Mode:")
    print("0. No Filtering (Raw Signal)")
    print("1. General EEG (1–30 Hz)")
    print("2. Alpha Rhythm (8–13 Hz)")
    print("3. Beta Activity (13–30 Hz)")
    print("4. Full EEG (0.5–45 Hz)")

    option = int(input("Enter option number (0–4): "))
    raw_filtered, label = apply_filter(raw, option)

    # Pick channels
    channels = ['Fp1', 'F3', 'O1']
    raw_pick = raw_filtered.copy().pick_channels(channels)

    # Time window
    start_sec = 11
    end_sec = 15
    sfreq = int(raw.info['sfreq'])
    start_sample = int(start_sec * sfreq)
    end_sample = int(end_sec * sfreq)

    # Get data in microvolts
    data, times = raw_pick[:, start_sample:end_sample]
    data_uV = data * 1e6  # convert to µV

    # Remove DC offset per channel
    data_uV = data_uV - np.mean(data_uV, axis=1, keepdims=True)

    print(f"Max amplitude in data (µV): {np.max(data_uV):.2f}")
    print(f"Min amplitude in data (µV): {np.min(data_uV):.2f}")

    max_amplitude = 150  # max amplitude allowed in µV
    min_spacing = 15     # minimum spacing between channels in µV

    # Calculate vertical spacing (at least min_spacing, but enough to separate channels)
    spacing = max(min_spacing, max_amplitude * 2)

    plt.figure(figsize=(12, 5))

    for idx, ch_data in enumerate(data_uV):
        y_offset = idx * spacing

        # Identify exceedances
        exceed_pos_idx = np.where(ch_data > max_amplitude)[0]
        exceed_neg_idx = np.where(ch_data < -max_amplitude)[0]

        # Clip data to ±max_amplitude
        clipped_data = np.clip(ch_data, -max_amplitude, max_amplitude)

        # Plot clipped data with offset
        plt.plot(times, clipped_data + y_offset, label=channels[idx])

        # Mark exceedances with red 'X' above or below clipped line
        if exceed_pos_idx.size > 0:
            plt.scatter(times[exceed_pos_idx], 
                        np.full_like(exceed_pos_idx, max_amplitude + y_offset),
                        marker='X', s=60, color='red', label='_nolegend_')
        if exceed_neg_idx.size > 0:
            plt.scatter(times[exceed_neg_idx], 
                        np.full_like(exceed_neg_idx, -max_amplitude + y_offset),
                        marker='X', s=60, color='red', label='_nolegend_')

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"EEG from {start_sec}s to {end_sec}s — {label}")
    plt.legend(loc='upper right')
    plt.grid(True)

    # Set y limits for clear channel separation
    plt.ylim(-max_amplitude, spacing * (len(channels) - 1) + max_amplitude)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
