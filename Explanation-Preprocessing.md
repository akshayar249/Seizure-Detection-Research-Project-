### Why You Applied a Band-Pass Filter

1. **Focus on EEG Bands**: The 0.5–40 Hz filter keeps delta, theta, alpha, beta, and lower gamma (0.5–40 Hz), which are key for brain activity, while cutting out irrelevant frequencies.

2. **Remove Low-Frequency Noise**: It eliminates slow drifts (below 0.5 Hz) like skin potentials, electrode drift, and movement artifacts, which were visible in your raw data.

3. **Remove High-Frequency Noise**: It cuts off noise above 40 Hz, like 50 Hz power line noise (common in India), muscle artifacts, and amplifier noise, as confirmed by your PSD plot.

4. **Improve Signal Quality**: Filtering boosts the signal-to-noise ratio by removing irrelevant noise, making the EEG data cleaner for further steps like wavelet denoising and adaptive filtering.

5. **Prepare for Next Steps**: It sets up the data for targeted preprocessing by focusing on the frequency range of interest, making subsequent steps more effective.


### Loading and Inspecting Raw EEG Data
The notebook loads an EEG dataset (`EEG_Cat_Study4_II_II_S1.bdf`) using MNE into a `raw` object, containing voltage measurements (in microvolts) from scalp electrodes. It visualizes 32 channels over ~23 minutes, showing:
- **Channels (Y-axis)**: Electrode positions (e.g., Fp1 for frontal pole) per the 10-20 system.
- **Time (X-axis)**: Duration in seconds.
- **Waveforms**: Voltage changes, appearing flat with small fluctuations due to low amplitude and noise (unwanted signals like muscle activity).

Metadata is printed:
- **Channels**: 73 total, including EEG (brain activity), EOG (eye movements, electrooculography), reference (M1, M2), and Status (event markers).
- **Sampling Rate**: 256 Hz (256 measurements/second), capturing brain activity up to 128 Hz, covering EEG bands (Delta: 0.5–4 Hz, Theta: 4–8 Hz, Alpha: 8–12 Hz, Beta: 12–30 Hz, Gamma: 30–100 Hz).
- **Data Shape**: 73 channels × 389,888 samples (~25 minutes).


### Bandpass Filtering
To remove noise (e.g., low-frequency drifts from movement or high-frequency electrical interference), a **bandpass filter** (keeping 0.5–40 Hz) is applied to a copy of the data (`raw_filtered`). This range includes key EEG bands. The original and filtered signals are plotted for comparison, and a **Power Spectral Density (PSD)** plot shows signal strength across frequencies (up to 100 Hz), checking for noise like 50 Hz power line interference.


### Wavelet Denoising
To further reduce noise within 0.5–40 Hz (e.g., muscle artifacts), **wavelet denoising** is used:
- **Process**: The filtered data is decomposed using the `db4` wavelet into 6 components (1 low-frequency approximation, 5 higher-frequency details). Noise is estimated and reduced via **soft thresholding**, preserving brain signals. The signal is reconstructed (`data_denoised`).
- **Visualization**: A 10-second plot of 32 channels shows smoother signals, with high-frequency noise reduced but brain waves (e.g., alpha) preserved. A PSD comparison (filtered vs. denoised) confirms reduced high-frequency noise (30–40 Hz).

### Adaptive Filtering for Eye Blinks
**Eye blink artifacts** (large voltage spikes in frontal channels like Fp1) are removed using **adaptive filtering** with EOG channels:
- **Process**: An EOG reference signal is created by averaging EOG channels (LVEOG, RVEOG, LHEOG, RHEOG), showing blink peaks. The **Least Mean Squares (LMS)** algorithm subtracts blink patterns from EEG channels. Parameters include an adaptation rate (0.01) and filter length (32 samples, ~125 ms).
- **Visualization**: Plots compare 10 seconds of data before and after. Frontal channels show reduced blink peaks, while others (e.g., Pz) remain unchanged. A PSD comparison shows slightly reduced low frequencies (0.5–4 Hz) due to blink removal, with EEG bands preserved.