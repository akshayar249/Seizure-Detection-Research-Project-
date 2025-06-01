import mne
import pywt
import numpy as np
from scipy.signal import lfilter

def load_data(fname):
    raw = mne.io.read_raw_bdf(fname, preload=True)
    raw.filter(1., 40., fir_design='firwin')  # Optional band-pass filter
    return raw


def wavelet_denoise(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [pywt.threshold(c, value=uthresh, mode='soft') if i > 0 else c
                     for i, c in enumerate(coeffs)]
    return pywt.waverec(coeffs_thresh, wavelet)

def adaptive_filter(signal, reference, mu=0.01, order=5):
    """LMS Adaptive Filter"""
    N = len(signal)
    y = np.zeros(N)
    e = np.zeros(N)
    w = np.zeros(order)
    for n in range(order, N):
        x = reference[n - order:n][::-1]
        y[n] = np.dot(w, x)
        e[n] = signal[n] - y[n]
        w = w + 2 * mu * e[n] * x
    return e

def denoise(raw):
    raw_data = raw.get_data()
    denoised = []

    for i, ch in enumerate(raw_data):
        print(f"Processing channel {i}")
        wavelet_clean = wavelet_denoise(ch)
        # Assume ECG is in channel -1 (or change according to your data)
        if raw_data.shape[0] > 1:
            ecg = raw_data[-1]
            final_clean = adaptive_filter(wavelet_clean, ecg)
        else:
            final_clean = wavelet_clean
        denoised.append(final_clean)

    # Replace original data
    raw._data = np.array(denoised)
    return raw

if __name__ == '__main__':
    raw = load_data('your_file.bdf')  # Replace with actual filename
    raw_clean = denoise(raw)
    raw_clean.plot()