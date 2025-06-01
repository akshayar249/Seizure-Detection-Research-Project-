import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_edf(
    'C:/Users/GANESH/AppData/Local/Programs/Python/Python313/chb01_01.edf', preload=True)

raw.pick_types(eeg=True)


raw_notch = raw.copy().notch_filter(freqs=50)  


raw_notch_car = raw_notch.copy().set_eeg_reference(ref_channels='average')


start_time = 10     
duration = 5        
stop_time = start_time + duration

channel_index = 0 
channel_name = raw.ch_names[channel_index]

sfreq = raw.info['sfreq']  
start_sample = int(start_time * sfreq)
stop_sample = int(stop_time * sfreq)

raw_data, times = raw[channel_index, start_sample:stop_sample]
notch_data, _ = raw_notch[channel_index, start_sample:stop_sample]
notch_car_data, _ = raw_notch_car[channel_index, start_sample:stop_sample]

plt.figure(figsize=(14, 6))
plt.plot(times, raw_data[0], label='Original', alpha=0.7)
plt.plot(times, notch_data[0], label='After Notch Filter', alpha=0.7)
plt.plot(times, notch_car_data[0], label='After Notch + CAR', alpha=0.7)

plt.title(f"Time-Domain Signal Comparison - Channel: {channel_name}\nWindow: {start_time}s to {stop_time}s")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("time_domain_comparison.png", dpi=300)  
plt.show()
