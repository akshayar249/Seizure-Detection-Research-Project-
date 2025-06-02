import pyarrow.parquet as pq
import numpy as np
from mne.decoding import CSP
import matplotlib.pyplot as plt
import time

file_path = "C://Users//GANESH//AppData//Local//Programs//Python//Python313//eeg_dataset_cleaned.parquet" 
sampling_rate = 256
epoch_sec = 2
epoch_len = sampling_rate * epoch_sec
stride = 2 * epoch_len 
max_epochs = 500

pf = pq.ParquetFile(file_path)
X_epochs = []
y_epochs = []

start_time = time.time()

for i in range(min(2, pf.num_row_groups)):  
    df = pf.read_row_group(i).to_pandas()

    if 'outcome' not in df.columns:
        continue

    signal_cols = [col for col in df.columns if col != "outcome"]
    eeg_data = df[signal_cols].values.T[:, ::2]  
    labels = df['outcome'].values[::2]

    for start in range(0, eeg_data.shape[1] - epoch_len, stride):
        end = start + epoch_len
        epoch = eeg_data[:, start:end]
        if epoch.shape[1] != epoch_len:
            continue
        label = int(round(np.mean(labels[start:end])))
        X_epochs.append(epoch)
        y_epochs.append(label)

        if len(X_epochs) >= max_epochs:
            break

    print(f"Processed row group {i+1}, total epochs so far: {len(X_epochs)}")
    if len(X_epochs) >= max_epochs:
        break

print(f" Finished preprocessing in {time.time() - start_time:.2f} seconds.")


X = np.stack(X_epochs)
y = np.array(y_epochs)

csp = CSP(n_components=4, log=True)
X_csp = csp.fit_transform(X, y)

explained_var = np.var(X_csp, axis=0) / np.sum(np.var(X_csp, axis=0))

plt.figure(figsize=(6, 4))
plt.bar(range(1, len(explained_var)+1), explained_var * 100, color="mediumseagreen")
plt.xlabel("CSP Component")
plt.ylabel("Variance (%)")
plt.title("CSP applied")
plt.grid(True)
plt.tight_layout()
plt.show()

n_components_to_plot = min(4, csp.filters_.shape[0])
fig, axes = plt.subplots(n_components_to_plot, 1, figsize=(10, 2.5 * n_components_to_plot), constrained_layout=True)

for i in range(n_components_to_plot):
    ax = axes[i] if n_components_to_plot > 1 else axes
    ax.bar(range(len(signal_cols)), csp.filters_[i], color='skyblue')
    ax.set_title(f"CSP Component {i+1} (Spatial Filter Weights)")
    ax.set_xticks(range(len(signal_cols)))
    ax.set_xticklabels(signal_cols, rotation=90, fontsize=8)
    ax.set_ylabel("Weight")
    ax.grid(True)

plt.suptitle("CSP Spatial Patterns (Filter Weights per EEG Channel)", fontsize=14)
plt.tight_layout()
plt.show()
