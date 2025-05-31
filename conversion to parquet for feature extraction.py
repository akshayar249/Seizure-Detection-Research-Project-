import os
import mne
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def preprocess_eeg_to_parquet(edf_path, output_dir, ica_components_to_exclude=None):
    # Step 1: Load EEG
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Step 2: Rename channels to a consistent format
    raw.rename_channels(lambda name: name.upper().replace('-', '_'))

    # Step 3: Pick EEG channels only
    raw.pick_types(eeg=True)

    # Step 4: Filter the data (1–40 Hz)
    raw.filter(l_freq=1., h_freq=40., fir_design='firwin', verbose=False)

    # Step 5: ICA for artifact removal
    ica = ICA(n_components=len(raw.ch_names), random_state=42, max_iter='auto')
    ica.fit(raw)

    # Step 6: Automatically or manually exclude components
    if ica_components_to_exclude is not None:
        ica.exclude = ica_components_to_exclude
    else:
        ica.exclude = [0]  # You can improve this with auto-detection

    # Step 7: Apply ICA
    raw_clean = ica.apply(raw.copy())

    # Step 8: Get cleaned EEG data
    data_clean = raw_clean.get_data()  # shape: (n_channels, n_times)

    # Step 9: Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.99, svd_solver='full')
    transformed = pca.fit_transform(data_clean.T)  # shape: (n_times, components)
    reconstructed = pca.inverse_transform(transformed).T  # shape: (n_channels, n_times)

    # Step 10: Create DataFrame with channel names and time index
    times = raw_clean.times  # in seconds
    channel_names = raw_clean.ch_names
    df = pd.DataFrame(reconstructed.T, columns=channel_names)
    df["time_sec"] = times
    df = df.set_index("time_sec")

    # Add source file column (optional, helpful when combining)
    file_id = os.path.splitext(os.path.basename(edf_path))[0]
    df["source_file"] = file_id

    # Step 11: Save as .parquet
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, f"{file_id}_cleaned.parquet")
    df.to_parquet(parquet_path, index=True)

    print(f"[✓] Processed and saved: {parquet_path}")
    return parquet_path

def combine_parquet_files(input_folder, output_filename="eeg_dataset_cleaned.parquet"):
    # List all .parquet files that end with _cleaned.parquet
    parquet_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith("_cleaned.parquet")
    ]

    if not parquet_files:
        print("❌ No cleaned parquet files found in the input folder.")
        return

    # Load and concatenate all parquet files
    combined_df = pd.concat([pd.read_parquet(f) for f in parquet_files], axis=0)

    # Save to one combined parquet file
    output_path = os.path.join(input_folder, output_filename)
    combined_df.to_parquet(output_path, index=True)

    print(f"[✅] Combined dataset saved at: {output_path}")

# ======= Main =========
if __name__ == "__main__":
    input_folder = "chbmit_data"         # Folder containing .edf files
    output_folder = "processed_parquet"  # Folder to store individual .parquet files

    edf_files = [f for f in os.listdir(input_folder) if f.endswith(".edf")]

    for edf_file in edf_files:
        edf_path = os.path.join(input_folder, edf_file)
        preprocess_eeg_to_parquet(edf_path, output_folder)

    # After processing all, combine them
    combine_parquet_files(output_folder)
