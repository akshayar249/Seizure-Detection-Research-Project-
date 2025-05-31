import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# --- Parameters ---
input_parquet = 'eeg_dataset_cleaned.parquet'  # Your parquet file path
label_col = 'outcome'  # Column name with seizure label: 0/1
fs = 256                # Sampling frequency in Hz
window_sec = 5          # Window length in seconds
window_size = fs * window_sec  # Samples per window (e.g., 1280 if fs=256 and 5 sec)
stride = window_size // 2       # 50% overlap

# --- Load dataset ---
print("Loading dataset...")
df = pd.read_parquet(input_parquet)
channels = [col for col in df.columns if col != label_col]
print(f"Channels: {channels}")
print(f"Total samples: {len(df)}")

# --- Prepare windows of raw EEG data ---
X_windows = []
y_windows = []

for start_idx in range(0, len(df) - window_size + 1, stride):
    segment = df.iloc[start_idx:start_idx + window_size]
    # Extract EEG data for channels, shape (window_size, num_channels)
    segment_data = segment[channels].values

    # Label is majority label in this segment
    segment_label = segment[label_col].mode()[0]

    X_windows.append(segment_data)
    y_windows.append(segment_label)

X = np.array(X_windows)  # shape: (num_windows, window_size, num_channels)
y = np.array(y_windows)

print(f"Total windows: {X.shape[0]}, Window shape: {X.shape[1:]}")

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Build CNN + LSTM model ---
model = Sequential([
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(window_size, len(channels))),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(filters=128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')  # Binary classification output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train model ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# --- Evaluate model ---
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
