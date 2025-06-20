import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Load and Segment Data ===
df = pd.read_csv(r'C:\Users\aksha\Desktop\new\mne_env\chbmit_preprocessed_data.csv')
X_raw = df.drop("Outcome", axis=1).values
y_raw = df["Outcome"].values

channels = 23
window_size = 256

num_windows = X_raw.shape[0] // window_size
X_raw = X_raw[:num_windows * window_size]
y_raw = y_raw[:num_windows * window_size]

X_segmented = X_raw.reshape(num_windows, window_size, channels).transpose(0, 2, 1)
y_segmented = y_raw.reshape(num_windows, window_size)
y_labels = (y_segmented.sum(axis=1) > (window_size / 2)).astype(int)

print("Segmented EEG shape:", X_segmented.shape)
print("Segmented labels shape:", y_labels.shape)

# === Normalize EEG per channel ===
mean = X_segmented.mean(axis=(0, 2), keepdims=True)
std = X_segmented.std(axis=(0, 2), keepdims=True)
X_segmented = (X_segmented - mean) / std

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_segmented, y_labels, test_size=0.2, stratify=y_labels, random_state=42
)

# === Class-Weighted Loss ===
class_counts = np.bincount(y_train)
class_weights = 1. / class_counts
weights = torch.tensor(class_weights, dtype=torch.float32)

# === DataLoaders ===
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                        torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# === LTI-CNN Model ===
class LTICNN(nn.Module):
    def __init__(self):
        super(LTICNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveMaxPool2d((23, 1))
        self.fc1 = nn.Linear(64 * 23, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 23, 256)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LTICNN().to(device)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 30

losses = []

# === Training Loop ===
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}")

# === Plot Loss Curve ===
plt.plot(range(1, epochs+1), losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("LTI-CNN Training Loss Curve")
plt.grid(True)
plt.show()

# === Evaluation ===
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

# === Classification Report ===
print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=["Preictal", "Ictal"]))

# === Confusion Matrix ===
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Preictal", "Ictal"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# === Save the Trained Model ===
torch.save(model.state_dict(), "lticnn_preictal_ictal.pth")
print("✅ Model saved as 'lticnn_preictal_ictal.pth'")
