import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)

# === Step 1: Load the data ===
features = pd.read_csv("chbmit_features_full.csv")  # Replace with your actual file path

# === Step 2: Separate features and label ===
X = features.drop(columns=['label'])
y = features['label']

# === Step 3: Handle NaNs (if any) ===
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# === Step 4: Feature scaling (important for SVM) ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# === Step 5: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# === Step 6: Train the SVM classifier ===
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

# === Step 7: Predictions and evaluation ===
y_pred = svm.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['preictal', 'ictal']))

# === Step 8: Confusion Matrix ===
class_names = ['preictal', 'ictal']
ConfusionMatrixDisplay.from_estimator(svm, X_test, y_test, display_labels=class_names, cmap='Blues')
plt.title("SVM Confusion Matrix (Full Feature Set)")
plt.show()

# === Step 9: ROC Curve ===
y_score = svm.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: preictal vs ictal")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# === Step 10: Classification Metrics Bar Chart ===
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
classes = ['preictal', 'ictal']
x = np.arange(len(classes))
width = 0.2

plt.figure(figsize=(10, 5))
plt.bar(x - width, precision, width=width, label='Precision')
plt.bar(x, recall, width=width, label='Recall')
plt.bar(x + width, f1, width=width, label='F1-Score')
plt.xticks(x, classes)
plt.xlabel("Class")
plt.ylabel("Score")
plt.title("Classification Metrics per Class")
plt.legend()
plt.grid(True)
plt.show()
