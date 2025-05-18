import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import numpy as np

# Load your pre-extracted features parquet file
df = pd.read_parquet('chbmit_extracted_features.parquet')  # replace with your file path

# Separate features and label
X = df.drop(columns=['label'])
y = df['label']

# Split into train-test (stratified to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize Random Forest with balanced class weights
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 75, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Grid search with 3-fold CV using F1 score (balance precision/recall)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# Predict on test set
y_pred = best_rf.predict(X_test)

print("Best parameters:", grid_search.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Adjust threshold for better recall
y_probs = best_rf.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# For example, find threshold for recall >= 0.5
desired_recall = 0.5
idx = np.argmax(recalls >= desired_recall)
opt_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

print(f"Optimal threshold for recall >= {desired_recall}: {opt_threshold:.3f}")

# Apply threshold
y_pred_adj = (y_probs >= opt_threshold).astype(int)

print("Classification report with adjusted threshold:")
print(classification_report(y_test, y_pred_adj))
