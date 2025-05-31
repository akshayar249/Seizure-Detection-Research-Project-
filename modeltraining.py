import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load feature-extracted parquet dataset
features_file = 'chbmit_features_full.parquet'  # Update path if needed
df = pd.read_parquet(features_file)

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Train/test split (with stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection: select top k features
k = 50  # You can tune this number
selector = SelectKBest(chi2, k=k)

# chi2 requires non-negative features; shift if necessary
min_feat = X_train_scaled.min()
if min_feat < 0:
    X_train_scaled_shifted = X_train_scaled - min_feat
    X_test_scaled_shifted = X_test_scaled - min_feat
else:
    X_train_scaled_shifted = X_train_scaled
    X_test_scaled_shifted = X_test_scaled

X_train_sel = selector.fit_transform(X_train_scaled_shifted, y_train)
X_test_sel = selector.transform(X_test_scaled_shifted)

# Train SVM classifier
svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm.fit(X_train_sel, y_train)

# Predict & evaluate
y_pred = svm.predict(X_test_sel)

print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
