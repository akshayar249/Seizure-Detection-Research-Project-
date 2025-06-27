import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === Load the dataset ===
file_path = "chbmit_features_full.csv"
df = pd.read_csv(file_path)

# === Step 1: Filter frequency-related features ===
frequency_keywords = [
    '_power_delta',
    '_power_theta',
    '_power_alpha',
    '_power_beta',
    '_power_gamma',
    '_ratio_theta_alpha',
    '_ratio_beta_delta',
    '_ratio_alpha_beta',
    '_ratio_gamma_beta'
]

# Filter columns that contain frequency-related keywords
frequency_columns = [col for col in df.columns if any(k in col for k in frequency_keywords)]

# Add label column
frequency_columns.append('label')

# Subset the dataframe to frequency features + label
df_freq = df[frequency_columns]

# === Step 2: Handle missing values ===
df_freq = df_freq.dropna()

# === Step 3: Split features and labels ===
X_freq = df_freq.drop(columns=['label'])
y_freq = df_freq['label']

# === Step 4: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_freq, y_freq, test_size=0.2, random_state=42)

# === Step 5: Initialize and train the Random Forest model ===
rf_freq = RandomForestClassifier(n_estimators=100, random_state=42)
rf_freq.fit(X_train, y_train)

# === Step 6: Make predictions ===
y_pred_freq = rf_freq.predict(X_test)

# === Step 7: Evaluate performance ===
accuracy_freq = accuracy_score(y_test, y_pred_freq)
report_freq = classification_report(y_test, y_pred_freq)
conf_matrix_freq = confusion_matrix(y_test, y_pred_freq)

# === Step 8: Display evaluation results ===
print(f"Accuracy (Frequency Features Only): {accuracy_freq:.4f}")
print("\nClassification Report:\n", report_freq)

# === Step 9: Plot the confusion matrix ===
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_freq, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Non-Seizure', 'Seizure'],
            yticklabels=['Non-Seizure', 'Seizure'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Frequency Features Only')
plt.tight_layout()
plt.show()



def plot_class_distribution(y_labels):
    plt.figure(figsize=(5, 3))
    sns.countplot(x=y_labels, palette="pastel")
    plt.title("Class Distribution (0 = Non-Seizure, 1 = Seizure)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(X_data):
    corr = X_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap of Frequency Features")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    indices = importances.argsort()[-top_n:][::-1]
    top_features = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=top_features, palette='viridis')
    plt.title("Top Frequency Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


plot_feature_importance(rf_freq, X_freq.columns)
plot_class_distribution(y_freq)
plot_correlation_heatmap(X_freq)














