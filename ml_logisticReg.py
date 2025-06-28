import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


df = pd.read_csv("chbmit_features_full.csv")


df_clean = df.dropna()


X = df_clean.drop(columns=["label"])
y = df_clean["label"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))


df_clean["predicted_label"] = model.predict(X_scaled)



joblib.dump(model, "seizure_model.pkl")
joblib.dump(scaler, "seizure_scaler.pkl")

print(" Model and scaler saved successfully.")


# Save predictions to CSV 
df_clean.to_csv("seizure_predictions.csv", index=False)
print("\n Predictions saved to 'seizure_predictions.csv'")

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['non-seizure', 'seizure']))




cm = confusion_matrix(y_test, y_pred)


labels = ['non-seizure', 'seizure']


fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=labels,
    yticklabels=labels,
    ylabel='True label',
    xlabel='Predicted label',
    title='Logistic Regression Confusion Matrix'
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

