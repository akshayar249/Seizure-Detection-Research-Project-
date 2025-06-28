import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv("chbmit_features_full.csv")
df_clean = df.dropna()

X = df_clean.drop(columns=["label"])
y = df_clean["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 1️⃣ Vary K for KNN
k_values = range(1, 16)
knn_accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn_accuracies.append(accuracy_score(y_test, y_pred))

# 2️⃣ Vary C for Logistic Regression
c_values = [0.01, 0.1, 1, 10, 100]
lr_accuracies = []
for c in c_values:
    lr = LogisticRegression(C=c, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    lr_accuracies.append(accuracy_score(y_test, y_pred))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# KNN plot
ax1.plot(k_values, knn_accuracies, marker='o', color='blue')
ax1.set_title("KNN Accuracy vs. Number of Neighbors (k)")
ax1.set_xlabel("k (Number of Neighbors)")
ax1.set_ylabel("Accuracy")
ax1.grid(True)

# Logistic Regression plot
ax2.plot(c_values, lr_accuracies, marker='s', color='green')
ax2.set_xscale('log')
ax2.set_title("Logistic Regression Accuracy vs. C")
ax2.set_xlabel("C (Inverse of Regularization Strength)")
ax2.set_ylabel("Accuracy")
ax2.grid(True)

plt.tight_layout()
plt.show()
