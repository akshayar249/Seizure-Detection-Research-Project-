 Logistic Regression and  K-Nearest Neighbors ML models for EEG Seizure Detection

This branch focuses on implementing and comparing **Logistic Regression** and **K-Nearest Neighbors (KNN)** models for EEG-based seizure detection using preprocessed features from the CHB-MIT dataset.

---

📁 Files Included

| File Name                    | Description                                      |
|-----------------------------|--------------------------------------------------|
| `ml_logisticReg.py`         | Logistic Regression model with performance metrics |
| `ml_KNN.py`                 | KNN classifier with confusion matrix evaluation   |
| `KNN_vs_LReg_param_tuning.py` | Accuracy comparison by varying `k` (KNN) and `C` (LR) |

---

 📊 Dataset

All scripts use the `chbmit_features_full.csv` file (not included in this branch) which contains pre-extracted EEG signal features and seizure labels.

 🧠 Model Overview

 🔹 Logistic Regression
- Regularization parameter `C` varied from 0.01 to 100
- Accuracy peaks at `C = 10`
- Simple, interpretable, high accuracy (~97.4%)

 🔹 K-Nearest Neighbors
- `k` varied from 1 to 15
- Accuracy highest at `k = 1`
- Non-parametric, sensitive to scaling and neighbor count


📚 Reference Dataset

CHB-MIT Scalp EEG Dataset   

📚 References

1. A. Shoeb and J. Guttag, “Application of machine learning to epileptic seizure detection,” ICML 2010.
2. U. R. Acharya et al., “Deep convolutional neural network for seizure detection,” Computers in Biology and Medicine, 2018.
3. M. T. Baharuddin et al., “Classification of EEG signals using KNN,” ICT4M, 2010.
4. L. Zhang et al., “Classification of seizure EEG using logistic regression,” IEEE BIBM, 2016.

🚀 Future Scope

- Real-time EEG seizure monitoring
- Transfer learning for patient adaptation
- Web-based or embedded deployment.
