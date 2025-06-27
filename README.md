# 🧠 Frequency-Domain EEG Seizure Detection with Random Forest

This project focuses on detecting seizures from EEG recordings using only **frequency-domain features** such as power bands (delta, theta, alpha, beta, gamma) and their ratios. The model is built using a Random Forest classifier and visualized with insightful analysis plots.

---

## 📊 Dataset

- **Source**: CHB-MIT Scalp EEG Dataset (preprocessed to CSV)
- **File Used**: `chbmit_features_full.csv`
- **Features**: Frequency-domain attributes from multiple EEG channels
- **Target**: `label` column (0 = Non-Seizure, 1 = Seizure)

---

## 🧪 Model Workflow

1. **Feature Selection**:  
   Only frequency-related columns are selected:
   - `_power_delta`, `_power_theta`, `_power_alpha`, `_power_beta`, `_power_gamma`
   - Ratios: `_ratio_theta_alpha`, `_ratio_alpha_beta`, etc.

2. **Data Preprocessing**:
   - Drop NA values
   - Split into `X` (features) and `y` (label)

3. **Training**:
   - 80/20 Train-Test split
   - RandomForestClassifier with 100 trees

4. **Evaluation**:
   - Accuracy, classification report
   - Visual analysis (confusion matrix, feature importance, etc.)

---

## 📈 Visual Analysis

| Graph | Description |
|-------|-------------|
| 🟪 Confusion Matrix | Validates performance across seizure/non-seizure |
| 🌿 Feature Importances | Highlights most predictive EEG bands and ratios |
| 📊 Class Distribution | Shows data balance (important for training) |
| 🔥 Correlation Heatmap | Identifies redundancy among frequency features |

---

## 📂 Files Included

| File | Description |
|------|-------------|
| `frequency_model.py` | Python script with training pipeline |
| `visualizations.py` | Optional plotting functions for insights |
| `Frequency_Domain_EEG_Analysis_Report.docx` | Project report with graph explanations |
| `README.md` | You're reading it :) |

---

## 🔧 Future Work

- Try CNN/LSTM using raw time-series windows
- Explore attention over frequency bands
- Handle data imbalance using SMOTE / Focal Loss
- Deploy web app with live predictions

---

## 🛡️ Author

- **Leeladitya Simma**
- Cybersecurity + Embedded Systems + AI for Neuroscience

---

## 🏁 Get Started

```bash
pip install pandas scikit-learn matplotlib seaborn
python frequency_model.py



