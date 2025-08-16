

 📌 Overview

This repository contains the implementation of an **automated EEG-based seizure detection framework** that combines advanced signal preprocessing techniques with **machine learning** and **deep learning** models.

The system classifies EEG signals into **seizure** and **non-seizure** states, providing a scalable solution for **real-time monitoring and clinical applications**.

The project explores multiple classifiers, including:

* **SVM**
* **KNN**
* **Logistic Regression**
* **Random Forest**
* **CNN+LTI**
* **CNN–BiLSTM–SCL**
* **SGCN–deepRNN**
* **CNN-DNN**

All models are evaluated on the **CHB-MIT Scalp EEG dataset** from PhysioNet.

---

## ⚡ Key Features

### 🧹 Signal Preprocessing Pipeline

* Independent Component Analysis (**ICA**)
* Principal Component Analysis (**PCA**)
* Empirical Mode Decomposition (**EMD**)
* Regression-based artifact removal (EOG/ECG)
* Bandpass & notch filtering

### 📐 Feature Extraction

* **Time-domain**: Mean, variance, kurtosis, Hjorth parameters
* **Frequency-domain**: Power spectral density (delta, theta, alpha, beta, gamma)
* **Band power ratios**: e.g., beta/delta, theta/alpha
* **Wavelet features**: DWT energy & entropy
* **Nonlinear measures**: entropy, fractal dimension

### 🤖 Models Implemented

* **Classical ML**: SVM, Logistic Regression, KNN, Random Forest
* **Deep Learning**: CNN+LTI, CNN–BiLSTM–SCL, SGCN–deepRNN, CNN-DNN

### 📈 Performance Highlights

* **KNN** → 98.2% accuracy (best performing traditional model)
* **Logistic Regression** → 97.4% accuracy, highly interpretable
* **CNN+LTI** → 94% accuracy, biologically inspired filtering
* **Deep models (BiLSTM, SGCN)** → Effective in spatiotemporal modeling

---

## 📂 Dataset

* **CHB-MIT Scalp EEG Database** (PhysioNet)
* Pediatric epilepsy patient recordings
* **Sampling rate:** 256 Hz
* **Patients:** 24 (∼900 hours of EEG data)
* Includes **seizure onset/offset annotations**

---

## 📊 Results

| Model                   | Input    | Accuracy  | Key Strengths                |
| ----------------------- | -------- | --------- | ---------------------------- |
| **KNN**                 | Features | **98.2%** | Simple, effective            |
| **Logistic Regression** | Features | 97.4%     | Fast, interpretable          |
| **SVM**                 | Features | 96%       | Robust, generalizable        |
| **Random Forest**       | Features | 96.6%     | Ensemble robustness          |
| **CNN+LTI**             | Raw EEG  | 94%       | Biologically inspired filter |
| **CNN–BiLSTM–SCL**      | Raw EEG  | 86%       | Temporal-spatial modeling    |
| **SGCN–deepRNN**        | Raw EEG  | 89%       | Graph + temporal learning    |
| **CNN-DNN**             | Raw EEG  | 92%       | Automatic feature learning   |

---

## 🔮 Future Scope

* Real-time deployment on **edge/embedded devices**
* Patient-independent generalization via **transfer learning**
* **Multimodal integration** with ECG, EMG, or video
* Improved **explainability** (e.g., SHAP, LIME, attention maps)
* **Seizure type classification** beyond binary detection

---

## 👩‍💻 Contributors

1. **Akshaya Ramesh**
2. **Diya Ghorpade**
3. **Siri Ganesh**
4. **Ananya Gupta**
5. **Leela Aditya Simma**

---




