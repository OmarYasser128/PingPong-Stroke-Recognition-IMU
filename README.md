# 🏓 Ping-Pong Stroke Recognition Using Smartphone IMU Data

This project presents an end-to-end machine learning framework for
automatic recognition of ping-pong strokes using inertial sensor data
collected from a smartphone. Accelerometer and gyroscope signals are
processed, transformed into meaningful features, and classified using
multiple machine learning models.

---

## 📌 Project Overview

- **Sensors**: Smartphone Accelerometer & Gyroscope
- **Sampling Rate**: ~100 Hz
- **Actions**: Forehand, Backhand (extendable)
- **Window Type**: One semantic window per stroke
- **Window Duration**: 1.5 – 4.5 seconds (actual movement)
- **Models**: Random Forest, SVM, KNN, Logistic Regression

---

## 📂 Dataset Description

The smartphone was securely attached to the playing hand of each
participant. Each trial corresponds to a complete stroke, including
preparation, execution, and follow-through.

### Recorded Signals
- Acceleration (x, y, z)
- Angular velocity (x, y, z)

### Derived Signals
- Acceleration magnitude
- Gyroscope magnitude

Each trial is labeled with:
- Stroke type
- Player identifier
- Trial number

---

## ⚙️ Feature Extraction

Each stroke window is converted into a fixed-length feature vector
using both time-domain and frequency-domain features:

### Time-Domain Features
- Mean
- Median
- Standard Deviation
- Variance
- Min / Max
- Range
- RMS
- Signal Magnitude Area (SMA)

### Frequency-Domain Features
- Dominant frequency (FFT)
- Spectral energy
- Spectral entropy

### Temporal Features
- Stroke duration
- Zero-crossing rate

---

## 🤖 Machine Learning Models

The following classifiers were evaluated:

- Random Forest
- Support Vector Machine (RBF)
- K-Nearest Neighbors
- Logistic Regression

All models were evaluated using **5-fold stratified cross-validation**.
Player identity was excluded from training to prevent subject-specific
bias.

---

## 📊 Results

- **Best Model**: Random Forest
- **Evaluation Metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix

### Model Comparison
![Model Comparison](results/model_comparison.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Feature Importance
![Feature Importance](results/feature_importance.png)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/train_models.py
