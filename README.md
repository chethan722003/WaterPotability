# 💧 Water Potability Predictor

A machine learning web application that predicts whether water is safe for drinking (potable) based on various chemical and physical attributes.

---

## 🚀 Features

- 🔍 Predict water potability using a trained ML model (Random Forest/XGBoost/SVM)
- 📊 Input 9 water quality features
- 📁 Upload CSV files for batch prediction
- ✅ Clean UI using Tailwind CSS
- 🧠 Trained on a real-world water quality dataset
- 📦 Packaged into a standalone `.exe` desktop app (PyInstaller)

---


---

## 📦 Installation

### dataset
https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability

### 🔧 Requirements

- Python 3.8+
- Flask
- scikit-learn
- pandas
- xgboost
- joblib

### 💻 Run Locally


```bash
git clone https://github.com/chethan722003/WaterPotability.git
cd WaterPotability
pip install -r requirements.txt
python app.py
