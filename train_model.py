# train_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv('data/water_potability.csv')

# 2. Handle missing values
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 3. Feature and label split
X = df_imputed.drop('Potability', axis=1)
y = df_imputed['Potability']

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Balance the dataset using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 7. Train RandomForest with best hyperparameters
model = RandomForestClassifier(
    class_weight='balanced',
    max_depth=None,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# 9. Save model and scaler
joblib.dump(model, 'model/water_quality_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("✅ Model and scaler saved successfully in 'model/' directory.")
