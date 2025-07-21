# main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("credit_scoring_dataset.csv")

# Encode categorical variables
categorical_cols = ["Employment_Status", "Marital_Status"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
data["Credit_Status"] = target_encoder.fit_transform(data["Credit_Status"])  # Good=1, Bad=0

# Split features and target
X = data.drop("Credit_Status", axis=1)
y = data["Credit_Status"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "XGBoost": XGBClassifier(eval_metric='logloss')  # Removed use_label_encoder
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n--- Training {name} ---")  # <-- Replaced emoji line
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
