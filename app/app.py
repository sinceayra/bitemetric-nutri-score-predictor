import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# 1. Define file paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "en.openfoodfacts.org.products.tsv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Specify exact columns to load (saves massive RAM on the 1.01 GB file)
FEATURES = [
    "energy_100g",
    "sugars_100g",
    "saturated-fat_100g",
    "sodium_100g",
    "fiber_100g",
    "proteins_100g",
]
TARGET = "nutrition_grade_fr"

print("Loading dataset (filtering to 7 core columns)...")
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    usecols=FEATURES + [TARGET],
    low_memory=False
)
print(f"Raw rows loaded: {len(df):,}")

# 3. Clean Data & Remove Outliers
df = df.dropna().copy()

# Standardize target labels to uppercase A-E
df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()
df = df[df[TARGET].isin(["A", "B", "C", "D", "E"])]

# Filter out impossible values (negative values or >100g per 100g product)
df = df[
    (df["energy_100g"] >= 0) & (df["energy_100g"] <= 4000) &
    (df["sugars_100g"] >= 0) & (df["sugars_100g"] <= 100) &
    (df["saturated-fat_100g"] >= 0) & (df["saturated-fat_100g"] <= 100) &
    (df["sodium_100g"] >= 0) & (df["sodium_100g"] <= 40) &
    (df["fiber_100g"] >= 0) & (df["fiber_100g"] <= 100) &
    (df["proteins_100g"] >= 0) & (df["proteins_100g"] <= 100)
]
print(f"Clean valid rows ready for training: {len(df):,}")

# 4. Feature Engineering (Nutritional Quality Ratios)
df["sugar_to_fiber_ratio"] = df["sugars_100g"] / (df["fiber_100g"] + 1.0)
df["protein_to_fat_ratio"] = df["proteins_100g"] / (df["saturated-fat_100g"] + 1.0)

ALL_FEATURES = FEATURES + ["sugar_to_fiber_ratio", "protein_to_fat_ratio"]

X = df[ALL_FEATURES]
y = df[TARGET]

# 5. Encode target labels (A, B, C, D, E -> 0, 1, 2, 3, 4)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 6. Stratified Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 7. Train XGBoost Multi-Class Classifier
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 8. Evaluate Model Performance
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 9. Print Feature Importances
print("Feature Importances:")
for feat, imp in sorted(zip(ALL_FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat:<22}: {imp:.4f}")

# 10. Save Trained Model & Label Encoder
model_path = os.path.join(MODEL_DIR, "xgb_nutriscore_model.pkl")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

joblib.dump(model, model_path)
joblib.dump(label_encoder, encoder_path)
print(f"\nModel saved to: {model_path}")
print(f"Encoder saved to: {encoder_path}")
