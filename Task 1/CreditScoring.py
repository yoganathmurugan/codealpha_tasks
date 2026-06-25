# -----------------------------
# 1. Import required libraries
# -----------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ---------------------------------------------------------
# 2. Create output folder
# ---------------------------------------------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 3. Load dataset from CSV
# ---------------------------------------------------------
file_path = "DataSet/dataset.csv"
df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------------------------------------
# 4. Set the target column
# ---------------------------------------------------------
df.columns = df.columns.str.strip()
target_column = "CreditScore"

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# ---------------------------------------------------------
# 5. Convert target labels to numeric if needed
# ---------------------------------------------------------
label_encoder = None

if df[target_column].dtype == "object":
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])
    class_names = label_encoder.classes_
else:
    class_names = sorted(df[target_column].dropna().unique())
    class_names = [str(c) for c in class_names]

# ---------------------------------------------------------
# 6. Feature engineering
# ---------------------------------------------------------
if "debt" in df.columns and "income" in df.columns:
    df["debt_to_income_ratio"] = df["debt"] / df["income"]

if "loan_amount" in df.columns and "income" in df.columns:
    df["loan_to_income_ratio"] = df["loan_amount"] / df["income"]

if "debt" in df.columns and "loan_amount" in df.columns and "income" in df.columns:
    df["financial_burden"] = (df["debt"] + df["loan_amount"]) / df["income"]

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ---------------------------------------------------------
# 7. Split features and target
# ---------------------------------------------------------
X = df.drop(columns=[target_column])
y = df[target_column]

# ---------------------------------------------------------
# 8. Identify numeric and categorical columns
# ---------------------------------------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric columns:")
print(numeric_features)

print("\nCategorical columns:")
print(categorical_features)

# ---------------------------------------------------------
# 9. Preprocessing pipelines
# ---------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ---------------------------------------------------------
# 10. Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 11. Build model pipeline
# ---------------------------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# ---------------------------------------------------------
# 12. Train the model
# ---------------------------------------------------------
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 13. Predict on test set
# ---------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# ---------------------------------------------------------
# 14. Evaluate the model
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

# ROC-AUC handling
try:
    if len(np.unique(y)) == 2:
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
except Exception:
    roc_auc = np.nan

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

if not np.isnan(roc_auc):
    print(f"ROC-AUC   : {roc_auc:.4f}")
else:
    print("ROC-AUC   : Not available")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------------------------------
# 15. Save evaluation metrics to CSV in output folder
# ---------------------------------------------------------
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Value": [accuracy, precision, recall, f1, roc_auc]
})

results_path = os.path.join(OUTPUT_DIR, "credit_scoring_results.csv")
results_df.to_csv(results_path, index=False)

print(f"\nSaved results to {results_path}")

# ---------------------------------------------------------
# 16. Confusion Matrix chart
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

confusion_matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(confusion_matrix_path)
plt.show()

print(f"Saved confusion matrix to {confusion_matrix_path}")

# ---------------------------------------------------------
# 17. ROC Curve chart
# ---------------------------------------------------------
roc_curve_path = os.path.join(OUTPUT_DIR, "roc_curve.png")

plt.figure(figsize=(7, 5))

try:
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    else:
        for i, class_label in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{class_label}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_curve_path)
    plt.show()

    print(f"Saved ROC curve to {roc_curve_path}")

except Exception as e:
    print(f"ROC curve could not be created: {e}")

# ---------------------------------------------------------
# 18. Feature importance
# ---------------------------------------------------------
feature_names = []

if len(categorical_features) > 0:
    onehot = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    encoded_cat_features = onehot.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(encoded_cat_features)
else:
    feature_names = numeric_features

importances = model.named_steps["classifier"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))

# Save feature importance CSV
feature_importance_csv_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
feature_importance_df.to_csv(feature_importance_csv_path, index=False)

print(f"Saved feature importance data to {feature_importance_csv_path}")

# Save feature importance chart
feature_importance_path = os.path.join(OUTPUT_DIR, "feature_importance.png")

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance_df.head(10),
    x="Importance",
    y="Feature",
    palette="viridis"
)
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(feature_importance_path)
plt.show()

print(f"Saved feature importance chart to {feature_importance_path}")

# ---------------------------------------------------------
# 19. Example prediction for one new customer
# ---------------------------------------------------------
sample_data = {}

for col in X.columns:
    if col in numeric_features:
        sample_data[col] = X[col].median()
    else:
        sample_data[col] = X[col].mode()[0]

new_customer = pd.DataFrame([sample_data])

new_prediction = model.predict(new_customer)[0]
new_probability = model.predict_proba(new_customer)[0]

print("\n===== NEW CUSTOMER PREDICTION =====")

if label_encoder is not None:
    predicted_class = label_encoder.inverse_transform([new_prediction])[0]
else:
    predicted_class = new_prediction

print("Predicted class:", predicted_class)
print("Class probabilities:", dict(zip(class_names, new_probability)))

# ---------------------------------------------------------
# 20. Save new customer prediction to CSV
# ---------------------------------------------------------
prediction_df = pd.DataFrame({
    "Class": class_names,
    "Probability": new_probability
})

prediction_path = os.path.join(OUTPUT_DIR, "new_customer_prediction.csv")
prediction_df.to_csv(prediction_path, index=False)

print(f"Saved new customer prediction to {prediction_path}")

print("\n===== ALL OUTPUT FILES SAVED INSIDE output/ FOLDER =====")