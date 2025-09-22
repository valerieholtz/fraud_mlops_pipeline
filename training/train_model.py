# training/train_model.py
"""
Train fraud detection model, log metrics & artifacts to MLflow,
and auto-promote the first model to Production if none exists yet.
Uses a two-step registration to avoid YAML serialization issues.
"""

import os
import sys
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient

# === Setup artifact directory ===
ARTIFACTS_DIR = "/workspace/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# === MLflow tracking directory ===
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
if tracking_uri.startswith("file:"):
    path = tracking_uri.replace("file:", "")
    os.makedirs(path, exist_ok=True)

mlflow.set_tracking_uri(tracking_uri)
print(f"[INFO] Using MLflow tracking URI: {tracking_uri}")

if mlflow.active_run():
    mlflow.end_run()

mlflow.autolog(disable=True)
mlflow.xgboost.autolog(disable=True)
mlflow.sklearn.autolog(disable=True)

EXCLUDED_COLUMNS = [
    "nameOrig", "nameDest",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]
TARGET_COLUMN = "isFraud"
EXPERIMENT_NAME = "fraud-detection-v1.4"

mlflow.set_experiment(EXPERIMENT_NAME)

# === Load dataset from SQLite ===
try:
    conn = sqlite3.connect("data/fraud.db")
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
except Exception as e:
    print("[ERROR] Could not load data from SQLite. Did you run init_db.py?")
    print(f"Details: {e}")
    sys.exit(1)

df = df.sample(n=10000, random_state=42)  # subset for test training
df.drop(columns=EXCLUDED_COLUMNS, inplace=True, errors="ignore")

# === Save class distribution plot ===
value_counts = df[TARGET_COLUMN].value_counts().sort_index()
value_counts.plot(kind="bar")
plt.title("Fraud vs Legit distribution")
plt.tight_layout()
plot_path = os.path.join(ARTIFACTS_DIR, "class_distribution.png")
plt.savefig(plot_path)
plt.close()

# === One-hot encode transaction type ===
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
type_encoded = encoder.fit_transform(df[["type"]])
type_df = pd.DataFrame(type_encoded, columns=encoder.get_feature_names_out(["type"]))
df = pd.concat(
    [df.drop(columns=["type"]).reset_index(drop=True), type_df.reset_index(drop=True)],
    axis=1,
)

# === Scale transaction amount ===
#scaler = MinMaxScaler()
#df["amount"] = scaler.fit_transform(df[["amount"]])

# === Train/test split ===
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# === Define Model ===
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    n_estimators=50,
    learning_rate=0.1,
    early_stopping_rounds=10,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)

# === Train, Evaluate, Log ===
with mlflow.start_run() as run:
    run_id = run.info.run_id
    artifact_uri = run.info.artifact_uri

    # Log class distribution plot
    mlflow.log_artifact(plot_path)

    # Train model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("roc_auc", float(auc))
    for cls in ["0", "1"]:
        mlflow.log_metric(f"precision_class_{cls}", float(report[cls]["precision"]))
        mlflow.log_metric(f"recall_class_{cls}", float(report[cls]["recall"]))
        mlflow.log_metric(f"f1_class_{cls}", float(report[cls]["f1-score"]))

    # Log confusion matrix
    cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.csv")
    cm_df.to_csv(cm_path, index=True)
    mlflow.log_artifact(cm_path)

    # Save & log feature names
    feat_path = os.path.join(ARTIFACTS_DIR, "feature_names.txt")
    with open(feat_path, "w") as f:
        for c in X.columns:
            f.write(c + "\n")
    mlflow.log_artifact(feat_path)

    # Log the trained model
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model"
    )

    print(f"[INFO] Training done. Run ID: {run_id}, ROC-AUC={auc:.3f}")

# === Registry + Promotion (two-step) ===
client = MlflowClient()
model_name = "fraud_model"

try:
    client.get_registered_model(model_name)
except Exception:
    client.create_registered_model(model_name)
    print(f"[INFO] Created registered model '{model_name}'")

artifact_uri_for_run = client.get_run(run_id).info.artifact_uri
model_source = f"{artifact_uri_for_run}/model"

mv = client.create_model_version(
    name=model_name,
    source=model_source,
    run_id=run_id,
    description="Registered from training pipeline (two-step registration)."
)

versions = client.search_model_versions(f"name='{model_name}'")
has_production = any(v.current_stage == "Production" for v in versions)

if not has_production:
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production"
    )
    print(f"[INFO] Auto-promoted {model_name} v{mv.version} to Production")
else:
    print(f"[INFO] Registered {model_name} v{mv.version}; kept existing Production model.")

print("[INFO] Done. You can now load with: mlflow.pyfunc.load_model('models:/fraud_model/Production')")
