import mlflow
from mlflow.tracking import MlflowClient
import os

# Use env var or fallback to local ./mlruns
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
print(f"[INFO] Using MLflow tracking URI: {tracking_uri}")

MODEL_NAME = "fraud_model"
client = MlflowClient()

# Get all versions of the model from registry
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise RuntimeError(f"No versions found for model {MODEL_NAME}")

# Sort by version number (latest first)
versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
latest = versions[0]

# Get metrics from the run that created this version
latest_run = client.get_run(latest.run_id)
new_auc = latest_run.data.metrics.get("roc_auc", 0.0)
print(f"Latest registered version v{latest.version} from run {latest.run_id}, ROC-AUC={new_auc}")

# Check current Production model
prod_versions = [v for v in versions if v.current_stage == "Production"]

if not prod_versions:
    print("No Production model yet → promoting latest model.")
    client.transition_model_version_stage(MODEL_NAME, latest.version, "Production")
else:
    prod = prod_versions[0]
    prod_run = client.get_run(prod.run_id)
    prod_auc = prod_run.data.metrics.get("roc_auc", 0.0)

    if new_auc > prod_auc:
        print(f"New model better ({new_auc:.3f} > {prod_auc:.3f}) → promoting.")
        client.transition_model_version_stage(MODEL_NAME, latest.version, "Production")
    else:
        print(f"Keeping current Production (new={new_auc:.3f}, old={prod_auc:.3f}).")
