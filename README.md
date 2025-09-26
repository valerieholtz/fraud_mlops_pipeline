# Fraud Detection MLOps Pipeline

This project demonstrates an end-to-end MLOps workflow for fraud detection using **XGBoost**, **MLflow**, **FastAPI**, **Docker**, and **GitHub Actions**.  
It includes data ingestion, training, experiment tracking, model registry, model promotion, serving, and automated retraining.

---

##  Project Structure

```
fraud_mlops_pipeline/
│── app/
│   ├── app.py                # FastAPI app (serves model from MLflow)
│   ├── feature_names.txt     # Saved feature names from training
│
│── training/
│   ├── train_model.py        # Model training & logging
│
│── ci/
│   ├── promote_if_better.py  # CI script: promote new model if better
│
│── data/
│   ├── fraud_transactions.csv    # Reduced dataset for testing/CI
│   ├── fraud.db                  # SQLite DB created by init_db.py
│   ├── generate_monthly_data.py  # Script to simulate 12 month of data for retraining
│
│── mlruns/                   # MLflow experiment & registry store
│── .env.example              # Example environment variables
│── requirements.txt          # Python dependencies
│── Dockerfile                # Container image for all services
│── docker-compose.yml        # Orchestration of API, training, MLflow UI
│── Makefile                  # Shortcuts (make up, make train, make promote, ...)
│── .github/workflows/mlops.yml # GitHub Actions workflow (automation)
```

---

##  Features

- SQLite for persistent fraud data (`data/fraud.db`)
- MLflow for experiment tracking, model logging, registry, and UI
- XGBoost binary classification model
- Automatic promotion if a new model outperforms Production
- FastAPI REST API with API key authentication
- Docker Compose for reproducible services:
  - **mlflow-ui** → experiment tracking dashboard  
  - **api** → model serving  
  - **trainer** → training job  
  - **promote** → promotion job
- GitHub Actions for CI/CD automation:
  - Scheduled retraining (monthly via cron)  
  - Model promotion  
  - Artifacts upload  

---

##  Purpose of `deploy_local_testing` Branch

The `deploy_local_testing` branch is an **experimental extension** of the pipeline that goes beyond local deployment.  
Here, the goal is to **simulate one full year of data drift and monthly retraining**, ensuring that the MLOps workflow can:  

- Handle imbalanced and evolving datasets across time.  
- Persist artifacts and model registry state between GitHub Actions jobs.  
- Automatically restart the API container and validate it with health checks and prediction probes.  
- Resolve schema mismatches by enforcing a consistent feature set (e.g., always including all transaction types).  
- Debug and harden the workflow with retries, artifact cleanup, and fallback strategies.  

This branch demonstrates that the pipeline is **robust against real-world challenges** like schema drift, registry persistence, and CI/CD timing issues.  
It validates that the fraud detection API remains accessible and up to date after every retraining cycle.  

---

##  Architecture

```mermaid
flowchart LR
    A[SQLite DB: fraud.db] --> B[Training Script train_model.py]
    B -->|Logs metrics, artifacts| C[MLflow Tracking Store mlruns/]
    B -->|Registers model| D[MLflow Model Registry]
    E[Promote Script promote_if_better.py] --> D
    D -->|Production model| F[FastAPI App app.py]
    F -->|REST API /predict| G[Client / Swagger UI]
    H[GitHub Actions Workflow] --> B
    H --> E
```

---

##  Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/fraud_mlops_pipeline.git
   cd fraud_mlops_pipeline
   ```

2. **Create `.env` file (based on `.env.example`)**
   ```env
   API_KEY=...
   PORT=8000
   MLFLOW_TRACKING_URI=file:///workspace/mlruns
   ```

3. **Build Docker images**
   ```bash
   docker compose build
   ```

---

##  Usage

### Start API + MLflow UI
```bash
docker compose up -d mlflow-ui api
```
- **API:** http://127.0.0.1:8000/docs  
- **MLflow:** http://127.0.0.1:5000  

### Train a model
```bash
docker compose run --rm trainer
```
Logs metrics/artifacts to MLflow and registers the model.

### Promote best model
```bash
docker compose run --rm promote
```
Compares new model with current Production and promotes if better.

### Test the API
```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -H 'x-api-key: fdtge784h'   -d '{"amount": 0.4, "step": 50, "TRANSFER": 1, "PAYMENT": 0, "CASH_OUT": 0, "DEBIT": 0}'
```

**Expected response:**
```json
{"prediction": 1.0}
```

---

##  Automation with GitHub Actions

The project includes a workflow in `.github/workflows/mlops.yml` that automates:

- Running the training pipeline (`train_model.py`)
- Running the promotion logic (`promote_if_better.py`)
- Scheduling retraining automatically once per month (cron job)
- Allowing manual retraining via the **Actions** tab in GitHub
- (Branch-specific) Simulating one year of drifted data, retraining monthly, and validating API health

This replaces the need for a heavy local Jenkins setup and runs entirely in GitHub’s cloud infrastructure.

---

##  Developer Shortcuts (Makefile)

Instead of typing long Docker commands, use:
```bash
make up        # start API + MLflow UI
make down      # stop everything
make logs      # tail logs
make train     # run training
make promote   # run promotion
```

---

##  Next Steps

- **Monitoring:** Add Prometheus + Grafana to monitor API latency & drift  
- **CI/CD:** Extend GitHub Actions to build and push Docker images to a registry  
- **Cloud:** Deploy containers to AWS ECS, GCP Cloud Run, or Kubernetes  
