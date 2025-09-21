 # 🚀 Fraud Detection MLOps Pipeline

This project demonstrates an **end-to-end MLOps workflow** for fraud detection using **XGBoost**, **MLflow**, **FastAPI**, and **Docker**.  
It includes **data ingestion, training, experiment tracking, model registry, model promotion, and serving**.

---

## 📂 Project Structure

fraud_mlops/
│── app/
│ ├── app.py # FastAPI app (serves model from MLflow)
│ ├── feature_names.txt # Saved feature names from training
│
│── training/
│ ├── train_model.py # Model training & logging
│
│── ci/
│ ├── promote_if_better.py # CI script: promote new model if better
│
│── data/
│ ├── fraud.db # SQLite database with fraud transactions
│
│── mlruns/ # MLflow experiment & registry store
│
│── .env # Environment variables (API_KEY, PORT, MLFLOW_TRACKING_URI)
│── requirements.txt # Python dependencies
│── Dockerfile # Container image for all services
│── docker-compose.yml # Orchestration of API, training, MLflow UI
│── Makefile # Shortcuts (make up, make train, make promote, ...)

markdown
Code kopieren

---

## 🔑 Features

- **SQLite** for persistent fraud data (`data/fraud.db`)
- **MLflow** for experiment tracking, model logging, registry, and UI
- **XGBoost** binary classification model
- **Automatic promotion** if a new model outperforms Production
- **FastAPI REST API** with API key authentication
- **Docker Compose** for reproducible services:
  - `mlflow-ui` → experiment tracking dashboard
  - `api` → model serving
  - `trainer` → training job
  - `promote` → promotion job

---

## 📊 Architecture

```mermaid
flowchart LR
    A[SQLite DB: fraud.db] --> B[Training Script train_model.py]
    B -->|Logs metrics, artifacts| C[MLflow Tracking Store mlruns/]
    B -->|Registers model| D[MLflow Model Registry]
    E[Promote Script promote_if_better.py] --> D
    D -->|Production model| F[FastAPI App app.py]
    F -->|REST API /predict| G[Client / Swagger UI]
⚙️ Setup
1. Clone the repository
bash
Code kopieren
git clone https://github.com/<your-repo>/fraud_mlops.git
cd fraud_mlops
2. Create .env file
In project root:

env
Code kopieren
API_KEY=fdtge784h
PORT=8000
MLFLOW_TRACKING_URI=file:///workspace/mlruns
3. Build Docker images
bash
Code kopieren
docker compose build
▶️ Usage
Start API + MLflow UI
bash
Code kopieren
docker compose up -d mlflow-ui api
API: http://127.0.0.1:8000/docs

MLflow: http://127.0.0.1:5000

Train a model
bash
Code kopieren
docker compose run --rm trainer
Logs metrics/artifacts to MLflow and registers the model.

Promote best model
bash
Code kopieren
docker compose run --rm promote
Compares new model with current Production and promotes if better.

Test the API
bash
Code kopieren
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -H 'x-api-key: fdtge784h' \
  -d '{"amount": 0.4, "step": 50, "TRANSFER": 1, "PAYMENT": 0, "CASH_OUT": 0, "DEBIT": 0}'
Expected response:


json
Code kopieren
{"prediction":1.0}

🔧 Developer Shortcuts (Makefile)
Instead of typing long Docker commands, use:

bash
Code kopieren
make up        # start API + MLflow UI
make down      # stop everything
make logs      # tail logs
make train     # run training
make promote   # run promotion
🛠️ Next Steps
Monitoring: Add Prometheus + Grafana to monitor API latency & drift

CI/CD: Integrate Jenkins (train → promote → deploy pipeline)

Cloud: Deploy containers to AWS ECS, GCP Cloud Run, or Kubernetes