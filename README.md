#  MLOps Drift Monitor

An end-to-end MLOps pipeline that monitors a churn prediction model in production, detects data drift automatically, and retrains itself when needed — with zero manual intervention.

** Live Dashboard:** *(add your Streamlit Cloud link here after deployment)*

---

## The Problem

Machine learning models degrade silently in production. Customer behavior shifts, new patterns emerge, and a model trained on last year's data slowly becomes less accurate — often without anyone noticing until it's too late.

This project builds a **self-monitoring system** that watches for that shift and reacts to it automatically.

## What It Does

1. Trains a churn prediction model on customer data
2. Simulates incoming "production" data batches over time, with deliberate feature drift
3. Uses **Evidently** to statistically detect when incoming data diverges from the training data
4. Visualizes drift trends on a live **Streamlit** dashboard
5. Runs a scheduled **GitHub Actions** workflow that checks for drift daily
6. **Automatically retrains and redeploys the model** when drift crosses a threshold — no human required

## Tech Stack

| Layer | Tool |
|---|---|
| Model | Scikit-learn (Logistic Regression) |
| Drift Detection | Evidently |
| Dashboard | Streamlit |
| Containerization | Docker |
| CI/CD & Automation | GitHub Actions |
| Data | [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

## Architecture

```
Reference Data → Train Model → Simulate Incoming Batches
                                      ↓
                              Evidently Drift Check
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                     ↓
             Drift Below Threshold              Drift Above Threshold
                    ↓                                     ↓
              Log & Continue                    Auto-Retrain Model
                                                          ↓
                                                  Commit New Model
                                                          ↓
                                              (all visualized on Dashboard)
```

## Project Structure

```
mlops-drift-monitor/
│
├── data/
│   ├── reference/          # training data snapshot
│   ├── incoming/           # simulated production batches
│   └── monitoring/         # drift summary output (JSON)
│
├── src/
│   ├── train.py             # trains and saves the churn model
│   ├── simulate_drift.py    # generates shifting "incoming" batches
│   ├── monitor.py           # runs Evidently drift detection
│   └── model/                # saved model, scaler, encoders
│
├── dashboard/
│   └── app.py                # Streamlit dashboard
│
├── .github/workflows/
│   ├── ci.yml                # runs tests on every push
│   └── retrain.yml           # daily drift check + auto-retrain
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone https://github.com/mahakkhan16/mlops-drift-monitor.git
cd mlops-drift-monitor
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

python src/train.py            # train the model
python src/simulate_drift.py   # generate simulated production batches
python src/monitor.py          # run drift detection
streamlit run dashboard/app.py # launch the dashboard
```

Then visit `http://localhost:8501`

## Running with Docker

```bash
docker build -t mlops-drift-monitor .
docker run -p 8501:8501 mlops-drift-monitor
```

Then visit `http://localhost:8501`

## CI/CD Pipeline

- **`ci.yml`** — runs on every push; re-runs training and monitoring scripts in a clean environment to catch breakages early
- **`retrain.yml`** — runs daily (and can also be triggered manually from the Actions tab); checks the latest drift share and automatically retrains + commits a new model if drift exceeds 15%

## How Drift Detection Works

Each incoming batch is compared against the original training data using Evidently's statistical drift tests (Wasserstein distance for numerical features, Jensen-Shannon distance for categorical features). If the share of drifted columns crosses a set threshold, the pipeline flags it and the retrain workflow kicks in automatically.

## What I Learned

Building this taught me that the hard part of ML isn't training a model — it's *keeping it correct over time*. Implementing drift detection and automated retraining gave me hands-on experience with the MLOps practices that separate a notebook experiment from a production system: monitoring, automation, and self-healing pipelines. I also worked through real infrastructure issues along the way — Docker networking errors, dependency conflicts, and CI permission settings — which mirrors the kind of debugging real ML engineering work involves.


Built by **Mahak Khan** — [LinkedIn](https://linkedin.com/in/mahak-khan-50ba99270)
