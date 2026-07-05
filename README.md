# SentiLens — AI Customer Sentiment Analysis

<img src="5_Dashboard/assets/logo.svg" width="72" alt="SentiLens logo">

**SentiLens** · *See what your customers feel.*

AI Major Capstone Project — **Option 1**. Classifies customer feedback as
Positive / Negative / Neutral with a fine-tuned DistilBERT transformer trained on
**real data** (the `cardiffnlp/tweet_eval` sentiment benchmark — 45,615 real
tweets for training, 12,284 held-out for testing), serves predictions through a
production-grade real-time API, stores everything in a database, and visualises
trends in an interactive dashboard.

## Folder structure — one folder per deliverable

| Folder | Deliverable | Core requirement |
|---|---|---|
| `1_Notebook/` | Jupyter notebook with executed outputs: preprocessing, EDA, baseline, DistilBERT training, evaluation | NLP preprocessing + transformer model |
| `2_Model/` | Fine-tuned model weights (written by the notebook) | — |
| `3_API/` | FastAPI real-time inference service + test suite | Real-time API inference |
| `4_Database/` | SQLite layer (WAL mode, indexed) + real-data seeder | Database integration |
| `5_Dashboard/` | Streamlit dashboard | Trend & issue analysis dashboard |
| `6_Report/` | Final analytical report + metrics + figures | Analytical report |

## Architecture

```
1_Notebook (train + evaluate) ──► 2_Model (saved weights)
                                        │
             POST /predict              ▼
Customer ──► 3_API (FastAPI) ──► 4_Database (SQLite) ──► 5_Dashboard (Streamlit)
```

## Setup (once)

Uses [uv](https://docs.astral.sh/uv/) with Python 3.11 (the system Python 3.9 is
end-of-life and too old for the ML stack):

```bash
cd "Sentiment Analysis Project"
uv venv --python 3.11 .venv
uv pip install -p .venv/bin/python -r requirements-full.txt
```

(The root `requirements.txt` is intentionally slim — it is what Streamlit
Community Cloud installs to host the dashboard.)

## How to run

**1. Train the model** (one-off; ~30 min on Apple Silicon / GPU). Either open the
notebook interactively or execute it headless — outputs are saved into the notebook:

```bash
cd 1_Notebook
../.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    sentiment_model_training.ipynb --ExecutePreprocessor.timeout=-1
```

This saves the model to `2_Model/distilbert-sentiment/` and writes
`6_Report/metrics.json` + confusion-matrix figures for the report.

**2. Seed the database with real classified feedback** (400 real tweets from the
held-out test split, classified by the trained model):

```bash
cd 4_Database && ../.venv/bin/python seed_real_data.py
```

**3. Start the API** (terminal 1):

```bash
cd 3_API && ../.venv/bin/python -m uvicorn main:app --port 8000
```

Interactive API docs: http://127.0.0.1:8000/docs — endpoints: `POST /predict`,
`POST /predict/batch`, `GET /feedback` (paginated), `GET /stats`, `GET /health`.
(Works even before training — it falls back to a public pretrained model.)

**4. Start the dashboard** (terminal 2):

```bash
cd 5_Dashboard && ../.venv/bin/python -m streamlit run app.py
```

Opens at http://localhost:8501 as a full product website:

- **Landing page** — hero, live statistics, feature grid, how-it-works and footer,
  rendered over an animated "Liquid Ether" WebGL background.
- **Login** — a pop-up dialog offering *Continue with Google* or email
  sign-in / account creation. Accounts are stored in the project database with
  PBKDF2-hashed passwords (200k iterations, per-user salt). The Google button
  uses Streamlit's native OIDC (`st.login`) when `[auth]` secrets are configured
  (Google Cloud OAuth client); otherwise it runs as a clearly-labelled demo SSO.
- **Dashboard** (after sign-in) — type feedback into the "Analyse new feedback"
  box; it calls the API live, stores the prediction, and the charts update.

## Tests

```bash
cd 3_API && ../.venv/bin/python -m pytest tests/ -v
```

8 tests cover health checks, all three sentiment classes, input validation,
batch inference, pagination, and stats — using a dummy model and a temporary
database so they run in ~1 second.

## Hosted demo (Streamlit Community Cloud)

The dashboard deploys straight from this GitHub repository:

1. Go to https://share.streamlit.io → **Create app** → pick this repo.
2. Set **Main file path** to `5_Dashboard/app.py` and deploy.

On first load the app bootstraps its database from
`4_Database/sample_feedback.csv` — a committed snapshot of 400+ real tweets
classified by the fine-tuned model — so all charts work immediately. The
"Analyse new feedback" box needs the inference API, which runs locally or via
Docker (the model weights are too large for GitHub); on the hosted demo it
shows a friendly error instead.

## Production deployment (Docker)

```bash
docker compose up --build
```

Builds two services (API on :8000, dashboard on :8501) sharing a persistent
volume for the database. Production hardening included: health checks, WAL-mode
SQLite with indexes, structured logging, CORS configuration, input validation,
global error handling, and environment-variable configuration
(`SENTIMENT_DB_PATH`, `SENTIMENT_MODEL_DIR`, `SENTIMENT_CORS_ORIGINS`).

For a real production environment, the documented next steps are PostgreSQL
instead of SQLite, API authentication, and HTTPS termination — see section 8 of
the report.
