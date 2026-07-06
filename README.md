---
title: CXSentinel
emoji: 🛡️
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: AI customer sentiment analysis dashboard
---

# CXSentinel — AI Customer Sentiment Analysis

<img src="5_Dashboard/assets/logo.svg" width="72" alt="CXSentinel logo">

**CXSentinel** · *Never Miss a Customer Signal.*

AI Major Capstone Project — **Option 1**. Classifies customer feedback as
Positive / Negative / Neutral with a fine-tuned DistilBERT transformer trained on
**real data** — the `cardiffnlp/tweet_eval` benchmark (45,615 human-labelled
tweets) plus, in the v2 model, real **Amazon and Yelp customer reviews**
(streamed from public datasets) and a labelled sample feedback file. Predictions
are served through a production-grade real-time API, stored in a database, and
visualised in a full product website with login, bulk upload and admin controls.

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

**1b. Train the v2 review model** (optional but recommended; ~40 min). Combines
real tweets with real Amazon (`amazon_polarity`) and Yelp (`yelp_review_full`)
customer reviews so the model handles e-commerce/marketplace feedback
(Amazon/noon/talabat-style text):

```bash
cd 1_Notebook && ../.venv/bin/python train_v2_reviews.py
```

Saves `2_Model/distilbert-sentiment-v2/` (the API automatically prefers v2)
and writes `6_Report/metrics_v2.json`.

**2. Seed the database with real classified feedback** (400 real tweets from the
held-out test split, classified by the trained model):

```bash
cd 4_Database && ../.venv/bin/python seed_real_data.py
```

**3. Start the dashboard** — the inference API starts **automatically**:

```bash
cd 5_Dashboard && ../.venv/bin/python -m streamlit run app.py
```

When the dashboard loads, it auto-starts the FastAPI inference service on port
8000 (if a trained model is present and the API isn't already running) and
keeps it up — so the Analyze features just work, with no separate step. The API
runs in its own process and persists independently.

You can still run the API on its own if you prefer (e.g. to see its logs or
docs at http://127.0.0.1:8000/docs — endpoints `POST /predict`,
`POST /predict/batch`, `GET /feedback`, `GET /stats`, `GET /health`):

```bash
cd 3_API && ../.venv/bin/python -m uvicorn main:app --port 8000
```

Or double-click **`Start CXSentinel.command`** to launch everything at once.

Opens at http://localhost:8501 as a full product website with three tabs:

- **Home Page** — hero, live statistics, feature grid, how-it-works and footer,
  rendered over an animated "Liquid Ether" WebGL background.
- **Analyze Page** (sign-in required) — flexible input: a single line or whole
  paragraph, **or a bulk CSV/TXT upload** (text column auto-detected, up to
  1,000 items per file) with progress, result charts and CSV export.
- **History Page** (sign-in required) — filters, KPIs, sentiment distribution,
  trend-over-time, issue analysis and exportable feedback table.
- **Admin Settings** (owner only) — user management, API/model status, SMTP
  status, master password change, and data reset/restore controls.

**Login & accounts** — a pop-up dialog offers *Continue with Google* or email
sign-up. New email accounts must verify a **6-digit email code**; with SMTP
credentials configured (an `[smtp]` section in `.streamlit/secrets.toml` or
`SMTP_HOST/PORT/USER/PASSWORD` env vars) the code is emailed, otherwise it is
shown on screen in clearly-labelled demo mode. Passwords are PBKDF2-hashed
(200k iterations, per-user salt). The Google button uses Streamlit's native
OIDC (`st.login`) when `[auth]` secrets are configured; otherwise demo SSO.

**Master (owner) login** — seeded on first run from `MASTER_EMAIL` /
`MASTER_PASSWORD` (env vars or a `[master]` secrets section). Defaults live in
`5_Dashboard/auth.py` — **change them before any public deployment.** The owner
account has `role='admin'` and unlocks the Admin Settings tab.

**Supabase authentication (optional)** — add a `[supabase]` section to
`.streamlit/secrets.toml` (or set `SUPABASE_URL` / `SUPABASE_ANON_KEY`):

```toml
[supabase]
url = "https://<project-ref>.supabase.co"
anon_key = "<anon public key>"
```

With these set, email sign-up/sign-in is verified through **Supabase Auth**
(Supabase sends the confirmation email); without them the app uses its local
PBKDF2 accounts with the built-in code verification. Sign-up collects first and
last name either way.

**Analysis extras** — bulk uploads are capped at 10 MB / 1,000 items (shown in
the UI); an **AI Confidence Weighting** slider sets the minimum model
confidence for automated handling (items below it are flagged for human
review, recalculating the KPIs, insights and urgent queue live); and the
report's **Download Report** tab generates a professional multi-page PDF
(figures, tables, insights, recommendations, methodology) via fpdf2 +
matplotlib.

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
Analyze Page needs the inference API, which runs locally or via Docker (the
model weights are too large for GitHub); on the hosted demo it shows a friendly
error instead.

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
