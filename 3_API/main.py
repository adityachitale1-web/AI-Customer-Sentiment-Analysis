"""Real-time sentiment inference API (core requirement 3).

Run from this folder:
    uvicorn main:app --port 8000

Interactive docs at http://127.0.0.1:8000/docs

Model loading order:
  1. The fine-tuned DistilBERT saved by the notebook to 2_Model/distilbert-sentiment/
  2. Fallback: a public pretrained 3-class sentiment model (cardiffnlp RoBERTa),
     so the API and dashboard work end-to-end even before training finishes.

Every prediction is stored in the SQLite database (4_Database/feedback.db).

Configuration (env vars):
  SENTIMENT_MODEL_DIR   path to a fine-tuned model directory (default: 2_Model/distilbert-sentiment)
  SENTIMENT_DB_PATH     path to the SQLite file (default: 4_Database/feedback.db)
  SENTIMENT_CORS_ORIGINS comma-separated allowed origins (default: * for the demo)
  SENTIMENT_TEST_MODE   "1" loads a lightweight dummy model (used by the test suite)
"""

import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "4_Database"))
import db  # noqa: E402  (4_Database/db.py)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sentiment-api")

FINE_TUNED_DIR = Path(os.getenv("SENTIMENT_MODEL_DIR",
                                PROJECT_ROOT / "2_Model" / "distilbert-sentiment"))
FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Normalise whatever label scheme the loaded model uses to our three classes
LABEL_MAP = {
    "negative": "Negative", "neutral": "Neutral", "positive": "Positive",
    "label_0": "Negative", "label_1": "Neutral", "label_2": "Positive",
}

state: dict = {}


class _DummyClassifier:
    """Deterministic stand-in used by the test suite (SENTIMENT_TEST_MODE=1)."""

    def __call__(self, texts, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else texts
        results = []
        for text in items:
            lowered = text.lower()
            if any(w in lowered for w in ("love", "great", "excellent", "amazing")):
                results.append({"label": "positive", "score": 0.98})
            elif any(w in lowered for w in ("hate", "terrible", "awful", "broken")):
                results.append({"label": "negative", "score": 0.97})
            else:
                results.append({"label": "neutral", "score": 0.90})
        return results


def load_classifier():
    if os.getenv("SENTIMENT_TEST_MODE") == "1":
        logger.info("Test mode: using dummy classifier")
        return _DummyClassifier(), "dummy (test mode)"
    from transformers import pipeline  # deferred: heavy import
    if FINE_TUNED_DIR.exists():
        logger.info("Loading fine-tuned model from %s", FINE_TUNED_DIR)
        return (pipeline("text-classification", model=str(FINE_TUNED_DIR)),
                "distilbert-finetuned")
    logger.warning("Fine-tuned model not found at %s — using fallback %s",
                   FINE_TUNED_DIR, FALLBACK_MODEL)
    return (pipeline("text-classification", model=FALLBACK_MODEL),
            "cardiffnlp-roberta (fallback)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    classifier, model_name = load_classifier()
    state["classifier"] = classifier
    state["model_name"] = model_name
    state["started_at"] = time.time()
    logger.info("API ready — model: %s, database: %s", model_name, db.DB_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Customer Sentiment Analysis API",
    description="Classifies customer feedback as Positive / Negative / Neutral "
                "and stores every prediction in the project database.",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = os.getenv("SENTIMENT_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": "Internal server error"})


# ---------------------------------------------------------------- schemas

class FeedbackIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      examples=["The delivery was fast but the box arrived damaged."])
    source: str = Field(default="api", max_length=50, examples=["web_form"])


class BatchIn(BaseModel):
    items: List[FeedbackIn] = Field(..., min_length=1, max_length=100)


class PredictionOut(BaseModel):
    id: int
    text: str
    sentiment: str
    confidence: float
    model: str
    latency_ms: float


# ---------------------------------------------------------------- helpers

def clean_text(text: str) -> str:
    # Same normalisation used at training time (see 1_Notebook)
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_and_store(feedback: FeedbackIn) -> PredictionOut:
    classifier = state.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    started = time.perf_counter()
    result = classifier(clean_text(feedback.text), truncation=True, max_length=128)[0]
    latency_ms = (time.perf_counter() - started) * 1000
    sentiment = LABEL_MAP.get(result["label"].lower())
    if sentiment is None:
        logger.error("Unexpected model label: %s", result["label"])
        raise HTTPException(status_code=500, detail="Unexpected model output")
    row_id = db.insert_feedback(feedback.text, sentiment, result["score"],
                                source=feedback.source)
    logger.info("predict id=%s sentiment=%s conf=%.3f latency=%.0fms source=%s",
                row_id, sentiment, result["score"], latency_ms, feedback.source)
    return PredictionOut(id=row_id, text=feedback.text, sentiment=sentiment,
                         confidence=round(result["score"], 4),
                         model=state["model_name"], latency_ms=round(latency_ms, 1))


# ---------------------------------------------------------------- endpoints

@app.get("/", tags=["meta"])
def root():
    return {"status": "ok", "model": state.get("model_name"), "docs": "/docs"}


@app.get("/health", tags=["meta"])
def health():
    try:
        total = db.total_count()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
    return {
        "status": "healthy",
        "model": state.get("model_name"),
        "database": str(db.DB_PATH),
        "stored_feedback": total,
        "uptime_seconds": round(time.time() - state.get("started_at", time.time()), 1),
    }


@app.post("/predict", response_model=PredictionOut, tags=["inference"])
def predict(feedback: FeedbackIn):
    return classify_and_store(feedback)


@app.post("/predict/batch", response_model=List[PredictionOut], tags=["inference"])
def predict_batch(batch: BatchIn):
    return [classify_and_store(item) for item in batch.items]


@app.get("/feedback", tags=["data"])
def list_feedback(limit: int = Query(default=100, ge=1, le=1000),
                  offset: int = Query(default=0, ge=0)):
    return {"total": db.total_count(),
            "items": db.fetch_feedback(limit=limit, offset=offset)}


@app.get("/stats", tags=["data"])
def stats():
    counts = db.sentiment_counts()
    return {"total": sum(counts.values()), "counts": counts,
            "daily": db.daily_counts(), "model": state.get("model_name")}
