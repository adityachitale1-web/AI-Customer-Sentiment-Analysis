"""In-process sentiment engine — the dashboard's built-in fallback.

Used whenever the FastAPI service isn't reachable (most importantly the hosted
Streamlit Cloud demo, which can't run a separate API process with the local
model weights). Mirrors the API's behaviour and response shape exactly:
same text cleaning, same label mapping, same database writes.

Model preference: the fine-tuned DistilBERT (v2 then v1) when the weights are
on disk; otherwise a public 3-class sentiment model downloaded from the
Hugging Face Hub (no credentials required).
"""

import re
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "4_Database"))
import db  # noqa: E402

FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABEL_MAP = {
    "negative": "Negative", "neutral": "Neutral", "positive": "Positive",
    "label_0": "Negative", "label_1": "Neutral", "label_2": "Positive",
}


def clean_text(text: str) -> str:
    # Same normalisation used at training time and in the API
    text = re.sub(r"https?://\S+|www\.\S+", "http", str(text))
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


def available() -> bool:
    """True when the transformers stack is installed (it is, on both the local
    env and the hosted deployment via requirements.txt)."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_resource(show_spinner="Loading The Built-In Analysis Model — First "
                                "Run Can Take A Minute…")
def _pipeline():
    from transformers import pipeline
    candidates = [
        (PROJECT_ROOT / "2_Model" / "distilbert-sentiment-v2",
         "distilbert-finetuned-v2 (built-in)"),
        (PROJECT_ROOT / "2_Model" / "distilbert-sentiment",
         "distilbert-finetuned (built-in)"),
    ]
    for model_dir, name in candidates:
        if model_dir.exists():
            return pipeline("text-classification", model=str(model_dir)), name
    return (pipeline("text-classification", model=FALLBACK_MODEL),
            "cardiffnlp-roberta (built-in)")


def predict_many(texts: list, source: str = "dashboard",
                 persist: bool = True) -> list:
    """Classify texts in-process. Returns dicts shaped like the API's
    PredictionOut (id, text, sentiment, confidence, scores, model, latency_ms)."""
    classifier, model_name = _pipeline()
    if persist:
        db.init_db()
    started = time.perf_counter()
    raw = classifier([clean_text(t) for t in texts], truncation=True,
                     max_length=128, top_k=None, batch_size=16)
    latency_ms = round((time.perf_counter() - started) * 1000 / max(len(texts), 1), 1)
    results = []
    for text, score_list in zip(texts, raw):
        scores = {}
        for item in score_list:
            mapped = LABEL_MAP.get(item["label"].lower())
            if mapped:
                scores[mapped] = round(float(item["score"]), 4)
        if not scores:
            continue
        sentiment = max(scores, key=scores.get)
        confidence = scores[sentiment]
        row_id = (db.insert_feedback(text, sentiment, confidence, source=source)
                  if persist else 0)
        results.append({"id": row_id, "text": text, "sentiment": sentiment,
                        "confidence": confidence, "scores": scores,
                        "model": model_name, "latency_ms": latency_ms})
    return results
