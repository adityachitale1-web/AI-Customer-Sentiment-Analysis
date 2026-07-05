"""Populate feedback.db with REAL customer feedback: a random sample of tweets
from the cardiffnlp/tweet_eval test split, classified by the trained model.

This replaces any previous demo rows so the dashboard shows genuine data
flowing through the actual model.

Usage (from 4_Database/):  ../.venv/bin/python seed_real_data.py [n_samples]
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from datasets import load_dataset
from transformers import pipeline

from db import DB_PATH, delete_all, init_db, insert_feedback, sentiment_counts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "2_Model" / "distilbert-sentiment"
FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABEL_MAP = {
    "negative": "Negative", "neutral": "Neutral", "positive": "Positive",
    "label_0": "Negative", "label_1": "Neutral", "label_2": "Positive",
}

N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 400
random.seed(42)


def main() -> None:
    init_db()

    print("Loading real feedback texts (cardiffnlp/tweet_eval test split)…")
    test_split = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")
    texts = random.sample(list(test_split["text"]), min(N_SAMPLES, len(test_split)))

    if MODEL_DIR.exists():
        print(f"Classifying with fine-tuned model: {MODEL_DIR}")
        classifier = pipeline("text-classification", model=str(MODEL_DIR))
    else:
        print(f"Fine-tuned model not found, using fallback: {FALLBACK_MODEL}")
        classifier = pipeline("text-classification", model=FALLBACK_MODEL)

    print(f"Running inference on {len(texts)} real texts…")
    results = classifier(texts, truncation=True, max_length=128, batch_size=32)

    removed = delete_all()
    if removed:
        print(f"Removed {removed} previous demo rows.")

    now = datetime.now()
    for text, result in zip(texts, results):
        sentiment = LABEL_MAP.get(result["label"].lower())
        if sentiment is None:
            continue
        # Spread entries over the past 45 days so the trend chart has shape
        created = (now - timedelta(days=random.uniform(0, 45))).isoformat(timespec="seconds")
        insert_feedback(text, sentiment, result["score"],
                        source="twitter", created_at=created)

    print(f"Inserted {len(texts)} real feedback rows into {DB_PATH}")
    print("Counts by sentiment:", sentiment_counts())


if __name__ == "__main__":
    main()
