"""CXSentinel model v2 — retrain on real e-commerce/review feedback.

Combines three real data sources so the model handles review-style text
(Amazon/noon/talabat-like) as well as short social feedback:

  1. cardiffnlp/tweet_eval sentiment (real tweets, human-labelled 3-class)
  2. yelp_review_full  (real Yelp reviews; stars 1-2=Negative, 3=Neutral, 4-5=Positive)
  3. amazon_polarity   (real Amazon reviews; Negative/Positive)
  4. the project's sample_uploads CSV (labelled customer feedback)

Saves to 2_Model/distilbert-sentiment-v2 (the API prefers v2 over v1) and
writes 6_Report/metrics_v2.json.

Run:  cd 1_Notebook && ../.venv/bin/python train_v2_reviews.py
"""

import csv
import io
import json
import re
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "2_Model" / "distilbert-sentiment-v2"
METRICS_PATH = PROJECT_ROOT / "6_Report" / "metrics_v2.json"
UPLOAD_CSV = PROJECT_ROOT / "4_Database" / "sample_uploads" / "sentiment-analysis.csv"

RANDOM_STATE = 42
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
MAX_LENGTH = 160
TWEETS_N = 12_000
YELP_PER_CLASS = 4_000     # 12k total
AMAZON_PER_CLASS = 2_000   # 4k total (binary: Negative/Positive)

rng = np.random.default_rng(RANDOM_STATE)


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "http", str(text))
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------ sources

def load_tweets():
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    train = ds["train"].to_pandas().sample(TWEETS_N, random_state=RANDOM_STATE)
    test = ds["test"].to_pandas()
    return train[["text", "label"]], test[["text", "label"]]


def load_yelp():
    """Stream Yelp reviews; balanced 3-class sample + a small held-out test set."""
    star_to_label = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}  # dataset labels are stars-1
    per_class_train = {0: [], 1: [], 2: []}
    per_class_test = {0: [], 1: [], 2: []}
    test_per_class = 500
    try:
        stream = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
        for ex in islice(stream, 400_000):
            label = star_to_label[ex["label"]]
            text = ex["text"][:1200]
            if len(per_class_train[label]) < YELP_PER_CLASS:
                per_class_train[label].append(text)
            elif len(per_class_test[label]) < test_per_class:
                per_class_test[label].append(text)
            if (all(len(v) >= YELP_PER_CLASS for v in per_class_train.values())
                    and all(len(v) >= test_per_class for v in per_class_test.values())):
                break
    except Exception as exc:
        print(f"! yelp_review_full unavailable ({exc}); skipping")
        return pd.DataFrame(columns=["text", "label"]), pd.DataFrame(columns=["text", "label"])
    train = pd.DataFrame([{"text": t, "label": lab}
                          for lab, texts in per_class_train.items() for t in texts])
    test = pd.DataFrame([{"text": t, "label": lab}
                         for lab, texts in per_class_test.items() for t in texts])
    return train, test


def load_amazon():
    per_class = {0: [], 1: []}
    try:
        stream = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
        for ex in islice(stream, 100_000):
            if len(per_class[ex["label"]]) < AMAZON_PER_CLASS:
                per_class[ex["label"]].append(ex["content"][:1200])
            if all(len(v) >= AMAZON_PER_CLASS for v in per_class.values()):
                break
    except Exception as exc:
        print(f"! amazon_polarity unavailable ({exc}); skipping")
        return pd.DataFrame(columns=["text", "label"])
    # amazon labels: 0=negative, 1=positive -> our 0/2
    return pd.DataFrame([{"text": t, "label": 0 if lab == 0 else 2}
                         for lab, texts in per_class.items() for t in texts])


def load_upload_csv():
    """The user-provided sample file: every line is one quoted CSV record."""
    if not UPLOAD_CSV.exists():
        return pd.DataFrame(columns=["text", "label"])
    rows = []
    raw_lines = UPLOAD_CSV.read_text().splitlines()
    for line in raw_lines[1:]:
        line = line.strip()
        if len(line) < 5:
            continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].replace('""', '"')
        parts = next(csv.reader(io.StringIO(line)))
        if len(parts) < 2:
            continue
        text, sentiment = parts[0].strip().strip('"'), parts[1].strip().title()
        if sentiment in ("Negative", "Neutral", "Positive") and text:
            rows.append({"text": text, "label": LABEL_NAMES.index(sentiment)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ train

def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tweets…")
    tweets_train, tweets_test = load_tweets()
    print(f"  tweets: {len(tweets_train):,} train / {len(tweets_test):,} test")
    print("Streaming Yelp reviews…")
    yelp_train, yelp_test = load_yelp()
    print(f"  yelp: {len(yelp_train):,} train / {len(yelp_test):,} test")
    print("Streaming Amazon reviews…")
    amazon_train = load_amazon()
    print(f"  amazon: {len(amazon_train):,} train")
    upload_df = load_upload_csv()
    print(f"  uploaded sample file: {len(upload_df):,} rows")

    train_df = pd.concat([tweets_train, yelp_train, amazon_train, upload_df],
                         ignore_index=True)
    train_df["text"] = train_df["text"].map(clean_text)
    train_df = train_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Combined training set: {len(train_df):,}")
    print(train_df["label"].value_counts().sort_index().rename(
        index=dict(enumerate(LABEL_NAMES))))

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def encode(df):
        ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH),
                      batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3,
        id2label=dict(enumerate(LABEL_NAMES)),
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./checkpoints_v2",
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=100,
            report_to="none",
            seed=RANDOM_STATE,
        ),
        train_dataset=encode(train_df),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    # ---------------- evaluate ----------------
    results = {}
    for name, df in [("tweet_eval_test", tweets_test), ("yelp_test", yelp_test)]:
        if df.empty:
            continue
        df = df.copy()
        df["text"] = df["text"].map(clean_text)
        preds = np.argmax(trainer.predict(encode(df)).predictions, axis=-1)
        results[name] = {
            "accuracy": round(float(accuracy_score(df["label"], preds)), 4),
            "macro_f1": round(float(f1_score(df["label"], preds, average="macro")), 4),
            "per_class": classification_report(df["label"], preds,
                                               target_names=LABEL_NAMES, output_dict=True),
            "n": int(len(df)),
        }
        print(f"{name}: acc {results[name]['accuracy']}  macroF1 {results[name]['macro_f1']}")

    trainer.save_model(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))
    results["train_size"] = int(len(train_df))
    results["sources"] = {"tweets": int(len(tweets_train)), "yelp": int(len(yelp_train)),
                          "amazon": int(len(amazon_train)), "uploaded_csv": int(len(upload_df))}
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Model saved to {SAVE_DIR}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
