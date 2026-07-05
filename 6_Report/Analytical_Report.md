# AI Customer Sentiment Analysis Dashboard — Analytical Report

**Author:** Aditya Chitale
**Course:** Artificial Intelligence Major Capstone Project (Option 1)
**Date:** 5 July 2026

---

## 1. Executive Summary

This project delivers an end-to-end AI system that automatically classifies customer feedback as **Positive**, **Negative**, or **Neutral** and visualises sentiment trends through an interactive dashboard. A fine-tuned DistilBERT transformer achieved **68.2% accuracy** and **0.680 macro F1** on a held-out test set of 12,284 real texts, outperforming a classical TF-IDF + Logistic Regression baseline (58.6% / 0.554) by **+9.6 accuracy points and +12.6 macro-F1 points**. The model is served through a real-time FastAPI endpoint; every prediction is persisted to a SQLite database and surfaced in a Streamlit dashboard for trend and issue analysis.

## 2. Problem Statement

Companies receive customer feedback continuously across channels (reviews, surveys, social media, support emails). Reading it manually does not scale, and delayed detection of negative sentiment spikes translates directly into churn and reputational damage. The goal is a system that (a) classifies feedback automatically, and (b) makes trends and recurring issues visible to non-technical stakeholders in real time.

## 3. Data

- **Dataset:** `cardiffnlp/tweet_eval` (sentiment task) — a standard academic benchmark of **real tweets** from the SemEval-2017 competition, labelled Negative / Neutral / Positive by human annotators.
- **Splits:** 45,615 train / 2,000 validation / 12,284 test (fixed benchmark splits — the test set was never seen during training).
- **Class balance:** Negative 15.5% / Neutral 45.3% / Positive 39.1%. Because the classes are imbalanced, we report **macro F1** (every class weighted equally) rather than accuracy alone.
- **Text characteristics:** short texts (median 20 words), informal language, emoticons, hashtags and URLs — representative of real customer-feedback channels.

## 4. Methodology

### 4.1 Preprocessing
Transformer models require minimal preprocessing — the WordPiece tokenizer handles casing and subwords, and aggressive cleaning (stopword removal, stemming) *degrades* contextual models. We applied only noise normalisation: URLs → `http`, user mentions → `@user`, whitespace collapsing. The identical cleaning function is used at training time and inside the inference API, preventing train/serve skew. The classical baseline additionally used lowercasing, English stopword removal and 1–2-gram TF-IDF features (20,000 max features).

### 4.2 Models
| | Baseline | Final model |
|---|---|---|
| Architecture | TF-IDF + Logistic Regression | DistilBERT (`distilbert-base-uncased`), fine-tuned |
| Why | Strong, fast classical reference | ~97% of BERT quality at 40% fewer parameters — suits real-time serving |
| Training | max_iter=1000, C=1.0, full 45,615-sample train set | 2 epochs, lr 2e-5, batch 16, max length 128, weight decay 0.01, 12,000-sample stratified subset (Apple M1 GPU/MPS) |

### 4.3 System architecture
```
Notebook (train + evaluate) ──► 2_Model/ (saved weights)
                                     │
        POST /predict                ▼
User ──► FastAPI (3_API) ──► SQLite (4_Database) ──► Streamlit dashboard (5_Dashboard)
```

## 5. Model Performance

All figures measured on the untouched 12,284-sample test set (`metrics.json` holds the raw numbers; `figures/confusion_matrices.png` the plots).

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.586 | 0.554 |
| **DistilBERT (fine-tuned)** | **0.682** | **0.680** |

Per-class results (DistilBERT, test set):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.706 | 0.674 | 0.690 | 3,972 |
| Neutral | 0.703 | 0.667 | 0.685 | 5,937 |
| Positive | 0.609 | 0.732 | 0.665 | 2,375 |

**Observations:**
- **The largest gain is on the Negative class** — the business-critical one. The baseline caught only 34% of negative feedback (recall 0.342, F1 0.457); DistilBERT doubles that recall to 0.674 (F1 0.690, **+23 F1 points**). Keyword models miss negativity expressed through context and sarcasm rather than obvious negative words.
- The transformer is also far more *balanced*: baseline per-class F1 ranged 0.46–0.66, DistilBERT sits in a tight 0.66–0.69 band across all three classes.
- The confusion matrices show remaining errors concentrate in the Neutral↔Negative and Neutral↔Positive boundaries — texts with mixed or implicit sentiment.
- For context, state-of-the-art results on this benchmark (RoBERTa trained on 58M tweets) reach ~0.72 macro F1; our 0.680 with a distilled model, 2 epochs and a 12k training subset is a strong result with a fraction of the compute, and the gap is a documented improvement path.
- Inference latency through the API is tens of milliseconds per request on the M1 (reported per-request in the API's `latency_ms` field) — comfortably real-time.

### 5.1 Model v2 — Extension To E-Commerce Review Text

To handle marketplace-style feedback (Amazon/noon/talabat-like reviews), a second
model was trained on a combined corpus of **28,096 real texts**: 12,000 tweets
(`cardiffnlp/tweet_eval`), 12,000 Yelp reviews (`yelp_review_full`, star ratings
mapped to 3 classes), 4,000 Amazon reviews (`amazon_polarity`) and a 96-item
labelled customer-feedback file. Training code: `1_Notebook/train_v2_reviews.py`;
metrics: `metrics_v2.json`.

| Test set | Accuracy | Macro F1 | n |
|---|---|---|---|
| Tweets (tweet_eval test) | 0.676 | 0.675 | 12,284 |
| **Customer reviews (held-out Yelp)** | **0.733** | **0.728** | 1,500 |

The v2 model gains strong review-domain performance (73.3% on a three-class task
that includes the difficult Neutral middle) while remaining statistically
unchanged on tweets (−0.6 points) — the API serves v2 by default.

## 6. Key Insights

1. **Context beats keywords for detecting unhappy customers.** The doubling of Negative-class recall is the single most valuable improvement: a keyword-based system would silently miss two-thirds of complaints; the transformer catches two-thirds *of them*.
2. **Sentiment is trackable in real time.** The dashboard's trend chart makes day-over-day sentiment shifts visible immediately, enabling a response before an issue snowballs.
3. **Issue analysis pinpoints root causes.** Frequent-word analysis of negative feedback surfaces concrete themes, turning raw complaints into an actionable priority list.
4. **Neutral is the hard class — and still valuable.** Neutral feedback often contains questions and pre-purchase signals; classifying it correctly routes it to sales/support rather than ignoring it.

## 7. Business / Practical Implications

- **Faster incident response:** a spike in negative sentiment after a release or delivery issue is visible within minutes, not weeks.
- **Support triage:** auto-routing negative feedback to support and neutral questions to sales saves manual reading time; at 68% accuracy the system works as a triage layer with human review of low-confidence predictions (the API returns a confidence score for exactly this purpose).
- **Product prioritisation:** recurring negative keywords provide a data-driven backlog signal.
- **Scalability:** the API architecture handles feedback from any channel (web forms, app reviews, email) through one endpoint — including a batch endpoint for bulk imports — with all history preserved in the database.

## 8. Limitations & Future Work

- **Accuracy headroom:** training on the full 45k set for more epochs, or starting from a domain-pretrained checkpoint (e.g. `cardiffnlp/twitter-roberta-base`), would close most of the ~4-point gap to state of the art.
- Sarcasm and mixed-sentiment texts remain the main error source; aspect-based sentiment analysis would let one text carry multiple sentiments per topic.
- The model is English-only; a multilingual model (e.g. XLM-RoBERTa) would extend coverage.
- SQLite (WAL mode) suits a single-host deployment; production at scale would use PostgreSQL, API authentication, and HTTPS termination. The Docker Compose setup in the repository is the deployment starting point.
- Periodic re-fine-tuning on the company's own labelled feedback would adapt the model to domain vocabulary.

## 9. Conclusion

All five core requirements — NLP preprocessing, a transformer-based model, real-time API inference, database integration, and a trend/issue dashboard — are implemented and integrated into a single working system, trained and evaluated on real human-labelled data. The fine-tuned DistilBERT model materially outperforms a classical baseline (+12.6 macro-F1 points), most dramatically on the business-critical task of catching negative feedback, and the dashboard converts its predictions into insights a business can act on.
