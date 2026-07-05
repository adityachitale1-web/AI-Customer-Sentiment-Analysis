"""API test suite. Run from 3_API/:  ../.venv/bin/python -m pytest tests/ -v

Uses SENTIMENT_TEST_MODE=1 (dummy classifier) and a temporary database so
tests are fast and never touch the real model or feedback.db.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ["SENTIMENT_TEST_MODE"] = "1"
os.environ["SENTIMENT_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test_feedback.db")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "model" in body


def test_predict_positive(client):
    resp = client.post("/predict", json={"text": "I love this product, it is great!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sentiment"] == "Positive"
    assert 0 <= body["confidence"] <= 1
    assert body["id"] > 0
    assert set(body["scores"]) == {"Negative", "Neutral", "Positive"}
    assert body["scores"]["Positive"] == body["confidence"]


def test_predict_negative(client):
    resp = client.post("/predict", json={"text": "This is terrible and broken."})
    assert resp.status_code == 200
    assert resp.json()["sentiment"] == "Negative"


def test_predict_neutral(client):
    resp = client.post("/predict", json={"text": "The package arrived on Tuesday."})
    assert resp.status_code == 200
    assert resp.json()["sentiment"] == "Neutral"


def test_predict_validation(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 3000}).status_code == 422


def test_batch_predict(client):
    resp = client.post("/predict/batch", json={"items": [
        {"text": "I love it"}, {"text": "I hate it"}, {"text": "It exists"},
    ]})
    assert resp.status_code == 200
    sentiments = [item["sentiment"] for item in resp.json()]
    assert sentiments == ["Positive", "Negative", "Neutral"]


def test_feedback_listing_and_stats(client):
    client.post("/predict", json={"text": "I love it", "source": "test_suite"})
    listing = client.get("/feedback?limit=5").json()
    assert listing["total"] >= 1
    assert len(listing["items"]) >= 1
    assert {"text", "sentiment", "confidence", "source", "created_at"} <= set(listing["items"][0])

    stats = client.get("/stats").json()
    assert stats["total"] >= 1
    assert "Positive" in stats["counts"]


def test_feedback_pagination_validation(client):
    assert client.get("/feedback?limit=0").status_code == 422
    assert client.get("/feedback?limit=5000").status_code == 422
