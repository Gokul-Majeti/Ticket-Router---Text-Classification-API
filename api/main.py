"""
FastAPI service that serves our text classifier.

Why FastAPI specifically:
- Automatic request validation (via Pydantic models below) — malformed
  requests get rejected with a clear error before ever touching the model.
- Auto-generated interactive docs at /docs, useful for testing without
  writing a separate client.
- Async-ready, so it can handle concurrent requests without blocking.
"""

import time
import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load model artifacts ONCE at startup, not per-request.
# Loading from disk on every request would add huge, unnecessary latency.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")
DB_PATH = os.path.join(BASE_DIR, "data", "predictions.db")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = FastAPI(
    title="Ticket Router API",
    description="Classifies incoming text into a category (business/entertainment/politics/sport/tech).",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / response schemas.
# Pydantic validates incoming JSON automatically — e.g. if `text` is missing
# or empty, the caller gets a 422 error before our code even runs.
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to classify")


class PredictResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict[str, float]
    latency_ms: float


# ---------------------------------------------------------------------------
# Minimal monitoring: log every prediction to SQLite.
# This is the raw material for monitoring later: confidence over time,
# class distribution drift, latency trends, volume spikes.
# We use SQLite (a file, not a server) because for a solo/small project it
# needs zero setup. Swap for Postgres later if this becomes multi-instance.
# ---------------------------------------------------------------------------
@contextmanager
def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text_preview TEXT,
                predicted_category TEXT,
                confidence REAL,
                latency_ms REAL
            )
        """)
        conn.commit()


def log_prediction(text: str, category: str, confidence: float, latency_ms: float):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, text_preview, predicted_category, confidence, latency_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                text[:100],  # only store a preview, not full text, for privacy + storage reasons
                category,
                confidence,
                latency_ms,
            ),
        )
        conn.commit()


@app.on_event("startup")
def startup():
    init_db()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Basic liveness check — used by load balancers / uptime monitors."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()

    vec = vectorizer.transform([req.text])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    scores = {cls: float(p) for cls, p in zip(classes, probs)}
    best_idx = probs.argmax()
    predicted_category = classes[best_idx]
    confidence = float(probs[best_idx])

    latency_ms = (time.perf_counter() - start) * 1000

    # Log every single prediction — this is what makes monitoring possible later.
    log_prediction(req.text, predicted_category, confidence, latency_ms)

    return PredictResponse(
        category=predicted_category,
        confidence=confidence,
        all_scores=scores,
        latency_ms=latency_ms,
    )


@app.get("/stats")
def stats():
    """
    Basic monitoring endpoint. In a real production system this data would
    feed a dashboard (Grafana, or a simple Streamlit app) rather than being
    read raw — but the underlying idea is the same: watch these numbers
    over time to catch problems before users complain.
    """
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        if total == 0:
            return {"total_predictions": 0}

        avg_confidence = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
        avg_latency = conn.execute("SELECT AVG(latency_ms) FROM predictions").fetchone()[0]
        low_confidence_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE confidence < 0.6"
        ).fetchone()[0]

        by_category = conn.execute(
            "SELECT predicted_category, COUNT(*) FROM predictions GROUP BY predicted_category"
        ).fetchall()

    return {
        "total_predictions": total,
        "avg_confidence": round(avg_confidence, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "low_confidence_predictions": low_confidence_count,
        "low_confidence_rate": round(low_confidence_count / total, 4),
        "category_distribution": {cat: count for cat, count in by_category},
    }
