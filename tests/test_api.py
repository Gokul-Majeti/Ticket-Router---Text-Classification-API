import sqlite3

import pytest
from fastapi.testclient import TestClient

from api import main

EXPECTED_CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Create a TestClient backed by an isolated temporary SQLite database."""
    db_file = tmp_path / "nested" / "test_predictions.db"
    monkeypatch.setattr(main, "DB_PATH", str(db_file))
    with TestClient(main.app) as test_client:
        yield test_client, db_file


def test_health_returns_ok(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_initializes_db_table(client):
    _, db_file = client
    assert db_file.exists()
    with sqlite3.connect(db_file) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "predictions" in tables


def test_stats_empty_database(client):
    test_client, _ = client
    response = test_client.get("/stats")
    assert response.status_code == 200
    assert response.json() == {"total_predictions": 0}


def test_predict_valid_sport_text_and_logs_to_db(client):
    test_client, db_file = client
    payload = {
        "text": (
            "The football team won the championship trophy after a thrilling "
            "final match at the stadium on Sunday."
        )
    }
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["category"] == "sport"
    assert 0.0 <= data["confidence"] <= 1.0
    assert set(data["all_scores"].keys()) == EXPECTED_CATEGORIES
    assert sum(data["all_scores"].values()) == pytest.approx(1.0, abs=1e-5)
    assert data["latency_ms"] >= 0.0

    with sqlite3.connect(db_file) as conn:
        rows = conn.execute(
            "SELECT text_preview, predicted_category, confidence, latency_ms FROM predictions"
        ).fetchall()

    assert len(rows) == 1
    preview, category, confidence, latency_ms = rows[0]
    assert preview == payload["text"][:100]
    assert category == data["category"]
    assert confidence == pytest.approx(data["confidence"])
    assert latency_ms == pytest.approx(data["latency_ms"])


def test_predict_truncates_long_text_preview_and_strips_whitespace(client):
    test_client, db_file = client
    inner_text = "Prime minister announces new parliament election schedule. " * 3
    response = test_client.post("/predict", json={"text": f"   {inner_text}   "})
    assert response.status_code == 200

    with sqlite3.connect(db_file) as conn:
        (preview,) = conn.execute("SELECT text_preview FROM predictions").fetchone()

    assert len(preview) == 100
    assert preview == inner_text.strip()[:100]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": "\n\t  "},
        {"text": None},
    ],
)
def test_predict_rejects_invalid_or_blank_text(client, payload):
    test_client, db_file = client
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422

    with sqlite3.connect(db_file) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
    assert count == 0


def test_stats_aggregates_multiple_predictions(client):
    test_client, _ = client
    samples = [
        "The team won the championship trophy after a thrilling final match.",
        "Software engineers released a new smartphone chip and cloud computing API.",
        "xyzzy",  # short unknown token -> low confidence (< 0.6)
    ]

    results = []
    for text in samples:
        res = test_client.post("/predict", json={"text": text})
        assert res.status_code == 200
        results.append(res.json())

    stats_res = test_client.get("/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()

    expected_low_conf = sum(1 for r in results if r["confidence"] < 0.6)
    expected_avg_conf = round(sum(r["confidence"] for r in results) / len(results), 4)
    expected_avg_lat = round(sum(r["latency_ms"] for r in results) / len(results), 2)
    expected_dist = {}
    for r in results:
        expected_dist[r["category"]] = expected_dist.get(r["category"], 0) + 1

    assert expected_low_conf >= 1
    assert stats["total_predictions"] == len(samples)
    assert stats["avg_confidence"] == expected_avg_conf
    assert stats["avg_latency_ms"] == expected_avg_lat
    assert stats["low_confidence_predictions"] == expected_low_conf
    assert stats["low_confidence_rate"] == round(expected_low_conf / len(samples), 4)
    assert stats["category_distribution"] == expected_dist
