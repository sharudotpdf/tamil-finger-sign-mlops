from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app import demo_api as api


class FakePredictor:
    def __init__(self):
        self.export_dir = ROOT / "fake-export"
        self.selection = {"model_version": "test-model-v1"}
        self.threshold = 0.5
        self.closed = False

    def predict_bytes(self, data: bytes, content_type: str | None = None) -> dict:
        return {
            "request_id": "test-request",
            "outcome": "accepted_sign",
            "character": "அ",
            "predicted_character": "அ",
            "raw_character": "அ",
            "score": 0.9,
            "score_is_calibrated_probability": False,
            "threshold": self.threshold,
            "category": "uyir",
            "vowel_component": "V01",
            "consonant_component": "FIST",
            "reason": "component_composition",
            "n_hands": 2,
            "first_n_hands": 2,
            "retry_used": False,
            "model_version": self.selection["model_version"],
            "latency_ms": 12.3,
            "annotated_image": "data:image/jpeg;base64,AA==",
        }

    def close(self) -> None:
        self.closed = True


def _labels() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "character": "அ",
            "dataset_folder": "1",
            "category": "uyir",
            "vowel_target": "V01",
            "consonant_target": "FIST",
        },
        {
            "character": "__background__",
            "dataset_folder": "background",
            "category": "background",
            "vowel_target": "__NA__",
            "consonant_target": "__NA__",
        },
    ])


@pytest.fixture
def client(monkeypatch):
    fake = FakePredictor()
    monkeypatch.setattr(api, "TLFS23Predictor", lambda root: fake)
    monkeypatch.setattr(api, "_load_label_table", lambda export_dir: _labels())
    monkeypatch.setattr(api, "_load_train_references", lambda: {})
    with TestClient(api.app) as test_client:
        yield test_client, fake


def test_health_endpoint(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test-model-v1"
    assert body["threshold"] == 0.5
    assert body["n_labels"] == 2


def test_lookup_endpoint(client):
    test_client, _ = client
    response = test_client.get("/lookup", params={"text": "அ"})
    assert response.status_code == 200
    body = response.json()
    assert body["all_supported"] is True
    assert body["segments"][0]["character"] == "அ"


def test_predict_endpoint(client):
    test_client, _ = client
    response = test_client.post(
        "/predict",
        files={"file": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "accepted_sign"
    assert body["model_version"] == "test-model-v1"
    assert body["predicted_label"]["character"] == "அ"


def test_predict_rejects_oversized_upload(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    response = test_client.post(
        "/predict",
        files={"file": ("too-big.jpg", b"12345", "image/jpeg")},
    )
    assert response.status_code == 413
