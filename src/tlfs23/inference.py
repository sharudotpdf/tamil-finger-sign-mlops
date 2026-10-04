"""Lokale Ende-zu-Ende-Inferenz fuer den exportierten TLFS23-Prototyp.

Die Inferenz verwendet exakt die in Notebook 05 freigegebene MediaPipe-Konfiguration,
das in Notebook 06 ausgewaehlte Modell und dessen Threshold. Hochgeladene Bilder
werden nur im Speicher verarbeitet; Runtime-Logs enthalten keine Bilddaten.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, UnidentifiedImageError

from . import features as f
from . import modeling as m
from .common import paths, read_json, sha256_file

MAX_IMAGE_PIXELS = 20_000_000
REQUIRED_EXPORT_FILES = {
    "model.joblib",
    "hand_landmarker.task",
    "selected_model.json",
    "05_feature_decision.json",
    "checksums.json",
    "features.py",
    "modeling.py",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_export_dir(root: Path) -> Path:
    """Findet das von Notebook 07 erzeugte Inferenzpaket deterministisch."""
    root = Path(root).resolve()
    p = paths(root)
    final_metrics = p["work"] / "final_test" / "metrics.json"
    if final_metrics.exists():
        metrics = read_json(final_metrics)
        relative = metrics.get("export_directory")
        if relative:
            candidate = (root / relative).resolve()
            if candidate.is_dir():
                return candidate

    selection_path = p["work"] / "modeling" / "selected_model.json"
    if not selection_path.exists():
        raise FileNotFoundError("selected_model.json fehlt. Notebook 06 zuerst abschliessen.")
    selection = read_json(selection_path)
    candidate = root / "models" / "export" / selection["model_version"]
    if not candidate.is_dir():
        raise FileNotFoundError(
            "Finales Exportpaket fehlt. Notebook 07 bis zum Exportabschnitt ausfuehren."
        )
    return candidate.resolve()


def verify_export(export_dir: Path) -> dict:
    """Prueft Vollstaendigkeit und die in Notebook 07 gespeicherten Checksums."""
    export_dir = Path(export_dir).resolve()
    missing = sorted(name for name in REQUIRED_EXPORT_FILES if not (export_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Export unvollstaendig; Dateien fehlen: {missing}")

    checksums = read_json(export_dir / "checksums.json")
    for name, expected in checksums.items():
        if Path(name).name != name:
            raise ValueError(f"Ungueltiger Dateiname in checksums.json: {name}")
        file = export_dir / name
        if not file.is_file() or sha256_file(file) != expected:
            raise ValueError(f"Exportdatei fehlt oder wurde veraendert: {name}")
    return checksums


def _decode_image(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("Leere Bilddatei.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Bild ist zu gross fuer die lokale Demo.")
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Datei konnte nicht als unterstuetztes Bild gelesen werden.") from exc


def _annotated_data_url(image: Image.Image, hands: list[np.ndarray]) -> str:
    """Zeichnet die detektierten Landmarkpunkte auf eine Kopie des Inferenzbildes."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    edges = [(0, b) for b in f.BASES] + [
        (i, i + 1) for b in f.BASES for i in range(b, b + 3)
    ]
    radius = max(2, int(min(canvas.size) * 0.006))

    for hand in hands:
        xx = hand[:, 0] * canvas.width
        yy = hand[:, 1] * canvas.height
        for a, b in edges:
            draw.line((float(xx[a]), float(yy[a]), float(xx[b]), float(yy[b])), width=max(1, radius // 2))
        for x, y in zip(xx, yy):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="white", width=max(1, radius // 2))

    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


class TLFS23Predictor:
    """Ein lokaler Predictor mit einem wiederverwendeten MediaPipe-Detektor."""

    def __init__(self, root: Path, export_dir: Path | None = None):
        self.root = Path(root).resolve()
        self.export_dir = (Path(export_dir).resolve() if export_dir else resolve_export_dir(self.root))
        verify_export(self.export_dir)

        # Demo soll mit exakt dem Code laufen, der mit dem Modell exportiert wurde.
        if sha256_file(Path(f.__file__)) != sha256_file(self.export_dir / "features.py"):
            raise ValueError("Aktuelles features.py stimmt nicht mit dem Export ueberein.")
        if sha256_file(Path(m.__file__)) != sha256_file(self.export_dir / "modeling.py"):
            raise ValueError("Aktuelles modeling.py stimmt nicht mit dem Export ueberein.")

        self.selection = read_json(self.export_dir / "selected_model.json")
        self.feature_decision = read_json(self.export_dir / "05_feature_decision.json")
        self.config = dict(self.feature_decision["config"])
        self.threshold = float(self.selection["threshold"])
        self.bundle = joblib.load(self.export_dir / "model.joblib")

        if self.bundle["model_version"] != self.selection["model_version"]:
            raise ValueError("Modellbundle und selected_model.json passen nicht zusammen.")
        if list(self.bundle["feature_names"]) != list(f.FEATURE_NAMES):
            raise ValueError("Feature-Schema des Modells stimmt nicht mit features.py ueberein.")

        self.detector = f.Detector(self.export_dir / "hand_landmarker.task", self.config)
        self._lock = threading.Lock()
        self.log_path = paths(self.root)["work"] / "runtime" / "inference.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.detector.close()

    def _detect(self, image: Image.Image) -> tuple[list[np.ndarray], int, bool, Image.Image]:
        image = image.copy()
        image.thumbnail((int(self.config["max_side"]), int(self.config["max_side"])), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.uint8)
        hands = self.detector.detect(rgb)
        first_n_hands = len(hands)
        retry_used = False

        if len(hands) < 2 and bool(self.config.get("retry", False)):
            retry_used = True
            enhanced = ImageEnhance.Contrast(image).enhance(1.1)
            enhanced = ImageEnhance.Brightness(enhanced).enhance(1.1)
            retried = self.detector.detect(np.asarray(enhanced, dtype=np.uint8))
            if len(retried) > len(hands):
                hands = retried

        hands = sorted(hands, key=lambda h: (float(h[0, 0]), float(h[0, 1])))
        return hands, first_n_hands, retry_used, image

    def predict_bytes(self, data: bytes, content_type: str | None = None) -> dict:
        request_id = uuid4().hex
        started = time.perf_counter()
        image_sha256 = _sha_bytes(data)

        with self._lock:
            image = _decode_image(data)
            hands, first_n_hands, retry_used, inference_image = self._detect(image)

            if len(hands) == 2:
                vector = f.pair_features(hands, inference_image.width, inference_image.height)
                status = "ok"
                x = vector.reshape(1, -1)
            else:
                status = "not_detected" if len(hands) == 0 else "incomplete_pair"
                x = np.full((1, len(f.FEATURE_NAMES)), np.nan, dtype=np.float32)

            frame = pd.DataFrame(
                [
                    {
                        "relative_path": "runtime_upload",
                        "sha256": image_sha256,
                        "character": "",
                        "category": "",
                        "vowel_target": "",
                        "consonant_target": "",
                        "status": status,
                        "variant": "original",
                        "split": "runtime",
                        "n_hands": len(hands),
                    }
                ]
            )

            row = m.predict_heads(self.bundle, frame, x, threshold=self.threshold).iloc[0]

        predicted = str(row["predicted_character"])
        raw_character = str(row["raw_character"])
        accepted = bool(row["accepted"])

        if accepted and predicted == m.BG:
            outcome = "no_sign"
            display_character = "Kein gueltiges Zeichen"
        elif accepted and predicted not in {m.BG, m.REJECT}:
            outcome = "accepted_sign"
            display_character = predicted
        else:
            outcome = "reject"
            display_character = "Zurueckgewiesen"

        latency_ms = float((time.perf_counter() - started) * 1000)
        result = {
            "request_id": request_id,
            "outcome": outcome,
            "character": display_character,
            "predicted_character": predicted,
            "raw_character": raw_character,
            "score": float(row["score"]),
            "score_is_calibrated_probability": False,
            "threshold": self.threshold,
            "category": str(row["pred_type"]),
            "vowel_component": str(row["pred_vowel"]),
            "consonant_component": str(row["pred_consonant"]),
            "reason": str(row["reason"]),
            "n_hands": int(len(hands)),
            "first_n_hands": int(first_n_hands),
            "retry_used": bool(retry_used),
            "model_version": str(self.selection["model_version"]),
            "latency_ms": latency_ms,
            "annotated_image": _annotated_data_url(inference_image, hands),
        }

        # Privacy-schonendes Betriebslog: keine Bildbytes, kein Dateiname, kein kompletter Bildhash.
        log_record = {
            "timestamp_utc": _utc(),
            "request_id": request_id,
            "model_version": result["model_version"],
            "outcome": outcome,
            "n_hands": result["n_hands"],
            "retry_used": result["retry_used"],
            "score": result["score"],
            "threshold": result["threshold"],
            "reason": result["reason"],
            "latency_ms": latency_ms,
            "content_type": content_type or "unknown",
            "input_bytes": len(data),
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        return result
