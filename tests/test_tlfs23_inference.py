from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tlfs23 import inference as inf


def _png_bytes(size=(4, 4)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_image_accepts_valid_image():
    image = inf._decode_image(_png_bytes())
    assert image.mode == "RGB"
    assert image.size == (4, 4)


def test_decode_image_rejects_empty_and_invalid_bytes():
    with pytest.raises(ValueError, match="Leere Bilddatei"):
        inf._decode_image(b"")

    with pytest.raises(ValueError, match="unterstuetztes Bild"):
        inf._decode_image(b"not-an-image")


def test_decode_image_rejects_too_many_pixels(monkeypatch):
    monkeypatch.setattr(inf, "MAX_IMAGE_PIXELS", 3)
    with pytest.raises(ValueError, match="zu gross"):
        inf._decode_image(_png_bytes((2, 2)))


def test_verify_export_checks_required_files_and_hashes(tmp_path):
    for name in inf.REQUIRED_EXPORT_FILES:
        if name == "checksums.json":
            continue
        (tmp_path / name).write_bytes(f"content:{name}".encode("utf-8"))

    checksums = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in inf.REQUIRED_EXPORT_FILES
        if name != "checksums.json"
    }
    (tmp_path / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")

    assert inf.verify_export(tmp_path) == checksums

    (tmp_path / "model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="veraendert"):
        inf.verify_export(tmp_path)
