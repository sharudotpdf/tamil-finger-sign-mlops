"""Lokale FastAPI-Demo fuer den TLFS23-Prototyp."""
from __future__ import annotations

import base64
import io
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps


HERE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        p for p in (HERE.parent, *HERE.parents)
        if (p / "src").is_dir() and (p / "models").is_dir()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("Projektroot mit src/ und models/ nicht gefunden.")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tlfs23.inference import TLFS23Predictor  # noqa: E402
from tlfs23 import data as d  # noqa: E402
from tlfs23.common import class_root, safe_path  # noqa: E402

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
predictor: TLFS23Predictor | None = None
label_table = pd.DataFrame()
train_reference: dict[str, dict] = {}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", str(text).strip())


def _load_label_table(export_dir: Path) -> pd.DataFrame:
    path = export_dir / "label_schema.csv"
    if not path.is_file():
        raise FileNotFoundError("label_schema.csv fehlt im Exportpaket.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"character", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"label_schema.csv fehlen Spalten: {sorted(missing)}")
    frame["character"] = frame["character"].map(_normalize)
    return frame


def _load_train_references() -> dict[str, dict]:
    """Optional: genau ein deterministisches Trainingsbild je Zeichen."""
    try:
        split, _ = d.load_split(PROJECT_ROOT)
    except Exception:
        return {}
    if "character" not in split or "split" not in split:
        return {}
    train = split[split["split"].eq("train")].copy()
    if train.empty:
        return {}
    train["character"] = train["character"].map(_normalize)
    chosen = (
        train.sort_values(["character", "sha256", "relative_path"])
        .groupby("character", sort=True)
        .head(1)
    )
    return {str(row["character"]): row.to_dict() for _, row in chosen.iterrows()}


def _public_label_record(row: pd.Series | dict) -> dict:
    record = dict(row)
    keys = ["character", "dataset_folder", "category", "vowel_target", "consonant_target"]
    return {key: str(record.get(key, "")) for key in keys}


def _segment_known_text(text: str) -> tuple[list[dict], list[str]]:
    """Segmentiert einen Tamil-String anhand der im exportierten Schema bekannten Zeichen."""
    normalized = _normalize(text)
    candidates = [
        c for c in label_table.get("character", pd.Series(dtype=str)).astype(str).tolist()
        if c and c != "__background__"
    ]
    candidates = sorted(set(candidates), key=lambda x: (-len(x), x))
    by_character = {
        str(row["character"]): _public_label_record(row)
        for _, row in label_table.iterrows()
    }

    segments: list[dict] = []
    unknown: list[str] = []
    i = 0
    while i < len(normalized):
        if normalized[i].isspace():
            i += 1
            continue
        match = next((c for c in candidates if normalized.startswith(c, i)), None)
        if match is None:
            unknown.append(normalized[i])
            i += 1
            continue
        segments.append(by_character[match])
        i += len(match)
    return segments, unknown


def _reference_data_url(character: str) -> dict:
    character = _normalize(character)
    row = train_reference.get(character)
    if row is None:
        raise HTTPException(status_code=404, detail="Kein Trainings-Referenzbild fuer dieses Zeichen gefunden.")
    try:
        base = class_root(PROJECT_ROOT)
        path = safe_path(base, row["relative_path"])
        with Image.open(path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((900, 700), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Referenzbild ist lokal nicht verfuegbar.") from exc

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return {
        "character": character,
        "dataset_folder": str(row.get("dataset_folder", "")),
        "reference_image": "data:image/jpeg;base64," + encoded,
        "source_split": "train",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, label_table, train_reference
    predictor = TLFS23Predictor(PROJECT_ROOT)
    label_table = _load_label_table(predictor.export_dir)
    train_reference = _load_train_references()
    try:
        yield
    finally:
        if predictor is not None:
            predictor.close()
            predictor = None
        label_table = pd.DataFrame()
        train_reference = {}


app = FastAPI(
    title="TLFS23 Tamil Finger Sign Demo",
    version="1.2",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        "status": "ok",
        "model_version": predictor.selection["model_version"],
        "threshold": predictor.threshold,
        "n_labels": int(len(label_table)),
        "reference_images_available": int(len(train_reference)),
    }


@app.get("/labels")
def labels():
    if label_table.empty:
        raise HTTPException(status_code=503, detail="Label table not ready")
    rows = [_public_label_record(row) for _, row in label_table.iterrows()]
    return {"n": len(rows), "labels": rows}


@app.get("/lookup")
def lookup(text: str = Query(..., min_length=1, max_length=100)):
    if label_table.empty:
        raise HTTPException(status_code=503, detail="Label table not ready")
    segments, unknown = _segment_known_text(text)
    return {
        "input": text,
        "normalized": _normalize(text),
        "segments": segments,
        "unknown": unknown,
        "all_supported": len(unknown) == 0 and len(segments) > 0,
    }


@app.get("/reference-image")
def reference_image(character: str = Query(..., min_length=1, max_length=8)):
    return _reference_data_url(character)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Bild ist groesser als 10 MB.")

    try:
        result = predictor.predict_bytes(data, content_type=file.content_type)

        # Die strukturierte Modellvorhersage gegen das exportierte Labelschema aufloesen.
        # So kann die UI neben Kategorie/Komponenten auch das konkrete Tamil-Zeichen zeigen.
        candidate = _normalize(result.get("raw_character", ""))
        match = label_table[label_table["character"].eq(candidate)]
        if not match.empty:
            result["predicted_label"] = _public_label_record(match.iloc[0])
        else:
            result["predicted_label"] = {
                "character": candidate,
                "dataset_folder": "",
                "category": result.get("category", ""),
                "vowel_target": result.get("vowel_component", ""),
                "consonant_target": result.get("consonant_component", ""),
            }
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inferenz fehlgeschlagen: {type(exc).__name__}") from exc


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TLFS23 Demo</title>
  <style>
    :root { color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #1f2024; color: #f5f5f7; }
    main { max-width: 1040px; margin: 48px auto; padding: 0 20px 60px; }
    .card { background: #2b2c31; border-radius: 18px; padding: 24px; margin: 18px 0; }
    h1 { margin-bottom: 8px; }
    h2 { margin-top: 0; }
    .muted { color: #b8b8c0; }
    input, button { font: inherit; }
    input[type=file] { width: 100%; margin: 18px 0; }
    input[type=text] { box-sizing: border-box; width: 100%; border: 1px solid #52535b; border-radius: 12px; padding: 13px 14px; background: #18191c; color: #f5f5f7; font-size: 20px; }
    button { border: 0; border-radius: 12px; padding: 12px 18px; cursor: pointer; font-weight: 700; }
    button:disabled { opacity: .55; cursor: wait; }
    .button-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    img { width: 100%; max-height: 470px; object-fit: contain; border-radius: 14px; background: #18191c; }
    .character { font-size: 64px; line-height: 1.1; margin: 8px 0 14px; }
    dl { display: grid; grid-template-columns: max-content 1fr; gap: 8px 14px; }
    dt { color: #b8b8c0; }
    dd { margin: 0; overflow-wrap: anywhere; }
    #error, #lookupError { color: #ffb4ab; }
    .lookup-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 16px; }
    .sign-card { background: #18191c; border-radius: 14px; padding: 16px; }
    .sign-char { font-size: 48px; margin-bottom: 8px; }
    .sign-card img { margin-top: 12px; max-height: 260px; }
    .table-wrap { overflow-x: auto; margin-top: 16px; }
    table { width: 100%; border-collapse: collapse; min-width: 720px; }
    th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #45464d; }
    th { color: #b8b8c0; font-size: 13px; }
    td:first-child { font-size: 22px; }
    .hidden { display: none; }
    @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>Tamil Finger Sign · lokale Demo</h1>
  <p class="muted">Referenzzeichen nachschlagen oder ein eigenes Bild klassifizieren.</p>

  <section class="card">
    <h2>Zeichen nachschlagen</h2>
    <p class="muted">Ein einzelnes Tamil-Zeichen oder direkt mehrere Zeichen, z. B. deinen Namen, eingeben.</p>
    <input id="lookupText" type="text" lang="ta" placeholder="Tamil-Zeichen eingeben …" autocomplete="off">
    <div class="button-row">
      <button id="lookupButton" type="button">Nachschlagen</button>
      <button id="toggleTable" type="button">Gesamte Tabelle anzeigen</button>
    </div>
    <p id="lookupError"></p>
    <div id="lookupResults" class="lookup-grid"></div>
    <div id="labelTableWrap" class="table-wrap hidden">
      <table>
        <thead>
          <tr>
            <th>Tamil-Zeichen</th>
            <th>Dataset-Ordner</th>
            <th>Kategorie</th>
            <th>Vokal-Komponente</th>
            <th>Konsonanten-Komponente</th>
          </tr>
        </thead>
        <tbody id="labelRows"></tbody>
      </table>
    </div>
  </section>

  <section class="card">
    <h2>Eigenes Foto testen</h2>
    <input id="file" type="file" accept="image/jpeg,image/png,image/webp">
    <button id="run">Vorhersage starten</button>
    <p id="error"></p>
  </section>

  <section class="grid">
    <div class="card">
      <h2>Eingabe / Landmarks</h2>
      <img id="preview" alt="Bildvorschau">
    </div>
    <div class="card">
      <h2>Ergebnis</h2>
      <div id="character" class="character">–</div>
      <dl>
        <dt>Status</dt><dd id="outcome">–</dd>
        <dt>Finales Zeichen</dt><dd id="finalCharacter">–</dd>
        <dt>Roh-Zeichen vor Threshold</dt><dd id="rawCharacter">–</dd>
        <dt>Dataset-Ordner</dt><dd id="datasetFolder">–</dd>
        <dt>Score</dt><dd id="score">–</dd>
        <dt>Threshold</dt><dd id="threshold">–</dd>
        <dt>Haende</dt><dd id="hands">–</dd>
        <dt>Retry</dt><dd id="retry">–</dd>
        <dt>Vorhergesagte Kategorie</dt><dd id="category">–</dd>
        <dt>Vokal</dt><dd id="vowel">–</dd>
        <dt>Konsonant</dt><dd id="consonant">–</dd>
        <dt>Grund</dt><dd id="reason">–</dd>
        <dt>Latenz</dt><dd id="latency">–</dd>
      </dl>
      <p class="muted">Der Score ist keine kalibrierte Wahrscheinlichkeit.</p>
    </div>
  </section>
</main>
<script>
const fileInput = document.getElementById('file');
const runButton = document.getElementById('run');
const error = document.getElementById('error');
const lookupText = document.getElementById('lookupText');
const lookupButton = document.getElementById('lookupButton');
const lookupError = document.getElementById('lookupError');
const lookupResults = document.getElementById('lookupResults');
const toggleTable = document.getElementById('toggleTable');
const labelTableWrap = document.getElementById('labelTableWrap');
const labelRows = document.getElementById('labelRows');
let labelsLoaded = false;

function textCell(value) {
  const td = document.createElement('td');
  td.textContent = value || '–';
  return td;
}

async function loadLabels() {
  if (labelsLoaded) return;
  const response = await fetch('/labels');
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Tabelle konnte nicht geladen werden.');
  labelRows.replaceChildren();
  for (const row of data.labels) {
    const tr = document.createElement('tr');
    tr.appendChild(textCell(row.character === '__background__' ? 'Background' : row.character));
    tr.appendChild(textCell(row.dataset_folder));
    tr.appendChild(textCell(row.category));
    tr.appendChild(textCell(row.vowel_target));
    tr.appendChild(textCell(row.consonant_target));
    labelRows.appendChild(tr);
  }
  labelsLoaded = true;
}

async function addReferenceImage(card, character) {
  const response = await fetch('/reference-image?character=' + encodeURIComponent(character));
  const data = await response.json();
  if (!response.ok) {
    const note = document.createElement('p');
    note.className = 'muted';
    note.textContent = 'Kein lokales Referenzbild verfügbar.';
    card.appendChild(note);
    return;
  }
  const img = document.createElement('img');
  img.src = data.reference_image;
  img.alt = 'Trainings-Referenzbild für ' + character;
  card.appendChild(img);
  const note = document.createElement('p');
  note.className = 'muted';
  note.textContent = 'Referenz: Trainingssplit · Ordner ' + data.dataset_folder;
  card.appendChild(note);
}

async function lookupCharacters() {
  const value = lookupText.value.trim();
  if (!value) {
    lookupError.textContent = 'Bitte mindestens ein Tamil-Zeichen eingeben.';
    lookupResults.replaceChildren();
    return;
  }
  lookupError.textContent = '';
  lookupButton.disabled = true;
  lookupResults.replaceChildren();
  try {
    const response = await fetch('/lookup?text=' + encodeURIComponent(value));
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Nachschlagen fehlgeschlagen.');

    if (data.unknown.length) {
      lookupError.textContent = 'Nicht im TLFS23-Schema erkannt: ' + data.unknown.join(' ');
    }
    if (!data.segments.length) return;

    for (const item of data.segments) {
      const card = document.createElement('div');
      card.className = 'sign-card';

      const char = document.createElement('div');
      char.className = 'sign-char';
      char.textContent = item.character;
      card.appendChild(char);

      const meta = document.createElement('div');
      meta.innerHTML = '<strong>Ordner:</strong> ' + (item.dataset_folder || '–') +
        '<br><strong>Kategorie:</strong> ' + (item.category || '–') +
        '<br><strong>Vokal:</strong> ' + (item.vowel_target || '–') +
        '<br><strong>Konsonant:</strong> ' + (item.consonant_target || '–');
      card.appendChild(meta);
      lookupResults.appendChild(card);
      await addReferenceImage(card, item.character);
    }
  } catch (e) {
    lookupError.textContent = e.message;
  } finally {
    lookupButton.disabled = false;
  }
}

lookupButton.addEventListener('click', lookupCharacters);
lookupText.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') lookupCharacters();
});

toggleTable.addEventListener('click', async () => {
  try {
    await loadLabels();
    labelTableWrap.classList.toggle('hidden');
    toggleTable.textContent = labelTableWrap.classList.contains('hidden')
      ? 'Gesamte Tabelle anzeigen'
      : 'Tabelle ausblenden';
  } catch (e) {
    lookupError.textContent = e.message;
  }
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) document.getElementById('preview').src = URL.createObjectURL(file);
});

runButton.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) { error.textContent = 'Bitte zuerst ein Bild auswaehlen.'; return; }
  error.textContent = '';
  runButton.disabled = true;
  runButton.textContent = 'Analysiere…';
  try {
    const body = new FormData();
    body.append('file', file);
    const response = await fetch('/predict', { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Unbekannter Fehler');

    document.getElementById('preview').src = data.annotated_image;
    document.getElementById('character').textContent = data.character;
    document.getElementById('outcome').textContent = data.outcome;

    const label = data.predicted_label || {};
    const finalSign = data.outcome === 'accepted_sign' ? data.predicted_character : '–';
    const rawSign = (data.raw_character && !['__REJECT__', '__background__'].includes(data.raw_character))
      ? data.raw_character
      : (data.raw_character === '__background__' ? 'Background' : '–');

    document.getElementById('finalCharacter').textContent = finalSign;
    document.getElementById('rawCharacter').textContent = rawSign;
    document.getElementById('datasetFolder').textContent = label.dataset_folder || '–';
    document.getElementById('score').textContent = data.score.toFixed(3);
    document.getElementById('threshold').textContent = data.threshold.toFixed(2);
    document.getElementById('hands').textContent = data.n_hands;
    document.getElementById('retry').textContent = data.retry_used ? 'ja' : 'nein';
    document.getElementById('category').textContent = label.category || data.category || '–';
    document.getElementById('vowel').textContent = label.vowel_target || data.vowel_component || '–';
    document.getElementById('consonant').textContent = label.consonant_target || data.consonant_component || '–';
    document.getElementById('reason').textContent = data.reason;
    document.getElementById('latency').textContent = data.latency_ms.toFixed(1) + ' ms';
  } catch (e) {
    error.textContent = e.message;
  } finally {
    runButton.disabled = false;
    runButton.textContent = 'Vorhersage starten';
  }
});
</script>
</body>
</html>
        """
    )
