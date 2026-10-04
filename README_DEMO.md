# TLFS23 FastAPI Demo

Die lokale Demo bietet zwei Funktionen:

1. **Zeichen nachschlagen**: Tamil-Zeichen oder mehrere Zeichen eingeben. Die App zeigt Dataset-Ordner, Kategorie, Vokal-/Konsonanten-Komponente und – wenn die TLFS23-Rohdaten lokal vorhanden sind – ein deterministisches Referenzbild aus dem **Trainingssplit**.
2. **Eigenes Bild testen**: JPG/PNG/WebP hochladen und mit dem exportierten finalen Modell klassifizieren.

## Dateien kopieren

- `src/tlfs23/inference.py` -> `src/tlfs23/inference.py`
- `app/demo_api.py` -> `app/demo_api.py`
- `app/__init__.py` -> `app/__init__.py`
- `requirements-demo.txt` -> Projektroot

Der Export aus Notebook 07 muss unter `models/export/<model_version>/` liegen und `label_schema.csv` enthalten.

## Start

```bash
cd /Users/sharu/Documents/DBU/tamil-finger-sign-mlops
source .venv/bin/activate
python -m uvicorn app.demo_api:app --host 127.0.0.1 --port 8000
```

Dann im Browser:

- Demo: `http://127.0.0.1:8000`
- API-Dokumentation: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## Zusätzliche Endpoints

- `GET /labels` – komplette exportierte Label-Tabelle
- `GET /lookup?text=...` – ein oder mehrere Tamil-Zeichen nachschlagen
- `GET /reference-image?character=...` – ein Trainings-Referenzbild laden, sofern die Rohdaten lokal verfügbar sind
- `POST /predict` – eigenes Bild klassifizieren

Das Referenzbild stammt ausschließlich aus dem Trainingssplit; der Testsplit wird nicht zur Demo-Auswahl verwendet.


## Version 1.2: konkretes Zeichen in der Ergebnisanzeige

Die Bildvorhersage zeigt nun explizit:

- das finale akzeptierte Tamil-Zeichen,
- das vom Modell zusammengesetzte Roh-Zeichen vor dem Threshold,
- den zugehoerigen Dataset-Ordner aus `label_schema.csv`,
- Kategorie sowie Vokal-/Konsonanten-Komponenten.

Bei einem Reject bleibt das finale Zeichen leer, die Roh-Vorhersage kann aber weiterhin sichtbar sein.
