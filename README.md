# Tamil Finger Spelling – MLOps-Prototyp

## Projektüberblick

Dieses Projekt implementiert einen lokal ausführbaren MLOps-Prototypen zur Erkennung von Tamil Finger Spelling auf Basis des TLFS23-Datensatzes.

Der Schwerpunkt liegt nicht ausschließlich auf der Modellgüte, sondern auf einer nachvollziehbaren ML-Pipeline mit reproduzierbarer Datenverarbeitung, Experiment-Tracking, Tests, CI, lokaler Bereitstellung und Laufzeit-Logging.

Die Verarbeitungskette umfasst:

1. Umgebungsprüfung
2. Dateninventur und Qualitätsprüfung
3. Duplikatprüfung und Kuration
4. Label-Schema und eingefrorenen Daten-Split
5. MediaPipe-Hand-Landmarks und Feature-Extraktion
6. Modellvergleich und Experiment-Tracking mit MLflow
7. finale Evaluation auf dem Holdout-Testsplit
8. lokalen Modell-Export und FastAPI-Demo

---

## Voraussetzungen

Empfohlen wird:

- Python 3.12
- Git
- eine lokale Python-Umgebung (`venv`)
- ausreichend Speicherplatz für den TLFS23-Datensatz

Die zentralen Python-Abhängigkeiten sind in `requirements.txt` aufgeführt.

Die im Projekt verwendete MediaPipe-Version ist auf `0.10.35` festgelegt.


---

## Projekt lokal einrichten

Repository klonen und in den Projektordner wechseln:

```bash
git clone https://github.com/sharudotpdf/tamil-finger-sign-mlops.git
cd tamil-finger-sign-mlops
```

Virtuelle Umgebung erstellen:

```bash
python3.12 -m venv .venv
```

Unter macOS/Linux aktivieren:

```bash
source .venv/bin/activate
```

Abhängigkeiten installieren:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Für die Ausführung der Notebooks in VS Code muss als Kernel die Python-Umgebung `.venv` ausgewählt werden.

---

## Datensatz

Für das Projekt wird der Datensatz **TLFS23 – Tamil Language Finger Spelling Image Dataset, Version 2** verwendet.

Die Rohdaten werden aufgrund ihrer Größe nicht im Git-Repository versioniert und müssen separat über Mendeley Data heruntergeladen werden:

[TLFS23 auf Mendeley Data](https://data.mendeley.com/datasets/39kzs5pxmk/2)

DOI: `10.17632/39kzs5pxmk.2`

Der Datensatz wird lokal standardmäßig unter folgendem Pfad erwartet:

```text
data/raw/TLFS23 - Tamil Language Finger Spelling Image Dataset 2/Dataset Folders/
---

### Lizenz und Quellenangabe

Der verwendete Datensatz **TLFS23 – Tamil Language Finger Spelling Image Dataset, Version 2** wurde über Mendeley Data veröffentlicht und steht unter der Lizenz **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

**Quelle:**

Chirranjeavi M, Bavesh Ram S, Gokulraj Varatharajan, Aaruran Sundaresh, Binoy Nair, Harikumar M E (2023):  
*TLFS23 – Tamil Language Finger Spelling Image Dataset*, Version 2, Mendeley Data.  
DOI: `10.17632/39kzs5pxmk.2`

**Lizenz:**

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

Die zugrunde liegenden Bilddaten sind nicht Bestandteil dieses Repositories. Die im Projekt erzeugten Manifeste, Features und Modellartefakte basieren auf einer lokal heruntergeladenen Kopie des Datensatzes.

## Datenverarbeitung ausführen

Die Notebooks werden in dieser Reihenfolge ausgeführt:

| Nr. | Notebook | Aufgabe |
|---|---|---|
| 00 | `00_environment_check (1).ipynb` | Python-Umgebung und Paketstände prüfen |
| 01 | `01_data_inventory (1).ipynb` | Datensatz inventarisieren |
| 02 | `02_data_quality_hash_audit_v2.ipynb` | Datei-, Pixel- und Hash-Prüfung |
| 03 | `03_duplicate_resolution_and_curation_v3.ipynb` | Duplikate behandeln und Daten kuratieren |
| 04 | `04_label_schema_and_split_v3.ipynb` | Label-Schema prüfen und Daten-Split einfrieren |
| 05 | `05_landmark_extraction_optimized.ipynb` | MediaPipe-Landmarks und 178 Merkmale extrahieren |
| 06 | `06_model_experiments_mlflow_v2.ipynb` | Modelle trainieren, vergleichen und mit MLflow protokollieren |
| 07 | `07_final_evaluation.ipynb` | finale Holdout-Evaluation und Modell-Export |

Die Notebooks sollten jeweils vollständig von oben nach unten ausgeführt werden.

Wichtige abgeleitete Daten werden unter anderem in folgenden Verzeichnissen gespeichert:

```text
data/manifests/tlfs23/
artifacts/tlfs23/
models/export/
```

Temporäre Caches und große reproduzierbare Zwischenartefakte werden über `.gitignore` ausgeschlossen.

---

## Modelltraining und Experiment-Tracking

Das Training und der Modellvergleich erfolgen in Notebook 06.

Verglichen werden mehrere Modellvarianten, darunter Dummy-Baseline, logistische Regression und Random Forest. Die Klassifikation ist strukturiert aufgebaut: Zeichenart, Vokal- und Konsonanten-Komponenten werden modelliert und anschließend deterministisch zum Tamil-Zeichen zusammengesetzt.

MLflow wird für das lokale Experiment-Tracking verwendet. Dabei werden unter anderem Parameter, Daten- und Codebezüge, Metriken sowie Modellartefakte protokolliert.

Die Modell- und Threshold-Auswahl erfolgt ausschließlich anhand der Validierungsdaten.

---

## Finale Evaluation

Die finale Evaluation erfolgt in Notebook 07 auf dem zuvor reservierten Holdout-Testsplit.

Der Testsplit wird erst nach Abschluss der Modell- und Threshold-Entscheidung verwendet. Ein Testzugriffs-Lock verhindert, dass die Modellwahl nachträglich anhand der Holdout-Ergebnisse angepasst wird.

Die finale Evaluation erzeugt unter anderem:

```text
artifacts/tlfs23/final_test/
```

Darin befinden sich beispielsweise Metriken, Klassifikationsbericht, Detection-Auswertung, Fehlerlisten und Konfusionsmatrizen.

Der exportierte Modellstand liegt unter:

```text
models/export/<model_version>/
```

und enthält das Modell, Label-Schema, Umgebungsinformationen, Prüfsummen und weitere für die Inferenz benötigte Dateien.

---

## Tests ausführen

Alle automatisierten Tests können im Projektroot ausgeführt werden:

```bash
source .venv/bin/activate
python -m pytest -q
```

Die Tests decken insbesondere folgende Bereiche ab:

- Daten- und Feature-Hilfsfunktionen
- Split- und Review-Logik
- Feature-Extraktion und Entwicklungsdaten-Bundles
- Inferenzfunktionen
- FastAPI-Endpunkte

Die Tests verwenden für isolierte Prüfungen künstliche Daten und simulierte Detektorausgaben. Sie ersetzen keine fachliche Evaluation des realen TLFS23-Datensatzes.

---

## Lokale Anwendung / API starten

Die lokale Demo basiert auf FastAPI.

Server starten:

```bash
source .venv/bin/activate
python -m uvicorn app.demo_api:app --host 127.0.0.1 --port 8000
```

Danach sind verfügbar:

- Demo: `http://127.0.0.1:8000`
- interaktive API-Dokumentation: `http://127.0.0.1:8000/docs`
- Health-Check: `http://127.0.0.1:8000/health`

Wichtige Endpunkte:

- `GET /health`
- `GET /labels`
- `GET /lookup?text=...`
- `GET /reference-image?character=...`
- `POST /predict`

Für `/predict` kann ein JPG-, PNG- oder WebP-Bild hochgeladen werden.

Referenzbilder werden ausschließlich aus dem Trainingssplit ausgewählt. Der Holdout-Testsplit wird nicht für die Demo verwendet.

---

## Laufzeit-Logging und Monitoring

Die API schreibt für Inferenzanfragen strukturierte JSONL-Logs nach:

```text
artifacts/tlfs23/runtime/inference.jsonl
```

Erfasst werden technische und vorhersagebezogene Metadaten, zum Beispiel:

- Zeitstempel
- Request-ID
- Modellversion
- Ergebnisstatus
- Anzahl erkannter Hände
- Retry-Nutzung der Handerkennung
- Konfidenzwert und Threshold
- Ablehnungsgrund
- Inferenzlatenz
- Eingabegröße

Das hochgeladene Bild selbst wird nicht gespeichert.

Das Monitoring-Konzept ist ausführlicher in `docs/MONITORING.md` beschrieben.

---

## CI/CD

Für die kontinuierliche Integration wird GitHub Actions verwendet.

Der Workflow liegt unter:

```text
.github/workflows/ci.yml
```

Er wird bei Pushes und Pull Requests auf `main` ausgeführt und führt unter anderem folgende Schritte durch:

1. Repository auschecken
2. Python 3.12 einrichten
3. Abhängigkeiten installieren
4. automatisierte Tests mit `pytest` ausführen
5. Import der FastAPI-Anwendung prüfen

Damit ist eine automatisierte CI-Prüfung des Repository-Stands umgesetzt.

Ein automatisches Deployment auf eine externe Produktionsumgebung ist bewusst nicht Bestandteil dieses lokalen Prototyps. Die Bereitstellung erfolgt lokal über FastAPI und über den versionierten Modell-Export.

---

## Reproduzierbarkeit

Für die Reproduzierbarkeit werden unter anderem folgende Mechanismen verwendet:

- feste Notebook-Reihenfolge
- reproduzierbare Datenmanifeste
- Hash-basierte Datenprüfung
- eingefrorener Split
- dokumentierte Label-Zuordnung
- definierte MediaPipe-Konfiguration
- Experiment-Tracking mit MLflow
- versionierte Modell- und Evaluationsartefakte
- automatisierte Tests
- GitHub Actions CI
- Prüfsummen im Modell-Export
- strukturierte Laufzeit-Logs

Die Rohdaten werden nicht im Repository gespeichert und müssen separat bereitgestellt werden.

---

## Zentrale Projektstruktur

```text
app/                       FastAPI-Anwendung
artifacts/                 Prüf-, Evaluations- und Laufzeitartefakte
configs/                   geprüfte Konfigurationen
 data/
   manifests/              Daten-, Kurations- und Splitmanifeste
   raw/                    lokale Rohdaten, nicht in Git
 docs/                     technische Dokumentation
 models/
   detector/               MediaPipe-Hand-Landmarker
   export/                 finaler Modell-Export
 notebooks/                Pipeline 00–07
 src/tlfs23/               wiederverwendbarer Python-Code
 tests/                    Unit- und Integrationstests
.github/workflows/         GitHub Actions CI
requirements.txt           direkte Projektabhängigkeiten
```

---

## Hinweise zur Abgabe

Für die LMS-Abgabe wurde eine ZIP bereitgestellt.


Bewertungsrelevante Konfigurationen, Manifeste, Tests, Quellcode, finale Evaluationsergebnisse und der benötigte Modell-Export sind als Bestandteil der Abgabe geblieben.
