# TLFS23 – Notebooks 04–07

Ergänzung zu deinem bestehenden lokalen Projekt. Die Notebooks 00–03 und alle Rohdaten bleiben unverändert.

## Einfügen

Kopiere **den Inhalt** dieses Pakets in deinen Projekt-Hauptordner. Die Unterordner werden zusammengeführt, nicht ersetzt:

```text
notebooks/04_label_schema_and_hand_structure.ipynb
notebooks/05_landmark_extraction.ipynb
notebooks/06_model_experiments_mlflow.ipynb
notebooks/07_final_evaluation.ipynb
src/tlfs23_workflow.py
tests/test_tlfs23_workflow.py
requirements_04_07.txt
docs/METHODEN_04_07.md
```

Die Funktionen im neuen Python-Modul verhindern doppelte Implementierungen von Vorverarbeitung,
Komposition und Inferenz. Sie werden auch von Tests und einer späteren API verwendet.
Das Modul wird aus den Notebooks automatisch über `src/` importiert. Keine Bearbeitung vorhandener Module notwendig.

## Voraussetzung aus Notebook 03

- `data/manifests/curation_manifest.csv`
- `artifacts/data_audit/01_inventory_summary.json`
- Originalbilder unter `data/raw/TLFS23 - Tamil Language Finger Spelling Image Dataset 2/Dataset Folders/`

Abweichender Speicherort: Umgebungsvariable `TLFS23_CLASS_ROOT` oder vorhandener Pfad aus Notebook 01.
Es werden ausschließlich freigegebene, lesbare und hash-eindeutige Kandidaten verwendet.

## Bestehende Umgebung verwenden

Im VS-Code-Terminal im Projektordner:

```bash
source .venv/bin/activate
python --version
python -m pip install -r requirements_04_07.txt
python -m pip check
python -m pytest tests/test_tlfs23_workflow.py -q
```

Keine neue virtuelle Umgebung, kein TensorFlow, keine Kaggle-API. Die Versionsbereiche sind
Installationsvoraussetzungen und **kein fertiges, auf deinem Mac validiertes Lockfile**.
Nach erfolgreichem lokalen Test den tatsächlichen Stand festhalten:

```bash
python -m pip freeze > requirements-macos-snapshot.txt
```

Jedes Notebook in VS Code mit deiner Python-3.12.12-Umgebung öffnen, von oben ausführen.
Jupyter Lab muss nicht separat gestartet werden.

## Reihenfolge und Freigaben

### 04 – zuerst die Labelzuordnung

1. Kuratiertes Manifest und kanonische Tamil-Tabelle lesen.
2. Die Stichprobe und den Split festlegen: standardmäßig bis zu 120 Bilder je Ordner.
3. **Nur Trainingsbeispiele** visuell betrachten.
4. `configs/label_schema_review.csv` ausfüllen: `character`, `verified`, `evidence`.
   Numerische Ordner werden nicht still den ersten Zeichen zugeordnet. Verwende Tabelle 1 oder die
   mitgelieferten Originalreferenzen deines Datensatzes. Background heißt `__background__`.
5. Die Faustregel und Ayudha-Darstellung prüfen, eine Review-Notiz schreiben, Freigabe aktivieren.

Die genaue Ordner-Zeichen-Zuordnung wurde hier **nicht** als erledigt vorgetäuscht.
Wir haben deinen CSV-Inhalt bzw. die lokalen Referenzbilder nicht vorliegen. Bei offenen Zuordnungen
bleibt das Training gesperrt. Du kannst die Vorlage hier teilen, damit die Zuordnung vor der Freigabe gemeinsam geprüft wird.

Die Eingabe `INCLUDE_FOLDERS=None` verwendet alle kuratierten Ordner. Eine begrenzte Liste ist vor dem
allerersten Split ebenfalls möglich; sie muss Background sowie Vokale, Konsonanten und einige Kombinationen enthalten.
Ein reiner Basiszeichenversuch würde die Zwei-Hand-Komposition nicht hinreichend prüfen.

### 05 – Detektor-Pilot vor umfangreicher Extraktion

Einmaliger Download des kleinen offiziellen Hand-Landmarker-Modells (nicht der Rohdaten).
Der erste Pilot verarbeitet drei Trainingsbilder je Ordner. Prüfe die gezeigten Landmark-Punkte.
`APPROVE_LANDMARK_APPROACH` startet erst danach die Train-/Validation-Extraktion.
Bei schlechten Ergebnissen **stoppen und den Ansatz prüfen**; keine schwierigen Beispiele heimlich entfernen.
Der Cache erlaubt Wiederaufnahme. Rohbilder werden einzeln geladen, Featureblöcke auf SSD gespeichert.
Testbilder werden hier nicht verarbeitet.

### 06 – drei begründete Modellvarianten

Dummy, logistische Regression und ressourcenbegrenzter Random Forest; keine breite Hyperparametersuche.
Pro Variante drei Klassifikatoren: Zeichenart, Vokal und Konsonant. Dies ist strukturierte Klassifikation,
kein neuronales Multi-Task-Training. Die Zusammensetzung erfolgt deterministisch.

MLflow nutzt eine lokale SQLite-Datei und protokolliert Parameter, Code, Datenversionen, Artefakte und
Validierungsmetriken. Eine volle Model Registry wird nicht behauptet. Die Wahl des Modells und des Score-Grenzwerts
verwendet nur Validation. Anschließend die Modellentscheidung im Notebook dokumentieren.

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/ml_pipeline/mlflow.db --port 5001
```

### 07 – einmaliger finaler Test und Export

`RUN_FINAL_TEST=True` erst bei festgelegten Entscheidungen. Detektionsfehler bleiben Teil der Evaluation.
Komponenten-Konfusionsmatrizen, Fehlerlisten und das Inferenzpaket werden erzeugt.
Ein erneuter Aufruf zeigt dieselben gespeicherten Ergebnisse; eine geänderte Auswahl wird blockiert.
Der freiwillige lokale Latenztest ist keine iPhone-Messung.

## Wichtige Grenzen

- Eine Faust ist nicht dasselbe wie eine fehlende Hand. Bei unvollständiger Erkennung wird zurückgewiesen.
- Die aktuelle Pipeline setzt für akzeptierte Zeichen **zwei detektierte Hände** voraus. Ist das für Ayudha
  oder andere Klassen ungeeignet, muss der Scope/Ansatz vor dem abschließenden Training geändert werden.
- Ayudha hat eine eigene Kategorie. `FIST + FIST` wird nicht allgemein als Background/Idle interpretiert.
- Ohne verifizierte Personen-/Session-IDs: Bildsplit, keine nachgewiesene Generalisierung auf neue Personen.
- Die Klassenstichprobe ist nicht die vollständige Datenbasis oder ihre natürliche Häufigkeitsverteilung.
- Es wird **nicht** geprüft, ob nie trainierte Vokal-Konsonanten-Paare erkannt werden.
- Modell-Scores sind nicht kalibriert. Auch hohe Scores können falsch sein. Background-Tests decken nicht
  sämtliche unbekannten Gesten ab. Es gibt keine automatische Produktionsfreigabe.
- Keine Aussage über Lernerfolg oder standardisierte Gebärdensprache.

## Git und formale Abgabe

Folgendes an die bestehende `.gitignore` anhängen:

```gitignore
# Recomputable caches and large local artifacts
artifacts/ml_pipeline/
models/detector/
models/export/
```

Versionieren: Notebooks, Quellcode, Tests, Konfigurationen und geprüfte Label-/Splitmanifeste.
Prüfe vor jedem Commit `git status`. Rohbilder und lokale Umgebungen bleiben außerhalb von Git.
Die finale **LMS-ZIP muss trotz .gitignore alle bewertungsrelevanten Ergebnisse enthalten**,
insbesondere das benötigte Modellpaket, Beispiele und die Startanleitung. Ein reiner GitHub-Quellcodeexport
ist dafür möglicherweise nicht ausreichend.

## Was danach noch kommt

Die vier Notebooks decken Labels, Features, Training, Tracking und Evaluation ab.
Lokale FastAPI, API-Integrationstests, GitHub Actions mit Bereitstellung, Logging/Monitoring,
Zielarchitektur, README-Abnahme und Reflexionsreport folgen separat. Kein weiteres Analyse-Notebook ist dafür erforderlich.

## Prüfung dieses Pakets

Siehe `docs/VALIDIERUNG.json`. Die Tests nutzen künstliche Bilder und simulierte Detektorausgaben;
das Klassifikatortraining und das Speichern/Laden werden mit echtem scikit-learn ausgeführt.
Kein echter TLFS23-Lauf, kein Mac-Hardwaretest, kein Live-MediaPipe- oder MLflow-Integrationstest wurde hier vorgetäuscht.
