# TLFS23 - Notebooks 02 bis 05

## Einsetzen
Die vier Notebooks nach `notebooks/` kopieren. Den gesamten Unterordner
`src/tlfs23/` nach `src/` kopieren und `tests/` ergänzen. Bestehende Dateien wie
`tlfs23_workflow.py`, Rohbilder, Modelle und frühere Auswertungen nicht löschen.
Die Notebooks 00 und 01 bleiben unverändert.

Im Projektordner, in der bestehenden `.venv`:

```bash
source .venv/bin/activate
python -m pip check
python -m pytest tests/test_tlfs23_data_features.py -q
```

Die benötigten Bibliotheken sind in `requirements_02_05.txt` aufgeführt.
Nur bei fehlenden Paketen installieren. Die Datei ist kein vollständiges Lockfile.
Verwendet wird die bereits getestete Projektkombination Python 3.12 und
MediaPipe 0.10.35. Bei Moduländerungen den Notebook-Kernel neu starten.

## Eingaben aus Notebook 01
- `data/manifests/file_inventory.csv`
- `artifacts/data_audit/01_inventory_summary.json`
- unveränderte Bilder im darin angegebenen Klassenordner

## Reihenfolge
| Notebook | Zentrale Ausgaben |
|---|---|
| 02 | `data/manifests/tlfs23/image_audit.csv`, `artifacts/tlfs23/02_audit.json` |
| 03 | `data/manifests/tlfs23/curation_manifest.csv`, `duplicate_groups.csv`, `near_duplicate_candidates.csv` |
| 04 | `configs/tlfs23/label_review.csv`, `data/manifests/tlfs23/split_manifest.csv`, `artifacts/tlfs23/04_split.json`, `04_hand_review.json` |
| 05 | Feature-Bundles unter `artifacts/tlfs23/feature_runs/`, Verweis `artifacts/tlfs23/development_features.json` |

Die abgeleiteten Daten werden getrennt von `artifacts/ml_pipeline/` und den
bisherigen Manifesten gespeichert. Gleiche Dateinamen der Notebooks vor dem
Ersetzen sichern. Einen eingefrorenen Split nicht nach Betrachtung des Holdouts
verändern.

### Notebook 02
Zuerst `AUDIT_MODE = "smoke"`, danach `"full"`.
Pro Bild wird vollständig dekodiert; beim Wiederholen werden unveränderte
Dateien nicht erneut dekodiert. Der Dateihash wird trotzdem erneut berechnet.
Die Prüfung nutzt einen SQLite-Cache und schreibt nur abgeleitete Dateien.
Pixelidentität bezieht sich auf RGB nach EXIF-Ausrichtung. Unterschiedliche
Bildbytes können dieselben Pixel ergeben; gleiche dHashes dagegen beweisen keine
Pixelidentität.

### Notebook 03
Die Curation behandelt sowohl gleiche Bytes als auch identische RGB-Pixel.
Deshalb müssen die Zahlen nicht exakt denen eines reinen Datei-Hash-Audits
entsprechen. Ein grober dHash wird nur als Prüfhinweis verwendet. Er ersetzt
weder eine Near-Duplicate-Prüfung noch Signer-/Session-IDs.

### Notebook 04
Vorhandene Einträge aus `configs/label_schema_review.csv` werden übernommen.
Provisorische Zuordnungen bleiben unbestätigt. Fehlende Zuordnungen anhand der
Originalreferenz ausfüllen; keine Ordnernummern aus der Reihenfolge erraten.
`FOLDER_TO_CHARACTER` erlaubt gezielte Korrekturen im Notebook. Kategorien und
Komponenten werden erst aus den Zeichen abgeleitet.

Wichtig für dieses Projekt: `EXPOSURE_FILES` muss alle bereits benutzten Bilder
abdecken. Die vorhandene `data/manifests/labelled_split_manifest.csv` enthält
den bisherigen ausgewählten Bestand. Weitere Pilotbilder außerhalb dieses
Bestands gegebenenfalls durch weitere Manifeste mit SHA-256 ergänzen. Keinen
leeren historischen Nachweis behaupten. Diese Angaben sind für die unabhängige
Holdout-Reservierung nötig; sie werden nicht durch eine neu formulierte
Darstellung der Notebooks entbehrlich.

Standard: 180 Trainingsbilder je Uyir-Mei-Klasse und 360 je Uyir-, Mei-, Ayudha-
und Background-Klasse. Validation und Holdout: jeweils 50 Bilder je Zeichen und
100 Background-Bilder. Der Code bricht ab, wenn nicht genügend unabhängige
Kandidaten vorliegen. Umfang dann vor Freigabe begründet festlegen.

Verifizierte Gruppen-IDs können als CSV mit `relative_path,group_id` übergeben
werden. Sie müssen alle Kandidaten abdecken. Ohne solche IDs bleibt die
Auswertung bildbezogen, auch wenn Datei- und Pixelduplikate ausgeschlossen sind.
Visuelle Handprüfung erst nach der Reservierung und nur auf Trainingsbildern.

### Notebook 05
Die vorhandene Datei `models/detector/hand_landmarker_v1.task` wird über ihren
im Projekt bekannten SHA-256 identifiziert. Es erfolgt kein automatischer Download.
Der initiale Test läuft in einem separaten Prozess mit Zeitlimit. Danach werden
höchstens 60 Trainingsbilder pro Konfiguration ausgewertet. Die Preview-Schritte
können durch ihren Cache wiederholt werden.

Erst nach visueller und quantitativer Prüfung `APPROVE_EXTRACTOR = True`
und eine konkrete `FEATURE_NOTE` eintragen. Danach kann `RUN_DEVELOPMENT = True`
gesetzt werden. Die Standard-Extraktion nutzt Blöcke mit 128 Bildern und maximal
300 Sekunden je Block. `timeout` ist ein Zeitlimit, keine erwartete Laufzeit.
Bei einem Abbruch bleibt der Notebook-Kernel erhalten; abgeschlossene Blöcke
bleiben gespeichert. Nicht mehrere Extraktionen derselben Konfiguration starten.
Bei einem Blockfehler das zugehörige `.log` prüfen, nicht den gesamten Cache löschen.

Ein geänderter Detektor, Code oder Augmentierungsparameter erhält eigene Cache-
Schlüssel. Die Freigabe bleibt an den Split gebunden. Ein bereits freigegebenes
Setup zuerst archivieren, bevor es absichtlich ersetzt wird.

Die Verarbeitung erzeugt 178 Merkmale: 87 je Hand plus 4 relative Paarmerkmale.
Die Gelenkwinkel werden als Kosinuswerte gespeichert. Fehlende/ungültige
Geometrie ist `NaN`; `feature_valid` und `status` bestimmen die weitere Nutzung.
Für die End-to-End-Evaluation dürfen fehlgeschlagene Zeilen nicht verschwinden.

Die optionale milde Variante betrifft nur Uyir/Mei im Training. Jedes Original
bleibt erhalten. `variant`, `sample_id` und Ausgangs-SHA dokumentieren die
Herkunft. Validation und Holdout werden nicht augmentiert. Es werden keine
physisch modifizierten Rohbilder gespeichert. MediaPipe selbst wird nicht trainiert.

## Anschluss an Notebook 06
Die neuen Artefakte sind nicht mit den bisherigen 129-dimensionalen Modellen
kompatibel. Das bisherige Notebook 06/07 deshalb noch nicht auf diesen Dateien
starten. Der vorgesehene Loader ist:

```python
from tlfs23 import features as f
rows, X, metadata = f.load_development(PROJECT_ROOT)
```

`variant == "original"` erlaubt den unveränderten Trainingsvergleich,
zusätzliche `mild_1`-Zeilen das Augmentierungsexperiment. Alle Validierungs-
metriken müssen auch Fehlversuche im Nenner behalten. Kategorie-Gewichtung,
Klassifikatorwahl und Thresholds werden in Notebook 06 festgelegt, nicht anhand
der Holdout-Ergebnisse.

## Git und Abgabe
Rohdaten, SQLite-Caches, Feature-Arrays, temporäre Dateien und Worker-Logs nicht
versehentlich zu Git hinzufügen. Prüfprotokolle, Konfigurationen, Quellcode und
Tests bleiben versionierbar. Die LMS-ZIP muss am Ende trotzdem alle für die
Bewertung benötigten Artefakte oder eine vollständige Erzeugungsanleitung enthalten.

## Validierung
Siehe `docs/VALIDIERUNG.json`. Die Tests benutzen künstliche Bilder, Hashes,
Split-Daten und simulierte Handlandmarks. Sie behaupten weder eine bestimmte
Detektionsgüte noch einen erfolgreichen Live-Lauf auf dem Mac.
