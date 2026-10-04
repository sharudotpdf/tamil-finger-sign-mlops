# Bildprüfung in Notebook 03 und 04

## Einfügen
1. Die vorhandenen Notebooks sichern. Die beiden Dateien unter `notebooks/` ersetzen.
2. `src/tlfs23/visual_review.py` in den bestehenden Ordner `src/tlfs23/` kopieren.
   `data.py`, `common.py`, `features.py` und `__init__.py` nicht ersetzen.
3. Optional die neue Testdatei unter `tests/` ergänzen.
4. Kernel neu starten und Notebook 03, danach 04 von oben ausführen.

Die gelieferten Notebooks enthalten keine vorweggenommenen Ergebnisse. Der Rohdatenbestand,
die Curation-Regeln, Trainingsumfänge und Detektoreinstellungen werden nicht geändert.
Es sind keine neuen Pakete und keine Widgets erforderlich. Audit- oder Feature-Caches nicht löschen.

## Notebook 03
- Identische Inhalte werden paarweise mit Datei- und Pixel-Hashes angezeigt.
- `DHASH_PAIRS_TO_SHOW = 6` begrenzt die Bildanzahl.
- `DHASH_START = 0` kann für weitere Gruppen erhöht werden.
- Die bestehende Prüfung sucht gleiche dHashes innerhalb desselben Ordners: Distanz 0.
  Sie ist keine vollständige Suche über positive Hamming-Distanzen.
- Optional eigene Einträge in `DHASH_NOTES` ergänzen; frühere Notizen bleiben erhalten.
  Kein Bild wird durch eine solche Notiz automatisch ausgeschlossen.

## Notebook 04
1. `FOLDER_TO_CHARACTER` enthält ausschließlich eigene, referenzgeprüfte Korrekturen.
2. Die Bildgalerie kommt vor `CONFIRM_FOLDER_MAPPING`. Beschriftungen geben den Eintrag
   der Prüftabelle wieder, nicht eine automatisch verifizierte Zeichenklasse.
3. `REVIEW_FOLDERS = []` und `REVIEW_PAGE = None` zeigen eine Auswahl aus den Kategorien.
   Für bestimmte Ordner: `REVIEW_FOLDERS = ["1", "2", "247", "Background"]`.
   Für alle Ordner seitenweise: Liste leer lassen und `REVIEW_PAGE = 0`, dann 1, 2 usw.
4. Erst nach dem Quellenabgleich aller verwendeten Labels: `CONFIRM_FOLDER_MAPPING = True`
   und `MAPPING_EVIDENCE` mit dem tatsächlichen Beleg füllen.
5. Die historische Dateiliste in `EXPOSURE_FILES` beibehalten/ergänzen. Anschließend
   `EXPOSURE_REVIEWED = True` setzen und `SPLIT_NOTE` ausfüllen. Frühere Datenverwendung
   nicht mit `NO_PREVIOUS_EXPOSURE = True` übergehen.
6. Die Handstruktur-Galerie zeigt nur Trainingsbilder mit Zeichen und Komponenten.
   Danach eigene Beobachtungen unter `HAND_REVIEW_NOTE` eintragen und erst dann
   `CONFIRM_HAND_STRUCTURE = True` setzen.
7. `Bereit für Notebook 05: True` bestätigt die erforderlichen Dateien/Freigaben.
   In Notebook 05 beginnt weiterhin der eigene MediaPipe-Pilot; dessen Freigaben werden nicht gesetzt.

## Sichtproben und Holdout
Vor der Datenaufteilung angebotene Bilder werden unter
`data/manifests/tlfs23/visual_review_exposure.csv` festgehalten. Beim ersten Split wird daraus
ein unveränderlicher Expositions-Snapshot erzeugt und zusätzlich zu den historischen Manifesten
berücksichtigt. So landen diese Bildinhalte nicht neu in Validation oder Holdout.
Nach einer vorhandenen Split-Freigabe werden diese beiden Teilmengen nicht als Review-Bilder gezeigt.
Ein bestehender Split wird nur konsistent geladen, nicht still neu gezogen. Bei einem Konflikt
werden die Daten nicht überschrieben. Die betreffenden Dateien nicht einfach löschen.

## Ergebnisdateien
- `artifacts/tlfs23/reviews/03_dhash_review.csv`: offene/eigene Paarbewertungen.
- `artifacts/tlfs23/reviews/*.html`: lokal gespeicherte Bildgalerien.
- `artifacts/tlfs23/reviews/04_mapping_examples.csv`: angezeigte Mapping-Beispiele.
- `artifacts/tlfs23/reviews/04_mapping_review.json`: selbst bestätigter Quellenabgleich.
- `artifacts/tlfs23/reviews/04_hand_examples.csv`: konkrete Handstruktur-Stichprobe.
- Unveränderte Schnittstellen: `split_manifest.csv`, `04_split.json`, `04_hand_review.json`.

Die HTML-Galerien enthalten eingebettete Bildkopien. Sie sind lokale Prüfarbeitsdateien,
keine Trainingsdaten. Vor Weitergabe gegebenenfalls Datensatz-Lizenz und Quellenangabe beachten.

## Prüfung
Im Projektordner und in der aktivierten Umgebung:

```bash
python -m pytest tests/test_tlfs23_visual_review.py -q
```

Die neue Datei enthält 19 Tests. Zusätzlich wurden die 44 vorhandenen Tests aus dem
Paket 02–05 mit den beiden ersetzten Notebooks ausgeführt: insgesamt 63 erfolgreich.
Der Gesamtdurchlauf verwendet künstliche Bilder und simulierte MediaPipe-Ausgaben.
Es wurden keine realen TLFS23-Labels oder Handgesten durch diese Tests fachlich bestätigt.
