# Methodische Grenzen und Quellen

## Abgrenzung
- Die Originalquelle legt die Zeichen-Ordner-Zuordnung fest. Die bereitgestellte
  Software verifiziert die Zuordnung nicht durch Raten der numerischen Reihenfolge.
- Identische Dateien und identische dekodierte Bilder werden getrennt erfasst.
  Gleiche grobe dHashes werden nur zur manuellen Prüfung vorgemerkt.
- Ein neuer Holdout ist gegen die dokumentierten bereits benutzten Datei- und
  Pixelinhalte abgegrenzt. Das beweist keine Unabhängigkeit von Personen,
  Aufnahmesitzungen oder nicht identischen Nachbarframes.
- Ausgeglichene Zeichenklassen ergeben keine gleich häufigen Komponenten oder
  Kategorien. Die aktiven Uyir-/Mei-Komponenten und FIST werden später separat bewertet.
- Mehr Daten, Gelenkwinkel und Augmentierung sind begründete Modellierungsentscheidungen,
  keine Garantie auf bessere Testergebnisse. Ein Vergleich ohne Augmentierung bleibt möglich.
- `FIST` bezeichnet ein fachliches Komponentenlabel. MediaPipe liefert keine
  tamilische Zeichenklasse; fehlende Detektion ist niemals ein FIST-Nachweis.
- Zwei erkannte Hände sind nicht automatisch zwei korrekt lokalisierte Hände.
  Visuelle Kontrollen und negative Background-Beispiele bleiben erforderlich.
- Die z-Koordinate ist eine geschätzte Tiefe in der Landmark-Darstellung.
  Fingerwinkel sind geometrische Merkmale, keine präzise biomechanische Messung.
- Bildaugmentierung bei festem MediaPipe erzeugt andere Merkmale, verbessert aber
  dessen Gewichte nicht. Detektionskonfiguration und Retry sind getrennte Entscheidungen.
- Der Retry verwendet keine Ground-Truth-Kategorie. Er muss in Training, Validation,
  finaler Evaluation und späterer API identisch angewendet werden.
- h0/h1 werden nach Handgelenk-x sortiert. Das ist eine technische Reihenfolge,
  keine automatische sprachliche Rollenerkennung. Spiegelungen bleiben ein offener
  Robustheitsaspekt; sie werden nicht als labelerhaltende Augmentierung angenommen.

## Quellen
1. Chirranjeavi M et al. TLFS23, Mendeley Data, Version 2, DOI 10.17632/39kzs5pxmk.2.
   https://data.mendeley.com/datasets/39kzs5pxmk/2
2. Bavesh Ram S et al. TLFS23 Tamil language fingerspelling dataset.
   Data in Brief 52 (2024), 109961. DOI 10.1016/j.dib.2023.109961. Tabelle 1.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10790027/
3. Python-Dokumentation: hashlib (SHA-256).
   https://docs.python.org/3.12/library/hashlib.html
4. Pillow: Image (verify, load, rotate) und ImageEnhance.
   https://pillow.readthedocs.io/en/stable/reference/Image.html
   https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html
5. Google AI Edge: Hand Landmarker für Python, IMAGE-Modus, Optionen und Koordinaten.
   https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
6. scikit-learn: Cross-validation, Datenaufteilung und Gruppen.
   https://scikit-learn.org/stable/modules/cross_validation.html

Die Implementierung und Merkmalsauswahl sind projektspezifisch. Die Quellen
belegen die Datensatzbeschreibung und verwendeten Schnittstellen, nicht die
Wirksamkeit der vorgeschlagenen Features auf TLFS23. Die eigene Prüfung und
Auswahl werden im Projekt dokumentiert. KI-Unterstützung ist in der Abgabe nach
den Vorgaben der Hochschule kenntlich zu machen.
