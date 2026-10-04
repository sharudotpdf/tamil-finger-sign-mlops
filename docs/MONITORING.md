# Monitoring und Laufzeit-Logging

## Zweck

Der Prototyp implementiert ein leichtgewichtiges Laufzeit-Monitoring für die lokale Tamil-Finger-Spelling-Inferenz-API.

Ziel ist es, das Verhalten der Inferenz nachvollziehbar zu machen, ohne eine umfangreiche produktive Monitoring-Infrastruktur aufzubauen.

## Laufzeit-Logging

Jede Vorhersageanfrage erzeugt einen strukturierten JSON-Logeintrag.

Der Logeintrag enthält unter anderem:

- Zeitstempel
- Request-ID
- Modellversion
- Ergebnis der Vorhersage
- Anzahl erkannter Hände
- Information darüber, ob ein erneuter Detektionsversuch genutzt wurde
- Konfidenzwert
- verwendeten Entscheidungsschwellenwert
- Grund für eine Ablehnung
- Inferenzlatenz
- Content-Type
- Größe der Eingabedatei

Die Laufzeit-Logs werden unter folgendem Pfad gespeichert:

`artifacts/tlfs23/runtime/inference.jsonl`

Das übermittelte Bild selbst wird nicht gespeichert.

## Monitoring-Kennzahlen

Aus den Laufzeit-Logs können verschiedene Kennzahlen abgeleitet werden.

### Anzahl der Anfragen

Anzahl der Inferenzanfragen über einen bestimmten Zeitraum.

### Fehlerrate

Anteil der Anfragen, bei denen ein technischer Fehler oder ein Fehler während der Inferenz auftritt.

### Ablehnungsrate

Anteil der Vorhersagen, die aufgrund des definierten Konfidenz-Schwellenwerts abgelehnt werden.

Ein deutlicher Anstieg kann darauf hinweisen, dass sich die eingehenden Daten von den Entwicklungsdaten unterscheiden oder die Vorhersagequalität abnimmt.

### Fehlerquote der Handerkennung

Anteil der Eingaben, bei denen keine verwertbaren Hand-Landmarks erkannt werden.

Diese Kennzahl ist besonders relevant, da das Klassifikationsmodell von einer erfolgreichen MediaPipe-Handerkennung abhängig ist.

### Inferenzlatenz

Dauer einer einzelnen Vorhersageanfrage.

Damit können mögliche Leistungsverschlechterungen der lokalen API erkannt werden.

### Verteilung der Konfidenzwerte

Veränderungen der Konfidenzwerte können darauf hinweisen, dass sich die eingehenden Daten von den Daten unterscheiden, mit denen das Modell entwickelt und evaluiert wurde.

## Drift-Indikatoren

Der Prototyp implementiert keine automatische statistische Drift-Erkennung.

Stattdessen können folgende Werte als operative Drift-Indikatoren genutzt werden:

- steigende Fehlerquote der Handerkennung
- steigende Ablehnungsrate
- sinkende Konfidenzwerte
- steigende Inferenzlatenz

Diese Kennzahlen beweisen keinen Model- oder Data-Drift, können aber darauf hinweisen, dass eine genauere Untersuchung oder erneute Modellevaluation notwendig ist.

## Datenschutz

Das Laufzeit-Monitoring speichert bewusst keine hochgeladenen Bilder.

Es werden ausschließlich technische und vorhersagebezogene Metadaten gespeichert, die für Monitoring und Fehleranalyse benötigt werden.

Dadurch wird die unnötige Speicherung potenziell sensibler Bilddaten vermieden.

## Umfang und Grenzen

Das Monitoring wurde für einen lokal ausführbaren MLOps-Prototypen entwickelt.

Für einen produktiven Einsatz könnte das Konzept beispielsweise um folgende Komponenten erweitert werden:

- zentrale Log-Speicherung
- Monitoring-Dashboards
- automatische Warnmeldungen
- statistische Drift-Erkennung
- Systeme wie Prometheus oder Grafana

Für den vorliegenden Prototypen steht jedoch die nachvollziehbare Erfassung zentraler Laufzeitinformationen im Vordergrund.
