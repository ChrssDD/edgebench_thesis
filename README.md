
# Methodik und Messprotokoll (EdgeBench)

Dieses Dokument definiert das Messverfahren, mit dem die Abfrageperformance von SQLite und DuckDB auf einem ressourcenbeschränkten Edge-Knoten reproduzierbar verglichen wird. Es beschreibt (i) Fairness- und Vergleichsregeln, (ii) Datenbasis, (iii) Workloads und Query-Semantik, (iv) Messablauf und Kennzahlen sowie (v) Logging, Reproduzierbarkeit und Validitätsrisiken.

---

## 1 Ziel und Vergleichsregeln

### 1.1 Untersuchungsziel
Verglichen wird die Latenz typischer Zeitreihen-Analytik-Queries aus der Gebäude- und Prozessautomatisierung auf einem Edge-Gerät. Der Fokus liegt auf Abfragezeiten unter verschiedenen Datenmengen, Tag-Anzahlen, Cache-Zuständen und Ressourcenprofilen.

### 1.2 Fairnessprinzipien
Der Vergleich gilt nur dann als interpretierbar, wenn folgende Regeln eingehalten werden:

1. **Semantische Äquivalenz**  
   SQLite und DuckDB müssen für jede Query inhaltlich gleiche Ergebnisse liefern (innerhalb definierter Toleranzen bei Gleitkommawerten). “Gleiches Problem” bedeutet gleiches Ergebnis auf gleicher Datenbasis, nicht identischer SQL-Text.

2. **Konstante physische Randbedingungen**  
   Gleiche Hardware, gleiche Storage-Anbindung, gleiche OS- und Kernel-Version, gleiche Datendateien, identische Tag-Anzahlen und Datenraten pro Messzelle.

3. **Isolierter Messpfad**  
   Pro Iteration wird genau die Query-Latenz gemessen. Setup- und Validierungsschritte werden außerhalb der Zeitmessung durchgeführt und separat geloggt.

4. **Cache-Regime als kontrollierter Faktor**  
   Warm- und Cold-Szenarien sind getrennte Versuchsbedingungen. Im Cold-Fall wird der Cachezustand vor jeder Iteration aktiv zurückgesetzt (oder alternativ vor jedem Block, wenn blockweise Kälte definiert ist). Warm-Fall bedeutet stabiler Page-Cache ohne Drops.

5. **Engine-spezifische Konfigurationen sind offen gelegt**  
   SQLite: Journaling-Mode, Synchronisation, Indizes.  
   DuckDB: Threads-Modus, Table-Scan vs Import.  
   Alles, was die Latenz beeinflusst, wird versioniert und im Ergebnisartefakt dokumentiert.

---

## 2 Testumgebung

### 2.1 Hardware und Betriebssystem
- Gerät: Raspberry Pi 5 (aarch64)
- OS: Debian 12 (bookworm)
- Kernel: 6.12.34+rpt-rpi-2712
- Storage: externe SSD via USB (UASP aktiv), ext4, Mount mit `noatime`
- PSI: Linux Pressure Stall Information ist verfügbar über `/proc/pressure/{cpu,memory,io}`

### 2.2 Messstabilisierung
- Messungen erfolgen im Leerlauf ohne interaktive Last
- Optionale Fixierung der CPU-Frequenz bzw. Protokollierung von Frequenz und Temperatur pro Iteration
- Wiederholungen und robuste Aggregation (Median, p95) zur Reduktion von Ausreißereinfluss

---

## 3 Datenbasis

### 3.1 Datenmodell (logisches Schema)
Zeitreihendaten werden als Messpunkte modelliert:

- `ts_ns`: Zeitstempel in Nanosekunden (Integer, monoton steigend)
- `tag`: Sensor- oder Kanal-ID (String oder Integer-Kategorie)
- `value`: Messwert (Float)
- optional: `quality`, `meta`, weitere Attribute je nach Generator

### 3.2 Datengenerierung und Determinismus
Die synthetische Datenbasis dient der kontrollierten Skalierung entlang zweier Achsen:

- **rows_per_tag**: Anzahl Messpunkte pro Tag
- **tags**: Anzahl paralleler Zeitreihen

Determinismus wird durch festen Seed und festen Referenzstart (`t0_ns`) hergestellt, sodass Datensätze identisch reproduziert werden können.

### 3.3 Dateiformate und Ladepfade
- SQLite: Laden in eine DB-Datei (WAL-Konfiguration, Indexpfade optional)
- DuckDB: Import aus Parquet oder direkter Scan (je nach Versuchsbedingung)

---

## 4 Workloads und Query-Semantik

### 4.1 Notation und Zeitsemantik
- Daten liegen als diskrete Samples vor, unregelmäßige Effekte können durch Jitter im Generator modelliert werden.
- Zeitfenster werden relativ zu einer Referenzzeit `t_ref` definiert.
- “Letzte N Minuten” bedeutet Intervall `(t_ref - N, t_ref]` in Nanosekunden, auf `ts_ns` angewandt.

### 4.2 Query-Klassen
Die Queries bilden drei typische Klassen ab:

1. **Vorverdichtung / Downsampling**  
   Gruppierung in Zeit-Buckets und Aggregation (z. B. Mittelwert).

2. **Gleitende Auswertung**  
   Aggregation über ein fixes Fenster relativ zu `t_ref` (z. B. 15 Minuten, 60 Minuten).

3. **Fensterbasierte Kennzahlen (Tumbling Window)**  
   Partitionierung in nicht überlappende Fenster (z. B. 10 Minuten) und Berechnung von Mittelwert und p95 pro Fenster.

---

## 5 Query-Definitionen (kanonisch)

Wichtig: Die folgende Spezifikation ist die “kanonische Semantik”. Die SQL-Implementierungen dürfen engine-spezifisch sein, solange die Semantik erfüllt wird.

### Q1 Downsample 5 Minuten (Bucket-Aggregation)
**Zweck:** Zeitliche Vorverdichtung, typisch für Trendanzeigen und Historian-Downsampling.

**Eingaben:**
- Tag-Menge `T` (z. B. alle Tags oder ein Subset)
- Zeitbereich `[t_start, t_end]`
- Bucket-Größe `B = 5 min`

**Ausgabe:**
- Pro `tag` und Bucket: `bucket_start`, `avg(value)`, optional `count`

**Semantik (formal):**
- Bucket-Index `b = floor((ts_ns - t0_ns) / B_ns)`
- Gruppierung nach `(tag, b)`

**Hinweis zur Performance-Erwartung:**
- Scan-dominant, profitiert von vektorisierten Aggregationen und effizientem Group-By.
- Indexnutzung ist typischerweise begrenzt, wenn ein großer Bereich gescannt wird.

---

### Q2 Last 15 Minuten Average (Endfenster-Aggregation)
**Zweck:** “Letzter Zustand” oder kurzfristige Glättung je Tag.

**Eingaben:**
- `t_ref`
- Fenster `W = 15 min`

**Ausgabe:**
- Pro Tag: `avg(value)` über `(t_ref - W, t_ref]`

**Semantik (formal):**
Für jedes `tag` gilt:
- `AVG_{tag,15m} = mean({ value | tag = tag_i AND ts_ns in (t_ref-W, t_ref] })`

**Hinweis zur Performance-Erwartung:**
- Selektiv, häufig indexfreundlich (Index auf `(tag, ts_ns)`).
- SQLite kann hier über Indexpfade stark profitieren.

---

### Q3 Last 60 Minuten Average (Endfenster-Aggregation)
Wie Q2, aber `W = 60 min`.  
Erwartung: Mehr Daten im Fenster, daher höherer Scananteil, Indexvorteil kann abnehmen.

---

### Q4 Tumbling 10 Minuten: Mean und p95 (Fenster-Report)
**Zweck:** Periodisches Reporting mit robusten Kennzahlen je Zeitfenster.

**Eingaben:**
- Zeitbereich `[t_start, t_end]`
- Fenstergröße `B = 10 min`

**Ausgabe:**
- Pro Tag und Fenster: `bucket_start`, `mean(value)`, `p95(value)`

**Semantik (Bucket):**
- Bucket wie in Q1, aber `B = 10 min`

**p95-Definition (diskret):**
- Für ein Fenster mit `n` Werten, sortiert aufsteigend `x[0..n-1]`
- Diskretes p95: `x[k]` mit `k = ceil(0.95 * n) - 1`, gekappt auf `[0, n-1]`

**Hinweis zur Engine-Ausrichtung:**
- DuckDB bietet Perzentilfunktionen, die je nach Funktion und Default-Interpolation anders sein können.
- Für Fairness muss in beiden Engines dieselbe diskrete Definition umgesetzt werden.

---

## 6 Messablauf

### 6.1 Experimentzelle und Faktoren
Eine Messzelle ist durch folgende Parameter eindeutig definiert:

- Engine: SQLite oder DuckDB
- Workload: Q1–Q4
- Datenmaß: `rows_per_tag` und `tags` (oder Gesamtzeilen, aber Achse muss klar dokumentiert sein)
- Cache-Regime: warm oder cold
- Threads: z. B. `t1` oder `auto` (DuckDB)
- Indexprofil (SQLite): z. B. Basisindex vs zusätzlicher `(tag, ts_ns)` Index
- RAM-Profil: Baseline oder Cap (falls cgroup-Limit aktiv)
- Ladepfad: DuckDB Import vs File-Scan (falls getestet)

### 6.2 Wiederholungen und Iterationen
- Pro Messzelle werden `R` Runs durchgeführt (z. B. 20)
- Pro Run werden `I` Iterationen gemessen (z. B. 10)
- Pro Iteration: Query ausführen, Latenz messen, Systemmetriken sampeln

Robuste Kennzahlen werden pro Run aus den `I` Iterationen berechnet und anschließend über Runs aggregiert.

### 6.3 Warm vs Cold (operativ)
**Warm:**
- Vor dem Run wird ein definierter “Warm-up” durchgeführt (z. B. 1–3 Ausführungen), die nicht in die Messung eingehen.
- Danach werden Iterationen ohne Cache-Drops gemessen.

**Cold:**
- Vor jeder Iteration wird der Cachezustand zurückgesetzt (oder blockweise vor einem Iterationsblock, wenn so definiert).
- Der Reset wird dokumentiert, und die Messung startet erst nach Abschluss des Drops.

Wichtig: “Cold” muss als konkrete Prozedur dokumentiert sein, sonst sind Ergebnisse nicht vergleichbar.

---

## 7 Kennzahlen und Aggregation

### 7.1 Latenz pro Iteration
- Gemessen wird die wall-clock Zeit der Query-Ausführung in Millisekunden.
- Pro Iteration entsteht ein Wert `lat_ms`.

### 7.2 Per-Run Statistiken
Aus den `I` Iterationen eines Runs:
- `p50` = Median der `lat_ms`
- `p95` = 95. Perzentil der `lat_ms` (diskret)
- optional: `p99`, `min`, `max`

### 7.3 Aggregation über Runs
Über `R` Runs wird robust zusammengefasst:
- Reported `p50` = Median der Run-`p50`
- Reported `p95` = Median der Run-`p95`
- Zusätzlich: Streuung (IQR oder Bootstrap-Konfidenzbänder)

### 7.4 Unsicherheitsabschätzung (Bootstrap)
Zur Quantifizierung der Unsicherheit:
- Pro Run werden aus den `I` Iterationen `B` Bootstrap-Stichproben gezogen (typisch `B = 2000`)
- Für jede Stichprobe wird die Zielkennzahl (Median oder p95) berechnet
- Aus der Bootstrap-Verteilung wird ein 95 Prozent Intervall `[low, high]` bestimmt
- Über Runs hinweg werden die unteren und oberen Grenzen robust aggregiert (Median der Lows und Median der Highs)

Hinweis: Bootstrap-Intervalle für hohe Perzentile bei sehr kleinen Stichproben (z. B. `I = 10`) sind primär zur Visualisierung der Streuung geeignet.

---

## 8 Systemmetriken und PSI (inkl. Erklärung der “Deltas”)

### 8.1 Was PSI liefert
Linux PSI stellt für CPU, Memory und IO Druckwerte bereit. Die Dateien in `/proc/pressure/*` enthalten u. a.:
- eine “some” und “full” Kategorie
- gleitende Durchschnittswerte über feste Zeitfenster
- kumulative Zähler (zeitbasierte Akkumulation seit Boot)

### 8.2 Warum in Plots “Delta” steht
Viele PSI-Werte sind **kumulativ**, also seit Systemstart aufaddiert. Um Belastung **pro Messintervall** sichtbar zu machen, wird pro Iteration ein Snapshot genommen und anschließend die Differenz zum vorherigen Snapshot gebildet:

- `delta = psi_total_now - psi_total_prev`

Interpretation:
- `delta` ist die zusätzliche Stall-Zeit, die während der Iteration (oder während des Iterationsintervalls) aufgelaufen ist.
- Dadurch lassen sich Iterationen vergleichen, ohne dass der absolute “seit Boot” Zähler dominiert.

### 8.3 Wie PSI im Kontext gelesen wird
- CPU-PSI hoch bei CPU-Konkurrenz oder Scheduler-Druck
- IO-PSI hoch bei blockierendem Storage-Backpressure
- Memory-PSI hoch bei Reclaim, Page-Fault Druck, RAM-Limits

Die Deltas werden zusammen mit Temperatur und Frequenz geloggt, um Latenzsprünge mechanistisch zu erklären.

---

## 9 Ergebnisartefakte und Reproduzierbarkeit

### 9.1 Protokollierte Metadaten
Pro Kampagne bzw. Messzelle werden gespeichert:
- Git-Commit oder Snapshot-ID
- Python-Version und Requirements
- Engine-Versionen (SQLite, DuckDB)
- Parameter der Messzelle (Tags, Rows, Cache, Threads, Indexprofil)
- DB-Größen (Dateigröße) und ggf. Parquet-Größe
- Laufzeit-Logs pro Iteration (Latenz, PSI-Deltas, Temperatur, Frequenz)

### 9.2 Struktur der Artefakte
Empfohlen:
- `results/` aggregierte Kennzahlen pro Zelle (z. B. JSON oder CSV)
- `logs/` Iterationslogs (z. B. NDJSON)
- `_audit/` Konsistenz- und Rerun-Protokolle
- `snapshots/` eingefrorene Umgebung (requirements, python_version, commit)

---

## 10 Validität und Grenzen

### 10.1 Interne Validität
Risiken:
- Unklare Cold-Prozedur oder inkonsistente Cache-Drops
- Vermischung von Indexprofilen (z. B. falsche DB-Datei)
- Hintergrundlast (Update-Dienste, Logging-Spikes)

Gegenmaßnahmen:
- Klare Prozeduren, Audit-Logs, feste Dateinamenkonvention, Wiederholungen und robuste Aggregation.

### 10.2 Externe Validität
Synthetische Daten ermöglichen kontrollierte Skalierung, können aber reale Eigenschaften nur approximieren:
- Verteilung und Autokorrelation realer Sensoren
- Missingness, Ausreißer, echte Event-Last

Gegenmaßnahme:
- Ein Real-Datensatz als Ankerpunkt im Parameterraum (zusätzlicher Validierungsbezug)
- Diskussion, welche Effekte generalisieren und welche generatorabhängig sein können

---

## 11 Kurzreferenz der Query-Schnittstellen (für Implementierung und Review)

### Q1 Downsample 5min
- Input: tag set, time range
- Output: per tag per 5min bucket avg

### Q2 Last 15min avg
- Input: t_ref
- Output: per tag avg in last 15min

### Q3 Last 60min avg
- Input: t_ref
- Output: per tag avg in last 60min

### Q4 Tumbling 10min mean + discrete p95
- Input: time range
- Output: per tag per 10min bucket mean and discrete p95

---

## 12 Checkliste vor einem Kampagnenlauf
- Datensatzparameter (rows_per_tag, tags) stimmen mit Manifest überein
- DB-Dateien eindeutig pro Indexprofil benannt
- Cache-Regime korrekt eingestellt und dokumentiert
- Threads-Konfiguration korrekt gesetzt
- Warm-up klar definiert und aus Messung ausgeschlossen
- Logs und Ergebnisartefakte werden in neue, eindeutige Kampagnenstruktur geschrieben
- Validierung (gleiches Ergebnis) zumindest stichprobenartig aktiv

---
