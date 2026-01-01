# EdgeBench

Benchmark-Repo zum Vergleich der Abfrageperformance von SQLite und DuckDB für typische Zeitreihen-Analytik auf Edge-Hardware (z. B. Raspberry Pi).

## Workflow

Der grundlegende Ablauf ist:
1. Daten generieren
2. Datenbanken laden
3. Queries messen
4. Ergebnisse als JSON speichern
5. Auswertung und Plots erzeugen

## Repository-Übersicht

- **config.py** – Zentrale Pfade (EDGE_ROOT sowie data/db/results/logs und Queries-Pfad)
- **generator/synth.py** – Deterministisches Erzeugen synthetischer Zeitreihen als CSV und Parquet
- **loader/sqlite_loader.py** – CSV nach SQLite (WAL/NORMAL, optionaler Extra-Index)
- **loader/duckdb_loader.py** – Parquet nach DuckDB (Tabelle measurements)
- **runner/run_bench.py** – Single-Benchmarks (eine Query mehrfach ausführen und JSON-Result schreiben)
- **runner/run_matrix.py** – Orchestrieren kompletter Experiment-Matrizen
- **queries/** – Engine-spezifische SQL-Dateien (queries/sqlite/*.sql und queries/duckdb/*.sql)
- **analysis/** – Auswertung (Summary, Break-even, Plots)

## Voraussetzungen

- **OS:** Linux wird empfohlen (Debian/Raspberry Pi passt gut)
- **Python:** 3.10 oder neuer
- **Pakete:** numpy, pandas, duckdb, pyarrow, psutil, matplotlib
- **Optional:** Root-Zugriff für echte Cold-Runs (Linux Page Cache droppen via /proc/sys/vm/drop_caches)
- **Optional:** PSI-Metriken (erfordern /proc/pressure/* im Kernel)

## Installation

```bash
# Repository klonen
git clone <repo-url>
cd edgebench

# Virtuelle Umgebung erstellen und aktivieren
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# oder
venv\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install --upgrade pip
pip install -r requirements.txt
```

**Hinweis:** Für `python -m` Aufrufe müssen die Ordner als Packages importierbar sein (mit `__init__.py`) und du musst im Repo-Root stehen. Alternativ lassen sich die Skripte direkt per Pfad ausführen: `python generator/synth.py ...`

## Konfiguration

### EDGE_ROOT

Das Output-Verzeichnis wird über die Umgebungsvariable `EDGE_ROOT` konfiguriert:

```bash
export EDGE_ROOT=/mnt/edgebench  # Standard
# oder
export EDGE_ROOT=~/edgebench     # Im Home-Verzeichnis
```

Standardmäßig: `/mnt/edgebench`

### Verzeichnisstruktur

Unterhalb von `EDGE_ROOT` wird folgende Verzeichnisstruktur erwartet:

```
edgebench/
├── data/
│   ├── <zeitreihe>.csv
│   └── <zeitreihe>.parquet
├── db/
│   ├── sqlite_<rows>x<tags>_noidx.db
│   ├── sqlite_<rows>x<tags>_idx.db
│   └── duck_<rows>x<tags>.db
├── results/
│   ├── <ergebnis>.json
│   ├── summary_with_cis.csv
│   └── ...
└── logs/
    ├── matrix_<RUN_ID>.log
    └── ...
```

## Quickstart

1. **Daten generieren:**

   Beispiel: 100000 rows pro Tag, 20 Tags, 1 Hz, über `generator/synth` mit `rows`, `tags`, `freq-ms` und `seed`.

   ```bash
   python generator/synth.py --rows 100000 --tags 20 --freq-ms 1000 --seed 42
   ```

   Das erzeugt `syn_<rows>x<tags>.csv` und `syn_<rows>x<tags>.parquet` unter `EDGE_ROOT/data`.

2. **DBs laden:**

   - SQLite ohne Extra-Index:

     ```bash
     python loader/sqlite_loader.py --csv EDGE_ROOT/data/syn_100000x20.csv --db sqlite_100000x20_noidx.db
     ```

   - SQLite mit Extra-Index (tag, ts_ns):

     ```bash
     python loader/sqlite_loader.py --csv EDGE_ROOT/data/syn_100000x20.csv --db sqlite_100000x20_idx.db --extra-index
     ```

   - DuckDB:

     ```bash
     python loader/duckdb_loader.py --parquet EDGE_ROOT/data/syn_100000x20.parquet --db duck_100000x20.db
     ```

3. **Single-Benchmark ausführen:**

   Beispiel für SQLite:

   ```bash
   python runner/run_bench.py --engine sqlite --db sqlite_100000x20_noidx.db --query downsample_1min --repeats 10
   ```

   Beispiel für DuckDB:

   ```bash
   python runner/run_bench.py --engine duckdb --db duck_100000x20.db --query downsample_1min --repeats 10 --threads 4
   ```

4. **Matrix-Runner für komplette Experimente nutzen:**

   ```bash
   python runner/run_matrix.py --sizes 1000,10000,100000 --tags 10,20,30 --repeats 5
   ```

5. **Ergebnisse auswerten:**

   ```bash
   python analysis/summarize_results.py
   ```

## Komponenten im Detail

- **Generator:** Erzeugt synthetische Zeitreihen mit `ts_ns` in Nanosekunden in gleichmäßigem Schritt (`freq-ms`), pro Tag eine sinus-basierte Kurve plus reproduzierbares Rauschen über `--seed` und schreibt CSV sowie Parquet. Mit `--t0-ns` lässt sich ein fixer Startzeitpunkt setzen, um die gleiche Zeitachse über Runs zu erhalten.

- **SQLite-Loader:** Schreibt in eine `measurements`-Tabelle mit Primärschlüssel `(ts_ns, tag)` und `WITHOUT ROWID` und setzt PRAGMAs wie WAL und `synchronous=NORMAL` sowie weitere Einstellungen (z. B. `page_size`, `cache_size`, `mmap_size`, `temp_store=MEMORY`). Optional kann ein zusätzlicher Index über `(tag, ts_ns)` erstellt werden.

- **DuckDB-Loader:** Materialisiert `measurements` aus Parquet, optional sortiert, und führt anschließend einen `CHECKPOINT` aus.

- **Single-Runner (`run_bench`):** Misst eine Query mehrfach und schreibt eine JSON-Datei nach `results/`. Wesentliche Parameter sind `engine` (sqlite oder duckdb), `db` (Pfad), `query` (Name; SQL wird aus `queries/<engine>/<query>.sql` geladen), `repeats`, `warm`, `threads` (nur DuckDB), `drop-caches` (root nötig), `max-run-s` (Zeitbudget, bricht ggf. früh ab) sowie `parquet/csv` als File-Scan-Helper für DuckDB. Die Output-Dateien werden nach dem Schema `<dbstem>_<engine>_<query>_<warm|cold>[_t{threads}]_{tbl|fs}.json` benannt.

- **Matrix-Runner (`run_matrix`):** Generiert Datensätze, baut DBs (SQLite idx/noidx und DuckDB), plant Query-Läufe (warm/cold; DuckDB auto/t1; optional file-scan), randomisiert die Reihenfolge und schreibt Logs und Manifeste, typischerweise `progress.ndjson` und `failures.ndjson` in `results` sowie `manifest.json` und `manifests/<RUN_ID>.json`, plus `logs/matrix_<RUN_ID>.log`.

## Queries

Queries liegen engine-spezifisch als Dateien unter `queries/sqlite/<name>.sql` und `queries/duckdb/<name>.sql`. Übliche Query-Namen sind `downsample_1min`, `downsample_5min`, `downsample_15min`, `last_15min_avg`, `last_60min_avg` und `window_10min_mean_p95`. DuckDB-Queries können optional Platzhalter `{PARQUET}` oder `{CSV}` nutzen.

## Analysis

Die Analysis-Komponente (`analysis/summarize_results.py`) liest `results/*.json` und erzeugt u. a. `summary_with_cis.csv`, `iterations_long.csv`, `break_even.csv`, `break_even_multi.csv` sowie `run_summary.json` in `results/`.

## Output-Formate

- `run_bench` schreibt pro Run eine JSON-Datei mit Metadaten (engine, db, query, repeats, warm, threads, drop_caches), Messwerten (durations_ns und aggregierte stats wie p50/p95/p99/mean/min/max in ms), Iterations-Metadaten (PSI before/after, Temperatur/Frequenz, RSS/CPU, Timestamps), Versionsinfos (Python, DuckDB/SQLite, pandas/numpy/pyarrow, optional git_rev) sowie zusätzlichen Feldern wie db_filesize_bytes, dataset_meta, scan_mode und truncated.

## Cold vs Warm

- **warm** aktiviert typischerweise DuckDB object cache und optionales Warmup.
- **drop-caches** vor jeder Iteration den Linux Page Cache droppt und damit ein härteres Cold-Setup abbildet (root nötig).

## Queries hinzufügen

Lege für jede Engine eine Datei unter `queries/sqlite/<name>.sql` und `queries/duckdb/<name>.sql` an und führe anschließend `run_bench` für sqlite und duckdb mit dem jeweiligen Query-Namen aus, um die Messung zu erzeugen.

## Troubleshooting

- **Permission denied bei drop-caches:** Root-Rechte fehlen, `sudo -E` benötigt.
- **Fehlende Query-Datei:** Prüfe `queries/<engine>/<query>.sql`.
- **DuckDB file-scan “no such table measurements”:** Nutze `{PARQUET}/{CSV}` im SQL oder übergib `--parquet/--csv`, damit `run_bench` eine TEMP VIEW `measurements` erstellen kann.
- **Fehlende PSI:** `/proc/pressure/*` existiert nicht, der Run ist trotzdem möglich, nur ohne PSI.
- **python -m funktioniert nicht:** Fehlende `__init__.py` Dateien oder nicht im Repo-Root, dann starte per Skriptpfad.

## Lizenz

License: Proprietary (All rights reserved). No use, copying, modification, or distribution without permission.


Copyright (c) 2026 Christian Dederer

All rights reserved.

No permission is granted to use, copy, modify, merge, publish, distribute, sublicense,
or sell copies of this software or its associated documentation, in whole or in part,
without prior written permission from the copyright holder.


## Note:
This Repo will be privated at 03/01/2026