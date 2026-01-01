# EdgeBench Tests (pytest)

Ziel:
- Schnell laufende Tests (ohne sudo), die Kernfunktionen absichern:
  - deterministische Datengenerierung
  - Loader erzeugen valide DBs
  - Queries sind semantisch konsistent (DuckDB vs SQLite)
  - Analyse kann minimale Artefakte parsen und Summary erzeugen

Ausführen:
cd /mnt/edgebench/src/edgebench
pytest -q
