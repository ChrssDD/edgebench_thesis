#config.py
from pathlib import Path
import os

EDGE_ROOT = Path(os.environ.get("EDGE_ROOT", "/mnt/edgebench")).resolve()
PROJECT_ROOT = Path(__file__).parent.resolve()

DATA_DIR    = EDGE_ROOT / "data"
DB_DIR      = DATA_DIR / "db"
RESULTS_DIR = EDGE_ROOT / "results"
LOG_DIR     = EDGE_ROOT / "logs"
QUERIES_DIR = PROJECT_ROOT / "queries"

for p in (DATA_DIR, DB_DIR, RESULTS_DIR, LOG_DIR, QUERIES_DIR):
    p.mkdir(parents=True, exist_ok=True)
