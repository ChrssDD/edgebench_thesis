# loader/sqlite_loader.py
import argparse, sqlite3, pandas as pd
from pathlib import Path
from config import DB_DIR

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS measurements (
  ts_ns INTEGER NOT NULL,
  tag   TEXT    NOT NULL,
  value REAL    NOT NULL,
  PRIMARY KEY (ts_ns, tag)
) WITHOUT ROWID;
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--db", type=Path, default=DB_DIR/"sqlite_edge.db")
    ap.add_argument("--extra-index", action="store_true",
                    help="Wenn gesetzt: Sekundärindex (tag, ts_ns) erzeugen.")
    args = ap.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(args.db); cur = con.cursor()

    # Reproduzierbare PRAGMAs / Fairness
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA page_size=4096;")
    cur.execute("PRAGMA cache_size=-200000;")   # ~200 MB
    cur.execute("PRAGMA mmap_size=268435456;")  # 256 MiB
    cur.execute("PRAGMA temp_store=MEMORY;")

    cur.execute(SCHEMA_SQL)

    # Bulk-Insert in EINER Transaktion; Indizes danach
    con.execute("BEGIN")
    for chunk in pd.read_csv(args.csv, chunksize=100_000):
        cur.executemany(
            "INSERT INTO measurements(ts_ns, tag, value) VALUES (?,?,?)",
            list(chunk.itertuples(index=False, name=None))
        )
    con.commit()

    # Nur wenn Zusatzindex gewünscht:
    if args.extra_index:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tag_ts ON measurements(tag, ts_ns);")
    con.commit()

    try:
        cur.execute("ANALYZE;")
        con.commit()
    except Exception:
        pass

    con.close()
    print(f"[ok] loaded into {args.db}")

if __name__ == "__main__":
    main()
