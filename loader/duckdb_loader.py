#loader/duckdb_loader.py
import argparse, duckdb
from pathlib import Path
from config import DB_DIR
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--db", type=Path, default=DB_DIR/"duck_edge.db")
    args = ap.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(args.db)
    con.execute("""
        CREATE OR REPLACE TABLE measurements AS
        SELECT ts_ns, tag, value
        FROM read_parquet(?)
        ORDER BY ts_ns, tag
    """, [str(args.parquet)])
    con.execute("CHECKPOINT")
    con.close()
    print(f"[ok] loaded into {args.db}")

if __name__ == "__main__":
    main()
