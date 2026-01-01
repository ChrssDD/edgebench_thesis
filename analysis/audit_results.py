# analysis/audit_results.py
import json
import pandas as pd
from pathlib import Path
from config import RESULTS_DIR, DB_DIR, LOG_DIR, EDGE_ROOT

def fmt_bytes(n):
    if n is None: return "?"
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def safe_read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def main():
    root = Path(RESULTS_DIR)
    print(f"=== PATHS ===")
    print(f"EDGE_ROOT:   {EDGE_ROOT}")
    print(f"RESULTS_DIR: {root}")
    print(f"DB_DIR:      {DB_DIR}")
    print(f"LOG_DIR:     {LOG_DIR}")

    print("\n=== DB FILES (name · size) ===")
    dbs = sorted(Path(DB_DIR).glob("*.db"))
    if not dbs:
        print("(none)")
    else:
        for p in dbs:
            try:
                sz = p.stat().st_size
            except Exception:
                sz = None
            print(f"{p.name:40s}  {fmt_bytes(sz)}")

    data_dir = Path(EDGE_ROOT) / "data"
    csvs = sorted(data_dir.glob("*.csv"))
    pqs  = sorted(data_dir.glob("*.parquet"))
    print("\n=== DATA FILES ===")
    print(f"CSV: {len(csvs)}  Parquet: {len(pqs)}")

    mani = root / "manifest.json"
    if mani.exists():
        m = json.loads(mani.read_text())
        print("\n=== MANIFEST ===")
        print(f"sizes: {m.get('sizes')}")
        print(f"tags:  {m.get('tags')}")
        print(f"repeats: {m.get('repeats')}, drop_caches: {m.get('drop_caches')}")
        print(f"timeout_s_per_bench: {m.get('timeout_s_per_bench')}, max_run_s: {m.get('max_run_s')}")
        print(f"run_id: {m.get('run_id')}")
    else:
        print("\n(manifest.json not found)")

    def read_ndjson(p):
        rows = []
        if p.exists():
            with p.open() as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try: rows.append(json.loads(line))
                    except Exception: pass
        return rows

    P = read_ndjson(root / "progress.ndjson")
    F = read_ndjson(root / "failures.ndjson")
    print("\n=== PROGRESS COUNTS ===")
    if not P:
        print("(no progress.ndjson)")
    else:
        from collections import Counter
        by_status = Counter(r.get("status") for r in P)
        print("status:", dict(by_status))
        key = lambda r: (r.get("engine"), r.get("query"), r.get("warm"))
        agg = Counter(key(r) for r in P)
        for (eng,q,w),cnt in sorted(agg.items()):
            print(f"{eng:6s}  {str(w):5s}  {q:24s}  count={cnt}")
    if F:
        print("\n=== FAILURES (last 5) ===")
        for r in F[-5:]:
            print(json.dumps(r, indent=2))

    summ_path = root / "summary_with_cis.csv"
    if not summ_path.exists():
        print("\n(summary_with_cis.csv not found)"); return
    summ = pd.read_csv(summ_path)
    if summ.empty:
        print("\n(summary_with_cis.csv empty)"); return

    print("\n=== BASIC COUNTS FROM SUMMARY ===")
    print("rows:", len(summ))
    print("queries:", sorted(summ['query'].dropna().unique()))
    print("warm:", sorted(summ['warm'].dropna().unique()))
    if "scan_mode" in summ.columns:
        print("scan_modes:", dict(summ['scan_mode'].value_counts()))
    if "dataset_rows" in summ.columns:
        print("sizes:", sorted(summ['dataset_rows'].dropna().unique()))

    def _sqlite_label_from_db(db_path: str) -> str:
        stem = Path(str(db_path)).stem
        if stem.endswith("_idx") or "_idx" in stem: return "sqlite_idx"
        if stem.endswith("_noidx") or "_noidx" in stem: return "sqlite_noidx"
        return "sqlite"

    def lab(row):
        if row["engine"] == "sqlite":
            return _sqlite_label_from_db(row.get("db",""))
        # duckdb
        t = row.get("threads")
        return f"duckdb_t{int(t)}" if pd.notna(t) else "duckdb"

    summ["label"] = [lab(r) for _, r in summ.iterrows()]
    print("labels:", sorted(summ['label'].unique()))

    print("\n=== COVERAGE (unique sizes per label) ===")
    cov = (summ.groupby(['query','warm','label'])['dataset_rows']
              .nunique().unstack('label').fillna(0).astype(int))
    print(cov)

    print("\n=== LABEL PRESENCE (need duckdb, duckdb_t1, sqlite_idx, sqlite_noidx) ===")
    need = {'duckdb','duckdb_t1','sqlite_idx','sqlite_noidx'}
    for (q,w), g in summ.groupby(['query','warm']):
        have = set(g['label'].unique())
        miss = need - have
        print(f"{(q,w)}: {'OK' if not miss else 'MISSING '+str(sorted(miss))}")

    print("\n=== SPEEDUP sqlite(best) / duckdb(auto) per (query,warm) ===")
    for (q,w), g in summ.groupby(['query','warm']):
        duck = g[(g['engine']=='duckdb') & (g['threads'].isna())].groupby('dataset_rows')['p50_ms'].min()
        sqli = g[g['engine']=='sqlite'].groupby('dataset_rows')['p50_ms'].min()
        sp = (sqli/duck).dropna()
        if sp.empty:
            print((q,w), "→ no overlap"); continue
        print(f"{(q,w)}: median={sp.median():.2f}  min={sp.min():.2f}  max={sp.max():.2f}")

    be = safe_read_csv(root / "break_even.csv")
    print("\n=== BREAK-EVEN (rows) ===")
    if be.empty:
        print("no break_even.csv")
    else:
        cols = ["query","warm","sqlite_variant","duckdb_profile","break_even_rows"] if "duckdb_profile" in be.columns else ["query","warm","sqlite_variant","break_even_rows"]
        print(be[cols].sort_values(cols).to_string(index=False))
        if "duckdb_profile" in be.columns:
            print("\nBE coverage by profile:")
            print(be.groupby(["duckdb_profile"])["break_even_rows"].apply(lambda s: s.notna().mean()).to_string())


if __name__ == "__main__":
    main()
