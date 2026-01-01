# analysis/diagnose.py
import argparse, json, re, sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import pandas as pd
import numpy as np

from config import RESULTS_DIR, DB_DIR

RE_SQLITE = re.compile(r"sqlite_(?P<rows>\d+)x(?P<tags>\d+)_(?P<variant>idx|noidx)\.db$")
RE_DUCK   = re.compile(r"duck_(?P<rows>\d+)x(?P<tags>\d+)\.db$")

def _safe_int(x) -> Optional[int]:
    try: return int(x)
    except: return None

def _sqlite_variant_from_dbpath(p: str) -> Optional[str]:
    m = RE_SQLITE.search(Path(p).name)
    return m.group("variant") if m else None

def _expected_from_name_sqlite(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    m = RE_SQLITE.match(path.name)
    if not m: return None, None, None
    return _safe_int(m.group("rows")), _safe_int(m.group("tags")), m.group("variant")

def _expected_from_name_duck(path: Path) -> Tuple[Optional[int], Optional[int]]:
    m = RE_DUCK.match(path.name)
    if not m: return None, None
    return _safe_int(m.group("rows")), _safe_int(m.group("tags"))

def _count_rows_sqlite(db_path: Path) -> Optional[int]:
    import sqlite3
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM measurements")
        n = cur.fetchone()[0]
        con.close()
        return int(n)
    except Exception:
        return None

def _count_rows_duck(db_path: Path) -> Optional[int]:
    try:
        import duckdb
        con = duckdb.connect(str(db_path))
        n = con.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
        con.close()
        return int(n)
    except Exception:
        return None

def _read_ndjson(p: Path) -> List[dict]:
    rows = []
    if p.exists():
        with p.open() as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: rows.append(json.loads(line))
                except Exception: pass
    return rows

def _load_summary() -> pd.DataFrame:
    p = Path(RESULTS_DIR) / "summary_with_cis.csv"
    if not p.exists():
        print("[warn] summary_with_cis.csv not found. Run: python -m analysis.summarize_results", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(p)
    # normalize
    if "sqlite_variant" not in df.columns:
        df["sqlite_variant"] = np.where(
            df["engine"].eq("sqlite") & df["db"].astype(str).str.contains("_idx"), "idx",
            np.where(df["engine"].eq("sqlite") & df["db"].astype(str).str.contains("_noidx"), "noidx", None)
        )
    return df

def _best_sqlite_curve(sub: pd.DataFrame, rows_col: str, metric: str) -> pd.Series:
    # take min across idx/noidx at each size
    return (sub[sub["engine"]=="sqlite"]
              .groupby(rows_col)[metric].min())

def _variant_sqlite_curve(sub: pd.DataFrame, rows_col: str, metric: str, variant: str) -> pd.Series:
    s = sub[(sub["engine"]=="sqlite") & (sub["sqlite_variant"]==variant)]
    return s.groupby(rows_col)[metric].min()

def _duck_curve(sub: pd.DataFrame, rows_col: str, metric: str, profile: str = "auto") -> pd.Series:
    """
    Return the DuckDB baseline curve for the given profile.
    profile ∈ {"auto","t1"}
    """
    if "threads" in sub.columns:
        if profile == "t1":
            s = sub[(sub["engine"] == "duckdb") & (sub["threads"] == 1)]
        else:
            s = sub[(sub["engine"] == "duckdb") & (sub["threads"].isna())]
    else:
        s = sub[sub["engine"] == "duckdb"]
    return s.groupby(rows_col)[metric].min()


def _explain_be(
    df: pd.DataFrame,
    query: str,
    warm: bool,
    variant: str,
    metric: str = "p95_ms",
    duckdb_profile: str = "auto",
    tags: int | None = None,
    min_rows_per_tag_for_window: int | None = None
) -> dict:
    base = df[
        (df["query"] == query) &
        (df["warm"] == warm) &
        (df["scan_mode"] == "table") &
        df["dataset_rows"].notna()
    ].copy()
    if tags is not None:
        base = base[base["dataset_tags"] == tags]
    if min_rows_per_tag_for_window is not None:
        rpt = base["dataset_rows"] / base["dataset_tags"]
        base = base.loc[rpt >= min_rows_per_tag_for_window]

    if base.empty:
        return {"query": query, "warm": warm, "variant": variant, "profile": duckdb_profile, "tags": tags, "reason": "no_data"}

    rows_col = "dataset_rows"
    duck = _duck_curve(base, rows_col, metric, profile=duckdb_profile)
    if variant == "best":
        sqlite = (base[base["engine"] == "sqlite"].groupby(rows_col)[metric].min())
    else:
        sqlite = (base[(base["engine"] == "sqlite") & (base["sqlite_variant"] == variant)]
                        .groupby(rows_col)[metric].min())

    duck_sizes   = set(duck.index.tolist())
    sqlite_sizes = set(sqlite.index.tolist())
    shared = sorted(duck_sizes & sqlite_sizes)
    if len(shared) == 0:
        return {"query": query, "warm": warm, "variant": variant, "profile": duckdb_profile, "tags": tags,
                "reason": "no_shared_sizes", "duck_only": sorted(duck_sizes - sqlite_sizes),
                "sqlite_only": sorted(sqlite_sizes - duck_sizes)}
    if len(shared) == 1:
        return {"query": query, "warm": warm, "variant": variant, "profile": duckdb_profile, "tags": tags,
                "reason": "insufficient_points", "shared_sizes": shared}

    delta = (sqlite.loc[shared] - duck.loc[shared]).values
    signs = np.sign(delta)
    has_cross = np.any(signs[:-1] * signs[1:] < 0)
    if not has_cross:
        return {"query": query, "warm": warm, "variant": variant, "profile": duckdb_profile, "tags": tags,
                "reason": "no_crossing_in_measured_range",
                "first_sign": int(signs[0]), "last_sign": int(signs[-1]), "shared_sizes": shared}
    return {"query": query, "warm": warm, "variant": variant, "profile": duckdb_profile, "tags": tags,
            "reason": "OK_crossing_found", "shared_sizes": shared}


def diagnose_break_even(duckdb_profile: str = "auto"):
    df = _load_summary()
    if df.empty:
        return
    all_tags = sorted(x for x in df["dataset_tags"].dropna().unique())
    print(f"\n=== BREAK-EVEN EXPLAINER (profile={duckdb_profile}; table only; per tags) ===")
    rows = []
    for q in sorted(df["query"].dropna().unique()):
        min_rpt = 3600 if q == "last_60min_avg" else (900 if q == "last_15min_avg" else None)
        for warm in (False, True):
            for tg in all_tags:
                for variant in ("best","idx","noidx"):
                    info = _explain_be(df, q, warm, variant, metric="p95_ms",
                                       duckdb_profile=duckdb_profile, tags=tg,
                                       min_rows_per_tag_for_window=min_rpt)
                    rows.append(info)
    rep = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
        print(rep.sort_values(["query","warm","tags","variant"]).to_string(index=False))



def check_runtime_env():
    print("\n=== RUNTIME ENV / HEALTH ===")
    prog = Path(RESULTS_DIR) / "progress.ndjson"
    fails = Path(RESULTS_DIR) / "failures.ndjson"
    P = _read_ndjson(prog)
    F = _read_ndjson(fails)

    # 1) timeouts summary
    from collections import Counter
    if F:
        print("Timeouts/errors by (engine, query, warm):")
        key = lambda r: (r.get("engine"), r.get("query"), r.get("warm"))
        cnt = Counter(key(r) for r in F)
        for (eng,q,w),c in sorted(cnt.items()):
            # w is bool; format it as string with width
            print(f"  {eng:6s} {str(w):5s} {q:24s} -> {c}")
    else:
        print("No failures.ndjson found or empty.")

    # 2) iterations / PSI / drop_caches
    it_path = Path(RESULTS_DIR) / "iterations_long.csv"
    if it_path.exists():
        it = pd.read_csv(it_path)
        if not it.empty:
            before_cols = [c for c in it.columns if c.startswith("psi_before_")]
            after_cols  = [c for c in it.columns if c.startswith("psi_after_")]
            psi_ok = (it[before_cols + after_cols].notna().sum(axis=1) > 0).mean()
            dc_col = "drop_caches_ok"
            dc_rate = it[dc_col].fillna(False).mean() if dc_col in it.columns else float("nan")
            print(f"PSI fields present on {psi_ok*100:.1f}% of iteration rows")
            print(f"drop_caches_ok true on {dc_rate*100:.1f}% of iteration rows")

            # Simple PSI ↔ latency (Spearman) per (query,warm)
            if "psi_after_cpu_some_avg10" in it.columns:
                print("\n=== PSI ↔ Latency (Spearman ρ) ===")
                for (q,w), g in it.groupby(["query","warm"]):
                    g2 = g[["dur_ms","psi_after_cpu_some_avg10"]].dropna()
                    if len(g2) >= 10:
                        # pandas corr spearman (no scipy dependency)
                        rho = g2.corr(method="spearman").iloc[0,1]
                        # FIXED: format bool as string, width 5
                        print(f"{q:24s}  warm={str(bool(w)):5s}  rho={rho:+.3f}")
        else:
            print("iterations_long.csv exists but is empty")
    else:
        print("iterations_long.csv not found (run: python -m analysis.extract_iterations)")


def check_dbs(mode: str):
    """
    mode: 'none' | 'all' | 'sample:N'
    Compares on TOTAL rows = rows_per_tag * tags.
    """
    print("\n=== DB INTEGRITY CHECK ===")
    sqlite_dbs = sorted(Path(DB_DIR).glob("sqlite_*.db"))
    duck_dbs   = sorted(Path(DB_DIR).glob("duck_*.db"))

    def choose(dbs: List[Path]) -> List[Path]:
        if mode == "all":
            return dbs
        m = re.match(r"sample:(\d+)", mode or "")
        if m:
            n = int(m.group(1))
            return dbs[:max(0, n)]
        return []  # none

    chk_sql = choose(sqlite_dbs)
    chk_du  = choose(duck_dbs)

    if not chk_sql and not chk_du:
        print("(skipping — use --check-dbs all | sample:N)")
        return

    rows = []
    for p in chk_sql:
        rpt, tags, variant = _expected_from_name_sqlite(p)
        got_total = _count_rows_sqlite(p)
        exp_total = (rpt * tags) if (rpt and tags) else None
        rows.append({
            "db": p.name, "engine": "sqlite", "variant": variant,
            "rows_per_tag": rpt, "tags": tags,
            "exp_total_rows": exp_total, "got_total_rows": got_total,
            "ok": (got_total == exp_total) if (got_total is not None and exp_total is not None) else None
        })
    for p in chk_du:
        rpt, tags = _expected_from_name_duck(p)
        got_total = _count_rows_duck(p)
        exp_total = (rpt * tags) if (rpt and tags) else None
        rows.append({
            "db": p.name, "engine": "duckdb", "variant": None,
            "rows_per_tag": rpt, "tags": tags,
            "exp_total_rows": exp_total, "got_total_rows": got_total,
            "ok": (got_total == exp_total) if (got_total is not None and exp_total is not None) else None
        })

    rep = pd.DataFrame(rows)
    if rep.empty:
        print("(no DBs checked)")
        return
    with pd.option_context("display.max_rows", None):
        print(rep.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="EdgeBench diagnostics: DB integrity, runtime health, BE explanations")
    ap.add_argument("--check-dbs", default="sample:6", choices=["none","all"] + [f"sample:{i}" for i in range(1,101)],
                    help="Verify row counts in DB files (can be slow). Default: sample:6")
    ap.add_argument("--duckdb-profile", choices=["auto","t1"], default="auto",
                    help="Which DuckDB baseline to use for BE explanation")
    args = ap.parse_args()

    check_dbs(args.check_dbs)
    check_runtime_env()
    diagnose_break_even(args.duckdb_profile)


if __name__ == "__main__":
    main()
