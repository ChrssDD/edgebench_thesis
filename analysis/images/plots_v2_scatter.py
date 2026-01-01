# analysis/images/plots_v2_scatter.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import apply_style
from .utils import ensure_dir

def _normalize_warm(v):
    s = str(v).strip().lower()
    return s in ("1","true","yes","warm")

def _select_rows(df, rows_target, rpt_target, strategy: str):
    cand = df.copy()
    if "dataset_rows" not in cand.columns or not cand["dataset_rows"].notna().any():
        return cand
    if rows_target is not None:
        vals = cand["dataset_rows"].unique()
        pick = min(vals, key=lambda x: abs(x - rows_target))
        return cand[cand["dataset_rows"] == pick]
    if rpt_target is not None and "dataset_tags" in cand.columns and cand["dataset_tags"].notna().any():
        rpt = cand["dataset_rows"] / cand["dataset_tags"]
        cand = cand.assign(_rpt=rpt)
        vals = cand["_rpt"].unique()
        pick = min(vals, key=lambda x: abs(x - rpt_target))
        return cand[cand["_rpt"] == pick].drop(columns=["_rpt"])
    if strategy == "min":
        pick = cand["dataset_rows"].min()
        return cand[cand["dataset_rows"] == pick]
    pick = cand["dataset_rows"].max()
    return cand[cand["dataset_rows"] == pick]

def _rho(x, y):
    if len(x) < 3:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))

def _subset(it, query, warm, engine, duck_profile, sqlite_variant, tags):
    df = it[it["query"] == query].copy()
    df["warm"] = df["warm"].apply(_normalize_warm)
    df = df[df["warm"] == warm]
    if tags is not None and "dataset_tags" in df.columns:
        df = df[df["dataset_tags"] == tags]
    if engine == "duckdb":
        df = df[df["engine"] == "duckdb"]
        if duck_profile == "t1":
            df = df[df["threads"] == 1]
        else:
            df = df[df["threads"].isna()]
    else:
        df = df[df["engine"] == "sqlite"]
        if sqlite_variant in ("idx","noidx"):
            df = df[df["sqlite_variant"] == sqlite_variant]
        # sqlite_variant == "best": lassen wir gemischt; Scatter zeigt ja Iterationen
    return df

def _col_for(kind: str):
    return "psi_io_full_diff_ms" if kind == "io" else "psi_cpu_some_diff_ms"

def scatter_one(it_long: pd.DataFrame, outdir: Path, *,
                query: str, warm: bool, engine: str,
                duck_profile: str, sqlite_variant: str,
                psi_kind: str, rows: float|None, rpt: float|None, select: str):
    base = _subset(it_long, query, warm, engine, duck_profile, sqlite_variant, tags=None)
    if base.empty:
        return None
    picked = _select_rows(base, rows, rpt, select)
    if picked.empty:
        return None
    psi_col = _col_for(psi_kind)
    sub = picked[["dur_ms", psi_col, "dataset_rows", "dataset_tags"]].dropna()
    if sub.empty:
        return None

    rows_val = int(sub["dataset_rows"].iloc[0]) if sub["dataset_rows"].notna().any() else None
    tags_val = int(sub["dataset_tags"].iloc[0]) if sub["dataset_tags"].notna().any() else None
    x = sub[psi_col].values.astype(float)
    y = sub["dur_ms"].values.astype(float)
    rho = _rho(x, y)
    n = len(x)

    label_engine = f"DuckDB({duck_profile})" if engine=="duckdb" else (f"SQLite({sqlite_variant})" if sqlite_variant else "SQLite")
    subdir = ensure_dir(outdir / psi_kind / ("warm" if warm else "cold") / (label_engine.replace("(","_").replace(")","")))
    fname = subdir / f"{query}__scatter_{psi_kind}__{label_engine}__rows{rows_val if rows_val else 'sel'}__tags{tags_val if tags_val else 'na'}.png"

    fig, ax = plt.subplots(figsize=(5.2,3.6))
    ax.scatter(x, y, s=14, alpha=0.65, edgecolor="none")
    ax.set_xlabel(("PSI Δ (I/O full, ms)" if psi_kind=="io" else "PSI Δ (CPU some, ms)"))
    ax.set_ylabel("Iteration latency (ms)")
    ttl = f"{query} — {'warm' if warm else 'cold'} — {label_engine}"
    if rows_val: ttl += f" — rows={rows_val}"
    if tags_val: ttl += f" (tags={tags_val})"
    ax.set_title(ttl)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.02, f"Spearman ρ = {rho:.3f}   n = {n}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=0.3))
    fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)
    return fname

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", required=True, help="Path to results/iterations_long.csv")
    ap.add_argument("--outdir", required=False, default="plots/psi_scatter")
    ap.add_argument("--queries", default="", help="Comma list; empty = all in iterations_long.csv")
    ap.add_argument("--warm", default="both", choices=["warm","cold","both"])
    ap.add_argument("--engine", default="duckdb", choices=["duckdb","sqlite"])
    ap.add_argument("--duck-profiles", default="auto,t1")
    ap.add_argument("--sqlite-variants", default="best", help="best,idx,noidx (only used if --engine=sqlite)")
    ap.add_argument("--psi", default="io,cpu", help="io,cpu or subset")
    ap.add_argument("--rows", type=float, default=None, help="Target TOTAL rows (nearest)")
    ap.add_argument("--rows-per-tag", type=float, default=None, help="Target rows-per-tag (nearest)")
    ap.add_argument("--select", default="max", choices=["max","min","nearest"])
    args = ap.parse_args()

    apply_style()
    it = pd.read_csv(args.iterations)
    queries = [q.strip() for q in args.queries.split(",") if q.strip()] or sorted(it["query"].dropna().unique())
    warms = ([True, False] if args.warm=="both" else ([True] if args.warm=="warm" else [False]))
    psi_kinds = [p.strip() for p in args.psi.split(",") if p.strip()]

    written = []
    if args.engine == "duckdb":
        profiles = [p.strip() for p in args.duck_profiles.split(",") if p.strip()]
        for q in queries:
            for w in warms:
                for prof in profiles:
                    for pk in psi_kinds:
                        f = scatter_one(it, Path(args.outdir), query=q, warm=w, engine="duckdb",
                                        duck_profile=("t1" if prof=="t1" else "auto"),
                                        sqlite_variant=None, psi_kind=("io" if pk=="io" else "cpu"),
                                        rows=args.rows, rpt=args.rows_per_tag, select=args.select)
                        if f: written.append(f)
    else:
        variants = [v.strip() for v in args.sqlite_variants.split(",") if v.strip()]
        for q in queries:
            for w in warms:
                for var in variants:
                    for pk in psi_kinds:
                        f = scatter_one(it, Path(args.outdir), query=q, warm=w, engine="sqlite",
                                        duck_profile=None, sqlite_variant=var, psi_kind=("io" if pk=="io" else "cpu"),
                                        rows=args.rows, rpt=args.rows_per_tag, select=args.select)
                        if f: written.append(f)

    print(f"[ok] wrote {len(written)} PSI scatter plots → {args.outdir}")

if __name__ == "__main__":
    main()
