# analysis/extract_tables_readonly.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_bool(x: Any) -> bool | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def _read_csv_required(p: Path) -> pd.DataFrame:
    if not p.is_file():
        raise FileNotFoundError(f"Missing required file: {p}")
    return pd.read_csv(p)


def _latex_escape(s: str) -> str:
    # minimal escape set (good enough for query names and small inline strings)
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("_", "\\_")
         .replace("%", "\\%")
         .replace("&", "\\&")
         .replace("#", "\\#")
         .replace("{", "\\{")
         .replace("}", "\\}")
    )


def _q_label(q: str) -> str:
    m = {
        "downsample_15min": "Downsample 15 min",
        "downsample_5min": "Downsample 5 min",
        "downsample_1min": "Downsample 1 min",
        "last_60min_avg": "Last 60 min avg",
        "last_15min_avg": "Last 15 min avg",
        "window_10min_mean_p95": "Window 10 min mean+p95",
    }
    return m.get(q, q)


def _fmt_int_rows(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    try:
        return f"{int(round(float(x))):d}"
    except Exception:
        return "--"


def _fmt_ms(x: Any) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "--"
        return f"{v:.2f}"
    except Exception:
        return "--"


def _fmt_x(x: Any) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "--"
        return f"{v:.2f}"
    except Exception:
        return "--"


def _winner_fmt(w: str | None, var: str | None) -> str:
    if not w:
        return "--"
    if w == "sqlite" and var:
        return f"SQLite ({var})"
    if w == "sqlite":
        return "SQLite"
    if w == "duckdb":
        return "DuckDB"
    return str(w)


def _fmt_rho_with_n(rho: Any, n: Any) -> str:
    """
    Formats Spearman rho with the number of valid pairs n.
    This makes "missing/blank" explainable:
      - n<3 => not enough pairs for a meaningful rank correlation
      - rho is NaN with n>=3 => can happen for constant/near-constant inputs
    """
    n_i: int | None = None
    try:
        if n is not None:
            n_f = float(n)
            if np.isfinite(n_f):
                n_i = int(round(n_f))
    except Exception:
        n_i = None

    r = float("nan")
    try:
        r = float(rho)
    except Exception:
        r = float("nan")

    if n_i is None or n_i < 3:
        return "n/a (n<3)"
    if not np.isfinite(r):
        return f"n/a (n={n_i})"
    return f"{r:.2f} (n={n_i})"


def _parse_be_row(
    df_be: pd.DataFrame, *, query: str, warm: bool, tags: int,
    metric: str, duck_profile: str, sqlite_variant: str
) -> dict:
    sub = df_be[
        (df_be["query"].astype(str) == str(query)) &
        (df_be["metric"].astype(str) == str(metric)) &
        (df_be["duckdb_profile"].astype(str) == str(duck_profile)) &
        (df_be["sqlite_variant"].astype(str) == str(sqlite_variant)) &
        (df_be["tags"].astype(int) == int(tags))
    ].copy()

    if "warm" in sub.columns:
        sub["warm_bool"] = sub["warm"].map(_to_bool)
        sub = sub[sub["warm_bool"] == bool(warm)]

    if sub.empty:
        return {
            "be_points_all": "",
            "be_first": None,
            "be_count": 0,
            "small_winner": None,
            "small_winner_variant": None,
            "large_winner": None,
            "large_winner_variant": None,
        }

    r = sub.iloc[0]
    pts = str(r.get("be_points", "") or "").strip()
    if pts.lower() == "nan":
        pts = ""

    pts_list: list[float] = []
    if pts:
        for part in pts.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                pts_list.append(float(part))
            except Exception:
                pass
    pts_list = sorted(pts_list)
    be_first = pts_list[0] if pts_list else None

    zones_raw = r.get("zones_json", "")
    small_w = small_var = large_w = large_var = None
    try:
        zones = json.loads(zones_raw) if isinstance(zones_raw, str) and zones_raw else []
        if isinstance(zones, list) and zones:
            z0 = zones[0]
            z1 = zones[-1]
            small_w = z0.get("winner_engine")
            small_var = z0.get("winner_variant")
            large_w = z1.get("winner_engine")
            large_var = z1.get("winner_variant")
    except Exception:
        pass

    return {
        "be_points_all": pts,
        "be_first": be_first,
        "be_count": int(len(pts_list)),
        "small_winner": small_w,
        "small_winner_variant": small_var,
        "large_winner": large_w,
        "large_winner_variant": large_var,
    }


def _tail_end_stats(
    df_sum: pd.DataFrame, *, query: str, warm: bool, tags: int,
    metric: str, duck_profile: str
) -> dict:
    d = df_sum.copy()
    d = d[d["scan_mode"].astype(str) == "table"]

    d["warm_bool"] = d["warm"].map(_to_bool)
    d = d[d["warm_bool"] == bool(warm)]
    d = d[d["query"].astype(str) == str(query)]
    d = d[d["dataset_tags"].astype(int) == int(tags)]

    if d.empty:
        return {"end_rows": None}

    # DuckDB selection
    duck = d[d["engine"].astype(str) == "duckdb"].copy()
    if duck_profile == "t1":
        duck = duck[duck["threads"] == 1]
    else:
        # "auto" is stored as NaN threads in your schema
        duck = duck[duck["threads"].isna()]

    # SQLite variants
    sqli = d[d["engine"].astype(str) == "sqlite"].copy()
    s_idx = sqli[sqli["sqlite_variant"].astype(str) == "idx"]
    s_noi = sqli[sqli["sqlite_variant"].astype(str) == "noidx"]

    # Find a max dataset_rows where both duck and at least one sqlite variant exist
    duck_rows = set(duck["dataset_rows"].dropna().astype(int).tolist())
    sqlite_rows = set(sqli["dataset_rows"].dropna().astype(int).tolist())
    common = sorted(duck_rows.intersection(sqlite_rows))
    if not common:
        return {"end_rows": None}

    end_rows = int(common[-1])

    duck_end = duck[duck["dataset_rows"].astype(int) == end_rows]
    sqli_end = sqli[sqli["dataset_rows"].astype(int) == end_rows]
    s_idx_end = s_idx[s_idx["dataset_rows"].astype(int) == end_rows]
    s_noi_end = s_noi[s_noi["dataset_rows"].astype(int) == end_rows]

    if duck_end.empty or sqli_end.empty:
        return {"end_rows": None}

    duck_val = float(duck_end[metric].median())

    # Best-of SQLite at end: min(median idx, median noidx) with fallback
    idx_val = float(s_idx_end[metric].median()) if not s_idx_end.empty else np.nan
    noi_val = float(s_noi_end[metric].median()) if not s_noi_end.empty else np.nan

    best_val = np.nan
    best_var = None
    if np.isfinite(idx_val) and np.isfinite(noi_val):
        if idx_val <= noi_val:
            best_val, best_var = idx_val, "idx"
        else:
            best_val, best_var = noi_val, "noidx"
    elif np.isfinite(idx_val):
        best_val, best_var = idx_val, "idx"
    elif np.isfinite(noi_val):
        best_val, best_var = noi_val, "noidx"

    speedup = (best_val / duck_val) if (np.isfinite(best_val) and duck_val > 0) else np.nan

    # Optional context columns (may be missing)
    def medcol(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or df.empty:
            return np.nan
        return float(pd.to_numeric(df[col], errors="coerce").median())

    def medintcol(df: pd.DataFrame, col: str) -> int | None:
        if col not in df.columns or df.empty:
            return None
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if v.empty:
            return None
        return int(round(float(v.median())))

    # Use the best variant rows for spearman/psi at end (if present)
    s_best_end = s_idx_end if best_var == "idx" else s_noi_end if best_var == "noidx" else sqli_end

    out = {
        "end_rows": end_rows,
        "duck_p95_ms": duck_val,
        "sqlite_best_p95_ms": float(best_val) if np.isfinite(best_val) else np.nan,
        "sqlite_best_variant": best_var or "",
        "speedup_x": float(speedup) if np.isfinite(speedup) else np.nan,

        # Spearman correlations (may be NaN if undefined)
        "spearman_io":  medcol(s_best_end, "spearman_io_vs_latency"),
        "spearman_cpu": medcol(s_best_end, "spearman_cpu_vs_latency"),

        # Number of valid (latency, PSI-delta) pairs that Spearman was computed on
        "pairs_io":     medintcol(s_best_end, "psi_pairs_n_io"),
        "pairs_cpu":    medintcol(s_best_end, "psi_pairs_n_cpu"),

        # PSI context (kept in CSV; not all shown in LaTeX by default)
        "psi_io_p95_ms":  medcol(s_best_end, "p95_psi_io_full_ms"),
        "psi_cpu_p95_ms": medcol(s_best_end, "p95_psi_cpu_some_ms"),
    }
    return out


def _write_latex_scorecard(rows: pd.DataFrame, out_tex: Path, *, title: str) -> None:
    lines = []
    lines.append("% Auto-generated by analysis.extract_tables_readonly.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_latex_escape(title)}}}")
    lines.append("\\begin{tabular}{llr r l l r}")
    lines.append("\\toprule")
    lines.append("Query & Mode & Tags & BE (rows) & Small winner & Large winner & End speedup \\\\")
    lines.append("\\midrule")

    for _, r in rows.iterrows():
        be = "--"
        if np.isfinite(r.get("be_first", np.nan)):
            be = _fmt_int_rows(r["be_first"])
            if int(r.get("be_count", 0)) > 1:
                be = f"{be} ({int(r['be_count'])})"
        small = r.get("small_winner_fmt", "--")
        large = r.get("large_winner_fmt", "--")
        spx = _fmt_x(r.get("speedup_x", np.nan))

        lines.append(
            f"{_latex_escape(str(r['query_label']))} & "
            f"{_latex_escape(str(r['mode']))} & "
            f"{int(r['tags'])} & "
            f"{be} & "
            f"{_latex_escape(str(small))} & "
            f"{_latex_escape(str(large))} & "
            f"{spx}\\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{2pt}")
    lines.append("\\begin{flushleft}\\footnotesize "
                 "BE = first break-even point for DuckDB(auto) vs SQLite(best) using p95 latency; "
                 "parentheses show number of break-even points (multi-crossings). "
                 "End speedup = SQLite(best) p95 divided by DuckDB(auto) p95 at the largest common dataset size."
                 "\\end{flushleft}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")


def _write_latex_tailend(rows: pd.DataFrame, out_tex: Path, *, title: str) -> None:
    lines = []
    lines.append("% Auto-generated by analysis.extract_tables_readonly.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_latex_escape(title)}}}")
    lines.append("\\begin{tabular}{llr r r r l r l l}")
    lines.append("\\toprule")
    lines.append("Query & Mode & Tags & End rows & DuckDB p95 (ms) & SQLite(best) p95 (ms) & Best var & Speedup & $\\rho_{IO}$ & $\\rho_{CPU}$ \\\\")
    lines.append("\\midrule")

    for _, r in rows.iterrows():
        rho_io = _fmt_rho_with_n(r.get("spearman_io"), r.get("pairs_io"))
        rho_cpu = _fmt_rho_with_n(r.get("spearman_cpu"), r.get("pairs_cpu"))

        lines.append(
            f"{_latex_escape(str(r['query_label']))} & "
            f"{_latex_escape(str(r['mode']))} & "
            f"{int(r['tags'])} & "
            f"{_fmt_int_rows(r.get('end_rows'))} & "
            f"{_fmt_ms(r.get('duck_p95_ms'))} & "
            f"{_fmt_ms(r.get('sqlite_best_p95_ms'))} & "
            f"{_latex_escape(str(r.get('sqlite_best_variant') or '--'))} & "
            f"{_fmt_x(r.get('speedup_x'))} & "
            f"{_latex_escape(rho_io)} & "
            f"{_latex_escape(rho_cpu)}\\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{2pt}")
    lines.append("\\begin{flushleft}\\footnotesize "
                 "Speedup > 1 means DuckDB is faster (lower p95). "
                 "Spearman $\\rho$ is reported for the SQLite(best) variant at the tail size, together with the number of valid (latency, PSI-delta) pairs $n$. "
                 "n/a indicates too few pairs (n<3) or undefined correlation (e.g., constant input) even when n is available."
                 "\\end{flushleft}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="READ-ONLY extraction from existing CSVs; writes tables into a separate export folder."
    )
    ap.add_argument("--in-root", required=True, help="Campaign root, e.g. /mnt/edgebench/campaigns/2025-12-17_run03")
    ap.add_argument("--out-root", required=True, help="Export root (will be created), e.g. /mnt/edgebench/exports/2025-12-17_run03")
    ap.add_argument("--tags", default="20,50", help="Comma-separated tags to include (default: 20,50)")
    ap.add_argument("--queries", default="", help="Comma-separated query names. Empty = auto-detect from summary CSV.")
    ap.add_argument("--metric", default="p95_ms", choices=["p50_ms", "p95_ms"], help="Latency metric for extraction (default p95_ms).")
    ap.add_argument("--duck-profile", default="auto", choices=["auto", "t1"], help="DuckDB profile for tail-end (default auto).")
    ap.add_argument("--sqlite-variant", default="best", choices=["best", "idx", "noidx"], help="SQLite selection for break-even multi (default best).")
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    in_results = in_root / "results"
    out_tables = out_root / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    be_path = in_results / "break_even_multi.csv"
    sum_path = in_results / "summary_with_cis.csv"

    df_be = _read_csv_required(be_path)
    df_sum = _read_csv_required(sum_path)

    # Normalize columns we rely on
    if "dataset_tags" not in df_sum.columns:
        raise RuntimeError("summary_with_cis.csv missing required column: dataset_tags")
    if "dataset_rows" not in df_sum.columns:
        raise RuntimeError("summary_with_cis.csv missing required column: dataset_rows")
    if args.metric not in df_sum.columns:
        raise RuntimeError(f"summary_with_cis.csv missing required metric column: {args.metric}")

    tags = [int(x.strip()) for x in args.tags.split(",") if x.strip()]
    if args.queries.strip():
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    else:
        queries = sorted(df_sum["query"].dropna().astype(str).unique().tolist())

    # Build rows
    out_rows = []
    for q in queries:
        for warm in (False, True):
            mode = "warm" if warm else "cold"
            for tg in tags:
                be = _parse_be_row(
                    df_be, query=q, warm=warm, tags=tg,
                    metric=args.metric, duck_profile="auto", sqlite_variant=args.sqlite_variant
                )
                tail = _tail_end_stats(
                    df_sum, query=q, warm=warm, tags=tg,
                    metric=args.metric, duck_profile=args.duck_profile
                )
                out_rows.append({
                    "query": q,
                    "query_label": _q_label(q),
                    "mode": mode,
                    "warm": warm,
                    "tags": tg,
                    "be_points_all": be["be_points_all"],
                    "be_first": be["be_first"],
                    "be_count": be["be_count"],
                    "small_winner_fmt": _winner_fmt(be["small_winner"], be["small_winner_variant"]),
                    "large_winner_fmt": _winner_fmt(be["large_winner"], be["large_winner_variant"]),
                    **tail,
                })

    df_out = pd.DataFrame(out_rows)

    # Persist CSVs
    csv_score = out_tables / f"scorecard_{args.metric}.csv"
    df_out.to_csv(csv_score, index=False)

    # Write LaTeX tables
    tex_score = out_tables / f"table_scorecard_{args.metric}.tex"
    _write_latex_scorecard(
        df_out,
        tex_score,
        title=f"Scorecard ({args.metric}) — break-even, dominance, and tail-end speedup",
    )

    tex_tail = out_tables / f"table_tail_end_{args.metric}.tex"
    _write_latex_tailend(
        df_out,
        tex_tail,
        title=f"Tail-end values ({args.metric}) at largest common dataset size",
    )

    # Small report
    rep = out_root / "EXTRACT_REPORT.txt"
    rep.write_text(
        "\n".join([
            "READ-ONLY extraction report",
            f"in_root:  {in_root}",
            f"be_csv:   {be_path}",
            f"sum_csv:  {sum_path}",
            f"out_root: {out_root}",
            "",
            f"queries:  {', '.join(queries)}",
            f"tags:     {tags}",
            f"metric:   {args.metric}",
            f"duck_profile(tail): {args.duck_profile}",
            f"sqlite_variant(BE): {args.sqlite_variant}",
            "",
            f"wrote: {csv_score}",
            f"wrote: {tex_score}",
            f"wrote: {tex_tail}",
        ]),
        encoding="utf-8",
    )

    print("[ok] wrote:")
    print(" ", csv_score)
    print(" ", tex_score)
    print(" ", tex_tail)
    print(" ", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
