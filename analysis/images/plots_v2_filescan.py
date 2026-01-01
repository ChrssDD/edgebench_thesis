# analysis/images/plots_v2_filescan.py
"""
Plots für Filescan-Vergleich (DuckDB table vs file) mit optionalen Break-Even-Overlays.

Erzeugt:
- Absolute Latenzkurven pro Query & warm/cold (DuckDB table vs DuckDB file; auto & t1)
- Ratio-Plots (file ÷ table)
- Optional: Break-Even-Linien aus compare_file_scan-Multi-CSV(s)

Beispielaufruf (pooled + optional per-format, falls vorhanden):
python -m analysis.images.plots_v2_filescan \
  --baseline-summary /mnt/edgebench/campaigns/2025-10-18_begrid_baseline_t20/results/summary_with_cis.csv \
  --filescan-summary   /mnt/edgebench/campaigns/2025-10-20T094010Z_begrid_filescan_t20/results/summary_with_cis.csv \
  --metrics p50_ms,p95_ms \
  --tags 20 \
  --be-multi-pooled /mnt/edgebench/campaigns/2025-10-20T094010Z_begrid_filescan_t20/results/break_even_multi_filescan.csv \
  --be-multi-by-format /mnt/edgebench/campaigns/2025-10-20T094010Z_begrid_filescan_t20/results/break_even_multi_filescan_by_format.csv \
  --outdir /mnt/edgebench/campaigns/2025-10-20T094010Z_begrid_filescan_t20/results/plots/filescan
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import apply_style
from .utils import ensure_dir


# ---------- IO ----------

def _need_cols(df: pd.DataFrame, need: list[str], where: str):
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{where} missing column(s): {miss}")

def _norm_bool_to_tf_str(v) -> str:
    s = str(v).strip().lower()
    return "True" if s in ("1","true","yes","warm") else "False"

def _norm_threads_to_key(v) -> str:
    s = str(v).strip().lower()
    if s in ("1","1.0","t1"):
        return "1"
    return "auto" if s in ("", "none", "nan", "auto") else s  # defensive

def load_summary(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    _need_cols(df, ["engine","query","warm","threads","scan_mode","dataset_rows"], str(p))

    # optionale Formatmarker -> normalisierte file_format
    has_parq = df.columns.str.lower().isin(["parquet"]).any()
    has_csvc = df.columns.str.lower().isin(["csv"]).any()

    file_fmt = []
    for _, r in df.iterrows():
        f = ""
        if "file_format" in df.columns and pd.notna(r.get("file_format")) and str(r["file_format"]).strip():
            f = str(r["file_format"]).strip().lower()
        else:
            if has_parq and str(r.get("parquet","")).strip().lower() in ("1","true","yes"):
                f = "parquet"
            elif has_csvc and str(r.get("csv","")).strip().lower() in ("1","true","yes"):
                f = "csv"
        file_fmt.append(f)
    df = df.copy()
    df["file_format"] = file_fmt

    # threads-Key für spätere BE-Map-Lookups (auto/1)
    thr_key = []
    for v in df["threads"]:
        if pd.isna(v):
            thr_key.append("auto")
        else:
            thr_key.append(_norm_threads_to_key(v))
    df["threads_key"] = thr_key

    # warm-Key als "True"/"False"-String (BE-CSV nutzt Strings)
    df["warm_key"] = df["warm"].map(_norm_bool_to_tf_str)

    return df


# ---------- BE-Loader ----------

def _parse_be_points(s: str) -> list[float]:
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            # still, skip invalid token
            continue
    return sorted(out)

def _load_be_multi_map(p: Path) -> dict:
    """
    Erwartete Spalten (aus scripts/compare_file_scan.py):
      query,warm,threads,tags,metric,file_format,be_points,be_count,zones_json,remark
    Map-Key: (query, warm('True'/'False'), threads('auto'/'1'), tags(int), file_format or "")
    """
    m = {}
    if p is None or not p.exists():
        return m
    df = pd.read_csv(p)
    need = {"query","warm","threads","tags","metric","file_format","be_points"}
    if not need.issubset(df.columns):
        return m
    for _, r in df.iterrows():
        key = (
            str(r.get("query","?")),
            _norm_bool_to_tf_str(r.get("warm","False")),
            _norm_threads_to_key(r.get("threads","auto")),
            int(r.get("tags", 0)) if pd.notna(r.get("tags")) else 0,
            str(r.get("file_format","") or "")
        )
        pts = _parse_be_points(str(r.get("be_points","")))
        # mehrere metrics möglich → verschachteln: m[(...)] ist dict metric->list
        metric = str(r.get("metric","p95_ms"))
        d = m.get(key, {})
        d[metric] = pts
        m[key] = d
    return m


# ---------- Pivots ----------

def pivot_duck_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[(df["engine"]=="duckdb") & (df["scan_mode"]=="table")].copy()
    auto = sub[sub["threads"].isna()].groupby(["query","warm","dataset_rows"], as_index=True)[metric].median().rename("auto")
    t1   = sub[sub["threads"]==1].groupby(["query","warm","dataset_rows"], as_index=True)[metric].median().rename("t1")
    out = pd.concat([auto, t1], axis=1)
    if out.empty:
        return out
    out.index = pd.MultiIndex.from_tuples(out.index, names=["query","warm","dataset_rows"])
    return out.sort_index()

def pivot_duck_file(df: pd.DataFrame, metric: str, file_format: str | None = None) -> pd.DataFrame:
    sub = df[(df["engine"]=="duckdb") & (df["scan_mode"]=="file")].copy()
    if file_format:
        sub = sub[sub["file_format"] == file_format]
    auto = sub[sub["threads"].isna()].groupby(["query","warm","dataset_rows"], as_index=True)[metric].median().rename("auto")
    t1   = sub[sub["threads"]==1].groupby(["query","warm","dataset_rows"], as_index=True)[metric].median().rename("t1")
    out = pd.concat([auto, t1], axis=1)
    if out.empty:
        return out
    out.index = pd.MultiIndex.from_tuples(out.index, names=["query","warm","dataset_rows"])
    return out.sort_index()


# ---------- Plot-Helfer ----------

def _series_or_nan(df: pd.DataFrame, col: str, rows: list[int]) -> pd.Series:
    if df is None or df.empty or col not in getattr(df, "columns", []):
        return pd.Series(index=rows, dtype=float)
    s = df.reset_index().set_index("dataset_rows")[col]
    return s.reindex(rows)

def _draw_be_lines(ax, be_points: list[float], *, line_style="--", line_alpha=0.8):
    for x in be_points:
        ax.axvline(float(x), linestyle=line_style, linewidth=1.1, color="0.5", alpha=line_alpha)


# ---------- Plotter: absolute & ratio ----------

def plot_abs(table_df: pd.DataFrame, file_df: pd.DataFrame, outdir: Path,
             metric: str, mode_label: str, format_label: str | None,
             be_map: dict | None, inferred_tags: int,
             line_style="--", line_alpha=0.8):
    """
    Absolute Latenzen: DuckDB table vs file (auto & t1) pro query & warm.
    Optional: Break-Even-Linien aus be_map (pooled oder by-format).
    """
    written=[]
    if (table_df is None or table_df.empty) and (file_df is None or file_df.empty):
        return written

    queries = sorted(set(table_df.index.get_level_values(0) if table_df is not None and not table_df.empty else [])
                     | set(file_df.index.get_level_values(0) if file_df is not None and not file_df.empty else []))
    for q in queries:
        for warm in (False, True):
            t = table_df.xs((q,warm), level=(0,1), drop_level=False) if (table_df is not None and (q,warm) in getattr(table_df, "index", [])) else None
            f = file_df .xs((q,warm), level=(0,1), drop_level=False) if (file_df  is not None and (q,warm) in getattr(file_df,  "index", [])) else None
            if t is None or f is None or t.empty or f.empty:
                continue
            rows = sorted(set(t.index.get_level_values("dataset_rows")) | set(f.index.get_level_values("dataset_rows")))
            for mode_name in ("auto","t1"):
                tt = _series_or_nan(t, mode_name, rows)
                ff = _series_or_nan(f, mode_name, rows)
                if tt.dropna().empty and ff.dropna().empty:
                    continue
                subdir = ensure_dir(outdir / "abs" / (format_label or "pooled") / metric / ("warm" if warm else "cold") / mode_name)
                fname = subdir / f"{q}__duckdb_{mode_name}__file_vs_table__{metric}.png"
                fig, ax = plt.subplots(figsize=(5.2,3.6))
                if tt.notna().any(): ax.plot(rows, tt.values, marker="o", lw=2, label="DuckDB table")
                if ff.notna().any(): ax.plot(rows, ff.values, marker="^", lw=2, linestyle="--", label=f"DuckDB file{(' ('+format_label+')') if format_label else ''}")
                ax.set_xscale("log")
                ax.set_xlabel("Total rows (log scale)")
                ax.set_ylabel(f"{metric} (ms)")
                ax.grid(True, which="both", alpha=0.35)
                ttl_fmt = f" — {format_label.upper()}" if format_label else ""
                ax.set_title(f"{q} — {'warm' if warm else 'cold'} — DuckDB({mode_name}){ttl_fmt}")
                ax.legend(loc="best", frameon=True, fontsize=9)

                # BE-Linien
                if be_map:
                    warm_key = "True" if warm else "False"
                    thr_key = "1" if mode_name == "t1" else "auto"
                    fmt_key = (format_label or "")
                    pts = be_map.get((q, warm_key, thr_key, inferred_tags, fmt_key), {}).get(metric, [])
                    if pts:
                        _draw_be_lines(ax, pts, line_style=line_style, line_alpha=line_alpha)

                fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)
                written.append(fname)
    return written

def plot_ratio(table_df: pd.DataFrame, file_df: pd.DataFrame, outdir: Path,
               metric: str, format_label: str | None,
               be_map: dict | None, inferred_tags: int,
               line_style="--", line_alpha=0.8):
    """
    Ratio file/table für DuckDB (auto & t1).
    Optional: Break-Even-Linien aus be_map (pooled oder by-format).
    """
    written=[]
    if (table_df is None or table_df.empty) and (file_df is None or file_df.empty):
        return written

    queries = sorted(set(table_df.index.get_level_values(0) if table_df is not None and not table_df.empty else [])
                     | set(file_df.index.get_level_values(0) if file_df is not None and not file_df.empty else []))
    for q in queries:
        for warm in (False, True):
            t = table_df.xs((q,warm), level=(0,1), drop_level=False) if (table_df is not None and (q,warm) in getattr(table_df, "index", [])) else None
            f = file_df .xs((q,warm), level=(0,1), drop_level=False) if (file_df  is not None and (q,warm) in getattr(file_df,  "index", [])) else None
            if t is None or f is None or t.empty or f.empty:
                continue
            rows = sorted(set(t.index.get_level_values("dataset_rows")) | set(f.index.get_level_values("dataset_rows")))
            for mode_name in ("auto","t1"):
                ts = t.reset_index().set_index("dataset_rows")[mode_name] if (t is not None and (mode_name in t.columns)) else None
                fs = f.reset_index().set_index("dataset_rows")[mode_name] if (f is not None and (mode_name in f.columns)) else None
                if ts is None or fs is None:
                    continue
                ratio = (fs / ts).reindex(rows).dropna()
                if ratio.empty:
                    continue
                subdir = ensure_dir(outdir / "ratio" / (format_label or "pooled") / metric / ("warm" if warm else "cold") / mode_name)
                fname = subdir / f"{q}__duckdb_{mode_name}__file_over_table_ratio__{metric}.png"
                fig, ax = plt.subplots(figsize=(5.2,3.6))
                ax.plot(ratio.index.values, ratio.values, marker="o", lw=2)
                ax.axhline(1.0, ls=":", lw=1.0, color="0.5")
                ax.set_xscale("log")
                ax.set_xlabel("Total rows (log scale)")
                ax.set_ylabel(f"file ÷ table — {metric}")
                ttl_fmt = f" — {format_label.upper()}" if format_label else ""
                ax.set_title(f"{q} — {'warm' if warm else 'cold'} — DuckDB({mode_name}){ttl_fmt}")
                ax.grid(True, which="both", alpha=0.35)

                # BE-Linien
                if be_map:
                    warm_key = "True" if warm else "False"
                    thr_key = "1" if mode_name == "t1" else "auto"
                    fmt_key = (format_label or "")
                    pts = be_map.get((q, warm_key, thr_key, inferred_tags, fmt_key), {}).get(metric, [])
                    if pts:
                        _draw_be_lines(ax, pts, line_style=line_style, line_alpha=line_alpha)

                fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)
                written.append(fname)
    return written


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-summary", required=True, help="baseline (table-scan) summary_with_cis.csv")
    ap.add_argument("--filescan-summary", required=True, help="filescan campaign summary_with_cis.csv")
    ap.add_argument("--outdir", required=False, default="plots/filescan")
    ap.add_argument("--metrics", default="p50_ms,p95_ms")
    ap.add_argument("--tags", type=int, default=None)
    # neue Optionen für BE-Overlays:
    ap.add_argument("--be-multi-pooled", required=False, help="break_even_multi_filescan.csv (pooled)")
    ap.add_argument("--be-multi-by-format", required=False, help="break_even_multi_filescan_by_format.csv")
    ap.add_argument("--be-line-style", default="--")
    ap.add_argument("--be-line-alpha", type=float, default=0.8)
    args = ap.parse_args()

    apply_style()

    base = load_summary(Path(args.baseline_summary))
    new  = load_summary(Path(args.filescan_summary))

    # optional auf Tags filtern
    if args.tags is not None and "dataset_tags" in base.columns:
        base = base[base["dataset_tags"] == args.tags]
    if args.tags is not None and "dataset_tags" in new.columns:
        new = new[new["dataset_tags"] == args.tags]

    outdir = ensure_dir(Path(args.outdir))

    # Tags ableiten, falls nicht via CLI gesetzt
    if args.tags is not None:
        inferred_tags = int(args.tags)
    else:
        cand = pd.concat([
            base.get("dataset_tags", pd.Series(dtype=float)),
            new.get("dataset_tags", pd.Series(dtype=float))
        ]).dropna().unique()
        inferred_tags = int(cand[0]) if len(cand) == 1 else 0

    # BE-Maps laden
    be_pool = _load_be_multi_map(Path(args.be_multi_pooled)) if args.be_multi_pooled else {}
    be_byfmt = _load_be_multi_map(Path(args.be_multi_by_format)) if args.be_multi_by_format else {}

    written=[]
    # welche Formate sind in der Filescan-Summary markiert?
    formats_present = sorted([f for f in ["csv","parquet"] if (len(new) and (new["file_format"] == f).any())])

    for metric in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        base_tbl = pivot_duck_table(base, metric)

        # pooled (ohne Formattrennung) — F4-Basis
        new_file_pooled = pivot_duck_file(new, metric, file_format=None)
        if not base_tbl.empty and not new_file_pooled.empty:
            written += plot_abs(
                base_tbl, new_file_pooled, outdir, metric, "pooled", None,
                be_map=be_pool, inferred_tags=inferred_tags,
                line_style=args.be_line_style, line_alpha=args.be_line_alpha
            )
            written += plot_ratio(
                base_tbl, new_file_pooled, outdir, metric, None,
                be_map=be_pool, inferred_tags=inferred_tags,
                line_style=args.be_line_style, line_alpha=args.be_line_alpha
            )

        # per-format Facetten — falls Marker vorhanden und ggf. passende BE-CSV
        for fmt in formats_present:
            new_file_fmt = pivot_duck_file(new, metric, file_format=fmt)
            if new_file_fmt.empty:
                continue
            written += plot_abs(
                base_tbl, new_file_fmt, outdir, metric, fmt, fmt,
                be_map=(be_byfmt or be_pool), inferred_tags=inferred_tags,
                line_style=args.be_line_style, line_alpha=args.be_line_alpha
            )
            written += plot_ratio(
                base_tbl, new_file_fmt, outdir, metric, fmt,
                be_map=(be_byfmt or be_pool), inferred_tags=inferred_tags,
                line_style=args.be_line_style, line_alpha=args.be_line_alpha
            )

    print(f"[ok] wrote {len(written)} filescan plots → {outdir}")

if __name__ == "__main__":
    main()
