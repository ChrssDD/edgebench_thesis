
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------- parsing helpers ---------------------------------

def parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]

def parse_csv_strs(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]

def parse_points(val: Any) -> list[float]:
    """
    break_even_multi.csv sometimes stores NaN or 'nan' in be_points.
    Only keep finite >0 points.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    out: list[float] = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            x = float(tok)
        except Exception:
            continue
        if math.isfinite(x) and x > 0:
            out.append(x)
    return sorted(out)

def parse_zones(val: Any) -> list[dict[str, Any]]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        z = json.loads(s)
        return z if isinstance(z, list) else []
    except Exception:
        return []


# ------------------------- core BE logic -----------------------------------

def _near(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= max(eps, 1e-9 * max(1.0, a, b))

def _near_log10(a: float, b: float, tol_log10: float) -> bool:
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return False
    if a <= 0 or b <= 0:
        return False
    return abs(math.log10(a) - math.log10(b)) <= tol_log10

def choose_primary(points: list[float], zones: list[dict[str, Any]], *, min_rel: float, min_abs_ms: float) -> float | None:
    """
    Primary = erster Break-Even-Punkt, der an einer "signifikanten" Zonen-Grenze liegt.
    Falls keiner signifikant ist: nimm den ersten Punkt.
    (entspricht derive_primary_map_from_multi in utils.py)
    """
    if not points:
        return None

    def significant_at(x: float) -> bool:
        for z in zones or []:
            a = float(z.get("start_rows", float("nan")))
            b = float(z.get("end_rows", float("nan")))
            if not (math.isfinite(a) and math.isfinite(b)):
                continue

            mrel = float(z.get("mean_rel_delta", 0.0) or 0.0)
            mabs = float(z.get("median_delta_ms", 0.0) or 0.0)

            if (mrel < min_rel) and (abs(mabs) < min_abs_ms):
                continue

            if _near(x, a) or _near(x, b):
                return True
        return False

    for x in points:
        if significant_at(x):
            return float(x)
    return float(points[0])

def select_multi_extras(
    points: list[float],
    zones: list[dict[str, Any]],
    *,
    top_k: int,
    min_rel: float,
    min_abs_ms: float,
    primary_x: float | None,
    dedupe_tol_log10: float,
) -> list[float]:
    """
    Selects the same type of "top-k multi" points as annotate_be_multi():
      - only points on significant zone boundaries (thresholded by min_rel/min_abs)
      - ranked by strongest effect around the boundary (median_delta_ms, then mean_rel_delta)
      - returns up to top_k points, excluding ones too close to primary_x in log10 space
    """
    if not points or top_k <= 0:
        return []

    def score_for_x(x: float) -> tuple[float, float, float]:
        best_abs, best_rel = -1.0, -1.0
        for z in zones or []:
            a = float(z.get("start_rows", float("nan")))
            b = float(z.get("end_rows", float("nan")))
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            if not (_near(x, a) or _near(x, b)):
                continue

            mabs = abs(float(z.get("median_delta_ms", 0.0) or 0.0))
            mrel = float(z.get("mean_rel_delta", 0.0) or 0.0)

            if (mrel < min_rel) and (mabs < min_abs_ms):
                continue

            if (mabs > best_abs) or (mabs == best_abs and mrel > best_rel):
                best_abs, best_rel = mabs, mrel
        # -x to prefer earlier points when ties remain
        return (best_abs, best_rel, -float(x))

    ranked = sorted(points, key=score_for_x, reverse=True)
    ranked = [x for x in ranked if score_for_x(x)[0] >= 0.0]

    top_points = ranked[:top_k]  # wie Plot
    out = []
    for x in top_points:
        if primary_x is not None and _near_log10(x, primary_x, dedupe_tol_log10):
            continue
        out.append(float(x))
        if len(out) >= top_k:
            break
    return out


# ------------------------- formatting (rows, ms, TeX) -----------------------

def fmt_rows(x: float | None) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "--"
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x/1_000:.2f}k"
    return f"{x:.0f}"

def fmt_ms(x: float | None) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "--"
    # keep compact for table
    if abs(x) >= 1000:
        return f"{x/1000:.2f}s"
    if abs(x) >= 10:
        return f"{x:.0f}ms"
    return f"{x:.2f}ms"

def fmt_rho(x: float | None) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "--"
    return f"{x:+.2f}"

def tex_escape(s: str) -> str:
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("_", r"\_")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    return s


# ------------------------- summary helpers (PSI + spearman) -----------------

def _psi_col(kind: str, stat: str) -> str:
    if kind == "io":
        return "p95_psi_io_full_ms" if stat == "p95" else "mean_psi_io_full_ms"
    else:
        return "p95_psi_cpu_some_ms" if stat == "p95" else "mean_psi_cpu_some_ms"

def _series_at_x(series: pd.Series, x: float) -> float | None:
    """Linear interpolation on x (dataset_rows)."""
    if series is None or series.empty:
        return None
    s = series.dropna().astype(float).sort_index()
    if s.empty or not (math.isfinite(x) and x > 0):
        return None
    xs = s.index.to_numpy(dtype=float)
    ys = s.to_numpy(dtype=float)
    if x < xs[0] or x > xs[-1]:
        return None
    # exact hit?
    if x in s.index:
        v = float(s.loc[x])
        return v if math.isfinite(v) else None
    v = float(np.interp(x, xs, ys))
    return v if math.isfinite(v) else None

def _cfg_mask(df: pd.DataFrame, *, query: str, warm: bool, tags: int, scan_mode: str,
              engine: str, profile: str | None = None, sqlite_variant: str | None = None) -> pd.DataFrame:
    d = df[(df["query"] == query) & (df["warm"] == warm) & (df["dataset_tags"] == tags) & (df["scan_mode"] == scan_mode) & (df["engine"] == engine)].copy()
    if engine == "duckdb":
        if profile == "t1":
            d = d[pd.notna(d["threads"]) & (d["threads"].astype("Int64") == 1)]
        else:
            d = d[d["threads"].isna()]
    else:
        if sqlite_variant in ("idx", "noidx"):
            d = d[d["sqlite_variant"] == sqlite_variant]
    return d

def _median_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    if df.empty or value_col not in df.columns:
        return pd.Series(dtype=float)
    s = df.groupby("dataset_rows")[value_col].median().sort_index()
    s.index = s.index.astype(float)
    return s

def best_sqlite_variant_at_x(df_summary: pd.DataFrame, *, query: str, warm: bool, tags: int, scan_mode: str, metric: str, x: float) -> str | None:
    """Pick idx vs noidx by interpolated latency at x."""
    a = _cfg_mask(df_summary, query=query, warm=warm, tags=tags, scan_mode=scan_mode, engine="sqlite", sqlite_variant="idx")
    b = _cfg_mask(df_summary, query=query, warm=warm, tags=tags, scan_mode=scan_mode, engine="sqlite", sqlite_variant="noidx")
    s_idx = _median_series(a, metric)
    s_no  = _median_series(b, metric)
    v_idx = _series_at_x(s_idx, x)
    v_no  = _series_at_x(s_no, x)
    if v_idx is None and v_no is None:
        return None
    if v_idx is None:
        return "noidx"
    if v_no is None:
        return "idx"
    return "idx" if v_idx <= v_no else "noidx"

def context_at_primary(
    df_summary: pd.DataFrame,
    *,
    query: str,
    warm: bool,
    tags: int,
    scan_mode: str,
    metric: str,
    primary_x: float,
    duck_profile: str,
    psi_stat: str,
) -> dict[str, float | str | None]:
    """
    Returns context values evaluated at primary_x:
      sqlite_variant (best by latency at x),
      psi_io/sqlite, psi_cpu/sqlite, psi_io/duck, psi_cpu/duck,
      spearman_io/sqlite, spearman_cpu/sqlite, spearman_io/duck, spearman_cpu/duck
    """
    sqlite_var = best_sqlite_variant_at_x(df_summary, query=query, warm=warm, tags=tags, scan_mode=scan_mode, metric=metric, x=primary_x)

    out: dict[str, float | str | None] = {"sqlite_variant": sqlite_var}

    # PSI series
    for kind in ("io", "cpu"):
        col = _psi_col(kind, psi_stat)
        d_sql = _cfg_mask(df_summary, query=query, warm=warm, tags=tags, scan_mode=scan_mode, engine="sqlite", sqlite_variant=sqlite_var)
        d_duck = _cfg_mask(df_summary, query=query, warm=warm, tags=tags, scan_mode=scan_mode, engine="duckdb", profile=duck_profile)

        out[f"psi_{kind}_sql"]  = _series_at_x(_median_series(d_sql, col), primary_x) if sqlite_var else None
        out[f"psi_{kind}_duck"] = _series_at_x(_median_series(d_duck, col), primary_x)

    # Spearman columns (local, from summary)
    for kind in ("io", "cpu"):
        scol = f"spearman_{kind}_vs_latency"
        d_sql = _cfg_mask(df_summary, query=query, warm=warm, tags=tags, scan_mode=scan_mode, engine="sqlite", sqlite_variant=sqlite_var)
        d_duck = _cfg_mask(df_summary, query=query, warm=warm, tags=tags, scan_mode=scan_mode, engine="duckdb", profile=duck_profile)

        out[f"rho_{kind}_sql"]  = _series_at_x(_median_series(d_sql, scol), primary_x) if sqlite_var else None
        out[f"rho_{kind}_duck"] = _series_at_x(_median_series(d_duck, scol), primary_x)

    return out


# ------------------------- TeX table writers --------------------------------

def make_be_cell(
    *,
    points_all: list[float],
    primary_x: float | None,
    extras: list[float],
    mode: str,
    show_counts: bool,
    with_context: bool,
    ctx: dict[str, float | str | None] | None,
) -> str:
    """
    Build one cell for one profile:
      - primary is bold (as in the legend)
      - extras are appended (the same "top-k multi" selection as in the plot)
      - optional context line: PSI and Spearman at primary (sql/duck)
    """
    if primary_x is None:
        return "--"

    shown = [primary_x] + (extras or [])
    # de-dup in log space for readability (rare duplicates)
    uniq: list[float] = []
    for x in shown:
        if not any(_near_log10(x, y, 1e-12) for y in uniq):
            uniq.append(float(x))

    # format: "P <primary> — <x1>; <x2>"
    parts: list[str] = []
    parts.append(rf"P \textbf{{{fmt_rows(primary_x)}}}")

    if mode in ("selected", "both") and extras:
        # Use en-dash when there is exactly one extra (range-like), else semicolons
        if len(extras) == 1:
            parts.append(rf"\textendash\ {fmt_rows(extras[0])}")
        else:
            parts.append(r"\textendash\ " + "; ".join(fmt_rows(x) for x in extras))

    if mode == "all":
        # Show all points, primary bold inside list
        def fmt_one(x: float) -> str:
            if _near(x, primary_x):
                return rf"\textbf{{{fmt_rows(x)}}}"
            return fmt_rows(x)
        parts = [ "; ".join(fmt_one(x) for x in points_all) ]  # replace
    cell = " ".join(parts).strip()

    if show_counts:
        if mode == "all":
            cell += rf" (n={len(points_all)})"
        else:
            cell += rf" (n={1+len(extras)}/{len(points_all)})"

    if with_context and ctx:
        sv = ctx.get("sqlite_variant")
        io_sql  = fmt_ms(ctx.get("psi_io_sql"))   # type: ignore[arg-type]
        io_duck = fmt_ms(ctx.get("psi_io_duck"))  # type: ignore[arg-type]
        cpu_sql  = fmt_ms(ctx.get("psi_cpu_sql"))   # type: ignore[arg-type]
        cpu_duck = fmt_ms(ctx.get("psi_cpu_duck"))  # type: ignore[arg-type]
        rio_sql  = fmt_rho(ctx.get("rho_io_sql"))   # type: ignore[arg-type]
        rio_duck = fmt_rho(ctx.get("rho_io_duck"))  # type: ignore[arg-type]
        rcpu_sql  = fmt_rho(ctx.get("rho_cpu_sql"))   # type: ignore[arg-type]
        rcpu_duck = fmt_rho(ctx.get("rho_cpu_duck"))  # type: ignore[arg-type]

        # compact second line; keep it optional / lightweight
        cell += (
            r"\newline\scriptsize "
            + rf"sql={sv or '--'}; IO@P {io_sql}/{io_duck} (\ensuremath{{\rho}}={rio_sql}/{rio_duck}); "
            + rf"CPU@P {cpu_sql}/{cpu_duck} (\ensuremath{{\rho}}={rcpu_sql}/{rcpu_duck})"
        )

    return cell

def write_tabular(out_path: Path, rows: list[dict[str, str]], caption_hint: str) -> None:
    """
    Uses p-columns to prevent overflow; requires booktabs (toprule/midrule/bottomrule).
    """
    lines: list[str] = []
    lines.append("% " + caption_hint)
    lines.append(r"\begin{tabular}{l p{0.39\linewidth} p{0.39\linewidth}}")
    lines.append(r"\toprule")
    lines.append(r"Query & DuckDB(auto) & DuckDB(t1) \\")
    lines.append(r"\midrule")
    for r in rows:
        q = r"\texttt{" + tex_escape(str(r["query"])) + "}"
        lines.append(f"{q} & {r['auto']} & {r['t1']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------- main ---------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--be-multi", required=True, help="Path to break_even_multi.csv")
    ap.add_argument("--summary", default="", help="Optional: summary_with_cis.csv (for PSI/Spearman context)")
    ap.add_argument("--outdir", required=True, help="Output dir for .tex fragments")
    ap.add_argument("--tags", default="20,50")
    ap.add_argument("--metrics", default="p50_ms,p95_ms")
    ap.add_argument("--scan-mode", default="table", choices=["table","file"])
    ap.add_argument("--min-rel", type=float, default=0.10)
    ap.add_argument("--min-abs-ms", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=2, help="How many extra multi-BE points to show (per profile).")
    ap.add_argument("--dedupe-tol-log10", type=float, default=0.01)
    ap.add_argument("--mode", default="selected", choices=["selected","all"], help="selected = matches plots; all = raw be_points list.")
    ap.add_argument("--show-counts", action="store_true", help="Append (n=shown/total) or (n=total).")
    ap.add_argument("--with-context", action="store_true", help="Append PSI + Spearman values at primary BE to each cell.")
    ap.add_argument("--psi-stat", default="p95", choices=["p95","mean"])

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    be_df = pd.read_csv(Path(args.be_multi))

    # normalize columns
    be_df["query"] = be_df["query"].astype(str).str.strip()
    be_df["metric"] = be_df["metric"].astype(str).str.strip()
    be_df["duckdb_profile"] = be_df["duckdb_profile"].astype(str).str.strip().str.lower()
    be_df["sqlite_variant"] = be_df["sqlite_variant"].astype(str).str.strip().str.lower()
    be_df["warm"] = be_df["warm"].astype(str).str.strip().str.lower().isin(("1","true","yes","warm"))
    be_df["tags"] = pd.to_numeric(be_df["tags"], errors="coerce").astype("Int64")

    tags_list = parse_csv_ints(args.tags)
    metrics = parse_csv_strs(args.metrics)

    # summary optional
    df_summary: Optional[pd.DataFrame] = None
    if args.with_context:
        if not args.summary:
            raise SystemExit("--with-context requires --summary")
        df_summary = pd.read_csv(Path(args.summary))
        # minimal type normalization
        df_summary["query"] = df_summary["query"].astype(str).str.strip()
        df_summary["scan_mode"] = df_summary["scan_mode"].astype(str).str.strip().str.lower()
        df_summary["engine"] = df_summary["engine"].astype(str).str.strip().str.lower()
        if "sqlite_variant" in df_summary.columns:
            df_summary["sqlite_variant"] = df_summary["sqlite_variant"].astype(str).str.strip().str.lower()
        if "threads" in df_summary.columns:
            df_summary["threads"] = pd.to_numeric(df_summary["threads"], errors="coerce")
        df_summary["warm"] = df_summary["warm"].astype(bool)
        df_summary["dataset_tags"] = pd.to_numeric(df_summary["dataset_tags"], errors="coerce").astype("Int64")
        df_summary["dataset_rows"] = pd.to_numeric(df_summary["dataset_rows"], errors="coerce")
        # keep scan-mode requested
        df_summary = df_summary[df_summary["scan_mode"] == args.scan_mode].copy()

    for tags in tags_list:
        for metric in metrics:
            for warm in (False, True):
                sub = be_df[
                    (be_df["tags"] == tags)
                    & (be_df["metric"] == metric)
                    & (be_df["warm"] == warm)
                    & (be_df["sqlite_variant"] == "best")
                    & (be_df["duckdb_profile"].isin(["auto", "t1"]))
                ].copy()

                if sub.empty:
                    continue

                rows_out: list[dict[str, str]] = []
                for q in sorted(sub["query"].unique()):
                    recs = sub[sub["query"] == q]
                    row = {"query": q, "auto": "--", "t1": "--"}

                    for prof in ("auto", "t1"):
                        r = recs[recs["duckdb_profile"] == prof]
                        if r.empty:
                            continue

                        r0 = r.iloc[0]
                        pts = parse_points(r0.get("be_points"))
                        zones = parse_zones(r0.get("zones_json"))
                        prim = choose_primary(pts, zones, min_rel=args.min_rel, min_abs_ms=args.min_abs_ms)
                        extras = select_multi_extras(
                            pts, zones,
                            top_k=args.top_k,
                            min_rel=args.min_rel,
                            min_abs_ms=args.min_abs_ms,
                            primary_x=prim,
                            dedupe_tol_log10=args.dedupe_tol_log10,
                        )

                        ctx = None
                        if args.with_context and df_summary is not None and prim is not None:
                            ctx = context_at_primary(
                                df_summary,
                                query=q, warm=warm, tags=tags, scan_mode=args.scan_mode,
                                metric=metric, primary_x=float(prim), duck_profile=prof,
                                psi_stat=args.psi_stat,
                            )

                        row[prof] = make_be_cell(
                            points_all=pts,
                            primary_x=prim,
                            extras=extras,
                            mode=args.mode,
                            show_counts=args.show_counts,
                            with_context=args.with_context,
                            ctx=ctx,
                        )

                    rows_out.append(row)

                tag_name = "warm" if warm else "cold"
                out_path = outdir / f"be_multi__tags{tags}__{metric}__{tag_name}.tex"
                hint = (
                    f"Multi break-even from break_even_multi.csv (sqlite_variant=best). "
                    f"primary = first significant boundary (bold); extras = top-k strongest (as in plots). "
                    f"tags={tags}, metric={metric}, {tag_name}; "
                    f"min_rel={args.min_rel}, min_abs_ms={args.min_abs_ms}, top_k={args.top_k}, dedupe_tol_log10={args.dedupe_tol_log10}; "
                    f"mode={args.mode}"
                    + (f"; context=PSI({args.psi_stat})+Spearman" if args.with_context else "")
                )
                write_tabular(out_path, rows_out, hint)

    print(f"[ok] wrote tables -> {outdir}")

if __name__ == "__main__":
    main()
