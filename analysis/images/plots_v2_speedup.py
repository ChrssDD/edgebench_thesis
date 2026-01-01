# analysis/images/plots_v2_speedup.py
import argparse
from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import RESULTS_DIR
from .style import apply_style
from .utils import (
    ensure_dir, load_summary,
    load_be_map, load_be_multi,
    derive_primary_map_from_multi,
    annotate_be_multi, render_be_combined_legend, BE_COLORS,
)

# ---------- speedup series ----------------------------------------------------
def _agg(series: pd.Series, stat: str) -> float:
    if stat == "median":
        return float(series.median())
    return float(series.min())  # default: "min" = best-of

def speedup_series(
    df: pd.DataFrame,
    *,
    query: str,
    warm: bool,
    metric: str,
    baseline: str,          # "auto" | "t1"
    scan_mode: str,
    stat: str = "min",      # "min" | "median"
) -> pd.Series:
    """
    Speedup = SQLite_best / DuckDB(baseline), je dataset_rows.
    SQLite_best = best-of (über SQLite-Varianten) nach Aggregation 'stat'.
    DuckDB(baseline): threads==1 für t1, threads isna() für auto.
    """
    d = df[(df["query"] == query) & (df["warm"] == warm)].copy()
    if scan_mode != "any":
        d = d[d["scan_mode"] == scan_mode].copy()
    if d.empty:
        return pd.Series(dtype=float)

    # DuckDB baseline auswählen
    if baseline == "t1":
        duck = d[(d["engine"] == "duckdb") & (d["threads"] == 1)].copy()
    else:
        duck = d[(d["engine"] == "duckdb") & (d["threads"].isna())].copy()

    sql = d[(d["engine"] == "sqlite")].copy()
    if duck.empty or sql.empty:
        return pd.Series(dtype=float)

    # Aggregation je dataset_rows
    duck_agg = duck.groupby("dataset_rows")[metric].apply(lambda s: _agg(s, stat))
    # Für SQLite: best-of je rows über alle Varianten
    sql_agg = (sql.groupby(["dataset_rows", "sqlite_variant"])[metric]
                  .apply(lambda s: _agg(s, stat))
                  .unstack("sqlite_variant")
                  .min(axis=1))

    both = pd.concat([duck_agg.rename("duck"), sql_agg.rename("sql")], axis=1).dropna()
    if both.empty:
        return pd.Series(dtype=float)

    s = (both["sql"] / both["duck"]).sort_index()
    s.name = f"speedup_{stat}"
    return s

# ---------- helpers for BE overlay -------------------------------------------
def _primary_for_baseline(primary_map: dict, *, query: str, warm: bool, tags: int | None, baseline: str) -> float | None:
    """
    primary_map ist 'be_map-ähnlich': Keys wie in derive_primary_map_from_multi().
    Wir extrahieren den Wert für das gewählte Profil (auto|t1), ggf. tag-spezifisch.
    """
    prof = baseline  # "auto" oder "t1"
    if tags is None:
        return primary_map.get((query, warm, "best", prof), None)
    v = primary_map.get((query, warm, "best", prof, tags), None)
    return v if v is not None else primary_map.get((query, warm, "best", prof), None)

def _width_for_outside(legend_outside: bool, be_box_outside: bool) -> float:
    return 6.8 if (legend_outside or be_box_outside) else 5.6

# ---------- single speedup plot ----------------------------------------------
def plot_speedup(
    df: pd.DataFrame,
    *,
    be_multi_map: dict,
    primary_be_map: dict,
    outdir: Path,
    query: str,
    warm: bool,
    metric: str,
    tags: int | None,
    scan_mode: str,
    baseline: str,                 # "auto" | "t1"
    stat: str,                     # "min" | "median"
    legend_outside: bool,
    be_box_outside: bool,
    fig_height: float,
    multi_top_k: int,
    multi_min_rel: float,
    multi_min_abs_ms: float,
    multi_dedupe_tol_log10: float,
    ylog: bool,
):
    s = speedup_series(
        df, query=query, warm=warm, metric=metric,
        baseline=baseline, scan_mode=scan_mode, stat=stat
    )
    if s.empty:
        return None

    subdir = ensure_dir(outdir / metric / ("warm" if warm else "cold") / baseline)
    fname = subdir / f"{query}__speedup_sqlite_best_over_duckdb_{baseline}__{metric}__{scan_mode}.png"

    fig, ax = plt.subplots(
        figsize=(_width_for_outside(legend_outside, be_box_outside), fig_height),
        constrained_layout=False
    )

    # Linie
    ax.plot(s.index.values, s.values, marker="o", linewidth=2.0)

    # Referenzlinien
    ax.axhline(1.0, ls=":", lw=1.0, color="0.5", zorder=1)
    if ylog:
        ax.set_yscale("log")

    # Achsen
    ax.set_xscale("log")
    ax.set_xlabel("Total rows (log scale)")
    ax.set_ylabel(f"Speedup (SQLite_best ÷ DuckDB({baseline})) — {metric}")
    title = f"{query} — {'warm' if warm else 'cold'} ({scan_mode})"
    if tags is not None:
        title += f" — tags={tags}"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)

    # Log-X Puffer, damit Linien am Rand sichtbar sind
    if len(s.index):
        x_min = float(np.nanmin(s.index.values))
        x_max = float(np.nanmax(s.index.values))
        if math.isfinite(x_min) and math.isfinite(x_max) and x_min > 0 and x_max > x_min:
            pad_log10 = 0.02
            ax.set_xlim(x_min / (10**pad_log10), x_max * (10**pad_log10))

    # Punkt-Annotation (letzter Punkt)
    if len(s.index) >= 1:
        ax.annotate(f"{s.values[-1]:.2f}× @ {int(s.index.values[-1]):,}",
                    xy=(s.index.values[-1], s.values[-1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)

    # ---- BE Overlay (nur für das gewählte baseline-Profil) -------------------
    metric_label = ("p95" if metric == "p95_ms" else "p50" if metric == "p50_ms" else metric)

    # Primary-Wert für das Profil herausziehen
    px = _primary_for_baseline(primary_be_map, query=query, warm=warm, tags=tags, baseline=baseline)
    primary_vals_for_box = {"auto": None, "t1": None}
    primary_vals_for_box[baseline] = px

    # Multi-Linien für dieses Profil zeichnen, Primary erzwingen
    selected_multi = {"auto": [], "t1": []}
    if be_multi_map:
        sel = annotate_be_multi(
            ax, be_multi_map,
            query=query, warm=warm, tags=tags, metric=metric,
            draw_profiles=(baseline,), sqlite_variant="best",
            top_k=multi_top_k,
            min_rel_delta=multi_min_rel,
            min_abs_delta_ms=multi_min_abs_ms,
            dedupe_primary_x={baseline: px},
            dedupe_tol_log10=multi_dedupe_tol_log10,
            colors=BE_COLORS,
            ensure_include_primary=True,
            include_primary_x={baseline: px},
        )
        selected_multi[baseline] = sel.get(baseline, [])

    # Box (dedupliziert Primary aus Multi-Liste)
    be_loc = "lower left"
    be_anchor = (1.02, 0.0) if be_box_outside else None
    render_be_combined_legend(
        ax,
        metric_label=metric_label,
        tags=tags,
        primary_map=primary_vals_for_box,
        multi_map=selected_multi,
        outside=be_box_outside,
        loc=be_loc,
        bbox_to_anchor=be_anchor,
        colors=BE_COLORS,
        zorder=100.0,
    )

    # Keine Serien-Legende nötig (nur eine Linie)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    return fname

# ---------- one panel helper for joined --------------------------------------
def _draw_speedup_panel(
    ax,
    dft: pd.DataFrame,
    *,
    be_multi_map: dict,
    primary_be_map: dict,
    query: str,
    warm: bool,
    metric: str,
    tags: int,
    scan_mode: str,
    baseline: str,
    stat: str,
    multi_top_k: int,
    multi_min_rel: float,
    multi_min_abs_ms: float,
    multi_dedupe_tol_log10: float,
):
    s = speedup_series(
        dft, query=query, warm=warm, metric=metric,
        baseline=baseline, scan_mode=scan_mode, stat=stat
    )
    if s.empty:
        return

    ax.plot(s.index.values, s.values, marker="o", linewidth=2.0)
    ax.axhline(1.0, ls=":", lw=1.0, color="0.5", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Total rows (log scale)")
    ax.set_ylabel(f"Speedup (SQLite_best ÷ DuckDB({baseline})) — {metric}")
    ax.set_title(f"tags={tags}")
    ax.grid(True, which="both", alpha=0.35)

    # X-Padding
    if len(s.index):
        x_min = float(np.nanmin(s.index.values))
        x_max = float(np.nanmax(s.index.values))
        if math.isfinite(x_min) and math.isfinite(x_max) and x_min > 0 and x_max > x_min:
            pad_log10 = 0.02
            ax.set_xlim(x_min / (10**pad_log10), x_max * (10**pad_log10))

    # BE Overlay nur für dieses baseline-Profil
    px = _primary_for_baseline(primary_be_map, query=query, warm=warm, tags=tags, baseline=baseline)
    sel = annotate_be_multi(
        ax, be_multi_map,
        query=query, warm=warm, tags=tags, metric=metric,
        draw_profiles=(baseline,), sqlite_variant="best",
        top_k=multi_top_k,
        min_rel_delta=multi_min_rel,
        min_abs_delta_ms=multi_min_abs_ms,
        dedupe_primary_x={baseline: px},
        dedupe_tol_log10=multi_dedupe_tol_log10,
        colors=BE_COLORS,
        ensure_include_primary=True,
        include_primary_x={baseline: px},
    )
    # Box innen (nur baseline-Spalte befüllen)
    metric_label = ("p95" if metric == "p95_ms" else "p50" if metric == "p50_ms" else metric)
    render_be_combined_legend(
        ax,
        metric_label=metric_label,
        tags=tags,
        primary_map={"auto": px if baseline == "auto" else None, "t1": px if baseline == "t1" else None},
        multi_map={"auto": sel.get("auto", []) if baseline == "auto" else [],
                   "t1":   sel.get("t1",   []) if baseline == "t1"   else []},
        outside=False,
        loc="upper left",
        bbox_to_anchor=None,
        colors=BE_COLORS,
        zorder=100.0,
    )

# ---------- joined (tags A|B) ------------------------------------------------
def plot_speedup_joined_pair(
    df: pd.DataFrame,
    *,
    be_multi_map: dict,
    primary_be_map: dict,
    outdir: Path,
    query: str,
    warm: bool,
    metric: str,
    scan_mode: str,
    tags_pair: tuple[int, int],
    baseline: str,
    stat: str,
    multi_top_k: int,
    multi_min_rel: float,
    multi_min_abs_ms: float,
    multi_dedupe_tol_log10: float,
    fig_height: float,
):
    a, b = tags_pair
    subdir = ensure_dir(outdir / "joined" / metric / ("warm" if warm else "cold") / baseline)
    fname  = subdir / f"{query}__speedup_joined_tags{a}_{b}__duck_{baseline}__{metric}__{scan_mode}.png"

    fig, axes = plt.subplots(1, 2, figsize=(10.0, fig_height), constrained_layout=False)
    plt.subplots_adjust(right=0.96, wspace=0.25)

    # Panel A
    dfa = df[df["dataset_tags"] == a] if "dataset_tags" in df.columns else df
    _draw_speedup_panel(
        axes[0], dfa,
        be_multi_map=be_multi_map,
        primary_be_map=primary_be_map,
        query=query, warm=warm, metric=metric, tags=a, scan_mode=scan_mode,
        baseline=baseline, stat=stat,
        multi_top_k=multi_top_k, multi_min_rel=multi_min_rel,
        multi_min_abs_ms=multi_min_abs_ms, multi_dedupe_tol_log10=multi_dedupe_tol_log10,
    )

    # Panel B
    dfb = df[df["dataset_tags"] == b] if "dataset_tags" in df.columns else df
    _draw_speedup_panel(
        axes[1], dfb,
        be_multi_map=be_multi_map,
        primary_be_map=primary_be_map,
        query=query, warm=warm, metric=metric, tags=b, scan_mode=scan_mode,
        baseline=baseline, stat=stat,
        multi_top_k=multi_top_k, multi_min_rel=multi_min_rel,
        multi_min_abs_ms=multi_min_abs_ms, multi_dedupe_tol_log10=multi_dedupe_tol_log10,
    )

    fig.suptitle(f"{query} — {'warm' if warm else 'cold'} ({scan_mode}) — DuckDB({baseline})", y=0.995)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    return fname

# ---------- CLI ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(RESULTS_DIR / "summary_with_cis.csv"))
    ap.add_argument("--break-even", default=str(RESULTS_DIR / "break_even.csv"))
    ap.add_argument("--break-even-multi", default=str(RESULTS_DIR / "break_even_multi.csv"))
    ap.add_argument("--outdir", default=str(RESULTS_DIR / "plots" / "main" / "speedup"))
    ap.add_argument("--metrics", default="p50_ms,p95_ms")
    ap.add_argument("--tags", type=int, default=None)
    ap.add_argument("--scan-mode", default="table", choices=["table", "file", "any"])
    ap.add_argument("--baselines", default="auto,t1")  # DuckDB profiles
    ap.add_argument("--speedup-stat", default="min", choices=["min", "median"])

    # BE source & multi parameters (identisch zu main)
    ap.add_argument("--be-source", default="multi", choices=["primary", "multi"],
                    help="Woher Primary-BE kommt: separate CSV (primary) oder aus break_even_multi abgeleitet (multi).")
    ap.add_argument("--multi-top-k", type=int, default=2)
    ap.add_argument("--multi-min-rel", type=float, default=0.10)
    ap.add_argument("--multi-min-abs-ms", type=float, default=0.0)
    ap.add_argument("--multi-dedupe-tol-log10", type=float, default=0.01)

    # Layout
    ap.add_argument("--legend-outside", action="store_true")
    ap.add_argument("--be-box-outside", action="store_true")
    ap.add_argument("--fig-height", type=float, default=4.8)
    ap.add_argument("--ylog", action="store_true")

    # Joined-Variante
    ap.add_argument("--make-joined", action="store_true")
    ap.add_argument("--joined-tags", default="20,50")
    ap.add_argument("--joined-outdir", default=str(RESULTS_DIR / "plots" / "main" / "speedup"))
    ap.add_argument("--joined-fig-height", type=float, default=4.8)

    args = ap.parse_args()
    apply_style()

    df_all = load_summary(Path(args.summary))
    df = df_all.copy()

    if args.tags is not None and "dataset_tags" in df.columns:
        df = df[df["dataset_tags"] == args.tags].copy()

    if args.scan_mode != "any":
        df = df[df["scan_mode"] == args.scan_mode].copy()
        df_all = df_all[df_all["scan_mode"] == args.scan_mode].copy()

    queries = sorted(df[(df["scan_mode"] == (args.scan_mode if args.scan_mode != "any" else "table"))]["query"].dropna().unique())
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    # BE maps laden
    be_multi_map = load_be_multi(Path(args.break_even_multi)) if Path(args.break_even_multi).is_file() else {}

    # Single-Tag Autodetect (nur für Single-Plots)
    inferred_tag = None
    if args.tags is None and "dataset_tags" in df.columns:
        uniq = sorted(df["dataset_tags"].dropna().unique())
        if len(uniq) == 1:
            inferred_tag = int(uniq[0])

    written = []
    for metric in metrics:
        # Primary-Map bestimmen
        if args.be_source == "multi":
            primary_be_map_single = derive_primary_map_from_multi(
                be_multi_map,
                preferred_metric=metric,
                tags=(args.tags if args.tags is not None else inferred_tag),
                min_rel_delta=args.multi_min_rel,
                min_abs_delta_ms=args.multi_min_abs_ms,
            )
        else:
            primary_be_map_single = load_be_map(Path(args.break_even), preferred_metric=metric) \
                if Path(args.break_even).is_file() else {}

        # Einzelplots
        for warm in (False, True):
            for baseline in baselines:
                for q in queries:
                    f = plot_speedup(
                        df,
                        be_multi_map=be_multi_map,
                        primary_be_map=primary_be_map_single,
                        outdir=Path(args.outdir),
                        query=q, warm=warm, metric=metric,
                        tags=(args.tags if args.tags is not None else inferred_tag),
                        scan_mode=(args.scan_mode if args.scan_mode != "any" else "table"),
                        baseline=baseline,
                        stat=args.speedup_stat,
                        legend_outside=args.legend_outside,
                        be_box_outside=args.be_box_outside,
                        fig_height=args.fig_height,
                        multi_top_k=args.multi_top_k,
                        multi_min_rel=args.multi_min_rel,
                        multi_min_abs_ms=args.multi_min_abs_ms,
                        multi_dedupe_tol_log10=args.multi_dedupe_tol_log10,
                        ylog=args.ylog,
                    )
                    if f:
                        written.append(f)

        # Joined-Variante
        if args.make_joined:
            parts = [p.strip() for p in args.joined_tags.split(",") if p.strip()]
            if len(parts) != 2:
                raise SystemExit(f"--joined-tags erwartet genau zwei Werte, z.B. 20,50 (bekommen: {args.joined_tags})")
            tA, tB = int(parts[0]), int(parts[1])

            # Primary-Map für beide Tags (union)
            if args.be_source == "multi":
                primary_be_map_joined = {}
                for tg in (tA, tB):
                    pm = derive_primary_map_from_multi(
                        be_multi_map,
                        preferred_metric=metric,
                        tags=tg,
                        min_rel_delta=args.multi_min_rel,
                        min_abs_delta_ms=args.multi_min_abs_ms,
                    )
                    primary_be_map_joined.update(pm)
            else:
                primary_be_map_joined = load_be_map(Path(args.break_even), preferred_metric=metric) \
                    if Path(args.break_even).is_file() else {}

            for warm in (False, True):
                for baseline in baselines:
                    for q in queries:
                        jf = plot_speedup_joined_pair(
                            df_all,
                            be_multi_map=be_multi_map,
                            primary_be_map=primary_be_map_joined,
                            outdir=Path(args.joined_outdir),
                            query=q, warm=warm, metric=metric,
                            scan_mode=(args.scan_mode if args.scan_mode != "any" else "table"),
                            tags_pair=(tA, tB),
                            baseline=baseline,
                            stat=args.speedup_stat,
                            multi_top_k=args.multi_top_k,
                            multi_min_rel=args.multi_min_rel,
                            multi_min_abs_ms=args.multi_min_abs_ms,
                            multi_dedupe_tol_log10=args.multi_dedupe_tol_log10,
                            fig_height=args.joined_fig_height,
                        )
                        if jf:
                            written.append(jf)

    print(f"[ok] wrote {len(written)} speedup plots → {args.outdir}")

if __name__ == "__main__":
    main()
