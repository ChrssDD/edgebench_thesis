# analysis/images/plots_v2_main.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from config import RESULTS_DIR
from .style import apply_style, OKABE_ITO
from .utils import (
    ensure_dir, load_summary, median_curve, ci_band, ORDER, MARKERS,
    load_be_map, annotate_be_lines,                # (import kept)
    load_be_multi, annotate_be_multi,              # multi (draw + selection)
    derive_primary_map_from_multi,                 # primary-from-multi
    render_be_combined_legend, BE_COLORS,          # consolidated box
)

# ---------------------------------------------------------------------------


PSI_COLORS = {
    ("sqlite", "noidx"): OKABE_ITO[0],  # black
    ("sqlite", "idx"):   OKABE_ITO[1],  # orange
    ("duckdb", "t1"):    OKABE_ITO[2],  # blue
    ("duckdb", "auto"):  OKABE_ITO[3],  # green
    ("sqlite", "any"):   "#888888",     # gray when aggregating both variants
}


def be_metrics_in_csv(be_csv: Path) -> set[str]:
    if not be_csv.is_file():
        return set()
    be = pd.read_csv(be_csv)
    if "metric" not in be.columns:
        return {"p95_ms"}
    return set(be["metric"].dropna().astype(str))


def _extract_primary_vals_from_map(primary_map: dict, *, query: str, warm: bool, tags: int | None) -> dict:
    """Return {'auto': x_or_None, 't1': x_or_None} from a be_map-like dict."""
    def _get(prof):
        if tags is None:
            return primary_map.get((query, warm, "best", prof), None)
        v = primary_map.get((query, warm, "best", prof, tags), None)
        return v if v is not None else primary_map.get((query, warm, "best", prof), None)
    return {"auto": _get("auto"), "t1": _get("t1")}


def _psi_col(kind: str, stat: str) -> str:
    if kind == "io":
        return "p95_psi_io_full_ms" if stat == "p95" else "mean_psi_io_full_ms"
    else:
        return "p95_psi_cpu_some_ms" if stat == "p95" else "mean_psi_cpu_some_ms"


def psi_curve(df: pd.DataFrame, query: str, warm: bool, scan_mode: str,
              *, engine: str = "duckdb", profile: str = "auto",
              sqlite_variant: str = "any", kind: str, stat: str) -> pd.Series:
    """
    Median PSI per dataset_rows, filtered by engine/profile/variant.
    engine: 'duckdb' | 'sqlite'
    profile: 'auto' | 't1'   (duckdb only)
    sqlite_variant: 'any' | 'idx' | 'noidx'   (sqlite only)
    kind: 'io' | 'cpu'; stat: 'p95' | 'mean'
    """
    col = _psi_col(kind, stat)
    d = df[(df["query"] == query) &
           (df["warm"] == warm) &
           (df["scan_mode"] == scan_mode) &
           (df["engine"] == engine)].copy()
    if d.empty or col not in d.columns:
        return pd.Series(dtype=float)

    if engine == "duckdb":
        if profile == "t1":
            d = d[d["threads"] == 1]
        else:
            d = d[d["threads"].isna()]
    else:  # sqlite
        if sqlite_variant in ("idx", "noidx"):
            d = d[d["sqlite_variant"] == sqlite_variant]  # 'any' keeps both

    if d.empty:
        return pd.Series(dtype=float)

    s = d.groupby("dataset_rows")[col].median().sort_index()
    s.name = f"{engine.upper()} PSI({kind}-{stat})"
    return s


# --- helpers ---------------------------------------------------------------

def _smooth_series(s: pd.Series, window: int = 3) -> pd.Series:
    """Centered rolling mean to reduce jagged PSI without SciPy."""
    if s is None or s.empty or window <= 1:
        return s
    return s.rolling(window=window, min_periods=1, center=True).mean()

def _draw_psi_panel(
    ax,
    df: pd.DataFrame,
    *,
    query: str,
    warm: bool,
    scan_mode: str,
    tags: int,
    kinds: list[str],
    engine_mode: str,             # 'duckdb' | 'sqlite' | 'both'
    profile: str,                 # duckdb: kept for signature
    sqlite_variant: str,          # sqlite: kept for signature
    stat: str,
    smooth_win: int = 3,
    show_legend: bool = False,    # NEW: default False → no legend in PSI panels
):
    """PSI-only panel for one tag (CPU or I/O per call)."""
    psi_opts = getattr(_draw_psi_panel, "_psi_opts", {}) or {}

    dft = df
    if "dataset_tags" in df.columns:
        dft = df[df["dataset_tags"] == tags].copy()

    engines = ["duckdb", "sqlite"] if engine_mode == "both" else [engine_mode]
    labels_drawn = False

    for eng in engines:
        if eng == "duckdb":
            for prof in psi_opts.get("duck_profiles", ["auto"]):
                for k in kinds:
                    s = psi_curve(
                        dft, query, warm, scan_mode,
                        engine="duckdb", profile=prof, sqlite_variant="any",
                        kind=k, stat=stat
                    )
                    if s.empty:
                        continue
                    s = _smooth_series(s, window=smooth_win)
                    color = PSI_COLORS.get(("duckdb", prof))
                    lab = f"DuckDB PSI {'I/O' if k=='io' else 'CPU'} {stat} ({prof})"
                    ax.plot(
                        s.index.values, s.values,
                        linestyle="--", marker="x", linewidth=1.6,
                        label=lab, color=color
                    )
                    labels_drawn = True
        else:  # sqlite
            for var in psi_opts.get("sqlite_variants", ["any"]):
                for k in kinds:
                    s = psi_curve(
                        dft, query, warm, scan_mode,
                        engine="sqlite", profile="auto", sqlite_variant=var,
                        kind=k, stat=stat
                    )
                    if s.empty:
                        continue
                    s = _smooth_series(s, window=smooth_win)
                    color = PSI_COLORS.get(("sqlite", var), PSI_COLORS[("sqlite", "any")])
                    suffix = "" if var == "any" else f" ({var})"
                    lab = f"SQLite PSI {'I/O' if k=='io' else 'CPU'} {stat}{suffix}"
                    ax.plot(
                        s.index.values, s.values,
                        linestyle="--", marker="x", linewidth=1.6,
                        label=lab, color=color
                    )
                    labels_drawn = True

    ax.set_xscale("log")
    ax.set_xlabel("Total rows (log scale)")
    ax.set_ylabel("PSI Δ (ms)")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_title(f"PSI — tags={tags}")

    if tags:
        for _, val in (("15", 900), ("60", 3600)):
            ax.axvline(val * tags, color="0.82", ls="--", lw=1)

    # No legend by default (single bottom legend from latency is enough).
    if show_legend and labels_drawn:
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, loc="upper left", frameon=True, fontsize=8)


def plot_joined_tags_split(
    df: pd.DataFrame,
    primary_be_map: dict,
    be_multi_map: dict | None,
    outdir: Path,
    *,
    query: str,
    warm: bool,
    metric: str,
    scan_mode: str,
    tags_pair: tuple[int, int] = (20, 50),
    be_mode: str = "multi",
    multi_top_k: int = 2,
    multi_min_rel: float = 0.10,
    multi_min_abs_ms: float = 0.0,
    multi_dedupe_tol_log10: float = 0.01,
    fig_height: float = 4.8,
):
    """
    Writes three separate 1×2 figures:
      1) Latency top row → '__latency.png'
      2) PSI CPU         → '__psi_cpu.png'
      3) PSI I/O         → '__psi_io.png'
    Returns list of filenames.
    """
    a, b = tags_pair
    subdir = ensure_dir(outdir / "joined" / metric / ("warm" if warm else "cold"))

    # Shared PSI opts
    psi_opts = getattr(_draw_psi_panel, "_psi_opts", {}) or {}
    stat     = psi_opts.get("stat", "p95")
    smooth   = psi_opts.get("smooth", 3)
    eng_mode = psi_opts.get("engine", "duckdb")

    written = []

    # ---- (1) LATENCY FIG ---------------------------------------------------
    lat_name = subdir / f"{query}__{metric}__joined_tags{a}_{b}__{scan_mode}__latency.png"
    figL, axesL = plt.subplots(
        1, 2, figsize=(10.0, max(3.8, fig_height)),
        constrained_layout=False, sharex="col"
    )
    plt.subplots_adjust(top=0.84, bottom=0.12, right=0.98, wspace=0.28)

    _draw_panel_for_tag(
        axesL[0], df,
        primary_be_map=primary_be_map, be_multi_map=be_multi_map, be_mode=be_mode,
        query=query, warm=warm, metric=metric, scan_mode=scan_mode, tags=a,
        multi_top_k=multi_top_k, multi_min_rel=multi_min_rel,
        multi_min_abs_ms=multi_min_abs_ms, multi_dedupe_tol_log10=multi_dedupe_tol_log10,
        be_loc="upper left", show_ylabel=True, panel_title=f"tags={a}",
    )
    _draw_panel_for_tag(
        axesL[1], df,
        primary_be_map=primary_be_map, be_multi_map=be_multi_map, be_mode=be_mode,
        query=query, warm=warm, metric=metric, scan_mode=scan_mode, tags=b,
        multi_top_k=multi_top_k, multi_min_rel=multi_min_rel,
        multi_min_abs_ms=multi_min_abs_ms, multi_dedupe_tol_log10=multi_dedupe_tol_log10,
        be_loc="upper left", show_ylabel=False, panel_title=f"tags={b}",
    )

    # Wide legend for latency only
    handles, labels = axesL[0].get_legend_handles_labels()
    if handles:
        figL.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.95),
            ncol=max(2, min(len(labels), 4)),
            frameon=True, fontsize=9, borderaxespad=0.0,
        )
    figL.suptitle(f"{query} — {'warm' if warm else 'cold'} ({scan_mode})", y=0.995)
    figL.savefig(lat_name, dpi=300)
    plt.close(figL)
    written.append(lat_name)

    # ---- (2) PSI CPU FIG ---------------------------------------------------
    cpu_name = subdir / f"{query}__{metric}__joined_tags{a}_{b}__{scan_mode}__psi_cpu.png"
    figC, axesC = plt.subplots(
        1, 2, figsize=(10.0, max(3.8, fig_height)),
        constrained_layout=False, sharex="col"
    )
    plt.subplots_adjust(bottom=0.18, right=0.98, wspace=0.28)

    _draw_psi_panel(
        axesC[0], df,
        query=query, warm=warm, scan_mode=scan_mode,
        tags=a, kinds=["cpu"], engine_mode=eng_mode,
        profile="auto", sqlite_variant="any",
        stat=stat, smooth_win=smooth, show_legend=False
    )
    _draw_psi_panel(
        axesC[1], df,
        query=query, warm=warm, scan_mode=scan_mode,
        tags=b, kinds=["cpu"], engine_mode=eng_mode,
        profile="auto", sqlite_variant="any",
        stat=stat, smooth_win=smooth, show_legend=False
    )
    axesC[0].set_title(f"PSI CPU — tags={a}")
    axesC[1].set_title(f"PSI CPU — tags={b}")
    figC.suptitle(f"{query} — {'warm' if warm else 'cold'} ({scan_mode})", y=0.995)
    figC.savefig(cpu_name, dpi=300)
    plt.close(figC)
    written.append(cpu_name)

    # ---- (3) PSI I/O FIG ---------------------------------------------------
    io_name = subdir / f"{query}__{metric}__joined_tags{a}_{b}__{scan_mode}__psi_io.png"
    figI, axesI = plt.subplots(
        1, 2, figsize=(10.0, max(3.8, fig_height)),
        constrained_layout=False, sharex="col"
    )
    plt.subplots_adjust(bottom=0.18, right=0.98, wspace=0.28)

    _draw_psi_panel(
        axesI[0], df,
        query=query, warm=warm, scan_mode=scan_mode,
        tags=a, kinds=["io"], engine_mode=eng_mode,
        profile="auto", sqlite_variant="any",
        stat=stat, smooth_win=smooth, show_legend=False
    )
    _draw_psi_panel(
        axesI[1], df,
        query=query, warm=warm, scan_mode=scan_mode,
        tags=b, kinds=["io"], engine_mode=eng_mode,
        profile="auto", sqlite_variant="any",
        stat=stat, smooth_win=smooth, show_legend=False
    )
    axesI[0].set_title(f"PSI I/O — tags={a}")
    axesI[1].set_title(f"PSI I/O — tags={b}")
    figI.suptitle(f"{query} — {'warm' if warm else 'cold'} ({scan_mode})", y=0.995)
    figI.savefig(io_name, dpi=300)
    plt.close(figI)
    written.append(io_name)

    return written



def _shade_be_transition_zones(ax, selected_multi: dict, *, palette: dict, width_log10: float = 0.012, max_per_profile: int = 3):
    """
    Thin translucent bands around each selected BE x (±width in log10 space).
    At most max_per_profile bands per profile to avoid clutter.
    """
    def _xs(items):
        for it in items or []:
            if isinstance(it, (int, float)):
                yield float(it)
            elif isinstance(it, dict) and "x" in it:
                yield float(it["x"])
            elif isinstance(it, (list, tuple)) and it:
                yield float(it[0])

    for prof, items in (selected_multi or {}).items():
        color = palette.get(prof, "#999")
        for i, x in enumerate(_xs(items)):
            if i >= max_per_profile:
                break
            left  = x / (10 ** width_log10)
            right = x * (10 ** width_log10)
            ax.axvspan(left, right, color=color, alpha=0.08, lw=0, zorder=0.5)

def _draw_be_pins(ax, selected_multi: dict, *, palette: dict, max_per_profile: int = 3, style: str = ":"):
    """Draw short vertical pins near the top edge at each BE x."""
    def _xs(items):
        for it in items or []:
            if isinstance(it, (int, float)):
                yield float(it)
            elif isinstance(it, dict) and "x" in it:
                yield float(it["x"])
            elif isinstance(it, (list, tuple)) and it:
                yield float(it[0])

    ymin, ymax = ax.get_ylim()
    y0 = ymin + 0.92 * (ymax - ymin)   # start of pins
    y1 = ymin + 0.995 * (ymax - ymin)  # end of pins (just under top)

    for prof, items in (selected_multi or {}).items():
        color = palette.get(prof, "#999")
        for i, x in enumerate(_xs(items)):
            if i >= max_per_profile:
                break
            ax.vlines(x, y0, y1, colors=color, linestyles=style, linewidth=1.8,
                      zorder=6.0, clip_on=False)


# ---------------------------------------------------------------------------

def plot_one(
    df: pd.DataFrame,
    primary_be_map: dict,
    outdir: Path,
    query: str,
    warm: bool,
    metric: str,
    tags: int | None,
    scan_mode: str = "table",
    *,
    be_multi_map: dict | None = None,
    be_mode: str = "multi",
    multi_top_k: int = 2,
    multi_min_rel: float = 0.10,
    multi_min_abs_ms: float = 0.0,
    multi_dedupe_tol_log10: float = 0.01,
    legend_outside: bool = False,
    be_box_outside: bool = False,
    fig_height: float = 4.8,
):
    cur = median_curve(df, query, warm, metric, scan_mode)
    if cur.empty:
        return None
    cib = ci_band(df, query, warm, metric, scan_mode)

    subdir = ensure_dir(outdir / metric / ("warm" if warm else "cold"))
    fname  = subdir / f"{query}__{metric}__vs_rows__{scan_mode}.png"

    fig, ax = plt.subplots(
        figsize=(6.8 if (legend_outside or be_box_outside) else 5.6, fig_height),
        constrained_layout=False
    )
    pv = cur.pivot(index="dataset_rows", columns="label", values=metric).sort_index()

    # series
    for lab in [c for c in ORDER if c in pv.columns]:
        x = pv.index.values
        y = pv[lab].values
        ls = "-" if "DuckDB" in lab else ":"
        ms = MARKERS.get(lab, "o")
        lw = 2.0
        if lab == "SQLite (noidx)":
            ax.plot(x, y, label=lab, linewidth=lw, marker=ms, ms=4, linestyle=ls,
                    markerfacecolor="none", markeredgewidth=1.6)
        else:
            ax.plot(x, y, label=lab, linewidth=lw, marker=ms, ms=4, linestyle=ls)

    # CI band
    if not cib.empty:
        low_col, high_col = f"ci_low_{metric}", f"ci_high_{metric}"
        for lab in [c for c in ORDER if c in pv.columns]:
            ci_lab = cib[cib["label"] == lab].sort_values("dataset_rows")
            if not ci_lab.empty and ci_lab[low_col].notna().any():
                ax.fill_between(ci_lab["dataset_rows"].values,
                                ci_lab[low_col].values, ci_lab[high_col].values,
                                alpha=0.18, linewidth=0)

    # axes + log padding
    ax.set_xscale("log")
    ax.set_xlabel("Total rows (log scale)")
    ax.set_ylabel(f"{metric} (ms)")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_title(f"{query} — {'warm' if warm else 'cold'} ({scan_mode})")

    if len(pv.index):
        x_min = float(np.nanmin(pv.index.values))
        x_max = float(np.nanmax(pv.index.values))
        if math.isfinite(x_min) and math.isfinite(x_max) and x_min > 0 and x_max > x_min:
            pad_log10 = 0.02
            ax.set_xlim(x_min / (10 ** pad_log10), x_max * (10 ** pad_log10))

    metric_label = ("p95" if metric == "p95_ms" else "p50" if metric == "p50_ms" else metric)

    primary_vals = {"auto": None, "t1": None}
    selected_multi_raw = {"auto": [], "t1": []}

    # PSI overlay (unchanged)
    if getattr(plot_one, "_psi_opts", None):
        po = plot_one._psi_opts
        kinds = po.get("kinds", [])
        stat  = po.get("stat", "p95")
        duck_profiles   = po.get("duck_profiles", ["auto"])
        sqlite_variants = po.get("sqlite_variants", ["any"])
        eng_mode = po.get("engine", "duckdb")
        if kinds:
            ax2 = ax.twinx()
            labels = []
            engines = ["duckdb", "sqlite"] if eng_mode == "both" else [eng_mode]
            for eng in engines:
                if eng == "duckdb":
                    for prof in duck_profiles:
                        for k in kinds:
                            s = psi_curve(df, query, warm, scan_mode,
                                          engine="duckdb", profile=prof,
                                          sqlite_variant="any", kind=k, stat=stat)
                            if s.empty:
                                continue
                            lab = f"DuckDB PSI {('I/O' if k=='io' else 'CPU')} {stat} ({prof})"
                            color = PSI_COLORS.get(("duckdb", prof))
                            ax2.plot(s.index.values, s.values, linestyle="--", marker="x",
                                     linewidth=1.6, label=lab, color=color)
                            labels.append(lab)
                else:
                    for var in sqlite_variants:
                        for k in kinds:
                            s = psi_curve(df, query, warm, scan_mode,
                                          engine="sqlite", profile="auto",
                                          sqlite_variant=var, kind=k, stat=stat)
                            if s.empty:
                                continue
                            suffix = "" if var == "any" else f" ({var})"
                            lab = f"SQLite PSI {('I/O' if k=='io' else 'CPU')} {stat}{suffix}"
                            color = PSI_COLORS.get(("sqlite", var), PSI_COLORS[("sqlite", "any")])
                            ax2.plot(s.index.values, s.values, linestyle="--", marker="x",
                                     linewidth=1.6, label=lab, color=color)
                            labels.append(lab)
            if labels:
                ax2.set_ylabel("PSI Δ (ms)")
                h2, l2 = ax2.get_legend_handles_labels()
                ax2.legend(h2, l2, loc="lower right", frameon=True, fontsize=8)

    # ---- BE lines: multi first, then primary overlay (thicker) ---------------
    if be_multi_map and be_mode in ("multi", "both"):
        primary_vals = _extract_primary_vals_from_map(primary_be_map, query=query, warm=warm, tags=tags)
        selected_multi_raw = annotate_be_multi(
            ax, be_multi_map,
            query=query, warm=warm, tags=tags, metric=metric,
            draw_profiles=("auto", "t1"), sqlite_variant="best",
            top_k=multi_top_k,
            min_rel_delta=multi_min_rel,
            min_abs_delta_ms=multi_min_abs_ms,
            dedupe_primary_x=primary_vals,
            dedupe_tol_log10=multi_dedupe_tol_log10,
            colors=BE_COLORS,
            ensure_include_primary=False,      # ← multi only
            include_primary_x=primary_vals,
        )

    # Always overlay primaries so they can't be swallowed by CI/grid
    annotate_be_lines(
        ax, primary_be_map,
        query=query, warm=warm, tags=tags,
        draw_profiles=("auto", "t1"), sqlite_variant="best",
        draw_vertical=True, line_alpha=1.0, line_width=2.2, colors=BE_COLORS
    )

    if tags:
        for _, val in (("15", 900), ("60", 3600)):
            ax.axvline(val * tags, color="0.82", ls="--", lw=1)

    if legend_outside:
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=9, borderaxespad=0.0)
    else:
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    # Pins (prefer real multi; else fall back to primaries)
    pins_map = selected_multi_raw if any((selected_multi_raw or {}).values()) \
               else {k: ([v] if v is not None else []) for k, v in (primary_vals or {}).items()}
    _draw_be_pins(ax, pins_map, palette=BE_COLORS, max_per_profile=2)

    # BE box: show all multi points
    be_loc = "lower left"
    be_anchor = (1.02, 0.0) if be_box_outside else None
    primary_vals_compact = {k: v for k, v in (primary_vals or {}).items() if v is not None}
    selected_multi_compact = {k: list(vals) for k, vals in (selected_multi_raw or {}).items() if vals}

    render_be_combined_legend(
        ax,
        metric_label=metric_label,
        tags=tags,
        primary_map=primary_vals_compact,
        multi_map=(selected_multi_compact if selected_multi_compact else None),
        outside=be_box_outside,
        loc=be_loc,
        bbox_to_anchor=be_anchor,
        colors=BE_COLORS,
        zorder=100.0,
    )

    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    return fname




def _draw_panel_for_tag(
    ax,
    df: pd.DataFrame,
    *,
    primary_be_map: dict,
    be_multi_map: dict | None,
    be_mode: str,
    query: str,
    warm: bool,
    metric: str,
    scan_mode: str,
    tags: int,
    multi_top_k: int,
    multi_min_rel: float,
    multi_min_abs_ms: float,
    multi_dedupe_tol_log10: float,
    be_loc: str,
    show_ylabel: bool,
    panel_title: str,
):
    """Latency panel for one tag."""
    dft = df
    if "dataset_tags" in df.columns:
        dft = df[df["dataset_tags"] == tags].copy()

    cur = median_curve(dft, query, warm, metric, scan_mode)
    cib = ci_band(dft, query, warm, metric, scan_mode)

    pv = cur.pivot(index="dataset_rows", columns="label", values=metric).sort_index()
    for lab in [c for c in ORDER if c in pv.columns]:
        x = pv.index.values
        y = pv[lab].values
        ls = "-" if "DuckDB" in lab else ":"
        ms = MARKERS.get(lab, "o")
        lw = 2.0
        if lab == "SQLite (noidx)":
            ax.plot(x, y, label=lab, linewidth=lw, marker=ms, ms=4, linestyle=ls,
                    markerfacecolor="none", markeredgewidth=1.6)
        else:
            ax.plot(x, y, label=lab, linewidth=lw, marker=ms, ms=4, linestyle=ls)

    if not cib.empty:
        low_col, high_col = f"ci_low_{metric}", f"ci_high_{metric}"
        for lab in [c for c in ORDER if c in pv.columns]:
            ci_lab = cib[cib["label"] == lab].sort_values("dataset_rows")
            if not ci_lab.empty and ci_lab[low_col].notna().any():
                ax.fill_between(ci_lab["dataset_rows"].values,
                                ci_lab[low_col].values, ci_lab[high_col].values,
                                alpha=0.18, linewidth=0)

    ax.set_xscale("log")
    if show_ylabel:
        ax.set_ylabel(f"{metric} (ms)")
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Total rows (log scale)")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_title(panel_title)

    if len(pv.index):
        x_min = float(np.nanmin(pv.index.values))
        x_max = float(np.nanmax(pv.index.values))
        if math.isfinite(x_min) and math.isfinite(x_max) and x_min > 0 and x_max > x_min:
            pad_log10 = 0.02
            ax.set_xlim(x_min / (10 ** pad_log10), x_max * (10 ** pad_log10))

    if tags:
        for _, val in (("15", 900), ("60", 3600)):
            ax.axvline(val * tags, color="0.82", ls="--", lw=1)

    # ---- BEs: multi first, primary overlay ----------------------------------
    primary_vals = {"auto": None, "t1": None}
    selected_multi_raw = {"auto": [], "t1": []}

    if be_multi_map and be_mode in ("multi", "both"):
        primary_vals = {
            "auto": primary_be_map.get((query, warm, "best", "auto", tags), primary_be_map.get((query, warm, "best", "auto"))),
            "t1":   primary_be_map.get((query, warm, "best", "t1",   tags), primary_be_map.get((query, warm, "best", "t1"))),
        }
        selected_multi_raw = annotate_be_multi(
            ax, be_multi_map,
            query=query, warm=warm, tags=tags, metric=metric,
            draw_profiles=("auto", "t1"), sqlite_variant="best",
            top_k=multi_top_k,
            min_rel_delta=multi_min_rel,
            min_abs_delta_ms=multi_min_abs_ms,
            dedupe_primary_x=primary_vals,
            dedupe_tol_log10=multi_dedupe_tol_log10,
            colors=BE_COLORS,
            ensure_include_primary=False,   # ← multi only
            include_primary_x=primary_vals,
        )

    # Primary overlay (thicker)
    annotate_be_lines(
        ax, primary_be_map,
        query=query, warm=warm, tags=tags,
        draw_profiles=("auto", "t1"), sqlite_variant="best",
        draw_vertical=True, line_alpha=1.0, line_width=2.2, colors=BE_COLORS
    )

    # Pins
    pins_map = selected_multi_raw if any((selected_multi_raw or {}).values()) \
               else {k: ([v] if v is not None else []) for k, v in (primary_vals or {}).items()}
    _draw_be_pins(ax, pins_map, palette=BE_COLORS, max_per_profile=2)

    # Compact BE box (show all multi points)
    metric_label = ("p95" if metric == "p95_ms" else "p50" if metric == "p50_ms" else metric)
    primary_vals_compact = {k: v for k, v in (primary_vals or {}).items() if v is not None}
    selected_multi_compact = {k: list(vals) for k, vals in (selected_multi_raw or {}).items() if vals}

    render_be_combined_legend(
        ax,
        metric_label=metric_label,
        tags=tags,
        primary_map=primary_vals_compact,
        multi_map=(selected_multi_compact if selected_multi_compact else None),
        outside=False,
        loc=be_loc,
        bbox_to_anchor=None,
        colors=BE_COLORS,
        zorder=100.0,
    )

    return primary_vals_compact, selected_multi_compact


    # BEs (no PSI overlay here)
    primary_vals = {"auto": None, "t1": None}
    selected_multi_raw = {"auto": [], "t1": []}

    if be_multi_map and be_mode in ("multi", "both"):
        primary_vals = {
            "auto": primary_be_map.get((query, warm, "best", "auto", tags), primary_be_map.get((query, warm, "best", "auto"))),
            "t1":   primary_be_map.get((query, warm, "best", "t1",   tags), primary_be_map.get((query, warm, "best", "t1"))),
        }
        selected_multi_raw = annotate_be_multi(
            ax, be_multi_map,
            query=query, warm=warm, tags=tags, metric=metric,
            draw_profiles=("auto", "t1"), sqlite_variant="best",
            top_k=multi_top_k,
            min_rel_delta=multi_min_rel,
            min_abs_delta_ms=multi_min_abs_ms,
            dedupe_primary_x=primary_vals,
            dedupe_tol_log10=multi_dedupe_tol_log10,
            colors=BE_COLORS,
            ensure_include_primary=True,
            include_primary_x=primary_vals,
        )

    # Pins (fallback to primary only for drawing)
    pins_map = selected_multi_raw if any((selected_multi_raw or {}).values()) \
               else {k: ([v] if v is not None else []) for k, v in (primary_vals or {}).items()}
    _draw_be_pins(ax, pins_map, palette=BE_COLORS, max_per_profile=2)

    # Compact BE box (legend)
    metric_label = ("p95" if metric == "p95_ms" else "p50" if metric == "p50_ms" else metric)
    primary_vals_compact = {k: v for k, v in (primary_vals or {}).items() if v is not None}
    selected_multi_compact = {k: vals[:1] for k, vals in (selected_multi_raw or {}).items() if vals}

    render_be_combined_legend(
        ax,
        metric_label=metric_label,
        tags=tags,
        primary_map=primary_vals_compact,
        multi_map=(selected_multi_compact if selected_multi_compact else None),
        outside=False,
        loc=be_loc,
        bbox_to_anchor=None,
        colors=BE_COLORS,
        zorder=100.0,
    )

    return primary_vals_compact, selected_multi_compact



def plot_joined_tags_pair(
    df: pd.DataFrame,
    primary_be_map: dict,
    be_multi_map: dict | None,
    outdir: Path,
    *,
    query: str,
    warm: bool,
    metric: str,
    scan_mode: str,
    tags_pair: tuple[int, int] = (20, 50),
    be_mode: str = "multi",
    multi_top_k: int = 2,
    multi_min_rel: float = 0.10,
    multi_min_abs_ms: float = 0.0,
    multi_dedupe_tol_log10: float = 0.01,
    fig_height: float = 4.8,
):
    """
    3×2 layout:
      Row 1 (oben):   Latenz (+ BE-Linien & kompakte BE-Box) für beide Tags
      Row 2 (mitte):  PSI CPU (p95/mean je nach --psi-stat) für beide Tags
      Row 3 (unten):  PSI I/O (p95/mean je nach --psi-stat) für beide Tags

    Breite Legende nur für die Latenz-Serien (oben links gesammelt), unten zentriert.
    """
    a, b = tags_pair
    subdir = ensure_dir(outdir / "joined" / metric / ("warm" if warm else "cold"))
    fname  = subdir / f"{query}__{metric}__joined_tags{a}_{b}__{scan_mode}.png"

    # etwas höheres Figure für 3 Reihen
    fig, axes = plt.subplots(
        3, 2,
        figsize=(10.0, max(9.0, fig_height + 3.2)),
        constrained_layout=False,
        sharex="col"
    )
    plt.subplots_adjust(top=0.84, bottom=0.12, right=0.98, wspace=0.28, hspace=0.35)

    # PSI-Optionen aus den global gesetzten _psi_opts auslesen
    psi_opts = getattr(_draw_panel_for_tag, "_psi_opts", {}) or {}
    stat     = psi_opts.get("stat", "p95")
    smooth   = psi_opts.get("smooth", 3)
    eng_mode = psi_opts.get("engine", "duckdb")

    # --- Row 1: Latenz (top) ------------------------------------------------
    _draw_panel_for_tag(
        axes[0,0], df,
        primary_be_map=primary_be_map, be_multi_map=be_multi_map, be_mode=be_mode,
        query=query, warm=warm, metric=metric, scan_mode=scan_mode, tags=a,
        multi_top_k=multi_top_k, multi_min_rel=multi_min_rel,
        multi_min_abs_ms=multi_min_abs_ms, multi_dedupe_tol_log10=multi_dedupe_tol_log10,
        be_loc="upper left", show_ylabel=True, panel_title=f"tags={a}",
    )
    _draw_panel_for_tag(
        axes[0,1], df,
        primary_be_map=primary_be_map, be_multi_map=be_multi_map, be_mode=be_mode,
        query=query, warm=warm, metric=metric, scan_mode=scan_mode, tags=b,
        multi_top_k=multi_top_k, multi_min_rel=multi_min_rel,
        multi_min_abs_ms=multi_min_abs_ms, multi_dedupe_tol_log10=multi_dedupe_tol_log10,
        be_loc="upper left", show_ylabel=False, panel_title=f"tags={b}",
    )

    # --- Row 2: PSI CPU (mittig) --------------------------------------------
    _draw_psi_panel(
        axes[1,0], df,
        query=query, warm=warm, scan_mode=scan_mode,
        tags=a, kinds=["cpu"], engine_mode=eng_mode,
        profile="auto", sqlite_variant="any",  # Signatur-kompatibel, Logik nutzt _psi_opts
        stat=stat, smooth_win=smooth
    )
    _draw_psi_panel(
        axes[1,1], df,
        query=query, warm=warm, scan_mode=scan_mode,
        tags=b, kinds=["cpu"], engine_mode=eng_mode,
        profile="auto", sqlite_variant="any",
        stat=stat, smooth_win=smooth
    )
    axes[1,0].set_title(f"PSI CPU — tags={a}")
    axes[1,1].set_title(f"PSI CPU — tags={b}")

    # --- Row 3: PSI I/O (unten) ---------------------------------------------
    _draw_psi_panel(
        axes[2,0], df,
        query=query, warm=warm, scan_mode=scan_mode,
        tags=a, kinds=["io"], engine_mode=eng_mode,
        profile="auto", sqlite_variant="any",
        stat=stat, smooth_win=smooth
    )
    _draw_psi_panel(
        axes[2,1], df,
        query=query, warm=warm, scan_mode=scan_mode,
        tags=b, kinds=["io"], engine_mode=eng_mode,
        profile="auto", sqlite_variant="any",
        stat=stat, smooth_win=smooth
    )
    axes[2,0].set_title(f"PSI I/O — tags={a}")
    axes[2,1].set_title(f"PSI I/O — tags={b}")

    # Breite Legende NUR für die Latenz-Serien
    handles, labels = axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.95),
            ncol=max(2, min(len(labels), 4)),
            frameon=True, fontsize=9, borderaxespad=0.0,
        )

    fig.suptitle(f"{query} — {'warm' if warm else 'cold'} ({scan_mode})", y=0.995)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(RESULTS_DIR / "summary_with_cis.csv"))
    ap.add_argument("--break-even", default=str(RESULTS_DIR / "break_even.csv"))
    ap.add_argument("--break-even-multi", default=str(RESULTS_DIR / "break_even_multi.csv"))
    ap.add_argument("--outdir", default=str(RESULTS_DIR / "plots" / "main"))
    ap.add_argument("--tags", type=int, default=None)
    ap.add_argument("--metrics", default="p50_ms,p95_ms")
    ap.add_argument("--scan-mode", default="table", choices=["table","file","any"])

    # Sources & layers
    ap.add_argument("--be-source", default="primary", choices=["primary","multi"])
    ap.add_argument("--be-mode", default="primary", choices=["none","primary","multi","both"])
    ap.add_argument("--multi-top-k", type=int, default=2)
    ap.add_argument("--multi-min-rel", type=float, default=0.10)
    ap.add_argument("--multi-min-abs-ms", type=float, default=0.0)
    ap.add_argument("--multi-dedupe-tol-log10", type=float, default=0.01)

    # placement
    ap.add_argument("--legend-outside", action="store_true")
    ap.add_argument("--be-box-outside", action="store_true")

    # figure sizes
    ap.add_argument("--fig-height", type=float, default=4.8)

    # joined (combined 3x2)
    ap.add_argument("--make-joined", action="store_true")
    ap.add_argument("--joined-tags", default="20,50")
    ap.add_argument("--joined-outdir", default=str(RESULTS_DIR / "plots" / "main"))
    ap.add_argument("--joined-fig-height", type=float, default=4.8)

    # NEW: split output (three files)
    ap.add_argument("--make-joined-split", action="store_true",
                    help="Write three separate images: latency, psi_cpu, psi_io.")

    # PSI overlay (for singles) + PSI-only panels (joined)
    ap.add_argument("--psi-overlay", default="none", choices=["none","io","cpu","both"])
    ap.add_argument("--psi-stat", default="p95", choices=["p95","mean"])
    ap.add_argument("--psi-profile", default="auto", choices=["auto","t1","both"])
    ap.add_argument("--psi-smooth", type=int, default=3, help="centered rolling window for PSI-only panels")
    ap.add_argument("--psi-engine", default="duckdb", choices=["duckdb","sqlite","both"],
                    help="Which engine's PSI to plot in PSI panels/overlay.")
    ap.add_argument("--psi-sqlite-variant", default="any", choices=["any","idx","noidx","both"],
                    help="For sqlite PSI: use any/idx/noidx (or both) runs.")

    args = ap.parse_args()
    apply_style()

    # enforce multi for this version
    args.be_mode = "multi"

    # load
    df_all = load_summary(Path(args.summary))
    df = df_all.copy()

    if args.tags is not None and "dataset_tags" in df.columns:
        df = df[df["dataset_tags"] == args.tags].copy()

    if args.scan_mode != "any":
        df = df[df["scan_mode"] == args.scan_mode].copy()
        df_all = df_all[df_all["scan_mode"] == args.scan_mode].copy()

    queries = sorted(df["query"].dropna().unique())
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    be_multi_map = load_be_multi(Path(args.break_even_multi)) if Path(args.break_even_multi).is_file() else {}

    # Single-tag autodetect (for singles)
    inferred_tag = None
    if args.tags is None and "dataset_tags" in df.columns:
        uniq = sorted(df["dataset_tags"].dropna().unique())
        if len(uniq) == 1:
            inferred_tag = int(uniq[0])

    # PSI options shared
    psi_kinds = []
    if args.psi_overlay == "io":
        psi_kinds = ["io"]
    elif args.psi_overlay == "cpu":
        psi_kinds = ["cpu"]
    elif args.psi_overlay == "both":
        psi_kinds = ["io", "cpu"]

    duck_profiles = ["auto", "t1"] if args.psi_profile == "both" else [args.psi_profile]
    sqlite_variants = ["noidx", "idx"] if args.psi_sqlite_variant == "both" else [args.psi_sqlite_variant]

    psi_opts = {
        "kinds": psi_kinds,
        "stat": args.psi_stat,
        "smooth": args.psi_smooth,
        "engine": args.psi_engine,           # duckdb|sqlite|both
        "duck_profiles": duck_profiles,      # list
        "sqlite_variants": sqlite_variants,  # list
    }
    _draw_psi_panel._psi_opts = psi_opts
    plot_one._psi_opts = psi_opts
    _draw_panel_for_tag._psi_opts = psi_opts

    written = []
    for metric in metrics:
        # primary map for singles
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

        # single plots
        for warm in (False, True):
            for q in queries:
                f = plot_one(
                    df, primary_be_map_single, Path(args.outdir),
                    q, warm, metric, (args.tags if args.tags is not None else inferred_tag),
                    scan_mode=(args.scan_mode if args.scan_mode != "any" else "table"),
                    be_multi_map=be_multi_map,
                    be_mode=args.be_mode,
                    multi_top_k=args.multi_top_k,
                    multi_min_rel=args.multi_min_rel,
                    multi_min_abs_ms=args.multi_min_abs_ms,
                    multi_dedupe_tol_log10=args.multi_dedupe_tol_log10,
                    legend_outside=args.legend_outside,
                    be_box_outside=args.be_box_outside,
                    fig_height=args.fig_height,
                )
                if f:
                    written.append(f)

        # joined plots (combined 3×2)
        if args.make_joined:
            parts = [p.strip() for p in args.joined_tags.split(",") if p.strip()]
            if len(parts) != 2:
                raise SystemExit(f"--joined-tags expects exactly two values, e.g., 20,50 (got: {args.joined_tags})")
            tA, tB = int(parts[0]), int(parts[1])

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
                for q in queries:
                    jf = plot_joined_tags_pair(
                        df_all,
                        primary_be_map_joined,
                        be_multi_map,
                        Path(args.joined_outdir),
                        query=q,
                        warm=warm,
                        metric=metric,
                        scan_mode=(args.scan_mode if args.scan_mode != "any" else "table"),
                        tags_pair=(tA, tB),
                        be_mode="multi",
                        multi_top_k=args.multi_top_k,
                        multi_min_rel=args.multi_min_rel,
                        multi_min_abs_ms=args.multi_min_abs_ms,
                        multi_dedupe_tol_log10=args.multi_dedupe_tol_log10,
                        fig_height=args.joined_fig_height,
                    )
                    if jf:
                        written.append(jf)

        # joined plots (SPLIT → three files)
        if args.make_joined_split:
            parts = [p.strip() for p in args.joined_tags.split(",") if p.strip()]
            if len(parts) != 2:
                raise SystemExit(f"--joined-tags expects exactly two values, e.g., 20,50 (got: {args.joined_tags})")
            tA, tB = int(parts[0]), int(parts[1])

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
                for q in queries:
                    files = plot_joined_tags_split(
                        df_all,
                        primary_be_map_joined,
                        be_multi_map,
                        Path(args.joined_outdir),
                        query=q,
                        warm=warm,
                        metric=metric,
                        scan_mode=(args.scan_mode if args.scan_mode != "any" else "table"),
                        tags_pair=(tA, tB),
                        be_mode="multi",
                        multi_top_k=args.multi_top_k,
                        multi_min_rel=args.multi_min_rel,
                        multi_min_abs_ms=args.multi_min_abs_ms,
                        multi_dedupe_tol_log10=args.multi_dedupe_tol_log10,
                        fig_height=args.joined_fig_height,
                    )
                    written.extend(files)

    print(f"[ok] wrote {len(written)} plots → {args.outdir}")




if __name__ == "__main__":
    main()
