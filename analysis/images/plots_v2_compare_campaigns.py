import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from .style import apply_style
from .utils import (
    ensure_dir,
    load_summary,
    load_be_multi,
    derive_primary_map_from_multi,
    load_be_map,
)

# Reuse Look&Feel aus plots_v2_main
from .plots_v2_main import _draw_panel_for_tag, _draw_psi_panel


# ----------------------------- helpers -------------------------------------

def _camp_name(p: Path) -> str:
    return p.name


def _normalize_dataset_rows(df, *, camp: Path):
    if "dataset_rows" in df.columns:
        return df

    for alt in ("total_rows", "rows_total", "dataset_total_rows", "n_rows", "rows"):
        if alt in df.columns:
            df = df.copy()
            df["dataset_rows"] = df[alt]
            return df

    if "dataset_tags" in df.columns:
        for rpt in ("rows_per_tag", "dataset_rows_per_tag", "rpt", "rows_per_series"):
            if rpt in df.columns:
                df = df.copy()
                df["dataset_rows"] = df[rpt] * df["dataset_tags"]
                return df

    raise SystemExit(
        "[err] summary CSV has no dataset_rows and no known fallback columns.\n"
        f"campaign: {camp}\n"
        f"columns: {list(df.columns)}"
    )


def _read_campaign(camp: Path, scan_mode: str):
    summ = camp / "results" / "summary_with_cis.csv"
    bem  = camp / "results" / "break_even_multi.csv"
    bep  = camp / "results" / "break_even.csv"

    if not summ.is_file():
        raise SystemExit(f"[err] missing: {summ}")

    df_all = load_summary(summ)
    df_all = _normalize_dataset_rows(df_all, camp=camp)

    if scan_mode != "any" and "scan_mode" in df_all.columns:
        df_all = df_all[df_all["scan_mode"] == scan_mode].copy()

    be_multi = load_be_multi(bem) if bem.is_file() else {}
    be_prim  = load_be_map(bep, preferred_metric="p95_ms") if bep.is_file() else {}

    return df_all, be_multi, be_prim


def _filter_df(df, *, query: str, warm: bool, scan_mode: str, tags: int):
    d = df
    if "query" in d.columns:
        d = d[d["query"] == query]
    if "warm" in d.columns:
        d = d[d["warm"] == warm]
    if scan_mode != "any" and "scan_mode" in d.columns:
        d = d[d["scan_mode"] == scan_mode]
    if "dataset_tags" in d.columns:
        d = d[d["dataset_tags"] == tags]
    return d


def _xrange_for_panel(df, *, query: str, warm: bool, scan_mode: str, tags: int):
    d = _filter_df(df, query=query, warm=warm, scan_mode=scan_mode, tags=tags)
    if d.empty or "dataset_rows" not in d.columns:
        return None
    x = d["dataset_rows"].dropna().astype(float).values
    if x.size == 0:
        return None
    x0 = float(np.nanmin(x))
    x1 = float(np.nanmax(x))
    if not np.isfinite(x0) or not np.isfinite(x1) or x0 <= 0 or x1 <= 0 or x1 <= x0:
        return None
    return (x0, x1)


def _overlap_xrange(ranges: list[tuple[float, float]]):
    if not ranges:
        return None
    lo = max(r[0] for r in ranges)
    hi = min(r[1] for r in ranges)
    if hi <= lo:
        return None
    return (lo, hi)


def _slice_to_xrange(df, *, query: str, warm: bool, scan_mode: str, tags: int, xlim):
    d = _filter_df(df, query=query, warm=warm, scan_mode=scan_mode, tags=tags)
    if d.empty:
        return d

    x0, x1 = xlim

    inside = d[(d["dataset_rows"] >= x0) & (d["dataset_rows"] <= x1)].copy()

    # NEU: einen “Ankerpunkt” links vom Overlap behalten (alle Zeilen dieser Größe)
    below = d[d["dataset_rows"] < x0]
    if not below.empty:
        r_keep = float(below["dataset_rows"].max())
        inside = pd.concat([below[below["dataset_rows"] == r_keep], inside], ignore_index=True)

    return inside



def _union_ylims(ax_list):
    ys = []
    for ax in ax_list:
        if ax is None:
            continue
        y0, y1 = ax.get_ylim()
        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
            ys.append((y0, y1))
    if not ys:
        return
    lo = min(y[0] for y in ys)
    hi = max(y[1] for y in ys)
    pad = 0.04 * (hi - lo)
    lo = max(0.0, lo - pad)
    hi = hi + pad
    for ax in ax_list:
        if ax is not None:
            ax.set_ylim(lo, hi)


def _parse_panel_spec(spec: str):
    """
    "B:4,A:20,A:50" -> [("B",4),("A",20),("A",50)]
    """
    out = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            raise SystemExit(f"[err] bad --panel-spec element: {p} (expected like A:20)")
        side, tag = p.split(":", 1)
        side = side.strip().upper()
        tag = int(tag.strip())
        if side not in ("A", "B"):
            raise SystemExit(f"[err] bad side in --panel-spec: {side} (use A or B)")
        out.append((side, tag))
    if len(out) != 3:
        raise SystemExit(f"[err] --panel-spec must have exactly 3 panels (got {len(out)}): {spec}")
    return out


def _filter_primary_map_to_xrange(pm: dict, xlim):
    if xlim is None:
        return pm
    x0, x1 = xlim
    return {k: v for k, v in pm.items() if isinstance(v, (int, float)) and (x0 <= float(v) <= x1)}


# ----------------------------- plotting ------------------------------------

def plot_row_1x3_latency(
    *,
    panels,  # list of dicts: {name, df, pm, mm, side, tag}
    outdir: Path,
    query: str, warm: bool, metric: str, scan_mode: str,
    be_mode: str, multi_top_k: int, multi_min_rel: float, multi_min_abs_ms: float, multi_dedupe_tol_log10: float,
    fig_height: float,
    xlim,
):
    sub = ensure_dir(outdir / "row_1x3" / metric / ("warm" if warm else "cold"))
    spec = "__".join([f"{p['side']}{p['tag']}" for p in panels])
    fname = sub / f"{query}__{metric}__row1x3__{spec}__{scan_mode}__latency.png"

    fig, axes = plt.subplots(1, 3, figsize=(14.5, max(3.9, fig_height)), constrained_layout=False, sharex="col", sharey=True)
    plt.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.98, wspace=0.26)

    legend_handles, legend_labels = None, None
    ax_list = []

    for i, p in enumerate(panels):
        ax = axes[i]
        ax_list.append(ax)

        title = f"tags={p['tag']} — {p['name']}"
        ok = True
        try:
            _draw_panel_for_tag(
                ax, p["df"],
                primary_be_map=p["pm"], be_multi_map=p["mm"], be_mode=be_mode,
                query=query, warm=warm, metric=metric, scan_mode=scan_mode, tags=p["tag"],
                multi_top_k=multi_top_k, multi_min_rel=multi_min_rel,
                multi_min_abs_ms=multi_min_abs_ms, multi_dedupe_tol_log10=multi_dedupe_tol_log10,
                be_loc="upper left", show_ylabel=(i == 0), panel_title=title,
            )
        except Exception:
            ok = False
            ax.text(0.5, 0.5, f"no data\n{title}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        if ok and legend_handles is None:
            h, l = ax.get_legend_handles_labels()
            if h:
                legend_handles, legend_labels = h, l

        if ok and xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.95),
            ncol=max(2, min(len(legend_labels), 4)),
            frameon=True, fontsize=9, borderaxespad=0.0,
        )

    _union_ylims(ax_list)

    xnote = f" | overlap X: [{xlim[0]:.0f}, {xlim[1]:.0f}]" if xlim is not None else ""
    fig.suptitle(f"{query} — {'warm' if warm else 'cold'} ({scan_mode}){xnote}", y=0.995)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    return fname


def plot_row_1x3_psi(
    *,
    kind: str,  # "cpu" or "io"
    panels,
    outdir: Path,
    query: str, warm: bool, metric: str, scan_mode: str,
    fig_height: float,
    xlim,
    psi_stat: str, psi_smooth: int, psi_engine: str, psi_profile: str, psi_sqlite_variant: str,
):
    sub = ensure_dir(outdir / "row_1x3" / metric / ("warm" if warm else "cold"))
    spec = "__".join([f"{p['side']}{p['tag']}" for p in panels])
    fname = sub / f"{query}__{metric}__row1x3__{spec}__{scan_mode}__psi_{kind}.png"

    duck_profiles = ["auto", "t1"] if psi_profile == "both" else [psi_profile]
    sqlite_variants = ["noidx", "idx"] if psi_sqlite_variant == "both" else [psi_sqlite_variant]
    _draw_psi_panel._psi_opts = {
        "stat": psi_stat,
        "smooth": psi_smooth,
        "engine": psi_engine,
        "duck_profiles": duck_profiles,
        "sqlite_variants": sqlite_variants,
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.5, max(3.9, fig_height)), constrained_layout=False, sharex="col", sharey=True)
    plt.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.98, wspace=0.26)

    ax_list = []
    legend_handles, legend_labels = None, None

    for i, p in enumerate(panels):
        ax = axes[i]
        ax_list.append(ax)
        title = f"PSI {kind.upper()} — tags={p['tag']} — {p['name']}"
        ok = True
        try:
            _draw_psi_panel(
                ax, p["df"],
                query=query, warm=warm, scan_mode=scan_mode,
                tags=p["tag"], kinds=[kind], engine_mode=psi_engine,
                profile="auto", sqlite_variant="any",
                stat=psi_stat, smooth_win=psi_smooth, show_legend=False
            )
            ax.set_title(title)
        except Exception:
            ok = False
            ax.text(0.5, 0.5, f"no data\n{title}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        if ok and legend_handles is None:
            h, l = ax.get_legend_handles_labels()
            if h:
                legend_handles, legend_labels = h, l

        if ok and xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.95),
            ncol=max(2, min(len(legend_labels), 4)),
            frameon=True, fontsize=9, borderaxespad=0.0,
        )

    _union_ylims(ax_list)

    xnote = f" | overlap X: [{xlim[0]:.0f}, {xlim[1]:.0f}]" if xlim is not None else ""
    fig.suptitle(f"{query} — PSI {kind.upper()} — {'warm' if warm else 'cold'} ({scan_mode}){xnote}", y=0.995)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    return fname


# ----------------------------- main ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camp-a", required=True)
    ap.add_argument("--camp-b", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--panel-spec", default="B:4,A:20,A:50",
                    help="Exactly 3 panels, e.g. 'B:4,A:20,A:50' (row order left->right).")

    ap.add_argument("--metrics", default="p50_ms,p95_ms")
    ap.add_argument("--scan-mode", default="table", choices=["table", "file", "any"])

    # What to render
    ap.add_argument("--layers", default="latency,psi_cpu,psi_io",
                    help="Comma list: latency,psi_cpu,psi_io")

    # BE selection (latency)
    ap.add_argument("--be-source", default="multi", choices=["multi", "primary"])
    ap.add_argument("--be-mode", default="primary", choices=["primary", "multi", "both", "none"])
    ap.add_argument("--multi-top-k", type=int, default=2)
    ap.add_argument("--multi-min-rel", type=float, default=0.10)
    ap.add_argument("--multi-min-abs-ms", type=float, default=0.0)
    ap.add_argument("--multi-dedupe-tol-log10", type=float, default=0.01)

    # PSI
    ap.add_argument("--psi-stat", default="p95", choices=["p95", "mean"])
    ap.add_argument("--psi-smooth", type=int, default=3)
    ap.add_argument("--psi-engine", default="duckdb", choices=["duckdb", "sqlite", "both"])
    ap.add_argument("--psi-profile", default="both", choices=["auto", "t1", "both"])
    ap.add_argument("--psi-sqlite-variant", default="any", choices=["any", "idx", "noidx", "both"])

    ap.add_argument("--fig-height", type=float, default=4.8)

    args = ap.parse_args()
    apply_style()

    campA = Path(args.camp_a)
    campB = Path(args.camp_b)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    dfA_all, mmA, _ = _read_campaign(campA, args.scan_mode)
    dfB_all, mmB, _ = _read_campaign(campB, args.scan_mode)

    nameA = _camp_name(campA)
    nameB = _camp_name(campB)

    panels_spec = _parse_panel_spec(args.panel_spec)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    layers  = [s.strip() for s in args.layers.split(",") if s.strip()]

    qA = set(dfA_all["query"].dropna().unique()) if "query" in dfA_all.columns else set()
    qB = set(dfB_all["query"].dropna().unique()) if "query" in dfB_all.columns else set()
    queries = sorted(qA | qB)

    written = 0

    for metric in metrics:
        for warm in (False, True):
            for q in queries:
                # (1) X-range pro Panel bestimmen -> Overlap
                ranges = []
                for side, tag in panels_spec:
                    df_src = dfA_all if side == "A" else dfB_all
                    r = _xrange_for_panel(df_src, query=q, warm=warm, scan_mode=args.scan_mode, tags=tag)
                    if r is None:
                        ranges = []
                        break
                    ranges.append(r)

                xlim = _overlap_xrange(ranges) if ranges else None
                if xlim is None:
                    # Kein sinnvoller Vergleich (kein Overlap oder missing)
                    continue

                # (2) Daten echt auf Overlap slicen
                panels = []
                all_have = True
                for side, tag in panels_spec:
                    if side == "A":
                        df_src, mm_src, nm = dfA_all, mmA, nameA
                    else:
                        df_src, mm_src, nm = dfB_all, mmB, nameB

                    d = _slice_to_xrange(df_src, query=q, warm=warm, scan_mode=args.scan_mode, tags=tag, xlim=xlim)
                    if d.empty:
                        all_have = False
                        break

                    # BE: primary-map pro Panel (und optional multi)
                    if args.be_source == "multi":
                        pm = derive_primary_map_from_multi(
                            mm_src, preferred_metric=metric, tags=tag,
                            min_rel_delta=args.multi_min_rel, min_abs_delta_ms=args.multi_min_abs_ms
                        )
                    else:
                        be_csv = (campA if side == "A" else campB) / "results" / "break_even.csv"
                        pm = load_be_map(be_csv, preferred_metric=metric) if be_csv.is_file() else {}

                    # Primary-BE auf Overlap filtern, damit Box/Lines nicht “außerhalb” suggerieren
                    pm = _filter_primary_map_to_xrange(pm, xlim)

                    panels.append({
                        "side": side,
                        "tag": tag,
                        "name": nm,
                        "df": d,
                        "pm": pm,
                        "mm": (mm_src if args.be_mode in ("multi", "both") else {}),
                    })

                if not all_have:
                    continue

                scan_mode = args.scan_mode if args.scan_mode != "any" else "table"

                if "latency" in layers:
                    plot_row_1x3_latency(
                        panels=panels,
                        outdir=outdir,
                        query=q, warm=warm, metric=metric, scan_mode=scan_mode,
                        be_mode=args.be_mode,
                        multi_top_k=args.multi_top_k,
                        multi_min_rel=args.multi_min_rel,
                        multi_min_abs_ms=args.multi_min_abs_ms,
                        multi_dedupe_tol_log10=args.multi_dedupe_tol_log10,
                        fig_height=args.fig_height,
                        xlim=xlim,
                    )
                    written += 1

                if "psi_cpu" in layers:
                    plot_row_1x3_psi(
                        kind="cpu",
                        panels=panels,
                        outdir=outdir,
                        query=q, warm=warm, metric=metric, scan_mode=scan_mode,
                        fig_height=args.fig_height,
                        xlim=xlim,
                        psi_stat=args.psi_stat,
                        psi_smooth=args.psi_smooth,
                        psi_engine=args.psi_engine,
                        psi_profile=args.psi_profile,
                        psi_sqlite_variant=args.psi_sqlite_variant,
                    )
                    written += 1

                if "psi_io" in layers:
                    plot_row_1x3_psi(
                        kind="io",
                        panels=panels,
                        outdir=outdir,
                        query=q, warm=warm, metric=metric, scan_mode=scan_mode,
                        fig_height=args.fig_height,
                        xlim=xlim,
                        psi_stat=args.psi_stat,
                        psi_smooth=args.psi_smooth,
                        psi_engine=args.psi_engine,
                        psi_profile=args.psi_profile,
                        psi_sqlite_variant=args.psi_sqlite_variant,
                    )
                    written += 1

    print(f"[ok] wrote {written} plots -> {outdir}")


if __name__ == "__main__":
    main()
