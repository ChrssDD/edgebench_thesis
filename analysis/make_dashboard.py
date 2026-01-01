import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from config import RESULTS_DIR

PLOTS = (RESULTS_DIR / "plots")
PLOTS.mkdir(parents=True, exist_ok=True)
IT_PATH = RESULTS_DIR / "iterations_long.csv"

def _col(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def _sqlite_label_from_db(db_path: str) -> str:
    stem = Path(str(db_path)).stem
    if stem.endswith("_idx") or "_idx" in stem:
        return "sqlite_idx"
    if stem.endswith("_noidx") or "_noidx" in stem:
        return "sqlite_noidx"
    return "sqlite"

def _add_label(sub: pd.DataFrame, thr_col: str | None):
    labels = []
    for _, row in sub.iterrows():
        if row["engine"] == "sqlite":
            labels.append(_sqlite_label_from_db(row["db"]))
        else:
            if thr_col and (thr_col in sub.columns) and pd.notna(row[thr_col]):
                labels.append(f"duckdb_t{int(row[thr_col])}")
            else:
                labels.append("duckdb")
    return sub.assign(label=labels)

def _load_break_even_multi(preferred_metric: str = "p95_ms"):
    p = RESULTS_DIR / "break_even.csv"
    if not p.exists():
        return {}, pd.DataFrame()
    be = pd.read_csv(p)
    if be.empty or "break_even_rows" not in be.columns or "sqlite_variant" not in be.columns:
        return {}, pd.DataFrame()

    # Normalize columns
    be["query"] = be["query"].astype(str)
    be["warm"]  = be["warm"].astype(str).map(lambda x: x.strip().lower() in ("true", "1"))
    if "duckdb_profile" not in be.columns:
        be["duckdb_profile"] = "auto"
    if "metric" not in be.columns:
        # Legacy files: assume they represent the preferred metric (p95 by default).
        be["metric"] = preferred_metric

    # Prefer p95; if not available (older outputs), fall back to p50
    if (be["metric"] == preferred_metric).any():
        pref = be[be["metric"] == preferred_metric].copy()
    else:
        # fallback to p50_ms if preferred not present
        fallback_metric = "p50_ms"
        pref = be[be["metric"] == fallback_metric].copy()

    # If multiple tags levels exist for the same (q,w,variant,profile), keep the largest tags
    if "tags" in pref.columns:
        pref = (pref.sort_values("tags")
                    .groupby(["query","warm","sqlite_variant","duckdb_profile"], as_index=False)
                    .tail(1))

    be_map = {
        (r["query"], r["warm"],
         str(r["sqlite_variant"]).lower(),
         str(r["duckdb_profile"]).lower()): r["break_even_rows"]
        for _, r in pref.iterrows()
        if pd.notna(r["break_even_rows"])
    }
    return be_map, be




PALETTE = {
    "sqlite":       "#1f77b4",
    "sqlite_noidx": "#1f77b4",
    "sqlite_idx":   "#1f77b4",
    "duckdb_t1":    "#ff7f0e",
    "duckdb":       "#2ca02c",
}
MARKERS = {
    "sqlite": "o",
    "sqlite_noidx": "o",
    "sqlite_idx": "o",
    "duckdb_t1": "s",
    "duckdb": "^",
}
ORDER = ["sqlite_noidx", "sqlite_idx", "sqlite", "duckdb_t1", "duckdb"]

def _friendly_label(col):
    return {
        "duckdb": "DuckDB (auto)",
        "duckdb_t1": "DuckDB (1 thread)",
        "sqlite_idx": "SQLite (idx)",
        "sqlite_noidx": "SQLite (no idx)",
        "sqlite": "SQLite",
    }.get(col, col)

def _fmt_rows(x, _):
    x = float(x)
    if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
    if x >= 1_000:     return f"{x/1_000:.0f}k"
    return f"{int(x)}"

def _apply_axes_style(ax, xlabel, ylabel, force_logy=False):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ymin, ymax = ax.get_ylim()
    if force_logy or (ymax / max(ymin, 1e-9) > 20):
        ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.tick_params(axis="both", which="both", labelsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_rows))
    for spine in ax.spines.values(): spine.set_alpha(0.3)

def _speedup_series_variant(sub, rows_col, thr_col, variant, baseline="auto"):
    """
    Returns a Series index=rows, values = (SQLite_variant p50) / (DuckDB_baseline p50)
    baseline ∈ {"auto","t1"}
    """
    # Pick DuckDB baseline
    if baseline == "t1" and thr_col and (thr_col in sub.columns):
        base_mask = (sub["engine"] == "duckdb") & (sub[thr_col] == 1)
    else:
        # auto threads (threads is NaN) or if no threads column, all duck rows
        base_mask = (sub["engine"] == "duckdb") & ((sub[thr_col].isna()) if (thr_col and thr_col in sub.columns) else True)
    base = sub[base_mask].groupby(rows_col)["p50_ms"].min()

    # SQLite variant slice
    s_sql = sub[sub["engine"] == "sqlite"].copy()
    if variant == "idx":
        s_sql = s_sql[s_sql["db"].astype(str).str.contains(r"_idx\b")]
    elif variant == "noidx":
        s_sql = s_sql[s_sql["db"].astype(str).str.contains(r"_noidx\b")]
    # variant == "best": keep both and take min
    s = s_sql.groupby(rows_col)["p50_ms"].min()

    sp = (s / base).replace([np.inf, -np.inf], pd.NA).dropna().sort_index()
    return sp


def main():
    summ = pd.read_csv(RESULTS_DIR/"summary_with_cis.csv")
    valp = RESULTS_DIR/"validation_suite.csv"
    vs = pd.read_csv(valp) if valp.exists() else pd.DataFrame()
    be_map, _ = _load_break_even_multi()
    it = pd.read_csv(IT_PATH) if IT_PATH.exists() else pd.DataFrame()

    scan_col = _col(summ, "scan_mode", "scan", "mode")
    rows_col = _col(summ, "dataset_rows", "rows")
    thr_col  = _col(summ, "threads")

    # ===== P50 vs rows (with BE lines for {auto,t1}) =====
    plot_files = []
    for q in sorted(summ["query"].dropna().unique()):
        for warm in (False, True):
            scans = sorted(summ[scan_col].dropna().unique()) if scan_col else [None]
            for scan in scans:
                sub = summ[(summ["query"] == q) & (summ["warm"] == warm)]
                if scan_col:
                    sub = sub[sub[scan_col] == scan]
                if "count" in sub.columns:
                    sub = sub[sub["count"].fillna(0) > 0]
                if (rows_col is None) or sub.empty:
                    continue
                lab = _add_label(sub, thr_col)
                g = (lab.groupby(["label", rows_col], as_index=False)["p50_ms"]
                       .median()
                       .sort_values([rows_col]))
                pv = g.pivot(index=rows_col, columns="label", values="p50_ms").sort_index()
                if pv.empty:
                    continue

                fig, ax = plt.subplots(figsize=(7.5, 4.2))
                for col in [c for c in ORDER if c in pv.columns]:
                    kwargs = {}
                    if col == "sqlite_noidx":
                        kwargs["markerfacecolor"] = "none"
                        kwargs["markeredgewidth"] = 1.8
                    ax.plot(
                        pv.index.values, pv[col].values,
                        marker=MARKERS.get(col,"o"), linewidth=2, markersize=5,
                        label=_friendly_label(col),
                    )
                _apply_axes_style(ax, xlabel="Rows (log scale)", ylabel="Median p50 (ms)")

                # BE lines for variants and both profiles
                for variant, style in (("best", (0.3, "--")), ("idx", (0.5, "-.")), ("noidx", (0.5, ":"))):
                    for prof, shade in (("auto", "0.3"), ("t1", "0.45")):
                        be_x = be_map.get((q, warm, variant, prof))
                        if pd.notna(be_x) if be_x is not None else False:
                            ax.axvline(be_x, linestyle=style[1], linewidth=1, color=shade)
                            y_top = ax.get_ylim()[1]
                            ax.text(
                                be_x, y_top*0.96,
                                f"{variant}/{prof} ≈ {be_x:,.0f}",
                                rotation=90, va="top", ha="right", fontsize=8
                            )

                ax.set_title(f"{q} — {'warm' if warm else 'cold'}", fontsize=11)
                ax.legend(loc="best", frameon=False, fontsize=9)
                fig.tight_layout()
                fname = f"{q}_p50_by_rows_w{int(warm)}" + (f"_{scan}" if scan_col else "") + ".png"
                fig.savefig(PLOTS/fname, dpi=150); plt.close(fig)
                plot_files.append(fname)

            # PSI scatter (optional)
            if not it.empty and "psi_after_cpu_some_avg10" in it.columns:
                it_sub = it[(it["query"] == q) & (it["warm"] == warm)].dropna(subset=["dur_ms","psi_after_cpu_some_avg10"])
                if not it_sub.empty:
                    fig2, ax2 = plt.subplots(figsize=(6.2, 4.0))
                    ax2.scatter(it_sub["psi_after_cpu_some_avg10"].values, it_sub["dur_ms"].values, s=12, alpha=0.35)
                    ax2.set_xlabel("PSI cpu avg10 (after)")
                    ax2.set_ylabel("Iteration duration (ms)")
                    ax2.grid(True, which="both", linewidth=0.5, alpha=0.35)
                    ax2.set_title(f"{q} — {'warm' if warm else 'cold'}: latency vs PSI", fontsize=11)
                    fig2.tight_layout()
                    fname2 = f"{q}_psi_corr_w{int(warm)}.png"
                    fig2.savefig(PLOTS/fname2, dpi=150); plt.close(fig2)
                    plot_files.append(fname2)

    # ===== Speedup curves vs DuckDB(auto) and vs DuckDB(t1) =====
    speedup_files = []
    for q in sorted(summ["query"].dropna().unique()):
        for warm in (False, True):
            scans = sorted(summ[scan_col].dropna().unique()) if scan_col else [None]
            for scan in scans:
                sub = summ[(summ["query"] == q) & (summ["warm"] == warm)]
                if scan_col:
                    sub = sub[sub[scan_col] == scan]
                if (rows_col is None) or sub.empty:
                    continue
                for baseline in ("auto","t1"):
                    sp_best  = _speedup_series_variant(sub, rows_col, thr_col, "best", baseline=baseline)
                    sp_idx   = _speedup_series_variant(sub, rows_col, thr_col, "idx",  baseline=baseline)
                    sp_noidx = _speedup_series_variant(sub, rows_col, thr_col, "noidx",baseline=baseline)
                    if all(x.empty for x in (sp_best, sp_idx, sp_noidx)):
                        continue

                    fig, ax = plt.subplots(figsize=(7.5, 4.2))
                    if not sp_best.empty:
                        ax.plot(sp_best.index.values, sp_best.values, marker="o", linewidth=2,
                                label=f"SQLite-best / DuckDB({baseline})")
                    if not sp_idx.empty:
                        ax.plot(sp_idx.index.values, sp_idx.values, marker="o", linewidth=1.8, linestyle="--",
                                label=f"SQLite-idx / DuckDB({baseline})")
                    if not sp_noidx.empty:
                        ax.plot(sp_noidx.index.values, sp_noidx.values, marker="o", linewidth=1.8, linestyle=":",
                                label=f"SQLite-noidx / DuckDB({baseline})")
                    ax.axhline(1.0, linestyle=":", linewidth=1, color="0.4")

                    _apply_axes_style(ax, xlabel="Rows (log scale)", ylabel="Speedup (sqlite ÷ duckdb)")
                    ax.set_title(f"{q} — {'warm' if warm else 'cold'}", fontsize=11)
                    ax.legend(loc="best", frameon=False, fontsize=9)

                    if not sp_best.empty:
                        x_last = sp_best.index.values[-1]; y_last = sp_best.values[-1]
                        ax.annotate(f"{y_last:.1f}× @ {x_last:,.0f}",
                                    xy=(x_last, y_last), xytext=(5, 5), textcoords="offset points", fontsize=9)

                    fig.tight_layout()
                    suffix = f"_{scan}" if scan_col else ""
                    fname = f"{q}_speedup_sqlite_over_duckdb-{baseline}_w{int(warm)}{suffix}.png"
                    fig.savefig(PLOTS/fname, dpi=150); plt.close(fig)
                    speedup_files.append(fname)

    # Warm <= Cold check
    key = ["engine", "query"] + ([scan_col] if scan_col else [])
    cold = summ[summ["warm"] == False].groupby(key)["p50_ms"].median()
    warmg = summ[summ["warm"] == True].groupby(key)["p50_ms"].median()
    cmp  = pd.concat([cold.rename("cold"), warmg.rename("warm")], axis=1).dropna()
    viol_warm = cmp[cmp["warm"] > cmp["cold"]].reset_index()

    # HTML
    html = []
    html.append("<html><head><meta charset='utf-8'><title>EdgeBench Dashboard</title></head><body>")
    html.append("<h1>EdgeBench Dashboard</h1>")

    if valp.exists():
        ok = (vs["status"] == "OK").sum()
        mism = (vs["status"] == "MISMATCH").sum()
        err = (vs["status"] == "ERROR").sum()
        html.append(f"<h2>Validation</h2><p>OK={ok} · MISMATCH={mism} · ERROR={err}</p>")
        html.append(vs.to_html(index=False))
    else:
        html.append("<h2>Validation</h2><p>validation_suite.csv nicht gefunden.</p>")

    be_csv = RESULTS_DIR/"break_even.csv"
    if be_csv.exists():
        be_full = pd.read_csv(be_csv)
        html.append("<h2>Break-even (DuckDB {auto,t1} vs SQLite variants)</h2>")
        html.append(be_full.to_html(index=False))

    html.append("<h2>Plots</h2>")
    for f in plot_files + speedup_files:
        html.append(f"<div><img src='plots/{f}' style='max-width: 1000px;'></div>")

    html.append("<h2>Checks</h2>")
    if viol_warm.empty:
        html.append("<p>Warm ≤ Cold: OK</p>")
    else:
        html.append("<h3>Warm langsamer als Cold</h3>")
        html.append(viol_warm.to_html(index=False))

    if IT_PATH.exists():
        html.append("<p>PSI correlation plots included where data was available.</p>")
    else:
        html.append("<p>No PSI iteration data found (iterations_long.csv missing).</p>")

    html.append("</body></html>")
    out = RESULTS_DIR/"dashboard.html"
    out.write_text("\n".join(html), encoding="utf-8")
    print(f"[ok] wrote {out} and {len(plot_files)+len(speedup_files)} plots → {PLOTS}")


if __name__ == "__main__":
    main()
