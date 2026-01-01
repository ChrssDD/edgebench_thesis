# analysis/images/plots_v2_ram.py
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- style / utils ----------
try:
    from .style import apply_style
    from .utils import ensure_dir
except Exception:
    def apply_style():
        plt.rcParams.update({
            "figure.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": False,
            "savefig.bbox": "tight",
        })
    def ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True); return p


# ---------- IO ----------
def _need_cols(df: pd.DataFrame, cols: list[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where} missing columns: {missing}")

def load_summary_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    _need_cols(df, ["engine","query","warm","threads","scan_mode","dataset_rows"], str(p))
    return df

def _normalize_warm(series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["1","true","yes","warm"])

def load_be_csv(p: Path, preferred_metric: str = "p95_ms", tags: int | None = None) -> pd.DataFrame | None:
    if not p.exists():
        return None
    be = pd.read_csv(p)
    need = {"query","warm","sqlite_variant","duckdb_profile","break_even_rows"}
    if not need.issubset(be.columns):
        return None
    be["query"] = be["query"].astype(str)
    be["warm"] = _normalize_warm(be["warm"])
    if "metric" not in be.columns:
        be["metric"] = preferred_metric
    if tags is not None and "tags" in be.columns:
        be = be[be["tags"] == tags]
    return be

def be_map_from_df(be: pd.DataFrame) -> dict:
    """
    Map: (query, warm, sqlite_variant, duckdb_profile, metric) -> float (break_even_rows or NaN)
    """
    out = {}
    if be is None or be.empty:
        return out
    for metric in sorted(be["metric"].unique()):
        dfm = be[be["metric"] == metric]
        for _, r in dfm.iterrows():
            key = (str(r["query"]), bool(r["warm"]),
                   str(r["sqlite_variant"]).lower(),
                   str(r["duckdb_profile"]).lower(),
                   metric)
            out[key] = r.get("break_even_rows", np.nan)
    return out


# ---- Multi-BE ----
def _parse_be_points(s: str) -> list[float]:
    """
    Robust, aber NICHT mehr still: gibt bei fehlerhaften Tokens eine Warnung aus.
    """
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception as e:
            warnings.warn(f"[break_even_multi.csv] Could not parse token '{tok}' as float: {e}")
    return sorted(out)

def load_be_multi_csv(p: Path, preferred_metric: str = "p95_ms", tags: int | None = None) -> pd.DataFrame | None:
    """
    Erwartete Spalten:
      query,warm,tags,metric,sqlite_variant,duckdb_profile,be_points,be_count,zones_json
    """
    if not p.exists():
        return None
    df = pd.read_csv(p)
    need = {"query","warm","metric","sqlite_variant","duckdb_profile","be_points"}
    if not need.issubset(df.columns):
        return None
    df["query"] = df["query"].astype(str)
    df["warm"] = _normalize_warm(df["warm"])
    if "metric" not in df.columns:
        df["metric"] = preferred_metric
    if tags is not None and "tags" in df.columns:
        df = df[df["tags"] == tags]
    df = df.copy()
    df["be_list"] = df["be_points"].apply(_parse_be_points)
    df["be_count"] = df["be_list"].apply(len)
    return df

def be_multi_map_from_df(df: pd.DataFrame) -> dict:
    """
    Map: (query, warm, sqlite_variant, duckdb_profile, metric) -> list[float]
    """
    out = {}
    if df is None or df.empty:
        return out
    for _, r in df.iterrows():
        key = (str(r["query"]), bool(r["warm"]),
               str(r["sqlite_variant"]).lower(),
               str(r["duckdb_profile"]).lower(),
               str(r["metric"]))
        out[key] = list(r["be_list"])
    return out


# ---------- Transforms ----------
def duck_pivot_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Wide DF index=(query,warm,dataset_rows), columns 'auto','t1' (DuckDB, table-scan), Aggregation=median.
    """
    sub = df[(df["engine"]=="duckdb") & (df["scan_mode"]=="table")].copy()
    auto = (sub[sub["threads"].isna()]
            .groupby(["query","warm","dataset_rows"], as_index=True)[metric].median()
            .rename("auto"))
    t1   = (sub[sub["threads"]==1]
            .groupby(["query","warm","dataset_rows"], as_index=True)[metric].median()
            .rename("t1"))
    wide = pd.concat([auto, t1], axis=1)
    wide.index = pd.MultiIndex.from_tuples(wide.index, names=["query","warm","dataset_rows"])
    return wide.sort_index()

def sqlite_pivot_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Wide DF index=(query,warm,dataset_rows), columns 'best','idx','noidx' (SQLite, table-scan), Aggregation=median.
    """
    sub = df[(df["engine"]=="sqlite") & (df["scan_mode"]=="table")].copy()
    if sub.empty:
        return pd.DataFrame()
    med = (sub.groupby(["sqlite_variant","query","warm","dataset_rows"], as_index=False)[metric]
               .median())
    piv = med.pivot_table(index=["query","warm","dataset_rows"],
                          columns="sqlite_variant", values=metric, aggfunc="first")
    # stelle sicher, dass idx/noidx existieren
    for col in ("idx","noidx"):
        if col not in piv.columns:
            piv[col] = np.nan
    piv["best"] = np.nanmin(piv[["idx","noidx"]].values, axis=1)
    piv = piv[["best","idx","noidx"]]
    piv.index = pd.MultiIndex.from_tuples(piv.index, names=["query","warm","dataset_rows"])
    return piv.sort_index()

def ratio_series(base_wide: pd.DataFrame, cap_wide: pd.DataFrame, col: str) -> pd.Series:
    """
    (cap / base) for specified column (e.g., 'auto','t1','best','idx','noidx'),
    aligned by (query,warm,dataset_rows).
    """
    if col not in base_wide.columns or col not in cap_wide.columns:
        return pd.Series(dtype=float)
    idx = base_wide.index.union(cap_wide.index)
    b = base_wide.reindex(idx)[col]
    c = cap_wide.reindex(idx)[col]
    s = (c / b).dropna()
    s.name = "ratio"
    return s


# ---------- Plotters ----------
def plot_ratio_per_query(s: pd.Series, outdir: Path, metric: str, cap_label: str,
                         facet_label: str, col_label: str):
    """
    Ratio (cap/base) vs rows per (query,warm).
    facet_label: 'DuckDB(auto)' o.ä.
    col_label  : Spaltenname in wide DF (auto/t1/best/idx/noidx) nur für Dateinamen.
    """
    written = []
    if s.empty:
        return written
    queries = sorted({ix[0] for ix in s.index})
    warms = [False, True]
    for q in queries:
        for w in warms:
            if (q, w) not in s.index:
                continue
            ss = s.xs((q, w), level=(0,1))
            if ss.empty:
                continue
            rows = ss.index.get_level_values("dataset_rows").values
            vals = ss.values

            subdir = ensure_dir(outdir / "ratio" / metric / ("warm" if w else "cold") / col_label / cap_label)
            fname = subdir / f"{q}__{col_label}__ramcap_{cap_label}_over_base__{metric}.png"

            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            ax.plot(rows, vals, marker="o", lw=2)
            ax.axhline(1.0, ls=":", lw=1.0, color="0.5")
            ax.set_xscale("log")
            ax.set_xlabel("Total rows (log scale)")
            ax.set_ylabel(f"cap ÷ base — {metric}")
            ax.grid(True, which="both", alpha=0.35)
            ax.set_title(f"{q} — {'warm' if w else 'cold'} — {facet_label}")
            fig.tight_layout()
            fig.savefig(fname, dpi=300)
            plt.close(fig)
            written.append(fname)
    return written


def plot_be_shift(base_be_map: dict, cap_be_map: dict, outdir: Path,
                  cap_label: str, metrics=("p50_ms","p95_ms"),
                  sqlite_variant="best", profiles=("auto","t1")):
    """
    Δ break-even rows (cap - base) pro query, warm, metric; EIN-BE-Tabelle (break_even.csv).
    """
    written = []
    for metric in metrics:
        for warm in (False, True):
            data = []
            queries = set()
            for prof in profiles:
                for key, base_val in base_be_map.items():
                    q, w, var, p, m = key
                    if var != sqlite_variant or p != prof or m != metric or w != warm:
                        continue
                    cap_val = cap_be_map.get((q, w, var, p, m), np.nan)
                    delta = np.nan if (pd.isna(base_val) or pd.isna(cap_val)) else (cap_val - base_val)
                    data.append({"query": q, "profile": prof, "delta": delta})
                    queries.add(q)
            if not data:
                continue
            df = pd.DataFrame(data)
            if df.empty:
                continue
            queries = sorted(df["query"].unique())
            pivot = df.pivot(index="query", columns="profile", values="delta").reindex(queries)
            if pivot.dropna(how="all").empty:
                continue

            subdir = ensure_dir(outdir / "be_shift" / metric / ("warm" if warm else "cold") / sqlite_variant / cap_label)
            fname = subdir / f"be_shift__{sqlite_variant}__{cap_label}__{metric}__w{int(warm)}.png"

            fig, ax = plt.subplots(figsize=(7.8, 3.8))
            x = np.arange(len(queries))
            cols = list(pivot.columns)
            width = 0.35 if len(cols) == 2 else 0.5
            offs = np.linspace(-(len(cols)-1)/2, (len(cols)-1)/2, len(cols)) * width
            for o, col in zip(offs, cols):
                y = pivot[col].values
                ax.bar(x + o, y, width*0.9, label=f"DuckDB({col})")
            ax.axhline(0.0, ls=":", lw=1.0, color="0.5")
            ax.set_xticks(x, queries, rotation=30, ha="right")
            ax.set_ylabel("Δ break-even rows (cap − base)")
            ax.set_title(f"Break-even shift vs SQLite({sqlite_variant}) — {metric}, {'warm' if warm else 'cold'} — {cap_label}")
            ax.legend(frameon=True, fontsize=9)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(fname, dpi=300)
            plt.close(fig)
            written.append(fname)
    return written


def _write_presence_log(presence_records: list[dict], outdir: Path, cap_label: str,
                        metric: str, warm: bool, k: int, sqlite_variant: str):
    if not presence_records:
        return None
    subdir = ensure_dir(outdir / "be_shift_multi" / "presence_changes" / f"k{k}" / metric / ("warm" if warm else "cold") / sqlite_variant / cap_label)
    csv = subdir / f"presence_changes__k{k}__{sqlite_variant}__{cap_label}__{metric}__w{int(warm)}.csv"
    pd.DataFrame(presence_records).to_csv(csv, index=False)
    return csv

def plot_be_shift_multi(base_multi_map: dict, cap_multi_map: dict, outdir: Path,
                        cap_label: str, *, metrics=("p50_ms","p95_ms"),
                        sqlite_variant="best", profiles=("auto","t1"),
                        k_max: int = 3, log_presence: bool = True):
    """
    Δ break-even für BE-Index k = 1..k_max (break_even_multi.csv).
    Zusätzlich wird protokolliert, wenn ein k-ter BE im Cap verschwindet/neu erscheint.
    """
    written = []
    for metric in metrics:
        for warm in (False, True):
            for k in range(1, k_max+1):
                rows = []
                presence_records = []
                # Schlüsselmenge bestimmen (Queries, die überhaupt vorkommen)
                keys_base = [(q, w, var, p, m)
                             for (q, w, var, p, m) in base_multi_map.keys()
                             if (w == warm and var == sqlite_variant and m == metric and p in profiles)]
                # Union über base & cap für vollständigen Präsenz-Check
                keys_cap  = [(q, w, var, p, m)
                             for (q, w, var, p, m) in cap_multi_map.keys()
                             if (w == warm and var == sqlite_variant and m == metric and p in profiles)]
                keys = sorted(set(keys_base) | set(keys_cap))

                for (q, w, var, p, m) in keys:
                    base_list = base_multi_map.get((q, w, var, p, m), [])
                    cap_list  = cap_multi_map.get((q, w, var, p, m), [])
                    base_has  = len(base_list) >= k
                    cap_has   = len(cap_list)  >= k

                    # Logging der Präsenz/Abwesenheit
                    presence_records.append({
                        "query": q,
                        "warm": warm,
                        "profile": p,
                        "metric": metric,
                        "k": k,
                        "baseline_has": base_has,
                        "cap_has": cap_has,
                        "baseline_value": (base_list[k-1] if base_has else np.nan),
                        "cap_value": (cap_list[k-1] if cap_has else np.nan),
                        "delta_cap_minus_base": (cap_list[k-1] - base_list[k-1]) if (base_has and cap_has) else np.nan,
                    })

                    # Nur wenn beide vorhanden sind, wird ein Balken gezeichnet
                    if base_has and cap_has:
                        delta = cap_list[k-1] - base_list[k-1]
                        rows.append({"query": q, "profile": p, "delta": delta})

                if log_presence:
                    _write_presence_log(presence_records, outdir, cap_label, metric, warm, k, sqlite_variant)

                if not rows:
                    continue
                df = pd.DataFrame(rows)
                if df.empty:
                    continue
                queries = sorted(df["query"].unique())
                pivot = df.pivot(index="query", columns="profile", values="delta").reindex(queries)
                if pivot.dropna(how="all").empty:
                    continue

                subdir = ensure_dir(outdir / "be_shift_multi" / f"k{k}" / metric / ("warm" if warm else "cold") / sqlite_variant / cap_label)
                fname = subdir / f"be_shift_multi__k{k}__{sqlite_variant}__{cap_label}__{metric}__w{int(warm)}.png"

                fig, ax = plt.subplots(figsize=(7.8, 3.8))
                x = np.arange(len(queries))
                cols = list(pivot.columns)
                width = 0.35 if len(cols) == 2 else 0.5
                offs = np.linspace(-(len(cols)-1)/2, (len(cols)-1)/2, len(cols)) * width
                for o, col in zip(offs, cols):
                    y = pivot[col].values
                    ax.bar(x + o, y, width*0.9, label=f"DuckDB({col})")
                ax.axhline(0.0, ls=":", lw=1.0, color="0.5")
                ax.set_xticks(x, queries, rotation=30, ha="right")
                ax.set_ylabel(f"Δ break-even k={k} (cap − base)")
                ax.set_title(f"Break-even shift (k={k}) vs SQLite({sqlite_variant}) — {metric}, {'warm' if warm else 'cold'} — {cap_label}")
                ax.legend(frameon=True, fontsize=9)
                ax.grid(True, axis="y", alpha=0.25)
                fig.tight_layout()
                fig.savefig(fname, dpi=300)
                plt.close(fig)
                written.append(fname)
    return written


# ---------- ECDF (optional, Rows-Auswahl steuerbar) ----------
def try_load_iterations(campaign_dir: Path) -> pd.DataFrame | None:
    p = campaign_dir / "results" / "iterations_long.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    need = {"query","warm","engine","threads","scan_mode","dur_ms"}
    if not need.issubset(df.columns):
        return None
    return df

def _select_rows(df, rows_target, rpt_target, strategy: str):
    """
    Wählt dataset_rows je nach Strategie:
      - strategy=max|min|nearest
      - rows_target: absolut bevorzugt (falls gesetzt), sonst
      - rpt_target : rows_per_tag (falls Spalten vorhanden), sonst
      - fallback: max
    """
    cand = df.copy()
    if "dataset_rows" not in cand.columns or not cand["dataset_rows"].notna().any():
        return cand  # nichts zu tun
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
    # strategiesteuerung ohne Ziel
    if strategy == "min":
        pick = cand["dataset_rows"].min()
        return cand[cand["dataset_rows"] == pick]
    # default: max oder nearest ohne Ziel -> max
    pick = cand["dataset_rows"].max()
    return cand[cand["dataset_rows"] == pick]

def plot_ecdf_cap_vs_base(it_base: pd.DataFrame, it_cap: pd.DataFrame,
                          outdir: Path, cap_label: str,
                          query: str, warm: bool, engine="duckdb", mode="auto",
                          rows_target: float | None = None,
                          rpt_target: float | None = None,
                          select_strategy: str = "max"):
    """
    ECDF-Vergleich (base vs cap) mit wählbarer Datensatzgröße.
    """
    written = []
    def pick(df):
        d = df[(df["query"]==query) & (df["warm"]==warm) &
               (df["engine"]==engine) & (df["scan_mode"]=="table")].copy()
        if mode == "t1":
            d = d[d["threads"]==1]
        else:
            d = d[d["threads"].isna()]
        d = _select_rows(d, rows_target, rpt_target, select_strategy)
        return d

    b = pick(it_base); c = pick(it_cap)
    if b.empty or c.empty or "dur_ms" not in b.columns or "dur_ms" not in c.columns:
        return written

    def ecdf_vals(arr):
        x = np.sort(np.asarray(arr, dtype=float))
        y = np.arange(1, len(x)+1)/len(x)
        return x, y

    subdir = ensure_dir(outdir / "ecdf" / ("warm" if warm else "cold") / mode / cap_label)
    # Dateiname informiert über Auswahl
    sel_tag = f"rows{int(rows_target)}" if rows_target is not None else (f"rpt{int(rpt_target)}" if rpt_target is not None else select_strategy)
    fname = subdir / f"{query}__{engine}_{mode}__ecdf_cap_vs_base__{cap_label}__{sel_tag}.png"

    xb, yb = ecdf_vals(b["dur_ms"].values)
    xc, yc = ecdf_vals(c["dur_ms"].values)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.step(xb, yb, where="post", label="base", linewidth=2)
    ax.step(xc, yc, where="post", label=f"cap {cap_label}", linewidth=2, linestyle="--")
    ax.set_xlabel("Iteration latency (ms)")
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{query} — {'warm' if warm else 'cold'} — {engine}({mode}) ECDF @ {sel_tag}")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    written.append(fname)
    return written


# ---------- CLI ----------
@dataclass
class Campaign:
    label: str
    root: Path
    summary: Path
    be_single: Path
    be_multi: Path

def parse_caps(raw_caps: list[str]) -> list[Campaign]:
    caps = []
    for s in raw_caps:
        if "=" not in s:
            raise ValueError("Each --cap must be LABEL=DIR")
        label, d = s.split("=", 1)
        root = Path(d)
        caps.append(Campaign(
            label=label.strip(),
            root=root,
            summary=root / "results" / "summary_with_cis.csv",
            be_single=root / "results" / "break_even.csv",
            be_multi=root  / "results" / "break_even_multi.csv",
        ))
    return caps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-campaign", required=True,
                    help="Path to baseline campaign dir (contains results/{summary_with_cis.csv,break_even*.csv})")
    ap.add_argument("--cap", action="append", required=True,
                    help="Repeatable: LABEL=DIR for each RAM-capped campaign (e.g., 512M=/path/to/cap512)")
    ap.add_argument("--outdir", default="plots/ram", help="Output directory for figures")
    ap.add_argument("--metrics", default="p50_ms,p95_ms")
    ap.add_argument("--tags", type=int, default=None)

    # Ratio switches
    ap.add_argument("--skip-ratio", action="store_true")
    ap.add_argument("--ratio-include", default="duckdb", choices=["duckdb","sqlite","both"],
                    help="Which engines to emit ratio plots for (cap/base)")
    ap.add_argument("--ratio-sqlite-variants", default="best",
                    help="Comma list among best,idx,noidx (only used if sqlite or both)")

    # BE shift
    ap.add_argument("--skip-be-shift", action="store_true")
    ap.add_argument("--be-source", default="both", choices=["single","multi","both"],
                    help="Which BE tables to use for shift plots")
    ap.add_argument("--be-kmax", type=int, default=3, help="Max BE index k to plot for multi")

    # ECDF
    ap.add_argument("--ecdf-queries", default="")
    ap.add_argument("--ecdf-rows", type=float, default=None, help="Target TOTAL rows for ECDF (nearest match)")
    ap.add_argument("--ecdf-rows-per-tag", type=float, default=None, help="Target rows-per-tag for ECDF (nearest match)")
    ap.add_argument("--ecdf-select", default="max", choices=["max","min","nearest"],
                    help="Selection strategy if no explicit target given")

    args = ap.parse_args()
    apply_style()

    base_root = Path(args.baseline_campaign)
    outdir   = ensure_dir(Path(args.outdir))

    # Load baseline
    base_sum = base_root / "results" / "summary_with_cis.csv"
    base_single = base_root / "results" / "break_even.csv"
    base_multi  = base_root / "results" / "break_even_multi.csv"

    base_df = load_summary_csv(base_sum)
    if args.tags is not None and "dataset_tags" in base_df.columns:
        base_df = base_df[base_df["dataset_tags"] == args.tags]

    base_be_df   = load_be_csv(base_single, tags=args.tags)
    base_be_map  = be_map_from_df(base_be_df)
    base_bem_df  = load_be_multi_csv(base_multi, tags=args.tags)
    base_bem_map = be_multi_map_from_df(base_bem_df)

    # Load caps
    caps = parse_caps(args.cap)

    written_total = 0

    for cap in caps:
        if not cap.summary.exists():
            print(f"[warn] {cap.label}: missing {cap.summary}")
            continue
        cap_df = load_summary_csv(cap.summary)
        if args.tags is not None and "dataset_tags" in cap_df.columns:
            cap_df = cap_df[cap_df["dataset_tags"] == args.tags]

        # R1: ratio (cap/base)
        if not args.skip_ratio:
            for metric in [m.strip() for m in args.metrics.split(",") if m.strip()]:
                if args.ratio_include in ("duckdb","both"):
                    base_wide = duck_pivot_table(base_df, metric)
                    cap_wide  = duck_pivot_table(cap_df,  metric)
                    if base_wide.empty or cap_wide.empty:
                        print(f"[i] {cap.label}: skip DuckDB ratio {metric} (no duckdb table data)")
                    else:
                        for mode in ("auto","t1"):
                            s = ratio_series(base_wide, cap_wide, mode)
                            written = plot_ratio_per_query(s, outdir, metric, cap.label,
                                                           facet_label=f"DuckDB({mode})", col_label=mode)
                            written_total += len(written)

                if args.ratio_include in ("sqlite","both"):
                    base_sw = sqlite_pivot_table(base_df, metric)
                    cap_sw  = sqlite_pivot_table(cap_df,  metric)
                    if base_sw.empty or cap_sw.empty:
                        print(f"[i] {cap.label}: skip SQLite ratio {metric} (no sqlite table data)")
                    else:
                        variants = [v.strip() for v in args.ratio_sqlite_variants.split(",") if v.strip()]
                        for var in variants:
                            if var not in base_sw.columns or var not in cap_sw.columns:
                                continue
                            s = ratio_series(base_sw, cap_sw, var)
                            written = plot_ratio_per_query(s, outdir, metric, cap.label,
                                                           facet_label=f"SQLite({var})", col_label=f"sqlite_{var}")
                            written_total += len(written)

        # R2: BE shift (single +/or multi)
        if not args.skip_be_shift:
            # single
            if args.be_source in ("single","both"):
                cap_be_df  = load_be_csv(cap.be_single, tags=args.tags)
                cap_be_map = be_map_from_df(cap_be_df)
                if base_be_map and cap_be_map:
                    written = plot_be_shift(base_be_map, cap_be_map, outdir, cap.label,
                                            metrics=tuple(m.strip() for m in args.metrics.split(",") if m.strip()),
                                            sqlite_variant="best", profiles=("auto","t1"))
                    written_total += len(written)
                else:
                    print(f"[i] {cap.label}: skip single-BE shift (missing break_even.csv in base or cap)")

            # multi
            if args.be_source in ("multi","both"):
                cap_bem_df  = load_be_multi_csv(cap.be_multi, tags=args.tags)
                cap_bem_map = be_multi_map_from_df(cap_bem_df)
                if base_bem_map and cap_bem_map:
                    written = plot_be_shift_multi(base_bem_map, cap_bem_map, outdir, cap.label,
                                                  metrics=tuple(m.strip() for m in args.metrics.split(",") if m.strip()),
                                                  sqlite_variant="best", profiles=("auto","t1"),
                                                  k_max=int(args.be_kmax), log_presence=True)
                    written_total += len(written)
                else:
                    print(f"[i] {cap.label}: skip multi-BE shift (missing break_even_multi.csv in base or cap)")

        # Optional ECDFs
        ecdf_qs = [q.strip() for q in args.ecdf_queries.split(",") if q.strip()]
        if ecdf_qs:
            it_base = try_load_iterations(base_root)
            it_cap  = try_load_iterations(cap.root)
            if it_base is None or it_cap is None:
                print(f"[i] {cap.label}: skip ECDF (iterations_long.csv not found)")
            else:
                for q in ecdf_qs:
                    for warm in (False, True):
                        for mode in ("auto","t1"):
                            written = plot_ecdf_cap_vs_base(
                                it_base, it_cap, outdir, cap.label, q, warm,
                                engine="duckdb", mode=mode,
                                rows_target=args.ecdf_rows,
                                rpt_target=args.ecdf_rows_per_tag,
                                select_strategy=args.ecdf_select
                            )
                            written_total += len(written)

    print(f"[ok] wrote ~{written_total} RAM-cap figures → {outdir}")

if __name__ == "__main__":
    main()
