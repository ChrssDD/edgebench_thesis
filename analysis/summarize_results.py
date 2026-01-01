# analysis/summarize_results.py
import os, json
from pathlib import Path
from config import RESULTS_DIR
import pandas as pd
import numpy as np
import re

RUN_GLOBS = ["*_tbl.json", "*_fs.json"]


def _psi_total(text: str | None, kind: str) -> int | None:
    """
    Extract the cumulative microseconds after 'total=' for a given PSI line kind ('some'|'full').
    Matches start-of-line 'kind' to avoid picking up the wrong section.
    """
    if not text or not isinstance(text, str):
        return None
    m = re.search(rf"(?m)^{kind}\s+.*?\btotal=(\d+)", text)
    return int(m.group(1)) if m else None


def _min_rows_per_tag_needed(query: str) -> int | None:
    # assumes 1 Hz data
    if query == "last_60min_avg":
        return 3600
    if query == "last_15min_avg":
        return 900
    return None


def bootstrap_ci(values, stat_fn, iters=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(42)
    vals = np.array(list(values), dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    boots = []
    n = len(vals)
    for _ in range(iters):
        sample = vals[rng.integers(0, n, size=n)]
        boots.append(stat_fn(sample))
    low = np.quantile(boots, alpha/2)
    high = np.quantile(boots, 1 - alpha/2)
    return float(low), float(high)


def q_disc(sample, q):
    s = np.sort(np.asarray(sample, dtype=float))
    if s.size == 0:
        return float("nan")
    k = int(np.ceil(len(s) * (q/100.0)))
    k = min(max(k, 1), len(s)) - 1
    return float(s[k])


def _sqlite_variant_from_db(db_path: str) -> str | None:
    if not isinstance(db_path, str):
        return None
    stem = Path(db_path).stem
    if stem.endswith("_idx") or "_idx" in stem:
        return "idx"
    if stem.endswith("_noidx") or "_noidx" in stem:
        return "noidx"
    return "unknown"


def is_run_payload(obj: dict) -> bool:
    return (
        isinstance(obj, dict)
        and obj.get("engine") in {"duckdb", "sqlite"}
        and isinstance(obj.get("durations_ns"), list)
        and len(obj["durations_ns"]) > 0
        and isinstance(obj.get("stats"), dict)
    )


def load_results():
    run_rows = []
    iter_rows = []
    ignored = []
    files = []
    for pat in RUN_GLOBS:
        files.extend(Path(RESULTS_DIR).glob(pat))
    for p in files:
        try:
            obj = json.loads(p.read_text())
        except Exception:
            ignored.append({"file": p.name, "reason": "json-parse-error"})
            continue
        if not is_run_payload(obj):
            ignored.append({"file": p.name, "reason": "not-a-run-schema"})
            continue

        # --- Latenz (bestehend) ---
        durs_ms = [float(x) / 1e6 for x in obj["durations_ns"]]
        s = sorted(durs_ms)
        n = len(s)

        # --- NEU: PSI je Iteration parsen & differenzieren (µs) ---
        io_diffs_us = []           # für Aggregation
        cpu_diffs_us = []          # für Aggregation
        io_pairs_lat_ms = []       # (lat_ms, io_diff_us) für Spearman
        cpu_pairs_lat_ms = []      # (lat_ms, cpu_diff_us) für Spearman
        rss_vals = []
        cpu_pcts = []

        for it_idx, it in enumerate(obj.get("iterations", [])):
            dur_ns = it.get("dur_ns")
            lat_ms = float(dur_ns) / 1e6 if isinstance(dur_ns, (int, float)) else None

            psi_b = it.get("psi_before") or {}
            psi_a = it.get("psi_after") or {}

            # I/O full
            io_b = _psi_total(psi_b.get("io"), "full")
            io_a = _psi_total(psi_a.get("io"), "full")
            if io_b is not None and io_a is not None:
                diff_us = max(0, io_a - io_b)
                io_diffs_us.append(diff_us)
                if lat_ms is not None:
                    io_pairs_lat_ms.append((lat_ms, diff_us))
            io_diff_ms_this = (max(0, io_a - io_b) / 1000.0) if (io_b is not None and io_a is not None) else float("nan")

            # CPU some
            cpu_b = _psi_total(psi_b.get("cpu"), "some")
            cpu_a = _psi_total(psi_a.get("cpu"), "some")
            if cpu_b is not None and cpu_a is not None:
                diff_us = max(0, cpu_a - cpu_b)
                cpu_diffs_us.append(diff_us)
                if lat_ms is not None:
                    cpu_pairs_lat_ms.append((lat_ms, diff_us))
            cpu_diff_ms_this = (max(0, cpu_a - cpu_b) / 1000.0) if (cpu_b is not None and cpu_a is not None) else float("nan")

            # Kontext
            rss = it.get("rss_bytes")
            if isinstance(rss, (int, float)):
                rss_vals.append(int(rss))
            cpu_pct = it.get("cpu_percent")
            if isinstance(cpu_pct, (int, float)):
                cpu_pcts.append(float(cpu_pct))

        # --- NEU: Aggregation (ms) mit diskretem p95 wie bei Latenz ---
        def to_ms_mean(us_list):
            return float(np.mean(us_list) / 1000.0) if us_list else float("nan")
        def to_ms_p95(us_list):
            return float(q_disc(us_list, 95) / 1000.0) if us_list else float("nan")

        mean_io_ms = to_ms_mean(io_diffs_us)
        p95_io_ms  = to_ms_p95(io_diffs_us)
        mean_cpu_ms = to_ms_mean(cpu_diffs_us)
        p95_cpu_ms  = to_ms_p95(cpu_diffs_us)

        # --- NEU: Spearman-Korrelationen (iterationsweise) ---
        def spearman_from_pairs(pairs):
            if len(pairs) < 3:
                return float("nan")
            lat, psi = zip(*pairs)
            try:
                return float(pd.Series(lat).corr(pd.Series(psi), method="spearman"))
            except Exception:
                return float("nan")

        rho_io  = spearman_from_pairs(io_pairs_lat_ms)
        rho_cpu = spearman_from_pairs(cpu_pairs_lat_ms)

        max_rss = max(rss_vals) if rss_vals else None
        mean_cpu_pct = float(np.mean(cpu_pcts)) if cpu_pcts else float("nan")

        # --- Row schreiben (bestehende + neue Felder) ---
        run_rows.append(
            {
                "file": p.name,
                "engine": obj.get("engine"),
                "query": obj.get("query"),
                "warm": obj.get("warm"),
                "threads": obj.get("threads"),
                "db": obj.get("db"),
                "sqlite_variant": (
                    _sqlite_variant_from_db(obj.get("db"))
                    if obj.get("engine") == "sqlite"
                    else None
                ),
                "scan_mode": obj.get("scan_mode"),

                # Latenz: DISCRETE quantiles (Type-1)
                "p50_ms": float(s[min(max(int(np.ceil(n * 0.50)), 1), n) - 1]),
                "p95_ms": float(s[min(max(int(np.ceil(n * 0.95)), 1), n) - 1]),
                "p99_ms": float(s[min(max(int(np.ceil(n * 0.99)), 1), n) - 1]),
                "mean_ms": float(np.mean(durs_ms)),
                "n": len(durs_ms),

                # --- NEU: PSI-Aggregate & Korrelationen ---
                "mean_psi_io_full_ms": mean_io_ms,
                "p95_psi_io_full_ms":  p95_io_ms,
                "mean_psi_cpu_some_ms": mean_cpu_ms,
                "p95_psi_cpu_some_ms":  p95_cpu_ms,
                "spearman_io_vs_latency":  rho_io,
                "spearman_cpu_vs_latency": rho_cpu,
                "psi_pairs_n_io":  len(io_pairs_lat_ms),
                "psi_pairs_n_cpu": len(cpu_pairs_lat_ms),

                # --- NEU: Ressourcen-Kontext ---
                "max_rss_bytes": max_rss,
                "mean_cpu_percent": mean_cpu_pct,

                # Dataset-Meta (bestehend)
                "dataset_rows": (obj.get("dataset_meta") or {}).get("rows"),
                "dataset_tags": (obj.get("dataset_meta") or {}).get("tags"),
            }
        )
        # --- Iterations-Langformat (eine Zeile je Iteration) ---
        ds_rows = (obj.get("dataset_meta") or {}).get("rows")
        ds_tags = (obj.get("dataset_meta") or {}).get("tags")
        for it_idx, it in enumerate(obj.get("iterations", [])):
            dur_ns = it.get("dur_ns")
            lat_ms = float(dur_ns) / 1e6 if isinstance(dur_ns, (int, float)) else float("nan")
            # PSI again per-iteration (robust fallback)
            def _psi_total_safe(blob, kind):
                if not isinstance(blob, dict):
                    return None
                return _psi_total(blob.get("io" if kind=="io" else "cpu"), "full" if kind=="io" else "some")
            io_b = _psi_total((it.get("psi_before") or {}).get("io"), "full")
            io_a = _psi_total((it.get("psi_after")  or {}).get("io"), "full")
            cpu_b = _psi_total((it.get("psi_before") or {}).get("cpu"), "some")
            cpu_a = _psi_total((it.get("psi_after")  or {}).get("cpu"), "some")
            io_ms  = (max(0, io_a - io_b) / 1000.0) if (isinstance(io_b, int) and isinstance(io_a, int)) else float("nan")
            cpu_ms = (max(0, cpu_a - cpu_b) / 1000.0) if (isinstance(cpu_b, int) and isinstance(cpu_a, int)) else float("nan")
            iter_rows.append({
                "file": p.name,
                "engine": obj.get("engine"),
                "query": obj.get("query"),
                "warm": obj.get("warm"),
                "threads": obj.get("threads"),
                "scan_mode": obj.get("scan_mode"),
                "sqlite_variant": _sqlite_variant_from_db(obj.get("db")) if obj.get("engine") == "sqlite" else None,
                "dataset_rows": ds_rows,
                "dataset_tags": ds_tags,
                "iter_index": it_idx,
                "dur_ms": lat_ms,
                "psi_io_full_diff_ms": io_ms,
                "psi_cpu_some_diff_ms": cpu_ms,
                "rss_bytes": it.get("rss_bytes"),
                "cpu_percent": it.get("cpu_percent"),
            })
    if ignored:
        (Path(RESULTS_DIR) / "ignored_meta.json").write_text(
            json.dumps(ignored, indent=2)
        )
        print(
            f"Ignored non-run jsons: {len(ignored)} → {Path(RESULTS_DIR) / 'ignored_meta.json'}"
        )
    return pd.DataFrame(run_rows), pd.DataFrame(iter_rows)



def break_even(
    df: pd.DataFrame,
    query: str,
    warm: bool,
    metric: str = "p50_ms",
    sqlite_choice: str = "best",  # {"best","idx","noidx"}
    duckdb_profile: str = "auto",  # {"auto","t1"}
    tags: int | None = None,
    min_rows_per_tag_for_window: int | None = None,
) -> float | None:
    """
    Compute break-even (TOTAL rows) between DuckDB(profile) and SQLite(variant),
    using linear interpolation on adjacent sizes. Returns None if no crossing.
    Aggregation je Größe: MEDIAN (konsistent mit Plots).
    Für sqlite_choice="best": punktweise min(idx_median, noidx_median).
    """
    base = df[
        (df["query"] == query)
        & (df["warm"] == warm)
        & (df["scan_mode"] == "table")
        & df["dataset_rows"].notna()
    ].copy()

    if tags is not None:
        base = base[base["dataset_tags"] == tags]

    if min_rows_per_tag_for_window is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rpt = base["dataset_rows"] / base["dataset_tags"]
        base = base.loc[rpt >= min_rows_per_tag_for_window]

    if base.empty:
        return None

    # DuckDB slice (Profil)
    if duckdb_profile == "t1":
        duck = base[(base["engine"] == "duckdb") & (base["threads"] == 1)]
    else:
        duck = base[(base["engine"] == "duckdb") & (base["threads"].isna())]
    if duck.empty:
        return None

    # SQLite slice
    sqli = base[base["engine"] == "sqlite"].copy()
    if sqli.empty:
        return None

    # --- Aggregation je Größe (Median) ---
    duck_g = duck.groupby("dataset_rows", as_index=False)[metric].median()
    duck_g["engine"] = "duckdb"

    if sqlite_choice in ("idx", "noidx"):
        sql_g = (
            sqli[sqli["sqlite_variant"] == sqlite_choice]
            .groupby("dataset_rows", as_index=False)[metric]
            .median()
        )
        sql_g["engine"] = "sqlite"
    else:
        # sqlite_choice == "best": pro Größe die bessere Variante (idx vs noidx)
        tmp = (
            sqli.groupby(["sqlite_variant", "dataset_rows"], as_index=False)[metric]
            .median()
        )
        pivv = tmp.pivot(
            index="dataset_rows", columns="sqlite_variant", values=metric
        )
        best_series = pivv.min(axis=1)  # punktweise min über Varianten
        sql_g = (
            best_series.reset_index()
            .rename(columns={0: metric, best_series.name: metric})
            .copy()
        )
        sql_g["engine"] = "sqlite"

    # gemeinsame Größen behalten
    both_rows = np.intersect1d(duck_g["dataset_rows"].values, sql_g["dataset_rows"].values)
    if both_rows.size < 2:
        return None
    duck_g = duck_g[duck_g["dataset_rows"].isin(both_rows)].copy()
    sql_g = sql_g[sql_g["dataset_rows"].isin(both_rows)].copy()

    # Zusammenführen und pivotieren: Spalten = {'duckdb','sqlite'}
    sub = pd.concat([duck_g, sql_g], ignore_index=True)
    piv = sub.pivot(index="dataset_rows", columns="engine", values=metric).dropna()
    if set(piv.columns) != {"duckdb", "sqlite"} or len(piv) < 2:
        return None

    piv = piv.sort_index()
    prev_r = prev_duck = prev_sql = None
    for r, row in piv.iterrows():
        if prev_r is not None:
            d0 = prev_sql - prev_duck
            d1 = row["sqlite"] - row["duckdb"]
            if d0 == 0:
                return float(prev_r)
            if d0 * d1 < 0:
                denom = (d1 - d0)
                if denom != 0:
                    t = -d0 / denom
                    if 0.0 <= t <= 1.0:
                        return float(prev_r + t * (r - prev_r))
        prev_r, prev_duck, prev_sql = r, row["duckdb"], row["sqlite"]
    return None  # no crossing in measured range


# --- helpers for multi break-even & dominance zones --------------------------
# Hinweis: Aggregation je Größe mit .median() (konsistent zu den Plots).

def _slice_curves(
    df: pd.DataFrame,
    *,
    query: str,
    warm: bool,
    metric: str,
    tags: int | None,
    duckdb_profile: str,
    min_rows_per_tag_for_window: int | None,
):
    base = df[
        (df["query"] == query)
        & (df["warm"] == warm)
        & (df["scan_mode"] == "table")
        & df["dataset_rows"].notna()
    ].copy()

    if tags is not None:
        base = base[base["dataset_tags"] == tags]

    if min_rows_per_tag_for_window is not None and "dataset_tags" in base.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            rpt = base["dataset_rows"] / base["dataset_tags"]
        base = base.loc[rpt >= min_rows_per_tag_for_window]

    if base.empty:
        return None, None, None

    # DuckDB series
    if duckdb_profile == "t1":
        duck = base[(base["engine"] == "duckdb") & (base["threads"] == 1)]
    else:
        duck = base[(base["engine"] == "duckdb") & (base["threads"].isna())]
    if duck.empty:
        return None, None, None

    # SQLite variants
    sqli = base[base["engine"] == "sqlite"].copy()
    idx = sqli[sqli["sqlite_variant"] == "idx"]
    noid = sqli[sqli["sqlite_variant"] == "noidx"]
    if idx.empty and noid.empty:
        return None, None, None

    # Median je Größe
    d_series = duck.groupby("dataset_rows")[metric].median()
    i_series = idx.groupby("dataset_rows")[metric].median() if not idx.empty else pd.Series(dtype=float)
    n_series = noid.groupby("dataset_rows")[metric].median() if not noid.empty else pd.Series(dtype=float)

    # gemeinsames X-Gitter
    x = sorted(set(d_series.index) | set(i_series.index) | set(n_series.index))
    d = pd.Series([d_series.get(r, np.nan) for r in x], index=x)
    i = pd.Series([i_series.get(r, np.nan) for r in x], index=x) if len(i_series) else None
    n = pd.Series([n_series.get(r, np.nan) for r in x], index=x) if len(n_series) else None

    # "best" = punktweises Minimum aus idx/noidx; Variante merken
    if i is not None and n is not None and len(i) and len(n):
        b_vals = np.fmin(i.values, n.values)
        b = pd.Series(b_vals, index=x)
        b_variant = pd.Series(np.where(np.less_equal(i.values, n.values), "idx", "noidx"), index=x)
    elif i is not None and len(i):
        b, b_variant = i.copy(), pd.Series(["idx"] * len(i), index=i.index)
    else:
        b, b_variant = n.copy(), pd.Series(["noidx"] * len(n), index=n.index)

    return d, {"idx": i, "noidx": n, "best": b, "best_variant": b_variant}, x


def _find_all_crossings(sql_s: pd.Series, duck_s: pd.Series, xs: list[float]) -> list[float]:
    """Alle Schnittpunkte (linear zwischen Messstufen) auf gemeinsamer X-Achse xs."""
    be = []
    prev_r = None
    prev_diff = None
    for r in xs:
        sd = duck_s.get(r, np.nan)
        ss = sql_s.get(r, np.nan)
        if not (np.isfinite(sd) and np.isfinite(ss)):
            continue
        diff = ss - sd  # >0: DuckDB schneller; <0: SQLite schneller
        if prev_r is not None and np.isfinite(prev_diff):
            if prev_diff == 0:
                be.append(float(prev_r))
            elif prev_diff * diff < 0:
                denom = (diff - prev_diff)
                if denom != 0:
                    t = -prev_diff / denom
                    if 0.0 <= t <= 1.0:
                        be_x = prev_r + t * (r - prev_r)
                        be.append(float(be_x))
        prev_r, prev_diff = r, diff
    return be


def _filter_crossings(be_list: list[float], *, eps_log10: float = 0.04) -> list[float]:
    """
    Cluster nahe BE-Punkte auf log10(X) und gib repräsentative Mitte je Cluster zurück.
    eps_log10 ~ 0.04 ≈ 9% in log-x.
    """
    if not be_list:
        return []
    arr = np.array(sorted(be_list), dtype=float)
    logs = np.log10(arr)
    kept = []
    cur = [arr[0]]
    cur_log_ref = logs[0]
    for x, lx in zip(arr[1:], logs[1:]):
        if lx - cur_log_ref <= eps_log10:
            cur.append(x)
        else:
            kept.append(float(np.median(cur)))
            cur = [x]
            cur_log_ref = lx
    kept.append(float(np.median(cur)))
    return kept


def _zones_from_crossings(
    kept_xs: list[float],
    duck: pd.Series,
    sql: pd.Series,
    *,
    min_rel_delta: float = 0.10,
    min_points: int = 2,
    best_variant: pd.Series | None = None,
    min_abs_delta_ms: float = 0.0,
):
    """
    Erzeuge Dominanzzonen zwischen BE-Punkten.
    Behalte nur Zonen, die „signifikant“ sind:
      - mittlere relative Abweichung >= min_rel_delta  ODER
      - |Median-Delta| >= min_abs_delta_ms
    Winner: 'duckdb' (diff>0) oder 'sqlite' (diff<0).
    Für "best" kann best_variant (pro X) die gewinnende SQLite-Variante liefern.
    """
    grid = sorted(set(duck.index) | set(sql.index))
    if not grid:
        return []

    cuts = [min(grid)] + list(kept_xs) + [max(grid)]
    zones = []
    for zid, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        pts = [r for r in grid if lo <= r <= hi]
        if len(pts) < min_points:
            continue

        diffs = []
        rels = []
        s_wins = []
        s_vars = []
        for r in pts:
            sd = duck.get(r, np.nan)
            ss = sql.get(r, np.nan)
            if not (np.isfinite(sd) and np.isfinite(ss)):
                continue
            dif = ss - sd  # >0: DuckDB schneller
            diffs.append(dif)
            denom = max(sd, ss, 1e-9)
            rels.append(dif / denom)
            s_wins.append("duckdb" if dif > 0 else "sqlite")
            if best_variant is not None and dif < 0:  # sqlite gewinnt
                s_vars.append(best_variant.get(r, "unknown"))

        if not diffs:
            continue

        med_delta = float(np.median(diffs))
        mean_rel = float(np.mean(np.abs(rels)))

        if (mean_rel < min_rel_delta) and (abs(med_delta) < min_abs_delta_ms):
            # instabile Mikro-Zone → verwerfen
            continue

        # Sieger bestimmen (Mehrheit), Fallback: Vorzeichen des Median-Deltas
        if s_wins:
            winner = max(set(s_wins), key=s_wins.count)
        else:
            winner = "duckdb" if med_delta > 0 else "sqlite" if med_delta < 0 else "tie"

        winner_variant = None
        if best_variant is not None and winner == "sqlite" and s_vars:
            winner_variant = max(set(s_vars), key=s_vars.count)

        zones.append(
            {
                "zone_id": zid,
                "start_rows": float(lo),
                "end_rows": float(hi),
                "x_len_rows": float(hi - lo),
                "x_len_log10": float(
                    np.log10(max(hi, 1.0)) - np.log10(max(lo, 1.0))
                ),
                "winner_engine": winner,
                "winner_variant": winner_variant,
                "median_delta_ms": med_delta,  # >0: DuckDB schneller
                "mean_rel_delta": mean_rel,
                "max_rel_delta": float(np.max(np.abs(rels))),
                "n_points": int(len(pts)),
            }
        )
    return zones


def _multi_be_for(
    df: pd.DataFrame,
    query: str,
    warm: bool,
    metric: str,
    tags: int,
    *,
    duckdb_profile: str,
    sqlite_choice: str,
    min_rows_per_tag_for_window: int | None,
    eps_log10: float = 0.04,
    min_rel_delta: float = 0.10,
    min_abs_delta_ms: float = 0.0,
):
    """Liefert (be_points, zones) für gegebene Wahl (DuckDB-Profil, SQLite-Variante/best)."""
    duck, sdict, x = _slice_curves(
        df,
        query=query,
        warm=warm,
        metric=metric,
        tags=tags,
        duckdb_profile=duckdb_profile,
        min_rows_per_tag_for_window=min_rows_per_tag_for_window,
    )
    if duck is None:
        return [], []

    if sqlite_choice == "idx":
        sql = sdict["idx"]
        best_var = None
    elif sqlite_choice == "noidx":
        sql = sdict["noidx"]
        best_var = None
    else:  # "best"
        sql = sdict["best"]
        best_var = sdict["best_variant"]

    if sql is None or sql.empty:
        return [], []

    # gemeinsame X nur, wo beide Werte haben
    xs = [
        r
        for r in x
        if (r in duck.index)
        and (r in sql.index)
        and np.isfinite(duck.get(r))
        and np.isfinite(sql.get(r))
    ]
    if len(xs) < 2:
        return [], []

    # alle (rohen) Schnittpunkte + Filter
    raw = _find_all_crossings(sql, duck, xs)
    kept = _filter_crossings(raw, eps_log10=eps_log10)
    zones = _zones_from_crossings(
        kept,
        duck,
        sql,
        min_rel_delta=min_rel_delta,
        min_points=2,
        best_variant=best_var,
        min_abs_delta_ms=min_abs_delta_ms,
    )
    return kept, zones


def write_break_even_multi(df_runs: pd.DataFrame, out_csv: Path) -> Path:
    """
    Erzeugt break_even_multi.csv:
      - alle signifikanten BE-Punkte (geclustert)
      - Dominanzzonen mit Deltas/Verhältnissen
      - für sqlite_choice in {"idx","noidx","best"} und duckdb_profile {"auto","t1"}
    """
    be_multi_rows = []
    all_tags = sorted(x for x in df_runs["dataset_tags"].dropna().unique())
    for q in sorted(df_runs["query"].dropna().unique()):
        min_rpt = _min_rows_per_tag_needed(q)
        for warm in (False, True):
            for tg in all_tags:
                for prof in ("auto", "t1"):
                    for metric in ("p50_ms", "p95_ms"):
                        for sqli in ("idx", "noidx", "best"):
                            be_list, zones = _multi_be_for(
                                df_runs,
                                q,
                                warm,
                                metric,
                                int(tg),
                                duckdb_profile=prof,
                                sqlite_choice=sqli,
                                min_rows_per_tag_for_window=min_rpt,
                                eps_log10=0.04,       # Clustering-Schwelle (log10)
                                min_rel_delta=0.10,   # 10% mittlere relative Abweichung
                                min_abs_delta_ms=0.0  # optional: z. B. 5.0
                            )
                            be_multi_rows.append(
                                {
                                    "query": q,
                                    "warm": warm,
                                    "tags": int(tg),
                                    "metric": metric,
                                    "sqlite_variant": sqli,
                                    "duckdb_profile": prof,
                                    "be_points": ";".join(
                                        f"{x:.6f}" for x in be_list
                                    ),
                                    "be_count": len(be_list),
                                    "zones_json": json.dumps(
                                        zones, separators=(",", ":")
                                    ),
                                }
                            )
    pd.DataFrame(be_multi_rows).to_csv(out_csv, index=False)
    print("Geschrieben:", out_csv)
    return out_csv


def main():
    df_runs, it_long = load_results()
    if df_runs.empty:
        print("Keine Run-Dateien gefunden, die dem Schema entsprechen (durations_ns etc.).")
        return
    if it_long.empty:
        print("Keine Long-Dateien gefunden, die dem Schema entsprechen (durations_ns etc.).")
        return

    # --- Bootstrap CIs (discrete quantiles) ---
    cis = []
    for fname in df_runs["file"]:
        obj = json.loads((Path(RESULTS_DIR) / fname).read_text())
        durs_ms = np.array([float(x) / 1e6 for x in obj["durations_ns"]], dtype=float)
        for metric, q in [("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99)]:
            low, high = bootstrap_ci(durs_ms, lambda s, qq=q: q_disc(s, qq))
            cis.append({"file": fname, "metric": metric, "ci_low": low, "ci_high": high})
    ci_df = pd.DataFrame(cis)
    if not ci_df.empty:
        ci_wide = ci_df.pivot(index="file", columns="metric", values=["ci_low", "ci_high"])
        ci_wide.columns = ["{}_{}".format(a, b) for a, b in ci_wide.columns.to_flat_index()]
        ci_wide = ci_wide.reset_index()
        summary = df_runs.merge(ci_wide, on="file", how="left")
    else:
        summary = df_runs.copy()

    # --- CI gating (keep CI only when n ≥ 10) ---
    if not summary.empty:
        summary["ci_ok"] = summary["n"] >= 10
        ci_cols = [c for c in summary.columns if c.startswith("ci_low_") or c.startswith("ci_high_")]
        if ci_cols:
            mask = summary["ci_ok"]
            for c in ci_cols:
                summary.loc[~mask, c] = np.nan

        # convenience for sanity checks (TOTAL / TAGS)
        with np.errstate(divide="ignore", invalid="ignore"):
            summary["rows_per_tag"] = summary["dataset_rows"] / summary["dataset_tags"]

    # Persist summary
    (Path(RESULTS_DIR) / "summary_with_cis.csv").write_text(summary.to_csv(index=False))
    if it_long is not None and not it_long.empty:
        it_path = Path(RESULTS_DIR) / "iterations_long.csv"
        it_long.to_csv(it_path, index=False)
        print("Geschrieben:", it_path)

    # --- Break-even table (per tags; both DuckDB profiles; emit p50 & p95) ---
    be_rows = []
    all_tags = sorted(x for x in df_runs["dataset_tags"].dropna().unique())  # für beide Loops
    for q in sorted(df_runs["query"].dropna().unique()):
        min_rpt = _min_rows_per_tag_needed(q)  # 3600 for last_60min, 900 for last_15min, else None
        for warm in (False, True):
            for tg in all_tags:
                for prof in ("auto", "t1"):
                    for metric in ("p50_ms", "p95_ms"):
                        be_idx = break_even(
                            df_runs, q, warm, metric=metric,
                            sqlite_choice="idx", duckdb_profile=prof,
                            tags=int(tg), min_rows_per_tag_for_window=min_rpt
                        )
                        be_noidx = break_even(
                            df_runs, q, warm, metric=metric,
                            sqlite_choice="noidx", duckdb_profile=prof,
                            tags=int(tg), min_rows_per_tag_for_window=min_rpt
                        )
                        # echtes "best": per punktweiser Best-of-Kurve
                        be_best = break_even(
                            df_runs, q, warm, metric=metric,
                            sqlite_choice="best", duckdb_profile=prof,
                            tags=int(tg), min_rows_per_tag_for_window=min_rpt
                        )

                        be_rows.extend([
                            {"query": q, "warm": warm, "tags": int(tg), "metric": metric,
                             "sqlite_variant": "idx", "duckdb_profile": prof, "break_even_rows": be_idx},
                            {"query": q, "warm": warm, "tags": int(tg), "metric": metric,
                             "sqlite_variant": "noidx", "duckdb_profile": prof, "break_even_rows": be_noidx},
                            {"query": q, "warm": warm, "tags": int(tg), "metric": metric,
                             "sqlite_variant": "best", "duckdb_profile": prof, "break_even_rows": be_best},
                        ])
    pd.DataFrame(be_rows).to_csv(Path(RESULTS_DIR) / "break_even.csv", index=False)

    # Compact overview remains
    agg = (
        df_runs.groupby(["engine", "query", "warm", "scan_mode"], dropna=False)["p50_ms"]
        .median()
        .reset_index()
    )
    (Path(RESULTS_DIR) / "run_summary.json").write_text(
        agg.to_json(orient="records", indent=2)
    )

    # --- NEU: Break-even-MULTI (alle Schnitte + Zonen) -----------------------
    write_break_even_multi(df_runs, Path(RESULTS_DIR) / "break_even_multi.csv")

    print(
        "Geschrieben:",
        Path(RESULTS_DIR) / "summary_with_cis.csv",
        Path(RESULTS_DIR) / "break_even.csv",
        Path(RESULTS_DIR) / "break_even_multi.csv",
    )


if __name__ == "__main__":
    main()
