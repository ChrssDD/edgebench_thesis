# analysis/images/utils.py
from __future__ import annotations
import math
import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

# ---- labels & mapping --------------------------------------------------------
def engine_label(row: pd.Series) -> str:
    if row["engine"] == "sqlite":
        var = str(row.get("sqlite_variant", ""))
        if var == "idx":   return "SQLite (idx)"
        if var == "noidx": return "SQLite (noidx)"
        return "SQLite"
    if row["engine"] == "duckdb":
        thr = row.get("threads", np.nan)
        if pd.notna(thr) and int(thr) == 1:
            return "DuckDB (t1)"
        return "DuckDB (auto)"
    return "?"

ORDER = ["SQLite (noidx)", "SQLite (idx)", "SQLite", "DuckDB (t1)", "DuckDB (auto)"]
MARKERS = {
    "SQLite (noidx)": "X",
    "SQLite (idx)": "s",
    "SQLite": "o",
    "DuckDB (t1)": "^",
    "DuckDB (auto)": "o",
}

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---- colors for BE annotations ----------------------------------------------
BE_COLORS = {"auto": "#D55E00", "t1": "#0072B2", "duck": "#4C72B0", "sql": "#DD8452"}

# ---- primary break-even (load & draw) ---------------------------------------
def load_be_map(be_csv: Path, preferred_metric: str = "p95_ms") -> dict:
    """
    Map:
      (query, warm, sqlite_variant, duckdb_profile[, tags]) -> break_even_rows (float)
    """
    if not Path(be_csv).is_file():
        return {}
    be = pd.read_csv(be_csv)
    if be.empty or "break_even_rows" not in be.columns:
        return {}

    be["query"] = be["query"].astype(str)
    be["warm"]  = be["warm"].astype(str).str.strip().str.lower().isin(("true","1","yes","warm"))
    if "duckdb_profile" not in be.columns:
        be["duckdb_profile"] = "auto"
    if "metric" not in be.columns:
        be["metric"] = preferred_metric

    # prefer requested metric, else p50, else everything
    if (be["metric"] == preferred_metric).any():
        pref = be[be["metric"] == preferred_metric].copy()
    elif (be["metric"] == "p50_ms").any():
        pref = be[be["metric"] == "p50_ms"].copy()
    else:
        pref = be.copy()

    has_tags = "tags" in pref.columns
    keycols = ["query", "warm", "sqlite_variant", "duckdb_profile"] + (["tags"] if has_tags else [])
    out: dict = {}
    for _, r in pref.iterrows():
        k = tuple(r[c] for c in keycols)
        v = r.get("break_even_rows", np.nan)
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[k] = float(v)
    return out

def _fmt_rows_for_box(x: float, xmin: float, xmax: float) -> str:
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return "n/a"
    if x <= xmin: return f"< {xmin:,.0f}"
    if x >= xmax: return f"> {xmax:,.0f}"
    if x >= 1_000_000: return f"{x/1_000_000:.2f}M"
    if x >= 1_000:     return f"{x/1_000:.2f}k"
    return f"{x:.0f}"

def annotate_be_lines(
    ax: Axes,
    be_map: dict,
    *,
    query: str,
    warm: bool,
    tags: int | None,
    draw_profiles: tuple[str, ...] = ("auto", "t1"),
    sqlite_variant: str = "best",
    draw_vertical: bool = True,
    line_alpha: float = 0.95,
    line_width: float = 1.8,
    colors: dict | None = None,
) -> dict:
    """
    Draw primary BE vertical lines and return:
      {'auto': x_or_None, 't1': x_or_None}

    Sichtbarkeits-Toleranz auf log10-Achse, damit Linien am Rand nicht verschwinden.
    """
    import math

    colors = colors or BE_COLORS

    def _lookup(q, w, variant, prof, tg):
        if tg is None:
            return be_map.get((q, w, variant, prof), None)
        v = be_map.get((q, w, variant, prof, tg), None)
        return v if v is not None else be_map.get((q, w, variant, prof), None)

    vals = {p: _lookup(query, warm, sqlite_variant, p, tags) for p in draw_profiles}

    xmin, xmax = ax.get_xlim()
    vis_tol_log10 = 0.005  # ~1.2% Toleranz in log10

    def _visible(x: float) -> bool:
        if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0):
            return False
        lmin = math.log10(max(xmin, 1e-12))
        lmax = math.log10(max(xmax, 1e-12))
        lx   = math.log10(x)
        return (lx >= lmin - vis_tol_log10) and (lx <= lmax + vis_tol_log10)

    if draw_vertical:
        for p in draw_profiles:
            x = vals.get(p)
            if _visible(x):
                ls = "--" if p == "auto" else "-."
                ax.axvline(x, linestyle=ls, linewidth=line_width, color=colors.get(p, "0.35"),
                           zorder=9, alpha=line_alpha)
    return vals


# ---- multi-break-even (load / derive primary / draw) -------------------------
def load_be_multi(be_multi_csv: Path) -> dict:
    """
    Load break_even_multi.csv:
      key = (query, warm, sqlite_variant, duckdb_profile, tags, metric)
      val = {"be_points":[floats], "zones":[{...}, ...]}
    """
    if not Path(be_multi_csv).is_file():
        return {}
    df = pd.read_csv(be_multi_csv)
    if df.empty:
        return {}

    def _parse_points(idx: int, s: str) -> list[float]:
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
                warnings.warn(f"[be_multi] bad be_point at row {idx}: {tok!r} ({e})")
        return out

    def _parse_zones(idx: int, s: str) -> list[dict]:
        if isinstance(s, str) and s.strip():
            try:
                z = json.loads(s)
                if isinstance(z, list):
                    return z
                warnings.warn(f"[be_multi] zones_json not a list at row {idx}")
            except Exception as e:
                warnings.warn(f"[be_multi] bad zones_json at row {idx}: {e}")
        return []

    out = {}
    for i, r in df.iterrows():
        key = (
            str(r.get("query", "")),
            bool(r.get("warm", False)),
            str(r.get("sqlite_variant", "best")),
            str(r.get("duckdb_profile", "auto")),
            int(r.get("tags")) if pd.notna(r.get("tags")) else None,
            str(r.get("metric", "p95_ms")),
        )
        out[key] = {
            "be_points": _parse_points(i, r.get("be_points", "")),
            "zones": _parse_zones(i, r.get("zones_json", "")),
        }
    return out

def derive_primary_map_from_multi(
    be_multi_map: dict,
    *,
    preferred_metric: str,
    tags: int | None,
    min_rel_delta: float = 0.10,
    min_abs_delta_ms: float = 0.0,
) -> dict:
    """
    Erzeuge eine be_map-ähnliche Dict:
      (query, warm, 'best', profile[, tags]) -> primary_be

    Regel: frühester BE, dessen angrenzende Zone signifikant ist:
      mean_rel_delta >= min_rel_delta ODER |median_delta_ms| >= min_abs_delta_ms
    Fallback: wenn keiner signifikant, nimm den frühesten BE.

    NEU: Wenn tags is None, konsolidiere über *alle* vorhandenen Tags und
         nimm je (q,w,prof) den frühesten (signifikanten) BE.
    """
    def _choose_from_record(rec: dict) -> float | None:
        be_pts = sorted(rec.get("be_points") or [])
        if not be_pts:
            return None
        zones = rec.get("zones") or []

        eps = 1e-6
        def _significant_for_x(x: float) -> bool:
            for z in zones:
                a = float(z.get("start_rows", float("nan")))
                b = float(z.get("end_rows", float("nan")))
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                mrel = float(z.get("mean_rel_delta", 0.0) or 0.0)
                mabs = float(z.get("median_delta_ms", 0.0) or 0.0)
                if (mrel < min_rel_delta) and (abs(mabs) < min_abs_delta_ms):
                    continue
                if abs(x - a) <= max(eps, 1e-9 * max(1.0, x)) or abs(x - b) <= max(eps, 1e-9 * max(1.0, x)):
                    return True
            return False

        chosen = next((float(x) for x in be_pts if _significant_for_x(x)), None)
        return float(be_pts[0]) if chosen is None else chosen

    out: dict = {}

    # tags gesetzt → wie bisher nur für diesen Tag
    if tags is not None:
        for (q, w, sqli, prof, tg, metric), rec in be_multi_map.items():
            if sqli != "best" or metric != preferred_metric or tg != tags:
                continue
            chosen = _choose_from_record(rec)
            if chosen is not None:
                out[(q, bool(w), "best", prof, int(tags))] = chosen
        return out

    # tags None → über alle Tags konsolidieren
    best_across: dict[tuple, float] = {}
    for (q, w, sqli, prof, tg, metric), rec in be_multi_map.items():
        if sqli != "best" or metric != preferred_metric:
            continue
        chosen = _choose_from_record(rec)
        if chosen is None:
            continue
        key = (q, bool(w), "best", prof)  # ohne Tag-Dimension
        best_across[key] = min(chosen, best_across.get(key, float("inf")))

    out.update({k: v for k, v in best_across.items() if math.isfinite(v)})
    return out



def annotate_be_multi(
    ax: Axes,
    be_multi_map: dict,
    *,
    query: str,
    warm: bool,
    tags: int | None,
    metric: str,
    draw_profiles: tuple[str, ...] = ("auto", "t1"),
    sqlite_variant: str = "best",
    top_k: int = 2,
    min_rel_delta: float = 0.10,
    min_abs_delta_ms: float = 0.0,
    dedupe_primary_x: dict | None = None,
    dedupe_tol_log10: float = 0.01,
    colors: dict | None = None,
    ensure_include_primary: bool = False,
    include_primary_x: dict | None = None,
) -> dict:
    """
    Zeichnet Top-K Multi-BE und gibt {'auto':[...], 't1':[...]} zurück.
    NEU:
      - Wenn es keinen Multi-Record (z.B. tags=None) gibt, wird dennoch die
        Primary-Linie gezeichnet (falls über include_primary_x gegeben).
      - Sichtbarkeit mit log-Toleranz.
    """
    import math
    colors = colors or BE_COLORS
    xmin, xmax = ax.get_xlim()
    vis_tol_log10 = 0.005

    def _visible(x: float) -> bool:
        if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0):
            return False
        lmin = math.log10(max(xmin, 1e-12))
        lmax = math.log10(max(xmax, 1e-12))
        lx   = math.log10(x)
        return (lx >= lmin - vis_tol_log10) and (lx <= lmax + vis_tol_log10)

    def _near_log10(a: float, b: float) -> bool:
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0 \
               and abs(math.log10(a) - math.log10(b)) <= dedupe_tol_log10

    def _select_top(be_pts: list[float], zones: list[dict]) -> list[float]:
        if not be_pts:
            return []
        eps = 1e-6
        by_x = sorted(be_pts)

        def _score_for_x(x: float) -> tuple[float, float, float]:
            best_abs, best_rel = -1.0, -1.0
            for z in zones or []:
                a = float(z.get("start_rows", float("nan")))
                b = float(z.get("end_rows", float("nan")))
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                if not (abs(x - a) <= max(eps, 1e-9 * max(1.0, x)) or abs(x - b) <= max(eps, 1e-9 * max(1.0, x))):
                    continue
                mabs = abs(float(z.get("median_delta_ms", 0.0) or 0.0))
                mrel = float(z.get("mean_rel_delta", 0.0) or 0.0)
                if (mrel < min_rel_delta) and (mabs < min_abs_delta_ms):
                    continue
                if (mabs > best_abs) or (mabs == best_abs and mrel > best_rel):
                    best_abs, best_rel = mabs, mrel
            return (best_abs, best_rel, -x)

        ranked = sorted(by_x, key=_score_for_x, reverse=True)
        ranked = [x for x in ranked if _score_for_x(x)[0] >= 0.0]
        return ranked[:top_k]

    selected: dict[str, list[float]] = {}
    for prof in draw_profiles:
        key = (query, warm, sqlite_variant, prof, tags, metric)
        rec = be_multi_map.get(key)
        kept: list[float] = []

        # Kandidaten aus Multi
        if rec:
            top_points = _select_top(rec.get("be_points", []), rec.get("zones", []))
            prim_for_dedupe = (dedupe_primary_x or {}).get(prof)
            for x in top_points:
                if prim_for_dedupe is not None and _near_log10(x, prim_for_dedupe):
                    continue
                kept.append(x)

        # Primary ggf. erzwingen – auch wenn es KEIN rec gibt
        if ensure_include_primary and include_primary_x:
            px = include_primary_x.get(prof)
            if isinstance(px, (int, float)) and math.isfinite(px) and px > 0:
                if not any(_near_log10(px, k) for k in kept):
                    kept = [px] + kept
                if len(kept) > top_k:
                    kept = kept[:top_k]

        # zeichnen
        drawn: list[float] = []
        for x in kept:
            if _visible(x):
                ls = ":" if prof == "auto" else (0, (3, 2, 1, 2))
                ax.axvline(x, linestyle=ls, linewidth=1.6, color=colors.get(prof, "0.5"),
                           zorder=7, alpha=0.9)
                drawn.append(x)
        selected[prof] = drawn

    return selected





# ---- combined BE legend (Primary + Multi) -----------------------------------
def render_be_combined_legend(
    ax: Axes,
    *,
    metric_label: str,
    tags: int | None,
    primary_map: dict,
    multi_map: dict | None,
    outside: bool = False,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] | None = None,
    colors: dict | None = None,
    zorder: float = 100.0,
) -> None:
    colors = colors or BE_COLORS
    xmin, xmax = ax.get_xlim()

    def _near_log10(a: float | None, b: float | None, tol: float = 0.01) -> bool:
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return False
        if a <= 0 or b <= 0:
            return False
        return abs(math.log10(a) - math.log10(b)) <= tol

    def _fmt(x: float) -> str:
        return _fmt_rows_for_box(x, xmin, xmax)

    # Header (single line)
    header = f"Break-even ({metric_label})"
    if tags is not None:
        header += f"  •  tags={tags}"

    lines: list[tuple[str, str]] = [(header, "black")]

    # One line per profile: "auto: P <primary> — <x1, x2, x3>"
    for prof in ("auto", "t1"):
        prim = primary_map.get(prof)
        pts = list(multi_map.get(prof, [])) if multi_map else []

        # remove primary from multi list if it slipped in
        pts = [x for x in pts if not _near_log10(x, prim)]

        parts = []
        if isinstance(prim, (int, float)) and math.isfinite(prim):
            parts.append(f"P {_fmt(prim)}")
        if pts:
            parts.append(", ".join(_fmt(x) for x in pts))

        pretty = " — ".join(parts) if parts else "—"
        lines.append((f"{prof}: {pretty}", colors.get(prof, "black")))

    items = [TextArea(t, textprops=dict(color=c, size=8)) for (t, c) in lines]
    vpack = VPacker(children=items, align="left", pad=0, sep=1)
    box = AnchoredOffsetbox(
        loc=loc, child=vpack, pad=0.3, frameon=True, borderpad=0.4,
        bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes if outside else None
    )
    box.patch.set_alpha(0.95)
    box.patch.set_facecolor("white")
    box.patch.set_edgecolor("0.8")
    box.set_zorder(zorder)
    ax.add_artist(box)





# ---- data shaping ------------------------------------------------------------
def load_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if "dataset_rows" not in df.columns:
        raise ValueError("summary_with_cis.csv missing dataset_rows")
    if "scan_mode" not in df.columns:
        raise ValueError("summary_with_cis.csv missing scan_mode")
    return df

def median_curve(df: pd.DataFrame, query: str, warm: bool, metric: str, scan_mode: str):
    sub = df[(df["query"] == query) & (df["warm"] == warm) & (df["scan_mode"] == scan_mode)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["label"] = sub.apply(engine_label, axis=1)
    g = (sub.groupby(["label","dataset_rows"], as_index=False)[metric]
           .median()
           .sort_values(["dataset_rows"]))
    return g

def ci_band(df: pd.DataFrame, query: str, warm: bool, metric: str, scan_mode: str):
    low, high = f"ci_low_{metric}", f"ci_high_{metric}"
    sub = df[(df["query"] == query) & (df["warm"] == warm) & (df["scan_mode"] == scan_mode)].copy()
    if {low, high, "ci_ok"}.issubset(sub.columns):
        sub = sub[sub["ci_ok"] == True].copy()
        sub["label"] = sub.apply(engine_label, axis=1)
        return (sub.groupby(["label","dataset_rows"], as_index=False)[[low,high]].median())
    return pd.DataFrame()
