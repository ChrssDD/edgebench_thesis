from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


# -----------------------
# parsing helpers
# -----------------------
def parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_csv_strs(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_points(s: Any) -> list[float]:
    """
    break_even_multi.csv stores points as "1234.0;5678.0;..."
    We drop non-finite values (nan/inf) so tables don't show "-- (n=1)".
    """
    if s is None:
        return []
    txt = str(s).strip()
    if not txt:
        return []

    out: list[float] = []
    for tok in txt.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            x = float(tok)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)

    # unique + sorted (stable)
    out = sorted(set(out))
    return out


def parse_zones(s: Any) -> list[dict]:
    if s is None:
        return []
    txt = str(s).strip()
    if not txt:
        return []
    try:
        z = json.loads(txt)
        return z if isinstance(z, list) else []
    except Exception:
        return []


# -----------------------
# primary selection
# -----------------------
def choose_primary(
    points: list[float],
    zones: list[dict],
    *,
    min_rel: float,
    min_abs_ms: float,
) -> float | None:
    """
    Primary = earliest break-even point that touches a "significant" zone.
    If none is significant: fallback to the first point.
    """
    if not points:
        return None

    eps = 1e-6

    def near(a: float, b: float) -> bool:
        return abs(a - b) <= max(eps, 1e-9 * max(1.0, a, b))

    def significant_at(x: float) -> bool:
        for z in zones or []:
            a = float(z.get("start_rows", float("nan")))
            b = float(z.get("end_rows", float("nan")))
            if not (math.isfinite(a) and math.isfinite(b)):
                continue

            mrel = float(z.get("mean_rel_delta", 0.0) or 0.0)
            mabs = float(z.get("median_delta_ms", 0.0) or 0.0)

            # zone is insignificant if BOTH thresholds are not met
            if (mrel < min_rel) and (abs(mabs) < min_abs_ms):
                continue

            # point is considered "anchored" if it is at zone boundary
            if near(x, a) or near(x, b):
                return True
        return False

    for x in points:
        if significant_at(x):
            return float(x)

    return float(points[0])


# -----------------------
# formatting
# -----------------------
def fmt_rows(x: float | None) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "--"
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x/1_000:.2f}k"
    return f"{x:.0f}"


def latex_escape_texttt(s: str) -> str:
    # for \texttt{...}
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("_", r"\_")
         .replace("%", r"\%")
         .replace("&", r"\&")
         .replace("#", r"\#")
         .replace("{", r"\{")
         .replace("}", r"\}")
    )


def fmt_points_multi(points: list[float], primary: float | None, *, show_n: bool = True) -> str:
    """
    Render all points; primary is bold.
    Example: "\\textbf{83.91k}; 99.87k (n=2)"
    """
    if not points:
        return "--"

    # try to match primary to one of the points (tolerant)
    prim_idx: int | None = None
    if primary is not None and math.isfinite(primary):
        for i, p in enumerate(points):
            if abs(p - primary) <= max(1e-6, 1e-9 * max(1.0, p, primary)):
                prim_idx = i
                break

    parts: list[str] = []
    for i, p in enumerate(points):
        txt = fmt_rows(p)
        if prim_idx is not None and i == prim_idx:
            txt = r"\textbf{" + txt + "}"
        parts.append(txt)

    s = "; ".join(parts)
    if show_n:
        s += f" (n={len(points)})"
    return s


def write_tabular(out_path: Path, rows: list[dict], caption_hint: str) -> None:
    lines: list[str] = []
    lines.append("% " + caption_hint)
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Query & DuckDB(auto) & DuckDB(t1) \\")
    lines.append(r"\midrule")

    for r in rows:
        q = r"\texttt{" + latex_escape_texttt(str(r["query"])) + "}"
        a = str(r.get("auto", "--"))
        t = str(r.get("t1", "--"))
        lines.append(f"{q} & {a} & {t} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------
# main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--be-multi", required=True, help="Path to break_even_multi.csv")
    ap.add_argument("--outdir", required=True, help="Output dir for .tex fragments")

    ap.add_argument("--tags", default="20,50")
    ap.add_argument("--metrics", default="p50_ms,p95_ms")

    ap.add_argument("--min-rel", type=float, default=0.10)
    ap.add_argument("--min-abs-ms", type=float, default=0.0)

    ap.add_argument("--sqlite-variant", default="best", help="e.g. best")
    ap.add_argument("--duckdb-profiles", default="auto,t1", help="comma list, e.g. auto,t1")

    ap.add_argument("--also-primary", action="store_true", help="Also write be_primary__... (primary only)")
    args = ap.parse_args()

    p = Path(args.be_multi)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(p)

    # normalize columns
    required = ["tags", "metric", "warm", "query", "sqlite_variant", "duckdb_profile", "be_points", "zones_json"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] missing columns in {p.name}: {missing}")

    df["warm"] = df["warm"].astype(str).str.strip().str.lower().isin(("1", "true", "yes", "warm"))
    df["sqlite_variant"] = df["sqlite_variant"].astype(str).str.strip().str.lower()
    df["duckdb_profile"] = df["duckdb_profile"].astype(str).str.strip().str.lower()
    df["query"] = df["query"].astype(str)

    tags_list = parse_csv_ints(args.tags)
    metrics = parse_csv_strs(args.metrics)
    profiles = [x.lower() for x in parse_csv_strs(args.duckdb_profiles)]
    sqlite_variant = str(args.sqlite_variant).strip().lower()

    for tags in tags_list:
        for metric in metrics:
            for warm in (False, True):
                sub = df[
                    (df["tags"] == tags)
                    & (df["metric"] == metric)
                    & (df["warm"] == warm)
                    & (df["sqlite_variant"] == sqlite_variant)
                    & (df["duckdb_profile"].isin(profiles))
                ].copy()

                if sub.empty:
                    continue

                rows_multi: list[dict] = []
                rows_primary: list[dict] = []

                for q in sorted(sub["query"].unique()):
                    recs = sub[sub["query"] == q]

                    out_multi = {"query": q, "auto": "--", "t1": "--"}
                    out_primary = {"query": q, "auto": "--", "t1": "--"}

                    for prof in profiles:
                        r = recs[recs["duckdb_profile"] == prof]
                        if r.empty:
                            continue

                        r0 = r.iloc[0]
                        pts = parse_points(r0.get("be_points"))
                        zones = parse_zones(r0.get("zones_json"))
                        prim = choose_primary(pts, zones, min_rel=args.min_rel, min_abs_ms=args.min_abs_ms)

                        out_multi[prof] = fmt_points_multi(pts, prim, show_n=True)
                        out_primary[prof] = fmt_rows(prim)

                    rows_multi.append(out_multi)
                    rows_primary.append(out_primary)

                tag = "warm" if warm else "cold"

                out_path_multi = outdir / f"be_multi__tags{tags}__{metric}__{tag}.tex"
                hint_multi = (
                    f"Multi break-even points from break_even_multi.csv (sqlite_variant={sqlite_variant}), "
                    f"primary is bold; tags={tags}, metric={metric}, {tag}"
                )
                write_tabular(out_path_multi, rows_multi, hint_multi)

                if args.also_primary:
                    out_path_primary = outdir / f"be_primary__tags{tags}__{metric}__{tag}.tex"
                    hint_primary = (
                        f"Primary break-even (derived from multi) (sqlite_variant={sqlite_variant}); "
                        f"tags={tags}, metric={metric}, {tag}"
                    )
                    write_tabular(out_path_primary, rows_primary, hint_primary)

    print(f"[ok] wrote tables -> {outdir}")


if __name__ == "__main__":
    main()
