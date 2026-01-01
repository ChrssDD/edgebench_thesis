# analysis/import_meas_log.py
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

# Entfernt ANSI Escape Codes (idf_monitor Farben etc.)
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# Robust: findet MEAS;... irgendwo in der Zeile (nicht nur am Anfang)
# Gruppen: ts_raw, tag, value
MEAS_RE = re.compile(
    r"MEAS;\s*(\d+)\s*;\s*([^;]+?)\s*;\s*([-+0-9.eE]+)",
    flags=0
)

def infer_unit_from_values(vals: np.ndarray) -> str:
    # Heuristik (dein ESP-Log ist in der Regel µs seit Boot)
    med = float(np.median(vals))
    if med >= 1e13:
        return "ns"
    if med >= 1e10:
        return "us"
    return "ms"

def to_ns(ts: np.ndarray, unit: str) -> np.ndarray:
    ts = ts.astype(np.int64)
    if unit == "ns":
        return ts
    if unit == "us":
        return ts * 1_000
    if unit == "ms":
        return ts * 1_000_000
    raise ValueError(f"unknown unit: {unit}")

def main():
    ap = argparse.ArgumentParser(
        description="Convert ESP-IDF idf_monitor log with MEAS;... lines into EdgeBench CSV/Parquet."
    )
    ap.add_argument("logfile", type=Path, help="Path to the text log (idf_monitor output).")
    ap.add_argument("--out-prefix", default="real_wx", help="Output prefix (written to DATA_DIR).")
    ap.add_argument("--unit", choices=["auto", "us", "ms", "ns"], default="auto", help="Unit of MEAS timestamp.")
    ap.add_argument("--relative", action="store_true", help="Shift time so min(ts_ns)=0 (recommended).")
    ap.add_argument(
        "--dedup",
        choices=["keep_last", "keep_first", "mean"],
        default="keep_last",
        help="Resolve duplicates on (ts_ns,tag) for SQLite PRIMARY KEY."
    )
    ap.add_argument(
        "--require-tags",
        default=None,
        help="Optional: comma-separated whitelist of tags to keep (e.g. wx_temp,wx_hum,wx_gas,wx_rain)."
    )
    a = ap.parse_args()

    if not a.logfile.exists():
        raise SystemExit(f"File not found: {a.logfile}")

    keep_tags = None
    if a.require_tags:
        keep_tags = {t.strip() for t in a.require_tags.split(",") if t.strip()}

    rows = []
    meas_seen = 0

    with a.logfile.open("r", errors="ignore") as f:
        for raw in f:
            # robust cleanup
            line = raw.replace("\x00", "")          # remove null bytes
            line = ANSI_RE.sub("", line)            # remove colors
            line = line.strip("\r\n")               # normalize newlines

            # find MEAS anywhere in the line
            m = MEAS_RE.search(line)
            if not m:
                continue

            meas_seen += 1
            try:
                ts_raw = int(m.group(1))
                tag = m.group(2).strip()
                val = float(m.group(3))
            except Exception:
                continue

            if not tag:
                continue
            if keep_tags is not None and tag not in keep_tags:
                continue

            rows.append((ts_raw, tag, val))

    if not rows:
        # helpful hint: show that we didn't parse anything although file exists
        raise SystemExit(
            "No MEAS;... lines parsed. "
            "If rg finds MEAS; lines, they may be malformed or use commas instead of dots for decimals."
        )

    df = pd.DataFrame(rows, columns=["ts_raw", "tag", "value"])

    unit = a.unit
    if unit == "auto":
        unit = infer_unit_from_values(df["ts_raw"].to_numpy(dtype=np.int64))

    df["ts_ns"] = to_ns(df["ts_raw"].to_numpy(dtype=np.int64), unit)

    if a.relative:
        df["ts_ns"] = df["ts_ns"] - int(df["ts_ns"].min())

    # deterministic ordering
    df = df.sort_values(["ts_ns", "tag"], kind="mergesort")

    # dedup for SQLite PK (ts_ns,tag)
    if a.dedup in ("keep_last", "keep_first"):
        keep = "last" if a.dedup == "keep_last" else "first"
        df = df.drop_duplicates(subset=["ts_ns", "tag"], keep=keep)
    else:
        df = df.groupby(["ts_ns", "tag"], as_index=False)["value"].mean()

    # stats
    tags = sorted(df["tag"].unique().tolist())
    span_s = (int(df["ts_ns"].max()) - int(df["ts_ns"].min())) / 1e9
    print(
        f"[ok] parsed rows={len(df)} tags={len(tags)} span_s={span_s:.1f} "
        f"unit={unit} relative={bool(a.relative)} meas_lines_seen={meas_seen}"
    )

    print("[tags]", ", ".join(tags))

    print("[stats] per-tag count + median dt (s)")
    for t in tags:
        sub = df[df["tag"] == t].sort_values("ts_ns")
        cnt = len(sub)
        if cnt >= 3:
            dts = np.diff(sub["ts_ns"].to_numpy(dtype=np.int64)) / 1e9
            med_dt = float(np.median(dts))
            p95_dt = float(np.quantile(dts, 0.95))
            print(f"  {t:10s} n={cnt:8d}  med_dt={med_dt:.6f}s  p95_dt={p95_dt:.6f}s")
        else:
            print(f"  {t:10s} n={cnt:8d}  med_dt=nan")

    # write outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DATA_DIR / f"{a.out_prefix}.csv"
    out_pq = DATA_DIR / f"{a.out_prefix}.parquet"

    df[["ts_ns", "tag", "value"]].to_csv(out_csv, index=False)
    df[["ts_ns", "tag", "value"]].to_parquet(out_pq, index=False)

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_pq}")

if __name__ == "__main__":
    main()
