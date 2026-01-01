# analysis/extract_iterations.py
import json, re
from pathlib import Path
import pandas as pd
from config import RESULTS_DIR

RUN_GLOBS = ["*_tbl.json", "*_fs.json"]

# Lines look like:
#   some avg10=0.00 avg60=0.00 avg300=0.00 total=0
PSI_RE = re.compile(
    r"^(?P<kind>some|full)\s+avg10=(?P<avg10>[0-9.]+)\s+avg60=(?P<avg60>[0-9.]+)\s+avg300=(?P<avg300>[0-9.]+)\s+total=(?P<total>\d+)\s*$"
)

def parse_psi_text(prefix: str, text: str | None) -> dict:
    out = {}
    if not text:
        return out
    for line in text.splitlines():
        m = PSI_RE.match(line.strip())
        if not m:
            continue
        kind = m.group("kind")  # 'some' or 'full'
        out[f"{prefix}_{kind}_avg10"]  = float(m.group("avg10"))
        out[f"{prefix}_{kind}_avg60"]  = float(m.group("avg60"))
        out[f"{prefix}_{kind}_avg300"] = float(m.group("avg300"))
        out[f"{prefix}_{kind}_total"]  = int(m.group("total"))
    return out

def main():
    rows = []
    for pat in RUN_GLOBS:
        for p in Path(RESULTS_DIR).glob(pat):
            obj = json.loads(p.read_text())
            if "durations_ns" not in obj or "iterations" not in obj:
                continue
            ds = (obj.get("dataset_meta") or {})
            for it in obj["iterations"]:
                r = {
                    "file": p.name,
                    "run_id": obj.get("run_id"),
                    "engine": obj.get("engine"),
                    "threads": obj.get("threads"),
                    "query": obj.get("query"),
                    "warm": obj.get("warm"),
                    "scan_mode": obj.get("scan_mode"),
                    "dataset_rows": ds.get("rows"),
                    "dataset_tags": ds.get("tags"),
                    "i": it.get("i"),
                    "dur_ms": float(it.get("dur_ns", 0))/1e6,
                    "temp_c": it.get("temp_c"),
                    "cpu_percent": it.get("cpu_percent"),
                    "rss_bytes": it.get("rss_bytes"),
                    "freq_khz_cpu0": it.get("freq_khz_cpu0"),
                    "drop_caches_ok": it.get("drop_caches_ok"),
                    "throttled_int": it.get("throttled_int"),
                    "t_wall_ns": it.get("t_wall_ns"),
                }
                psi_b = it.get("psi_before") or {}
                psi_a = it.get("psi_after") or {}
                # keep resource dimension in column names
                for res in ("cpu", "memory", "io"):
                    r.update(parse_psi_text(f"psi_before_{res}", psi_b.get(res)))
                    r.update(parse_psi_text(f"psi_after_{res}",  psi_a.get(res)))
                rows.append(r)
    df = pd.DataFrame(rows)
    out = Path(RESULTS_DIR) / "iterations_long.csv"
    df.to_csv(out, index=False)
    print(f"[ok] wrote {out} with {len(df)} rows")

if __name__ == "__main__":
    main()
