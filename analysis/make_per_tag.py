# analysis/make_per_tag.py
"""
Erzeugt be_per_tag.csv aus allen *_tbl.json in RESULTS_DIR.
Spalten: engine, query, warm, rows_total, tags, rows_per_tag, p50_ms, p95_ms, file
Aufruf:  python -m analysis.make_per_tag
"""

from __future__ import annotations
import json, csv
from pathlib import Path
from config import RESULTS_DIR  # nutzt euren Kampagnen-Results-Ordner

def main() -> None:
    results = Path(RESULTS_DIR)
    out = results / "be_per_tag.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(results.glob("*_tbl.json"))
    rows = []

    for p in files:
        try:
            o = json.loads(p.read_text())
        except Exception:
            continue
        dm = o.get("dataset_meta") or {}
        total = dm.get("rows")
        tags  = dm.get("tags")
        rpt = (total / tags) if (isinstance(total, int) and isinstance(tags, int) and tags > 0) else None
        stats = o.get("stats") or {}

        rows.append([
            o.get("engine"),
            o.get("query"),
            "warm" if o.get("warm") else "cold",
            total,
            tags,
            rpt,
            stats.get("p50_ms"),
            stats.get("p95_ms"),
            p.name
        ])

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine","query","warm","rows_total","tags","rows_per_tag","p50_ms","p95_ms","file"])
        w.writerows(rows)

    print(f"[ok] wrote {out}  (rows={len(rows)})")

if __name__ == "__main__":
    main()
