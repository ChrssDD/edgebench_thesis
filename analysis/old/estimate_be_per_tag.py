# analysis/estimate_be_per_tag.py
import csv, math
from pathlib import Path
from collections import defaultdict
from config import RESULTS_DIR

def load_rows():
    rows = []
    with open(Path(RESULTS_DIR)/"be_per_tag.csv") as f:
        r = csv.DictReader(f)
        for x in r:
            try:
                rpt = float(x["rows_per_tag"]) if x["rows_per_tag"] else None
                p95 = float(x["p95_ms"]) if x["p95_ms"] else None
            except: 
                rpt = p95 = None
            rows.append({
                "engine": x["engine"],
                "query": x["query"],
                "warm": x["warm"],
                "rpt": rpt,
                "p95": p95,
                "file": x["file"],
            })
    return rows

def best_sqlite(rows):
    # bestes p95 je (query,warm,rpt) über sqlite_idx/noidx
    best = {}
    for x in rows:
        if x["engine"]!="sqlite" or x["p95"] is None or x["rpt"] is None: 
            continue
        k=(x["query"], x["warm"], x["rpt"])
        best[k]= min(best.get(k, math.inf), x["p95"])
    return best

def duck(rows):
    dd = {}
    for x in rows:
        if x["engine"] != "duckdb" or x["p95"] is None or x["rpt"] is None:
            continue
        # nur AUTO nehmen: _t1_ in Dateiname ausschließen
        if "_t1_" in (x.get("file") or ""):
            continue
        k = (x["query"], x["warm"], x["rpt"])
        dd[k] = x["p95"]
    return dd

def estimate_cross(x_pts):
    # x_pts: Liste (rpt, delta) mit delta = sqlite_best - duckdb
    # Rückgabe: "below_min"/"above_max"/float(BE) linear interpoliert
    x_pts=sorted(x_pts)
    if all(d>0 for _,d in x_pts):  # DuckDB schneller überall (delta>0)
        return "below_min"
    if all(d<0 for _,d in x_pts):  # SQLite schneller überall
        return "above_max"
    for (x1,d1),(x2,d2) in zip(x_pts, x_pts[1:]):
        if d1==0: return x1
        if d1*d2<0:
            # linearer Schnittpunkt zwischen (x1,d1) und (x2,d2)
            t = abs(d1)/(abs(d1)+abs(d2))
            return x1 + t*(x2-x1)
    return None

def main():
    rows=load_rows()
    best_sql = best_sqlite(rows)
    dduck    = duck(rows)

    by_qw=defaultdict(list)
    for (q,w,rpt), s95 in best_sql.items():
        d95 = dduck.get((q,w,rpt))
        if d95 is None: 
            continue
        by_qw[(q,w)].append((rpt, s95 - d95))  # delta>0 → Duck schneller

    print("query,warm,break_even_rows_per_tag,remark")
    for (q,w), pts in sorted(by_qw.items()):
        be = estimate_cross(pts)
        if isinstance(be,str):
            remark = "Duck faster at all sampled sizes" if be=="below_min" else "SQLite faster at all sampled sizes"
            print(f"{q},{w},{be},{remark}")
        elif be is None:
            print(f"{q},{w},n/a,insufficient data")
        else:
            print(f"{q},{w},{int(round(be))},linear interpolation")

if __name__=="__main__":
    main()
