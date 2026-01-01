# analysis/validate_suite.py
import subprocess, json, sys
from pathlib import Path
from config import RESULTS_DIR

QUERIES = [
  "downsample_1min","downsample_5min","downsample_15min",
  "last_15min_avg","last_60min_avg","window_10min_mean_p95"
]

def run_case(sqlite_db, duck_db, q):
    cmd = [sys.executable, "-m", "analysis.validate_results",
           "--query", q, "--sqlite-db", sqlite_db, "--duckdb-db", duck_db]
    if q == "window_10min_mean_p95":
        cmd += ["--rtol","1e-3","--atol","1e-6"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        status = "OK" if out.strip().startswith("OK") else "MISMATCH"
        return {"query": q, "status": status, "output": out}
    except subprocess.CalledProcessError as e:
        return {"query": q, "status": "ERROR", "output": e.output}

def main():
    if len(sys.argv)!=3:
        print("Usage: python -m analysis.validate_suite <sqlite_db> <duck_db>"); sys.exit(2)
    sqlite_db, duck_db = sys.argv[1], sys.argv[2]
    results = [run_case(sqlite_db, duck_db, q) for q in QUERIES]
    out_json = RESULTS_DIR / "validation_suite.json"
    out_csv  = RESULTS_DIR / "validation_suite.csv"
    out_json.write_text(json.dumps(results, indent=2))
    import pandas as pd
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_json} and {out_csv}")
    ok = sum(r["status"]=="OK" for r in results)
    mism = sum(r["status"]=="MISMATCH" for r in results)
    err = sum(r["status"]=="ERROR" for r in results)
    print(f"Summary: OK={ok} MISMATCH={mism} ERROR={err}")

if __name__ == "__main__":
    main()
