#runner/run_bench.py
import argparse, json, time, os, platform, shutil
from datetime import datetime, timezone
from pathlib import Path
from config import RESULTS_DIR, QUERIES_DIR
import psutil
import numpy as np

def run_sqlite(db, sql):
    import sqlite3
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute(sql); _ = cur.fetchall(); con.close()

def run_duckdb(db, sql, threads=None, setup_stmts=None, disable_object_cache=False):
    """
    setup_stmts: list of raw SQL strings executed before 'sql' (e.g. CREATE TEMP VIEW ...).
    disable_object_cache: turn off DuckDB object cache (useful to enforce 'cold' on file scans)
    """
    import duckdb
    con = duckdb.connect(db)
    if disable_object_cache:
        con.execute("PRAGMA enable_object_cache=false")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    if setup_stmts:
        for stmt in setup_stmts:
            con.execute(stmt)
    con.execute(sql); _ = con.fetchall()
    con.close()

def run_once(engine, db, sql, threads=None, setup_stmts=None, warm=False):
    if engine == "sqlite":
        return run_sqlite(db, sql)
    return run_duckdb(db, sql, threads, setup_stmts, disable_object_cache=(not warm))

def get_versions():
    import sqlite3, duckdb, pandas as pd, numpy as np, pyarrow
    def get_git_rev():
        try:
            import subprocess, pathlib
            root = pathlib.Path(__file__).resolve().parent
            return subprocess.check_output(
                ["git","-C", str(root), "rev-parse","--short","HEAD"], text=True
            ).strip()
        except Exception:
            return None
    return {
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "sqlite3": sqlite3.sqlite_version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version()},
        "machine": platform.machine(),
        "git_rev": get_git_rev(),
    }

def get_db_filesize(path):
    try:
        return os.path.getsize(path)
    except Exception:
        return None

def read_psi():
    base = Path("/proc/pressure"); d = {}
    for k in ("cpu","memory","io"):
        try: d[k] = base.joinpath(k).read_text().strip()
        except: d[k] = None
    return d

def read_temp_c():
    try:
        temps = psutil.sensors_temperatures()
        for key in ("cpu-thermal", "cpu_thermal", "soc_thermal"):
            if key in temps and temps[key]:
                return max(getattr(s, "current", None) for s in temps[key] if getattr(s, "current", None) is not None)
    except Exception:
        pass
    for path in ("/sys/class/thermal/thermal_zone0/temp",):
        try:
            with open(path) as f: return int(f.read().strip())/1000.0
        except Exception: pass
    return None

def read_freq_khz():
    for cpu in (f"cpu{i}" for i in range(os.cpu_count() or 4)):
        p = f"/sys/devices/system/cpu/{cpu}/cpufreq/scaling_cur_freq"
        try:
            with open(p) as f: return int(f.read().strip())
        except Exception: pass
    return None

def drop_caches():
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f: f.write("3\n")
        time.sleep(0.5)
        return True
    except Exception:
        return False

def pct_disc(vals, p):
    """Diskretes Perzentil (ceil(n*p) Ordnungstatistik, 1-basiert)."""
    if not vals: return None
    s = sorted(vals)
    n = len(s)
    k = int(np.ceil(n * (p/100.0)))
    k = min(max(k,1), n) - 1
    return s[k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["sqlite","duckdb"], required=True)
    ap.add_argument("--db", type=str, required=True)
    ap.add_argument("--query", default="downsample_5min")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--threads", type=int, default=None)  # duckdb only
    ap.add_argument("--parquet", type=str, default=None)  # file-scan helper
    ap.add_argument("--csv", type=str, default=None)      # file-scan helper
    ap.add_argument("--sleep-ms", type=int, default=200)
    ap.add_argument("--drop-caches", action="store_true", help="Drop Linux page cache before each iteration (root needed)")
    ap.add_argument("--max-run-s", type=int, default=None,
                    help="Zeitbudget für diesen run_bench (alle Wiederholungen zusammen)")
    a = ap.parse_args()

    # SQL laden
    sql = (QUERIES_DIR / a.engine / f"{a.query}.sql").read_text()

    # DuckDB: Platzhalter ersetzen oder TEMP VIEW bauen
    setup = None
    if a.engine == "duckdb":
        if a.csv and "{CSV}" in sql:
            sql = sql.replace("{CSV}", a.csv)
        if a.parquet and "{PARQUET}" in sql:
            sql = sql.replace("{PARQUET}", a.parquet)
        elif a.parquet:
            q = a.parquet.replace("'", "''")
            setup = [f"CREATE OR REPLACE TEMP VIEW measurements AS SELECT * FROM read_parquet('{q}')"]
        elif a.csv:
            q = a.csv.replace("'", "''")
            setup = [f"CREATE OR REPLACE TEMP VIEW measurements AS SELECT * FROM read_csv_auto('{q}', HEADER=TRUE)"]

    # Optionaler Warmup
    if a.warm:
        run_once(a.engine, a.db, sql, a.threads, setup_stmts=setup, warm=True)

    # Messungen
    proc = psutil.Process(os.getpid()); _ = proc.cpu_percent(None)
    durs = []; it_meta = []; psi = {"start": read_psi()}
    t_budget0 = time.time()
    for i in range(a.repeats):
        if a.max_run_s is not None and (time.time() - t_budget0) >= a.max_run_s:
            break
        dc_ok = drop_caches() if a.drop_caches else False
        meta = {
            "i": i,
            "psi_before": read_psi(),
            "temp_c": read_temp_c(),
            "freq_khz_cpu0": read_freq_khz(),
            "drop_caches_ok": bool(dc_ok),
            "t_wall_ns": time.time_ns(),
        }
        t0 = time.perf_counter_ns()
        run_once(a.engine, a.db, sql, a.threads, setup_stmts=setup, warm=a.warm)
        dur = time.perf_counter_ns() - t0
        durs.append(dur)
        try:
            mi = proc.memory_info(); meta["rss_bytes"] = getattr(mi, "rss", None)
        except Exception: meta["rss_bytes"] = None
        try:
            meta["cpu_percent"] = proc.cpu_percent(interval=0.0)
        except Exception: meta["cpu_percent"] = None
        meta.update({"psi_after": read_psi(), "dur_ns": dur})
        it_meta.append(meta)
        time.sleep(a.sleep_ms/1000)
    psi["end"] = read_psi()

    # Stats (robust even if no iterations ran due to --max-run-s)
    if durs:
        stats = {
            "count": len(durs),
            "p50_ms": round(pct_disc(durs,50)/1e6,3),
            "p95_ms": round(pct_disc(durs,95)/1e6,3),
            "p99_ms": round(pct_disc(durs,99)/1e6,3),
            "mean_ms": round(sum(durs)/len(durs)/1e6,3),
            "min_ms": round(min(durs)/1e6,3),
            "max_ms": round(max(durs)/1e6,3),
        }
    else:
        # Keep schema minimal but valid; dashboard/summarize can handle count==0
        stats = {"count": 0}



    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db_stem = Path(a.db).stem
    tag = "warm" if a.warm else "cold"
    thr = (f"_t{a.threads}" if (a.engine=="duckdb" and a.threads) else "")
    mode = "_fs" if (a.parquet or a.csv) else "_tbl"
    out = RESULTS_DIR / f"{db_stem}_{a.engine}_{a.query}_{tag}{thr}{mode}.json"

    def detect_dataset_meta(engine, db, scan_mode=None, file_path=None):
        try:
            if scan_mode == "file" and file_path:
                import duckdb
                con = duckdb.connect()  # in-memory
                fp = file_path.replace("'", "''")
                if file_path.lower().endswith(".parquet"):
                    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{fp}')").fetchone()[0]
                    tags = con.execute(f"SELECT COUNT(DISTINCT tag) FROM read_parquet('{fp}')").fetchone()[0]
                else:
                    rows = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{fp}', HEADER=TRUE, SAMPLE_SIZE=-1)").fetchone()[0]
                    tags = con.execute(f"SELECT COUNT(DISTINCT tag) FROM read_csv_auto('{fp}', HEADER=TRUE, SAMPLE_SIZE=-1)").fetchone()[0]
                con.close()
            elif engine == "sqlite":
                import sqlite3
                con = sqlite3.connect(db); cur = con.cursor()
                cur.execute("SELECT COUNT(*) FROM measurements"); rows = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT tag) FROM measurements"); tags = cur.fetchone()[0]
                con.close()
            else:
                import duckdb
                con = duckdb.connect(db)
                rows = con.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
                tags = con.execute("SELECT COUNT(DISTINCT tag) FROM measurements").fetchone()[0]
                con.close()
            return {"rows": int(rows), "tags": int(tags)}
        except Exception:
            return None

    payload = {
      "engine": a.engine, "db": a.db, "query": a.query,
      "repeats": a.repeats, "warm": a.warm, "threads": a.threads,
      "sleep_ms": a.sleep_ms, "drop_caches": a.drop_caches,
      "started_at": datetime.now(timezone.utc).isoformat(),
      "durations_ns": durs, "iterations": it_meta,
      "stats": stats, "psi": psi,
      "hostname": os.uname().nodename, "kernel": os.uname().release,
      "versions": get_versions(),
      "db_filesize_bytes": get_db_filesize(a.db),
      "dataset_meta": detect_dataset_meta(a.engine, a.db,
                                          scan_mode=("file" if (a.parquet or a.csv) else "table"),
                                          file_path=(a.parquet or a.csv)),
      "env": {
        "cpus": os.cpu_count(),
        "ram_total_bytes": psutil.virtual_memory().total if hasattr(psutil, "virtual_memory") else None,
        "disk_root": shutil.disk_usage("/")._asdict() if hasattr(shutil, "disk_usage") else None,
      },
      "scan_mode": ("file" if (a.parquet or a.csv) else "table"),
      "run_id": os.environ.get("EDGE_RUN_ID"),
      "truncated": (a.max_run_s is not None and (time.time() - t_budget0) >= a.max_run_s),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[ok] wrote {out}")
    print(stats)

if __name__ == "__main__":
    main()
