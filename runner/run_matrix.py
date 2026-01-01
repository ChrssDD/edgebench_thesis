#runner/run_matrix.py
import argparse, itertools, json, random, subprocess, sys, time, uuid, os
from pathlib import Path
from config import EDGE_ROOT, DATA_DIR, DB_DIR, RESULTS_DIR, LOG_DIR

PROGRESS = RESULTS_DIR / "progress.ndjson"
FAILURES = RESULTS_DIR / "failures.ndjson"
MANIFEST = RESULTS_DIR / "manifest.json"
MANI_DIR = RESULTS_DIR / "manifests"
MANI_DIR.mkdir(parents=True, exist_ok=True)
MANI_INDEX = MANI_DIR / "index.json"

SUMMARY  = RESULTS_DIR / "run_summary.json"

RUN_ID = os.environ.get("EDGE_RUN_ID") or uuid.uuid4().hex[:8]
LOG_DIR.mkdir(parents=True, exist_ok=True)
MATRIX_LOG = LOG_DIR / f"matrix_{RUN_ID}.log"

def tee_line(line, bench_log=None, also_stderr=False):
    target = sys.stderr if also_stderr else sys.stdout
    target.write(line); target.flush()
    with open(MATRIX_LOG, "a") as ml: ml.write(line)
    if bench_log:
        with open(bench_log, "a") as bl: bl.write(line)

def sh_stream(cmd, bench_log=None, env=None, timeout_s=None):
    """Stream child stdout/stderr direkt in Dateien; vermeidet Capture-Buffers."""
    start_line = "+ " + " ".join(cmd) + "\n"
    tee_line(start_line, bench_log=bench_log)
    with open(MATRIX_LOG, "a") as ml, \
         (open(bench_log, "a") if bench_log else open(os.devnull, "w")) as bl:
        with subprocess.Popen(cmd, stdout=bl, stderr=bl, text=True, env=env) as p:
            try:
                rc = p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                finally:
                    rc = 124  # gängiger Timeout-Code
                    tee_line(f"[timeout] {' '.join(cmd)} after {timeout_s}s\n",
                             bench_log=bench_log, also_stderr=True)
    return rc

def gen_dataset(rows, tags, seed=42, freq_ms=1000, t0_ns=None, verbose=False):
    csv = DATA_DIR / f"syn_{rows}x{tags}.csv"
    pq  = DATA_DIR / f"syn_{rows}x{tags}.parquet"
    if csv.exists() and pq.exists():
        return csv, pq
    cmd = [sys.executable, "-m", "generator.synth",
           "--rows", str(rows), "--tags", str(tags),
           "--freq-ms", str(freq_ms), "--seed", str(seed)]
    if t0_ns is not None: cmd += ["--t0-ns", str(t0_ns)]
    rc = sh_stream(cmd, bench_log=str(MATRIX_LOG))
    if rc != 0:
        raise RuntimeError("Datengenerierung fehlgeschlagen")
    return csv, pq

def load_sqlite(csv, index=False, db=None, verbose=False):
    db = db or (DB_DIR / f"sqlite_edge_{'idx' if index else 'noidx'}.db")
    if db.exists(): db.unlink()
    cmd = [sys.executable, "-m", "loader.sqlite_loader", str(csv), "--db", str(db)]
    if index: cmd.append("--extra-index")
    rc = sh_stream(cmd, bench_log=str(MATRIX_LOG))
    if rc != 0:
        raise RuntimeError("SQLite-Load fehlgeschlagen")
    return db

def load_duckdb(parquet, db=None, verbose=False):
    db = db or (DB_DIR / "duck_edge.db")
    if db.exists(): db.unlink()
    cmd = [sys.executable, "-m", "loader.duckdb_loader", str(parquet), "--db", str(db)]
    rc = sh_stream(cmd, bench_log=str(MATRIX_LOG))
    if rc != 0:
        raise RuntimeError("DuckDB-Load fehlgeschlagen")
    return db

def expected_outfile(engine, db, query, warm, threads=None, parquet=None, csv=None):
    db_stem = Path(db).stem
    tag = "warm" if warm else "cold"
    thr = (f"_t{threads}" if (engine=="duckdb" and threads) else "")
    mode = "_fs" if (parquet or csv) else "_tbl"
    return RESULTS_DIR / f"{db_stem}_{engine}_{query}_{tag}{thr}{mode}.json"

def log_progress(rec):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def log_failure(rec):
    with open(FAILURES, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# OPTIONAL: in-process execution (power-user)
def run_bench_inproc(engine, db, query, warm, repeats, threads, parquet, csv, drop_caches, max_run_s):
    from runner.run_bench import main as bench_main
    argv = [sys.argv[0], "--engine", engine, "--db", str(db),
            "--query", query, "--repeats", str(repeats)]
    if warm: argv.append("--warm")
    if threads is not None: argv += ["--threads", str(threads)]
    if parquet: argv += ["--parquet", str(parquet)]
    if csv: argv += ["--csv", str(csv)]
    if drop_caches: argv.append("--drop-caches")
    if max_run_s is not None: argv += ["--max-run-s", str(max_run_s)]
    old_argv = sys.argv
    try:
        sys.argv = argv
        bench_main()
        return 0
    finally:
        sys.argv = old_argv

def run_bench_exec(engine, db, query, warm, repeats=9, threads=None, drop_caches=False,
                   parquet=None, csv=None, overwrite=False, exec_mode="subprocess",
                   timeout_s=None, max_run_s=None):
    out = expected_outfile(engine, db, query, warm, threads, parquet, csv)
    if out.exists() and not overwrite:
        log_progress({
            "ts": time.time(), "status": "skipped_existing",
            "out": str(out), "engine": engine, "query": query, "warm": warm,
            "threads": threads, "parquet": bool(parquet), "csv": bool(csv)
        })
        return True

    bench_log = LOG_DIR / f"bench_{RUN_ID}_{Path(db).stem}_{engine}_{query}_" \
                          f"{'warm' if warm else 'cold'}{('_t'+str(threads)) if (engine=='duckdb' and threads) else ''}" \
                          f"{'_fs' if (parquet or csv) else '_tbl'}.log"

    cmd = [sys.executable, "-m", "runner.run_bench",
           "--engine", engine, "--db", str(db),
           "--query", query, "--repeats", str(repeats)]
    if warm: cmd.append("--warm")
    if threads is not None: cmd += ["--threads", str(threads)]
    if drop_caches: cmd.append("--drop-caches")
    if parquet: cmd += ["--parquet", str(parquet)]
    if csv: cmd += ["--csv", str(csv)]
    if max_run_s is not None: cmd += ["--max-run-s", str(max_run_s)]

    t0 = time.time()
    if exec_mode == "inproc":
        rc = run_bench_inproc(engine, db, query, warm, repeats, threads, parquet, csv, drop_caches, max_run_s)
    else:
        env = os.environ.copy()
        env.setdefault("PYTHONHASHSEED", "0")
        rc = sh_stream(cmd, bench_log=str(bench_log), env=env, timeout_s=timeout_s)
    dur = time.time() - t0

    rec = {
        "ts": time.time(), "engine": engine, "query": query, "warm": warm,
        "threads": threads, "parquet": parquet, "csv": csv,
        "db": str(db), "duration_s": round(dur, 3),
        "returncode": rc, "out_file": str(out),
        "bench_log": str(bench_log),
        "run_id": RUN_ID,
    }
    if rc == 0:
        rec["status"] = "ok"
        log_progress(rec); return True
    else:
        rec["status"] = "timeout" if rc == 124 else "error"
        if timeout_s: rec["timeout_s"] = timeout_s
        log_progress(rec); log_failure(rec); return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100000,1000000,5000000,10000000")
    ap.add_argument("--tags", default="10,50,100")
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--drop-caches", action="store_true")
    ap.add_argument("--with-file-scan", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exec-mode", choices=["subprocess","inproc"], default="subprocess")
    ap.add_argument("--timeout-s-per-bench", type=int, default=None,
                    help="hartes Zeitlimit je runner.run_bench Prozess")
    ap.add_argument("--max-run-s", type=int, default=None,
                    help="Budget innerhalb von runner.run_bench (früher Abbruch der Wiederholungen)")
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    tags  = [int(x) for x in a.tags.split(",")]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "sizes": sizes, "tags": tags, "repeats": a.repeats,
        "with_file_scan": a.with_file_scan, "overwrite": a.overwrite,
        "drop_caches": a.drop_caches, "run_id": RUN_ID,
        "generated_at": __import__("datetime").datetime.utcnow().isoformat()+"Z",
        "edge_root": str(EDGE_ROOT), "results_dir": str(RESULTS_DIR),
        "db_dir": str(DB_DIR), "data_dir": str(DATA_DIR),
        "exec_mode": a.exec_mode,
        "timeout_s_per_bench": a.timeout_s_per_bench,
        "max_run_s": a.max_run_s,
    }

    MANIFEST.write_text(json.dumps(manifest_payload, indent=2))

    (MANI_DIR / f"{RUN_ID}.json").write_text(json.dumps(manifest_payload, indent=2))
    
    try:
        idx = json.loads(MANI_INDEX.read_text()) if MANI_INDEX.exists() else {"runs": []}
    except Exception:
        idx = {"runs": []}
    idx["runs"] = [r for r in idx["runs"] if r.get("run_id") != RUN_ID]
    idx["runs"].append({"run_id": RUN_ID, "manifest": f"manifests/{RUN_ID}.json", "generated_at": manifest_payload["generated_at"]})
    MANI_INDEX.write_text(json.dumps(idx, indent=2))

    # old stuff
    # RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # MANIFEST.write_text(json.dumps({
    #     "sizes": sizes, "tags": tags, "repeats": a.repeats,
    #     "with_file_scan": a.with_file_scan, "overwrite": a.overwrite,
    #     "drop_caches": a.drop_caches, "run_id": RUN_ID,
    #     "generated_at": __import__("datetime").datetime.utcnow().isoformat()+"Z",
    #     "edge_root": str(EDGE_ROOT), "results_dir": str(RESULTS_DIR),
    #     "db_dir": str(DB_DIR), "data_dir": str(DATA_DIR),
    #     "exec_mode": a.exec_mode,
    #     "timeout_s_per_bench": a.timeout_s_per_bench,
    #     "max_run_s": a.max_run_s,
    # }, indent=2))

    # datasets
    datasets = []
    for rows, tg in itertools.product(sizes, tags):
        csv, pq = gen_dataset(rows, tg)
        sqlite_db_noidx = load_sqlite(csv, index=False, db=DB_DIR / f"sqlite_{rows}x{tg}_noidx.db")
        sqlite_db_idx   = load_sqlite(csv, index=True,  db=DB_DIR / f"sqlite_{rows}x{tg}_idx.db")
        duck_db         = load_duckdb(pq, db=DB_DIR / f"duck_{rows}x{tg}.db")
        datasets.append((rows, tg, csv, pq, sqlite_db_noidx, sqlite_db_idx, duck_db))

    # plan
    ds_queries = ["downsample_1min","downsample_5min","downsample_15min"]
    win_queries = ["window_10min_mean_p95"]
    last_queries = ["last_15min_avg","last_60min_avg"]
    queries = ds_queries + win_queries + last_queries

    matrix = []
    for rows, tg, csv, pq, sqlite_db_noidx, sqlite_db_idx, duck_db in datasets:
        for q in queries:
            for warm in (False, True):
                matrix.append(("sqlite", sqlite_db_noidx, q, warm, None, None, None))
                matrix.append(("sqlite", sqlite_db_idx,   q, warm, None, None, None))
                matrix.append(("duckdb", duck_db, q, warm, None, None, None))
                matrix.append(("duckdb", duck_db, q, warm, 1,    None, None))
                if a.with_file_scan:
                    matrix.append(("duckdb", duck_db, q, warm, None, pq,   None))

    random.seed(1337); random.shuffle(matrix)
    tee_line(f"[plan] total_runs={len(matrix)} run_id={RUN_ID}\n")

    ok_cnt = err_cnt = skip_cnt = 0
    for eng, db, q, warm, thr, pq, csv in matrix:
        drop_flag = (a.drop_caches and (not warm))
        out = expected_outfile(eng, db, q, warm, thr, pq, csv)
        if out.exists() and not a.overwrite:
            skip_cnt += 1
            log_progress({
                "ts": time.time(), "status": "skipped_existing",
                "out": str(out), "engine": eng, "query": q, "warm": warm,
                "threads": thr, "parquet": bool(pq), "csv": bool(csv), "run_id": RUN_ID
            })
            continue

        ok = run_bench_exec(eng, db, q, warm, repeats=a.repeats, threads=thr,
                            drop_caches=drop_flag, parquet=str(pq) if pq else None,
                            csv=str(csv) if csv else None, overwrite=a.overwrite,
                            exec_mode=a.exec_mode, timeout_s=a.timeout_s_per_bench,
                            max_run_s=a.max_run_s)
        if ok: ok_cnt += 1
        else: err_cnt += 1

    SUMMARY.write_text(json.dumps({
        "ok": ok_cnt, "errors": err_cnt, "skipped": skip_cnt,
        "total_planned": len(matrix), "run_id": RUN_ID
    }, indent=2))
    tee_line(f"[done] ok={ok_cnt} errors={err_cnt} skipped={skip_cnt} planned={len(matrix)} run_id={RUN_ID}\n")

if __name__ == "__main__":
    main()
