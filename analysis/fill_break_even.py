# analysis/fill_break_even.py
# Fill missing shared-size runs so break-even can be computed.
# - Plans DuckDB(profiles: auto,t1), SQLite(idx), SQLite(noidx) across chosen TOTAL sizes & tags
# - ONLY runs if the expected JSON is missing (idempotent)
# - True cold = sudo + --drop-caches; warm = normal user
# - Then rebuilds summary, dashboard, and (optionally) diagnose
#
# Usage (from repo root, venv on):
#   python -m analysis.fill_break_even
#   python -m analysis.fill_break_even --totals 10000,100000,1000000,25000 --tags 20,50
#   python -m analysis.fill_break_even --rows-per-tag 100000,1000000 --tags 20,50
#   python -m analysis.fill_break_even --queries last_60min_avg,downsample_5min --max-run-s 240
#   python -m analysis.fill_break_even --no-warm             # cold only
#   python -m analysis.fill_break_even --no-cold             # warm only
#   python -m analysis.fill_break_even --no-summarize        # just run benches
#   python -m analysis.fill_break_even --diagnose            # also run diagnose at the end
#   python -m analysis.fill_break_even --duckdb-profiles auto,t1
#   python -m analysis.fill_break_even --autogen
#   python -m analysis.fill_break_even --dry-run
#
# Notes:
# - Totals are PER DATASET. Filenames use rows-per-tag: rows/tag = TOTAL ÷ TAGS.
# - Worst-case time estimate is printed before runs (planned × max_run_s).

from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import json

from config import RESULTS_DIR, DB_DIR

DEFAULT_QUERIES = [
    "downsample_1min",
    "downsample_5min",
    "downsample_15min",
    "last_15min_avg",
    "last_60min_avg",
    "window_10min_mean_p95",
]

def _run(cmd, env, use_sudo: bool):
    if use_sudo:
        # preserve env (e.g., EDGE_ROOT, PYTHONHASHSEED) and run with sudo for true cold
        cmd = ["sudo", "-E", "env"] + [f"{k}={v}" for k, v in env.items()] + cmd
    return subprocess.run(cmd, text=True)

def parse_csv_ints(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out

def parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def expected_outfile(
    engine: str,
    db_path: str | Path,
    query: str,
    warm: bool,
    threads: Optional[int] = None,
    parquet: Optional[str] = None,
    csv: Optional[str] = None,
) -> Path:
    """
    Mirror runner.run_bench() naming:
      <results>/<dbstem>_<engine>_<query>_<warm|cold>[_t{threads}]{_fs|_tbl}.json
    """
    db_stem = Path(db_path).stem
    tag = "warm" if warm else "cold"
    thr = (f"_t{threads}" if (engine == "duckdb" and threads) else "")
    mode = "_fs" if (parquet or csv) else "_tbl"
    return Path(RESULTS_DIR) / f"{db_stem}_{engine}_{query}_{tag}{thr}{mode}.json"

def build_db_paths(rows_per_tag_list: Iterable[int], tags: int) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Filenames encode rows-per-tag:
      duck_{rows_per_tag}x{tags}.db
      sqlite_{rows_per_tag}x{tags}_idx.db
      sqlite_{rows_per_tag}x{tags}_noidx.db
    """
    ducks, sqli_idx, sqli_noi = [], [], []
    for rpt in rows_per_tag_list:
        ducks.append(Path(DB_DIR) / f"duck_{rpt}x{tags}.db")
        sqli_idx.append(Path(DB_DIR) / f"sqlite_{rpt}x{tags}_idx.db")
        sqli_noi.append(Path(DB_DIR) / f"sqlite_{rpt}x{tags}_noidx.db")
    return ducks, sqli_idx, sqli_noi

def run_once(
    engine: str,
    db: Path,
    query: str,
    warm: bool,
    repeats: int,
    max_run_s: int | None,
    threads: Optional[int] = None,
) -> int:
    """
    Launch runner.run_bench for one (engine,db,query,warm) combo.
    - warm=True  -> normal user, no drop-caches
    - warm=False -> sudo + --drop-caches (true cold start)
    """
    cmd = [
        sys.executable, "-m", "runner.run_bench",
        "--engine", engine,
        "--db", str(db),
        "--query", query,
        "--repeats", str(repeats),
    ]
    if warm:
        cmd.append("--warm")
    else:
        cmd.append("--drop-caches")   # required for real COLD runs

    if engine == "duckdb" and threads is not None:
        cmd += ["--threads", str(threads)]

    if max_run_s is not None:
        cmd += ["--max-run-s", str(max_run_s)]

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")

    use_sudo = not warm   # cold needs sudo for a proper cache drop
    argv = (["sudo","-E","env"] + [f"{k}={v}" for k, v in env.items()] + cmd) if use_sudo else cmd
    print("+ " + " ".join(shlex.quote(c) for c in argv))

    try:
        p = _run(cmd, env, use_sudo)
        return p.returncode
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[error] failed to launch: {e}", file=sys.stderr)
        return 1

def is_truncated(path: Path) -> bool:
    try:
        obj = json.loads(path.read_text())
        return bool(obj.get("truncated"))
    except Exception:
        return False

def run_if_missing(
    engine: str,
    db: Path,
    query: str,
    warm: bool,
    repeats: int,
    max_run_s: int | None,
    dry_run: bool,
    threads: Optional[int] = None
) -> bool:
    """
    If the expected output JSON is missing, run the bench.
    Returns True if we executed (or would execute in dry-run), False if skipped.
    """
    out = expected_outfile(engine, db, query, warm, threads=threads)
    if out.exists() and not is_truncated(out):
        print(f"[skip] {out.name} already exists (not truncated)")
        return False
    if not db.exists():
        print(f"[warn] DB missing: {db}")
        return False
    if dry_run:
        thr_tag = (f" t{threads}" if (engine == 'duckdb' and threads) else "")
        print(f"[plan] would run {engine}{thr_tag} {db.name} {query} {'warm' if warm else 'cold'}")
        return True
    print(f"[run ] {engine}{(' t'+str(threads)) if (engine=='duckdb' and threads) else ''} {db.name} {query} {'warm' if warm else 'cold'}")
    rc = run_once(engine, db, query, warm, repeats, max_run_s, threads)
    if rc != 0:
        print(f"[fail] rc={rc} {engine} {db.name} {query} {'warm' if warm else 'cold'}", file=sys.stderr)
    return True

def call_module(mod: str) -> None:
    cmd = [sys.executable, "-m", mod]
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

def main():
    ap = argparse.ArgumentParser(description="Fill missing shared-size benches to unlock break-even.")
    ap.add_argument("--totals", default="10000,100000,1000000",
                    help="Comma-separated TOTAL rows per dataset. Will be converted to rows-per-tag for filenames: <rows_per_tag>x<tags>.")
    ap.add_argument("--tags", default="20,50",
                    help="Comma-separated tag counts, e.g. 20,50 (default matches Exposé)")
    ap.add_argument("--rows-per-tag", default=None,
                    help="Comma-separated rows per tag levels (e.g. 100000,1000000,10000000). If set, overrides --totals conversion.")
    ap.add_argument("--queries", default=",".join(DEFAULT_QUERIES),
                    help="Comma-separated list of queries to run")
    ap.add_argument("--duckdb-profiles", default="auto,t1",
                    help="Comma-separated subset of {auto,t1} to run for DuckDB (default: both)")
    ap.add_argument("--repeats", type=int, default=7, help="Repeats per run (default: 7)")
    ap.add_argument("--max-run-s", type=int, default=180, help="Budget inside runner.run_bench per (engine,db,query,warm,profile)")
    ap.add_argument("--no-cold", action="store_true", help="Skip cold runs")
    ap.add_argument("--no-warm", action="store_true", help="Skip warm runs")
    ap.add_argument("--no-summarize", action="store_true", help="Do not rebuild summary/dashboard after runs")
    ap.add_argument("--diagnose", action="store_true", help="Run analysis.diagnose at the end")
    ap.add_argument("--dry-run", action="store_true", help="Print plan/time estimate, do not execute benches")
    ap.add_argument("--autogen", action="store_true",
                    help="If a DB is missing, synthesize data and load DBs before running benches.")

    a = ap.parse_args()
    totals = parse_csv_ints(a.totals)
    queries = parse_csv_strs(a.queries)
    tags_list = parse_csv_ints(a.tags)
    rpt_list_cli = parse_csv_ints(a.rows_per_tag) if a.rows_per_tag else None
    duck_profiles = [p.strip().lower() for p in a.duckdb_profiles.split(",") if p.strip().lower() in ("auto", "t1")]
    if not duck_profiles:
        duck_profiles = ["auto"]

    if a.no_cold and a.no_warm:
        print("[nothing to do] both --no-cold and --no-warm were set", file=sys.stderr)
        return

    planned = executed = 0

    def loop(engine: str, dbs: List[Path], threads_for_duck=None):
        nonlocal planned, executed
        for db in dbs:
            for q in queries:
                warm_states = ([False] if a.no_warm else ([True] if a.no_cold else [False, True]))
                for warm in warm_states:
                    if engine == "duckdb" and threads_for_duck is not None:
                        for thr in threads_for_duck:
                            planned += 1
                            ran = run_if_missing(engine, db, q, warm, a.repeats, a.max_run_s, a.dry_run, threads=thr)
                            if ran: executed += 1
                    else:
                        planned += 1
                        ran = run_if_missing(engine, db, q, warm, a.repeats, a.max_run_s, a.dry_run, threads=None)
                        if ran: executed += 1

    # ---- PLAN header ----
    print("=== PLAN ===")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"DB_DIR:      {DB_DIR}")
    print(f"Totals:      {totals}  (tags={tags_list})")
    if rpt_list_cli:
        print(f"Rows/tag:    {rpt_list_cli}  (explicit via --rows-per-tag)")
    print(f"DuckDB:      profiles={duck_profiles}")
    print(f"Queries:     {queries}")
    print(f"Warm/Cold:   {'warm ' if not a.no_warm else ''}{'cold' if not a.no_cold else ''}")
    print(f"Repeats:     {a.repeats}, max-run-s: {a.max_run_s}")
    if a.dry_run:
        print("[dry-run] will not execute benches")
    print(f"Autogen DBs: {'yes' if a.autogen and not a.dry_run else 'no'}")

    # ---- PLAN loops (by tag cardinality) ----
    for tags in tags_list:
        rows_per_tag_list = (rpt_list_cli if rpt_list_cli
                             else [max(t // tags, 1) for t in totals])

        # helpful print: TOTAL = rows_per_tag * tags
        print(
            "Rows/tag:    "
            f"{rows_per_tag_list}  "
            "(filenames use rows-per-tag; TOTAL = rows/tag × tags → "
            + ", ".join([f"{rpt * tags}" for rpt in rows_per_tag_list]) + ")"
        )

        # Optionally autogenerate missing DBs for this (rpt, tags) grid
        if a.autogen and not a.dry_run:
            for rpt in rows_per_tag_list:
                ensure_dbs(rpt, tags)

        ducks, sqli_idx, sqli_noi = build_db_paths(rows_per_tag_list, tags)

        # DuckDB (auto and/or t1)
        thr_list: List[Optional[int]] = []
        if "auto" in duck_profiles:
            thr_list.append(None)
        if "t1" in duck_profiles:
            thr_list.append(1)
        if duck_profiles == ["auto"]:
            thr_list = [None]

        loop("duckdb", ducks, threads_for_duck=thr_list)
        # SQLite idx/noidx
        loop("sqlite", sqli_idx)
        loop("sqlite", sqli_noi)

    # ---- Worst-case time estimate ----
    if a.max_run_s is not None:
        total_sec = planned * max(a.max_run_s, 1)
        hrs = int(total_sec // 3600)
        mins = int((total_sec % 3600) // 60)
        secs = int(total_sec % 60)
        print(f"[estimate] worst-case wall time (sequential): {planned} runs × {a.max_run_s}s = {total_sec}s  (~{hrs}h {mins}m {secs}s)")
        print("           (DB autogen/IO overhead not included; real time is usually less)")
    else:
        print("[estimate] no --max-run-s → worst-case unbounded (depends on data + queries)")

    print(f"=== DONE RUNS === planned={planned} executed={executed} (including skips/dry-run)")

    if not a.no_summarize:
        print("=== BUILD ARTIFACTS ===")
        call_module("analysis.summarize_results")
        call_module("analysis.make_dashboard")
        call_module("analysis.audit_results")
        if a.diagnose:
            call_module("analysis.diagnose")

def ensure_dbs(rows_per_tag: int, tags: int) -> None:
    csv = Path(DB_DIR).parent / f"syn_{rows_per_tag}x{tags}.csv"
    pq  = Path(DB_DIR).parent / f"syn_{rows_per_tag}x{tags}.parquet"
    duck = Path(DB_DIR) / f"duck_{rows_per_tag}x{tags}.db"
    s_idx = Path(DB_DIR) / f"sqlite_{rows_per_tag}x{tags}_idx.db"
    s_noi = Path(DB_DIR) / f"sqlite_{rows_per_tag}x{tags}_noidx.db"

    # generate data if needed
    if not csv.exists() or not pq.exists():
        cmd = [sys.executable, "-m", "generator.synth",
               "--rows", str(rows_per_tag), "--tags", str(tags),
               "--freq-ms", "1000", "--seed", "42"]
        print("+ " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    # load DBs if missing
    if not s_noi.exists():
        cmd = [sys.executable, "-m", "loader.sqlite_loader", str(csv), "--db", str(s_noi)]
        print("+ " + " ".join(cmd)); subprocess.run(cmd, check=True)
    if not s_idx.exists():
        cmd = [sys.executable, "-m", "loader.sqlite_loader", str(csv), "--db", str(s_idx), "--extra-index"]
        print("+ " + " ".join(cmd)); subprocess.run(cmd, check=True)
    if not duck.exists():
        cmd = [sys.executable, "-m", "loader.duckdb_loader", str(pq), "--db", str(duck)]
        print("+ " + " ".join(cmd)); subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
