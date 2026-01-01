#/mnt/edgebench/src/edgebench/tests/test_loader_duckdb.py
from __future__ import annotations

from pathlib import Path
import duckdb

from conftest import run_module


def test_duckdb_loader_creates_table(edge_env, tmp_path: Path):
    gen = run_module("generator.synth",
                     ["--rows", "900", "--tags", "20", "--freq-ms", "1000", "--seed", "42", "--t0-ns", "1700000000000000000"],
                     env=edge_env)
    assert gen.returncode == 0, gen.stderr

    pq = tmp_path / "data" / "syn_900x20.parquet"
    assert pq.exists()

    db = tmp_path / "data" / "db" / "duck_900x20.db"

    p = run_module("loader.duckdb_loader", [str(pq), "--db", str(db)], env=edge_env)
    assert p.returncode == 0, p.stderr
    assert db.exists()

    con = duckdb.connect(str(db))
    try:
        n = con.execute("SELECT COUNT(*) FROM measurements;").fetchone()[0]
        assert int(n) == 900 * 20
    finally:
        con.close()
