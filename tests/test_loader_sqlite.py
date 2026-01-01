#/mnt/edgebench/src/edgebench/tests/test_loader_sqlite.py
from __future__ import annotations

import sqlite3
from pathlib import Path

from conftest import run_module


def _index_names(db: Path) -> set[str]:
    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
        return {r[0] for r in cur.fetchall()}
    finally:
        con.close()


def _rowcount(db: Path) -> int:
    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM measurements;")
        return int(cur.fetchone()[0])
    finally:
        con.close()


def test_sqlite_loader_creates_db_and_optional_index(edge_env, tmp_path: Path):
    # generate data
    gen = run_module("generator.synth",
                     ["--rows", "900", "--tags", "20", "--freq-ms", "1000", "--seed", "42", "--t0-ns", "1700000000000000000"],
                     env=edge_env)
    assert gen.returncode == 0, gen.stderr

    csv = tmp_path / "data" / "syn_900x20.csv"
    assert csv.exists()

    db_noidx = tmp_path / "data" / "db" / "sqlite_900x20_noidx.db"
    db_idx   = tmp_path / "data" / "db" / "sqlite_900x20_idx.db"

    # load noidx
    p1 = run_module("loader.sqlite_loader", [str(csv), "--db", str(db_noidx)], env=edge_env)
    assert p1.returncode == 0, p1.stderr
    assert db_noidx.exists()

    # load idx
    p2 = run_module("loader.sqlite_loader", [str(csv), "--db", str(db_idx), "--extra-index"], env=edge_env)
    assert p2.returncode == 0, p2.stderr
    assert db_idx.exists()

    # rowcount = rows * tags
    assert _rowcount(db_noidx) == 900 * 20
    assert _rowcount(db_idx)   == 900 * 20

    idxs_no = _index_names(db_noidx)
    idxs_ix = _index_names(db_idx)

    # should exist only in idx DB
    assert "idx_tag_ts" not in idxs_no
    assert "idx_tag_ts" in idxs_ix
