#/mnt/edgebench/src/edgebench/tests/test_queries_equivalence.py
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import duckdb

from conftest import run_module

NS_1MIN  = 60_000_000_000
NS_15MIN = 900_000_000_000
NS_60MIN = 3_600_000_000_000


def _round_rows(rows: list[tuple], nd: int = 9) -> list[tuple]:
    out = []
    for r in rows:
        rr = []
        for v in r:
            if isinstance(v, float):
                rr.append(round(v, nd))
            else:
                rr.append(v)
        out.append(tuple(rr))
    return out


def _sqlite_fetch(db: Path, sql: str) -> list[tuple]:
    con = sqlite3.connect(str(db))
    try:
        cur = con.cursor()
        cur.execute(sql)
        return cur.fetchall()
    finally:
        con.close()


def _duck_fetch(db: Path, sql: str) -> list[tuple]:
    con = duckdb.connect(str(db))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _sort(rows: list[tuple]) -> list[tuple]:
    return sorted(rows, key=lambda x: tuple(str(v) for v in x))


def _assert_equal(a: list[tuple], b: list[tuple]):
    assert len(a) == len(b)
    for i, (ra, rb) in enumerate(zip(a, b)):
        assert len(ra) == len(rb)
        for j, (va, vb) in enumerate(zip(ra, rb)):
            if isinstance(va, (int, str)) and isinstance(vb, (int, str)):
                assert va == vb
            elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                fa = float(va); fb = float(vb)
                assert math.isfinite(fa) and math.isfinite(fb)
                assert abs(fa - fb) <= 1e-6 + 1e-6 * abs(fb)
            else:
                assert str(va) == str(vb)


def sql_downsample_15min() -> str:
    return f"""
    SELECT
      (ts_ns - (ts_ns % {NS_15MIN})) AS bucket_start_ns,
      tag,
      AVG(value) AS avg_v,
      MIN(value) AS min_v,
      MAX(value) AS max_v
    FROM measurements
    GROUP BY 1,2
    ORDER BY 1,2;
    """.strip()


SQL_LAST_15_SQLITE = f"""
SELECT tag, AVG(value) AS avg15
FROM measurements
WHERE ts_ns >= (SELECT MAX(ts_ns) - {NS_15MIN} FROM measurements)
GROUP BY tag
ORDER BY tag;
""".strip()

SQL_LAST_15_DUCK = f"""
SELECT tag, AVG(value) AS avg15
FROM measurements
WHERE ts_ns >= (SELECT MAX(ts_ns) - {NS_15MIN}::BIGINT FROM measurements)
GROUP BY tag
ORDER BY tag;
""".strip()

SQL_LAST_60_SQLITE = f"""
SELECT tag, AVG(value) AS avg60
FROM measurements
WHERE ts_ns >= (SELECT MAX(ts_ns) - {NS_60MIN} FROM measurements)
GROUP BY tag
ORDER BY tag;
""".strip()

SQL_LAST_60_DUCK = f"""
SELECT tag, AVG(value) AS avg60
FROM measurements
WHERE ts_ns >= (SELECT MAX(ts_ns) - {NS_60MIN}::BIGINT FROM measurements)
GROUP BY tag
ORDER BY tag;
""".strip()


def test_queries_match_duckdb_vs_sqlite(edge_env, tmp_path: Path):
    # generate + load both DBs
    gen = run_module("generator.synth",
                     ["--rows", "900", "--tags", "20", "--freq-ms", "1000", "--seed", "42", "--t0-ns", "1700000000000000000"],
                     env=edge_env)
    assert gen.returncode == 0, gen.stderr

    csv = tmp_path / "data" / "syn_900x20.csv"
    pq  = tmp_path / "data" / "syn_900x20.parquet"

    sqlite_db = tmp_path / "data" / "db" / "sqlite_900x20_idx.db"
    duck_db   = tmp_path / "data" / "db" / "duck_900x20.db"

    p1 = run_module("loader.sqlite_loader", [str(csv), "--db", str(sqlite_db), "--extra-index"], env=edge_env)
    assert p1.returncode == 0, p1.stderr

    p2 = run_module("loader.duckdb_loader", [str(pq), "--db", str(duck_db)], env=edge_env)
    assert p2.returncode == 0, p2.stderr

    # downsample 15min
    a = _round_rows(_sort(_duck_fetch(duck_db, sql_downsample_15min())))
    b = _round_rows(_sort(_sqlite_fetch(sqlite_db, sql_downsample_15min())))
    _assert_equal(a, b)

    # last 15 min
    a = _round_rows(_sort(_duck_fetch(duck_db, SQL_LAST_15_DUCK)))
    b = _round_rows(_sort(_sqlite_fetch(sqlite_db, SQL_LAST_15_SQLITE)))
    _assert_equal(a, b)

    # last 60 min
    a = _round_rows(_sort(_duck_fetch(duck_db, SQL_LAST_60_DUCK)))
    b = _round_rows(_sort(_sqlite_fetch(sqlite_db, SQL_LAST_60_SQLITE)))
    _assert_equal(a, b)
