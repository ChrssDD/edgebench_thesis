#/mnt/edgebench/src/edgebench/tests/test_generator_determinism.py
from __future__ import annotations

from pathlib import Path

from conftest import run_module, sha256_file


def test_generator_is_deterministic(edge_env, tmp_path: Path):
    # fixed t0_ns -> no dependence on wall-clock time
    args = ["--rows", "900", "--tags", "20", "--freq-ms", "1000", "--seed", "42", "--t0-ns", "1700000000000000000"]
    p1 = run_module("generator.synth", args, env=edge_env)
    assert p1.returncode == 0, p1.stderr

    csv1 = tmp_path / "data" / "syn_900x20.csv"
    pq1  = tmp_path / "data" / "syn_900x20.parquet"
    assert csv1.exists()
    assert pq1.exists()

    h_csv_1 = sha256_file(csv1)
    h_pq_1  = sha256_file(pq1)

    # run again into SAME EDGE_ROOT; file should be identical (overwrite with same bytes)
    p2 = run_module("generator.synth", args, env=edge_env)
    assert p2.returncode == 0, p2.stderr

    h_csv_2 = sha256_file(csv1)
    h_pq_2  = sha256_file(pq1)

    assert h_csv_1 == h_csv_2
    assert h_pq_1 == h_pq_2
