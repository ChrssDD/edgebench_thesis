#/mnt/edgebench/src/edgebench/tests/test_analysis_summarize_results.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from conftest import run_module


def test_summarize_results_creates_summary(edge_env, tmp_path: Path):
    # create minimal results dir with one valid run json
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_obj = {
        "engine": "sqlite",
        "query": "downsample_15min",
        "warm": True,
        "threads": None,
        "scan_mode": "table",
        "db": str(tmp_path / "data" / "db" / "sqlite_900x20_idx.db"),
        "durations_ns": [1_000_000, 1_100_000, 1_200_000, 1_050_000, 1_090_000, 1_110_000, 1_130_000],
        "stats": {"p50_ns": 1_100_000, "p95_ns": 1_200_000},
        "dataset_meta": {"rows": 18_000, "tags": 20},
        "iterations": [
            {"dur_ns": 1_000_000, "psi_before": {"cpu": "", "io": ""}, "psi_after": {"cpu": "", "io": ""}}
        ]
    }

    (results_dir / "sqlite_900x20_idx_sqlite_downsample_15min_warm_tbl.json").write_text(json.dumps(run_obj))

    # run analysis.summarize_results
    p = run_module("analysis.summarize_results", [], env=edge_env)
    assert p.returncode == 0, p.stderr

    out = results_dir / "summary_with_cis.csv"
    assert out.exists()

    df = pd.read_csv(out)
    # minimal expected columns
    for c in ["engine", "query", "warm", "scan_mode", "p50_ms", "p95_ms", "dataset_rows", "dataset_tags"]:
        assert c in df.columns
