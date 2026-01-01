#/mnt/edgebench/src/edgebench/tests/test_artifact_schema.py
from __future__ import annotations

import json
from pathlib import Path


def test_expected_artifact_fixture_schema():
    p = Path(__file__).parent / "fixtures" / "expected_artifact_min.json"
    obj = json.loads(p.read_text())

    required = [
        "engine", "query", "warm", "scan_mode", "durations_ns", "stats", "dataset_meta", "iterations"
    ]
    for k in required:
        assert k in obj

    assert obj["engine"] in {"sqlite", "duckdb"}
    assert isinstance(obj["durations_ns"], list) and len(obj["durations_ns"]) > 0
    assert isinstance(obj["stats"], dict)
    assert isinstance(obj["dataset_meta"], dict)
    assert isinstance(obj["iterations"], list)
