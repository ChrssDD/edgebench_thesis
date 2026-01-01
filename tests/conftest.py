#/mnt/edgebench/src/edgebench/tests/conftest.py
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest


REPO_ROOT = Path("/mnt/edgebench/src/edgebench").resolve()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_module(module: str, args: List[str], env: Dict[str, str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", module] + args
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)


@pytest.fixture()
def edge_env(tmp_path: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["EDGE_ROOT"] = str(tmp_path)
    env["PYTHONHASHSEED"] = "0"
    return env
