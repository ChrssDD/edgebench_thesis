# generator/synth.py
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from config import DATA_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--tags", type=int, required=True)
    ap.add_argument("--freq-ms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42, help="RNG-Seed für reproduzierbares Rauschen")
    ap.add_argument("--t0-ns", type=int, default=None, help="Startzeitpunkt der Serie in ns")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Zeitachse (ns): Jetzt - N*Δt ... bis Jetzt (exakt aufsteigend)
    step_ns = int(a.freq_ms * 1_000_000)
    if a.t0_ns is None:
        t_end = time.time_ns()
        t_start = t_end - (a.rows-1)*step_ns
    else:
        t_start = a.t0_ns
        t_end   = t_start + (a.rows-1)*step_ns

    ts_ns = np.arange(t_start, t_end + 1, step_ns, dtype=np.int64)
    
    tags = [f"t{i}" for i in range(a.tags)]
    rows = []
    for i, tag in enumerate(tags):
        phase = i * 0.3
        base  = np.sin(np.linspace(0, 40, a.rows) + phase) * 10 + i*0.5
        noise = rng.normal(0, 0.2, a.rows)
        vals  = (base + noise).astype("float64")
        rows.append(pd.DataFrame({"ts_ns": ts_ns, "tag": tag, "value": vals}))

    df = pd.concat(rows, ignore_index=True)
    csv_path = DATA_DIR / f"syn_{a.rows}x{a.tags}.csv"
    pq_path  = DATA_DIR / f"syn_{a.rows}x{a.tags}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)
    print(f"[ok] wrote {csv_path} and {pq_path}")

if __name__ == "__main__":
    main()
