# analysis/real_prepare_syn.py
import argparse, json, os
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to real parquet (ts_ns,tag,value)")
    ap.add_argument("--rows-per-tag", default="10000,20000,50000,100000,150000")
    ap.add_argument("--out-prefix", default="syn", help="Must stay 'syn' for run_matrix compatibility")
    a = ap.parse_args()

    edge_root = Path(os.environ.get("EDGE_ROOT", "/mnt/edgebench")).resolve()
    data_dir = edge_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    src = Path(a.src)
    if not src.is_absolute():
        src = (edge_root / a.src).resolve()
    if not src.exists():
        raise SystemExit(f"[error] src not found: {src}")

    sizes = [int(x.strip()) for x in a.rows_per_tag.split(",") if x.strip()]

    df = pd.read_parquet(src)
    need = {"ts_ns", "tag", "value"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[error] missing columns: need {need}, have {set(df.columns)}")

    df = df.sort_values(["tag", "ts_ns"], kind="mergesort")
    tags = sorted(df["tag"].unique().tolist())
    tag_count = len(tags)
    rpt_min = int(df.groupby("tag").size().min())

    print(f"[ok] src={src}")
    print(f"[ok] tags={tag_count} ({', '.join(tags)})")
    print(f"[ok] rows_total={len(df)} min_rows_per_tag={rpt_min}")

    manifest = {
        "edge_root": str(edge_root),
        "src": str(src),
        "tags": tag_count,
        "tag_names": tags,
        "rows_total": int(len(df)),
        "min_rows_per_tag": rpt_min,
        "slices": []
    }

    for n in sizes:
        if n > rpt_min:
            print(f"[skip] rpt={n} (not enough rows)")
            continue

        sl = pd.concat([g.head(n) for _, g in df.groupby("tag", sort=False)], ignore_index=True)
        sl = sl.sort_values(["tag", "ts_ns"], kind="mergesort")

        # IMPORTANT: file names must match run_matrix gen_dataset() expectations
        csv_path = data_dir / f"{a.out_prefix}_{n}x{tag_count}.csv"
        pq_path  = data_dir / f"{a.out_prefix}_{n}x{tag_count}.parquet"

        sl.to_parquet(pq_path, index=False)
        sl.to_csv(csv_path, index=False)

        meta = {
            "rows_per_tag": n,
            "tags": tag_count,
            "rows_total": int(len(sl)),
            "csv": str(csv_path),
            "parquet": str(pq_path),
            "span_h": float((sl["ts_ns"].max() - sl["ts_ns"].min()) / 1e9 / 3600.0),
        }
        manifest["slices"].append(meta)

        print(f"[wrote] {csv_path.name}")
        print(f"[wrote] {pq_path.name}")

    (data_dir / "real_as_syn_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ok] wrote {data_dir/'real_as_syn_manifest.json'}")

if __name__ == "__main__":
    main()
