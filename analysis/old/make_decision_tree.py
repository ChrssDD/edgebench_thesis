import math
import pandas as pd
from pathlib import Path
from config import RESULTS_DIR

def main():
    p = RESULTS_DIR / "break_even.csv"
    if not p.exists():
        print("[err] break_even.csv not found. Run summarize_results first.")
        return
    be = pd.read_csv(p)
    if "duckdb_profile" not in be.columns:
        be["duckdb_profile"] = "auto"
    be = be.dropna(subset=["break_even_rows"])

    rows = []
    for (q, warm, prof), g in be.groupby(["query","warm","duckdb_profile"]):
        for variant in ("idx","noidx","best"):
            vrow = g[g["sqlite_variant"] == variant]
            be_val = float(vrow["break_even_rows"].iloc[0]) if not vrow.empty else math.nan
            rows.append({"query": q, "warm": warm, "duckdb_profile": prof,
                         "sqlite_variant": variant, "break_even_rows": be_val})

    out_csv = RESULTS_DIR / "decision_tree_table.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}")

    dot = ['digraph G { rankdir=LR; node [shape=box, fontsize=10];']
    for r in rows:
        label = f"{r['query']}\\nwarm={bool(r['warm'])}\\nprof={r['duckdb_profile']}\\n{r['sqlite_variant']} BE≈{r['break_even_rows']:.0f}"
        name = f"n_{abs(hash((r['query'], r['warm'], r['duckdb_profile'], r['sqlite_variant'])))}"
        dot.append(f'{name} [label="{label}"];')
    dot.append("}")
    out_dot = RESULTS_DIR / "decision_tree.dot"
    out_dot.write_text("\n".join(dot), encoding="utf-8")
    print(f"[ok] wrote {out_dot}  (render with: dot -Tpng decision_tree.dot -o decision_tree.png)")

if __name__ == "__main__":
    main()
