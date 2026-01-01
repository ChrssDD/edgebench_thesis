#!/usr/bin/env python3
# analysis/image/ram_report.py
# Full version embedded here (see previous attempt for detailed docstring)
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

REQ_SUM_COLS = {"engine","query","warm","threads","scan_mode","dataset_rows"}
REQ_BE_COLS  = {"query","warm","sqlite_variant","duckdb_profile","break_even_rows"}
REQ_BEM_COLS = {"query","warm","metric","sqlite_variant","duckdb_profile","be_points"}

def need_cols(df, need, where):
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{where} missing columns: {miss}")

def norm_warm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["1","true","yes","warm"])

def parse_bem_points(s: str):
    pts = []
    if not isinstance(s, str):
        return pts
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok: continue
        try:
            pts.append(float(tok))
        except Exception as e:
            print(f"[warn] break_even_multi: could not parse '{tok}': {e}")
    return sorted(pts)

from dataclasses import dataclass
@dataclass
class Camp:
    label: str
    root: Path
    summary: Path
    be: Path
    bem: Path

def load_camp(root: Path, label: str) -> Camp:
    return Camp(
        label=label,
        root=root,
        summary=root / "results" / "summary_with_cis.csv",
        be=root / "results" / "break_even.csv",
        bem=root / "results" / "break_even_multi.csv",
    )

def load_summary(p: Path, tags: int|None) -> pd.DataFrame:
    df = pd.read_csv(p)
    need_cols(df, REQ_SUM_COLS, str(p))
    if tags is not None and "dataset_tags" in df.columns:
        df = df[df["dataset_tags"] == tags]
    return df

def load_be(p: Path, tags: int|None, metric: str) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    be = pd.read_csv(p)
    if not REQ_BE_COLS.issubset(be.columns):
        return pd.DataFrame()
    be["warm"] = norm_warm(be["warm"])
    if "metric" not in be.columns:
        be["metric"] = metric
    if tags is not None and "tags" in be.columns:
        be = be[be["tags"] == tags]
    return be

def load_bem(p: Path, tags: int|None, metric: str) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if not REQ_BEM_COLS.issubset(df.columns):
        return pd.DataFrame()
    df["warm"] = norm_warm(df["warm"])
    if "metric" not in df.columns:
        df["metric"] = metric
    if tags is not None and "tags" in df.columns:
        df = df[df["tags"] == tags]
    df["be_list"] = df["be_points"].apply(parse_bem_points)
    df["be_count"] = df["be_list"].apply(len)
    return df

def duck_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[(df["engine"]=="duckdb") & (df["scan_mode"]=="table")].copy()
    if sub.empty:
        return pd.DataFrame()
    auto = (sub[sub["threads"].isna()]
            .groupby(["query","warm","dataset_rows"], as_index=True)[metric].median()
            .rename("auto"))
    t1   = (sub[sub["threads"]==1]
            .groupby(["query","warm","dataset_rows"], as_index=True)[metric].median()
            .rename("t1"))
    wide = pd.concat([auto, t1], axis=1)
    wide.index = pd.MultiIndex.from_tuples(wide.index, names=["query","warm","dataset_rows"])
    return wide.sort_index()

def ratio_area_above_one(base_wide: pd.DataFrame, cap_wide: pd.DataFrame,
                         col: str, *, warm: bool) -> pd.DataFrame:
    out = []
    base_qs = {ix[0] for ix in base_wide.index}
    cap_qs  = {ix[0] for ix in cap_wide.index}
    qs = sorted(base_qs | cap_qs)
    for q in qs:
        try:
            bq = base_wide.xs((q, warm), level=(0,1))
            cq = cap_wide.xs((q, warm), level=(0,1))
        except KeyError:
            continue
        idx = sorted(set(bq.index) | set(cq.index))
        if not idx: continue
        b = bq.reindex(idx)[col]
        c = cq.reindex(idx)[col]
        r = (c / b).astype(float)
        r = r.replace([np.inf, -np.inf], float("nan")).dropna()
        if len(r) < 2:
            area = 0.0
        else:
            xs = np.array(list(r.index), dtype=float)
            ys = np.maximum(0.0, r.values - 1.0)
            lx = np.log10(np.maximum(xs, 1.0))
            area = float(np.trapz(ys, lx))
        out.append({"query": q, "area": area})
    return pd.DataFrame(out).sort_values("area", ascending=False)

def be_shift_scores(base_be: pd.DataFrame, cap_be: pd.DataFrame,
                    metric: str, warm: bool) -> pd.DataFrame:
    bb = base_be[(base_be["metric"]==metric) & (base_be["warm"]==warm) & (base_be["sqlite_variant"]=="best")]
    cc = cap_be [(cap_be ["metric"]==metric) & (cap_be ["warm"]==warm) & (cap_be ["sqlite_variant"]=="best")]
    if bb.empty or cc.empty:
        return pd.DataFrame()
    rows = []
    for prof in ("auto","t1"):
        b = bb[bb["duckdb_profile"]==prof][["query","break_even_rows"]].rename(columns={"break_even_rows":"be_base"})
        c = cc[cc["duckdb_profile"]==prof][["query","break_even_rows"]].rename(columns={"break_even_rows":"be_cap"})
        j = b.merge(c, on="query", how="outer")
        j["delta"] = j["be_cap"] - j["be_base"]
        rows.append(j.assign(profile=prof))
    out = pd.concat(rows, ignore_index=True)
    out["abs_delta"] = out["delta"].abs()
    return out

def presence_changes(base_bem: pd.DataFrame, cap_bem: pd.DataFrame,
                     metric: str, warm: bool, k:int=1) -> pd.DataFrame:
    ba = base_bem[(base_bem["metric"]==metric) & (base_bem["warm"]==warm) & (base_bem["sqlite_variant"]=="best")]
    ca = cap_bem [(cap_bem ["metric"]==metric) & (cap_bem ["warm"]==warm) & (cap_bem ["sqlite_variant"]=="best")]
    rows = []
    for prof in ("auto","t1"):
        b = ba[ba["duckdb_profile"]==prof][["query","be_list"]].copy()
        c = ca[ca["duckdb_profile"]==prof][["query","be_list"]].copy()
        j = b.merge(c, on="query", how="outer", suffixes=("_base","_cap"))
        def kth(lst, k):
            if isinstance(lst, list) and len(lst)>=k:
                return lst[k-1]
            return float("nan")
        j["base_has"] = j["be_list_base"].apply(lambda lst: isinstance(lst, list) and len(lst)>=k)
        j["cap_has"]  = j["be_list_cap"].apply(lambda lst: isinstance(lst, list) and len(lst)>=k)
        j["base_k"]   = j["be_list_base"].apply(lambda lst: kth(lst,k))
        j["cap_k"]    = j["be_list_cap"].apply(lambda lst: kth(lst,k))
        j["delta_k"]  = j["cap_k"] - j["base_k"]
        rows.append(j.assign(profile=prof))
    return pd.concat(rows, ignore_index=True)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--cap", action="append", required=True, help="LABEL=PATH")
    ap.add_argument("--tags", type=int, default=None)
    ap.add_argument("--metric", default="p95_ms")
    ap.add_argument("--warm", type=int, default=0, choices=[0,1])
    ap.add_argument("--mode", default="t1", choices=["auto","t1"])
    ap.add_argument("--outdir", default="ram_report")
    args = ap.parse_args()

    outdir = Path(args.outdir); (outdir/"report").mkdir(parents=True, exist_ok=True)

    def _load(root, label):
        return {
            "label": label,
            "root": Path(root),
            "summary": Path(root)/"results"/"summary_with_cis.csv",
            "be": Path(root)/"results"/"break_even.csv",
            "bem": Path(root)/"results"/"break_even_multi.csv",
        }

    base = _load(args.baseline, "baseline")
    bsum = load_summary(base["summary"], args.tags)
    bbe  = load_be(base["be"],  args.tags, args.metric)
    bbem = load_bem(base["bem"], args.tags, args.metric)
    b_duck = duck_pivot(bsum, args.metric)

    overview_rows = []
    topq_rows = []

    for s in args.cap:
        if "=" not in s:
            raise SystemExit(f"--cap needs LABEL=PATH, got: {s}")
        label, path = s.split("=", 1)
        camp = _load(path, label)
        if not camp["summary"].exists():
            print(f"[warn] skip {label}: missing {camp['summary']}"); continue
        csum = load_summary(camp["summary"], args.tags)
        cbe  = load_be(camp["be"],  args.tags, args.metric)
        cbem = load_bem(camp["bem"], args.tags, args.metric)

        be_df = be_shift_scores(bbe, cbe, args.metric, bool(args.warm))
        total_abs = float(be_df["abs_delta"].sum()) if not be_df.empty else 0.0
        mean_abs  = float(be_df["abs_delta"].mean()) if not be_df.empty else 0.0
        n_valid   = int(be_df["abs_delta"].notna().sum()) if not be_df.empty else 0

        pres = presence_changes(bbem, cbem, args.metric, bool(args.warm), k=1)
        disappeared = int(((pres["base_has"]==True) & (pres["cap_has"]==False)).sum()) if not pres.empty else 0
        appeared    = int(((pres["base_has"]==False) & (pres["cap_has"]==True)).sum()) if not pres.empty else 0

        c_duck = duck_pivot(csum, args.metric)
        if not b_duck.empty and not c_duck.empty:
            ra = ratio_area_above_one(b_duck, c_duck, args.mode, warm=bool(args.warm))
            top = ra.head(3).copy(); top["cap_label"] = label
            topq_rows.extend(top.assign(mode=args.mode).to_dict(orient="records"))
            ratio_total = float(ra["area"].sum())
            ratio_top1  = float(top["area"].iloc[0]) if len(top) else 0.0
        else:
            ratio_total = 0.0
            ratio_top1  = 0.0

        overview_rows.append({
            "cap": label, "metric": args.metric, "warm": args.warm, "mode": args.mode,
            "be_total_abs_delta_rows": total_abs,
            "be_mean_abs_delta_rows":  mean_abs,
            "be_num_valid": n_valid,
            "multi_k1_disappeared": disappeared,
            "multi_k1_appeared": appeared,
            "ratio_area_sum": ratio_total,
            "ratio_area_top1": ratio_top1,
        })

    ov = pd.DataFrame(overview_rows).sort_values(["metric","warm","cap"])
    oq = pd.DataFrame(topq_rows)

    (outdir/"report"/"overview.csv").write_text(ov.to_csv(index=False))
    (outdir/"report"/"top_queries.csv").write_text(oq.to_csv(index=False))

    lines = []
    for _, r in ov.iterrows():
        cap = r["cap"]; warm = int(r["warm"]); metric = r["metric"]; mode = r["mode"]
        wtag = "warm" if warm==1 else "cold"
        lines.append(f"% BE-Shift (single), {metric}, {wtag}, cap={cap}")
        lines.append(fr"\includegraphics[width=\linewidth]{{figs/ram/be_shift/{metric}/{wtag}/best/{cap}/be_shift__best__{cap}__{metric}__w{warm}.png}}")
        if not oq.empty:
            top_for_cap = oq[oq["cap_label"]==cap].sort_values("area", ascending=False).head(1)
            if not top_for_cap.empty:
                q = top_for_cap.iloc[0]["query"]
                lines.append(f"% Ratio (DuckDB {mode}), top query = {q}")
                lines.append(fr"\includegraphics[width=\linewidth]{{figs/ram/ratio/{metric}/{wtag}/{mode}/{cap}/{q}__{mode}__ramcap_{cap}_over_base__{metric}.png}}")
        lines.append("")
    (outdir/"report"/"latex_snippets.tex").write_text("\n".join(lines))

    md = []
    md.append(f"# RAM-cap Auswahlbericht\n")
    md.append(f"- Baseline: `{args.baseline}`")
    md.append(f"- Metric: `{args.metric}`  Warm: `{args.warm}`  Mode: `{args.mode}`  Tags: `{args.tags}`\n")
    if not ov.empty:
        md.append("## Kennzahlen pro Cap\n")
        md.append(ov.to_markdown(index=False))
        md.append("")
    if not oq.empty:
        md.append("## Top-Queries (Ratio-Fläche über 1.0)\n")
        md.append(oq.sort_values(['cap_label','area'], ascending=[True,False]).to_markdown(index=False))
        md.append("")
    (outdir/"report"/"README.md").write_text("\n".join(md))

    print("[ok] wrote:")
    for p in ["overview.csv","top_queries.csv","latex_snippets.tex","README.md"]:
        print(" -", (outdir/"report"/p))

if __name__ == "__main__":
    main()
