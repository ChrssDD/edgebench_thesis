#analysis/validate_results.py
import argparse, sqlite3, duckdb, pandas as pd
from config import QUERIES_DIR

def df_for_sqlite(db, sql):  return pd.read_sql_query(sql, sqlite3.connect(db))
def df_for_duckdb(db, sql):  return duckdb.connect(db).execute(sql).df()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--sqlite-db", required=True)
    ap.add_argument("--duckdb-db", required=True)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    a = ap.parse_args()

    sql_s = (QUERIES_DIR/"sqlite"/f"{a.query}.sql").read_text()
    sql_d = (QUERIES_DIR/"duckdb"/f"{a.query}.sql").read_text()

    s = df_for_sqlite(a.sqlite_db, sql_s)
    d = df_for_duckdb(a.duckdb_db, sql_d)

    # align by common columns, sort same order
    common = [c for c in s.columns if c in d.columns]
    s = s[common].sort_values(common).reset_index(drop=True)
    d = d[common].sort_values(common).reset_index(drop=True)

    ok = True; issues=[]
    if list(s.columns)!=list(d.columns):
        ok=False; issues.append(f"Columns differ: {list(s.columns)} vs {list(d.columns)}")
    elif len(s)!=len(d):
        ok=False; issues.append(f"Row count differ: {len(s)} vs {len(d)}")
    else:
        for col in common:
            if pd.api.types.is_numeric_dtype(s[col]):
                diff = (s[col]-d[col]).abs()
                if not (diff <= (a.atol + a.rtol*d[col].abs())).all():
                    ok=False; issues.append(f"Numeric mismatch in {col} (max diff {diff.max()})")
            else:
                if not (s[col].astype(str)==d[col].astype(str)).all():
                    ok=False; issues.append(f"Non-numeric mismatch in {col}")

    print("OK" if ok else "MISMATCH", "; ".join(issues))

if __name__=="__main__":
    main()
