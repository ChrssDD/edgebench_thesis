-- SQLite: echtes 10-Minuten-Gleitfenster (Minutenbasis) + diskreter p95, stabil & deterministisch
WITH minutes AS (
  SELECT
    (ts_ns - (ts_ns % 60000000000)) AS bucket_min_ns,
    tag,
    ROUND(AVG(value), 9) AS avg_v              -- stabilize AVG
  FROM measurements
  GROUP BY 1,2
),
mean_win AS (
  SELECT
    bucket_min_ns,
    tag,
    AVG(avg_v) OVER (
      PARTITION BY tag
      ORDER BY bucket_min_ns
      ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS mean10
  FROM minutes
),
frame AS (
  SELECT
    m1.bucket_min_ns AS t,
    m1.tag,
    m2.bucket_min_ns AS t2,
    m2.avg_v,
    ROW_NUMBER() OVER (
      PARTITION BY m1.tag, m1.bucket_min_ns
      ORDER BY m2.avg_v, m2.bucket_min_ns, m2.tag     -- deterministic
    ) AS rn,
    COUNT(*) OVER (
      PARTITION BY m1.tag, m1.bucket_min_ns
    ) AS n
  FROM minutes m1
  JOIN minutes m2
    ON m2.tag = m1.tag
   AND m2.bucket_min_ns BETWEEN m1.bucket_min_ns - (9*60000000000) AND m1.bucket_min_ns
),
p95_idx AS (
  -- correct discrete p95 index: ceil(n*0.95) but without CEIL()
  -- For integer n: ceil(n*95/100) == (n*95 + 99)/100 using integer division
  SELECT
    t,
    tag,
    CAST( ((MAX(n) * 95) + 99) / 100 AS INT ) AS k
  FROM frame
  GROUP BY 1,2
),
p95_win AS (
  SELECT
    f.t AS bucket_min_ns,
    f.tag,
    f.avg_v AS p95
  FROM frame f
  JOIN p95_idx k
    ON k.t = f.t AND k.tag = f.tag AND k.k = f.rn
)
SELECT m.bucket_min_ns, m.tag, mw.mean10, pw.p95
FROM (SELECT DISTINCT bucket_min_ns, tag FROM minutes) AS m
JOIN mean_win mw ON mw.bucket_min_ns = m.bucket_min_ns AND mw.tag = m.tag
JOIN p95_win pw ON pw.bucket_min_ns = m.bucket_min_ns AND pw.tag = m.tag
ORDER BY 1,2;
