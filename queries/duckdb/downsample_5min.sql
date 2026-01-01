-- DuckDB: 5-Minuten-Tumbling-Buckets
SELECT
  (ts_ns - (ts_ns % 300000000000)) AS bucket_start_ns,  -- 5 min in ns
  tag,
  AVG(value) AS avg_v,
  MIN(value) AS min_v,
  MAX(value) AS max_v
FROM measurements
GROUP BY 1,2
ORDER BY 1,2;
