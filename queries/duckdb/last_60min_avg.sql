-- DuckDB: letzter 60-Minuten-Horizont (Overflow-sicher)
SELECT
  tag,
  AVG(value) AS avg60
FROM measurements
WHERE ts_ns >= (
  SELECT MAX(ts_ns) - 3600000000000::BIGINT  -- 60 * 60 * 1e9 als BIGINT
  FROM measurements
)
GROUP BY tag
ORDER BY tag;
