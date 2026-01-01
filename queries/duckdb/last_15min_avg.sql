-- DuckDB: letzter 15-Minuten-Horizont (Overflow-sicher)
SELECT
  tag,
  AVG(value) AS avg15
FROM measurements
WHERE ts_ns >= (
  SELECT MAX(ts_ns) - 900000000000::BIGINT  -- 15 * 60 * 1e9 als BIGINT
  FROM measurements
)
GROUP BY tag
ORDER BY tag;
