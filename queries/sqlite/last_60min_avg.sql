-- SQLite: letzter 60-Minuten-Horizont
SELECT tag, AVG(value) AS avg60
FROM measurements
WHERE ts_ns >= (SELECT MAX(ts_ns) - (60 * 60 * 1000000000) FROM measurements)
GROUP BY tag
ORDER BY tag;
