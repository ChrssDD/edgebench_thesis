-- SQLite: letzter 15-Minuten-Horizont
SELECT tag, AVG(value) AS avg15
FROM measurements
WHERE ts_ns >= (SELECT MAX(ts_ns) - (15 * 60 * 1000000000) FROM measurements)
GROUP BY tag
ORDER BY tag;
