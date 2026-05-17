-- Sample queries for local validation.

SELECT COUNT(*) AS document_count
FROM documents;

SELECT COUNT(*) AS chunk_count
FROM chunks;

SELECT
    source_name,
    COUNT(*) AS chunks
FROM chunks
GROUP BY source_name
ORDER BY chunks DESC;
