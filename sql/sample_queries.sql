-- Sample queries for local validation.
--
-- The analytics queries below are examples for producing the checked-in
-- sample outputs under data/sample_outputs/. They are not a production
-- text-to-SQL agent and are not executed by the Streamlit app.

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

-- Sample output: data/sample_outputs/top_complaint_types.csv
SELECT
    complaint_type,
    COUNT(*) AS request_count
FROM nyc_311_service_requests_clean
GROUP BY complaint_type
ORDER BY request_count DESC
LIMIT 5;

-- Sample output: data/sample_outputs/requests_by_borough.csv
SELECT
    borough,
    COUNT(*) AS request_count
FROM nyc_311_service_requests_clean
GROUP BY borough
ORDER BY request_count DESC;

-- Sample output: data/sample_outputs/agency_request_volume.csv
SELECT
    agency,
    agency_name,
    COUNT(*) AS request_count
FROM nyc_311_service_requests_clean
GROUP BY agency, agency_name
ORDER BY request_count DESC
LIMIT 5;

-- Sample output: data/sample_outputs/backlog_summary.csv
SELECT
    status,
    COUNT(*) AS request_count
FROM nyc_311_service_requests_clean
WHERE status IN ('Open', 'In Progress', 'Overdue', 'Closed Last 7 Days')
GROUP BY status
ORDER BY request_count DESC;
