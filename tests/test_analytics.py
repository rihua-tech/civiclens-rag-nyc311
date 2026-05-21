from src.analytics.simple_analytics import answer_analytics_question, load_sample_output


def test_sample_analytics_outputs_can_be_loaded():
    sample_files = [
        "top_complaint_types.csv",
        "requests_by_borough.csv",
        "agency_request_volume.csv",
        "backlog_summary.csv",
    ]

    for sample_file in sample_files:
        rows = load_sample_output(sample_file)
        assert rows


def test_top_complaint_types_answer_returns_useful_text():
    response = answer_analytics_question("What are the top complaint types?")

    assert response["mode"] == "analytics"
    assert "top sample complaint types" in response["answer"]
    assert "Noise - Residential" in response["answer"]
    assert response["sources"][0]["source_path"] == "data/sample_outputs/top_complaint_types.csv"


def test_borough_volume_answer_returns_useful_text():
    response = answer_analytics_question("Which borough has the highest complaint volume?")

    assert response["mode"] == "analytics"
    assert "Brooklyn" in response["answer"]
    assert "highest sample complaint volume" in response["answer"]
    assert response["sources"][0]["source_path"] == "data/sample_outputs/requests_by_borough.csv"


def test_unknown_analytics_question_returns_safe_fallback():
    response = answer_analytics_question("What is the average close time by hour?")

    assert response["mode"] == "fallback"
    assert "predefined sample analytics questions" in response["answer"]
    assert response["sources"] == []
    assert response["retrieved_chunks"] == []
