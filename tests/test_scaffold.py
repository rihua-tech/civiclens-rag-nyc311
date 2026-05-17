from pathlib import Path


def test_starter_files_exist():
    repo_root = Path(__file__).resolve().parents[1]
    expected_files = [
        "README.md",
        "requirements.txt",
        "docker-compose.yml",
        "app/streamlit_app.py",
        "sql/schema.sql",
    ]

    missing_files = [path for path in expected_files if not (repo_root / path).is_file()]

    assert missing_files == []
