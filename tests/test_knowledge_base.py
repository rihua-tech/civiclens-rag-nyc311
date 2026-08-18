from pathlib import Path

from src.ingestion.load_documents import load_source_manifest


def test_repository_manifest_is_small_valid_and_distinguishes_source_categories():
    sources = load_source_manifest()

    assert 2 <= len(sources) <= 10
    categories = {source["source_category"] for source in sources}
    assert "external_nyc311" in categories
    assert "civiclens_project" in categories
    assert all(str(source["path"]).startswith(("docs/", "README.md")) for source in sources)
    assert all(not str(source["path"]).startswith("data/raw/") for source in sources)


def test_curated_field_guide_covers_required_nyc311_concepts():
    field_guide = Path("docs/knowledge/nyc311-service-request-fields.md").read_text(
        encoding="utf-8"
    ).lower()

    required_terms = [
        "complaint type",
        "closed date",
        "status",
        "agency",
        "borough",
        "location",
        "created date",
        "resolution action updated date",
    ]

    for term in required_terms:
        assert term in field_guide


def test_knowledge_directory_contains_no_large_files():
    knowledge_files = [path for path in Path("docs/knowledge").rglob("*") if path.is_file()]

    assert knowledge_files
    assert all(path.stat().st_size < 250_000 for path in knowledge_files)
