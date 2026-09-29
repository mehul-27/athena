from backend.config import Settings


def test_research_defaults_match_audited_engine_contract():
    settings = Settings(_env_file=None)
    assert settings.research_max_rounds == 20
    assert settings.research_max_time == 300
    # Lowered from 18: a high floor forced ~18 rounds of LLM work before the
    # stop decision could fire, which is what made free-tier research crawl.
    assert settings.research_min_rounds == 2
    assert settings.research_max_empty_rounds == 2
    assert settings.research_max_stagnant_rounds == 2
    assert settings.research_batch_extraction is True
    assert settings.research_extraction_batch_size == 3
    assert settings.research_max_urls_per_round == 3
    assert settings.research_max_content_chars == 15000
    assert settings.research_max_report_tokens == 16384
    assert settings.research_extraction_timeout_seconds == 90
    assert settings.research_planning_timeout_seconds == 90
    assert settings.research_query_timeout_seconds == 120
    assert settings.research_extraction_concurrency == 3
    assert settings.research_synthesis_window == 10
    assert settings.research_run_timeout_seconds == 1800
    assert settings.research_search_provider == ""
    # Role models default to blank = "provider default" (no behaviour change).
    assert settings.research_fast_model == ""
    assert settings.research_strong_model == ""


def test_research_data_dir_defaults_under_data(tmp_path):
    settings = Settings(_env_file=None, data_dir=str(tmp_path))
    assert settings.resolved_research_data_dir() == tmp_path / "research"
