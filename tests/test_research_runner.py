import asyncio

from backend.config import Settings
from backend.research.runner import ResearchRunner
from backend.research.store import ResearchStore


class NoCallRouter:
    configured = True
    provider_names = ["stub"]


def test_store_rejects_invalid_research_ids(tmp_path):
    settings = Settings(_env_file=None, data_dir=str(tmp_path / "data"))
    store = ResearchStore(settings)
    assert store.path_for("../bad") is None
    assert store.path_for("rp-0123456789ab") is not None


def test_store_round_trip_and_listing(tmp_path):
    settings = Settings(_env_file=None, data_dir=str(tmp_path / "data"))
    store = ResearchStore(settings)
    record = {
        "query": "q", "status": "done", "result": "r", "raw_report": "raw",
        "sources": [], "raw_findings": [], "stats": {}, "category": None,
        "started_at": 1.0, "completed_at": 2.0,
    }
    store.save("rp-0123456789ab", record)
    assert store.load("rp-0123456789ab") == record
    listed = store.list()[0]
    assert listed["research_id"] == "rp-0123456789ab"
    assert {key: value for key, value in listed.items() if key != "research_id"} == record
    assert store.delete("rp-0123456789ab") is True
    assert store.load("rp-0123456789ab") is None


def test_runner_status_unknown_id_is_none(tmp_path):
    settings = Settings(_env_file=None, data_dir=str(tmp_path / "data"))
    runner = ResearchRunner(settings, NoCallRouter())
    assert runner.status("rp-0123456789ab") is None


def test_runner_start_creates_research_id_and_running_status(tmp_path):
    settings = Settings(_env_file=None, data_dir=str(tmp_path / "data"), research_run_timeout_seconds=60)
    runner = ResearchRunner(settings, NoCallRouter())

    async def start_and_check():
        started = runner.start("question", max_rounds=1)
        assert started["research_id"].startswith("rp-")
        assert runner.status(started["research_id"])["status"] == "running"
        runner.cancel(started["research_id"])
        await asyncio.sleep(0)

    asyncio.run(start_and_check())


def test_hide_unhide_image_changes_regenerated_report(tmp_path):
    settings = Settings(_env_file=None, data_dir=str(tmp_path / "data"))
    runner = ResearchRunner(settings, NoCallRouter())
    research_id = "rp-0123456789ab"
    image_url = "https://cdn.example.test/hero.jpg"
    runner.store.save(research_id, {
        "query": "Question",
        "status": "done",
        "result": "## Answer\\nText",
        "raw_report": "## Answer\\nText",
        "sources": [{"url": "https://example.test", "title": "Source", "image": image_url}],
        "raw_findings": [],
        "stats": {},
        "category": None,
        "started_at": 1,
        "completed_at": 2,
        "hidden_images": [],
    })

    first = runner.report_html(research_id)
    assert image_url in first
    assert runner.update(research_id, hidden_images=[image_url])
    hidden = runner.report_html(research_id)
    assert image_url not in hidden
    assert "Show hidden (1)" in hidden
    assert runner.update(research_id, hidden_images=[])
    restored = runner.report_html(research_id)
    assert image_url in restored


def test_format_report_contains_summary_and_raw_markdown():
    from backend.research.runner import format_research_report

    report = format_research_report("q", "# Final", {"Rounds": 2, "Queries": 4, "URLs": 3}, 12.5)
    assert "## Research Summary" in report
    assert "**Rounds:** 2" in report
    assert "# Final" in report
