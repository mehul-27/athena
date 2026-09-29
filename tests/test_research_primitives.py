from datetime import datetime

from backend.research.json_utils import parse_json_array, parse_json_object
from backend.research.quality import is_low_quality
from backend.research.sources import extract_raw_findings, extract_sources
from backend.research.text import strip_thinking


def test_parse_json_array_clean_and_fenced():
    assert parse_json_array('["one", "two"]') == ["one", "two"]
    assert parse_json_array('```json\n["one", "two"]\n```') == ["one", "two"]


def test_parse_json_array_repairs_truncation():
    assert parse_json_array('["query one", "query two", "unfinished') == ["query one", "query two"]


def test_parse_json_array_prefers_last_parseable_array():
    text = 'Example: ["example"]\nActual answer: ["real one", "real two"]'
    assert parse_json_array(text) == ["real one", "real two"]


def test_parse_json_array_garbage_returns_empty():
    assert parse_json_array("no JSON here") == []


def test_parse_json_object_clean_fenced_and_prose_wrapped():
    expected = {"summary": "relevant"}
    assert parse_json_object('{"summary":"relevant"}') == expected
    assert parse_json_object('```json\n{"summary":"relevant"}\n```') == expected
    assert parse_json_object('Result: {"summary":"relevant"} thanks') == expected
    assert parse_json_object("not JSON") is None


def test_strip_thinking_tags_and_dangling_openers():
    assert strip_thinking("Before <think>private reasoning</think> Answer") == "Before Answer"
    assert strip_thinking("Answer <thinking time='1'>secret</thinking>") == "Answer"
    assert strip_thinking("Final <thought>secret</thought>") == "Final"
    assert strip_thinking("Answer <think>unfinished reasoning") == "Answer"


def test_strip_thinking_gemma_channels():
    assert strip_thinking("<|channel>thought\nprivate notes<channel|><|channel>response\n## Final answer<channel|>") == "## Final answer"


def test_strip_thinking_qwen_preamble_preserves_sectioned_answer():
    raw = "Thinking Process:\n1. Read context.\n\n## Answer\nLUMEN-5831-ORBIT"
    assert strip_thinking(raw) == "## Answer\nLUMEN-5831-ORBIT"


def test_strip_thinking_prompt_echo_preserves_bold_answer_heading():
    raw = "The user asks: what is it?\n\n**Answer:** 42."
    assert strip_thinking(raw) == "**Answer:** 42."


def test_strip_thinking_qwen_preamble_without_heading_matches_source_behavior():
    # The source regex consumes a leading reasoning block to end-of-string when
    # no heading/bold delimiter introduces the final answer.
    assert strip_thinking("Thinking Process: private notes\n\nplain final answer") == ""


def test_strip_thinking_clean_text_is_unchanged():
    assert strip_thinking("A clean report.\n\nSecond paragraph.") == "A clean report.\n\nSecond paragraph."


def test_quality_filter_markers_and_valid_text():
    assert is_low_quality("This content is insufficient to answer")
    assert is_low_quality("Cookie consent banner")
    assert is_low_quality("The paper studies cookie consent rates across regions")  # source marker matches phrases verbatim
    assert is_low_quality("")
    assert is_low_quality(None)


def test_source_extraction_deduplicates_and_filters():
    findings = [
        {"url": "https://a.test", "title": "A", "summary": "Useful evidence", "og_image": "https://a.test/cover.jpg"},
        {"url": "https://a.test", "title": "A duplicate", "summary": "More evidence"},
        {"url": "https://junk.test", "title": "Junk", "summary": "No relevant information"},
    ]
    assert extract_sources(findings) == [
        {"url": "https://a.test", "title": "A", "image": "https://a.test/cover.jpg"}
    ]
    assert extract_raw_findings(findings) == [
        {"url": "https://a.test", "title": "A", "summary": "Useful evidence"},
        {"url": "https://a.test", "title": "A duplicate", "summary": "More evidence"},
    ]


def test_source_findings_falls_back_to_evidence():
    result = extract_raw_findings([{"url": "https://a.test", "evidence": "E" * 2200}])
    assert len(result[0]["summary"]) == 2000
