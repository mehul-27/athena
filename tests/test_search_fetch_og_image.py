from unittest.mock import patch

import httpx

from backend.config import Settings
from search.fetch import fetch_webpage_content


def _response(url, html):
    return httpx.Response(
        200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=html.encode(),
        request=httpx.Request("GET", url),
    )


def test_html_fetch_extracts_absolute_og_image(monkeypatch):
    settings = Settings(_env_file=None, web_fetch_timeout=3)
    html = """<html><head><title>Report</title>
      <meta property="og:image" content="https://cdn.example.test/hero.jpg">
      </head><body><main>Readable content</main></body></html>"""
    monkeypatch.setattr("search.fetch.assert_public_url", lambda url: url)
    monkeypatch.setattr(httpx.Client, "stream", lambda self, method, url, **kwargs: _FakeContext(_response(url, html)))

    result = fetch_webpage_content("https://example.test/page", settings=settings)

    assert result["success"] is True
    assert result["og_image"] == "https://cdn.example.test/hero.jpg"


def test_html_fetch_resolves_relative_image_url(monkeypatch):
    settings = Settings(_env_file=None, web_fetch_timeout=3)
    html = '<html><head><meta property="og:image" content="/images/hero.jpg"></head><body>Body</body></html>'
    monkeypatch.setattr("search.fetch.assert_public_url", lambda url: url)
    monkeypatch.setattr(httpx.Client, "stream", lambda self, method, url, **kwargs: _FakeContext(_response(url, html)))

    result = fetch_webpage_content("https://example.test/articles/story", settings=settings)

    assert result["og_image"] == "https://example.test/images/hero.jpg"


def test_html_fetch_without_og_image_returns_empty_string(monkeypatch):
    settings = Settings(_env_file=None, web_fetch_timeout=3)
    html = "<html><head><title>Report</title></head><body>Readable content</body></html>"
    monkeypatch.setattr("search.fetch.assert_public_url", lambda url: url)
    monkeypatch.setattr(httpx.Client, "stream", lambda self, method, url, **kwargs: _FakeContext(_response(url, html)))

    result = fetch_webpage_content("https://example.test/page", settings=settings)

    assert result["og_image"] == ""


class _FakeContext:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, *args):
        return False
