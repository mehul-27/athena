"""Athena search package: web search, news search, page fetching.

Search is a **separate capability**. It is never wired into RAG Chat — RAG Chat
stays document-only.
"""

from search.fetch import fetch_webpage_content
from search.news import news_search
from search.web import call_provider, provider_chain, web_search

__all__ = ["fetch_webpage_content", "web_search", "news_search", "call_provider", "provider_chain"]
