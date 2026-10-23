"""Tests for the web-lookup tools — no network, every fetch is injected.

The published chapter list of a book whose own TOC was destroyed is a real source
of truth, but a test suite must never depend on the internet. Every test here
therefore drives :func:`search_web` / :func:`fetch_web_page` with a fake
``fetch`` and pins the behaviour that matters: the result shape, the provider
fallback, the redirect unwrapping, and the bounds (result count, page size,
public-http(s)-only urls).
"""

from __future__ import annotations

import json
import urllib.error

from book_translator.core.constants import (
    MAX_WEB_FETCH_CHARS,
    MAX_WEB_RESULTS,
    MIN_WEB_FETCH_CHARS,
)
from book_translator.stages.agent.tools import (
    SPLIT_TOOLS,
    tool_names,
)
from book_translator.stages.agent.tools import (
    fetch_web_page as fetch_page_tool,
)
from book_translator.stages.agent.tools import (
    web_search as web_search_tool,
)
from book_translator.stages.agent.tools.registry import SPLIT
from book_translator.stages.analysis.websearch import (
    check_url,
    fetch_web_page,
    search_web,
)

BRAVE_HTML = """
<html><body><main>
  <div class="snippet svelte-1" data-pos="0" data-type="web">
    <div class="result-body"><div class="result-wrapper"><div class="result-content svelte-2">
      <a href="https://books.google.com/books/about/Staring_at_the_Sun.html?id=BwozrXMHYk4C" class="l1">
        <div class="site-name-content"><cite class="snippet-url">books.google.com</cite></div>
        <div class="title search-snippet-title line-clamp-1" title="Staring at the Sun - Irvin D. Yalom">Staring at the Sun - Irvin D. Yalom</div>
      </a>
      <div class="generic-snippet"><div class="content desktop-default-regular t-primary">
        RECOGNIZING DEATH ANXIETY COVERT DEATH ANXIETY THE AWAKENING EXPERIENCE
      </div></div>
    </div></div></div>
  </div>
  <div class="snippet svelte-1" data-pos="1" data-type="web">
    <div class="result-body"><div class="result-wrapper"><div class="result-content svelte-2">
      <a href="https://openlibrary.org/books/OL20734548M/Staring_at_the_sun" class="l1">
        <div class="title search-snippet-title" title="Staring at the sun by Irvin D. Yalom">Staring at the sun by Irvin D. Yalom</div>
      </a>
      <div class="generic-snippet"><div class="content">The mortal wound. Recognizing death anxiety.</div></div>
    </div></div></div>
  </div>
</main></body></html>
"""

DDG_HTML = """
<html><body>
  <div class="result results_links results_links_deep web-result">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fopenlibrary.org%2Fbooks%2FOL20734548M&rut=abc">Staring at the sun</a>
    <a class="result__snippet" href="#">Table of contents: the mortal wound, recognizing death anxiety.</a>
  </div>
</body></html>
"""

DDG_CHALLENGE = "<html><body>Unfortunately, bots use DuckDuckGo too. Please complete the challenge</body></html>"


def _fetcher(pages, calls=None):
    """``fetch(url, timeout) -> (status, content_type, charset, body)``."""

    def fetch(url, timeout):
        if calls is not None:
            calls.append(url)
        for needle, response in pages.items():
            if needle in url:
                return response
        raise AssertionError(f"unexpected url {url}")

    return fetch


# --------------------------------------------------------------------------- #
# Searching
# --------------------------------------------------------------------------- #


def test_search_parses_brave_results():
    report = search_web(
        "Staring at the Sun Yalom table of contents",
        fetch=_fetcher({"search.brave.com": (200, "text/html", "utf-8", BRAVE_HTML)}),
    )
    assert report["provider"] == "brave"
    assert report["result_count"] == 2
    first, second = report["results"]
    assert first["url"].startswith("https://books.google.com/")
    assert first["title"] == "Staring at the Sun - Irvin D. Yalom"
    assert "RECOGNIZING DEATH ANXIETY" in first["snippet"]
    assert second["url"].startswith("https://openlibrary.org/")
    # The agent is told the data is external and untrusted.
    assert "untrusted" in report["note"]


def test_search_unwraps_the_duckduckgo_redirect():
    report = search_web(
        "irvin yalom staring at the sun",
        fetch=_fetcher(
            {
                "search.brave.com": (429, "text/html", "utf-8", "Too Many Requests"),
                "html.duckduckgo.com": (200, "text/html", "utf-8", DDG_HTML),
            }
        ),
    )
    assert report["provider"] == "duckduckgo"
    assert report["results"][0]["url"] == "https://openlibrary.org/books/OL20734548M"
    assert report["results"][0]["snippet"].startswith("Table of contents")


def test_search_falls_back_when_the_first_provider_fails():
    calls: list[str] = []
    report = search_web(
        "query",
        fetch=_fetcher(
            {
                "search.brave.com": (503, "text/html", "utf-8", "busy"),
                "html.duckduckgo.com": (200, "text/html", "utf-8", DDG_HTML),
            },
            calls,
        ),
    )
    assert report["provider"] == "duckduckgo"
    assert any("search.brave.com" in url for url in calls)
    assert any("html.duckduckgo.com" in url for url in calls)


def test_search_treats_an_anti_bot_page_as_a_provider_failure():
    report = search_web(
        "query",
        fetch=_fetcher(
            {
                "search.brave.com": (429, "text/html", "utf-8", "slow down"),
                "html.duckduckgo.com": (202, "text/html", "utf-8", DDG_CHALLENGE),
            }
        ),
    )
    assert "error" in report
    assert [a["provider"] for a in report["attempts"]] == ["brave", "duckduckgo"]
    assert "challenge" in json.dumps(report["attempts"])
    assert "hint" in report


def test_search_reports_a_network_error_instead_of_raising():
    def broken(url, timeout):
        raise urllib.error.URLError("dns failure")

    report = search_web("query", fetch=broken)
    assert report["error"] == "no search provider returned results"
    assert all("URLError" in a["error"] for a in report["attempts"])


def test_search_rejects_an_empty_query_without_touching_the_network():
    def never(url, timeout):  # pragma: no cover - must not run
        raise AssertionError("the network must not be used")

    assert "error" in search_web("   ", fetch=never)


def test_search_caps_the_number_of_results():
    html = "".join(
        f'<div data-type="web"><a href="https://example.com/{i}">t</a>'
        f'<div class="title" title="title {i}">title {i}</div>'
        f'<div class="content">snippet {i}</div></div>'
        for i in range(30)
    )
    report = search_web("q", max_results=99, fetch=_fetcher({"search.brave.com": (200, "text/html", "utf-8", html)}))
    assert report["result_count"] == MAX_WEB_RESULTS


# --------------------------------------------------------------------------- #
# Reading a page
# --------------------------------------------------------------------------- #

PAGE = """
<html><head><title>Staring at the sun | Open Library</title>
<style>body{color:red}</style><script>var x = "The mortal wound";</script></head>
<body>
  <h1>Staring at the sun</h1>
  <div>Table of contents</div>
  <ul><li>The mortal wound</li><li>Recognizing death anxiety</li><li>The awakening experience</li></ul>
  <p>By Irvin D. Yalom &amp; friends.</p>
</body></html>
"""


def test_fetch_returns_block_separated_text_and_the_title():
    report = fetch_web_page(
        "https://openlibrary.org/books/OL20734548M",
        fetch=_fetcher({"openlibrary.org": (200, "text/html; charset=utf-8", "utf-8", PAGE)}),
        resolve_dns=False,
    )
    assert report["title"] == "Staring at the sun | Open Library"
    lines = report["text"].splitlines()
    assert "The mortal wound" in lines
    assert "Recognizing death anxiety" in lines
    assert "By Irvin D. Yalom & friends." in lines
    # script and style contents are not page text
    assert "color:red" not in report["text"]
    assert 'var x' not in report["text"]
    assert report["truncated"] is False


def test_fetch_clips_the_text_and_says_so():
    big = "<html><body><p>" + ("chapter list " * 400) + "</p></body></html>"
    report = fetch_web_page(
        "https://example.com/",
        max_chars=300,
        fetch=_fetcher({"example.com": (200, "text/html", "utf-8", big)}),
        resolve_dns=False,
    )
    assert report["truncated"] is True
    assert len(report["text"]) == 300


def test_fetch_has_a_floor_so_a_tiny_request_stays_useful():
    big = "<html><body><p>" + ("chapter list " * 400) + "</p></body></html>"
    report = fetch_web_page(
        "https://example.com/",
        max_chars=1,
        fetch=_fetcher({"example.com": (200, "text/html", "utf-8", big)}),
        resolve_dns=False,
    )
    assert len(report["text"]) == MIN_WEB_FETCH_CHARS


def test_fetch_caps_max_chars_whatever_the_agent_asks():
    big = "<html><body>" + "<p>paragraph</p>" * 20_000 + "</body></html>"
    report = fetch_web_page(
        "https://example.com/",
        max_chars=10_000_000,
        fetch=_fetcher({"example.com": (200, "text/html", "utf-8", big)}),
        resolve_dns=False,
    )
    assert len(report["text"]) <= MAX_WEB_FETCH_CHARS


def test_fetch_reads_a_json_document_as_text():
    body = json.dumps({"table_of_contents": [{"title": "The mortal wound"}]})
    report = fetch_web_page(
        "https://openlibrary.org/books/OL20734548M.json",
        fetch=_fetcher({"openlibrary.org": (200, "application/json", "utf-8", body)}),
        resolve_dns=False,
    )
    assert "error" not in report
    assert "The mortal wound" in report["text"]


def test_fetch_turns_an_http_error_into_a_payload():
    def fetch(url, timeout):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    report = fetch_web_page("https://example.com/blocked", fetch=fetch, resolve_dns=False)
    assert report["error"] == "HTTP 403"
    assert "another search result" in report["hint"]


# --------------------------------------------------------------------------- #
# URL policy
# --------------------------------------------------------------------------- #


def test_url_policy_refuses_anything_but_public_http_addresses():
    assert check_url("") is not None
    assert check_url("file:///etc/passwd") is not None
    assert check_url("ftp://example.com/x") is not None
    assert check_url("http://localhost:8080/admin") is not None
    assert check_url("http://127.0.0.1/") is not None
    assert check_url("http://10.0.0.5/") is not None
    assert check_url("http://169.254.169.254/latest/meta-data/") is not None
    assert check_url("https://openlibrary.org/books/OL20734548M") is None


def test_fetch_refuses_a_private_address_without_fetching():
    def never(url, timeout):  # pragma: no cover - must not run
        raise AssertionError("the network must not be used")

    report = fetch_web_page("http://127.0.0.1:1234/", fetch=never)
    assert "private" in report["error"] or "local" in report["error"]


# --------------------------------------------------------------------------- #
# The LangChain tools themselves
# --------------------------------------------------------------------------- #


def test_the_tools_are_bound_to_the_split_stage_only():
    names = [tool.name for tool in SPLIT_TOOLS]
    assert "web_search" in names
    assert "fetch_web_page" in names
    assert "web_search" in tool_names(SPLIT)
    # stage 2 must not gain network access through the shared group
    assert "web_search" not in tool_names("preprocess")
    assert "fetch_web_page" not in tool_names("preprocess")


def test_the_tool_wrappers_return_json_and_refuse_private_urls():
    payload = json.loads(fetch_page_tool.invoke({"url": "http://127.0.0.1/private"}))
    assert "error" in payload
    empty = json.loads(web_search_tool.invoke({"query": ""}))
    assert "error" in empty
