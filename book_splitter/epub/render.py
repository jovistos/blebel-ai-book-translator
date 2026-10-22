"""Markdown -> XHTML for the EPUB package (stage 11).

The Markdown the pipeline publishes is deliberately narrow: ATX headings,
blank-line paragraphs, bullet and numbered lists, blockquotes, thematic breaks,
emphasis and one-line image references. This module renders exactly that with
``markdown-it-py`` and does the three things a plain renderer cannot:

* it gives every heading a stable ``id``, because the EPUB navigation document
  links to them;
* it turns a paragraph that holds nothing but a picture into a ``<figure>``, so
  the stylesheet can centre and scale it;
* it reports each chapter's first level-1 heading as the chapter title, which is
  what the navigation document and every spine item's ``<title>`` use.

Raw HTML is escaped rather than passed through: the text must be XHTML, and an
unclosed ``<br>`` from the source would make the whole document invalid. Pictures
written as HTML ``<img>`` tags are the one exception — they are converted to
Markdown image syntax first, so they survive the escaping and reach the package.

``markdown-it-py`` is imported lazily: a pipeline that never builds an EPUB does
not need it, and a missing install is reported as an actionable
:class:`~book_splitter.core.errors.MissingDependencyError` instead of an
``ImportError`` from deep inside the renderer.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import MissingDependencyError
from ..documents.images import HTML_IMAGE_RE

#: An opening heading tag, its inline body and its closing tag.
HEADING_RE = re.compile(r"<h([1-6])>(.*?)</h\1>", re.DOTALL)
#: The same, after :func:`_tag_headings` added the ``id`` attribute.
ANCHORED_HEADING_RE = re.compile(r"<h([1-6])(?:\s[^>]*)?>(.*?)</h\1>", re.DOTALL)
#: Any tag, used to read a heading's text out of its rendered fragment.
TAG_RE = re.compile(r"<[^>]+>")
#: A paragraph that contains one picture and nothing else.
PICTURE_PARAGRAPH_RE = re.compile(r"<p>\s*(<img\b[^>]*?/?>)\s*</p>")
#: Any rendered picture tag, self-closing or not.
IMAGE_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
#: ``src``/``alt`` of an HTML picture tag (the pipeline preserves those tags).
HTML_IMG_SRC_RE = re.compile(
    r"""\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""", re.IGNORECASE
)
HTML_IMG_ALT_RE = re.compile(
    r"""\balt\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""", re.IGNORECASE
)
_WHITESPACE_RE = re.compile(r"\s+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

#: The install command a missing renderer reports.
MARKDOWN_IT_HINT = "pip install 'markdown-it-py>=3.0'"


@dataclass
class RenderedMarkdown:
    """One chapter, rendered: the body fragment plus what the package needs."""

    body: str
    headings: list[dict[str, Any]] = field(default_factory=list)
    title: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        """Visible characters, tags excluded — the size a reader would feel."""
        return len(html.unescape(TAG_RE.sub("", self.body)).strip())


def render_markdown(text: str, *, source: str = "") -> RenderedMarkdown:
    """Render one Markdown document to an XHTML body fragment."""
    renderer = _renderer()
    warnings: list[str] = []
    prepared = _html_pictures_to_markdown(str(text or ""), warnings, source=source)
    body = renderer.render(prepared)
    body = _tag_headings(body, warnings, source=source)
    body = PICTURE_PARAGRAPH_RE.sub(r"<figure>\1</figure>", body)
    headings = _headings(body)
    title = next((item["text"] for item in headings if item["level"] == 1), "")
    if not title:
        title = next((item["text"] for item in headings if item["text"]), "")
    return RenderedMarkdown(
        body=body,
        headings=headings,
        title=_plain(title),
        warnings=warnings,
    )


def plain_text(fragment: str) -> str:
    """The visible text of a rendered fragment, with entities decoded."""
    return _plain(fragment)


def rewrite_images(
    body: str, handler: Callable[[str, str], str]
) -> tuple[str, list[str]]:
    """Rewrite every ``<img>`` of a rendered body through ``handler``.

    ``handler`` receives the ``src`` as written and the whole tag, and returns the
    replacement markup (``""`` drops the picture). The sources encountered are
    returned in reading order, which is what a package writer needs to copy each
    picture exactly once.
    """
    seen: list[str] = []

    def replace(match: re.Match[str]) -> str:
        tag = match.group(0)
        src = _attribute(HTML_IMG_SRC_RE, tag)
        if src:
            seen.append(src)
        return handler(src, tag)

    return IMAGE_TAG_RE.sub(replace, body), seen


# --------------------------------------------------------------------------- #
# the renderer
# --------------------------------------------------------------------------- #
def _renderer():
    """A configured ``MarkdownIt`` instance, or an actionable dependency error."""
    try:
        from markdown_it import MarkdownIt
    except ImportError as exc:  # pragma: no cover - exercised by the error test
        raise MissingDependencyError(
            "stage 11 builds the EPUB with markdown-it-py, which is not installed; "
            f"install it with: {MARKDOWN_IT_HINT}"
        ) from exc
    return MarkdownIt(
        "default",
        {
            # The book's own text is not HTML: escape it, so the XHTML stays valid.
            "html": False,
            # ``<hr />`` and ``<img />``: self-closing tags are XHTML-safe.
            "xhtmlOut": True,
            # The pipeline writes one paragraph per line already; a soft line
            # break is a space, exactly as Markdown says.
            "breaks": False,
            "typographer": False,
            "linkify": False,
        },
    )


# --------------------------------------------------------------------------- #
# headings
# --------------------------------------------------------------------------- #
def _tag_headings(body: str, warnings: list[str], *, source: str) -> str:
    """Give every heading an ``id`` the navigation document can link to."""
    used: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        level = match.group(1)
        inner = match.group(2)
        anchor = _anchor(_plain(inner), used)
        return f'<h{level} id="{anchor}">{inner}</h{level}>'

    tagged = HEADING_RE.sub(replace, body)
    if not used and "<h" in body:
        warnings.append(f"{source}: headings could not be identified")
    return tagged


def _headings(body: str) -> list[dict[str, Any]]:
    """The heading inventory of a rendered body, in reading order."""
    return [
        {
            "level": int(match.group(1)),
            "anchor": _anchor_of(match.group(0)),
            "text": _plain(match.group(2)),
        }
        for match in ANCHORED_HEADING_RE.finditer(body)
    ]


def _anchor_of(heading_tag: str) -> str:
    match = re.match(r'<h\d id="([^"]*)"', heading_tag)
    return match.group(1) if match else ""


def _anchor(text: str, used: set[str]) -> str:
    """A stable, URL-safe fragment identifier, unique inside one document."""
    plain = unicodedata.normalize("NFKD", _plain(text))
    slug = _SLUG_RE.sub("-", plain.encode("ascii", "ignore").decode("ascii").lower())
    slug = slug.strip("-")[:48] or "section"
    if not slug[0].isalpha():
        slug = f"h-{slug}"
    candidate = slug
    bump = 2
    while candidate in used:
        candidate = f"{slug}-{bump}"
        bump += 1
    used.add(candidate)
    return candidate


def _plain(fragment: str) -> str:
    """Readable text out of an HTML fragment."""
    return _WHITESPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", fragment))).strip()


# --------------------------------------------------------------------------- #
# pictures written as HTML
# --------------------------------------------------------------------------- #
def _html_pictures_to_markdown(text: str, warnings: list[str], *, source: str) -> str:
    """Rewrite ``<img src=... alt=...>`` as ``![alt](src)``.

    Markdown images are rendered by the parser; raw HTML tags would be escaped
    into visible text. A tag with no ``src`` carries no picture, so it is dropped
    and reported.
    """

    def replace(match: re.Match[str]) -> str:
        tag = match.group(0)
        src = _attribute(HTML_IMG_SRC_RE, tag)
        if not src:
            warnings.append(f"{source}: an <img> tag without src was dropped")
            return ""
        alt = _attribute(HTML_IMG_ALT_RE, tag) or "Image"
        escaped_alt = alt.replace("[", "\\[").replace("]", "\\]")
        return f"![{escaped_alt}]({src})"

    return HTML_IMAGE_RE.sub(replace, text)


def _attribute(pattern: re.Pattern[str], tag: str) -> str:
    match = pattern.search(tag)
    if not match:
        return ""
    return next((group for group in match.groups() if group is not None), "")


__all__ = [
    "MARKDOWN_IT_HINT",
    "RenderedMarkdown",
    "plain_text",
    "render_markdown",
    "rewrite_images",
]
