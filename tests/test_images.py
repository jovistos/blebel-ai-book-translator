"""Offline tests for the image-reference helpers every stage shares.

The helpers are deliberately small, so these tests pin the exact shapes the
pipeline must recognise and the marker round-trip stage 7 relies on:

* Markdown and raw-HTML pictures are both found, in order;
* a line holding only a picture is protected from cleaning;
* ``protect_images`` -> ``restore_images`` is byte-for-byte reversible;
* a dropped, doubled or invented marker is reported, never ignored.
"""

from __future__ import annotations

from book_translator.documents.images import (
    has_images,
    image_count,
    image_refs,
    image_spans,
    image_token_problems,
    is_image_line,
    protect_images,
    restore_images,
    strip_images,
)

MARKDOWN = "![Image](../01-markdown/images/rabbit.png)"
HTML = '<img src="images/hat.jpg" alt="A hat" />'


def test_markdown_and_html_images_are_found_in_order():
    text = f"Before\n\n{MARKDOWN}\n\n{HTML}\n\nAfter"

    assert image_refs(text) == [MARKDOWN, HTML]
    assert image_count(text) == 2
    assert has_images(text) is True
    assert image_spans(text)[0] == (text.index(MARKDOWN), text.index(MARKDOWN) + len(MARKDOWN))


def test_an_image_only_line_is_recognised_but_prose_with_an_image_is_not():
    assert is_image_line(MARKDOWN)
    assert is_image_line(f"  {HTML}  ")
    assert not is_image_line(f"Alice saw {MARKDOWN}")
    assert not is_image_line("An ordinary line.")
    assert strip_images(f"Alice saw {MARKDOWN}") == "Alice saw "


def test_all_caps_alt_text_does_not_turn_a_picture_into_a_heading():
    from book_translator.documents.markdown import is_plain_heading_candidate

    assert is_plain_heading_candidate("ALICE")
    assert not is_plain_heading_candidate("![ALICE](../01-markdown/images/rabbit.png)")


def test_protect_and_restore_are_byte_for_byte_reversible():
    source = f"Alice looked.\n\n{MARKDOWN}\n\nThen {HTML} appeared.\n"
    protected, mapping = protect_images(source)

    assert MARKDOWN not in protected and HTML not in protected
    assert list(mapping.values()) == [MARKDOWN, HTML]
    assert "[[IMG_1]]" in protected and "[[IMG_2]]" in protected

    restored, missing, duplicated = restore_images(protected, mapping)
    assert restored == source
    assert missing == [] and duplicated == []


def test_protect_uses_a_marker_that_is_not_already_in_the_text():
    source = f"[[IMG_1]] is the author's own text.\n\n{MARKDOWN}\n"
    protected, mapping = protect_images(source)

    assert "[[IMG_1]] is the author's own text." in protected
    assert len(mapping) == 1
    token = next(iter(mapping))
    assert token != "[[IMG_1]]"
    assert restore_images(protected, mapping)[0] == source


def test_a_dropped_doubled_or_invented_marker_is_reported():
    mapping = {"[[IMG_1]]": MARKDOWN, "[[IMG_2]]": HTML}

    assert image_token_problems("[[IMG_1]] [[IMG_2]]", mapping) == []

    dropped = image_token_problems("[[IMG_1]] only", mapping)
    assert dropped and "dropped" in dropped[0] and "[[IMG_2]]" in dropped[0]

    doubled = image_token_problems("[[IMG_1]] [[IMG_1]] [[IMG_2]]", mapping)
    assert doubled and "repeated" in doubled[0]

    invented = image_token_problems("[[IMG_1]] [[IMG_2]] [[IMG_9]]", mapping)
    assert invented and "invented" in invented[0]

    unexpected = image_token_problems("[[IMG_9]]", {})
    assert unexpected and "invented" in unexpected[0]
