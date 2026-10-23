"""Rule group: one chapter's non-title heading decisions.

Every entry must name a real heading line (or a plain ALL-CAPS candidate for
``promote``), cannot target the chapter title, and cannot repeat a line. The two
warnings tell the agent when it ignored the inspector (kept a suspicious heading)
and when it never reviewed one.
"""

from __future__ import annotations

from book_translator.core.constants import HEADING_DECISIONS

from .base import ChapterPlanDraft


def check_chapter_headings(draft: ChapterPlanDraft) -> None:
    structure = draft.structure
    title_line = structure.title_line
    heading_by_line = {int(item["line_number"]): item for item in structure.headings}
    # Index the suspicious list once: the per-heading lookup used to scan it.
    suspicious_by_line = {
        int(item["line_number"]): item for item in structure.suspicious_headings
    }
    claimed = draft.claimed

    for index, entry in enumerate(draft.plan.get("headings") or []):
        line = entry["line_number"]
        decision = entry["decision"]
        label = f"heading #{index + 1} (line {line})"
        if decision not in HEADING_DECISIONS:
            draft.fail(f"{label}: decision must be one of {list(HEADING_DECISIONS)}")
            continue
        if line < 1 or line > len(draft.lines):
            draft.fail(f"{label}: out of range (chapter has {len(draft.lines)} lines)")
            continue
        if line == title_line:
            draft.fail(
                f"{label}: the chapter title is controlled by title.action, not headings"
            )
            continue
        record = heading_by_line.get(line)
        if record is None:
            draft.fail(f"{label}: not a heading or a plain ALL-CAPS heading candidate")
            continue
        is_markdown_header = bool(record.get("level"))
        if decision == "promote":
            if is_markdown_header:
                draft.fail(f"{label}: already has '#' markup, so it cannot be promoted")
                continue
            if record.get("suggested") != "promote":
                draft.fail(f"{label}: not a plain ALL-CAPS heading candidate")
                continue
        elif decision == "demote" and not is_markdown_header:
            draft.fail(f"{label}: only a Markdown header can be demoted")
            continue
        if line in claimed:
            draft.fail(f"{label}: another heading decision already targets this line")
            continue
        claimed.add(line)
        suspicious = suspicious_by_line.get(line)
        if decision == "keep" and suspicious is not None:
            draft.warn(
                f"{label}: the inspector suggested {suspicious['suggested']} but the plan keeps it"
            )

    omitted = [
        item
        for item in structure.suspicious_headings
        if int(item["line_number"]) not in claimed
    ]
    if omitted:
        draft.warn(
            f"{len(omitted)} suspicious heading(s) were not reviewed and will be kept as-is: "
            + ", ".join(str(item["line_number"]) for item in omitted[:8])
        )


__all__ = ["check_chapter_headings"]
