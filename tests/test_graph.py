"""Tests for the LangGraph pipeline, using a scripted model double."""

from __future__ import annotations

import json

from book_splitter.agent import build_graph
from langchain_core.messages import AIMessage, HumanMessage


class ScriptedModel:
    """Minimal chat-model double: replays canned AIMessages."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_messages = []

    def invoke(self, messages, *_, **__):
        self.seen_messages.append(list(messages))
        index = min(len(self.seen_messages) - 1, len(self.responses) - 1)
        return self.responses[index]


def _tool_call(name, args, call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _invoke_graph(model, source, output_dir, **extra):
    app = build_graph(model=model)
    state = {
        "source_path": str(source),
        "output_dir": str(output_dir),
        "front_matter_lines": 200,
        "max_attempts": 4,
        "include_front_matter": True,
        "dry_run": False,
        "attempts": 0,
        "usage": {},
        "log": [],
        "feedback": [],
        "messages": [],
        **extra,
    }
    return app.invoke(state, config={"recursion_limit": 40})


def test_graph_runs_full_pipeline_with_scripted_model(plain, tmp_path, plan):
    mapping = plan(
        [(12, "01-predgovor.md"), (18, "02-dolazak.md"), (28, "03-oluja.md"),
         (38, "04-odlazak.md"), (44, "05-biljeske.md")]
    )
    model = ScriptedModel([
        _tool_call("validate_split_map_tool", {"file_path": str(plain), "split_mapping": mapping}, "c1"),
        _tool_call("propose_split_map", {"split_mapping": mapping, "strategy": "toc_titles"}, "c2"),
    ])
    final = _invoke_graph(model, plain, tmp_path)

    assert final["validation"]["ok"], final["validation"]["issues"]
    assert final["manifest"]["file_count"] == 6
    assert final["verification"]["ok"]
    assert final["verification"]["reassembles_byte_for_byte"]
    assert (tmp_path / "01-predgovor.md").is_file()
    assert (tmp_path / "_manifest.json").is_file()
    # scout + 2 agent turns
    assert final["attempts"] == 2


def test_graph_slices_a_prose_chapter_start_submitted_by_the_model(plain, tmp_path):
    """A chapter whose heading is gone, submitted as role=prose-chapter + evidence."""
    mapping = [
        {"line_number": 12, "filename": "01-predgovor.md"},
        {"line_number": 18, "filename": "02-dolazak.md"},
        {
            "line_number": 20,
            "filename": "03-dolazak-nastavak.md",
            "role": "prose-chapter",
            "evidence": "printed TOC lists a section here; no heading survives in the file",
        },
        {"line_number": 28, "filename": "04-oluja.md"},
        {"line_number": 44, "filename": "05-biljeske.md"},
    ]
    model = ScriptedModel([
        _tool_call(
            "propose_split_map",
            {"split_mapping": mapping, "strategy": "prose_chapters"},
            "c1",
        ),
    ])
    final = _invoke_graph(model, plain, tmp_path)

    assert final["validation"]["ok"], final["validation"]["issues"]
    assert final["validation"]["prose_chapter_count"] == 1
    assert final["verification"]["reassembles_byte_for_byte"]
    assert (tmp_path / "03-dolazak-nastavak.md").is_file()
    # The evidence the model gave is echoed into the trace, not silently dropped.
    assert final["proposed_splits"][2]["evidence"].startswith("printed TOC")


def test_graph_rejects_a_bad_plan_and_loops_back(plain, tmp_path, plan):
    bad = plan([(3, "01-toc.md"), (5, "02-toc.md"), (7, "03-toc.md")])  # all inside the TOC
    good = plan([(12, "01-predgovor.md"), (18, "02-dolazak.md"), (28, "03-oluja.md")])
    model = ScriptedModel([
        _tool_call("propose_split_map", {"split_mapping": bad, "strategy": "heuristics"}, "c1"),
        _tool_call("propose_split_map", {"split_mapping": good, "strategy": "toc_titles"}, "c2"),
    ])
    final = _invoke_graph(model, plain, tmp_path)

    assert final["attempts"] == 2, "the graph must ask the model to correct the map"
    assert final["validation"]["ok"]
    assert final["verification"]["ok"]
    # The correction prompt told the model exactly what was wrong.
    joined = " ".join(
        m.content if isinstance(m, HumanMessage) else "" for m in model.seen_messages[-1]
    )
    assert "inside the printed table of contents" in joined


def test_graph_blocks_slicing_before_validation(plain, tmp_path, plan):
    mapping = plan([(12, "01-a.md"), (18, "02-b.md"), (28, "03-c.md")])
    model = ScriptedModel([
        _tool_call(
            "slice_book_by_line_numbers",
            {"file_path": str(plain), "split_mapping": mapping, "output_dir": str(tmp_path)},
            "c1",
        ),
        _tool_call("propose_split_map", {"split_mapping": mapping, "strategy": "toc_titles"}, "c2"),
    ])
    final = _invoke_graph(model, plain, tmp_path)
    blocked = [
        m for m in final["messages"] if getattr(m, "name", "") == "slice_book_by_line_numbers"
    ]
    assert blocked and "blocked until the map passes validation" in blocked[0].content
    assert final["verification"]["ok"]


def test_graph_reports_unknown_tools(plain, tmp_path, plan):
    mapping = plan([(12, "01-a.md"), (18, "02-b.md"), (28, "03-c.md")])
    model = ScriptedModel([
        _tool_call("not_a_real_tool", {}, "c1"),
        _tool_call("propose_split_map", {"split_mapping": mapping, "strategy": "toc_titles"}, "c2"),
    ])
    final = _invoke_graph(model, plain, tmp_path)
    assert any("unknown tool" in line for line in final["log"])
    assert final["verification"]["ok"]


def test_graph_gives_up_when_the_model_never_submits(plain, tmp_path):
    model = ScriptedModel([AIMessage(content="I am not sure what to do.")])
    final = _invoke_graph(model, plain, tmp_path, max_attempts=3)
    assert final["attempts"] == 3
    assert "manifest" not in final or not final.get("manifest")
    assert final["result"]["ok"] is False
    assert not list(tmp_path.glob("*.md"))


def test_the_last_turn_is_reserved_for_submitting_the_map(plain, tmp_path):
    """The failure this guards against: the model spends every turn inspecting.

    A run of *We Need to Talk About Kevin* used all six turns on `find_lines_matching`,
    `inspect_line_window`, `web_search` and `inspect_book_tail`, then gave up with
    nothing written. On the final allowed turn the agent is now told that only a
    submission is useful, while earlier turns are left un-nagged.
    """
    model = ScriptedModel([
        _tool_call("scan_markdown_headers", {"file_path": str(plain)}, f"c{i}")
        for i in range(1, 4)
    ])
    final = _invoke_graph(model, plain, tmp_path, max_attempts=3)

    assert final["attempts"] == 3
    assert final["result"]["ok"] is False
    first_turn = "\n".join(str(m.content) for m in model.seen_messages[0])
    last_turn = "\n".join(str(m.content) for m in model.seen_messages[-1])
    assert "FINAL TURN" not in first_turn
    assert "FINAL TURN" in last_turn
    assert "propose_split_map" in last_turn


def test_a_map_submitted_on_the_final_turn_still_slices(plain, tmp_path, plan):
    """The final-turn nudge must not close the door it is pointing at."""
    mapping = plan([(12, "01-predgovor.md"), (18, "02-dolazak.md"), (28, "03-oluja.md")])
    model = ScriptedModel([
        _tool_call("scan_markdown_headers", {"file_path": str(plain)}, "c1"),
        _tool_call("scan_markdown_headers", {"file_path": str(plain)}, "c2"),
        _tool_call("propose_split_map", {"split_mapping": mapping, "strategy": "heuristics"}, "c3"),
    ])
    final = _invoke_graph(model, plain, tmp_path, max_attempts=3)

    assert final["attempts"] == 3
    assert final["validation"]["ok"], final["validation"]["issues"]
    assert final["verification"]["ok"]
    assert final["verification"]["reassembles_byte_for_byte"]


def test_graph_dry_run_validates_but_writes_nothing(plain, tmp_path, plan):
    mapping = plan([(12, "01-a.md"), (18, "02-b.md"), (28, "03-c.md")])
    model = ScriptedModel([
        _tool_call("propose_split_map", {"split_mapping": mapping, "strategy": "toc_titles"}, "c1"),
    ])
    final = _invoke_graph(model, plain, tmp_path, dry_run=True)
    assert final["validation"]["ok"]
    assert final["manifest"]["dry_run"] is True
    assert "verification" not in final or not final.get("verification")
    assert not list(tmp_path.glob("*.md"))


def test_llm_context_never_contains_book_body_text(sample, tmp_path, plan):
    """The core promise: chapter prose must never reach the model."""
    body_markers = [
        "Kao bolnički kardiolog, video sam i obra",
        "Nesporno je da seksualna aktivnost može da ima snažan uticaj",
        "Danas već ina ljudi prihvataju ideju da prekomjeran stres",
    ]
    mapping = plan([(46, "01-a.md"), (78, "02-b.md"), (136, "03-c.md"), (170, "04-d.md")])
    model = ScriptedModel([
        _tool_call("propose_split_map", {"split_mapping": mapping, "strategy": "markdown_headers"}, "c1"),
    ])
    _invoke_graph(model, sample, tmp_path)

    seen = "\n".join(str(m.content) for batch in model.seen_messages for m in batch)
    for marker in body_markers:
        assert marker not in seen, f"book body leaked into the model context: {marker!r}"
    # ...while the structural signal it *is* allowed to see is present.
    assert "GLAVA 1" in seen
    assert "printed_toc_block" in seen


def test_llm_context_is_bounded(sample, tmp_path, plan):
    mapping = plan([(46, "01-a.md"), (78, "02-b.md"), (136, "03-c.md"), (170, "04-d.md")])
    model = ScriptedModel([
        _tool_call("extract_front_matter", {"file_path": str(sample), "max_lines": 100000}, "c1"),
        _tool_call("propose_split_map", {"split_mapping": mapping, "strategy": "toc_titles"}, "c2"),
    ])
    _invoke_graph(model, sample, tmp_path)
    longest = max(
        (len(str(m.content)) for batch in model.seen_messages for m in batch), default=0
    )
    assert longest <= 25_000, "a single tool result exceeded the context guardrail"


def test_scout_report_lists_no_body_text(plain):
    from book_splitter.agent.nodes.scout import build_scout_report
    from book_splitter.agent.tools import scan_markdown_headers
    from book_splitter.analysis.headers import assess_headers
    from book_splitter.analysis.toc import detect_toc_block
    from book_splitter.documents.io import load_lines

    lines = load_lines(str(plain))
    headers = scan_markdown_headers(str(plain))
    report = build_scout_report(
        source=str(plain),
        line_count=len(lines),
        headers_payload=headers,
        toc_block=detect_toc_block(lines),
        assessment=assess_headers(headers["headers"], len(lines)),
        native_toc={"available": False},
    )
    assert "Ova knjiga je nastala iz niza razgovora" not in report
    assert "SCOUT REPORT" in report
    assert "printed_toc_block" in report
    # The model is told its turn budget from the start, so the last turn is kept
    # for the submission rather than for one more inspection.
    assert "budget:" in report
    assert "propose_split_map" in report


def test_a_failed_split_still_writes_its_trace(monkeypatch, tmp_path, plain):
    """A run that never proposes a plan has no output folder yet.

    That is precisely the run whose trace explains the failure (proposed map,
    validation issues, tool log), so ``write_trace`` must create the folder
    instead of raising FileNotFoundError and losing the evidence.
    """
    from book_splitter.agent import runner
    from book_splitter.core.constants import TRACE_FILENAME

    silent = ScriptedModel([AIMessage(content="I cannot decide on a split map.")])
    real_build = runner.build_graph
    monkeypatch.setattr(runner, "build_graph", lambda settings=None: real_build(model=silent))

    out_dir = tmp_path / "01-chapters"
    result = runner.run_split(str(plain), str(out_dir), max_attempts=2, recursion_limit=30)

    assert not result.ok
    assert result.error == "failed to produce a valid chapter map"
    trace = out_dir / TRACE_FILENAME
    assert trace.is_file()
    payload = json.loads(trace.read_text(encoding="utf-8"))
    assert payload["result"]["ok"] is False
    assert payload["tool_log"], "the agent's own narration must survive a failed run"
    assert not payload["validation"]  # no plan ever reached the validator


# --------------------------------------------------------------------------- #
# tool binding: the splitter sees only its own stage's tools
# --------------------------------------------------------------------------- #
class _ToolBinder:
    """A chat-model double that records what ``bind_tools`` was handed."""

    def __init__(self) -> None:
        self.tools = []
        self.kwargs = {}

    def bind_tools(self, tools, **kwargs):
        self.tools = list(tools)
        self.kwargs = kwargs
        return self


def test_the_splitter_binds_only_its_stage_toolset():
    from book_splitter.agent.llm import bind_tools
    from book_splitter.agent.tools import SPLIT_TOOLS
    from book_splitter.toolkit.preprocess import PREPROCESS_TOOLS

    binder = _ToolBinder()
    bind_tools(binder)

    assert [tool.name for tool in binder.tools] == [tool.name for tool in SPLIT_TOOLS]
    assert "collect_chapter_headers" not in {tool.name for tool in binder.tools}
    assert binder.kwargs.get("parallel_tool_calls") is False
    # The other stage's tools are real, registered tools; they are just not the
    # splitter's to bind.
    assert "collect_chapter_headers" in {tool.name for tool in PREPROCESS_TOOLS}


def test_the_splitter_tool_node_dispatches_only_its_stage_table():
    from book_splitter.agent.graph import SPLIT
    from book_splitter.stage_graph import make_tool_node

    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "collect_chapter_headers",
                        "args": {"chapter_path": "x.md"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        "validation": {},
    }
    out = make_tool_node(spec=SPLIT)()(state)
    message = out["messages"][0]
    assert message.status == "error"
    assert "unknown tool" in message.content
    assert any("unknown tool collect_chapter_headers" in line for line in out["log"])
