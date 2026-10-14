"""Tests for the XML-tagged transcript renderer."""

import xml.etree.ElementTree as ET
from itertools import pairwise

from opencode_power_pack.models import (
    PartKind,
    Session,
    TokenBreakdown,
    Transcript,
    TranscriptMessage,
    TranscriptPart,
    TranscriptTodo,
)
from opencode_power_pack.transcript import (
    TranscriptOptions,
    render_transcript,
    render_transcripts,
)


def _session(**changes: object) -> Session:
    values: dict[str, object] = {
        "id": "session-root",
        "title": "Fix naïve parser",
        "created": 1_000,
        "updated": 5_000,
        "project_id": "project-root",
        "directory": "/workspace/acme/widget",
        "repository": "widget",
        "agent": "build",
        "provider": "openai",
        "model": "gpt-test",
        "cost": 1.25,
        "tokens": TokenBreakdown(
            input=10, output=20, reasoning=3, cache_read=4, cache_write=5
        ),
    }
    values.update(changes)
    return Session(**values)  # type: ignore[arg-type]


def _message(**changes: object) -> TranscriptMessage:
    values: dict[str, object] = {"id": "message-1", "role": "user"}
    values.update(changes)
    return TranscriptMessage(**values)  # type: ignore[arg-type]


def _transcript(**changes: object) -> Transcript:
    values: dict[str, object] = {"session": _session()}
    values.update(changes)
    return Transcript(**values)  # type: ignore[arg-type]


def test_golden_two_message_session() -> None:
    message_user = _message(
        id="message-user",
        role="user",
        agent="plan",
        model="gpt-test",
        provider="openai",
        created=1_000,
        parts=(TranscriptPart(kind=PartKind.TEXT, text="Needle in user text."),),
    )
    message_assistant = _message(
        id="message-assistant",
        role="assistant",
        agent="build",
        model="gpt-test",
        provider="openai",
        created=2_000,
        parts=(
            TranscriptPart(kind=PartKind.TEXT, text="The assistant replies."),
            TranscriptPart(
                kind=PartKind.TOOL,
                tool_name="bash",
                tool_title="Check repository status",
                tool_status="completed",
            ),
        ),
    )
    transcript = _transcript(messages=(message_user, message_assistant))

    expected = (
        "<oc-session\n"
        '  id="session-root"\n'
        '  title="Fix naïve parser"\n'
        '  repository="widget"\n'
        '  directory="/workspace/acme/widget"\n'
        '  agent="build"\n'
        '  model="gpt-test"\n'
        '  provider="openai"\n'
        '  created="1970-01-01T00:00:01Z"\n'
        '  updated="1970-01-01T00:00:05Z"\n'
        '  cost="1.2500"\n'
        '  tokens="42"\n'
        '  messages="2"\n'
        ">\n"
        '<oc-message n="1" role="user" agent="plan" model="gpt-test" '
        'provider="openai" time="1970-01-01T00:00:01Z">\n'
        "<oc-text>\n"
        "Needle in user text.\n"
        "</oc-text>\n"
        "</oc-message>\n"
        '<oc-message n="2" role="assistant" agent="build" model="gpt-test" '
        'provider="openai" time="1970-01-01T00:00:02Z">\n'
        "<oc-text>\n"
        "The assistant replies.\n"
        "</oc-text>\n"
        '<oc-tool name="bash" status="completed" '
        'title="Check repository status" />\n'
        "</oc-message>\n"
        "</oc-session>"
    )
    assert render_transcript(transcript) == expected
    assert render_transcripts([transcript]) == expected + "\n"


def test_timestamps_truncate_fractional_seconds() -> None:
    session = _session(created=1_757_961_303_442, updated=1_757_961_303_471)
    message = _message(created=1_757_961_303_442)
    transcript = Transcript(session=session, messages=(message,))
    output = render_transcript(transcript)
    assert 'created="2025-09-15T18:35:03Z"' in output
    assert 'updated="2025-09-15T18:35:03Z"' in output
    assert 'time="2025-09-15T18:35:03Z"' in output
    assert "." not in output.split('created="')[1].split('"')[0]


def test_verbatim_fidelity_survives_byte_for_byte() -> None:
    body = (
        "## A heading\n"
        "\n"
        "---\n"
        "\n"
        "- `bash`\n"
        "\n"
        "<details>reasoning-shaped</details>\n"
        "\n"
        "```python\n"
        "print('hi')\n"
        "```\n"
    )
    transcript = _transcript(
        messages=(_message(parts=(TranscriptPart(kind=PartKind.TEXT, text=body),)),)
    )
    output = render_transcript(transcript)
    extracted = output.split("<oc-text>\n", 1)[1].split("\n</oc-text>", 1)[0]
    assert extracted == body


def test_reasoning_excluded_by_default_and_included_with_flag() -> None:
    transcript = _transcript(
        messages=(
            _message(
                parts=(
                    TranscriptPart(kind=PartKind.REASONING, text="private thoughts"),
                ),
            ),
        )
    )
    default_output = render_transcript(transcript)
    assert "private thoughts" not in default_output
    assert "<oc-reasoning>" not in default_output
    assert default_output.count("<oc-message") == 1
    assert "/>" in default_output  # message self-closes with no body

    included_output = render_transcript(
        transcript, TranscriptOptions(include_reasoning=True)
    )
    assert "<oc-reasoning>\nprivate thoughts\n</oc-reasoning>" in included_output


def test_self_closed_vs_paired_message() -> None:
    empty_message = _transcript(messages=(_message(role="assistant", created=9_000),))
    output = render_transcript(empty_message)
    assert '<oc-message n="1" role="assistant" time="1970-01-01T00:00:09Z" />' in output

    with_body = _transcript(
        messages=(
            _message(
                role="assistant",
                created=9_000,
                parts=(TranscriptPart(kind=PartKind.TEXT, text="hi"),),
            ),
        )
    )
    output = render_transcript(with_body)
    assert '<oc-message n="1" role="assistant" time="1970-01-01T00:00:09Z">' in output
    assert output.endswith("</oc-message>\n</oc-session>")


def test_self_closed_vs_paired_tool() -> None:
    transcript = _transcript(
        messages=(
            _message(
                parts=(
                    TranscriptPart(kind=PartKind.TOOL, tool_name="bash"),
                    TranscriptPart(
                        kind=PartKind.TOOL,
                        tool_name="x",
                        tool_input="input body",
                    ),
                ),
            ),
        )
    )
    self_closed = render_transcript(transcript)
    assert '<oc-tool name="bash" />' in self_closed
    assert '<oc-tool name="x" />' in self_closed

    paired = render_transcript(transcript, TranscriptOptions(include_tool_inputs=True))
    assert '<oc-tool name="bash" />' in paired
    assert (
        '<oc-tool name="x">\n<oc-input>\ninput body\n</oc-input>\n</oc-tool>' in paired
    )


def test_tool_attribute_order_and_fallback_name() -> None:
    transcript = _transcript(
        messages=(
            _message(
                parts=(
                    TranscriptPart(kind=PartKind.TOOL, tool_title="No name here"),
                    TranscriptPart(kind=PartKind.TOOL, tool_name="x", tool_title="   "),
                    TranscriptPart(kind=PartKind.TOOL, tool_name="x", tool_status=""),
                ),
            ),
        )
    )
    output = render_transcript(transcript)
    assert '<oc-tool name="tool" title="No name here" />' in output
    assert '<oc-tool name="x" />' in output


def test_tool_part_with_child_session_id_attribute() -> None:
    part = TranscriptPart(
        kind=PartKind.TOOL,
        tool_name="task",
        tool_title="Explore",
        tool_status="completed",
        child_session_id="session-child",
    )
    transcript = _transcript(messages=(_message(parts=(part,)),))
    output = render_transcript(transcript)
    assert (
        '<oc-tool name="task" status="completed" title="Explore" '
        'session="session-child" />' in output
    )


def test_patch_part_lists_files_and_is_omitted_when_empty() -> None:
    with_files = _transcript(
        messages=(
            _message(
                parts=(TranscriptPart(kind=PartKind.PATCH, files=("a.py", "b.py")),)
            ),
        )
    )
    output = render_transcript(with_files)
    assert '<oc-patch files="a.py, b.py" />' in output

    without_files = _transcript(
        messages=(_message(parts=(TranscriptPart(kind=PartKind.PATCH, files=()),)),)
    )
    empty_output = render_transcript(without_files)
    assert "<oc-patch" not in empty_output


def test_file_part_attached_and_omitted_without_filename() -> None:
    with_filename = _transcript(
        messages=(
            _message(
                parts=(TranscriptPart(kind=PartKind.FILE, filename="bin/debugpy"),)
            ),
        )
    )
    assert '<oc-file name="bin/debugpy" />' in render_transcript(with_filename)

    without_filename = _transcript(
        messages=(_message(parts=(TranscriptPart(kind=PartKind.FILE),)),)
    )
    assert "<oc-file" not in render_transcript(without_filename)


def test_agent_part() -> None:
    transcript = _transcript(
        messages=(
            _message(
                parts=(TranscriptPart(kind=PartKind.AGENT, agent_name="pr-reviewer"),)
            ),
        )
    )
    assert '<oc-agent name="pr-reviewer" />' in render_transcript(transcript)


def test_compaction_part_auto_and_manual() -> None:
    auto = _transcript(
        messages=(
            _message(
                parts=(TranscriptPart(kind=PartKind.COMPACTION, compaction_auto=True),),
            ),
        )
    )
    assert '<oc-compaction auto="true" />' in render_transcript(auto)

    manual = _transcript(
        messages=(
            _message(
                parts=(
                    TranscriptPart(kind=PartKind.COMPACTION, compaction_auto=False),
                ),
            ),
        )
    )
    assert '<oc-compaction auto="false" />' in render_transcript(manual)


def test_message_error_attribute() -> None:
    transcript = _transcript(
        messages=(
            _message(
                parts=(TranscriptPart(kind=PartKind.TEXT, text="Partial reply"),),
                error="MessageAbortedError",
            ),
        )
    )
    output = render_transcript(transcript)
    assert 'error="MessageAbortedError"' in output
    assert '<oc-message n="1" role="user" error="MessageAbortedError">' in output


def test_tool_bodies_excluded_by_default_and_each_flag_independent() -> None:
    part = TranscriptPart(
        kind=PartKind.TOOL,
        tool_name="bash",
        tool_input='{"command": "git status --short"}',
        tool_output="M README.md",
    )
    transcript = _transcript(messages=(_message(parts=(part,)),))

    default_output = render_transcript(transcript)
    assert "<oc-input" not in default_output
    assert "<oc-output" not in default_output

    inputs_only = render_transcript(
        transcript, TranscriptOptions(include_tool_inputs=True)
    )
    assert "<oc-input>" in inputs_only
    assert "<oc-output" not in inputs_only
    assert '{"command": "git status --short"}' in inputs_only

    outputs_only = render_transcript(
        transcript, TranscriptOptions(include_tool_outputs=True)
    )
    assert "<oc-output>" in outputs_only
    assert "<oc-input" not in outputs_only

    both = render_transcript(
        transcript,
        TranscriptOptions(include_tool_inputs=True, include_tool_outputs=True),
    )
    assert "<oc-input>" in both
    assert "<oc-output>" in both


def test_truncation_attribute_exact_value() -> None:
    part = TranscriptPart(kind=PartKind.TOOL, tool_name="x", tool_input="0123456789")
    transcript = _transcript(messages=(_message(parts=(part,)),))
    output = render_transcript(
        transcript, TranscriptOptions(include_tool_inputs=True, max_tool_chars=5)
    )
    assert '<oc-input truncated="5">\n01234\n</oc-input>' in output


def test_truncation_disabled_for_zero_and_negative() -> None:
    part = TranscriptPart(kind=PartKind.TOOL, tool_name="x", tool_input="0123456789")
    transcript = _transcript(messages=(_message(parts=(part,)),))

    zero_output = render_transcript(
        transcript, TranscriptOptions(include_tool_inputs=True, max_tool_chars=0)
    )
    assert "truncated" not in zero_output
    assert "<oc-input>\n0123456789\n</oc-input>" in zero_output

    negative_output = render_transcript(
        transcript, TranscriptOptions(include_tool_inputs=True, max_tool_chars=-1)
    )
    assert "truncated" not in negative_output
    assert "0123456789" in negative_output


def test_truncation_not_applied_when_content_fits_exactly() -> None:
    part = TranscriptPart(kind=PartKind.TOOL, tool_name="x", tool_input="01234")
    transcript = _transcript(messages=(_message(parts=(part,)),))
    output = render_transcript(
        transcript, TranscriptOptions(include_tool_inputs=True, max_tool_chars=5)
    )
    assert "truncated" not in output


def test_child_session_nests_inside_matching_tool() -> None:
    child_session = Session(
        id="session-child",
        title="Explore",
        created=None,
        updated=None,
        project_id=None,
        directory=None,
    )
    child = _transcript(session=child_session)
    parent = _transcript(
        messages=(
            _message(
                parts=(
                    TranscriptPart(
                        kind=PartKind.TOOL,
                        tool_name="task",
                        tool_title="Explore the codebase",
                        child_session_id="session-child",
                    ),
                ),
            ),
        ),
        children=(child,),
    )
    output = render_transcript(parent)
    expected = (
        '<oc-tool name="task" title="Explore the codebase" session="session-child">\n'
        "<oc-session\n"
        '  id="session-child"\n'
        '  title="Explore"\n'
        '  messages="0"\n'
        ">\n"
        "</oc-session>\n"
        "</oc-tool>"
    )
    assert expected in output


def test_unmatched_child_lands_at_end_of_session() -> None:
    unmatched = _transcript(session=_session(id="session-orphan", title="Orphan"))
    parent = _transcript(
        messages=(_message(parts=(TranscriptPart(kind=PartKind.TEXT, text="hi"),)),),
        children=(unmatched,),
    )
    output = render_transcript(parent)
    assert output.index("</oc-message>") < output.index('id="session-orphan"')
    assert output.endswith("</oc-session>\n</oc-session>")


def test_two_levels_of_child_nesting() -> None:
    grandchild = _transcript(session=_session(id="session-gc", title="Grandchild"))
    child = _transcript(
        session=_session(id="session-child", title="Child"),
        messages=(
            _message(
                parts=(
                    TranscriptPart(
                        kind=PartKind.TOOL,
                        tool_name="task",
                        child_session_id="session-gc",
                    ),
                ),
            ),
        ),
        children=(grandchild,),
    )
    parent = _transcript(
        messages=(
            _message(
                parts=(
                    TranscriptPart(
                        kind=PartKind.TOOL,
                        tool_name="task",
                        child_session_id="session-child",
                    ),
                ),
            ),
        ),
        children=(child,),
    )
    output = render_transcript(parent)
    root_index = output.index('id="session-root"')
    child_index = output.index('id="session-child"')
    gc_index = output.index('id="session-gc"')
    assert root_index < child_index < gc_index


def test_attribute_escaping() -> None:
    session = _session(title='Title with "quotes" & <tags>\nand a newline')
    transcript = _transcript(session=session)
    output = render_transcript(transcript)
    assert (
        'title="Title with &quot;quotes&quot; &amp; &lt;tags&gt; and a newline"'
        in output
    )
    assert "\n" not in output.split('title="')[1].split('"')[0]


def test_nonce_forces_suffixed_prefix_and_is_deterministic() -> None:
    body = "here is a literal <oc-text> tag in content"
    transcript = _transcript(
        messages=(_message(parts=(TranscriptPart(kind=PartKind.TEXT, text=body),)),)
    )
    first = render_transcript(transcript)
    second = render_transcript(transcript)
    assert first == second

    # A suffixed prefix must be in use throughout the whole document.
    import re

    match = re.search(r"<(oc-[0-9a-f]+-)session", first)
    assert match is not None
    prefix = match.group(1)
    assert f"<{prefix}message" in first
    assert f"<{prefix}text>" in first
    assert f"<{prefix}text" not in body  # sanity: not accidentally in source

    extracted = first.split(f"<{prefix}text>\n", 1)[1].split(f"\n</{prefix}text>", 1)[0]
    assert extracted == body


def test_nonce_not_triggered_by_unrelated_content() -> None:
    transcript = _transcript(
        messages=(
            _message(parts=(TranscriptPart(kind=PartKind.TEXT, text="ordinary"),)),
        )
    )
    output = render_transcript(transcript)
    assert "<oc-session" in output
    assert "<oc-text>" in output


def test_nonce_excludes_bodies_not_included_by_options() -> None:
    # A tool input containing "<oc-" should not force a nonce unless
    # include_tool_inputs is set, since it will not appear in the document.
    part = TranscriptPart(
        kind=PartKind.TOOL, tool_name="x", tool_input="contains <oc-text> literally"
    )
    transcript = _transcript(messages=(_message(parts=(part,)),))
    default_output = render_transcript(transcript)
    assert "<oc-session" in default_output
    assert "<oc-tool" in default_output

    with_input = render_transcript(
        transcript, TranscriptOptions(include_tool_inputs=True)
    )
    assert not with_input.startswith("<oc-session")
    import re

    match = re.search(r"<(oc-[0-9a-f]+-)session", with_input)
    assert match is not None


def test_multiple_transcripts_have_no_separator() -> None:
    first = _transcript(session=_session(id="s1", title="First"))
    second = _transcript(session=_session(id="s2", title="Second"))
    output = render_transcripts([first, second])
    assert output.count("<oc-session") == 2
    assert "---" not in output
    first_index = output.index('id="s1"')
    second_index = output.index('id="s2"')
    assert first_index < second_index
    assert output.endswith("\n")
    assert not output.endswith("\n\n")
    # The first session's close tag must be immediately followed by the
    # second session's open tag on the next line, with no blank line.
    boundary = output.index("</oc-session>\n<oc-session")
    assert boundary >= 0


def test_todos_rendered_with_status_and_omitted_when_none() -> None:
    transcript = _transcript(
        todos=(
            TranscriptTodo(content="Investigate the parser", status="completed"),
            TranscriptTodo(content="Write the tests", status="pending"),
            TranscriptTodo(content="Untouched", status=None),
        )
    )
    output = render_transcript(transcript)
    assert '<oc-todo status="completed">Investigate the parser</oc-todo>' in output
    assert '<oc-todo status="pending">Write the tests</oc-todo>' in output
    assert "<oc-todo>Untouched</oc-todo>" in output


def test_todos_omitted_when_empty() -> None:
    transcript = _transcript(todos=())
    assert "<oc-todos" not in render_transcript(transcript)


def test_metadata_attributes_omitted_for_none_values() -> None:
    session = Session(
        id="minimal",
        title="Minimal",
        created=None,
        updated=None,
        project_id=None,
        directory=None,
    )
    transcript = _transcript(session=session)
    output = render_transcript(transcript)
    for label in (
        "repository=",
        "directory=",
        "parent=",
        "agent=",
        "model=",
        "provider=",
        "created=",
        "updated=",
        "cost=",
        "tokens=",
        "archived=",
    ):
        assert label not in output
    assert 'id="minimal"' in output
    assert 'messages="0"' in output


def test_model_without_provider_has_no_provider_attribute() -> None:
    session = _session(model="gpt-test", provider=None)
    transcript = _transcript(session=session)
    output = render_transcript(transcript)
    assert 'model="gpt-test"' in output
    assert "provider=" not in output


def test_tokens_omitted_when_total_is_none() -> None:
    session = _session(
        tokens=TokenBreakdown(
            input=10, output=None, reasoning=3, cache_read=4, cache_write=5
        )
    )
    transcript = _transcript(session=session)
    output = render_transcript(transcript)
    assert "tokens=" not in output


def test_tokens_omitted_when_absent() -> None:
    session = _session(tokens=None)
    transcript = _transcript(session=session)
    output = render_transcript(transcript)
    assert "tokens=" not in output


def test_archived_attribute_true_only_when_true() -> None:
    active = _transcript(session=_session(archived=None))
    assert "archived=" not in render_transcript(active)

    inactive = _transcript(session=_session(archived=False))
    assert "archived=" not in render_transcript(inactive)

    archived = _transcript(session=_session(archived=True))
    assert 'archived="true"' in render_transcript(archived)


def test_blank_title_is_omitted_not_rendered_as_blank() -> None:
    none_title = _transcript(session=_session(title=None))
    assert "title=" not in render_transcript(none_title)

    blank_title = _transcript(session=_session(title="   "))
    assert "title=" not in render_transcript(blank_title)


def test_output_ends_with_single_trailing_newline_and_no_trailing_ws() -> None:
    grandchild = _transcript(
        session=_session(id="session-gc", title="Grandchild"),
        todos=(TranscriptTodo(content="Nested todo", status="completed"),),
    )
    child = _transcript(
        session=_session(id="session-child", title="Child session"),
        messages=(
            _message(
                id="m-child",
                role="assistant",
                agent="build",
                model="gpt-test",
                created=1_500,
                parts=(
                    TranscriptPart(kind=PartKind.TEXT, text="Child reply"),
                    TranscriptPart(kind=PartKind.REASONING, text="thinking"),
                    TranscriptPart(
                        kind=PartKind.TOOL,
                        tool_name="bash",
                        tool_title="Run tests",
                        tool_status="completed",
                        tool_input="pytest -q",
                        tool_output="5 passed",
                    ),
                    TranscriptPart(kind=PartKind.PATCH, files=("a.py",)),
                    TranscriptPart(kind=PartKind.FILE, filename="bin/debugpy"),
                    TranscriptPart(kind=PartKind.AGENT, agent_name="pr-reviewer"),
                    TranscriptPart(kind=PartKind.COMPACTION, compaction_auto=True),
                ),
                error="MessageAbortedError",
            ),
        ),
        todos=(
            TranscriptTodo(content="Investigate", status="completed"),
            TranscriptTodo(content="Ship", status="in_progress"),
        ),
        children=(grandchild,),
    )
    root = _transcript(
        messages=(
            _message(
                id="m-user",
                role="user",
                created=1_000,
                parts=(TranscriptPart(kind=PartKind.TEXT, text="Root message"),),
            ),
        ),
        todos=(TranscriptTodo(content="Root todo", status="pending"),),
        children=(child,),
    )

    options = TranscriptOptions(
        include_reasoning=True,
        include_tool_inputs=True,
        include_tool_outputs=True,
    )
    output = render_transcripts([root, root], options)

    assert output.endswith("\n")
    assert not output.endswith("\n\n")
    lines = output.split("\n")
    assert lines[-1] == ""
    for line in lines:
        assert line == line.rstrip(), f"trailing whitespace in line: {line!r}"
    for previous, current in pairwise(lines):
        assert not (previous == "" and current == ""), "found a double blank line"


def test_round_trip_via_elementtree() -> None:
    """Parse the rendered document as XML and check ``oc-text`` bodies match.

    This only works because the fixture content below is Markdown-heavy but
    free of raw ``<``/``&`` characters. Verbatim bodies are deliberately not
    escaped, so content containing those characters is not guaranteed to be
    well-formed XML; that is an accepted limitation of this test, not of the
    renderer.
    """
    user_text = "## Heading\n\nSome *markdown* text.\n\n- item one\n- item two"
    assistant_text = "Here is a fenced block:\n\n```\ncode line\n```\n\nDone."
    transcript = _transcript(
        messages=(
            _message(
                id="m-user",
                role="user",
                created=1_000,
                parts=(TranscriptPart(kind=PartKind.TEXT, text=user_text),),
            ),
            _message(
                id="m-assistant",
                role="assistant",
                created=2_000,
                parts=(TranscriptPart(kind=PartKind.TEXT, text=assistant_text),),
            ),
        )
    )
    output = render_transcript(transcript)
    wrapped = f"<root>\n{output}\n</root>"
    root = ET.fromstring(wrapped)
    text_elements = root.findall(".//oc-text")
    assert len(text_elements) == 2
    assert text_elements[0].text == f"\n{user_text}\n"
    assert text_elements[1].text == f"\n{assistant_text}\n"
