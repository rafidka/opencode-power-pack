"""XML-tagged rendering for conversation transcripts.

This module is pure presentation: it takes the storage-independent
:class:`~opencode_power_pack.models.Transcript` tree and turns it into text
whose structure is expressed entirely as ``oc-`` prefixed XML tags. Message
content is arbitrary Markdown written by users and models, so every
Markdown-shaped token in the output (headings, rules, list items, fenced
code) is verbatim conversation content; the transcript's own structure never
borrows that syntax. This module performs no database access and no CLI
wiring, so it can be exercised entirely from in-memory fixtures and reused
by any output format built on top of it later.
"""

import hashlib
import re
from collections.abc import Sequence
from datetime import UTC, datetime

from opencode_power_pack.models import (
    PartKind,
    Session,
    TokenBreakdown,
    Transcript,
    TranscriptMessage,
    TranscriptPart,
    TranscriptTodo,
    ValueModel,
)

_DEFAULT_TAG_PREFIX = "oc-"
_NONCE_MARKER = "<oc-"
_NONCE_START_LENGTH = 4

_ATTR_ESCAPES = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    (">", "&gt;"),
    ('"', "&quot;"),
)


class TranscriptOptions(ValueModel):
    """Options controlling how much detail a transcript renders."""

    include_reasoning: bool = False
    include_tool_inputs: bool = False
    include_tool_outputs: bool = False
    max_tool_chars: int = 2000


_DEFAULT_OPTIONS = TranscriptOptions()


def render_transcripts(
    transcripts: Sequence[Transcript],
    options: TranscriptOptions = _DEFAULT_OPTIONS,
) -> str:
    """Render several transcripts as one document with no separator between them.

    Consecutive ``<oc-session>`` elements are already unambiguous, so nothing
    is inserted between sessions. The tag prefix is resolved once across
    every transcript, including nested children, so the whole document uses
    one consistent prefix.
    """
    prefix = _resolve_prefix(transcripts, options)
    documents = [
        _render_session(transcript, options, prefix) for transcript in transcripts
    ]
    return "\n".join(documents) + "\n"


def render_transcript(
    transcript: Transcript,
    options: TranscriptOptions = _DEFAULT_OPTIONS,
) -> str:
    """Render one transcript, including any sub-agent children, as one document."""
    prefix = _resolve_prefix((transcript,), options)
    return _render_session(transcript, options, prefix)


def _tag_suffix(attrs: str) -> str:
    return f" {attrs}" if attrs else ""


def _escape_attr(value: str) -> str:
    result = value
    for character, escaped in _ATTR_ESCAPES:
        result = result.replace(character, escaped)
    return re.sub(r"[\r\n\t]", " ", result)


def _attrs_line(pairs: Sequence[tuple[str, str | None]]) -> str:
    parts: list[str] = []
    for name, value in pairs:
        if value is None or value.strip() == "":
            continue
        parts.append(f'{name}="{_escape_attr(value)}"')
    return " ".join(parts)


def _format_timestamp(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms // 1_000, UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _tokens_str(tokens: TokenBreakdown | None) -> str | None:
    if tokens is None or tokens.total is None:
        return None
    return str(tokens.total)


def _truncate(content: str, max_chars: int) -> tuple[str, int]:
    if max_chars <= 0 or len(content) <= max_chars:
        return content, 0
    omitted = len(content) - max_chars
    return content[:max_chars], omitted


def _render_body(prefix: str, tag: str, content: str, truncated: int = 0) -> str:
    attrs = f' truncated="{truncated}"' if truncated > 0 else ""
    return f"<{prefix}{tag}{attrs}>\n{content}\n</{prefix}{tag}>"


def _render_todo(todo: TranscriptTodo, prefix: str) -> str:
    attrs = _attrs_line([("status", todo.status)])
    return f"<{prefix}todo{_tag_suffix(attrs)}>{todo.content}</{prefix}todo>"


def _render_todos(todos: Sequence[TranscriptTodo], prefix: str) -> str:
    tag = f"{prefix}todos"
    lines = [f"<{tag}>", *(_render_todo(todo, prefix) for todo in todos), f"</{tag}>"]
    return "\n".join(lines)


def _render_tool(
    part: TranscriptPart,
    options: TranscriptOptions,
    prefix: str,
    matched_child: Transcript | None,
) -> str:
    name = part.tool_name or "tool"
    attrs = _attrs_line(
        [
            ("name", name),
            ("status", part.tool_status),
            ("title", part.tool_title),
            ("session", part.child_session_id),
        ]
    )
    children: list[str] = []
    if options.include_tool_inputs and part.tool_input:
        content, omitted = _truncate(part.tool_input, options.max_tool_chars)
        children.append(_render_body(prefix, "input", content, omitted))
    if options.include_tool_outputs and part.tool_output:
        content, omitted = _truncate(part.tool_output, options.max_tool_chars)
        children.append(_render_body(prefix, "output", content, omitted))
    if matched_child is not None:
        children.append(_render_session(matched_child, options, prefix))

    tag = f"{prefix}tool"
    if not children:
        return f"<{tag}{_tag_suffix(attrs)} />"
    return "\n".join([f"<{tag}{_tag_suffix(attrs)}>", *children, f"</{tag}>"])


def _render_part(
    part: TranscriptPart,
    options: TranscriptOptions,
    prefix: str,
    children_by_id: dict[str, Transcript],
) -> tuple[str | None, str | None]:
    """Render one part, returning its chunk and the child session id it consumed."""
    if part.kind is PartKind.TEXT:
        if not part.text:
            return None, None
        return _render_body(prefix, "text", part.text), None
    if part.kind is PartKind.REASONING:
        if not options.include_reasoning or not part.text:
            return None, None
        return _render_body(prefix, "reasoning", part.text), None
    if part.kind is PartKind.TOOL:
        matched = (
            children_by_id.get(part.child_session_id) if part.child_session_id else None
        )
        chunk = _render_tool(part, options, prefix, matched)
        consumed = matched.session.id if matched is not None else None
        return chunk, consumed
    if part.kind is PartKind.FILE:
        if part.filename is None:
            return None, None
        attrs = _attrs_line([("name", part.filename)])
        return f"<{prefix}file{_tag_suffix(attrs)} />", None
    if part.kind is PartKind.AGENT:
        attrs = _attrs_line([("name", part.agent_name)])
        return f"<{prefix}agent{_tag_suffix(attrs)} />", None
    if part.kind is PartKind.COMPACTION:
        auto_value = (
            None
            if part.compaction_auto is None
            else ("true" if part.compaction_auto else "false")
        )
        attrs = _attrs_line([("auto", auto_value)])
        return f"<{prefix}compaction{_tag_suffix(attrs)} />", None
    return None, None


def _render_message(
    message: TranscriptMessage,
    index: int,
    options: TranscriptOptions,
    prefix: str,
    children_by_id: dict[str, Transcript],
) -> tuple[str, set[str]]:
    body_chunks: list[str] = []
    consumed: set[str] = set()
    for part in message.parts:
        chunk, used = _render_part(part, options, prefix, children_by_id)
        if chunk is not None:
            body_chunks.append(chunk)
        if used is not None:
            consumed.add(used)

    attrs = _attrs_line(
        [
            ("n", str(index)),
            ("role", message.role),
            ("agent", message.agent),
            ("model", message.model),
            ("provider", message.provider),
            (
                "time",
                _format_timestamp(message.created)
                if message.created is not None
                else None,
            ),
            ("cost", f"{message.cost:.4f}" if message.cost is not None else None),
            ("tokens", _tokens_str(message.tokens)),
            ("error", message.error),
        ]
    )

    tag = f"{prefix}message"
    if not body_chunks:
        return f"<{tag}{_tag_suffix(attrs)} />", consumed
    lines = [f"<{tag}{_tag_suffix(attrs)}>", *body_chunks, f"</{tag}>"]
    return "\n".join(lines), consumed


def _session_attr_lines(session: Session, message_count: int) -> list[str]:
    pairs: list[tuple[str, str | None]] = [
        ("id", session.id),
        ("title", session.title),
        ("repository", session.repository),
        ("directory", session.directory),
        ("parent", session.parent_id),
        ("agent", session.agent),
        ("model", session.model),
        ("provider", session.provider),
        (
            "created",
            _format_timestamp(session.created) if session.created is not None else None,
        ),
        (
            "updated",
            _format_timestamp(session.updated) if session.updated is not None else None,
        ),
        ("cost", f"{session.cost:.4f}" if session.cost is not None else None),
        ("tokens", _tokens_str(session.tokens)),
        ("messages", str(message_count)),
        ("archived", "true" if session.archived else None),
    ]
    lines: list[str] = []
    for name, value in pairs:
        if value is None or value.strip() == "":
            continue
        lines.append(f'  {name}="{_escape_attr(value)}"')
    return lines


def _render_session(
    transcript: Transcript,
    options: TranscriptOptions,
    prefix: str,
) -> str:
    session = transcript.session
    children_by_id = {child.session.id: child for child in transcript.children}
    consumed: set[str] = set()
    body_chunks: list[str] = []

    for index, message in enumerate(transcript.messages, start=1):
        chunk, used = _render_message(message, index, options, prefix, children_by_id)
        body_chunks.append(chunk)
        consumed |= used

    if transcript.todos:
        body_chunks.append(_render_todos(transcript.todos, prefix))

    for child in transcript.children:
        if child.session.id not in consumed:
            body_chunks.append(_render_session(child, options, prefix))

    tag = f"{prefix}session"
    lines = [f"<{tag}", *_session_attr_lines(session, len(transcript.messages)), ">"]
    lines.extend(body_chunks)
    lines.append(f"</{tag}>")
    return "\n".join(lines)


def _collect_bodies(
    transcripts: Sequence[Transcript], options: TranscriptOptions
) -> list[str]:
    bodies: list[str] = []
    for transcript in transcripts:
        _collect_transcript_bodies(transcript, options, bodies)
    return bodies


def _collect_transcript_bodies(
    transcript: Transcript, options: TranscriptOptions, bodies: list[str]
) -> None:
    for message in transcript.messages:
        for part in message.parts:
            if part.kind is PartKind.TEXT:
                if part.text:
                    bodies.append(part.text)
            elif part.kind is PartKind.REASONING:
                if options.include_reasoning and part.text:
                    bodies.append(part.text)
            elif part.kind is PartKind.TOOL:
                if options.include_tool_inputs and part.tool_input:
                    content, _ = _truncate(part.tool_input, options.max_tool_chars)
                    bodies.append(content)
                if options.include_tool_outputs and part.tool_output:
                    content, _ = _truncate(part.tool_output, options.max_tool_chars)
                    bodies.append(content)
    for todo in transcript.todos:
        bodies.append(todo.content)
    for child in transcript.children:
        _collect_transcript_bodies(child, options, bodies)


def _resolve_prefix(
    transcripts: Sequence[Transcript], options: TranscriptOptions
) -> str:
    """Resolve the ``oc-`` tag prefix, falling back to a nonce on collision.

    Every verbatim body that the given options will actually include is
    checked for the literal string ``<oc-``. If none contains it, the plain
    prefix is safe. Otherwise a deterministic suffix derived from those
    bodies' hash is grown until it no longer collides.
    """
    bodies = _collect_bodies(transcripts, options)
    if not any(_NONCE_MARKER in body for body in bodies):
        return _DEFAULT_TAG_PREFIX

    digest = hashlib.sha256("\n".join(bodies).encode("utf-8")).hexdigest()
    length = len(digest)
    n = _NONCE_START_LENGTH
    while n <= length:
        candidate = f"oc-{digest[:n]}-"
        if not any(f"<{candidate}" in body for body in bodies):
            return candidate
        n += 1
    return f"oc-{digest}-"
