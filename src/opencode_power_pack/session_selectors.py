"""Session selector parsing and resolution shared by mutating commands.

Selectors arrive as command arguments or on standard input, which lets
``op2 sessions list`` feed ``op2 sessions show`` and ``op2 sessions delete``.
A selector is a full session ID or any prefix that identifies exactly one
session.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import IO, cast

from .errors import UsageError
from .models import RecordModel, Session

_MAX_AMBIGUOUS_CANDIDATES = 5


class SessionSelector(RecordModel):
    """A user-supplied session reference and where it came from."""

    text: str
    origin: str

    def describe(self) -> str:
        """Return the selector and its origin for error messages."""
        return f"{self.text!r} ({self.origin})"


class SessionResolver:
    """Resolve selectors to sessions by exact ID first, then unique prefix."""

    def __init__(self, sessions: Iterable[Session]) -> None:
        self._by_id = {session.id: session for session in sessions}

    def known(self, text: str) -> bool:
        """Return whether ``text`` is exactly a known session ID."""
        return text in self._by_id

    def resolve(self, selector: SessionSelector) -> Session:
        """Return the single session identified by ``selector``."""
        exact = self._by_id.get(selector.text)
        if exact is not None:
            return exact
        matches = sorted(
            (
                session
                for identifier, session in self._by_id.items()
                if identifier.startswith(selector.text)
            ),
            key=lambda session: session.id,
        )
        if not matches:
            raise UsageError(f"no session matches {selector.describe()}")
        if len(matches) > 1:
            shown = [session.id for session in matches[:_MAX_AMBIGUOUS_CANDIDATES]]
            listed = ", ".join(shown)
            if len(matches) > _MAX_AMBIGUOUS_CANDIDATES:
                listed += ", ..."
            raise UsageError(
                f"{selector.describe()} is ambiguous; "
                f"{len(matches)} sessions match: {listed}"
            )
        return matches[0]

    def resolve_all(self, selectors: Sequence[SessionSelector]) -> tuple[Session, ...]:
        """Resolve every selector, keeping first-seen order and dropping repeats."""
        resolved: dict[str, Session] = {}
        for selector in selectors:
            session = self.resolve(selector)
            resolved.setdefault(session.id, session)
        return tuple(resolved.values())


def selectors_from_arguments(values: Sequence[str]) -> tuple[SessionSelector, ...]:
    """Return selectors for command-line session arguments."""
    return tuple(
        SessionSelector(text=value.strip(), origin="argument")
        for value in values
        if value.strip()
    )


def selectors_from_stdin(
    stream: IO[str], resolver: SessionResolver
) -> tuple[SessionSelector, ...]:
    """Return selectors read from ``stream``.

    JSON and JSON Lines input contribute their ``id`` fields. Any other input is
    treated as one record per line: fields are split on tabs, or on whitespace
    when the line has none, and the field naming a session wins. This accepts the
    tab-separated ``sessions list`` output as well as bare IDs.
    """
    text = stream.read()
    if not text.strip():
        return ()
    structured = _selectors_from_json(text)
    if structured is not None:
        return structured
    selectors: list[SessionSelector] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        origin = f"stdin line {number}"
        selectors.append(
            SessionSelector(
                text=_field_naming_a_session(line, resolver, origin), origin=origin
            )
        )
    return tuple(selectors)


def _field_naming_a_session(line: str, resolver: SessionResolver, origin: str) -> str:
    fields = [
        field.strip()
        for field in (line.split("\t") if "\t" in line else line.split())
        if field.strip()
    ]
    if not fields:
        raise UsageError(f"{origin} contains no session ID")
    if len(fields) == 1:
        return fields[0]
    exact = [field for field in fields if resolver.known(field)]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise UsageError(f"{origin} names {len(exact)} sessions: " + ", ".join(exact))
    raise UsageError(
        f"{origin} contains no recognizable session ID; "
        "pipe `sessions list` output or one session ID per line"
    )


def _selectors_from_json(text: str) -> tuple[SessionSelector, ...] | None:
    stripped = text.lstrip()
    if not stripped.startswith(("[", "{")):
        return None
    documents = _json_documents(text)
    selectors: list[SessionSelector] = []
    for index, document in enumerate(documents, start=1):
        origin = "stdin json" if len(documents) == 1 else f"stdin json record {index}"
        selectors.extend(_selectors_from_document(document, origin))
    return tuple(selectors)


def _json_documents(text: str) -> tuple[object, ...]:
    try:
        return (json.loads(text),)
    except ValueError:
        pass
    documents: list[object] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            documents.append(json.loads(line))
        except ValueError as error:
            raise UsageError(f"stdin line {number} is not valid JSON") from error
    return tuple(documents)


def _selectors_from_document(
    document: object, origin: str
) -> tuple[SessionSelector, ...]:
    if isinstance(document, str):
        return (SessionSelector(text=document.strip(), origin=origin),)
    if isinstance(document, Mapping):
        mapping = cast(Mapping[str, object], document)
        for key in ("id", "sessionId", "sessionID"):
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return (SessionSelector(text=value.strip(), origin=origin),)
        session = mapping.get("session")
        if isinstance(session, Mapping):
            return _selectors_from_document(cast(Mapping[str, object], session), origin)
        raise UsageError(f"{origin} has no session id field")
    if isinstance(document, Sequence):
        selectors: list[SessionSelector] = []
        for item in cast(Sequence[object], document):
            selectors.extend(_selectors_from_document(item, origin))
        return tuple(selectors)
    raise UsageError(f"{origin} is not a session record")
