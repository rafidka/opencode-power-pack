"""Assembly of full conversation transcripts from V2 session records."""

import sqlite3
from collections.abc import Mapping

from ..models import Session, Transcript, TranscriptQuery
from .messages import session_messages


def session_transcripts(
    connection: sqlite3.Connection,
    query: TranscriptQuery,
    sessions_by_id: Mapping[str, Session],
) -> tuple[Transcript, ...]:
    """Build one transcript per requested session, preserving query order.

    Unknown session IDs are skipped. Sub-agent transcripts are inlined into
    ``children`` when requested, following ``subagent`` tool calls recursively
    and guarding against cycles and repeats within a branch.
    """
    return tuple(
        _build_transcript(connection, query, sessions_by_id, session, frozenset())
        for session_id in query.session_ids
        if (session := sessions_by_id.get(session_id)) is not None
    )


def _build_transcript(
    connection: sqlite3.Connection,
    query: TranscriptQuery,
    sessions_by_id: Mapping[str, Session],
    session: Session,
    ancestors: frozenset[str],
) -> Transcript:
    messages = session_messages(connection, session.id)
    children: tuple[Transcript, ...] = ()
    if query.include_children:
        next_ancestors = ancestors | {session.id}
        child_ids = dict.fromkeys(
            part.child_session_id
            for message in messages
            for part in message.parts
            if part.child_session_id is not None
        )
        children = tuple(
            _build_transcript(
                connection, query, sessions_by_id, child_session, next_ancestors
            )
            for child_id in child_ids
            if child_id not in next_ancestors
            and (child_session := sessions_by_id.get(child_id)) is not None
        )
    return Transcript(session=session, messages=messages, children=children)
