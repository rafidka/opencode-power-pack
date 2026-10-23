"""What sessions did within a time window: active days, tickets, and pull requests.

Activity is attributed to root sessions, folding in their sub-agent sessions,
because an ``implement`` sub-agent often creates the pull request for work its
parent planned. Only messages sent within the window count.
"""

import re
import sqlite3
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import cast
from zoneinfo import ZoneInfo

from ..models import (
    ActivityQuery,
    CreatedPullRequest,
    Session,
    SessionActivity,
)
from ._utils import content, json_object, string
from .messages import ASSISTANT, USER, content_blocks, tool_name
from .sessions import filtered_sessions

_TICKET = re.compile(r"\b([A-Za-z]{2,10})-([0-9]{1,6})\b")
_NOT_TICKETS = frozenset({"UTF", "SHA", "ISO", "RFC", "GPT", "PEP", "TLS", "SSL"})
"""Upper-case prefixes that look like ticket keys but name standards or models."""

_PULL_REQUEST_URL = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/pull/(\d+)")
_STACK_CREATED = re.compile(r"Created PR #(\d+) for (\S+)")
_CREATE_COMMAND = re.compile(
    r"(?:^|&&|\|\||;|\|)\s*(?:timeout\s+\S+\s+)?gh\s+(?:pr\s+create|stack\s+submit)\b",
    re.MULTILINE,
)
"""``gh pr create`` or ``gh stack submit`` in command position, not merely quoted."""
_CD_PREFIX = re.compile(r"^\s*cd\s+(\S+)\s*&&")


def session_activity(
    connection: sqlite3.Connection, query: ActivityQuery
) -> tuple[SessionActivity, ...]:
    """Return the activity of root sessions with a user message in the window."""
    roots = {
        session.id: session
        for session in filtered_sessions(connection, query.filters)
        if session.parent_id is None
    }
    parents: dict[str, str | None] = {
        row["id"]: row["parent_id"]
        for row in connection.execute("SELECT id, parent_id FROM session_v2")
    }
    window = "time_created >= ?" + (
        "" if query.until is None else " AND time_created < ?"
    )
    bounds = (query.since,) if query.until is None else (query.since, query.until)
    zone = ZoneInfo(query.time_zone)
    collectors: dict[str, _Collector] = {}
    for row in connection.execute(
        "SELECT session_id, type, time_created, data FROM session_message "
        f"WHERE {window} AND type IN (?, ?) ORDER BY time_created, session_id, seq",
        (*bounds, USER, ASSISTANT),
    ):
        root_id = _root(row["session_id"], parents)
        root = roots.get(root_id)
        if root is None:
            continue
        collector = collectors.get(root_id)
        if collector is None:
            collector = collectors[root_id] = _Collector(root, query.ticket_prefixes)
        data = json_object(row["data"])
        if row["type"] == USER:
            collector.user_message(row["time_created"], data, zone)
        else:
            collector.assistant_message(data)
    activities = [
        collector.activity() for collector in collectors.values() if collector.days
    ]
    activities.sort(key=lambda activity: activity.session.id)
    activities.sort(key=lambda activity: activity.session.updated or -1, reverse=True)
    return tuple(activities)


def _root(session_id: str, parents: dict[str, str | None]) -> str:
    seen = {session_id}
    while (parent := parents.get(session_id)) is not None and parent not in seen:
        seen.add(parent)
        session_id = parent
    return session_id


class _Collector:
    def __init__(self, session: Session, ticket_prefixes: frozenset[str]) -> None:
        self.session = session
        self.prefixes = frozenset(prefix.upper() for prefix in ticket_prefixes)
        self.days: set[str] = set()
        self.user_messages = 0
        self.tickets: Counter[str] = Counter()
        self.created: dict[tuple[object, ...], CreatedPullRequest] = {}
        self.referenced: dict[str, None] = {}
        self._tickets(session.title)

    def user_message(
        self, created: int, data: dict[str, object], zone: ZoneInfo
    ) -> None:
        self.user_messages += 1
        self.days.add(datetime.fromtimestamp(created / 1_000, zone).date().isoformat())
        text = string(data.get("text"))
        self._tickets(text)
        self._references(text)

    def assistant_message(self, data: dict[str, object]) -> None:
        for block in content_blocks(data):
            block_type = string(block.get("type"))
            if block_type == "text":
                self._references(string(block.get("text")))
            elif block_type == "tool":
                self._tool(block)

    def activity(self) -> SessionActivity:
        by_number = {
            int(match.group(1)): url
            for url in self.referenced
            if (match := _PULL_REQUEST_URL.fullmatch(url)) is not None
        }
        created = [
            pr.model_copy(update={"url": by_number[pr.number]})
            if pr.url is None and pr.number in by_number
            else pr
            for pr in self.created.values()
        ]
        created_urls = {pr.url for pr in created if pr.url}
        return SessionActivity(
            session=self.session,
            active_days=tuple(sorted(self.days)),
            user_messages=self.user_messages,
            tickets=tuple(
                ticket
                for ticket, _ in sorted(
                    self.tickets.items(), key=lambda item: (-item[1], item[0])
                )
            ),
            pull_requests_created=tuple(created),
            pull_requests_referenced=tuple(
                url for url in self.referenced if url not in created_urls
            ),
        )

    def _tool(self, block: dict[str, object]) -> None:
        state = _object(block.get("state"))
        raw_input = state.get("input")
        serialized = content(raw_input)
        self._tickets(serialized)
        self._references(serialized)
        if tool_name(block) != "shell":
            return
        values = _object(raw_input)
        command = string(values.get("command")) or ""
        if _CREATE_COMMAND.search(command) is None:
            return
        cd = _CD_PREFIX.match(command)
        directory = (
            string(values.get("workdir"))
            or (cd.group(1) if cd else None)
            or self.session.directory
        )
        output = "\n".join(
            text
            for item in _objects(state.get("content"))
            if (text := string(item.get("text"))) is not None
        )
        for match in _PULL_REQUEST_URL.finditer(output):
            pr = CreatedPullRequest(
                number=int(match.group(1)), url=match.group(0), directory=directory
            )
            self.created.setdefault(("url", pr.url), pr)
        for match in _STACK_CREATED.finditer(output):
            pr = CreatedPullRequest(
                number=int(match.group(1)), branch=match.group(2), directory=directory
            )
            self.created.setdefault(("stack", pr.number, directory), pr)

    def _tickets(self, text: str | None) -> None:
        if not text:
            return
        for match in _TICKET.finditer(text):
            prefix, number = match.group(1), match.group(2)
            if self.prefixes:
                if prefix.upper() in self.prefixes:
                    self.tickets[f"{prefix.upper()}-{number}"] += 1
            elif prefix.isupper() and prefix not in _NOT_TICKETS:
                self.tickets[f"{prefix}-{number}"] += 1

    def _references(self, text: str | None) -> None:
        if text:
            for match in _PULL_REQUEST_URL.finditer(text):
                self.referenced.setdefault(match.group(0), None)


def _object(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _objects(value: object) -> Iterable[dict[str, object]]:
    if not isinstance(value, list):
        return ()
    return (
        cast("dict[str, object]", item)
        for item in cast("list[object]", value)
        if isinstance(item, dict)
    )
