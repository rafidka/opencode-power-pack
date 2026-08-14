"""Shared, storage-independent contracts for OpenCode project and session commands."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class OutputFormat(StrEnum):
    """Output encodings supported by commands that return structured data."""

    TABLE = "table"
    JSON = "json"
    JSONL = "jsonl"


@dataclass(frozen=True, slots=True)
class AppContext:
    """Process-wide command settings supplied by the root CLI callback."""

    database: Path | None = None


class ArchiveMode(StrEnum):
    """Whether a query includes active, archived, or all sessions."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    ALL = "all"


class ListSort(StrEnum):
    """Session fields available for list ordering."""

    UPDATED = "updated"
    CREATED = "created"
    COST = "cost"
    TOKENS = "tokens"
    TITLE = "title"


class SortDirection(StrEnum):
    """Ordering direction for list queries."""

    ASC = "asc"
    DESC = "desc"


class SearchSource(StrEnum):
    """The session content field in which a search match was found."""

    TITLE = "title"
    TEXT = "text"
    TODO = "todo"
    REASONING = "reasoning"
    TOOL_INPUT = "tool-input"
    TOOL_OUTPUT = "tool-output"


@dataclass(frozen=True, slots=True)
class Project:
    """An OpenCode project, its canonical worktree, and known directory aliases."""

    id: str
    worktree: str
    name: str | None = None
    directory_aliases: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        """Return the canonical worktree path."""
        return self.worktree

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic machine-output representation of this project."""
        return {
            "id": self.id,
            "worktree": self.worktree,
            "path": self.path,
            "name": self.name,
            "directoryAliases": list(self.directory_aliases),
        }


@dataclass(frozen=True, slots=True)
class Repository:
    """A repository candidate with a stable display name and canonical path."""

    display_name: str
    project_id: str
    path: str
    aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic machine-output representation of this repository."""
        return {
            "displayName": self.display_name,
            "projectId": self.project_id,
            "path": self.path,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class TokenBreakdown:
    """Token counts, where ``total`` is the sum when every component is known."""

    input: int | None = None
    output: int | None = None
    reasoning: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None

    @property
    def total(self) -> int | None:
        """Return all five components summed, or ``None`` when any is unknown."""
        values = (
            self.input,
            self.output,
            self.reasoning,
            self.cache_read,
            self.cache_write,
        )
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    def as_dict(self) -> dict[str, int | None]:
        """Return the nested, camel-case token representation."""
        return {
            "input": self.input,
            "output": self.output,
            "reasoning": self.reasoning,
            "cacheRead": self.cache_read,
            "cacheWrite": self.cache_write,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class Session:
    """A native-compatible session summary enriched with optional v1 metadata."""

    id: str
    title: str | None
    created: int | None
    updated: int | None
    project_id: str | None
    directory: str | None
    parent_id: str | None = None
    repository: str | None = None
    agent: str | None = None
    provider: str | None = None
    model: str | None = None
    cost: float | None = None
    tokens: TokenBreakdown | None = None
    archived: bool | None = None
    archived_at: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the stable camel-case machine-output representation."""
        return {
            "id": self.id,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "projectId": self.project_id,
            "directory": self.directory,
            "parentId": self.parent_id,
            "repository": self.repository,
            "agent": self.agent,
            "provider": self.provider,
            "model": self.model,
            "cost": self.cost,
            "tokens": None if self.tokens is None else self.tokens.as_dict(),
            "archived": self.archived,
            "archivedAt": self.archived_at,
        }


@dataclass(frozen=True, slots=True)
class SessionFilter:
    """Common immutable filters shared by list and search queries."""

    repository_project_ids: frozenset[str] = frozenset()
    include_children: bool = False
    archive_mode: ArchiveMode = ArchiveMode.ACTIVE
    agents: frozenset[str] = frozenset()
    providers: frozenset[str] = frozenset()
    models: frozenset[str] = frozenset()
    since: int | None = None
    until: int | None = None


@dataclass(frozen=True, slots=True)
class ListSessionsQuery:
    """A storage-independent query for listing sessions."""

    filters: SessionFilter = field(default_factory=SessionFilter)
    sort: ListSort = ListSort.UPDATED
    direction: SortDirection = SortDirection.DESC
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class SearchSessionsQuery:
    """A storage-independent query for searching session content."""

    text: str
    filters: SessionFilter = field(default_factory=SessionFilter)
    sources: frozenset[SearchSource] = frozenset(
        {SearchSource.TITLE, SearchSource.TEXT}
    )
    roles: frozenset[str] = frozenset()
    case_sensitive: bool = False
    snippet_length: int = 200
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class SearchSessionMatch:
    """A session match with its source, optional message role, and snippet."""

    session: Session
    source: SearchSource
    role: str | None
    snippet: str | None
    match_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic machine-output representation of this match."""
        return {
            "session": self.session.as_dict(),
            "source": self.source.value,
            "role": self.role,
            "snippet": self.snippet,
            "matchCount": self.match_count,
        }


@dataclass(frozen=True, slots=True)
class ListSessionsOutput:
    """An optional aggregate list result; commands may emit ``sessions`` directly."""

    sessions: tuple[Session, ...]
    total: int


@dataclass(frozen=True, slots=True)
class SearchSessionsOutput:
    """An optional aggregate search result for callers that need query metadata."""

    query: SearchSessionsQuery
    matches: tuple[SearchSessionMatch, ...]
    total: int
