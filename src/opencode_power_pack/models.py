"""Shared, storage-independent contracts for OpenCode project and session commands."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic.alias_generators import to_camel


class ValueModel(BaseModel):
    """Base class for immutable values passed between layers.

    Field names are used as-is. Internal values, queries, events, and anything
    persisted in a private cache belong here, so that the published output
    contract can change independently of them.
    """

    model_config = ConfigDict(frozen=True)


class RecordModel(ValueModel):
    """Base class for records that commands emit as machine-readable output.

    Fields are named in snake case and serialized in camel case, so Python code
    and the published JSON contract each keep their own conventions without a
    hand-written conversion.
    """

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        alias_generator=to_camel,
    )


class OutputFormat(StrEnum):
    """Output encodings supported by commands that return structured data."""

    TABLE = "table"
    JSON = "json"
    JSONL = "jsonl"


class AppContext(ValueModel):
    """Process-wide command settings supplied by the root CLI callback."""

    database: Path | None = None


class ArchiveMode(StrEnum):
    """Whether a query includes active, archived, or all sessions."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    ALL = "all"


class TrivialMode(StrEnum):
    """Whether a query includes substantive, trivial, or all sessions."""

    SUBSTANTIVE = "substantive"
    TRIVIAL = "trivial"
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
    REASONING = "reasoning"
    TOOL_INPUT = "tool-input"
    TOOL_OUTPUT = "tool-output"


class PartKind(StrEnum):
    """The part kinds that carry conversation content."""

    TEXT = "text"
    REASONING = "reasoning"
    TOOL = "tool"
    FILE = "file"
    COMPACTION = "compaction"
    AGENT = "agent"


class Project(RecordModel):
    """An OpenCode project, its canonical worktree, and known directory aliases."""

    id: str
    worktree: str
    name: str | None = None
    directory_aliases: tuple[str, ...] = ()

    @computed_field
    @property
    def path(self) -> str:
        """Return the canonical worktree path."""
        return self.worktree


class Repository(RecordModel):
    """A repository candidate with a stable display name and canonical path."""

    display_name: str
    project_id: str
    path: str
    aliases: tuple[str, ...] = ()


class TokenBreakdown(RecordModel):
    """Token counts, where ``total`` is the sum when every component is known."""

    input: int | None = None
    output: int | None = None
    reasoning: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None

    @computed_field
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


class Session(RecordModel):
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


class SessionFilter(ValueModel):
    """Common immutable filters shared by list and search queries."""

    repository_project_ids: frozenset[str] = frozenset()
    include_children: bool = False
    archive_mode: ArchiveMode = ArchiveMode.ACTIVE
    trivial_mode: TrivialMode = TrivialMode.SUBSTANTIVE
    excluded_session_ids: frozenset[str] = frozenset()
    agents: frozenset[str] = frozenset()
    providers: frozenset[str] = frozenset()
    models: frozenset[str] = frozenset()
    since: int | None = None
    until: int | None = None


class ListSessionsQuery(ValueModel):
    """A storage-independent query for listing sessions."""

    filters: SessionFilter = Field(default_factory=SessionFilter)
    sort: ListSort = ListSort.UPDATED
    direction: SortDirection = SortDirection.DESC
    limit: int | None = None


class SearchSessionsQuery(ValueModel):
    """A storage-independent query for searching session content."""

    text: str
    filters: SessionFilter = Field(default_factory=SessionFilter)
    sources: frozenset[SearchSource] = frozenset(
        {SearchSource.TITLE, SearchSource.TEXT}
    )
    roles: frozenset[str] = frozenset()
    case_sensitive: bool = False
    snippet_length: int = 200
    limit: int | None = None


class SearchSessionMatch(RecordModel):
    """A session match with its source, optional message role, and snippet."""

    session: Session
    source: SearchSource
    role: str | None
    snippet: str | None
    match_count: int


class ListSessionsOutput(ValueModel):
    """An optional aggregate list result; commands may emit ``sessions`` directly."""

    sessions: tuple[Session, ...]
    total: int


class SearchSessionsOutput(ValueModel):
    """An optional aggregate search result for callers that need query metadata."""

    query: SearchSessionsQuery
    matches: tuple[SearchSessionMatch, ...]
    total: int


class TranscriptPart(RecordModel):
    """One rendered unit of a message.

    A flat union discriminated by ``kind``, because parts are shallow and a
    flat record keeps the renderer a simple dispatch. ``text`` carries the
    body of ``TEXT`` and ``REASONING``. The ``tool_*`` fields describe a
    ``TOOL`` call, where ``tool_input`` and ``tool_output`` are
    already-serialized text. ``child_session_id`` is set for a ``subagent``
    tool call that spawned a sub-agent session. ``filename`` names a ``FILE``
    attachment. ``compaction_auto`` records whether a ``COMPACTION`` was
    automatic. ``agent_name`` names an ``AGENT`` mention.
    """

    kind: PartKind
    text: str | None = None
    tool_name: str | None = None
    tool_title: str | None = None
    tool_status: str | None = None
    tool_input: str | None = None
    tool_output: str | None = None
    child_session_id: str | None = None
    filename: str | None = None
    compaction_auto: bool | None = None
    agent_name: str | None = None


class TranscriptMessage(RecordModel):
    """One turn in a conversation, with the parts it produced.

    ``role`` is ``user``, ``assistant``, or ``compaction`` for the point where
    OpenCode summarized earlier context. ``error`` holds the kind of failure
    recorded against the turn, most commonly ``aborted``.
    """

    id: str
    role: str
    created: int | None = None
    agent: str | None = None
    model: str | None = None
    provider: str | None = None
    cost: float | None = None
    tokens: TokenBreakdown | None = None
    error: str | None = None
    parts: tuple[TranscriptPart, ...] = ()


class Transcript(RecordModel):
    """A full conversation for one session, with any spawned sub-agent transcripts.

    ``children`` is recursive and holds sub-agent transcripts; it is only
    populated when they were requested.
    """

    session: Session
    messages: tuple[TranscriptMessage, ...] = ()
    children: tuple[Transcript, ...] = ()


class TranscriptQuery(ValueModel):
    """A storage-independent query for reading transcripts.

    This deliberately covers only what changes which rows are read; how much
    of each part is rendered is a presentation concern owned by the Markdown
    renderer, so no reasoning or tool-body flags belong here.
    """

    session_ids: tuple[str, ...] = ()
    include_children: bool = False


class CreatedPullRequest(RecordModel):
    """A pull request created by ``gh pr create`` or ``gh stack submit``.

    ``gh stack submit`` reports a number and branch but no URL, so ``url`` is
    absent for those; ``directory`` is where the command ran, for looking the
    pull request up with ``gh``.
    """

    number: int | None = None
    url: str | None = None
    branch: str | None = None
    directory: str | None = None


class SessionActivity(RecordModel):
    """What one root session, with its sub-agents, did within a time window.

    ``active_days`` are local dates, in the requested time zone, on which a user
    message was sent. ``tickets`` are ordered by how often they were mentioned.
    """

    session: Session
    active_days: tuple[str, ...]
    user_messages: int
    tickets: tuple[str, ...] = ()
    pull_requests_created: tuple[CreatedPullRequest, ...] = ()
    pull_requests_referenced: tuple[str, ...] = ()


class ActivityQuery(ValueModel):
    """A storage-independent query for session activity within a window."""

    filters: SessionFilter = Field(default_factory=SessionFilter)
    since: int
    until: int | None = None
    time_zone: str
    ticket_prefixes: frozenset[str] = frozenset()
