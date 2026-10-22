"""Terminal feedback while the semantic index synchronizes.

``sessions index`` shows a progress bar. ``sessions search`` synchronizes
before every query, so it also says up front when there are sessions to embed,
because that is the one time a search is noticeably slow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn

from opencode_power_pack.errors import OpencodePowerPackError, RuntimeCommandError

if TYPE_CHECKING:
    from opencode_power_pack.semantic import SemanticSyncEvent, SemanticSyncPlanned


class SyncProgressRenderer:
    """Render synchronization events as transient progress bars."""

    def __init__(self, console: Console) -> None:
        self._progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        )
        self._started = False
        self._documents_task: TaskID | None = None
        self._sessions_task: TaskID | None = None
        self._deleted_task: TaskID | None = None

    def __call__(self, event: SemanticSyncEvent) -> None:
        try:
            from opencode_power_pack.semantic import (
                SemanticEmbeddingBatchCompleted,
                SemanticSessionSynchronized,
                SemanticSyncPlanned,
            )

            if isinstance(event, SemanticSyncPlanned):
                self._plan(event)
                return
            if isinstance(event, SemanticEmbeddingBatchCompleted):
                self._advance(self._documents_task, event.document_count)
                return
            if isinstance(event, SemanticSessionSynchronized):
                self._advance(self._sessions_task, 1)
                return
            self._advance(self._deleted_task, len(event.session_ids))
        except OpencodePowerPackError:
            raise
        except Exception as error:
            raise RuntimeCommandError(
                f"could not render indexing progress: {error}"
            ) from error

    def _plan(self, event: SemanticSyncPlanned) -> None:
        if self._started:
            return
        if (
            event.document_count == 0
            and event.changed_session_count == 0
            and event.deleted_session_count == 0
        ):
            return
        self._progress.start()
        self._started = True
        if event.document_count:
            self._documents_task = self._progress.add_task(
                "Embedding documents", total=event.document_count
            )
        if event.changed_session_count:
            self._sessions_task = self._progress.add_task(
                "Synchronizing sessions", total=event.changed_session_count
            )
        if event.deleted_session_count:
            self._deleted_task = self._progress.add_task(
                "Deleting sessions", total=event.deleted_session_count
            )

    def _advance(self, task_id: TaskID | None, amount: int) -> None:
        if task_id is not None:
            self._progress.advance(task_id, amount)

    def stop(self) -> None:
        if self._started:
            self._progress.stop()
            self._started = False


class SearchSyncNotice:
    """Announce the sessions a search has to embed, then show their progress.

    The notice goes to stderr whatever the output format, so machine-readable
    stdout is never affected; the progress bar is shown only on a terminal.
    Synchronizations with nothing to embed stay silent.
    """

    def __init__(self, console: Console, *, show_progress: bool) -> None:
        self._console = console
        self._progress = SyncProgressRenderer(console) if show_progress else None

    def __call__(self, event: SemanticSyncEvent) -> None:
        from opencode_power_pack.semantic import SemanticSyncPlanned

        if isinstance(event, SemanticSyncPlanned) and event.document_count:
            self._console.print(_notice(event), highlight=False, soft_wrap=True)
        if self._progress is not None:
            self._progress(event)

    def stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()


def _notice(event: SemanticSyncPlanned) -> str:
    sessions = event.changed_session_count
    chunks = event.document_count
    return (
        f"Indexing {sessions} new or changed {_plural(sessions, 'session')} "
        f"({chunks} {_plural(chunks, 'chunk')} to embed) before searching…"
    )


def _plural(count: int, noun: str) -> str:
    return noun if count == 1 else f"{noun}s"
