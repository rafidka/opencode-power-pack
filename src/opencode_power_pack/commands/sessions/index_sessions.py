"""The ``sessions index`` command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console

from opencode_power_pack.database import resolve_database_path
from opencode_power_pack.errors import OpencodePowerPackError, UsageError
from opencode_power_pack.models import OutputFormat
from opencode_power_pack.output import emit_json, emit_table

from ._index_progress import SyncProgressRenderer

if TYPE_CHECKING:
    from opencode_power_pack.semantic import (
        CohereEmbedProvider,
        SemanticService,
        SemanticSyncResult,
    )


def _stderr_console() -> Console:
    return Console(stderr=True)


def _progress_renderer(output_format: OutputFormat) -> SyncProgressRenderer | None:
    if output_format is not OutputFormat.TABLE:
        return None
    console = _stderr_console()
    if not console.is_terminal:
        return None
    return SyncProgressRenderer(console)


def _semantic_dependencies() -> tuple[
    str,
    int,
    type[CohereEmbedProvider],
    type[SemanticService],
]:
    from opencode_power_pack.semantic import (
        COHERE_DIMENSION,
        COHERE_MODEL,
        CohereEmbedProvider,
        SemanticService,
    )

    return COHERE_MODEL, COHERE_DIMENSION, CohereEmbedProvider, SemanticService


def register(app: typer.Typer) -> None:
    """Register the semantic index synchronization command."""
    if not any(command.name == "index" for command in app.registered_commands):
        app.command("index")(index_sessions)


def index_sessions(
    context: typer.Context,
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Clear and recreate semantic chunks."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Scan session revisions even if the source is current."
        ),
    ] = False,
    semantic_index: Annotated[
        Path | None,
        typer.Option(
            "--semantic-index",
            metavar="PATH",
            help="Use PATH for the semantic index sidecar.",
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format: table, json, or jsonl."),
    ] = OutputFormat.TABLE,
) -> None:
    """Synchronize the semantic index for OpenCode sessions."""
    try:
        _run_index_sessions(
            context,
            rebuild=rebuild,
            force=force,
            semantic_index=semantic_index,
            output_format=output_format,
        )
    except OpencodePowerPackError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(error.exit_code) from None


def _run_index_sessions(
    context: typer.Context,
    *,
    rebuild: bool,
    force: bool,
    semantic_index: Path | None,
    output_format: OutputFormat,
) -> None:
    if force and rebuild:
        raise UsageError("--force cannot be used with --rebuild")

    model, dimension, provider_type, service_type = _semantic_dependencies()
    provider = provider_type()

    from opencode_power_pack.cli import get_app_context

    app_context = get_app_context(context)
    database_path = resolve_database_path(app_context.database)
    service = service_type(database_path, provider, sidecar_path=semantic_index)
    progress = _progress_renderer(output_format)
    try:
        result = service.sync(
            force=force,
            rebuild=rebuild,
            on_progress=progress,
        )
    finally:
        if progress is not None:
            progress.stop()
    _emit_result(result, service, output_format, model, dimension)


def _emit_result(
    result: SemanticSyncResult,
    service: SemanticService,
    output_format: OutputFormat,
    model: str,
    dimension: int,
) -> None:
    payload = {
        "noOp": result.no_op,
        "indexedSessionIds": list(result.indexed_session_ids),
        "deletedSessionIds": list(result.deleted_session_ids),
        "indexedSessionCount": result.indexed_session_count,
        "deletedSessionCount": result.deleted_session_count,
        "sourceChanged": result.source_changed,
        "provider": "cohere",
        "model": model,
        "dimension": dimension,
        "sourceDatabase": str(service.source_database),
        "sidecarPath": str(service.sidecar_path),
    }
    if output_format in {OutputFormat.JSON, OutputFormat.JSONL}:
        emit_json(payload)
        return

    if result.no_op:
        typer.echo("Semantic index is already current.")
    else:
        typer.echo(
            "Semantic index indexed "
            f"{result.indexed_session_count} and removed "
            f"{result.deleted_session_count} sessions."
        )
    if result.source_changed:
        typer.echo("Warning: source changed during indexing; run another sync.")
    emit_table(
        ("Provider", "Model", "Dimension", "Source DB", "Sidecar"),
        (
            (
                "cohere",
                model,
                dimension,
                service.source_database,
                service.sidecar_path,
            ),
        ),
    )
