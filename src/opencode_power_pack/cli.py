"""Root Typer application and executable entry point."""

from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .commands.sessions import app as sessions_app
from .errors import OpencodePowerPackError, RuntimeCommandError, exit_for_error
from .models import AppContext

app = typer.Typer(
    name="opencode-power-pack",
    help="Useful command-line tools for working with OpenCode data.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(sessions_app, name="sessions")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def get_app_context(context: typer.Context) -> AppContext:
    """Return the root :class:`AppContext` for a nested Typer command context."""
    app_context = context.find_root().obj
    if not isinstance(app_context, AppContext):
        msg = "CLI application context was not initialized"
        raise RuntimeCommandError(msg)
    return app_context


@app.callback()
def root(
    context: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
    database: Annotated[
        Path | None,
        typer.Option(
            "--database",
            envvar="OPENCODE_DATABASE",
            metavar="PATH",
            help="Use PATH as the OpenCode database instead of the default location.",
        ),
    ] = None,
) -> None:
    """Run an OpenCode Power Pack command."""
    context.obj = AppContext(database=database)


def main() -> None:
    """Execute the command-line application with expected-error handling."""
    try:
        app()
    except OpencodePowerPackError as error:
        exit_for_error(error)
