"""Registration point for commands that inspect OpenCode sessions."""

from collections.abc import Callable
from importlib import import_module

import typer

CommandRegistrar = Callable[[typer.Typer], None]

app = typer.Typer(
    help="List and search sessions stored by OpenCode.",
    no_args_is_help=True,
)


def register_command(registrar: CommandRegistrar) -> CommandRegistrar:
    """Register a sessions command against this group and return the registrar."""
    registrar(app)
    return registrar


def register_available_commands() -> None:
    """Load optional command modules that expose a ``register`` function."""
    package = __name__
    for module_name in ("list_sessions", "search_sessions"):
        qualified_name = f"{package}.{module_name}"
        try:
            module = import_module(qualified_name)
        except ModuleNotFoundError as error:
            if error.name == qualified_name:
                continue
            raise
        registrar = getattr(module, "register", None)
        if not callable(registrar):
            msg = f"{qualified_name} must expose a callable register(app)"
            raise TypeError(msg)
        registrar(app)


register_available_commands()
