"""Expected failures raised by command implementations."""

from typing import NoReturn

from .output import error_console


class OpencodePowerPackError(Exception):
    """Base class for errors that should be shown without a traceback."""

    exit_code = 1


class UsageError(OpencodePowerPackError):
    """An invalid value or combination of command-line arguments."""

    exit_code = 2


class RuntimeCommandError(OpencodePowerPackError):
    """An expected failure while executing a command."""


class DatabaseError(RuntimeCommandError):
    """A database could not be located, opened, or queried."""


class TimeParseError(UsageError):
    """A supplied timestamp does not follow the supported time contract."""


def exit_for_error(error: OpencodePowerPackError) -> NoReturn:
    """Render an expected command error to stderr and terminate the process."""
    error_console.print(f"Error: {error}")
    raise SystemExit(error.exit_code)
