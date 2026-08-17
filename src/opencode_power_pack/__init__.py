"""OpenCode Power Pack command-line tools."""

__version__ = "0.1.0"


def main() -> None:
    """Run the command-line interface."""
    from .cli import main as cli_main

    cli_main()


__all__ = ["__version__", "main"]
