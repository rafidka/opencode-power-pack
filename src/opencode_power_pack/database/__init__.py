"""Public facade for read-only OpenCode database access."""

from .client import OpenCodeDatabase, resolve_database_path

__all__ = ("OpenCodeDatabase", "resolve_database_path")
