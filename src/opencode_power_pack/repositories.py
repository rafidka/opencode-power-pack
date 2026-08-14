"""Repository selection helpers for OpenCode projects."""

import os
from collections.abc import Sequence
from pathlib import Path

from .errors import UsageError
from .models import Repository


class RepositoryNotFoundError(UsageError):
    """A repository selector did not match a known repository."""


class AmbiguousRepositoryError(UsageError):
    """A repository selector matched more than one known repository."""


class RepositoryResolver:
    """Resolve concise repository selectors against known repositories."""

    def __init__(self, repositories: Sequence[Repository]) -> None:
        self._repositories = tuple(repositories)

    def resolve(self, query: str) -> Repository:
        """Return the uniquely best repository match for ``query``."""
        normalized = _normalize(query)
        folded = normalized.casefold()
        matches: list[tuple[int, Repository]] = []
        for repository in self._repositories:
            rank = self._match_rank(repository, folded)
            if rank is not None:
                matches.append((rank, repository))
        if not matches:
            msg = f"no repository matches {query!r}"
            raise RepositoryNotFoundError(msg)
        best = min(rank for rank, _ in matches)
        candidates = [repository for rank, repository in matches if rank == best]
        if len(candidates) != 1:
            names = ", ".join(
                f"{repository.display_name} ({repository.path})"
                for repository in candidates
            )
            msg = f"repository selector {query!r} is ambiguous: {names}"
            raise AmbiguousRepositoryError(msg)
        return candidates[0]

    def resolve_many(self, queries: Sequence[str]) -> frozenset[str]:
        """Resolve selectors and return their union of project identifiers."""
        return frozenset(self.resolve(query).project_id for query in queries)

    def resolve_cwd(self, path: Path) -> Repository:
        """Resolve a working directory by its deepest containing repository path."""
        target = _normalize(os.path.abspath(os.fspath(path)))
        candidates: list[tuple[int, Repository]] = []
        for repository in self._repositories:
            paths = (repository.path, *repository.aliases)
            depths = [
                len(_parts(candidate))
                for candidate in paths
                if _contains(candidate, target)
            ]
            if depths:
                candidates.append((max(depths), repository))
        if not candidates:
            msg = f"no repository contains {os.fspath(path)!r}"
            raise RepositoryNotFoundError(msg)
        deepest = max(depth for depth, _ in candidates)
        matches = [repository for depth, repository in candidates if depth == deepest]
        if len(matches) != 1:
            names = ", ".join(
                f"{repository.display_name} ({repository.path})"
                for repository in matches
            )
            msg = f"working directory {os.fspath(path)!r} is ambiguous: {names}"
            raise AmbiguousRepositoryError(msg)
        return matches[0]

    @staticmethod
    def _match_rank(repository: Repository, query: str) -> int | None:
        if repository.project_id.casefold() == query:
            return 0
        paths = tuple(
            _normalize(path) for path in (repository.path, *repository.aliases)
        )
        folded_paths = tuple(path.casefold() for path in paths)
        if query in folded_paths:
            return 1
        basenames = tuple(os.path.basename(path) for path in folded_paths)
        if query in basenames:
            return 2
        if any(name.startswith(query) for name in basenames):
            return 3
        if any(query in name for name in basenames):
            return 4
        if any(
            query in component for path in folded_paths for component in _parts(path)
        ):
            return 5
        if any(query in path for path in folded_paths):
            return 6
        return None


def _normalize(path: str) -> str:
    return os.path.normpath(os.path.expanduser(path))


def _parts(path: str) -> tuple[str, ...]:
    return tuple(part for part in Path(path).parts if part not in {os.sep, ""})


def _contains(parent: str, child: str) -> bool:
    try:
        return os.path.commonpath((_normalize(parent), child)) == _normalize(parent)
    except ValueError:
        return False
