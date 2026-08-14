from pathlib import Path

import pytest

from opencode_power_pack.models import Repository
from opencode_power_pack.repositories import (
    AmbiguousRepositoryError,
    RepositoryResolver,
)


def test_resolves_ranked_selectors_and_cwd_without_existing_paths() -> None:
    resolver = RepositoryResolver(
        (
            Repository("one", "first", "/work/alpha", ("/old/alpha",)),
            Repository("two", "second", "/work/beta"),
        )
    )

    assert resolver.resolve("first").display_name == "one"
    assert resolver.resolve("alpha").project_id == "first"
    assert resolver.resolve_many(("alpha", "beta", "alpha")) == {"first", "second"}
    assert resolver.resolve_cwd(Path("/old/alpha/nested")).project_id == "first"


def test_ambiguous_repository_selector_lists_candidates() -> None:
    resolver = RepositoryResolver(
        (
            Repository("one", "one", "/work/api-client"),
            Repository("two", "two", "/work/api-server"),
        )
    )

    with pytest.raises(AmbiguousRepositoryError, match="api"):
        resolver.resolve("api")
