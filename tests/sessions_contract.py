"""Published JSON shapes shared by the search command tests."""

# The published JSON shape, written out rather than derived from the models, so a
# change to how they serialize cannot quietly change the contract.
EXPECTED_SESSION = {
    "id": "ses_123",
    "title": "A title\nwith whitespace",
    "created": 1,
    "updated": 1_700_000_000_000,
    "projectId": "project-1",
    "directory": "/work/example",
    "parentId": None,
    "repository": "example-repo",
    "agent": None,
    "provider": None,
    "model": None,
    "cost": None,
    "tokens": None,
    "archived": None,
    "archivedAt": None,
}
