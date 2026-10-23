"""Tests for the sessions log command and session activity extraction."""

import json
from pathlib import Path

import pytest
from opencode_v2 import (
    assistant,
    create_database,
    insert_message,
    insert_project,
    insert_session,
    text,
    tool,
    user,
)
from typer.testing import CliRunner

from opencode_power_pack.cli import app
from opencode_power_pack.database import OpenCodeDatabase
from opencode_power_pack.models import ActivityQuery, SessionFilter, TrivialMode

runner = CliRunner()

_DAY = 24 * 60 * 60 * 1_000
_START = 1_792_216_000_000  # 2026-10-17T05:46:40Z, 2026-10-16 22:46 in Vancouver
_PR = "https://github.com/acme/widget/pull/{}"


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "opencode.db"
    connection = create_database(path)
    insert_project(connection, "p", "/work/widget", "widget")
    insert_project(connection, "q", "/work/briefing", "briefing")

    insert_session(
        connection, "root", title="PROJ-12 parser fix", updated=_START + 2 * _DAY
    )
    insert_message(
        connection,
        "root",
        "r-u1",
        "user",
        user(f"Work on proj-12 and PROJ-34, see {_PR.format(1)}; UTF-8 is fine"),
        created=_START,
    )
    insert_message(
        connection,
        "root",
        "r-a1",
        "assistant",
        assistant(
            text(f"Opened {_PR.format(2)}"),
            tool(
                "shell",
                input={"command": "cd /work/widget && gh pr create --fill"},
                output=_PR.format(2),
            ),
            tool(
                "shell",
                input={"command": "grep -r 'gh stack submit' ."},
                output="Created PR #99 for not/real",
                call_id="quoted",
            ),
        ),
        created=_START + 1,
    )
    insert_message(
        connection, "root", "r-u2", "user", user("Next day"), created=_START + _DAY
    )
    insert_message(
        connection, "root", "r-old", "user", user("PROJ-77"), created=_START - _DAY
    )

    insert_session(
        connection, "child", parent_id="root", title="Implement", updated=_START
    )
    insert_message(
        connection,
        "child",
        "c-a1",
        "assistant",
        assistant(
            tool(
                "shell",
                input={"command": "gh stack submit --auto", "workdir": "/work/widget"},
                output=(
                    "✓ Created PR #3 for me/feat/three\n"
                    "✓ Created PR #4 for me/feat/four"
                ),
            ),
            text(f"PR {_PR.format(4)} is ready"),
        ),
        created=_START + 2,
    )

    insert_session(
        connection,
        "briefing",
        project_id="q",
        directory="/work/briefing",
        title="Slack briefing",
        updated=_START,
    )
    insert_message(
        connection,
        "briefing",
        "b-u1",
        "user",
        user("Briefing mentioning PROJ-99 " + "x" * 400),
        created=_START,
    )

    insert_session(connection, "idle", title="Old work", updated=_START - _DAY)
    insert_message(
        connection, "idle", "i-u1", "user", user("x" * 400), created=_START - _DAY
    )
    connection.commit()
    connection.close()
    return path


def _activity(database: Path, **changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "filters": SessionFilter(trivial_mode=TrivialMode.ALL),
        "since": _START,
        "until": _START + 3 * _DAY,
        "time_zone": "America/Vancouver",
    }
    values.update(changes)
    with OpenCodeDatabase(database) as opened:
        activities = opened.session_activity(ActivityQuery(**values))  # pyright: ignore[reportArgumentType]
    return {activity.session.id: activity for activity in activities}


def test_activity_folds_sub_agents_into_their_root(database: Path) -> None:
    activities = _activity(database)
    assert set(activities) == {"root", "briefing"}
    root = activities["root"]
    assert root.user_messages == 2
    assert root.active_days == ("2026-10-16", "2026-10-17")
    assert [(pr.number, pr.url, pr.branch) for pr in root.pull_requests_created] == [
        (2, _PR.format(2), None),
        (3, None, "me/feat/three"),
        (4, _PR.format(4), "me/feat/four"),
    ]
    assert root.pull_requests_created[1].directory == "/work/widget"
    assert root.pull_requests_referenced == (_PR.format(1),)


def test_tickets_are_upper_case_keys_ordered_by_mentions(database: Path) -> None:
    root = _activity(database)["root"]
    assert root.tickets == ("PROJ-12", "PROJ-34")


def test_ticket_prefixes_match_any_case(database: Path) -> None:
    root = _activity(database, ticket_prefixes=frozenset({"proj"}))["root"]
    assert root.tickets == ("PROJ-12", "PROJ-34")


def test_window_counts_only_messages_sent_within_it(database: Path) -> None:
    activities = _activity(database, since=_START + _DAY)
    assert set(activities) == {"root"}
    assert activities["root"].user_messages == 1
    assert activities["root"].pull_requests_created == ()


def _log(database: Path, *options: str) -> str:
    result = runner.invoke(
        app,
        [
            "--database",
            str(database),
            "sessions",
            "log",
            "--since",
            "2026-10-17T00:00:00Z",
            "--until",
            "2026-10-20T00:00:00Z",
            "--tz",
            "America/Vancouver",
            *options,
        ],
    )
    assert result.exit_code == 0, result.output
    return result.stdout


def test_markdown_groups_by_main_ticket(database: Path) -> None:
    output = _log(database)
    assert output.startswith(
        "# Work log: 2026-10-16 to 2026-10-19 (America/Vancouver)\n"
    )
    assert "## PROJ-12\n\n- **PROJ-12 parser fix** · widget" in output
    assert "  - Created #3 me/feat/three (in /work/widget)" in output
    assert f"  - Created {_PR.format(4)}" in output
    assert "  - Also mentions PROJ-34" in output
    assert "## PROJ-99" in output


def test_exclude_repo_and_json(database: Path) -> None:
    records = json.loads(
        _log(database, "--exclude-repo", "briefing", "--format", "json")
    )
    assert [record["session"]["id"] for record in records] == ["root"]
    assert records[0]["pullRequestsCreated"][0]["url"] == _PR.format(2)


def test_rejects_unknown_time_zone(database: Path) -> None:
    result = runner.invoke(
        app, ["--database", str(database), "sessions", "log", "--tz", "Mars/Base"]
    )
    assert result.exit_code != 0
    assert "unknown time zone" in result.output


def test_system_time_zone_prefers_tz_then_localtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opencode_power_pack.commands.sessions import log_sessions

    monkeypatch.setenv("TZ", ":Europe/Paris")
    assert log_sessions.system_time_zone() == "Europe/Paris"

    monkeypatch.delenv("TZ")
    link = tmp_path / "localtime"
    link.symlink_to("/usr/share/zoneinfo/America/Toronto")
    monkeypatch.setattr(log_sessions, "_LOCALTIME", str(link))
    assert log_sessions.system_time_zone() == "America/Toronto"

    monkeypatch.setattr(log_sessions, "_LOCALTIME", str(tmp_path / "missing"))
    assert log_sessions.system_time_zone() == "UTC"
