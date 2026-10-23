---
name: work-log
description: Write a standup or weekly update from the user's OpenCode sessions — what they worked on, grouped by ticket, with the PRs they opened — show it in chat and, when configured, send it to the user on Slack. Use when asked for a standup, a work log, a weekly update, "what did I do yesterday/this week", or a summary of recent work for a status update.
---

# Work log

Builds the update from `op2 sessions log`, which reports each session active in
a window with the days it was used, the tickets it mentioned, and the pull
requests it created. Read-only everywhere except for at most one Slack message
to the user themselves.

## 0. Load the settings

Read `${XDG_CONFIG_HOME:-$HOME/.config}/opencode-power-pack/config.json` if it
exists. Every field is optional:

```json
{
  "time_zone": "America/Toronto",
  "slack_user_id": "U0123456789",
  "exclude_repos": ["scratch"]
}
```

- `time_zone`: IANA zone for days and the window. Defaults to the system
  zone, which on a remote machine is often UTC rather than the user's.
- `slack_user_id`: who to send the update to on Slack. Without it, the update
  is shown in chat only.
- `exclude_repos`: repositories whose sessions are not work to report, such as
  automated jobs.

Below, `$ZONE` is `time_zone`. When it is not set, drop `TZ=$ZONE` and
`--tz "$ZONE"` from the commands so they use the system zone.

## 1. Pick the window

Use what the user asks for; otherwise:

- **Standup** (the default): the previous workday, from 00:00 until now.
  Monday looks back to Friday.
- **Weekly update**: Monday 00:00 of the current week until now, or the
  previous week when asked on a Monday.

`--since` and `--until` need ISO 8601 with an offset:

```sh
n=$(case $(TZ=$ZONE date +%u) in 1) echo 3;; 7) echo 2;; *) echo 1;; esac)
TZ=$ZONE date -d "today 00:00 $n days ago" -Iseconds
```

## 2. Gather

```sh
op2 sessions log --since <start> --tz "$ZONE" --format json
```

Add `--exclude-repo <name>` for each entry in `exclude_repos`.

Each record has `session` (title, repository, directory), `activeDays`,
`userMessages`, `tickets` (most mentioned first), `pullRequestsCreated`, and
`pullRequestsReferenced`. A `gh stack submit` pull request has a `number`,
`branch`, and `directory` but may have no `url`.

Then, in parallel, enrich what is there with whichever tools are available.
Skip any that are missing or fail, and note the gap in the footer:

- The main ticket of each session, from the issue tracker, for its title and
  state. With the Linear CLI: `linear issue view <ID> --json --no-comments`.
  Only tickets that are some session's first ticket; later ones are often just
  mentioned in passing.
- Each created pull request: `gh pr view <url-or-number> --json title,state,url`,
  run from its `directory` when there is no URL.
- The user's own merges in the window:
  `gh search prs --author @me --merged --merged-at ">=<start date>" --json title,url,repository --limit 30`.

## 3. Read sessions where needed

A title is usually enough. For sessions with many messages or no clear title,
read the end of the conversation to see what was concluded:

```sh
op2 sessions show <session-id>
```

Skip sessions that were only questions or explanations with nothing decided,
unless they took a large part of the window.

## 4. Write it

Use Slack mrkdwn so the same text works in chat and in Slack. Group by ticket,
then put unticketed work by repository. Write outcomes, not activity: "Added
retry backoff to the uploader (PR)" rather than "Worked on the uploader". One
line per item, most significant first.

Standup:

```
*Standup: <Weekday, Mon D>* (since <window start>)

*Done*
• <PROJ-123 title>: <outcome>. <url|PR #N> _(merged)_
• <repo>: <outcome>

*In progress*
• <ticket or repo>: <where it stands, what's next>

*Blockers / questions*
• <only if a session ended blocked or waiting on someone; otherwise omit>

_From N sessions. <gaps, e.g. "issue tracker unavailable">_
```

A weekly update uses the same sections, titled `*Week of <Mon D>*`, and may
run longer. Omit an empty section. Never invent outcomes: if a session's
result is unclear, say what it was about rather than what it achieved. If the
user's instructions require messages written by an agent to carry an
attribution line, add it at the top.

## 5. Deliver

1. Show the update in chat.
2. If `slack_user_id` is set and a Slack tool that sends messages is
   available, send the update to that user ID, which delivers it as a direct
   message to themselves. Confirm in one line that it was sent, or why not.
   If either is missing, skip this step silently.

Do not post anywhere else, including the issue tracker or channels, unless the
user asks in this conversation.
