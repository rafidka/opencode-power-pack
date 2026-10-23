---
name: distill
description: Turn OpenCode sessions into durable topic notes — how a system works, decisions and why, commands that worked, gotchas — saved as Markdown in the user's notes repository, where the recall skill finds them. Use when asked to distill, save, or take notes on a conversation ("save this", "remember this", "write this up as a note"), to backfill notes from past sessions, or when recall keeps finding the same topic in raw sessions.
---

# Distill

Notes live in `${OP2_NOTES_DIR:-$HOME/notes/opencode}`, a local git
repository with one Markdown file per topic. They are written for a future
session that knows nothing about this one, and are what the `recall` skill
searches first.

## What belongs in a note

Keep knowledge that will still be true and useful in a month:

- how a system or component works, and why it is built that way;
- decisions, with the reasoning and the alternatives rejected;
- procedures and the exact commands that worked;
- gotchas, failure modes, and how they were diagnosed.

Leave out task progress, PR review back-and-forth, anything only about a
change that has since merged, and anything already in the repository's own
documentation. A session can produce no note at all; say so rather than
writing a thin one. Never copy secrets, tokens, or credentials into a note,
even ones that appear in a session.

## 1. Choose the sources

- Nothing specified: the current conversation. You already have it in context.
- Session IDs, or a description to find them by:
  `op2 sessions search "<description>" --exclude "$OPENCODE_SESSION_ID" --format jsonl -n 5`
- Read each with `op2 sessions show <session-id>`. Add
  `--include-tool-outputs --max-tool-chars 1000` when exact commands or output
  matter.

## 2. Find existing notes

Prepare the notes repository on first use:

```sh
dir="${OP2_NOTES_DIR:-$HOME/notes/opencode}"
[ -d "$dir/.git" ] || { mkdir -p "$dir" && git -C "$dir" init -q; }
```

List titles and tags with `grep -H -m 2 -E '^(title|tags):' "$dir"/*.md`, and
grep for the topic's distinctive terms. **Update a matching note instead of
creating another.** Notes are per topic, not per session.

## 3. Write

File name: the topic in kebab case, such as `billing-invoice-retries.md`.
Format:

```markdown
---
title: Billing invoice retries
tags: [billing, queues, retries]
repos: [billing-service]
sources:
  - ses_0123456789abcdefghijklmn  # 2026-10-22 Why invoices were sent twice
created: 2026-10-23
updated: 2026-10-23
status: current
---

# Billing invoice retries

## Summary

Two or three sentences a reader could stop after.

## Details

How it works, decisions and why. Use code references (`path/to/file.py`,
function names) rather than pasted code.

## Commands

Commands that worked, with what they are for.

## Gotchas

## Open questions
```

Omit empty sections. When updating a note: merge rather than append, add the
new session to `sources`, set `updated`, and when something it said is no
longer true, correct it and say what changed and when, rather than keeping
both versions. Set `status: superseded` and link the replacement when a note
is replaced by another. Separate what was verified, such as by running a
command or reading code, from what was inferred, and write the inferred parts
as such.

## 4. Commit

```sh
git -C "$dir" add -A && git -C "$dir" commit -q -m "distill: <note titles>"
```

If the commit fails, for example because GPG signing needs an unlock, leave
the changes uncommitted and say so. Never push: the repository holds internal
information and has no remote unless the user adds one.

Finish with one line per note: created or updated, its path, and its title.

## Backfilling

To distill many past sessions, such as "distill my billing-service sessions
from September":

1. List candidates: `op2 sessions list --repo <repo> --since <time> --format json`.
2. Pick sessions whose titles suggest durable knowledge: explanations,
   investigations, spikes, setup, and design discussions. Skip briefings and
   routine implementation.
3. Give batches of about five sessions to `general` sub-agents in parallel.
   Each reads its sessions and **returns draft notes as text without writing
   files**, along with the existing note each draft should update, if any.
4. Merge drafts on the same topic, write the files yourself, and commit once.
   Then list what was created and updated so the user can review the diff with
   `git -C "$dir" show`.
