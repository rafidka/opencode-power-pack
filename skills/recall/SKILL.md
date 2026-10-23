---
name: recall
description: Search the user's past OpenCode sessions and distilled notes before re-deriving an answer. Use proactively, without being asked, before explaining how a system works (internals, infrastructure, deployments, secrets, setup steps) or re-investigating something that may have been worked out before, and whenever the user refers to a past discussion ("we talked about this", "last time", "remember when"). Skip it for questions answered by the code in front of you.
---

# Recall

People ask about the same systems across many sessions and repositories, and
the answer often already exists in an earlier session or note. Check before
working it out again. A recall should cost a few seconds, so keep it to the
steps below and move on when nothing relevant turns up.

## 1. Search

Run these in parallel with the shell tool. Phrase the query as a full
natural-language question, not keywords.

```sh
op2 sessions search "<question>" --exclude "$OPENCODE_SESSION_ID" --format jsonl -n 5
```

```sh
grep -ril -- "<distinctive term>" "${OP2_NOTES_DIR:-$HOME/notes/opencode}" 2>/dev/null
```

- `--exclude` keeps the current conversation and its sub-agents out of the
  results.
- `search` may first print `Indexing N new or changed sessions…` on stderr.
  That is expected.
- If `search` fails (no `COHERE_API_KEY`, network), fall back to literal
  matching with one or two distinctive terms, such as an error message, a
  resource name, or a ticket ID:

  ```sh
  op2 sessions grep "<term>" --exclude "$OPENCODE_SESSION_ID" --format jsonl -n 10
  ```

- Notes are Markdown files written by the `distill` skill. A matching note is
  usually better than a raw session: it is already summarized and deduplicated.
  Read it with the read tool.

## 2. Judge relevance

Each search result has a `score`, Cohere Rerank's estimate that the snippet is
useful. Above about 0.5 is usually on topic; below about 0.3 rarely is. Use the
title, date, repository, and snippet as well, not the score alone. If nothing
is relevant, say nothing about the recall and answer normally.

## 3. Read the best one or two

```sh
op2 sessions show <session-id>
```

This prints the conversation without tool inputs or outputs, which is usually
enough. When the exact command or output matters, add
`--include-tool-inputs --include-tool-outputs --max-tool-chars 1000`.

## 4. Use it

- Treat what you find as a lead, not a fact. Systems change: check it against
  the current code, configuration, or documentation before relying on it, and
  say when you could not.
- Tell the user briefly where it came from, for example "We worked this out on
  Oct 18 in `infra` (Secret Manager key registration)", then answer. Do
  not paste the old conversation.
- If the earlier answer turns out to be wrong or outdated, say so.
- If the same topic keeps coming up and has no note yet, suggest running the
  `distill` skill on those sessions.
