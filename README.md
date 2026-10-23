# opencode-power-pack

`opencode-power-pack` inspects OpenCode data already stored on your machine.
The installed command is `opencode-power-pack`; `op2` is a shorter alias.

## Install

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv tool install .
op2 --help
```

For development, run `uv sync --all-groups`.

## Database and safety

OpenCode V2 is required. By default, the CLI reads:

```
${XDG_DATA_HOME:-$HOME/.local/share}/opencode/opencode.db
```

Use another database with the global `--database` option or
`OPENCODE_DATABASE`:

```bash
op2 --database /path/to/opencode.db sessions list
OPENCODE_DATABASE=/path/to/opencode.db op2 sessions grep "timeout"
```

The database is opened read-only with SQLite `mode=ro` and `query_only`. Only
V2's `session_v2` and `session_message` tables are read. A V1 database is
rejected: V2 copies V1 sessions into its own tables when it first starts and
stops updating the V1 tables afterwards, so reading them would show stale data.
`sessions delete` is the exception to read-only access: it does not write to
SQLite itself but asks OpenCode to perform the change, and only against the
database used by OpenCode's background service.

## Sessions

### List

```bash
# Recent root sessions from every repository
op2 sessions list

# Restrict to a repository and cap results
op2 sessions list --repo ~/src/my-project --max-count 20

# Use the repository containing the current directory
op2 sessions list --cwd

# Machine-readable output
op2 sessions list --format jsonl --sort updated --reverse
```

`--repo` may be repeated and cannot be combined with `--cwd`. By default only
active, substantive root sessions are returned. Use `--include-children`,
`--include-archived`, or `--archived-only` as needed. Common filters are
`--agent`, `--provider`, `--model`, `--since`, and `--until` (which is
exclusive). Sort with `--sort` (`updated`, `created`, `cost`, `tokens`, or
`title`); formats are `table`, `json`, and `jsonl`.

`table` is meant for reading. On a terminal it renders a bordered table; when
stdout is redirected or piped it prints the same columns as tab-separated
values with no header, so results can be fed to other commands.

### Trivial sessions

Throwaway sessions, such as tests and greetings, are hidden from `sessions
list`, `sessions search`, and `sessions grep`. `--include-trivial` shows them
alongside everything else and `--trivial-only` shows nothing else, which makes
cleaning them up a pipe:

```bash
op2 sessions list --trivial-only
op2 sessions list --trivial-only | op2 sessions delete --yes
```

A session is trivial when all of these hold: it is a root session with no
sub-agent sessions, it has not been updated for 24 hours, it made no tool
calls, it has at most one user message, and its user and assistant text is
shorter than 300 characters in total. Titles are ignored, because OpenCode
titles a session from its first message, so one titled "Test message" may
have gone on to do real work. Session selectors in `show` and `delete` still
resolve trivial sessions. The thresholds are constants in
`database/trivial.py`.

### Show

```bash
# A single conversation on stdout
op2 sessions show ses_8a3f2c

# Write a file to reference from another conversation
op2 sessions show ses_8a3f2c -o conversation.md

# The most recent conversation in the current repository
op2 sessions list --cwd --max-count 1 | op2 sessions show

# Several related conversations in one document, for summarizing
op2 sessions search "rate limit" --max-count 3 | op2 sessions show

# Full fidelity, including what every tool call sent and returned
op2 sessions show ses_8a3f2c --include-reasoning --include-tool-inputs --include-tool-outputs
```

`sessions show` renders one or more stored sessions as a transcript, suitable
for pasting or `@`-referencing into another conversation, or for feeding to a
summarizer. Output goes to stdout by default; `-o`/`--output` writes to a file
instead, overwriting it, matching shell redirection.

The transcript's structure is XML tags; message content is embedded verbatim,
abridged here to a session, one message, and a couple of tool calls:

```
<oc-session
  id="ses_f76551cebffexrwFNP7m17fSwk"
  title="Conversation dump to markdown command"
  repository="opencode-power-pack"
  agent="orchestrate"
  model="claude-opus-5"
  provider="cohere-llm-gateway"
  created="2026-09-15T18:15:03Z"
  updated="2026-09-15T18:56:21Z"
  cost="0.0000"
  tokens="1907779"
  messages="32"
>
<oc-message n="374" role="assistant" agent="plan" model="claude-opus-5" time="2026-09-11T19:18:04Z">
<oc-text>
Here's the guide. One decision up front, because the two failure modes overlap.

## Which task exposes both

**The `uccs` project inside `submittal-review`.** Its corpus is exactly the pair you want:
</oc-text>
<oc-tool name="shell" status="completed" title="git status --short" />
<oc-tool name="subagent" status="completed" title="Explore the codebase" session="ses_04fccdc8cffe">
</oc-tool>
</oc-message>
</oc-session>
```

Every Markdown token in the output is verbatim conversation content; the
transcript's own structure is XML tags only. Text inside `<oc-text>` is
byte-for-byte verbatim and never escaped or indented, because message content
is arbitrary Markdown written by users and models and re-escaping it would
corrupt it. Attribute values, being metadata rather than content, are
XML-escaped as usual.

Tags carry an `oc-` prefix rather than plain names like `<message>` or
`<file>`, because unprefixed names collide with real conversation content —
`<file>` alone appears in 141 message bodies, from the `read` tool's own
output format — while `oc-` collides in none. As a last-resort guarantee, if
content ever contains the literal `<oc-`, the whole document switches to a
deterministic hash-suffixed prefix such as `oc-a3f9-`.

Sub-agent sessions started by the `subagent` tool nest their own transcript inside
the `<oc-tool>` element that spawned them, recursively, only with
`--include-children`; otherwise that element is empty, as in the example
above. Tool inputs and outputs, when opted in, appear as `<oc-input>` and
`<oc-output>` children of `<oc-tool>`; a truncated `<oc-output>` carries a
`truncated` attribute giving the number of characters omitted. Tools are named
as V2 names them, including in sessions recorded by V1 (`bash` is `shell`,
`task` is `subagent`, `apply_patch` is `patch`). V2 stores no tool titles, so
the title is taken from the call's description, command, path, or similar
input.

Session selectors work exactly like `sessions delete`:
full IDs or any prefix identifying exactly one session, given as arguments or
read from standard input when no argument is given, including the `table`,
`json`, or `jsonl` output of `sessions list`, `sessions search`, or
`sessions grep`. Multiple
sessions are consecutive `<oc-session>` elements with no separator, which is
what makes `op2 sessions search "..." | op2 sessions show` useful for feeding
several related conversations into a single summary.

By default the transcript contains user and assistant text plus one line per
tool call giving the tool, its title, and its status. Tool inputs and outputs
account for roughly 90% of stored conversation bytes, so including them turns
a 4,000–11,000 token transcript into a 40,000–350,000 token one; they are
opt-in via `--include-tool-inputs` and `--include-tool-outputs`, each capped
per call by `--max-tool-chars` (default 2000, 0 for unlimited). Reasoning is
omitted by default, consistent with `sessions grep`, and included with
`--include-reasoning`. OpenCode's own reminders and notices are
not part of the transcript; compactions appear as a `compaction` message.

### Delete

```bash
# One or more session IDs, or any prefix identifying exactly one session
op2 sessions delete ses_8a3f2c --yes

# Delete the sessions of a repository that were already archived
op2 sessions list --repo ~/src/spike --archived-only | op2 sessions delete --yes
```

The command takes session IDs as arguments, and reads them from standard input
when no argument is given. Piped input may be the tab-separated `table` output
of `sessions list`, `sessions search`, or `sessions grep`, their `json` or
`jsonl` output, or one
session ID per line. Selectors resolve only when they identify a single session;
ambiguous prefixes produce an error and nothing is changed.

Deletion is permanent and also removes child sessions, messages, and the
session's event history, so it asks for confirmation unless `--yes` is passed,
and refuses to run without `--yes` when it cannot prompt.

The command does not modify SQLite directly; it calls OpenCode's own session
endpoint through `opencode api`, which keeps cascading rules owned by OpenCode.
`opencode api` finds or starts OpenCode's background service and authenticates
with it, so the `opencode` CLI must be on `PATH`. The service always uses its own
database, so the command refuses to run when the service is not the process
holding the selected database open. Exit status is `1` if any session failed,
and each failure is reported with its session ID.

There is no archive command, because OpenCode V2 has no API for archiving a
session. Sessions archived before the upgrade are still skipped by
`sessions list`, `sessions search`, and `sessions grep` unless
`--include-archived` or
`--archived-only` is given.

### Search

`sessions search` finds sessions by meaning, from a natural-language
description. It requires `COHERE_API_KEY` to already be set in the process
environment; configure it through your shell environment or secret manager
before invoking `op2`.

```bash
# The 5 sessions most relevant to a description
op2 sessions search "why did deployment fail?"

# Restrict to the current repository, as JSON
op2 sessions search "authentication issue" --cwd --format json

# Explicitly synchronize the index, or rescan/recreate it
op2 sessions index
op2 sessions index --force
op2 sessions index --rebuild --format json
```

Every search first brings the semantic index up to date. When there are new or
changed sessions to embed, it says so on stderr before doing it, for example
`Indexing 3 new or changed sessions (41 chunks to embed) before searching…`,
followed by a progress bar on a terminal. Searches with nothing to embed say
nothing. Machine-readable output on stdout is never affected. Running
`op2 sessions index` ahead of time moves that wait out of the search.

Indexing uses Cohere's `embed-v5.0-fast` model with 1024 dimensions; Cohere
failures include safe, actionable status guidance without exposing request
content or credentials. An index built with a different model is rejected;
rebuild it with `op2 sessions index --rebuild`.

A search pre-ranks indexed chunks by cosine similarity, takes the best 50 (at
most 3 from any one session), and rescores them with Cohere's
`rerank-v5.0-fast`. Each session is ranked by its best reranked chunk, and its
score is Rerank's calibrated relevance: roughly the probability that the chunk
is useful. `--no-rerank` skips the Rerank call and ranks by cosine similarity
alone. The candidate count, per-session cap, and default result count are
constants in `semantic/service.py`.

Search covers titles and user/assistant text; `--in title` or `--in text`
restricts that, and repeated `--role user` or `--role assistant` filters message
content. It also accepts the list repository, archive, trivial, session, and time
filters, `-n`/`--max-count` (default 5), `--snippet-length`, and the `table`,
`json`, or `jsonl` output formats.

### Grep

`sessions grep` finds sessions containing literal text. It needs no API key
and sends nothing anywhere.

```bash
# Titles and user/assistant text containing a phrase
op2 sessions grep "migration failure"

# Search the current repository with a shorter snippet
op2 sessions grep "rate limit" --cwd --snippet-length 120

# Search titles and opted-in tool output
op2 sessions grep "deploy" --in title --include-tool-outputs --format json
```

Matching is literal and case-insensitive unless `--case-sensitive` is given, and
every match is returned unless `--max-count` limits it. Grep accepts the same
filters as search. `--include-reasoning`, `--include-tool-inputs`, and
`--include-tool-outputs` opt into content that search never indexes. Repository
selectors resolve only when they uniquely identify a repository; ambiguous
selectors produce an error.

## Privacy and performance

The CLI searches current materialized session data, not event history.
Reasoning and tool input/output may be sensitive or large, so they are omitted
unless explicitly requested. Broad searches scan selected session content.
`sessions show` follows the same rule: reasoning and tool inputs and
outputs are omitted unless explicitly requested.

For semantic indexing, only session titles and product-visible user/assistant
text are sent to Cohere, and reranking sends the query with those same chunks.
Reasoning, tool inputs, and tool outputs are excluded. Credentials are never
stored in the semantic sidecar.

OpenCode's database remains read-only for every reading command. The derived
semantic sidecar is stored
under `${XDG_CACHE_HOME:-~/.cache}/opencode-power-pack/`, separately for each
source database. `op2 sessions index` accepts `--semantic-index` to use a
different sidecar path. Semantic searches lazily synchronize before querying;
there is no daemon or background watcher. A stored fingerprint
of the main SQLite database plus its WAL skips scans when unchanged. SQLite's
SHM reader-lock metadata is intentionally not fingerprinted. A fingerprint that
cannot be read is treated as absent, so a sidecar written by another release
costs a rescan rather than an error.
When it changes, per-session `time_updated` revisions compare stable chunk IDs
and embed only new or modified chunks. Unchanged vectors are retained, while
removed chunks and sessions are deleted. Initial indexing can make paid API
calls; later searches against an unchanged source embed only the query and rerank
its candidates.

## Development

```bash
uv run pytest
uv run ruff check .
uv run pyright
uv run op2 --help
uv run op2 sessions list --help
uv run op2 sessions show --help
uv run op2 sessions search --help
uv run op2 sessions grep --help
uv run op2 sessions index --help
uv run op2 sessions delete --help
```
