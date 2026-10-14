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

By default, the CLI reads:

```
${XDG_DATA_HOME:-$HOME/.local/share}/opencode/opencode.db
```

Use another database with the global `--database` option or
`OPENCODE_DATABASE`:

```bash
op2 --database /path/to/opencode.db sessions list
OPENCODE_DATABASE=/path/to/opencode.db op2 sessions search "timeout"
```

The database is opened read-only with SQLite `mode=ro` and `query_only`.
`sessions archive` and `sessions delete` are the exceptions: they do not write to
SQLite themselves but ask OpenCode to perform the change. When they have to start
their own server, `--database` must name a standard
`<data home>/opencode/opencode.db` path, because OpenCode resolves its database
from its data directory rather than from an option.

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
active root sessions are returned. Use `--include-children`,
`--include-archived`, or `--archived-only` as needed. Common filters are
`--agent`, `--provider`, `--model`, `--since`, and `--until` (which is
exclusive). Sort with `--sort` (`updated`, `created`, `cost`, `tokens`, or
`title`); formats are `table`, `json`, and `jsonl`.

`table` is meant for reading. On a terminal it renders a bordered table; when
stdout is redirected or piped it prints the same columns as tab-separated
values with no header, so results can be fed to other commands.

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
<oc-tool name="bash" status="completed" title="Check repository status" />
<oc-tool name="task" status="completed" title="Explore the codebase" session="ses_04fccdc8cffe">
</oc-tool>
</oc-message>
<oc-todos>
<oc-todo status="completed">Investigate the parser</oc-todo>
</oc-todos>
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

Sub-agent sessions started by the `task` tool nest their own transcript inside
the `<oc-tool>` element that spawned them, recursively, only with
`--include-children`; otherwise that element is empty, as in the example
above. Tool inputs and outputs, when opted in, appear as `<oc-input>` and
`<oc-output>` children of `<oc-tool>`; a truncated `<oc-output>` carries a
`truncated` attribute giving the number of characters omitted.

Session selectors work exactly like `sessions archive` and `sessions delete`:
full IDs or any prefix identifying exactly one session, given as arguments or
read from standard input when no argument is given, including the `table`,
`json`, or `jsonl` output of `sessions list` or `sessions search`. Multiple
sessions are consecutive `<oc-session>` elements with no separator, which is
what makes `op2 sessions search "..." | op2 sessions show` useful for feeding
several related conversations into a single summary.

By default the transcript contains user and assistant text plus one line per
tool call giving the tool, its title, and its status. Tool inputs and outputs
account for roughly 90% of stored conversation bytes, so including them turns
a 4,000–11,000 token transcript into a 40,000–350,000 token one; they are
opt-in via `--include-tool-inputs` and `--include-tool-outputs`, each capped
per call by `--max-tool-chars` (default 2000, 0 for unlimited). Reasoning is
omitted by default, consistent with `sessions search`, and included with
`--include-reasoning`. `--include-todos` appends the session todo list.

### Archive and delete

```bash
# One or more session IDs, or any prefix identifying exactly one session
op2 sessions archive ses_8a3f2c ses_4b91
op2 sessions delete ses_8a3f2c --yes

# Archive everything that has not been touched in 90 days
op2 sessions list --until 90d | op2 sessions archive

# Delete the sessions of a repository that were already archived
op2 sessions list --repo ~/src/spike --archived-only | op2 sessions delete --yes
```

Both commands take session IDs as arguments, and read them from standard input
when no argument is given. Piped input may be the tab-separated `table` output
of `sessions list` or `sessions search`, their `json` or `jsonl` output, or one
session ID per line. Selectors resolve only when they identify a single session;
ambiguous prefixes produce an error and nothing is changed.

Archiving is reversible in OpenCode and keeps all session content; archived
sessions are skipped by `sessions list` and `sessions search` unless
`--include-archived` or `--archived-only` is given. Deletion is permanent and
also removes child sessions, messages, and the session's event history, so it
asks for confirmation unless `--yes` is passed, and refuses to run without
`--yes` when it cannot prompt.

Neither command modifies SQLite directly; each calls OpenCode's own session
endpoints, which keeps cascading rules owned by OpenCode. If a local OpenCode
server is already running on the same database — `opencode serve`, `opencode web`,
or a TUI — it is reused, so the change is fast and its connected clients see the
result immediately. A server is identified by the database file it holds open, not
by its port, so a server backed by a different data directory is never used.
Otherwise a short-lived `opencode serve` is started and stopped around the batch,
which requires the `opencode` CLI on `PATH` and pays roughly a second of startup
once per command. Exit status is `1` if any session failed, and each failure is
reported with its session ID.

### Search

```bash
# Literal search (the default) of titles and user/assistant text
op2 sessions search "migration failure"

# Search the current repository with a shorter snippet
op2 sessions search "rate limit" --cwd --snippet-length 120

# Search only titles and opted-in todo content
op2 sessions search "deploy" --in title --include-todos --format json
```

Searches title and text by default. `--in title` or `--in text` restricts the
ordinary sources; `--include-todos`, `--include-reasoning`,
`--include-tool-inputs`, and `--include-tool-outputs` opt into additional
content. Use repeated `--role user` or `--role assistant` to filter message
content. Search also accepts the list repository, archive, session, and time
filters, plus `--case-sensitive`, `--max-count`, `--snippet-length`, and the
`table`, `json`, or `jsonl` output formats.

Search terms are literal. Repository selectors resolve only when they uniquely
identify a repository; ambiguous selectors produce an error.

### Semantic search

Semantic search finds sessions from a natural-language description. Semantic
search and indexing require `COHERE_API_KEY` to already be set in the process
environment. Configure it through your shell environment or secret manager
before invoking `op2`.

```bash
# Search by meaning instead of literal terms
op2 sessions search "why did deployment fail?" --semantic

# JSON results
op2 sessions search "authentication issue" --semantic --format json

# Explicitly synchronize the index, or rescan/recreate it
op2 sessions index
op2 sessions index --force
op2 sessions index --rebuild --format json
```

Semantic indexing uses Cohere's fixed `embed-v4.0` model with 1024 dimensions.
Table-format indexing shows interactive progress on a TTY; Cohere failures include
safe, actionable status guidance without exposing request content or credentials.

Semantic search supports the usual title/text selection, `--role`, repository,
session, archive, agent/provider/model, and time filters. It returns 10 results
by default; use `-n`/`--max-count` to change that. `--case-sensitive` and the
sensitive `--include-todos`, `--include-reasoning`, `--include-tool-inputs`, and
`--include-tool-outputs` flags cannot be used with `--semantic`.

## Privacy and performance

The CLI searches current materialized session data, not event history. Todos,
reasoning, and tool input/output may be sensitive or large, so they are omitted
unless explicitly requested. Broad searches scan selected session content.
`sessions show` follows the same rule: reasoning, todos, and tool inputs and
outputs are omitted unless explicitly requested.

For semantic indexing, only session titles and product-visible user/assistant
text are sent to Cohere. Reasoning, todos, tool inputs, and tool outputs are
excluded. Credentials are never stored in the semantic sidecar.

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
calls; later searches against an unchanged source embed only the query.

## Development

```bash
uv run pytest
uv run ruff check .
uv run pyright
uv run op2 --help
uv run op2 sessions list --help
uv run op2 sessions show --help
uv run op2 sessions search --help
uv run op2 sessions index --help
uv run op2 sessions archive --help
uv run op2 sessions delete --help
```
