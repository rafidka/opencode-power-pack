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

For semantic indexing, only session titles and product-visible user/assistant
text are sent to Cohere. Reasoning, todos, tool inputs, and tool outputs are
excluded. Credentials are never stored in the semantic sidecar.

OpenCode's database remains read-only. The derived semantic sidecar is stored
under `${XDG_CACHE_HOME:-~/.cache}/opencode-power-pack/`, separately for each
source database. `op2 sessions index` accepts `--semantic-index` to use a
different sidecar path. Semantic searches lazily synchronize before querying;
there is no daemon or background watcher. A stored fingerprint
of the main SQLite database plus its WAL skips scans when unchanged. SQLite's
SHM reader-lock metadata is intentionally not fingerprinted.
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
uv run op2 sessions search --help
uv run op2 sessions index --help
```
