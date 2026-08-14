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
# Search titles and user/assistant text
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

## Privacy and performance

The CLI searches current materialized session data, not event history. Todos,
reasoning, and tool input/output may be sensitive or large, so they are omitted
unless explicitly requested. Broad searches scan selected session content.

## Development

```bash
uv run pytest
uv run ruff check .
uv run pyright
uv run op2 --help
uv run op2 sessions list --help
uv run op2 sessions search --help
```
