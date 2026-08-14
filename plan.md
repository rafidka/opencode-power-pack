# Add `opencode-sessions`

Create a fast global OpenCode session-listing command by moving the existing
implementation from `~/dotfiles` into this repository.

## Requirements

- Add an executable at `bin/opencode-sessions`.
- Read OpenCode's SQLite database directly at
  `${XDG_DATA_HOME:-$HOME/.local/share}/opencode/opencode.db`.
- Open the database read-only with `sqlite3 -readonly`.
- List sessions from every project ordered by `time_updated` descending.
- Support these options:
  - `-n`, `--max-count N`
  - `--format table|json`
  - `-h`, `--help`
- Default to table output and include all sessions.
- JSON output should use the same field names as `opencode session list --format json`:
  `id`, `title`, `updated`, `created`, `projectId`, and `directory`.
- Validate dependencies, database presence, option values, and unknown options
  with useful stderr errors.
- Do not modify the OpenCode database.

## Implementation

1. Inspect `~/dotfiles/bin/opencode-sessions` and preserve its current behavior
   unless a correction is needed for this repository.
2. Create `bin/opencode-sessions` with Bash strict mode:
   `set -euo pipefail`.
3. Make the file executable.
4. Add a root `README.md` that describes the repository, installation via
   `PATH`, the `sqlite3` dependency, and usage examples.
5. Add `opencode-sessions` to the README's tools list.
6. Remove `bin/opencode-sessions` from `~/dotfiles` after confirming the new
   copy works. Remove its corresponding entries from `~/dotfiles/bin/help` and
   `~/dotfiles/AGENTS.md`.

## Validation

Run from this repository:

```bash
bash -n bin/opencode-sessions
bin/opencode-sessions --help
bin/opencode-sessions --max-count 2
bin/opencode-sessions --format json --max-count 1
git diff --check
```

Confirm the JSON result has the expected fields and that table output includes
sessions from directories other than the current one. Do not commit unless
explicitly requested.
