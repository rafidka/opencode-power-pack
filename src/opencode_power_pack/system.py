"""Read-only inspection of the local operating system.

Used to confirm that OpenCode's background service is the process writing to a
given database. The implementation reads Linux ``/proc``; where that is not
available, the answer is unknown and callers decide how to proceed.
"""

import os
from pathlib import Path


def process_holds_file(pid: int, path: Path) -> bool | None:
    """Return whether process ``pid`` has ``path`` open, or ``None`` if unknown."""
    directory = f"/proc/{pid}/fd"
    target = os.fspath(path)
    try:
        entries = tuple(os.scandir(directory))
    except FileNotFoundError:
        return None if not Path("/proc/self/fd").exists() else False
    except OSError:
        return None
    for entry in entries:
        try:
            if os.readlink(entry.path) == target:
                return True
        except OSError:
            continue
    return False
