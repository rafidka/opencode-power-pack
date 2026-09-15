"""Read-only inspection of the local operating system.

Used to find an OpenCode server that is already serving a given database. The
implementation reads Linux ``/proc``; on platforms that do not expose it, the
queries return nothing and callers fall back to starting their own server.
"""

import os
import socket
from pathlib import Path

_LOOPBACK = "127.0.0.1"
_LISTENING_STATE = "0A"
_SOCKET_LINK_PREFIX = "socket:["
_TCP_TABLES = ("/proc/net/tcp", "/proc/net/tcp6")
_LOCAL_ADDRESS_FIELD = 1
_STATE_FIELD = 3
_UID_FIELD = 7
_INODE_FIELD = 9
_TCP_FIELD_COUNT = 10


def free_port() -> int:
    """Return a currently unused local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind((_LOOPBACK, 0))
        return int(candidate.getsockname()[1])


def ports_serving_file(path: Path) -> tuple[int, ...]:
    """Return the listening ports of this user's processes that hold ``path`` open.

    Identifying a server by the file it has open, rather than by the port it
    answers on, is what makes it safe to reuse: a server backed by a different
    database is never a candidate.
    """
    ports_by_inode = _listening_ports_by_inode()
    if not ports_by_inode:
        return ()
    target = os.fspath(path)
    found: set[int] = set()
    for pid in _own_process_ids():
        holds_file = False
        ports: list[int] = []
        for link in _open_file_links(pid):
            if link == target:
                holds_file = True
            elif link.startswith(_SOCKET_LINK_PREFIX):
                port = ports_by_inode.get(_socket_inode(link))
                if port is not None:
                    ports.append(port)
        if holds_file:
            found.update(ports)
    return tuple(sorted(found))


def _listening_ports_by_inode() -> dict[int, int]:
    uid = os.getuid()
    ports: dict[int, int] = {}
    for table in _TCP_TABLES:
        for row in _rows(table):
            fields = row.split()
            if (
                len(fields) < _TCP_FIELD_COUNT
                or fields[_STATE_FIELD] != _LISTENING_STATE
            ):
                continue
            try:
                if int(fields[_UID_FIELD]) != uid:
                    continue
                port = int(fields[_LOCAL_ADDRESS_FIELD].rsplit(":", 1)[1], 16)
                ports[int(fields[_INODE_FIELD])] = port
            except ValueError:
                continue
    return ports


def _rows(table: str) -> tuple[str, ...]:
    try:
        return tuple(Path(table).read_text().splitlines()[1:])
    except OSError:
        return ()


def _own_process_ids() -> tuple[int, ...]:
    uid = os.getuid()
    identifiers: list[int] = []
    try:
        entries = tuple(os.scandir("/proc"))
    except OSError:
        return ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid == uid:
                identifiers.append(int(entry.name))
        except OSError:
            continue
    return tuple(identifiers)


def _open_file_links(pid: int) -> tuple[str, ...]:
    links: list[str] = []
    try:
        with os.scandir(f"/proc/{pid}/fd") as entries:
            for entry in entries:
                try:
                    links.append(os.readlink(entry.path))
                except OSError:
                    continue
    except OSError:
        return ()
    return tuple(links)


def _socket_inode(link: str) -> int:
    try:
        return int(link[len(_SOCKET_LINK_PREFIX) : -1])
    except ValueError:
        return -1
