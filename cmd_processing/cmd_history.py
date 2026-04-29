"""Command history — persisted to cmd_history.json in the project root."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

_DEFAULT_SHOW = 20
_MAX_ENTRIES = 1000


def _history_path() -> Path:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return Path(project_root) / "cmd_history.json"


def _load() -> list:
    p = _history_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []


def _save(entries: list) -> None:
    _history_path().write_text(json.dumps(entries, indent=2))


def record(cmd: str) -> None:
    """Append a command to the history file (trimmed to _MAX_ENTRIES)."""
    cmd = cmd.strip()
    if not cmd:
        return
    entries = _load()
    n = (entries[-1]["n"] + 1) if entries else 1
    entries.append({
        "n": n,
        "date": datetime.now().isoformat(timespec="seconds"),
        "cmd": cmd,
    })
    if len(entries) > _MAX_ENTRIES:
        entries = entries[-_MAX_ENTRIES:]
    _save(entries)


def get_nth(n: int) -> Optional[str]:
    """Return the command with history number n (1-based). None if not found."""
    entries = _load()
    for e in entries:
        if e["n"] == n:
            return e["cmd"]
    return None


def format_history(last_n: int = _DEFAULT_SHOW) -> str:
    entries = _load()
    if not entries:
        return "No command history yet"
    shown = entries[-last_n:]
    lines = [f"{e['n']:4d}  {e['date']}  {e['cmd']}" for e in shown]
    return "Command history:\n" + "\n".join(lines)
