"""The journal — the system of record.

Append-only JSONL, one file per calendar day under local/journal/. Every entry
carries a monotonic `seq` that never resets, so "since seq N" is a complete
subscription protocol and replay order is never ambiguous. Each line is
fsynced: this file is what crash recovery replays, and a lost tail would mean
recovering to a state the hardware has already left.

Entry kinds:
  transition  the Night machine moved            (from/to filled)
  target      the Target machine moved           (data.dso, from/to filled)
  rejected    an event was offered and a guard refused (guard filled) — the
              audit trail of every time a guard saved the roof
  note        an annotation with no state effect — the migration target for
              post_social_message

Replay tolerates a truncated final line (a crash mid-write) by dropping it:
the entry's transition never fully happened from the reader's point of view,
and the reconciliation pass (Phase 3) squares the journal against the sensors
anyway.
"""
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class Entry:
    seq: int
    ts: str
    kind: str                      # transition | target | rejected | note
    event: str
    source: str                    # nina|chat|vision|watchdog|conductor|operator|spark|publish|shadow
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    guard: Optional[str] = None
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        # Wire names match the plan's schema ("from"/"to" are reserved-ish in
        # Python, hence the field names differ from the wire).
        d["from"] = d.pop("from_state")
        d["to"] = d.pop("to_state")
        return json.dumps(d, separators=(",", ":"))

    @staticmethod
    def from_json(line: str) -> "Entry":
        d = json.loads(line)
        d["from_state"] = d.pop("from", None)
        d["to_state"] = d.pop("to", None)
        return Entry(**d)


class Journal:
    """Writer + reader over the day files. Thread-safe appends."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = self._recover_head()

    # ---------------------------------------------------------- writing

    def append(self, kind: str, event: str, source: str, *,
               from_state: Optional[str] = None, to_state: Optional[str] = None,
               guard: Optional[str] = None, data: Optional[dict] = None,
               ts: Optional[str] = None) -> Entry:
        with self._lock:
            self._seq += 1
            entry = Entry(seq=self._seq,
                          ts=ts or datetime.now().astimezone().isoformat(timespec="seconds"),
                          kind=kind, event=event, source=source,
                          from_state=from_state, to_state=to_state,
                          guard=guard, data=data or {})
            path = self._file_for_today()
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(entry.to_json() + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return entry

    # ---------------------------------------------------------- reading

    def entries_since(self, seq: int, limit: int = 1000) -> list:
        """Entries with seq > `seq`, oldest first, across day files."""
        out = []
        for day_file in self._files_newest_last():
            for e in self._read_file(day_file):
                if e.seq > seq:
                    out.append(e)
                    if len(out) >= limit:
                        return out
        return out

    def replay(self, days: int = 2) -> Iterator[Entry]:
        """All entries from the last `days` day files, oldest first.

        Two days by default because a night spans midnight: recovering at
        03:00 needs yesterday's file to see the night begin.
        """
        files = self._files_newest_last()[-days:] if days else self._files_newest_last()
        for f in files:
            yield from self._read_file(f)

    def head(self) -> int:
        return self._seq

    # ---------------------------------------------------------- internals

    def _file_for_today(self) -> Path:
        return self.root / (date.today().isoformat() + ".jsonl")

    def _files_newest_last(self) -> list:
        return sorted(self.root.glob("*.jsonl"))

    def _read_file(self, path: Path) -> Iterator[Entry]:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return
        for i, line in enumerate(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                yield Entry.from_json(line)
            except (json.JSONDecodeError, TypeError):
                # A malformed line mid-file is corruption worth knowing about;
                # a malformed LAST line is the expected crash-mid-write case
                # and is silently dropped — its transition never happened.
                if i < len(raw.splitlines()) - 1:
                    raise

    def _recover_head(self) -> int:
        """Highest seq on disk, tolerant of a truncated tail."""
        best = 0
        for f in self._files_newest_last():
            for e in self._read_file(f):
                if e.seq > best:
                    best = e.seq
        return best
