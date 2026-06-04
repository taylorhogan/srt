"""Thread-safe in-memory job registry for the SRT job-based web chat UI.

A *job* is the unit of work created when a user issues a command. Each job has a
stable ``id``, a lifecycle (``QUEUED -> RUNNING -> DONE/ERROR/CANCELLED``), an
append-only ``log`` of entries, and an optional ``progress`` value. Job events are
broadcast to connected WebSocket clients through the shared ``message_bus``
subscriber queues, so the UI can render one self-updating card per job and keep
each job's log isolated.

See ``docs/UI_DESIGN.md`` for the design intent.

Threading model:
- ``web_server._dispatch_command`` creates a job, binds it to the dispatch thread
  via :func:`set_current_job`, and runs the command. Every ``post_message`` made
  while that binding is active is appended to the job's log.
- The long-running commands hand off to a worker thread via :func:`spawn`, which
  re-binds the job id inside the worker and owns the terminal transition. The
  dispatch thread detects this handoff (``async_handoff``) and does not finalize.
"""

import logging
import threading
import time
from typing import Any, Callable, Optional

from cmd_processing import message_bus

_logger = logging.getLogger(__name__)

# ── Lifecycle states ──────────────────────────────────────────────
QUEUED = "QUEUED"
RUNNING = "RUNNING"
DONE = "DONE"
ERROR = "ERROR"
CANCELLED = "CANCELLED"
_TERMINAL = frozenset({DONE, ERROR, CANCELLED})

# The pinned "Observatory" feed that collects untagged messages (scheduler
# pushes, unsolicited notifications) now that the linear transcript is gone.
SYSTEM_JOB_ID = "system"
SYSTEM_STATUS = "SYSTEM"

_jobs: dict = {}
_order: list = []            # creation order of job ids (newest appended last)
_lock = threading.RLock()
_counter = 0
_local = threading.local()   # holds .job_id for the current thread
_cancelled: set = set()      # job ids for which a cancel was requested

# First command word -> (kind, resource_class). ``kind`` drives the card's
# colour-coded left edge; ``resource_class`` lets the UI reason about queuing.
_CLASSIFY = {
    "image":       ("image",  "mount_exclusive"),
    "schedule":    ("image",  "mount_exclusive"),
    "slew":        ("slew",   "mount_exclusive"),
    "goto":        ("slew",   "mount_exclusive"),
    "tonight":     ("quick",  "quick"),
    "best":        ("quick",  "quick"),
    "status":      ("quick",  "quick"),
    "calendar":    ("quick",  "quick"),
    "speedtest":   ("quick",  "quick"),
    "history":     ("quick",  "quick"),
    "help":        ("quick",  "quick"),
    "?":           ("quick",  "quick"),
    "show":        ("quick",  "archive"),
    "latest":      ("quick",  "archive"),
    # super-user long-running jobs
    "optics":      ("search", "archive"),
    "drift":       ("search", "archive"),
    "stack":       ("search", "archive"),
    "snr":         ("search", "archive"),
    "transit":     ("search", "archive"),
    "bad":         ("search", "archive"),
    "stats":       ("search", "archive"),
    "update":      ("quick",  "quick"),
    "focus":       ("focus",  "mount_exclusive"),
    "autofocus":   ("focus",  "mount_exclusive"),
}


def init() -> None:
    """Reset the registry and (re)create the pinned system job."""
    global _jobs, _order, _counter, _cancelled, _resources
    with _lock:
        _jobs = {}
        _order = []
        _counter = 0
        _cancelled = set()
        _resources = {}
        _ensure_system_locked()


def _ensure_system_locked() -> dict:
    job = _jobs.get(SYSTEM_JOB_ID)
    if job is None:
        job = {
            "id": SYSTEM_JOB_ID,
            "title": "Observatory",
            "command": None,
            "kind": "system",
            "resource_class": "quick",
            "status": SYSTEM_STATUS,
            "created_at": time.time(),
            "started_at": time.time(),
            "finished_at": None,
            "progress": None,
            "async_handoff": False,
            "log": [],
        }
        _jobs[SYSTEM_JOB_ID] = job
        _order.append(SYSTEM_JOB_ID)
    return job


# ── Classification helpers ────────────────────────────────────────

def classify(command_text: str) -> tuple[str, str]:
    word = (command_text or "").strip().split(" ")[0].lower().lstrip("@")
    return _CLASSIFY.get(word, ("quick", "quick"))


def make_title(command_text: str) -> str:
    """Derive a human-readable provisional title from the raw command."""
    text = (command_text or "").strip()
    if not text:
        return "Command"
    words = text.split()
    verb = words[0].lower()
    target = " ".join(words[1:]).strip()
    pretty = {
        "image": "Imaging",
        "schedule": "Scheduling",
        "slew": "Slewing to",
        "goto": "Slewing to",
        "tonight": "Best object tonight",
        "best": "Best date for",
        "show": "Survey image",
        "latest": "Latest frame",
        "status": "Observatory status",
        "calendar": "Calendar",
        "transit": "Transit search",
        "snr": "SNR analysis",
        "stack": "Stacking",
        "optics": "Optical metrics",
        "drift": "Drift analysis",
        "bad": "Bad-frame scan",
        "stats": "Image statistics",
        "help": "Help",
    }.get(verb)
    if pretty is None:
        return text if len(text) <= 48 else text[:45] + "…"
    return f"{pretty} {target}".strip() if target else pretty


# ── Current-thread job binding ────────────────────────────────────

def set_current_job(job_id: Optional[str]) -> None:
    _local.job_id = job_id


def get_current_job() -> Optional[str]:
    return getattr(_local, "job_id", None)


def clear_current_job() -> None:
    _local.job_id = None


# ── Event broadcast ───────────────────────────────────────────────

def _broadcast(event: dict) -> None:
    try:
        message_bus.broadcast({"type": "job_event", "data": event})
    except Exception:
        _logger.exception("Failed to broadcast job event")


# ── Lifecycle ─────────────────────────────────────────────────────

def create(command_text: str, status: str = QUEUED) -> str:
    """Create a job from a command string and broadcast it. Returns the job id."""
    global _counter
    kind, resource_class = classify(command_text)
    with _lock:
        _counter += 1
        job_id = f"job_{_counter}"
        now = time.time()
        job = {
            "id": job_id,
            "title": make_title(command_text),
            "command": command_text,
            "kind": kind,
            "resource_class": resource_class,
            "status": status,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "progress": None,
            "async_handoff": False,
            "log": [],
        }
        _jobs[job_id] = job
        _order.append(job_id)
        snapshot = dict(job)
    _broadcast({"kind": "created", "job": snapshot})
    return job_id


def transition(job_id: str, status: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None or job["status"] == status:
            return
        # Never move a terminal job again.
        if job["status"] in _TERMINAL:
            return
        job["status"] = status
        now = time.time()
        if status == RUNNING and job["started_at"] is None:
            job["started_at"] = now
        if status in _TERMINAL:
            job["finished_at"] = now
            if status == DONE and job["progress"] is not None:
                job["progress"] = 100
        patch = {
            "kind": "update",
            "id": job_id,
            "status": job["status"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "progress": job["progress"],
        }
    _broadcast(patch)


def set_progress(job_id: str, value: Optional[float]) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["progress"] = value
    _broadcast({"kind": "update", "id": job_id, "progress": value,
                "status": job["status"], "started_at": job["started_at"],
                "finished_at": job["finished_at"]})


def append_log(job_id: Optional[str], entry: dict) -> None:
    """Append a log entry (a message_bus message dict) to a job and broadcast it.

    Falls back to the pinned system job when ``job_id`` is unknown/None.
    """
    with _lock:
        job = _jobs.get(job_id) if job_id else None
        if job is None:
            job = _ensure_system_locked()
        job["log"].append(entry)
        status = job["status"]
        jid = job["id"]
    _broadcast({"kind": "log", "id": jid, "entry": entry, "status": status})


def mark_async(job_id: str) -> None:
    """Flag that a worker thread has taken ownership of a job's terminal state."""
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["async_handoff"] = True


def is_async(job_id: str) -> bool:
    with _lock:
        job = _jobs.get(job_id)
        return bool(job and job.get("async_handoff"))


# ── Cancellation (cooperative) ────────────────────────────────────

# Canonical cancellation signal; defined in utils so low-level processing
# modules can raise it without importing cmd_processing. Re-exported here so
# callers can use jobs.Cancelled.
from utils.cancellation import Cancelled  # noqa: E402,F401


# job id -> list of cancellable resources (e.g. ProcessPoolExecutor). On cancel
# we best-effort tear these down so pools die promptly instead of draining.
_resources: dict = {}


def request_cancel(job_id: str) -> bool:
    with _lock:
        job = _jobs.get(job_id)
        if job is None or job["status"] in _TERMINAL or job_id == SYSTEM_JOB_ID:
            return False
        _cancelled.add(job_id)
        resources = list(_resources.get(job_id, ()))
    # Tear down outside the lock — shutdown() can block briefly.
    for resource in resources:
        _terminate_resource(resource)
    return True


def is_cancelled(job_id: Optional[str]) -> bool:
    if not job_id:
        return False
    with _lock:
        return job_id in _cancelled


def raise_if_cancelled(job_id: Optional[str]) -> None:
    """Raise :class:`Cancelled` if a cancel has been requested for this job."""
    if is_cancelled(job_id):
        raise Cancelled()


def cancel_cb_for(job_id: Optional[str]) -> Callable[[], bool]:
    """Return a zero-arg ``() -> bool`` predicate reporting this job's cancel
    state — the shape long analysis functions accept as ``cancel_cb``."""
    return lambda: is_cancelled(job_id)


def register_resource(job_id: Optional[str], resource: Any) -> None:
    """Register a cancellable resource (e.g. a process/thread pool) for a job so
    :func:`request_cancel` can tear it down promptly. No-op without a job id."""
    if not job_id:
        return
    with _lock:
        _resources.setdefault(job_id, []).append(resource)


def unregister_resource(job_id: Optional[str], resource: Any) -> None:
    """Drop a previously registered resource (call in the worker's ``finally``)."""
    if not job_id:
        return
    with _lock:
        bucket = _resources.get(job_id)
        if bucket and resource in bucket:
            bucket.remove(resource)
            if not bucket:
                del _resources[job_id]


def _terminate_resource(resource: Any) -> None:
    """Best-effort teardown of a cancellable resource. Never raises."""
    try:
        if hasattr(resource, "shutdown"):           # concurrent.futures Executor
            resource.shutdown(wait=False, cancel_futures=True)
        elif hasattr(resource, "terminate"):        # multiprocessing.Pool / Process
            resource.terminate()
        elif callable(resource):                    # plain cleanup callback
            resource()
    except Exception:
        _logger.exception("Failed to terminate cancellable resource")


# ── Removal ───────────────────────────────────────────────────────

def remove(job_id: str) -> bool:
    """Remove a finished job card and broadcast a 'removed' event.

    Only terminal jobs may be removed; the pinned system feed and any
    live (queued/running) job are refused so a removed job's later
    ``append_log`` cannot fall back into the system feed.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None or job_id == SYSTEM_JOB_ID or job["status"] not in _TERMINAL:
            return False
        del _jobs[job_id]
        if job_id in _order:
            _order.remove(job_id)
        _cancelled.discard(job_id)
        _resources.pop(job_id, None)
    _broadcast({"kind": "removed", "id": job_id})
    return True


# ── Worker spawn (replaces threading.Thread for long commands) ────

def spawn(target: Callable[..., Any], args: tuple = (), kwargs: Optional[dict] = None) -> threading.Thread:
    """Run ``target`` on a daemon thread that inherits the caller's current job.

    The job's terminal transition is owned by this worker: ``DONE`` on normal
    return, ``ERROR`` on exception (a ``CANCELLED`` set during the run wins). The
    dispatch thread detects the handoff via :func:`is_async` and does not finalize.
    """
    kwargs = kwargs or {}
    job_id = get_current_job()
    if job_id:
        mark_async(job_id)

    def _runner():
        set_current_job(job_id)
        try:
            target(*args, **kwargs)
            if job_id:
                finalize(job_id, DONE)
        except Cancelled:
            # Cooperative cancellation — terminal CANCELLED, not an error.
            if job_id:
                append_log(job_id, message_bus.make_entry("■ Cancelled."))
                transition(job_id, CANCELLED)
        except Exception:
            _logger.exception("Job worker failed")
            if job_id:
                append_log(job_id, message_bus.make_entry("✕ Job failed — see log."))
                finalize(job_id, ERROR)
        finally:
            clear_current_job()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t


def finalize(job_id: str, status: str) -> None:
    """Terminal transition that honours a pending cancel request."""
    if is_cancelled(job_id):
        status = CANCELLED
    transition(job_id, status)


# ── Snapshot for (re)connect rehydration ──────────────────────────

def snapshot() -> list:
    with _lock:
        return [dict(_jobs[jid]) for jid in _order if jid in _jobs]
