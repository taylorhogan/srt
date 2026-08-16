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

import atexit
import logging
import multiprocessing
import os
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

# Per-job log cap. The system feed is never deleted, so without a bound its log
# grows forever and leaks memory (observed: the web server reached ~5 GB over
# ~30 h and wedged its event loop). Keep the most recent entries; trim the
# oldest in bulk past a slack margin to amortize the cost of trimming.
_MAX_JOB_LOG = 2000
_JOB_LOG_TRIM_SLACK = 256

_jobs: dict = {}
_order: list = []            # creation order of job ids (newest appended last)
_lock = threading.RLock()
_counter = 0
_local = threading.local()   # holds .job_id for the current thread

# Job-id tracing. Posts route to a card by the job bound to the calling thread,
# and that binding has to survive dispatch -> spawn -> child process. When it
# does not, everything silently lands in the system feed instead, which is a
# symptom with no stack trace. Set SRT_JOBTRACE=1 to log the id at each hop.
_TRACE = os.environ.get("SRT_JOBTRACE", "") not in ("", "0", "false", "False")
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
    "process":     ("search", "archive"),
    "publish":     ("search", "archive"),
    "purge":       ("quick",  "archive"),
    "skysolve":    ("search", "archive"),
    "snr":         ("search", "archive"),
    "transit":     ("search", "archive"),
    "hr":          ("search", "archive"),
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
        "hr": "H–R diagram",
        "stack": "Stacking",
        "process": "Colour process",
        "publish": "Publish image",
        "purge": "Purge flats",
        "skysolve": "Sky camera plate solve",
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
    if _TRACE:
        _logger.info("JOBTRACE bind job=%s pid=%d thread=%s",
                     job_id, os.getpid(), threading.current_thread().name)


def get_current_job() -> Optional[str]:
    return getattr(_local, "job_id", None)


def clear_current_job() -> None:
    _local.job_id = None


# ── Cross-process job handoff ─────────────────────────────────────
# The end-of-night scripts NINA launches (end.bat/smessage.bat → end.py/
# smessage.py) run in their own processes, so posts they make have no job
# binding and fall through to the system feed — where the roof-close
# spectrogram etc. is effectively invisible. doit_cmd persists its job id to a
# file at run start; those scripts adopt it so their posts land on the imaging
# job's card, same as the roof-open posts made in-process.
_IMAGING_JOB_FILE = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "imaging_job.txt"
)
_IMAGING_JOB_MAX_AGE_SEC = 24 * 3600


def persist_imaging_job() -> None:
    """Write the calling thread's job id to ``imaging_job.txt`` (best-effort)."""
    job_id = get_current_job()
    if not job_id:
        return
    try:
        with open(_IMAGING_JOB_FILE, "w") as f:
            f.write(job_id)
    except OSError:
        _logger.exception("Failed to persist imaging job id")


def adopt_imaging_job() -> None:
    """Bind this thread to the persisted imaging job id, if any.

    No-op when the thread already has a job (an in-process caller like the
    emergency stop must keep its own card) or when the file is older than a
    day (a manual end-sequence run should not resurrect last week's card).
    The id is only forwarded with posts; the server falls back to the system
    feed if the job no longer exists, so a stale id degrades gracefully.
    """
    if get_current_job():
        return
    try:
        if time.time() - os.path.getmtime(_IMAGING_JOB_FILE) > _IMAGING_JOB_MAX_AGE_SEC:
            return
        with open(_IMAGING_JOB_FILE) as f:
            job_id = f.read().strip()
    except OSError:
        return
    if job_id:
        set_current_job(job_id)


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


def get_job(job_id: str) -> Optional[dict]:
    """Return a copy of a job dict, or None if the id is unknown."""
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


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
            # An id we do not know is a message from a worker whose card is
            # gone — nearly always a child that outlived a server restart. It
            # still has to go somewhere, but it must not read as observatory
            # output, because that looks like the system reporting work nobody
            # started. See _watch_parent for why those children now die.
            if job_id and job_id != SYSTEM_JOB_ID:
                entry = dict(entry)
                entry["text"] = f"[orphan {job_id}] {entry.get('text', '')}"
            job = _ensure_system_locked()
        log = job["log"]
        log.append(entry)
        # Bound the log so the immortal system feed can't grow without limit.
        if len(log) > _MAX_JOB_LOG + _JOB_LOG_TRIM_SLACK:
            del log[: len(log) - _MAX_JOB_LOG]
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
        already_requested = job_id in _cancelled
        _cancelled.add(job_id)
        resources = list(_resources.get(job_id, ()))
    # Tear down outside the lock — shutdown() can block briefly.
    for resource in resources:
        _terminate_resource(resource)
    # Acknowledge on the card immediately; the worker may not reach its next
    # cancel checkpoint for a while (or ever, if it predates checkpoints).
    if not already_requested:
        append_log(job_id, message_bus.make_entry(
            "■ Cancel requested — stopping at the next safe point."))
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


# ── Process-isolated worker spawn (true CPU parallelism, no GIL) ───

# Live child processes, tracked so we can tear them down on shutdown.
_child_procs: set = set()


def _watch_parent(conn: Any, job_id: Optional[str]) -> None:
    """Exit this child as soon as the parent server goes away.

    A process-isolated job is only ever meaningful to the server that started
    it: its progress goes to a card in the server's job registry and its output
    is a file the server names back to the user. Once the parent is gone the
    work is unreachable, and an hour-long stack left running is not harmless —
    it saturates the disk the new server needs and overwrites results with
    output nobody can see. Blocking on a pipe that only the parent holds open
    turns "parent died" into a plain EOF, however it died.
    """
    try:
        conn.recv()                       # never written to; blocks until EOF
    except (EOFError, OSError):
        pass
    except Exception:
        _logger.exception("parent watchdog failed; leaving the job running")
        return
    _logger.warning("Parent server exited — abandoning job %s (pid %d)",
                    job_id, os.getpid())
    # _exit, not sys.exit: the work is deep inside numpy/astroalign and the
    # point is to stop now, not to unwind.
    os._exit(3)


def _child_main(job_id: Optional[str], target: Callable[..., Any],
                args: tuple, kwargs: Optional[dict],
                death_conn: Any = None) -> None:
    """Entry point that runs inside a freshly spawned child process.

    Binds the job id so the worker's posts route to the right card — this
    process never initialises the in-process message bus, so
    ``social_server.post_social_message`` uses its HTTP ``/api/post`` fallback,
    which we tag with the job id. Outcome is signalled via the exit code
    (0 = ok, 1 = error, 2 = cancelled); the parent monitor owns the job's
    terminal transition.
    """
    import sys as _sys
    kwargs = kwargs or {}
    # Importing config configures this process's file logging.
    try:
        from configs import config
        config.data()
    except Exception:
        pass
    if _TRACE or job_id is None:
        _logger.info("JOBTRACE child pid=%d received job=%s", os.getpid(), job_id)
    set_current_job(job_id)
    if death_conn is not None:
        threading.Thread(target=_watch_parent, args=(death_conn, job_id),
                         daemon=True, name="parent-watchdog").start()
    # Process-isolated jobs (snr/stack: astroalign over dozens of 62MP frames
    # across thread pools) otherwise run at NORMAL priority and can saturate the
    # machine, starving the real-time USB camera capture — which corrupts not
    # only `live` frames but the roof/park vision-safety snapshots that share
    # that camera. Drop this child below normal so it always yields to real-time
    # work; the diagnostic just takes a little longer when the box is busy.
    try:
        if hasattr(os, "nice"):                       # POSIX (the Spark)
            os.nice(10)
        else:                                         # Windows
            import ctypes
            _BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            _k32 = ctypes.windll.kernel32
            _k32.GetCurrentProcess.restype = ctypes.c_void_p
            _k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            if not _k32.SetPriorityClass(_k32.GetCurrentProcess(), _BELOW_NORMAL_PRIORITY_CLASS):
                _logger.debug("SetPriorityClass failed (err %s)", ctypes.get_last_error())
    except Exception:
        _logger.debug("could not lower job process priority", exc_info=True)
    try:
        target(*args, **kwargs)
        _sys.exit(0)
    except Cancelled:
        _sys.exit(2)
    except SystemExit:
        raise
    except Exception:
        _logger.exception("Process-isolated job failed")
        try:
            from cmd_processing import social_server
            social_server.post_social_message("✕ Job failed — see log.")
        except Exception:
            pass
        _sys.exit(1)


def spawn_process(target: Callable[..., Any], args: tuple = (),
                  kwargs: Optional[dict] = None) -> "multiprocessing.Process":
    """Run ``target`` in its own OS process — true parallelism, no shared GIL.

    Mirrors :func:`spawn` for heavy, GIL-bound commands. The child reports
    progress/results back to its job card over the existing job-id-tagged
    ``/api/post`` HTTP path; a monitor thread maps the child's exit code to the
    terminal state (DONE/ERROR, or CANCELLED if a cancel was requested).
    Cancellation tears the process down via :func:`register_resource`
    (``Process.terminate``).
    """
    kwargs = kwargs or {}
    job_id = get_current_job()
    if _TRACE or job_id is None:
        _logger.info("JOBTRACE spawn_process captured job=%s on thread=%s "
                     "(None here means the child cannot route its posts)",
                     job_id, threading.current_thread().name)
    if job_id:
        mark_async(job_id)

    ctx = multiprocessing.get_context("spawn")
    # Parent-death detector. daemon=False (below) means the OS will happily let
    # these children outlive us, and the `update` command exits via os._exit, so
    # the atexit teardown never runs — the result was 75-minute stacks still
    # grinding the disk and posting to a job registry that no longer had their
    # card. The child holds the read end of this pipe and nothing ever writes to
    # it; whenever this process goes away — clean exit, os._exit, crash or
    # kill — the write end closes and the child sees EOF. Unlike a pid check it
    # cannot be fooled by pid reuse, and it needs no psutil.
    death_r, death_w = ctx.Pipe(duplex=False)
    # daemon=False: workers may create their own pools, which daemonic
    # processes are forbidden from doing.
    proc = ctx.Process(target=_child_main,
                       args=(job_id, target, args, kwargs, death_r),
                       daemon=False)
    proc.start()
    death_r.close()          # only the child may hold the read end
    proc._srt_death_w = death_w   # keep the write end alive as long as proc is
    if job_id:
        register_resource(job_id, proc)
    with _lock:
        _child_procs.add(proc)

    def _monitor():
        proc.join()
        with _lock:
            _child_procs.discard(proc)
        if not job_id:
            return
        unregister_resource(job_id, proc)
        if is_cancelled(job_id):
            finalize(job_id, CANCELLED)
        elif proc.exitcode == 0:
            finalize(job_id, DONE)
        else:
            finalize(job_id, ERROR)

    threading.Thread(target=_monitor, daemon=True).start()
    return proc


def terminate_child_procs() -> None:
    """Kill any live process-isolated job children.

    Called explicitly from the `update` restart path, which exits via os._exit
    and so never runs the atexit hook below. The children's own parent watchdog
    is the backstop for every other way this process can die.
    """
    _terminate_child_procs()


@atexit.register
def _terminate_child_procs() -> None:
    """Best-effort teardown of any live child processes on interpreter exit."""
    with _lock:
        procs = list(_child_procs)
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass


def finalize(job_id: str, status: str) -> None:
    """Terminal transition that honours a pending cancel request."""
    if is_cancelled(job_id):
        status = CANCELLED
    transition(job_id, status)


# ── Snapshot for (re)connect rehydration ──────────────────────────

def snapshot() -> list:
    with _lock:
        return [dict(_jobs[jid]) for jid in _order if jid in _jobs]
