"""Shared cancellation primitive.

Lives in ``utils`` (a low-level package) so processing modules like
``stacking.stacker`` can raise a common cancellation signal without depending
upward on ``cmd_processing``. ``cmd_processing.jobs`` re-exports :class:`Cancelled`
so callers can refer to ``jobs.Cancelled``.
"""


class Cancelled(Exception):
    """Raised inside a worker when its job has been cancelled.

    Long-running analysis functions accept a ``cancel_cb: () -> bool`` and raise
    this at their checkpoints; the command wrapper catches it, posts a
    "cancelled" message, and returns so the job lands in the CANCELLED state.
    """
