import time
from collections import deque
import threading
from typing import Optional


class RateLimiter:
    """
    A thread-safe rate limiter that allows at most `max_calls` calls in any sliding window
    of `period` seconds (default: 60 seconds for per-minute limiting).

    When the limit is exceeded, additional calls are ignored (the check returns False).
    """

    def __init__(self, max_calls: int, period: float = 60.0):
        if max_calls <= 0:
            raise ValueError("max_calls must be > 0")
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()

    def __call__(self) -> bool:
        """
        Attempt to acquire permission for a call.

        Returns:
            True  - if the call is allowed (and it is recorded)
            False - if the rate limit is exceeded (call should be ignored)
        """
        with self.lock:
            now = time.time()

            # Expire old timestamps outside the sliding window
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()

            # Check if we can allow this call
            if len(self.calls) >= self.max_calls:
                return False

            # Allow and record the call
            self.calls.append(now)
            return True


# Example usage:

# Create a shared limiter: max 10 calls per minute
api_limiter = RateLimiter(max_calls=10, period=60.0)


def make_api_call(data: str) -> Optional[str]:
    """
    Example function that respects the rate limit.
    If rate-limited, we ignore/skip the call and return None.
    """
    if not api_limiter():
        print("Rate limit exceeded - ignoring this call")
        return None

    # Proceed with the actual work (e.g., real API request)
    print(f"Processing: {data}")
    return f"Result for {data}"


# Test it quickly
if __name__ == "__main__":
    for i in range(15):
        result = make_api_call(f"request-{i}")
        time.sleep(0.1)  # Simulate some time between calls

    print("\nWait a bit for the window to slide...")
    time.sleep(65)  # Wait more than 60 seconds

    for i in range(5):
        result = make_api_call(f"new-request-{i}")