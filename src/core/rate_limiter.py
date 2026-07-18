import asyncio
import time


class RateLimiter:
    """Tracks requests and tokens in a rolling 60s window, blocks until safe to proceed."""

    def __init__(self, max_requests_per_min: int, max_tokens_per_min: int):
        self.max_rpm = max_requests_per_min
        self.max_tpm = max_tokens_per_min
        self._lock = asyncio.Lock()
        self._request_log: list[float] = []
        self._token_log: list[tuple[float, int]] = []

    async def acquire(self, tokens: int):
        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - 60

                self._request_log = [t for t in self._request_log if t > cutoff]
                self._token_log = [(t, n) for t, n in self._token_log if t > cutoff]

                used_tokens = sum(n for _, n in self._token_log)
                used_requests = len(self._request_log)

                if used_requests < self.max_rpm and used_tokens + tokens <= self.max_tpm:
                    self._request_log.append(now)
                    self._token_log.append((now, tokens))
                    return

                oldest_times = []
                if self._request_log:
                    oldest_times.append(self._request_log[0])
                if self._token_log:
                    oldest_times.append(self._token_log[0][0])
                oldest = min(oldest_times) if oldest_times else now
                wait_time = max(0.05, oldest + 60 - now)

            await asyncio.sleep(min(wait_time, 1.0))