"""
Bounded retry with exponential backoff for transient external calls.

Dependency-free (stdlib only) so it can be imported from the foundational
modules (db, prices, edgar_ingest) without creating import cycles. Used to make
the pipeline resilient to transient blips in the LLM / Polygon / SEC EDGAR /
Postgres calls — a single dropped connection or a 503 should retry, not fail the
whole job. Callers that want graceful degradation still wrap the final raise in
their own try/except (e.g. price fetch returns [] after retries are exhausted).
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Tuple, Type, TypeVar

T = TypeVar("T")

# Default transient-failure policy. Deterministic backoff (no jitter) so behaviour
# is reproducible in tests; the serial worker means there's no thundering-herd to
# de-correlate anyway.
DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 0.5   # seconds; nth retry waits base * 2**(n-1)
DEFAULT_MAX_DELAY = 8.0    # cap per-wait


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    label: str = "",
) -> T:
    """Call ``fn()``, retrying on ``retry_on`` exceptions with exponential backoff.

    Returns ``fn()``'s result on the first success. After ``attempts`` failures
    the last exception is re-raised (so callers keep their existing error
    handling / fallback). Non-``retry_on`` exceptions propagate immediately.
    """
    last: BaseException | None = None
    name = label or getattr(fn, "__name__", "call")
    for n in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as e:
            last = e
            if n == attempts:
                break
            delay = min(base_delay * (2 ** (n - 1)), max_delay)
            print(f"⚠️  retry {n}/{attempts - 1} for {name} after "
                  f"{type(e).__name__}: {e} — retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    assert last is not None  # loop ran ≥ once, so a failure was recorded
    raise last


def with_retry(
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
):
    """Decorator form of :func:`retry_call`."""
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            return retry_call(
                lambda: fn(*args, **kwargs),
                attempts=attempts, base_delay=base_delay, max_delay=max_delay,
                retry_on=retry_on, label=fn.__name__,
            )
        return wrapper
    return deco
