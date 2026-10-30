"""Short HTTP 429 retries.

A delay written in hours, or any parsed delay over 60 seconds, is not retried.
Up to eight retries sleep the advertised delay. Logs record the exception class
only.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

MAX_QUOTA_RETRIES = 8
MAX_RETRY_SECONDS = 60.0

_HOUR = re.compile(r"(\d+(?:\.\d+)?)\s*hours?")
_SECONDS = re.compile(
    r"retry(?:\s+in|\s+of about|\s+delay)?[^\d]{0,40}(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b",
    re.IGNORECASE,
)
_RETRY_DELAY = re.compile(
    r"retrydelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE
)


def is_quota_error(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    status = getattr(exc, "status_code", None)
    if code == 429 or status == 429:
        return True
    return False


def is_thinking_budget_error(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    status = getattr(exc, "status_code", None)
    if code != 400 and status != 400:
        return False
    return "thinking" in str(exc).lower()


def advertised_delay_seconds(exc: BaseException) -> float | None:
    """Return a sleep duration, or None when the call must not be retried."""

    text = str(exc)
    if _HOUR.search(text):
        return None
    match = _RETRY_DELAY.search(text) or _SECONDS.search(text)
    if match is None:
        return None
    delay = float(match.group(1))
    if delay > MAX_RETRY_SECONDS:
        return None
    return delay


def run_with_quota_retries[T](
    call: Callable[[], T],
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_retries: int = MAX_QUOTA_RETRIES,
) -> T:
    """Call ``call``. Retry a short 429 up to ``max_retries`` times."""

    retries = 0
    while True:
        try:
            return call()
        except Exception as exc:
            delay = advertised_delay_seconds(exc) if is_quota_error(exc) else None
            if delay is None or retries >= max_retries:
                raise
            logger.warning("provider quota retry (%s)", type(exc).__name__)
            sleep(delay)
            retries += 1
