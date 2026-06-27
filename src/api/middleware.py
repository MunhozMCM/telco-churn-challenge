"""Latency middleware — times each request and exposes it as a header + log."""

from __future__ import annotations

import time

from fastapi import Request
from loguru import logger


async def add_process_time_header(request: Request, call_next):
    """Measure wall-clock latency, attach ``X-Process-Time``, and log it.

    Uses ``perf_counter`` (monotonic, high-resolution). ``/health`` is excluded
    from the access log to avoid drowning the logs in orchestrator pings.
    """
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.6f}"

    if request.url.path != "/health":
        logger.bind(
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=round(elapsed * 1000, 2),
        ).info("request handled")
    return response
