"""aiohttp /health server on HEALTH_PORT.

A tiny liveness endpoint shared by the live bot (``mode="live"``) and the
headless harness (``mode="harness"``). It answers
``GET /health -> {"status": "ok", "mode": ..., "guild": ...}`` and performs
NO Discord I/O of any kind — it never imports discord and never dials out.
"""

from __future__ import annotations

import asyncio
import json
import signal

from aiohttp import web

__all__ = ["build_health_app", "health", "serve_health"]

_MODE_KEY = web.AppKey("mode", str)
_GUILD_KEY = web.AppKey("guild_id", str)


async def health(request: web.Request) -> web.Response:
    """``GET /health`` — static liveness payload for the configured guild."""
    return web.json_response(
        {
            "status": "ok",
            "mode": request.app[_MODE_KEY],
            "guild": request.app[_GUILD_KEY],
        }
    )


def build_health_app(*, mode: str, guild_id: str) -> web.Application:
    """Build the health-only aiohttp app (no other routes, no Discord)."""
    app = web.Application()
    app[_MODE_KEY] = mode
    app[_GUILD_KEY] = guild_id
    app.router.add_get("/health", health)
    return app


async def serve_health(
    *,
    host: str,
    port: int,
    mode: str,
    guild_id: str,
) -> int:
    """Run the health endpoint until SIGINT/SIGTERM; return the exit code.

    Prints one compact JSON readiness line to stdout (flushed) so callers can
    detect startup, then blocks until killed. Shuts the server down cleanly
    on signal so ``kill <pid>`` exits 0.
    """
    app = build_health_app(mode=mode, guild_id=guild_id)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    print(
        json.dumps({"serving": True, "mode": mode, "port": port, "guild": guild_id}),
        flush=True,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await runner.cleanup()
    return 0
