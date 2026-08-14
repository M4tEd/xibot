"""Unit tests for the aiohttp health endpoint (songbot/bot/health.py).

The handler is exercised directly via aiohttp's mocked request — no socket,
no network, no Discord anything. The real server path (bind 127.0.0.1:3108,
SIGTERM-clean shutdown) is covered by the harness CLI integration test and
the worker's manual verification.
"""

from __future__ import annotations

import json

from aiohttp.test_utils import make_mocked_request

from songbot.bot.health import build_health_app, health


async def test_health_returns_status_mode_guild() -> None:
    app = build_health_app(mode="harness", guild_id="guild-1")
    request = make_mocked_request("GET", "/health", app=app)

    response = await health(request)

    assert response.status == 200
    assert response.content_type == "application/json"
    body = json.loads(response.text)
    assert body == {"status": "ok", "mode": "harness", "guild": "guild-1"}


async def test_health_reflects_live_mode_and_configured_guild() -> None:
    app = build_health_app(mode="live", guild_id="999888777")
    request = make_mocked_request("GET", "/health", app=app)

    response = await health(request)

    body = json.loads(response.text)
    assert body == {"status": "ok", "mode": "live", "guild": "999888777"}


def test_app_routes_health_get() -> None:
    app = build_health_app(mode="harness", guild_id="g")
    routes = {(r.method, r.resource.canonical) for r in app.router.routes()}
    assert ("GET", "/health") in routes
