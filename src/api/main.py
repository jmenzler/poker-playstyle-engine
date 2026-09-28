"""Local study API; remote access requires an external authentication boundary."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src._log import get_logger
from src.api import jobs as api_jobs
from src.api.deps import get_config

log = get_logger("api.main")

_CORS_ORIGINS = [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "tauri://localhost",
    "https://tauri.localhost",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background JobRegistry sweeper on app boot; cancel on shutdown."""
    cfg = get_config()
    sweep_task = asyncio.create_task(api_jobs.sweeper_loop(ttl_s=cfg.job_ttl_s, interval_s=300.0))
    log.info("api.main.startup", host=cfg.fastapi_host, port=cfg.fastapi_port)
    try:
        yield
    finally:
        sweep_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweep_task
        log.info("api.main.shutdown")


app = FastAPI(title="poker-engine study", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
async def health() -> dict:
    """Operational ping. Used by Tauri frontend status bar."""
    return {"status": "ok", "n_jobs": len(api_jobs.registry)}


def _include_routers() -> None:
    """Import every router and attach to the app.

    Deferred to function scope to avoid circular imports with src.api.deps
    (which routers depend on). An ImportError here is fatal by design — see the
    call site — so a missing/broken router fails startup loudly rather than
    booting an app with silently dropped endpoints.
    """
    from src.api.autoloop import router as autoloop_router
    from src.api.dashboard import router as dashboard_router
    from src.api.decisions import router as decisions_router
    from src.api.edit_node import router as edit_node_router
    from src.api.eval import router as eval_router
    from src.api.eval_suite import router as eval_suite_router
    from src.api.gaps import router as gaps_router
    from src.api.hand_ranges import router as hand_ranges_router
    from src.api.hands import router as hands_router
    from src.api.hm3 import router as hm3_router
    from src.api.ingest import router as ingest_router
    from src.api.leaks import router as leaks_router
    from src.api.patches import router as patches_router
    from src.api.probe import router as probe_router
    from src.api.similar import router as similar_router
    from src.api.suppressions import router as suppressions_router
    from src.api.verify import router as verify_router

    routers = (
        leaks_router,
        similar_router,
        probe_router,
        patches_router,
        dashboard_router,
        hands_router,
        hand_ranges_router,
        edit_node_router,
        ingest_router,
        verify_router,
        eval_router,
        eval_suite_router,
        suppressions_router,
        decisions_router,
        hm3_router,
        autoloop_router,
        gaps_router,
    )
    for r in routers:
        app.include_router(r)
    log.info("api.main.routers_attached", n=len(routers))


# Import routers at module load. An ImportError in any router file fails the
# whole app — surface that loudly rather than booting with endpoints silently
# missing (ERR-01).
_include_routers()

# --- StaticFiles mount for built React/Vite frontend --------------------------
# Mount is AFTER routers so /api/* routes win over the catch-all static handler.
# Conditional: dev checkouts without a built dist/ still boot cleanly.
from pathlib import Path  # noqa: E402 (after routers intentionally)

_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
_FRONTEND_INDEX = _FRONTEND_DIST / "index.html"
if _FRONTEND_DIST.is_dir():
    from fastapi import Request
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str, request: Request):
        """SPA fallback — serve static asset if present, else index.html.

        StaticFiles(html=True) only falls back on directories, so deep routes
        like /replayer/12345 return 404. Custom catch-all serves the file when
        it exists under dist/, otherwise the SPA index so react-router can
        resolve the route client-side.
        """
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_INDEX)

    log.info("api.main.spa_fallback_mounted", path=str(_FRONTEND_DIST))
else:
    log.warning("api.main.static_missing", path=str(_FRONTEND_DIST))
