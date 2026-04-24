#!/usr/bin/env python3
"""Local FastAPI server that exposes Refresh/Resolve buttons from the Vercel sites.

Runs on localhost:8787 by default. The Vercel-deployed HTML reads the host from
the ?api= query param (or falls back to http://localhost:8787) and POSTs to:

  POST /refresh/assists    → python3 scripts/paper_trade.py --threshold 0.60
  POST /refresh/rebounds   → python3 scripts/rebound_paper_trade.py
  POST /resolve/assists    → python3 scripts/paper_trade.py --resolve
  POST /resolve/rebounds   → python3 scripts/rebound_paper_trade.py --resolve
  GET  /health             → {"ok": true}

Auth: if the env var EDGEFINDER_API_SECRET is set, every mutating request must
include a matching X-API-Secret header. Leave it unset for local-only use.

CORS: the browser on edgefinder-*.vercel.app and localhost:* can POST here. Other
origins are blocked.

Usage:
  pip install fastapi uvicorn
  python3 scripts/api.py                       # listens on 127.0.0.1:8787
  EDGEFINDER_API_SECRET=secret123 python3 scripts/api.py
  python3 scripts/api.py --host 0.0.0.0 --port 9000
"""
import os
import re
import sys
import asyncio
import argparse
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable  # use the same interpreter the server runs on
API_SECRET = os.getenv("EDGEFINDER_API_SECRET", "")

# Per-action timeouts (S4 fix). Refresh hits NBA API + runs the model and can
# legitimately take many minutes. Resolve just reads yesterday's rows and
# updates them — if it runs longer than 2 minutes, something is hung and
# continuing to hold the async lock only blocks future refreshes.
REFRESH_TIMEOUT = int(os.getenv("EDGEFINDER_REFRESH_TIMEOUT", "1200"))  # 20 min
RESOLVE_TIMEOUT = int(os.getenv("EDGEFINDER_RESOLVE_TIMEOUT", "120"))   # 2 min

# Only these (model, action) combos map to real scripts.
COMMANDS: dict[tuple[str, str], list[str]] = {
    ("assists", "refresh"):  [PY, "scripts/paper_trade.py", "--threshold", "0.60"],
    ("assists", "resolve"):  [PY, "scripts/paper_trade.py", "--resolve"],
    ("rebounds", "refresh"): [PY, "scripts/rebound_paper_trade.py"],
    ("rebounds", "resolve"): [PY, "scripts/rebound_paper_trade.py", "--resolve"],
}

# Action → timeout mapping so refresh can hang for minutes waiting on the NBA
# API while resolve stays on a tight leash.
ACTION_TIMEOUTS: dict[str, int] = {
    "refresh": REFRESH_TIMEOUT,
    "resolve": RESOLVE_TIMEOUT,
}

# One-at-a-time guard per (model, action) so a double-click can't spawn two
# copies of the same long-running script against the same Supabase rows.
_locks: dict[tuple[str, str], asyncio.Lock] = {
    key: asyncio.Lock() for key in COMMANDS
}

app = FastAPI(title="EdgeFinder Local API", version="1.0.0")

# edgefinder-assists.vercel.app, edgefinder-rebounds.vercel.app, their preview
# deployments, plus any localhost port for dev.
#
# allow_private_network=True opts into Chrome's Private Network Access
# handshake — when an HTTPS public site reaches http://localhost, Chrome
# sends `Access-Control-Request-Private-Network: true` on the preflight
# and expects `Access-Control-Allow-Private-Network: true` back. Starlette
# 1.0's CORSMiddleware emits that header when this flag is on; without it,
# the preflight fails with "Disallowed CORS private-network".
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https://edgefinder-[a-z0-9-]+\.vercel\.app$|^http://localhost:\d+$|^http://127\.0\.0\.1:\d+$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Secret"],
    allow_private_network=True,
)


def _check_auth(request: Request) -> None:
    if not API_SECRET:
        return  # auth disabled
    sent = request.headers.get("x-api-secret", "")
    if sent != API_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Bad or missing X-API-Secret header")


async def _run_script(model: str, action: str) -> dict:
    key = (model, action)
    cmd = COMMANDS.get(key)
    if cmd is None:
        raise HTTPException(status_code=404, detail=f"Unknown endpoint: /{action}/{model}")

    lock = _locks[key]
    if lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{action} for {model} is already running — wait for it to finish",
        )

    async with lock:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        timeout = ACTION_TIMEOUTS.get(action, RESOLVE_TIMEOUT)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise HTTPException(
                status_code=504,
                detail=f"{action}/{model} exceeded {timeout}s timeout — process killed",
            )

        output = (stdout or b"").decode("utf-8", errors="replace")
        # Send back just the tail — the full script log can be 100+ KB.
        tail = output[-4000:] if len(output) > 4000 else output

        if proc.returncode != 0:
            # Try to pull the last INFO/WARN/ERROR line for a cleaner message.
            last_signal = ""
            for m in re.finditer(r"\| (INFO|WARNING|ERROR) \| .+", tail):
                last_signal = m.group(0)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": f"{cmd[1]} exited with code {proc.returncode}",
                    "summary": last_signal or "no log signal captured",
                    "log_tail": tail,
                },
            )

        return {"ok": True, "model": model, "action": action, "log_tail": tail}


@app.get("/health")
def health():
    return {"ok": True, "auth_required": bool(API_SECRET),
            "models": sorted({m for m, _ in COMMANDS}),
            "timeouts": {"refresh": REFRESH_TIMEOUT, "resolve": RESOLVE_TIMEOUT}}


@app.post("/refresh/{model}")
async def refresh(model: str, request: Request):
    _check_auth(request)
    return await _run_script(model, "refresh")


@app.post("/resolve/{model}")
async def resolve(model: str, request: Request):
    _check_auth(request)
    return await _run_script(model, "resolve")


@app.exception_handler(HTTPException)
async def _http_exc_handler(request: Request, exc: HTTPException):
    # Normalize detail so HTML's `await res.text()` lands on useful content.
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": exc.detail},
    )


def main():
    parser = argparse.ArgumentParser(description="EdgeFinder local API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)

    print(f"EdgeFinder API listening on http://{args.host}:{args.port}")
    print(f"  auth: {'REQUIRED (X-API-Secret)' if API_SECRET else 'disabled'}")
    print(f"  endpoints: /refresh/{{assists,rebounds}}, /resolve/{{assists,rebounds}}, /health")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
