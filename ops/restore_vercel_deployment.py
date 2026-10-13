#!/usr/bin/env python3
"""Download the source files of a Vercel CLI deployment into a local folder.

Vercel keeps the uploaded source of every CLI deployment, so a project whose
local folder was lost can be restored from its last deployment.

Usage:
    VERCEL_TOKEN=... python3 restore_vercel_deployment.py DEPLOYMENT_ID TEAM_ID DEST_DIR

Example (EdgeFinder NBA app, production deployment of 2026-05-27):
    VERCEL_TOKEN=... python3 restore_vercel_deployment.py \
        dpl_FaQ4UxZJ9782c5fh1kHExeLrd3vr team_Xr2jyDONLp0U1ASqG8vWmuZI ~/Projects/edgefinder-nba-pro

Create the token at https://vercel.com/account/tokens (scope: the team that owns
the project). Only the Python standard library is used. Build output (dist/),
node_modules/ and .vercel/ are skipped.
"""
import base64
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

API = "https://api.vercel.com"
SKIP_PREFIXES = ("dist/", "node_modules/", ".vercel/")


def get(url: str, token: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read(), resp.headers.get("Content-Type", "")


def walk(nodes, prefix=""):
    """Yield (path, uid) for every file in the deployment file tree."""
    for node in nodes:
        name = node.get("name", "")
        path = f"{prefix}/{name}" if prefix else name
        if node.get("type") == "directory":
            yield from walk(node.get("children", []), path)
        elif node.get("type") == "file":
            yield path, node.get("uid")


def relative_paths(files):
    """CLI deployments nest the whole project under a top-level 'src' folder; strip it."""
    strip = "src/" if files and all(p.startswith("src/") for p, _ in files) else ""
    return [(p[len(strip):], uid) for p, uid in files]


def decode(raw: bytes, content_type: str) -> bytes:
    """The files endpoint wraps content as {"data": "<base64>"}; unwrap when present."""
    if "json" in content_type:
        try:
            body = json.loads(raw)
        except ValueError:
            return raw
        if isinstance(body, dict) and isinstance(body.get("data"), str) and set(body) <= {"data", "encoding"}:
            return base64.b64decode(body["data"])
    return raw


def main() -> None:
    token = os.environ.get("VERCEL_TOKEN", "")
    if len(sys.argv) != 4 or not token:
        print(__doc__)
        sys.exit(2)
    dep, team, dest = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]).expanduser()

    try:
        body, _ = get(f"{API}/v6/deployments/{dep}/files?teamId={team}", token)
    except urllib.error.HTTPError as e:
        print(f"Listing files failed: HTTP {e.code} {e.read()[:300]!r}")
        sys.exit(1)
    files = relative_paths(list(walk(json.loads(body))))
    if not files:
        print("No files in this deployment (only CLI/API deployments keep their source).")
        sys.exit(1)

    written = 0
    for rel, uid in files:
        if rel.startswith(SKIP_PREFIXES) or not uid:
            continue
        try:
            raw, ctype = get(f"{API}/v7/deployments/{dep}/files/{uid}?teamId={team}", token)
        except urllib.error.HTTPError as e:
            print(f"  FAILED {rel}: HTTP {e.code}")
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(decode(raw, ctype))
        written += 1
        print(f"  wrote {rel}")
    print(f"done: {written} files in {dest}")


if __name__ == "__main__":
    main()
