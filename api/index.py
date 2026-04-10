"""Main dashboard route — serves the projection table."""
import json
from flask import Flask, render_template
from lib.config import logger
from lib.db import get_cached_projections

app = Flask(__name__, template_folder="../templates")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def dashboard(path):
    cached = get_cached_projections()

    rows = cached if cached else []
    if rows:
        avg = round(sum(r.get("expected_assists", 0) for r in rows) / len(rows), 2)
        top = max(rows, key=lambda x: x.get("expected_assists", 0))
        summary = {
            "total_players": len(rows),
            "avg_expected_assists": avg,
            "highest_projection_player": top.get("player", "N/A"),
            "highest_projection_value": top.get("expected_assists", 0),
        }
    else:
        summary = {
            "total_players": 0, "avg_expected_assists": 0.0,
            "highest_projection_player": "N/A", "highest_projection_value": 0.0,
        }

    metrics = {
        "market_players": len(rows),
        "skipped_missing_logs": 0,
        "skipped_incomplete_market": 0,
    }

    return render_template(
        "index.html",
        rows=rows,
        summary=summary,
        last_updated=rows[0].get("generated_at", "Never") if rows else "Never",
        last_market_fetch="See /api/cron/refresh",
        status="Healthy" if rows else "Waiting for data",
        last_error="",
        metrics=metrics,
    )
