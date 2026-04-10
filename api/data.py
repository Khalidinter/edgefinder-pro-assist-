"""JSON API endpoint for projection data."""
from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/api/data")
def api_data():
    try:
        from lib.db import get_cached_projections
        rows = get_cached_projections()
    except Exception:
        rows = []

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

    return jsonify({
        "rows": rows,
        "summary": summary,
        "status": "Healthy" if rows else "Waiting",
    })
