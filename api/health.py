"""Health check — tests imports and Supabase connection."""
from flask import Flask, jsonify
import traceback

app = Flask(__name__)


@app.route("/api/health")
def health():
    checks = {}

    # Test basic imports
    try:
        import pandas as pd
        checks["pandas"] = f"OK ({pd.__version__})"
    except Exception as e:
        checks["pandas"] = f"FAIL: {e}"

    try:
        import requests
        checks["requests"] = f"OK ({requests.__version__})"
    except Exception as e:
        checks["requests"] = f"FAIL: {e}"

    try:
        from nba_api.stats.static import players
        p = players.find_players_by_full_name("LeBron James")
        checks["nba_api"] = f"OK (found {len(p)} players)"
    except Exception as e:
        checks["nba_api"] = f"FAIL: {e}"

    # Test Supabase
    try:
        from lib.config import get_supabase, SUPABASE_URL
        sb = get_supabase()
        checks["supabase_url"] = SUPABASE_URL[:30] + "..." if SUPABASE_URL else "MISSING"
        res = sb.schema("assist_model").table("model_projections").select("id").limit(1).execute()
        checks["supabase"] = f"OK (connected)"
    except Exception as e:
        checks["supabase"] = f"FAIL: {traceback.format_exc()[-200:]}"

    # Test Odds API config
    try:
        from lib.config import ODDS_API_KEY
        checks["odds_api_key"] = "SET" if ODDS_API_KEY else "MISSING"
    except Exception as e:
        checks["odds_api_key"] = f"FAIL: {e}"

    all_ok = all("OK" in str(v) or v == "SET" for v in checks.values())
    return jsonify({"status": "healthy" if all_ok else "degraded", "checks": checks})
