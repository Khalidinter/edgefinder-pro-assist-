"""Vercel Cron: daily cache refresh. Fetches live market + builds projections.
Uses fast path (position-based fallbacks) to stay under function timeout."""
import os
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/api/cron/refresh")
def cron_refresh():
    from lib.config import logger
    from lib.model import (
        fetch_live_market, current_nba_season_string, prettify_player_name,
        get_player_logs_df, build_assist_projection_from_logs,
        get_team_context_tables, infer_current_team_from_logs,
        resolve_opponent_team, normalize_name, american_to_implied,
        american_to_decimal, clamp, utc_now_str,
        TEAM_ABBR_TO_NAME, POSITION_PA_MULTIPLIERS, MIN_GAMES_REQUIRED,
        NB_ALPHA, MARKET,
    )
    from lib.db import save_projections

    auth = request.headers.get("Authorization", "")
    cron_secret = os.getenv("CRON_SECRET")
    if cron_secret and auth != f"Bearer {cron_secret}":
        if os.getenv("VERCEL_ENV"):
            return jsonify({"error": "Unauthorized"}), 401

    logger.info("Cron refresh triggered")

    market, err, fetched_at = fetch_live_market()
    if market is None:
        return jsonify({"error": err, "fetched_at": fetched_at}), 500

    season = current_nba_season_string()

    # Load team context once
    ctx, ctx_err = get_team_context_tables(season)
    ctx = ctx or {}
    if ctx_err:
        logger.warning(ctx_err)

    lap, laa = 100.0, 25.0
    if ctx:
        pv = [v["pace"] for v in ctx.values() if "pace" in v]
        av = [v["opp_ast_allowed"] for v in ctx.values() if "opp_ast_allowed" in v]
        if pv: lap = sum(pv) / len(pv)
        if av: laa = sum(av) / len(av)

    rows = []
    metrics = {"market_players": 0, "projected_players": 0,
               "skipped_missing_logs": 0, "skipped_incomplete_market": 0}
    gen_at = utc_now_str()

    for norm, lines_payload in market.items():
        name = prettify_player_name(norm)
        metrics["market_players"] += 1

        for line, odds in lines_payload.items():
            if "over_odds" not in odds or "under_odds" not in odds:
                metrics["skipped_incomplete_market"] += 1
                continue

            # Fast path: get logs only (no tracking API call)
            result = get_player_logs_df(name, season)
            if result is None:
                metrics["skipped_missing_logs"] += 1
                continue

            logs, player_id, team_id = result
            if len(logs) < MIN_GAMES_REQUIRED:
                metrics["skipped_missing_logs"] += 1
                continue

            # Use position-based PA multiplier (fast, no API call)
            from nba_api.stats.static import players
            info = players.find_player_by_id(player_id)
            pos = info.get("position", "") if info else ""
            pa_mult = POSITION_PA_MULTIPLIERS.get(pos, 1.80)
            pa_src = "position"

            team = infer_current_team_from_logs(logs)
            home, away = odds.get("home_team", ""), odds.get("away_team", "")
            opp = resolve_opponent_team(team, home, away)

            tp = ctx.get(team, {}).get("pace", lap)
            op = ctx.get(opp, {}).get("pace", lap)
            oaa = ctx.get(opp, {}).get("opp_ast_allowed", laa)

            venue = "Home" if team and team == next(
                (v for v in TEAM_ABBR_TO_NAME.values() if normalize_name(v) == normalize_name(home)), None
            ) else "Away"

            proj = build_assist_projection_from_logs(
                logs=logs, market_line=line, player_name=name,
                team_pace=tp, opponent_pace=op, opponent_ast_allowed=oaa,
                league_avg_pace=lap, league_avg_ast_allowed=laa,
                pa_multiplier=pa_mult, pa_source=pa_src, venue=venue,
            )
            if proj is None:
                metrics["skipped_missing_logs"] += 1
                continue

            metrics["projected_players"] += 1
            gen_at = proj.get("generated_at", gen_at)
            ea = proj["expected_assists"]

            oo, uo = odds.get("over_odds"), odds.get("under_odds")
            fo, fu = american_to_implied(oo), american_to_implied(uo)
            ft = fo + fu
            mio = fo / ft if ft > 0 else 0.5
            mfo = round(mio * 100, 1) if ft > 0 else None

            mop = proj["over_prob"]
            eo = round((mop - mio) * 100, 1)
            eu = round(((1 - mop) - (1 - mio)) * 100, 1)
            do_, du = american_to_decimal(oo), american_to_decimal(uo)
            evo = round((mop * do_ - 1) * 100, 1)
            evu = round(((1 - mop) * du - 1) * 100, 1)

            if eo >= eu:
                bs, be, bev, bd = "OVER", eo, evo, do_
            else:
                bs, be, bev, bd = "UNDER", eu, evu, du
            kr = (be / 100) / (bd - 1) if bd > 1 else 0
            kf = round(clamp(kr / 4, 0, 0.05) * 100, 2)

            proj["team"] = team or "Unknown"
            proj["opponent"] = opp or "Unknown"
            proj["venue"] = venue

            rows.append({
                "player": proj["display_name"], "team": proj["team"],
                "opponent": proj["opponent"], "venue": proj["venue"],
                "projected_minutes": proj["projected_minutes"],
                "creation_activity": proj["creation_activity"],
                "conversion_rate": proj["conversion_rate"],
                "pace_factor": proj["pace_factor"], "opponent_factor": proj["opponent_factor"],
                "venue_factor": proj.get("venue_factor", 1.0), "rest_days": proj["rest_days"],
                "expected_assists": ea, "market_line": line,
                "line_diff": round(ea - line, 2),
                "over_prob": round(mop * 100, 1), "market_fair_over": mfo,
                "edge_over": eo, "edge_under": eu,
                "ev_over": evo, "ev_under": evu,
                "best_side": bs, "best_edge": be, "best_ev": bev, "kelly_pct": kf,
                "confidence": proj.get("confidence", "C"),
                "pa_source": pa_src,
                "player_alpha": proj.get("player_alpha", NB_ALPHA),
                "games_used": proj.get("games_used", 0),
                "role_flag": proj.get("role_flag", "stable"),
                "minutes_change_pct": proj.get("minutes_change_pct", 0),
                "ast_rate_change_pct": proj.get("ast_rate_change_pct", 0),
                "book": odds.get("book", "Unknown"), "market": odds.get("market", MARKET),
                "over_odds": f"+{oo}" if isinstance(oo, (int, float)) and oo > 0 else str(oo),
                "under_odds": f"+{uo}" if isinstance(uo, (int, float)) and uo > 0 else str(uo),
                "generated_at": gen_at,
            })

    rows.sort(key=lambda x: abs(x["best_edge"]), reverse=True)

    if rows:
        avg = round(sum(r["expected_assists"] for r in rows) / len(rows), 2)
        top = max(rows, key=lambda x: x["expected_assists"])
        summary = {"total_players": len(rows), "avg_expected_assists": avg,
                    "highest_projection_player": top["player"],
                    "highest_projection_value": top["expected_assists"]}
    else:
        summary = {"total_players": 0, "avg_expected_assists": 0.0,
                    "highest_projection_player": "N/A", "highest_projection_value": 0.0}

    save_projections(rows, summary, metrics)

    return jsonify({
        "status": "ok",
        "players_projected": metrics["projected_players"],
        "market_players": metrics["market_players"],
        "fetched_at": fetched_at,
        "generated_at": gen_at,
    })
