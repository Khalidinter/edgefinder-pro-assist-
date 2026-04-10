"""Backtest endpoint — runs model against historical data, stores results."""
import os
from flask import Flask, request, render_template, jsonify

template_path = os.path.join(os.path.dirname(__file__), '..', 'templates')
app = Flask(__name__, template_folder=template_path)


@app.route("/api/backtest")
def backtest_page():
    from lib.config import logger
    from lib.model import (
        build_assist_projection_from_logs, get_pa_data, get_multi_season_logs,
        current_nba_season_string, BACKTEST_FALLBACK_PLAYERS, MIN_GAMES_REQUIRED,
    )
    from lib.db import save_backtest_run, get_cached_projections

    n = int(request.args.get("n", 20))
    n = min(max(n, 5), 60)
    season_str = request.args.get("seasons", current_nba_season_string())
    seasons = [s.strip() for s in season_str.split(",")]
    fmt = request.args.get("format", "html")

    # Get player list
    cached = get_cached_projections()
    player_names = list({r.get("player") for r in cached if r.get("player")})[:15]
    if not player_names:
        player_names = BACKTEST_FALLBACK_PLAYERS

    results = []
    for player_name in player_names:
        log_result = get_multi_season_logs(player_name, seasons)
        if log_result is None:
            continue
        full_logs, player_id, team_id = log_result
        if len(full_logs) < n + MIN_GAMES_REQUIRED:
            continue

        pa_mult, pa_src, trk_conv = get_pa_data(player_id, team_id, seasons[-1])

        for i in range(n, 0, -1):
            train = full_logs.iloc[:-i].copy()
            actual_game = full_logs.iloc[-i]
            if len(train) < MIN_GAMES_REQUIRED:
                continue

            actual_ast = float(actual_game["AST"])
            matching = [r for r in cached if r.get("player") == player_name]
            market_line = matching[0].get("market_line", round(train["AST"].mean() - 0.5)) if matching else round(train["AST"].mean() - 0.5)

            proj = build_assist_projection_from_logs(
                logs=train, market_line=market_line, player_name=player_name,
                pa_multiplier=pa_mult, pa_source=pa_src, tracking_conversion=trk_conv,
            )
            if proj is None:
                continue

            predicted = proj["expected_assists"]
            model_over = proj["over_prob"]
            actual_over = 1 if actual_ast > market_line else 0
            model_call = "OVER" if model_over > 0.5 else "UNDER"
            correct = (model_call == "OVER" and actual_over == 1) or (model_call == "UNDER" and actual_over == 0)

            results.append({
                "player": player_name,
                "season": ",".join(seasons),
                "game_date": str(actual_game["GAME_DATE"].date()) if hasattr(actual_game["GAME_DATE"], "date") else str(actual_game["GAME_DATE"]),
                "predicted": predicted,
                "actual": int(actual_ast),
                "market_line": market_line,
                "line_source": "synthetic",
                "model_over_pct": round(model_over * 100, 1),
                "model_call": model_call,
                "actual_result": "OVER" if actual_over else "UNDER",
                "correct": correct,
                "error": round(predicted - actual_ast, 2),
                "player_alpha": proj.get("player_alpha"),
                "confidence": proj.get("confidence"),
            })

    # Compute summary
    summary, calibration = {}, {}
    if results:
        total = len(results)
        correct_count = sum(1 for r in results if r["correct"])
        hit_rate = round(correct_count / total * 100, 1)
        mae = round(sum(abs(r["error"]) for r in results) / total, 2)
        avg_error = round(sum(r["error"] for r in results) / total, 2)

        buckets = {"50-55": [], "55-60": [], "60-65": [], "65-70": [], "70+": []}
        for r in results:
            sp = r["model_over_pct"] if r["model_call"] == "OVER" else 100 - r["model_over_pct"]
            hit = 1 if r["correct"] else 0
            if sp < 55: buckets["50-55"].append(hit)
            elif sp < 60: buckets["55-60"].append(hit)
            elif sp < 65: buckets["60-65"].append(hit)
            elif sp < 70: buckets["65-70"].append(hit)
            else: buckets["70+"].append(hit)

        for b, hits in buckets.items():
            if hits:
                calibration[b] = {"n": len(hits), "hit_rate": round(sum(hits) / len(hits) * 100, 1)}

        summary = {
            "seasons": seasons, "players_tested": len(player_names),
            "total_predictions": total, "correct": correct_count,
            "hit_rate": hit_rate, "mae": mae, "avg_error": avg_error,
        }

        try:
            save_backtest_run({**summary, "calibration": calibration, "line_source": "synthetic"}, results)
        except Exception as e:
            logger.warning("Failed to save backtest: %s", e)

    bt = {"results": results, "summary": summary, "calibration": calibration, "n_games_back": n}

    if fmt == "json":
        return jsonify(bt)
    return render_template("backtest.html", **bt)
