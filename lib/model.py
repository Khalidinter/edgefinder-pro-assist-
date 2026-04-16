"""
EdgeFinder Assist Model — Core projection engine.
Extracted from app.py. Stateless — all caching done via Supabase.
"""
import re
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nba_api.stats.endpoints import playergamelogs, leaguedashteamstats, playerdashptpass
from nba_api.stats.static import players

from lib.db import get_cached_game_logs, save_game_logs
from lib.config import (
    ODDS_API_KEY, SPORT, MARKET, BOOKMAKER,
    MIN_GAMES_REQUIRED, NB_ALPHA, logger,
)
from lib.backtest_utils import get_project_root

import xgboost as xgb

_ASSIST_XGB_MODEL = None

def _get_assist_xgb_model():
    """Lazy-load the trained binary classifier (thread-unsafe but fine for single-process use)."""
    global _ASSIST_XGB_MODEL
    if _ASSIST_XGB_MODEL is None:
        model_path = get_project_root() / "models" / "binary_classifier.json"
        if model_path.exists():
            m = xgb.XGBClassifier()
            m.load_model(str(model_path))
            _ASSIST_XGB_MODEL = m
            logger.info("Loaded assist XGB model from %s", model_path)
    return _ASSIST_XGB_MODEL

# ── HTTP Session ──
retry_strategy = Retry(
    total=3, connect=3, read=3, status=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http_session = requests.Session()
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)

# ── Team Maps ──
TEAM_ABBR_TO_NAME = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

ALIAS_MAP = {
    "sga": "shai gilgeous alexander",
    "shai gilgeous-alexander": "shai gilgeous alexander",
    "steph curry": "stephen curry",
    "lebron james sr": "lebron james",
    "donovan mitchell jr": "donovan mitchell",
    "la clippers": "los angeles clippers",
}

POSITION_PA_MULTIPLIERS = {
    "Guard": 1.90, "G": 1.90, "Forward": 1.75, "F": 1.75,
    "Center": 1.65, "C": 1.65, "Guard-Forward": 1.82, "G-F": 1.82,
    "Forward-Guard": 1.82, "F-G": 1.82, "Forward-Center": 1.70,
    "F-C": 1.70, "Center-Forward": 1.70, "C-F": 1.70,
}

POSITION_CONVERSION_DEFAULTS = {
    "Guard": 0.52, "G": 0.52, "Forward": 0.57, "F": 0.57,
    "Center": 0.61, "C": 0.61, "Guard-Forward": 0.54, "G-F": 0.54,
    "Forward-Guard": 0.54, "F-G": 0.54, "Forward-Center": 0.59,
    "F-C": 0.59, "Center-Forward": 0.59, "C-F": 0.59,
}

BACKTEST_FALLBACK_PLAYERS = [
    "Trae Young", "Tyrese Haliburton", "LeBron James", "Luka Doncic",
    "Jalen Brunson", "Darius Garland", "James Harden", "De'Aaron Fox",
    "Cade Cunningham", "Nikola Jokic", "Shai Gilgeous-Alexander",
    "Jayson Tatum", "Tyrese Maxey", "Dejounte Murray", "Fred VanVleet",
]


# ── Helpers ──
def normalize_name(name: Any) -> str:
    if not name:
        return ""
    clean = str(name).lower().strip()
    clean = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", clean)
    clean = clean.replace("-", " ")
    clean = re.sub(r"[^a-z\s]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return ALIAS_MAP.get(clean, clean)


def prettify_player_name(norm_name: str) -> str:
    return " ".join(w.capitalize() for w in norm_name.split())


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def current_nba_season_string() -> str:
    now = datetime.now()
    if now.month >= 10:
        start_year = now.year
        end_year = (now.year + 1) % 100
    else:
        start_year = now.year - 1
        end_year = now.year % 100
    return f"{start_year}-{end_year:02d}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def weighted_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    weights = list(range(1, len(values) + 1))
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def safe_minutes_to_float(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if ":" in s:
        try:
            mins, secs = s.split(":")
            return float(mins) + (float(secs) / 60.0)
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def american_to_implied(odds: Any) -> float:
    try:
        if odds is None or odds == 0:
            return 0.0
        if odds < 0:
            return (-odds) / ((-odds) + 100)
        return 100 / (odds + 100)
    except (TypeError, ZeroDivisionError):
        return 0.0


def american_to_decimal(odds: Any) -> float:
    try:
        if odds is None or odds == 0:
            return 1.0
        if odds < 0:
            return 1.0 + (100.0 / abs(odds))
        return 1.0 + (odds / 100.0)
    except (TypeError, ZeroDivisionError):
        return 1.0


# ── Odds API ──
def fetch_nba_events() -> Tuple[Optional[List[Dict]], Optional[str]]:
    if not ODDS_API_KEY:
        return None, "ODDS_API_KEY missing"
    try:
        res = http_session.get(
            f"https://api.the-odds-api.com/v4/sports/{SPORT}/events",
            params={"apiKey": ODDS_API_KEY}, timeout=10,
        )
        res.raise_for_status()
        return res.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Failed to fetch events: {e}"


def fetch_event_props(event_id: str) -> Tuple[Optional[Dict], Optional[str]]:
    if not ODDS_API_KEY:
        return None, "ODDS_API_KEY missing"
    try:
        res = http_session.get(
            f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds",
            params={
                "apiKey": ODDS_API_KEY, "regions": "us",
                "markets": MARKET, "bookmakers": BOOKMAKER, "oddsFormat": "american",
            }, timeout=10,
        )
        res.raise_for_status()
        return res.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Failed to fetch event props for {event_id}: {e}"


def fetch_live_market() -> Tuple[Optional[Dict], Optional[str], str]:
    fetched_at = utc_now_str()
    events, err = fetch_nba_events()
    if events is None:
        return None, err, fetched_at

    live_market: Dict[str, Any] = {}
    parsed = 0
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        odds_payload, odds_err = fetch_event_props(event_id)
        if odds_payload is None:
            logger.warning(odds_err)
            continue
        parsed += 1
        home_team = odds_payload.get("home_team", "")
        away_team = odds_payload.get("away_team", "")
        for book in odds_payload.get("bookmakers", []):
            for market_block in book.get("markets", []):
                for outcome in market_block.get("outcomes", []):
                    raw_name = outcome.get("description", "")
                    bet_type = outcome.get("name")
                    line = outcome.get("point")
                    price = outcome.get("price")
                    if not raw_name or bet_type is None or line is None or price is None:
                        continue
                    norm = normalize_name(raw_name)
                    if not norm:
                        continue
                    if norm not in live_market:
                        live_market[norm] = {}
                    if line not in live_market[norm]:
                        live_market[norm][line] = {
                            "book": book.get("title", BOOKMAKER),
                            "market": market_block.get("key", MARKET),
                            "event_id": event_id,
                            "home_team": home_team, "away_team": away_team,
                        }
                    if bet_type == "Over":
                        live_market[norm][line]["over_odds"] = price
                    elif bet_type == "Under":
                        live_market[norm][line]["under_odds"] = price

    logger.info("Live market loaded. events=%s players=%s", parsed, len(live_market))
    return live_market, None, fetched_at


# ── NBA API: Player Logs ──
def find_player_id(player_name: str) -> Optional[int]:
    matches = players.find_players_by_full_name(player_name)
    if not matches:
        return None
    exact = normalize_name(player_name)
    for m in matches:
        if normalize_name(m.get("full_name", "")) == exact:
            return m["id"]
    return matches[0]["id"]


def get_player_logs_df(
    player_name: str, season: str
) -> Optional[Tuple[pd.DataFrame, int, int]]:
    player_id = find_player_id(player_name)
    if player_id is None:
        return None
    try:
        df = playergamelogs.PlayerGameLogs(
            player_id_nullable=player_id, season_nullable=season,
            timeout=10,
        ).get_data_frames()[0]
        if df.empty:
            return None
        df = df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["MIN_FLOAT"] = df["MIN"].apply(safe_minutes_to_float)
        df["AST"] = pd.to_numeric(df["AST"], errors="coerce").fillna(0.0)
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        team_id = int(df.iloc[-1]["TEAM_ID"]) if "TEAM_ID" in df.columns else 0
        # Cache fresh logs for future fallback
        try:
            records = []
            for _, row in df.iterrows():
                records.append({
                    "player_name": player_name,
                    "player_id": player_id,
                    "team_id": team_id,
                    "team_abbr": str(row.get("TEAM_ABBREVIATION", "")),
                    "season": season,
                    "game_date": str(row["GAME_DATE"].date()),
                    "game_id": str(row.get("GAME_ID", "")),
                    "matchup": str(row.get("MATCHUP", "")),
                    "minutes": round(float(row.get("MIN_FLOAT", 0)), 1),
                    "assists": int(row.get("AST", 0)),
                    "points": int(row.get("PTS", 0)),
                    "rebounds": int(row.get("REB", 0)),
                    "turnovers": int(row.get("TOV", 0)),
                    "fga": int(row.get("FGA", 0)),
                    "fgm": int(row.get("FGM", 0)),
                })
            save_game_logs(records)
        except Exception:
            pass
        return df, player_id, team_id
    except Exception as e:
        logger.warning("Failed to fetch logs for %s: %s — trying cache", player_name, e)
        return _load_cached_logs(player_name, player_id, season)


def _load_cached_logs(
    player_name: str, player_id: int, season: str
) -> Optional[Tuple[pd.DataFrame, int, int]]:
    """Reconstruct a logs DataFrame from the Supabase game-log cache."""
    cached = get_cached_game_logs(player_name, season)
    if not cached or len(cached) < 5:
        return None
    df = pd.DataFrame(cached)
    # Map cached column names → model column names
    df = df.rename(columns={
        "game_date": "GAME_DATE",
        "team_abbr": "TEAM_ABBREVIATION",
        "team_id": "TEAM_ID",
        "matchup": "MATCHUP",
        "assists": "AST",
        "points": "PTS",
        "rebounds": "REB",
        "turnovers": "TOV",
        "fga": "FGA",
        "fgm": "FGM",
        "minutes": "MIN",
    })
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN_FLOAT"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0)
    df["AST"] = pd.to_numeric(df["AST"], errors="coerce").fillna(0.0)
    df = df[df["MIN_FLOAT"] > 0].sort_values("GAME_DATE").reset_index(drop=True)
    if df.empty:
        return None
    team_id = int(df.iloc[-1]["TEAM_ID"]) if "TEAM_ID" in df.columns else 0
    logger.info("Using cached logs for %s %s (%d games)", player_name, season, len(df))
    return df, player_id, team_id


def get_multi_season_logs(
    player_name: str, seasons: List[str]
) -> Optional[Tuple[pd.DataFrame, int, int]]:
    all_frames = []
    pid, tid = None, 0
    for season in seasons:
        result = get_player_logs_df(player_name, season)
        if result is not None:
            df, pid, tid = result
            all_frames.append(df)
        time.sleep(0.6)
    if not all_frames:
        return None
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)
    return combined, pid, tid


# ── Tracking Data ──
def fetch_player_tracking_data(
    player_id: int, team_id: int, season: str
) -> Optional[Dict[str, float]]:
    try:
        time.sleep(0.6)
        df = playerdashptpass.PlayerDashPtPass(
            player_id=player_id, team_id=team_id,
            season=season, per_mode_simple="Totals",
            timeout=10,
        ).get_data_frames()[0]
        if df.empty:
            return None
        total_ast = pd.to_numeric(df["AST"], errors="coerce").fillna(0).sum()
        total_fga = pd.to_numeric(df["FGA"], errors="coerce").fillna(0).sum()
        total_fgm = pd.to_numeric(df["FGM"], errors="coerce").fillna(0).sum()
        if total_ast <= 0:
            return None
        missed = total_fga - total_fgm
        potential = total_ast + missed
        return {
            "pa_ratio": clamp(potential / total_ast, 1.3, 2.8),
            "tracking_conversion": clamp(total_ast / potential, 0.25, 0.80),
            "total_ast": float(total_ast),
            "potential_ast": float(potential),
        }
    except Exception as e:
        logger.warning("Tracking fetch failed for player_id=%s: %s", player_id, e)
        return None


def get_pa_data(
    player_id: int, team_id: int, season: str
) -> Tuple[float, str, Optional[float]]:
    tracking = fetch_player_tracking_data(player_id, team_id, season)
    if tracking is not None:
        return tracking["pa_ratio"], "tracking", tracking["tracking_conversion"]
    info = players.find_player_by_id(player_id)
    if info:
        pos = info.get("position", "")
        if pos in POSITION_PA_MULTIPLIERS:
            return POSITION_PA_MULTIPLIERS[pos], "position", None
    return 1.80, "default", None


# ── Team Context ──
def get_team_context_tables(season: str) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        adv = leaguedashteamstats.LeagueDashTeamStats(
            season=season, season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame",
            timeout=10,
        ).get_data_frames()[0]
        opp = leaguedashteamstats.LeagueDashTeamStats(
            season=season, season_type_all_star="Regular Season",
            measure_type_detailed_defense="Opponent", per_mode_detailed="PerGame",
            timeout=10,
        ).get_data_frames()[0]
        pace_map = adv.set_index("TEAM_NAME")["PACE"].to_dict()
        ast_col = "AST" if "AST" in opp.columns else "OPP_AST"
        opp_ast_map = opp.set_index("TEAM_NAME")[ast_col].to_dict()
        ctx = {}
        for t in sorted(set(pace_map) | set(opp_ast_map)):
            ctx[t] = {"pace": float(pace_map.get(t, 100)), "opp_ast_allowed": float(opp_ast_map.get(t, 25))}
        return ctx, None
    except Exception as e:
        return {}, f"Team context failed: {e}"


def infer_current_team_from_logs(logs: pd.DataFrame) -> Optional[str]:
    if logs is None or logs.empty or "TEAM_ABBREVIATION" not in logs.columns:
        return None
    return TEAM_ABBR_TO_NAME.get(str(logs.iloc[-1]["TEAM_ABBREVIATION"]).upper())


def resolve_opponent_team(player_team: Optional[str], home: str, away: str) -> Optional[str]:
    if not player_team:
        return None
    nh, na = normalize_name(home), normalize_name(away)
    ch = next((v for v in TEAM_ABBR_TO_NAME.values() if normalize_name(v) == nh), home)
    ca = next((v for v in TEAM_ABBR_TO_NAME.values() if normalize_name(v) == na), away)
    if player_team == ch:
        return ca
    if player_team == ca:
        return ch
    return None


# ── Model Components ──
def get_rest_days(logs: pd.DataFrame) -> int:
    if logs.empty or "GAME_DATE" not in logs.columns:
        return 2
    try:
        return int(max((pd.Timestamp.now() - logs["GAME_DATE"].max()).days, 0))
    except Exception:
        return 2


def rest_factor_from_days(d: int) -> float:
    if d <= 1: return 0.97
    if d == 2: return 1.00
    if d == 3: return 1.02
    return 1.03


def negative_binomial_prob_over_line(mean: float, line: float, alpha: float = NB_ALPHA) -> float:
    if mean is None or mean <= 0:
        return 0.0
    threshold = int(math.floor(line)) + 1
    variance = mean + alpha * (mean ** 2)
    if variance <= mean:
        variance = mean + 1e-6
    r = (mean ** 2) / (variance - mean)
    p = r / (r + mean)
    cdf = 0.0
    for k in range(threshold):
        log_pmf = math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1) + r * math.log(p) + k * math.log(1 - p)
        cdf += math.exp(log_pmf)
    return clamp(1 - cdf, 0.0, 1.0)


def compute_player_alpha(assists: pd.Series) -> float:
    a = assists.dropna()
    if len(a) < 10:
        return NB_ALPHA
    m = a.mean()
    if m < 0.5:
        return NB_ALPHA
    v = a.var(ddof=1)
    if v <= m:
        return 0.10
    return clamp((v - m) / (m ** 2), 0.10, 1.20)


def compute_venue_factor(df: pd.DataFrame, venue: str) -> float:
    if "MATCHUP" not in df.columns or venue == "Unknown":
        return 1.0
    df2 = df.copy()
    df2["IS_HOME"] = ~df2["MATCHUP"].astype(str).str.contains("@")
    home, away = df2[df2["IS_HOME"]], df2[~df2["IS_HOME"]]
    if home.empty or away.empty:
        return 1.0
    overall = df2["AST"].sum() / df2["MIN_FLOAT"].sum() if df2["MIN_FLOAT"].sum() > 0 else 1.0
    if overall <= 0:
        return 1.0
    rate = (home if venue == "Home" else away)
    r = rate["AST"].sum() / rate["MIN_FLOAT"].sum() if rate["MIN_FLOAT"].sum() > 0 else overall
    return clamp(r / overall, 0.95, 1.05)


def detect_role_change(logs: pd.DataFrame) -> Dict[str, Any]:
    result = {"flag": "stable", "minutes_change_pct": 0.0, "ast_rate_change_pct": 0.0}
    if logs is None or len(logs) < 10:
        return result
    df = logs.copy()
    df["MIN_FLOAT"] = df["MIN"].apply(safe_minutes_to_float)
    df["AST"] = pd.to_numeric(df["AST"], errors="coerce").fillna(0)
    df = df[df["MIN_FLOAT"] > 0]
    if len(df) < 10:
        return result
    last3, season = df.tail(3), df
    sm, rm = season["MIN_FLOAT"].mean(), last3["MIN_FLOAT"].mean()
    sr = season["AST"].sum() / season["MIN_FLOAT"].sum() if season["MIN_FLOAT"].sum() > 0 else 0
    rr = last3["AST"].sum() / last3["MIN_FLOAT"].sum() if last3["MIN_FLOAT"].sum() > 0 else 0
    mc = ((rm - sm) / sm * 100) if sm > 0 else 0
    ac = ((rr - sr) / sr * 100) if sr > 0 else 0
    result["minutes_change_pct"] = round(mc, 1)
    result["ast_rate_change_pct"] = round(ac, 1)
    if mc > 20 or ac > 25:
        result["flag"] = "elevated"
    elif mc < -20 or ac < -25:
        result["flag"] = "reduced"
    return result


def compute_confidence_grade(
    logs_df: pd.DataFrame, projected: float, line: float, pa_source: str
) -> str:
    score = 0.0
    n = len(logs_df)
    m = logs_df["AST"].mean()
    s = logs_df["AST"].std(ddof=1) if n > 1 else m
    cv = (s / m) if m > 0 else 1.0
    score += 35 if cv <= 0.25 else 28 if cv <= 0.40 else 18 if cv <= 0.55 else 10 if cv <= 0.70 else 5
    score += 25 if n >= 50 else 20 if n >= 30 else 12 if n >= 15 else 8 if n >= 10 else 3
    last10 = logs_df.tail(min(10, n))
    mm, ms = last10["MIN_FLOAT"].mean(), last10["MIN_FLOAT"].std(ddof=1) if len(last10) > 1 else 0
    mcv = (ms / mm) if mm > 0 else 1.0
    score += 20 if mcv <= 0.08 else 15 if mcv <= 0.15 else 10 if mcv <= 0.25 else 4
    dist = abs(projected - line)
    score += 15 if dist <= 0.5 else 12 if dist <= 1.0 else 7 if dist <= 2.0 else 3
    if pa_source == "tracking":
        score += 5
    if score >= 75: return "A"
    if score >= 55: return "B"
    if score >= 35: return "C"
    return "D"


# ── Main Projection ──
def build_assist_projection_from_logs(
    logs: pd.DataFrame, market_line: float, player_name: str,
    team_pace: float = 100.0, opponent_pace: float = 100.0,
    opponent_ast_allowed: float = 25.0,
    league_avg_pace: float = 100.0, league_avg_ast_allowed: float = 25.0,
    pa_multiplier: float = 1.80, pa_source: str = "default",
    tracking_conversion: Optional[float] = None, venue: str = "Unknown",
) -> Optional[Dict[str, Any]]:
    if logs is None or logs.empty or len(logs) < MIN_GAMES_REQUIRED:
        return None
    df = logs.copy()
    df["MIN_FLOAT"] = df["MIN"].apply(safe_minutes_to_float)
    df["AST"] = pd.to_numeric(df["AST"], errors="coerce").fillna(0)
    df = df[df["MIN_FLOAT"] > 0].copy()
    if len(df) < MIN_GAMES_REQUIRED:
        return None

    last5 = df.tail(5).copy()
    last10 = df.tail(min(10, len(df))).copy()

    proj_min = clamp(weighted_mean(last5["MIN_FLOAT"].tolist()), 12.0, 42.0)

    # Opportunity rate
    last10["PA_EST"] = last10["AST"] * pa_multiplier
    pa_rates = (last10["PA_EST"] / last10["MIN_FLOAT"]).replace([math.inf, -math.inf], 0).fillna(0)
    opp_rate = weighted_mean(pa_rates.tolist())
    df["PA_EST"] = df["AST"] * pa_multiplier
    season_pa = df["PA_EST"].sum() / df["MIN_FLOAT"].sum() if df["MIN_FLOAT"].sum() > 0 else opp_rate
    opp_rate = clamp(0.70 * opp_rate + 0.30 * season_pa, 0.03, 0.45)

    # Conversion rate (INDEPENDENT from pa_multiplier)
    last15 = df.tail(min(15, len(df))).copy()
    gl_conv = last15["AST"].sum() / (last15["MIN_FLOAT"].sum() * opp_rate) if (last15["MIN_FLOAT"].sum() * opp_rate) > 0 else 0.54
    gl_conv = clamp(gl_conv, 0.25, 0.80)
    if tracking_conversion is not None:
        conv_rate = 0.70 * tracking_conversion + 0.30 * gl_conv
    else:
        info = players.find_players_by_full_name(player_name)
        pos = info[0].get("position", "") if info else ""
        pos_conv = POSITION_CONVERSION_DEFAULTS.get(pos, 0.54)
        conv_rate = 0.50 * pos_conv + 0.50 * gl_conv
    conv_rate = clamp(conv_rate, 0.25, 0.80)

    rest = get_rest_days(df)
    rf = rest_factor_from_days(rest)
    pf = clamp(((team_pace + opponent_pace) / 2) / league_avg_pace if league_avg_pace > 0 else 1.0, 0.94, 1.06)
    of = clamp(opponent_ast_allowed / league_avg_ast_allowed if league_avg_ast_allowed > 0 else 1.0, 0.90, 1.10)

    r5 = last5["AST"].sum() / last5["MIN_FLOAT"].sum() if last5["MIN_FLOAT"].sum() > 0 else 0
    rs = df["AST"].sum() / df["MIN_FLOAT"].sum() if df["MIN_FLOAT"].sum() > 0 else r5
    tf = clamp((r5 / rs) if rs > 0 else 1.0, 0.92, 1.08)
    vf = compute_venue_factor(df, venue)
    mrf = clamp(proj_min / 28.0, 0.82, 1.0)

    exp_ast = clamp(proj_min * opp_rate * conv_rate * rf * pf * of * tf * vf * mrf, 0.1, 20.0)
    alpha = compute_player_alpha(df["AST"])
    nb_prob = negative_binomial_prob_over_line(exp_ast, market_line, alpha=alpha)

    # Try XGB model (primary); fall back to NB heuristic
    xgb_model = _get_assist_xgb_model()
    if xgb_model is not None:
        r10 = last10["AST"].sum() / last10["MIN_FLOAT"].sum() if last10["MIN_FLOAT"].sum() > 0 else r5
        ast_std = float(last10["AST"].std(ddof=1)) if len(last10) > 1 else 0.0
        fga_pm = last5["FGA"].sum() / last5["MIN_FLOAT"].sum() if "FGA" in last5.columns and last5["MIN_FLOAT"].sum() > 0 else 0.0
        pts_pm = last5["PTS"].sum() / last5["MIN_FLOAT"].sum() if "PTS" in last5.columns and last5["MIN_FLOAT"].sum() > 0 else 0.0
        tov_pm = last5["TOV"].sum() / last5["MIN_FLOAT"].sum() if "TOV" in last5.columns and last5["MIN_FLOAT"].sum() > 0 else 0.0
        is_home_flag = 1 if venue == "Home" else 0
        pred_minus = exp_ast - market_line
        line_minus_l10 = market_line - (r10 * proj_min)
        xgb_features = [[
            proj_min, r5, r10, rs, ast_std,
            fga_pm, pts_pm, tov_pm,
            team_pace, opponent_pace, opponent_ast_allowed,
            rest, is_home_flag, 1 if rest <= 1 else 0, len(df),
            market_line, pred_minus, line_minus_l10,
        ]]
        try:
            prob_over = float(xgb_model.predict_proba(xgb_features)[0, 1])
            prob_source = "xgb"
        except Exception as e:
            logger.warning("XGB inference failed: %s", e)
            prob_over = nb_prob
            prob_source = "nb_fallback"
    else:
        prob_over = nb_prob
        prob_source = "nb_fallback"

    role = detect_role_change(df)
    conf = compute_confidence_grade(df, exp_ast, market_line, pa_source)

    return {
        "display_name": player_name, "generated_at": utc_now_str(),
        "projected_minutes": round(proj_min, 1),
        "creation_activity": round(opp_rate, 3), "conversion_rate": round(conv_rate, 3),
        "pace_factor": round(pf, 3), "opponent_factor": round(of, 3),
        "venue_factor": round(vf, 3), "rest_days": rest,
        "expected_assists": round(exp_ast, 2), "over_prob": prob_over,
        "player_alpha": round(alpha, 3), "confidence": conf,
        "pa_source": pa_source, "games_used": len(df),
        "role_flag": role["flag"],
        "minutes_change_pct": role["minutes_change_pct"],
        "ast_rate_change_pct": role["ast_rate_change_pct"],
        "prob_source": prob_source,
    }


def build_player_projection(
    player_name: str, market_line: float, season: str, odds_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    result = get_player_logs_df(player_name, season)
    if result is None:
        return None
    logs, player_id, team_id = result
    if len(logs) < MIN_GAMES_REQUIRED:
        return None

    pa_mult, pa_src, trk_conv = get_pa_data(player_id, team_id, season)

    ctx, err = get_team_context_tables(season)
    ctx = ctx or {}
    if err:
        logger.warning(err)

    team = infer_current_team_from_logs(logs)
    home, away = odds_data.get("home_team", ""), odds_data.get("away_team", "")
    opp = resolve_opponent_team(team, home, away)

    lap, laa = 100.0, 25.0
    if ctx:
        pv = [v["pace"] for v in ctx.values() if "pace" in v]
        av = [v["opp_ast_allowed"] for v in ctx.values() if "opp_ast_allowed" in v]
        if pv: lap = sum(pv) / len(pv)
        if av: laa = sum(av) / len(av)

    tp = ctx.get(team, {}).get("pace", lap)
    op = ctx.get(opp, {}).get("pace", lap)
    oaa = ctx.get(opp, {}).get("opp_ast_allowed", laa)

    venue = "Home" if team and team == next(
        (v for v in TEAM_ABBR_TO_NAME.values() if normalize_name(v) == normalize_name(home)), None
    ) else "Away"

    model = build_assist_projection_from_logs(
        logs=logs, market_line=market_line, player_name=player_name,
        team_pace=tp, opponent_pace=op, opponent_ast_allowed=oaa,
        league_avg_pace=lap, league_avg_ast_allowed=laa,
        pa_multiplier=pa_mult, pa_source=pa_src,
        tracking_conversion=trk_conv, venue=venue,
    )
    if model is None:
        return None
    model["team"] = team or "Unknown"
    model["opponent"] = opp or "Unknown"
    model["venue"] = venue
    return model


# ── Row Builder ──
def build_projection_rows(
    live_market: Dict[str, Any], season: Optional[str] = None,
) -> Tuple[List[Dict], Dict, Dict[str, int], str]:
    season = season or current_nba_season_string()
    rows, metrics = [], {
        "market_players": len(live_market), "projected_players": 0,
        "skipped_missing_logs": 0, "skipped_bad_line": 0, "skipped_incomplete_market": 0,
    }
    gen_at = utc_now_str()

    for norm, lines_payload in live_market.items():
        name = prettify_player_name(norm)
        for line, odds in lines_payload.items():
            if "over_odds" not in odds or "under_odds" not in odds:
                metrics["skipped_incomplete_market"] += 1
                continue
            proj = build_player_projection(name, line, season, odds)
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
                "pa_source": proj.get("pa_source", "default"),
                "player_alpha": proj.get("player_alpha", NB_ALPHA),
                "games_used": proj.get("games_used", 0),
                "role_flag": proj.get("role_flag", "stable"),
                "minutes_change_pct": proj.get("minutes_change_pct", 0),
                "ast_rate_change_pct": proj.get("ast_rate_change_pct", 0),
                "book": odds.get("book", "Unknown"), "market": odds.get("market", MARKET),
                "over_odds": f"+{oo}" if isinstance(oo, (int, float)) and oo > 0 else str(oo),
                "under_odds": f"+{uo}" if isinstance(uo, (int, float)) and uo > 0 else str(uo),
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
    return rows, summary, metrics, gen_at
