import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SPORT = "basketball_nba"
MARKET = "player_assists"
BOOKMAKER = "draftkings"
SCHEMA = "assist_model"

MIN_GAMES_REQUIRED = 5
NB_ALPHA = 0.35

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("edgefinder")


def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)
