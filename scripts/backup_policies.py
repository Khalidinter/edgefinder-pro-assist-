"""One-shot: write RLS backup artifacts to data/backup/2026-04-23/.

Inputs: inline data (policies/rls/roles/indexes) + the two MCP overflow files
on disk for grants and columns. Outputs: six JSON files with audit headers,
plus policies.sql with CREATE POLICY rollback statements.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

BACKUP_DIR = Path("data/backup/2026-04-23")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump(filename: str, source_query: str, rows: list) -> None:
    payload = {
        "captured_at": NOW,
        "source_query": source_query,
        "row_count": len(rows),
        "rows": rows,
    }
    out = BACKUP_DIR / filename
    out.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"  wrote {out} ({len(rows)} rows)")


# --- Inline data from MCP responses ---------------------------------------

POLICIES = [
    {"schemaname":"public","tablename":"am_backtest_results","policyname":"am_read_bt_res","permissive":"PERMISSIVE","roles":"{anon}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_backtest_results","policyname":"am_write_bt_res","permissive":"PERMISSIVE","roles":"{anon}","cmd":"INSERT","qual":None,"with_check":"true"},
    {"schemaname":"public","tablename":"am_backtest_results","policyname":"anon_read_backtest_results","permissive":"PERMISSIVE","roles":"{public}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_backtest_results","policyname":"service_write_backtest_results","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
    {"schemaname":"public","tablename":"am_backtest_runs","policyname":"am_read_bt_runs","permissive":"PERMISSIVE","roles":"{anon}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_backtest_runs","policyname":"am_write_bt_runs","permissive":"PERMISSIVE","roles":"{anon}","cmd":"INSERT","qual":None,"with_check":"true"},
    {"schemaname":"public","tablename":"am_backtest_runs","policyname":"anon_read_backtest_runs","permissive":"PERMISSIVE","roles":"{public}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_backtest_runs","policyname":"service_write_backtest_runs","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
    {"schemaname":"public","tablename":"am_game_logs","policyname":"am_read_logs","permissive":"PERMISSIVE","roles":"{anon}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_game_logs","policyname":"am_write_logs","permissive":"PERMISSIVE","roles":"{anon}","cmd":"INSERT","qual":None,"with_check":"true"},
    {"schemaname":"public","tablename":"am_game_logs","policyname":"anon_read_game_logs","permissive":"PERMISSIVE","roles":"{public}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_game_logs","policyname":"service_write_game_logs","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
    {"schemaname":"public","tablename":"am_paper_trades","policyname":"pt_read","permissive":"PERMISSIVE","roles":"{anon}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_paper_trades","policyname":"pt_update","permissive":"PERMISSIVE","roles":"{anon}","cmd":"UPDATE","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_paper_trades","policyname":"pt_write","permissive":"PERMISSIVE","roles":"{anon}","cmd":"INSERT","qual":None,"with_check":"true"},
    {"schemaname":"public","tablename":"am_player_tracking","policyname":"am_read_tracking","permissive":"PERMISSIVE","roles":"{anon}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_player_tracking","policyname":"am_write_tracking","permissive":"PERMISSIVE","roles":"{anon}","cmd":"INSERT","qual":None,"with_check":"true"},
    {"schemaname":"public","tablename":"am_player_tracking","policyname":"anon_read_tracking","permissive":"PERMISSIVE","roles":"{public}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_player_tracking","policyname":"service_write_tracking","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
    {"schemaname":"public","tablename":"am_projections","policyname":"am_delete_proj","permissive":"PERMISSIVE","roles":"{anon}","cmd":"DELETE","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_projections","policyname":"am_read_proj","permissive":"PERMISSIVE","roles":"{anon}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_projections","policyname":"am_write_proj","permissive":"PERMISSIVE","roles":"{anon}","cmd":"INSERT","qual":None,"with_check":"true"},
    {"schemaname":"public","tablename":"am_projections","policyname":"anon_read_projections","permissive":"PERMISSIVE","roles":"{public}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_projections","policyname":"service_write_projections","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
    {"schemaname":"public","tablename":"am_team_context","policyname":"am_read_context","permissive":"PERMISSIVE","roles":"{anon}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_team_context","policyname":"am_write_context","permissive":"PERMISSIVE","roles":"{anon}","cmd":"INSERT","qual":None,"with_check":"true"},
    {"schemaname":"public","tablename":"am_team_context","policyname":"anon_read_team_context","permissive":"PERMISSIVE","roles":"{public}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"am_team_context","policyname":"service_write_team_context","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
    {"schemaname":"public","tablename":"closing_prices","policyname":"service_role_all","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
    {"schemaname":"public","tablename":"kv_store","policyname":"service_role_all","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
    {"schemaname":"public","tablename":"pipeline_runs","policyname":"anon insert pipeline_runs","permissive":"PERMISSIVE","roles":"{anon}","cmd":"INSERT","qual":None,"with_check":"true"},
    {"schemaname":"public","tablename":"pipeline_runs","policyname":"anon read pipeline_runs","permissive":"PERMISSIVE","roles":"{anon}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"rb_paper_trades","policyname":"Allow all for service role","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
    {"schemaname":"public","tablename":"rb_projections","policyname":"Public read rb_projections","permissive":"PERMISSIVE","roles":"{public}","cmd":"SELECT","qual":"true","with_check":None},
    {"schemaname":"public","tablename":"rb_projections","policyname":"Service write rb_projections","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
    {"schemaname":"public","tablename":"server_runs","policyname":"service_role_all","permissive":"PERMISSIVE","roles":"{public}","cmd":"ALL","qual":"true","with_check":"true"},
]

RLS_STATE = [
    {"schema":"public","table":"am_backtest_results","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"am_backtest_runs","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"am_game_logs","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"am_paper_trades","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"am_player_tracking","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"am_projections","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"am_team_context","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"bets","relrowsecurity":False,"relforcerowsecurity":False},
    {"schema":"public","table":"closing_prices","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"edges","relrowsecurity":False,"relforcerowsecurity":False},
    {"schema":"public","table":"kv_store","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"pipeline_runs","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"player_games","relrowsecurity":False,"relforcerowsecurity":False},
    {"schema":"public","table":"rb_paper_trades","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"rb_projections","relrowsecurity":True,"relforcerowsecurity":False},
    {"schema":"public","table":"server_runs","relrowsecurity":True,"relforcerowsecurity":False},
]

ROLES = [
    {"rolname":"anon","rolsuper":False,"rolinherit":True,"rolcreaterole":False,"rolcreatedb":False,"rolcanlogin":False,"rolbypassrls":False},
    {"rolname":"authenticated","rolsuper":False,"rolinherit":True,"rolcreaterole":False,"rolcreatedb":False,"rolcanlogin":False,"rolbypassrls":False},
    {"rolname":"authenticator","rolsuper":False,"rolinherit":False,"rolcreaterole":False,"rolcreatedb":False,"rolcanlogin":True,"rolbypassrls":False},
    {"rolname":"postgres","rolsuper":False,"rolinherit":True,"rolcreaterole":True,"rolcreatedb":True,"rolcanlogin":True,"rolbypassrls":True},
    {"rolname":"service_role","rolsuper":False,"rolinherit":True,"rolcreaterole":False,"rolcreatedb":False,"rolcanlogin":False,"rolbypassrls":True},
]

INDEXES = [
    {"schemaname":"public","tablename":"am_backtest_results","indexname":"backtest_results_pkey","indexdef":"CREATE UNIQUE INDEX backtest_results_pkey ON public.am_backtest_results USING btree (id)"},
    {"schemaname":"public","tablename":"am_backtest_results","indexname":"idx_am_bt_player","indexdef":"CREATE INDEX idx_am_bt_player ON public.am_backtest_results USING btree (player)"},
    {"schemaname":"public","tablename":"am_backtest_results","indexname":"idx_am_bt_run","indexdef":"CREATE INDEX idx_am_bt_run ON public.am_backtest_results USING btree (run_id)"},
    {"schemaname":"public","tablename":"am_backtest_runs","indexname":"backtest_runs_pkey","indexdef":"CREATE UNIQUE INDEX backtest_runs_pkey ON public.am_backtest_runs USING btree (id)"},
    {"schemaname":"public","tablename":"am_game_logs","indexname":"historical_game_logs_pkey","indexdef":"CREATE UNIQUE INDEX historical_game_logs_pkey ON public.am_game_logs USING btree (id)"},
    {"schemaname":"public","tablename":"am_game_logs","indexname":"historical_game_logs_player_id_game_date_key","indexdef":"CREATE UNIQUE INDEX historical_game_logs_player_id_game_date_key ON public.am_game_logs USING btree (player_id, game_date)"},
    {"schemaname":"public","tablename":"am_game_logs","indexname":"idx_am_hgl_date","indexdef":"CREATE INDEX idx_am_hgl_date ON public.am_game_logs USING btree (game_date)"},
    {"schemaname":"public","tablename":"am_game_logs","indexname":"idx_am_hgl_player","indexdef":"CREATE INDEX idx_am_hgl_player ON public.am_game_logs USING btree (player_name)"},
    {"schemaname":"public","tablename":"am_game_logs","indexname":"idx_am_hgl_season","indexdef":"CREATE INDEX idx_am_hgl_season ON public.am_game_logs USING btree (season)"},
    {"schemaname":"public","tablename":"am_paper_trades","indexname":"am_paper_trades_date_player_uniq","indexdef":"CREATE UNIQUE INDEX am_paper_trades_date_player_uniq ON public.am_paper_trades USING btree (prediction_date, player_id)"},
    {"schemaname":"public","tablename":"am_paper_trades","indexname":"am_paper_trades_pkey","indexdef":"CREATE UNIQUE INDEX am_paper_trades_pkey ON public.am_paper_trades USING btree (id)"},
    {"schemaname":"public","tablename":"am_paper_trades","indexname":"idx_pt_date","indexdef":"CREATE INDEX idx_pt_date ON public.am_paper_trades USING btree (prediction_date)"},
    {"schemaname":"public","tablename":"am_paper_trades","indexname":"idx_pt_resolved","indexdef":"CREATE INDEX idx_pt_resolved ON public.am_paper_trades USING btree (resolved)"},
    {"schemaname":"public","tablename":"am_player_tracking","indexname":"player_tracking_pkey","indexdef":"CREATE UNIQUE INDEX player_tracking_pkey ON public.am_player_tracking USING btree (id)"},
    {"schemaname":"public","tablename":"am_player_tracking","indexname":"player_tracking_player_id_season_key","indexdef":"CREATE UNIQUE INDEX player_tracking_player_id_season_key ON public.am_player_tracking USING btree (player_id, season)"},
    {"schemaname":"public","tablename":"am_projections","indexname":"idx_am_proj_expires","indexdef":"CREATE INDEX idx_am_proj_expires ON public.am_projections USING btree (expires_at)"},
    {"schemaname":"public","tablename":"am_projections","indexname":"idx_am_proj_player","indexdef":"CREATE INDEX idx_am_proj_player ON public.am_projections USING btree (player)"},
    {"schemaname":"public","tablename":"am_projections","indexname":"model_projections_pkey","indexdef":"CREATE UNIQUE INDEX model_projections_pkey ON public.am_projections USING btree (id)"},
    {"schemaname":"public","tablename":"am_team_context","indexname":"team_context_pkey","indexdef":"CREATE UNIQUE INDEX team_context_pkey ON public.am_team_context USING btree (id)"},
    {"schemaname":"public","tablename":"am_team_context","indexname":"team_context_season_team_name_key","indexdef":"CREATE UNIQUE INDEX team_context_season_team_name_key ON public.am_team_context USING btree (season, team_name)"},
    {"schemaname":"public","tablename":"bets","indexname":"bets_pkey","indexdef":"CREATE UNIQUE INDEX bets_pkey ON public.bets USING btree (id)"},
    {"schemaname":"public","tablename":"bets","indexname":"idx_bets_book","indexdef":"CREATE INDEX idx_bets_book ON public.bets USING btree (book)"},
    {"schemaname":"public","tablename":"bets","indexname":"idx_bets_category","indexdef":"CREATE INDEX idx_bets_category ON public.bets USING btree (category)"},
    {"schemaname":"public","tablename":"bets","indexname":"idx_bets_game_date","indexdef":"CREATE INDEX idx_bets_game_date ON public.bets USING btree (game_date)"},
    {"schemaname":"public","tablename":"bets","indexname":"idx_bets_outcome_game_date","indexdef":"CREATE INDEX idx_bets_outcome_game_date ON public.bets USING btree (outcome, game_date)"},
    {"schemaname":"public","tablename":"bets","indexname":"idx_bets_player","indexdef":"CREATE INDEX idx_bets_player ON public.bets USING btree (player)"},
    {"schemaname":"public","tablename":"bets","indexname":"idx_bets_resolved","indexdef":"CREATE INDEX idx_bets_resolved ON public.bets USING btree (resolved)"},
    {"schemaname":"public","tablename":"closing_prices","indexname":"closing_prices_pkey","indexdef":"CREATE UNIQUE INDEX closing_prices_pkey ON public.closing_prices USING btree (game_date, player_key)"},
    {"schemaname":"public","tablename":"closing_prices","indexname":"idx_closing_prices_date","indexdef":"CREATE INDEX idx_closing_prices_date ON public.closing_prices USING btree (game_date)"},
    {"schemaname":"public","tablename":"edges","indexname":"edges_pkey","indexdef":"CREATE UNIQUE INDEX edges_pkey ON public.edges USING btree (id)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_book","indexdef":"CREATE INDEX idx_edges_book ON public.edges USING btree (book)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_created","indexdef":"CREATE INDEX idx_edges_created ON public.edges USING btree (created_at DESC)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_created_at","indexdef":"CREATE INDEX idx_edges_created_at ON public.edges USING btree (created_at DESC)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_es_tier","indexdef":"CREATE INDEX idx_edges_es_tier ON public.edges USING btree (es_tier)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_game_date","indexdef":"CREATE INDEX idx_edges_game_date ON public.edges USING btree (game_date)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_player","indexdef":"CREATE INDEX idx_edges_player ON public.edges USING btree (player)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_player_book","indexdef":"CREATE INDEX idx_edges_player_book ON public.edges USING btree (player, book)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_player_game_date","indexdef":"CREATE INDEX idx_edges_player_game_date ON public.edges USING btree (player, game_date DESC)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_player_stat","indexdef":"CREATE INDEX idx_edges_player_stat ON public.edges USING btree (player, stat, side)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_player_stat_side","indexdef":"CREATE INDEX idx_edges_player_stat_side ON public.edges USING btree (player, stat, side)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_resolved","indexdef":"CREATE INDEX idx_edges_resolved ON public.edges USING btree (resolved)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_resolved_game_date","indexdef":"CREATE INDEX idx_edges_resolved_game_date ON public.edges USING btree (resolved, game_date)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_stat_side_line","indexdef":"CREATE INDEX idx_edges_stat_side_line ON public.edges USING btree (stat, side, line)"},
    {"schemaname":"public","tablename":"edges","indexname":"idx_edges_stat_side_resolved","indexdef":"CREATE INDEX idx_edges_stat_side_resolved ON public.edges USING btree (stat, side, resolved)"},
    {"schemaname":"public","tablename":"kv_store","indexname":"kv_store_pkey","indexdef":"CREATE UNIQUE INDEX kv_store_pkey ON public.kv_store USING btree (key)"},
    {"schemaname":"public","tablename":"pipeline_runs","indexname":"idx_pipeline_runs_lookup","indexdef":"CREATE INDEX idx_pipeline_runs_lookup ON public.pipeline_runs USING btree (pipeline, run_type, run_at DESC)"},
    {"schemaname":"public","tablename":"pipeline_runs","indexname":"pipeline_runs_pkey","indexdef":"CREATE UNIQUE INDEX pipeline_runs_pkey ON public.pipeline_runs USING btree (id)"},
    {"schemaname":"public","tablename":"player_games","indexname":"idx_pg_game_date","indexdef":"CREATE INDEX idx_pg_game_date ON public.player_games USING btree (game_date)"},
    {"schemaname":"public","tablename":"player_games","indexname":"idx_pg_player_date","indexdef":"CREATE INDEX idx_pg_player_date ON public.player_games USING btree (player, game_date DESC)"},
    {"schemaname":"public","tablename":"player_games","indexname":"idx_pg_player_opponent","indexdef":"CREATE INDEX idx_pg_player_opponent ON public.player_games USING btree (player, opponent)"},
    {"schemaname":"public","tablename":"player_games","indexname":"idx_pg_player_stat","indexdef":"CREATE INDEX idx_pg_player_stat ON public.player_games USING btree (player, stat)"},
    {"schemaname":"public","tablename":"player_games","indexname":"player_games_pkey","indexdef":"CREATE UNIQUE INDEX player_games_pkey ON public.player_games USING btree (id)"},
    {"schemaname":"public","tablename":"rb_paper_trades","indexname":"rb_paper_trades_date_player_uniq","indexdef":"CREATE UNIQUE INDEX rb_paper_trades_date_player_uniq ON public.rb_paper_trades USING btree (prediction_date, player_id)"},
    {"schemaname":"public","tablename":"rb_paper_trades","indexname":"rb_paper_trades_pkey","indexdef":"CREATE UNIQUE INDEX rb_paper_trades_pkey ON public.rb_paper_trades USING btree (id)"},
    {"schemaname":"public","tablename":"rb_projections","indexname":"rb_projections_expires_at_idx","indexdef":"CREATE INDEX rb_projections_expires_at_idx ON public.rb_projections USING btree (expires_at)"},
    {"schemaname":"public","tablename":"rb_projections","indexname":"rb_projections_pkey","indexdef":"CREATE UNIQUE INDEX rb_projections_pkey ON public.rb_projections USING btree (id)"},
    {"schemaname":"public","tablename":"server_runs","indexname":"idx_server_runs_ts","indexdef":"CREATE INDEX idx_server_runs_ts ON public.server_runs USING btree (ts DESC)"},
    {"schemaname":"public","tablename":"server_runs","indexname":"server_runs_pkey","indexdef":"CREATE UNIQUE INDEX server_runs_pkey ON public.server_runs USING btree (id)"},
]


# --- Extract rows from MCP overflow files --------------------------------

def extract_rows(path: Path) -> list:
    """MCP tool-result files nest JSON several layers deep.

    Layer 1: outer list [{"type":"text","text":"<stringified JSON>"}]
    Layer 2: text is a JSON-encoded string whose value is {"result":"..."}
    Layer 3: result value is a plain string containing <untrusted-data-XXX>[rows]</untrusted-data-XXX>
    """
    outer = json.loads(path.read_text())
    inner_text = outer[0]["text"]
    inner_obj = json.loads(inner_text)
    result_str = inner_obj["result"]
    m = re.search(r"<untrusted-data-[^>]+>\s*(\[.*?\])\s*</untrusted-data", result_str, re.DOTALL)
    if not m:
        raise RuntimeError(f"could not find untrusted-data block in {path}")
    return json.loads(m.group(1))


GRANTS_FILE = Path("/Users/khalidhassan/.claude/projects/-Users-khalidhassan-Projects-edgefinder-assist-pro/f9294654-3d22-4e88-9e80-8511b63260b3/tool-results/mcp-6212ee66-8063-47ec-80f1-c469e9824345-execute_sql-1776990674143.txt")
COLUMNS_FILE = Path("/Users/khalidhassan/.claude/projects/-Users-khalidhassan-Projects-edgefinder-assist-pro/f9294654-3d22-4e88-9e80-8511b63260b3/tool-results/mcp-6212ee66-8063-47ec-80f1-c469e9824345-execute_sql-1776990684906.txt")

GRANTS = extract_rows(GRANTS_FILE)
COLUMNS = extract_rows(COLUMNS_FILE)


# --- Write JSON artifacts -------------------------------------------------

print("Writing JSON artifacts...")
dump("policies.json",
     "SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check FROM pg_policies WHERE schemaname = 'public' ORDER BY tablename, policyname;",
     POLICIES)
dump("rls_state.json",
     "SELECT n.nspname, c.relname, c.relrowsecurity, c.relforcerowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname='public' AND c.relkind='r' ORDER BY c.relname;",
     RLS_STATE)
dump("roles.json",
     "SELECT rolname, rolsuper, rolinherit, rolcreaterole, rolcreatedb, rolcanlogin, rolbypassrls FROM pg_roles WHERE rolname IN ('anon','authenticated','service_role','postgres','authenticator') ORDER BY rolname;",
     ROLES)
dump("indexes.json",
     "SELECT schemaname, tablename, indexname, indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY tablename, indexname;",
     INDEXES)
dump("grants.json",
     "SELECT grantee, table_schema, table_name, privilege_type, is_grantable FROM information_schema.role_table_grants WHERE table_schema='public' AND grantee IN ('anon','authenticated','service_role','postgres','authenticator','supabase_admin','supabase_auth_admin','supabase_storage_admin','dashboard_user','PUBLIC') ORDER BY table_name, grantee, privilege_type;",
     GRANTS)
dump("columns.json",
     "SELECT table_name, column_name, ordinal_position, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema='public' ORDER BY table_name, ordinal_position;",
     COLUMNS)


# --- Generate CREATE POLICY rollback SQL ---------------------------------

def roles_sql(roles_str: str) -> str:
    """Postgres dumps {anon} or {public,anon}; convert to SQL."""
    inner = roles_str.strip("{}")
    roles = [r.strip('"') for r in inner.split(",") if r.strip()]
    # Postgres serializes PUBLIC as "public" pseudo-role; emit as PUBLIC for clarity
    emitted = []
    for r in roles:
        emitted.append("PUBLIC" if r == "public" else r)
    return ", ".join(emitted)


def policy_sql(p: dict) -> str:
    name = p["policyname"]
    table = f"public.{p['tablename']}"
    permissive = p["permissive"].upper()  # PERMISSIVE | RESTRICTIVE
    cmd = p["cmd"].upper()  # SELECT | INSERT | UPDATE | DELETE | ALL
    to_clause = roles_sql(p["roles"])
    parts = [
        f'CREATE POLICY "{name}"',
        f"ON {table}",
        f"AS {permissive}",
        f"FOR {cmd}",
        f"TO {to_clause}",
    ]
    if p["qual"] is not None:
        parts.append(f"USING ({p['qual']})")
    if p["with_check"] is not None:
        parts.append(f"WITH CHECK ({p['with_check']})")
    return "  " + "\n  ".join(parts) + ";"


print("\nGenerating policies.sql...")
lines = [
    "-- EdgeFinder RLS policy rollback — captured " + NOW,
    "-- Source: pg_policies WHERE schemaname = 'public'",
    "-- " + "-" * 70,
    "-- To restore: run this file against the database as a role that can CREATE POLICY.",
    "-- First drop any conflicting policies by name, or wrap in BEGIN; DROP POLICY ...; etc.",
    "-- " + "-" * 70,
    "",
]
current_table = None
for p in POLICIES:
    if p["tablename"] != current_table:
        current_table = p["tablename"]
        lines.append(f"\n-- ==== {current_table} ====")
    lines.append(policy_sql(p))
    lines.append("")

# Also emit ALTER TABLE ... ENABLE ROW LEVEL SECURITY for every table that has it on.
lines.append("\n-- ==== ALTER TABLE … ENABLE ROW LEVEL SECURITY ====")
for r in RLS_STATE:
    if r["relrowsecurity"]:
        lines.append(f"ALTER TABLE public.{r['table']} ENABLE ROW LEVEL SECURITY;")
# And explicitly note which are OFF
lines.append("")
lines.append("-- Tables with RLS OFF at capture time (DO NOT re-enable without policies):")
for r in RLS_STATE:
    if not r["relrowsecurity"]:
        lines.append(f"--   public.{r['table']}")

sql_path = BACKUP_DIR / "policies.sql"
sql_path.write_text("\n".join(lines) + "\n")
print(f"  wrote {sql_path}")


# --- Verification --------------------------------------------------------

print("\n=== VERIFICATION ===")
print(f"  policies.json row_count = {len(POLICIES)}")
policy_sql_count = sum(1 for line in sql_path.read_text().splitlines() if line.startswith("  CREATE POLICY"))
print(f"  policies.sql CREATE POLICY count = {policy_sql_count}")
assert policy_sql_count == len(POLICIES), f"ASYMMETRY: json has {len(POLICIES)} policies, sql has {policy_sql_count} CREATE POLICYs"
print("  ✓ policy count matches")

print(f"  rls_state.json row_count = {len(RLS_STATE)}")
assert len(RLS_STATE) == 16, f"expected 16 public tables, got {len(RLS_STATE)}"
print("  ✓ 16 public tables")

alter_count = sum(1 for r in RLS_STATE if r["relrowsecurity"])
print(f"  ALTER TABLE ... ENABLE RLS count in SQL = {alter_count}")
print(f"  grants.json row_count = {len(GRANTS)}")
print(f"  columns.json row_count = {len(COLUMNS)}")
print(f"  indexes.json row_count = {len(INDEXES)}")
print(f"  roles.json row_count = {len(ROLES)}")

# Spot-check: pt_write should be anon INSERT with check=true
pt_write = next((p for p in POLICIES if p["tablename"] == "am_paper_trades" and p["policyname"] == "pt_write"), None)
assert pt_write and pt_write["cmd"] == "INSERT" and pt_write["roles"] == "{anon}" and pt_write["with_check"] == "true", f"pt_write shape wrong: {pt_write}"
print("  ✓ spot-check pt_write: anon INSERT with check=true")

# Spot-check: service_write_projections should be public ALL
swp = next((p for p in POLICIES if p["tablename"] == "am_projections" and p["policyname"] == "service_write_projections"), None)
assert swp and swp["cmd"] == "ALL" and swp["roles"] == "{public}", f"service_write_projections shape wrong: {swp}"
print("  ✓ spot-check service_write_projections: public ALL (the bug)")

# Spot-check: bets has RLS OFF
bets_rls = next(r for r in RLS_STATE if r["table"] == "bets")
assert not bets_rls["relrowsecurity"], "expected bets RLS OFF"
print("  ✓ spot-check bets: RLS OFF (matches audit)")

print("\nAll checks passed.")
