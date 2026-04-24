-- EdgeFinder RLS policy rollback — captured 2026-04-24T00:35:39Z
-- Source: pg_policies WHERE schemaname = 'public'
-- ----------------------------------------------------------------------
-- To restore: run this file against the database as a role that can CREATE POLICY.
-- First drop any conflicting policies by name, or wrap in BEGIN; DROP POLICY ...; etc.
-- ----------------------------------------------------------------------


-- ==== am_backtest_results ====
  CREATE POLICY "am_read_bt_res"
  ON public.am_backtest_results
  AS PERMISSIVE
  FOR SELECT
  TO anon
  USING (true);

  CREATE POLICY "am_write_bt_res"
  ON public.am_backtest_results
  AS PERMISSIVE
  FOR INSERT
  TO anon
  WITH CHECK (true);

  CREATE POLICY "anon_read_backtest_results"
  ON public.am_backtest_results
  AS PERMISSIVE
  FOR SELECT
  TO PUBLIC
  USING (true);

  CREATE POLICY "service_write_backtest_results"
  ON public.am_backtest_results
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== am_backtest_runs ====
  CREATE POLICY "am_read_bt_runs"
  ON public.am_backtest_runs
  AS PERMISSIVE
  FOR SELECT
  TO anon
  USING (true);

  CREATE POLICY "am_write_bt_runs"
  ON public.am_backtest_runs
  AS PERMISSIVE
  FOR INSERT
  TO anon
  WITH CHECK (true);

  CREATE POLICY "anon_read_backtest_runs"
  ON public.am_backtest_runs
  AS PERMISSIVE
  FOR SELECT
  TO PUBLIC
  USING (true);

  CREATE POLICY "service_write_backtest_runs"
  ON public.am_backtest_runs
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== am_game_logs ====
  CREATE POLICY "am_read_logs"
  ON public.am_game_logs
  AS PERMISSIVE
  FOR SELECT
  TO anon
  USING (true);

  CREATE POLICY "am_write_logs"
  ON public.am_game_logs
  AS PERMISSIVE
  FOR INSERT
  TO anon
  WITH CHECK (true);

  CREATE POLICY "anon_read_game_logs"
  ON public.am_game_logs
  AS PERMISSIVE
  FOR SELECT
  TO PUBLIC
  USING (true);

  CREATE POLICY "service_write_game_logs"
  ON public.am_game_logs
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== am_paper_trades ====
  CREATE POLICY "pt_read"
  ON public.am_paper_trades
  AS PERMISSIVE
  FOR SELECT
  TO anon
  USING (true);

  CREATE POLICY "pt_update"
  ON public.am_paper_trades
  AS PERMISSIVE
  FOR UPDATE
  TO anon
  USING (true);

  CREATE POLICY "pt_write"
  ON public.am_paper_trades
  AS PERMISSIVE
  FOR INSERT
  TO anon
  WITH CHECK (true);


-- ==== am_player_tracking ====
  CREATE POLICY "am_read_tracking"
  ON public.am_player_tracking
  AS PERMISSIVE
  FOR SELECT
  TO anon
  USING (true);

  CREATE POLICY "am_write_tracking"
  ON public.am_player_tracking
  AS PERMISSIVE
  FOR INSERT
  TO anon
  WITH CHECK (true);

  CREATE POLICY "anon_read_tracking"
  ON public.am_player_tracking
  AS PERMISSIVE
  FOR SELECT
  TO PUBLIC
  USING (true);

  CREATE POLICY "service_write_tracking"
  ON public.am_player_tracking
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== am_projections ====
  CREATE POLICY "am_delete_proj"
  ON public.am_projections
  AS PERMISSIVE
  FOR DELETE
  TO anon
  USING (true);

  CREATE POLICY "am_read_proj"
  ON public.am_projections
  AS PERMISSIVE
  FOR SELECT
  TO anon
  USING (true);

  CREATE POLICY "am_write_proj"
  ON public.am_projections
  AS PERMISSIVE
  FOR INSERT
  TO anon
  WITH CHECK (true);

  CREATE POLICY "anon_read_projections"
  ON public.am_projections
  AS PERMISSIVE
  FOR SELECT
  TO PUBLIC
  USING (true);

  CREATE POLICY "service_write_projections"
  ON public.am_projections
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== am_team_context ====
  CREATE POLICY "am_read_context"
  ON public.am_team_context
  AS PERMISSIVE
  FOR SELECT
  TO anon
  USING (true);

  CREATE POLICY "am_write_context"
  ON public.am_team_context
  AS PERMISSIVE
  FOR INSERT
  TO anon
  WITH CHECK (true);

  CREATE POLICY "anon_read_team_context"
  ON public.am_team_context
  AS PERMISSIVE
  FOR SELECT
  TO PUBLIC
  USING (true);

  CREATE POLICY "service_write_team_context"
  ON public.am_team_context
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== closing_prices ====
  CREATE POLICY "service_role_all"
  ON public.closing_prices
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== kv_store ====
  CREATE POLICY "service_role_all"
  ON public.kv_store
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== pipeline_runs ====
  CREATE POLICY "anon insert pipeline_runs"
  ON public.pipeline_runs
  AS PERMISSIVE
  FOR INSERT
  TO anon
  WITH CHECK (true);

  CREATE POLICY "anon read pipeline_runs"
  ON public.pipeline_runs
  AS PERMISSIVE
  FOR SELECT
  TO anon
  USING (true);


-- ==== rb_paper_trades ====
  CREATE POLICY "Allow all for service role"
  ON public.rb_paper_trades
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== rb_projections ====
  CREATE POLICY "Public read rb_projections"
  ON public.rb_projections
  AS PERMISSIVE
  FOR SELECT
  TO PUBLIC
  USING (true);

  CREATE POLICY "Service write rb_projections"
  ON public.rb_projections
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== server_runs ====
  CREATE POLICY "service_role_all"
  ON public.server_runs
  AS PERMISSIVE
  FOR ALL
  TO PUBLIC
  USING (true)
  WITH CHECK (true);


-- ==== ALTER TABLE … ENABLE ROW LEVEL SECURITY ====
ALTER TABLE public.am_backtest_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.am_backtest_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.am_game_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.am_paper_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.am_player_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.am_projections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.am_team_context ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.closing_prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.kv_store ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rb_paper_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rb_projections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.server_runs ENABLE ROW LEVEL SECURITY;

-- Tables with RLS OFF at capture time (DO NOT re-enable without policies):
--   public.bets
--   public.edges
--   public.player_games
