-- EdgeFinder Supabase RLS — Phase 2 DRAFT (not applied). Review, then run in the
-- SQL editor as one transaction. Project: jnirimzrhunjdtyvkhtt.
--
-- Prerequisites, in order:
--   1. Every client is on the new API keys. Browsers (both dashboards in this repo,
--      edgefinder-nba-pro, edgefinder.cognivaultlabs.com) use the publishable key,
--      which runs as role `anon`. The Cloudflare worker and this repo's Python
--      scripts use a secret key, which runs as `service_role` and BYPASSES RLS, so
--      none of the policies below affect them.
--   2. The legacy anon / service_role JWTs are disabled in Supabase ->
--      Settings -> API Keys (the service_role JWT was found in public bundles).
--   3. Smoke-test all three sites right after running this.
--
-- What it does:
--   A. Enables RLS on the three tables that still have it off.
--   B. Drops the 11 "TO PUBLIC ... FOR ALL" policies that currently let the public
--      key insert/update/delete. service_role never needed them.
--   C. Adds explicit `anon` policies for exactly what the browsers do today:
--        read : edges, bets, player_games, closing_prices, kv_store, server_runs,
--               rb_paper_trades (its only policy today is one of the ALL policies)
--        write: bets (the NBA app's tracker), kv_store rows whose key starts with
--               'user/' (the NBA app's settings / book balances)
--      Everything else stays read-only for anon or service-role-only.
--   Note: anyone holding the publishable key can still insert bets rows; that is
--   the NBA app's current design (client-side passcode only). Move bet writes
--   behind the worker later if that matters.

begin;

-- A. Enable RLS where it is still off
alter table public.bets         enable row level security;
alter table public.edges        enable row level security;
alter table public.player_games enable row level security;

-- B. Drop the TO PUBLIC ALL policies (Phase 2 targets from the 2026-04-24 receipt)
drop policy if exists "service_write_backtest_results" on public.am_backtest_results;
drop policy if exists "service_write_backtest_runs"    on public.am_backtest_runs;
drop policy if exists "service_write_game_logs"        on public.am_game_logs;
drop policy if exists "service_write_tracking"         on public.am_player_tracking;
drop policy if exists "service_write_projections"      on public.am_projections;
drop policy if exists "service_write_team_context"     on public.am_team_context;
drop policy if exists "service_role_all"               on public.closing_prices;
drop policy if exists "service_role_all"               on public.kv_store;
drop policy if exists "Allow all for service role"     on public.rb_paper_trades;
drop policy if exists "Service write rb_projections"   on public.rb_projections;
drop policy if exists "service_role_all"               on public.server_runs;

-- C. Explicit anon policies
create policy "anon_read_edges"          on public.edges          for select to anon using (true);
create policy "anon_read_player_games"   on public.player_games   for select to anon using (true);
create policy "anon_read_closing_prices" on public.closing_prices for select to anon using (true);
create policy "anon_read_server_runs"    on public.server_runs    for select to anon using (true);
create policy "rb_read"                  on public.rb_paper_trades for select to anon using (true);

create policy "anon_read_bets"   on public.bets for select to anon using (true);
create policy "anon_insert_bets" on public.bets for insert to anon with check (true);
create policy "anon_update_bets" on public.bets for update to anon using (true) with check (true);
create policy "anon_delete_bets" on public.bets for delete to anon using (true);

create policy "anon_read_kv_store"   on public.kv_store for select to anon using (true);
create policy "anon_insert_user_kv"  on public.kv_store for insert to anon with check (key like 'user/%');
create policy "anon_update_user_kv"  on public.kv_store for update to anon using (key like 'user/%') with check (key like 'user/%');

commit;

-- Verify afterwards: every table should show relrowsecurity = true and no policy
-- with roles = {public} and cmd = ALL should remain.
select c.relname, c.relrowsecurity
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public' and c.relkind = 'r' order by 1;

select tablename, policyname, cmd, roles::text
from pg_policies where schemaname = 'public' order by 1, 2;
