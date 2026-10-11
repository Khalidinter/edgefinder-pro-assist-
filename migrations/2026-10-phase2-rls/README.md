# Phase 2 RLS (draft, 2026-10-11)

Follow-up to `../2026-04-24-phase1-rls/`. Drafted after a 2026-10-07 audit found the
project's legacy `service_role` JWT embedded in two public front-end bundles
(edgefinder-nba-pro.vercel.app and edgefinder.cognivaultlabs.com).

Order of operations:

1. Create a secret key (`sb_secret_...`) in Supabase -> Settings -> API Keys and move the
   Cloudflare worker (`edgefinder-server`) and this repo's `.env` to it.
2. Rebuild the front-ends with the publishable key (`sb_publishable_...`). The two
   dashboards in `sites/` already use it.
3. Disable the legacy keys in Supabase -> Settings -> API Keys.
4. Run `policies.sql` in the SQL editor, then open each site and confirm data loads and
   the NBA app can still save a bet / settings.

Not applied automatically: the policies encode an assumption about what the browser
apps read and write (see the header of `policies.sql`). Adjust before running if the
NBA app writes other tables.
