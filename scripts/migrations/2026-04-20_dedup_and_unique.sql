-- 2026-04-20: Deduplicate paper-trade rows and add UNIQUE constraints.
--
-- WHY:
--   Before this migration, both GitHub Actions and the local scheduled task
--   ran save_predictions() against rb_paper_trades and am_paper_trades with
--   no unique constraint, so each day's predictions were inserted twice.
--   Dashboard aggregate metrics (PnL, hit rate, ROI) were inflated 2x starting
--   2026-04-17 on the rebound side.
--
-- WHAT:
--   Step 1 (destructive): delete duplicate rows, keeping the earliest by id.
--   Step 2: add UNIQUE(prediction_date, player_id) so future duplicates 409.
--   Step 3: (optional) verify row counts drop to expected values.
--
-- HOW TO APPLY:
--   Run block-by-block in the Supabase SQL editor. DO NOT run blindly — each
--   DELETE block prints the number of rows it removed; sanity-check before
--   moving to the UNIQUE constraint. The code-side upsert header shipped with
--   this change (Prefer: resolution=merge-duplicates) already prevents NEW
--   duplicates once the UNIQUE constraints exist; until then it no-ops.
--
-- ROLLBACK:
--   UNIQUE constraints can be dropped: ALTER TABLE ... DROP CONSTRAINT ...;
--   Deleted rows cannot be restored — take a snapshot before running Step 1.


-- ═══════════════════════════════════════════════════════════════════
-- STEP 0 (read-only): report current duplication before we touch anything
-- ═══════════════════════════════════════════════════════════════════
SELECT
    'rb_paper_trades' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT (prediction_date, player_id)) AS unique_keys,
    COUNT(*) - COUNT(DISTINCT (prediction_date, player_id)) AS duplicate_rows
FROM rb_paper_trades
UNION ALL
SELECT
    'am_paper_trades',
    COUNT(*),
    COUNT(DISTINCT (prediction_date, player_id)),
    COUNT(*) - COUNT(DISTINCT (prediction_date, player_id))
FROM am_paper_trades;


-- ═══════════════════════════════════════════════════════════════════
-- STEP 1 (destructive): delete duplicate rows, keep earliest id per key
-- ═══════════════════════════════════════════════════════════════════
-- Rebounds
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY prediction_date, player_id
               ORDER BY id ASC
           ) AS rn
    FROM rb_paper_trades
)
DELETE FROM rb_paper_trades
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);

-- Assists
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY prediction_date, player_id
               ORDER BY id ASC
           ) AS rn
    FROM am_paper_trades
)
DELETE FROM am_paper_trades
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);


-- ═══════════════════════════════════════════════════════════════════
-- STEP 2: add UNIQUE constraint so future duplicates are rejected (or
-- merged server-side by the upsert header the paper-trade scripts send).
-- ═══════════════════════════════════════════════════════════════════
ALTER TABLE rb_paper_trades
    ADD CONSTRAINT rb_paper_trades_date_player_uniq
    UNIQUE (prediction_date, player_id);

ALTER TABLE am_paper_trades
    ADD CONSTRAINT am_paper_trades_date_player_uniq
    UNIQUE (prediction_date, player_id);


-- ═══════════════════════════════════════════════════════════════════
-- STEP 3 (read-only): confirm duplicates are gone and constraints exist
-- ═══════════════════════════════════════════════════════════════════
SELECT
    'rb_paper_trades' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT (prediction_date, player_id)) AS unique_keys
FROM rb_paper_trades
UNION ALL
SELECT
    'am_paper_trades',
    COUNT(*),
    COUNT(DISTINCT (prediction_date, player_id))
FROM am_paper_trades;

SELECT conname, conrelid::regclass, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conname IN (
    'rb_paper_trades_date_player_uniq',
    'am_paper_trades_date_player_uniq'
);
