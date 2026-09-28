-- migrations/022_drop_strategy_nodes_superseded_by.sql
-- DP-level patch unification made the per-cluster supersede model obsolete: patches
-- now coexist as DP-keyed Milvus rows and every strategy_node stays active=TRUE.
-- strategy_nodes.superseded_by is therefore always NULL and read by nothing — drop it.
-- (weight_overlay.superseded_by is a separate concern and is left intact.)
ALTER TABLE strategy_nodes DROP COLUMN IF EXISTS superseded_by;
