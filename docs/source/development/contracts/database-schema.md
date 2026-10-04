# Database Schema Contract

This page documents the SQLite schema used to persist screening runs. It is
generated from the actual implementation:

- schema bootstrap and compatibility updates —
  `src/molblender/persistence/store/schema_bootstrap.py`
- write and query facade — `src/molblender/persistence/backend.py`
- result queries — `src/molblender/persistence/store/results_query.py`
- HPO trial persistence — `src/molblender/persistence/hpo_trial_ops.py`
- JSON and folder import — `src/molblender/persistence/migration.py`

## Database location and entry points

The default database path is produced by `get_default_screening_db_path()` and is
`<output root>/screening/screening_results.db`. A run may override it with
`ScreeningConfig.db_path`.

`ScreeningResultsDB` is the write and query facade. Opening it initializes the
schema through `initialize_schema()`, which is idempotent: every statement is
either `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`, or an
`ALTER TABLE` guarded by a `PRAGMA table_info` check. Opening an existing
database therefore never destroys data.

## Tables

### `screening_sessions`

One row per screening run. `session_id` is the primary key.

| Column | Type | Default | Notes |
| --- | --- | --- | --- |
| `session_id` | TEXT | — | Primary key |
| `timestamp` | TEXT | — | Run timestamp, indexed descending |
| `task_type` | TEXT | — | Task family |
| `primary_metric` | TEXT | — | Metric name used for ranking |
| `dataset_name` | TEXT | `NULL` | Dataset label |
| `dataset_csv_path` | TEXT | `NULL` | Source CSV |
| `dataset_size` | INTEGER | `NULL` | Sample count |
| `cv_folds` | INTEGER | `3` | Configured folds |
| `test_size` | REAL | `0.2` | Test fraction |
| `random_state` | INTEGER | `42` | Seed |
| `resolved_workers` | INTEGER | `-1` | Resolved worker count; backfilled from `n_jobs` when added |
| `n_jobs` | INTEGER | `-1` | Legacy parallel field |
| `train_indices` / `test_indices` / `val_indices` | TEXT | `NULL` | JSON index arrays |
| `success` | BOOLEAN | `TRUE` | Run outcome |
| `n_models_evaluated`, `n_representations`, `n_unique_models` | INTEGER | `0` | Summary counters |
| `best_score`, `mean_score`, `std_score` | REAL | `0.0` | Summary statistics |
| `created_at`, `updated_at` | TIMESTAMP | `CURRENT_TIMESTAMP` | Row timestamps |
| `session_metadata` | TEXT | `NULL` | JSON metadata |

Columns added by later migrations: `cache_dir`, `split_column`, `input_column`,
`split_fingerprint`, `cohort_fingerprint`, `data_fingerprint`,
`identity_status` (default `pending`), `session_kind` (default `screening`),
`merge_provenance`.

### `model_results`

One row per evaluated model/representation combination.

| Column | Type | Default | Notes |
| --- | --- | --- | --- |
| `id` | INTEGER | autoincrement | Primary key |
| `session_id` | TEXT | — | Foreign key to `screening_sessions` |
| `model_name` | TEXT | — | Model identifier |
| `representation_name` | TEXT | — | Representation identifier |
| `representation_config`, `model_config` | TEXT | — | JSON configuration |
| `primary_metric` | REAL | `NULL` | Selection score; backfilled from the legacy `score` column |
| `rank` | INTEGER | `NULL` | Ranking position |
| `hpo_score` | REAL | `NULL` | Migrated from the legacy `hpo_cv_score` |
| `cv_fold_scores` | TEXT | `NULL` | JSON fold scores; migrated from the legacy `cv_scores` |
| `training_time` | REAL | `0.0` | Seconds |
| `n_features` | INTEGER | `0` | Feature count |
| `model_params` | TEXT | `NULL` | JSON parameters |
| `predictions` | TEXT | `NULL` | JSON predictions |
| `feature_importance` | TEXT | `NULL` | JSON importances |
| `model_artifact` | BLOB | `NULL` | Serialized model, only when `save_models` is enabled |
| `stage` | INTEGER | `1` | `1` for Stage 1, `2` for HPO |
| `hpo_stage`, `hpo_method` | TEXT | `NULL` / `grid` | HPO provenance |
| `best_params` | TEXT | `NULL` | JSON best parameter set |
| `primary_metric_name` | TEXT | `NULL` | Metric name behind `primary_metric` |
| `train_indices`, `test_indices`, `val_indices` | TEXT | `NULL` | JSON index arrays |
| `all_metrics` | TEXT | `NULL` | JSON mapping of every metric, including scoped keys |
| `grid_search_results` | TEXT | `NULL` | JSON search history; migrated from the legacy `all_cv_results` |
| `result_identity_key` | TEXT | `NULL` | Identity digest for reuse and deduplication |
| `evaluation_status` | TEXT | `NULL` | Outcome marker |
| `created_at` | TIMESTAMP | `CURRENT_TIMESTAMP` | Row timestamp |

Columns added by later migrations: `stage1_compatibility_key`,
`source_session_id`, `source_compatibility_key`.

### `dataset_info`

One row per session, holding the cohort payload used for downstream analysis.
`session_id` is both primary key and foreign key.

| Column | Type | Notes |
| --- | --- | --- |
| `session_id` | TEXT | Primary key, foreign key to `screening_sessions` |
| `target_column` | TEXT | Target column name |
| `dataset_n_train_samples`, `dataset_n_test_samples` | INTEGER | Cohort sizes |
| `test_true_values`, `train_true_values` | TEXT | JSON label arrays (required) |
| `test_input_data`, `train_input_data` | TEXT | JSON input arrays |
| `test_smiles_data`, `train_smiles_data` | TEXT | JSON SMILES arrays |
| `val_true_values`, `val_input_data`, `val_smiles_data` | TEXT | Validation cohort payloads |
| `train_sample_indices`, `test_sample_indices`, `val_sample_indices` | TEXT | JSON index arrays |
| `val_indices` | TEXT | JSON validation indices |

### `hpo_trials`

One row per evaluated HPO trial, written in real time during the search. Unlike
`model_results`, which stores the final best parameter set with full prediction
metrics, `hpo_trials` stores lightweight cross-validation-only trial records, so
a search can resume from the main screening database and a dashboard can show
intermediate progress without reading `optuna_storage.db`.

| Column | Type | Default | Notes |
| --- | --- | --- | --- |
| `id` | INTEGER | autoincrement | Primary key |
| `session_id` | TEXT | — | Foreign key to `screening_sessions` |
| `model_name`, `representation_name` | TEXT | — | Trial target |
| `hpo_stage` | TEXT | `NULL` | Stage label |
| `hpo_method` | TEXT | — | Search method |
| `trial_number` | INTEGER | — | Trial index |
| `trial_status` | TEXT | — | Status string |
| `params` | TEXT | — | JSON parameter set |
| `cv_scorer_score`, `cv_natural_score` | REAL | `NULL` | Score in scorer orientation and natural orientation |
| `cv_fold_scores` | TEXT | `NULL` | JSON fold scores |
| `cv_std` | REAL | `NULL` | Fold standard deviation |
| `training_time` | REAL | `NULL` | Seconds |
| `is_best` | INTEGER | `0` | Marks the winning trial |
| `split_fingerprint`, `cohort_fingerprint` | TEXT | `NULL` | Provenance fingerprints |
| `train_fold_scores`, `fold_indices`, `fold_score_source`, `fold_classification_diags` | TEXT | `NULL` | Fold diagnostics |
| `overfit_gap` | TEXT | `NULL` | JSON per-fold train-validation gaps |
| `overfit_gap_mean`, `overfit_gap_std` | REAL | `NULL` | Gap summary |
| `fold_train_sizes`, `fold_validation_sizes` | TEXT | `NULL` | JSON fold sizes |
| `created_at` | TIMESTAMP | `CURRENT_TIMESTAMP` | Row timestamp |

### `stage1_result_links`

Cross-session Stage 1 reuse links.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | INTEGER | Primary key |
| `source_result_id` | INTEGER | Foreign key to `model_results(id)`, `ON DELETE CASCADE` |
| `target_session_id` | TEXT | Foreign key to `screening_sessions(session_id)`, `ON DELETE CASCADE` |
| `linked_at` | TIMESTAMP | Link timestamp |

`UNIQUE (source_result_id, target_session_id)`.

## Indexes

| Index | Table | Columns | Kind |
| --- | --- | --- | --- |
| `idx_model_results_session` | `model_results` | `session_id` | index |
| `idx_model_results_score` | `model_results` | `primary_metric DESC` | index |
| `idx_model_results_rank` | `model_results` | `rank` | index |
| `idx_model_results_session_stage` | `model_results` | `session_id, stage` | index |
| `idx_model_results_session_score` | `model_results` | `session_id, primary_metric DESC` | index |
| `idx_model_results_compat_key` | `model_results` | `stage1_compatibility_key` | index |
| `idx_model_results_stage1_effective` | `model_results` | `session_id, stage, representation_name, model_name, stage1_compatibility_key` | index |
| `idx_model_results_identity_unique` | `model_results` | `result_identity_key` | **unique, partial** (`WHERE result_identity_key IS NOT NULL`) |
| `idx_sessions_timestamp` | `screening_sessions` | `timestamp DESC` | index |
| `idx_stage1_links_target` | `stage1_result_links` | `target_session_id` | index |
| `idx_stage1_links_source` | `stage1_result_links` | `source_result_id` | index |
| `idx_stage1_links_target_source` | `stage1_result_links` | `target_session_id, source_result_id` | index |
| `idx_hpo_trials_trial_identity` | `hpo_trials` | `session_id, model_name, representation_name, hpo_stage, hpo_method, trial_number, split_fingerprint` | unique |
| `idx_hpo_trials_lookup` | `hpo_trials` | `session_id, model_name, representation_name, hpo_stage, hpo_method, split_fingerprint` | index |

The partial unique index on `result_identity_key` is deliberate: it is limited to
non-NULL values so legacy rows without an identity digest can still be read and
upgraded lazily instead of being rejected on write.

## Constraints

- `model_results.session_id` and `dataset_info.session_id` reference
  `screening_sessions(session_id)`; `hpo_trials.session_id` does the same.
- `stage1_result_links` cascades on delete from both sides.
- `hpo_trials` enforces a seven-column trial identity so `INSERT OR REPLACE`
  cannot degenerate into a plain insert.
- Nullable key columns in the `hpo_trials` identity are normalized to `''`
  before writing. `NULL = NULL` is unknown under a `UNIQUE` constraint, so
  writing raw `NULL` would make the copied row invisible to later identity
  matching.

## Migration and version policy

There is no numeric schema version. Compatibility is expressed as idempotent,
additive steps, all of which are safe to run repeatedly:

1. Create missing tables and indexes with `IF NOT EXISTS`.
2. Add missing columns with `ALTER TABLE`, each guarded by `PRAGMA table_info`.
3. Backfill renamed legacy columns:
   - `hpo_cv_score` → `hpo_score`
   - `all_cv_results` → `grid_search_results`
   - `cv_scores` → `cv_fold_scores`
4. Backfill `primary_metric` from the legacy `score` column when it is `NULL`,
   and default `primary_metric_name` to `pearson_r` when it is `NULL`.
5. Add the `hpo_trials` fold-diagnostic columns by `ALTER TABLE`.
6. Rebuild `hpo_trials` when the seven-column `UNIQUE` constraint is missing.

Two ordering rules are load-bearing:

- The `hpo_trials` deduplication migration must run **before**
  `idx_hpo_trials_trial_identity` is created. Creating the unique index on a
  legacy table that still contains duplicate rows raises `IntegrityError` and
  would brick the bootstrap.
- The legacy `idx_hpo_trials_session` four-column index is dropped before the
  rebuild because it conflicts with the new identity key.

The `hpo_trials` rebuild keeps exactly one row per identity key. It partitions by
`COALESCE(column, '')` over the seven key columns, orders by `created_at`
descending with `id DESC` as a tie-break, and uses a window function rather than
`MAX(created_at)` so that two rows written in the same second cannot both
survive and violate the new constraint.

## Serialization conventions

Every structured payload is stored as JSON text, never as a native SQLite type:

- index arrays — `train_indices`, `test_indices`, `val_indices`,
  `*sample_indices`, `fold_indices`
- metrics — `all_metrics`, `cv_fold_scores`, `train_fold_scores`,
  `overfit_gap`, `grid_search_results`
- configuration and parameters — `representation_config`, `model_config`,
  `model_params`, `best_params`, `params`, `session_metadata`
- dataset payloads — `*_true_values`, `*_input_data`, `*_smiles_data`

`model_artifact` is the only binary column and is populated only when
`ScreeningConfig.save_models` is enabled.

## Read and write entry points

| Concern | Entry point |
| --- | --- |
| Session and result writes | `ScreeningResultsDB.create_session`, `add_model_result`, `persist_screening_result`, `update_session_summary` |
| Session reads | `ScreeningResultsDB.get_session_results`, `list_sessions`, `load_comprehensive_results`, `get_all_database_results` |
| Single-session result record | `get_session_results_record` in `store/results_query.py` |
| Dataset payload | `ScreeningResultsDB.save_dataset_info`, `load_dataset_input_smiles_for_session` |
| HPO trials | `save_hpo_trial`, `load_hpo_trials`, `count_hpo_trials`, `mark_best_hpo_trial` |
| Export | `ScreeningResultsDB.export_to_json` |
| Statistics and deletion | `ScreeningResultsDB.get_database_stats`, `delete_session` |
| Import from JSON or a results folder | `migrate_json_to_sqlite`, `migrate_folder_to_sqlite`, `list_database_sessions` |

The JSON and folder importers fail closed: a record that does not validate is
rejected rather than partially inserted, and importing an existing session is
only accepted when the proposed snapshot is equivalent to the stored one.

## Compatibility notes

- Legacy columns are kept and still read; new columns are additive.
- Rows written before an identity digest existed have `result_identity_key IS
  NULL` and are unaffected by the partial unique index.
- `resolved_workers` supersedes `n_jobs`; when the column is added, existing
  rows are backfilled from `n_jobs`.
- `primary_metric` supersedes the legacy `score` column, and the backfill runs on
  every bootstrap until no `NULL` values remain.

## Related pages

- {doc}`configuration` — the fields that populate `screening_sessions`.
- {doc}`metrics-and-results` — the JSON shape stored in `all_metrics`.
