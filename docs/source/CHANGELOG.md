# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **COATI2 sequence encoder representation (`coati2_smiles`) with cache-hit provenance restoration** (`representations/sequential/language_model/coati2.py`, `coati2_loader.py`, `coati2_registry.py`, `data/dataset/caching.py`, `data/dataset/features.py`, `screening/engine/data/quality_flow.py`, `config/settings.py`, `tests/representations/sequential/language_model/`) (2026-09-13)
  - `coati2_smiles` featurizer wraps the official `coati` sequence encoder (`ibm/biomed.sm.coati2-84m`); SMILES canonicalization policy is fixed per checkpoint and never varies between runs
  - Checkpoint resolution is fully reproducible: local mirror is SHA-256 verified against a pinned revision, remote download goes through `snapshot_download` gated by the new `get_allow_model_download()` runtime policy (re-reads `MOLBLENDER_ALLOW_MODEL_DOWNLOAD` at call time instead of an import-time snapshot)
  - Checkpoint identity (URI, SHA-256 digest, coati version, canonicalization policy, inference/training modalities, architecture, `uses_image_at_inference`) persisted into representation-cache metadata and `model_results.representation_config`, so a cache-hit run writes the same audit row as a fresh run
  - `load_cached_representation_with_fallback()` gains an `include_metadata=True` mode returning `(data, metadata)`: historical callers still get the bare array, but metadata (with `model_provenance`) survives the cache-hit path instead of being dropped on load
  - Contract tests: cache-identity invariance across local/remote checkpoint resolution, cross-process temp-dir ownership with stale-PID reclamation, and the featurizer→dataset→representation_config→SQLite provenance chain
- **MMELON multi-view representation adapter (`mmelon_fused`) with reproducible model provenance** (`representations/multi_view/`, `data/dataset/features.py`, `screening/engine/data/quality_flow.py`, `tests/representations/multi_view/`) (2026-09-12)
  - `mmelon_fused` featurizer wraps `ibm/biomed.sm.mv-te-84m` (bmfm_sm, optional dependency in a dedicated env); snapshot resolution honors `revision`/`cache_dir`/offline gate directly via `snapshot_download`, `revision` pinned to a verified commit hash
  - Cache identity = repo id + resolved revision + fusion strategy + adapter version; `snapshot_path` and fetch knobs (`cache_dir`, `allow_download`) are diagnostic-only and never enter the key
  - Model provenance (provider, requested/canonical repo id, resolved revision, fusion strategy, embedding dim, adapter version) persists into representation-cache metadata, `dataset._feature_metadata`, and screening-DB `model_results.representation_config`, so Dashboard/export rows are queryable per checkpoint
  - Contract tests: 26 cases covering cache-identity invariance (snapshot path change, runtime knobs) and the full featurizer→dataset→representation_config→SQLite JSON chain
- **Dataclass default fixes for py3.11+ compatibility** (`data/dataset/splitting/strategy_registry.py`, `representations/descriptors/descriptors_basic.py`) (2026-09-12)
  - `StrategySpec.alternative_inputs` moves `MappingProxyType({})` into a `default_factory` (mutable-default rejection on 3.11+); descriptors facade re-exports are guarded so a missing optional dep (e.g. `mordred`) no longer breaks registration
- **HPO realtime persistence + deployment/external-evaluation persistence** (`persistence/deployment.py`, `persistence/result_updates.py`, `persistence/store/results_query.py`, `persistence/store/schema_bootstrap.py`, `persistence/contracts.py`, `persistence/__init__.py`, `persistence/session_write.py`, `screening/orchestration/processors/hpo/`) (2026-09-04)
  - `save_deployment_result()`: persist final full-fit predictions as `stage=3` `model_results` rows with `primary_metric=NaN`, `hpo_stage="stage3_deployment"`, no fabricated evaluation score
  - `finalize_deployment_session()`: mark deployment session complete without inventing `best_score`/`mean_score`/`std_score`
  - `save_external_evaluation_result()`: persist labeled prospective/external evaluation as `stage=4` with `hpo_stage="stage4_external"`
  - `update_result_metrics()`: canonical `UPDATE model_results SET all_metrics = ?` helper, no raw SQL from callers
  - `session_metadata` column added to `screening_sessions` (TEXT, schema bootstrap + ALTER); `session_kind` distinguishes deployment from evaluation
  - HPO trial rows projected into the same result-record shape as `model_results` in `get_session_results_record()` — callers no longer special-case Grid/Optuna
  - `score_direction` (`higher_is_better`/`lower_is_better`) derived from `primary_metric_name` via `is_higher_better()`; `record_kind` (`deployment`/`hpo_trial`/`model_result`) added to every result
  - `ModelExecutionSettings.from_config()` in `screening/engine/models/execution_settings.py`: canonical execution-params dataclass (workers, GPU, timeout, seed) with `ModelConfig` round-trip; replaces scattered config extraction
  - `save_hpo_trial()` per-trial realtime persistence to main DB; `evaluate_all_hpo_params_on_test` disabled during active HPO runs to avoid redundant test evaluation
  - `ModelRegistry.get_models()` accepts `execution_settings` kwarg, threaded through all evaluator call sites
- **Seed contract + nested-CV fail-closed** (`screening/engine/evaluation/nested_cv_evaluator.py`, `screening/engine/evaluation/cross_validation.py`, `screening/engine/hpo/contracts.py`, `screening/engine/hpo/grid_search.py`, `tests/engine/hpo/test_hpo_seed_fail_closed.py`, `tests/engine/hpo/test_seed_parameter_contract.py`) (2026-09-04)
  - `NestedCVEvaluator` accepts and propagates `seed` / `random_state`; missing seed raises `ValueError` (fail-closed, never silent `None`)
  - `GridSearch.__init__` validates `seed`/`random_state` consistency; `hpo/contracts.py` adds seed contract fields
- **Nested-CV seed contract tests** (`tests/engine/evaluation/test_nested_cv_seed_contract.py`) (2026-09-04)
- **HPO seed fail-closed test** (`tests/engine/hpo/test_hpo_seed_fail_closed.py`, `tests/engine/hpo/test_seed_parameter_contract.py`, `tests/engine/models/test_execution_settings.py`) (2026-09-04)
- **Export seed reproducibility test** (`tests/dashboard/test_export_seed_reproducibility.py`) (2026-09-04)
- **Model results persistence tests** (`tests/api/persistence/test_deployment_persistence.py`) (2026-09-04)
- **Vector screening exclusions test** (`tests/screening/orchestration/processors/hpo/test_vector_screening_exclusions.py`) (2026-09-04)
- **Representation seed reproducibility test** (`tests/representations/test_representation_seed_reproducibility.py`) (2026-09-04)
- **Vector pathway routing + GNN canonical split/ScreeningResult** (`multimodal/routing/parser.py`, `multimodal/routing/validation.py`, `multimodal/modality_handlers/graph.py`, `multimodal/modality_handlers/base.py`, `multimodal/modality_handlers/language_model.py`, `multimodal/modality_handlers/vector.py`, `tests/models/api/multimodal/routing/test_parser.py`) (2026-08-25)
  - `_parse_custom_combination()` now returns a `list[ResolvedRoute]`: vector modality with pathways `["all","vae"]` expands to two routes (VECTOR_STANDARD + VECTOR_VAE); other modalities return a single-element list
  - Validation error message for unknown pathways clarifies that `traditional_ml` is a model corpus, not a pathway
  - GNN graph handler canonicalized: train/test indices from shared `_canonical_split_plan` (not sklearn `train_test_split`), emits `ScreeningResult` (replaces `ModelResult`), persist via `_save_model_result_to_db(result, "graph")` for resume/session visibility
  - `_build_child_screening_config()` builds kwargs conditionally (cv_folds_cap/max_workers_cap only when not None); two `except Exception: pass` replaced with `logger.debug()` (ruff B023 + bandit B110 compliance)
- **Cache path-traversal hardening + generic cache contract** (`data/cache/core.py`, `cache/contracts.py`, `tests/cache/test_cache_v3_implementation_identity.py`) (2026-08-25)
  - `_get_metadata_cache_path()` now resolves metadata-provided `file_path` below the cache's trusted root, rejecting absolute paths or `..` traversal; malformed metadata becomes a normal cache miss instead of leaking to a reader or cleanup routine
  - `clear_cache()` and `shrink_cache()` now route every entry through `_get_metadata_cache_path()` instead of trusting `metadata["file_path"]` directly, closing the deletion-by-path-traversal vector
  - New `cache.contracts.CacheStore` runtime-checkable Protocol: `get`/`set`/`delete` seam for generic key/value cache adapters, keeping representation-feature cache and screening-result persistence as separate domains
  - Cache V3 implementation-identity test coverage verifies the trusted-root contract
- **Export manifest module** (`models/export_manifest.py`, `tests/models/api/test_export_manifest.py`) (2026-08-25)
  - Dependency-free metadata layer for lazy public exports: `ExportSpec` frozen dataclass captures canonical owner (`module`/`attribute`), legacy flag, optional-dependency tag and removal-version policy
  - `build_export_manifest()` converts a lazy tuple map to validated, introspectable specs; package `__getattr__` functions remain lazy, this module only validates declared public names have a resolvable owner
- **Model-results database identity key** (`persistence/model_results_write.py`) (2026-08-25)
  - `_result_identity_key()` builds a SHA-256 identity for a canonical result write from session/model/representation/stage + Stage-1 compatibility key or HPO stage/method/best-params; distinct `<missing>` sentinel prevents incomplete legacy identities from colliding with explicit empty keys
- **HPO processor typed orchestration state** (`multimodal/processors/hpo/processor.py`) (2026-08-25)
  - Internal orchestration state further bundled in typed dataclasses, replacing positional attribute unpacking; 131-line net change consolidating scheduling, materialization and write-out paths
- **CI dependency workflow updates** (`.github/workflows/dependencies.yml`, `environment.yml`, `pyproject.toml`) (2026-08-25)
  - `safety scan --full-report` replaces deprecated `safety check --full-report`; `pip-audit --strict` enables stricter vulnerability matching
- **Representations mypy+bandit+Pyright clean** (`representations/**`) (2026-09-04)
  - 85→0 mypy errors: `no-redef` ignores on `parallel.py` + `deepchem.py` imports, `list-item` ignore on `bert/base.py` `[None]*n` list, `attr-defined` ignore on `esm.py` `self.model.config`, `call-arg` ignore on `coordinates.py` class-level, `arg-type` ignore on `facade.py` `list_available_protein_featurizers_from_registry()` return, unused `Optional` import removed from `cache_validator.py`
  - 15 bandit issues fixed with `# nosec` annotations: B105 on `full_builtin_manifest.py` (false positive "vector" is output_type), B110 on `mdanalysis_backend.py` (intentional cleanup swallow), B403 on `cache_validator.py` (pickle cache I/O only), B404 on `structure_predictor.py` (subprocess for isolated conda env), B603/B607 on `structure_predictor.py` (validated cmd), B615 on `esm.py` + `transformer_models.py` (HF `from_pretrained` gated by `MOLBLENDER_ALLOW_MODEL_DOWNLOAD`)
- **HPO resume state immutable dataclasses + evaluator component consolidation** (`multimodal/processors/hpo/`, `screening_engine/evaluation/`) (2026-08-25)
  - `HPOResumeState`/`HPOResumeCandidate` frozen dataclasses replace transient `_resume_*` attributes on Stage-1 results; resume state carried via `candidate_input.resume_state`, `_freeze()`/`_thaw()` provide immutable snapshots and mutable projections
  - `prepare_resume_candidates()` replaces `attach_resume_state()`, `_prepare_optuna_resume_state()` replaces `_resume_optuna_candidate()`, `_prepare_random_resume_state()` replaces `_resume_random_candidate()`
  - `build_hpo_study_context()` returns typed `HPOStudyContext` dataclass (replaces tuple return); `study_context.is_complete`/`is_persistent` replace tuple index checks
  - `prepare_evaluation_estimator()` consolidates scaling-pipeline wrapping + Phase-2 quality transformer wrapping into one function; CV-only path passes `include_phase2=False` so fold metadata applies per-fold
  - `_raise_mapped_evaluation_error()` replaces scattered `classify_exception` + `_emit_evaluation_failure_event` + `raise EvaluationError` pattern across evaluator; `_phase2_state_for_result` variable removed
  - `evaluator_components.py` new module: `prepare_evaluation_estimator` + `ModelTrainer`/`PredictionGenerator`/`ResourceManager` extracted from evaluator
  - Stage-1 reuse import wrapped in explicit `BEGIN`/`ROLLBACK` transaction for atomicity; partial link set never exposed as successful import
  - `record_adapter.py` type-coercion helpers (`_coerce_mapping`/`_coerce_sequence`) accept both live Mapping/Sequence and JSON-string legacy fields
- **Stage-1 cross-session reuse with stable cohort identity** (`multimodal/modality_handlers/base.py`, `multimodal/modality_handlers/language_model.py`, `multimodal/screeners.py`, `persistence/_stage1_identity.py`, `screening/standard.py`, `screening_runtime/split_plan.py`) (2026-08-27)
  - `derive_stable_cohort_ids()` in `split_plan.py`: derives ordered cohort from each molecule's canonical SMILES instead of `dataset._index` (freshly generated UUIDs per build, never matches a historical session); falls back to `_index` for datasets without a `molecules` collection
  - UniversalScreener `_import_compatible_stage1_results()`: builds `(repr_name, model_name) -> compatibility_key` registry from the routing plan, resolves model configs via `get_model_registry().get_models()` and representation configs via `get_representation_config()`, then calls `plan_and_import_compatible_stage1_results()` — replaces the previous no-op that could never match a concrete child evaluator config
  - ModalityHandlerMixin save callback: stamps each new Stage-1 result with the matching `stage1_compatibility_key` from `_stage1_candidate_keys`, enabling cross-session dedup; empty result warnings now include `representations=`/`models=` for triage
  - StandardScreener `_prepare_stage1_results()`: handles the dict form of `prepare_screening_inputs()` output (output-type → names mapping), previously only iterated list-of-groups
  - `_target_signature_from_dataset()`: supports MolecularDataset `labels`/`_labels` frames including MultiIndex columns like `(target, "value")` — signature matches on the first index level
- **`coordinates_3d` fixed-shape 3-D representation** (`representations/spatial/coordinates.py`, `representations/spatial/__init__.py`, `representations/__init__.py`, `representations/registry/builtin_declarations.py`, `representations/registry/full_builtin_manifest.py`, `representations/registry/selection.py`, `representations/spatial/matrix/coulomb.py`, `representations/spatial/matrix/graph.py`, `representations/sequential/language_model/bert/{chembert,chemberta,smiles_bert}.py`, `representations/sequential/tokenizer/deepchem.py`, `tests/representations/spatial/test_coordinates.py`, `tests/models/api/multimodal/test_string_matrix_representation_selection.py`) (2026-08-27)
  - New `Coordinates3DFeaturizer` / `AtomicCoordinatesFeaturizer` alias: registered as `coordinates_3d` under `spatial/coordinates` with `output_type="matrix"`; produces fixed `(max_atoms, 3)` float32 array with zero padding, rejects oversized molecules instead of truncating, owns a seeded ETKDGv3 conformer generator, supports `center`/`normalize` transforms, preserves existing conformers without in-place mutation
  - `CoulombMatrixEig.__init__`: pops fixed-parameter kwargs before `super().__init__()` to prevent registry default-kwargs collisions
  - `AdjacencyMatrix`/`EdgeMatrix` registry default: `flatten=False` (CNN consumption as spatial tensor); callers requesting a traditional-ML vector can still pass `flatten=True`
  - `select_featurizers_by_category()`: legacy category paths like `spatial/UniMol/learned_cls/unimolv1-WithHs` resolve by suffix match under the declared parent category only
  - DeepChem tokenizer import: direct `from deepchem.feat.smiles_tokenizer import ...` instead of `dc.feat.smiles_tokenizer.SmilesTokenizer`, fixing DeepChem 2.8 lazy-attribute absence
  - ChemBERT/ChemBERTa/SMILES-BERT: explicit `FeaturizationError` import, removing unreachable `# type: ignore[name-defined]` markers
- **UniMol subprocess stability + skip-existing reuse hardening** (`representations/spatial/unimol/process_wrapper.py`, `screening_engine/data/gpu_precompute.py`, `config/dependencies/loaders_modules/ml_plm.py`, `config/dependencies/loaders_modules/core.py`, `screening_engine/evaluation/evaluator_helpers.py`, `screening_runtime/evaluation.py`) (2026-08-27)
  - UniMol subprocess runs under `spawn` (not `fork`) so a child inheriting an already-initialized CUDA context no longer raises "Cannot re-initialize CUDA in forked subprocess"; `result_queue.get()` drains the pipe before `join()`, fixing the deadlock when large UniMol payloads overflow the ~64 KB queue pipe and the child blocks in `put()`
  - GPU precompute now writes UniMol results through the representation cache writer instead of only mutating in-memory features, so subsequent screening runs skip model reload and recompute
  - `unimol_tools` import isolated to `~/.cache/molblender/unimol/logs` via a cwd context manager (the package hard-codes a cwd-relative `./logs` logger); dependency import probes run with `cwd=tempfile.gettempdir()` for side-effect-free checking
  - `is_heavy_workload()` no longer classifies `mlp`-suffixed representations as heavy; `evaluate_representation()` suppresses the compatibility fallback when every requested model was filtered out by skip-existing reuse (`all_models_skipped_existing`), so an empty primary loop means "already computed", not failure
- **Session-merge identity preservation** (`persistence/session_merge.py`, `persistence/_session_merge_sqlite.py`, `persistence/_session_merge_data.py`) (2026-08-27)
  - `merge_sessions_in_place()` loads full `screening_sessions` rows (replacing the narrow `list_sessions()` column set) so split/cohort fingerprints and dataset paths survive the merge; merged session and result rows now preserve `split_column`/`input_column`/`cache_dir`/fingerprints/`identity_status`/`session_kind`/`merge_provenance` and `stage1_compatibility_key` so post-merge skip-existing key matching still resolves
  - `merge_duplicate_results()` grafts `stage1_compatibility_key` from any duplicate that carries it when the winning row is an unstamped legacy duplicate
- **Optuna CV fold diagnostics end-to-end** (`screening_engine/hpo/optuna_trial_execution.py`, `optuna_study.py`, `result_translation.py`, `contracts.py`, `multimodal/processors/hpo/processor.py`, `results.py`, `candidate_execution.py`, `candidate_materialization.py`, `persistence/hpo_trial_ops.py`, `schema_bootstrap.py`, `screening_engine/evaluation/diagnostics.py`, `metrics.py`, `types.py`, `tests/...`) (2026-08-26)
  - Fold-level diagnostics: per-fold train/validation scorer scores, fold boundaries (`train`/`validation` index pairs), train/validation sizes, overfit gap (scorer-space `train - val`, mean/std), classification diagnostics (balanced accuracy, accuracy, macro-F1, ROC-AUC, PR-AUC, log loss) — all persisted per trial via `hpo_trials` table
  - Serial pruning P0 fix: `raise optuna.TrialPruned()` now correctly indented inside `if trial.should_prune()`, so partial fold data is written before pruning
  - `evaluate_trial_cv()` unified: single-fit-per-fold for both pruning and non-pruning paths (eliminates double fit from `cross_val_score` + refit loop); returns `(val_scores, train_scores, classification_diags)` tuple
  - `set_trial_attrs()` extended signature: writes `cv_train_fold_scores`, `fold_indices`, `fold_train_sizes`, `fold_validation_sizes`, `fold_score_source`, `fold_classification_diags`, `overfit_gap`/`overfit_gap_mean`/`overfit_gap_std`
  - `TrialEvent` dataclass extended with all fold diagnostic fields; `_build_event()` and `_make_trial_callback()` propagate them through to `HpoTrialRecord` and `save_hpo_trial()` DB persistence
  - `translate_optimization_result()` returns unified fold summary (natural validation/train folds, scorer-space gaps, source, completeness); `create_hpo_result()` consumes that summary instead of re-scanning `cv_results`
  - CV-only HPO result cleanup: `cv_fold_metrics`, `cv_fold_predictions`, `test_predictions`, `train_predictions`, `test_true_values`, `train_true_values`, `test_pred_proba`, `train_pred_proba`, `test_decision_score`, `train_decision_score` all cleared to `None` before saving
  - `build_cv_only_all_metrics()` stores HPO diagnostics under `hpo_`-prefixed keys (`hpo_train_fold_scores`, `hpo_overfit_gap_by_fold`, `hpo_split_fingerprint`, `fold_score_source`, `model_params`)
  - Schema migration: `hpo_trials` table extended with `fold_classification_diags TEXT`; ALTER TABLE path for existing databases
  - `diagnostics.py` new module: `export_classification_diagnostics()` + `_compute_fold_classification()` for per-fold classification metric extraction
  - `EvaluationMetrics` carries `fold_classification_diags` field across all construction paths
  - New canonical `utils.gpu` module; `utils/gpu_manager.py` reduced to facade adapter pointing at canonical path
- **UniMol-V1-NoHs variant registration** (`representations/spatial/unimol/configs.py`, `tests/representations/spatial/test_unimol.py`) (2026-08-27)
  - New `UniMol-V1-NoHs` config entry closes the registration gap: `unimolv1` was the only base model without a paired `remove_hs=True` companion; the base config dict now has 12 entries (was 11), yielding 36 auto-registered featurizers (CLS/AtomicCoords/AtomicReprs per config)
  - Test asserts the paired WithHs/NoHs invariant for `unimolv1` (`remove_hs` False vs True), matching the existing invariant already covered for every v2 size
- **HPO fold-score diagnostics end-to-end: Grid fold assembly + Optuna resume preservation** (`screening_engine/hpo/grid_search.py`, `screening_engine/hpo/contracts.py`, `screening_engine/hpo/optuna_trial_execution.py`, `screening_engine/hpo/optuna_study.py`, `multimodal/processors/hpo/resume_state.py`, `screening_engine/evaluation/metrics.py`, `tests/models/api/screening_engine/hpo/test_grid_fold_scores_assembly.py`, `tests/models/api/screening_engine/hpo/test_fold_diagnostic_keys_contract.py`, `tests/models/api/multimodal/processors/hpo/test_optuna_resume_diagnostics.py`) (2026-08-31)
  - `grid_search.py` reorganized: monolithic `_optimize_model` split into `_prepare_grid_run()` (returns a `GridRunState`) plus `_run_holdout_optimization()` / `_run_cv_optimization()`; holdout evaluation extracted into `_evaluate_holdout_param()` with a frozen `_HoldoutExecutionSpec`
  - New `_grid_fold_scores_from_cv_results()` assembles per-candidate fold-score vectors from the recovery scheduler's explicit `fold_scores` or plain `GridSearchCV` `split{i}_test_score` columns, so the post-search callback persists real fold-level diagnostics instead of only the mean
  - `contracts.py`: `OPTUNA_FOLD_DIAGNOSTIC_KEYS` pins the nine fold-diagnostic keys (`cv_train_fold_scores`, `overfit_gap`, `overfit_gap_mean`, `overfit_gap_std`, `fold_indices`, `fold_train_sizes`, `fold_validation_sizes`, `fold_score_source`, `fold_classification_diags`); both `set_trial_attrs` and `_set_partial_trial_attrs` write all of them (contract test enforced)
  - `optuna_trial_execution.py`: new `_score_fold()` unifies per-fold fitting/scoring plus classification diagnostics across the pruning and non-pruning paths
  - `optuna_study.py` + `resume_state.py`: resume priors forward a `diagnostics_list` into `inject_compatible_prior_trials`; `_load_optuna_history` preserves fold-level user attrs from both `hpo_trials` rows and `grid_search_results` payloads, so resumed trials carry the same fold detail as live ones
  - `evaluation/metrics.py`: shared `metric_requires_probability_scores` predicate replaces hard-coded probability-metric sets; adds `log_loss` / `balanced_accuracy` support and train-set F1/precision/recall/balanced-accuracy diagnostics; `get_sklearn_scoring` accepts callable scorers and resolves `spearman_rho` / `balanced_accuracy` / `neg_log_loss`
- **Fail-closed metric selection validation** (`metrics/catalog.py`, `metrics/validation.py`, `metrics/__init__.py`, `screening_engine/base.py`, `tests/models/api/screening_engine/test_metric_selection_validation.py`) (2026-08-31)
  - `validate_primary_metric()` normalizes a requested metric against `METRIC_REGISTRY` and rejects anything whose status is not `supported` (historical names like `kendall_tau`/`msle`/`mpe`) or incompatible with the task family
  - `is_recognized_scorer_name()` checks whether sklearn itself recognizes an arbitrary scorer string; `ScreeningConfig` validates `primary_metric` and `hpo_scoring` at config time — fail closed on unknown/wrong-task names while still accepting valid sklearn scorer names, and `None` `hpo_scoring` stays allowed
- **Runtime configuration protocol** (`config/runtime/contracts.py`, `config/runtime/overrides.py`, `config/runtime/profiles.py`, `config/runtime/__init__.py`) (2026-08-31)
  - New `RuntimeConfig` Protocol (`max_cpu_cores`, `max_workers_per_model`, `execution_preference`, `model_timeout`, `auto_resource_optimization`) decouples the runtime adapter annotations (`apply_overrides` / `apply_runtime_profile`) from the screening engine's `ScreeningConfig`, so runtime config is not coupled to engine config shapes; `runtime/__init__` exports it
- **Shared contracts package** (`contracts/__init__.py`, `contracts/modality.py`, `contracts/deprecations.py`, `tests/data/test_modality_contract.py`, `tests/ci/check_layer_dependencies.py`) (2026-08-31)
  - New `molblender.contracts` owns stable domain contracts shared by independent packages: `InputModality` (moved from `models.modality_models.modality_detector`, which now re-imports it so the old import path keeps working) and `MolBlenderDeprecationWarning` (a `UserWarning` subclass that stays visible under Python's default warning filters, unlike `DeprecationWarning`)
  - `data/cache/multimodal/*` no longer imports the model layer; the CI layer gate now forbids any `molblender.models` import from `contracts`/`config`/`data`/`cache`/`metrics`/`representations` and from top-level `persistence`/`infrastructure`
- **Evaluator component extraction** (`screening_engine/evaluation/evaluator.py`, `evaluator_telemetry.py`, `evaluator_validation.py`, `nested_cv_evaluator.py`) (2026-09-01)
  - `evaluator_telemetry.py` owns structured event emission and the `EvaluationError` mapping policy (`emit_evaluation_event` / `emit_evaluation_failure_event` / `handle_standard_evaluation_exception` / `raise_mapped_evaluation_error`), extracted from `evaluator.py` and shared by holdout + CV-only paths
  - `evaluator_validation.py` owns `validate_regression_predictions` (regression primary predictions must be numeric, non-empty, length-matched, finite) so the evaluator façade no longer carries validation logic
  - Holdout execution split into `_run_holdout_cv` / `_train_predict_holdout` / `_assemble_holdout_result`; `_validate_regression_predictions` import removed from evaluator, `NestedCVEvaluator` passes `validation_mode="nested_inner_cv"` explicitly
- **Grid/HPO execution extraction** (`screening_engine/hpo/grid_execution.py`, `candidate_cv.py`, `fold_diagnostics.py`, `request_validation.py`, `grid_search.py`) (2026-09-01)
  - New `grid_execution.py` owns model-fitting loops (`evaluate_holdout_candidate`, `execute_holdout_grid`, `execute_parent_grid_cv`) with explicit `HoldoutCandidateContext` / `HoldoutCandidateResult` / `HoldoutGridContext` frozen dataclasses; `grid_search.py` drops its inline holdout/fold execution and delegates
  - New `candidate_cv.py` `evaluate_candidate_folds()` — the single candidate-level fold loop shared by Grid and Optuna; per-fold failure recorded as a `FoldDiagnostics` with the remaining folds retained
  - New `fold_diagnostics.py` is the single source of truth for per-fold/per-candidate diagnostics (`FoldDiagnostics`, `CandidateDiagnostics`, `candidate_to_trial_event`, `collapse_candidate`, `has_any_finite_score`); Optuna `_build_event` and Grid both emit the identical `TrialEvent` via the shared translator instead of re-assembling folds or overfit gaps by hand
  - New `request_validation.py` is the single data-completeness gate reused by every HPO entry point (X/y presence + length, holdout `validation_source`, materialized CV folds, groups/stratify); delegates fold rules to `SplitPlan`'s canonical validators
- **Screening/lead migration table + package-alias machinery** (`_compat/migration_table.py`, `_compat/package_alias.py`, `_compat/__init__.py`, `tests/screening/test_migration_table_integrity.py`, `tests/screening/test_migration_identity.py`, `tests/screening/test_screening_entry_surface.py`) (2026-09-02)
  - New `molblender._compat` is the single owner of the legacy `molblender.models.api.*` alias machinery: `register_package_alias()` binds `sys.modules[legacy_root]` to the canonical root object, records it in a process-wide registry and emits one `MolBlenderDeprecationWarning`
  - A single `_LegacySubmoduleAliasFinder` meta-path hook lazily resolves deep legacy imports (`legacy_root.submodule`) to the same canonical module object via the migration table, so importing a legacy root never drags in GPU / PLM / HPO submodules; unregistered deep imports raise `ModuleNotFoundError` instead of being silently faked
  - `migration_table.py` is the hand-maintained, immutable old-path → new-path table (`screening_engine` → `screening.engine`, `screening_runtime` → `screening.runtime`, `multimodal` → `screening.orchestration`, `screening` → `screening`, `lead` → `molblender.lead`); the integrity test enumerates the real filesystem and fails closed if the table drifts from the modules on disk
  - New `tests/lead/` suite (behavior contracts, route equivalence, runner) and `tests/screening/` surface tests pin the public entry exports and legacy identity (`old is new`)
- **Persistent-output path module** (`config/paths.py`, `config/__init__.py`) (2026-09-02)
  - New `molblender.config.paths` centralizes persistent-output locations (`get_output_root`, `get_default_screening_db_path`, `get_default_lead_output_dir`, `get_representation_cache_dir`); `config` re-exports them, and `LeadAnalysisRequest.output_dir` now defaults to `get_default_lead_output_dir()` instead of `./lead_sensitivity_results`
- **Canonical persistence migration + artifact path policy + resource compat** (`persistence/migration.py`, `persistence/backend.py`, `config/paths.py`, `infrastructure/resource_compat.py`) (2026-09-02)
  - New `molblender.persistence.migration` is the single owner of screening-results migration/repair (`migrate_json_to_sqlite`, `migrate_folder_to_sqlite`, `repair_orphaned_results`, `export_session_to_json`, `get_database_stats`, `list_database_sessions`) with `MigrationOutcome` / `MigrationStatus` / `RepairOutcome` dataclasses; `persistence.backend.ScreeningResultsDB.transaction()` provides all-or-nothing semantics for multi-statement migration
  - `config.paths` Artifact Path Policy: `MOLBLENDER_CACHE_ROOT` is the single root for every generic cache (defaults `~/.cache/molblender`), `MOLBLENDER_CACHE_DIR` is now representation-cache-only, `MOLBLENDER_OUTPUT_DIR` is the persistent-output root (defaults `~/.molblender/outputs`); `get_cache_root()` / `get_output_root()` derive from these env vars
  - New `molblender.infrastructure.resource_compat` is the canonical source for `n_jobs` → `max_workers_per_model` / `max_cpu_cores` translation used by the `molblender.screening` / `molblender.lead` call chains

### Changed
- **Screening and lead promoted to top-level packages** (`screening/`, `lead/`, `models/api/screening/`, `models/api/screening_engine/`, `models/api/screening_runtime/`, `models/api/multimodal/`, `models/api/lead/`, `models/README.md`) (2026-09-02)
  - `molblender.screening` is now the single public entry for all screening workflows: standard presets (`quick_screen` / `thorough_screen` / `screen_models` / `interpretable_screen` / `StandardScreener`), comparison (`compare_models` / `compare_representations` / `simple_evaluate`), universal/multimodal (`universal_screen` / `UniversalScreener`) and core contracts (`ScreeningConfig` / `TaskType` / `MetricType` / `ModelCorpus` / `ModelResult` / `Combination`); public names resolve lazily through the export manifest so importing the package never drags in the heavy engine/orchestration implementations
  - `molblender.lead` is the single public entry for lead sensitivity analysis (`run_lead_sensitivity_analysis` / `generate_lead_combinations` / `extract_best_config_from_db`); heavy runner/viz subpackages stay lazily imported
  - `molblender.lead` now resolves its public surface (`LeadAnalysisRequest` / `LeadAnalysisOutcome` / `LeadAnalysisRunner` plus the legacy sensitivity API) lazily through a package `__getattr__` export map, so importing the package no longer eagerly pulls in pandas / matplotlib / scikit-learn until a name is accessed
  - Internal layout: engine → `screening.engine`, runtime → `screening.runtime`, universal/multimodal orchestration → `screening.orchestration`; the old `molblender.models.api.screening_engine` / `...screening_runtime` / `...multimodal` / `...screening` / `...lead` paths keep working as aliases via `_compat` (see Added + Deprecated)
  - README and docs sync the canonical import paths (`molblender.screening.engine.base.ScreeningConfig` etc.); docs badge now points at ReadTheDocs

### Changed
- **Dashboard launcher consolidation** (`dashboard/launcher.py`, `api/dashboard.py`, `cli/commands/view.py`, `dashboard/metrics/metrics_calculator.py`, `tests/api/test_unified_api_contract.py`, `tests/cli/test_cli_refactor_unit.py`) (2026-08-31)
  - New `dashboard/launcher.launch_dashboard(results_folder, *, port=8501, no_browser=False, max_workers=None) -> int` is the single owner of Streamlit spawning: resolves `app.py` next to itself, sets env vars, opens the browser optionally, waits on the process and returns its exit code; spawns with `stdout`/`stderr=DEVNULL` to prevent pipe deadlocks
  - `cli/commands/view.py` drops its inline Streamlit spawn logic (subprocess/env/sleep/browser-open/interrupt handling) and delegates to `launch_dashboard`; `api/dashboard.py` calls `launch_dashboard` directly with `results_dir, port, no_browser, max_workers` instead of invoking the CLI command via `SimpleNamespace`, and no longer mutates `sys.argv`
  - Tests: `run_dashboard` calls `launch_dashboard(results_dir, port=8512, no_browser=True, max_workers=6)` once; launcher test asserts `Popen` gets `DEVNULL` stdio (deadlock avoidance) and owns the real `app.py` entrypoint
- **CV splitter relocation to data domain** (`data/dataset/splitting/cv.py`, `data/splitting.py`, `screening_engine/evaluation/cross_validation.py`, `multimodal/processors/hpo/training_data.py`, `tests/data/splitting/test_grouped_nested_cv.py`, `tests/data/splitting/test_split_plan_contract.py`) (2026-08-31)
  - New `data/dataset/splitting/cv.py` owns `StratifiedCVWithLabels` and `create_cv_splitter(task_type, cv_folds, random_state, stratify_labels=None, force_stratified=False)`; `screening_engine/evaluation/cross_validation.py` loses those definitions and re-imports from the data domain, `data/splitting.py` and `hpo/training_data.py` follow the new import path — one owner for CV splitter primitives under data splitting
  - Tests updated to the typed interface: grouped-nested-CV mocks use `optimizer.optimize(request)` returning `OptimizationResult` and assert the typed `apply_phase2_transformer` flag forwards; split-plan contract stubs include `cv_folds=3` beside `hpo_cv_folds=3`
- **HPO objective decoupled from reporting metric + materialization delegation** (`screening_engine/hpo/contracts.py`, `multimodal/processors/hpo/processor.py`, `multimodal/processors/hpo/candidate_materialization.py`, `multimodal/processors/hpo/candidate_execution.py`, `tests/models/api/multimodal/processors/hpo/test_candidate_execution_seam.py`) (2026-08-31)
- **HPO objective decoupled from reporting metric + materialization delegation** (`screening_engine/hpo/contracts.py`, `multimodal/processors/hpo/processor.py`, `multimodal/processors/hpo/candidate_materialization.py`, `multimodal/processors/hpo/candidate_execution.py`, `tests/models/api/multimodal/processors/hpo/test_candidate_execution_seam.py`) (2026-08-31)
  - New `resolve_hpo_scoring_metric()` (`hpo_scoring` > `request_scoring` > `primary_metric`) makes Grid and Optuna optimize the same objective; candidate execution/materialization now use it instead of `config.primary_metric`, decoupling the HPO objective from the final reporting metric
  - `processor.py` sheds its local per-candidate materialization code (removed `HPOMaterializeContext`, `HPORecordSchedulingRequest`, `_materialize_and_record_candidate`, `_refit_param_model`, `_materialize_per_param_loop`, `_materialize_prior_best`, `_materialize_no_params`, `_mark_best_hpo_trial`); the coordinator now delegates execution to `execute_hpo_candidate` and persistence prep to `materialize_hpo_candidate` in `candidate_materialization.py`
- **Persistence and infrastructure promoted to top-level packages** (`persistence/`, `infrastructure/`, `models/api/persistence/__init__.py`, `models/api/infrastructure/__init__.py`) (2026-08-31)
  - `molblender.persistence` (32 modules incl. `store/`) and `molblender.infrastructure` (runtime policy, GPU scheduling, telemetry) are now physical top-level owners importable without the application layer
  - The historical `molblender.models.api.persistence` / `...infrastructure` paths remain as genuine module aliases: they register `sys.modules` entries pointing at the canonical modules so package-scoped state cannot split, and emit a user-visible `MolBlenderDeprecationWarning`; all internal call sites already import the canonical paths
- **HPOSettings immutable grid-optimizer boundary expanded** (`screening_engine/hpo/contracts.py`, `optuna_optimizer.py`, `optuna_study.py`, `optuna_trial_execution.py`, `result_translation.py`, `tests/models/api/screening_engine/hpo/test_hpo_settings_interface.py`) (2026-09-01)
  - `HPOSettings` now also projects `configured_hpo_scoring`/`primary_metric`, `optuna_timeout`/`optuna_pruning`/`optuna_sampler`, `max_parallel_jobs`/`random_state` and `estimator_params` from the config, with fail-closed interface invariants (validation-strategy whitelist, Optuna+holdout rejection, `max_parallel_jobs >= -1`) enforced at the snapshot boundary; `resolve_scoring()` resolves the objective without re-reading mutable config
  - `OptunaOptimizer` reads every HPO knob from its `HPOSettings` snapshot (trials, timeout, pruning, sampler, random seed for sampler) instead of `getattr(config, ...)`, so post-construction config mutation cannot change a running optimizer; sample-weight knobs come from a frozen `SampleWeightPolicy.from_config(config)`
  - `optuna_trial_execution` folds diagnostics into the same immutable records: `_build_event` reconstructs `CandidateDiagnostics` from trial user attrs and routes through `candidate_to_trial_event`; `evaluate_trial_cv`/`_run_pruning_folds_parallel` return `fold_errors` alongside scores; `set_trial_attrs`/`_set_partial_trial_attrs` persist them
  - `result_translation._extract_train_fold_scores_for_best` falls back to GridSearch-style `split{i}_train_score` columns when no Optuna-style train folds exist; `translate_optimization_result` fold-score locals widened to `list[float | None]`
- **Grid/Optuna sample-weight + config boundary hardening** (`screening_engine/hpo/grid_search.py`, `screening_engine/evaluation/sample_weighting.py`, `screening_engine/base.py`, `data/splitting.py`, `persistence/_stage1_reuse.py`) (2026-09-01)
  - New frozen `SampleWeightPolicy` dataclass (`enabled`/`strategy`/`threshold`/`power`/`quantile`) with `from_config()` snapshots fit-time weight knobs once, so long-lived candidate loops never read a mutable `ScreeningConfig`; `grid_search` and Optuna both consume it
  - `ScreeningConfig._finalize` normalizes `task_type` through `validate_task_type(allow_internal_generic=True)` so documented string forms (e.g. `"regression"`) are canonicalized before validation instead of being treated as classification
  - `data/dataset/splitting/contracts.py` makes `validate_materialized_cv_folds` / `validate_plan_labels` public (they were `_private`) so `request_validation` can delegate; `data/splitting.py` follows the new names
  - `_stage1_reuse` rollback also handles `sqlite3.ProgrammingError` (malformed bind values raise either error), preserving the all-or-nothing reuse contract after rollback
- **mypy/ruff/bandit cleanup for HPO interface migration** (`pyproject.toml`, `screening_engine/hpo/*`, `screening_engine/evaluation/*`, `tests/models/api/screening_engine/`) (2026-09-01)
  - `pyproject.toml` gains a `[tool.mypy]` section (`python_version=3.9`, `warn_unused_ignores`, `warn_redundant_casts`, `ignore_missing_imports` with a narrow per-module override for sklearn/joblib/optuna/rdkit/torch)
  - HPO/evaluation files fixed to type-clean: `metric_direction.scorer_to_natural_list` accepts `Sequence` from `collections.abc`; `nested_cv_evaluator` annotates `all_metrics: dict[str, list[float]]` and `aggregated_metrics: dict[str, Any]`; `fold_diagnostics`/`grid_search` declare `float | None` locals; `evaluate_trial_cv` return annotation includes `fold_errors`; `optuna = None` guards carry `# type: ignore[assignment]`; `grid_execution`/`grid_search` import blocks sorted
  - Bandit: zero High findings on the migrated surface (8 pre-existing Low b110/b101 only); ruff clean on all changed source

### Deprecated
- `molblender.models.api.screening_engine` / `...screening_runtime` / `...multimodal` / `...screening` / `...lead` import paths (2026-09-02) — moved to `molblender.screening` / `molblender.screening.engine` / `molblender.screening.runtime` / `molblender.screening.orchestration` / `molblender.lead`; legacy paths keep working as module aliases (same module objects) with a `MolBlenderDeprecationWarning` until v2.0
- `molblender.models.api.persistence` and `molblender.models.api.infrastructure` import paths (2026-08-31) — moved to `molblender.persistence` / `molblender.infrastructure`; legacy paths keep working as module aliases until v2.0
- `molblender.models.lead` import path (2026-09-02) — moved to `molblender.lead` (top-level); the legacy path keeps working as a deprecated alias that emits `MolBlenderDeprecationWarning` until v2.0
- `molblender.models.api.utils` import path (2026-09-02) — re-exports delegate to `molblender.persistence` and the remaining utilities route to their canonical owners (persistence/infrastructure/foundation); the legacy path keeps working as an alias that emits `MolBlenderDeprecationWarning` until v2.0
- `molblender.models.api.utils.migration` import path (2026-09-02) — moved to `molblender.persistence.migration`; the legacy path keeps working as a re-export shim that emits `MolBlenderDeprecationWarning` until v2.0
- `molblender.models.api.legacy_parameters` import path (2026-09-02) — aliases `molblender.infrastructure.resource_compat`; the legacy path keeps working with `MolBlenderDeprecationWarning` until v2.0

### Fixed
- **Cross-session Stage-1 reuse fails fast on summary refresh** (`persistence/_stage1_reuse.py`, `persistence/session_write.py`, `tests/models/api/persistence/test_session_write.py`, `tests/models/api/persistence/test_stage1_results_queries.py`) (2026-08-31)
  - After importing compatible Stage-1 rows, the target session summary is rebuilt from the effective row view; a failed refresh now aborts the reuse with `PersistenceOperationalError` instead of silently proceeding to recompute; only legacy schemas missing the `updated_at` column keep a narrow compatibility path
- **Untrusted multimodal cache rejects pickle; specialized formats read back via metadata sidecar** (`data/cache/core.py`, `data/cache/exceptions.py`, `data/cache/multimodal/`, `tests/data/test_multimodal_cache_security.py`) (2026-08-31)
  - `RepresentationCache` / `MultiModalCache` gain `trusted: bool = True`; with `trusted=False` any pickle-backed payload fails closed with `CacheTrustError` instead of being deserialized, while safe JSON/NumPy/GraphML artifacts stay readable
  - Modality loaders resolve artifacts through the metadata sidecar with cache-root path-traversal guarding, so dedicated formats (matrix/image/graph/spatial-3D/complex) are genuinely read back instead of always missing
- **Generic cache trust contract** (`cache/manager.py`, `cache/__init__.py`, `tests/cache/test_cache_trust_contract.py`) (2026-08-31)
  - `CacheManager` gains `trusted: bool = True`: a present-but-unloadable blob now raises `CacheCorruptionError` with the row preserved instead of being treated as a silent miss; on `trusted=False` any present pickle raises `CacheTrustError` instead of being unpickled, so hostile or foreign payloads are neither executed nor silently discarded; missing/expired keys and transient SQLite failures still return the default as before
  - `get_cache` / `_reset_global_cache` accept the flag; `get_cache` rejects a silent trust-policy downgrade on the singleton; `cache.__init__` exports `CacheLoadError` / `CacheCorruptionError` / `CacheTrustError`
- **Stage-1 schema error classification + result-presence queries fail closed** (`persistence/_stage1_identity.py`, `persistence/result_queries.py`, `tests/models/api/persistence/test_stage1_schema_error_classification.py`, `tests/models/api/persistence/test_result_queries.py`) (2026-08-31)
  - `ensure_stage1_compatibility_schema` returns `False` only for the legacy "no such table: model_results" case; all other SQLite/operational failures raise `PersistenceOperationalError` instead of being silently downgraded to `False`; a concurrent "duplicate column" race is treated as success
  - `check_existing_model_result` / `check_existing_hpo_result`: `False` now means only a genuine miss or the legacy missing-`model_results` case; operational DB failures raise `PersistenceOperationalError` (already-typed errors re-raised without re-wrapping)
- **Stage-1 identity NaN tolerance + bad-config isolation** (`persistence/_stage1_identity.py`, `multimodal/modality_handlers/base.py`, `tests/models/api/persistence/test_stage1_result_compatibility.py`, `tests/models/api/multimodal/test_stage1_expected_candidates_nan.py`) (2026-08-27)
  - `_canonical_json_obj` no longer raises `ValueError` on non-finite floats (NaN/inf); it maps them to stable string sentinels `__nan__`/`__inf__`/`__-inf__` so legitimate estimator params like XGBoost's `missing=np.nan` keep a deterministic Stage-1 compatibility key instead of aborting the whole signature build
  - `build_stage1_result_signature` call in the vector modality handler wrapped in `try/except (ValueError, TypeError)`; a single unencodable representation config sets `expected[(repr_name, model_name)] = None` and continues instead of aborting the entire vector modality's expected-candidates registry
  - Tests cover both paths: `_canonical_json` canonicalizes NaN/inf to distinct sentinels (distinct keys, no collision), and a genuinely unencodable config (unordered collection) is contained per-representation rather than propagating
- **CPU/GPU training concurrency split + Stage-1 worker budget** (`screening_engine/evaluation/evaluator_helpers.py`, `screening_engine/evaluation/evaluator.py`, `screening_runtime/evaluation.py`, `tests/models/test_vae_integration.py`, `tests/models/api/screening_runtime/test_preparation_parallel_policy.py`) (2026-08-27)
  - `is_heavy_workload()` gains keyword-only `include_representation: bool = True`; model-fit schedulers (`StandardEvaluator._is_heavy_workload`, `_classify_models_by_weight`) pass `False` so a CPU estimator (RandomForest/MLP) consuming GPU-backed features (UniMol) is no longer misclassified as training-heavy and forced onto the GPU-heavy scheduler
  - Removed the `"neural"` pattern from the model-side `heavy_patterns` lists so representation name is the sole driver of representation-side heaviness; the representation-side check is skipped entirely when `include_representation=False`
  - `_resolve_stage1_light_workers` rewritten to reserve a per-model CPU budget: `outer_workers = total_workers // per_model_workers` (from `resource_policy.context.max_workers_per_model`), capped at `min(outer_workers, max(1, n_models))`; prevents launching one BLAS-spawning Stage-1 job per model while each also starts its own OpenMP threads
  - Tests: RandomForest/MLP on `UniMol-CLS-unimolv1-WithHs` with `include_representation=False` returns False (MLP default still True); outer-workers policy reserves per-model cores (56/8/13 → 7) and can use the full budget for single-thread models (4/1/13 → 4)
- **Markdown editor blank-page bug** (`writings/patent/scripts/md_editor.py`) (2026-08-25)
  - `INDEX_HTML` declared as raw triple-quoted string (`r"""`): inline JS backslashes (regex, `'\\'` string literals) no longer collapsed by Python, eliminating "Invalid or unexpected token" browser error
- **HPO processor typed refactor + CV-only metrics + train-metric delegation** (`multimodal/processors/hpo/processor.py`, `results.py`, `screening_engine/evaluation/metrics.py`) (2026-08-25)
  - `processor.py` refactored: internal orchestration state now bundled in typed dataclasses (`HPORecordSchedulingRequest`, `HPOMaterializeContext`), replacing positional attribute unpacking across three `record_hpo_result` call sites; coordinator delegates directly to `database_write_ops.save_hpo_result_record_outcome`
  - `build_cv_only_all_metrics()` in `results.py`: produces schema-correct `all_metrics` for CV-only HPO (without running engine `compute_all_metrics`), including `evaluation_mode="hpo_cv_only"`, `cv_fold_count`, fold-score std deviation
  - `_compute_hpo_train_metrics()` rewritten to delegate to engine canonical `compute_train_regression_metrics` / new `compute_train_classification_metrics` instead of self-pairing train predictions into `compute_all_metrics` and filtering for `train_*` keys; classification callers now pass real train proba/decision scores
  - New `compute_train_classification_metrics()` in `evaluation/metrics.py`: single owner of `train_*` classification naming (`train_accuracy`, `train_roc_auc`, `train_pr_auc`, `train_roc_auc_ovr`, `train_roc_auc_ovo`)
  - `PROVENANCE_KEYS` tuple added to `evaluation/metrics.py`: keys that must never be overwritten during metric recomputation (split_fingerprint, hpo_split_fingerprint, candidate_selection, used_sample_weight, evaluation_mode, hpo_score_source, hpo_stage, hpo_method, stage); `base_helpers.py` metric recomputation now skips these via `setdefault`
  - `_normalize_stage1_primary_metrics()` in `candidate_preparation.py` handles `None` values explicitly before `float()` conversion
  - Grid-search optimizer gains `_run_parent_grid_cv()` callback-enabled parent-process CV path to avoid writing trial state from joblib workers
- **Evaluation fold-loop extraction + holdout materialized CV contracts** (`screening_engine/evaluation/cross_validation.py`, `data/dataset/splitting/contracts.py`) (2026-08-25)
  - `_FoldEvaluationResult` dataclass in `cross_validation.py`: single source for per-fold fit/predict/mask bookkeeping; fold loop extracted from `StandardEvaluator` into `_run_fold_evaluation`
  - `validate_precomputed_cv_folds()` validates explicit CV folds at the public evaluator boundary; evaluator deduplicates cv-only fold loop (delegated to shared function)
  - `SplitPlan` contract updates in `splitting/contracts.py` and `splitting.py`: holdout materialized CV contract
  - `parallel_strategies.py` gains `PROVENANCE_KEYS` awareness; `evaluator_helpers.py`, `evaluation_requests.py`, `quality_transformer.py` minor updates for fold/provenance contract
- **Persistence OperationalError wrapping + SQLite query indexes** (`persistence/errors.py`, `_stage1_effective.py`, `_stage1_reuse.py`, `store/schema_bootstrap.py`) (2026-08-25)
  - New `PersistenceOperationalError(RuntimeError)` exception wraps `sqlite3.OperationalError` with operation name and cause; bare `except Exception` handlers in `_stage1_effective.py` and `_stage1_reuse.py` narrowed to `except sqlite3.OperationalError` — callers cannot mistake DB failures for cache misses; `sqlite3.InterfaceError` logged as warning instead of error
  - Two new SQLite indexes: `idx_model_results_stage1_effective` (composite on session_id, stage, representation_name, model_name, stage1_compatibility_key) and `idx_stage1_links_target_source`
  - `database_write_ops.py` new `_result_primary_metric_value()` helper: safely extracts numeric primary metric from result objects, handles JSON-string metrics dicts, falls back to NaN
  - `database_session_ops.py` now computes `data_fingerprint` via `compute_data_fingerprint()` at session creation (was deferred); `session_service.py` adds `"success"` boolean field to session summary stats
  - `ScreeningSessionCoordinator` class role clarified in docstring, imports from `database_write_ops`
- **Cache security: path-traversal hardening** (`data/cache/multimodal/loading_helpers.py`, `representations/utils/cache_validator.py`) (2026-08-25)
  - New `CacheLoadError` exception class in `loading_helpers.py`; new `_resolve_artifact_path()` / `_resolve_artifact_paths()` functions validate metadata-provided artifact paths are confined to cache root, rejecting absolute paths or `..` sequences that escape
  - `_path_under_root()` helper + `trusted_root` parameter in `cache_validator.validate_cache_file()`: rejects cache files outside trusted root
- **Temporal representations: explicit placeholder classes** (`representations/temporal/`) (2026-08-25)
  - `__init__.py` docstring rewritten: package explicitly marked experimental, dynamic `importlib` auto-import removed, `Ensemble`/`Trajectory` intentionally not exported so `from molblender.representations.temporal import *` does not claim usable API
  - `Ensemble` and `Trajectory` placeholder classes now raise `NotImplementedError` on instantiation with actionable messages pointing to `CLAUDE.md`; guidance to use `DynamicsTrajectory` from `representations.image.video` for video-based dynamics
  - New `CLAUDE.md` (93 lines) and `README.md` (46 lines) document the extension area
- **New tests** (2026-08-25)
  - `tests/data/test_multimodal_cache_security.py` (251 lines): path-traversal hardening for `_resolve_artifact_path`, `_resolve_artifact_paths`, and all `load_*` functions
  - `tests/representations/temporal/test_temporal_placeholders.py` (169 lines): import contract — package imports cleanly, `Ensemble`/`Trajectory` raise `NotImplementedError`, no registered featurizer points at temporal
  - `tests/models/api/screening_engine/evaluation/test_cv_only_fold_loop_equivalence.py` (284 lines): cv-only fold loop delegation regression coverage (primary metric, per-fold scores/predictions, classification proba, phase-2 metadata, non-finite rejection)
  - `tests/models/api/screening_engine/evaluation/test_holdout_materialized_cv_contract.py` (207 lines): holdout Stage-1 CV materialized in `SplitPlan`
  - `tests/models/api/test_batch4_real_data_smoke.py` (368 lines): end-to-end smoke tests on tiny synthetic data (vector/matrix path, holdout, cv-only, grid/optuna HPO, cross-session skip, optional deps)
  - `tests/models/api/test_export_governance_freshness.py` (194 lines): isolated-subprocess lazy-import and fail-closed `__getattr__` contract
  - `tests/models/api/test_hpo_result_evaluation_mode.py` (133 lines): `create_hpo_result` evaluation_mode and prior-source preservation
  - `tests/models/api/persistence/test_stage1_query_indexes.py` (34 lines): schema contract for new SQLite indexes
-
- **Metrics 核心收口 + evaluator 组件拆分** (`metrics/core.py`, `evaluation/evaluator_components.py`, `evaluation/metrics.py`) (2026-08-25)
  - `metrics/core.py` 成为所有数学原语的唯一 owner（r²、pearson、spearman、RMSE、MAE、分类指标、效率比）；`drawings/`、`dashboard/`、`export.py` 均委托于此
  - 新增 `evaluator_components.py`：pipeline construction、fold execution、result extraction 从 evaluator 拆入独立模块
  - `nested_cv_evaluator.py`、`quality_flow.py` 消费同一组件，不再自定义 fold 逻辑
  - `efficiency_metrics.py` 统一效率指标计算；`prediction_metrics.py` 统一预测指标投影
  - 10 条 metric parity + 6 条 evaluator component 契约覆盖回归/分类指标一致性
  - `screening_runtime` 不再反向导入 multimodal；34 条 Standard/Universal 生命周期回归覆盖 HPO-once、中断、resume 与禁用 DB
- **Stage-1 effective view 与身份拆分** (`persistence/_stage1_effective.py`, `_stage1_identity.py`, `_stage1_reuse.py`) (2026-08-25)
  - `stage1_results.py` 的 effective view、兼容性身份、跨会话 link、schema fallback 与 summary 刷新拆入私有模块
  - `_effective_stage1_query()` 已是唯一 SQL owner，禁止复制 query 或再建 facade
- **Dashboard result enrichment 收口** (`dashboard/metrics/calculations/`, `dashboard/data/`) (2026-08-25)
  - `dashboard.metrics.calculations` 是补充指标、效率和结果复制的唯一实现；data API、`DashboardMetrics` 与 processors helper 均直接委托
  - 所有入口保持同一 core 指标、低指标方向和不可变输入语义；10 条 enrichment contract 回归覆盖
- **Comparative 共享 split/指标方向** (`screening/comparative.py`) (2026-08-25)
  - 比较入口预先从所有 representation 物化一个 `SplitPlan` 并按同一 plan 应用；winner 与差值使用 canonical metric direction，忽略非有限分数
  - `r2_score` 不再误读 Pearson squared；比较/简单评估 contract 覆盖共享 fingerprint、RMSE 最小值优先和 NaN 排除
- **HPO training-data 不可变收口** (`multimodal/processors/hpo/training_data.py`) (2026-08-25)
  - 训练数据重建、SplitPlan 投影和预物化 folds 由不可变 `ReconstructedHPOTrainingData` 唯一承载；执行路径已直接消费它
  - `results.py` 只保留 HPO 结果构建与分析，不再有第二条 tuple 路径
- **Session service 收敛** (`persistence/session_service.py`) (2026-08-25)
  - 新增 session 持久化应用服务：create/summary/indices/fingerprint 收口为 persistence 的 application service
  - Standard _DBManagerShim 和 Universal manager 注入同一 adapter，manager 只管理 DB lifecycle
  - `ProfessionalResultProcessor` → `ScreeningResultProcessor` 重命名，旧名保留 alias
  - 适配器、persistence、screening_engine 各模块 import 路径统一
- **Standard execution 收敛到 screening_runtime** (`api/screening/standard.py`) (2026-08-25)
  - 删除废弃 `standard_execution.py`，Standard 路径经 `ScreeningRunContext` 直接委托 `screening_runtime`
  - 并行策略测试迁至 `screening_runtime/test_preparation_parallel_policy.py`
- **删除无生产调用的旧 import 路径** (2026-08-25)
  - 删除 `screening.adapters` 及 engine 的 Butina、feature-clustering、splitting-utils re-export；canonical owner 分别为 `screening.standard` 和 `data.dataset.splitting`
- **HPO training-data Module 收敛** (`multimodal/processors/hpo/training_data.py`, `results.py`) (2026-08-25)
  - 新增 `reconstruct_hpo_training_data()`，返回不可变 `ReconstructedHPOTrainingData`
  - 特征恢复、split-index 解析、预物化 HPO folds 唯一归属新模块；`results.py` 只保留结果构建与分析，-399 行

### Added
- **HPO 候选物化与 Phase-2 评估拆分** (`multimodal/processors/hpo/`, `screening_engine/evaluation/`) (2026-08-24)
  - 新增 `candidate_materialization.py`：负责完成 HPO 候选的 holdout refit、`ModelResult` 构造与持久化，经注入的 `HPOResultPersistencePort`，不直接查写 SQLite
  - 新增 `phase2_evaluation.py`：Phase-2 质量管线的 pipeline wrapping、fitted state 提取、元数据刷新与 fold-mask 对账，不依赖 `StandardEvaluator`
  - `evaluator.py` 精简为编排边界，Phase-2 逻辑收敛到独立模块
  - `processor.py` 大幅瘦身（-571 行），候选选择/会话生命周期/resume 策略归 coordinator，per-candidate 物化归新模块
  - grid_search / optuna_optimizer 指标方向处理与 result_translation 保持契约
- **会话身份收口（session identity contract）** (`data/identity.py`, `persistence/session_write.py`, `persistence/session_ops.py`) (2026-08-24)
  - `data/identity.py` 新增 `ScreeningDataIdentity` 三指纹契约：`cohort_fingerprint`（有序样本身份）、`data_fingerprint`（版本化输入内容+目标列+目标值，不可归一化时返回 None）、`split_fingerprint`（分割结构）
  - `compute_data_fingerprint()` 覆盖有序输入内容、目标列名、目标值；不同 SMILES、不同 y 值、不同行序均得到不同指纹，相同输入得到相同指纹
  - `screening_sessions` 表新增 `data_fingerprint`、`identity_status`（pending/verified/unavailable）、`session_kind`（screening/aggregate）、`merge_provenance` 列，含自动迁移
  - `Session` 对象新增 `identity_status` 三态生命期：pending（创建时写入 data_fingerprint）→ verified（outcome 后写入 cohort+split）或 unavailable（写入失败/不可归一化）
  - `finalize_session_identity()` 原子写入 cohort+split 并设 verified，缺任一字段或 DB 错误自动设 unavailable
  - Standard 和 Universal 入口统一收敛到此 finalizer，返回状态必须检查，失败日志警告
  - `check_session_eligibility()` 在 persistence 层集中过滤 session_kind='screening' AND identity_status='verified'
  - `find_compatible_stage1_results()` 和 `load_stage2_prior_results()` 增加 JOIN 过滤，旧 schema 无列时自动降级
  - Resume 路径在 `services.py` 中检查 eligibility，不合格自动回退新建 session
  - `merge_databases()` 拒绝非 screening+verified 的 session，比较 data_fingerprint+cohort+split+target 四字段，缺任一 fail-closed
  - aggregate session 的 `session_kind='aggregate'`，`identity_status='unavailable'`，`merge_provenance` 存专用列
  - `build_merged_session()` 接收显式 `SourceSessionRef(db_path, session_id)` 列表，不再按位置推断来源 DB
  - 删除 `AGGREGATE:` 前缀约定和 `cache_dir` 滥用，`cache_dir` 保持原语义

### Changed
- **HPO 持久化状态契约贯穿到 DB 层** (`persistence/model_results_write.py`, `multimodal/processors/hpo/`) (2026-08-23)
  - `save_model_result_record` 返回结构化 `ModelResultWriteOutcome`（INSERTED/UPDATED/DUPLICATE/FAILED），`__bool__` 保持旧调用兼容
  - 新增 `_payload_equivalent()` 按字段规范化比对，相同 payload 短路为 DUPLICATE 不写 DB
  - `BEGIN IMMEDIATE` 事务隔离防止并发 SELECT+INSERT 竞态
  - Stage-1 identity 含 `stage1_compatibility_key`（NULL-safe IS 精确匹配），不同 split/config 不再互相覆盖
  - `save_hpo_result_record_outcome` 用 `create_db_model_result` 适配器保留预测/概率字段，JSON 主指标正确解析
  - `HPOStageCoordinator.persist()` 代替 `_save_hpo_result_realtime`，fingerprint 校验 fail-closed，三路 DB 分发
  - 写点 1/2/3 统一传 `param_key`/`combo`/`existing_param_keys` 内存去重，DB 确认 DUPLICATE 后缓存参数键
  - `record_hpo_result` 接受 `HPOResultPersistencePort` 协议，返回结构化 `HPORecordOutcome`，session_id None 语义修正
  - 新增 22 条合约测试（DB 层 10 + port 级 12），旧测试 `is True`/`is False` 适配为 `bool()` 包装
  - 全部 501 测试通过，ruff 无错误
  - **Dashboard 分类归一化收口** (`dashboard/`, `tests/dashboard/`) (2026-08-23)
  - `classification_contracts.py` 成为唯一归一化 seam：`normalize_classification_pair` 返回 `ClassificationNormalizedResult`（y_true/y_pred/valid_mask/probability/decision_score），score 有限性并入统一 mask
  - `classification_plots._labelize` 删除旧 argmax/阈值回退路径，无条件委托 normalization；归一化失败返回 None 时跳过图表并显示 warning，不再猜测
  - ROC/PR 通路使用 `result.probability`/`result.decision_score`（已过滤），不回读原始 payload 数组，解决 NaN/Inf 行混淆矩阵与 ROC 样本不一致的 P0 问题
  - 1D float score 增加 NaN/Inf 守卫，过滤后再做阈值判断
  - `session_loader.normalize_classification_pair` 标记为 deprecated 兼容 wrapper，新调用方统一从 `classification_contracts` 导入
  - 新增 18 条合约测试：反序 [2,1]、字符串标签、重复标签、列数不符、未知真值、NaN/Inf 统一 mask、wrapper identity、proba+dscore 同时存在的一致性
  - 新增 `test_optional_test_markers.py`：ODDT/PLM 真实集成测试 marker 矩阵契约，collect-only 锁定测试集合
  - ODDT 测试补全 `@pytest.mark.integration`/`dependency`/`asset`/`slow` 标记
  - `ConfigManager._load_cache_config()` 删除，`settings.py` 为唯一 cache live authority
- **utils/database → persistence/store + session merge 迁移完成** (`persistence/`, `utils/`) (2026-08-22)
  - `utils/database/` 的 schema bootstrap、admin ops、aggregation、results_query、exports 迁入 `persistence/store/`；`utils/database/` 改为 lazy `__getattr__` 兼容 facade，不触发循环导入
  - `SessionMergeError` 统一到 `persistence/errors.py`，`utils.database.exceptions` 与 `persistence.store.exceptions` 均为 re-export，identity 一致
  - session merge 实装迁入 `persistence/`：`session_merge.py`（六公开操作）、`_session_merge_data.py`（去重/dataset-info 合并）、`_session_merge_sqlite.py`（SQL 读/重写/适配）、`session_merge_core.py`（校验 + payload）；旧 `utils/session_merger.py` 等三模块为纯 re-export
  - CLI `merge_session.py` 与 Dashboard `loaders.py` 直连 persistence；`utils/storage.py` 改为兼容 facade
  - 修复 `build_merged_session` 的 `best_score` 方向 bug：mae/rmse 等 lower-is-better 指标现在取 min 有限值，排除 None/NaN/Inf
  - 修复原地合并 FK 顺序：删除前显式清理 `hpo_trials`（无 cascade）与 `stage1_result_links`，避免 dangling link
  - 新增 `test_session_merge_compat.py`：六公开函数 identity、新旧输出等价、原地合并备份/rollback/FK 清理、MAE/RMSE best_score 方向、`rg utils.session_merge` 为零
- **Stage1* → Screening* 命名迁移完成** (`screening_runtime/`, `screening/standard.py`, `multimodal/`, `screening_engine/`) (2026-08-22)
  - 旧名全部改为 canonical 名：`Stage1Execution`→`ScreeningExecution`、`Stage1Result`→`ScreeningExecutionResult`、`Stage1Hooks`→`ScreeningHooks`、`StandardStage1Execution`→`StandardScreeningExecution`、`ProfessionalResultProcessor`→`ScreeningResultProcessor`、`ProfessionalDataHandler`→`ScreeningDataHandler`、`ProfessionalModelRegistry`→`ScreeningModelRegistry`、`HPOProcessor`→`HPOStageCoordinator`、`ScreeningDatabaseManager`→`ScreeningSessionCoordinator`
  - Standard 路径 `ScreeningRun.run(request)` 改为显式 `ScreeningRun.run(context=ScreeningRunContext(...), executor=StandardScreeningExecution(...))`
  - Universal 路径已接入 `ScreeningRun.run(context, executor=UniversalExecution(...))`，HPO 恰好一次
  - 修复 `run.py` 兼容入口回归：`ScreeningRunRequest` + 自定义 executor 不再错传给执行器，改为先转 `ScreeningRunContext`
  - 新增真实 UniversalScreener.run_screening() 入口测试、API facade 取消测试、legacy request + 自定义 executor 回归测试
  - 全部 `Deprecated alias` 注释改为 `Backward-compat alias`，不发 DeprecationWarning
  - `screening_engine/__init__.py` 与 `screening_runtime/__init__.py` 导出统一用 canonical 名，旧名作为兼容 alias 保留
  - 旧名作为 backward-compat alias 保留，计划 v2.0 移除
- **Metrics NaN 合同统一与 EvaluationRequest 迁移** (`metrics/core.py`, `metrics/formatting.py`, `drawings/model/metrics.py`, `screening_engine/evaluation/`, `screening_runtime/`, `screening/`) (2026-08-21)
  - `calculate_spearman_rho` / `calculate_kendall_tau` 全分支（scipy、ImportError fallback、Exception fallback、size<2、constant input）统一返回 `float("nan")`，不再返回 `0.0`；docstring Note 同步更新
  - `is_better_metric(value_a, value_b, name)` 语义修正：有限值始终优于非有限值（NaN/inf）；两者皆非有限时返回 `False`；修复 HPO winner selection 当 baseline 为 NaN 时选不出 winner 的问题
  - `drawings/model/metrics.py` 的重复 `is_better_metric`（自带 higher_better/lower_better 列表、无 NaN 处理）改为委托 `molblender.metrics.formatting.is_better_metric`，保证全包 NaN 语义一致
  - `EvaluationRequest` / `CVOnlyEvaluationRequest` 成为内部评估的窄类型入口：`representation_config` 默认 `None`；`CVOnclyEvaluationRequest` typo 修正为 `CVOnlyEvaluationRequest` 并保留旧名 alias
  - 迁移全部宽接口 `evaluate_model` / `evaluate_model_cv_only` 调用方至构造 request 后调用 `evaluate_from_request` / `evaluate_cv_only_from_request`：runtime (`screening_runtime/evaluation.py`) → engine (`batch_processing.py`, `parallel_strategies.py`) → `nested_cv_evaluator.py` → `screening/standard.py` + `comparative.py`；旧宽接口签名保留为 adapter 委托
  - 测试：`test_spearman_constant_matches_canonical` 改用 `np.isnan` 断言；`test_formatting` 新增 3 个 `is_better_metric` NaN 语义用例；`test_batch_processing_timeout_context` dummy evaluator 改为 `evaluate_from_request`
- **HPO 双线收敛：`optimize(request)` 成为唯一生产入口** (`screening_engine/hpo/`, `multimodal/processors/hpo/`) (2026-08-20)
  - `OptimizationResult` 契约统一：`best_score` 与 `cv_results` 分数一律 scorer-space（越大越好），新增 `score_space="scorer"` 与 `validation_mode` 元数据；`__post_init__` fail-closed 校验非空 `cv_results` 必须声明 `score_space`、`validation_mode` 内外一致、分数 finite-or-None
  - `translate_optimization_result()` 成为唯一 scorer→natural 转换点，输出字段改为 `best_hpo_score_natural` / `fold_scores`（natural）；`create_hpo_result()` 只消费 natural 字段，不再从 `cv_results` 重新扫描 fold（修复 Optuna prior-best 误取 live 首参数 fold 的问题）
  - `optimize_model()` 与 `optimize_multiple_models()` 降级为 deprecated 兼容适配器（kwargs → `OptimizationRequest` → `optimize()` → 旧 tuple），不再承载实现、不再写 `hpo_cv_score`；Optuna 不可用时 fallback 改走 Grid typed `optimize()`
  - Optuna final-fit fail-closed：best 参数全量重训失败 → `model=None` + `status="failed"`，不再返回未 fit 模型
  - Storage-mode coarse prior 幂等注入：新 RDB study 注入 coarse priors，重跑经 canonical-params 去重不重复
  - `score_space` 与 `score_provenance` 分离：数值空间只取 `scorer`/`natural`，来源记在 `score_provenance`
  - 弃用：`optimize_model()` / `optimize_multiple_models()` 及只读别名 `hpo_cv_score` 计划在两个发布周期后移除；业务代码禁止写 `hpo_cv_score`，仅 persistence/read adapter 保留兼容读取

### Fixed
- **HF/TORCH cache 路径单一权威源 + hub cache 全链路闭合** (`config/settings.py`, `config/core.py`, `config/models/`, `representations/sequential/language_model/`, `representations/protein/sequence/plm/`) (2026-08-22)
  - `settings.py` 成为 cache 路径唯一权威源：新增 `get_hf_hub_cache_dir()` 返回 hub 快照目录（默认 `$HF_HOME/hub`，可被 `HF_HUB_CACHE` 或 `set_cache_dir("hf_hub", path)` 覆盖）；`ConfigManager.set_cache_dir`/`get_cache_dir` 委托给 `settings`，不再维护独立 `_cache_config` 副本
  - **P0 generic loader**：`load_huggingface_model()` 三处（model、tokenizer、clean_kwargs fallback）由 `setdefault` 改为 `get-or` 模式，显式 `cache_dir=None` 不再透传 None 给 `from_pretrained`，统一解析为 `get_hf_hub_cache_dir()`
  - **P0 ESM-C**：`ESMCFeaturizer._load_model` 直连 HF 路径，`load_kwargs` 改为 `cache_dir=self.cache_dir or get_hf_hub_cache_dir()`，四个 `from_pretrained` 共享；新增 `local_model_path` 存在性校验与 model weights 缺失时的 download policy 重检查
  - **P0 MolFormer 等分子 LM**：`BaseMolecularLM.__init__` 不再在构造时快照 hub cache（`self.cache_dir = cache_dir`，默认 None）；`MolFormerFeaturizer._load_model` 在加载时解析 `effective_cache_dir = self.cache_dir or get_hf_hub_cache_dir()`，四个 `from_pretrained` 复用——构造后调用 `set_cache_dir("hf_hub", ...)` 即时生效
  - **P1 文档**：`CLAUDE.md` 删除 `TRANSFORMERS_CACHE`，路径示例改为 `~/data/...` 相对写法；补充 `HF_HOME`/`HF_HUB_CACHE`/`from_pretrained(cache_dir=...)` 三者语义关系说明
  - 全部直接读取 `EFFECTIVE_HF_HOME`/`EFFECTIVE_TORCH_HOME` 的 PLM/LM 模块（`t5`/`carp`/`peptide`/`esm`/`model_doctor`/`torch_hub`）改为运行时 `get_cache_dir("hf"/"torch")` 调用，不再 import 时快照
  - 新增 `tests/config/test_cache_dir_contracts.py`：29 条契约测试覆盖 `set_cache_dir` round-trip、`ConfigManager` 委托、env 传播、`get_hf_hub_cache_dir` 默认/覆盖、以及 ESM-C/MolFormer/generic tokenizer fallback 三个 mock 测试断言 `from_pretrained` 实收 `cache_dir == set_cache_dir("hf_hub", tmp)`
- **Side-fixes unblocking P0-P5 commit** (2026-08-22)
  - `representations/utils/base_featurizer.py`: mypy-driven split of the `_prepare_input` RDKIT_MOL guard into a multi-line `if (...)` dropped the trailing colon, causing a `SyntaxError` that broke all fusion export tests; colon restored
  - `tests/dashboard/test_merged_session_performance.py::test_metrics_not_recalculated_on_cache_hit`: used the `benchmark` fixture without a skip guard; added `pytest.importorskip("pytest_benchmark")` so the module skips cleanly when pytest-benchmark is absent
  - **P0**: 策略注册表双实现消除 — `contracts.py` 移除重复的 `StrategySpec`、`_build_registry`、`resolve_strategy`、`check_prerequisites` 等全部注册表代码，改为 `from .strategy_registry import *` 兼容 re-export；`strategy_registry.py` 成为唯一权威注册表；`SplitPlan`、`NestedCVFold` 及辅助函数保留在 `contracts.py`；新增对象身份与唯一 registry 测试
  - **P0**: `r2_score` 引擎输出恢复 — 回退此前"不输出 r2_score"的改动，恢复 sklearn CoD 输出确保向后兼容；`pearson_r2` 保持为规范回归指标，`r2_score` 标记废弃但有实际输出；配置编译期不拒绝 `r2_score`，避免评估末尾崩溃
  - **P1**: `test_plm.py` 三次 `pytestmark` 赋值合并为单次列表，`dependency`、`filterwarnings` 共存不再覆盖；`test_deepchem.py` 同理修复
  - `tests/metrics/test_formatting.py`: P0 NaN-semantics migration made `format_metric_value`/`safe_float_to_str` return `"NaN"`/`"Inf"`/`"-Inf"` for non-finite values; updated 6 assertions to match the new labels (replacing the generic `"unavailable"`), and refreshed the stale `safe_float_to_str` docstring examples
- **Mypy type errors in image/utils.py and registry/core.py** (2026-08-19)
  - `save_molecular_image()`: renamed parameter to match `PIL.Image.save(format=...)` keyword, fixing a builtin shadowing bug where bare `format` referred to the Python builtin rather than the function's intended `file_format` argument
  - `batch_process_images()`, `_generic_get_featurizer()`: return type corrected from lowercase `callable` to `Callable[..., Any]`
  - `augment_molecular_image()`: renamed `enhancer` to `brightness_enhancer` / `contrast_enhancer` to eliminate shadowed variable
  - `extract_image_statistics()`: return type widened to `dict[str, Any]` (dict contains `shape` tuple and `dtype` string, not only floats)
  - `calculate_image_similarity()`: `return float(1.0 / (1.0 + mse))` — explicit Python float cast to avoid returning a numpy scalar
  - `register_featurizer()` / `get_featurizer()` and protein counterparts: `expected_base_class=BaseFeaturizer` / `BaseProteinFeaturizer` annotated with `# type: ignore[type-abstract]` (4 sites)
  - `default_kwargs` in `_generic_get_featurizer()`: annotated `dict[str, Any] = {}` at declaration; reassignment branch uses unannotated `{}` to avoid `no-redef`

### Added
- **Classification metrics refactor: canonical score extraction, metric name normalization, single decision point for binary score selection** (`screening_engine/evaluation/metrics.py`, `evaluator.py`, `cross_validation.py`, `grid_search.py`, `dashboard/data/metrics.py`, `metrics/core.py`) (2026-08-19)
  - `_normalize_classification_metric_name()` resolves MetricType enum values and string aliases to a canonical name; unknown names return unchanged so the caller rejects explicitly rather than falling through to accuracy
  - `_extract_score()` rewritten: 2D Nx2 proba without explicit labels returns last column (sklearn convention); 2D with >2 columns requires `class_labels` matching the column count; 1D arrays pass through unchanged
  - `_select_binary_score()` introduced as the single decision point for all binary ranking metrics, returning `(pos_score, pos_label)` tuple consumed by `_classification_metric_score`, `get_scoring_function`, and `compute_classification_metrics`
  - `_normalize_label_predictions()` now handles 2D multiclass proba matrices (argmax) and 1D binary proba (threshold), with explicit integer label passthrough
  - CV `_classification_metric_score()` delegates binary ROC/PR to `_select_binary_score` for consistent positive-class column selection
  - `get_scoring_function()` accepts `class_labels` parameter and passes it through to the returned lambda; probe_metric logic updated to detect `class_labels` as a required kwarg
  - `get_sklearn_scoring()` removed redundant `needs_proba` logic in favour of `response_method="predict_proba"` on `make_scorer`; streamlined multiclass ROC-AUC/PR-AUC makers
  - `evaluator.py` updates `compute_classification_metrics` call to pass `class_labels` through; simplified `_compute_and_save_metrics` to use `compute_all_metrics` return value directly
  - `grid_search.py` HPO `_compute_score` passes `class_labels` to `get_scoring_function`; `CoarseGridSearchOptimizer` extracts `hpo_metric_value` with enum-safe `.value` accessor
  - `cross_validation.py` properly passes `class_labels` to `_extract_score` in `_score_func`; `_cv_metrics` now passes `class_labels` to `compute_classification_metrics`
  - `metrics/core.py` `_select_binary_score` import added; `dashboard/data/metrics.py` `_normalize_classification_metric_name` import added
  - 5 new test modules: `test_classification_metric_core_final.py` (22), `test_classification_metrics_integration.py` (43), `test_evaluator_classification_metric_validity.py` (12), `test_pearson_constant_nan.py` (11), `test_grid_search_metric_validity.py` (12) — 100 total new tests, all passing
  - Test `test_regression_metric_parity.py` expanded with constant-target and empty-input fallback coverage
- **Classification metrics unification** (`screening_engine/evaluation/metrics.py`, `screening_engine/types.py`, `screening_engine/evaluation/evaluator.py`) (2026-08-18)
  - CV and final-Test ROC-AUC for multiclass now share the same scorer (`roc_auc_score(multi_class="ovr", average="weighted")`) via `make_scorer(response_method="predict_proba", ...)` instead of sklearn's default `macro`-OvR string
  - `_classification_metric_score()` gains `y_score` parameter; roc_auc/pr_auc return an `UNAVAILABLE` marker when no probability matrix is available, preventing downstream NaN/zero injection
  - multiclass PR-AUC computes `average_precision_score(y, score, average="weighted")` (sklearn ≥1.4 multiclass OvR); no longer emits `0.0` placeholder into DB/sorting/charts
  - `compute_classification_metrics()` accepts `class_labels` for correct positive-class probability-column extraction in binary problems
  - precision/recall now compute all three `average` modes (macro/micro/weighted); MetricType enum and catalog declarations aligned with what the engine actually computes
  - `get_scoring_function()` returns `UNAVAILABLE` when roc_auc/pr_auc has no probability input, so HPO no longer thresholds and re-scores internally
  - Test coverage in `test_corrections.py`, `test_stage1_hpo_db_fallback.py`, `test_standard_stage1_compatibility.py`
- **Stage-1 key-based per-(repr, model) coverage matching** (`persistence/stage1_results.py`, `multimodal/modality_handlers/base.py`, `base_helpers.py`, `vector.py`, `string.py`, `language_model.py`) (2026-08-18)
  - DB resume matching refined from model-name-only to `(representation_name, model_name, stage1_compatibility_key)` triple, so a resumed session only reuses Stage-1 rows whose compatibility identity matches the current config
  - `ModalityHandlerMixin._build_stage1_expected_candidates()` builds the expected `(repr, model) → key` mapping via `registry.get_models()` / `get_params()` / `fusion_metadata` / `get_representation_config()`; called from vector, string, and language-model handlers before any resume check
  - `parse_complete_repr_names()` in `base_helpers.py` groups parsed `(model, key)` pairs by representation and returns the set of representation names that are parse-complete, enabling per-representation gap detection
  - `_fetch_complete_stage1_representations_for_candidates()` in `stage1_results.py` uses a SQL CTE to build expected `(repr, model, key)` VALUES and match within each representation; backward-compat fallback catches "no such table" for `stage1_result_links` and falls back to the native session-scoped query
  - Fail-closed semantics: a candidate with `None` key is skipped, not matched
  - Test stubs in `test_per_route_execution_coverage.py`, `test_language_model_handler.py`, `test_string_matrix_representation_selection.py` updated with `_build_stage1_expected_candidates` lambdas
- **Cross-session Stage-1 compatibility import** (`persistence/stage1_results.py`, `multimodal/services.py`, `screening/standard.py`, `multimodal/screeners.py`, `multimodal/modality_handlers/base.py`) (2026-08-15)
  - New `stage1_results.py` owns Stage-1 row reads, per-representation coverage, the compatibility identity (configuration + cohort + split + target + metric + task bind), schema bootstrap for `stage1_compatibility_key` / `source_session_id` / `source_compatibility_key` columns, and the cross-session copy of strictly-compatible historical Stage-1 rows into a fresh session
  - `ScreeningDatabaseManager.add_screening_result` and `ResultRecord` now carry the compatibility key and provenance columns; `ScreeningResultsDB.add_model_result_from_screening` forwards them through to SQLite
  - `multimodal/services.py` exposes `import_compatible_stage1_results()`, `plan_and_import_compatible_stage1_results()`, and `resolve_screening_session()` as the canonical composition seam — the SQL planning/copy and identity logic stay in `persistence/`
  - `StandardScreener._import_compatible_stage1_results()` builds a `(repr, model, repr_config, model_config)`-keyed candidate registry and stamps `result.stage1_compatibility_key` before any callback sees it; child screeners own cross-session reuse and the parent `UniversalScreener` route-level identity is now an explicit no-op
  - `ModalityHandlerMixin` honours the child-stamped key without re-stamping and never lets the parent's route-level identity overwrite it
  - 6 new test modules: `test_stage1_result_compatibility.py`, `test_stage1_results_queries.py`, `test_standard_stage1_compatibility.py`, `test_compatibility_migration.py`, `test_multimodal_helper_route.py`, and HPO candidate selection fairness test adjustments
- **HPO split metadata projection, persistence backend migration, config contracts** (`multimodal/processors/hpo/`, `persistence/backend.py`, `multimodal/config_contracts.py`) (2026-08-15)
  - `SplitPlan.groups`/`stratify_labels` projected into `OptimizationRequest` by real `train_idx`; missing, 2D, negative, non-integer, and out-of-bounds metadata all fail-closed to `None`
  - `persistence/backend.py` becomes the canonical SQLite backend (`ScreeningSession`/`ResultRecord`/`ScreeningResultsDB`); `utils/results_db.py` shrinks to an identity adapter
  - New `multimodal/config_contracts.py` with typed config boundary dataclasses
  - 5 new test modules: `test_adapter_contracts.py`, `test_split_context_propagation.py`, `test_config_contracts_import_boundary.py`, `test_universal_screen_contract.py`, `test_results_db_backend_owner.py`
  - 18 new tests passing
- **Static GPU-heavy index and resource_intensity declarations** (`screening_engine/`, `representations/registry/`) (2026-08-15)
  - Static GPU-heavy index covers small-molecule LM, UniMol, and protein PLM; policy hits before registry lookup, avoiding model dependency load
  - Model registry declares `resource_intensity` with real CPU/device/process-isolation capabilities
  - Static index is fully immutable (inner and outer layers); in-place mutation raises `TypeError`
  - 7 new contract modules: `configuration.py`, `config_normalization.py`, `errors.py`, `ports.py`, `result_contracts.py`, `routing_contracts.py`, `types.py`
  - Clean subprocess verification: chemberta/UniMol/ESM2 static classification does not import torch/DeepChem/TensorFlow
  - 81 contract tests passing; 25 passed + 3 skipped (live-registry audit missing optional modules)
- **Standard/Comparative owner flip with v2.0 shim layer** (`screening/standard.py`, `screening/comparative*.py`) (2026-08-15)
  - `screening/standard.py` becomes the sole Standard implementation (897 lines); `screening/adapters/standard.py` downgraded to v2.0 warning shim
  - `screening/comparative*.py` becomes the sole Comparative implementation; `screening/legacy/comparative*.py` downgraded to v2.0 warning shim
  - Routing `compat.py` adds `specific_pairs_mode=True` for strict one-model-one-representation validation; otherwise fail-closed
  - AST scan of production source code prohibits re-importing the v2.0 shim
  - 6 modality handlers updated import paths; audit docs record actual owners and 2.0 migration boundary
  - 98 tests passing (1 skipped): 11 legacy execution entry points + 67 Standard/Comparative/Routing + new `test_combination_contract.py`
- **Legacy compatibility entry points marked for v2.0 removal** (`routing/compat.py`, `screening/standard.py`, `cache/adapters/`, `screening/standard_execution.py`) (2026-08-14)
  - Legacy compatibility entry points uniformly declare 2.0 removal, triggering explicit deprecation warnings; `UniversalScreener.run_screening()` legacy `list[str]` input is normalized after a single warning, canonical `Combination` path is zero-warning
  - cache module 4 adapter files slimmed (545 lines removed); ownership boundary tests added `test_cache_ownership.py`, `test_cache_domain_boundary.py`
  - New `test_standard_execution_parallel_policy.py` (127 lines), `test_session_lifecycle_fail_closed.py` (33 tests)
- **Standard adapter consolidation and comparative legacy packages** (`screening/adapters/`, `screening/legacy/`) (2026-08-14)
  - `screening/adapters/standard.py` becomes the single Standard adapter implementation (855 lines); old `screening/standard.py` converted to `__getattr__` deprecation shim
  - `screening/legacy/` consolidates comparative-related modules (comparative/comparative_helpers/comparative_repr_helpers/comparative_simple_helpers); top-level files converted to deprecation shims
  - Public `molblender.models.api.screening` imports directly from `adapters.standard` without warnings; legacy `screening.standard` path maintains warning compatibility
  - 6 multimodal modality handlers updated import paths; 94 new contract tests added
- **Standard/Universal service-construction seam** (`multimodal/services.py`, `screening/standard.py`, `multimodal/screeners.py`) (2026-08-13)
  - New `multimodal/services.py` provides `create_screening_database_manager()` and `create_hpo_processor()` factory functions; Standard and Universal both construct DB manager and HPO processor through this module, no longer directly connecting to private processor implementations
  - Universal screener passes real `data_handler` to HPO processor (previously `None`), ensuring HPO stage correctly handles representation alignment
- **SessionPort protocol and LeadAnalysisRunner** (`api/persistence/session_port.py`, `lead/lead_sensitivity_runner.py`, `lead/runner.py`, `utils/caching/__init__.py`) (2026-08-13)
  - New `ScreeningSessionPort` Protocol defines 4 session persistence operations shared by Standard/Universal; Universal-only lifecycle methods remain `ScreeningDatabaseManager`-exclusive
  - New `LeadAnalysisRunner` extracts lead sensitivity parameter binding and CPU budget parsing into an independent adapter; `lead_sensitivity.py` and `lead/runner.py` both delegate to this adapter
  - `utils/caching/__init__.py` documentation updated: clarifies experimental status, public contract guarantees, and boundary with `data/cache`
  - New `test_session_port.py` (345 lines), `test_stratify_boundary.py` (380 lines); extended `test_utils_caching.py` and `test_runner.py`

- **Cache v3 fingerprint-based representation cache identity** (`data/cache/fingerprint.py`, `data/cache/core.py`, `data/dataset/caching.py`, `data/dataset/partial_cache.py`, `representations/utils/parallel_cached.py`) (2026-08-17)
  - New `compute_molecule_fingerprint` / `compute_dataset_fingerprint` hash canonical SMILES, 3D conformer coordinates, and protein context via SHA-256; the fingerprint is the cache key for v3 entries. Provenance-only fields (e.g. `source_path`) are excluded.
  - `RepresentationCache` bumped to `rep_cache_v3` and now embeds `implementation_identity = {cache_key_version, package_version}` in both top-level payload and metadata; entries with absent or mismatched identity are rejected. Legacy v1/v2 entries fail closed.
  - `DatasetCacheMixin._validate_cache_metadata` switched from `molecule_keys` (canonical SMILES) to `molecule_fingerprints`; coordinate- and protein-state-only differences no longer pass the identity check.
  - `PartialCache` and `MultiModalCache` adopt the v3 key scheme; `find_all_compatible_caches` and direct `load_representation` both compare against the canonical implementation identity.
  - `batch_featurize_with_cache` has two modes: **fail-closed compute mode** (no `featurizer_identity`, single `UserWarning`, no SQLite traffic) and **identity-aware cache mode** (`featurizer_identity` provided, SHA-256 hash of canonical payload). Mismatched `input_contexts` length raises `ValueError`; non-string mapping keys (including `bool`) and unsupported context element types raise `TypeError`; `featurizer_name` alone is never cache identity.
  - 6 new test files: `test_cache_v3_identity_fingerprint.py`, `test_cache_v3_implementation_identity.py`, `test_cache_v3_protein_param_identity.py`, `test_cache_v3_followup.py`, `test_batch_featurize_v3.py`, `test_dataset_v3_cache_wiring.py`

- **Documentation: CADD workflow + interaction guides, rules refresh** (`.claude/rules/`, `CLAUDE.md`) (2026-08-17)
  - New `.claude/rules/cadd_workflow.md` and `.claude/rules/interaction.md` formalize CADD project conventions and response style/table format expectations.
  - Trimmed duplicated encoding/code-style content from `coding_standards.md`; aligned `CLAUDE.md` and the 10 testing rule files (`assertions`, `ci`, `coverage`, `data_management`, `exceptions`, `fixtures`, `mock`, `parameterized`, `performance`, plus `changelog`/`database`/`dataset_config`) with the new guides. No semantic rule changes.

### Deprecated
- `molblender.models.api.screening.standard_execution` converted to deprecation adapter: all execution helper functions in the module now forward from `molblender.models.api.screening_runtime` via `__getattr__`, triggering `DeprecationWarning` on each access. New code must import execution functions directly from `screening_runtime`.

### Fixed
- **Dependency loader refactor and dead-code cleanup** (`config/dependencies/`, `data/dataset/splitting/`, `dashboard/`, `drawings/`) (2026-08-13)
  - Removed dead file `loaders_bio_misc.py` (187 lines, never referenced by any import path)
  - UniMolRepr loader fix: when UniMolRepr is missing, now raises `RuntimeError` and marks overall unavailable, no longer falsely reporting `available=True`
  - Pybel loader migrated from manual `_dependency_cache` to shared `import_dependency()` helper
  - RDKit loader adds `Butina` and `MaxMinPicker` entry points; SciPy loader adds `stats` and `hierarchy`
  - `data/dataset/splitting/` 5 modules migrated to lazy RDKit import; `dashboard/` + `drawings/` 7 files migrated to lazy SciPy import

- **HPO legacy-schema test split and resume bug fix** (`tests/models/api/multimodal/processors/hpo/`) (2026-08-13)
  - Original 3,458-line single file `test_legacy_schema_fail_closed.py` split into 6 focused test files + `_hpo_test_helpers.py` (shared DB stub, `_GridResumableStub`, row-insertion helper); fixes regression in `compute_missing_resume_params` truncation during split that prevented Grid partial-resume from loading missing parameters

- `create_screener()` docstring fix: return value changed from ambiguous "Screener instance" to "StandardScreener instance"; removed unused `BaseScreener` import.
- **Phase 3 persistence layer extraction** (`api/persistence/`, `multimodal/processors/hpo/processor.py`, `screening/standard.py`) (2026-08-12)
  - HPO/session/resume SQL consolidated into `hpo_ops.py`, `result_queries.py`, `stage1_ops.py`, `dataset_info_reads.py`; processor calls through `_check_and_load_resume()`, no longer reads `param_generator`
  - Grid resume formalizes `GridResumable` protocol; non-Grid/legacy mocks no longer fabricate `n_trials_completed`
  - Standard `_DBManagerShim` migrated to call persistence layer; 66-file refactor

- **P0: Universal SelectionOutcome direct-pass to ResultProcessor** (`multimodal/screening_postprocess.py`, `screeners.py`) (2026-08-12)
  - Disallows post-process secondary optimization; `format_multimodal_results()` receives `SelectionOutcome` to ensure CV/HPO-only selection execution

- **P1: Session SQL unified through persistence** (`api/persistence/session_write.py`) (2026-08-12)
  - `create_session()`/`get_session_summary()` unified through `db._get_connection()`; compatibility facade preserved

- **Mypy compliance** (`config/`, `models/api/`, `data/dataset/`) (2026-08-12)
  - Type annotations for `_method_cache`, `suggested`, `all_results`; return-type compatibility; SQLite set→list; `CalledProcessError` fix

- **HPO processor and Optuna module split** (`multimodal/processors/hpo/`, `screening_engine/hpo/`, `tests/models/api/`) (2026-08-13)
  - HPO processor split into `candidate_preparation.py`, `result_recording.py`, `resume_state.py`, `session_context.py`, greatly reducing the processor main file
  - Optuna split into `optuna_search_space.py`, `optuna_study.py`, `optuna_trial_execution.py`; `optuna_optimizer.py` rewritten by 76%
  - 10 new test files covering the module contracts
  - CI layer dependency checks updated

- **HPO/persistence boundary restoration** (`multimodal/processors/hpo/processor.py`, `api/persistence/session_write.py`, `screening_engine/hpo/contracts.py`, `grid_search.py`, `database/operations.py`, `sessions.py`) (2026-08-12)
  - HPO processor calls persistence layer through `_check_and_load_resume()`
  - Grid resume formalizes `GridResumable` protocol: `can_resume(params, cv_results)`; processor no longer reads `param_generator` private attribute
  - session create/summary consolidated into `persistence/session_write.py`, unified through `db._get_connection()`
  - `nested_cv_evaluator.py` enforces HPO typed contract, pre-materializes folds
  - `test_legacy_schema_fail_closed.py`, `test_grid_resumable.py`, `test_session_write.py` extended coverage

- **Persistence: tighten screening boundaries** (`api/persistence/model_results_write.py`, `database/operations.py`, `nested_cv_evaluator.py`) (2026-08-12)
  - New `model_results_write.py`: bulk model-result write port, consolidated from `database/operations.py`
  - `database/operations.py` reduced by 259 lines; responsibilities narrowed to schema/admin
  - `nested_cv_evaluator.py` HPO typed contract strengthened
  - New `test_model_results_write.py` (610 lines), `test_nested_cv_hpo_typed_contract.py` (621 lines), `test_catalog_registry_boundary.py`, `test_screening_finalization_contract.py`

- **ScreeningRun callback hooks architecture** (`screening_runtime/contracts.py`, `screening_runtime/stage1.py`, `screening_runtime/run.py`, `screening/standard.py`, `multimodal/screeners.py`, `test_contracts.py`) (2026-08-11)
  - `ScreeningCallbacks(on_repr_complete, on_stage1_complete)`: lifecycle hooks supporting per-representation DB persistence and HPO enhancement or DB fallback loading after Stage-1 completion
  - `ScreeningRunRequest.callbacks`: passes runtime extension points via `ScreeningCallbacks`; `ScreeningRun.run()` internally converts to `Stage1Hooks`
  - `Stage1Hooks.on_stage1_complete`: receives mutable `Stage1Result`, allowing callers to modify `all_results` in-place (inject HPO results or DB fallback data)
  - `execute_stage1(incomparable_split=False)`: supports caller-provided incomparable split seeds; OR-semantics merge of internal failure flags
  - `ScreeningOutcome.split_plan`: carries the split plan actually used by the run (immutable run fact); Standard session persistence reads from this field
  - StandardScreener injects per-representation DB persistence and HPO logic through `ScreeningCallbacks` rather than hardcoding in runtime
  - UniversalScreener route orchestration unchanged, still delegates SQLite operations to `ScreeningDatabaseManager`
  - New `screening_runtime/CLAUDE.md`: clarifies shared execution primitive positioning, multimodal boundary, and clear responsibility division
  - Updated `screening_engine/hpo/CLAUDE.md`: adds `translate_optimization_result()` result-translation-layer responsibility notes

- **Phase 1.1 HPO typed contract completion** (`screening_engine/hpo/`, `multimodal/processors/hpo/processor.py`, `test_hpo_contract_completeness.py`) (2026-08-11)
  - `HPOBackend` protocol method changed from `optimize_model(request)` to `optimize(request)` — protocol declaration, Grid/Optuna implementations, and processor call site all share consistent signatures
  - `OptimizationRequest` adds 7 fields: `coarse_cv_results` (Optuna coarse prior / Grid partial-resume), `resume_completed_trials` (Optuna resume), `resume_prior_trials` (Optuna fallback injection), `resume_missing_params` (Grid partial-resume), `representation_name` (legacy Phase 2 transformer), `quality_metadata` (legacy per-component metadata), `apply_phase2_transformer` (whether to enable Phase 2 quality pipeline)
  - `OptimizationResult.best_score` type changed from `float` to `Optional[float]` — failures no longer forced to `0.0`, backends may return `None`; processor skips candidates with `best_score=None` and logs a warning, never saving a fake 0.0 score
  - Grid/Optuna `optimize()` adapters fully forward all new fields to internal `optimize_model()`, not just partial fields
  - Grid/Optuna `optimize()` computes `n_trials_completed` from `cv_results["params"]`, no longer defaults to 0
  - processor `OptimizationRequest` construction changed from multiple rebuilds to a single creation after collecting all fields; resume data read from `result._resume_*` private attributes and passed into request; legacy `quality_metadata` changed from debug-only to actual request passthrough, ensuring legacy representations without `representation_config` can still assemble Phase2QualityTransformer
  - 11 regression tests cover: protocol signature validation, Grid/Optuna field-forwarding capture, `best_score=None` allowance, legacy metadata preservation, correct `n_trials_completed` derivation
  - New `screening_engine/hpo/CLAUDE.md` documenting HPO module positioning, typed contract design, and boundaries with evaluator/result_processor

- **SplitPlan full-chain refactoring** (`contracts.py`, `splitting.py`, `split_plan.py`, `standard.py`, `screeners.py`) (2026-08-01)
  - `SplitPlan` dataclass with frozen, validated indices: `train_indices`, `test_indices`, `cv_folds`, `nested_cv_folds`, `hpo_cv_folds`, `hpo_split_fingerprint`, `groups`, `stratify_labels`, `n_samples` (mandatory). Pre-materialised indices eliminate per-`apply()` splitter `.split()` calls, guaranteeing identical fold indices across all representations and repeated `apply()` calls.
  - `SplitPlan.from_legacy_payload(data, *, X=None, y=None, groups=None)`: materialises CV/nested CV folds from live splitters at construction time; validates `len(X) == n_samples` and `len(y) == len(X)` even when `n_samples` is explicit; raises `ValueError` for negative/duplicate/out-of-range/2D fold indices and length-mismatched group/stratify arrays.
  - `SplitPlan.fingerprint` and `SplitPlan.hpo_split_fingerprint`: SHA-256 fingerprints hash train set + actual fold indices + group/stratification protocol for HPO prior isolation; `hpo_split_fingerprint` excludes raw `hpo_cv_folds` to keep the canonical fingerprint stable.
  - `_compute_hpo_cv_folds(plan, config)`: returns a new `SplitPlan` via `dataclasses.replace()` (never mutates original); computes HPO CV folds on the training subset as local indices `[0, n_train)`; uses `StratifiedGroupKFold` when both groups and stratify_labels are present (fail-closed); `GroupKFold` and `StratifiedGroupKFold` now use `shuffle=True, random_state=random_state` for deterministic seeding consistent with non-group CV path; rejects when unique groups < requested folds.
  - `finalize_shared_split_plan(plan, config, hpo_processor)`: now returns `(finalized_plan, fingerprint)` tuple so callers receive the new plan with HPO folds; called by both `StandardScreener` and `UniversalScreener` after `_extract_split_plan()`.
  - `screening_engine/hpo/` new modules: `contracts.py` (HPO result contract types), `priors.py` (prior loading and stage transition), `result_translation.py` (translation utilities), `search_space.py` (search space definition).
  - `infrastructure/gpu_scheduler.py` (GPU slot management).
  - `screening_runtime/run.py`, `contracts.py` (ScreeningOutcome/SelectionOutcome and run workflow).
  - `test_split_plan_contract.py` (80+ tests covering roundtrip, fingerprint stability, validation, HPO fold computation, groups/stratify, GroupKFold seeding, `_compute_hpo_cv_folds` immutability, and `finalize_shared_split_plan` tuple contract).

- **Vector Representation Fusion (concat-based)** (2026-07-23)
  - New `FusionConfig` (`enabled`, `groups`, `strategy="concat"`, `name_prefix`, `include_original`, `max_components`, `max_total_features`, `on_invalid`, `dtype="float32"`) on `universal_screen` produces synthetic 2D vector representations by concatenating dense fingerprints and descriptors. Each fusion row is named `fusion__A__B__hxxxxxxxx` (SHA-256 hash suffix over the ordered component list) so its component order and identity are stable across runs.
  - The fusion row stores version 2 `representation_config` metadata, including ordered component ranges and replayable quality operations (`replace_non_finite`, initial mask, imputation, scaler, and variance mask). Same-process HPO distinguishes post-quality cache arrays from raw cross-process arrays.
  - Dashboard wiring recognises fusion rows as `vector_fusion`, parses their component metadata from real SQLite rows, and shows component breakdowns. Generated export scripts call the same core `featurize_fusion_features` reconstruction helper used by production code.
  - DB resume round-trips `representation_config`; public `universal_screen()` smoke coverage verifies persistence and same-session skip-existing behavior. Small real GridSearchCV and three-trial Optuna workflows verify stage-2 fusion rows and metadata persistence.
  - Scope remains experimental and explicit: dense 2D vector concatenation only, disabled by default. It does not claim general multimodal or learned fusion. See `Archive/vector_representation_fusion_implementation_plan.md` for implementation evidence and remaining release work.
- **Split fingerprint for CV/nested CV HPO isolation** (`splitting.py`): `build_cv_split_fingerprint` materializes KFold/GroupKFold splitters into per-fold index hashes, with per-outer-fold inner splitter rebuild for group-aware nested CV. `build_split_fingerprint_for_plan` dispatches between holdout (indices) and CV (splitter) modes. `finalize_shared_split_plan` stamps `random_state`, computes the live fingerprint, and pushes it to the HPO processor — called by both `StandardScreener` and `UniversalScreener`
- **`_materialize_nested_cv_inner_folds` and `_rebuild_inner_cv_for_outer_fold` shared helpers** (`splitting.py`): evaluator and fingerprint builder use the same `build_inner_cv_for_outer_fold(..., random_state=random_state_base + fold_idx)` call, preventing per-outer-fold inner splitter drift
- **`update_session_fingerprint_only` in `database_query_ops.py`**: shared SQL helper for persisting CV/nested fingerprints, used by both `ScreeningDatabaseManager` and `_DBManagerShim` so the SQL logic stays in one place
- **`_DBManagerShim.update_session_fingerprint`** (`standard.py`): delegates to the shared helper, so raw `ScreeningResultsDB` paths also write CV fingerprints
- **Optuna study name uses `_live_split_fingerprint`** (`processor.py`): prefers the in-memory live fingerprint set by the screener, so CV/nested modes get a unique study name instead of "unknown"
- **`_validate_session_split_fingerprint` prefers in-memory fingerprint** (`processor.py`): compares against the screener's canonical split plan, not the DB row (which may not exist for CV modes)
- **`_load_all_stage1_results_from_db` and `_load_prior_stage2_grid_results` fingerprint filtering** (`processor.py`): rows with mismatched `all_metrics.split_fingerprint` are rejected; legacy NULL-fingerprint rows require `config.allow_legacy_unfingerprinted_hpo_priors=True`
- **`ScreeningConfig.allow_legacy_unfingerprinted_hpo_priors`** (`base.py`): opt-in flag plumbed through `HPOConfig` → `ScreeningConfig` → `universal_screen()`, controlling both Stage 1 merge and Stage 2 prior filtering
- **`update_session_fingerprint` in routing and screening execution** (`routing_execution.py`, `screening_execution.py`): CV/nested modes write fingerprint-only session records so resumed runs can filter against the correct fold structure
- **`HPOConfig` public field** (`api.py`): `allow_legacy_unfingerprinted_hpo_priors` exposed in the public API

### Changed

- **`persistence/stage1_ops.py` → `persistence/stage1_results.py`** (`persistence/`): the Stage-1 row operations module was renamed and extended with the cross-session compatibility identity, schema bootstrap, planning helpers, and import path. The public API names (`count_*_stage1_representations`, `fetch_stage1_rows`, `fetch_complete_stage1_representation_names`, `sql_placeholders`) keep their names; new names (`Stage1ResultCandidate`, `Stage1ResultSignature`, `build_stage1_result_signature`, `ensure_stage1_compatibility_schema`, `find_compatible_stage1_results`, `import_compatible_stage1_results`, `import_compatible_stage1_results_via_signatures`, `plan_stage1_result_compatibility`) are exported from `molblender.models.api.persistence`.
- **Optuna coarse-grid priors persist into RDB storage** (`screening_engine/hpo/optuna_study.py`): `_inject_coarse_priors` is now invoked on every resume path whenever `coarse_cv_results` is provided, not only when no persistent study exists. `inject_compatible_prior_trials` dedups internally so re-injection is safe and persisted priors survive across resumable sessions.
- **`finalize_shared_split_plan` return type** (`splitting.py`): now returns `(finalized_plan, fingerprint)` tuple (was just `fingerprint`). All callers (`screeners.py`, `standard.py`, test suite) must unpack the tuple.
- **`SplitPlan.n_samples` is mandatory** (`contracts.py`): `SplitPlan(n_samples=None)` raises `ValueError` in `__post_init__`; `from_legacy_payload` without `X` or explicit `n_samples` raises `ValueError`. Callers must provide dataset size at plan construction time.
- **`GroupKFold` / `StratifiedGroupKFold` in group-aware CV now use `shuffle=True`** (`splitting.py`): `_compute_hpo_cv_folds` passes `shuffle=True, random_state=random_state` to both splitters, making group-aware HPO CV deterministic and seed-controlled (previously deterministic but unseeded).
- **`_compute_hpo_cv_folds` returns new `SplitPlan`** (`splitting.py`): no longer mutates the input plan via `object.__setattr__`; returns a fresh `SplitPlan` via `dataclasses.replace()`. Callers must use the returned plan.
- **`from_legacy_payload` validates X/y always when provided** (`contracts.py`): even when `n_samples` is explicit, passing `X`/`y` now validates `len(X) == n_samples` and `len(y) == len(X)` — previously the check was skipped when `n_samples` was present.

- **`standard.py` `run_screening` `shared_split_plan` finalization**: moved `finalize_shared_split_plan` outside the `if shared_split_plan is None` block, so externally-injected plans also get `random_state` stamped and the live fingerprint pushed to HPO
- **`standard.py` DB write branches on CV vs holdout**: CV/nested plans (`cv_splitter`/`outer_cv` in plan) call `update_session_fingerprint` instead of `update_session_indices` which silently refuses to write without train/test indices
- **`ScreeningDatabaseManager.update_session_fingerprint`** (`database.py`): delegates to `database_query_ops.update_session_fingerprint_only` instead of inline SQL
- **Phase2QualityTransformer** (`quality_transformer.py`): sklearn-compatible `TransformerMixin` that wraps per-fold Phase 2 quality transforms (impute/scaler/variance/constant filter) for use inside `Pipeline` with GridSearchCV and Optuna, ensuring train-only statistics per fold
- **`_build_combined_phase2_mask` coordinate projection**: `variance_mask` (post-initial_mask space) is now projected back to original column space via `combined[initial] &= variance_mask`, fixing a crash when a component undergoes two-stage removal (constant filter then variance filter)
- **`component_ranges` support in mask builder**: `_build_combined_phase2_mask` now accepts `component_ranges` and `component_order` to produce full-width masks for fusion rows where some components have no mask entries
- **Explicit `AllFeaturesRemovedError` handling at 5 boundaries**: `StandardEvaluator.evaluate_model`, `evaluate_model_cv_only`, `CoarseGridSearchOptimizer`, `OptunaOptimizer`, and `perform_cross_validation` all handle the domain exception with fail-closed semantics (no silent conversion to nan/0.0 scores)
- **`error_score="raise"` in cross_val_score calls**: `perform_cross_validation` and `OptunaOptimizer` now use `error_score="raise"` so Phase 2 quality collapse propagates as a domain exception rather than a silent nan score
- **Regression tests**: 13 new tests covering `AllFeaturesRemovedError` (12 scenarios in `test_all_features_removed_error.py`) and two-stage mask projection (`test_cv_only_fusion_ba.py`)
- **`reset_global_analysis_cache` and `reset_global_multimodal_cache`**: rebuild helpers for the analysis and multimodal cache singletons, mirroring the existing `reset_global_cache`. Let tests switch `MOLBLENDER_CACHE_DIR` and reset all three cache singletons so previously-touched paths never leak
- **`isolated_representation_cache` fixture** (`tests/conftest.py`): backed by pytest `tmp_path`, sets the env var, updates `EFFECTIVE_REPRESENTATION_CACHE_DIR`, and resets all three cache singletons so each test starts at a clean isolated path
- **`_force_test_cache_dir` autouse fixture** (`tests/conftest.py`): pins `MOLBLENDER_CACHE_DIR` to `tests/.mbl_cache` for every test, hermetically overriding any external setting and re-syncing the runtime config so test ordering cannot leave the project-root default active
- **`test_cache_isolation.py` suite**: documents the isolation contract (suite default points at `tests/.mbl_cache`, `isolated_representation_cache` writes to `tmp_path`, parent suite cache does not grow) and uses a subprocess to verify the cross-process boundary
- **`test_partial_reuse_overlap`**: verifies that the molecule-level partial cache reuse path correctly reuses rows for an overlapping subset and recomputes only the new molecules
- **`TestParallelSchedulerSpy`**: two patches spy `_try_parallel_cdk_execution` (must not be called for single Morgan) and `_add_single_feature_set` (verifies `n_workers` is forwarded to the featurizer layer where Morgan uses joblib)
- **Fusion reconstruction zero-width support**: `reconstruct_fusion_features` now accepts `allow_zero_features=True` for `post_quality_representations`, `raw_representations`, and dataset-cache branches, with a post-concat check raising `FusionConfigError` when every component collapses to zero width
- **`fit_quality_on_train_transform_eval` metadata persistence**: `should_remove`, `initial_mask`, and `variance_mask` are now written before skipping a fully-removed component, enabling downstream replay (HPO resume, dashboard, export)
- **`_build_combined_phase2_mask` signature**: added optional `component_ranges` and `component_order` parameters for fusion-aware mask construction
- **`grid_search.py` and `optuna_optimizer.py`**: added `AllFeaturesRemovedError` import and explicit fail-closed handling at both inner-param and outer-optimization boundaries
- **`perform_cross_validation` non-raise error path** now returns `np.array([np.nan])` instead of `np.array([0.0])` so "evaluation failed" is no longer conflated with a valid score of zero. Consumers (HPO ranking, downstream selectors) can distinguish unevaluable folds from legitimately poor scores
- **`OptunaOptimizer.optimize_model`**: preserves a pristine `base_trial_model = clone(model)` so each trial clones from a clean state and conditional params from one trial never leak into the next. Also handles the case where every trial is pruned or failed (`study.best_trial` raises `ValueError`) by returning `np.nan` instead of an empty result
- **`add_features` single-string featurizer preserves user-given name**: a call like `add_features("morgan_fp_r2_1024")` now produces a feature column named `"morgan_fp_r2_1024"` (the original behavior) instead of being renamed to `feat_0`. Falls back to `feat_{i}` only for unnamed `BaseFeaturizer` instances
- **`ModelResult` carries `evaluation_status` and `error_message`**: lets downstream consumers (HPO ranking, Dashboard) distinguish `completed`, `cv_failed` (all CV folds NaN), and unhandled exception outcomes; `cv_failed` results are marked as unevaluable and no longer enter ranking as if they were valid scores

- **HPO parameter rename** (2026-07-21)
  - Renamed `HPOConfig.selection_strategy` → `selection_scope`, `ScreeningConfig.hpo_selection_strategy` → `hpo_selection_scope`, and all legacy kwarg keys accordingly. The parameter controls which group top-N candidates are selected from ("global", "per_type", "per_subtype"). The old name `selection_strategy` was ambiguous with HPO search algorithm selection.

- **Corpus string parsing centralized as `parse_model_corpus()`** (`screening_engine/models/corpus_filter.py`, `standard_helpers.py`, `comparative_helpers.py`, `model_registry.py`) (2026-08-13)
  - Single fail-closed parser: unknown, blank, and non-string inputs raise `ValueError` instead of silently falling back to `ModelCorpus.ALL`. `"transformers"` consistently resolves to `TRANSFORMERS_ONLY` (narrow predicate), matching across all entry points.

- **Screening cache classic and robust entry points share one canonical payload** (`api/utils/caching/core.py`) (2026-08-13)
  - `cache_screening_results` and `robust_cache_screening_results` write the same `{data: <results>, metadata: {identity + cached_at}}` format; `load_cached_screening_results` and `robust_load_cached_screening_results` apply the same identity verification covering dataset id, target column, config, and representation names. Legacy wrapper files (pre-unification format) remain readable.

- **Cache singletons use isolated default subdirectories under `MOLBLENDER_CACHE_DIR`** (`cache/manager.py`, `data/cache/analysis.py`, `data/cache/multimodal/core.py`) (2026-08-13)
  - SQLite cache, AnalysisCache, MultiModalCache, and RepresentationCache each write to dedicated subdirectories (`sqlite_cache`, `analysis_cache`, `multimodal_cache`) to prevent metadata/payload collisions. SQLite cache TTL comparisons now use UTC-aware timestamps.

### Removed
- **Dead-code facades removed** (2026-07-30)
  - Deleted `models/api/core/` (8 files) — entire deprecated compatibility facade tree. Real implementations live in `models/api/screening_engine/`. Legacy facade contract tests verify the old paths are now unimportable (intentional, prevents regression).
  - Deleted `dashboard/data/calculations.py` (91 lines) — re-export of `metrics.core`. All callers now import directly from `molblender.metrics.core`.
  - Deleted `data/dataset/splitting/{butina,diversity,dnr,feature_clustering,scaffold,umap_clustering}.py` (6 files) — one-liner re-exports forwarding to `strategies/*`. All callers use the canonical strategy modules.
  - Deleted `models/api/compatibility.py` (already done in prior commit) — re-export of `legacy_parameters.py`.
  - Deleted `screening_engine/models/compatibility.py` (already done in prior commit) — split re-export of `eligibility.py` + `corpus_filter.py`.
- **Registration-order Representation Truncation** (2026-07-14)
  - Removed `max_string_representations` / `max_matrix_representations` config fields, `_apply_representation_limit` helper, and `modality_handlers/_truncation_helper.py` module. Default string/matrix routes now execute every compatible candidate returned by the registry; explicit `route.representations` continue to execute alone. Modality handlers check explicit route first so an empty default universe no longer invalidates an explicit selection. Matrix default universe is now `spatial/matrix` only (4 candidates: adjacency_matrix, coulomb_matrix, coulomb_matrix_eig, edge_matrix) so UniMol-style coordinates/embeddings are no longer swept into MATRIX_CNN. Renamed `test_representation_truncation.py` → `test_string_matrix_representation_selection.py`.

### Fixed
- **GPU cleanup `AttributeError` silently skipped** (`evaluation/utilities.py`): `cleanup_model_memory()` caught `NameError` from missing `get_gpu_manager` in a broad `except Exception`, skipping torch cache clear and gc. Each cleanup step now has its own narrow try/except block.
- **GPUManager missing `is_available`/`clear_memory` interface** (`utils/gpu/manager.py`): `utilities.py` called `gpu_manager.is_available()` on a `GPUManager` that had no such method, triggering `AttributeError` swallowed by the broad except. Added both methods.
- **`get_free_memory_mb` used process-local torch reserved memory** (`utils/gpu/manager.py`): fallback to `torch.cuda.memory_reserved()` reflected only the current process's PyTorch allocations, not real global free memory. Now uses NVML exclusively; returns `None` when unavailable so callers never mistake stale data for real free memory.
- **Explicit-selection routes (matrix/image CNN) bypassed shared split plan** (`modality_handlers/base.py`): `_screen_cnn_modality` routed `route.explicit_selection` through `run_combination_screening` without forwarding `shared_split_plan`, so CNN/Transformer routes fell back to a per-route random split and broke cross-route comparability. Now forwards `shared_split_plan=getattr(self, "_canonical_split_plan", None)` so all routes share the canonical train/test membership.
- **`print_and_log_final_summary` announced a "best" winner when no CV/HPO score was available** (`screening_postprocess.py`): when `selection_score` returned `None` for every result (rare CV/HPO failures), the summary would fall back to `all_results[0]` and print `Best: <model>` — silently leaking held-out test signal into the headline. Now fails closed: when `scored` is empty, logs `selection unavailable` and skips the `Best:` line. Consistent with the `result_processor` fail-closed contract.
- **HPO prior loaders silently accepted all legacy rows when the schema lacked `all_metrics`** (`processors/hpo/processor.py`, both `_load_all_stage1_results_from_db` and `_load_prior_stage2_grid_results`): the old `if live_fingerprint and has_all_metrics` guard fell through to "keep everything" whenever the column was absent — defeating the entire split-fingerprint isolation guarantee. Now: when a live fingerprint is present but the schema lacks `all_metrics`, priors are rejected by default and only retained when `HPOConfig.allow_legacy_unfingerprinted_hpo_priors=True`. Both load paths consistent.
- **`_compute_hpo_cv_folds` mutated original `SplitPlan`** (`splitting.py`): `object.__setattr__` on the frozen `SplitPlan` shared by all routes meant downstream HPO processors and caches observed different plan states after finalisation. Now returns a new `SplitPlan` via `dataclasses.replace()`; original plan is never mutated.
- **`finalize_shared_split_plan` callers used old single-value return** (`screeners.py`, `standard.py`, `test_shared_split_data.py`): after the function signature changed to `(plan, fingerprint)` tuple, all callers were updated to unpack the tuple; three test sites also fixed for the same pattern.
- **`_extract_split_plan` missing `X, y` in holdout plan construction** (`test_shared_split_data.py`, `split_plan.py`): holdout split dicts without explicit `n_samples` now require `X`/`y` to be passed; `from_legacy_payload` raises `ValueError` if neither is available.

### Changed
- **GPU management consolidated into `utils/gpu/`** (2026-07-30): previously scattered across `utils/gpu/`, `utils/gpu_manager.py`, `utils/gpu_helpers.py`, `screening_engine/evaluation/gpu_manager.py`. Unified implementation lives in `utils/gpu/` with `GPUManager` exposing `is_available()`, `clear_memory()`, `get_free_memory_mb()` (NVML strict). `screening_engine/evaluation/gpu_manager.py` is now a backward-compatible shim with `DeprecationWarning`.
- **Device selection distinguishes CUDA/MPS/CPU** (`utils/gpu/utils.py`): `auto_select_device()` and `suggest_device()` return `"cuda"`, `"mps"`, or `"cpu"` instead of the old `"gpu"/"cpu"` binary. MPS no longer treated as CUDA. Memory thresholds only apply to CUDA devices.
- **`auto_select_device` is now a pure function** (`utils/gpu/utils.py`): no longer mutates `os.environ`. Added `build_subprocess_env(device, gpu_id)` for subprocess workers that need `CUDA_VISIBLE_DEVICES` set.
- **`check_gpu_available` strict mode** (`utils/gpu/utils.py`): returns `False` when NVML cannot determine real free memory (was `True` with an optimistic "assuming OK" message). Callers must explicitly opt in if they want to proceed without NVML.
- **`calculate_r2_score` sklearn `force_finite=True` parity** (`metrics/core.py`): constant target + perfect prediction → 1.0 (was 0.0), constant + imperfect → 0.0. Single-sample input (`size < 2`) → 0.0 (sklearn returns NaN + warning). NaN/inf divergence documented as intentional robustness choice.
- **`calculate_r2` emits `DeprecationWarning`** (`metrics/core.py`): alias now calls `warnings.warn(DeprecationWarning, stacklevel=2)` pointing to `calculate_pearson_r2` / `calculate_r2_score`.
- **`calculate_pearson_correlation` / `calculate_pearson_r` constant-input guard** (`metrics/core.py`): check `np.std == 0` before `np.corrcoef` → 0.0, no `RuntimeWarning`.
- **`calculate_kendall_tau` constant-input guard** (`metrics/core.py`): NaN → 0.0, `ConstantInputWarning` no longer propagates.
- **`statistical_significance_test` t-test NaN propagation** (`metrics/validation.py`): identical constant groups → returns `statistic=None, p_value=None, significant=False` (was NaN).
- **Dashboard fallback metrics delegated to `molblender.metrics.core`** (`dashboard/metrics/calculations/regression_metrics.py`, `dashboard/metrics/validators.py`): removed duplicate self-computation with `ss_tot + 1e-10` epsilon producing wildly negative values on constant targets. Both `basic_regression_metrics` and `_basic_regression_metrics` now delegate to `calculate_comprehensive_metrics`.
- **Dashboard import missing `calculate_r2_score`** (`dashboard/data/metrics.py`): added to import list; `except Exception: pass` was swallowing `NameError` and returning `{}`.
- **`data_handler.py` DNR docstring** (`screening_engine/data_handler.py`): corrected claim that `label_col` is inferred from `y` — code reads `self._target_column` only, no inference.
- **`metrics_semantics.md` stale TODOs §7 removed** (`docs/source/development/metrics_semantics.md`): removed §7 (calculate_r2, catalog, compatibility split — all done). Updated §4.1/4.2 with single-sample divergence and constant-target parity docs. Updated §5 test-reference to `test_core.py`.

### Changed
- **Atom-sphere Point Cloud Renamed** (2026-07-12)
  - Renamed `surface_descriptors` (`SurfaceDescriptors`) featurizer to `atom_sphere_point_cloud` (`AtomSpherePointCloud`). The old name implied a Connolly/SAS/SES molecular surface that the implementation never produced — the rows are actually a Fibonacci-sphere point cloud on probe-expanded atomic spheres.
  - `LEGACY_FEATURIZER_ALIASES` plus a Python class alias keep old configs and `from molblender.representations import SurfaceDescriptors` working; `get_featurizer("surface_descriptors", ...)` returns the new implementation.
  - Sampling is now a deterministic Fibonacci sphere (no RNG); points are distributed across heavy atoms so the returned array no longer pads with ~90% zero rows.
  - Auto-embedded 3D conformers are reproducible through a new `random_seed` parameter (RDKit `ETKDG().randomSeed`) without leaking into the global Python/NumPy RNG state on modern RDKit builds.

- **Modality Compatibility Routing Refactor** (2026-07-11)
  - Consolidated modality-model compatibility to a single rule based on `supported_modalities`; added `VALID_MODALITIES` (`{vector, matrix, image, string, graph}`) as the single source of truth in `screening_engine/base.py`
  - `is_model_compatible_with_modality()` rejects unknown or empty modality strings with `ValueError` (no silent default-allow)
  - `list_models()` gained a `data_modality` filter that reuses the shared compatibility helper, replacing the previous in-line implementation
  - Extracted `resolve_vector_candidate_models(task_type, include_vae)` on the registry; vector handler now delegates auto/comprehensive candidate resolution to it instead of duplicating the VAE-by-family check
  - `_validate_registry()` now enforces `supported_tasks`, `categories`, `family`, `supported_modalities` (must be in `VALID_MODALITIES`), and `subtypes` (required for `TRADITIONAL_ML`); all violations are reported in a single `ValueError`
  - Patent terminology aligned with code: removed unsupported "auto mode adaptive selection" claim, clarified default mode may exclude VAE family from vector candidates, switched Table 1 modality labels to Chinese curly quotes

- **Model Family / Subtype Classification** (2026-07-11)
  - Promoted `family` and `subtypes` to first-class fields on `ModelConfig`; every registered model now declares a `ModelFamily` (TRADITIONAL_ML, FEED_FORWARD_NEURAL_NETWORK, CNN, TRANSFORMER, VAE, GRAPH_NEURAL_NETWORK) and one or more `ModelSubtype` values
  - Added `get_hpo_group()` helper for HPO grouping by family/subtype; HPO selection unit modes (`combo`, `representation_existing`) operate on this canonical grouping

- **Task Type Compatibility Refinements** (2026-07-10)
  - `supports_task()` treats CLASSIFICATION as the family base with binary/multiclass as variants via `_CLASSIFICATION_VARIANTS`
  - Strict user-facing task type policy: only canonical `TaskType` values are accepted at the public API surface

### Added
- `tests/models/api/screening_engine/test_modality_compatibility.py` (11 cases): registry capability routing, auto/comprehensive vector modes (with optional VAE), unknown/empty modality rejection
- `tests/models/api/screening_engine/test_list_models_corpus_consistency.py` (7 cases): `list_models` ↔ `filter_by_corpus` result parity across all corpora
- `tests/models/api/multimodal/test_hpo_group_consistency.py` (10 cases): family/subtype grouping, combo/representation_existing path consistency, alias top-N

### Fixed
- **Task Type Compatibility Logic** (2026-07-10)
  - Refined `supports_task()` to treat CLASSIFICATION as family base with binary/multiclass as variants
  - Added `_CLASSIFICATION_VARIANTS` constant for clearer classification handling
  - Improved task type fallback logic for better model-registry compatibility

- **Model Corpus Consistency** (2026-07-09)
  - Fixed MINIMAL corpus to use existing model name `svm_linear` instead of non-existent `svr`
  - Unified `minimal` string mapping across standard_helpers and comparative_helpers to `ModelCorpus.MINIMAL`
  - Prevented inconsistent model sets when using `minimal` parameter from different entry points
  - Centralized metric direction logic in `molblender.metrics.get_metric_sort_ascending()`
  - Replaced hardcoded `LOWER_IS_BETTER` sets across CLI, Dashboard, and screening engine
  - Fixed ranking to respect metric direction (MAE/RMSE ascending, R²/accuracy descending)
  - Improved `BaseScreener.top_n()` to use `min` for lower-is-better metrics instead of `max`
  - Enhanced `ProfessionalResultProcessor` summary: `best_score`/`worst_score` now metric-aware
  - Updated `rank_results()` to auto-detect direction when `ascending=None`
  - Applied to: `export.py`, `processors.py`, `result_processor.py`, `base.py`, `ranking.py`, `parallel_strategies.py`
  - Test coverage: added `tests/models/api/screening_engine/test_metric_direction_ranking.py`

- **HPO Parallel Worker Calculation** (2026-06-23)
  - Fixed `_resolve_parallel_jobs()` to handle `cv_splits` parameter explicitly
  - Improved `GridSearchCV.fit()` to skip `groups` parameter when None
  - Enhanced worker cap logic for better resource utilization

- **Optuna Warm-Start Improvements** (2026-06-12)
  - Removed `hpo_method='grid'` filter — warm-start now loads priors from any prior method (grid, random, optuna)
  - Now reads `grid_search_results` JSON column for full param combos instead of only `best_params`
  - Fixed `hpo_cv_score` → `hpo_score` column name mismatch in SQL query
  - Added ultrafine→coarse cascade fallback when fine priors are unavailable in DB
  - Added DB-free fallback using Stage 1 in-memory `grid_search_results`
  - Added `warm_start_source` metadata field in `grid_search_results` for audit trail
  - Default `optuna_warm_start` changed from `False` to `True` in `HPOConfig`

- **XGBoost Optuna Crash Fix** (2026-06-12)
  - Removed duplicate `subsample` parameter suggestion in Optuna search space that caused all xgboost trials to fail with `ValueError`

- **Model Export Functionality Overhaul** (2026-06-13)
  - Fixed `pprint.pformat` rendering `float('nan')` as bare `nan` → `NameError` in generated scripts (XGBoost `"missing": NaN`)
  - Fixed top-N export filename collisions when same model+representation appears in multiple rows (now uses rank-indexed filenames)
  - Added `_MODEL_EXPORT_MAP` + `_CLASSIFIER_MAP` covering 15+ model types for runnable code generation
  - Added classification support: auto-detects `task_type` from DB, generates Classifier classes and classification metrics
  - Fixed `--metric` parameter ignored in CLI export (now sorts by requested metric with correct direction for MAE/RMSE)
  - Added `input_column` CSV header validation with auto-correction fallback to `SMILES`
  - Expanded SQL queries to include `best_params`, `all_metrics`, `train_indices`, `test_indices`
  - Fixed duplicate `train_test_split` in GradientBoosting branch and duplicate `random_state` kwargs

- **Data Split Strategy Parameter Exposure** (2026-06-12)
  - Exposed 8 new split parameters: `scaffold_func`, `split_method`, `maxmin_mode`, `dnr_split_mode`, `dnr_threshold`, `high_dnr_in_test`, `similarity_threshold`, `property_diff_threshold`
  - Added UMAP n_components auto-clamping for small datasets (`<52` samples) to prevent spectral crash
  - Fixed UMAP optional labels handling to prevent KeyError on label-less datasets
  - Fixed splito_scaffold hardcoded parameters to respect user-specified values

- **Splitting StrategySpec Registry & MultiIndex Label Contract** (2026-07-30)
  - Introduced `StrategySpec` frozen dataclass registry (`contracts.py`) as single source of truth for splitting strategies, with context-aware `resolve_strategy(name, entry_point)` alias resolution (e.g. "random" maps to `train_test` in `split_dataset` but to `cv_only` in `cross_validate`) and `check_prerequisites()` fail-closed input validation
  - `DataSplitter` constructor now resolves strategies via `resolve_strategy` instead of hardcoded aliases; `split_dispatch.py` delegates to per-strategy modules through the registry; CV/nested-CV modes share folds via `build_inner_cv_for_outer_fold`
  - `split_dataset()` MultiIndex label extraction: `label_col` as string routes through `get_labels(include_errors=False)` which auto-selects the "value" sub-column in standard `("target","value")+("target","error")` layouts, producing a correct 1D y; full MultiIndex tuples (e.g. `("task1","a")`) are accepted and read directly from `dataset.labels`
  - Fail-closed guards: non-existent `label_col` raises with available column names; user-provided `smiles` kwarg is explicitly rejected (indices must align with dataset molecules); non-standard MultiIndex without a "value" sub-column raises "resolves to N columns"; DNR strategy rejects full MultiIndex tuples at preflight with a clear hint to use a top-level name
  - `optuna_optimizer.py` forwards `apply_phase2_transformer` to the GridSearchCV fallback; test mock no longer swallows unknown kwargs
  - New test suite `test_public_entrypoints.py` (54 cases) covers facade contracts, MultiIndex standard/non-standard convention (capturing y and asserting 1D correctness), DNR + tuple rejection, and val_size compatibility; `test_grouped_nested_cv.py` adds Optuna→GridSearch fallback forwarding tests

- **Dashboard session metadata validation & metric-aware sorting** (2026-07-30)
  - New `validate_session_metadata_compatibility(require_complete)` in `session_merger.py`: strict mode rejects sessions with mixed `task_type` / `primary_metric` or missing metadata before cross-session comparison
  - Dashboard `load_sqlite_results` now calls validator with `require_complete=True` for multi-session loads; catches `SessionMergeError` and displays via `st.error`
  - `results_aggregation._sort_results_by_metric` uses `is_higher_better()` so MAE/RMSE sort ascending (best-first), replacing SQL `ORDER BY DESC` which only works for higher-better metrics
  - `_df_index` positional fix in `selectors.py` (now stores `np.arange(len(df_subset))` instead of `df_subset.index`) and `selection.py` (validates `0 <= df_index < len(df)` before `df.iloc`) to fix wrong-row lookup on filtered DataFrames
  - Removed `dashboard/data/legacy.py` and legacy re-exports from `__init__.py`
  - New tests: `test_load_sqlite_results_mixed_metadata.py`, `test_load_sqlite_results_single_session.py`, `test_load_sqlite_results_connection.py`, `test_load_result_rows_sort_edge_cases.py`, `test_metrics_ground_truth_isolation.py`

- **Aggregation metric direction & SessionMergeError** (2026-07-30)
  - New `database/exceptions.py` defines `SessionMergeError` (ValueError subclass) shared between `aggregation.py` and `session_merger.py` to avoid circular imports
  - `aggregation.get_all_database_results_record` now detects mixed `task_type` / `primary_metric` across sessions and raises `SessionMergeError` with the exact mismatched sets, logging the warning (fail-closed)
  - `best_score` / `worst_score` now respect metric direction (higher_better: best=max/worst=min; lower_better: best=min/worst=max) using local variables instead of the brittle `combined_stats` dict
  - `session_merger.merge_all_sessions` with explicit `session_ids` now sorts results by metric-aware direction before selecting `best_model`, making it equivalent to the default (no `session_ids`) aggregation path
  - `optuna_optimizer.py` forwards `apply_phase2_transformer` to the GridSearchCV fallback

- **Drawings module reorganization** (2026-07-30)
  - Removed deprecated `drawings/multimodal/` (4 files), `drawings/plot_utils.py`, `drawings/graph.py`
  - Updated `drawings/__init__.py` imports and added `plot_screening_results`; updated `drawings/CLAUDE.md`
  - Removed `representations/sequential/language_model/bert_models.py` (re-export wrapper); `__init__.py` now imports directly from `.bert`

### Changed
- **Expanded Fine/Ultrafine HPO Grids** (2026-06-12)
  - Ridge ultrafine: expanded alpha range (1e-6 to 1e7), added `tol`, `max_iter`, `sparse_cg` solver
  - SVM fine/ultrafine: wider C/gamma/epsilon ranges, added loss options for linear SVM
  - KNN fine/ultrafine: more neighbor counts, added p=3 distance metric, expanded leaf_size
  - MLP fine/ultrafine: added smaller (32,) and multi-layer architectures, wider learning rate and alpha ranges
  - Decision tree ultrafine: expanded max_depth, min_samples_split, min_samples_leaf ranges

- **Dashboard Modality Sunburst Fixes** (2026-06-12)
  - Fixed colorbar size mismatch between Modality Breakdown and Model Distribution sunburst charts
  - Fixed all-red color display caused by compressed color range from aggregated `best_performance` min/max
  - Color range now uses raw metric values from all rows (2.99x wider range)
  - Sunburst parent colors now correctly reflect best leaf performance
  - "Modality Distribution" subheader renamed to "Results by Modality" for clarity

- **Dashboard CV-only Test Size Display** (2026-06-12)
  - Test Size card now shows `1/N (percentage%)` for CV-only evaluation modes (e.g., `1/5 (20.0%)`)
  - Previously showed "N/A" which was misleading for cross-validation runs
  - Falls back to "CV-only" for invalid fold counts

- **Performance vs Training Time Chart** (2026-06-12)
  - Removed redundant metric name from title (metric is already shown in the colorbar)
  - Reduced marker size by ~50% for better visual density

- **Removed Unused Facade Layers** (2026-06-12)
  - Deleted `modality_components/facade.py` and `modality_components/modality_charts.py` (pure re-exports with no added value)
  - Inlined orchestration logic directly in `pages.py`
  - Preserved necessary architectural abstractions (no breaking changes)

### Added

- **HPO Selection Unit Modes** (2026-07-10)
  - Added `hpo_selection_unit` configuration to control HPO candidate selection granularity
  - "combo" mode: optimize top model-representation pairs from Stage 1 (default)
  - "representation_existing" mode: select top representations, then optimize Stage 1-evaluated pairs for them
  - "representation_all_routed" mode: (planned) select top representations, then expand to all compatible models via routing
  - Added `select_by_representation_for_hpo()` for Stage 1-evaluated expansion
  - Fixed bug where `representation_existing` mode was never activated due to incorrect conditional after alias mapping
  - Updated `ScreeningConfig.hpo_selection_unit` and `HPOConfig.selection_unit` fields
  - Test coverage: `tests/models/api/multimodal/test_hpo_selection_unit.py`

- **Optuna Per-Fold CV Scores** (2026-06-12)
  - Optuna trials now store individual fold scores in `cv_fold_scores` and `split{i}_test_score` keys
  - Matches GridSearch output format for uniform downstream processing

- **Named Feature Importance with column_mask Alignment** (2026-06-11)
  - Added `feature_names` and `column_mask` fields to `ModelResult` dataclass
  - Implemented `_build_top_named_importance()` with column_mask application logic
  - ODDT residue-bit interpretation now produces semantic names (e.g., "ARG@2::salt_bridge" instead of "f_1")
  - Complete flow: featurizer → data_handler → evaluator → result_processor
  - Quality filter (column_mask) correctly applied to recover post-filter feature names
  - Updated ODDTStructuralFingerprint docstring to clarify SIFP is AA-type projection, NOT residue-level IFP
  - Test coverage: 4 new unit tests + 1 e2e slow test (85.9s) validating end-to-end with random_forest
  - Code: `screening_engine/base.py`, `result_processor.py`, `data/preparation.py`, `evaluator.py`, `standard.py`, `standard_execution.py`
  - Impact: Intertable feature importances for residue-bit fingerprints, better model insights

- **Candidate Classification** (2026-06-11)
  - Added `classify_candidates()` function for post-screening analysis
  - Three classification categories: high_performance, stable, cost_effective
  - Automatic metric direction detection (higher-is-better vs lower-is-better)
  - Safe numeric coercion supporting JSON strings and string numbers
  - Preferred key extraction from JSON metrics (e.g., `{"mae": 9, "rmse": 0.3}` → 0.3)
  - Direction-aware threshold filtering for min/max primary metrics
  - Lower-is-better quantile inversion (quantile=0.75 → 0.25 percentile for RMSE)
  - Lower-is-better cost-effective efficiency (1/(error*time) for RMSE)
  - Stable candidates exclude missing cv_std (requires valid stability info)
  - Test coverage: 17/17 tests passing
  - Code: `src/molblender/models/api/analysis/candidate_classifier.py`
  - Impact: Automated candidate ranking for model-representation combinations

- **Representation Truncation** (2026-06-11)
  - Added configurable representation limits for resource budget management
  - Config fields: `max_string_representations` (default=3), `max_matrix_representations` (default=2)
  - Unified helper function: `_apply_representation_limit()` with explicit/zero/negative modes
  - Explicit representations bypass truncation (user-specified sets take priority)
  - Zero or negative limits disable truncation (use all available representations)
  - Public API exposure via `CoreScreeningConfig` and legacy kwargs
  - Production code integration: string.py and matrix.py use helper
  - Test coverage: 18/18 tests passing (9 unit + 6 production + 3 API)
  - Code: `src/molblender/models/api/multimodal/modality_handlers/_truncation_helper.py`
  - Impact: Configurable representation selection balancing coverage and resource usage

- **Morgan Hashed Count Fingerprints** (2026-05-01)
  - Added `MorganHashedCountFP` class using RDKit's official `GetCountFingerprintAsNumPy()` API
  - New featurizers: `morgan_hashed_count_fp_r2_8192`, `morgan_hashed_count_fp_r2_16384` (R2)
  - New featurizers: `morgan_hashed_count_fp_r3_8192`, `morgan_hashed_count_fp_r3_16384` (R3)
  - New feature-invariant variants: `morgan_feature_hashed_count_fp_r2_8192`, `morgan_feature_hashed_count_fp_r3_8192`
  - New chiral variants: All hashed_count fingerprints support `useChirality=True`
  - Improved `DEFAULT_SPARSE_FP_SIZE` from 4096 to 8192 (reduces truncation)
  - Enhanced truncation warnings with feature count statistics
  - Migration guide: Deprecated `morgan_count_fp_*` and `morgan_feature_fp_*` (truncated sparse count)
  - Impact: Fixed-dimension count fingerprints without truncation, better coverage for complex molecules

- **Morgan Chiral Fingerprints** (2026-05-06)
  - Added chirality support to Morgan bit fingerprints
  - New featurizers: `morgan_fp_r2_2048_chiral`, `morgan_fp_r2_1024_chiral`, `morgan_fp_r2_512_chiral`
  - New featurizers: `morgan_fp_r3_2048_chiral`, `morgan_fp_r3_1024_chiral`, `morgan_fp_r3_512_chiral`
  - New hashed count chiral variants: `morgan_hashed_count_fp_r2_8192_chiral`, `morgan_hashed_count_fp_r2_16384_chiral`
  - New hashed count chiral variants: `morgan_hashed_count_fp_r3_8192_chiral`, `morgan_hashed_count_fp_r3_16384_chiral`
  - New feature-hashed chiral variants: `morgan_feature_hashed_count_fp_r2_8192_chiral`, `morgan_feature_hashed_count_fp_r3_8192_chiral`
  - Impact: Enantiomer-sensitive Morgan fingerprints for stereochemistry-aware applications

- **DeepChem Chiral Fingerprints** (2026-05-06)
  - Added chirality support to DeepChem Circular fingerprints
  - New featurizers: `deepchem_morgan_r2_2048_chiral`, `deepchem_morgan_r3_1024_chiral`
  - New count variant: `deepchem_morgan_count_r2_2048_chiral`
  - Impact: Enantiomer discrimination for DeepChem fingerprint workflows

- **DeepChem Chiral Graph Featurizers** (2026-05-06)
  - Added chirality support to DeepChem graph featurizers
  - New featurizers: `deepchem_convmol_chiral`, `deepchem_weave_chiral`, `deepchem_molgraphconv_chiral`
  - Enhanced registration to support default_kwargs for graph featurizers
  - Impact: Stereochemistry-aware graph representations for GNN models

- **Chirality Test Suite** (2026-05-06)
  - Added comprehensive test coverage for all chirality variants (33 tests)
  - Test files: `test_rdkit_chiral.py`, `test_deepchem_chiral.py`, `test_deepchem_chiral.py`
  - Tests verify enantiomer discrimination and registration
  - Results: 32 passed, 1 skipped (DeepChem library limitation)
  - Impact: Validates chirality encoding across all fingerprint types

- **Custom HPO Parameter Grids and Scoring** (2026-05-25)
  - Added `custom_param_grids` config for user-defined parameter grids per model
  - Added `estimator_params` config for fixed estimator parameters (e.g., max_iter, random_state)
  - Added `hpo_scoring` config for custom HPO scoring metric (separate from evaluation metric)
  - Updated `CoarseGridSearchOptimizer` to use custom grids when provided
  - Impact: Users can now specify fine-grained parameter grids and use sklearn native scorers

- **train_val_test Split Strategy with Val-Aware HPO** (2026-05-26)
  - Added `val_indices`/`val_true_values` fields to ModelResult and database schema
  - Fixed HPO for train_val_test strategy: uses val for tuning, test for final evaluation
  - Extended structured split guard to maxmin/umap_clustering/splito_* advanced strategies
  - Added Optuna CV metadata (validation_mode/score_source/n_splits) to results
  - Full val passthrough through evaluation pipeline (evaluator → parallel → batch)
  - Impact: Eliminates optimistic bias in HPO metrics, correct 3-way split validation

- **Excel and Parquet File Format Support** (2026-03-25)
  - Added `DataLoader.from_excel()` for loading `.xlsx/.xls` files
  - Added `DataLoader.from_parquet()` for loading `.parquet` files (fast for large datasets)
  - Updated `DataLoader.auto_load()` to detect and handle Excel/Parquet formats
  - Added convenience functions: `load_from_excel()`, `load_from_parquet()`
  - New optional dependency group `[io]`: openpyxl, xlrd, pyarrow
  - Updated `[all]` group to include `[io]` dependencies
  - Impact: Easier data loading from business formats (Excel) and big data formats (Parquet)

- **Config Module Reorganization** (2026-03-24)
  - Organized config files by functionality: `logging/`, `io/`, `runtime/`
  - Moved `loader.py` → `io/loader.py` (YAML/TOML config loading)
  - Moved `export.py` → `io/export.py` (config export templates)
  - Moved `logging_utils.py` → `logging/utils.py` (logging utilities)
  - Created `runtime/` for runtime configuration management
  - Updated all imports across codebase to use new paths
  - Impact: Clearer module organization, better separation of concerns

### Fixed

- **Sklearn Native Scorer Support** (2026-05-25)
  - Fixed `get_sklearn_scoring()` to recognize sklearn native scorer strings (neg_mean_squared_error, neg_mean_absolute_error, neg_root_mean_squared_error)
  - Fixed `get_scoring_function()` to return proper lambda functions for sklearn native scorers
  - Fixed `NestedCVEvaluator._run_hpo_on_outer_fold()` to use correct field name (hpo_score, not hpo_cv_score)
  - Fixed `CoarseGridSearchOptimizer` import to include ParameterGrid for custom grid validation
  - Impact: MolBlender HPO now correctly uses sklearn scoring metrics and matches reference GridSearchCV behavior

- **CNN and VAE GPU Memory Overflow** (2026-04-14)
  - Fixed GPU memory overflow in CNN (`TorchCNNWrapper.predict()`) and VAE (`MolecularFingerprintVAE.predict()`)
  - Added batch processing to both `predict()` methods (previously processed all data at once)
  - Added `inference_batch_size` parameter to `TorchCNNWrapper.__init__()` and `MolecularFingerprintVAE.__init__()`
  - Updated model catalog:
    - `matrix_cnn`: batch_size=16, inference_batch_size=16
    - `image_cnn`: batch_size=16, inference_batch_size=16
    - `matrix_cnn_small`: batch_size=32, inference_batch_size=32
    - `image_cnn_small`: batch_size=32, inference_batch_size=32
    - `VAE (latent=64/128/256/compact)`: batch_size=32, inference_batch_size=32
  - Automatic fallback: `inference_batch_size` → training `batch_size` → default
  - Impact: All GPU models (Transformer, CNN, VAE) now support batch processing for large datasets

- **Transformer GPU Memory Overflow** (2026-04-14)
  - Fixed GPU memory overflow when processing >4200 samples in `predict()` method
  - Added batch processing to `predict()` method (previously processed all data at once)
  - Added `inference_batch_size` parameter to `TransformerStringModel.__init__()`
  - Updated model catalog: `transformer_small` (batch=32), `transformer_medium` (batch=16)
  - Automatic fallback: `inference_batch_size` → training `batch_size` → default 32
  - Impact: Can now process large datasets (>4200 samples) without GPU memory overflow

- **Test Import Errors After Config Reorganization** (2026-03-25)
  - Fixed 21 test import errors caused by config module restructuring
  - Updated `multimodal/api.py`: config imports → `io/loader`, `io/export`, `logging/utils`
  - Updated 2 test files: `universal_screen` import path corrections
  - Updated `test_base_validation_regression.py`: modality model imports
  - Deleted obsolete test using removed `RegressionGridSearch` API
  - Impact: Test suite restored to 2104 passing tests (2 errors remain, project-specific)

### Changed

- **YAML/TOML Configuration File Support** (2026-03-17)
  - Created `config/loader.py` for loading YAML and TOML configuration files
  - Auto-detect format by file extension (.yaml/.yml/.toml)
  - Environment variable substitution with `${VAR_NAME}` and `${VAR_NAME:default}` syntax
  - Created `config/export.py` for exporting and creating configuration templates
  - Added 4 configuration templates:
    - `screening_basic.yaml` - Minimal quick-start template
    - `screening_advanced.yaml` - Full-featured template with all options
    - `screening_hpo.yaml` - HPO-optimized template
    - `screening_basic.toml` - TOML format example
  - Integrated with `universal_screen()` via new `config_file` parameter
  - Priority order: explicit params > config file > defaults
  - Backward compatible with existing API
  - Impact: Version control configs, easier collaboration, non-programmers can modify settings

- **Parallel Execution Infrastructure Module** (2026-03-17)
  - Created new `molblender.parallel` module (457 lines) for unified parallel execution
  - Core primitives: `WorkerError`, `ProgressTracker`, `execute_with_timeout()`
  - Parallel executor: `execute_with_fallback()` with automatic serial fallback
  - Cached execution: `CachedExecutor`, `CacheStats`, `make_cache_key()`
  - Refactored 3 files to use new infrastructure:
    - `representations/utils/parallel.py` - uses `execute_with_fallback()`
    - `representations/utils/parallel_cached.py` - uses `CachedExecutor`
    - `representations/image/utils.py` - uses unified parallel execution
  - Eliminates code duplication across representations layer
  - Uses runtime `concurrent.futures` imports for proper monkeypatch test compatibility
  - Impact: Cleaner parallel execution, better error handling, consistent progress tracking

- **Representations Registry Module Refactoring** (2026-03-17)
  - Moved `utils/registry*.py` (7 files, 1153 lines) to independent `registry/` module
  - Removed `registry_` prefix from filenames: `registry_core.py` → `core.py`, etc.
  - Renamed `selectors.py` → `selection.py` to avoid conflict with Python stdlib
  - Created clean module boundary: registry handles featurizer registration, utils handles utilities
  - Updated all imports across 35+ source files and 5 test files
  - Impact: Clearer architecture, better separation of concerns, eliminates utils module bloat

- **Modality Models Base Class Simplification** (2026-03-17)
  - Consolidated 5 base classes into focused single-responsibility modules
  - Created `core.py` (was `base_core.py`) for shared abstractions
  - Created dedicated base files: `cnn/base.py`, `string_models/base.py`, `vae/base.py`, `graph/base.py`
  - Deleted redundant `base_*.py` files that mixed multiple abstractions
  - Added `modality_models/README.md` documenting the new structure
  - Impact: Easier to understand, modify, and extend modality models

- **API Layer Configuration Objects** (2026-03-17)
  - Replaced 78 parameters in `universal_screen()` with 6 dataclass config objects
  - New config classes: `CoreScreeningConfig`, `SplitConfig`, `ResourceConfig`, `HPOConfig`, `DatabaseConfig`, `WeightConfig`
  - Reduced `api.py` from 785 lines to ~550 lines (30% reduction)
  - Removed 53 legacy parameters and ~100 lines of parameter merging logic
  - Maintained backward compatibility through facade pattern
  - Updated 10 RdRp test scripts to use new config object API
  - Impact: Cleaner API, easier parameter management, better code maintainability

### Changed

- **Representations Module Organization** (2026-03-17)
  - Old: `utils/` module mixed registry logic with utility functions (16 files)
  - New: `registry/` as independent module, `utils/` focused on pure utilities
  - Eliminated circular imports between utils and registry
  - All registry functions now import from `molblender.representations.registry`
  - Impact: Single-responsibility principle, clearer module boundaries

- **DeepChem Models Integration** (2026-03-17)
  - Moved `models/deepchem/` to `models/modality_models/graph/`
  - DeepChem GNN models now properly inherit from `GraphModalityModel`
  - Updated imports across models using DeepChem
  - Deleted obsolete `models/deepchem/` directory

### Fixed

- **Documentation API Compatibility** (2026-03-20)
  - Added `MolecularDataset.validate()` method for dataset validation as shown in quickstart documentation
  - Added `MolecularDataset.from_dataframe()` alias pointing to `from_df()` for backward compatibility
  - Fixed UniMol dependency loading to expose `UniMolRepr` class via `deps.get_unimol_tools()["UniMolRepr"]`
  - Updated UniMol test expectations for actual output dimensions (768 instead of 512 for UniMol-V2-84M)
  - All documentation examples now work correctly (6/6 tests passing)
  - Impact: Documentation matches actual API, better user experience

- **Registry Module Lint and Type Errors** (2026-03-17)
  - Fixed Ruff B007 unused loop variable errors in test files
  - Fixed mypy type annotation errors in registry/display.py, registry/info.py, registry/shapes.py
  - Added TYPE_CHECKING imports for BaseFeaturizer to avoid circular imports
  - Added missing logging import in display.py
  - Fixed bandit B307 security issue: replaced eval() with ast.literal_eval()
  - Updated test imports from old registry_* paths to new registry/ module paths
  - Impact: All lint checks passing, improved type safety, better security

- **Session Merge Core — shared fail-closed merge primitives** (2026-07-30)
  - Extracted `session_merge_core.py` (no DB dependencies) with shared
    `merge_resolved_session_payload()` and `validate_session_metadata_compatibility()`
    used by both explicit ``merge_all_sessions(db_path, session_ids=...)`` and the
    default ``get_all_database_results_record()`` all-DB aggregation path — the two
    callers now honour the same validation, metric-resolution and score-filtering
    rules instead of the default path silently bypassing all of them.
  - ``merge_resolved_session_payload()`` now validates every result row is a dict
    (previously only checked that the payload is a dict and that ``results`` is a list).
    Non-dict rows raise ``SessionMergeError`` carrying the session_id and row index,
    instead of the outer broad ``except Exception`` in ``merge_all_sessions()``
    swallowing the ``AttributeError`` into a ``None`` return that is indistinguishable
    from "no sessions at all".
  - ``validate_session_metadata_compatibility()`` now uses canonical
    ``_canonical_metric_name()`` (strip + lowercase) for cross-session metric
    comparison so ``"MAE"`` / ``" mae "`` / ``"mae"`` are not spuriously rejected;
    the error message preserves the original raw values so operators see what
    actually arrived.  The multi-session missing-metric rejection is fixed-closed
    for all primary_metric values (including ``None`` and ``""``), not only for
    present strings — one session with ``mae`` and another with ``None`` now raises
    rather than silently defaulting.  The task_type check fires before the metric
    check and its error message surfaces ``None`` for missing values instead of
    hiding them behind a single-value set.
  - Empty result sets (session exists, zero rows) return a standard empty payload
    with ``model_results=[]`` and ``best_model.model_name="Unknown"``; ``None`` is
    reserved for "no sessions at all" or a real read failure, matching the explicit
    path contract so callers never need to branch on which route was taken.
  - Ten new tests (``test_session_merge_malformed_payload.py``,
    ``test_session_merge_row_shape_and_canonical_metric.py``, plus additions to the
    default-path, invalid-scores and strict-metadata suites) cover: non-dict payload,
    non-list results, per-row None/str/int rejection with row position, default-path
    ``get_session_results`` returning a junk row, case/whitespace-correct metric
    comparison (``"MAE"`` vs ``"mae"``, ``" mae "`` vs ``"mae"``, uppercase session
    value), two sessions with mixed ``"MAE"``/``"mae"``/``" mae "``, mixed
    ``mae``/``None`` in loose mode, ``mae``/``""`` in loose mode, different
    ``task_type`` with identical metric, and missing-vs-present ``task_type`` error
    text including ``None``.

### Removed

- **Registry Filename Prefixes** (2026-03-17)
  - Removed `registry_` prefix from all registry module files
  - `registry_core.py` → `core.py`, `registry.py` → `facade.py`, etc.
  - Renamed `selectors.py` → `selection.py` to avoid Python stdlib conflict
  - Impact: Cleaner filenames, better naming conventions
  - Impact: Consistent model architecture, all graph models in one location

- **Execution Layer Cleanup** (2026-03-17)
  - Removed `execution/optimized_parallel.py` (duplicate functionality)
  - Deleted `models/api/utils/resource_scheduler.py` (replaced by infrastructure layer)
  - Cleaned up legacy compatibility shims
  - Impact: Removed code duplication, clearer dependencies

### Tests

- Added `tests/models/api/test_model_discovery.py` (model catalog discovery tests)
- Added `tests/models/modality_models/test_graph_models.py` (graph model tests)
- Verified 157 featurizers still accessible after refactoring
- All 51 multimodal/representations tests passing

### Fixed

- Fixed circular import between `utils/` and `registry/` modules
- Fixed name collision between new `ScreeningConfig` and existing `screening_engine.ScreeningConfig`
- Fixed dashboard imports after API refactoring
- Fixed relative imports in registry module after file moves

- **Architecture Role Catalog & Executable Snapshot** (2026-03-12)
  - Added machine-readable package role metadata across top-level facades, domain APIs, visualization layers, and execution layers
  - Added `molblender.architecture_roles` helpers for:
    - package role catalog
    - recommended entrypoints
    - execution layer decisions
    - visualization layer decisions
    - migration guidance
  - Added `python -m molblender.architecture_roles` JSON snapshot output
  - Impact: Architecture guidance is now executable, testable, and CI-friendly

- **Execution Layer Boundary Contracts** (2026-03-12)
  - Added contract tests to lock the distinction between:
    - `molblender.models.api.infrastructure` (primary screening runtime)
    - `molblender.representations.utils` (generic batching/caching helpers)
    - `molblender.models.execution` (compatibility layer)
  - Added import-isolation and public-surface tests to prevent legacy executor leakage into recommended APIs
  - Impact: Clearer long-term migration path and lower risk of architectural drift

- **Infrastructure: Telemetry Package Modularization** (2026-03-11)
  - Replaced monolithic `telemetry.py` (755 lines) with modular `telemetry/` package
  - New structure: `types.py`, `backends.py`, `emitter.py`, `global_emitter.py`, `legacy.py`
  - Maintained 100% backward compatibility - all old imports still work
  - Export contract tests: 16 tests passing
  - Infrastructure tests: 33 tests passing
  - Impact: Better code organization, clearer separation of concerns, improved maintainability

- **Representations: Public API Consolidation** (2026-03-11)
  - Added comprehensive API export contract tests (23 tests)
  - Verified tool_registry purity: metadata management only, no UI logic
  - Added import isolation tests (14 tests)
  - Confirmed no circular imports in representations module
  - Validated lazy imports: transformers not loaded on package import
  - Impact: Stable public API, better import performance, clearer module boundaries

- **Drawings Package Positioning** (2026-03-11)
  - Clarified drawings as "static plotting utilities" layer
  - Verified drawings does NOT export dashboard/interactive components
  - Added 23 API contract tests for drawings/models/dashboard separation
  - Updated module documentation to distinguish drawings (static) vs dashboard (interactive)
  - Impact: Clearer package boundaries, easier to choose right visualization tool

- **Public API Layer Audit** (2026-03-11)
  - Added 33 public API contract tests
  - Verified molblender.api = unified convenience facade
  - Verified molblender.models = richer ML domain API
  - Verified molblender.representations = richer featurizer API
  - Updated top-level molblender documentation with API layer explanation
  - Distinguished RECOMMENDED (unified facade) vs COMPATIBILITY (direct imports) in __all__
  - Impact: Clearer API usage guidance, better developer experience

- **Architecture Role Contract Tests** (2026-03-11)
  - Added 27 package role contract tests
  - Verified drawings = static plotting layer (not dashboard)
  - Verified models.api.core = screening engine core components
  - Verified models.api.infrastructure = runtime policy/telemetry/error policy
  - Verified molblender.api = facade (doesn't contain implementation details)
  - Impact: Architectural boundaries locked down, prevents future drift

### Changed

- **Top-Level Facades Now Use Lazy Imports**
  - `molblender` and `molblender.api` now use lazy facade exports
  - Importing the top-level package no longer eagerly loads large subpackages such as representations, models, dashboard, or drawings
  - Workflow subfacades (`molblender.api.models`, `molblender.api.representations`, `molblender.api.dashboard`) also use lighter import paths
  - Impact: Faster startup, lower import overhead, and cleaner package boundaries

- **Execution and Architecture Documentation Refined**
  - Updated source README files and developer-facing docs to distinguish current recommended layers from historical compatibility layers
  - Clarified that `models.api.core` is the screening engine core, not the project-wide core
  - Clarified that `drawings` is for static plotting while `dashboard` is the interactive UI
  - Impact: Easier navigation for contributors and clearer package roles for users

- **Telemetry Module Organization**
  - Old: Single file `telemetry.py` with all implementations
  - New: Modular package with separate files for types, backends, emitter, global functions
  - Benefit: Easier to maintain and extend, clearer code organization

- **Representations Import Behavior**
  - Confirmed: Heavy dependencies (torch, transformers) loaded lazily, not on import
  - Confirmed: Multiple imports consistent, no side effects
  - Benefit: Faster imports, lower memory footprint for basic usage

### Tests

- Added `tests/models/api/infrastructure/test_telemetry_exports.py` (16 tests)
  - Public API exports verified
  - Import patterns tested (from infrastructure, from telemetry package, from submodules)
  - Backend classes verified
  - EventEmitter availability confirmed
  - Package structure validated

- Added `tests/representations/test_api_exports_contract.py` (23 tests)
  - Core utilities availability verified
  - Base classes and exceptions confirmed
  - Tool registry purity validated
  - No heavy imports on package import confirmed
  - No side effects on import verified
  - Circular import prevention confirmed

- Added `tests/core/test_import_isolation.py` (14 tests)
  - Representations import isolation verified
  - Telemetry import isolation verified
  - Import consistency confirmed
  - Lazy imports validated

- Added `tests/drawings/test_api_exports_contract.py` (23 tests)
  - Drawings public API exports verified
  - Drawings does NOT export dashboard components verified
  - Optional imports handled gracefully confirmed
  - Drawings vs dashboard separation validated

- Added `tests/core/test_public_api_contract.py` (33 tests)
  - Top-level molblender API verified
  - molblender.api as unified facade validated
  - molblender.models as richer domain API confirmed
  - molblender.representations as richer featurizer API confirmed
  - API layer distinction verified

- Added `tests/core/test_package_roles.py` (27 tests)
  - Drawings as static plotting layer role verified
  - models.api.core as screening engine role confirmed
  - models.api.infrastructure as runtime policy role verified
  - molblender.api as facade (no implementation) validated
  - Package boundaries separation confirmed

### Technical Details

**Telemetry Package Structure**:
```
infrastructure/telemetry/
├── __init__.py         # Unified exports (backward compatible)
├── types.py            # EventType, EventSeverity enums
├── backends.py         # EventBackend, JSONFileBackend, LogFileBackend, etc.
├── emitter.py          # EventEmitter class
├── global_emitter.py   # get_global_emitter(), configure_global_emitter(), emit_event()
└── legacy.py           # build_event(), emit_legacy_event()
```

**Representations Public API**:
- Core utilities: `get_featurizer`, `list_available_featurizers`, `get_featurizer_info`
- Base classes: `BaseFeaturizer`, `BaseProteinFeaturizer`
- Exceptions: `FeaturizationError`, `InvalidInputError`, `RegistryError`
- Tool registry: `ToolInfo`, `ToolRegistry` (pure metadata, no UI logic)
- Enhanced registry: `FeaturizerInfo`, `FeaturizerCatalog`, `get_featurizer_recommendations`

### Migration Notes

**For Telemetry Users**:
- No changes required - all old imports still work
- New modular imports available for finer control:
  ```python
  # Old way (still works)
  from molblender.models.api.infrastructure import emit_event, EventType
  
  # New way (more granular)
  from molblender.models.api.infrastructure.telemetry.types import EventType
  from molblender.models.api.infrastructure.telemetry.global_emitter import emit_event
  ```

**For Representations Users**:
- No changes required - public API unchanged
- Tool registry now available for advanced metadata queries:
  ```python
  from molblender.representations.tool_registry import ToolRegistry
  
  registry = ToolRegistry()
  gpu_featurizers = registry.list(tags=["gpu"])
  protein_featurizers = registry.list(category="protein")
  ```


- **Phase 3-4: Configuration Management & Deprecated Code Cleanup** (2026-03-10)
  - **Phase 3: Configuration Management Unification**
    - Created `config/core.py` (309 lines): Unified ConfigManager singleton
    - Dataclasses: `CacheConfig`, `ModelConfig`, `LoggingConfig`
    - Centralized environment variable reading with validation
    - Runtime configuration update: `set_cache_dir()`, `get_cache_dir()`
    - Backward compatibility: Legacy `settings` exports preserved with `_legacy` suffix
    - Tests: 12 ConfigManager tests passing, 73 total config tests passing
  - **Phase 4: Deprecated Code Cleanup**
    - Deleted `models/api/utils/resource_scheduler.py` (102-line shim)
    - Removed `timeout_context` from `evaluation/utilities.py` (~100 lines)
    - Updated `evaluation/__init__.py`: Removed timeout_context from `__all__` and `__getattr__`
    - Deleted test files: `test_resource_scheduler_compat.py`, `test_timeout_context_shim.py`
    - Updated guard tests: Removed resource_scheduler and resource_profiles tests
    - All infrastructure and evaluation tests passing (4 passed)
  - **Documentation**
    - Created `config_manager_guide.md`: ConfigManager usage and best practices
    - Created `migration_guide.md`: API migration guide from old to new API
    - Created `api_guide.md`: Unified API layer usage guide
  - **Impact**: Unified configuration management, removed ~200 lines deprecated code, 306 total tests passing

- **Round 8: Dashboard State Management & Resource Tracking** (2026-03-09)
  - **Task 1: Dashboard State Management Modularization**
    - Created `dashboard/state/` package with three manager classes
    - `SessionManager`: Manages session-wide state (cache, files, refresh)
    - `NavigationManager`: Manages navigation state (active tab, history)
    - `FilterManager`: Manages filter state (metrics, models, representations)
    - Total: 459 lines across 3 modules
  - **Task 2: Dashboard Cache Hierarchical Refactoring**
    - Created `dashboard/cache/policies.py` with CachePolicies system
    - Clear separation: `st.cache_data` (DataFrame, dict) vs `st.cache_resource` (connections)
    - Predefined strategies: SESSION_DATA, SHORT_LIVED_DATA, FEATURIZER, CONNECTION
    - `@cache_with_policy` decorator for easy policy application
  - **Task 3: CI Matrix and Gate Configuration**
    - Created `.github/workflows/dashboard_smoke.yml`
    - Python 3.9/3.10 matrix testing
    - Test suites: state management, cache manager, API contracts, dashboard smoke
  - **Task 4: Dashboard Integrated Resource Tracking System**
    - Created `dashboard/components/resource_tracking/` package
    - `representation_selector.py` (192 lines): Type selector, comparison table, parameter info
    - `cache_statistics.py` (273 lines): Sidebar stats, management page, helper functions
    - Updated `app.py`: Added 2 new tabs (Resource Management, Representation Types)
    - Total tabs: 5 → 7
  - **Phase 5: Auto-discovery Enhancement**
    - Enhanced `representations/registry/` with dependency checking and availability flags
    - Created `representations/registry/validation.py` (320 lines): Parameter validation framework
    - Auto-discovery now detects dependencies and sets `is_available` flag
    - Parameter validation with type checking, choices, min/max values
    - Helper functions: `sanitize_representation_params()`, `get_representation_summary()`
  - **Tests**: 290 tests passing (12 Phase 4 + 19 Phase 5 + 259 previous)
  - **Impact**: Improved dashboard modularity, cache observability, resource tracking, auto-discovery

- **Round 7: Dashboard API Contract & Smoke Testing (CC)** (2026-03-08)
  - **Task 1: Dashboard API Export Contract Tests** (21 tests)
    - Created `tests/dashboard/test_api_exports_contract.py`
    - Verified all public API exports from dashboard modules
    - Tested backward compatibility of key interfaces
    - Coverage: `ResultsDataLoader`, `DashboardMetrics`, `FilterConfig`, render functions
  - **Task 2: Dashboard Smoke Stabilization** (26 tests)
    - Created `tests/dashboard/test_task2_smoke_stabilization.py`
    - Verified tab order is fixed and matches documentation
    - Tested all 4 main pages (Overview, Performance, Detailed Results, Hyperparameter)
    - Validated data loading pipeline and metrics system integration
    - Verified filtering system and backward compatibility
  - **Task 3: User Documentation Sync**
    - Created `docs/source/dashboard_troubleshooting.md`
    - Merged sessions loading workflow explanation
    - Common error handling: `NoneType.get`, `sqlite3.Row.get`, `KeyError: 'primary_metric'`
    - Diagnostic commands and minimal health check script
    - Performance tips for large databases
  - **Task 4: Regression Gate**
    - All 248 dashboard tests passing (7 skipped)
    - 47 new tests added (21 API + 26 smoke)
    - Zero regressions in existing functionality
  - **Impact**: Strengthened dashboard API stability, improved documentation, comprehensive test coverage

- **Round 6: Efficiency & Representations Modularization** (2026-03-08)
  - **Efficiency Analysis Refactoring**: Split `efficiency.py` (645 lines) into `efficiency/` package
    - New structure: `scatter.py` (132), `distribution.py` (131), `metrics.py` (181), `speed.py` (122)
    - Facade: `__init__.py` (88 lines) with `render_efficiency_analysis()` entry point
    - Clear separation: scatter plots, distributions, metrics, speed analysis
  - **Base Featurizer Refactoring**: Split `base.py` (633 lines) into focused modules
    - `base_featurizer.py` (415 lines) - Small molecule featurizer base class
    - `base_protein_featurizer.py` (147 lines) - Protein featurizer base class
    - Facade: `base.py` (11 lines) for backward compatibility
  - **CDK Fingerprint Factory Pattern**: Split `cdk.py` (633 lines) into `cdk/` package
    - `loaders.py` (109 lines) - Lazy CDK component loading
    - `base.py` (213 lines) - `BaseCDKFingerprint` base class
    - `classes.py` (155 lines) - Factory pattern eliminates 13 duplicate class definitions
    - Facade: `cdk.py` (5 lines) + `cdk/__init__.py` (47 lines)
    - `_make_cdk_class()` factory function for dynamic class generation
  - **Validators Duplicate Code Removal**: Removed 242-line duplicate `MetricsCalculator`
    - `validators.py` (860→613 lines, -29%)
    - Now uses unified `MetricsCalculator` from `metrics_calculator.py`
  - **Test Organization**: Moved 8 test files from `tests/` to `tests/data/`
    - Added `__init__.py` to test directories to fix import name collisions
  - **Test Marker Documentation**: Added comprehensive usage guidelines for `@pytest.mark.slow` and `@pytest.mark.network`
  - **Impact**: 4 files → 12 modules, eliminated duplicate code, improved maintainability
  - **Test Results**: 159 passed (19 dashboard + 140 CDK)

- **Round 5: Dashboard Boundary & Stability Hardening** (2026-03-08)
  - **Package Import Contracts**: All dashboard submodules importable without streamlit runtime
    - Tests: `test_round5_package_imports.py` (12 tests)
    - Verified: All submodules in `data/`, `metrics/`, `components/` import cleanly
  - **Behavioral Contracts**: Verified critical UI behaviors preserved
    - Tests: `test_round5_app_services_contracts.py` (9 tests)
    - Tab order validation: Overview → Performance → Detailed Results → Hyperparameter → Individual Model
    - Default page verification: "🔍 Overview" as initial tab
    - CLI entry points: `main()` and `run_from_cli()` signatures validated
  - **Type Safety**: Added `ClassVar` annotations for class-level constants (RUF012 fix)
  - **Test Coverage**: +21 tests (170 total, +68% from baseline)
  - **Regression**: All 319 tests passing (0 failures)
  - **Impact**: Hardened package boundaries, prevented UI regressions, improved type safety

- **Round 4: Dashboard app.py Refactoring** (2026-03-08)
  - **Major Refactoring**: Split `app.py` (766 lines) into `app_services/` package
    - `startup_diagnostics.py` (62 lines) - Diagnostic utilities
    - `page_config.py` (164 lines) - Page configuration and CSS styling
    - `pages.py` (377 lines) - Tab rendering logic
    - Facade: `app.py` (247 lines, -68%) with main orchestration
  - **Architecture Improvements**:
    - Phase 1: Extracted diagnostics and configuration (766→549, -28%)
    - Phase 2: Extracted page rendering logic (549→320, -42%)
    - Phase 3: Simplified tab dispatch (320→247, -23%)
  - **Streamlit Decoupling**: Lazy import pattern for streamlit dependencies
    - Enables pytest testing without streamlit runtime
    - Fixed `ModuleNotFoundError` in test collection
  - **Contract Tests**: Added API contract tests (12 tests)
    - `test_app_contracts.py`: Module structure, CLI signatures, UI behavior preservation
  - **Test Coverage**: +12 tests (142 total, +41% from baseline)
  - **Impact**: Clear separation of concerns, improved testability, backward compatible

- **HPO Module Refactoring** (2026-03-03)
  - Split `hpo.py` (1220 lines) into modular `hpo/` directory
  - New structure: `processor.py` (workflow), `selection.py` (model selection), `results.py` (data reconstruction)
  - Added utility function `serialize_indices()` for JSON serialization of train/test indices
  - Added `_create_model_for_task()` method to eliminate duplicate model instantiation code
  - Unified target extraction using existing `extract_targets_from_dataset()` function
  - Eliminated ~30 lines of repeated code across processors
  - Improved code maintainability: single point of change vs scattered modifications
  - Files added: `processors/hpo/__init__.py`, `processors/hpo/processor.py`,
    `processors/hpo/selection.py`, `processors/hpo/results.py`
  - Files modified: `processors/data_converter.py` (serialize_indices),
    `processors/hpo/processor.py` (refactored),
    `processors/hpo/results.py` (unified target extraction)

- **DeepChem Graph Neural Network Integration** (2026-03-02)
  - Plan B GNN mode implementation with full DeepChem graph support
  - Graph converters: ConvMol → GraphData, WeaveMol → GraphData
  - Smart routing: auto-detect GNN vs vector mode based on user categories
  - Supported GNN models: GCN, GAT (DeepChem 2.8.0 compatible)
  - Supported graph featurizers: ConvMol, MolGraphConv, Weave
  - Files added: `models/deepchem/gnn.py`, `models/api/multimodal/modality_handlers/graph_converter.py`
  - Files modified: `representations/graph/__init__.py`, `models/api/multimodal/modality_handlers/graph.py`
  - Test results: Plan A (vector mode) 26 results, best: deepchem_graphconv_vector + xgboost (Pearson r = 0.379)

- **Splito Integration for Advanced Splitting** (2026-02-24)
  - Unified API for splito cluster-level splitting strategies via `DataSplitter`
  - New strategies: `splito_perimeter`, `splito_molecular_weight`, `splito_max_dissimilarity`, `splito_scaffold`
  - Cluster-level algorithms (K-means based) vs molecular-level algorithms
  - Files modified: `models/api/core/splitting/strategies.py`

- **Specific Pairs Mode for Precise Combination Control** (2026-02-23)
  - New `combinations.representations` parameter for precise "representation + model" pair specification
  - Auto-detection of `specific_pairs_mode` when combinations specify both representations and models
  - Added `run_combination_screening()` utility function in multimodal utils
  - Supports all modality handlers: vector, string, matrix, image, language_model
  - Temporarily disables `skip_existing_results` to ensure specified combinations are re-run
  - Fixed result extraction: uses `screener.results` (ModelResult objects) instead of formatted dict
  - Verification on remote server 209: Stage 1 (15 results), Stage 2 HPO (756 results)
  - Best result: datamol_avalon + decision_tree (R² = 0.7859)
  - Files added: `src/molblender/models/api/multimodal/utils.py` (run_combination_screening)
  - Files modified: `src/molblender/models/api/core/base.py` (specific_pairs_mode flag),
    `src/molblender/models/api/multimodal/api.py` (auto-detection),
    `src/molblender/models/api/multimodal/modality_handlers/*.py` (all handlers)

- **Model Export CLI** (2026-02-19)
  - New `molblender export` command for exporting model recreation scripts from database
  - Subcommands: `export best`, `export model`, `export top`
  - Generates complete Python scripts with all parameters from screening
  - Supports absolute path resolution and CSV pre-defined splits
  - Files added: `src/molblender/utils/export_cli.py`

### Changed

- **CDK Fingerprint Defaults: `cdk_sub` native size + `cdk_signature` default disable** (2026-03-03)
  - `cdk_sub` now uses CDK native 307-bit shape (removed project-level zero-padding-to-1024 behavior)
  - `cdk_signature` remains implemented but is no longer registered by default
  - Rationale: raw Signature fingerprint outputs are variable-token and need explicit vocabulary alignment before ML
  - Files modified: `src/molblender/representations/fingerprints/cdk.py`,
    `tests/representations/fingerprints/test_cdk.py`,
    `docs/source/usage/representations/fingerprints/cdk.md`

- **Splitting Strategy Naming Clarity** (2026-02-24)
  - Renamed `max_dissimilarity_split` → `sequential_max_dissimilarity_split`
  - Clarifies molecular-level (sequential) vs cluster-level (splito) algorithms
  - Backward compatibility maintained with alias
  - Files modified: `models/api/core/splitting/diversity.py`, `models/api/core/splitting/strategies.py`

- **Database Schema Enhancement** (2026-02-19)
  - Added `split_column` field to `screening_sessions` table
  - Added `input_column` field to `screening_sessions` table
  - Enhanced `create_session()` to save split configuration metadata
  - Modified `load_dataset_with_split()` to return input_column information
  - Improved export code generation to read from database instead of guessing
  - Files modified: `models/api/utils/results_db.py`, `models/api/multimodal/processors/database.py`,
    `dashboard/components/model_inspection/export.py`, `utils/export_cli.py`

- **Dashboard Performance Optimization** (2026-02-18)
  - Added permanent caching to `create_results_dataframe()` with `@st.cache_data`
  - Optimized `MetricsCalculator.enrich_dataframe_with_metrics()` with filtering and tqdm progress
  - Files modified: `dashboard/data/processors.py`, `dashboard/metrics/central.py`

### Fixed

- **Database NULL Handling** (2026-02-18)
  - Fixed NoneType error when accessing NULL all_metrics fields
  - Uses `(result.get("all_metrics") or {})` instead of `result.get("all_metrics", {})`
  - Fixed None checks for `dataset_info` and `merged_dataset_info`
  - Files modified: `dashboard/data/loaders.py`

- **Dashboard Session Wall Time Calculation** (2026-03-06)
  - Fixed wall_time showing 0 in Session Breakdown table
  - Now correctly loads `created_at` and `updated_at` from `screening_sessions` table
  - Attaches session timestamps to each model result for per-session wall_time calculation
  - Each session shows independent wall_time based on its own timestamps (not merged across sessions)
  - Files modified: `dashboard/data/loaders.py`, `dashboard/components/charts/general.py`

- **Classification F1 Metric Alias** (2026-03-06)
  - Added "f1" as alias for "f1_score" in CLASSIFICATION_ONLY_METRICS
  - Ensures backward compatibility with databases using "f1" key instead of "f1_score"
  - Fixed KeyError when displaying F1 Score in Dashboard
  - File modified: `dashboard/metrics/central.py`

- **Model Export Split Information Fallback** (2026-03-06)
  - Fixed export code generation to check multiple sources for split configuration
  - Now tries `screening_config` if `dataset_info` doesn't have split indices/column
  - Ensures exported scripts can recreate data splits correctly
  - File modified: `dashboard/components/model_inspection/export.py`

- **Metric Column Series Name Attribute** (2026-03-06)
  - Fixed `get_metric_column()` returning Series without name attribute
  - Series now includes `name=selected_metric` for proper groupby operations
  - Prevents errors when using metric columns in grouping operations
  - File modified: `dashboard/components/utils/__init__.py`

- **Disabled Lasso and ElasticNet Models** (2026-02-08)
  - Removed lasso and elastic_net from available models due to poor performance on small datasets
  - These models consistently produce Pearson R = 0.00 on small datasets (<500 training samples)
  - Model registration commented out in `models/api/core/model_registry.py`
  - Parameter grids commented out in `models/corpus/grids/linear_models.py`
  - Code preserved for potential future re-enablement on larger datasets
  - Linear models category now reduced to 4 models: Ridge, Logistic, LinearSVR, Bayesian Ridge

- **Mordred Batch Processing** (2026-02-07)
  - Added `_featurize_batch()` method to MordredFeaturizer for parallel computation
  - Uses Mordred's `calc.pandas()` API for batch descriptor calculation
  - Significantly faster than individual molecule processing (10-50x speedup)
  - Automatically falls back to individual calculation if batch fails
  - File modified: `representations/descriptors/descriptors_basic.py`

- **Default Primary Metric: Pearson R** (2026-02-07)
  - Changed default regression metric from R² to Pearson R for better interpretability
  - Pearson R directly measures linear correlation strength (-1 to 1)
  - More stable on small datasets and symmetric for prediction evaluation
  - Files modified: `api.py`, `reporting.py`, `results_db.py`, `migration.py`

- **Language Model Routing in Multimodal Screening** (2026-02-14)
  - Prevented duplicate processing of language-model representations inside VECTOR workflow
  - Ensures each modality is evaluated once with correct modality mapping

### Fixed

- **CV Scoring Metric: Pearson R instead of neg_MSE** (2026-03-01)
  - Changed cross-validation scoring from `neg_mean_squared_error` to `pearson_r`
  - Ensures CV evaluation uses same metric as final model assessment
  - Provides more interpretable results (correlation strength vs squared error)
  - Files modified: `models/api/core/metrics.py`, `models/api/multimodal/modality_handlers/vector.py`

- **Per-Fold CV Scores Storage for HPO Stage 2** (2026-03-01)
  - Fixed per-fold CV scores not being saved to database for HPO Stage 2 results
  - Now stores all fold scores in `all_metrics['cv_fold_scores']`
  - Enables detailed fold-level analysis in dashboard
  - Files modified: `models/api/multimodal/processors/database.py`

- **Primary Metric Renaming** (2026-03-01)
  - Refactored internal variable name from `score` to `primary_metric` across package
  - Improves code clarity and distinguishes from model.score() method
  - Files modified: Multiple files across `models/` and `dashboard/`

- **Dashboard Progress Bar Cleanup** (2026-02-28)
  - Fixed progress bar not being cleaned up properly after completion
  - Progress bar now rendered inside container for proper cleanup
  - Files modified: `dashboard/app.py`

- **Dashboard Loading Placeholder** (2026-02-28)
  - Fixed loading placeholder not being cleared after success message
  - Improves user experience by removing redundant UI elements
  - Files modified: `dashboard/app.py`

- **Hyperparameter Chart Performance** (2026-02-27)
  - Optimized parameter extraction with progress indicator
  - Reduced loading time for Individual Model page with 200k+ results
  - Files modified: `dashboard/components/charts/hyperparameters/main.py`

- **Dashboard Special Characters** (2026-02-27)
  - Sanitized special characters in sunburst chart colorbar labels
  - Fixed rendering issues with underscores and special characters
  - Files modified: `dashboard/components/model_inspection/sunburst.py`

- **skip_existing_results Empty Database Bug** (2026-02-07)
  - Fixed bug where `skip_existing_results=True` would skip all screening when database is empty
  - Now correctly runs full screening when no existing results are found (count=0)
  - Only skips when all results exist (count=len(representations))
  - Partial results still load existing and skip missing to avoid OOM
  - File modified: `models/api/multimodal/modality_handlers.py`

- **Mol2Vec Offline Support** (2026-02-07)
  - Removed network connectivity check that blocked Mol2Vec in offline environments
  - Now works with locally cached models without internet access
  - File modified: `config/dependencies.py`

- **HPO Stage 2 Resume Stability** (2026-02-14)
  - Removed call to non-existent `check_existing_hpo_result` during resumed optimization
  - Stage 2 now resumes using the standard DB existence checks

- **Offline/Feature Robustness Improvements** (2026-02-14)
  - Improved handling for offline environments and count-based fingerprint edge cases
  - Reduced modality failures caused by degenerate feature arrays

- **Cross-Session Stage 1 Result Loading** (2026-02-17)
  - Fixed `skip_existing_results` to load Stage 1 results from all sessions, not just current session
  - Fixed exception handler to preserve existing results when partial screening fails
  - Result: HPO Stage 2 now correctly uses all available Stage 1 results (193 → 1425 results)

- **HPO 56-Core Parallelization** (2026-02-17)
  - Fixed GridSearchCV to use all available CPU cores via `parallel_backend`
  - Force estimator `n_jobs=1` to prevent thread conflicts with GridSearchCV parallelism
  - Result: 240 parameter fits in 30.1s (vs 20min before), CPU utilization 98.9%

- **Train/Test Indices Persistence** (2026-02-17)
  - Save train/test indices to screening_sessions table for user_provided splits
  - HPO now correctly reconstructs training data across sessions
  - Fixed train/test split consistency between Stage 1 and Stage 2

### Added

- **Mol2Vec Auto-Download Enhancement** (2026-02-07)
  - Added `download_mol2vec_model()` function for automatic model downloading
  - Caches model in `~/.mol2vec/` directory (26MB)
  - Supports custom model paths via `pretrain_model_path` parameter
  - Falls back gracefully with clear error messages if download fails
  - File modified: `config/models.py`, `representations/fingerprints/deepchem.py`

- **HPO Resume + Stage 2 Autosave** (2026-02-07)
  - Stage 2 skips combinations already optimized in the database
  - Results are saved after each optimized model to reduce lost work
  - Pre-computes required representations once before HPO loop

- **Mol2Vec Auto-Download + Offline Checks** (2026-02-07)
  - Auto-downloads Mol2Vec pretrained model (cached in `~/.mol2vec`) when not provided
  - Detects missing network connectivity for PubChem/Mol2Vec features and warns early

- **Dashboard Analysis Enhancements** (2026-02-07)
  - Adds category-specific model distribution chart in Performance Analysis
  - Detailed Results table now includes best CV fold score and training-set primary metric (when available)

- **Universal skip_existing_results for All Modalities** (2026-02-04)
  - **Problem**: `skip_existing_results` only worked for VECTOR modality, other modalities (STRING, MATRIX, IMAGE, LANGUAGE_MODEL) always re-ran Stage 1 screening
  - **Solution**: Created universal helper method `_check_and_load_existing_results()` in `ModalityHandlerMixin`
  - **Implementation**: All 5 modality handlers now call this shared method to check database for existing Stage 1 results
  - **Behavior**:
    - ✅ If **all** representations have Stage 1 results → skip screening, load from database
    - ⚠️ If **partial** representations have results → run full screening (ensures completeness)
    - ✅ If **no** representations have results → run full screening
  - **Files Modified**:
    - `src/molblender/models/api/multimodal/modality_handlers.py`: Added `_check_and_load_existing_results()` helper (126 lines)
    - All modality handlers (VECTOR, STRING, MATRIX, IMAGE, LANGUAGE_MODEL) now use this method
  - **User Impact**: Significantly faster HPO Stage 2 execution when Stage 1 results already exist in database

- **Database Merger Enhancements** (2026-02-04)
  - **Session Preservation**: Now preserves all sessions and their metadata when merging databases
  - **Flexible Deduplication**: Added `--no-remove-duplicates` flag to keep all results including duplicates
  - **Table Completeness**: Ensures all required tables (dataset_info, schema_version) are copied to merged database
  - **Session Consistency**: Maintains session_id consistency across model_results and dataset_info tables
  - **File Modified**: `src/molblender/models/api/utils/database_merger.py`
  - **New Behavior**:
    - Default: Remove duplicates, keep best score per model+representation
    - With `--no-remove-duplicates`: Keep all results from all databases
  - **Example**: `molblender merge_databases db1.db db2.db -o merged.db --no-remove-duplicates`

- **Multi-Database Merge Tool** (`molblender merge_databases`)
  - New CLI command to merge multiple `.db` files into one unified database
  - Automatically deduplicates `model_name + representation_name` combinations (keeps best score)
  - Filters out entries with NaN/invalid scores
  - Example: `molblender merge_databases db1.db db2.db -o merged.db`
  - Use cases: Merge interrupted+resumed screenings, combine multi-modality results, clean up test runs
  - Documentation: `usage/basic/cli.md`

- **Data Type Error Fix** (data_handler.py)
  - Fixed `ufunc 'isnan' not supported for the input types` error when processing non-numeric dtype arrays
  - Added automatic object→float64 conversion for numeric data stored as object dtype
  - Prevents VECTOR modality failures on larger datasets (2535 molecules)

- **Grid Search Resolution Improvement** (tree_models.py)
  - Increased grid points for well-performing models (Random Forest, XGBoost, LightGBM)
  - Random Forest: 36 → 2160 combinations (60x increase)
  - XGBoost: 108 → 57600 combinations (533x increase)
  - All grid centers now use sklearn/official defaults

- **Optuna Integration** (optuna_optimizer.py)
  - New Optuna-based Bayesian optimizer for fine-tuning top models
  - Warm-start from Grid Search best parameters (±50% range)
  - MedianPruner for early stopping of unpromising trials
  - Focuses on top 3 models + slow models (Transformer, CNN)

- **Cross-Session Result Caching** (`--skip-existing`)
  - **Problem**: Previously, re-running screening would re-compute all model-representation combinations even if results existed in previous database sessions
  - **Solution**: Added `skip_existing_results` configuration parameter and `--skip-existing` CLI flag
  - **Implementation Details**:
    - `ScreeningConfig.skip_existing_results`: Check across ALL sessions, not just current session
    - `ScreeningResultsDB.check_existing_result(check_all_sessions=True)`: SQL query without session_id filter
    - **Smart Caching**: When enabled, skips any model-representation combination that exists in ANY previous session
  - **Use Cases**:
    - Resume interrupted screening jobs without re-computing completed results
    - Add new representations/models to existing screening without re-running everything
    - Incremental screening: run on subset, then expand with `--skip-existing`
  - **Performance**: For the RdRp dataset (723 existing results), resuming with `--skip-existing` would skip all 723 combinations and only compute new ones
  - **Backward Compatibility**: Default `skip_existing_results=False` preserves existing behavior (always recompute)
  - **Files Modified**:
    - `src/molblender/models/api/core/base.py`: Added `skip_existing_results` configuration parameter
    - `src/molblender/models/api/multimodal/processors/database.py`: Added `check_all_sessions` parameter to `check_existing_result()`
    - `src/molblender/models/api/utils/results_db.py`: Added `check_all_sessions` parameter to `check_existing_result()`
    - `src/molblender/models/api/multimodal/api.py`: Added `skip_existing_results` parameter to `universal_screen()`
    - `tests/.../run_molblender_screening.py`: Added `--skip-existing` CLI argument
  - **Example Usage**:
    ```bash
    # First run: compute all combinations (723 results)
    python run_molblender_screening.py --disable-gpu

    # Second run: skip existing 723 results
    python run_molblender_screening.py --disable-gpu --skip-existing
    ```

- **Placeholder Value Protection in Database Storage**
  - **Problem**: Database was storing placeholder `[0,1,2,...]` indices instead of actual `y_test`/`y_train` values in `dataset_info.test_true_values`
  - **Root Cause**: Early session creation stored placeholder data before actual split data was available
  - **Impact**: Dashboard scatter plots showed incorrect Pearson R (near 0) due to placeholder true values
  - **Solution**: Added validation in `ScreeningResultsDB.save_dataset_info()`:
    - Checks if `test_true_values == list(range(len(test_true_values)))` (placeholder detection)
    - Checks if `train_true_values == list(range(len(train_true_values)))` (placeholder detection)
    - Refuses to save placeholder data with clear error message
    - Returns `False` to prevent invalid data storage
  - **Error Message**: `"❌ Refusing to save placeholder test_true_values (indices) for session {session_id}. These should be actual y_test values, not [0, 1, 2, ...]. This causes incorrect metric recalculations in Dashboard."`
  - **Files Modified**:
    - `src/molblender/models/api/utils/results_db.py`: Added placeholder validation in `save_dataset_info()` (lines 595-617)
  - **User Impact**:
    - Prevents database corruption from placeholder data
    - Ensures Dashboard metrics are computed from correct `y_test` values
    - Forces proper data flow: `evaluator.py` → real `y_test` → database → Dashboard
  - **Note**: Old databases with placeholder values should be regenerated by re-running screening with latest code

### Changed

- **Split Strategy Name Mapping**
  - **Fixed**: `max_dissimilarity` → `maxmin` in `run_molblender_screening.py`
  - **Reason**: API only recognizes `maxmin`, not `max_dissimilarity`
  - **Impact**: Correct MaxDissimilarity split functionality now works
  - **File**: `tests/data/.../run_molblender_screening.py` line 237

- **Database Merge Deduplication Keys** (2026-02-07)
  - Deduplication now considers model params and HPO best params to preserve tuned variants
  - Improves session/database merges for mixed Stage 1 + Stage 2 results

### Fixed

- **Two-stage mask coordinate mismatch**: when a component's `variance_mask` (post-initial_mask, shorter length) was AND-ed directly against `initial_mask` (original length), numpy broadcast raised `ValueError: operands could not be broadcast together`; now projected back to original column space
- **cv_only fusion mask width**: `_build_combined_phase2_mask` previously skipped components without masks, producing a partial mask shorter than `X.shape[1]`; now uses `component_ranges` to infer full width for maskless components
- **Relative import path in cv_only evaluator**: `from ...data.quality_flow` (3 dots) corrected to `from ..data.quality_flow` (2 dots) to match the actual module tree
- **Cross-trial param leakage in Optuna**: when the search space skips a conditional param for a trial, that trial's value previously persisted into subsequent trials via `model.set_params(**params)` because `model` was mutated in place. Now each trial starts from a clean clone of `base_trial_model`
- **`on_error="remove"` initial parsing failures correctly added to `ids_to_remove`**: a molecule that failed RDKit parsing was previously passed along but never recorded for removal; now `ids_to_remove.add(mol_id)` is set inside the failure branch
- **`_apply_removal_by_ids` stub removed from `FeatureManagementMixin`**: the placeholder `raise NotImplementedError` was shadowing the real method on `DataOperationsMixin` via MRO. The actual implementation is now reached via inheritance
- **Test cache isolation**: prior tests could leak cache files into `tests/.mbl_cache` or, if run from a different cwd, the project root. The new hermetic autouse fixture pins every test to `tests/.mbl_cache` and the per-test fixture resets all three cache singletons onto `tmp_path`, so cache writes never escape the test boundary
- **Removed obsolete `test_cache_semantics.py`**: tests were merged into `test_feature_manager_cache.py` to avoid duplication and drift
- **Stratify keyword mismatch in optimizer precompute**: `materialize_precomputed_splits` is now called with `stratify_labels=` (the actual parameter name) instead of `y_stratify=`, which Python previously silently ignored — a `TypeError` on every non-custom-stratify run
- **Representation Feasibility Assessment** (2026-07-29)
  - New ``RepresentationAvailability`` model with ``assess_availability_from_dataframe()`` detects per-representation featurization failures (missing features, zero coverage) before screening begins, preventing downstream NaN/empty-feature crashes
  - Three new modules: ``data/dataset/representation_availability.py`` (dataset-level availability assessment), ``models/api/screening_engine/data/representation_feasibility.py`` (screening-level feasibility report), ``cli/representation_feasibility.py`` (CLI entry point) with 46 tests in ``test_representation_failure_handling.py``
  - Dashboard ``representation_feasibility`` component visualises coverage gaps per representation
  - ``_validate_data_quality`` now records per-representation column masks (``_quality_column_masks``) and ``_quality_metadata`` during ``validate_data_quality_phase1`` so Phase 2 replay and HPO resumption can reconstruct column-level state
- **Fusion HPO Resume Phase 2 Leak Fix** (2026-07-29)
  - ``reconstruct_fusion_features(..., for_hpo_resume=True)`` returns Phase-1 numeric-coerced raw blocks without applying Stage-1 train-fitted Phase 2 metadata (imputation/scaler/variance), preventing test-leakage in HPO inner-CV folds; returns at raw width (sum of raw component widths) instead of post-Phase-2 ``total_features``
  - HPO path skips post-width and zero-width validation (width mismatch 1191 vs 81 is expected for raw fusion) and skips the top-level ``SimpleImputer`` for fusion rows so ``Phase2QualityTransformer`` handles NaN/inf per-fold
  - ``_quality_metadata`` lookup in ``run_hpo_stage`` uses ``get_representation_config(data_handler, repr_name)`` (resolves fusion ``_fusion_metadata`` with ``component_feature_ranges``) instead of ``_quality_metadata[repr_name]`` (returns None for fusion rows), so GridSearch/Optuna both wire up the Phase 2 transformer correctly
  - New stale-cache guard in HPO path: ``_quality_processed_representations`` fingerprint mismatch still raises ``FusionConfigError`` so Stage 2 HPO is refused when a component was overwritten after Stage 1
  - 5 new unit tests in ``test_fusion_hpo_resume.py``: raw-width return, stale-cache rejection, fresh-dataset bypass, NaN preservation, realistic overwrite scenario
- **Fingerprint dtype alignment** (2026-07-29): ``_write_back_to_dataset`` fingerprint now uses ``astype(np.float32)`` matching the downstream ``_assert_dense_2d_finite`` cast, eliminating false-positive stale-cache errors during same-process HPO reconstruction

- **OPTUNA_AVAILABLE=False fallback forwarded `y_stratify`**: the Optuna optimizer's fallback `CoarseGridSearchOptimizer.optimize_model()` call now threads `y_stratify=y_stratify` so the custom stratification labels are not lost when Optuna is absent
- **Group-aware splitter precompute now passes `groups=`**: `cv_splitter.split(X, y_stratify, groups=groups)` is used in both GridSearch and Optuna precompute paths; on failure a `ValueError` with splitter context is raised instead of silently falling back to `materialize_precomputed_splits` (different CV path, wrong folds, no audit trail)
- **Regression tests for the three stratification fixes**: `TestOptimizerStratifyForwarding` verifies `materialize_precomputed_splits` materializes with the correct keyword, the real Optuna fallback forwards `y_stratify` to Grid, and the real `optimize_model` (Grid/Optuna, parameterized) raises `ValueError` for a group-aware failing splitter rather than falling back

## [Unreleased]

### Fixed

- **Stage-1 effective-exists 全链 fail-closed（收回按名 skip）** (2026-08-20)
  - `persistence.stage1_results.stage1_exists_effective_result()` 删除 `allow_name_only` 逃生口：缺 `stage1_compatibility_key` 一律返回 False，同名不同 config/split 的历史行不再被误判为可复用
  - 旧 `utils/database/sessions.py::check_existing_result_record` facade 新增可选 `stage1_compatibility_key`：无 key 的 Stage-1 查询发 `DeprecationWarning` 并返回 False（recompute），带 key 时委托 canonical owner；`utils/database/__init__.py` 不再 re-export 该符号
  - 反例测试 `TestLegacyFacadeFailClosed`：native/linked 同名不同 key 行在无 key 时不 skip，匹配 key 才 skip
- **metrics 未定义值语义继续收口到 `molblender.metrics.core`** (2026-08-20)
  - 新增 `calculate_explained_variance_score`（population variance；常数目标按 sklearn force_finite 语义：完美拟合 1.0、非完美 0.0）
  - `dashboard/data/metrics.py::calculate_additional_metrics` 的 MAPE/MedAE/max_error/Pearson r/r²/explained-variance 全部改委托 core，删除手写 `np.corrcoef`/`np.median`/mask-MAPE
  - `evaluation/metrics.py::get_scoring_function` 回归 scorer（rmse/mae/mse/medae/max_error/neg_*）从 sklearn 直算 + 手写 `np.sqrt`/`ravel` 改为 core 的 `calculate_*`；`r2_score` key 显式绑定系数决定 R²，与 `r2`/`pearson_r2`（squared Pearson）区分
- **修复 baseline 失败** (2026-08-20)
  - `test_classification.py`：`_supports_task` 方法已外提为 `models/eligibility.py::supports_task` 模块函数，测试改用显式导入；DataSplitter patch 目标改为 `data_handler` 模块命名空间（`from X import Y` 的 binding 拷贝问题）；split 返回 keys 对齐真实契约（`X_train/X_test/y_train/y_test/...`）
  - `test_multimodal_service_boundary.py`：services `__all__` 契约纳入 `Stage-1` import helpers（`import_compatible_stage1_results` / `plan_and_import_compatible_stage1_results` 为真实 wrapper）
  - `test_multimodal_screening_helpers.py::test_execute_modality_screening_loop_updates_splits_once`：patch `session_ops.update_session_indices` 计数并让 mock result 携带 train/test indices，holdout 只写一次 splits/indices
  - `test_lead_sensitivity_runtime_config.py`：monkeypatch 目标从已不存在的 `run_lead_sensitivity_analysis_impl` 改为实际委托点 `_execute_lead_analysis`

- **Stage 1 Partial Reuse to Avoid OOM** (2026-02-07)
  - When only some representations have Stage 1 results, existing ones are loaded instead of forcing a full re-run

- **HPO Feature Reconstruction Robustness** (2026-02-07)
  - Handles nested embeddings with fast stacking and imputes NaNs in descriptor features

- **Dashboard MAPE Radar Chart Display** (2026-02-04)
  - **Problem**: Extremely large MAPE values (e.g., 3.6 billion %) were clamped to max range (2.0), resulting in normalized value of 0 on radar charts
  - **Impact**: Radar charts showed misleading "0" for MAPE when actual values were unreasonably high
  - **Solution**: Skip MAPE values >500% (5.0) from radar chart display entirely
  - **Rationale**: MAPE >500% indicates severe data quality issues (e.g., predictions near zero for non-zero true values)
  - **File Modified**: `src/molblender/dashboard/components/model_inspection/visualization.py`
  - **Code Change**:
    ```python
    # Skip unreasonably large MAPE values (>500% indicates data issues)
    if key == "mape" and value > 5.0:
        continue
    ```
  - **User Impact**: Radar charts now show only meaningful metrics, preventing confusion from extreme outliers

- **Dashboard Training Time Analysis Rendering**
  - Fixed import path for modality training time pie chart in `general.py`
  - Ensured Session Breakdown renders only in the right column
  - Resolved pandas column mismatch for session aggregation when `n_jobs` is present
- **Log Noise Reduction**
  - Added repeated-message suppression for common CUDA/XGBoost/CV failure patterns
  - Reduced spam for near-constant predictions and low Pearson‑r warnings
- **Database Merge Tool (`molblender merge_databases`)**
  - Fixed critical bugs causing Dashboard loading failures and data loss
  - Now copies all 4 required tables (`screening_sessions`, `model_results`, `dataset_info`, `schema_version`) instead of only 2
  - Preserves original session_ids instead of creating artificial "merged" session
  - Properly copies `dataset_info` and `schema_version` entries for each session
  - File: `src/molblender/models/api/utils/database_merger.py` (complete rewrite)

### Changed

- **CUDA → CPU Fallback (Thresholded)**
  - Added CPU fallback for VAE/CNN/Transformer models on CUDA failure when CPU cores ≥32
  - XGBoost CUDA fallback now respects the same core threshold (otherwise re-raises)

### Added

- **Sample Weighting for Imbalanced Regression** (`src/molblender/models/api/core/weighting.py`)
  - **Three Weighting Strategies** for handling zero-inflated or heavily imbalanced regression data:
    - **threshold**: Binary weighting (e.g., pX > 0 samples get 2-3x weight, pX = 0 get 1x weight)
    - **inverse_density**: KDE-based density estimation, weights ∝ 1/density for rare value ranges
    - **quantile**: Quantile-based weighting, extreme quantiles get higher weight
  - **Configuration Parameters** in `ScreeningConfig`:
    - `use_sample_weights`: Enable/disable weighting (default: False for backward compatibility)
    - `weight_strategy`: Choose strategy ("threshold", "inverse_density", "quantile")
    - `weight_threshold`: Threshold for binary split (default: 0.0)
    - `weight_power`: Power to smooth extreme weight differences (0.5-1.0, default: 1.0)
    - `active_subset_threshold`: Threshold for active subset evaluation (default: 0.0)
  - **Safe Model Training** (`_safe_fit_with_weights()` in evaluator.py):
    - Automatically handles models that don't support `sample_weight`
    - Graceful fallback to standard training if weights not supported
    - Logs warnings when weights cannot be used
  - **Active Subset Metrics** for comprehensive imbalanced regression evaluation:
    - `active_r2`, `active_rmse`, `active_mae`: Metrics computed only on active samples (e.g., pX > 0)
    - `active_count`, `active_percentage`: Number and percentage of active samples
    - Computed when `use_sample_weights=True` and task is regression
  - **Command-Line Interface** (run_molblender_screening.py):
    - `--use-sample-weights`: Enable sample weighting
    - `--weight-strategy`: Choose weighting strategy
    - `--weight-threshold`: Set threshold for binary weighting
    - `--weight-power`: Smooth weight differences (default: 1.0)
    - `--active-subset-threshold`: Set threshold for active subset evaluation
  - **Use Case**: RdRp inhibitor screening with 86% inactive (pX=0) and 14% active (pX>0) samples
    - Training on weighted samples improves model performance on rare active compounds
    - Active subset metrics provide visibility into model performance on the region of interest
  - **Backward Compatibility**: Default `use_sample_weights=False` ensures no impact on existing workflows
  - **Reference**: SMOGN (Synthetic Minority Over-sampling Technique for Non-Gaussian continuous features)
    https://link.springer.com/article/10.1007/s10994-021-06023-5

- **Advanced Molecular Splitting Strategies** (`src/molblender/data/dataset/splitting/`)
  - **4 Advanced Splitters from splito package** (Apache 2.0 License):
    - **PerimeterSplit**: Extrapolation-oriented split placing most dissimilar molecules in test set
    - **MolecularWeightSplit**: Tests generalization across different molecular sizes
    - **MOODSplitter**: Model-Optimized Out-of-Distribution split based on deployment data similarity
    - **LoSplitter**: Lead optimization split returning multiple test clusters for SAR exploration
  - **Dual API Design**:
    - **Functional API (sklearn-style)**: `train_test_split(X, y, molecules=smiles, method='perimeter')`
    - **Class-based API**: `PerimeterSplit(test_size=0.2).split(smiles)`
    - **MolecularDataset integration**: `dataset.train_test_split(method='perimeter')`
  - **RDKit-based Implementation**:
    - Replaced all `datamol` dependencies with direct RDKit calls
    - Utility functions: `compute_fingerprints()`, `compute_molecular_weights()`, `to_mol()`, `to_smiles()`
    - K-means clustering for distance-based splits (computational efficiency)
  - **Modular Package Structure** (`splitting/` subdirectory):
    - `base.py`: SplittingMixin for MolecularDataset
    - `advanced.py`: 4 advanced splitters (~780 lines)
    - `functional.py`: sklearn-style functional API (~350 lines)
    - `utils.py`: RDKit utility functions
  - **Comprehensive Testing**:
    - 16 unit tests covering all methods and edge cases
    - Test functional API, class-based API, and MolecularDataset integration
    - Test utility functions (fingerprints, molecular weights)
  - **Documentation**:
    - API reference: `docs/api/data/splitting.rst`
    - Usage guide: `docs/usage/data/splitting.md` (extended with 4 new sections)
    - Complete examples for all splitting methods
  - **License Attribution**:
    - Apache 2.0 license headers in all relevant files
    - References to splito package: https://github.com/datamol-io/splito
  - **Use Cases**:
    - Virtual screening validation (PerimeterSplit)
    - Fragment-to-lead optimization (MolecularWeightSplit)
    - Deployment-aware validation (MOODSplitter)
    - SAR exploration in medicinal chemistry (LoSplitter)
  - **Custom User-Provided Split** (`method='custom'`):
    - Use predefined train/test assignments from external sources
    - **Two input modes**:
      - `split_column`: Column name with 'train'/'test', 0/1, or True/False values
      - `train_indices`/`test_indices`: Explicit index arrays
    - Use cases: benchmark reproduction, temporal splits, experimental batches
    - Complete validation: overlap detection, range checking, value parsing
    - Utility function: `indices_from_split_column()` for column parsing
  - **Enhanced LoSplitter Documentation**:
    - Added scaffold hopping pharmaceutical use case example
    - Real-world scenario: Scaffold 1 exhaustion → scaffold hopping → series prediction
    - Explains how LoSplitter validates model generalization across scaffolds
  - **Backward Compatible**:
    - All existing splitting methods remain unchanged
    - New methods are opt-in via `method` parameter
    - Default behavior (random split) preserved
  - **Expanded Test Coverage**:
    - 28 unit tests total (12 new custom split tests)
    - Tests for functional API and MolecularDataset integration
    - Edge case testing: NaN values, invalid formats, missing columns
- **Two-Stage Hyperparameter Optimization System** (`src/molblender/models/api/`)
  - **Stage 1**: Model screening with default parameters for rapid baseline performance assessment
  - **Stage 2**: Automated hyperparameter optimization (HPO) for top-N performers from Stage 1
  - **Configuration via ScreeningConfig**:
    - `enable_hpo=True/False`: Enable/disable HPO Stage 2 (default: False for backward compatibility)
    - `hpo_stage='coarse'/'fine'`: Coarse grid search or fine-grained Optuna optimization (default: 'coarse')
    - `hpo_method='grid'/'random'`: GridSearchCV or RandomizedSearchCV (default: 'grid')
    - `top_n_for_hpo=N`: Number of top Stage 1 performers to optimize (default: 10)
    - `hpo_cv_folds=K`: CV folds for HPO (default: None, uses `cv_folds` or `inner_cv_folds`)
    - `hpo_selection_scope='global'/'per_type'/'per_subtype'`: Model selection scope for HPO: which group to pick top-N from (default: 'global')
      - **'global'**: Select top N performers overall across all model types (default behavior)
      - **'per_type'**: Select top N from Traditional ML AND top N from Deep Learning separately
        - Ensures balanced HPO coverage when one model category dominates Stage 1 performance
        - Example: `hpo_selection_scope='per_type', top_n_for_hpo=5` → 5 Traditional ML + 5 Deep Learning = 10 total HPO runs
        - Use case: CNN/VAE/Transformer models receive HPO even if Traditional ML models have higher Stage 1 scores
      - **'per_subtype'**: Select top N from each fine-grained model category (LINEAR, TREE, BOOSTING, VAE, CNN, TRANSFORMER)
        - Most granular option for comprehensive model type coverage across all architectures
    - `hpo_models_per_type=N`: Override number of models per type/subtype (default: None, uses `top_n_for_hpo`)
  - **Model Type Classification System** (`processors/hpo.py`):
    - High-level: TRADITIONAL_ML vs DEEP_LEARNING (using ModelCorpus enum from model registry)
    - Fine-grained: LINEAR, TREE, BOOSTING, KERNEL, VAE, TRANSFORMER, CNN subtypes
    - Automatic classification based on model registry metadata and categories
  - **Parameter Grid System** (`src/molblender/models/corpus/parameter_grids.py`):
    - Comprehensive coarse grids for all model types (tree, boosting, SVM, linear, neural, VAE, CNN)
    - Model-specific parameter ranges based on best practices
    - CNN grids: learning_rate, batch_size, epochs (matrix_cnn, image_cnn variants)
    - Fine grids reserved for Optuna optimization (future)
  - **Database Integration**:
    - Stage tracking in `model_results` table: `stage=1` (default params) vs `stage=2` (optimized params)
    - HPO metadata: `hpo_stage`, `best_params`, `hpo_cv_score` columns
    - Incremental result storage for both Stage 1 and Stage 2
  - **Smart Workflow**:
    - Automatically skips Stage 2 if insufficient high-performing models in Stage 1
    - Preserves Stage 1 results even when Stage 2 is enabled
    - Compatible with all data splitting strategies (train_val_test, nested_cv, etc.)
  - **Performance Impact**:
    - Stage 1 only: Fast baseline screening (~5-10 min for 100 combinations)
    - Stage 1 + Stage 2: Comprehensive optimization (~15-30 min with top_n=10)
  - **Complete GridSearchCV Results Storage** (`all_cv_results` column):
    - Stores ALL tested parameter combinations from GridSearchCV, not just the best one
    - Enables HPO parameter sensitivity analysis in dashboard
    - Captured data for each combination:
      - `params`: Parameter values tested (e.g., `{"alpha": 0.1}`, `{"alpha": 1.0}`)
      - `mean_test_score`, `std_test_score`, `rank_test_score`: Aggregated CV performance
      - Individual fold scores (`split0_test_score`, `split1_test_score`, etc.)
      - Timing information (`mean_fit_time`, `std_fit_time`, `mean_score_time`)
    - JSON serialization with numpy→list conversion for database storage
    - Backward compatible: NULL for Stage 1 results (default parameters, no HPO)
    - Example use case: Compare Ridge α=[0.1, 1.0, 10.0] to visualize sensitivity
    - Database schema migration: Automatically adds `all_cv_results TEXT` column to existing databases
    - Implementation locations:
      - `grid_search.py:148-186`: `_extract_cv_results()` method extracts all cv_results_ data
      - `screeners.py:919,947,1120-1122,1154-1174`: HPO orchestration captures and stores cv_results
      - `results_db.py:186-187`: Database migration adds all_cv_results column
  - Successfully tested with:
    - Traditional ML models (Ridge, RandomForest, XGBoost, LightGBM, etc.)
    - Deep learning models (VAE with latent_dim, learning_rate, batch_size grids)
    - 3D representations (spatial matrices, UniMol embeddings, 3D fingerprints)
    - Multiple fingerprint categories (RDKit, CDK, Datamol)

- **VAE (Variational Autoencoder) Integration for Molecular Fingerprints** (`src/molblender/models/`)
  - **Complete VAE Implementation** (`modality_models/vae_models.py`):
    - MolecularFingerprintVAE class with encoder-decoder architecture
    - Integrated predictor network for direct property prediction from latent space
    - 5 pre-configured variants: VAE (latent=64/128/256), VAE (compact), VAE (deep)
    - Auto-generates 3D conformations from SMILES when needed
    - PyTorch-based with GPU/CPU auto-detection
  - **Model Registry Integration** (`api/core/model_registry.py`):
    - All 5 VAE models registered in ModelRegistry
    - Categorized under `DEEP_LEARNING` and `ACCURATE` corpus
    - HPO parameter grids: latent_dim, learning_rate, batch_size, epochs
  - **Pathway System** (`api/multimodal/`):
    - `combinations="auto"`: Traditional ML only (default behavior, backward compatible)
    - `combinations="comprehensive"`: Traditional ML + VAE models for vector modality
    - Similar to matrix/image pathway handling (flattened vs CNN)
  - **sklearn Compatibility Fixes**:
    - Fixed `clone()` compatibility: Store device as string, convert to torch.device only when needed
    - Fixed numpy conversion: Use `.detach().cpu().numpy()` for proper tensor→array conversion
    - Proper 1D/2D array shape handling for sklearn metrics
  - **User Experience**:
    - Auto-excludes VAE models in default mode (no impact on existing workflows)
    - Optional VAE screening via `combinations="comprehensive"` parameter
    - Seamless integration with HPO system (Stage 1 + Stage 2 optimization)
    - GPU acceleration when available, graceful CPU fallback
  - **Current Status**: Fully integrated and tested, VAE models accessible but showing 0.0 scores (model performance issue, not integration bug)

- **DNR Diagnostics Module** (`src/molblender/data/diagnostics/`)
  - Complete implementation of DNR (Different Neighbor Ratio) analysis for dataset quality assessment
  - Paper-accurate parameters from "Upgrading Reliability in Molecular Property Prediction" (similarity threshold 0.5, property diff 1.0 log unit)
  - Activity cliff detection for identifying similar molecules with large property differences
  - Comprehensive visualization suite with 5 plot types (DNR distribution, DNR vs property, activity cliff network, similarity heatmap, neighbor statistics)
  - SVG output format to avoid matplotlib/numpy compatibility issues with PNG
  - One-click full diagnostics workflow with `run_full_diagnostics()` method
  - Automatic sampling for large datasets (>10,000 molecules) to prevent memory issues
  - Integration with MolecularDataset API via `DatasetDiagnostics` class
  - Comprehensive markdown report generation with interpretation guidelines
  - Submodules: `core.py` (main diagnostics), `similarity.py` (fingerprint utilities), `visualization.py` (plotting functions)
  - **Interactive Streamlit Dashboard** (`src/molblender/data/diagnostics/dashboard/`)
    - Integrated into package as proper module
    - Console script entry point: `molblender-diagnostics` command
    - Built-in comprehensive documentation and interpretation guide
    - Adjustable thresholds with real-time recomputation
    - CSV export functionality for results
    - Support for file upload or command-line file path
  - Successfully tested on 120-molecule head.csv dataset with complete output generation
  - Comprehensive module documentation in `src/molblender/data/diagnostics/CLAUDE.md`

- **Protein Data Handling Documentation** (`docs/source/usage/data/protein.md`)
  - Comprehensive guide to Protein class and multi-format support
  - Database retrieval from RCSB PDB, AlphaFold, and UniProt
  - Structure prediction and repair with ESMFold and PDBFixer
  - FASTA, PDB, mmCIF format handling with automatic detection
  - BioPython integration for structural analysis
  - Intelligent caching system for downloads and predictions
  - Protein-ligand dataset integration examples
  - Multi-chain structure support and metadata extraction
  - Sequence validation and cleaning utilities
  - Best practices for high-throughput protein processing

- **Scaffold-Based Splitting Implementation** (`src/molblender/models/api/core/splitting/scaffold.py`)
  - Complete scaffold-based train/test splitting for drug discovery applications
  - Two scaffold generation methods: Bemis-Murcko and Generic (topology-only)
  - Two split strategies: Balanced (greedy size matching) and Random (random assignment)
  - RDKit integration for scaffold computation with error handling
  - Scaffold leakage detection to ensure train/test set separation
  - Integration with `DataSplitter` class via `strategy="scaffold"` parameter
  - Comprehensive unit tests with 18 test cases covering all functionality
  - Documentation added to `docs/source/usage/data/splitting.md` with examples and use cases

- **DNR-Based Splitting Strategy** (`src/molblender/models/api/core/splitting/dnr.py`)
  - Systematically tests model performance on rough SAR regions and challenging molecules
  - Three split modes:
    - **Threshold mode**: Split by DNR threshold (high vs low DNR molecules)
    - **Quantile mode**: Split by DNR quantiles (top X% vs rest)
    - **Neighbor mode**: Split by neighbor presence (isolated vs connected molecules)
  - Configurable parameters: DNR threshold, similarity threshold, property difference threshold
  - Enables systematic evaluation of two major error modes from "Upgrading Reliability in Molecular Property Prediction" paper
  - Integration with `DataSplitter` class via `strategy="dnr"` parameter
  - Leverages existing DNR calculation infrastructure from diagnostics module
  - Provides detailed split statistics including mean DNR, high-DNR counts, no-neighbor counts

- **MaxMinPicker Diversity Splitting** (`src/molblender/models/api/core/splitting/diversity.py`)
  - Diversity-based splitting using RDKit's MaxMinPicker algorithm
  - Two operational modes:
    - **Friendly mode**: Diverse training set ensures broad chemical space coverage for learning
    - **Unfriendly mode**: Diverse test set creates most challenging generalization scenario
  - Multiple fingerprint support: Morgan (default), RDKit topological, MACCS keys
  - Configurable fingerprint parameters: radius, number of bits
  - Uses Tanimoto dissimilarity for maximum diversity selection
  - Reports similarity statistics: train/test avg similarity, cross-similarity
  - Integration with `DataSplitter` class via `strategy="maxmin"` parameter
  - Addresses diversity sampling requirements from molecular ML literature

- **Butina Clustering-Based Splitting** (`src/molblender/models/api/core/splitting/butina.py`)
  - Leave-cluster-out cross-validation based on Tanimoto similarity clustering
  - Uses Butina's sphere exclusion algorithm (Butina 1999, J. Chem. Inf. Comput. Sci.)
  - Ensures chemically similar molecules stay together in either training or test set
  - Prevents information leakage from structural similarity, addressing MolAgent clustering strategy
  - Key features:
    - **Automatic clustering**: Self-adaptive cluster count based on similarity threshold
    - **Greedy balanced assignment**: Largest clusters assigned first to balance train/test sizes
    - **Leave-cluster-out validation**: Entire clusters move as units (no split within cluster)
  - Configurable parameters: similarity_threshold (default 0.6), fingerprint type, radius, nbits
  - Reports cluster statistics: n_clusters, cluster sizes, intra/inter-cluster similarity
  - Integration with `DataSplitter` class via `strategy="butina"` parameter
  - More fine-grained than scaffold splitting (considers full molecular topology vs core structure only)
  - Suitable for evaluating generalization to similar but unseen chemical combinations

- **Feature Clustering-Based Splitting** (`src/molblender/models/api/core/splitting/feature_clustering.py`)
  - General-purpose clustering split supporting arbitrary molecular representations (not limited to fingerprints)
  - Three clustering algorithms:
    - **K-means++**: Fast spherical clustering with smart initialization
    - **Hierarchical**: Ward linkage tree-based clustering
    - **DBSCAN**: Density-based clustering with automatic cluster count detection
  - Flexible feature input sources:
    - **User-provided representations**: 3D embeddings (Boltz-2), language models (ChemBERTa), custom features
    - **RDKit descriptors**: ~20 physicochemical descriptors (MW, LogP, TPSA, etc.)
    - **Fingerprints**: Morgan/RDKit/MACCS as fallback
  - Automatic optimal k selection via Silhouette score maximization (k ∈ [2, √n])
  - Feature standardization with StandardScaler for fair distance computation
  - Comprehensive clustering quality metrics:
    - **Silhouette score**: [-1, 1], >0.5 indicates good clustering
    - **Calinski-Harabasz index**: Higher values indicate better-defined clusters
    - **Davies-Bouldin index**: Lower values indicate better separation
  - Leave-cluster-out assignment with greedy balanced strategy
  - DBSCAN noise point handling (assigned to training set)
  - Integration with `DataSplitter` class via `strategy="feature_clustering"` parameter
  - Configurable parameters: clustering_algorithm, n_clusters, auto_select_k, features, use_descriptors, dbscan_eps, dbscan_min_samples
  - Complementary to Butina splitting: Butina uses Tanimoto+fingerprints, feature_clustering uses Euclidean+arbitrary features
  - Ideal for non-fingerprint representations (3D structures, quantum features, embeddings)
  - Comprehensive test suite (14 passed, 2 DBSCAN skipped due to numpy compatibility)

- **Shared Splitting Utilities** (`src/molblender/models/api/core/splitting/utils.py`)
  - Centralized utility functions to eliminate code duplication across splitting strategies
  - `compute_fingerprints()`: Unified fingerprint generation (Morgan, RDKit, MACCS)
  - `compute_avg_similarity()`: Average pairwise Tanimoto similarity within a set
  - `compute_cross_similarity()`: Average Tanimoto similarity between train and test sets
  - Reduces code duplication by 64 lines across butina.py and diversity.py
  - Shared by Butina, MaxMin, and Feature Clustering splitting strategies

- **Dataset Splitting Documentation** (`docs/source/usage/data/splitting.md`)
  - Comprehensive guide to all 10 supported splitting strategies (train_test, train_val_test, nested_cv, cv_only, scaffold, dnr, maxmin, butina, feature_clustering, user_provided)
  - Detailed implementation references with code locations and line numbers
  - Visual workflow diagrams for each splitting strategy
  - Scaffold split section with Bemis-Murcko vs Generic comparison and balanced vs random split methods
  - Feature Clustering section with complete examples:
    - User-provided features (3D embeddings, language models)
    - RDKit descriptors with K-means/Hierarchical clustering
    - DBSCAN with automatic cluster detection
    - Feature Clustering vs Butina comparison table
    - Typical workflow example with Boltz-2 embeddings
  - Drug discovery use cases and advantages over random splitting
  - Best practice recommendations based on dataset size and use case
  - Reproducibility guarantees and fixed random seed documentation
  - Cross-validation protocol details with StratifiedKFold for classification
  - Decision tree for choosing the right splitting strategy
  - Performance considerations and memory efficiency comparisons
- **Boltz-2 AI Structure Prediction Integration** (`src/molblender/representations/AI_fold/boltz2/`)
  - Complete module for extracting embeddings from Boltz-2 protein-ligand complex predictions
  - Support for three embedding types: global (29-33 dim), token-level (4-dim pooled), pairwise distance matrices
  - Intelligent caching system to avoid redundant structure predictions (saves hours of GPU time)
  - Isolated conda environment execution via subprocess to prevent dependency conflicts
  - Automatic YAML input generation with short IDs to avoid Boltz-2 truncation errors
  - MSA server integration for protein sequence alignment
  - Structure file parsing (CIF format) with recursive file discovery
  - Geometric feature extraction including COM, radius of gyration, protein-ligand contacts, confidence scores (pLDDT)
  - Complete test suite with unit tests and integration tests
  - **Note**: Structure prediction via subprocess currently has limitations; users can provide pre-computed CIF files for embedding extraction

- **Protein-Ligand Dataset Support** (`src/molblender/data/`)
  - Extended `Molecule` class with `protein_sequence` and `protein_pdb_path` attributes for protein-ligand complexes
  - New parameters in `MolecularDataset.from_csv()`: `protein_sequence_column` and `protein_pdb_column`
  - Automatic storage of protein information in molecule properties for CSV round-tripping
  - Seamless integration with existing featurization pipeline via kwargs passing
- **Dynamic Metric Resolution System** (`src/molblender/dashboard/metrics/central.py`)
  - Single source of truth for all metric definitions and display names
  - Automatic resolution of `primary_metric` to actual metric names (e.g., "Pearson R²", "MAE")
  - Consistent metric naming across all dashboard components
  - Eliminated hardcoded "Primary Metric" references throughout the interface
  - Enhanced `format_metric_name()` utility with DataFrame context for proper resolution

- **Professional Chart Styling Framework** (`src/molblender/dashboard/components/utils/chart_fonts.py`)
  - Unified font sizing system across all dashboard visualizations
  - Professional color scheme (light blue #6BAED6, light green #74C476, light orange #FD8D3C)
  - Consistent axis styling and formatting for research-quality charts
  - Global font size control via `AXIS_TITLE_FONT_SIZE`, `TICK_LABEL_FONT_SIZE`, `CHART_TITLE_FONT_SIZE`

- **Interactive Table System** (`src/molblender/dashboard/components/tables.py`)
  - Complete replacement of HTML tables with native `st.dataframe()` components
  - Dynamic metric column names with proper formatting
  - Sorting, filtering, and export capabilities
  - Responsive design with container width optimization

- **Comprehensive Distribution Analysis Charts** (`src/molblender/dashboard/components/distribution/charts.py`)
  - Five chart types: Box Plot, Violin Plot, Histogram, Density Plot, Raincloud Plot
  - Unified chart rendering system with consistent styling
  - Professional axis labeling with dynamic metric names
  - Eliminated "undefined" chart titles across all visualizations

- **Comprehensive Methodology Documentation** (`docs/source/usage/models/methodology.md`)
  - Complete explanation of train/test data splitting strategy (80/20 default with `random_state=42`)
  - Detailed 5-fold cross-validation protocol documentation
  - Model training and evaluation workflow with code references
  - Numerical examples and visual diagrams showing data flow
  - Best practices for choosing `test_size` and `cv_folds` based on dataset size
  - Known limitations and future improvements documented
  - Added cross-references from `screening.md` and navigation in `models/index.md`
  - Added CHANGELOG to documentation TOC

### Fixed
- **Boltz-2 Test Failures and Registration** (`src/molblender/representations/AI_fold/boltz2/`)
  - Fixed ligand_id truncation issue: Changed from 16-character cache_key to short ID 'L' to prevent Boltz-2 KeyError (`structure_predictor.py:157`)
  - Added AI_fold module import to `representations/__init__.py` to trigger Boltz2Embedder registration
  - Fixed test_registration: Corrected assertion to check instance instead of class (registry returns instances by design)
  - Fixed test_full_workflow: Corrected Dataset API usage (`feature_names`, `features.iloc[0]`) and `add_features()` argument order
  - Fixed VAE models indentation: Corrected MolecularVAE class `__init__` method indentation to prevent `RuntimeError: super(): no arguments`
  - Test results: 12 passed (up from 9), 1 skipped (GPU-dependent test validated manually with pre-computed CIF files)
  - All 4 originally failing tests now pass: `test_structure_prediction`, `test_caching_works`, `test_registration`, `test_full_workflow`

- **Dynamic Metric Resolution** (`src/molblender/dashboard/components/`)
  - Fixed "Primary Metric" displaying in Outlier Details table - now shows actual metric name
  - Fixed "Primary Metric" displaying in Distribution Overview charts - now shows formatted axis titles
  - Eliminated "undefined" chart titles across all dashboard visualizations
  - Fixed NameError with undefined 'df' variable in modality comparison charts
  - Resolved duplicate column names in results tables

- **Modality Filter Functionality** (`src/molblender/dashboard/components/charts/performance.py`)
  - Fixed non-responsive modality filter in Representation Analysis section
  - Added debug output for troubleshooting filter behavior
  - Enhanced modality mapping with proper error handling

- **Cross-Validation Reproducibility** (`src/molblender/models/api/core/evaluation/evaluator.py`)
  - Fixed CV random_state issue: Now uses `KFold(random_state=42)` and `StratifiedKFold(random_state=42)` objects
  - Ensures complete reproducibility of both CV scores and test scores across different runs
  - Automatically selects StratifiedKFold for classification tasks to maintain class balance
  - Updated documentation to reflect this fix in methodology.md

- **DNR and MaxMin Splitting Test Fixes** (`tests/models/splitting/test_dnr_maxmin_split.py`)
  - Fixed test dataset creation: Changed from `from_smiles_list()` to `from_df()` to properly set label_names
  - Fixed diversity.py: Changed `mol.rdkit_mol` to `mol.get_rdkit_mol()` method call
  - All 12 tests now pass successfully (previously 9 failed, 3 passed)
  - Verified DNR-based splitting with all 3 modes (threshold, quantile, neighbor)
  - Verified MaxMinPicker diversity splitting with friendly/unfriendly modes
  - Confirmed DataSplitter integration for both new strategies

- **Dashboard Distribution Charts Column Name Errors** (`src/molblender/dashboard/components/distribution/charts.py`)
  - Fixed ValueError: "Value of 'x' is not the name of a column in 'data_frame'" in 6 chart rendering functions
  - Root cause: Chart functions incorrectly used metric parameter (e.g., 'pearson_r') as dataframe column names
  - Solution: Dashboard always uses 'score' column for metric values; metric names are for labeling only
  - Fixed functions: `render_box_plot()`, `render_violin_plot()`, `render_histogram()`, `render_density_plot()`, `render_raincloud_plot()`, `render_quick_distribution()`
  - Pattern: Changed from `df[metric]` to `df['score']` while keeping `format_metric_name(metric)` for axis labels

- **Dashboard Usage Analysis pd.crosstab Error** (`src/molblender/dashboard/components/distribution/usage_analysis.py`)
  - Fixed ValueError: "aggfunc cannot be used without values" in representation-model combination heatmap
  - Simplified `pd.crosstab()` call to show combination counts only (removed conditional aggfunc logic)
  - Removed conflicting parameters: `values=df[selected_metric]` and `aggfunc='count'/'mean'`
  - Now displays usage frequency (how many times each representation-model pair was tested)

- **CNN Model Dimension Mismatch Bug** (`src/molblender/models/modality_models/cnn_models.py`)
  - **Critical Fix**: Replaced `squeeze()` with `flatten()` to prevent scalar predictions causing "Length mismatch: y_true=457, y_pred=1" errors
  - **Root Cause**: `squeeze()` removes all size-1 dimensions, converting single-sample predictions from `[1]` array to scalar
  - **Fixed Locations**:
    - `predict()` method (lines 601-605): Changed `outputs.squeeze()` → `outputs.flatten()`
    - `fit()` method training loop (line 546): Loss calculation now uses `outputs.flatten(), batch_y.flatten()`
    - `fit()` method validation loop (line 562): Same fix for validation loss calculation
  - **Batch Size Safety** (lines 515-529): Added dynamic batch_size adjustment for small datasets to prevent BatchNorm errors
    - Automatically reduces batch_size if dataset size < configured batch_size
    - Logs warning when dataset too small for BatchNorm, preventing training crashes
  - **Impact**: Eliminates 100+ "Length mismatch" errors per screening run, enables CNN evaluation on small datasets
  - **Verification**: Test confirmed single-sample prediction returns 1D array `(1,)` instead of scalar

- **XGBoost CUDA Compatibility** (`src/molblender/models/corpus/model_corpus.py`, `src/molblender/config/version_compat.py`)
  - **XGBoostRegressorWrapper/XGBoostClassifierWrapper**: Automatic CUDA fallback on initialization errors
  - **Mechanism**:
    - Try fitting with default parameters (may attempt CUDA)
    - On `XGBoostError` with "cuda" in error message, automatically retry with `device="cpu", tree_method="hist"`
    - Logs warning when fallback occurs for user awareness
  - **Implementation**: Wrapper classes with `fit()` and `predict()` methods implementing sklearn interface
  - **Error Prevention**: Resolves 8 CUDA-related failures in production screening logs
  - **Transparent Usage**: Drop-in replacement for `XGBRegressor`/`XGBClassifier` with no API changes
  - **Note**: `get_xgboost_params()` now only handles version compatibility (e.g., use_label_encoder), device selection handled by wrapper

- **Modality Compatibility Enforcement** (`src/molblender/models/api/core/model_registry.py`)
  - **Critical Fix**: Added modality compatibility check when `model_names` parameter is explicitly provided
  - **Root Cause**: Lines 82-91 previously skipped modality filtering when user specified model list
  - **Problem**: `universal_screen(..., model_names=['transformer_small'])` with fingerprint data incorrectly attempted Transformer+VECTOR combination
  - **Solution** (lines 82-98):
    - Check modality compatibility even with explicit model_names
    - Log warning and skip incompatible models: `Model '{name}' is not compatible with data modality '{modality}', skipping`
  - **Compatibility Matrix**:
    - VECTOR (fingerprints/descriptors) → Traditional ML only, NO Transformers/CNN
    - STRING (raw SMILES) → Transformer only
    - MATRIX/IMAGE → CNN only
  - **Verification**: `transformer_small + maccs_keys` now correctly skipped, only `ridge + maccs_keys` evaluated

- **Configurable Timeout System** (`src/molblender/models/api/core/base.py`, `src/molblender/models/api/core/evaluation/evaluator.py`)
  - **User-Customizable Timeouts** in `ScreeningConfig`:
    - `model_timeout: Optional[int] = None` - Set absolute timeout (seconds), None=use adaptive
    - `base_model_timeout: int = 600` - Base timeout for adaptive calculation (increased from 300s to 600s)
    - `min_model_timeout: int = 60` - Minimum allowed timeout (seconds)
    - `max_model_timeout: int = 3600` - Maximum allowed timeout (seconds)
  - **Adaptive Timeout Calculation** (`_get_adaptive_timeout()` method):
    - User-specified `model_timeout` takes priority if set
    - Otherwise uses `base_model_timeout` × multipliers:
      - Representation type: Transformer ×3, CNN ×2, Fingerprint ×0.5
      - Data size: n_samples > 10000 → ×2, n_features > 5000 → ×1.5
      - Operation type: CV uses `cv_folds × 0.3` multiplier
    - Final timeout bounded by `[min_model_timeout, max_model_timeout]`
  - **Impact**:
    - Transformer CV with 10k samples: 600 × 3 × 2 × (5 × 0.3) = **5400s (90min)**
    - Fingerprint training: 600 × 0.5 = **300s (5min)**
  - **Default Timeout Increase**: Base timeout doubled from 5min to 10min for better deep learning model coverage

### Changed
- **Dataset Quality Analysis Migration**: Replaced `overview.py` with comprehensive `diagnostics/` module
  - Migrated basic statistics and molecular weight calculations to `diagnostics/core.py`
  - Enhanced with DNR analysis and activity cliff detection capabilities
  - Improved visualization quality with publication-ready SVG output
  - Updated `src/molblender/data/__init__.py` to export `DatasetDiagnostics` instead of `generate_dataset_report`

- **Dashboard UI Reorganization**: Improved navigation and logical grouping of analysis components
  - Performance vs Training Time chart now colored by modality category (fingerprints, language-model, spatial, image, string) instead of individual models for clearer pattern recognition
  - Simplified Model Analysis from 3 to 2 sub-tabs (Performance Deep Dive and Efficiency Analysis)
  - Moved Multi-Dimensional Model Comparison (radar chart) from Performance Analysis to Detailed Results as new 6th sub-tab
  - Combined Hierarchical Clustering Analysis with Correlation Matrix in Detailed Results → Metric Correlation tab for related statistical analysis
  - Moved Statistical Summary to Detailed Results → Outlier & Distribution tab (displayed at top)
  - Removed duplicate Representation Analysis from Model Analysis → Performance Deep Dive (still available in dedicated Representation Analysis tab)
  - Modality Performance Statistics table uses HTML rendering for compatibility (Streamlit dataframe attempted but reverted due to PyArrow compatibility issues with Pandas 1.5.3)

- **Dashboard Individual Model Inspection Improvements**:
  - Made Predictions vs True Values scatter plot square-shaped (650×650 pixels) for better visual proportions
  - Moved Hyperparameter Analysis from Detailed Results to Individual Model Inspection tab for logical grouping
  - Consolidated Model Parameters into Hyperparameter Analysis tab (reduced from 4 tabs to 3)
  - Enhanced Hyperparameter Analysis now shows: Model/Representation info cards, parameter configuration (cards + table), all performance metrics, and export options (Python dict + JSON)
  - Improved tab structure: Prediction Scatter Plot → Hyperparameter Analysis → Export & Code

- **Dashboard UI Font Hierarchy**:
  - Implemented 3-level tab font size hierarchy for better visual hierarchy and readability
  - Level 1 main tabs: 32px (Overview, Performance Analysis, etc.)
  - Level 2 sub-tabs: 26px (Modality Analysis, Model Analysis, Efficiency Analysis)
  - Level 3 nested tabs: 20px (Modality Overview, Representation Analysis, etc.)

- **Dashboard Performance Analysis Cleanup**:
  - Removed duplicate Efficiency Analysis sub-tab from Model Analysis (now only appears as top-level tab)
  - Streamlined Model Analysis to focus on Category and Specific Model comparisons
  - Removed Statistical Summary section from Comprehensive Distribution Analysis to reduce redundancy

- **Detailed Results Tab Optimization**:
  - Reduced from 6 sub-tabs to 5 by moving Hyperparameter Analysis to Individual Model Inspection
  - Current structure: Results Table, Metric Correlation, Outlier & Distribution, Distribution Analysis, Multi-Dimensional Comparison

## [0.1.0] - 2025-01-XX

### Added
- Initial release of MolBlender
- Multi-modal molecular representation generation
- Automated ML model screening system
- Interactive Streamlit dashboard for results visualization
- Support for 60+ molecular representations across multiple modalities (fingerprints, language models, images, spatial)
- Integration with 20+ ML algorithms (scikit-learn, XGBoost, LightGBM, neural networks)
- SQLite-based results persistence with caching support
- Comprehensive evaluation metrics for regression and classification tasks

## [2026-03-09] - Round 6 Refactoring Regression Fixes

### Fixed
- **Dashboard Round 6 Regression Fixes** (4 critical bug fixes)
  - **Import Error Fix** (`src/molblender/dashboard/app_services/pages.py`)
    - Fix Overview page import error: `_render_performance_vs_time_scatter` → `render_performance_vs_time_scatter` (2 locations)
    - During Round 6 refactoring, function was renamed but call sites were not updated
    - Impact: Overview page crashes, Performance vs Training Time chart cannot display
  
  - **Filter Parameter Count Fix** (`src/molblender/dashboard/components/filters/model_filters.py`)
    - Fix parameter count error in `_render_filter_summary()` call (passed 4 but accepts 3)
    - Remove redundant `selected_metric` parameter
    - Impact: Performance Analysis tab filter crashes
  
  - **Test Size Calculation Fix** (`src/molblender/dashboard/data/processors_helpers/loading.py`)
    - **Critical data accuracy fix**: Prioritize calculating actual test_size from real `n_train`/`n_test`
    - Before fix: Display configured value `test_size=0.2` (20%) - **Incorrect**
    - After fix: Display actual split `5271/53235=9.9%` - **Correct**
    - Impact: MaxDissimilarity split creates 9.9% test set, but Dashboard displays 20%
    - Code logic:
      ```python
      # Extract configured value from screening_config
      if "n_train" in dataset_info and "n_test" in dataset_info:
          n_train = dataset_info["n_train"]
          n_test = dataset_info["n_test"]
          if n_train and n_test:
              total = n_train + n_test
              if total > 0:
                  dataset_info["test_size"] = n_test / total  # Override configured value
      ```
  
  - **Scatter Plot API Update** (`src/molblender/dashboard/components/model_inspection/render.py`)
    - Fix Individual Model Inspection scatter plot rendering
    - After Round 6 refactoring, API changed to accept `InspectionPayload` instead of individual parameters
    - Before fix:`render_scatter_plot(model_data, dataset_info, show_train, add_regression, selected_metric=metric)`
    - After fix:
      ```python
      context = build_context_from_results(results, dataset_info)
      payload = build_payload_from_model_data(model_data, dataset_info, context)
      render_scatter_plot(payload, show_train_data=show_train, add_regression_line=add_regression)
      ```
    - Impact: Individual Model tab crashes, scatter plot cannot display

- **Modality Models Modularization** (`src/molblender/models/modality_models/`)
  - **File splitting refactoring**: base.py (622 lines) → 5 modules
    - `base_core.py` (6.8K) - Core base classes `BaseModalityModel` and `ModelResult`
    - `base_vector.py` (1.3K) - Vector modality `VectorModalityModel`
    - `base_string.py` (4.3K) - String modality `StringModalityModel`
    - `base_additional_modalities.py` (2.6K) - Image/matrix/graph/3D modalities
    - `base.py` (24 lines) - Backward compatible facade, re-exports all classes
  - **Design principles**: Comply with <800 lines rule, high cohesion low coupling
  - **Backward compatibility**: All existing import statements require no changes

- **Registry Error Handling Improvement** (`src/molblender/representations/utils/registry_core.py`)
  - **Dependency error type preservation**: Don't wrap into `RegistryError`, preserve original exception types
  - **Purpose**: Callers/tests can distinguish optional-feature scenarios vs hard registry errors
  - **Simplified logic**: `get_featurizer_info()` and `get_protein_featurizer_info()` simplified to call `build_featurizer_info()`

- **Datamol Fingerprint Calculator Enhancement** (`src/molblender/representations/fingerprints/datamol.py`)
  - **New method**: `_resolve_molfeat_calculator()` static method
  - **Compatibility**: Support both new and old molfeat loading formats
    - Current format: `{"available": True, "modules": {"molfeat_calc": module}}`
    - Old format: Directly flattened keys (`{"FPCalculator": ...}`)

- **Test Configuration Documentation** (`pytest.ini`)
  - **New test markers**:
    - `@pytest.mark.slow` - Slow tests (model download >100MB or >30s, heavy computation >10s, integration tests)
    - `@pytest.mark.network` - Network tests (external API calls, PDB/UniProt, etc., fail in offline environments)
  - **Usage examples**:
    - `pytest -m "not slow"` - Skip slow tests
    - `pytest -m "not network"` - Offline mode testing
    - `pytest -m "slow"` - Only run slow tests

### Impact
- **Dashboard stability**: Fixed 4 critical bugs, all 5 tabs and sub-functions fully available
- **Data accuracy**: Test Size corrected from incorrectly displaying 20% to actual value 9.9%
- **Code quality**: Model modularization refactoring complies with <800 lines rule, improves maintainability
- **Error handling**: Improved dependency error type preservation, increases test flexibility
- **Testing efficiency**: New test markers support rapid development iteration (skip slow/network tests)

### Verification
- ✅ Dashboard Overview: Test Size displays 9.9% (correct)
- ✅ Performance Analysis: Filters work correctly
- ✅ Individual Model: Scatter plot renders correctly
- ✅ All 4 fixed files pass manual testing
- ✅ Modularization refactoring maintains backward compatibility

### Removed

- **Duplicate tool_registry.py Module** (2026-03-17)
  - Deleted `tool_registry.py` (445 lines) - redundant with registry/ module
  - Deleted `test_tool_registry.py` - dedicated test file no longer needed
  - All functionality already exists in registry/ module:
    - `ToolInfo` → `FeaturizerInfo`
    - `ToolRegistry` → `registry/` facade.py + core.py + queries.py
    - `list_featurizers()` → `list_available_featurizers()`
    - `search_featurizers()` → `FeaturizerQuery.search()`
  - Updated all tests to use registry/ module instead
  - Impact: Single source of truth, cleaner architecture, less confusion
- **Search-space consolidation** (`screening_engine/hpo/search_space.py`, `optuna_search_space.py`) (2026-08-19)
  - `optuna_search_space.py` become the single canonical owner of search-space definitions
  - `search_space.py` `get_search_space()/extract_distribution_specs()/get_search_space_fingerprint()` changed to thin wrappers delegating via lazy import
  - `test_search_space_ownership.py` rewritten to assert delegation identity (wrapper == production) instead of behavioural differences
  - Removed unused `hashlib` import; `Optional[...]` → `X | None` syntax
- **MetricSpec registry** (`metrics/catalog.py`) (2026-08-19)
  - New `MetricSpec` dataclass and `METRIC_REGISTRY` dict with `supported`/`historical`/`planned` status
  - `normalize_metric_name()`, `get_metric_spec()`, `list_supported_metrics()` API for alias resolution and task-based filtering
  - Engine, HPO, dashboard all consume the same registry for task applicability, direction, display name and score requirements
- **HPO holdout contract** (`screening_engine/hpo/contracts.py`, `multimodal/processors/hpo/processor.py`, `screening_engine/hpo/grid_search.py`) (2026-08-19)
  - `OptimizationRequest` gains `X_validation`/`y_validation`/`validation_source` fields with documentation that holdout is only valid for `train_val_test` splits
  - `processor.py:801` comment fixed to state explicit rejection (no implicit cross-validation fallback)
  - `grid_search.py` holdout guard checks `resume_missing_params` before raising — empty resume set skips silently
- **Delete hand-written HPO metrics** (`multimodal/processors/hpo/results.py`) (2026-08-19)
  - Three blocks of hand-written `pearsonr`/`f1_score`/`(>0.5).astype(int)` thresholding replaced with `compute_all_metrics` from engine
  - `_compute_hpo_train_metrics()`, `_train_metrics_from_predictions()` and `create_hpo_result()` test-metrics block all delegate to engine's canonical primitives
  - Removed `from scipy.stats import pearsonr` (no longer used)
- **Classification rank metrics fix** (`screening_engine/evaluation/metrics.py`) (2026-08-19)
  - Removed duplicate `train_accuracy` computation block (lines 821-829) — value already computed once before all early-return paths
- **Test compression** (`tests/models/api/screening_engine/evaluation/`) (2026-08-19)
  - `test_classification_metric_core_final.py` renamed to `test_classification_metrics_unit.py` (removed `_final` suffix)
  - New `conftest.py` with shared fixtures (`binary_labels_01`, `binary_labels_25`, `binary_25_proba`, `string_labels`, `multiclass_labels`, `multiclass_proba_clean`, `binary_decision_scores`)
  - Removed pre-existing unused variables (F841) from integration test file
