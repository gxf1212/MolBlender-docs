# Screening Configuration Contract

This page documents the configuration surface that drives a screening run. It is
generated from the actual implementation rather than from historical notes:

- `ScreeningConfig` — `src/molblender/screening/engine/base.py`
- configuration compiler — `src/molblender/screening/orchestration/configuration.py`
- normalization and validation rules — `src/molblender/screening/engine/config_normalization.py`

`ScreeningConfig` is the engine-level contract. Callers normally obtain it from
the public compiler `compile_universal_screen_config()`, which assembles grouped
configuration objects (screening, split, resource, HPO, database, weighting,
fusion) into a `CompiledScreeningConfig`. Direct construction of
`ScreeningConfig` is also supported and runs the same registry validation, so
both paths fail closed on the same inputs.

## Construction and validation order

`ScreeningConfig.__post_init__()` performs five steps in a fixed order; each step
can reject the configuration:

1. Normalize `task_type` through `validate_task_type()`, so a documented string
   such as `"regression"` is stored in canonical form.
2. Resolve `hpo_validation_strategy`, merging the legacy `hpo_mode` alias.
3. Apply the preset (`quick`, `standard`, `performance`; `custom` applies
   nothing). The preset only overrides fields that are still at their default.
4. Validate the HPO method, validation strategy, and split strategy combination
   through `validate_hpo_configuration()`.
5. Reject unsupported `primary_metric` and `hpo_scoring` values against the
   metric registry for the resolved task family.

Steps 4 and 5 run *after* the preset, so a preset that enables HPO is still
checked against the capability matrix.

## Required fields

| Field | Type | Notes |
| --- | --- | --- |
| `task_type` | `TaskType` or string | Normalized to canonical form; every string is treated as classification before normalization, which is why normalization happens first |
| `primary_metric` | `MetricType` or string | Must be a registry metric that the engine can compute for the task family; historical and planned metrics are rejected at construction time |

## Presets

| Preset | `cv_folds` | Default DL epochs (`vae` / `transformer` / `cnn`) | HPO |
| --- | --- | --- | --- |
| `quick` | `1` | `15` / `10` / `10` | disabled |
| `standard` (default) | `3` | `50` / `30` / `30` | unchanged (default off) |
| `performance` | `5` | `50` / `30` / `30` | enabled, `hpo_stage="coarse"` |
| `custom` | unchanged | unchanged | unchanged |

A preset value is applied only when the field still holds its default value, so
an explicit user value always wins. The compiler path always passes
`preset="custom"` and supplies every field itself.

## Split configuration

`split_strategy` selects the cohort layout. The accepted values are validated by
`VALID_STRATEGIES` in the configuration compiler:

```text
train_test, train_val_test, nested_cv, cv_only, scaffold, dnr,
max_dissimilarity, maxmin, butina, feature_clustering, umap_clustering,
splito_perimeter, splito_molecular_weight, splito_max_dissimilarity,
splito_scaffold, splito_mood, user_provided
```

| Strategy | Cohorts produced | Required parameters |
| --- | --- | --- |
| `train_test` | train, test | `test_size` |
| `train_val_test` | train, val, test | `0 < val_size < 1` |
| `nested_cv` | outer CV folds with an inner CV loop | `outer_cv_folds >= 2`, `inner_cv_folds >= 2`; group-aware inner or outer strategies require `groups_column` or `groups` |
| `cv_only` | cross-validation folds only | `cv_folds` |
| `user_provided` | caller-supplied indices | `user_splits` with `train_indices` and `test_indices`, or a dataset carrying `split_info` metadata |
| molecular strategies (`scaffold`, `dnr`, `maxmin`, `max_dissimilarity`, `butina`, `feature_clustering`, `umap_clustering`, `splito_*`) | strategy-specific train/test (or train/val/test) partitions | strategy-specific parameters such as `scaffold_func`, `dnr_threshold`, `butina_similarity_threshold`, `n_clusters`, `umap_*`, `splito_*` |

### Shared-split provenance

Screening shares one split across representations so that cross-representation
ranking is meaningful. Two fields govern its fail-closed behaviour:

- `enforce_split_reference_for_x_dependent` (default `True`): an X-dependent
  strategy (`feature_clustering`, `umap_clustering`, `max_dissimilarity`,
  `butina`, `maxmin`) requires `split_reference_representation` to be set
  explicitly. Without a fixed reference, the shared split would be seeded by the
  first dict entry of `representations`, which is arbitrary across runs.
- `allow_incomparable_splits` (default `False`): an opt-in escape hatch. When
  `True`, a shared-split failure falls back to per-representation splits and tags
  every result with `split_comparability="incomparable"`, which excludes it from
  cross-split gap comparisons in the Dashboard.

`evaluate_holdout_cv` (default `True`) is independent of `cv_only` and
`nested_cv`: standard holdout evaluation may additionally run a diagnostic CV on
the training cohort.

## Cross-validation and nested CV

| Field | Default | Meaning |
| --- | --- | --- |
| `cv_folds` | `5` | Folds for plain CV and for the diagnostic holdout CV |
| `outer_cv_folds` | `5` | Outer folds under `nested_cv` |
| `inner_cv_folds` | `3` | Inner folds used for HPO under `nested_cv` |
| `outer_split_strategy` | `kfold` | Outer split strategy under `nested_cv` |
| `inner_split_strategy` | `kfold` | Inner split strategy under `nested_cv` |
| `groups_column` | `None` | Column holding group labels for group-aware strategies |
| `groups` | `None` | Pre-extracted group labels |
| `stratify` | `None` | Pre-extracted stratification labels (production path for regression stratification) |

`stratify_column`, `stratify_bins`, and `stratify_labels` are experimental
placeholders: they are not wired end-to-end and may be silently ignored. Use
`stratify` instead.

## Hyperparameter optimization

HPO is off by default (`enable_hpo=False`). When it is enabled, the method and
validation strategy must satisfy the capability matrix enforced by
`validate_hpo_configuration()`:

| `hpo_method` | `hpo_validation_strategy` | `split_strategy` | Allowed |
| --- | --- | --- | --- |
| `grid` | `cross_validation` | any | yes |
| `grid` | `holdout` | `train_val_test`, `user_provided` | yes |
| `grid` | `holdout` | any other | no |
| `random` | `cross_validation` | any | yes |
| `optuna` | `cross_validation` | any | yes |
| `optuna` | `holdout` | any | no |
| any | `holdout` | `cv_only`, `nested_cv` | no |

`hpo_validation_strategy` defaults to `cross_validation` when unset. The legacy
`hpo_mode` field is still accepted as an `InitVar` and normalized to the same
vocabulary; if both are supplied and normalize to different values, construction
raises `ValueError`.

| Field | Default | Meaning |
| --- | --- | --- |
| `hpo_stage` | `coarse` | `coarse`, `fine`, `ultrafine`, `customized`; automatically set to `customized` when `custom_param_grids` is supplied |
| `hpo_method` | `grid` | `grid`, `random`, `optuna` |
| `hpo_cv_folds` | `None` | CV folds for HPO; `None` falls back to `cv_folds` or `inner_cv_folds` |
| `top_n_for_hpo` | `10` | Stage-1 candidates promoted to HPO |
| `hpo_selection_scope` | `global` | `global`, `per_type`, `per_subtype` |
| `hpo_selection_unit` | `combo` | `combo`, `representation_existing` (alias `representation`); `representation_all_routed` is planned |
| `hpo_scoring` | `None` | Separate HPO objective; must be a registry metric for the task family or a scikit-learn scorer name |
| `optuna_n_trials` | `50` | Trial budget for `hpo_method="optuna"` |
| `optuna_timeout` | `None` | Total Optuna timeout in seconds |
| `optuna_pruning` | `True` | Median pruning of unpromising trials |
| `optuna_sampler` | `tpe` | `tpe`, `random`, `cmaes`, `nsga2`, `nsga3` |
| `optuna_warm_start` | `True` | Run the coarse grid first and inject the trials as priors |

Two fail-closed flags govern test-set usage:

- `allow_test_fallback` (default `False`): when `False`, a test-set winner is
  never used as a ranking fallback.
- `evaluate_all_hpo_params_on_test` (default `False`): diagnostic only; when
  `True`, every HPO parameter set is also evaluated on the test cohort.

`allow_legacy_unfingerprinted_hpo_priors` (default `False`) controls whether HPO
priors and Stage-1 rows with a `NULL` split fingerprint are accepted.

## Resource and parallelism

Runtime parallelism uses three distinct concepts, and mixing them up is the most
common configuration error:

| Field | Default | Meaning |
| --- | --- | --- |
| `max_cpu_cores` | `-1` | Total CPU budget for the screening process; `-1` means auto. Resolved into combination-level worker count by `resolve_parallel_workers_from_config()` |
| `max_workers_per_model` | `1` | Per-model worker cap for estimators with internal parallelism (for example RandomForest or XGBoost) |
| `n_jobs` | — | Deprecated public alias that historically overlapped with `max_workers_per_model`; normalize it through the shared compatibility shim instead of storing new logic inline |

| Field | Default | Meaning |
| --- | --- | --- |
| `parallel_models` | `True` | Enable parallel model evaluation |
| `parallel_backend` | `multiprocessing` | `multiprocessing`, `threading`, `sequential` |
| `model_batch_size` | `4` | Models evaluated per parallel batch |
| `max_parallel_jobs` | `-1` | Maximum parallel jobs for model evaluation; `-1` means auto |
| `auto_resource_optimization` | `True` | Infrastructure-driven resource scheduling |
| `execution_preference` | `balanced` | `speed`, `memory`, `balanced` |
| `gpu_devices` | `None` | Explicit CUDA device ids; unset means auto-detect |
| `enable_heavy_gpu_scheduler` | `False` | GPU slot policy for heavy-model scheduling and GPU-aware featurization |
| `isolate_heavy_models` | `True` | Run inherently heavy estimators (VAE, CNN, transformer, torch-backed) in the bounded heavy queue. This is a safety guarantee, not a GPU policy, and is deliberately independent of `enable_heavy_gpu_scheduler` |
| `heavy_jobs_per_gpu` | `1` | Concurrent heavy jobs per GPU |
| `heavy_max_parallel_jobs` | `None` | Total heavy-job cap; `None` means the number of GPU slots |
| `heavy_model_keywords` | `[]` | Extra name keywords treated as heavy in addition to the registry classification |
| `representation_loading_mode` | `eager` | `eager` keeps every representation resident; `bounded_batch` loads, evaluates, and releases a memory-bounded batch |
| `representation_batch_size` | `None` | Chunk size in `bounded_batch` mode |
| `representation_memory_budget_mb` | `None` | Memory ceiling for one bounded batch |
| `representation_load_workers` | `1` | Loading parallelism |
| `representation_prefetch` | `False` | Prefetch the next batch |
| `task_parallel` | `False` | Opt-in task-level parallelism over `(representation, model)` pairs instead of representation-level scheduling |
| `task_timeout_seconds` | `None` | Hard per-combination timeout for the process-backed task pool |
| `task_pool_memory_budget_mb` | `None` | Task-pool process memory budget, decoupled from the loading budget |
| `task_pool_memory_fraction` | `0.5` | Fraction of usable commit headroom available to the task pool |
| `task_pool_min_free_commit_mb` | `None` | Free commit to keep in reserve |
| `task_pool_memory_context_factor` | `2.0` | Multiplier turning representation context bytes into a per-task memory debit |
| `disable_task_pool_scheduler` | `False` | Force the legacy static pool-wave path |

`bounded_batch` requires `representation_batch_size` or
`representation_memory_budget_mb`; otherwise compilation fails closed.

The legacy `ResourceConfig` (`resource_config`, default `None`) exists only for
compatibility with older execution helpers. New workflows should use the
`ScreeningConfig` runtime fields above.

## Timeouts

| Field | Default | Meaning |
| --- | --- | --- |
| `model_timeout` | `None` | Explicit per-model timeout in seconds; `None` means adaptive |
| `base_model_timeout` | `600` | Base for adaptive calculation |
| `min_model_timeout` | `60` | Lower bound |
| `max_model_timeout` | `3600` | Upper bound |

## Storage and resume

| Field | Default | Meaning |
| --- | --- | --- |
| `enable_db_storage` | `True` | Persist results to SQLite |
| `db_path` | `None` | Database path; falls back to the default screening database |
| `resume_session_id` | `None` | Resume a session and skip completed combinations |
| `skip_existing_results` | `True` | Skip results already present in the resumed session |
| `save_models` | `False` | Persist fitted models |
| `enable_caching` | `True` | Enable representation caching |
| `verbose` | `1` | Verbosity level |

## Feature selection, weighting, fusion, and fallback

| Field | Default | Meaning |
| --- | --- | --- |
| `enable_feature_selection` | `True` | Drop zero-variance features |
| `zero_variance_threshold_default` | `0.95` | Default threshold |
| `zero_variance_threshold_fingerprints` | `0.99` | Fingerprint threshold |
| `zero_variance_threshold_matrices` | `0.95` | Matrix threshold |
| `zero_variance_threshold_images` | `0.95` | Image threshold |
| `use_sample_weights` | `False` | Enable sample weighting for imbalanced regression |
| `weight_strategy` | `threshold` | `threshold`, `quantile`, `tail_distance` |
| `weight_threshold` | `0.0` | Threshold for the `threshold` strategy |
| `weight_quantile` | `0.75` | Quantile for the `quantile` strategy |
| `weight_power` | `0.6` | Smoothing exponent for weight differences |
| `enable_compatibility_fallback` | `True` | Retry failed routes with a vectorized traditional-ML fallback |
| `force_compatibility_fallback` | `False` | Skip the primary route and use the fallback directly |
| `compatibility_fallback_max_features` | `256` | Apply dimensionality reduction above this width |
| `compatibility_fallback_memory_threshold_gb` | `None` | Use the fallback above this representation size |
| `compatibility_fallback_model_names` | `None` | Override the fallback model set |
| `enable_representation_fusion` | `False` | Master switch for concat-only fusion; every fusion field below is ignored when `False` |
| `fusion_groups` | `None` | Ordered component lists; each list emits one fusion row |
| `fusion_strategy` | `concat` | Only `concat` is supported |
| `fusion_include_original` | `True` | Keep the original single representations alongside fusion rows |
| `fusion_name_prefix` | `fusion` | Prefix for generated names |
| `fusion_max_components` | `4` | Per-fusion component cap |
| `fusion_max_total_features` | `None` | Optional post-filter width cap |
| `fusion_on_invalid` | `skip` | `skip` or `raise` for groups that fail to build |

Sparse input is rejected at fusion build time. `splito_candidate_methods` is a
Python-API-only field: it expects splitter objects and is not serializable to a
YAML or config-file representation.

## Model selection and combinations

| Field | Default | Meaning |
| --- | --- | --- |
| `models` | `None` | Optional model-name filter used by resume checks |
| `excluded_models` | `None` | Model names dropped globally |
| `excluded_model_representations` | `None` | `(model, representation)` pairs to drop |
| `combinations` | `None` | Explicit `Combination` objects for precise control |
| `specific_pairs_mode` | `False` | `True` when combinations specify exact pairs |
| `force_include_combos` | `None` | HPO combinations included regardless of selection score |
| `estimator_params` | `None` | Fixed estimator parameters per model |
| `custom_param_grids` | `None` | Custom HPO grids per model |
| `default_dl_epochs` | `{"vae": 50, "transformer": 30, "cnn": 30}` | Deep-learning epoch defaults, overridden by presets |

## Mutual exclusion and conflict summary

| Situation | Behaviour |
| --- | --- |
| `hpo_method="optuna"` with `hpo_validation_strategy="holdout"` | Rejected: Optuna supports cross-validation only |
| `hpo_validation_strategy="holdout"` with a split that produces no validation cohort | Rejected: only `train_val_test` and `user_provided` are accepted |
| `split_strategy` in `cv_only` or `nested_cv` with non-CV HPO validation | Rejected: these splits have no holdout validation cohort |
| `hpo_mode` and `hpo_validation_strategy` both supplied and conflicting | Rejected with `ValueError` after normalization |
| `primary_metric` that the registry cannot compute for the task family | Rejected at construction time |
| `hpo_scoring` that is neither a registry metric nor a scikit-learn scorer name | Rejected at construction time |
| X-dependent split strategy without `split_reference_representation` | Rejected unless `enforce_split_reference_for_x_dependent=False` |
| `representation_loading_mode="bounded_batch"` without a batch size or memory budget | Rejected at compilation time |
| `enable_representation_fusion=False` | All fusion fields are ignored rather than rejected |
| `custom_param_grids` supplied with a standard `hpo_stage` | `hpo_stage` is silently promoted to `customized` |

## Deprecated and experimental fields

- `hpo_mode` — deprecated `InitVar`; use `hpo_validation_strategy`.
- `n_jobs` — deprecated alias overlapping `max_workers_per_model`.
- `resource_config` (legacy `ResourceConfig`) — kept for compatibility with older
  execution helpers.
- `stratify_column`, `stratify_bins`, `stratify_labels` — experimental, not wired
  end-to-end.
- `hpo_selection_unit="representation_all_routed"` — planned, not implemented.

## Related pages

- {doc}`metrics-and-results` — how configured cohorts appear as metric scopes.
