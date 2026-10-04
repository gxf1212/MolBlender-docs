# ADR-0005: Unified Metric Scope and Result Schema

## Status

✅ Accepted

## Context

`all_metrics` is a heterogeneous bag: performance values, provenance strings and
free-form fields share one namespace. Six result-producing paths write into it
(`standard`, `holdout_external_test`, `cv_only`, `nested_cv`, `hpo_cv_only`, and
the HPO holdout branch), and historically they wrote *bare* keys such as
`roc_auc`. A bare key is ambiguous: it means the external test set for
holdout-style rows but the CV/HPO aggregate for the CV family, so a consumer
that reads `result["all_metrics"]["roc_auc"]` can silently report a CV score as
a test score.

Adding prefixed keys (`train_*`, `test_*`, ...) without a rule would be worse:
consumers would have to know which producer wrote the row, and a whitelist that
matches only bare names would filter the prefixed keys out before they ever
reach a chart selector.

## Decision

1. **`scope + metric` is the normalized internal form.** Producers project every
   bare performance key onto `{scope}_{metric}` through
   `screening.engine.evaluation.metrics.project_metrics_with_scope`. The
   projection is whitelist-driven (`PERFORMANCE_METRIC_KEYS`, derived from
   `MetricType`), so provenance keys, already-scoped keys, `train_*` keys and
   free-form fields such as `selection_source` are structurally excluded — a
   double prefix like `val_test_*` can never be produced.
2. **Bare keys stay supported and mean exactly one thing**: the row's canonical
   scope. Canonical scope resolves in this order —
   `primary_metric_scope` → `metric_scope` → `evaluation_mode` (mapped by
   `EVALUATION_MODE_SCOPES`).
3. **An explicit prefixed key always wins over a bare key.** Resolution
   (`get_scoped_metric` in the engine, `resolve_scoped_metric` in the Dashboard)
   fails closed: when the requested scope is absent it returns `None`, never a
   value borrowed from another cohort.
4. **Split provenance travels with the row.** Producers persist
   `split_fingerprint` and `split_comparability`; consumers must gate any
   cross-cohort comparison on `split_comparability == "comparable"`.
5. **`PERFORMANCE_METRICS` membership tests must first strip the scope prefix**
   (`metric_base_name`). The whitelist has no built-in scope awareness, so a
   caller that compares raw keys drops every scoped metric.
6. `metric_schema_version` (currently `2`) is a marker written by producers. It
   is not read anywhere and carries no migration semantics; schema evolution is
   handled by the compatibility rules in ADR-0007.

## Consequences

- The scope vocabulary is duplicated by design and must be kept in sync:
  `EVALUATION_MODE_SCOPES` / `_SCOPE_BY_MODE` on the engine side and
  `KNOWN_SCOPES` on the Dashboard side. A new split strategy has to update both
  plus `evaluator.py`'s projection gate, or rows silently lose their scope.
- Adding a metric to `MetricType` automatically makes it scope-projectable,
  which is the intended behaviour and requires no whitelist edit.
- Consumers that used bare keys keep working, but they now receive the canonical
  scope only; anything cross-scope must go through the resolver.
- `serialize_model_result` coerces values with `float()`, which drops string
  provenance from the returned dict. The SQLite row keeps the original mapping,
  so provenance must be read from the database, not from the returned summary.

## Migration

No data migration. Older rows are read under decision 2; new rows carry both
forms so either reader works. Contract tests cover each evaluation mode before a
new scope key is exposed in the Dashboard.
