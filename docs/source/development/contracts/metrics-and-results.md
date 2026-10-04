# Metrics and Results Contract

This page defines the result shape shared by screening, persistence, reporting,
and the Dashboard. The contract is deliberately independent of Streamlit so it
also applies to Python return values, SQLite rows, JSON exports, and CSV views.

## Result layers

`all_results` is a list of model/representation result records. A record may
contain direct fields such as `primary_metric`, `primary_metric_name`, and
`all_metrics`. The `all_metrics` mapping is the authoritative source for
additional performance values when it is present.

The normalized metric identity is:

```text
scope + metric
```

Examples are `train_roc_auc`, `val_roc_auc`, `test_roc_auc`, `cv_roc_auc`,
`hpo_roc_auc`, and `outer_cv_roc_auc`.

## Scope vocabulary

| Scope | Meaning | Typical evaluation mode |
| --- | --- | --- |
| `train` | Metrics calculated on the training cohort | holdout evaluation |
| `val` | Metrics calculated on a validation cohort | holdout or HPO |
| `test` | Metrics calculated on the held-out test cohort | standard/holdout evaluation |
| `cv` | Cross-validation score | `cv_only` |
| `hpo` | HPO validation score | `hpo_cv_only` |
| `outer_cv` | Outer-fold score from nested CV | `nested_cv` |

The complete vocabulary is shared with the Dashboard scope resolver. A result
may contain only a subset of these scopes.

## Bare and prefixed keys

New result producers should write prefixed keys when more than one scope is
available. Existing bare keys remain supported for compatibility. A bare key
such as `roc_auc` is interpreted as the result's canonical scope, determined in
this order:

1. `primary_metric_scope`
2. `metric_scope`
3. `evaluation_mode`

An explicit prefixed key always takes precedence over a bare key. A bare key is
never copied into another scope merely because that scope is requested.

## Scope metadata

`metric_scope` records the producer's evaluation mode. `primary_metric_scope`
records the scope represented by the primary metric. `evaluation_mode` is kept
as a compatibility and diagnostic field.

When split provenance is available, producers should also persist
`split_fingerprint`, `cohort_fingerprint`, and `split_comparability`. Dashboard
gap comparisons are valid only when the relevant cohorts are comparable.

## Missing and failed values

- A missing scope is represented by an absent key or `None`; consumers must not
  fabricate it from another scope.
- `NaN` is not a valid persisted metric value.
- Failure values follow the metric catalog: higher-is-better metrics use the
  documented zero failure value, while error metrics use positive infinity.
- Metadata such as `metric_scope`, fingerprints, timestamps, and resource
  counts is not a performance metric and must not appear in metric selectors.

## Dashboard behavior

The metric selector uses the unscoped base name, for example `roc_auc`. A scope
selector next to it resolves the selected value for each row. Only scopes carrying at least one numeric value for the selected metric are
offered (a cohort present only for another metric is not listed); the default
is the primary (as-reported) view, which leaves the row's canonical scope
intact.

Selecting a scope writes the resolved values into the selected metric column,
so the existing charts render that cohort without any chart-specific code. Rows
without that scope keep an empty value — nothing is copied from another cohort.
`cv`, `hpo`, and `outer_cv` are selectable scopes but are not silently mixed
with holdout cohorts.

The Detailed Results table shows the `train`, `val` and `test` columns plus an
overfitting gap (train-val) and a generalization gap (val-test). These gaps are
direction-aware: for a higher-is-better metric, `train - val` and `val - test`
are positive when the earlier cohort scores higher; for a lower-is-better
metric, the subtraction direction is reversed, so a positive gap always means
the earlier cohort did better. The gap columns come from the shared
`add_split_gap_columns` helper, the single production path used by the results
table.

A gap is produced only when every row reports
`split_comparability == "comparable"`. Any other value — including a missing one,
which counts as not audited — blocks the gap columns and the reason is shown
with the status breakdown, for example
`split provenance does not support cross-split gaps (comparable=3, incomparable=1)`.
Blocking is the correct outcome, not a fallback: a gap between cohorts that were
never verified as comparable would be a fabricated number. Callers that
deliberately want the raw subtraction can opt out per call
(`require_comparable=False`), which no Dashboard page does.

## Compatibility policy

Older SQLite and JSON results remain readable when they contain only bare
metrics or only test metrics. The reader may derive a base metric column from
the persisted payload, but it must preserve the original `all_metrics` mapping.
Schema changes that alter key meaning require a migration and a changelog
entry. New scope keys must be covered by contract tests before they are exposed
in the Dashboard.
