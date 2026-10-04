# Dashboard Smoke Tests

This page is the maintainer reference for the Dashboard smoke layer: what
behaviour must hold, which code owns it, and which tests cover it. It is not a
user guide; for usage see {doc}`../usage/dashboard/index`.

All behaviour below is owned by
`molblender/dashboard/components/scope_selector.py` and
`molblender/dashboard/metrics/scope.py`, and is covered by the tests listed at
the end of each section.

## Default: primary, as reported

`render_scope_selector()` always offers `Primary (as reported)` first and
defaults to it (`index=0`). Returning `None` means "leave the DataFrame
untouched", which preserves the pre-scope Dashboard behaviour for users who do
not opt into a cohort.

Tests: `tests/dashboard/test_scope_selector.py`.

## Only scopes with values for the current metric

The selector derives its options from `available_scopes_for_metric()`, not from
`available_scopes()`. A scope is offered only when the currently selected metric
has at least one numeric value in that cohort, so a cohort that would render as
an all-empty chart is never listed.

When a chosen cohort has holes, the selector reports the missing count and leaves
those rows empty. A missing value is never copied from another cohort and never
falls back to the canonical scope.

Tests: `tests/dashboard/test_scope_selector.py`,
`tests/dashboard/test_scope_evaluation_modes.py`,
`tests/dashboard/test_scope_end_to_end.py`.

## Switching between train, validation, and test

Selecting `train`, `val`, or `test` resolves the metric column through
`apply_scope_to_frame()` (exposed as `apply_scope()`), which projects the chosen
cohort into the selected metric column on a copy of the frame. Existing charts
read the projected column unchanged, and sorting still uses the original
selection score rather than the projected value.

The projection is per-metric: a scope that exists for `roc_auc` but not for
`rmse` is not offered when `rmse` is selected.

Tests: `tests/dashboard/test_scope_evaluation_modes.py`,
`tests/dashboard/test_scope_end_to_end.py`.

## Mixed evaluation modes

A result set may mix standard holdout rows with `cv_only`, `hpo_cv_only`, and
`nested_cv` rows. Those rows carry their own scopes (`cv`, `hpo`, `outer_cv`),
are offered by the same selector, and are never mixed into a holdout gap. Their
presence must not hide a valid holdout gap for the rows that do carry
train/val/test values.

Tests: `tests/dashboard/test_split_comparison.py`,
`tests/dashboard/test_scope_end_to_end.py`.

## Gap fail-closed behaviour

Cross-split gaps come from `add_split_gap_columns()` and follow four rules:

1. A gap is computed only for rows that carry both sides of the pair; a row with
   a partial cohort does not participate.
2. `train-val` and `val-test` are audited independently, and each pair reports
   its own blocking reason.
3. A gap is produced only when every participating row reports
   `split_comparability == "comparable"`. A payload that is missing or corrupted
   counts as `not_audited`, never as comparable.
4. Gaps are direction-aware: a positive gap always means the earlier cohort
   scored better, for both higher-is-better and lower-is-better metrics.

When a gap is withheld, the selector explains why through
`split_gap_blocked_reason()` instead of silently showing no column. The results
table and the panel builder share the same gap policy, so both surfaces agree.

Tests: `tests/dashboard/test_split_comparison.py`,
`tests/dashboard/test_render_results_table.py`,
`tests/dashboard/test_scope_end_to_end.py`.

## Running the Dashboard smoke layer

The smoke layer splits into two groups with different dependency requirements.
Both are run from the repository root in the conda `work` environment.

### Group A — scope and provenance contract tests (no Streamlit required)

```bash
python -m pytest tests/dashboard/test_split_comparison.py \
    tests/dashboard/test_scope_evaluation_modes.py -q
```

These two files import only pandas, pytest, and `molblender.dashboard.metrics.scope`,
which is Streamlit-free by design. They are the group to run when the `dashboard`
extra is unavailable, and they cover scope resolution, participation rules, gap
fail-closed behaviour, and direction awareness.

### Group B — selector, render, and AppTest smoke tests (requires the `dashboard` extra)

```bash
python -m pytest tests/dashboard/test_scope_selector.py \
    tests/dashboard/test_render_results_table.py \
    tests/dashboard/test_scope_end_to_end.py -q
```

These files import Streamlit directly or import modules that do:
`test_scope_selector.py` and `test_scope_end_to_end.py` use
`streamlit.testing.v1.AppTest`, and `test_render_results_table.py` imports
`molblender.dashboard.components.tables`, which imports Streamlit at module
level. Without the `dashboard` extra (Streamlit, Plotly, PyArrow) they fail
during collection, not during the test body.

Running Group A and Group B together is what produced **76 passed** in this
pass. Do not assume the combined command works on a Streamlit-free environment.

The full Dashboard suite (`pytest tests/dashboard`) has **not** been executed in
the pass that produced this page; see {doc}`compatibility` for the split between
declared support and executed verification.

## Related pages

- {doc}`contracts/metrics-and-results` - scope vocabulary and gap semantics.
- {doc}`testing` - test layers and reproducible commands.
