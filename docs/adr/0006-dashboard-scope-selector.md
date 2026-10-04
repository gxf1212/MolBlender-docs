# ADR-0006: Dashboard Uses a Scope Selector, Not Per-Split Pages

## Status

✅ Accepted

## Context

Once results can carry `train`, `val`, `test`, `cv`, `hpo` and `outer_cv` values,
the obvious UI answer is a dedicated "Train/Val/Test comparison" tab. That
answer copies every chart, filter and export path once per split, and it still
does not help `cv` / `hpo` / `outer_cv` rows, which would need yet another page.
Cross-*database* comparison is already owned by `experiment_comparison.py`, so a
new comparison surface would also overlap that responsibility.

## Decision

1. **One scope selector per page**, rendered next to the existing metric
   selector on Overview, Performance Analysis and Detailed Results.
2. **The selector only offers cohorts that carry a numeric value for the
   selected metric** (`available_scopes_for_metric`). A cohort that exists only
   for another metric is not offered, so a chosen cohort always renders data
   for the metric being viewed. Nothing is inferred for a cohort that is not in
   the data.
3. **The default is `Primary (as reported)`**, which leaves the DataFrame
   untouched — existing users see exactly what they saw before.
4. **Charts are reused, not rewritten.** Selecting a scope writes the resolved
   per-row values into the selected metric column
   (`apply_scope_to_frame`), so modality sunbursts, model comparisons,
   efficiency plots, distributions and the results table all follow the choice
   without chart-level code.
5. **`cv`, `hpo` and `outer_cv` share the control** but are never mixed into a
   holdout gap: gaps exist only between `train` / `val` / `test`.
6. **Gaps are direction-aware** — a positive gap always means the earlier cohort
   did better — and are withheld unless every row reports
   `split_comparability == "comparable"`. The results table renders `train`,
   `val`, `test` plus an overfitting gap (train-val) and a generalization gap
   (val-test) through the shared `add_split_gap_columns` helper. When a gap is
   withheld, the page states the reason and the status breakdown instead of
   printing a number.
7. **Selection and ranking are unchanged by the display scope.** Ranking keeps
   using the run's selection score (`hpo_score`, CV mean, ...); the selector
   changes what is displayed, not how models were chosen.

## Consequences

- No duplicated chart code, and a future scope (for example an external
  validation cohort) only needs an entry in the shared vocabulary.
- Rows missing the chosen cohort stay empty, so a chart can legitimately show
  fewer points than the default view; a caption reports how many rows are
  affected so the difference is never mistaken for a filter.
- Display and selection diverge on purpose: a user can view Train metrics while
  the leaderboard order still reflects the validation-based selection. The
  metric cards and gap captions are the place where this is explained.
- The selector is Streamlit-UI only; all policy (scope resolution, gap
  direction, provenance gating) lives in `dashboard/metrics/scope.py`, which
  imports no Streamlit and is covered by unit tests.

## Migration

None. The default view is unchanged, and result sets without scope keys simply
fall back to the single canonical cohort.
