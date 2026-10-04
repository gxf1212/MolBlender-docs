# ADR-0007: Legacy Metric-Key and SQLite Compatibility

## Status

✅ Accepted

## Context

Result databases produced before the scope contract contain only bare metric
keys, and many contain only the test value. They are long-lived artifacts that
users keep opening with newer versions of the Dashboard, so they cannot be
invalidated. At the same time the schema has no database-level version: there is
no `PRAGMA user_version`, and both `persistence/store/schema_bootstrap.py` and
`dashboard/data/loaders/migrations.py` evolve tables by probing
`PRAGMA table_info` / `PRAGMA index_list` and applying idempotent `ALTER TABLE`
or rebuild-and-copy steps.

## Decision

1. **Old rows are readable forever.** A bare key is interpreted as the row's
   canonical scope, resolved as `primary_metric_scope` → `metric_scope` →
   `evaluation_mode`. When none of those are present, the row has no provable
   scope and scope-dependent views must fail closed rather than assume `test`.
2. **Prefixed keys win when present.** A legacy row that carries both forms is
   read through the prefixed key; the bare value is never copied into a missing
   train, validation or test value.
3. **Schema evolution stays probe-based and idempotent.** Adding a column means
   adding an `ALTER TABLE` step guarded by a `PRAGMA table_info` check, not a
   version number. `metric_schema_version` remains a written-only marker.
4. **Provenance gates are fail-closed for legacy rows.** A row without
   `split_comparability` counts as not audited, so cross-split gaps are hidden
   for pre-contract databases even when train/val/test values exist. Showing a
   gap between cohorts that were never verified as comparable would be a
   fabricated number.
5. **The reader may derive columns, never rewrite meaning.** The Dashboard may
   add a derived base-metric column, but it must preserve the original
   `all_metrics` mapping on the row.
6. **New scope keys need contract tests before exposure.** A scope is offered in
   the selector only once standard, holdout, CV, HPO and nested-CV rows have been
   covered by tests.

## Consequences

- Pre-contract databases keep working with reduced functionality: per-cohort
  values are shown, gaps are not. The UI states why, so the omission is
  explicable rather than mysterious.
- Any change that alters the meaning of an existing key is a breaking change: it
  needs a migration, a changelog entry and a major/minor version bump under
  SemVer, because the result schema is part of the compatibility promise.
- The absence of `PRAGMA user_version` means migrations must remain safe to run
  repeatedly and in any order; a migration that assumed a version step would
  break older files.

## Migration

No one-off migration is required or planned. Compatibility is handled at read
time by decisions 1–2; new artifacts additionally carry prefixed keys and split
provenance.
