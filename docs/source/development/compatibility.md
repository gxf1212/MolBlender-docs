# Compatibility Contract

This page records what MolBlender promises across Python versions, optional
dependency sets, older databases, deprecated import paths, and Streamlit-free
environments. It is generated from the actual packaging metadata and runtime
code:

- packaging metadata — `pyproject.toml`
- schema compatibility — `src/molblender/persistence/store/schema_bootstrap.py`
- deprecation category — `src/molblender/contracts/deprecations.py`
- legacy API facades — `src/molblender/models/api/`
- Streamlit-free contract — `src/molblender/dashboard/metrics/scope.py`

## Python version support

| Item | Value |
| --- | --- |
| `requires-python` | `>=3.9` |
| Classifiers | Python 3.9, 3.10, 3.11 |
| Type-check target | `python_version = "3.9"` |

The classifiers are the **declared support range**. They are not, by themselves,
evidence of a verification run on every listed version; see
[Verification scope](#verification-scope) for what has actually been executed.

Two consequences follow from the 3.9 floor:

- Modules that use PEP 604 unions such as `str | None` in annotations must carry
  `from __future__ import annotations`. `molblender/dashboard/metrics/scope.py`
  does exactly that. Code without the future import must keep `Optional[...]`
  forms so the annotations stay evaluable at runtime on 3.9.
- Newer lint releases may suggest rewriting `Optional[X]` as `X | None`. That
  suggestion is a version-driven style difference, not a defect; treat it as
  noise unless the module already imports `annotations` from `__future__`.

## Optional dependency matrix

Optional features are installed through extras. The core install already
includes DeepChem-backed features, so the `deepchem` extra is a compatibility
alias with no additional packages.

| Extra | Purpose | Notable pins |
| --- | --- | --- |
| `deepchem` | Compatibility alias; DeepChem features are in the core install | — |
| `io` | Excel and Parquet file support | `pyarrow>=10.0.0,<15.0.0` |
| `cheminformatics_ml` | `chemprop`, `mol2vec` | — |
| `openbabel` | Molecular file conversion, requires `MOLBLENDER_USE_OPENBABEL=true` | — |
| `additional_representations` | `pubchempy`, `selfies`, `schnetpack` | — |
| `protein` | Protein sequence and structure models | `biopython>=1.79`, `transformers>=4.0.0` |
| `spatial` | 3D featurizers | `unimol-tools`, `dscribe`, `ase` |
| `gnn` | PyTorch-based graph networks | `torch>=2.0.0,<2.5.0`, `dgl>=1.0.0,<2.0.0`, `torch-geometric>=2.3.0,<2.6.0` |
| `md` | Molecular dynamics trajectory analysis | `mdanalysis` |
| `ifp` | Protein-ligand interaction fingerprints | `prolif`, `MDAnalysis` |
| `drawing` | Plotting utilities | `scipy>=1.6,<1.13` |
| `models` | Traditional and deep learning estimators | `scikit-learn>=1.1`, `joblib>=1.4`, `scipy>=1.6,<1.13` |
| `dashboard` | Interactive dashboard | `streamlit>=1.28.0`, `plotly>=5.0.0`, `pyarrow>=10.0.0,<15.0.0` |
| `all` | Convenience group over the feature extras | — |
| `docs` | Documentation build | `sphinx`, `furo`, `myst-parser`, `sphinx-design` |
| `tests` | Test execution | `pytest` |

Two pin families are deliberate rather than incidental:

- `scipy<1.13` and `pyarrow<15` — compatibility with `numpy<2.0`.
- `torch<2.5` and `dgl<2.0` — the GNN compatibility matrix; newer Torch majors
  are evaluated separately before being widened.

`joblib>=1.4` is required because the parallel paths forward `initializer=`.

Optional dependencies are loaded lazily, so importing MolBlender never requires
an extra to be installed. A feature that needs a missing extra raises at the
point of use, not at import time.

## Legacy SQLite compatibility

Older screening databases remain readable and are upgraded in place. The rules
are enforced by the schema bootstrap and are described in detail in
{doc}`contracts/database-schema`:

- every statement is idempotent, so opening an existing database never destroys
  data;
- columns are only added, never removed or retyped;
- renamed legacy columns are backfilled into their successors
  (`hpo_cv_score` → `hpo_score`, `all_cv_results` → `grid_search_results`,
  `cv_scores` → `cv_fold_scores`);
- `primary_metric` is backfilled from the legacy `score` column, and
  `primary_metric_name` defaults to `pearson_r` when it is `NULL`;
- validation cohort columns (`val_*`) are added to `screening_sessions`,
  `model_results`, and `dataset_info` on open;
- rows written before identity digests existed keep
  `result_identity_key IS NULL` and are unaffected by the partial unique index.

There is no numeric schema version to migrate against: the bootstrap derives the
required state from `PRAGMA table_info` and `sqlite_master` on every open.

Result-level compatibility is separate from schema compatibility. A result whose
split provenance is missing is treated as *not audited*, never as comparable, so
Dashboard gap comparisons stay fail-closed on old rows.

## Deprecated imports and APIs

Deprecated import paths emit `MolBlenderDeprecationWarning`. The category derives
from `UserWarning`, not `DeprecationWarning`, so it stays visible under Python's
default filter stack and can still be silenced by a filter registered after
MolBlender is imported:

```python
import warnings
from molblender.contracts import MolBlenderDeprecationWarning

warnings.filterwarnings("ignore", category=MolBlenderDeprecationWarning)
```

| Deprecated | Replacement | Removal |
| --- | --- | --- |
| `molblender.models.api.utils` (and its `persistence` / `infrastructure` facades) | `molblender.persistence`, `molblender.reporting`, `molblender.screening.engine.input_validation`, `molblender.validation` | v2.0 |
| `ScreeningConfig.n_jobs` | `max_workers_per_model` for per-model parallelism, `max_cpu_cores` for the total budget | — |
| `ScreeningConfig.hpo_mode` | `hpo_validation_strategy` | — |
| `ResourceConfig` (legacy runtime config) | `ScreeningConfig` runtime fields | — |
| `model_results.score` | `model_results.primary_metric` | — |
| `model_results.hpo_cv_score` | `hpo_score` | — |
| `model_results.all_cv_results` | `grid_search_results` | — |
| `model_results.cv_scores` | `cv_fold_scores` | — |

Experimental fields are documented separately in {doc}`contracts/configuration`:
`stratify_column`, `stratify_bins`, and `stratify_labels` are accepted but may be
silently ignored because they are not wired end-to-end.

## Streamlit-free boundary

Streamlit is an optional dependency, declared only by the `dashboard` extra. The
result contract is deliberately independent of it:

- `molblender.dashboard.metrics.scope` is Streamlit-free by design; it operates
  on the raw `all_metrics` payload and pandas frames only.
- Data preparation modules build result frames without importing Streamlit.
- UI modules import Streamlit lazily, inside the rendering function, so
  importing a dashboard module does not require Streamlit to be installed.

The practical consequence: metric scope resolution, gap computation, and
provenance gating can be used from the Python API, from tests, and from
CI without a Streamlit installation, and the same code path produces the
Dashboard's numbers.

## Verifying compatibility

The table below lists the checks that enforce this contract. Commands are the
source of truth; a green run of them is what turns a declared guarantee into a
verified one.

| Check | Command |
| --- | --- |
| Layer and import rules | `python tests/ci/check_layer_dependencies.py` |
| Packaging metadata assertions | `pytest tests/ci/test_ci_smoke.py` |
| Deprecation category behaviour | `pytest tests/test_contracts_deprecations.py` |
| Schema bootstrap on a fresh database | `pytest tests/persistence/test_database_schema_doc_contract.py` |
| Configuration contract | `pytest tests/screening/test_configuration_doc_contract.py` |

## Verification scope

This section separates what the project guarantees from what has actually been
executed, so that a declared range is never read as a verified result.

Declared, not re-verified on every version in this pass:

- Python 3.10 and 3.11 support (verification for this pass ran on Python 3.9).
- Optional extras are declared in packaging metadata; installing and exercising
  every extra has not been repeated in this pass.

Executed in this pass:

- the layer and import rules, the packaging metadata assertions, the
  deprecation category tests, the schema bootstrap contract tests, and the
  configuration contract tests listed above;
- the scope, results-table, and mixed-mode Dashboard contract tests.

Explicitly **not** executed in this pass, and therefore not claimed here:

- a full Dashboard test-suite run;
- any re-validation against a real benchmark database. Result-shape risks in this
  pass were covered by small synthetic fixtures, which exercise the same code
  paths but are not a benchmark-scale verification.

## Related pages

- {doc}`contracts/configuration` — configuration fields and deprecated options.
- {doc}`contracts/database-schema` — tables, indexes, and migration rules.
- {doc}`contracts/metrics-and-results` — scope vocabulary shared by all consumers.
