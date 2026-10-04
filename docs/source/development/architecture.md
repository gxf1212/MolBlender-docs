# Architecture Overview

This document provides a high-level overview of MolBlender's architecture and design principles.

## System Design Philosophy

MolBlender is built around four core principles:

1. **Modularity**: Components are independent and interchangeable
2. **Registry-Based**: Dynamic registration of featurizers and models
3. **Lazy Loading**: Dependencies loaded only when needed
4. **Type Safety**: Comprehensive type hints throughout

Since the 2026-03 architecture consolidation, MolBlender also follows two packaging rules:

5. **Facade First**: public entrypoints stay lightweight and stable
6. **Explicit Role Boundaries**: workflow facades, engine internals, runtime infrastructure, and compatibility layers are separated

## Current Package Roles

MolBlender now exposes package-role metadata directly in code. The main roles are:

| Package | Role | Stability |
|---------|------|-----------|
| `molblender` | Top-level convenience/discovery surface | Supported |
| `molblender.api` | Unified public facade | Recommended |
| `molblender.models` | Domain models facade | Recommended |
| `molblender.representations` | Domain representations facade | Recommended |
| `molblender.data` | Data-domain facade | Recommended |
| `molblender.data.dataset` | Dataset subdomain | Recommended |
| `molblender.data.io` | IO support module | Supported |
| `molblender.data.molecule` | Molecule subdomain | Supported |
| `molblender.data.protein` | Protein subdomain | Supported |
| `molblender.data.diagnostics` | Data diagnostics subdomain | Specialized |
| `molblender.data.cache` | Data cache subdomain | Supported |
| `molblender.data.cache.multimodal` | Multimodal cache subdomain | Supported |
| `molblender.data.preprocessing` | Data preprocessing subdomain | Supported |
| `molblender.drawings` | Static plotting utilities | Supported |
| `molblender.dashboard` | Interactive dashboard UI | Supported |
| `molblender.data.diagnostics.dashboard` | Interactive diagnostics dashboard | Specialized |
| `molblender.screening.engine` | Screening engine core | Internal |
| `molblender.screening.orchestration` | Screen workflow facade | Recommended |
| `molblender.screening.runtime` | Screening execution runtime | Internal |
| `molblender.infrastructure` | Resource policy, GPU scheduling | Internal |
| `molblender.representations.utils` | Generic batching/caching helpers | Supported |

For day-to-day workflow code, prefer `molblender.api`, `molblender.models`, or
`molblender.representations`. The top-level `molblender` package remains a
supported discovery surface, but it is no longer the primary recommended entry
layer for user examples.

The workflow facade `molblender.api.dashboard.run_dashboard()` delegates
directly to the same `view` command implementation used by the CLI, instead of
mutating `sys.argv` or relying on a separate dashboard-only CLI module.
The UI package itself, `molblender.dashboard.run_dashboard()`, remains a
lower-level Streamlit launcher that can optionally accept a results path.

(metadata_routing)=
## Scikit-learn Metadata Routing

Some MolBlender splitters inherit scikit-learn estimator helpers that expose
metadata-routing methods such as `set_split_request()`. The generated API docs
may cross-reference this section when documenting inherited scikit-learn
behavior. These routing helpers are implementation details of estimator
interoperability, not a separate MolBlender runtime layer.

For tooling or CI diagnostics, the package inventory is documented in this
guide; there is no separate CLI command for an architecture snapshot.

## Architecture Diagrams

The three diagrams below are the C4 first two levels plus one data-flow view.
They only show components that exist in the code; there is no separate service,
HTTP API, or message broker in MolBlender.

### System context

```{mermaid}
graph LR
    user[User script / notebook] --> api[molblender.api]
    user --> cli[molblender.cli]
    cli --> api
    cli --> view[cli: view command]
    api --> screen[molblender.screening]
    view --> dash[molblender.dashboard]
    screen --> db[(SQLite screening_results.db)]
    dash --> db
    screen --> deps[Optional dependencies:<br/>RDKit, DeepChem, scikit-learn,<br/>PyTorch, Optuna]
    dash --> ui[Optional dependencies:<br/>Streamlit, Plotly, PyArrow]
```

Entry points that really exist:

- `molblender.api` — `create_screener`, `screen_models`, `get_featurizer`,
  `list_featurizers`, `get_featurizer_info`, `load_results`,
  `load_dashboard_data`, `run_dashboard`.
- `molblender.cli` — `view`, `export`, `info`, `list_representations`,
  `merge_databases`, `merge_session`, `model_doctor`, `calculate_metrics`.
- `molblender.dashboard` — the Streamlit UI, launched through
  `molblender.api.run_dashboard()` or the `view` command.

### Container responsibilities

```{mermaid}
graph TD
    api[molblender.api] --> screen[molblender.screening]
    screen --> engine[screening.engine]
    screen --> orch[screening.orchestration]
    screen --> runtime[screening.runtime]
    orch --> repr[molblender.representations]
    orch --> data[molblender.data]
    orch --> pers[molblender.persistence]
    orch --> metrics[molblender.metrics]
    runtime --> infra[molblender.infrastructure]
    data --> cache[data.cache]
    dash[molblender.dashboard] --> pers
    dash --> metrics
    dash --> data
```

| Container | Responsibility |
| --- | --- |
| `molblender.data` | Datasets, molecules, proteins, IO typing, quality diagnostics, cache |
| `molblender.representations` | Featurizer registry and modality-specific representations |
| `molblender.screening` | Workflow facade (`orchestration`), engine internals (`engine`), execution (`runtime`) |
| `molblender.persistence` | SQLite sessions, model results, dataset payloads, HPO trials |
| `molblender.metrics` | Metric catalog and task-aware metric validation |
| `molblender.infrastructure` | Resource policy, GPU scheduling, execution context |
| `molblender.dashboard` | Result exploration UI; reads persistence and metrics, and never writes screening results. One exception exists: `dashboard/data/loaders/migrations.py` runs legacy schema compatibility updates on an old database it opens for reading (additive `ALTER TABLE`, column renames, and backfills of `primary_metric` / `primary_metric_name`), so opening an old database from the Dashboard is not strictly read-only |

### Screening data flow

```{mermaid}
graph LR
    ds[dataset] --> split[split / cohort<br/>data.dataset.splitting]
    split --> repr[representation<br/>featurizer + cache]
    repr --> eval[evaluation and HPO<br/>screening.engine.evaluation]
    eval --> proc[result processor<br/>screening.orchestration.processors]
    proc --> am[all_metrics<br/>+ split_comparability<br/>+ split_fingerprint]
    proc --> store[(SQLite / all_results)]
    store --> dash[Dashboard loaders]
    dash --> sel[scope selector<br/>train / val / test / cv / hpo / outer_cv]
    sel --> gaps[add_split_gap_columns<br/>direction-aware gaps]
```

Two rules anchor the right-hand side of this flow:

- `all_metrics` is the authoritative per-row payload; `split_comparability`
  records whether the cohorts behind a row are comparable.
- The Dashboard never recomputes provenance. `add_split_gap_columns()` derives
  `{train,val,test}_{metric}` columns and the two gaps, and a gap is produced
  only for rows that actually carry both sides of the pair and report
  `comparable` provenance. See {doc}`contracts/metrics-and-results`.

## Core Components

### 1. Representations System

The representations system generates molecular features across multiple modalities.

**Key Classes:**
- `BaseFeaturizer`: Abstract base for all featurizers
- `FeaturizerRegistry`: Central registry for all featurizers
- `get_featurizer()`: Factory function to instantiate featurizers

**Modalities:**
- **VECTOR**: Traditional fingerprints, descriptors (e.g., Morgan, MACCS, RDKit descriptors)
- **STRING**: Raw SMILES/SELFIES for Transformers
- **MATRIX**: 2D matrices (e.g., adjacency, Coulomb)
- **IMAGE**: Molecular images (2D drawings, 3D renders)
- **LANGUAGE_MODEL**: Pre-computed embeddings (ChemBERTa, MolFormer)

**Design Pattern:**
```python
@register_featurizer("morgan_fp_r2_1024", 
                     category="fingerprints",
                     shape=(1024,))
class MorganFingerprint(BaseFeaturizer):
    EXPECTED_INPUT_TYPE = InputType.RDKIT_MOL
    OUTPUT_SHAPE = (1024,)
    
    def _featurize(self, mol, **kwargs):
        # Input is guaranteed to be RDKit Mol
        return generate_fingerprint(mol)
```

### 2. Data Management

Handles molecular data loading, validation, and preprocessing.

Current data packaging is split into three primary layers plus one supporting utility subdomain:

- `molblender.data.dataset`: dataset structures and public splitting helpers
- `molblender.data.io`: shared input typing and compatibility parsing
- `molblender.data.molecule`: single-molecule objects and file loading
- `molblender.data.protein`: protein objects and sequence/PDB helpers
- `molblender.data.diagnostics`: dataset-quality analysis
- `molblender.data.cache`: cache implementations
- `molblender.data.cache.multimodal`: specialized modality-aware cache storage
- `molblender.data.preprocessing`: feature preparation, balancing, and temporal split helpers

The diagnostics subdomain also lazily exposes its specialized UI as
`molblender.data.diagnostics.dashboard`, keeping pre-modeling dataset analysis
separate from the main results dashboard.
Likewise, `molblender.data.cache` lazily exposes its `multimodal` cache submodule.
Shared primitive validation lives in `molblender.validation`; domain-specific
validation is distributed across `data`, `metrics`, and `screening` subdomains.

**Key Classes:**
- `Molecule`: Single molecule wrapper with lazy evaluation
- `MolecularDataset`: Collection of molecules with features
- `InputType`: Enum for supported input formats

**Features:**
- Lazy feature computation
- Automatic caching
- Error resilience
- Memory efficient

### 3. Models System

Automated machine learning with multi-modal support.

**Components:**
- **Model Corpus**: Pre-defined model configurations with parameter grids
- **Modality Detector**: Automatic detection of data modalities
- **Screening Engine Core**: Dataset handling, evaluation, HPO, result processing
- **Runtime Infrastructure**: ExecutionContext, ResourcePolicy, telemetry, error classification
- **Compatibility Layers**: Legacy execution helpers kept for backward compatibility

**Screening Workflow:**
1. Detect available modalities in dataset
2. Select compatible models
3. Execute with optimal parallelization
4. Store results in SQLite database
5. Generate interactive dashboard

### 4. Dependency Management

Unified system for handling optional dependencies.

**Pattern:**
```python
from molblender.config import dependencies as deps

def featurize(self, mol):
    rdkit = deps.get_rdkit()  # Auto-raises error if missing
    Chem = rdkit['Chem']
    # Use RDKit...
```

**Benefits:**
- Graceful degradation when deps missing
- Clear error messages
- Centralized dependency checking
- Easy mocking for tests

## Data Flow

### Feature Generation Pipeline

```
Input (SMILES/SDF/CSV)
    ↓
Molecule Object
    ↓
BaseFeaturizer._prepare_input()
    ↓
Concrete Featurizer._featurize()
    ↓
Numpy Array
```

### Screening Pipeline

```
MolecularDataset
    ↓
Detect Modalities
    ↓
Select Compatible Models
    ↓
Parallel/Sequential Execution
    ↓
SQLite Database
    ↓
Dashboard Visualization
```

### Execution Layer Decisions

Screening execution has three distinct layers:

| Layer | Status | Purpose |
|-------|--------|---------|
| `molblender.screening.runtime` | Primary | Active screening execution, split planning, stage orchestration |
| `molblender.infrastructure` | Supporting | Resource policy, GPU scheduling, execution context |
| `molblender.representations.utils` | Supporting | Generic batching/caching helpers outside the screening engine |

The public workflow APIs (`molblender.screening`) delegate to `molblender.screening.runtime` for execution and to `molblender.infrastructure` for resource policy, keeping orchestration logic separate from platform concerns.

## Extension Points

### Adding New Featurizers

1. Inherit from `BaseFeaturizer`
2. Use `@register_featurizer` decorator
3. Implement `_featurize()` method
4. Define `EXPECTED_INPUT_TYPE` and `OUTPUT_SHAPE`

See {doc}`adding_features/featurizers` for details.

### Adding New Models

1. Create wrapper class (if needed)
2. Add to model corpus with parameter grid
3. Define modality compatibility
4. Register in screening system

See {doc}`adding_features/models` for details.

### Adding New Modalities

1. Define modality type
2. Implement detector logic
3. Create compatible models
4. Update screening logic

See {doc}`adding_features/modalities` for details.

## Performance Considerations

### Parallelization Strategy

**Light Tasks** (fingerprints, descriptors):
- Parallel across combinations when allowed by runtime policy
- CPU-bound, benefit from multiprocessing

**Heavy Tasks** (CNN, Transformers):
- Often routed through more conservative execution modes
- GPU utilization happens inside each task
- Runtime policy avoids oversubscription and GPU memory conflicts

### Caching Strategy

**Feature Caching:**
- Representations cached after first computation
- Invalidated on parameter changes
- Stored in `.mbl_cache/` directory

**Model Results:**
- SQLite database stores all results
- Skip completed combinations
- Support incremental updates

## Code Organization

```
src/molblender/
├── representations/     # Feature generators and representation catalogs
│   ├── fingerprints/   # Traditional fingerprints
│   ├── descriptors/    # Molecular descriptors
│   ├── sequential/     # Language models
│   ├── spatial/        # 3D representations
│   ├── graph/          # Graph representations
│   ├── image/          # Molecular images
│   ├── protein/        # Protein-specific
│   └── utils/          # Base classes, registry
├── data/               # Data management
│   ├── molecule.py     # Single molecule
│   ├── dataset/        # Dataset classes
│   └── io.py           # Input handling
├── models/             # ML models
│   ├── corpus/         # Model definitions
│   └── modality_models/ # Modality-specific wrappers
├── screening/          # Screening workflow
│   ├── engine/         # Model registry, combination selection
│   ├── orchestration/  # Universal screen, screen_models, config contracts
│   └── runtime/        # Execution, split planning, stage orchestration
├── infrastructure/     # Resource policy, GPU scheduling
├── persistence/        # SQLite storage (sessions, results, HPO)
├── drawings/           # Static plotting utilities
├── config/             # Global configuration
└── dashboard/          # Streamlit visualization
```

## Public Surface Recommendations

For user-facing code, prefer:

- `molblender.api` for quick scripts and tutorials
- `molblender.models` for richer screening workflows
- `molblender.representations` for deeper featurizer work
- `molblender.drawings` for static figures
- `molblender.dashboard` for interactive result exploration

Avoid using internal packages such as `molblender.screening.runtime` or `molblender.infrastructure` unless you are extending MolBlender itself.

## Design Patterns

### Registry Pattern
Featurizers and models are registered at import time, allowing dynamic discovery and instantiation.

### Factory Pattern
`get_featurizer()` and `get_model()` functions act as factories.

### Strategy Pattern
Different execution strategies (parallel vs sequential) based on task weight.

### Lazy Evaluation
Features computed only when accessed, with automatic caching.

## See Also

- {doc}`adding_features/index` - Extend MolBlender
- {doc}`style` - Code conventions
- {doc}`testing` - Testing guidelines
