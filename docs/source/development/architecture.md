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
| `molblender.models.api.core` | Screening engine core | Internal |
| `molblender.models.api.infrastructure` | Screening runtime infrastructure | Internal |
| `molblender.execution` | Generic execution helpers | Supported |
| `molblender.models.execution` | Legacy model execution | Compatibility |

This metadata can be queried programmatically:

```python
from molblender.architecture_roles import (
    get_package_role_catalog,
    get_recommended_entrypoints,
    get_execution_layer_decisions,
)

catalog = get_package_role_catalog()
recommended = get_recommended_entrypoints()
execution_layers = get_execution_layer_decisions()
```

For day-to-day workflow code, prefer `molblender.api`, `molblender.models`, or
`molblender.representations`. The top-level `molblender` package remains a
supported discovery surface, but it is no longer the primary recommended entry
layer for user examples.

Likewise, `molblender.models.execution` remains available for compatibility,
but package-level imports such as `from molblender.models.execution import
ParallelExecutor` now emit deprecation warnings that point new screening runtime
work toward `molblender.models.api.infrastructure`.

For tooling or CI diagnostics, MolBlender can emit a JSON architecture snapshot:

```bash
python -m molblender.architecture_roles
```

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

Execution-related code now has three distinct layers:

| Layer | Status | Purpose |
|-------|--------|---------|
| `molblender.models.api.infrastructure` | Primary | Active runtime policy for screening workflows |
| `molblender.execution` | Supported | Generic execution/batching helpers outside the screening engine |
| `molblender.models.execution` | Compatibility | Legacy executor/checkpoint APIs retained for older imports |

This distinction matters because the public workflow APIs should depend on `models.api.infrastructure`, not on the legacy executor packages.

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
│   ├── api/            # Screening engine, runtime infrastructure, results DB
│   ├── corpus/         # Model definitions
│   ├── modality_models/ # Modality-specific wrappers
│   └── execution/      # Legacy execution compatibility layer
├── execution/          # Generic execution helpers
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

Avoid using internal packages such as `molblender.models.api.core` or `molblender.models.api.infrastructure` unless you are extending MolBlender itself.

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
