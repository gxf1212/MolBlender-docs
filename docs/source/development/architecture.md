# Architecture Overview

This document provides a high-level overview of MolBlender's architecture and design principles.

## System Design Philosophy

MolBlender is built around four core principles:

1. **Modularity**: Components are independent and interchangeable
2. **Registry-Based**: Dynamic registration of featurizers and models
3. **Lazy Loading**: Dependencies loaded only when needed
4. **Type Safety**: Comprehensive type hints throughout

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
- **Parallel Executor**: Smart parallelization (light tasks in parallel, heavy tasks sequential)
- **Checkpoint Manager**: Crash recovery and incremental updates

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
- Parallel across all combinations
- CPU-bound, benefit from multiprocessing

**Heavy Tasks** (CNN, Transformers):
- Sequential execution
- GPU utilization within each task
- Prevents GPU memory conflicts

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
├── representations/     # Feature generators
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
│   ├── api/            # User-facing API
│   ├── corpus/         # Model definitions
│   ├── modality_models/ # Modality-specific wrappers
│   └── execution/      # Parallel execution
├── config/             # Global configuration
└── dashboard/          # Streamlit visualization
```

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
