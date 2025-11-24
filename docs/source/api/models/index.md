# Models API Reference

Automated machine learning API for molecular property prediction with multi-modal support.

## Overview

The models module provides comprehensive machine learning capabilities:

- **Automated Screening**: Test 28+ models across multiple representations
- **Multi-Modal Support**: VECTOR, STRING, MATRIX, IMAGE modalities
- **Intelligent Scheduling**: Optimal CPU/GPU utilization
- **Results Management**: SQLite database with dashboard visualization

## Quick Example

```python
from molblender.models.api import universal_screen
from molblender.data import MolecularDataset

# Load dataset
dataset = MolecularDataset.from_csv("data.csv",
                                     input_column="SMILES",
                                     label_columns=["activity"])

# One-command screening
results = universal_screen(dataset, target_column="activity")

# View results
# molblender view ./results_folder
```

## API Documentation

```{toctree}
:maxdepth: 2
:hidden:

screening
corpus
modality_models
core
utils
```

### Main Sections

- **{doc}`screening`** - Core screening functions (`universal_screen`, `quick_screen`)
- **{doc}`corpus`** - Model definitions and parameter grids
- **{doc}`modality_models`** - Modality-specific wrappers (CNN, VAE, Transformers)
- **{doc}`core`** - Core types and base classes
- **{doc}`utils`** - Database, validation, caching utilities

## See Also

- {doc}`../../usage/models/index` - Usage guide with examples
- {doc}`../../development/architecture` - System design overview
