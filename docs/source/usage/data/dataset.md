# Dataset

Comprehensive dataset handling with flexible data loading, feature computation, and batch operations.

## Overview

The `MolecularDataset` class provides a unified container for managing collections of molecules along with their associated data (labels, features, and weights). It handles:

- **Multiple input formats** (SMILES, SDF, CSV, Excel)
- **Lazy feature computation** with intelligent caching
- **Batch operations** for efficient processing
- **Data splitting** strategies for ML workflows
- **Error resilience** with graceful handling of invalid molecules

```{admonition} Key Features
:class: tip

- Automatic format detection and conversion
- Memory-efficient processing for large datasets
- Direct compatibility with scikit-learn and PyTorch
- Comprehensive data validation and cleaning
```

## Quick Start

```python
from molblender.data import MolecularDataset

# Load from CSV
dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    label_columns=["activity", "toxicity"]
)

# Add molecular features
dataset.add_features("morgan_fp_r2_1024", n_workers=4)

# Split for ML
train_set, test_set = dataset.split(test_size=0.2, random_state=42)

print(f"Training: {len(train_set)}, Test: {len(test_set)}")
```

## Topics

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 📊 **Dataset Basics**
:link: dataset_basics
:link-type: doc

Learn how to create, load, and manipulate molecular datasets from various file formats.
:::

:::{grid-item-card} 🔀 **Dataset Splitting**
:link: splitting
:link-type: doc

Professional splitting strategies including random, stratified, scaffold, and time-based splits.
:::

::::

## Dataset Workflow

```{mermaid}
graph LR
    A[Data Sources] --> B[MolecularDataset]
    B --> C[Feature Computation]
    C --> D[Data Splitting]
    D --> E[ML Training]

    A1[CSV/Excel] --> A
    A2[SDF Files] --> A
    A3[SMILES List] --> A

    D1[Random Split] --> D
    D2[Scaffold Split] --> D
    D3[Stratified Split] --> D
```

## Common Operations

### Loading Data

```python
# From SMILES list
dataset = MolecularDataset.from_smiles(
    ["CCO", "CCN", "CCC"],
    properties=[{"activity": 1}, {"activity": 0}, {"activity": 1}]
)

# From SDF file
dataset = MolecularDataset.from_sdf(
    "compounds.sdf",
    label_columns=["IC50", "LogP"]
)
```

### Feature Computation

```python
# Single featurizer
dataset.add_features("morgan_fp_r2_1024", n_workers=4)

# Multiple featurizers
dataset.add_features([
    "morgan_fp_r2_1024",
    "rdkit_descriptors",
    "maccs_keys"
], n_workers=4)
```

### Data Splitting

```python
# Random split
train, test = dataset.split(test_size=0.2, random_state=42)

# Stratified split
train, test = dataset.split(
    test_size=0.2,
    stratify='activity',
    random_state=42
)

# Scaffold split
train, test = dataset.scaffold_split(
    test_size=0.2,
    random_state=42
)
```

## Integration with ML Libraries

### Scikit-learn

```python
from sklearn.ensemble import RandomForestRegressor

# Get features and labels
X = dataset.get_feature_matrix("morgan_fp_r2_1024")
y = dataset.properties["activity"].values

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
```

### PyTorch

```python
import torch
from torch.utils.data import DataLoader

# Convert to PyTorch dataset
torch_dataset = dataset.to_pytorch(
    feature_name="morgan_fp_r2_1024",
    target_name="activity"
)

# Create DataLoader
dataloader = DataLoader(torch_dataset, batch_size=32, shuffle=True)
```

## See Also

- [Molecule Objects](molecule.md) - Individual molecular representations
- [Protein Handling](protein.md) - Protein-specific dataset management
- [Data Management Overview](index.md) - Complete data module guide

```{toctree}
:maxdepth: 1
:hidden:

dataset_basics
splitting
```
