# Basics

Fundamental concepts and usage patterns for working with MolBlender.

## Introduction

This section covers the essential building blocks you need to understand before diving into specific components of MolBlender. Whether you're just starting out or need a reference for core concepts, you'll find practical guides on the featurizer system, configuration management, and data representation formats.

Key topics include:
- **Featurizer System**: Understanding how to discover, instantiate, and use molecular and protein featurizers
- **Configuration**: Managing global settings, dependencies, and environment setup
- **Data Shapes & Modalities**: Understanding the different representation formats and their dimensions

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🔧 **Featurizers**
:link: featurizers
:link-type: doc

Understanding the featurizer system and registry
:::

:::{grid-item-card} ⚙️ **Configuration**
:link: config
:link-type: doc

Global settings and dependency management
:::

:::{grid-item-card} 📐 **Data Shapes**
:link: shapes
:link-type: doc

Understanding modalities and tensor dimensions
:::

::::

## Quick Start

```python
import molblender as pm

# List available featurizers
featurizers = pm.list_available_featurizers()
print(f"Available featurizers: {len(featurizers)}")

# Get a specific featurizer
featurizer = pm.get_featurizer("morgan_fp_r2_1024")

# Generate features
from rdkit import Chem
mol = Chem.MolFromSmiles("CCO")
features = featurizer.featurize(mol)
print(f"Feature shape: {features.shape}")
```

## See Also

- [Representations](../representations/index.md) - Detailed guides for specific representation types
- [Data Management](../data/index.md) - Working with molecular datasets
- [Machine Learning Models](../models/index.md) - Model screening and optimization

```{toctree}
:maxdepth: 1
:hidden:

featurizers
config
shapes
```
