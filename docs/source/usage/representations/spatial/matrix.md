# Matrix Featurizers

Matrix representations encode molecular structure as 2D arrays capturing spatial relationships, connectivity, and atomic properties.

## Introduction

Matrix featurizers transform molecules into structured numerical formats suitable for machine learning. They capture different aspects of molecular structure:

- **Spatial information**: 3D atomic positions and interactions (Coulomb Matrix)
- **Connectivity**: 2D graph structure and bond types (Adjacency/Edge Matrix)
- **Fixed dimensions**: Matrices padded/truncated to `max_atoms` for ML compatibility

## Installation

```bash
# Optional backends for Coulomb Matrix
pip install deepchem  # Default backend
pip install dscribe ase  # Alternative backend
```

## Quick Start

```python
import molblender as pm
from molblender.data.molecule import Molecule

# Create a molecule with 3D coordinates
mol = Molecule.from_smiles("CCO", embed3d=True)

# Generate Coulomb Matrix
cm = pm.get_featurizer("coulomb_matrix", max_atoms=10)
features = cm.featurize(mol)
print(features.shape)  # (100,) - flattened 10x10 matrix

# Generate Adjacency Matrix (no 3D needed)
adj = pm.get_featurizer("adjacency_matrix", max_atoms=10, flatten=False)
matrix = adj.featurize("CCO")  # Direct SMILES input
print(matrix.shape)  # (10, 10) - 2D connectivity matrix
```

## Available Featurizers

:::{list-table} **Matrix Featurizers**
:header-rows: 1
:widths: 25 20 55

* - Featurizer Key
  - Output Shape
  - Description
* - `coulomb_matrix`
  - (max_atoms²,) or (max_atoms, max_atoms)
  - Atomic positions and nuclear charges
* - `coulomb_matrix_eig`
  - (max_atoms,)
  - Eigenvalues of Coulomb matrix (invariant)
* - `adjacency_matrix`
  - (max_atoms²,) or (max_atoms, max_atoms)
  - Binary connectivity (1 if bonded, 0 otherwise)
* - `edge_matrix`
  - (max_atoms²,) or (max_atoms, max_atoms)
  - Bond orders (1.0, 2.0, 3.0, 1.5 for aromatic)
:::

## Coulomb Matrix

Encodes pairwise atomic interactions based on positions and nuclear charges.

::::{tab-set}

:::{tab-item} Theory
The Coulomb matrix elements are defined as:

$$M_{ij} = \begin{cases}
0.5 Z_i^{2.4} & \text{if } i = j \\
\frac{Z_i Z_j}{|R_i - R_j|} & \text{if } i \neq j
\end{cases}$$

Where:
- $Z_i$: Nuclear charge of atom $i$
- $|R_i - R_j|$: Euclidean distance between atoms
:::

:::{tab-item} Basic Usage
```python
import molblender as pm
from molblender.data.molecule import Molecule

# Basic Coulomb Matrix
cm = pm.get_featurizer("coulomb_matrix", 
    max_atoms=20,    # Matrix size
    flatten=True     # Return 1D array
)

# Automatic 3D generation for SMILES
features = cm.featurize("CCO")
print(features.shape)  # (400,) = 20*20 flattened

# Pre-computed 3D molecule
mol_3d = Molecule.from_smiles("CCO", embed3d=True)
features_3d = cm.featurize(mol_3d)
```
:::

:::{tab-item} Advanced Options
```python
# Custom parameters
cm_custom = pm.get_featurizer("coulomb_matrix",
    max_atoms=30,
    remove_hydrogens=True,   # Ignore H atoms
    permutation='random',    # For data augmentation
    upper_tri=True,         # Only upper triangle
    flatten=False,          # Keep 2D matrix
    backend='dscribe',      # Use DScribe backend
    sigma=0.1,             # Noise for random permutation
    seed=42                # Reproducibility
)

# Permutation strategies:
# - 'sorted_l2': Sort by L2 norm (default)
# - 'eigenspectrum': Return eigenvalues only
# - 'random': Random permutation for augmentation
# - 'none': Keep original atom order
```
:::

::::

```{admonition} 3D Coordinates Required
:class: warning
Coulomb Matrix requires 3D atomic positions. MolBlender automatically generates them if missing, but pre-computed coordinates are recommended for consistency.
```

## Coulomb Matrix Eigenvalues

Permutation-invariant representation using eigenvalues of the Coulomb matrix.

```python
import numpy as np

# Eigenvalue representation
cm_eig = pm.get_featurizer("coulomb_matrix_eig", 
    max_atoms=20,
    remove_hydrogens=False
)

# Returns sorted eigenvalues
eigenvalues = cm_eig.featurize("c1ccccc1")  # Benzene
print(eigenvalues.shape)  # (20,) - one value per max atom
print(f"Non-zero eigenvalues: {np.count_nonzero(eigenvalues)}")
```

## Adjacency Matrix

Binary matrix representing molecular connectivity.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🔗 **Features**
- Binary values (0 or 1)
- Symmetric matrix
- No bond type information
- No 3D coordinates needed
:::

:::{grid-item-card} 📊 **Example**
```python
import molblender as pm

adj = pm.get_featurizer("adjacency_matrix",
    max_atoms=10,
    remove_hydrogens=True,
    flatten=False
)

# Benzene connectivity
matrix = adj.featurize("c1ccccc1")
print(matrix[:6, :6])  # 6x6 ring
# [[0. 1. 0. 0. 0. 1.]
#  [1. 0. 1. 0. 0. 0.]
#  [0. 1. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 1. 0.]
#  [0. 0. 0. 1. 0. 1.]
#  [1. 0. 0. 0. 1. 0.]]
```
:::

::::

## Edge Matrix

Encodes bond types as numerical values in the connectivity matrix.

```python
import molblender as pm

# Edge matrix with bond orders
edge = pm.get_featurizer("edge_matrix", 
    max_atoms=15,
    remove_hydrogens=False,
    flatten=True
)

# Mixed bond types example
mol = "C=CC#N"  # Double and triple bonds
features = edge.featurize(mol)

# Bond order mapping:
# 0.0 = no bond
# 1.0 = single bond
# 2.0 = double bond
# 3.0 = triple bond
# 1.5 = aromatic bond
```

## Batch Processing

Process multiple molecules efficiently:

```python
import molblender as pm

# List of molecules
molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CCC"]

# Any matrix featurizer
featurizer = pm.get_featurizer("adjacency_matrix", max_atoms=20)

# Process batch - returns list of arrays
features = featurizer.featurize(molecules)
print(len(features))  # 4 results
print(features[0].shape)  # (400,) for first molecule

# Parallel processing
features_parallel = featurizer.featurize(molecules, n_workers=4)

# Handle invalid molecules - returns None for failures
molecules_with_invalid = ["CCO", "invalid_smiles", "c1ccccc1"]
results = featurizer.featurize(molecules_with_invalid)
print(f"Valid: {sum(1 for r in results if r is not None)}")
print(f"Invalid: {sum(1 for r in results if r is None)}")
```

## Practical Tips

```{admonition} Best Practices
:class: tip

1. **Size selection**: Set `max_atoms` based on your largest molecule plus buffer
2. **Hydrogen handling**: Remove for heavy-atom focus, keep for detailed interactions
3. **Flattening**: Use flattened for traditional ML, keep 2D for specialized architectures
4. **Backend choice**: DeepChem (default) is simpler, DScribe offers more options
5. **Permutation**: Use `sorted_l2` for consistency, `eigenspectrum` for invariance
```

## Use Cases

:::{list-table}
:header-rows: 1
:widths: 30 70

* - Featurizer
  - Applications
* - Coulomb Matrix
  - Quantum property prediction, energy estimation, 3D-aware models
* - Coulomb Eigenvalues
  - Invariant molecular descriptors, similarity metrics
* - Adjacency Matrix
  - Graph neural networks, topology analysis, connectivity patterns
* - Edge Matrix
  - Bond-aware predictions, reaction modeling, chemical transformations
:::

## References

- [Original Coulomb Matrix Paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.058301)
- [DScribe Documentation](https://singroup.github.io/dscribe/latest/)
- [DeepChem Featurizers](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html)
- {doc}`/api/representations/spatial/matrix`

```{toctree}
:maxdepth: 1
:hidden:
```