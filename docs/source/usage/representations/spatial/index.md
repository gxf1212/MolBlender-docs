# Spatial Representations

Generate 3D spatial molecular representations that capture geometric structure and inter-atomic relationships for machine learning applications.

## Introduction

Spatial representations utilize the 3D coordinates of molecules to encode geometric and spatial information that 2D representations cannot capture. These are essential for tasks involving molecular properties dependent on 3D structure, such as:

- Drug-target binding affinity prediction
- Molecular property prediction (solubility, toxicity)
- Conformational analysis and ensemble studies
- 3D-QSAR modeling

MolBlender automatically handles conformer generation when 3D coordinates are missing, ensuring your workflow continues seamlessly.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 📊 **Matrix Representations**
:link: matrix
:link-type: doc
Fixed-size matrix encodings of molecular geometry
:::

:::{grid-item-card} 🧬 **3D Coordinates**
Raw atomic positions with normalization options
:::

:::{grid-item-card} 🤖 **Uni-Mol Features**
:link: unimol
:link-type: doc
Pre-trained 3D molecular embeddings
:::

:::{grid-item-card} ⚡ **Auto-Conformers**
Automatic 3D structure generation
:::
::::

## Quick Start

```python
import molblender as mbl

# Get spatial featurizers
coulomb = mbl.get_featurizer("coulomb_matrix")
coords = mbl.get_featurizer("coordinates_3d") 
unimol = mbl.get_featurizer("UniMol-CLS-unimolv2-1.1b-WithHs")  # If available

# Single molecule
smiles = "CCO"
coulomb_features = coulomb.featurize(smiles)
coord_features = coords.featurize(smiles)

print(f"Coulomb matrix shape: {coulomb_features.shape}")    # (529,) = 23×23 flattened
print(f"3D coordinates shape: {coord_features.shape}")      # (69,) = 23×3 flattened

# Batch processing with automatic conformer generation
molecules = ["CCO", "CCN", "CCC"]
batch_features = coulomb.featurize(molecules, n_workers=4)
```

## Matrix Representations

### Available Matrix Types

| Featurizer Name | Shape | Description |
|---|---|---|
| `coulomb_matrix` | (529,) | Coulomb interaction matrix (23×23) |
| `coulomb_matrix_eig` | (23,) | Eigenvalues of Coulomb matrix |
| `adjacency_matrix` | (529,) | Molecular connectivity matrix |
| `edge_matrix` | (529,) | Bond type encoding matrix |

### Coulomb Matrix

Encodes nuclear repulsion and atomic distances in a rotation-invariant matrix:

```python
import molblender as mbl
import numpy as np

# Initialize Coulomb matrix featurizer
coulomb = mbl.get_featurizer("coulomb_matrix")

# Generate features for ethanol
matrix_flat = coulomb.featurize("CCO")
matrix = matrix_flat.reshape(23, 23)  # Reshape to matrix form

# Matrix properties
print(f"Matrix symmetry: {np.allclose(matrix, matrix.T)}")  # True
print(f"Diagonal elements (nuclear charges²): {np.diag(matrix)[:3]}")
print(f"Off-diagonal elements (Coulomb interactions): {matrix[0,1]}")

# For larger molecules, matrix is automatically padded/truncated
large_mol = coulomb.featurize("C" * 30)  # Long alkane
print(f"Still same shape: {large_mol.shape}")  # (529,)
```

### Eigenvalue Representation

Rotation and translation invariant descriptor from Coulomb matrix:

```python
coulomb_eig = mbl.get_featurizer("coulomb_matrix_eig")

# Eigenvalues are sorted in descending order
eigenvals = coulomb_eig.featurize("CCO")
print(f"Eigenvalue shape: {eigenvals.shape}")      # (23,)
print(f"Largest eigenvalue: {eigenvals[0]}")       # Highest energy
print(f"Smallest eigenvalue: {eigenvals[-1]}")     # Lowest energy

# Same molecule, different conformers give similar eigenvalues
eig1 = coulomb_eig.featurize("CC(=O)O")  # Acetic acid
eig2 = coulomb_eig.featurize("CC(=O)O")  # Same molecule
correlation = np.corrcoef(eig1, eig2)[0,1]
print(f"Conformer stability: {correlation:.3f}")   # Usually > 0.9
```

### Adjacency and Edge Matrices

Encode molecular connectivity and bond types:

```python
# Adjacency matrix (binary connectivity)
adj = mbl.get_featurizer("adjacency_matrix")
adj_matrix = adj.featurize("CCO").reshape(23, 23)

# Edge matrix (bond order information)
edge = mbl.get_featurizer("edge_matrix") 
edge_matrix = edge.featurize("CCO").reshape(23, 23)

print("Bond analysis for ethanol (CCO):")
print(f"C-C bond (adjacency): {adj_matrix[0,1]}")   # 1.0 (connected)
print(f"C-C bond (edge type): {edge_matrix[0,1]}")  # 1.0 (single bond)
print(f"C=O bond would be: 2.0 (double bond)")
```

## Coordinates Representations

### Raw 3D Coordinates

Access atomic positions directly with preprocessing options:

```python
# Initialize with custom parameters
coords = mbl.get_featurizer("coordinates_3d", max_atoms=30, center=True, normalize=True)

# Get 3D coordinates
positions = coords.featurize("CCO")
coords_matrix = positions.reshape(30, 3)  # (max_atoms, 3)

print(f"Atom positions shape: {coords_matrix.shape}")
print(f"First atom position: {coords_matrix[0]}")
print(f"Coordinates centered: {np.allclose(coords_matrix[:3].mean(axis=0), 0)}")

# Compare normalized vs. non-normalized
coords_raw = mbl.get_featurizer("coordinates_3d", normalize=False)
raw_pos = coords_raw.featurize("CCO").reshape(30, 3)
print(f"Coordinate scale difference: {np.max(np.linalg.norm(raw_pos[:3], axis=1)):.2f}")
```

### Distance Matrix Representations

```python
dist_matrix = mbl.get_featurizer("distance_matrix")

# Get pairwise distances
distances = dist_matrix.featurize("CCO").reshape(23, 23)

print("Distance analysis for ethanol:")
print(f"C-C distance: {distances[0,1]:.2f} Å")
print(f"C-O distance: {distances[1,2]:.2f} Å") 
print(f"Max distance: {distances.max():.2f} Å")
```

## Uni-Mol Representations

Pre-trained 3D molecular embeddings (requires `unimol_tools` installation):

```bash
pip install unimol_tools
```

```python
# Available Uni-Mol models
unimol_models = [
    "UniMol-CLS-unimolv1-WithHs",      # 512-dim embeddings
    "UniMol-CLS-unimolv2-1.1b-WithHs", # Latest model with H
    "UniMol-CLS-unimolv2-1.1b-NoHs"   # Without explicit H
]

try:
    unimol = mbl.get_featurizer("UniMol-CLS-unimolv2-1.1b-WithHs")
    
    # Single molecule embedding
    embedding = unimol.featurize("CCO")
    print(f"Uni-Mol embedding shape: {embedding.shape}")  # (512,)
    
    # Batch processing (GPU accelerated if available)
    molecules = ["CCO", "CCN", "CCC", "C1CCCCC1"]
    batch_embeddings = unimol.featurize(molecules, n_workers=1)  # GPU batch
    
    print(f"Batch shape: {len(batch_embeddings)} molecules")
    print(f"Each embedding: {batch_embeddings[0].shape}")
    
except ImportError:
    print("Uni-Mol requires: pip install unimol_tools")
```

## Conformer Handling

MolBlender automatically generates 3D conformers when needed:

### Automatic Generation

```python
# Molecules without 3D coordinates get conformers automatically
coulomb = mbl.get_featurizer("coulomb_matrix")

# These all work seamlessly:
from_smiles = coulomb.featurize("CCO")                    # SMILES → 3D
from_rdkit = coulomb.featurize(mbl.Molecule("CCO").mol)    # RDKit Mol → 3D  
from_file = coulomb.featurize("molecule.sdf")             # File → 3D

print("All generate 3D features automatically")
```

### Multi-Conformer Analysis

```python
# Generate multiple conformers for ensemble analysis
ensemble_coulomb = mbl.get_featurizer("coulomb_matrix_ensemble", n_conformers=10)

# Get features from multiple conformers
conformer_features = ensemble_coulomb.featurize("CC(C)C(=O)O")  # Flexible molecule
print(f"Ensemble features shape: {conformer_features.shape}")   # (10, 529)

# Analyze conformer diversity
std_per_feature = conformer_features.std(axis=0)
print(f"Feature variability: {std_per_feature.mean():.3f}")
```

### Custom Conformer Generation

```python
# Configure conformer generation parameters
custom_coords = mbl.get_featurizer("coordinates_3d", 
                                 conformer_method="ETKDG",    # Algorithm
                                 optimize_conformer=True,      # Energy minimize
                                 random_seed=42)               # Reproducible

features = custom_coords.featurize("C1CCCCC1")  # Cyclohexane
```

## Batch Processing & Performance

### Parallel Processing

```python
import time

# Large molecule list
molecules = ["C" * i for i in range(5, 25)]  # Alkanes of increasing size

# Serial vs parallel comparison
coulomb = mbl.get_featurizer("coulomb_matrix")

start_time = time.time()
serial_results = [coulomb.featurize(mol) for mol in molecules]
serial_time = time.time() - start_time

start_time = time.time()
parallel_results = coulomb.featurize(molecules, n_workers=4)
parallel_time = time.time() - start_time

print(f"Serial time: {serial_time:.2f}s")
print(f"Parallel time: {parallel_time:.2f}s")
print(f"Speedup: {serial_time/parallel_time:.1f}x")
```

### Memory Management

```python
# For very large datasets, use batch processing
def process_large_dataset(smiles_list, batch_size=1000):
    coulomb = mbl.get_featurizer("coulomb_matrix")
    results = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        batch_features = coulomb.featurize(batch, n_workers=4)
        results.extend(batch_features)
        
        # Optional: Clear GPU memory if using Uni-Mol
        if i % 5000 == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results
```

## Integration with Dataset

```python
from molblender.data import MolecularDataset

# Create dataset with 3D features
molecules = ["CCO", "CCN", "CCC"]
dataset = MolecularDataset.from_smiles(molecules)

# Add spatial features
dataset.add_features("coulomb_matrix", n_workers=4)
dataset.add_features("coordinates_3d", n_workers=4)

# Access features
print("Dataset with spatial features:")
print(dataset.features.columns.tolist())
print(f"Feature shapes: {[f.shape for f in dataset.features.iloc[0]]}")

# Save/load with spatial features preserved
dataset.to_pickle("spatial_dataset.pkl")
```

## Troubleshooting

### Common Issues

**Conformer generation fails:**
```python
# Check for problematic molecules
problematic = []
molecules = ["CCO", "C#C", "[H]"]  # Include edge cases

coulomb = mbl.get_featurizer("coulomb_matrix")
for mol in molecules:
    try:
        result = coulomb.featurize(mol)
        print(f"{mol}: Success, shape {result.shape}")
    except Exception as e:
        print(f"{mol}: Failed - {e}")
        problematic.append(mol)
```

**Memory issues with large molecules:**
```python
# Use size limits
coords = mbl.get_featurizer("coordinates_3d", max_atoms=50)  # Increase limit
large_mol = "C" * 100  # 100-carbon chain
try:
    result = coords.featurize(large_mol)
    print("Large molecule handled successfully")
except Exception as e:
    print(f"Size limit exceeded: {e}")
```

**Performance optimization:**
```python
# Monitor conformer generation time
import logging
logging.getLogger("molblender.representations.spatial").setLevel(logging.INFO)

# This will show conformer generation progress
coulomb = mbl.get_featurizer("coulomb_matrix")
result = coulomb.featurize("CC(C)(C)C(=O)O")  # Branched molecule
```

## References

- [Coulomb Matrix (2012)](https://doi.org/10.1103/PhysRevLett.108.058301) - Original Coulomb matrix paper
- [Uni-Mol (2023)](https://openreview.net/forum?id=6K2RM6wVqKu) - Pre-trained 3D molecular representations
- [RDKit Conformers](https://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules) - 3D conformer generation

```{toctree}
:maxdepth: 1
:hidden:

matrix
unimol
```