# RDKit Fingerprints

Generate molecular fingerprints using RDKit's comprehensive fingerprinting algorithms for similarity search, machine learning, and molecular analysis.

## Introduction

RDKit fingerprints encode molecular structure as fixed-length bit vectors or count vectors. MolBlender provides a unified interface to all major RDKit fingerprint types with consistent API, flexible input handling, and optimized batch processing.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🔄 **Morgan Fingerprints**
Circular fingerprints capturing atom neighborhoods
:::

:::{grid-item-card} 🌐 **Topological Fingerprints**
Path-based structural fingerprints
:::

:::{grid-item-card} 🔑 **MACCS Keys**
166 predefined structural keys
:::

:::{grid-item-card} 👥 **Atom Pair Fingerprints**
Encode atom pairs and distances
:::

:::{grid-item-card} 🔀 **Torsion Fingerprints**
Four-atom topological paths
:::

:::{grid-item-card} ⚡ **Batch Processing**
Parallel computation support
:::
::::

## Quick Start

```python
import molblender as mbl
import numpy as np

# Three ways to provide molecular input
# Method 1: Direct SMILES string
featurizer = mbl.get_featurizer('morgan_fp_r2_2048')
fp_smiles = featurizer.featurize('CCO')
print(fp_smiles.shape)  # (2048,)
print(fp_smiles.dtype)  # uint8

# Method 2: MolBlender Molecule object
mol = mbl.Molecule.from_smiles('CCO')
fp_mol = featurizer.featurize(mol)
print(np.array_equal(fp_smiles, fp_mol))  # True

# Method 3: RDKit Mol object
rdkit_mol = mbl.mol_from_input('CCO')
fp_rdkit = featurizer.featurize(rdkit_mol)
print(np.array_equal(fp_mol, fp_rdkit))  # True
```

## Available Featurizers

:::{list-table} **RDKit Fingerprint Featurizers**
:header-rows: 1
:widths: 30 15 15 40

* - Featurizer Key
  - Type
  - Size
  - Description
* - `rdkit_fp_2048`
  - Binary
  - 2048
  - RDKit topological fingerprint (default)
* - `rdkit_fp_1024`
  - Binary
  - 1024
  - RDKit topological fingerprint (medium)
* - `rdkit_fp_512`
  - Binary
  - 512
  - RDKit topological fingerprint (compact)
* - `morgan_fp_r2_2048`
  - Binary
  - 2048
  - Morgan fingerprint, radius=2 (ECFP4-like)
* - `morgan_fp_r2_1024`
  - Binary
  - 1024
  - Morgan fingerprint, radius=2, medium size
* - `morgan_fp_r2_512`
  - Binary
  - 512
  - Morgan fingerprint, radius=2, compact
* - `morgan_fp_r3_2048`
  - Binary
  - 2048
  - Morgan fingerprint, radius=3 (ECFP6-like)
* - `morgan_fp_r3_1024`
  - Binary
  - 1024
  - Morgan fingerprint, radius=3, medium size
* - `morgan_fp_r2_2048_chiral`
  - Binary
  - 2048
  - Morgan fingerprint with chirality (R2)
* - `morgan_fp_r3_2048_chiral`
  - Binary
  - 2048
  - Morgan fingerprint with chirality (R3)
* - `morgan_hashed_count_fp_r2_8192`
  - Hashed Count
  - 8192
  - Morgan hashed count (R2, fixed dim, no truncation)
* - `morgan_hashed_count_fp_r2_16384`
  - Hashed Count
  - 16384
  - Morgan hashed count (R2, minimal collision)
* - `morgan_hashed_count_fp_r3_8192`
  - Hashed Count
  - 8192
  - Morgan hashed count (R3, fixed dim, no truncation)
* - `morgan_feature_hashed_count_fp_r2_8192`
  - Hashed Count
  - 8192
  - Morgan feature hashed count (pharmacophore)
* - `morgan_hashed_count_fp_r2_8192_chiral`
  - Hashed Count
  - 8192
  - Morgan hashed count with chirality (R2)
* - `morgan_count_fp_r2`
  - Sparse Count
  - 4096
  - [DEPRECATED] Morgan sparse count (use hashed_count instead)
* - `morgan_feature_fp_r2`
  - Sparse Count
  - 4096
  - [DEPRECATED] Morgan feature sparse count (use hashed_count instead)
* - `maccs_keys`
  - Binary
  - 167
  - MACCS structural keys
* - `atom_pair_fp`
  - Binary
  - 2048
  - Atom pair fingerprint
* - `atom_pair_count_fp`
  - Count
  - 4096
  - Atom pair counts
* - `torsion_fp`
  - Binary
  - 2048
  - Topological torsion fingerprint
* - `torsion_count_fp`
  - Count
  - 4096
  - Topological torsion counts
:::

```{tip}
Copy any featurizer key from the table above and use it directly with `mbl.get_featurizer()`!
```

## Fingerprint Types

### Morgan Fingerprints

Circular fingerprints that encode atom environments within a specified radius. Known as ECFP (Extended Connectivity Fingerprints) when radius=2 gives ECFP4.

::::{tab-set}

:::{tab-item} Basic Usage
```python
# Binary Morgan fingerprints
morgan = mbl.get_featurizer('morgan_fp_r2_2048')
fp = morgan.featurize('c1ccccc1O')  # phenol
print(f"Active bits: {np.sum(fp)}")  # Number of set bits
print(f"Fingerprint density: {np.sum(fp) / len(fp):.3f}")  # Sparsity

# Hashed count fingerprints (recommended, no truncation)
morgan_hashed = mbl.get_featurizer('morgan_hashed_count_fp_r2_8192')
fp_hashed = morgan_hashed.featurize('c1ccccc1O')
print(f"Non-zero features: {np.count_nonzero(fp_hashed)}")
print(f"Max count: {np.max(fp_hashed)}")  # Full count information preserved
```
:::

:::{tab-item} Hashed Count vs Sparse Count
```python
# ✅ RECOMMENDED: Hashed count (fixed dimension, no truncation)
morgan_hashed = mbl.get_featurizer('morgan_hashed_count_fp_r2_8192')
fp_hashed = morgan_hashed.featurize('complex_molecule')
# Output: Fixed 8192 dimensions, all counts preserved via hashing

# ⚠️  DEPRECATED: Sparse count (truncated, loses information)
morgan_sparse = mbl.get_featurizer('morgan_count_fp_r2')
fp_sparse = morgan_sparse.featurize('complex_molecule')
# Output: Variable dimensions, features beyond 4092 may be truncated
# Warning: "Sparse fingerprint: 15/4207 features (0.4%) truncated"

# 💡 Use hashed_count for complex molecules to avoid information loss
```
:::

:::{tab-item} Custom Parameters
```python
from molblender.representations.fingerprints.rdkit import (
    MorganBitFP, MorganHashedCountFP
)

# Custom binary Morgan with chirality
custom_morgan = MorganBitFP(
    radius=3,           # Larger radius captures more context
    nBits=4096,         # More bits for less collisions
    useChirality=True,  # Include stereochemistry
    useBondTypes=True,  # Consider bond types
    useFeatures=False   # Use atom types, not pharmacophores
)

# Custom hashed count with feature invariants
custom_hashed = MorganHashedCountFP(
    radius=2,
    fpSize=16384,       # Larger space reduces hash collisions
    useFeatures=True,   # Use pharmacophore features
    useChirality=True   # Include stereochemistry
)
```
:::

:::{tab-item} Applications
- **Similarity searching**: Find molecules with similar scaffolds
- **Machine learning**: Features for QSAR/QSPR models (hashed_count recommended)
- **Scaffold hopping**: Identify molecules with similar pharmacophores
- **Virtual screening**: Fast similarity-based filtering
- **Chiral separation**: Stereochemistry-aware clustering with chiral variants
:::

::::

```{warning}
**Morgan Count Migration Guide**

The old `morgan_count_fp_*` and `morgan_feature_fp_*` fingerprints use sparse count vectors that may truncate features beyond the default size. For new projects, use the new hashed count variants:

**Old (Deprecated)** → **New (Recommended)**
- `morgan_count_fp_r2` → `morgan_hashed_count_fp_r2_8192`
- `morgan_feature_fp_r2` → `morgan_feature_hashed_count_fp_r2_8192`
- `morgan_count_fp_r3` → `morgan_hashed_count_fp_r3_8192`
- `morgan_feature_fp_r3` → `morgan_feature_hashed_count_fp_r3_8192`

Benefits: No truncation, fixed dimensions, better coverage for complex molecules.
```

#### Available Morgan Variants

**Binary Morgan (Bit Vectors)**
- `morgan_fp_r2_512`, `morgan_fp_r2_1024`, `morgan_fp_r2_2048`
- `morgan_fp_r3_512`, `morgan_fp_r3_1024`, `morgan_fp_r3_2048`
- `morgan_fp_r2_512_chiral`, `morgan_fp_r2_1024_chiral`, `morgan_fp_r2_2048_chiral`
- `morgan_fp_r3_512_chiral`, `morgan_fp_r3_1024_chiral`, `morgan_fp_r3_2048_chiral`

**Hashed Count Morgan (Fixed Dimension, No Truncation)**
- `morgan_hashed_count_fp_r2_8192`, `morgan_hashed_count_fp_r2_16384`
- `morgan_hashed_count_fp_r3_8192`, `morgan_hashed_count_fp_r3_16384`
- `morgan_feature_hashed_count_fp_r2_8192` (pharmacophore features)
- `morgan_feature_hashed_count_fp_r3_8192` (pharmacophore features)
- All variants support chiral versions with `_chiral` suffix

**Sparse Count Morgan (Deprecated, May Truncate)**
- `morgan_count_fp_r2`, `morgan_count_fp_r3`
- `morgan_feature_fp_r2`, `morgan_feature_fp_r3`

### Topological Fingerprints

RDKit's native fingerprints based on paths between atoms.

```python
# Different sizes for different applications
rdkit_fp_large = mbl.get_featurizer('rdkit_fp_2048')  # More detailed
rdkit_fp_small = mbl.get_featurizer('rdkit_fp_512')   # Memory efficient

# Compare fingerprint density
mol = 'CC(C)CC(C)(C)O'  # Complex branched molecule
fp_large = rdkit_fp_large.featurize(mol)
fp_small = rdkit_fp_small.featurize(mol)

print(f"Large FP active bits: {np.sum(fp_large)}/{len(fp_large)}")
print(f"Small FP active bits: {np.sum(fp_small)}/{len(fp_small)}")
```

### MACCS Keys

Fixed set of 166 structural keys (+ 1 unused bit) designed for substructure screening.

```python
maccs = mbl.get_featurizer('maccs_keys')

# Compare different functional groups
molecules = {
    'alcohol': 'CCO',
    'ketone': 'CC(=O)C',
    'amine': 'CCN',
    'aromatic': 'c1ccccc1'
}

for name, smiles in molecules.items():
    fp = maccs.featurize(smiles)
    print(f"{name}: {np.sum(fp)} keys present")
    # Each bit corresponds to a specific substructure
```

### Atom Pair Fingerprints

Encode all atom pairs and their topological distances.

```python
# Binary atom pairs
atom_pair = mbl.get_featurizer('atom_pair_fp')
fp = atom_pair.featurize('CCOC(=O)C')  # ethyl acetate

# Count-based atom pairs (captures frequency)
atom_pair_count = mbl.get_featurizer('atom_pair_count_fp')
fp_count = atom_pair_count.featurize('CCOC(=O)C')
print(f"Unique atom pairs: {np.count_nonzero(fp_count)}")
```

### Torsion Fingerprints

Capture four consecutive bonded atoms (torsion angles in 2D).

```python
# Torsions need at least 4 atoms
torsion = mbl.get_featurizer('torsion_fp')

# Small molecule - few torsions
fp_small = torsion.featurize('CCCC')  # n-butane
print(f"Butane torsions: {np.sum(fp_small)}")

# Larger molecule - more torsions
fp_large = torsion.featurize('CC(C)CC(=O)NC1CCCCC1')
print(f"Complex molecule torsions: {np.sum(fp_large)}")
```

## Batch Processing

Process multiple molecules efficiently with automatic parallelization:

```python
# Sample dataset
smiles_list = [
    'CCO',                    # ethanol
    'CC(C)O',                 # isopropanol
    'c1ccccc1',               # benzene
    'CC(=O)O',                # acetic acid
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # caffeine
]

featurizer = mbl.get_featurizer('morgan_fp_r2_2048')

# Sequential processing
fps_sequential = featurizer.featurize_many(smiles_list)
print(f"Output shape: {fps_sequential.shape}")  # (5, 2048)

# Parallel processing (automatically uses all cores)
fps_parallel = featurizer.featurize_many(smiles_list, n_jobs=-1)

# With error handling
fps, errors = featurizer.featurize_many(
    smiles_list + ['invalid_smiles'],  # Add invalid molecule
    return_errors=True
)
print(f"Processed: {len(fps)}, Failed: {len(errors)}")
# errors contains tuples of (index, exception)
```

## Integration Examples

### Similarity Search

```python
from sklearn.metrics.pairwise import cosine_similarity

# Reference molecule
reference = 'CC(=O)Oc1ccccc1C(=O)O'  # aspirin
featurizer = mbl.get_featurizer('morgan_fp_r2_2048')
ref_fp = featurizer.featurize(reference).reshape(1, -1)

# Database to search
database = [
    'CC(=O)Oc1ccccc1',          # phenyl acetate
    'CC(=O)Nc1ccccc1C(=O)O',    # similar to aspirin
    'c1ccccc1',                 # benzene
    'CCCCCCCC'                  # octane
]

# Calculate similarities
db_fps = featurizer.featurize(database)
similarities = cosine_similarity(ref_fp, db_fps)[0]

# Rank by similarity
for idx, (smiles, sim) in enumerate(sorted(zip(database, similarities), 
                                           key=lambda x: x[1], reverse=True)):
    print(f"{idx+1}. {smiles}: {sim:.3f}")
```

## References

- [RDKit Documentation](https://www.rdkit.org/docs/index.html)
- [Fingerprints in RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity)
- [Morgan Fingerprints Paper](https://pubs.acs.org/doi/10.1021/ci100050t)
- [MACCS Keys Description](https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py)

```{toctree}
:maxdepth: 1
:hidden:
```