# TopologyNet 3D

Generate persistent homology-based topological fingerprints for molecular structures, capturing global shape and cavity features that traditional descriptors miss.

## Introduction

TopologyNet 3D implements Element-Specific Persistent Homology (ESPH) to analyze molecular topology across multiple distance scales. This approach captures structural features like connected components, cycles, and voids that are crucial for understanding binding site complementarity and molecular recognition.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🔬 **Persistent Homology**
Multi-scale topological analysis across distance thresholds
:::

:::{grid-item-card} ⚛️ **Element-Specific**
Separate analysis for C, N, O, S, P atomic groups
:::

:::{grid-item-card} 🌐 **Multi-Dimensional**
0D (components), 1D (cycles), 2D (voids)
:::

:::{grid-item-card} 📊 **Fixed Output**
512-dimensional vector for ML compatibility
:::

::::

## Quick Start

```python
import molblender as mbl

# Basic usage
featurizer = mbl.get_featurizer('topology_net_3d')
features = featurizer.featurize('c1ccc2[nH]c3ccccc3c2c1')  # indole
print(f"Shape: {features.shape}")  # (512,)
print(f"Non-zero features: {np.count_nonzero(features)}")

# Custom parameters
custom_featurizer = mbl.get_featurizer(
    'topology_net_3d',
    max_distance=10.0,  # Shorter range analysis
    n_bins=4           # Fewer distance bins
)
features = custom_featurizer.featurize('CCO')
print(f"Custom features: {features.shape}")
```

## Parameters

:::{list-table} **TopologyNet 3D Parameters**
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Default
  - Description
* - `max_distance`
  - 15.0
  - Maximum distance (Ångströms) for topological analysis
* - `n_bins`
  - 8
  - Number of distance bins for multi-scale analysis
:::

## Algorithm Overview

::::{tab-set}

:::{tab-item} Topological Features
1. **Element Grouping**: Atoms grouped by atomic number (C, N, O, S, etc.)
2. **Distance Matrix**: Pairwise Euclidean distances for each element group
3. **Multi-Scale Analysis**: Features extracted at 8 distance thresholds (0-15Å)
4. **Homology Computation**:
   - **0D**: Connected components (molecular fragments)
   - **1D**: Cycles and loops (ring systems)
   - **2D**: Voids and cavities (binding pockets)
:::

:::{tab-item} Feature Encoding
The 512-dimensional output contains:

- **0-170**: Element-specific connected components across distance scales
- **171-340**: Cycle/loop features for different elements  
- **341-512**: Void/cavity features and spatial relationships

Each distance bin contributes ~64 features, with element-specific weighting.
:::

:::{tab-item} Applications
- **Scaffold Hopping**: Identify molecules with similar topology but different chemistry
- **Binding Site Analysis**: Capture pocket shape and complementarity
- **Virtual Screening**: Topology-based similarity for hit identification
- **Lead Optimization**: Understand how structural changes affect binding
:::

::::

## Usage Examples

### Single Molecule Analysis

```python
# Analyze topological complexity
molecules = {
    'simple': 'CCO',                    # ethanol
    'aromatic': 'c1ccccc1',             # benzene
    'bicyclic': 'c1ccc2ccccc2c1',       # naphthalene  
    'complex': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # caffeine
}

featurizer = mbl.get_featurizer('topology_net_3d')

for name, smiles in molecules.items():
    features = featurizer.featurize(smiles)
    complexity = np.sum(features > 0.1)  # Count significant features
    print(f"{name}: {complexity} topological features")
```

### Batch Processing

```python
# Process compound library
compound_library = [
    'CC(=O)Oc1ccccc1C(=O)O',        # aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # caffeine
    'CCN(CC)C(=O)c1ccc2c(c1)ncn2C',  # theophylline
    'c1ccc(cc1)C(=O)O',             # benzoic acid
    'CC1=CC=C(C=C1)C(=O)O'          # p-toluic acid
]

featurizer = mbl.get_featurizer('topology_net_3d')

# Parallel processing
features = featurizer.featurize_many(compound_library, n_jobs=4)
print(f"Processed {len(features)} compounds")
print(f"Each feature shape: {features[0].shape}")  # (512,)

# Analyze topological diversity
features_array = np.array(features)
feature_variance = np.var(features_array, axis=0)
diverse_features = np.sum(feature_variance > 0.01)
print(f"Diverse topological features: {diverse_features}")
```

### Integration with Machine Learning

```python
from molblender.data import MolecularDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Load molecular dataset
dataset = MolecularDataset.from_csv(
    "binding_data.csv",
    smiles_column="SMILES", 
    target_column="binding_affinity"
)

# Add topological features
dataset.add_features("topology_net_3d", n_workers=4)

# Train topology-based model
X = np.array(dataset.features["topology_net_3d"])
y = dataset.targets["binding_affinity"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"Topology model CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

## Feature Interpretation

### Understanding Output Values

```python
# Analyze feature distribution
featurizer = mbl.get_featurizer('topology_net_3d')
features = featurizer.featurize('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')  # caffeine

# Split by feature type
components = features[:170]      # Connected components
cycles = features[171:340]       # Cycle features  
voids = features[341:512]        # Void features

print(f"Component features (avg): {np.mean(components):.3f}")
print(f"Cycle features (avg): {np.mean(cycles):.3f}")
print(f"Void features (avg): {np.mean(voids):.3f}")

# Identify most important topological features
important_indices = np.where(features > np.percentile(features, 95))[0]
print(f"Top 5% features: {len(important_indices)} positions")
```

### Similarity Analysis

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compare topological similarity
reference = 'c1ccc2[nH]c3ccccc3c2c1'  # carbazole
compounds = [
    'c1ccc2c(c1)[nH]c3ccccc23',      # carbazole (identical)
    'c1ccc2c(c1)nc3ccccc3c2=O',      # phenazin-1-one (similar)
    'c1ccc2ccccc2c1',                # naphthalene (different)
    'CCCCCCCC'                       # octane (very different)
]

featurizer = mbl.get_featurizer('topology_net_3d')

# Calculate topological fingerprints
ref_fp = featurizer.featurize(reference).reshape(1, -1)
comp_fps = featurizer.featurize_many(compounds)

# Compute similarities
similarities = cosine_similarity(ref_fp, comp_fps)[0]

for smiles, sim in zip(compounds, similarities):
    print(f"Similarity to carbazole: {sim:.3f} - {smiles}")
```

## Performance Characteristics

### Computation Time

:::{list-table} **Performance Benchmarks**
:header-rows: 1
:widths: 40 30 30

* - Dataset Size
  - Sequential
  - Parallel (4 cores)
* - 100 molecules
  - 5-10 seconds
  - 2-3 seconds
* - 1000 molecules  
  - 50-100 seconds
  - 15-30 seconds
* - 10000 molecules
  - 8-15 minutes
  - 3-6 minutes
:::

### Memory Usage

- **Per molecule**: ~1-5 MB during computation
- **1000 molecules**: ~100-500 MB total memory
- **Features storage**: 512 × 4 bytes = 2 KB per molecule

## References

- [Persistent Homology for Molecular Analysis](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00619)
- [TopologyNet Paper](https://arxiv.org/abs/2010.01196)
- [RDKit 3D Conformer Generation](https://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules)

```{toctree}
:maxdepth: 1
:hidden:
```