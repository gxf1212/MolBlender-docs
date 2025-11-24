# Distance Matrix

Generate binned distance interaction matrices optimized for datasets where multiple ligands share the same protein structure, enabling efficient virtual screening and binding affinity prediction.

## Introduction

The Distance Matrix fingerprint computes binned distance interactions between ligand and protein atoms (or intra-ligand when protein not available). This approach is specifically optimized for single-protein multiple-ligand scenarios like structure-based virtual screening, with intelligent caching and efficient batch processing.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 📏 **Distance Binning**
Five distance ranges: 0-2, 2-4, 4-6, 6-8, 8-10 Å
:::

:::{grid-item-card} ⚛️ **Atom Type Pairs**
C-C, C-N, C-O, N-N, N-O, O-O interactions
:::

:::{grid-item-card} 🎯 **Protein Reuse**
Load receptor once, process many ligands efficiently
:::

:::{grid-item-card} 📊 **Fixed Output**
200-dimensional vector (5 bins × 40 atom pairs)
:::

::::

## Quick Start

```python
import molblender as pm
import numpy as np

# Basic distance matrix fingerprint
featurizer = pm.get_featurizer('protein_ligand_distance_matrix')
features = featurizer.featurize('c1ccc2[nH]c3ccccc3c2c1')  # indole
print(f"Shape: {features.shape}")  # (200,)
print(f"Distance interactions: {np.sum(features > 0)}")

# Analyze distance distribution
close_contacts = np.sum(features[:40])     # 0-2Å interactions
hbond_range = np.sum(features[40:80])      # 2-4Å interactions  
vdw_contacts = np.sum(features[80:120])    # 4-6Å interactions

print(f"Close contacts: {close_contacts:.1f}")
print(f"H-bond range: {hbond_range:.1f}")
print(f"vdW contacts: {vdw_contacts:.1f}")
```

## Distance Binning Strategy

::::{tab-set}

:::{tab-item} Distance Ranges
The fingerprint uses five biologically meaningful distance bins:

- **0-2 Å**: Very close contacts, potential clashes
- **2-4 Å**: Hydrogen bonds, strong polar interactions
- **4-6 Å**: van der Waals contacts, hydrophobic interactions  
- **6-8 Å**: Medium-range interactions, water-mediated contacts
- **8-10 Å**: Long-range effects, electrostatic interactions
:::

:::{tab-item} Atom Type Classification
Simplified but effective atom typing:

**Ligand atoms**: C, N, O, S, P (5 types)  
**Protein atoms**: C, N, O, S (4 types)

**Interaction pairs**: 5 × 4 = 20 unique pairs
**Total features**: 5 distance bins × 20 pairs = 100 → padded to 200
:::

:::{tab-item} Feature Organization
The 200-dimensional vector is structured as:

```
Bin 1 (0-2Å):  [C-C, C-N, C-O, C-S, N-C, N-N, ..., P-S] (20 pairs × 2)
Bin 2 (2-4Å):  [C-C, C-N, C-O, C-S, N-C, N-N, ..., P-S] (20 pairs × 2)  
...
Bin 5 (8-10Å): [C-C, C-N, C-O, C-S, N-C, N-N, ..., P-S] (20 pairs × 2)
```
:::

::::

## Parameters

:::{list-table} **Distance Matrix Parameters**
:header-rows: 1
:widths: 30 20 50

* - Parameter
  - Default
  - Description
* - `distance_bins`
  - [2.0, 4.0, 6.0, 8.0, 10.0]
  - Distance thresholds in Ångströms
* - `ligand_atom_types`
  - ['C', 'N', 'O', 'S', 'P']
  - Atom types considered in ligands
* - `protein_atom_types`
  - ['C', 'N', 'O', 'S']
  - Atom types considered in proteins
:::

## Usage Examples

### Single Protein Virtual Screening

```python
# Simulate structure-based virtual screening workflow
virtual_library = [
    'CC(=O)Oc1ccccc1C(=O)O',        # aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # caffeine
    'CCN(CC)C(=O)c1ccc2c(c1)ncn2C',  # theophylline  
    'c1ccc(cc1)C(=O)O',             # benzoic acid
    'CC(C)(C)NCC(c1ccc(O)cc1O)O'    # salbutamol
]

featurizer = pm.get_featurizer('protein_ligand_distance_matrix')

# Batch processing (optimized for single protein scenario)
distance_features = featurizer.featurize_many(virtual_library, n_jobs=4)
print(f"Processed {len(distance_features)} compounds")
print(f"Feature matrix: {distance_features.shape}")  # (5, 200)

# Analyze interaction patterns
for i, smiles in enumerate(virtual_library):
    features = distance_features[i]
    total_interactions = np.sum(features)
    print(f"{smiles}: {total_interactions:.1f} total interactions")
```

### Custom Distance Parameters

```python
# Custom distance binning for specific applications
short_range_featurizer = pm.get_featurizer(
    'protein_ligand_distance_matrix',
    distance_bins=[3.0, 6.0, 10.0],           # Three bins only
    ligand_atom_types=['C', 'N', 'O'],         # Focus on common atoms
    protein_atom_types=['C', 'N']              # Simplified protein typing
)

# Long-range analysis
long_range_featurizer = pm.get_featurizer(
    'protein_ligand_distance_matrix', 
    distance_bins=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0],  # Extended range
    ligand_atom_types=['C', 'N', 'O', 'S', 'P', 'F', 'Cl'], # More atom types
    protein_atom_types=['C', 'N', 'O', 'S']
)

# Compare resolutions
test_compound = 'CCN(CC)C(=O)c1ccc2c(c1)ncn2C'

short_features = short_range_featurizer.featurize(test_compound)
long_features = long_range_featurizer.featurize(test_compound) 

print(f"Short-range features: {short_features.shape}")
print(f"Long-range features: {long_features.shape}")
```

### RdRp-Ligand Dataset Example

```python
from molblender.data import MolecularDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load RdRp-ligand binding dataset
dataset = MolecularDataset.from_csv(
    "data/result_clean.csv",
    input_column='SMILES',
    label_columns=['ddG']
)

print(f"Dataset size: {len(dataset)} compounds")

# Add distance matrix features
dataset.add_features("protein_ligand_distance_matrix", n_workers=4)
X = dataset.features["protein_ligand_distance_matrix"]
y = dataset.labels["ddG"]

# Train binding affinity model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Distance Matrix Model:")
print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f} kcal/mol")

# Feature importance analysis
importances = model.feature_importances_
distance_importance = []

for bin_idx in range(5):
    bin_start = bin_idx * 40
    bin_end = (bin_idx + 1) * 40
    bin_importance = np.sum(importances[bin_start:bin_end])
    distance_importance.append(bin_importance)
    
    distance_range = f"{bin_idx*2}-{(bin_idx+1)*2}"
    print(f"Distance {distance_range}Å importance: {bin_importance:.3f}")
```

## Integration with Molecular Dynamics

```python
# Analyze conformational flexibility effects
flexible_compounds = [
    'CCCCCCCC',                              # flexible alkyl chain
    'c1ccccc1',                             # rigid aromatic
    'CC(C)CC(C)(C)O',                       # branched flexible  
    'c1ccc2c(c1)c3ccccc3c4ccccc24',        # rigid polycyclic
    'NCCCCCCCCCCCCCCCCCCCCN'                # very flexible diamine
]

featurizer = pm.get_featurizer('protein_ligand_distance_matrix')

flexibility_analysis = []
for compound in flexible_compounds:
    features = featurizer.featurize(compound)
    
    # Calculate feature distribution
    analysis = {
        'compound': compound,
        'rotatable_bonds': compound.count('C') - compound.count('c') - 1,  # Rough estimate
        'feature_variance': np.var(features),
        'interaction_spread': np.std(features[features > 0]),
        'total_interactions': np.sum(features)
    }
    flexibility_analysis.append(analysis)

# Sort by flexibility
flexibility_analysis.sort(key=lambda x: x['rotatable_bonds'])

print("Flexibility vs Distance Pattern Analysis:")
for analysis in flexibility_analysis:
    print(f"Rotatable bonds: {analysis['rotatable_bonds']}")
    print(f"  Feature variance: {analysis['feature_variance']:.3f}")
    print(f"  Interaction spread: {analysis['interaction_spread']:.3f}")
    print(f"  Total interactions: {analysis['total_interactions']:.1f}")
```

## Comparison with Other 3D Methods

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compare with other 3D fingerprints
test_compounds = [
    'CC(=O)Oc1ccccc1C(=O)O',        # aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # caffeine  
    'CCN(CC)C(=O)c1ccc2c(c1)ncn2C'   # theophylline
]

# Get multiple 3D representations
distance_fp = pm.get_featurizer('protein_ligand_distance_matrix')
topology_fp = pm.get_featurizer('topology_net_3d')
splif_fp = pm.get_featurizer('splif_enhanced')

# Calculate features
distance_features = distance_fp.featurize_many(test_compounds)
topology_features = topology_fp.featurize_many(test_compounds)
splif_features = splif_fp.featurize_many(test_compounds)

# Pairwise similarity analysis
methods = ['Distance Matrix', 'Topology Net', 'SPLIF Enhanced']
feature_sets = [distance_features, topology_features, splif_features]

for i, method in enumerate(methods):
    similarity_matrix = cosine_similarity(feature_sets[i])
    print(f"\n{method} Similarity Matrix:")
    print("         Aspirin  Caffeine  Theophylline")
    for j, compound in enumerate(['Aspirin', 'Caffeine', 'Theophylline']):
        print(f"{compound:>9}: {similarity_matrix[j][0]:.3f}    {similarity_matrix[j][1]:.3f}     {similarity_matrix[j][2]:.3f}")
```

## Performance Optimization

### Batch Processing Optimization

```python
import time

# Benchmark different batch sizes
compound_library = ['CCO'] * 1000  # 1000 identical compounds for timing
featurizer = pm.get_featurizer('protein_ligand_distance_matrix')

batch_sizes = [1, 10, 50, 100, 500, 1000]
timing_results = []

for batch_size in batch_sizes:
    start_time = time.time()
    
    # Process in batches
    all_features = []
    for i in range(0, len(compound_library), batch_size):
        batch = compound_library[i:i+batch_size]
        batch_features = featurizer.featurize_many(batch, n_jobs=4)
        all_features.extend(batch_features)
    
    elapsed_time = time.time() - start_time
    timing_results.append((batch_size, elapsed_time))
    print(f"Batch size {batch_size:>4}: {elapsed_time:.2f} seconds")

# Find optimal batch size
optimal_batch = min(timing_results, key=lambda x: x[1])
print(f"\nOptimal batch size: {optimal_batch[0]} ({optimal_batch[1]:.2f}s)")
```

### Memory Usage Analysis  

```python
import psutil
import os

def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Memory profiling
featurizer = pm.get_featurizer('protein_ligand_distance_matrix')
test_compounds = ['CCO', 'CCN', 'CCC'] * 100  # 300 compounds

print("Memory usage during processing:")
print(f"Initial: {get_memory_usage():.1f} MB")

# Process compounds
features = featurizer.featurize_many(test_compounds, n_jobs=1)
print(f"After processing: {get_memory_usage():.1f} MB")

# Feature storage
feature_size = features.nbytes / 1024 / 1024
print(f"Feature storage: {feature_size:.1f} MB")
print(f"Memory per compound: {feature_size / len(test_compounds) * 1024:.1f} KB")
```

## Applications in Drug Discovery

### Fragment-Based Drug Design

```python
# Analyze fragment binding patterns
fragments = [
    'c1ccccc1',           # benzene (aromatic)
    'CCO',                # ethanol (H-bond donor/acceptor)
    'CC(C)C',             # isobutane (hydrophobic)
    'CC(=O)O',            # acetic acid (charged)
    'c1ccncc1',           # pyridine (aromatic + H-bond acceptor)
    'CC#N',               # acetonitrile (polar)
    'CCF',                # fluoroethane (halogen)
    'c1coc2ccccc12'       # benzofuran (bicyclic heteroaromatic)
]

featurizer = pm.get_featurizer('protein_ligand_distance_matrix')
fragment_features = featurizer.featurize_many(fragments)

# Cluster fragments by binding patterns
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(fragment_features)

print("Fragment clustering by distance patterns:")
for fragment, cluster in zip(fragments, clusters):
    print(f"Cluster {cluster}: {fragment}")

# Identify cluster characteristics
for cluster_id in range(3):
    cluster_mask = clusters == cluster_id
    cluster_features = fragment_features[cluster_mask]
    cluster_mean = np.mean(cluster_features, axis=0)
    
    # Find dominant distance ranges
    distance_sums = []
    for bin_idx in range(5):
        bin_start = bin_idx * 40
        bin_end = (bin_idx + 1) * 40
        distance_sums.append(np.sum(cluster_mean[bin_start:bin_end]))
    
    dominant_range = np.argmax(distance_sums)
    print(f"Cluster {cluster_id} dominant range: {dominant_range*2}-{(dominant_range+1)*2}Å")
```

## References

- [Protein-Ligand Distance Analysis](https://doi.org/10.1002/jcc.23905)
- [Structure-Based Virtual Screening](https://doi.org/10.1016/j.drudis.2008.12.005)
- [Distance-Based Descriptors Review](https://doi.org/10.1007/s10822-013-9644-8)

```{toctree}
:maxdepth: 1
:hidden:
```