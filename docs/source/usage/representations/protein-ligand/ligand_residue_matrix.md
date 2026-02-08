# Atomic Distance Matrix

Generate high-resolution atomic distance matrices between ligand atoms and protein binding site residues with automatic cavity detection and fixed-length output for machine learning.

## Introduction

The Ligand Residue Distance Matrix creates detailed atomic distance matrices preserving full spatial information without pharmacophore abstractions. Unlike PLIF/SPLIF approaches, this method provides raw distance data with automatic binding site standardization and ML-ready fixed dimensions.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🎯 **Auto-Cavity Detection**
Automatic binding site identification within 3.0-4.5Å
:::

:::{grid-item-card} ⚛️ **Full Atomic Resolution**
Raw distances, no pharmacophore binning
:::

:::{grid-item-card} 📐 **Matrix Format**
30×200 distance matrix (6000 features total)
:::

:::{grid-item-card} 🤖 **ML-Ready**
Fixed dimensions across all molecules
:::

:::{grid-item-card} 📊 **Dataset Consistent**
Standardized cavity from most frequent residues
:::

:::{grid-item-card} 🔬 **Spatial Sampling**
Farthest-point sampling for diverse atom selection
:::

::::

## Quick Start

```python
import molblender as mbl
import numpy as np

# Basic atomic distance matrix
featurizer = mbl.get_featurizer('ligand_residue_distance_matrix')
features = featurizer.featurize('c1ccc2[nH]c3ccccc3c2c1')  # indole
print(f"Shape: {features.shape}")  # (6000,)

# Reshape to matrix format for analysis
distance_matrix = features.reshape(30, 200)  # 30 ligand atoms × 200 cavity positions
print(f"Matrix shape: {distance_matrix.shape}")
print(f"Non-zero interactions: {np.count_nonzero(distance_matrix)}")

# Analyze distance distribution
close_contacts = np.sum(distance_matrix < 3.0)  # Contacts < 3Å
interactions = np.sum(distance_matrix < 6.0)    # Interactions < 6Å
print(f"Close contacts (<3Å): {close_contacts}")
print(f"Total interactions (<6Å): {interactions}")
```

## Matrix Architecture

::::{tab-set}

:::{tab-item} Matrix Dimensions
The 6000-dimensional output represents a **30×200 distance matrix**:

- **30 rows**: Maximum ligand atoms (zero-padded if fewer)
- **200 columns**: Standardized cavity positions  
- **Values**: Raw distances in Ångströms (0.0 for padding)

```python
# Matrix interpretation
features = featurizer.featurize('CCO')  # ethanol (3 atoms)
matrix = features.reshape(30, 200)

# First 3 rows contain real atom distances
real_atoms = matrix[:3, :]  # 3 real atoms
padding = matrix[3:, :]     # 27 rows of padding (should be zeros)

print(f"Real atom distances: {np.count_nonzero(real_atoms)}")
print(f"Padding zeros: {np.count_nonzero(padding) == 0}")
```
:::

:::{tab-item} Cavity Structure  
**200 cavity positions** = **50 residues × 4 atoms/residue**

Automatic cavity detection process:
1. Find all protein atoms within `cavity_cutoff` of ligand
2. Group atoms by residue (chain, name, number)
3. Rank residues by importance (number of cavity atoms)
4. Select diverse atoms per residue using farthest-point sampling
5. Create standardized 200-position cavity template

```python
# Cavity analysis (conceptual)
def analyze_cavity_positions(features):
    matrix = features.reshape(30, 200)
    
    # Analyze per-residue contributions (4 atoms per residue)
    residue_distances = []
    for res_idx in range(50):
        res_start = res_idx * 4
        res_end = res_start + 4
        res_distances = matrix[:, res_start:res_end]
        avg_distance = np.mean(res_distances[res_distances > 0])
        residue_distances.append(avg_distance)
    
    return residue_distances
```
:::

:::{tab-item} Distance Values
**Distance interpretation**:
- **0.0**: No interaction or padded position
- **1.0-3.0Å**: Very close contacts, H-bonds, potential clashes
- **3.0-6.0Å**: van der Waals interactions, hydrophobic contacts
- **6.0-10.0Å**: Weak interactions, water-mediated contacts  
- **>10.0Å**: Distant relationships, structural context
:::

::::

## Parameters

:::{list-table} **Atomic Distance Matrix Parameters**
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Default
  - Description
* - `cavity_cutoff`
  - 4.0
  - Distance threshold for cavity detection (Å)
* - `max_cavity_residues`
  - 50
  - Maximum standardized cavity residues
* - `atoms_per_residue`
  - 4
  - Average atoms selected per residue
* - `max_ligand_atoms`
  - 30
  - Maximum expected ligand atoms
:::

## Usage Examples

### High-Resolution Binding Analysis

```python
# Analyze binding interactions at atomic resolution
drug_compounds = [
    'CC(=O)Oc1ccccc1C(=O)O',        # aspirin (13 atoms)
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # caffeine (14 atoms)  
    'CCN(CC)C(=O)c1ccc2c(c1)ncn2C',  # theophylline (15 atoms)
    'c1ccc(cc1)C(=O)O',             # benzoic acid (8 atoms)
    'CC(C)(C)NCC(c1ccc(O)cc1O)O'    # salbutamol (13 atoms)
]

featurizer = mbl.get_featurizer('ligand_residue_distance_matrix')

for compound in drug_compounds:
    features = featurizer.featurize(compound)
    matrix = features.reshape(30, 200)
    
    # Count real atoms (non-zero rows)
    real_atoms = np.count_nonzero(np.sum(matrix, axis=1))
    
    # Analyze contact patterns
    very_close = np.sum(matrix < 2.0)   # Potential clashes
    hbond_range = np.sum((matrix >= 2.0) & (matrix < 3.5))  # H-bond distance
    vdw_range = np.sum((matrix >= 3.5) & (matrix < 6.0))    # vdW interactions
    
    print(f"{compound}:")
    print(f"  Real atoms: {real_atoms}")
    print(f"  Very close contacts: {very_close}")
    print(f"  H-bond distance: {hbond_range}")
    print(f"  vdW interactions: {vdw_range}")
```

### Custom Cavity Parameters

```python
# Fine-grained cavity detection
detailed_featurizer = mbl.get_featurizer(
    'ligand_residue_distance_matrix',
    cavity_cutoff=3.5,              # Tighter cavity definition
    max_cavity_residues=30,         # Fewer residues
    atoms_per_residue=3,            # Fewer atoms per residue
    max_ligand_atoms=20             # Smaller ligand assumption
)

# Broader cavity analysis  
broad_featurizer = mbl.get_featurizer(
    'ligand_residue_distance_matrix',
    cavity_cutoff=5.0,              # Broader cavity
    max_cavity_residues=60,         # More residues
    atoms_per_residue=5,            # More atoms per residue  
    max_ligand_atoms=40             # Larger ligand support
)

# Compare cavity definitions
test_molecule = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # caffeine

detailed_features = detailed_featurizer.featurize(test_molecule)
broad_features = broad_featurizer.featurize(test_molecule)

print(f"Detailed cavity: {detailed_features.shape}")
print(f"Broad cavity: {broad_features.shape}")

# Compare interaction counts
detailed_matrix = detailed_features.reshape(-1, detailed_featurizer.atoms_per_residue * detailed_featurizer.max_cavity_residues)
broad_matrix = broad_features.reshape(-1, broad_featurizer.atoms_per_residue * broad_featurizer.max_cavity_residues)

print(f"Detailed interactions: {np.count_nonzero(detailed_matrix)}")
print(f"Broad interactions: {np.count_nonzero(broad_matrix)}")
```

### Structure-Activity Relationships

```python
from molblender.data import MolecularDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# Load binding affinity dataset
dataset = MolecularDataset.from_csv(
    "kinase_binding.csv",
    smiles_column="SMILES", 
    target_column="pIC50"
)

# Generate atomic distance matrices
dataset.add_features("ligand_residue_distance_matrix", n_workers=4)
X = dataset.features["ligand_residue_distance_matrix"]
y = dataset.targets["pIC50"]

print(f"Dataset: {X.shape[0]} compounds, {X.shape[1]} distance features")

# Dimensionality reduction for analysis
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

print(f"PCA explained variance: {pca.explained_variance_ratio_[:5].sum():.1%} (first 5 components)")

# Cross-validation with full distance matrix
rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores_full = cross_val_score(rf_full, X, y, cv=5, scoring='r2')

# Cross-validation with PCA features
rf_pca = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores_pca = cross_val_score(rf_pca, X_reduced, y, cv=5, scoring='r2')

print(f"Full distance matrix R²: {cv_scores_full.mean():.3f} ± {cv_scores_full.std():.3f}")
print(f"PCA-reduced R²: {cv_scores_pca.mean():.3f} ± {cv_scores_pca.std():.3f}")

# Feature importance analysis
rf_full.fit(X, y)
importances = rf_full.feature_importances_

# Reshape to analyze spatial patterns
importance_matrix = importances.reshape(30, 200)

# Find most important ligand atom positions
ligand_importance = np.sum(importance_matrix, axis=1)
important_atoms = np.argsort(ligand_importance)[-5:]  # Top 5 ligand positions

# Find most important cavity positions  
cavity_importance = np.sum(importance_matrix, axis=0)
important_positions = np.argsort(cavity_importance)[-5:]  # Top 5 cavity positions

print(f"Most important ligand positions: {important_atoms}")
print(f"Most important cavity positions: {important_positions}")
```

### Binding Mode Analysis

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Analyze different binding modes
similar_compounds = [
    'c1ccc2c(c1)nc3ccccc3c2N',      # phenazin-1-amine
    'c1ccc2c(c1)nc3ccccc3c2O',      # phenazin-1-ol
    'c1ccc2c(c1)nc3ccccc3c2=O',     # phenazin-1-one
    'c1ccc2c(c1)[nH]c3ccccc23',     # carbazole
    'c1ccc2c(c1)sc3ccccc23'         # dibenzothiophene
]

featurizer = mbl.get_featurizer('ligand_residue_distance_matrix')
binding_features = featurizer.featurize_many(similar_compounds)

# Cluster by binding patterns
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(binding_features)

print("Binding mode clusters:")
for compound, cluster in zip(similar_compounds, clusters):
    print(f"Cluster {cluster}: {compound}")

# Analyze cluster differences
for cluster_id in range(2):
    cluster_mask = clusters == cluster_id
    cluster_features = binding_features[cluster_mask]
    
    # Average distance matrix for this cluster
    cluster_mean = np.mean(cluster_features, axis=0)
    cluster_matrix = cluster_mean.reshape(30, 200)
    
    # Find binding hotspots
    contact_density = np.sum(cluster_matrix < 4.0, axis=0)  # Contacts per cavity position
    hotspots = np.argsort(contact_density)[-10:]  # Top 10 hotspot positions
    
    print(f"Cluster {cluster_id} hotspots: positions {hotspots}")
    print(f"Average contact density: {np.mean(contact_density):.1f}")
```

### Advanced Spatial Analysis

```python
# Spatial correlation analysis
def analyze_spatial_correlations(features, molecule_name):
    """Analyze correlations between ligand atoms and cavity positions."""
    matrix = features.reshape(30, 200)
    
    # Remove padding (zero rows/columns)
    non_zero_rows = np.any(matrix > 0, axis=1)
    non_zero_cols = np.any(matrix > 0, axis=0)
    
    active_matrix = matrix[non_zero_rows][:, non_zero_cols]
    
    print(f"{molecule_name}:")
    print(f"  Active matrix: {active_matrix.shape}")
    
    # Distance statistics
    distances = active_matrix[active_matrix > 0]
    print(f"  Distance range: {distances.min():.2f} - {distances.max():.2f} Å")
    print(f"  Mean distance: {distances.mean():.2f} Å")
    print(f"  Contact ratio: {np.sum(distances < 4.0) / len(distances):.1%}")
    
    return active_matrix

# Analyze spatial patterns for different scaffolds
scaffolds = {
    'benzene': 'c1ccccc1',
    'naphthalene': 'c1ccc2ccccc2c1',
    'anthracene': 'c1ccc2cc3ccccc3cc2c1',
    'phenanthrene': 'c1ccc2c(c1)ccc3ccccc23'
}

featurizer = mbl.get_featurizer('ligand_residue_distance_matrix')

spatial_data = {}
for name, smiles in scaffolds.items():
    features = featurizer.featurize(smiles)
    active_matrix = analyze_spatial_correlations(features, name)
    spatial_data[name] = active_matrix
```

## Performance Characteristics

### Computational Performance

:::{list-table} **Processing Times (CPU)**
:header-rows: 1
:widths: 30 35 35

* - Dataset Size  
  - Sequential
  - Parallel (4 cores)
* - 100 molecules
  - 8-15 seconds
  - 3-6 seconds
* - 1000 molecules
  - 80-150 seconds
  - 25-45 seconds
* - 10000 molecules
  - 12-25 minutes
  - 4-8 minutes
:::

### Memory Requirements

- **Per molecule**: ~25 KB (6000 × 4 bytes)
- **1000 molecules**: ~25 MB feature storage
- **Processing memory**: ~1-10 MB per molecule during computation
- **Cavity cache**: ~1-10 MB per protein structure

## Applications in Drug Discovery

### Fragment Growing and Linking

```python
# Analyze fragment extension opportunities
core_fragment = 'c1ccccc1'  # benzene core
extended_fragments = [
    'c1ccccc1C',              # methyl extension
    'c1ccccc1O',              # hydroxyl extension
    'c1ccccc1N',              # amino extension
    'c1ccccc1C(=O)O',         # carboxyl extension
    'c1ccccc1c2ccccc2',       # biphenyl extension
]

featurizer = mbl.get_featurizer('ligand_residue_distance_matrix')

core_features = featurizer.featurize(core_fragment)
core_matrix = core_features.reshape(30, 200)

print("Fragment extension analysis:")
for fragment in extended_fragments:
    ext_features = featurizer.featurize(fragment)
    ext_matrix = ext_features.reshape(30, 200)
    
    # Compare binding patterns
    similarity = cosine_similarity([core_features], [ext_features])[0][0]
    
    # New contacts introduced
    core_contacts = np.sum(core_matrix < 4.0)
    ext_contacts = np.sum(ext_matrix < 4.0) 
    new_contacts = ext_contacts - core_contacts
    
    print(f"{fragment}:")
    print(f"  Similarity to core: {similarity:.3f}")
    print(f"  New contacts: {new_contacts}")
```

### Structure-Based Lead Optimization

```python
# Lead optimization series analysis
lead_series = [
    'c1ccc2c(c1)[nH]c3ccccc23',       # carbazole (lead)
    'c1ccc2c(c1)[nH]c3cc(F)ccc23',    # 3-fluoro
    'c1ccc2c(c1)[nH]c3cc(Cl)ccc23',   # 3-chloro  
    'c1ccc2c(c1)[nH]c3cc(Br)ccc23',   # 3-bromo
    'c1ccc2c(c1)[nH]c3cc(I)ccc23',    # 3-iodo
    'c1ccc2c(c1)[nH]c3cc(O)ccc23',    # 3-hydroxy
    'c1ccc2c(c1)[nH]c3cc(N)ccc23'     # 3-amino
]

featurizer = mbl.get_featurizer('ligand_residue_distance_matrix')
series_features = featurizer.featurize_many(lead_series)

# Lead compound reference
lead_features = series_features[0]
lead_matrix = lead_features.reshape(30, 200)

print("Lead optimization analysis:")
for i, compound in enumerate(lead_series[1:], 1):
    derivative_features = series_features[i]
    derivative_matrix = derivative_features.reshape(30, 200)
    
    # Calculate binding pattern changes
    difference_matrix = derivative_matrix - lead_matrix
    
    # Identify regions of change
    significant_changes = np.abs(difference_matrix) > 1.0  # Changes > 1Å
    change_regions = np.sum(significant_changes)
    
    # Overall similarity
    similarity = cosine_similarity([lead_features], [derivative_features])[0][0]
    
    print(f"{compound}:")
    print(f"  3D similarity: {similarity:.3f}")
    print(f"  Significant changes: {change_regions} positions")
```

## References

- [Atomic Distance Matrices in QSAR](https://doi.org/10.1021/ci00062a008)
- [Protein-Ligand Interaction Profiling](https://doi.org/10.1002/jcc.23905)
- [Machine Learning with Distance Matrices](https://doi.org/10.1007/s10822-016-9969-8)
- [Binding Site Comparison Methods](https://doi.org/10.1002/prot.25238)

```{toctree}
:maxdepth: 1
:hidden:
```