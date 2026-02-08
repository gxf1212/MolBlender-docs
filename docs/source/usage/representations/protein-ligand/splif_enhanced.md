# Enhanced SPLIF

Generate Structural Protein-Ligand Interaction Fingerprints (SPLIF) with advanced pharmacophore classification and distance-binned interaction patterns for protein-ligand complex analysis.

## Introduction

Enhanced SPLIF captures protein-ligand interactions through sophisticated pharmacophore typing and multi-distance analysis. Unlike traditional SPLIF, this implementation uses modern atom classification schemes and provides detailed distance-resolved interaction profiles suitable for structure-based drug design.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 💊 **Pharmacophore-Based**
Seven interaction types: donors, acceptors, aromatic, hydrophobic, charged
:::

:::{grid-item-card} 📏 **Distance-Resolved**
Eight distance bins from 0-8Å capturing different interaction strengths
:::

:::{grid-item-card} 🎯 **Interaction Counting**
Count-based encoding preserving interaction frequency information
:::

:::{grid-item-card} 📊 **Fixed Dimensions**
2048-dimensional vector (8 bins × 64 pharmacophore pairs × 4)
:::

::::

## Quick Start

```python
import molblender as mbl
import numpy as np

# Basic pharmacophore interaction analysis
featurizer = mbl.get_featurizer('splif_enhanced')
features = featurizer.featurize('CC(=O)Nc1ccc(O)cc1')  # acetaminophen
print(f"Shape: {features.shape}")  # (2048,)
print(f"Total interactions: {np.sum(features)}")

# Analyze interaction patterns
strong_interactions = features[:256]  # 0-2Å interactions
hbond_interactions = features[256:512]  # 2-3Å interactions (H-bonds)
print(f"Strong contacts: {np.sum(strong_interactions)}")
print(f"H-bond potential: {np.sum(hbond_interactions)}")
```

## Pharmacophore Classification

:::{list-table} **Pharmacophore Types and Definitions**
:header-rows: 1
:widths: 20 25 55

* - Type
  - RDKit Definition
  - Chemical Examples
* - `hb_donor`
  - N.3, N.2, O.3 with H
  - Primary amines, alcohols, amides
* - `hb_acceptor` 
  - N.2, O.2, O.3, S.2
  - Carbonyls, ethers, pyridine N
* - `aromatic`
  - C.ar, N.ar in rings
  - Benzene, pyridine, indole
* - `hydrophobic`
  - C.3, C.2, C.1
  - Alkyl chains, aromatic carbons
* - `positive`
  - N.4, N.pl3+
  - Quaternary ammonium, protonated amines
* - `negative`
  - O.co2-, P.3-, S.o2-
  - Carboxylates, phosphates, sulfates
* - `polar`
  - O.2, N.2, S.2, P.3
  - General polar atoms not classified above
:::

## Distance Binning

Enhanced SPLIF analyzes interactions across eight distance ranges:

::::{tab-set}

:::{tab-item} Distance Ranges
- **0-2.0 Å**: Very close contacts, potential clashes
- **2.0-3.0 Å**: Hydrogen bonds, strong polar interactions  
- **3.0-4.0 Å**: Close van der Waals contacts
- **4.0-5.0 Å**: Medium-range interactions
- **5.0-6.0 Å**: Weak interactions, water-mediated contacts
- **6.0-7.0 Å**: Long-range electrostatic effects
- **7.0-8.0 Å**: Distant interactions, structural context
- **>8.0 Å**: Ignored (beyond typical binding interactions)
:::

:::{tab-item} Feature Organization
The 2048-dimensional vector is structured as:

```
Bin 1 (0-2Å):  [donor-donor, donor-acceptor, ..., polar-polar] (64 pairs)
Bin 2 (2-3Å):  [donor-donor, donor-acceptor, ..., polar-polar] (64 pairs)
...
Bin 8 (7-8Å): [donor-donor, donor-acceptor, ..., polar-polar] (64 pairs)
```

Total: 8 bins × 64 pair types × 4 replicates = 2048 features
:::

:::{tab-item} Interpretation
- **High values in 2-3Å**: Strong hydrogen bonding potential
- **Hydrophobic-hydrophobic**: Lipophilic interaction strength
- **Aromatic patterns**: π-π stacking and π-cation potential  
- **Charge interactions**: Electrostatic complementarity
:::

::::

## Usage Examples

### Pharmacophore Analysis

```python
# Compare molecules with different interaction profiles
molecules = {
    'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',      # H-bond donor/acceptor
    'benzene': 'c1ccccc1',                    # Aromatic only
    'ethanol': 'CCO',                         # H-bond donor/acceptor
    'octane': 'CCCCCCCC',                     # Hydrophobic only
    'arginine': 'NC(CCCNC(N)=N)C(=O)O'       # Positive charge
}

featurizer = mbl.get_featurizer('splif_enhanced')

for name, smiles in molecules.items():
    features = featurizer.featurize(smiles)
    
    # Analyze interaction types
    hbond_strength = np.sum(features[256:512])  # 2-3Å range (H-bonds)
    aromatic_score = np.sum(features[512:768])  # 3-4Å range (π-interactions)
    hydrophobic_score = np.sum(features[768:1024])  # 4-5Å range
    
    print(f"{name}:")
    print(f"  H-bond potential: {hbond_strength:.1f}")
    print(f"  Aromatic score: {aromatic_score:.1f}")  
    print(f"  Hydrophobic score: {hydrophobic_score:.1f}")
```

### Batch Processing for Virtual Screening

```python
# Process drug-like compound library
drug_library = [
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',      # caffeine
    'CC(C)(C)NCC(c1ccc(O)cc1O)O',        # salbutamol
    'CCN(CC)C(=O)c1ccc2c(c1)ncn2C',      # theophylline
    'CC(=O)Oc1ccccc1C(=O)O',             # aspirin
    'c1ccc(cc1)C(=O)O'                    # benzoic acid
]

featurizer = mbl.get_featurizer('splif_enhanced')

# Parallel processing for large libraries
features = featurizer.featurize_many(drug_library, n_jobs=4)
print(f"Processed {len(features)} compounds")

# Analyze interaction diversity
interaction_profiles = []
for i, compound_features in enumerate(features):
    profile = {
        'compound': drug_library[i],
        'total_interactions': np.sum(compound_features),
        'hbond_strength': np.sum(compound_features[256:512]),
        'hydrophobic_strength': np.sum(compound_features[768:1024])
    }
    interaction_profiles.append(profile)

# Sort by interaction strength
sorted_profiles = sorted(interaction_profiles, 
                        key=lambda x: x['total_interactions'], 
                        reverse=True)

for profile in sorted_profiles:
    print(f"{profile['compound']}: {profile['total_interactions']:.1f} interactions")
```

### Structure-Activity Relationship Analysis

```python
from molblender.data import MolecularDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression

# Load SAR dataset
dataset = MolecularDataset.from_csv(
    "kinase_inhibitors.csv",
    smiles_column="SMILES",
    target_column="IC50_nM"
)

# Generate SPLIF features
dataset.add_features("splif_enhanced", n_workers=4)
X = dataset.features["splif_enhanced"]
y = -np.log10(dataset.targets["IC50_nM"])  # Convert to pIC50

# Feature selection to identify key interactions
selector = SelectKBest(f_regression, k=100)
X_selected = selector.fit_transform(X, y)

# Train interaction-based model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_selected, y)

# Identify most important interaction patterns
feature_importance = model.feature_importances_
selected_indices = selector.get_support(indices=True)

print("Top 10 interaction patterns:")
for i in np.argsort(feature_importance)[-10:]:
    global_idx = selected_indices[i]
    distance_bin = global_idx // 256
    pair_type = global_idx % 64
    print(f"  Distance {distance_bin*1+2:.1f}Å, pair {pair_type}: {feature_importance[i]:.3f}")
```

## Integration with Binding Affinity Prediction

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load protein-ligand binding dataset  
dataset = MolecularDataset.from_csv(
    "binding_affinities.csv",
    smiles_column="SMILES", 
    target_column="delta_G"
)

# Add SPLIF features for interaction analysis
dataset.add_features("splif_enhanced", n_workers=4)

# Prepare data
X = dataset.features["splif_enhanced"]
y = dataset.targets["delta_G"]

# Cross-validation with interaction fingerprints
model = RandomForestRegressor(n_estimators=200, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"SPLIF Model Performance:")
print(f"CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Feature importance analysis
model.fit(X, y)
importances = model.feature_importances_

# Identify critical interaction distances
distance_importance = []
for dist_bin in range(8):
    bin_start = dist_bin * 256
    bin_end = (dist_bin + 1) * 256
    bin_importance = np.sum(importances[bin_start:bin_end])
    distance_importance.append(bin_importance)
    print(f"Distance {dist_bin*1+2:.1f}Å importance: {bin_importance:.3f}")

# Plot distance-resolved importance
plt.figure(figsize=(10, 6))
distances = [f"{i*1+2:.1f}" for i in range(8)]
plt.bar(distances, distance_importance)
plt.xlabel("Distance Range (Å)")
plt.ylabel("Feature Importance") 
plt.title("SPLIF Distance-Resolved Feature Importance")
plt.show()
```

## Performance Characteristics

### Computational Performance

:::{list-table} **Processing Times (CPU)**
:header-rows: 1
:widths: 30 35 35

* - Dataset Size
  - Sequential Processing
  - Parallel (4 cores)
* - 100 molecules
  - 2-5 seconds
  - 1-2 seconds
* - 1000 molecules
  - 20-50 seconds  
  - 8-15 seconds
* - 10000 molecules
  - 3-8 minutes
  - 1-3 minutes
:::

### Memory Requirements

- **Per molecule**: ~0.5-2 MB during computation
- **Feature storage**: 2048 × 4 bytes = 8 KB per molecule  
- **1000 molecules**: ~8 MB feature storage + ~200 MB processing

## Advanced Applications

### Pharmacophore-Based Clustering

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Cluster compounds by interaction patterns
featurizer = mbl.get_featurizer('splif_enhanced')
features = featurizer.featurize_many(compound_library)

# Focus on hydrogen bonding interactions (2-3Å range)
hbond_features = features[:, 256:512]

# Cluster by H-bonding patterns
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(hbond_features)

# Visualize with PCA
pca = PCA(n_components=2)
features_2d = pca.fit_transform(hbond_features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("Pharmacophore-Based Clustering (H-bond Patterns)")
plt.colorbar(scatter, label="Cluster")
plt.show()
```

## References

- [Original SPLIF Paper](https://pubs.acs.org/doi/10.1021/ci050274i)
- [Pharmacophore Modeling Review](https://doi.org/10.1016/j.drudis.2010.06.013)
- [RDKit Pharmacophore Features](https://www.rdkit.org/docs/GettingStartedInPython.html#pharmacophore-fingerprints)

```{toctree}
:maxdepth: 1
:hidden:
```