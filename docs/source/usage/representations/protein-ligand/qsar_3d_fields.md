# 3D QSAR Fields

Generate CoMFA/CoMSIA-style molecular interaction fields for 3D-QSAR modeling and structure-based drug design.

## Introduction

3D QSAR Fields computes molecular interaction fields around ligands and binding sites, capturing electrostatic, steric, and hydrophobic properties at discrete 3D grid points. This approach enables traditional CoMFA-style 3D-QSAR analysis with modern computational efficiency.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} ⚡ **Electrostatic Fields**
Coulombic potential with Gasteiger charges
:::

:::{grid-item-card} 🔴 **Steric Fields**
Lennard-Jones potentials with van der Waals radii
:::

:::{grid-item-card} 🌊 **Hydrophobic Fields**
Element-specific hydrophobicity with Gaussian decay
:::

:::{grid-item-card} 📐 **Regular Grid**
1.0Å spacing, 8Å radius, 512 grid points maximum
:::

:::{grid-item-card} 🎯 **CoMFA Compatible**
Field values suitable for PLS regression
:::

:::{grid-item-card} 📊 **Fixed Output**
1000-dimensional vector (333+333+334 fields)
:::

::::

## Quick Start

```python
import molblender as mbl
import numpy as np

# Basic 3D field calculation
featurizer = mbl.get_featurizer('qsar_3d_fields')
features = featurizer.featurize('CCO')  # ethanol
print(f"Shape: {features.shape}")  # (1000,)

# Extract field components
electrostatic = features[:333]   # Electrostatic potential
steric = features[333:666]       # Steric clashes/interactions
hydrophobic = features[666:999]  # Hydrophobic complementarity

print(f"Electrostatic range: {np.min(electrostatic):.2f} to {np.max(electrostatic):.2f}")
print(f"Steric range: {np.min(steric):.2f} to {np.max(steric):.2f}")
print(f"Hydrophobic range: {np.min(hydrophobic):.2f} to {np.max(hydrophobic):.2f}")
```

## Field Types and Calculations

::::{tab-set}

:::{tab-item} Electrostatic Field
**Method**: Gasteiger partial charges with Coulombic potential

**Formula**: $V_{elec}(r) = \sum_i \frac{q_i}{|r - r_i|}$

where:
- $q_i$ = Gasteiger partial charge on atom $i$
- $r_i$ = Position of atom $i$  
- $r$ = Grid point position

**Units**: Elementary charge units per Ångström  
**Interpretation**: 
- Positive values: Repulsion for positive charges
- Negative values: Attraction for positive charges
:::

:::{tab-item} Steric Field
**Method**: Lennard-Jones 6-12 potential with van der Waals radii

**Formula**: $V_{steric}(r) = 4\epsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]$

**Cutoffs**: Values limited to [-10, +100] to prevent singularities

**Interpretation**:
- High positive values: Steric clashes (unfavorable)
- Negative values: Favorable van der Waals interactions  
- Zero values: No steric interaction
:::

:::{tab-item} Hydrophobic Field  
**Method**: Gaussian decay with element-specific hydrophobicity

**Formula**: $V_{hydro}(r) = \sum_i h_i \times \exp\left(\frac{-d_i^2}{2\sigma^2}\right)$

where:
- $h_i$ = Element hydrophobicity (C=+1.0, N=-0.5, O=-0.8)
- $d_i$ = Distance from atom $i$ to grid point
- $\sigma$ = 2.0 Å (Gaussian width)

**Interpretation**:
- Positive values: Favor hydrophobic interactions
- Negative values: Favor polar interactions
:::

::::

## Parameters

:::{list-table} **3D QSAR Fields Parameters**
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Default
  - Description
* - `grid_spacing`
  - 1.0
  - Grid point spacing in Ångströms
* - `field_radius`
  - 8.0
  - Maximum distance from molecule center (Å)
* - `max_grid_points`
  - 512
  - Maximum number of grid points (performance limit)
:::

```{note}
Grid size is automatically limited to 8×8×8 = 512 points for computational efficiency. For larger molecules, the grid spacing is automatically adjusted.
```

## Usage Examples

### Single Molecule Field Analysis

```python
# Analyze different molecule types
molecules = {
    'polar': 'CCO',                    # ethanol
    'aromatic': 'c1ccccc1',            # benzene  
    'charged': 'CC(=O)O',              # acetic acid
    'complex': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # caffeine
}

featurizer = mbl.get_featurizer('qsar_3d_fields')

for name, smiles in molecules.items():
    features = featurizer.featurize(smiles)
    
    # Extract field statistics
    electrostatic = features[:333]
    steric = features[333:666] 
    hydrophobic = features[666:999]
    
    print(f"{name.title()} molecule:")
    print(f"  Electrostatic variation: {np.std(electrostatic):.3f}")
    print(f"  Steric complexity: {np.std(steric):.3f}")
    print(f"  Hydrophobic character: {np.mean(hydrophobic):.3f}")
```

### Custom Grid Parameters

```python
# High-resolution analysis for small molecules
high_res_featurizer = mbl.get_featurizer(
    'qsar_3d_fields',
    grid_spacing=0.5,      # Finer grid
    field_radius=6.0       # Smaller radius
)

# Low-resolution for large molecules  
fast_featurizer = mbl.get_featurizer(
    'qsar_3d_fields',
    grid_spacing=1.5,      # Coarser grid
    field_radius=10.0      # Larger radius
)

# Compare resolutions
molecule = 'CC(C)(C)NCC(c1ccc(O)cc1O)O'  # salbutamol

high_res_features = high_res_featurizer.featurize(molecule)
fast_features = fast_featurizer.featurize(molecule)

print(f"High resolution: {high_res_features.shape}")
print(f"Fast calculation: {fast_features.shape}")
```

### 3D-QSAR Model Building

```python
from molblender.data import MolecularDataset
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load QSAR dataset
dataset = MolecularDataset.from_csv(
    "steroid_activity.csv",
    smiles_column="SMILES",
    target_column="log_activity" 
)

# Generate 3D field descriptors
dataset.add_features("qsar_3d_fields", n_workers=4)

# Prepare data for 3D-QSAR
X = dataset.features["qsar_3d_fields"]
y = dataset.targets["log_activity"]

# Standardize features (important for field values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PLS regression (traditional 3D-QSAR approach)
pls_model = PLSRegression(n_components=5)
cv_scores = cross_val_score(pls_model, X_scaled, y, cv=5, scoring='r2')

print(f"3D-QSAR PLS Model:")
print(f"CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Train final model
pls_model.fit(X_scaled, y)

# Component analysis
print(f"\nPLS Component Weights:")
for i in range(5):
    weights = pls_model.x_weights_[:, i]
    
    # Field contributions  
    elec_contrib = np.sum(np.abs(weights[:333]))
    steric_contrib = np.sum(np.abs(weights[333:666]))
    hydro_contrib = np.sum(np.abs(weights[666:999]))
    
    total = elec_contrib + steric_contrib + hydro_contrib
    print(f"Component {i+1}:")
    print(f"  Electrostatic: {elec_contrib/total:.1%}")
    print(f"  Steric: {steric_contrib/total:.1%}")
    print(f"  Hydrophobic: {hydro_contrib/total:.1%}")
```

### Field-Based Virtual Screening

```python
from sklearn.metrics.pairwise import cosine_similarity

# Reference compound with known activity
reference_compound = 'CC(=O)Oc1ccccc1C(=O)O'  # aspirin

# Virtual compound library
virtual_library = [
    'CC(=O)Oc1ccccc1',                    # phenyl acetate (similar)
    'CC(=O)Nc1ccc(O)cc1',                 # acetaminophen (different mechanism)  
    'c1ccc(cc1)C(=O)O',                   # benzoic acid (simpler)
    'CC(C)(C)OC(=O)Nc1ccc(O)cc1',         # protected acetaminophen
    'CCN(CC)C(=O)c1ccc2c(c1)ncn2C'        # theophylline (unrelated)
]

featurizer = mbl.get_featurizer('qsar_3d_fields')

# Calculate reference field
ref_fields = featurizer.featurize(reference_compound).reshape(1, -1)

# Screen virtual library
library_fields = featurizer.featurize_many(virtual_library)

# Field-based similarity
similarities = cosine_similarity(ref_fields, library_fields)[0]

# Rank by 3D field similarity
results = list(zip(virtual_library, similarities))
results.sort(key=lambda x: x[1], reverse=True)

print("Field-based virtual screening results:")
for i, (smiles, similarity) in enumerate(results):
    print(f"{i+1}. Similarity: {similarity:.3f} - {smiles}")
```

## Advanced Applications

### Field Component Analysis

```python
import matplotlib.pyplot as plt

# Analyze field contributions across compound series
alcohol_series = ['CO', 'CCO', 'CCCO', 'CCCCO', 'CCCCCO']
featurizer = mbl.get_featurizer('qsar_3d_fields')

field_analysis = []
for alcohol in alcohol_series:
    features = featurizer.featurize(alcohol)
    
    analysis = {
        'compound': alcohol,
        'carbon_count': alcohol.count('C'),
        'electrostatic_var': np.var(features[:333]),
        'steric_mean': np.mean(features[333:666]),
        'hydrophobic_mean': np.mean(features[666:999])
    }
    field_analysis.append(analysis)

# Plot trends
carbon_counts = [a['carbon_count'] for a in field_analysis]
electrostatic_vars = [a['electrostatic_var'] for a in field_analysis]
hydrophobic_means = [a['hydrophobic_mean'] for a in field_analysis]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(carbon_counts, electrostatic_vars, 'bo-')
plt.xlabel('Carbon Count')
plt.ylabel('Electrostatic Field Variance')
plt.title('Electrostatic Complexity vs Chain Length')

plt.subplot(1, 2, 2) 
plt.plot(carbon_counts, hydrophobic_means, 'ro-')
plt.xlabel('Carbon Count')
plt.ylabel('Mean Hydrophobic Field')
plt.title('Hydrophobicity vs Chain Length')

plt.tight_layout()
plt.show()
```

### CoMFA/CoMSIA Compatibility

```python
# Extract field grids for visualization (conceptual)
def extract_field_grids(features, grid_size=8):
    """Reshape 1D features back to 3D grids."""
    # This is conceptual - actual implementation would need grid metadata
    electrostatic = features[:333]
    steric = features[333:666]
    hydrophobic = features[666:999]
    
    # Pad to cube if needed
    n_points = grid_size ** 3
    if len(electrostatic) < n_points:
        electrostatic = np.pad(electrostatic, (0, n_points - len(electrostatic)))
    
    # Reshape to 3D grid
    elec_grid = electrostatic[:n_points].reshape(grid_size, grid_size, grid_size)
    
    return {
        'electrostatic': elec_grid,
        'steric': steric[:n_points].reshape(grid_size, grid_size, grid_size),
        'hydrophobic': hydrophobic[:n_points].reshape(grid_size, grid_size, grid_size)
    }

# Example usage
features = featurizer.featurize('CCO')
field_grids = extract_field_grids(features)

print(f"Grid shapes: {field_grids['electrostatic'].shape}")
print(f"Electrostatic range: {np.min(field_grids['electrostatic'])} to {np.max(field_grids['electrostatic'])}")
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
  - 10-20 seconds
  - 4-8 seconds
* - 1000 molecules
  - 100-200 seconds
  - 30-60 seconds
* - 10000 molecules
  - 15-30 minutes
  - 5-10 minutes
:::

### Memory Requirements

- **Per molecule**: ~2-10 MB during computation
- **Feature storage**: 1000 × 4 bytes = 4 KB per molecule
- **Grid calculation**: Temporary 3D arrays ~8³ × 3 fields × 8 bytes

## Applications in Drug Discovery

### Lead Optimization

```python
# Analyze SAR for lead optimization
lead_series = [
    'c1ccc2c(c1)[nH]c3ccccc23',         # carbazole (lead)
    'c1ccc2c(c1)[nH]c3cc(Cl)ccc23',     # 3-chloro derivative
    'c1ccc2c(c1)[nH]c3cc(F)ccc23',      # 3-fluoro derivative  
    'c1ccc2c(c1)[nH]c3cc(O)ccc23',      # 3-hydroxy derivative
    'c1ccc2c(c1)[nH]c3cc(N)ccc23'       # 3-amino derivative
]

featurizer = mbl.get_featurizer('qsar_3d_fields')
lead_features = featurizer.featurize_many(lead_series)

# Compare field patterns
lead_ref = lead_features[0]  # Original lead
for i, derivative in enumerate(lead_series[1:], 1):
    similarity = cosine_similarity([lead_ref], [lead_features[i]])[0][0]
    
    # Field differences
    field_diff = lead_features[i] - lead_ref
    elec_change = np.mean(field_diff[:333])
    steric_change = np.mean(field_diff[333:666])
    hydro_change = np.mean(field_diff[666:999])
    
    print(f"Derivative {i}: {derivative}")
    print(f"  3D similarity: {similarity:.3f}")
    print(f"  Electrostatic change: {elec_change:+.3f}")
    print(f"  Steric change: {steric_change:+.3f}")
    print(f"  Hydrophobic change: {hydro_change:+.3f}")
```

## References

- [CoMFA Original Paper](https://pubs.acs.org/doi/10.1021/jm00112a004)
- [CoMSIA Methodology](https://pubs.acs.org/doi/10.1021/jm940835a)
- [3D-QSAR Review](https://doi.org/10.1016/j.drudis.2010.11.005)
- [Molecular Interaction Fields](https://doi.org/10.1002/jcc.540150703)

```{toctree}
:maxdepth: 1
:hidden:
```