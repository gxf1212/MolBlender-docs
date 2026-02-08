# Molecular Descriptors

Generate interpretable molecular descriptors that capture physicochemical properties and structural features for machine learning and cheminformatics applications.

## Introduction

Molecular descriptors are numerical representations that encode specific physicochemical properties of molecules. Unlike fingerprints, descriptors have clear chemical interpretations and are essential for:

- QSAR/QSPR modeling with interpretable features
- Drug discovery (Lipinski's rule, ADMET properties)
- Chemical space analysis and similarity searching
- Feature selection and mechanistic understanding

MolBlender provides access to comprehensive descriptor collections from RDKit and Mordred, with intelligent handling of missing values and 3D conformer requirements.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🧮 **RDKit Descriptors**
200+ physicochemical and topological descriptors
:::

:::{grid-item-card} 📊 **Mordred Descriptors**
1800+ comprehensive molecular descriptors
:::

:::{grid-item-card} 🎯 **Descriptor Sets**
Curated collections for specific applications
:::

:::{grid-item-card} 🔧 **Missing Values**
Robust handling of calculation failures
:::
::::

## Quick Start

```python
import molblender as mbl

# Get molecular descriptor featurizers
rdkit_desc = mbl.get_featurizer("rdkit_all_descriptors")
mordred_2d = mbl.get_featurizer("mordred_descriptors_2d")
mordred_all = mbl.get_featurizer("mordred_descriptors_all")

# Single molecule
smiles = "CCO"
rdkit_features = rdkit_desc.featurize(smiles)
mordred_features = mordred_2d.featurize(smiles)

print(f"RDKit descriptors: {rdkit_features.shape}")    # (~200,)
print(f"Mordred 2D: {mordred_features.shape}")        # (1613,)
print(f"Mordred all: {mordred_all.featurize(smiles).shape}")  # (1826,)

# Batch processing
molecules = ["CCO", "CCN", "CCC"]
batch_descriptors = rdkit_desc.featurize(molecules, n_workers=4)
print(f"Batch shape: {len(batch_descriptors)} molecules")
```

## RDKit Descriptors

### Available Descriptor Sets

| Featurizer Name | Count | Description |
|---|---|---|
| `rdkit_all_descriptors` | ~200 | All RDKit 2D/3D descriptors |
| `rdkit_essential_descriptors` | 20 | Core physicochemical properties |
| `rdkit_lipinski_descriptors` | 8 | Lipinski rule-of-five descriptors |
| `rdkit_admet_descriptors` | 15 | ADMET-relevant properties |

### Essential Descriptors

Start with the most important physicochemical properties:

```python
import molblender as mbl
import numpy as np

# Get essential descriptors (most interpretable)
essential = mbl.get_featurizer("rdkit_essential_descriptors")

# Calculate for a drug-like molecule
drug_molecule = "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2"  # Acetaminophen
descriptors = essential.featurize(drug_molecule)

# Interpret key descriptors (approximate indices)
print("Essential Descriptor Analysis:")
print(f"Molecular Weight: {descriptors[0]:.1f} g/mol")
print(f"LogP (lipophilicity): {descriptors[1]:.2f}")
print(f"TPSA (polar surface area): {descriptors[2]:.1f} Ų")
print(f"H-bond donors: {int(descriptors[3])}")
print(f"H-bond acceptors: {int(descriptors[4])}")
print(f"Rotatable bonds: {int(descriptors[5])}")
print(f"Aromatic rings: {int(descriptors[6])}")

# Get descriptor names for interpretation
descriptor_names = essential.get_feature_names()
for i, (name, value) in enumerate(zip(descriptor_names[:7], descriptors[:7])):
    print(f"{name}: {value:.3f}")
```

### Lipinski's Rule of Five

Evaluate drug-likeness using Lipinski descriptors:

```python
lipinski = mbl.get_featurizer("rdkit_lipinski_descriptors")

# Test different molecules
molecules = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 
    "Large peptide": "CC(C)C[C@H](NC(=O)[C@H](CC(C)C)NC(=O)C)C(=O)N[C@@H](CC(=O)O)C(=O)O"
}

print("Lipinski Rule of Five Analysis:")
for name, smiles in molecules.items():
    desc = lipinski.featurize(smiles)
    
    # Lipinski criteria (approximate indices)
    mw = desc[0]      # Molecular weight < 500
    logp = desc[1]    # LogP < 5
    hbd = desc[2]     # H-bond donors < 5
    hba = desc[3]     # H-bond acceptors < 10
    
    violations = 0
    violations += 1 if mw > 500 else 0
    violations += 1 if logp > 5 else 0
    violations += 1 if hbd > 5 else 0
    violations += 1 if hba > 10 else 0
    
    print(f"{name:12}: MW={mw:.1f}, LogP={logp:.2f}, "
          f"HBD={int(hbd)}, HBA={int(hba)}, violations={violations}")
```

### ADMET Descriptors

Properties relevant for absorption, distribution, metabolism, elimination, and toxicity:

```python
admet = mbl.get_featurizer("rdkit_admet_descriptors")

# Analyze ADMET properties
compound = "CN1CCN(CC1)C2=CC=C(C=C2)C(=O)N"  # Drug-like compound
admet_props = admet.featurize(compound)

print("ADMET Property Analysis:")
# These indices are approximate - check actual feature names
print(f"TPSA (absorption): {admet_props[0]:.1f} Ų")      # < 140 good
print(f"LogP (distribution): {admet_props[1]:.2f}")      # 1-3 ideal
print(f"Fraction Csp3: {admet_props[2]:.3f}")           # > 0.25 good
print(f"Aromatic atoms: {int(admet_props[3])}")         # Fewer better
print(f"Molecular complexity: {admet_props[4]:.1f}")    # BertzCT

# Get all feature names for proper interpretation
feature_names = admet.get_feature_names()
for i, (name, value) in enumerate(zip(feature_names, admet_props)):
    print(f"{name}: {value:.3f}")
```

### 3D Descriptors

RDKit automatically includes 3D descriptors when molecules have conformers:

```python
# All descriptors include 3D if available
all_desc = mbl.get_featurizer("rdkit_all_descriptors")

# Molecule with flexible geometry
flexible_mol = "CCCCCC(C)C(=O)O"  # Branched carboxylic acid
descriptors = all_desc.featurize(flexible_mol)

print(f"Total descriptors (2D + 3D): {len(descriptors)}")

# 3D descriptors are automatically included if conformers exist
# Examples: PMI1, PMI2, PMI3, RadiusOfGyration, InertialShapeFactor
print("3D shape descriptors included automatically")
```

## Mordred Descriptors

Mordred provides the most comprehensive descriptor collection available:

### 2D vs All Descriptors

```python
# 2D descriptors only (no 3D conformer needed)
mordred_2d = mbl.get_featurizer("mordred_descriptors_2d")
desc_2d = mordred_2d.featurize("CCO")

# All descriptors (includes 3D, conformer generated automatically)  
mordred_all = mbl.get_featurizer("mordred_descriptors_all")
desc_all = mordred_all.featurize("CCO")

print(f"Mordred 2D descriptors: {desc_2d.shape[0]}")   # 1613
print(f"Mordred all descriptors: {desc_all.shape[0]}") # 1826
print(f"3D-specific descriptors: {desc_all.shape[0] - desc_2d.shape[0]}")

# Check for missing values
nan_count_2d = np.isnan(desc_2d).sum()
nan_count_all = np.isnan(desc_all).sum()

print(f"Missing values in 2D: {nan_count_2d}")
print(f"Missing values in all: {nan_count_all}")
```

### Descriptor Categories

Mordred organizes descriptors into chemical categories:

```python
# Get detailed descriptor information
mordred = mbl.get_featurizer("mordred_descriptors_all", 
                           fill_value=0,      # Replace NaN with 0
                           ignore_3D=False)   # Include 3D descriptors

# Analyze complex molecule
complex_mol = "CC1=C2C=C(C=CC2=NN1C3=CC=CC=C3C)C(=O)NC4=CC=C(C=C4)F"
descriptors = mordred.featurize(complex_mol)

print(f"Mordred descriptor categories:")
print(f"Total descriptors: {len(descriptors)}")
print(f"Non-zero descriptors: {np.count_nonzero(descriptors)}")
print(f"Descriptor range: {descriptors.min():.3f} to {descriptors.max():.3f}")

# Sample some key descriptor types
print(f"\nSample descriptors:")
print(f"First 10 values: {descriptors[:10]}")
print(f"Indices 100-110: {descriptors[100:110]}")
```

### Performance Comparison

Compare descriptor calculation performance:

```python
import time

molecules = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(C)C(=O)O"] * 10  # 50 molecules

# Time RDKit descriptors
rdkit_desc = mbl.get_featurizer("rdkit_all_descriptors")
start = time.time()
rdkit_results = rdkit_desc.featurize(molecules, n_workers=4)
rdkit_time = time.time() - start

# Time Mordred 2D descriptors
mordred_2d = mbl.get_featurizer("mordred_descriptors_2d")
start = time.time()
mordred_2d_results = mordred_2d.featurize(molecules, n_workers=4)
mordred_2d_time = time.time() - start

# Time Mordred all descriptors  
mordred_all = mbl.get_featurizer("mordred_descriptors_all")
start = time.time()
mordred_all_results = mordred_all.featurize(molecules, n_workers=4)
mordred_all_time = time.time() - start

print("Performance Comparison (50 molecules):")
print(f"RDKit (~200 desc): {rdkit_time:.2f}s, {len(rdkit_results[0])} features")
print(f"Mordred 2D (1613): {mordred_2d_time:.2f}s, {len(mordred_2d_results[0])} features")
print(f"Mordred all (1826): {mordred_all_time:.2f}s, {len(mordred_all_results[0])} features")
```

## Descriptor Collections

### Curated Sets for Applications

```python
# Define custom descriptor collections for specific use cases

def get_drug_discovery_descriptors(smiles_list):
    """Get descriptors optimized for drug discovery"""
    essential = mbl.get_featurizer("rdkit_essential_descriptors")
    lipinski = mbl.get_featurizer("rdkit_lipinski_descriptors") 
    admet = mbl.get_featurizer("rdkit_admet_descriptors")
    
    results = []
    for smiles in smiles_list:
        desc_essential = essential.featurize(smiles)
        desc_lipinski = lipinski.featurize(smiles)
        desc_admet = admet.featurize(smiles)
        
        # Combine descriptors
        combined = np.concatenate([desc_essential, desc_lipinski, desc_admet])
        results.append(combined)
    
    return results

def get_interpretable_descriptors(smiles_list):
    """Get most interpretable descriptors"""
    rdkit_desc = mbl.get_featurizer("rdkit_all_descriptors")
    
    # Select interpretable descriptors by name
    interpretable_indices = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25]  # Key indices
    
    results = []
    for smiles in smiles_list:
        all_desc = rdkit_desc.featurize(smiles)
        interpretable = all_desc[interpretable_indices]
        results.append(interpretable)
    
    return results

# Usage
drug_compounds = ["CC1=CC=C(C=C1)C(=O)O", "CN1CCN(CC1)C2=CC=CC=C2"]
drug_desc = get_drug_discovery_descriptors(drug_compounds)
interpretable_desc = get_interpretable_descriptors(drug_compounds)

print(f"Drug discovery descriptors: {len(drug_desc[0])} features")
print(f"Interpretable descriptors: {len(interpretable_desc[0])} features")
```

### Chemical Space Analysis

Use descriptors to analyze chemical space:

```python
# Analyze descriptor distributions across different molecule types
molecule_classes = {
    "Small drugs": ["CCO", "CC(=O)O", "CN", "CCC"],
    "Natural products": ["C[C@H]1CC[C@H](C[C@H]1O)C(=O)O", 
                        "CC(=O)OC1=CC=CC=C1C(=O)O"],
    "Peptides": ["CC(C)C[C@H](NC(=O)C)C(=O)N[C@@H](CC(=O)O)C(=O)O"],
    "Aromatics": ["c1ccccc1", "c1ccc2ccccc2c1", "c1ccc(cc1)c2ccccc2"]
}

rdkit_desc = mbl.get_featurizer("rdkit_essential_descriptors")

print("Chemical Space Analysis:")
for class_name, molecules in molecule_classes.items():
    descriptors = [rdkit_desc.featurize(mol) for mol in molecules]
    descriptors = np.array(descriptors)
    
    # Analyze key properties
    mw_mean = np.mean(descriptors[:, 0])  # Molecular weight
    logp_mean = np.mean(descriptors[:, 1])  # LogP
    tpsa_mean = np.mean(descriptors[:, 2])  # TPSA
    
    print(f"{class_name:15}: MW={mw_mean:.1f}, LogP={logp_mean:.2f}, TPSA={tpsa_mean:.1f}")
```

## Handling Missing Values

### Missing Value Strategies

```python
# Different strategies for handling missing values
strategies = [
    ("default", {}),                           # NaN preserved
    ("zero_fill", {"fill_value": 0}),         # Replace NaN with 0
    ("mean_fill", {"fill_value": "mean"}),    # Replace with mean
    ("drop_nan", {"drop_nan_features": True}) # Remove features with NaN
]

problematic_mol = "[Pt]"  # Metal complex - may cause descriptor failures

for strategy_name, params in strategies:
    try:
        mordred = mbl.get_featurizer("mordred_descriptors_2d", **params)
        descriptors = mordred.featurize(problematic_mol)
        nan_count = np.isnan(descriptors).sum()
        
        print(f"{strategy_name:10}: shape={descriptors.shape}, NaN count={nan_count}")
        
    except Exception as e:
        print(f"{strategy_name:10}: Failed - {e}")
```

### Robust Batch Processing

Handle descriptor failures gracefully in batch processing:

```python
def robust_descriptor_calculation(smiles_list, featurizer_name="rdkit_all_descriptors"):
    """Calculate descriptors with error handling"""
    featurizer = mbl.get_featurizer(featurizer_name, fill_value=0)
    
    results = []
    failed_molecules = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            descriptors = featurizer.featurize(smiles)
            
            # Check for excessive missing values
            nan_fraction = np.isnan(descriptors).sum() / len(descriptors)
            if nan_fraction > 0.5:  # More than 50% missing
                print(f"Warning: {smiles} has {nan_fraction:.1%} missing descriptors")
            
            results.append(descriptors)
            
        except Exception as e:
            print(f"Failed to calculate descriptors for {smiles}: {e}")
            failed_molecules.append((i, smiles))
            results.append(None)
    
    return results, failed_molecules

# Test with diverse molecules including edge cases
test_molecules = [
    "CCO",           # Simple alcohol
    "CC(=O)O",       # Carboxylic acid  
    "[Pt]",          # Metal (may fail)
    "C" * 50,        # Very large molecule
    "invalid"        # Invalid SMILES
]

descriptors, failures = robust_descriptor_calculation(test_molecules)

print(f"Successfully calculated: {sum(1 for d in descriptors if d is not None)}/{len(test_molecules)}")
print(f"Failed molecules: {len(failures)}")
for idx, smiles in failures:
    print(f"  {idx}: {smiles}")
```

## Integration with Machine Learning

### Feature Selection

Use descriptor interpretability for feature selection:

```python
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

# Generate descriptors for a dataset
molecules = [
    "CCO", "CCN", "CCC", "CC(C)C", "CCCC", "c1ccccc1",
    "CC(=O)O", "CCO", "CN", "CC(C)C(=O)O"
]

# Simulate target property (e.g., solubility)
np.random.seed(42)
target = np.random.normal(0, 1, len(molecules))

# Calculate descriptors
rdkit_desc = mbl.get_featurizer("rdkit_essential_descriptors")
X = np.array([rdkit_desc.featurize(mol) for mol in molecules])
y = target

# Feature selection
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)

# Get selected feature indices
selected_features = selector.get_support(indices=True)
feature_names = rdkit_desc.get_feature_names()

print("Selected features for ML model:")
for idx in selected_features:
    score = selector.scores_[idx]
    print(f"  {feature_names[idx]}: score={score:.3f}")

print(f"\nReduced from {X.shape[1]} to {X_selected.shape[1]} features")
```

### Dataset Integration

```python
from molblender.data import MolecularDataset

# Create dataset with descriptor features
molecules = ["CCO", "CCN", "CCC", "c1ccccc1"]
dataset = MolecularDataset.from_smiles(molecules)

# Add multiple descriptor types
dataset.add_features("rdkit_essential_descriptors", n_workers=4)
dataset.add_features("rdkit_lipinski_descriptors", n_workers=4) 
dataset.add_features("mordred_descriptors_2d", n_workers=4)

# Access descriptor features
print("Dataset with descriptor features:")
print(dataset.features.columns.tolist())
print(f"Feature dimensions: {[f.shape for f in dataset.features.iloc[0]]}")

# Combine all descriptors
all_descriptors = []
for idx in range(len(dataset)):
    combined = np.concatenate([
        dataset.features.iloc[idx, 0],  # RDKit essential
        dataset.features.iloc[idx, 1],  # RDKit Lipinski  
        dataset.features.iloc[idx, 2]   # Mordred 2D
    ])
    all_descriptors.append(combined)

all_descriptors = np.array(all_descriptors)
print(f"Combined descriptors shape: {all_descriptors.shape}")
```

## Performance Tips

### Optimization Strategies

```python
# 1. Choose appropriate descriptor set
# For initial screening: use essential descriptors
# For detailed analysis: use comprehensive sets

# 2. Handle missing values efficiently
mordred_fast = mbl.get_featurizer("mordred_descriptors_2d", 
                                fill_value=0,        # Faster than mean
                                ignore_3D=True)      # Skip 3D calculations

# 3. Use parallel processing for large datasets
def process_large_dataset(smiles_list, batch_size=1000):
    """Process large datasets in batches"""
    featurizer = mbl.get_featurizer("rdkit_all_descriptors")
    
    all_results = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        batch_results = featurizer.featurize(batch, n_workers=4)
        all_results.extend(batch_results)
        print(f"Processed batch {i//batch_size + 1}")
    
    return all_results

# 4. Memory-efficient processing
def memory_efficient_descriptors(smiles_list, output_file):
    """Calculate descriptors with memory management"""
    import pickle
    
    featurizer = mbl.get_featurizer("rdkit_essential_descriptors")
    
    with open(output_file, 'wb') as f:
        for i, smiles in enumerate(smiles_list):
            descriptors = featurizer.featurize(smiles)
            pickle.dump((smiles, descriptors), f)
            
            # Clear memory periodically
            if i % 1000 == 0:
                print(f"Processed {i} molecules")

# Example usage for large datasets
large_smiles = ["CCO"] * 10000  # Simulate large dataset
# results = process_large_dataset(large_smiles)
print("Use process_large_dataset() for >1000 molecules")
```

## References

- [RDKit Descriptors](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors) - Complete list of RDKit descriptors
- [Mordred Documentation](https://mordred-descriptor.github.io/documentation/) - Comprehensive descriptor library
- [ADMET Properties Review](https://doi.org/10.1021/acs.chemrestox.9b00157) - ADMET descriptor applications
- [Lipinski's Rule](https://doi.org/10.1016/S0169-409X(00)00129-0) - Drug-likeness criteria

```{toctree}
:maxdepth: 1
:hidden:

descriptors_basic
```