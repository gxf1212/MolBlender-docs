# DeepChem Fingerprints

This guide describes how to use DeepChem-based fingerprint featurizers in MolBlender. These featurizers provide a bridge to the molecular fingerprinting capabilities in the DeepChem library, with consistent interfaces and integration with MolBlender's featurizer registry.

## Overview

DeepChem is a powerful library for deep learning in drug discovery, materials science, and quantum chemistry. It provides a variety of fingerprinting techniques that MolBlender makes available through a consistent interface. These include standard molecular fingerprints (MACCS, Morgan/ECFP), as well as more advanced representations like Mol2Vec and MAT.

## Installation Requirements

To use DeepChem fingerprints, you need:

```bash
# Core requirement
pip install deepchem

# For PubChem fingerprints
pip install pubchempy

# For Mol2Vec fingerprints
pip install mol2vec
```

Mol2Vec uses a pre-trained model. MolBlender will **auto-download and cache** it on first use
(default cache: `~/.mol2vec`). If you're offline, download the model manually and pass
`pretrain_model_path` when creating the featurizer.

PubChem fingerprints require **network access**. In offline or restricted environments,
`deepchem_pubchem` will be skipped with a warning.

DeepChem installation can sometimes be complex due to its dependencies. If you encounter issues, consider using Conda:

```bash
conda install -c conda-forge deepchem
```

## Available DeepChem Fingerprints

MolBlender provides access to the following DeepChem fingerprints:

| Featurizer Name | Description | Output Format | Dependencies |
|:----------------|:------------|:--------------|:-------------|
| `deepchem_maccs` | MACCS Keys fingerprint | 167-bit array | DeepChem |
| `deepchem_morgan_r2_2048` | Morgan/ECFP4 fingerprint | 2048-bit array | DeepChem |
| `deepchem_morgan_r3_1024` | Morgan/ECFP6 fingerprint | 1024-bit array | DeepChem |
| `deepchem_morgan_count_r2_2048` | Morgan count-based fingerprint | Sparse dictionary | DeepChem |
| `deepchem_pubchem` | PubChem fingerprint | 881-bit array | DeepChem, PubChemPy |
| `deepchem_mol2vec` | Mol2Vec embeddings | 300-dim vector | DeepChem, mol2vec |
| `deepchem_mat` | Molecular Attention Transformer | Complex output | DeepChem |

## Basic Usage

```python
from molblender.representations import get_featurizer

# 1. Create a DeepChem fingerprint featurizer
featurizer = get_featurizer("deepchem_maccs")

# 2. Process a single molecule (SMILES input)
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
fp = featurizer.featurize(smiles)
print(f"MACCS fingerprint shape: {fp.shape}")  # (167,)

# 3. Process multiple molecules
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]  # Ethanol, Benzene, Acetic acid
fps = featurizer.featurize(smiles_list)
print(f"Number of fingerprints: {len(fps)}")  # 3
```

## Advanced Usage

### Listing Available Fingerprints

You can list all available DeepChem fingerprints:

```python
from molblender.representations.fingerprints.deepchem import list_deepchem_fingerprints

# Print information about all available DeepChem fingerprints
list_deepchem_fingerprints()
```

### Using Morgan Count Fingerprints

The count-based Morgan fingerprint returns a sparse dictionary instead of a bit vector:

```python
featurizer = get_featurizer("deepchem_morgan_count_r2_2048")
count_fp = featurizer.featurize("CCO")  # Returns a dictionary of {feature_idx: count}

# Example output:
# {1523: 1, 890: 2, 901: 1, ...}
```

### Using Mol2Vec Embeddings

Mol2Vec is a word2vec-inspired model that generates continuous vector representations:

```python
# Auto-download model on first use (cached in ~/.mol2vec)
mol2vec = get_featurizer("deepchem_mol2vec")

# Optional: specify a custom pre-trained model
custom_mol2vec = get_featurizer("deepchem_mol2vec", 
                                pretrain_model_path="/path/to/model.pkl")

embedding = mol2vec.featurize("CCO")
print(f"Mol2Vec embedding shape: {embedding.shape}")  # (300,)
```

### Parallel Processing

DeepChem featurizers support parallel processing for batch operations:

```python
# Process in parallel with 4 workers
fps = featurizer.featurize(smiles_list, n_workers=4)
```

## Comparing with Other Fingerprints

DeepChem fingerprints can be used alongside other implementations:

```python
# Compare DeepChem MACCS with RDKit MACCS
dc_maccs = get_featurizer("deepchem_maccs")
rdkit_maccs = get_featurizer("maccs_keys")

dc_fp = dc_maccs.featurize("CCO")
rdkit_fp = rdkit_maccs.featurize("CCO")

print(f"DeepChem MACCS: {dc_fp.shape}")  # (167,)
print(f"RDKit MACCS: {rdkit_fp.shape}")   # (167,)
```

## Customizing Fingerprints

You can customize the fingerprint parameters when creating the featurizer:

```python
# Custom Morgan fingerprint with chirality
custom_morgan = get_featurizer("deepchem_morgan_r2_2048", 
                              radius=2,       # ECFP4 radius
                              size=1024,      # Output bit vector size
                              chiral=True,    # Include chirality
                              features=True)  # Use atom features (FCFP-like)
```

## Troubleshooting

| Problem | Possible Cause | Solution |
|:--------|:---------------|:---------|
| `ModuleNotFoundError: No module named 'deepchem'` | DeepChem not installed | Install with `pip install deepchem` |
| `DependencyNotFoundError: PubChemPy is required...` | Missing PubChemPy | Install with `pip install pubchempy` |
| `DependencyNotFoundError: mol2vec is required...` | Missing mol2vec | Install with `pip install mol2vec` |
| `Mol2Vec model download failed` | No network / blocked download | Manually download `model_300dim.pkl` and pass `pretrain_model_path` |
| `ValueError: This class requires RDKit to be installed...` | RDKit missing | Install RDKit with `pip install rdkit` or via conda |
| PubChem fingerprints unavailable | No network or PubChem blocked | Ensure connectivity or skip `deepchem_pubchem` |
| Slow performance with PubChem fingerprints | Network requests | PubChem fingerprints may require internet connectivity |

## When to Use DeepChem Fingerprints

- When you need to maintain compatibility with DeepChem workflows
- When using specific DeepChem models that expect these fingerprints
- For advanced representations like Mol2Vec that aren't available in other packages
- When building ensemble methods that combine multiple fingerprint implementations

## See Also

- {doc}`/api/representations/fingerprints/deepchem` - API documentation
- {doc}`/usage/representations/fingerprints/rdkit` - RDKit fingerprints
- {doc}`/usage/representations/fingerprints/cdk` - CDK fingerprints
- {doc}`/usage/representations/fingerprints/datamol` - Molfeat/Datamol fingerprints
- [DeepChem Documentation](https://deepchem.readthedocs.io/) - DeepChem official docs

```{toctree}
:hidden:
```
