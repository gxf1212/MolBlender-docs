# Datamol Fingerprint

This guide explains how to use the Datamol/Molfeat fingerprint featurizers implemented in MolBlender. These featurizers wrap the `molfeat.calc.FPCalculator` class and provide consistent access to a range of popular molecular fingerprints.

## Background

`molfeat.calc.FPCalculator` is a unified interface to compute molecular fingerprints from SMILES strings or RDKit `Mol` objects. MolBlender integrates this through `BaseDatamolFPCalculator`, enabling standardized use and registry-based loading of featurizers.

Each featurizer is registered under a specific name (e.g., `datamol_maccs`, `datamol_ecfp4_2048`), making it easy to plug into existing workflows using the featurizer registry.

## Installation Requirements

To use this module, you must have `molfeat` installed:

```bash
pip install molfeat
```

For certain fingerprint types, additional dependencies may be required:
- For MAP4 fingerprints: `pip install map4-ojmb`

## Available Fingerprints

The following fingerprints are registered and usable if `molfeat` is available:

| Fingerprint Name | Description | Bit Size |
|------------------|-------------|----------|
| `datamol_maccs` | MACCS Keys | 167 |
| `datamol_erg` | Extended Reduced Graph | 315 |
| `datamol_estate` | EState | 79 |
| `datamol_ecfp4_2048` | Morgan ECFP4 | 2048 |
| `datamol_ecfp6_2048` | Morgan ECFP6 | 2048 |
| `datamol_fcfp4_2048` | Morgan FCFP4 | 2048 |
| `datamol_topological` | Path-Based Topological | 2048 |
| `datamol_atompair` | Atom Pair | 2048 |
| `datamol_rdkit` | RDKit Topological | 2048 |
| `datamol_avalon` | Avalon | 512 |
| `datamol_ecfp4_count` | Morgan ECFP4 Count | Variable |
| `datamol_fcfp4_count` | Morgan FCFP4 Count | Variable |
| `datamol_atompair_count` | Atom Pair Count | Variable |
| `datamol_secfp` | SMILES Extended Connectivity | 2048 |
| `datamol_map4` | MAP4 (Requires map4-ojmb) | 1024 |


## Basic Usage

### Simple Example

```python
from molblender.representations import get_featurizer

# Example SMILES input
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine

# Get the Avalon fingerprint featurizer
featurizer = get_featurizer("datamol_avalon")

# Featurize the SMILES
features = featurizer(smiles)

print(features.shape)  # Should print: (512,)
```

### Batch Processing

```python
smiles_list = [
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(=O)OC1=CC=CC=C1C(=O)O",      # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
]

# Get fingerprints for all molecules
features_list = featurizer.featurize(smiles_list)

# Process in parallel (optional)
features_list = featurizer.featurize(smiles_list, n_workers=4)
```

### Using with RDKit Molecules

```python
from rdkit import Chem

# Create an RDKit molecule
mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

# Featurize the RDKit molecule directly
features = featurizer.featurize(mol)
```

### Custom Parameters

```python
# Override default parameters for ECFP4
custom_featurizer = get_featurizer(
    "datamol_ecfp4_2048", 
    radius=3,          # Change radius from 2 to 3
    length=1024,       # Change bit length from 2048 to 1024
    useChirality=True  # Add chirality information
)
```

## Advanced Usage

### Available Fingerprints Check

To list all available Datamol fingerprints:

```python
from molblender.representations.fingerprints.datamol import list_datamol_fingerprints

# Print info about all available datamol fingerprints
list_datamol_fingerprints()
```

### Error Handling and Testing

When working with fingerprints that might have dependency issues:

```python
from molblender.representations import get_featurizer

smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

for name in ["datamol_map4", "datamol_avalon"]:
    try:
        featurizer = get_featurizer(name)
        output = featurizer(smiles)
        print(f"{name}: {output.shape}")
    except Exception as e:
        print(f"{name} failed with error: {e}")
```

## Internals

* Featurizers are registered using the `register_featurizer` function during module import.
* Each fingerprint is identified by its `fp_name`, which maps to molfeat's supported types (e.g., `ecfp`, `maccs`, `erg`).
* The featurizer tries to determine output shape (`OUTPUT_SHAPE`) from `fp_name` and bit size hints.

## Reference Fingerprint Names

These are the fingerprint types supported by the underlying molfeat library:

```python
['maccs', 'avalon', 'ecfp', 'fcfp', 'topological', 'atompair', 'rdkit',
 'pattern', 'layered', 'map4', 'secfp', 'erg', 'estate',
 'avalon-count', 'rdkit-count', 'ecfp-count', 'fcfp-count',
 'topological-count', 'atompair-count']
```

## Notes

* If `molfeat` is missing, featurizers won't register and will raise `DependencyNotFoundError`.
* Internally, all conversion errors and dependency issues raise `FeaturizationError`.
* For count-based fingerprints, the output size may vary based on the input molecule's complexity.

## See Also

* {doc}`/api/representations/fingerprints/datamol` - API documentation
* {doc}`/usage/representations/fingerprints/rdkit` - RDKit fingerprints
* {doc}`/usage/representations/fingerprints/cdk` - CDK fingerprints

```{toctree}
:hidden:
```