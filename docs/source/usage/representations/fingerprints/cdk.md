# CDK Fingerprints

This guide describes how to use the CDK (Chemistry Development Kit) fingerprint featurizers in MolBlender. These featurizers leverage the Java-based CDK library via the `CDK-pywrapper` Python package to generate a diverse range of molecular fingerprints.

## Overview

CDK fingerprints are widely used in cheminformatics for molecular similarity searching, QSAR modeling, and virtual screening. They provide a different implementation compared to RDKit fingerprints, which can be valuable for ensemble approaches or when specific CDK fingerprint types are needed.

## Installation Requirements

To use CDK fingerprints, you need:

```bash
# Install CDK-pywrapper
pip install CDK-pywrapper

# Ensure you have Java installed (JDK 8 or higher recommended)
# You can check your Java installation with:
java -version
```

Additionally, a working RDKit installation is required for molecule handling:

```bash
# Either via pip
pip install rdkit

# Or via conda (recommended)
conda install -c conda-forge rdkit
```

## Available CDK Fingerprints

MolBlender provides access to the following CDK fingerprints:

| Featurizer Name | Description | Bit Length | Use Case |
|:----------------|:------------|:-----------|:---------|
| `cdk` | Standard CDK fingerprint | Variable | General-purpose molecule similarity |
| `cdk_ext` | Extended CDK fingerprint | Variable | Enhanced standard fingerprint with more features |
| `cdk_graph` | Graph-only CDK fingerprint | Variable | Purely topological features |
| `cdk_estate` | E-State fingerprint | Variable | Electrotopological state indices |
| `cdk_maccs` | MACCS keys | **166** | MDL MACCS key-based substructure patterns |
| `cdk_pubchem` | PubChem CACTVS fingerprint | **881** | PubChem's CACTVS-based substructure keys |
| `cdk_sub` | Substructure fingerprint | Variable | Predefined chemical substructure patterns |
| `cdk_kr` | Klekota-Roth fingerprint | Variable | Substructures from Klekota-Roth, useful for bioactivity |
| `cdk_ap2d` | Atom Pairs 2D fingerprint | Variable | Atom-pair relationships and distances |
| `cdk_hybrid` | Hybridization fingerprint | Variable | Atom hybridization state patterns |
| `cdk_lingo` | Lingo fingerprint | Variable | N-character SMILES substrings as features |
| `cdk_shortest_path` | Shortest Path fingerprint | Variable | Path-based descriptors between atoms |
| `cdk_signature` | Signature fingerprint | Variable | Atom environment signatures |
| `cdk_circular` | Circular (ECFP-like) fingerprint | Variable | Morgan/ECFP-like circular environments |

**Notes**:
- Only MACCS and PubChem have fixed, guaranteed sizes
- Other fingerprints' lengths may vary based on molecule complexity
- All CDK fingerprints are returned as binary (0/1) arrays of type uint8

## Basic Usage

```python
from molblender.representations import get_featurizer
from rdkit import Chem

# 1. Create a featurizer using the registry
featurizer = get_featurizer("cdk_maccs")

# 2. Option A: Pass a SMILES string directly
fp = featurizer.featurize("CCO")  # Ethanol
print(f"MACCS fingerprint shape: {fp.shape}")  # (166,)

# 2. Option B: Pass an RDKit molecule
mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
fp = featurizer.featurize(mol)
print(f"MACCS fingerprint for aspirin: {fp.shape}")  # (166,)

# 3. Process multiple molecules
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]  # Ethanol, Benzene, Acetic acid
fps = featurizer.featurize(smiles_list)
print(f"Number of fingerprints: {len(fps)}")  # 3
```

## Advanced Usage

### Parallel Processing

CDK fingerprints support parallel processing, but with an important caveat:

```python
# Process molecules in parallel (using threading, not multiprocessing)
# This is important because the JVM doesn't work well with multiprocessing
fps = featurizer.featurize(smiles_list, n_workers=4)  # Uses threading
```

### Handling Different Input Types

The CDK featurizers accept both SMILES strings and RDKit Mol objects:

```python
from molblender.representations import get_featurizer

featurizer = get_featurizer("cdk_pubchem")

# Processing both SMILES and RDKit Mol in the same batch
from rdkit import Chem
inputs = [
    "CCO",                     # SMILES string
    Chem.MolFromSmiles("c1ccccc1"),  # RDKit Mol object
    "CC(=O)O"                  # SMILES string
]

fps = featurizer.featurize(inputs)
print(f"PubChem fingerprint shape: {fps[0].shape}")  # (881,)
```

### Comparing CDK and RDKit Fingerprints

```python
from molblender.representations import get_featurizer
import numpy as np
from rdkit import Chem

# Create featurizers
cdk_maccs = get_featurizer("cdk_maccs")
rdkit_maccs = get_featurizer("maccs_keys")

# Featurize the same molecule
mol = Chem.MolFromSmiles("CCO")
cdk_fp = cdk_maccs.featurize(mol)
rdkit_fp = rdkit_maccs.featurize(mol)

# Compare the outputs
print(f"CDK MACCS shape: {cdk_fp.shape}")    # (166,)
print(f"RDKit MACCS shape: {rdkit_fp.shape}")  # (167,)

# Note the difference: RDKit includes a "counted bits" at position 0
# that CDK doesn't have (hence 167 vs 166 bits)
```

## Performance Considerations

CDK fingerprints have certain characteristics to be aware of:

1. **JVM Overhead**: The first featurization operation incurs JVM startup costs
2. **Threading vs Multiprocessing**: CDK featurizers use threading for parallelization (not multiprocessing)
3. **Memory Usage**: The CDK-Java bridge requires more memory than pure Python implementations
4. **Speed**: CDK fingerprints are typically slower than their RDKit counterparts but offer different implementation details

## Troubleshooting

| Problem | Possible Cause | Solution |
|:--------|:---------------|:---------|
| `ImportError: No module named CDK_pywrapper` | CDK-pywrapper not installed | `pip install CDK-pywrapper` |
| `java.lang.UnsatisfiedLinkError` | Java not found or incompatible | Ensure Java 8+ is installed and check `JAVA_HOME` environment variable |
| `DependencyNotFoundError: CDK components not loaded` | CDK backend initialization failed | Check Java installation and CDK-pywrapper version |
| `TypeError: an integer is required` | Type conversion issues | Ensure RDKit mol objects are properly sanitized |
| `OutOfMemoryError: Java heap space` | Running out of JVM memory | Increase Java heap size via environment variables (`_JAVA_OPTIONS="-Xmx2g"`) |

## When to Use CDK Fingerprints

- When you need specific CDK implementations not available in RDKit
- For ensemble approaches combining multiple fingerprint sources
- When working with existing workflows that expect CDK fingerprints
- For comparative studies between different cheminformatics toolkits

## See Also

- {doc}`/api/representations/fingerprints/cdk` - API documentation
- {doc}`/usage/representations/fingerprints/rdkit` - RDKit fingerprints alternative
- {doc}`/usage/representations/fingerprints/datamol` - Molfeat/Datamol fingerprints
- [CDK Official Documentation](https://cdk.github.io/) - Chemistry Development Kit docs
- [CDK-pywrapper GitHub](https://github.com/pzc/cdk-pywrapper) - Python wrapper for CDK

```{toctree}
:hidden:
```