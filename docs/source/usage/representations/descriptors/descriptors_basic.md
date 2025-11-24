# Molecular Descriptors

Molecular descriptors are numerical values that characterize properties of molecules. Unlike fingerprints (which focus on structural patterns), descriptors typically represent physiochemical properties such as molecular weight, logP, and topological indices.

MolBlender provides access to two major descriptor calculation engines:

1. **RDKit Descriptors**: All descriptors available through RDKit's `Descriptors.CalcMolDescriptors()` function
2. **Mordred Descriptors**: Comprehensive descriptor set from the Mordred library

```{toctree}
:maxdepth: 1
:hidden:

```

## RDKit Descriptors

RDKit provides a wide range of molecular descriptors (200+) including physical properties, topological indices, and more. These are accessed through the `RDKitAllDescriptorsFeaturizer`.

### Basic Usage

```python
from molblender import get_featurizer

# Create a featurizer using the registry system
featurizer = get_featurizer("rdkit_all_descriptors")

# Calculate descriptors for a single molecule (SMILES string)
descriptors = featurizer.featurize("CCO")
print(f"Generated {len(descriptors)} descriptors")
# Output: Generated 208 descriptors

# The output is a numpy array of float values
print(descriptors[:5])
# Example output: [13.01  30.07   2.    22.    10.  ]
```

### Batch Processing

You can process multiple molecules at once:

```python
from molblender import get_featurizer

# Get a featurizer from the registry
featurizer = get_featurizer("rdkit_all_descriptors")

# Calculate descriptors for multiple molecules at once
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
descriptors_batch = featurizer.featurize(smiles_list)

print(f"Number of molecules: {len(descriptors_batch)}")
print(f"Descriptors per molecule: {len(descriptors_batch[0])}")
# Example output: Number of molecules: 3, Descriptors per molecule: 208
```

## Mordred Descriptors

The Mordred library offers 1800+ descriptors organized in various categories. MolBlender provides easy access through the `MordredFeaturizer`.

### Basic Usage

```python
from molblender import get_featurizer

# Create a featurizer for 2D descriptors only (default)
featurizer_2d = get_featurizer("mordred_descriptors_2d")

# Create a featurizer for both 2D and 3D descriptors
featurizer_all = get_featurizer("mordred_descriptors_all")

# Calculate 2D descriptors from SMILES
descriptors_2d = featurizer_2d.featurize("CCO")
print(f"Generated {len(descriptors_2d)} 2D descriptors")
# Example output: Generated 1613 descriptors

# For 3D descriptors, the featurizer will automatically generate coordinates
descriptors_all = featurizer_all.featurize("CCO")
print(f"Generated {len(descriptors_all)} total descriptors")
# Example output: Generated 1826 descriptors
```

### Using Specific Descriptor Subsets

You can select specific descriptors to calculate by creating custom featurizer instances:

```python
from molblender import get_featurizer

# Create a featurizer with only selected descriptors
featurizer_subset = get_featurizer("mordred_descriptors_2d", 
                                   descriptor_subset=["SLogP", "TPSA", "nHeavyAtom", "MW", "nRing"])

descriptors = featurizer_subset.featurize("CCO")
print(f"Generated {len(descriptors)} selected descriptors")
# Example output: Generated 5 selected descriptors
```

### Available Mordred Featurizers

MolBlender provides several pre-configured Mordred descriptor featurizers:

```python
from molblender import list_available_featurizers

# List all available descriptor featurizers
descriptors = list_available_featurizers(categorized=True)
print("Available descriptor featurizers:")
for name in descriptors.get("descriptors", []):
    print(f"  - {name}")
```

## Handling Failed Descriptor Calculations

Some descriptors might fail to calculate for certain molecules. By default, failed calculations return `np.nan`. You can customize this behavior:

```python
from molblender import get_featurizer

# Create a featurizer with custom fill value for failed calculations
featurizer = get_featurizer("rdkit_all_descriptors", fill_value=0.0)

# This also works for Mordred descriptors
mordred_featurizer = get_featurizer("mordred_descriptors_2d", fill_value=-999)

# Test with a problematic molecule
descriptors = featurizer.featurize("C")
print(f"Descriptors calculated with custom fill value")
```

## API Reference

For complete details on all parameters and methods, see the API documentation:

- {doc}`/api/representations/descriptors/descriptors_basic`

TODO:

- [] test `def test_descriptor_subset_valid(self)` failed
