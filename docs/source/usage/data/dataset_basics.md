# Dataset Basics

The `MolecularDataset` class provides a unified container for managing collections of molecules along with their associated data (labels, features, and weights).

This guide covers the fundamental operations for creating and manipulating molecular datasets.

## Creating Datasets

### From SMILES Strings

```python
from molblender.data import MolecularDataset, InputType

# Create a dataset from SMILES strings
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]  # Ethanol, Benzene, Acetic acid
ids = ["mol1", "mol2", "mol3"]
properties = [
    {"logP": -0.31, "activity": 1},
    {"logP": 2.13, "activity": 0},
    {"logP": -0.17, "activity": 1}
]

dataset = MolecularDataset.from_smiles_list(
    smiles_list=smiles_list,
    ids=ids,
    properties_list=properties
)

print(f"Dataset size: {len(dataset)}")
print(f"Labels: {dataset.label_names}")
```

### From CSV File

```python
from molblender.data import MolecularDataset, InputType

# Create a dataset from a CSV file containing SMILES
dataset = MolecularDataset.from_csv(
    filepath="compounds.csv",
    input_column="SMILES",           # Column containing SMILES
    id_column="ID",                  # Column for molecule IDs
    label_columns=["pIC50", "LogP"], # Columns to use as labels
    label_error_map={"pIC50": "pIC50_stderr"}, # Map labels to error columns
    mol_input_type=InputType.SMILES, # Specify input type
    sanitize=True                    # Sanitize molecules during creation
)

# Access label values and associated errors
labels_with_errors = dataset.get_labels(names=["pIC50"], include_errors=True)
print(labels_with_errors.head())
# Output:
#          pIC50        
#         value  error
# mol1     7.2    0.3
# mol2     6.5    0.2
```

### From CSV with Both SMILES and Structure Files

```python
from molblender.data import MolecularDataset, InputType

# Create a dataset that handles both SMILES and structure files
dataset = MolecularDataset.from_dual_input_csv(
    filepath="compounds.csv",
    smiles_column="SMILES",           # Column with SMILES
    structure_column="FilePath",      # Column with file paths
    structure_type=InputType.PDB_FILE, # Type of structure files 
    primary_input="smiles",           # Use SMILES as primary input
    label_columns=["Activity", "LogP"],
    label_error_map={"Activity": "Activity_error"}
)

# Access molecules with both representations
mol = dataset.get_molecule(0)
print(f"SMILES: {mol.smiles}")
print(f"Structure file: {mol.get_property('FilePath')}")
```

### From SDF File

```python
from molblender.data import MolecularDataset

# Create a dataset from an SDF file
dataset = MolecularDataset.from_sdf(
    filepath="compounds.sdf",
    id_tag="ID",                      # SDF tag for molecule IDs
    label_tags=["IC50", "LogP"],      # SDF tags to use as labels
    label_error_suffix="_err",        # Suffix for error tags (IC50_err)
    include_all_properties=True       # Include all SDF properties
)
```

## Accessing Dataset Contents

### Molecules and Basic Properties

```python
# Get basic dataset information
print(f"Dataset size: {len(dataset)}")
print(f"Label names: {dataset.label_names}")
print(f"Feature names: {dataset.feature_names}")

# Access molecules by position (0-based index)
first_mol = dataset.get_molecule(0)  
print(f"First molecule: {first_mol.smiles}")

# Access molecules by ID
mol = dataset.get_molecule("mol1")
print(f"Molecule ID: {mol.id}")
print(f"SMILES: {mol.smiles}")
print(f"Properties: {mol.properties}")

# Get all molecule IDs
all_ids = dataset.ids
print(f"All IDs: {list(all_ids)[:5]}")
```

### Labels, Features and Weights

```python
# Get all labels (as DataFrame)
all_labels = dataset.get_labels()

# Get specific labels with their errors
activity_data = dataset.get_labels(names=["Activity"], include_errors=True)
print(activity_data.head())

# Access specific features
morgan_fps = dataset.get_features(names=["morgan_fp_r2_2048"])

# Get weights
sample_weights = dataset.get_weights(names=["weight"])
```

### Iterating Through Samples

```python
# Iterate through all samples
for mol_id, molecule, labels, features, weights in dataset.itersamples():
    print(f"ID: {mol_id}")
    print(f"SMILES: {molecule.smiles}")
    print(f"Activity: {labels.get('Activity')}")
    print(f"Has Morgan FP: {'morgan_fp_r2_2048' in features}")
    print("-" * 40)
```

## Adding Features

```python
import molblender as mbl
from molblender.data import MolecularDataset

# List available featurizers
available_featurizers = mbl.list_available_featurizers()
print(f"Available featurizers: {available_featurizers[:5]}...")

# Add a single feature set 
dataset.add_features(
    featurizer_inputs="morgan_fp_r2_2048",  # Use registered featurizer name
    n_workers=4,                            # Use 4 parallel workers
    on_error="store_none"                   # Store None for failed molecules
)

# Add multiple feature sets at once
dataset.add_features(
    featurizer_inputs=["maccs_keys", "rdkit_fp_2048"],
    feature_names=["maccs", "rdkit"],       # Custom names for feature columns
    n_workers=4
)

print(f"Available features: {dataset.feature_names}")
```

### Combining Features

```python
# Concatenate features horizontally (e.g., combine fingerprints)
dataset.concatenate_features(
    feature_names=["morgan_fp_r2_2048", "maccs"],
    new_feature_name="combined_fingerprint",
    axis=1,                                 # Horizontal concatenation
    on_error="nan"                          # Fill errors with NaN
)
```

### Handling Feature Generation Errors

```python
# Check for featurization failures
failures = dataset.get_featurization_failures("morgan_fp_r2_2048")
print(f"Number of failures: {len(failures)}")

# Drop molecules that failed featurization
dataset.drop_featurization_failures("morgan_fp_r2_2048")
print(f"Dataset size after dropping failures: {len(dataset)}")
```

## Working with Dual Representation Molecules

The `MolecularDataset` can store both SMILES and structural data for the same molecules:

```python
# Structure-centric approach
dataset_3d = MolecularDataset.from_dual_input_csv(
    filepath="compounds.csv",
    smiles_column="SMILES",
    structure_column="FilePath",
    structure_type=InputType.PDB_FILE,
    primary_input="structure",  # Structure files as primary input
    label_columns=["Activity"]
)

# Access data
mol = dataset_3d.get_molecule(0)
print(f"Structure file: {mol.input_data}")           # Primary input
print(f"SMILES from property: {mol.get_property('SMILES')}")  # Secondary input
```

## FAQ

**Q: Can we generate SMILES from a structure?**  
A: Yes, if a molecule is loaded from a structure file, you can access its SMILES representation via the `molecule.smiles` property, which will attempt to generate a SMILES string using RDKit.

**Q: Can we get molecules by both order and ID?**  
A: Yes, the `get_molecule()` method can retrieve molecules by:
- Positional index: `dataset.get_molecule(0)` (first molecule)
- ID: `dataset.get_molecule("mol1")` (molecule with ID "mol1")

**Q: How can I read both SMILES and structure data from one CSV file?**  
A: Use the `from_dual_input_csv` method to easily load both representations:

```python
dataset = MolecularDataset.from_dual_input_csv(
    filepath="compounds.csv",
    smiles_column="SMILES",
    structure_column="FilePath",
    structure_type=InputType.PDB_FILE,
    primary_input="smiles"  # Or "structure" if you prefer
)
```

For more detailed information, see the {doc}`/api/data/dataset` API reference.