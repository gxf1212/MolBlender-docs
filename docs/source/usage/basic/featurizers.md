# Featurizers

MolBlender offers a comprehensive suite of tools for generating diverse molecular and protein representations. These representations, often called "features" or "embeddings," are crucial for training machine learning models in cheminformatics and bioinformatics. MolBlender provides a unified interface for molecular featurization, supporting both small molecules and proteins. All featurizers aim to follow a consistent API pattern, making it easy to switch between different representations.

Featurizers transform molecular structures (e.g., from SMILES strings, SDF/PDB files) or protein sequences into numerical representations suitable for machine learning. The library organizes featurizers by type and provides tools to discover, instantiate, and use them effectively.

## Core Workflow

Generating representations with MolBlender typically involves the following steps:

1.  **Discover Available Featurizers**: Identify the representations MolBlender offers for your specific data type (small molecules or proteins).
2.  **Get a Specific Featurizer**: Instantiate the desired featurizer, usually by its unique registered name, and optionally customize its parameters.
3.  **Prepare Input Data**: Ensure your input data (like SMILES strings, RDKit Mol objects, or protein sequences) is in a format the featurizer can process. MolBlender featurizers often handle common input types directly.
4.  **Generate Representations**: Apply the featurizer to your molecule(s) or sequence(s) to obtain the numerical features.

Let's explore each of these steps in more detail.

---
## 1. Discovering Available Featurizers

MolBlender maintains separate registries for featurizers designed for small molecules and those for proteins.

### Listing Featurizer Names
You can easily get a list of registered names:

```python
import molblender as pm

# List all available small molecule featurizer names
sm_featurizer_names = pm.list_available_featurizers()
print(f"Total small molecule featurizers: {len(sm_featurizer_names)}")
if sm_featurizer_names:
    print(f"Examples: {sm_featurizer_names[:3]}")
# Example Output:
# Total small molecule featurizers: 72
# Examples: ['UniMol-CLS-unimolv1-WithHs', 'UniMol-CLS-unimolv2-1.1b-NoHs', 'UniMol-CLS-unimolv2-1.1b-WithHs']

# List all available protein featurizer names
protein_featurizer_names = pm.list_available_protein_featurizers()
print(f"\nTotal protein featurizers: {len(protein_featurizer_names)}")
if protein_featurizer_names:
    print(f"Examples: {protein_featurizer_names[:3]}")
# Example Output:
# Total protein featurizers: 12
# Examples: ['ankh', 'carp', 'esm2']
```

### Categorized and Pretty-Printed View
For a more structured and user-friendly overview, especially in an interactive console, you can print a categorized tree of available featurizers:

```python
import molblender as pm

# Pretty print small molecule featurizers in a categorized tree
pm.print_available_featurizers(categorized=True, registry_type='small_molecule')

# Pretty print protein featurizers
pm.print_available_featurizers(categorized=True, registry_type='protein')
```
An example output for small molecule featurizers might look like:
```
Available Small Molecule Featurizers
====================================
Total: 72 featurizers in 10 categories

├── fingerprints (45 items)
│   ├── rdkit (20 items)
│   │   ├── morgan (8 items)
│   │   │   ├── morgan_fp_r2_2048
│   │   │   ├── morgan_fp_r3_2048
│   │   │   └── ...
│   │   └── maccs (1 item)
│   │       └── maccs_keys
│   ├── deepchem (10 items)
│   │   ├── circular_fp
│   │   └── ...
│   └── cdk (15 items)
├── descriptors (25 items)
│   ├── rdkit (20 items)
│   └── mordred (5 items)
└── spatial (20 items)
    ├── matrix (10 items)
    │   ├── coulomb_matrix
    │   └── ...
    └── unimol (10 items)
```

### Getting Detailed Information with Shape
To learn more about a specific featurizer, including its underlying class, default parameters, output shape, and description:

```python
import molblender as pm

# Get info for a small molecule featurizer
info_morgan = pm.get_featurizer_info('morgan_fp_r2_1024')
if info_morgan:
    print("\nInfo for 'morgan_fp_r2_1024':")
    print(f"  class: {info_morgan['class']}")
    print(f"  shape: {info_morgan['shape']}")  # Shape at top level!
    print(f"  shape_type: {info_morgan['shape_type']}")
    print(f"  default_kwargs: {info_morgan['default_kwargs']}")
    print(f"  description: {info_morgan['metadata']['description']}")
# Example Output:
# Info for 'morgan_fp_r2_1024':
#   class: <class 'molblender.representations.fingerprints.rdkit.MorganBitFP'>
#   shape: (1024,)
#   shape_type: fixed
#   default_kwargs: {'radius': 2, 'nBits': 1024}
#   description: Morgan fingerprint (radius 2, 1024 bits) (via RDKit)

# Get info for a protein featurizer
info_esm = pm.get_protein_featurizer_info('esm2_t6_8M_UR50D')
if info_esm:
    print("\nInfo for 'esm2_t6_8M_UR50D':")
    print(f"  class: {info_esm['class']}")
    print(f"  shape: {info_esm['shape']}")  # CLS token shape
    print(f"  per_residue_shape: {info_esm.get('per_residue_shape', 'N/A')}")
    print(f"  default_kwargs: {info_esm['default_kwargs']}")
# Example Output:
# Info for 'esm2_t6_8M_UR50D':
#   class: <class 'molblender.representations.protein.sequence.plm.ESM2Featurizer'>
#   shape: (320,)
#   per_residue_shape: (None, 320)
#   default_kwargs: {'plm_key': 'esm2_t6_8M_UR50D', ...}

# Quick shape check for any featurizer
def get_shape(name):
    info = pm.get_featurizer_info(name) or pm.get_protein_featurizer_info(name)
    return info['shape'] if info else None

print(f"\nMorgan FP shape: {get_shape('morgan_fp_r2_1024')}")  # (1024,)
print(f"Count FP shape: {get_shape('ecfp4_count')}")  # 'dynamic'
```

### Checking Shapes of All CDK Fingerprints
Here's how to examine the shapes of all CDK fingerprints:

```python
import molblender as pm

# Get all CDK fingerprints
cdk_fingerprints = pm.select_featurizers_by_category("fingerprints/cdk")
print(f"Found {len(cdk_fingerprints)} CDK fingerprints\n")

# Check shapes of all CDK fingerprints
cdk_shapes = {}
for fp_name in sorted(cdk_fingerprints):
    info = pm.get_featurizer_info(fp_name)
    if info:
        shape = info['shape']
        shape_type = info['shape_type']
        desc = info['metadata']['description'][:50] + "..."
        
        # Group by shape
        if shape not in cdk_shapes:
            cdk_shapes[shape] = []
        cdk_shapes[shape].append((fp_name, desc))
        
        print(f"{fp_name}:")
        print(f"  Shape: {shape} ({shape_type})")
        print(f"  Desc: {desc}\n")

# Summary by shape
print("\nCDK Fingerprints grouped by shape:")
for shape, fps in sorted(cdk_shapes.items(), key=lambda x: str(x[0])):
    print(f"\nShape {shape}: {len(fps)} fingerprints")
    for name, desc in fps[:3]:  # Show first 3
        print(f"  - {name}")
    if len(fps) > 3:
        print(f"  ... and {len(fps) - 3} more")

# Example Output:
# Found 15 CDK fingerprints
#
# cdk_atompairs2d:
#   Shape: (780,) (fixed)
#   Desc: CDK Atom Pairs 2D fingerprint (780 bits)...
#
# cdk_circular:
#   Shape: (1024,) (fixed)
#   Desc: CDK Circular/ECFP-like fingerprint (1024 bits)...
#
# CDK Fingerprints grouped by shape:
# Shape (780,): 1 fingerprints
#   - cdk_atompairs2d
# Shape (1024,): 8 fingerprints
#   - cdk_circular
#   - cdk_extended
#   - cdk_graphonly
#   ... and 5 more
```

### Finding Featurizers by Shape
You can also search for featurizers based on their output shape:

```python
import molblender as pm

# Find all 2048-bit fingerprints
def find_by_shape(target_shape, category=None):
    """Find all featurizers with a specific shape."""
    if category:
        candidates = pm.select_featurizers_by_category(category)
    else:
        candidates = pm.list_available_featurizers()
    
    results = []
    for name in candidates:
        info = pm.get_featurizer_info(name)
        if info and info['shape'] == target_shape:
            results.append(name)
    return results

# Find all 2048-dimensional fingerprints
fp_2048 = find_by_shape((2048,), category="fingerprints")
print(f"Found {len(fp_2048)} fingerprints with shape (2048,):")
for fp in fp_2048[:5]:
    print(f"  - {fp}")
if len(fp_2048) > 5:
    print(f"  ... and {len(fp_2048) - 5} more")

# Find all dynamic-shape featurizers
dynamic_feats = find_by_shape("dynamic")
print(f"\nFound {len(dynamic_feats)} featurizers with dynamic shapes:")
for feat in dynamic_feats[:3]:
    info = pm.get_featurizer_info(feat)
    print(f"  - {feat}: {info['metadata']['description'][:40]}...")
```

### Shape Compatibility Check
When building ensemble models, it's important to check shape compatibility:

```python
import molblender as pm

# Check if multiple featurizers have compatible shapes
featurizers_to_check = [
    'morgan_fp_r2_2048',
    'rdkit_fp_2048',
    'datamol_ecfp4',
    'maccs_keys'  # Different shape!
]

shapes = {}
for name in featurizers_to_check:
    info = pm.get_featurizer_info(name)
    if info:
        shapes[name] = info['shape']

print("Shape compatibility check:")
for name, shape in shapes.items():
    print(f"  {name}: {shape}")

# Check if all shapes are the same
unique_shapes = set(shapes.values())
if len(unique_shapes) == 1:
    print("\n✓ All featurizers have compatible shapes!")
else:
    print(f"\n✗ Found {len(unique_shapes)} different shapes: {unique_shapes}")
    print("  Consider using only featurizers with matching shapes for direct stacking.")

# Group by compatible shapes
from collections import defaultdict
shape_groups = defaultdict(list)
for name, shape in shapes.items():
    shape_groups[shape].append(name)

print("\nCompatible groups:")
for shape, names in shape_groups.items():
    print(f"  Shape {shape}: {', '.join(names)}")
```

### Overview of Featurizer Categories
MolBlender groups featurizers into logical categories:
-   **Small Molecule Categories (Examples):**
    -   `fingerprints`: Various 2D fingerprints (Morgan, RDKit topological, Atom Pairs, etc.).
    -   `fingerprints/cdk`: CDK-based fingerprints (often require Java).
    -   `fingerprints/datamol`: Fingerprints leveraging the Datamol library.
    -   `descriptors`: Collections of 0D/1D/2D/3D molecular descriptors (e.g., from RDKit, Mordred).
    -   `graph`: Representations of molecules as graphs (e.g., for Graph Neural Networks).
    -   `spatial` (or `matrix`, `spatial_learned`): Matrix-based (Adjacency, Coulomb) or learned 3D representations (UniMol).
    -   `deepchem`: Featurizers compatible with or wrapping DeepChem functionalities.
-   **Protein Categories (Examples):**
    -   `protein_language_models`: Embeddings from Protein Language Models (ESM-2, ProtT5, CARP, Ankh, PepBERT).
    -   *(Future categories might include structural protein features, etc.)*
  

---

## 2. Using Featurizers

### Getting a Specific Featurizer Instance

Once you know the registered name of the featurizer:

```python
import molblender as pm

# For a small molecule featurizer (e.g., MACCS keys)
maccs_featurizer = pm.get_featurizer('maccs_keys')
print(f"Instantiated: {maccs_featurizer}")
# Example Output: MACCSKeysFP(name='maccs_keys')

# For a protein featurizer (e.g., a small ESM2 model)
# Replace 'esm2_t6_8M_UR50D' with an actual registered key from list_available_protein_featurizers()
protein_plm_featurizer = pm.get_protein_featurizer('esm2_t6_8M_UR50D')
print(f"Instantiated: {protein_plm_featurizer}")
# Example Output: ESM2Featurizer(name='esm2_t6_8M_UR50D', model_name='esm2_t6_8M_UR50D', ...)
```
You can often override default parameters by passing them as keyword arguments to `get_featurizer` or `get_protein_featurizer`. For example:
`morgan_featurizer_custom = pm.get_featurizer('morgan_fp_r2_1024', nBits=512, radius=3)`

### Preparing Input Data

MolBlender featurizers are designed to accept common input formats directly.
-   **Small Molecules**: Input can typically be SMILES strings or pre-processed RDKit `Mol` objects. Utilities like {py:func}`molblender.data.io.mol_from_input` are used internally to standardize inputs (e.g., parse SMILES, sanitize RDKit Mols).
-   **Proteins**: Input is typically a protein sequence string. Future utilities like `molblender.data.protein.protein_from_input` will handle various sources like FASTA files, PDB IDs, etc., to yield a standardized sequence.

For most use cases, you can pass your raw data (like a SMILES string or protein sequence) directly to the `featurize` method.

### Generating Representations

The `featurize()` method (or calling the featurizer instance directly) computes the features.

```python
import molblender as pm

# Using a pre-instantiated featurizer (e.g., from the previous step)
# For this example, let's get a MACCS keys featurizer
maccs_featurizer = pm.get_featurizer('maccs_keys')

# Featurize a single small molecule (SMILES string)
smiles_ethanol = "CCO"
features_single = maccs_featurizer.featurize(smiles_ethanol) # Or maccs_featurizer(smiles_ethanol)
if features_single is not None:
    print(f"MACCS features for '{smiles_ethanol}' (shape: {features_single.shape})")
    # Expected Output: MACCS features for 'CCO' (shape: (167,))

# Featurize a list of small molecules
smiles_list = ["CCO", "CCC", "c1ccccc1"] # Ethanol, Propane, Benzene
features_batch = maccs_featurizer.featurize(smiles_list)
print(f"\nProcessed {len(features_batch)} molecules for MACCS keys:")
for i, fp_vector in enumerate(features_batch):
    if fp_vector is not None:
        print(f"  Molecule '{smiles_list[i]}': fingerprint shape {fp_vector.shape}")
    else:
        print(f"  Molecule '{smiles_list[i]}': featurization failed.")
```

---
## 3. Handling Different Input Types

MolBlender aims for flexibility in input.

### Small Molecule Inputs

Most small molecule featurizers accept:
-   SMILES strings: `"CCO"`
-   RDKit `Mol` objects: `Chem.MolFromSmiles("CCO")`
-   Some may also accept InChI strings if supported by internal conversion.

```python
from rdkit import Chem
import molblender as pm

# Assume 'featurizer' is a small molecule featurizer instance, e.g.,
# featurizer = pm.get_featurizer('rdkit_fp_2048') 
# For demonstration, we re-initialize it here if the previous block wasn't run
try:
    featurizer = pm.get_featurizer('rdkit_fp_2048')
except pm.RegistryError:
    print("Skipping 'Handling Different Input Types' example as 'rdkit_fp_2048' is not available.")
    featurizer = None

if featurizer:
    # From SMILES string
    features_smiles = featurizer("CCO")
    print(f"Features from SMILES: {features_smiles.shape if features_smiles is not None else 'Failed'}")

    # From RDKit molecule object
    mol = Chem.MolFromSmiles("CCC")
    if mol:
        features_mol = featurizer(mol)
        print(f"Features from RDKit Mol: {features_mol.shape if features_mol is not None else 'Failed'}")

    # Batch with mixed types (if underlying _prepare_input handles them consistently)
    # For simplicity, it's often best to provide a batch of consistent types (all SMILES or all Mols)
    # molecules_mixed_batch = ["CC(=O)O", Chem.MolFromSmiles("c1cnccc1")]
    # features_mixed_list = featurizer(molecules_mixed_batch)
    # print(f"Processed {len(features_mixed_list)} from mixed batch.")
```

### Protein Sequence Inputs

Protein featurizers (like PLMs) typically expect raw amino acid sequences:

```python
import molblender as pm

# Assume 'protein_featurizer' is an instance, e.g.,
# protein_featurizer = pm.get_protein_featurizer('esm2_t6_8M_UR50D')
# For demonstration, we re-initialize it here
try:
    protein_featurizer = pm.get_protein_featurizer('esm2_t6_8M_UR50D')
except (pm.RegistryError, pm.DependencyNotFoundError):
    print("Skipping protein input example as 'esm2_t6_8M_UR50D' is not available.")
    protein_featurizer = None


if protein_featurizer:
    # Single sequence
    sequence = "MKTAYIAKQRQISFVKSHFSRQ"
    embedding = protein_featurizer(sequence)
    if embedding is not None:
        print(f"Protein embedding shape: {embedding.shape}")
        # Example Output: Protein embedding shape: (320,) for esm2_t6_8M

    # Multiple sequences
    sequences_batch = ["MKTAYIAKQRQISFVKSHFSRQ", "VITAL", "LIVMATGSMQALPFDVQEWQLSGPRA"]
    embeddings_batch = protein_featurizer(sequences_batch)
    print(f"Processed {len(embeddings_batch)} protein sequences.")
```

### Using `mol_from_input` and `protein_from_input` Explicitly

For more complex scenarios or pre-processing, you might use MolBlender's data input functions directly before featurization.

```python
import molblender as pm
from molblender.data.io import mol_from_input # Assuming direct import path
# from molblender.data.protein import protein_from_input # Conceptual

# Small molecule from SMILES
ethanol_mol_obj = mol_from_input("CCO", input_type="smiles")
if ethanol_mol_obj:
    morgan_featurizer = pm.get_featurizer('morgan_fp_r2_1024')
    if morgan_featurizer:
        ethanol_fp = morgan_featurizer.featurize(ethanol_mol_obj)
        # print(f"Ethanol fingerprint shape: {ethanol_fp.shape if ethanol_fp is not None else 'Failed'}")

# Protein sequence (already a string, protein_from_input would standardize further)
# protein_seq = "MKTAYI"
# standardized_protein_seq = protein_from_input(protein_seq) # Example
# esm_featurizer = pm.get_protein_featurizer('esm2_t6_8M_UR50D') # Use an actual key
# if esm_featurizer and standardized_protein_seq:
#     protein_embedding = esm_featurizer.featurize(standardized_protein_seq)
#     # print(f"Protein ESM embedding shape: {protein_embedding.shape if protein_embedding is not None else 'Failed'}")
```

---
## 4. Parallel Processing and Batching

The `featurize()` method can process single items or a list of items. For multiple items, you can leverage parallel processing by setting the `n_workers` argument.

```python
import molblender as pm

# Example: Featurizing a larger list of SMILES using multiple workers
# featurizer = pm.get_featurizer('morgan_fp_r2_1024') # Ensure featurizer is defined
try:
    featurizer = pm.get_featurizer('morgan_fp_r2_1024')
except pm.RegistryError:
    print("Skipping parallel processing example as 'morgan_fp_r2_1024' is not available.")
    featurizer = None

if featurizer:
    large_smiles_list = ["CCO", "CCC", "c1ccccc1", "CC(=O)O", "CNC"] * 200  # Example: 1000 molecules
    
    # Process with 4 worker processes
    # The progress_bar=True option will show a TQDM progress bar if tqdm is installed.
    features_parallel = featurizer.featurize(large_smiles_list, n_workers=4, progress_bar=True)
    print(f"Processed {len(features_parallel)} molecules in parallel.")
    # `features_parallel` is a list of NumPy arrays or Nones

# Parallel processing for protein sequences (PLMs are often better with n_workers=1 or GPU)
# protein_featurizer = pm.get_protein_featurizer('esm2_t6_8M_UR50D') # Ensure featurizer is defined
# large_sequences_list = ["MKTAYIAKQRQISFVKSHFSRQ"] * 100
# embeddings_parallel = protein_featurizer.featurize(large_sequences_list, n_workers=1) # Or adjust based on model
# print(f"Processed {len(embeddings_parallel)} protein sequences.")
```

### Batch Processing Pipeline Example

For very large datasets that may not fit into memory all at once, or to manage resources, you can process data in smaller batches.

```python
import numpy as np
import molblender as pm

def process_dataset_in_batches(data_list, featurizer_name='morgan_fp_r2_1024', 
                               batch_size=500, n_workers=2):
    """Processes a list of inputs (e.g., SMILES) in batches."""
    print(f"Initializing featurizer: {featurizer_name}")
    # Get the featurizer instance
    featurizer = pm.get_featurizer(featurizer_name)
    if not featurizer: # Should not happen if name is valid and deps are met
        print(f"Could not get featurizer {featurizer_name}")
        return None, []

    all_features_collected = []
    valid_item_indices = []
    
    print(f"Processing {len(data_list)} items in batches of {batch_size} using {n_workers} worker(s).")
    for i in range(0, len(data_list), batch_size):
        current_batch = data_list[i : i + batch_size]
        print(f"  Processing batch {i // batch_size + 1} ({len(current_batch)} items)...")
        # Featurize the current batch
        batch_feature_results = featurizer.featurize(current_batch, n_workers=n_workers)
        
        # Collect valid features and their original indices
        for j, feat_vector in enumerate(batch_feature_results):
            if feat_vector is not None:
                all_features_collected.append(feat_vector)
                valid_item_indices.append(i + j) # Store the original index of the valid item
    
    if not all_features_collected:
        print("No valid features were generated.")
        return np.array([]), []

    # Stack all collected features into a single NumPy matrix
    # This assumes all valid feature vectors have the same length.
    try:
        feature_matrix = np.vstack(all_features_collected)
    except ValueError as e:
        print(f"Error stacking features: {e}. This might happen if feature vectors have inconsistent lengths.")
        # Fallback: return list of arrays if stacking fails
        return all_features_collected, valid_item_indices 
    
    print(f"Finished processing {len(data_list)} items.")
    print(f"Successfully featurized: {len(valid_item_indices)} items.")
    print(f"Final feature matrix shape: {feature_matrix.shape}")
    
    return feature_matrix, valid_item_indices

# Example usage:
# smiles_dataset = ["CCO", "CCC", "c1ccccc1", "INVALID_SMILES", "CC(=O)O"] * 500 # 2500 items
# feature_matrix_result, valid_idx = process_dataset_in_batches(smiles_dataset, batch_size=100)
```

---
## 5. Customizing Featurizers

You can customize featurizers by:
1.  Passing parameters to `get_featurizer()` or `get_protein_featurizer()`.
2.  Directly instantiating the featurizer class with desired parameters.

```python
import molblender as pm

# Option 1: Pass parameters via get_featurizer
# Override default radius and nBits for a Morgan fingerprint
custom_morgan1 = pm.get_featurizer(
    'morgan_fp_r2_1024', 
    radius=3, 
    nBits=512, 
    useChirality=True
)
# print(custom_morgan1) 
# Expected: MorganBitFP(name='morgan_fp_r2_1024', radius=3, nBits=512, useChirality=True, ...)

# Option 2: Direct instantiation from the class
from molblender.representations.fingerprints.rdkit import MorganBitFP # Example path

custom_morgan2 = MorganBitFP(radius=4, nBits=4096, useChirality=True)
# print(custom_morgan2)
# Expected: MorganBitFP(class=MorganBitFP, radius=4, nBits=4096, useChirality=True, ...)
```

---
## 6. Error Handling

MolBlender's featurization system is designed to handle common issues gracefully.

### Missing Dependencies
Featurizers are registered in MolBlender even if their underlying dependencies (like RDKit, Mordred, specific PLM libraries) are not installed. An error (typically `DependencyNotFoundError`) will only be raised when you try to *instantiate or use* such a featurizer.

```python
import molblender as pm

# Listing always works, showing all featurizers MolBlender *knows about*
all_sm_featurizers = pm.list_available_featurizers()
print(f"Total registered small molecule featurizers: {len(all_sm_featurizers)}")

# Attempting to get a featurizer whose dependencies are missing will fail
# For example, if RDKit is not installed:
# try:
#     morgan_featurizer = pm.get_featurizer('morgan_fp_r2_1024')
# except pm.DependencyNotFoundError as e:
#     print(f"Dependency error: {e}")
# Example Output if RDKit missing:
# Dependency error: RDKit is required for RDKit MorganBitFP. Install RDKit (e.g., pip install rdkit-pypi).
```

### Invalid Inputs
When featurizing a batch of inputs, if some inputs are invalid (e.g., unparsable SMILES strings, sequences with invalid characters for a PLM), the `featurize()` method (with default `strict=False`) will return `None` for those specific items, allowing the rest of the batch to be processed.

```python
import molblender as pm

# Assume 'morgan_featurizer' is already defined (e.g., from pm.get_featurizer('morgan_fp_r2_1024'))
# For this example, let's quickly get one if the previous block wasn't run:
try:
    morgan_featurizer = pm.get_featurizer('morgan_fp_r2_1024')
except pm.RegistryError:
    print("Skipping invalid input example as 'morgan_fp_r2_1024' is not available.")
    morgan_featurizer = None

if morgan_featurizer:
    smiles_with_invalid = ["CCO", "INVALID_SMILES_STRING", "CCC", "CC(C)(C)C(=O)O"]
    feature_results = morgan_featurizer.featurize(smiles_with_invalid)

    print("\\nResults for batch with invalid SMILES:")
    for i, (smi, feat) in enumerate(zip(smiles_with_invalid, feature_results)):
        if feat is None:
            print(f"  Failed to featurize: '{smi}' (result is None)")
        else:
            print(f"  Featurized '{smi}': shape {feat.shape}")
# Example Output:
# Results for batch with invalid SMILES:
#   Featurized 'CCO': shape (1024,)
#   Failed to featurize: 'INVALID_SMILES_STRING' (result is None)
#   Featurized 'CCC': shape (1024,)
#   Featurized 'CC(C)(C)C(=O)O': shape (1024,)
```
If you set `strict=True` when calling `featurize(..., strict=True)` for parallel processing (`n_workers > 1`), an *unhandled* exception in a worker process would cause the main `featurize` call to raise a {py:class}`~molblender.representations.utils.exceptions.FeaturizationError` (wrapping a `WorkerError`). However, `InvalidInputError` from SMILES parsing is typically handled within the worker by returning `None`, so `strict=True` will not cause `featurize()` to raise an error for this specific handled case.

---
## 7. Further Examples

### Fingerprint Comparison

```python
import molblender as pm

# Compare different types of fingerprints for the same molecule
# Ensure these featurizer keys are registered in your MolBlender setup
fingerprint_keys = ['morgan_fp_r2_1024', 'rdkit_fp_2048', 'maccs_keys']
example_smiles = "Cc1ccccc1O" # o-Cresol

print(f"\\nComparing fingerprints for: {example_smiles}")
for fp_key in fingerprint_keys:
    try:
        featurizer = pm.get_featurizer(fp_key)
        features = featurizer(example_smiles) # Using __call__ shortcut for featurize
        if features is not None:
            print(f"  {fp_key}: shape {features.shape}, sum of bits {features.sum()}")
        else:
            print(f"  {fp_key}: Failed to featurize.")
    except pm.RegistryError:
        print(f"  {fp_key}: Featurizer not available.")
# Example Output:
# Comparing fingerprints for: Cc1ccccc1O
#   morgan_fp_r2_1024: shape (1024,), sum of bits 17.0
#   rdkit_fp_2048: shape (2048,), sum of bits 124.0
#   maccs_keys: shape (167,), sum of bits 21.0
```

### Descriptor Calculation

```python
import molblender as pm

# Calculate RDKit descriptors
# Ensure 'rdkit_all_descriptors' is registered
try:
    rdkit_desc_calculator = pm.get_featurizer('rdkit_all_descriptors')
    ethanol_descriptors = rdkit_desc_calculator.featurize("CCO")
    if ethanol_descriptors is not None:
        print(f"\\nNumber of RDKit descriptors for Ethanol: {len(ethanol_descriptors)}")
        # Example Output: Number of RDKit descriptors for Ethanol: 210 (varies with RDKit version)
except pm.RegistryError:
    print("\\n'rdkit_all_descriptors' not available.")

# Calculate Mordred descriptors (2D only)
# Ensure 'mordred_descriptors_2d' is registered and Mordred is installed
try:
    mordred_desc_calculator = pm.get_featurizer('mordred_descriptors_2d')
    ethanol_mordred_desc = mordred_desc_calculator.featurize("CCO")
    if ethanol_mordred_desc is not None:
        print(f"Number of Mordred 2D descriptors for Ethanol: {len(ethanol_mordred_desc)}")
        # Example Output: Number of Mordred 2D descriptors for Ethanol: 1613 (varies with Mordred version)
except (pm.RegistryError, pm.DependencyNotFoundError):
    print("\\n'mordred_descriptors_2d' not available or Mordred library missing.")

```

---
## 8. Best Practices

1.  **Discover First**: Use `pm.list_available_featurizers()` or `pm.print_available_featurizers()` to see what's available and their registered names.
2.  **Check Return Values**: When featurizing batches, always check for `None` in the returned list, as individual items might fail.
3.  **Choose Appropriately**: Select featurizers based on your task:
    -   **Fingerprints** (e.g., Morgan, MACCS) are good for similarity searches, basic screening models.
    -   **Descriptors** (e.g., RDKit, Mordred) are often used for QSAR modeling.
    -   **Graph Representations** are for Graph Neural Networks (GNNs).
    -   **Spatial/3D Representations** (e.g., Coulomb Matrix, UniMol) are for models that can leverage 3D information.
    -   **Protein Language Model Embeddings** (e.g., ESM, ProtT5) for sequence-based protein tasks.
4.  **Parallel Processing**: For datasets with thousands of items, use the `n_workers` parameter in the `featurize()` method to speed up computation. Note that some featurizers (like GPU-based PLMs) may perform best with `n_workers=1`.
5.  **Caching Results**: For computationally expensive featurizations (e.g., some 3D descriptors, PLM embeddings for very long sequences), consider caching the generated features locally to avoid re-computation.

---
## 9. API Reference

### Core Functions for Featurizer Access:
-   {py:func}`molblender.representations.list_available_featurizers`
-   {py:func}`molblender.representations.get_featurizer`
-   {py:func}`molblender.representations.get_featurizer_info`
-   {py:func}`molblender.representations.list_available_protein_featurizers`
-   {py:func}`molblender.representations.get_protein_featurizer`
-   {py:func}`molblender.representations.get_protein_featurizer_info`
-   *(Utility for printing, if exposed at top level)* `{py:func}`molblender.print_available_featurizers`

### Base Classes:
-   {py:class}`molblender.representations.utils.base.BaseFeaturizer`
-   *(If defined and relevant)* `{py:class}`molblender.representations.protein.base.BaseProteinFeaturizer`

---
## 10. Related Links

-   {doc}`/usage/representations/fingerprints/index` (Overview of fingerprint types)
-   {doc}`/usage/representations/descriptors/index` (Overview of descriptor types)
-   {doc}`/usage/representations/spatial/index` (Overview of spatial/3D types)
-   {doc}`/usage/representations/protein/index` (Overview of protein representation types)
-   {doc}`/api/representations/index` (Root of the representations API documentation)



```{toctree}
:maxdepth: 1
:hidden:

```