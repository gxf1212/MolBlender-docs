# Common Featurizer Shapes

## Small Molecule Fingerprints

### Standard Fingerprints
```python
# MACCS Keys (always 166 or 167 bits)
"maccs_keys": (167,)         # RDKit version
"cdk_maccs": (166,)          # CDK version
"datamol_maccs": (167,)      # Datamol version

# PubChem
"cdk_pubchem": (881,)        # PubChem CACTVS fingerprint

# EState
"cdk_estate": (79,)          # E-State atom types
"datamol_estate": (79,)      # E-State via datamol

# Klekota-Roth
"cdk_kr": (4860,)            # Large substructure fingerprint
"cdk_klekota_roth": (4860,)  # Alias

# ERG (Extended Reduced Graph)
"datamol_erg": (315,)
```

### Configurable Fingerprints
```python
# Morgan/ECFP (size specified in name)
"morgan_fp_r2_2048": (2048,)
"morgan_fp_r2_1024": (1024,)
"morgan_fp_r2_512": (512,)
"datamol_ecfp4": (2048,)     # Default
"datamol_ecfp4_1024": (1024,)

# RDKit Topological
"rdkit_fp_2048": (2048,)
"rdkit_fp_1024": (1024,)
"rdkit_fp_512": (512,)

# Atom Pair
"atom_pair_fp": (2048,)      # Default configurable

# Torsion
"torsion_fp": (2048,)        # Default configurable
```

### Dynamic/Variable Length
```python
# Count fingerprints
"morgan_count_fp_r2": "dynamic"
"morgan_feature_fp_r2": "dynamic"
"atom_pair_count_fp": "dynamic"
"torsion_count_fp": "dynamic"
"datamol_ecfp4_count": "dynamic"
"datamol_fcfp4_count": "dynamic"
"datamol_atompair_count": "dynamic"
```

## Molecular Descriptors

### RDKit Descriptors
```python
"rdkit_descriptors_all": (~200,)  # Exact count depends on RDKit version
# Can determine dynamically:
# from rdkit.Chem import Descriptors
# shape = (len(Descriptors.descList),)
```

### Mordred Descriptors
```python
"mordred_descriptors_all": (1826,)  # All 2D + 3D descriptors
"mordred_descriptors_2d": (1613,)   # Only 2D descriptors
"mordred_descriptors_custom": "dynamic"  # When subset selected
```

## Graph Representations
```python
# Most graph representations are dynamic
"deepchem_graphconv": "dynamic"
"deepchem_weave": "dynamic"
"deepchem_mol_graph_conv": "dynamic"
```

## Spatial/3D Representations

### Matrix Representations
```python
# Fixed size matrices (padded)
"coulomb_matrix": (100, 100)      # max_atoms × max_atoms
"coulomb_matrix_eig": (100,)      # Just eigenvalues
"adjacency_matrix": (100, 100)
"edge_matrix": (100, 100)
```

### Learned 3D Representations
```python
# UniMol variants
"unimol_cls_no_h": (512,)         # CLS token
"unimol_cls_all_h": (512,)
"unimol_cls_pocket": (512,)
"unimol_atomic_repr": "dynamic"    # Per-atom: (n_atoms, 512)
```

## Protein Representations

### Protein Language Models (PLMs)
```python
# ESM-2 Family
"esm2_t6_8M": (320,)
"esm2_t12_35M": (480,)
"esm2_t30_150M": (640,)
"esm2_t33_650M": (1280,)
"esm2_t36_3B": (2560,)
"esm2_t48_15B": (5120,)

# ESM-C
"esmc_300m": (960,)
"esmc_600m": (1152,)

# Other PLMs
"prot_t5_xl": (1024,)
"prot_t5_xxl": (1024,)
"ankh_base": (768,)
"ankh_large": (1536,)
"carp_38M": (512,)
"carp_76M": (512,)
"carp_640M": (1280,)
"pepbert": (768,)

# Note: These are CLS token shapes
# Per-residue shapes would be (sequence_length, embedding_dim)
```

### Sequence-Based Features
```python
"amino_acid_one_hot": (None, 20)  # Variable length × 20 amino acids
"amino_acid_properties": (None, N)  # N physicochemical properties
"position_specific_scoring_matrix": (None, 20)
```

## Image Representations
```python
# 2D molecular images
"mol_image_2d": (224, 224, 3)     # Height × Width × Channels
"mol_grid_image": (128, 128, 1)   # Grayscale

# 3D voxel grids
"mol_voxel_grid": (32, 32, 32, N)  # 3D grid with N channels
```

## Sequential/String Representations
```python
# SMILES-based
"smiles": "dynamic"               # Variable length string
"canonical_smiles": "dynamic"
"smiles_bert_embedding": (768,)   # From SMILES-BERT model

# InChI
"inchi": "dynamic"
"inchi_key": (27,)               # Fixed length InChI key
```

## Usage Example
```python
import molblender as pm

# Quick shape lookup
def print_shapes_for_category(category):
    featurizers = pm.select_featurizers_by_category(category)
    print(f"\nShapes for {category}:")
    for name in sorted(featurizers):
        info = pm.get_featurizer_info(name)
        if info:
            shape = info['shape']
            print(f"  {name}: {shape}")

# Check all MACCS variants
print_shapes_for_category("maccs")

# Find all 2048-dimensional representations
all_2048 = []
for name in pm.list_available_featurizers():
    shape = pm.get_featurizer_shape(name)
    if shape == (2048,):
        all_2048.append(name)
print(f"\nFound {len(all_2048)} featurizers with shape (2048,)")
```
