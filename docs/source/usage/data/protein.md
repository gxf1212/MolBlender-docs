# Protein Data Handling

Comprehensive protein data management with support for sequences, structures, and multiple input formats.

## Overview

The `Protein` class provides a unified interface for working with protein data in MolBlender. It handles:

- **Multiple input formats** (FASTA, PDB, mmCIF, sequences)
- **Database retrieval** (RCSB PDB, AlphaFold, UniProt)
- **Structure prediction and repair** (via ESMFold, PDBFixer)
- **BioPython integration** for structural analysis
- **Automatic caching** of downloaded and generated structures

```{admonition} Key Features
:class: tip

- Automatic format detection and conversion
- Intelligent caching to avoid redundant downloads
- Sequence validation with error handling
- Multi-chain structure support
- Integration with protein-ligand featurizers
```

## Quick Start

### Creating Protein Objects

```python
from molblender.data.protein import Protein

# From amino acid sequence
protein = Protein.from_sequence("MKTIIALSYIFCLVFA")

# From FASTA string
fasta_str = """>sp|P12345|EXAMPLE_HUMAN Example protein
MKTIIALSYIFCLVFAGDGT
AVEKTVAWAVEKLLKC"""
protein = Protein.from_fasta_string(fasta_str)

# From FASTA file
protein = Protein.from_fasta_file("protein.fasta")

# From PDB file
protein = Protein.from_pdb_file("1abc.pdb")

# From PDB ID (downloads automatically)
protein = Protein.from_pdb_id("1ABC")

# From AlphaFold database
protein = Protein.from_alphafold_id("P12345")

# From UniProt ID
protein = Protein.from_uniprot_id("P12345")
```

### Accessing Protein Data

```python
# Access sequence
print(f"Sequence: {protein.sequence}")
print(f"Length: {len(protein.sequence)} residues")

# Access structure (BioPython Structure object)
if protein.structure:
    print(f"Structure: {protein.structure}")
    print(f"Chains: {list(protein.structure.get_chains())}")

# Access metadata
print(f"Metadata: {protein.metadata}")
```

## Input Formats

### 1. Amino Acid Sequences

Directly provide protein sequences using single-letter amino acid codes.

```python
# Simple sequence
protein = Protein.from_sequence("MKTIIALSYIFCLVFA")

# With structure prediction (requires ESMFold)
protein = Protein.from_sequence(
    "MKTIIALSYIFCLVFA",
    predict_structure=True  # Generates 3D structure
)

# Check if structure is available
if protein.structure:
    print("3D structure available")
```

#### Sequence Validation

Sequences are automatically validated and cleaned:

```python
# Valid amino acids (standard 20 + some non-standard)
valid_sequence = "ACDEFGHIKLMNPQRSTVWY"  # ✅ Valid

# Invalid characters are replaced with 'X'
protein = Protein.from_sequence("MKTX12ALSYX")
# Becomes: "MKTXXALSYX" with warning
```

**Supported amino acids:**
- **Standard 20**: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- **Non-standard**: B (Asx), Z (Glx), X (Unknown), U (Selenocysteine), O (Pyrrolysine)
- **Simulation residues**: ACE, NME, NHE, etc.

### 2. FASTA Format

#### FASTA String

```python
fasta_string = """>sp|P12345|PROT_HUMAN Protein example
MKTIIALSYIFCLVFA
GDGTAVEKTVAWAVEK
LLKCAGFAAVGGF"""

protein = Protein.from_fasta_string(fasta_string)

# Metadata is automatically extracted
print(protein.metadata)
# {'seqrecord_id': 'sp|P12345|PROT_HUMAN',
#  'seqrecord_name': 'sp|P12345|PROT_HUMAN',
#  'seqrecord_description': 'Protein example',
#  'seqrecord_annotations': {...}}
```

#### FASTA File

```python
# Single-sequence FASTA
protein = Protein.from_fasta_file("protein.fasta")

# Multi-sequence FASTA (returns list)
from molblender.data.protein import proteins_from_fasta

proteins = proteins_from_fasta("multi_protein.fasta")
print(f"Loaded {len(proteins)} proteins")

for i, protein in enumerate(proteins):
    print(f"Protein {i+1}: {len(protein.sequence)} residues")
```

### 3. PDB Format

PDB (Protein Data Bank) files contain 3D structural information.

```python
# From local PDB file
protein = Protein.from_pdb_file("1abc.pdb")

# With structure repair (fixes missing atoms)
protein = Protein.from_pdb_file(
    "1abc.pdb",
    fix_structure=True  # Uses PDBFixer
)

# Extract sequence from structure
print(f"Sequence from structure: {protein.sequence}")

# Access chains
for chain in protein.structure.get_chains():
    print(f"Chain {chain.id}: {len(list(chain.get_residues()))} residues")
```

#### PDB File Features

- **Automatic sequence extraction** from structure
- **Multi-chain support** - handles complexes and assemblies
- **Structure repair** - fixes missing atoms, residues
- **Metadata extraction** - resolution, R-factor, etc.

### 4. mmCIF Format

Modern crystallographic information file format.

```python
# From CIF file
protein = Protein.from_cif_file("1abc.cif")

# With structure repair
protein = Protein.from_cif_file(
    "1abc.cif",
    fix_structure=True
)
```

CIF files provide the same functionality as PDB files but with better support for large structures.

## Database Retrieval

### RCSB Protein Data Bank

Download experimentally determined structures directly from PDB.

```python
# Download by PDB ID
protein = Protein.from_pdb_id("1ABC")

# Choose format (PDB or CIF)
protein = Protein.from_pdb_id("1ABC", file_format='cif')

# With structure repair
protein = Protein.from_pdb_id(
    "1ABC",
    fix_structure=True
)

# Files are automatically cached
# Second call loads from cache (much faster)
protein = Protein.from_pdb_id("1ABC")  # Instant!
```

#### Batch Download

```python
from molblender.data.protein import fetch_multiple_from_pdb

pdb_ids = ["1ABC", "2XYZ", "3DEF"]
proteins = fetch_multiple_from_pdb(pdb_ids)

# Results dictionary
for pdb_id, protein in proteins.items():
    if protein:
        print(f"{pdb_id}: {len(protein.sequence)} residues")
    else:
        print(f"{pdb_id}: Failed to download")
```

### AlphaFold Database

Access AI-predicted protein structures from AlphaFold.

```python
# Download AlphaFold prediction by UniProt ID
protein = Protein.from_alphafold_id("P12345")

# AlphaFold structures come with confidence scores
print(protein.metadata.get('source'))  # 'alphafold'

# Structures are cached automatically
cache_dir = protein.cache_dir
print(f"Cached in: {cache_dir}")
```

```{note}
AlphaFold provides predicted structures for most proteins in UniProt. Predictions include per-residue confidence scores (pLDDT).
```

### UniProt Database

Retrieve sequence information from UniProt.

```python
# Fetch from UniProt
protein = Protein.from_uniprot_id("P12345")

# UniProt metadata is automatically extracted
print(protein.metadata.get('uniprot_id'))
print(protein.metadata.get('organism'))
print(protein.metadata.get('function'))
```

## Universal Input Handler

Don't know the input type? Use `from_input()` for automatic detection.

```python
# Automatic format detection
protein = Protein.from_input("MKTIIALSYIFCLVFA")           # Detects: sequence
protein = Protein.from_input("protein.fasta")              # Detects: FASTA file
protein = Protein.from_input("1abc.pdb")                   # Detects: PDB file
protein = Protein.from_input("1ABC")                       # Detects: PDB ID
protein = Protein.from_input(">sp|P12345\nMKTII...")       # Detects: FASTA string

# Manual type specification (optional)
from molblender.data import InputType

protein = Protein.from_input(
    "1ABC",
    input_type=InputType.PDB_ID
)
```

## Structure Operations

### Structure Prediction

Generate 3D structures from sequences using ESMFold.

```python
# Create protein with sequence
protein = Protein.from_sequence("MKTIIALSYIFCLVFA")

# Predict 3D structure
protein.predict_structure_esmfold()

# Structure is now available
if protein.structure:
    print("Structure predicted successfully")
    print(f"Number of atoms: {len(list(protein.structure.get_atoms()))}")
```

```{warning}
Structure prediction requires ESMFold to be installed and configured. This is a compute-intensive operation that may require GPU resources.
```

### Structure Repair

Fix common issues in protein structures.

```python
# Load protein with issues
protein = Protein.from_pdb_file("broken_structure.pdb")

# Check for missing atoms
if protein.has_missing_atoms():
    print("Structure has missing atoms")

# Repair structure
protein.fix_structure()

# Save repaired structure
protein.write_structure("repaired.pdb")
```

**What gets fixed:**
- Missing heavy atoms
- Missing hydrogen atoms
- Non-standard residues
- Crystallographic waters (optional)

### Working with Structures

```python
# Access BioPython structure object
structure = protein.structure

# Iterate over chains
for chain in structure.get_chains():
    print(f"Chain {chain.id}")

    # Iterate over residues
    for residue in chain:
        print(f"  Residue: {residue.get_resname()}")

        # Iterate over atoms
        for atom in residue:
            print(f"    Atom: {atom.name} at {atom.coord}")

# Get specific chain
chain_A = protein.get_chain('A')

# Get chain sequences
sequences = protein.get_chain_sequences()
for chain_id, seq in sequences.items():
    print(f"Chain {chain_id}: {seq}")
```

## Caching System

All downloads and structure predictions are automatically cached.

### Default Cache Location

```python
# Default: .molblender_cache/proteins/ in current directory
protein = Protein.from_pdb_id("1ABC")
print(protein.cache_dir)
# Output: /path/to/cwd/.molblender_cache/proteins/
```

### Custom Cache Directory

```python
# Specify custom cache location
protein = Protein.from_pdb_id(
    "1ABC",
    cache_dir="/path/to/custom/cache"
)

# All Protein methods support cache_dir parameter
protein = Protein.from_sequence(
    "MKTIIALSYIFCLVFA",
    cache_dir="/shared/cache",
    predict_structure=True  # Predicted structure will be cached
)
```

### Cache Benefits

1. **Faster loading**: Cached files load instantly
2. **Reduced network traffic**: No redundant downloads
3. **Offline capability**: Work without internet after initial download
4. **Structure persistence**: Predicted structures are saved for reuse

## Integration with MolecularDataset

Protein data integrates seamlessly with molecular datasets for protein-ligand modeling.

### Loading Protein-Ligand Datasets

```python
from molblender.data import MolecularDataset

# Load dataset with protein information
dataset = MolecularDataset.from_csv(
    "protein_ligand_data.csv",
    input_column="ligand_smiles",
    label_columns=["binding_affinity"],
    protein_sequence_column="protein_sequence",  # Protein sequences
    protein_pdb_column="pdb_path"               # Optional: PDB files
)

# Access protein information
for i, mol in enumerate(dataset):
    protein_seq = mol.protein_sequence
    protein_pdb = mol.protein_pdb_path

    if protein_seq:
        print(f"Molecule {i}: Ligand with {len(protein_seq)}-residue protein")
```

### Protein-Ligand Featurizers

```python
# Add protein-ligand features
dataset.add_features([
    "topology_net_3d",      # 3D topological features
    "splif_enhanced",       # Enhanced SPLIF fingerprints
    "qsar_3d_fields",      # 3D QSAR field descriptors
])

# Protein information is automatically passed to featurizers
features = dataset.get_features(["topology_net_3d"])
```

## File I/O Operations

### Saving Protein Data

```python
# Save structure to file
protein.write_structure("output.pdb", format='pdb')
protein.write_structure("output.cif", format='cif')

# Export to FASTA
fasta_string = protein.to_fasta(
    header="my_protein",
    description="Custom protein sequence"
)
print(fasta_string)
# >my_protein Custom protein sequence
# MKTIIALSYIFCLVFA...

# Write FASTA to file
with open("protein.fasta", "w") as f:
    f.write(fasta_string)
```

### Loading from Various Sources

```python
# Auto-detect file format
protein = Protein.from_input("protein_file.xxx")

# Explicitly specify format
from molblender.data import InputType

protein = Protein.from_input(
    "ambiguous_file",
    input_type=InputType.PDB_FILE
)
```

## Advanced Usage

### Multi-Chain Proteins

```python
# Load multi-chain structure
protein = Protein.from_pdb_id("1ABC")

# Get all chain sequences
sequences = protein.get_chain_sequences()

for chain_id, sequence in sequences.items():
    print(f"Chain {chain_id}: {len(sequence)} residues")
    print(f"  Sequence: {sequence[:50]}...")

# Extract specific chain
chain_A = next(
    chain for chain in protein.structure.get_chains()
    if chain.id == 'A'
)
```

### Metadata Extraction

```python
# Get comprehensive metadata
metadata = protein.get_metadata()

print(metadata)
# {
#   'source': 'pdb',
#   'pdb_id': '1ABC',
#   'resolution': 2.1,
#   'r_factor': 0.185,
#   'organism': 'Homo sapiens',
#   'chains': ['A', 'B'],
#   'num_residues': 352,
#   ...
# }
```

### Custom Sequence Processing

```python
from molblender.data.protein import (
    STANDARD_AMINO_ACIDS,
    validate_sequence
)

# Check if sequence contains only standard amino acids
sequence = "MKTIIALSYIFCLVFA"
is_standard = all(aa in STANDARD_AMINO_ACIDS for aa in sequence)

# Validate and clean sequence
cleaned = validate_sequence("MKTX12@ALSY#IFCLVFA")
print(cleaned)  # "MKTXXALSYXIFCLVFA" (invalid chars → X)
```

## Error Handling

### Robust Loading

```python
# Handle download failures
try:
    protein = Protein.from_pdb_id("INVALID")
except Exception as e:
    print(f"Failed to load: {e}")
    # Fallback to sequence
    protein = Protein.from_sequence("MKTIIALSYIFCLVFA")

# Check structure availability
if protein.structure is None:
    print("No structure available")
    # Try to predict
    protein.predict_structure_esmfold()
```

### Validation

```python
# Sequence validation with detailed errors
from molblender.representations.utils.exceptions import InvalidInputError

try:
    protein = Protein.from_sequence("")  # Empty sequence
except InvalidInputError as e:
    print(f"Invalid input: {e}")

# Structure validation
if protein.structure and protein.has_missing_atoms():
    print("Warning: Structure has missing atoms")
    protein.fix_structure()
```

## Performance Tips

```{admonition} Best Practices
:class: tip

**Caching:**
- Always use a persistent cache directory for production
- Share cache across projects to avoid redundant downloads

**Batch Operations:**
- Use `fetch_multiple_from_pdb()` for batch downloads
- Use `proteins_from_fasta()` for multi-sequence files

**Memory Management:**
- Process large protein lists in chunks
- Clear unnecessary structure data after feature extraction

**Structure Prediction:**
- Only predict structures when absolutely necessary
- Cache predicted structures for reuse
- Consider using AlphaFold DB instead of on-the-fly prediction
```

## Examples

### Example 1: Protein-Ligand Dataset

```python
from molblender.data import MolecularDataset

# Load protein-ligand binding data
dataset = MolecularDataset.from_csv(
    "binding_data.csv",
    input_column="ligand_smiles",
    label_columns=["pIC50"],
    protein_sequence_column="target_sequence"
)

# Add protein-ligand features
dataset.add_features([
    "topology_net_3d",
    "ligand_residue_matrix"
])

# Train model (example)
from molblender.models import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="pIC50",
    modality_categories=["fingerprints/protein-ligand"]
)
```

### Example 2: Structure-Based Analysis

```python
# Download and analyze protein structure
protein = Protein.from_pdb_id("1ABC", fix_structure=True)

# Extract structural features
sequences = protein.get_chain_sequences()
metadata = protein.get_metadata()

# Export for external tools
protein.write_structure("cleaned_1ABC.pdb")
fasta = protein.to_fasta()

# Use in featurization
from molblender.representations.protein.structure import MaSIFFeaturizer

featurizer = MaSIFFeaturizer()
features = featurizer.featurize([protein])
```

### Example 3: High-Throughput Processing

```python
# Process multiple proteins efficiently
pdb_ids = ["1ABC", "2XYZ", "3DEF", "4GHI"]

proteins = fetch_multiple_from_pdb(
    pdb_ids,
    file_format='pdb',
    fix_structure=True,
    cache_dir="/shared/cache"
)

# Extract sequences for all proteins
sequences = {
    pdb_id: protein.sequence
    for pdb_id, protein in proteins.items()
    if protein is not None
}

print(f"Successfully processed {len(sequences)}/{len(pdb_ids)} proteins")
```

## API Reference

For detailed API documentation, see:
- {doc}`../../api/data/index` - Data module API reference
- {doc}`molecule` - Molecule class documentation
- {doc}`dataset` - Dataset management guide

## Related Topics

- {doc}`molecule` - Small molecule handling
- {doc}`dataset` - Dataset operations
- {doc}`../representations/protein/index` - Protein featurizers

## Summary

```{admonition} Key Takeaways
:class: tip

1. **Unified Interface**: Single `Protein` class handles all input formats
2. **Auto-Detection**: `from_input()` automatically detects format type
3. **Database Integration**: Direct download from PDB, AlphaFold, UniProt
4. **Smart Caching**: Automatic caching of downloads and predictions
5. **BioPython Compatible**: Full integration with BioPython ecosystem
6. **Validation**: Automatic sequence validation and structure repair
7. **ML Ready**: Seamless integration with MolBlender featurizers
```
