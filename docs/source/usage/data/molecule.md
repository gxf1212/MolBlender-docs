# Molecule

```{toctree}
:maxdepth: 1
:hidden:
```

## Overview

The `Molecule` class is a core abstraction in MolBlender for handling chemical entities in a flexible and representation-agnostic way. It wraps various input types (SMILES, RDKit Mol objects, coordinate files, etc.) and provides access to commonly used molecular representations, with caching and error tracking.

This class is designed for use in molecular datasets and feature pipelines, ensuring reproducibility and clarity when managing molecular data.

## Main Features

* Supports multiple input types: SMILES strings, RDKit Mol objects, and file paths (e.g., SDF, PDB).
* Converts input into standardized formats on demand.
* Caches computed representations: RDKit Mol, SMILES, 3D coordinates, PDB block.
* Tracks generation errors and allows selective regeneration.
* Includes user-specified metadata and file origins.

## Initialization

```python
mol = Molecule(input_data, input_type=None, mol_id=None, properties=None, source_path=None, **kwargs)
```

* `input_data`: Original molecule input (string, object, or file content).
* `input_type`: Enum hint from `InputType` (e.g., `SMILES`, `SDF_FILE`).
* `mol_id`: Optional unique identifier (auto-generated UUID if not provided).
* `properties`: Dictionary of labels, targets, metadata.
* `source_path`: File path from which `input_data` was loaded.
* `kwargs`: Additional arguments passed to internal parsers.

## Accessing Representations

### RDKit Mol

```python
mol.get_rdkit_mol(force_reload=False, **parser_kwargs)
```

Returns a [`rdkit.Chem.Mol`](https://www.rdkit.org/docs/api/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol) object parsed from the input. Uses internal `mol_from_input()` function. You can control parsing behavior via `parser_kwargs` such as:

* `sanitize=True`
* `addHs=True`
* `embed3d=True`

### SMILES

```python
mol.get_smiles(force_reload=False)
```

Returns a canonical SMILES string. If the original input is SMILES, it returns it directly; otherwise, it attempts conversion from RDKit Mol.

### Coordinates

```python
mol.get_coordinates(force_reload=False)
```

Returns a NumPy array of 3D coordinates, extracted from the first conformer in the RDKit Mol object. Shape: `(n_atoms, 3)`.

### PDB Block

```python
mol.get_pdb_block(force_reload=False)
```

Returns a string containing the molecule in PDB format, generated via RDKit. This is useful for visual inspection or exporting to downstream tools.

## Error Handling

Each representation method caches errors encountered during generation. These errors can be inspected or cleared by reloading.

If a representation fails to generate (e.g., due to a malformed SMILES or file), `None` is returned, and the error is cached for later inspection.

## Example

```python
from molblender.data import Molecule
from molblender.data.io import InputType

mol = Molecule("CCO", input_type=InputType.SMILES)
print(mol.get_smiles())        # -> 'CCO'
print(mol.get_rdkit_mol())     # -> <rdkit.Chem.Mol object>
print(mol.get_coordinates())   # -> numpy array or None
```

## API References

* Input parsing: {doc}`/api/data/index`
* RDKit feature extraction: {doc}`/api/representations/fingerprints/rdkit`
* DeepChem and others: {doc}`/api/representations/fingerprints/deepchem`

---

This class plays a foundational role in building chemically aware applications using MolBlender. Downstream featurizers and dataset containers depend on its unified interface.
