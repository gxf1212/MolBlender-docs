# GRADE and X-GRADE

MolBlender exposes CDPKit's GRAIL descriptors as two protein-ligand complex
representations:

│Key│Dimensions│Ligand-only segment│Interaction segment│
│`grade_3d`│35│13│22│
│`xgrade_3d`│177│31│146│

Install the optional backend with `pip install molblender[grade]`. Both
representations are continuous geometric descriptors and can be screened with
ordinary vector models such as random forests, gradient boosting, SVMs and
linear models.

## Input Contract

GRADE requires a receptor PDB and a ligand pose with real 3D coordinates in
the same coordinate frame. MolBlender never generates a conformer or inserts a
zero vector for missing complex data. A failed receptor or pose is recorded as
a normal per-sample featurization failure.

For one receptor shared by all ligands:

```python
from molblender.representations import get_featurizer

grade = get_featurizer("grade_3d", protein_path="receptor.pdb")
features = grade.featurize(molecules)
```

For a receptor path stored per CSV row:

```python
from molblender.data import MolecularDataset, InputType

dataset = MolecularDataset.from_csv(
    "complexes.csv",
    input_column="ligand_sdf_path",
    mol_input_type=InputType.SDF_FILE,
    protein_pdb_column="receptor_pdb_path",
    label_columns=["affinity"],
    strict=True,
)
dataset.add_features("xgrade_3d", n_workers=4, on_error="raise")
```

Supplying both a fixed receptor and a different row-specific receptor is an
error. The feature cache includes ligand coordinates and receptor file bytes,
so replacing either structure invalidates the cached representation.

## Protocol

The adapter mirrors the official GRADE preprocessing: nonstandard receptor
residues are removed except single-atom metals, isolated standard fragments
are removed, the receptor environment is extracted within 21 angstrom of the
ligand, and CDPKit's `prepareForGRAILDescriptorCalculation` is applied before
descriptor calculation. Charge normalization is disabled by default, matching
the official command-line script, and can be enabled explicitly with
`normalize_charges=True`.

GRADE interaction terms are model descriptors, not physical binding energies.
In particular, electrostatic terms do not model a system-specific dielectric
environment.

## Sources And Licensing

The implementation calls the public CDPKit API and does not copy the upstream
batch script. GRADE's reference implementation and CDPKit carry LGPL terms;
CDPKit remains an optional runtime dependency rather than bundled code.

- GRADE repository: <https://github.com/molinfo-vienna/GRADE>
- CDPKit GRAIL API: <https://cdpkit.org/cdpl_api_doc/python_api_doc/classCDPL_1_1GRAIL_1_1GRAILDescriptorCalculator.html>
- X-GRADE API: <https://cdpkit.org/cdpl_api_doc/python_api_doc/classCDPL_1_1GRAIL_1_1GRAILXDescriptorCalculator.html>
