# Boltz-2 AI Structure Prediction

Extract geometric and structural embeddings from Boltz-2 AI-predicted protein-ligand complex structures for downstream machine learning tasks like binding affinity prediction.

## Introduction

Boltz-2 is a state-of-the-art AI model for protein-ligand structure prediction. This module enables extraction of three types of embeddings from Boltz-2 predictions: global geometric features, token-level atomic features, and pairwise distance matrices. The implementation uses intelligent caching and isolated conda environment execution to provide seamless integration with MolBlender's ML pipeline.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🤖 **AI Structure Prediction**
Leverage Boltz-2's transformer-based architecture for protein-ligand complex prediction
:::

:::{grid-item-card} 🎯 **Three Embedding Types**
Global (29-33 dim), Token (4-dim pooled), Pairwise (distance matrices)
:::

:::{grid-item-card} 💾 **Intelligent Caching**
MD5-based caching saves 3-5 minutes per prediction on GPU
:::

:::{grid-item-card} 🔒 **Environment Isolation**
Subprocess execution prevents dependency conflicts with main environment
:::

::::

## Installation

Boltz-2 must be installed in a separate conda environment to avoid dependency conflicts:

```bash
# Create dedicated Boltz-2 environment
conda create -n boltz2 python=3.11
conda activate boltz2
pip install boltz

# Verify installation
conda run -n boltz2 python -c "import boltz; print(boltz.__version__)"

# Install BioPython in main environment (for CIF parsing)
pip install biopython
```

## Quick Start

```python
import molblender as pm
from rdkit import Chem

# Create Boltz-2 embedder
embedder = pm.get_featurizer(
    'Boltz2Embedder',
    embedding_type='global',  # 'global', 'token', or 'pairwise'
    pooling_method='mean',    # For token-level: 'mean' or 'max'
    boltz2_conda_env='boltz2' # Your Boltz-2 environment name
)

# Single molecule with protein sequence
mol = Chem.MolFromSmiles("CCO")
protein_seq = "MKLVWGSNKKAAYDIL..."
embedding = embedder.featurize([mol], protein_sequence=protein_seq)[0]
print(f"Embedding shape: {embedding.shape}")  # (29,) for global

# From CSV dataset
dataset = pm.MolecularDataset.from_csv(
    "data.csv",
    input_column='smiles',
    protein_sequence_column='protein_seq'
)

# Add Boltz-2 features
dataset.add_features(
    featurizer_inputs=embedder,
    feature_names='boltz2_global'
)
```

## Embedding Types

:::{list-table} **Boltz-2 Embedding Options**
:header-rows: 1
:widths: 20 20 60

* - Type
  - Dimension
  - Description
* - `global`
  - 29-33
  - Whole-complex geometric features: COM, radius of gyration, protein-ligand contacts, atom composition, confidence scores
* - `token`
  - 4 (pooled)
  - Per-atom features (x, y, z, B-factor) with mean/max pooling across all atoms
* - `pairwise`
  - Variable
  - Upper triangle of pairwise distance matrix between all atoms
:::

### Global Features (29-33 dimensions)

The global embedding captures comprehensive geometric properties:

1. **Basic Structural (10 features)**:
   - Total atom count
   - Center of mass (x, y, z)
   - Radius of gyration
   - Bounding box dimensions (x, y, z)
   - Protein/ligand atom counts

2. **Protein-Ligand Contacts (4 features)**:
   - Contact counts at 3.5Å, 4.5Å, 6.0Å, 8.0Å distance thresholds

3. **Distance Statistics (4 features)**:
   - Minimum, mean, maximum distances
   - Ligand radius

4. **Confidence Scores (3 features)**:
   - Mean, standard deviation, minimum pLDDT (B-factor from Boltz-2)

5. **Atom Composition (8 features)**:
   - Fractional counts of C, N, O, S, P, F, Cl, Br atoms

6. **Ligand-Specific (0-4 features)**:
   - Ligand COM and radius (if ligand detected, adds 4 dimensions)

## Parameters

:::{list-table} **Boltz2Embedder Parameters**
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Default
  - Description
* - `embedding_type`
  - 'global'
  - Type of embedding: 'global', 'token', or 'pairwise'
* - `pooling_method`
  - 'mean'
  - Pooling for token-level embeddings: 'mean' or 'max'
* - `use_structure_cache`
  - True
  - Enable caching of predicted structures
* - `boltz2_conda_env`
  - 'boltz2'
  - Name of conda environment with Boltz-2 installed
* - `cache_dir`
  - None
  - Custom cache directory (default: ~/.cache/molblender/boltz2/structures/)
* - `timeout`
  - 600
  - Timeout in seconds for Boltz-2 prediction
:::

## Usage Examples

### Basic Embedding Extraction

```python
# Global geometric features
embedder_global = pm.get_featurizer('Boltz2Embedder', embedding_type='global')
mol = pm.mol_from_input("CC(=O)Nc1ccc(O)cc1")  # acetaminophen
protein_seq = "MKLVWGSNKKAAYDILKHQRHGYIEGKQTMEWVMSGNKNSRYNMTFEKHKAQEEARKLFNEIAQMKEERGITLADQDSRKLSDMGFGFDLRSSGEKLLADGLMRFFKTMSALKLKERIDAKLENSGFNISGHYQFLVKRVNESDPKIKDFDFVMKFSLEELGREIEEFVNKYTKLEFQRQEVDEIIKTADQVNQMHDFVQTYKNFKEFGVEYLPHGGFVTNIDDWIKKLNQMPQEIAVDMVPGKPMCVESFSDYPPLGRFAVRDMRQTVAVGVIKAVDKKAAGAGKRK"

embedding = embedder_global.featurize([mol], protein_sequence=protein_seq)[0]

print(f"Shape: {embedding.shape}")
print(f"Mean COM: {embedding[1:4].mean():.2f}")
print(f"Radius of gyration: {embedding[4]:.2f}")
print(f"Contacts at 3.5Å: {embedding[10]:.0f}")
```

### Token-Level Embeddings

```python
# Extract per-atom features with different pooling
embedder_token_mean = pm.get_featurizer(
    'Boltz2Embedder',
    embedding_type='token',
    pooling_method='mean'
)

embedder_token_max = pm.get_featurizer(
    'Boltz2Embedder',
    embedding_type='token',
    pooling_method='max'
)

# Mean pooling (average atomic features)
token_mean = embedder_token_mean.featurize([mol], protein_sequence=protein_seq)[0]
print(f"Mean pooled shape: {token_mean.shape}")  # (4,)

# Max pooling (maximum atomic features)
token_max = embedder_token_max.featurize([mol], protein_sequence=protein_seq)[0]
print(f"Max pooled shape: {token_max.shape}")  # (4,)
```

### Batch Processing with Caching

```python
# Process multiple ligands against same protein
ligands = [
    "CC(=O)Nc1ccc(O)cc1",        # acetaminophen
    "CC(=O)Oc1ccccc1C(=O)O",     # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine
]

protein_sequence = "MKLVWGSNKKAAYDIL..."

embedder = pm.get_featurizer('Boltz2Embedder', embedding_type='global')
mols = [pm.mol_from_input(smi) for smi in ligands]

# First call: Runs Boltz-2 predictions (~3-5 min each)
embeddings_first = [embedder.featurize([mol], protein_sequence=protein_sequence)[0]
                    for mol in mols]

# Second call: Uses cache (instant)
embeddings_second = [embedder.featurize([mol], protein_sequence=protein_sequence)[0]
                     for mol in mols]

# Verify cache worked
import numpy as np
for i, (emb1, emb2) in enumerate(zip(embeddings_first, embeddings_second)):
    print(f"Ligand {i}: Cache match = {np.allclose(emb1, emb2)}")
```

### Binding Affinity Prediction

```python
from molblender.data import MolecularDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load protein-ligand dataset with affinity values
dataset = pm.MolecularDataset.from_csv(
    "kinase_binding.csv",
    input_column='smiles',
    protein_sequence_column='protein_seq',
    target_column='binding_affinity'
)

# Generate Boltz-2 global embeddings
embedder = pm.get_featurizer('Boltz2Embedder', embedding_type='global')
dataset.add_features(
    featurizer_inputs=embedder,
    feature_names='boltz2_global'
)

# Prepare training data
X = np.array(dataset.features['boltz2_global'])
y = dataset.targets['binding_affinity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Boltz-2 Affinity Model:")
print(f"  R² Score: {r2:.3f}")
print(f"  RMSE: {rmse:.3f}")
```

### Feature Importance Analysis

```python
import matplotlib.pyplot as plt

# Train model and analyze features
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Define feature groups
feature_names = [
    'n_atoms', 'COM_x', 'COM_y', 'COM_z', 'radius_gyration',
    'bbox_x', 'bbox_y', 'bbox_z', 'n_protein', 'n_ligand',
    'contacts_3.5A', 'contacts_4.5A', 'contacts_6.0A', 'contacts_8.0A',
    'dist_min', 'dist_mean', 'dist_max', 'ligand_radius',
    'plddt_mean', 'plddt_std', 'plddt_min',
    'frac_C', 'frac_N', 'frac_O', 'frac_S',
    'frac_P', 'frac_F', 'frac_Cl', 'frac_Br'
]

# Plot top 10 features
top_indices = np.argsort(importances)[-10:]
plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[top_indices])
plt.yticks(range(10), [feature_names[i] for i in top_indices])
plt.xlabel('Feature Importance')
plt.title('Top 10 Boltz-2 Features for Binding Affinity')
plt.tight_layout()
plt.show()
```

## Advanced Usage

### Extract from Pre-Computed CIF Files

If you already have Boltz-2 CIF structure files, extract embeddings directly:

```python
from molblender.representations.AI_fold.boltz2.embedding_extractor import Boltz2EmbeddingExtractor
from pathlib import Path

extractor = Boltz2EmbeddingExtractor()

# Extract from existing CIF
cif_file = Path("predictions/complex_model_0.cif")
embedding = extractor.extract_embedding(
    cif_file=cif_file,
    embedding_type='global'
)

print(f"Extracted embedding: {embedding.shape}")
```

### Custom Cache Directory

```python
from pathlib import Path

# Use custom cache location
embedder = pm.get_featurizer(
    'Boltz2Embedder',
    embedding_type='global',
    cache_dir=Path("/scratch/boltz2_cache")
)

# Cache will be stored at /scratch/boltz2_cache/<md5_hash>.cif
```

### Force Re-prediction (Ignore Cache)

```python
# Access predictor and force re-prediction
embedder = pm.get_featurizer('Boltz2Embedder')

structure_file = embedder.predictor.predict_structure(
    ligand_smiles="CCO",
    protein_sequence="MKLVW...",
    force_repredict=True  # Ignore cache, run Boltz-2 again
)

print(f"New prediction: {structure_file}")
```

## Caching System

### How It Works

1. **Cache Key Generation**: MD5 hash of `(ligand_SMILES, protein_sequence)` tuple
2. **Cache Location**: `~/.cache/molblender/boltz2/structures/<hash>.cif`
3. **Cache Hit**: If CIF exists, skip 3-5 minute GPU prediction
4. **Cache Miss**: Run Boltz-2 prediction and save CIF for future use

### Benefits

- **Time Savings**: ~3-5 minutes per complex on GPU
- **Reproducibility**: Same inputs always return identical embeddings
- **Disk Efficiency**: CIF files are ~200KB each
- **Transparent**: Cache is automatically managed by the system

### Cache Management

```python
# Check cache status
embedder = pm.get_featurizer('Boltz2Embedder')
cache_dir = embedder.predictor.cache_dir
print(f"Cache directory: {cache_dir}")
print(f"Cached structures: {len(list(cache_dir.glob('*.cif')))}")

# Clear cache manually if needed
import shutil
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("Cache cleared")
```

## Performance Characteristics

### Computational Requirements

:::{list-table} **Resource Requirements**
:header-rows: 1
:widths: 30 35 35

* - Component
  - Without Cache
  - With Cache Hit
* - GPU Required
  - Yes (NVIDIA)
  - No
* - Time per Complex
  - 3-5 minutes
  - <1 second
* - Network Required
  - Yes (MSA server)
  - No
* - Memory Usage
  - ~4 GB GPU
  - ~100 MB CPU
:::

### Processing Benchmarks

:::{list-table} **Processing Times**
:header-rows: 1
:widths: 30 35 35

* - Dataset Size
  - First Run (no cache)
  - Second Run (cached)
* - 10 complexes
  - 30-50 minutes
  - 5-10 seconds
* - 100 complexes
  - 5-8 hours
  - 1-2 minutes
* - 1000 complexes
  - 50-80 hours
  - 10-20 minutes
:::

```{tip}
For large-scale screening, run Boltz-2 predictions as a separate batch job first to populate the cache, then extract embeddings quickly.
```

## Technical Details

### YAML Input Format

Boltz-2 requires YAML input with short IDs (to avoid truncation):

```yaml
version: 1
sequences:
- protein:
    id: A  # Short ID required (not "protein")
    sequence: MKLVWGSNKKAAYDIL...
- ligand:
    id: L  # Short ID required (not "ligand")
    smiles: CCO
```

### Subprocess Execution

Boltz-2 runs via isolated conda environment:

```python
cmd = [
    "conda", "run", "-n", "boltz2",
    "boltz", "predict", "input.yaml",
    "--out_dir", "output/",
    "--use_msa_server"  # Required for MSA generation
]
```

### Output Path Detection

Boltz-2 outputs to: `output_dir/boltz_results_<hash>/predictions/<id>/<id>_model_0.cif`

The module uses recursive search to locate CIF files automatically.

### Ligand Detection Heuristic

Atoms are classified as protein vs ligand using BioPython's residue detection:

```python
for atom in all_atoms:
    residue = atom.get_parent()
    if residue.id[0] == ' ':  # Standard residue
        protein_atoms.append(atom)
    else:  # Heteroatom (ligand, water)
        ligand_atoms.append(atom)
```

## Limitations

- **GPU Requirement**: Structure prediction requires NVIDIA GPU (embedding extraction is CPU-only)
- **Internet Connection**: MSA server requires network access for protein sequence alignment
- **Prediction Reliability**: Subprocess-based prediction may fail silently; recommend using pre-computed CIF files for production
- **Ligand Detection**: Relies on heteroatom flags; may misclassify unusual molecules

## Troubleshooting

### "No module named 'boltz'"

```bash
# Ensure Boltz-2 is installed in correct environment
conda activate boltz2
pip install boltz

# Verify
python -c "import boltz; print(boltz.__version__)"
```

### "Boltz-2 did not generate any CIF output"

Check these common issues:

1. **GPU Availability**:
   ```bash
   nvidia-smi  # Verify GPU is accessible
   ```

2. **MSA Server Access**:
   ```bash
   curl -I https://api.colabfold.com  # Check connectivity
   ```

3. **Boltz-2 Logs**:
   Inspect stdout/stderr for prediction failure messages

### "BioPython is required"

```bash
# Install BioPython in main environment (not boltz2 env)
pip install biopython
```

### Silent Prediction Failures

If predictions fail without errors, use pre-computed CIF files:

```python
# Extract from existing CIF instead of predicting
from molblender.representations.AI_fold.boltz2.embedding_extractor import Boltz2EmbeddingExtractor

extractor = Boltz2EmbeddingExtractor()
embedding = extractor.extract_embedding("my_structure.cif", "global")
```

## References

- [Boltz-2 Paper](https://www.biorxiv.org/content/10.1101/2024.11.19.624167v1)
- [Boltz-2 GitHub](https://github.com/jwohlwend/boltz)
- [BioPython Documentation](https://biopython.org/)
- [Protein-Ligand Interaction Fingerprints Review](https://doi.org/10.1002/cmdc.202000582)

```{toctree}
:maxdepth: 1
:hidden:
```
