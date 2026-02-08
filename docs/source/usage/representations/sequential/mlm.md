# Molecular Language Models

```{toctree}
:maxdepth: 1
:hidden:
```

Molecular Language Models (MLMs) are transformer-based models pre-trained on millions of chemical structures to learn meaningful molecular representations. MolBlender provides a unified interface to state-of-the-art MLMs for efficient molecular featurization.

## Quick Start

```python
import molblender as mbl
from molblender.data import Molecule

# List available molecular language models
mlm_featurizers = [f for f in mbl.list_available_featurizers() 
                   if 'language_model' in f]
print(mlm_featurizers)
# Output: ['chemberta', 'chemberta-zinc250k', 'chemberta-pubchem10m',
#          'smilesbert', 'smilesbert-bindingdb', 'chembert', 
#          'molbert', 'molformer', 'selformer']

# Get a molecular language model featurizer
featurizer = mbl.get_featurizer('chemberta')

# Featurize a single molecule (various input formats)
# From SMILES
embedding = featurizer("CCO")
print(f"Embedding shape: {embedding.shape}")
# Output: Embedding shape: (768,)

# From Molecule object
mol = Molecule.from_smiles("CCO")
embedding = featurizer(mol)

# From RDKit Mol
from rdkit import Chem
rdkit_mol = Chem.MolFromSmiles("CCO")
embedding = featurizer(rdkit_mol)

# Batch processing
smiles_list = ["CCO", "CC(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C"]
embeddings = featurizer(smiles_list, n_workers=4)
print(f"Processed {len(embeddings)} molecules")
# Output: Processed 3 molecules
```

## Available Models

````{card}
**Pre-trained Molecular Language Models**
^^^

| Key | Model | HF Model Name | Embedding Dim | Parameters | Description |
|-----|-------|---------------|---------------|------------|-------------|
| `chemberta` | ChemBERTa v1 | seyonec/ChemBERTa-zinc-base-v1 | 768 | 85M | Original ChemBERTa trained on 250K ZINC |
| `chemberta-zinc250k` | ChemBERTa v1 Large | seyonec/ChemBERTa-zinc250k-v1 | 768 | 85M | ChemBERTa on extended ZINC250K dataset |
| `chemberta-pubchem10m` | ChemBERTa PubChem | seyonec/PubChem10M_SMILES_BPE_450k | 768 | 85M | ChemBERTa trained on 10M PubChem molecules |
| `smilesbert` | SMILES-BERT | unikei/bert-base-smiles | 768 | 110M | BERT model trained on SMILES strings |
| `smilesbert-bindingdb` | SMILES-BERT BindingDB | JuIm/SMILES_BERT | 768 | 110M | SMILES-BERT on 50K BindingDB molecules |
| `chembert` | ChemBERT | jonghyunlee/ChemBERT_ChEMBL_pretrained | 256 | 85M | BERT trained on ChEMBL v33 database |
| `molbert` | MolBERT | BenevolentAI/MolBERT | 512 | 85M | Multi-task model (may require access) |
| `molformer` | MolFormer-XL | ibm/MoLFormer-XL-both-10pct | 768 | 47M | IBM's efficient transformer on 1.1B molecules |
| `selformer` | SELFormer | HUBioDataLab/SELFormer | 768 | 86M | Model with local attention mechanism |

````

:::{important}
**Security Notice**: Due to CVE-2025-32434, all models require PyTorch >= 2.6. Models will not load with older torch versions.
:::

## Installation

### Core Dependencies
```bash
# Basic installation
pip install molblender

# Required for molecular language models
pip install transformers torch>=2.6

# For GPU acceleration (recommended)
pip install torch>=2.6 --index-url https://download.pytorch.org/whl/cu118
```

## Detailed Usage

### Model Configuration

```python
# Basic usage with default settings
featurizer = mbl.get_featurizer('chemberta')

# Custom configuration
featurizer = mbl.get_featurizer(
    'chemberta',
    pooling='cls',        # Pooling strategy: 'mean', 'cls', 'max'
    max_length=256,       # Maximum sequence length
    device='cuda',        # Device: 'cuda', 'cpu', or specific GPU
    cache_dir='/path/to/cache'  # Custom model cache directory
)

# Check model info
output_info = featurizer.get_output_info()
print(f"Output type: {output_info[0]}, Shape: {output_info[1]}")
# Output: Output type: <class 'numpy.ndarray'>, Shape: (768,)
```

### Pooling Strategies

Different pooling strategies extract molecular representations differently:

```python
# Mean pooling (default) - average over all tokens
mean_featurizer = mbl.get_featurizer('chemberta', pooling='mean')

# CLS pooling - use first token representation
cls_featurizer = mbl.get_featurizer('chemberta', pooling='cls')

# Max pooling - maximum values across tokens
max_featurizer = mbl.get_featurizer('chemberta', pooling='max')

# Compare representations
smiles = "CC(C)CC1=CC=C(C=C1)C(C)C"  # Ibuprofen
mean_emb = mean_featurizer(smiles)
cls_emb = cls_featurizer(smiles)
max_emb = max_featurizer(smiles)

print(f"Mean embedding: {mean_emb[:5]}")
print(f"CLS embedding: {cls_emb[:5]}")
print(f"Max embedding: {max_emb[:5]}")
```

### Batch Processing

```python
# Process multiple molecules efficiently
molecules = [
    "CCO",                    # Ethanol
    "CC(=O)O",               # Acetic acid
    "CC(C)CC1=CC=C(C=C1)C(C)C",  # Ibuprofen
    "CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen isomer
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
]

# Serial processing
embeddings = featurizer(molecules)

# Parallel processing
embeddings = featurizer(molecules, n_workers=4)

# Handle failures gracefully
for i, (mol, emb) in enumerate(zip(molecules, embeddings)):
    if emb is not None:
        print(f"Molecule {i}: {emb.shape}")
    else:
        print(f"Molecule {i}: Failed to process")
```

### Working with Molecular Datasets

```python
from molblender.data import MolecularDataset

# Load dataset
dataset = MolecularDataset.from_csv(
    'molecules.csv',
    smiles_column='SMILES',
    label_columns=['activity']
)

# Featurize entire dataset
featurizer = mbl.get_featurizer('molformer')
X = dataset.featurize(featurizer, n_workers=8)
y = dataset.get_labels('activity')

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Use for machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"R² score: {score:.3f}")
```

### Model Comparison

```python
# Compare different models on the same molecules
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Test molecules
test_molecules = {
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "cyclohexane": "C1CCCCC1"
}

# Compare ChemBERTa variants
models = ['chemberta', 'chemberta-zinc250k', 'chemberta-pubchem10m']
embeddings = {}

for model_key in models:
    featurizer = mbl.get_featurizer(model_key)
    embeddings[model_key] = {}
    for name, smiles in test_molecules.items():
        embeddings[model_key][name] = featurizer(smiles)

# Calculate similarities
for model in models:
    print(f"\n{model} similarities:")
    emb = embeddings[model]
    
    # Aromatic similarity
    sim_aromatic = cosine_similarity(
        [emb["benzene"]], [emb["toluene"]]
    )[0, 0]
    print(f"  Benzene-Toluene: {sim_aromatic:.3f}")
    
    # Functional group effect
    sim_functional = cosine_similarity(
        [emb["benzene"]], [emb["phenol"]]
    )[0, 0]
    print(f"  Benzene-Phenol: {sim_functional:.3f}")
    
    # Different structure
    sim_different = cosine_similarity(
        [emb["benzene"]], [emb["cyclohexane"]]
    )[0, 0]
    print(f"  Benzene-Cyclohexane: {sim_different:.3f}")
```

### GPU Memory Management

```python
# For large batches, process in chunks to avoid OOM
def featurize_large_dataset(smiles_list, featurizer, chunk_size=100):
    all_embeddings = []
    
    for i in range(0, len(smiles_list), chunk_size):
        chunk = smiles_list[i:i + chunk_size]
        embeddings = featurizer(chunk)
        all_embeddings.extend(embeddings)
        
        # Clear GPU cache if using CUDA
        if featurizer.device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()
    
    return all_embeddings

# Example with 10,000 molecules
large_dataset = ["C" * (i % 50 + 1) for i in range(10000)]
featurizer = mbl.get_featurizer('molformer', device='cuda')
embeddings = featurize_large_dataset(large_dataset, featurizer)
print(f"Processed {len(embeddings)} molecules")
```

## Troubleshooting

:::{dropdown} **Model loading fails with torch.load error**
Ensure you have PyTorch >= 2.6:
```bash
pip install torch>=2.6
```
This is required due to security vulnerability CVE-2025-32434.
:::

:::{dropdown} **MolBERT access error**
The MolBERT model may require special access. If you encounter access errors, consider using ChemBERTa as an alternative:
```python
featurizer = mbl.get_featurizer('chemberta')  # Similar performance
```
:::

:::{dropdown} **Out of memory errors**
1. Reduce batch size
2. Use CPU instead of GPU
3. Process in chunks (see GPU Memory Management example)
4. Use a smaller model (e.g., MolFormer has only 47M parameters)
:::

:::{dropdown} **Slow inference speed**
1. Enable GPU acceleration: `device='cuda'`
2. Use batch processing instead of single molecules
3. Consider using MolFormer (optimized for efficiency)
4. Reduce max_length if molecules are short
:::

## API Reference

{doc}`/api/representations/sequential/language_model/mlm`

## Related Links

- [ChemBERTa GitHub](https://github.com/seyonechithrananda/bert-loves-chemistry)
- [SMILES-BERT Paper](https://dl.acm.org/doi/10.1145/3307339.3342186)
- [ChemBERT HuggingFace](https://huggingface.co/jonghyunlee/ChemBERT_ChEMBL_pretrained)
- [MolBERT Paper](https://arxiv.org/abs/2011.13230)
- [MolFormer GitHub](https://github.com/IBM/molformer)
- [SELFormer GitHub](https://github.com/HUBioDataLab/SELFormer)
- [HuggingFace Models Hub](https://huggingface.co/models)
- 