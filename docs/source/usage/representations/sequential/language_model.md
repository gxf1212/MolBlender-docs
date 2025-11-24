# Language Models for SMILES

This guide covers using pre-trained language models to generate embeddings from SMILES strings in MolBlender.

## Overview

Language models trained on SMILES strings can capture chemical patterns and generate meaningful molecular embeddings. These models treat SMILES as sequences of tokens, similar to how natural language processing models handle text.

## Available Models

### ChemBERTa

ChemBERTa is a BERT-like model pre-trained on large collections of SMILES strings:

```python
from molblender.representations import get_featurizer

# Load ChemBERTa model
model = get_featurizer("smiles-lm-chemberta")

# Generate embedding for a molecule
embedding = model.featurize("CC=O")  # Returns 768-dimensional vector

# Batch processing
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
embeddings = model.featurize(smiles_list)
```

### Model Variants

Different model sizes and training variants are available:

```python
# Different model sizes
chemberta_base = get_featurizer("smiles-lm-chemberta-base")
chemberta_large = get_featurizer("smiles-lm-chemberta-large")

# Fine-tuned variants
chemberta_finetuned = get_featurizer("smiles-lm-chemberta-finetuned")
```

## Usage Patterns

### Single Molecule Embedding

```python
from molblender.representations import get_featurizer

model = get_featurizer("smiles-lm-chemberta")

# Simple molecule
smiles = "CCO"  # Ethanol
embedding = model.featurize(smiles)

print(f"Embedding shape: {embedding.shape}")  # (768,)
print(f"Embedding type: {type(embedding)}")   # numpy.ndarray
```

### Batch Processing

```python
# Process multiple molecules efficiently
molecules = [
    "CCO",           # Ethanol
    "c1ccccc1",      # Benzene
    "CC(=O)O",       # Acetic acid
    "CC(C)C",        # Isobutane
]

embeddings = model.featurize(molecules, n_workers=4)

# Result is list of arrays
for i, emb in enumerate(embeddings):
    print(f"Molecule {i}: shape {emb.shape}")
```

### Integration with Datasets

```python
from molblender.data import MolecularDataset

# Load dataset
dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    label_columns=["activity"]
)

# Add language model features
dataset.add_features("smiles-lm-chemberta", n_workers=4)

# Use embeddings for ML
features = dataset.features["smiles-lm-chemberta"]
labels = dataset.labels["activity"]

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(features.tolist(), labels)
```

## Model Configuration

### Custom Parameters

Some models support custom configuration:

```python
# Model with custom parameters
model = get_featurizer(
    "smiles-lm-chemberta",
    max_length=512,        # Maximum SMILES length
    pooling_strategy="cls", # How to pool token embeddings
    normalize=True         # Normalize output embeddings
)
```

### Memory and Performance

Language models can be memory-intensive:

```python
# For large batches, process in chunks
def process_large_batch(smiles_list, batch_size=100):
    results = []
    model = get_featurizer("smiles-lm-chemberta")
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        batch_embeddings = model.featurize(batch)
        results.extend(batch_embeddings)
    
    return results

# Process 10,000 molecules in batches
large_smiles_list = ["CCO"] * 10000  # Example
embeddings = process_large_batch(large_smiles_list)
```

## Applications

### Similarity Search

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Generate embeddings for query and database
query_embedding = model.featurize("CCO")
database_embeddings = model.featurize([
    "CCC", "CCCO", "c1ccccc1", "CC(=O)O"
])

# Calculate similarities
similarities = cosine_similarity(
    query_embedding.reshape(1, -1),
    database_embeddings
)[0]

# Find most similar molecules
most_similar_idx = np.argmax(similarities)
print(f"Most similar molecule index: {most_similar_idx}")
print(f"Similarity score: {similarities[most_similar_idx]:.3f}")
```

### Clustering

```python
from sklearn.cluster import KMeans

# Generate embeddings for molecule library
molecules = ["CCO", "CCC", "CCCO", "c1ccccc1", "CC(=O)O", "c1ccc(O)cc1"]
embeddings = model.featurize(molecules)

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Group molecules by cluster
for cluster_id in range(3):
    cluster_molecules = [mol for i, mol in enumerate(molecules) if clusters[i] == cluster_id]
    print(f"Cluster {cluster_id}: {cluster_molecules}")
```

### Feature Engineering

```python
# Combine with other representations
fingerprint_model = get_featurizer("morgan_fp_r2_1024")
language_model = get_featurizer("smiles-lm-chemberta")

def combined_features(smiles):
    # Get both fingerprint and language model features
    fp = fingerprint_model.featurize(smiles)
    lm = language_model.featurize(smiles)
    
    # Concatenate features
    return np.concatenate([fp, lm])

# Use combined features
combined = combined_features("CCO")
print(f"Combined feature shape: {combined.shape}")  # (1024 + 768,)
```

## Model Details

### ChemBERTa Architecture

- **Base Model**: BERT architecture adapted for SMILES
- **Training Data**: Large corpus of SMILES strings
- **Tokenization**: Character-level or subword tokenization
- **Output**: 768-dimensional embeddings (base model)

### Supported Input Formats

- SMILES strings
- Canonical SMILES
- Isomeric SMILES

### Performance Considerations

- **Memory Usage**: ~1-4GB for model weights
- **Processing Speed**: ~10-100 molecules/second depending on length
- **Batch Size**: Larger batches more efficient but require more memory

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use model checkpointing
2. **Invalid SMILES**: Model may fail on malformed SMILES strings
3. **Long Sequences**: Very long SMILES may be truncated

### Error Handling

```python
def robust_featurization(smiles_list):
    results = []
    failed_indices = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            embedding = model.featurize(smiles)
            results.append(embedding)
        except Exception as e:
            print(f"Failed to process SMILES {i}: {smiles}")
            print(f"Error: {e}")
            results.append(None)
            failed_indices.append(i)
    
    return results, failed_indices
```

## See Also

- {doc}`tokenizer` - SMILES tokenization
- {doc}`mlm` - Masked language modeling
- {doc}`../fingerprints/index` - Traditional fingerprints for comparison
- {doc}`../../basic/featurizers` - General featurizer usage