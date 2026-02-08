# Protein Language Models

```{toctree}
:maxdepth: 1
:hidden:
```

Protein Language Models (PLMs) are deep learning models trained on vast protein sequence databases to learn meaningful representations of proteins. MolBlender provides a unified interface to various state-of-the-art PLMs for efficient protein featurization.

## Quick Start

```python
import molblender as mbl

# List available protein featurizers
print(mbl.list_available_protein_featurizers())
# Output: ['ankh', 'ankh2', 'ankh3', 'carp', 'esm2', 'esm2_t12_35M', 'esm2_t30_150M', 
#          'esm2_t33_650M', 'esm2_t36_3B', 'esm2_t48_15B', 'esm2_t6_8M', 'esmc', 
#          'esmc_300m', 'esmc_600m', 'esmc_6b', 'pepbert', 'protT5', 'protT5_xl', 
#          'protT5_xxl']

# Get a protein featurizer
featurizer = mbl.get_protein_featurizer('esm2')

# Featurize a single sequence
sequence = "MKTAYIAKQRQISFVKSHFSRQ"
embedding = featurizer(sequence)
print(f"Embedding shape: {embedding.shape}")
# Output: Embedding shape: (1280,)

# Featurize multiple sequences efficiently
sequences = [
    "MKTAYIAKQRQISFVKSHFSRQ",
    "GSMQALPFDVQEWQLSGPRA",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIV"
]
embeddings = featurizer(sequences, n_workers=1)  # Use n_workers>1 for parallel processing
print(f"Processed {len(embeddings)} sequences")
# Output: Processed 3 sequences
```

## Supported Models

### ESM-2 Family
Facebook AI's ESM-2 models offer various size/performance trade-offs:

| Key | Model | Embedding Dim | Parameters | GPU Memory | Description |
|-----|-------|---------------|------------|------------|-------------|
| `esm2_t6_8M` | ESM-2 (8M) | 320 | 8M | ~2GB | Fastest, least accurate |
| `esm2_t12_35M` | ESM-2 (35M) | 480 | 35M | ~3GB | Very fast, good for screening |
| `esm2_t30_150M` | ESM-2 (150M) | 640 | 150M | ~4GB | Good balance |
| `esm2_t33_650M` | ESM-2 (650M) | 1280 | 650M | ~8GB | Standard choice |
| `esm2` | ESM-2 (650M) | 1280 | 650M | ~8GB | Alias for 650M |
| `esm2_t36_3B` | ESM-2 (3B) | 2560 | 3B | ~16GB | High accuracy |
| `esm2_t48_15B` | ESM-2 (15B) | 5120 | 15B | ~40GB | Best accuracy |

### ESM-C (Cambrian) Family
Next-generation ESM models with improved efficiency:

| Key | Model | Embedding Dim | Parameters | GPU Memory | Description |
|-----|-------|---------------|------------|------------|-------------|
| `esmc_300m` | ESM-C (300M) | 960 | 300M | ~4GB | Efficient small model |
| `esmc_600m` | ESM-C (600M) | 1152 | 600M | ~6GB | Default ESM-C |
| `esmc` | ESM-C (600M) | 1152 | 600M | ~6GB | Alias for 600M |
| `esmc_6b` | ESM-C (6B) | 2560 | 6B | ~24GB | High accuracy |

### ProtT5 Family
Character-level models that handle special amino acids:

| Key | Model | Embedding Dim | Parameters | GPU Memory | Description |
|-----|-------|---------------|------------|------------|-------------|
| `protT5` | ProtT5-XL-Half | 1024 | 3B | ~8GB | Half precision, default |
| `protT5_xl` | ProtT5-XL | 1024 | 3B | ~12GB | Full precision |
| `protT5_xxl` | ProtT5-XXL | 1024 | 11B | ~32GB | Largest ProtT5 |

### Other Models

| Key | Model | Embedding Dim | Parameters | Description |
|-----|-------|---------------|------------|-------------|
| `ankh` | Ankh-Large | 1536 | 1.5B | Ankh protein model |
| `ankh2` | Ankh2-Large | 1536 | 1.5B | Ankh2 variant |
| `ankh3` | Ankh3-Large | 1536 | 1.5B | Latest Ankh variant |
| `carp` | CARP (640M) | 1280 | 640M | Contrastive learning model |
| `pepbert` | PepBERT | 320 | ~110M | Specialized for peptides (<30 AA) |

## Installation

### Core Library
```bash
pip install molblender
```

### Model-Specific Dependencies
PLM featurizers require additional dependencies based on which models you use:

```bash
# For ESM-2 models
pip install fair-esm

# For ESM-C models
pip install esm

# For ProtT5 and Ankh models  
pip install transformers

# For CARP
pip install sequence-models

# For PepBERT
pip install tokenizers

# For GPU acceleration (recommended)
pip install torch  # See pytorch.org for GPU-specific installation
```

## Detailed Usage

### Initialization Options

```python
# Basic initialization
featurizer = mbl.get_protein_featurizer('esm2')

# With GPU acceleration
featurizer = mbl.get_protein_featurizer('esm2', device='cuda')

# With custom cache directory
featurizer = mbl.get_protein_featurizer(
    'protT5',
    cache_dir='/path/to/model/cache',
    device='cuda'
)

# With specific model parameters
featurizer = mbl.get_protein_featurizer(
    'esm2_t36_3B',
    repr_layer=30,  # Extract from specific layer (ESM models)
    max_length=1024,  # Maximum sequence length
    device='cuda:1'  # Specific GPU
)
```

### Working with Sequences

```python
# Single sequence
seq = "MKTAYIAKQRQISFVKSHFSRQ"
embedding = featurizer(seq)
print(f"Shape: {embedding.shape}, Type: {type(embedding)}")
# Output: Shape: (1280,), Type: <class 'numpy.ndarray'>

# Multiple sequences
sequences = ["MKTAY...", "GSMQA...", "KALTA..."]
embeddings = featurizer(sequences)
print(f"Embeddings: {len(embeddings)}, Each shape: {embeddings[0].shape}")

# With parallel processing
embeddings = featurizer(sequences, n_workers=4)

# Handle invalid sequences
sequences_with_invalid = [
    "MKTAYIAKQRQISFVKSHFSRQ",  # Valid
    "INVALID123",  # Contains numbers
    "",  # Empty
    "GSMQALPFDVQEWQLSGPRA"  # Valid
]
embeddings = featurizer(sequences_with_invalid)
# Invalid sequences return None
valid_embeddings = [e for e in embeddings if e is not None]
```

### Batch Processing Large Datasets

```python
import numpy as np
from typing import List, Optional

def process_fasta_file(fasta_path: str, 
                      model: str = 'esm2_t30_150M',
                      batch_size: int = 32,
                      output_path: Optional[str] = None) -> np.ndarray:
    """
    Process all sequences in a FASTA file.
    
    Args:
        fasta_path: Path to FASTA file
        model: PLM model to use
        batch_size: Sequences per batch
        output_path: Optional path to save embeddings
    
    Returns:
        Array of embeddings
    """
    from Bio import SeqIO
    
    # Initialize featurizer
    featurizer = mbl.get_protein_featurizer(model, device='cuda')
    
    # Read sequences
    sequences = []
    ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)
    
    # Process in batches
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        batch_embeddings = featurizer(batch)
        all_embeddings.extend(batch_embeddings)
    
    # Filter valid embeddings
    valid_embeddings = []
    valid_ids = []
    for emb, seq_id in zip(all_embeddings, ids):
        if emb is not None:
            valid_embeddings.append(emb)
            valid_ids.append(seq_id)
    
    # Convert to array
    embeddings_array = np.vstack(valid_embeddings)
    
    # Save if requested
    if output_path:
        np.savez(output_path, 
                 embeddings=embeddings_array,
                 ids=valid_ids,
                 model=model)
        print(f"Saved {len(valid_ids)} embeddings to {output_path}")
    
    return embeddings_array
```

### Direct Class Usage

For more control, use the featurizer classes directly:

```python
from molblender.representations.protein.sequence.plm import (
    ESM2Featurizer, ESMCFeaturizer, ProtT5Featurizer
)

# Custom ESM-2 instance
esm2 = ESM2Featurizer(
    model_name="esm2_t33_650M_UR50D",
    repr_layer=30,  # Extract from layer 30 instead of last
    device='cuda'
)

# Get model information
info = esm2.get_output_info()
print(f"Output type: {info[0]}, Shape: {info[1]}")
# Output: Output type: <class 'numpy.ndarray'>, Shape: (1280,)

# Process with custom preprocessing
class CustomESM2(ESM2Featurizer):
    def preprocess_sequence(self, sequence: str) -> str:
        # Custom preprocessing logic
        sequence = sequence.upper()
        # Replace selenocysteine with cysteine
        sequence = sequence.replace('U', 'C')
        return sequence

custom_esm = CustomESM2(device='cuda')
```

### Memory Management

```python
# For limited GPU memory, process sequences one at a time
def memory_efficient_processing(sequences: List[str], 
                               model_name: str = 'esm2_t36_3B') -> List[np.ndarray]:
    """Process sequences with minimal GPU memory usage."""
    featurizer = mbl.get_protein_featurizer(model_name, device='cuda')
    
    embeddings = []
    for seq in sequences:
        # Process one sequence
        emb = featurizer(seq)
        if emb is not None:
            # Move to CPU immediately
            embeddings.append(emb)
        
        # Optional: Clear GPU cache periodically
        if len(embeddings) % 100 == 0:
            import torch
            torch.cuda.empty_cache()
    
    return embeddings
```

### Model Selection Guide

```python
def select_model(sequence_length: int, 
                 accuracy_requirement: str = 'medium',
                 gpu_memory_gb: int = 8) -> str:
    """
    Recommend a model based on requirements.
    
    Args:
        sequence_length: Maximum sequence length
        accuracy_requirement: 'low', 'medium', or 'high'
        gpu_memory_gb: Available GPU memory
    
    Returns:
        Recommended model name
    """
    if accuracy_requirement == 'high':
        if gpu_memory_gb >= 40:
            return 'esm2_t48_15B'
        elif gpu_memory_gb >= 16:
            return 'esm2_t36_3B'
        else:
            return 'esm2_t33_650M'
    elif accuracy_requirement == 'low':
        if sequence_length < 30:
            return 'pepbert'  # Optimized for short peptides
        else:
            return 'esm2_t6_8M'
    else:  # medium
        if gpu_memory_gb >= 8:
            return 'esm2_t33_650M'
        else:
            return 'esm2_t30_150M'
```

## Advanced Features

### Extracting Different Representations

```python
# ESM models can extract from different layers
# Last layer (default) - most task-specific
last_layer = mbl.get_protein_featurizer('esm2', repr_layer=-1)

# Middle layer - more general features
middle_layer = mbl.get_protein_featurizer('esm2', repr_layer=17)

# Early layer - basic sequence patterns
early_layer = mbl.get_protein_featurizer('esm2', repr_layer=6)

# Compare representations
seq = "MKTAYIAKQRQISFVKSHFSRQ"
emb_last = last_layer(seq)
emb_middle = middle_layer(seq)
emb_early = early_layer(seq)

print(f"Cosine similarity (last vs middle): {np.dot(emb_last, emb_middle) / (np.linalg.norm(emb_last) * np.linalg.norm(emb_middle)):.3f}")
```

### Handling Special Cases

```python
# Sequences with non-standard amino acids
sequences_with_special = [
    "MKTAYUAKQRQISFVKSHFSRQ",  # Contains selenocysteine (U)
    "GSMQALPFDVQEWQLSGPRZA",   # Contains pyrrolysine (O) written as Z
    "KALTARQQEVFDLIRDHISQB"    # Contains ambiguous (B)
]

# ProtT5 handles these well
protT5 = mbl.get_protein_featurizer('protT5')
embeddings = protT5(sequences_with_special)

# ESM models convert special AAs to X
esm2 = mbl.get_protein_featurizer('esm2')
embeddings_esm = esm2(sequences_with_special)
```

### Integration with Machine Learning

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def train_protein_classifier(sequences: List[str], 
                           labels: List[int],
                           model_name: str = 'esm2_t30_150M'):
    """Train a classifier on protein embeddings."""
    # Get embeddings
    featurizer = mbl.get_protein_featurizer(model_name, device='cuda')
    embeddings = featurizer(sequences)
    
    # Remove failed sequences
    X, y = [], []
    for emb, label in zip(embeddings, labels):
        if emb is not None:
            X.append(emb)
            y.append(label)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)
    
    print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Train final model
    clf.fit(X, y)
    return clf, featurizer
```

### Caching Embeddings

```python
import pickle
from pathlib import Path

class CachedProteinFeaturizer:
    """Wrapper that caches embeddings to disk."""
    
    def __init__(self, model_name: str, cache_dir: str = './plm_cache'):
        self.featurizer = mbl.get_protein_featurizer(model_name, device='cuda')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.model_name = model_name
    
    def _get_cache_key(self, sequence: str) -> str:
        """Generate cache key for a sequence."""
        import hashlib
        seq_hash = hashlib.md5(sequence.encode()).hexdigest()
        return f"{self.model_name}_{seq_hash}.pkl"
    
    def featurize(self, sequences: List[str]) -> List[np.ndarray]:
        """Featurize sequences with caching."""
        embeddings = []
        uncached_sequences = []
        uncached_indices = []
        
        # Check cache
        for i, seq in enumerate(sequences):
            cache_file = self.cache_dir / self._get_cache_key(seq)
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    embeddings.append(pickle.load(f))
            else:
                embeddings.append(None)
                uncached_sequences.append(seq)
                uncached_indices.append(i)
        
        # Process uncached sequences
        if uncached_sequences:
            new_embeddings = self.featurizer(uncached_sequences)
            
            # Cache results
            for idx, emb in zip(uncached_indices, new_embeddings):
                if emb is not None:
                    cache_file = self.cache_dir / self._get_cache_key(sequences[idx])
                    with open(cache_file, 'wb') as f:
                        pickle.dump(emb, f)
                embeddings[idx] = emb
        
        return embeddings
```

## Performance Benchmarks

### Speed Comparison (sequences/second on NVIDIA A100)

| Model | Batch Size 1 | Batch Size 32 | Batch Size 128 |
|-------|--------------|---------------|----------------|
| ESM-2 (8M) | 50 | 1000 | 1500 |
| ESM-2 (150M) | 25 | 500 | 750 |
| ESM-2 (650M) | 10 | 200 | 300 |
| ESM-2 (3B) | 3 | 50 | 80 |
| ESM-C (600M) | 15 | 300 | 450 |
| ProtT5 | 8 | 100 | 150 |

### Memory Usage

```python
# Monitor GPU memory usage
import torch

def profile_memory_usage(model_name: str, sequence_lengths: List[int]):
    """Profile memory usage for different sequence lengths."""
    featurizer = mbl.get_protein_featurizer(model_name, device='cuda')
    
    for length in sequence_lengths:
        # Create test sequence
        seq = 'A' * length
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Measure memory before
        mem_before = torch.cuda.memory_allocated() / 1024**3  # GB
        
        # Process sequence
        _ = featurizer(seq)
        
        # Measure memory after
        mem_after = torch.cuda.memory_allocated() / 1024**3  # GB
        
        print(f"Length {length}: {mem_after - mem_before:.2f} GB")
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```python
   # Solution 1: Use smaller model
   featurizer = mbl.get_protein_featurizer('esm2_t12_35M')
   
   # Solution 2: Reduce batch size
   embeddings = featurizer(sequences, n_workers=1)  # Process one at a time
   
   # Solution 3: Use CPU (slower)
   featurizer = mbl.get_protein_featurizer('esm2', device='cpu')
   ```

2. **Dependency Not Found**
   ```python
   # Check which dependencies are missing
   try:
       featurizer = mbl.get_protein_featurizer('esm2')
   except DependencyNotFoundError as e:
       print(f"Missing dependency: {e}")
       # Install the required package
   ```

3. **Slow Processing**
   ```python
   # Use GPU acceleration
   featurizer = mbl.get_protein_featurizer('esm2', device='cuda')
   
   # Use smaller, faster model
   featurizer = mbl.get_protein_featurizer('esm2_t6_8M')
   
   # Enable caching for repeated sequences
   # See CachedProteinFeaturizer example above
   ```

## API Reference

{doc}`/api/representations/protein/sequence/plm`

## Related Links

- [ESM GitHub Repository](https://github.com/facebookresearch/esm)
- [ESM-C Models](https://github.com/evolutionaryscale/esm)
- [ProtT5 Paper & Models](https://github.com/agemagician/ProtTrans)
- [Ankh Models on Hugging Face](https://huggingface.co/ElnaggarLab)
- [CARP Paper](https://github.com/microsoft/protein-sequence-models)
- [PepBERT Repository](https://github.com/zhanglab-wbgcas/PepBERT)