# Sequential Representations

This guide provides an overview of sequential representations in MolBlender. Sequential representations treat molecules as strings or sequences of tokens, applying techniques from natural language processing to chemical structures.

## Overview

Sequential representations are based on the idea that molecules can be described as sequences (like SMILES strings) rather than as graphs or 3D structures. This perspective enables the application of powerful sequence-based machine learning models, including:

- Simple string-based representations (raw SMILES)
- Tokenization approaches (breaking SMILES into meaningful tokens)
- Pre-trained language models (learning chemical patterns from large datasets)

## Types of Sequential Representations

MolBlender provides three main types of sequential representations:

| Type | Description | Example |
|:-----|:------------|:--------|
| **String-based** | Raw SMILES or simplified string formats | `"CC(=O)OC1=CC=CC=C1C(=O)O"` |
| **Tokenized** | SMILES broken into meaningful tokens | `["C", "C", "(", "=", "O", ")"...]` |
| **Language Models** | Vector embeddings from pre-trained models | 768-dimensional vector |

## When to Use Sequential Representations

Sequential representations are particularly useful for:

1. **Text-based ML Models**: When using NLP or sequence-based models
2. **Transfer Learning**: Leveraging pre-trained knowledge from large datasets
3. **Simplicity**: When simpler representations are easier to work with than graphs
4. **Generative Tasks**: For SMILES or molecular structure generation
5. **Out-of-the-box Solutions**: Using powerful pre-trained models without fine-tuning

## Available Sequential Representations

### String-Based

Simple string processing for SMILES:

```python
from molblender.representations.utils.mol_strings import (
    standardize_smiles,
    canonical_smiles,
    randomize_smiles
)

# Standardize a SMILES string
standardized = standardize_smiles("C(C)=O")  # "CC=O"

# Get canonical SMILES
canonical = canonical_smiles("C(C)=O")  # "CC=O"

# Generate randomized SMILES for data augmentation
randomized = randomize_smiles("CC=O")  # "O=CC" or other valid variant
```

### Tokenization

Breaking SMILES into tokens:

```python
from molblender.representations import get_featurizer

# Basic SMILES tokenizer
tokenizer = get_featurizer("DeepChem-BasicSmilesTokenizer")
tokens = tokenizer.featurize("CC=O")  # ["C", "C", "=", "O"]

# Vocabulary-based tokenizer (returns token IDs)
vocab_tokenizer = get_featurizer("DeepChem-SmilesTokenizer", vocab_file="path/to/vocab.txt")
token_ids = vocab_tokenizer.featurize("CC=O")  # [12, 12, 22, 19]
```

### Language Models

SMILES language models for embedding generation:

```python
from molblender.representations import get_featurizer

# Load a ChemBERTa model
model = get_featurizer("smiles-lm-chemberta")

# Generate an embedding
embedding = model.featurize("CC=O")  # 768-dimensional vector
```

## Integration with Other Representations

Sequential representations can be combined with other types:

- **Fingerprints**: Use language model embeddings as input to fingerprint generation
- **Graph Representations**: Compare or ensemble with graph-based methods
- **3D Representations**: Complement with spatial information for full molecule description

## Submodules

MolBlender organizes sequential representations into several key submodules:

- **strings**: Basic SMILES string manipulation and normalization
- **tokenizer**: SMILES tokenization and vocabulary handling
- **language_model**: Pre-trained models for SMILES embedding

## See Also

- {doc}`/usage/representations/sequential/tokenizer` - SMILES tokenization
- {doc}`/usage/representations/sequential/language_model` - Language models for SMILES
- {doc}`/usage/representations/fingerprints/index` - Molecular fingerprints
- {doc}`/usage/representations/graph/index` - Graph-based representations

```{toctree}
:maxdepth: 1
:hidden:

tokenizer
language_model
mlm
```