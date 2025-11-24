# Masked Language Model API

API reference for masked language modeling functionality with SMILES.

## Overview

This module provides masked language modeling capabilities for SMILES strings, enabling pre-training and fine-tuning of transformer models on molecular data.

## Classes

### ChemBERTFeaturizer

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.mlm.ChemBERTFeaturizer
   :members:
   :show-inheritance:
```

### ChemBERTaFeaturizer

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.mlm.ChemBERTaFeaturizer
   :members:
   :show-inheritance:
```

### BaseMolecularLM

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.mlm.BaseMolecularLM
   :members:
   :show-inheritance:
```

## Functions

### MLM Training Utilities

```{eval-rst}
.. automodule:: molblender.representations.sequential.language_model.mlm
   :members:
   :show-inheritance:
```

## Usage Examples

### Basic MLM Usage

```python
from molblender.representations.sequential.language_model import SmilesMLMFeaturizer

# Create MLM featurizer
mlm_model = SmilesMLMFeaturizer(
    model_name="chemberta-mlm",
    mask_probability=0.15
)

# Generate masked predictions
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
masked_predictions = mlm_model.predict_masked(smiles)
```

### Fine-tuning on Custom Data

```python
from molblender.representations.sequential.language_model import ChemBERTaMLM

# Initialize model for fine-tuning
model = ChemBERTaMLM(
    pretrained_model="chemberta-base",
    vocab_size=1000,
    max_length=512
)

# Fine-tune on custom SMILES dataset
training_smiles = ["CCO", "c1ccccc1", "CC(=O)O", ...]
model.fine_tune(training_smiles, epochs=10, batch_size=32)
```

### Custom MLM Configuration

```python
from molblender.representations.sequential.language_model import SMILESBERTMLMConfig

# Custom configuration
config = SMILESBERTMLMConfig(
    vocab_size=2000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    mask_token_id=103,
    pad_token_id=0
)

# Use config to initialize model
model = ChemBERTaMLM(config=config)
```

## Model Architecture

### Transformer Components

The MLM models use standard transformer architecture:

- **Embedding Layer**: Converts tokens to embeddings
- **Positional Encoding**: Adds positional information
- **Transformer Blocks**: Multi-head attention + feed-forward
- **MLM Head**: Predicts masked tokens

### Training Objectives

- **Masked Language Modeling**: Predict randomly masked tokens
- **Token Type Prediction**: Distinguish different token types
- **Position Prediction**: Learn positional relationships

## Configuration Parameters

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 1000 | Size of the vocabulary |
| `hidden_size` | int | 768 | Hidden dimension size |
| `num_hidden_layers` | int | 12 | Number of transformer layers |
| `num_attention_heads` | int | 12 | Number of attention heads |
| `intermediate_size` | int | 3072 | Feed-forward intermediate size |
| `max_position_embeddings` | int | 512 | Maximum sequence length |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask_probability` | float | 0.15 | Probability of masking tokens |
| `replace_probability` | float | 0.8 | Probability of replacing with [MASK] |
| `random_probability` | float | 0.1 | Probability of random replacement |
| `learning_rate` | float | 1e-4 | Learning rate for training |
| `batch_size` | int | 32 | Training batch size |

## Performance Considerations

### Memory Usage

- **Model Size**: ~110M parameters for base model
- **Training Memory**: 4-8GB GPU memory recommended
- **Inference**: 1-2GB for typical batch sizes

### Training Tips

1. **Data Preparation**: Clean and canonicalize SMILES
2. **Batch Size**: Start with smaller batches, increase gradually
3. **Learning Rate**: Use warmup scheduling
4. **Validation**: Monitor perplexity and reconstruction accuracy

## See Also

- {doc}`../../../../usage/representations/sequential/mlm` - Usage guide
- {doc}`../../../../usage/representations/sequential/language_model` - Language model usage