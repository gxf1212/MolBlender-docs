# Molecular Language Model API

API reference for SMILES language-model featurizers exposed through
`molblender.representations.sequential.language_model`.

## Overview

MolBlender currently exposes **featurizers** built on top of pretrained
molecular language models. This page documents the public featurizer surface,
not historical pretraining-only MLM helper classes.

## Core Base Class

### BaseMolecularLM

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.BaseMolecularLM
   :members:
   :no-index:
   :show-inheritance:
```

## BERT-Family Featurizers

### ChemBERTFeaturizer

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.ChemBERTFeaturizer
   :members:
   :no-index:
   :show-inheritance:
```

### ChemBERTaFeaturizer

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.ChemBERTaFeaturizer
   :members:
   :no-index:
   :show-inheritance:
```

### MolBERTFeaturizer

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.MolBERTFeaturizer
   :members:
   :no-index:
   :show-inheritance:
```

### SMILESBERTFeaturizer

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.SMILESBERTFeaturizer
   :members:
   :no-index:
   :show-inheritance:
```

## Transformer-Family Featurizers

### MolFormerFeaturizer

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.MolFormerFeaturizer
   :members:
   :no-index:
   :show-inheritance:
```

### XMOLFeaturizer

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.XMOLFeaturizer
   :members:
   :no-index:
   :show-inheritance:
```

### SELFormerFeaturizer

```{eval-rst}
.. autoclass:: molblender.representations.sequential.language_model.SELFormerFeaturizer
   :members:
   :no-index:
   :show-inheritance:
```

## Usage Examples

### Basic ChemBERTa Usage

```python
from molblender.representations.sequential.language_model import ChemBERTaFeaturizer

featurizer = ChemBERTaFeaturizer(
    model_name="seyonec/ChemBERTa-zinc-base-v1",
    pooling="mean",
)

embeddings = featurizer.featurize(["CCO", "c1ccccc1"])
```

### MolFormer Usage

```python
from molblender.representations.sequential.language_model import MolFormerFeaturizer

featurizer = MolFormerFeaturizer(
    model_name="ibm/MoLFormer-XL-both-10pct",
    pooling="cls",
)

embeddings = featurizer.featurize(["CC(=O)O", "CCN(CC)CC"])
```

## See Also

- {doc}`../../../../usage/representations/sequential/mlm` - Usage guide
- {doc}`../index` - Sequential representations index
