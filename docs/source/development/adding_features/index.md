# Adding Features

Guides for extending MolBlender with new capabilities.

## Quick Navigation

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🧪 **New Featurizers**
:link: featurizers
:link-type: doc
Add molecular representations
:::

:::{grid-item-card} 🤖 **New Models**
:link: models
:link-type: doc
Add ML algorithms
:::

:::{grid-item-card} 📊 **New Modalities**
:link: modalities
:link-type: doc
Add data types
:::

::::

## Overview

MolBlender's modular architecture makes it easy to extend with new:
- **Featurizers**: Molecular representation methods
- **Models**: Machine learning algorithms
- **Modalities**: Data format types

Each extension type has specific requirements and patterns documented in the guides below.

```{toctree}
:maxdepth: 2
:hidden:

featurizers
models
modalities
```

## Before You Start

1. Review {doc}`../architecture` to understand the system design
2. Check {doc}`../style` for code conventions
3. Set up your {doc}`../setup` development environment
4. Read {doc}`../testing` for test requirements

## General Principles

### Registration Pattern

Most MolBlender features use the registry pattern:

```python
@register_featurizer("my_feature", category="fingerprints", shape=(1024,))
class MyFeaturizer(BaseFeaturizer):
    ...
```

### Dependency Management

Use the unified dependency system:

```python
from molblender.config import dependencies as deps

def _featurize(self, mol):
    rdkit = deps.get_rdkit()  # Raises clear error if missing
    return rdkit['AllChem'].GetMorganFingerprintAsBitVect(mol, 2)
```

### Error Handling

Use custom exceptions:

```python
from molblender.representations.utils.exceptions import (
    FeaturizationError,
    InvalidInputError
)

if not valid:
    raise InvalidInputError("Invalid molecule")
```

## See Also

- {doc}`../architecture` - System design
- {doc}`../contributing` - Contribution workflow
- {doc}`../testing` - Test guidelines
