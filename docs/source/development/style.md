# Code Style Guidelines

MolBlender follows PEP 8 with additional conventions for scientific Python code.

## Python Style

### General

- **Line length**: 100 characters (slightly relaxed for scientific code)
- **Indentation**: 4 spaces (no tabs)
- **Encoding**: UTF-8
- **Imports**: Organized by standard → third-party → local

### Naming Conventions

```python
# Classes: PascalCase
class MorganFingerprint:
    pass

# Functions/methods: snake_case
def calculate_features():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_ATOMS = 100

# Private: leading underscore
def _internal_helper():
    pass
```

### Type Hints

**Always provide type hints** for:
- Function parameters
- Return values
- Class attributes

```python
from typing import List, Optional, Union
import numpy as np

def featurize(
    molecules: List[str],
    n_jobs: int = 1,
    cache_dir: Optional[str] = None
) -> np.ndarray:
    """Generate features."""
    ...
```

## Docstring Format

Use **Google style** docstrings:

```python
def complex_function(
    param1: str,
    param2: int = 10,
    flag: bool = False
) -> dict:
    """One-line summary.

    Detailed description if needed. Can span multiple
    lines and include references.

    Args:
        param1: Description of param1.
        param2: Description with default value.
        flag: Boolean flag description.

    Returns:
        Dictionary containing results with keys:
        - 'value': The computed value
        - 'status': Success status

    Raises:
        ValueError: If param1 is empty.
        RuntimeError: If computation fails.

    Example:
        >>> result = complex_function("test", param2=5)
        >>> result['value']
        42
    """
    ...
```

## Import Organization

```python
# Standard library
import os
import sys
from typing import List, Optional

# Third-party
import numpy as np
import pandas as pd
from rdkit import Chem

# Local imports
from molblender.config import dependencies as deps
from molblender.representations.utils.base import BaseFeaturizer
```

## Error Handling

Use specific exceptions:

```python
from molblender.representations.utils.exceptions import (
    FeaturizationError,
    InvalidInputError,
    DependencyNotFoundError
)

def process_molecule(mol: str) -> np.ndarray:
    """Process molecule."""
    if not mol:
        raise InvalidInputError("Empty molecule string")
    
    try:
        result = compute_features(mol)
    except Exception as e:
        raise FeaturizationError(f"Failed to featurize: {e}")
    
    return result
```

## Code Formatting

Use **Black** for automatic formatting:

```bash
# Format all code
black src/ tests/

# Check without modifying
black --check src/
```

## See Also

- {doc}`contributing` - Contribution workflow
- {doc}`testing` - Testing guidelines
