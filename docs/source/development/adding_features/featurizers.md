# Adding New Featurizers

Guide for adding new molecular representation methods to MolBlender.

(Content to be expanded - see development/adding_features/index for overview)

## Quick Template

```python
from molblender.representations.utils.base import BaseFeaturizer
from molblender.representations.utils.registry import register_featurizer
from molblender.data.io import InputType
import numpy as np

@register_featurizer(
    "my_featurizer",
    category="fingerprints",
    shape=(1024,),
    description="My custom featurizer"
)
class MyFeaturizer(BaseFeaturizer):
    EXPECTED_INPUT_TYPE = InputType.RDKIT_MOL
    OUTPUT_SHAPE = (1024,)
    
    def __init__(self, n_bits: int = 1024):
        super().__init__()
        self.n_bits = n_bits
    
    def _featurize(self, mol, **kwargs) -> np.ndarray:
        # Input is guaranteed to be RDKit Mol
        return generate_features(mol, self.n_bits)
```

See {doc}`../architecture` for more details.
