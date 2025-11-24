# Custom User-Provided Splits

Use predefined train/test assignments from external sources or implement custom splitting logic.

```{admonition} Overview
:class: note

Custom splitting allows you to:
- Use existing train/test assignments from benchmark datasets
- Apply pre-computed temporal or experimental splits
- Integrate splits from external pipelines or publications
- Implement domain-specific splitting criteria
```

**Navigation**: {doc}`index` > Custom Splits

---

## When to Use Custom Splits

✅ **Use custom splits when:**
- Reproducing specific benchmark results with exact same splits
- Dataset has inherent temporal or experimental structure
- External experts have defined optimal splits for your domain
- Need same split across multiple experiments for consistency
- Implementing specialized splitting logic not covered by built-in strategies

❌ **Don't use when:**
- Built-in strategies cover your needs (simpler and validated)
- No pre-existing split assignments available
- Standard random splitting is sufficient

---

## Two Input Modes

### Mode 1: Split Column from Metadata

Use a column in your dataset that indicates train/test assignment.

```python
from molblender.data import MolecularDataset

# Dataset with a pre-existing split column
dataset = MolecularDataset.from_csv(
    "benchmark_data.csv",
    input_column="SMILES",
    label_columns=["pIC50"]
)

# Assuming metadata has a 'split' column with 'train'/'test' values
dataset.metadata['split'] = ['train', 'train', 'test', 'train', 'test', ...]

train_ds, test_ds = dataset.train_test_split(
    method='custom',
    split_column='split'
)
```

### Mode 2: Explicit Index Arrays

Provide pre-computed train/test indices directly.

```python
# Predefined indices (e.g., from temporal split, experimental batches)
train_indices = [0, 1, 2, 5, 6, 7, 10, 11, 12]
test_indices = [3, 4, 8, 9, 13, 14]

train_ds, test_ds = dataset.train_test_split(
    method='custom',
    train_indices=train_indices,
    test_indices=test_indices
)
```

---

## Supported Split Column Formats

| Format | Train Value | Test Value | Example |
|--------|-------------|------------|---------|
| String | `'train'` | `'test'` | `['train', 'test', 'train']` |
| Numeric | `0` | `1` | `[0, 1, 0, 0, 1]` |
| Boolean | `False` | `True` | `[False, True, False]` |

All formats are automatically recognized and parsed by the utility function `indices_from_split_column()`.

### Example: String Format

```python
# CSV with split column
# SMILES,activity,split
# CCO,5.2,train
# c1ccccc1,6.1,test
# CC(=O)O,4.8,train

dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    label_columns=["activity"]
)

# Split column automatically detected
train, test = dataset.train_test_split(
    method='custom',
    split_column='split'  # Column name in metadata
)
```

### Example: Numeric Format

```python
# Split column with 0/1 values
dataset.metadata['split'] = [0, 1, 0, 0, 1, 0, 1]  # 0=train, 1=test

train, test = dataset.train_test_split(
    method='custom',
    split_column='split'
)
```

### Example: Boolean Format

```python
# Split column with True/False values
dataset.metadata['is_test'] = [False, True, False, False, True]

train, test = dataset.train_test_split(
    method='custom',
    split_column='is_test'
)
```

---

## Functional API

```python
from molblender.data.dataset.splitting import train_test_split
import numpy as np

# Your data
X = np.array([...])
y = np.array([...])

# Mode 1: Using split column
split_column_values = ['train', 'test', 'train', 'test', 'train']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    method='custom',
    split_column=split_column_values  # Pass values directly
)

# Mode 2: Using explicit indices
train_idx = [0, 2, 4]
test_idx = [1, 3]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    method='custom',
    train_indices=train_idx,
    test_indices=test_idx
)
```

---

## Use Cases

### 1. Benchmark Reproduction

Use exact same train/test splits from published papers to ensure fair comparison.

```python
# Load benchmark dataset with predefined splits
dataset = MolecularDataset.from_csv(
    "benchmark_qm9.csv",
    input_column="SMILES",
    label_columns=["homo"]
)

# Split column from benchmark paper
# Ensures results are comparable to published literature
train, test = dataset.train_test_split(
    method='custom',
    split_column='split'  # Pre-defined in benchmark
)

# Train and evaluate
from molblender.models import universal_screen
results = universal_screen(
    dataset=train,
    target_column="homo",
    # ... model parameters
)

# Report: "Using benchmark splits from [Paper et al. 2024]"
print(f"Benchmark R²: {results['best_model']['test_r2']:.3f}")
```

### 2. Temporal Splits

Train on older data, test on newer data to simulate real-world deployment.

```python
import pandas as pd

# Load dataset with timestamps
df = pd.read_csv("time_series_molecules.csv")
dataset = MolecularDataset.from_df(
    df,
    input_column="SMILES",
    label_columns=["activity"]
)

# Split by date: Before 2023 → train, 2023+ → test
df['split'] = df['date'].apply(lambda d: 'test' if d >= '2023-01-01' else 'train')
dataset.metadata['split'] = df['split'].values

train, test = dataset.train_test_split(
    method='custom',
    split_column='split'
)

# Test set simulates future deployment
```

### 3. Experimental Design

Group experimental batches together to account for batch effects.

```python
# Dataset from multiple experimental batches
# Batch 1-3 → train, Batch 4 → test

dataset.metadata['batch'] = [1, 1, 2, 2, 3, 3, 4, 4]
dataset.metadata['split'] = ['train' if b < 4 else 'test'
                              for b in dataset.metadata['batch']]

train, test = dataset.train_test_split(
    method='custom',
    split_column='split'
)

# Validate batch-to-batch generalization
```

### 4. External Annotations

Use expert-curated splits for domain-specific validation.

```python
# Medicinal chemist provided expert split
# Based on synthetic accessibility and IP considerations

expert_train_indices = load_expert_annotations("train_molecules.txt")
expert_test_indices = load_expert_annotations("test_molecules.txt")

train, test = dataset.train_test_split(
    method='custom',
    train_indices=expert_train_indices,
    test_indices=expert_test_indices
)
```

---

## Validation

MolBlender automatically validates custom splits to prevent errors:

### Overlap Detection

```python
# Error: Indices overlap between train and test
train_idx = [0, 1, 2, 3]
test_idx = [2, 3, 4, 5]  # 2 and 3 overlap!

# Raises ValueError
train, test = dataset.train_test_split(
    method='custom',
    train_indices=train_idx,
    test_indices=test_idx
)
# ValueError: Train and test indices overlap: {2, 3}
```

### Range Checking

```python
# Error: Index out of range
train_idx = [0, 1, 2]
test_idx = [3, 4, 100]  # 100 > dataset size!

# Raises IndexError
train, test = dataset.train_test_split(
    method='custom',
    train_indices=train_idx,
    test_indices=test_idx
)
# IndexError: Test indices contain out-of-range values
```

### Value Parsing

```python
# Error: Invalid split column values
dataset.metadata['split'] = ['train', 'test', 'validation', 'train']  # 'validation' invalid!

# Raises ValueError
train, test = dataset.train_test_split(
    method='custom',
    split_column='split'
)
# ValueError: Split column contains invalid values: {'validation'}
# Expected: 'train'/'test', 0/1, or True/False
```

---

## Advanced Custom Logic

For more complex splitting logic, create indices programmatically then use custom split:

### Example: Property-Based Split

```python
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

# Custom split based on molecular weight
def mw_split(dataset, mw_threshold=300, test_size=0.2):
    """Split by MW: heavy molecules in test, light in train."""

    # Compute molecular weights
    mws = []
    for mol_obj in dataset.molecules:
        rdkit_mol = mol_obj.get_rdkit_mol()
        mw = Descriptors.MolWt(rdkit_mol)
        mws.append(mw)

    mws = np.array(mws)

    # Sort by MW
    sorted_idx = np.argsort(mws)
    n_test = int(len(dataset) * test_size)

    # Heaviest molecules → test
    test_idx = sorted_idx[-n_test:]
    train_idx = sorted_idx[:-n_test]

    return train_idx, test_idx

# Apply custom split
train_idx, test_idx = mw_split(dataset, mw_threshold=300, test_size=0.2)

train, test = dataset.train_test_split(
    method='custom',
    train_indices=train_idx,
    test_indices=test_idx
)

print(f"Train MW range: {np.mean([...]) :.1f}")
print(f"Test MW range: {np.mean([...]):.1f}")
```

### Example: Combining Multiple Strategies

```python
# First: Scaffold split to get novel scaffolds
from molblender.models.api.core.splitting import scaffold_split

scaffold_result = scaffold_split(
    X=dataset.features.values,
    y=dataset.labels.values,
    smiles=dataset.get_smiles(),
    test_size=0.3
)

# Then: DNR split within scaffold test set to get challenging molecules
scaffold_test_idx = scaffold_result['test_indices']
scaffold_test_dataset = dataset.subset(scaffold_test_idx)

from molblender.models.api.core.splitting import dnr_split
dnr_result = dnr_split(
    dataset=scaffold_test_dataset,
    mode="threshold",
    dnr_threshold=0.4
)

# Final split: Novel scaffolds + activity cliffs
train_idx = scaffold_result['train_indices']
test_idx = scaffold_test_idx[dnr_result['test_indices']]

train, test = dataset.train_test_split(
    method='custom',
    train_indices=train_idx,
    test_indices=test_idx
)

# Result: Most challenging test set (novel scaffolds + high DNR)
```

---

## Utility Functions

### `indices_from_split_column()`

Parse split column values into train/test indices.

```python
from molblender.data.dataset.splitting import indices_from_split_column

# Your split column
split_values = ['train', 'test', 'train', 'test', 'train']

# Parse to indices
train_idx, test_idx = indices_from_split_column(split_values)

print(f"Train indices: {train_idx}")  # [0, 2, 4]
print(f"Test indices: {test_idx}")    # [1, 3]
```

**Supported formats:**
- Strings: `'train'`/`'test'`
- Numeric: `0`/`1`
- Boolean: `False`/`True`

---

## Best Practices

```{admonition} Custom Split Best Practices
:class: tip

**Do:**
- ✅ Document the source and rationale for custom splits
- ✅ Validate indices before use (check overlaps, ranges)
- ✅ Report custom split methodology in publications
- ✅ Use fixed indices for reproducibility across experiments

**Don't:**
- ❌ Manually cherry-pick favorable train/test assignments
- ❌ Use custom splits without clear justification
- ❌ Ignore validation errors (fix indices instead)
- ❌ Change split assignments between experiments
```

---

## Comparison with Other Strategies

| Aspect | Custom Split | Built-in Strategies |
|--------|-------------|---------------------|
| **Control** | Full control | Pre-defined algorithms |
| **Reproducibility** | User-managed | Automatic with random_state |
| **Validation** | Manual validation needed | Built-in validation |
| **Use Case** | Specialized scenarios | General scenarios |
| **Complexity** | Higher (requires indices) | Lower (parameters only) |

**When to prefer custom split:**
- Need exact benchmark reproduction
- Have domain expertise for optimal splitting
- Implementing novel splitting strategy

**When to prefer built-in:**
- Standard ML scenarios
- No pre-existing splits available
- Want automated validation

---

## Related Strategies

- **For basic splits**: See {doc}`basic_strategies`
- **For chemical structure**: See {doc}`chemical_strategies`
- **For property-based**: See {doc}`property_strategies`
- **For advanced OOD**: See {doc}`advanced_strategies`

---

**Navigation**: {doc}`index` | Previous: {doc}`advanced_strategies` | Next: {doc}`best_practices`
