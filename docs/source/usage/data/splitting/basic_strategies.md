# Basic Splitting Strategies

Standard machine learning splitting strategies for most use cases.

```{admonition} Overview
:class: note

This page covers the four fundamental splitting strategies used in standard machine learning:
- **train_test**: Simple 2-way split (most common)
- **train_val_test**: 3-way split for hyperparameter optimization
- **nested_cv**: Nested cross-validation for unbiased HPO evaluation
- **cv_only**: Cross-validation only (small datasets)
```

**Navigation**: {doc}`index` > Basic Strategies

---

## 1. Train/Test Split (Default)

The standard two-way split used for most screening tasks.

### When to Use

✅ **Use train_test when:**
- Dataset has ≥500 samples
- Standard model screening without HPO
- Quick baseline performance assessment
- Industry applications (most common scenario)

❌ **Don't use when:**
- Dataset < 100 samples (use cv_only instead)
- Heavy hyperparameter tuning needed (use train_val_test)
- Academic research requiring unbiased HPO (use nested_cv)

### Configuration

```python
from molblender.models import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_test",  # Default
    test_size=0.2,                # 20% test set
    cv_folds=5,                   # 5-fold CV on training set
    random_state=42
)
```

### How It Works

```
Dataset (1000 samples)
    ↓
train_test_split(test_size=0.2, random_state=42)
    ├─→ Training Set: 800 samples (80%)
    │      ↓
    │   5-Fold Cross-Validation
    │   ├─→ Fold 1: train 640, val 160
    │   ├─→ Fold 2: train 640, val 160
    │   ├─→ Fold 3: train 640, val 160
    │   ├─→ Fold 4: train 640, val 160
    │   └─→ Fold 5: train 640, val 160
    │      ↓
    │   Final model: trained on all 800 samples
    │
    └─→ Test Set: 200 samples (20%)
           ↓
        Final evaluation
```

**Key Points:**
- Test set is **never used for training or hyperparameter tuning**
- Cross-validation on training set provides robust performance estimate
- Final model trained on entire training set, evaluated on test set
- Fixed `random_state=42` ensures reproducibility

### Implementation Details

**Code Location**: `src/molblender/models/api/core/splitting/strategies.py:26-84`

```python
def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Standard train/test split.

    For classification, uses StratifiedShuffleSplit to maintain class balance.
    For regression, uses regular train_test_split with shuffling.
    """
    if stratify is not None:
        # Stratified split for classification
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, stratify))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        # Regular split for regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'split_type': 'train_test'
    }
```

### Example

```python
from molblender.data import MolecularDataset
from molblender.models import universal_screen

# Load dataset
dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    label_columns=["activity"]
)

# Standard train/test screening
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    task_type="regression",
    split_strategy="train_test",
    test_size=0.2,
    cv_folds=5,
    random_state=42
)

# Access results
print(f"Best model: {results['best_model']['model_name']}")
print(f"Test R²: {results['best_model']['test_r2']:.3f}")
print(f"CV R² (mean): {results['best_model']['cv_r2_mean']:.3f}")
```

---

## 2. Train/Val/Test Split

Three-way split for scenarios involving hyperparameter optimization.

### When to Use

✅ **Use train_val_test when:**
- Dataset has ≥1000 samples (preferably ≥5000)
- Performing hyperparameter optimization (HPO)
- Model selection among multiple architectures
- Need dedicated validation set separate from test set

❌ **Don't use when:**
- Dataset < 1000 samples (insufficient for 3-way split)
- No hyperparameter tuning (use train_test instead)
- Academic research requiring unbiased HPO (use nested_cv)

### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_val_test",
    test_size=0.15,      # 15% test set
    val_size=0.15,       # 15% validation set
    random_state=42
)
```

### How It Works

```
Dataset (1000 samples)
    ↓
First split: (train+val) vs test
    ├─→ Temp Set: 850 samples (85%)
    │      ↓
    │   Second split: train vs val
    │   ├─→ Training Set: 700 samples (70%)
    │   └─→ Validation Set: 150 samples (15%)
    │
    └─→ Test Set: 150 samples (15%)
```

**Workflow:**
1. **Training set**: Train models with different hyperparameters
2. **Validation set**: Select best hyperparameters
3. **Test set**: Final unbiased evaluation of best model

### Implementation Details

**Code Location**: `splitting/strategies.py:86-169`

```python
def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Three-way split: train / validation / test.

    Best for HPO when you have sufficient data.
    """
    # First split: (train+val) vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=True
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp if stratify is not None else None,
        shuffle=True
    )

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'split_type': 'train_val_test'
    }
```

### Example with HPO

```python
# Large dataset with hyperparameter optimization
results = universal_screen(
    dataset=large_dataset,  # ≥5000 samples
    target_column="pIC50",
    split_strategy="train_val_test",
    test_size=0.15,
    val_size=0.15,
    enable_hpo=True,         # Enable Stage 2 HPO
    hpo_stage="coarse",      # Coarse grid search
    top_n_for_hpo=10,
    random_state=42
)

# Validation set used for HPO, test set for final evaluation
print(f"HPO best params: {results['best_model']['best_params']}")
print(f"Test R²: {results['best_model']['test_r2']:.3f}")
```

---

## 3. Nested Cross-Validation

Provides unbiased performance estimates for hyperparameter optimization.

### When to Use

✅ **Use nested_cv when:**
- Academic research or publications
- Need unbiased performance estimate with HPO
- Comparing multiple model selection strategies
- Dataset size 500-5000 samples (sweet spot)

❌ **Don't use when:**
- Dataset < 500 samples (computational cost too high)
- Industry deployment (train_val_test more practical)
- No hyperparameter tuning (use train_test)

### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="nested_cv",
    outer_cv_folds=5,     # Outer CV for performance estimation
    inner_cv_folds=3,     # Inner CV for hyperparameter tuning
    random_state=42
)
```

### How It Works

```
Dataset (1000 samples)
    ↓
Outer CV (5 folds) - for performance estimation
    ├─→ Fold 1: dev 800, test 200
    │      ↓
    │   Inner CV (3 folds on dev set) - for HPO
    │   ├─→ Inner Fold 1: train 533, val 267
    │   ├─→ Inner Fold 2: train 533, val 267
    │   └─→ Inner Fold 3: train 534, val 266
    │      ↓
    │   Best hyperparameters → test on outer test (200)
    │
    ├─→ Fold 2: dev 800, test 200
    │   (repeat inner CV...)
    ...
    └─→ Fold 5: dev 800, test 200
           ↓
        Average performance across 5 outer folds
```

**Key Insight**: Each outer fold gets its own best hyperparameters from inner CV, preventing optimistic bias.

### Why Nested CV?

```{admonition} The Bias Problem
:class: warning

**Single CV + HPO = Biased**:
- Use CV to select hyperparameters
- Report CV score as performance
- ❌ Optimistically biased (hyperparameters tuned on validation folds)

**Nested CV = Unbiased**:
- Outer CV: Never sees the test fold during HPO
- Inner CV: Tunes hyperparameters only on development folds
- ✅ Unbiased estimate (each test fold is truly held out)
```

### Implementation Details

**Code Location**: `splitting/strategies.py:215-284`

```python
def get_nested_cv_splitter(
    n_samples: int,
    outer_cv_folds: int = 5,
    inner_cv_folds: int = 3,
    random_state: int = 42,
    is_classification: bool = False
) -> Dict[str, Any]:
    """
    Get nested cross-validation splitters.

    Nested CV provides unbiased performance estimates when doing HPO.
    - Outer CV: For performance estimation
    - Inner CV: For hyperparameter tuning
    """
    if is_classification:
        outer_cv = StratifiedKFold(
            n_splits=outer_cv_folds,
            shuffle=True,
            random_state=random_state
        )
        inner_cv = StratifiedKFold(
            n_splits=inner_cv_folds,
            shuffle=True,
            random_state=random_state + 1  # Different seed
        )
    else:
        outer_cv = KFold(
            n_splits=outer_cv_folds,
            shuffle=True,
            random_state=random_state
        )
        inner_cv = KFold(
            n_splits=inner_cv_folds,
            shuffle=True,
            random_state=random_state + 1
        )

    return {
        'outer_cv': outer_cv,
        'inner_cv': inner_cv,
        'split_type': 'nested_cv'
    }
```

### Example

```python
# Research-grade evaluation with HPO
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="nested_cv",
    outer_cv_folds=5,
    inner_cv_folds=3,
    enable_hpo=True,
    random_state=42
)

# Report unbiased performance for publication
print(f"Nested CV R² (unbiased): {results['best_model']['nested_cv_r2_mean']:.3f}")
print(f"Nested CV R² (std): {results['best_model']['nested_cv_r2_std']:.3f}")
```

---

## 4. CV-Only Strategy

Pure cross-validation without a separate test set, for small datasets.

### When to Use

✅ **Use cv_only when:**
- Dataset < 100 samples (very small)
- Cannot afford to hold out test set
- Exploratory analysis or proof-of-concept
- Maximum data utilization required

❌ **Don't use when:**
- Dataset ≥100 samples (use train_test instead)
- Need truly independent test set
- Model selection or HPO involved (optimistically biased)

### Configuration

```python
results = universal_screen(
    dataset=small_dataset,  # < 100 samples
    target_column="activity",
    split_strategy="cv_only",
    cv_folds=5,
    random_state=42
)
```

### How It Works

```
Dataset (80 samples)
    ↓
5-Fold Cross-Validation (no separate test set)
    ├─→ Fold 1: train 64, val 16
    ├─→ Fold 2: train 64, val 16
    ├─→ Fold 3: train 64, val 16
    ├─→ Fold 4: train 64, val 16
    └─→ Fold 5: train 64, val 16
           ↓
        Average CV score as final metric
```

**Important**: No truly independent test set. Every sample is used for validation in one fold.

### Limitations

```{warning}
**CV-only doesn't provide independent test set**:
- Performance estimates may be optimistically biased
- Model selection on CV scores can lead to overfitting
- Use only when dataset size prohibits train/test split
- Consider LOOCV (Leave-One-Out CV) for datasets < 30 samples
```

### Example

```python
from molblender.data import MolecularDataset

# Very small dataset
small_dataset = MolecularDataset.from_csv(
    "small_molecules.csv",  # Only 80 molecules
    input_column="SMILES",
    label_columns=["activity"]
)

# CV-only screening
results = universal_screen(
    dataset=small_dataset,
    target_column="activity",
    split_strategy="cv_only",
    cv_folds=5,  # Or 10 for smaller datasets
    random_state=42
)

# Only CV score available (no independent test score)
print(f"CV R² (mean): {results['best_model']['cv_r2_mean']:.3f}")
print(f"CV R² (std): {results['best_model']['cv_r2_std']:.3f}")
```

---

## Cross-Validation Details

### Adaptive Fold Adjustment

MolBlender automatically adjusts cross-validation folds based on dataset size.

**Code Location**: `evaluation/evaluator.py:288-299`

```python
def _cross_validate(self, model, X, y):
    """Perform cross-validation with automatic fold adjustment."""

    cv_folds = self.config.cv_folds
    n_samples = len(y)

    # Ensure we don't have more folds than samples
    if cv_folds > n_samples:
        logger.warning(f"cv_folds={cv_folds} > n_samples={n_samples}, "
                      f"using cv_folds={n_samples}")
        cv_folds = n_samples

    # Minimum 2 folds required
    if cv_folds < 2:
        logger.warning(f"cv_folds={cv_folds} invalid, using cv_folds=2")
        cv_folds = 2
```

### Classification vs Regression

**Code Location**: `evaluation/evaluator.py:302-315`

```python
# Create CV splitter based on task type
if self.config.task_type in [TaskType.CLASSIFICATION,
                             TaskType.BINARY_CLASSIFICATION]:
    # Use StratifiedKFold for classification
    cv_splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=self.config.random_state
    )
else:
    # Use KFold for regression
    cv_splitter = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=self.config.random_state
    )
```

### Stratified Sampling (Classification)

For classification tasks, MolBlender automatically uses **StratifiedKFold** to maintain class balance across folds.

**Benefits:**
- Maintains class distribution in each fold
- Prevents fold-to-fold variance from class imbalance
- More reliable CV scores for imbalanced datasets

**Example:**
```python
# Classification dataset with imbalanced classes
# Class 0: 800 samples, Class 1: 200 samples

# Without stratification (bad):
# Fold 1 might have: Class 0: 195, Class 1: 5 (95% vs 5%)
# Fold 2 might have: Class 0: 165, Class 1: 35 (83% vs 17%)

# With StratifiedKFold (good):
# All folds maintain: Class 0: 160, Class 1: 40 (80% vs 20%)
```

---

## Reproducibility Guarantees

All basic splitting strategies use **fixed random seeds** to ensure reproducibility.

### What's Reproducible

✅ **Guaranteed reproducible** (with same `random_state`):
- Train/test split indices
- Cross-validation fold assignments
- Model training (if model uses same seed)
- Final test scores

⚠️ **May vary slightly**:
- Training time (system load dependent)
- Memory usage (Python GC behavior)

### Example

```python
# Run 1
results1 = universal_screen(
    dataset=dataset,
    target_column="activity",
    random_state=42  # Fixed
)

# Run 2 (different session)
results2 = universal_screen(
    dataset=dataset,
    target_column="activity",
    random_state=42  # Same seed
)

# Guarantee: Identical splits
assert (results1['test_indices'] == results2['test_indices']).all()
```

**→ See {doc}`best_practices` for more on reproducibility**

---

## Quick Comparison

| Strategy | Test Set | Validation | HPO Support | Dataset Size | Computational Cost |
|----------|----------|------------|-------------|--------------|-------------------|
| **train_test** | Yes (20%) | CV on train | Limited | ≥500 | Low |
| **train_val_test** | Yes (15%) | Dedicated (15%) | Full | ≥1000 | Low |
| **nested_cv** | CV outer folds | CV inner folds | Unbiased | 500-5000 | High |
| **cv_only** | No | CV only | ❌ Biased | <100 | Medium |

---

## Related Strategies

- **For drug discovery**: See {doc}`chemical_strategies` (scaffold split)
- **For diversity testing**: See {doc}`property_strategies` (maxmin split)
- **For research validation**: See {doc}`advanced_strategies` (perimeter, MOOD splits)
- **For custom needs**: See {doc}`custom_splits`

---

**Navigation**: {doc}`index` | Next: {doc}`chemical_strategies`
