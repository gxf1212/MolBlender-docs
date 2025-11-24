# Evaluation Methodology

Complete guide to MolBlender's data splitting, cross-validation, and model evaluation protocols.

## Overview

MolBlender follows rigorous machine learning best practices to ensure fair model comparison and reproducible results. This document provides a detailed explanation of:

- How training and test sets are created
- How 5-fold cross-validation is performed
- How models are trained and evaluated
- How to interpret metrics in the Dashboard
- Code references for understanding the implementation

```{admonition} Key Principle
:class: tip

All models are evaluated on the **same test set** with **fixed random seeds** to ensure fair comparison and reproducibility.
```

## Data Splitting Strategy

```{admonition} Advanced Splitting Strategies
:class: seealso

This section covers the **internal train/test splitting** used during model screening. For comprehensive dataset splitting strategies (scaffold split, temporal split, etc.) to prepare your data **before** running screening, see {doc}`/usage/data/splitting`.
```

### Train/Test Split

MolBlender uses sklearn's `train_test_split` to divide your dataset into training and test sets.

#### Default Configuration

```python
# Default parameters in ScreeningConfig (base.py:215-217)
cv_folds: int = 5            # Cross-validation folds
test_size: float = 0.2       # Test set proportion (20%)
random_state: int = 42       # Random seed for reproducibility
```

#### Implementation Details

**Code Location**: `src/molblender/models/api/core/data_handler.py:120-125`

```python
def split_data(self, X: np.ndarray, y: np.ndarray):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=self.config.test_size,        # Default: 0.2 (20%)
        random_state=self.config.random_state,  # Default: 42
        shuffle=True                            # Shuffle before splitting
    )
    return {
        'X_train': X_train,  # 80% of data
        'X_test': X_test,    # 20% of data
        'y_train': y_train,
        'y_test': y_test
    }
```

#### Numerical Example

For a dataset with **1000 molecules**:

```
Original Dataset: 1000 samples
    ↓
split_data(test_size=0.2, random_state=42)
    ├─→ Training Set: 800 samples (80%)
    └─→ Test Set: 200 samples (20%)
```

#### Stratified Splitting for Classification

For classification tasks, MolBlender automatically uses **stratified sampling** to maintain class balance:

**Code Location**: `data_handler.py:108-117`

```python
if stratify is None and self.config.task_type in [TaskType.CLASSIFICATION, ...]:
    unique_labels = np.unique(y)
    if len(unique_labels) < len(y) * 0.5:  # Not too many classes
        stratify = y  # Enable stratification

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=self.config.test_size,
        random_state=self.config.random_state
    )
```

### Reproducibility Guarantee

The `random_state=42` parameter ensures that:
- The same dataset will always be split the same way
- Results are reproducible across different runs
- Different users get identical train/test splits

```{warning}
**Current Limitation**: Each representation is split independently (`standard.py:87`). While `random_state=42` is fixed, if representations have different sample counts (e.g., due to missing values), they may have slightly different test sets. This is a known issue and will be addressed in a future release.
```

## Cross-Validation Protocol

### 5-Fold Cross-Validation

Cross-validation is performed **only on the training set** to estimate model performance without touching the test set.

#### Why Cross-Validation?

- **Reduces overfitting**: Tests model on multiple train/validation splits
- **Better performance estimation**: Averages over 5 different validation sets
- **Efficient use of data**: Uses all training data for both training and validation

#### Implementation Details

**Code Location**: `src/molblender/models/api/core/evaluation/evaluator.py:285-327`

```python
def _cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray):
    """Perform cross-validation with fixed random_state for reproducibility."""

    cv_folds = self.config.cv_folds  # Default: 5

    # Create KFold object with fixed random_state for reproducibility
    if self.config.task_type in [TaskType.CLASSIFICATION, ...]:
        # Use StratifiedKFold for classification to maintain class balance
        cv_splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.config.random_state  # Fixed: 42
        )
    else:
        # Use KFold for regression
        cv_splitter = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.config.random_state  # Fixed: 42
        )

    # Perform cross-validation with fixed splitting
    cv_scores = cross_val_score(
        model, X, y,                              # X, y are TRAINING data only
        cv=cv_splitter,                           # Use splitter object with fixed random_state
        scoring=scoring,                          # e.g., 'r2' for regression
        n_jobs=self.config.max_workers_per_model  # Parallel execution
    )

    return cv_scores  # Returns [score1, score2, score3, score4, score5]
```

**Cross-Validation Call**: `evaluator.py:139`

```python
# Called during model evaluation
cv_scores = self._cross_validate(model, X_train, y_train)
```

#### How sklearn Splits the Data

When you call `cross_val_score(model, X_train, y_train, cv=cv_splitter)`, sklearn internally:

```python
# Pseudo-code for sklearn's internal logic
# MolBlender creates KFold with fixed random_state
kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # ✅ Fixed random_state

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    # Split training data into CV train and validation
    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

    # Train model on CV training set
    cloned_model = clone(model)
    cloned_model.fit(X_cv_train, y_cv_train)

    # Evaluate on CV validation set
    score = scoring(cloned_model, X_cv_val, y_cv_val)
    scores.append(score)

return np.array(scores)  # [0.85, 0.87, 0.84, 0.86, 0.88]
```

#### Numerical Example

For a training set with **800 samples**:

```
Training Set: 800 samples
    ↓
5-Fold Cross-Validation
    ├─→ Fold 1: CV_train 640 samples (80%) | CV_val 160 samples (20%) → score1 = 0.85
    ├─→ Fold 2: CV_train 640 samples (80%) | CV_val 160 samples (20%) → score2 = 0.87
    ├─→ Fold 3: CV_train 640 samples (80%) | CV_val 160 samples (20%) → score3 = 0.84
    ├─→ Fold 4: CV_train 640 samples (80%) | CV_val 160 samples (20%) → score4 = 0.86
    └─→ Fold 5: CV_train 640 samples (80%) | CV_val 160 samples (20%) → score5 = 0.88
        ↓
Mean CV Score = (0.85 + 0.87 + 0.84 + 0.86 + 0.88) / 5 = 0.86
Std CV Score = 0.015
```

```{admonition} Reproducibility
:class: tip

MolBlender now uses `KFold(random_state=42)` for cross-validation, ensuring **complete reproducibility** of both CV scores and test scores across different runs.
```

## Model Training and Evaluation

### Final Model Training

After cross-validation, the model is trained on the **entire training set** to maximize performance.

**Code Location**: `evaluator.py:167-183`

```python
# Train final model using ALL training data
with timeout_context(train_timeout, f"Train {model_name}+{representation_name}"):
    n_samples = X_train.shape[0]  # 800 samples
    logger.debug(f"Training {model_name} on {n_samples} samples")

    # Train on full training set (not just one fold)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
```

### Test Set Evaluation

The trained model is evaluated on the **held-out test set**.

**Code Location**: `evaluator.py:192-210`

```python
# Predict on test set
test_pred = model.predict(X_test)   # 200 predictions
train_pred = model.predict(X_train) # 800 predictions (for diagnostics)

# Compute test metrics
test_metrics = compute_metrics(y_test, test_pred)
# test_metrics = {'r2': 0.87, 'rmse': 0.52, 'mae': 0.41, ...}
```

### Metrics Explained

MolBlender reports **two types of metrics**:

| Metric Type | Source | Purpose | Dashboard Display |
|------------|--------|---------|-------------------|
| **CV Scores** | 5-fold cross-validation on training set | Estimate model robustness | Mean ± Std in tables |
| **Test Scores** | Final evaluation on test set | True generalization performance | **Primary metrics** |

#### What Dashboard Shows

```{admonition} Dashboard Metrics Source
:class: important

The **primary metrics** displayed in the Dashboard (Best Metric, Mean Metric, etc.) are **test set scores**, not CV scores. CV scores are shown as auxiliary information to assess model stability.
```

**Example Dashboard Display**:

```
Best Model: Random Forest + Morgan FP
├─ Test R² = 0.87 ← Main metric (from test set)
├─ CV R² = 0.86 ± 0.015 ← Auxiliary (from cross-validation)
├─ Test RMSE = 0.52
└─ Training Time = 12.3s
```

## Complete Evaluation Workflow

### Visual Overview

```
Original Dataset: 1000 molecules
    ↓
[data_handler.split_data() - data_handler.py:120]
    ├─→ X_train, y_train: 800 samples (80%)
    │       ↓
    │   [evaluator._cross_validate() - evaluator.py:139]
    │       ↓
    │   [sklearn.cross_val_score(cv=5) - evaluator.py:302]
    │   ├─→ Fold 1: CV_train 640, CV_val 160 → score1
    │   ├─→ Fold 2: CV_train 640, CV_val 160 → score2
    │   ├─→ Fold 3: CV_train 640, CV_val 160 → score3
    │   ├─→ Fold 4: CV_train 640, CV_val 160 → score4
    │   └─→ Fold 5: CV_train 640, CV_val 160 → score5
    │       ↓
    │   cv_scores = [0.85, 0.87, 0.84, 0.86, 0.88]
    │   mean_cv = 0.86, std_cv = 0.015
    │       ↓
    │   [model.fit(X_train, y_train) - evaluator.py:167]
    │   Final model trained on ALL 800 training samples
    │
    └─→ X_test, y_test: 200 samples (20%)
            ↓
        [model.predict(X_test) - evaluator.py:193]
        test_pred = [pred1, pred2, ..., pred200]
            ↓
        [compute_metrics(y_test, test_pred)]
        test_score = 0.87 ← **Dashboard displays this**
```

### Step-by-Step Execution

1. **Data Loading**: Load dataset with molecules and target values
2. **Feature Extraction**: Compute molecular representations (fingerprints, descriptors, etc.)
3. **Train/Test Split**: Divide data into 80% train (800) and 20% test (200)
4. **Cross-Validation**: 5-fold CV on training set to get CV scores
5. **Final Training**: Train model on entire training set (800 samples)
6. **Test Evaluation**: Predict on test set (200 samples) and compute test scores
7. **Result Storage**: Save all metrics to SQLite database
8. **Dashboard Display**: Show test scores as primary metrics

## Best Practices

### Choosing test_size

The `test_size` parameter controls the train/test split ratio. Choose based on your dataset size:

| Dataset Size | Recommended test_size | Train Samples | Test Samples | Rationale |
|-------------|----------------------|---------------|--------------|-----------|
| < 500 | `0.3` (30%) | 350 | 150 | Need sufficient test samples for reliable evaluation |
| 500 - 5000 | `0.2` (20%) | 4000 | 1000 | Default - balanced split |
| 5000 - 50000 | `0.15` (15%) | 42500 | 7500 | More data for training |
| > 50000 | `0.1` (10%) | 90000 | 10000 | Large test set still provides good statistics |

**Example**:

```python
# For a small dataset with 300 molecules
results = universal_screen(
    dataset=small_dataset,
    target_column="activity",
    test_size=0.3,  # Use 30% for testing (90 molecules)
    cv_folds=3      # Fewer folds to ensure enough data per fold
)

# For a large dataset with 100,000 molecules
results = universal_screen(
    dataset=large_dataset,
    target_column="solubility",
    test_size=0.1,  # Only 10% for testing (10,000 molecules)
    cv_folds=5      # Standard 5-fold CV
)
```

### Choosing cv_folds

The `cv_folds` parameter controls the number of cross-validation folds:

| Dataset Size | Recommended cv_folds | CV Train | CV Val | Rationale |
|-------------|---------------------|----------|--------|-----------|
| < 100 | `3` | 67 | 33 | Avoid too-small validation sets |
| 100 - 1000 | `5` (default) | 640 | 160 | Standard choice, good balance |
| 1000 - 10000 | `5` or `3` | 6400 | 1600 | Use 3 for speed |
| > 10000 | `3` | 24000 | 8000 | Faster, minimal performance loss |

**Speed vs Accuracy Trade-off**:

```python
# Fast screening for exploration (2-3x faster)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    cv_folds=3  # 3-fold CV
)

# Thorough screening for final evaluation (default)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    cv_folds=5  # 5-fold CV
)

# Very thorough (research/publication, 2x slower)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    cv_folds=10  # 10-fold CV
)
```

### Ensuring Fair Comparison

To ensure all models are compared fairly:

1. **Use the same random_state** (default: 42)
2. **Use the same test_size** for all screening runs
3. **Use the same cv_folds** for all models

```python
# Good: All models compared on same splits
config_params = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

results1 = universal_screen(dataset, "activity", **config_params)
results2 = compare_models(dataset, "activity", "morgan_fp", **config_params)
results3 = compare_representations(dataset, "activity", "random_forest", **config_params)
```

```{warning}
**Current Limitation**: Each representation may use slightly different train/test splits (see Known Limitations section). For critical comparisons, verify that all representations have the same sample count before splitting.
```

## Known Limitations

### Limitation 1: Per-Representation Splitting

**Issue**: Each representation triggers a separate `split_data()` call.

**Code Location**: `standard.py:84-90`

```python
for repr_name, X in prepared_data['representations'].items():
    y = prepared_data['targets']
    split_data = self.data_handler.split_data(X, y)  # ⚠️ Called per representation

    X_train, X_test = split_data['X_train'], split_data['X_test']
    y_train, y_test = split_data['y_train'], split_data['y_test']
```

**Impact**:
- **Minor** if all representations have the same sample count
- **Moderate** if representations have missing values (different sample counts)

**Workaround**:
- Use `random_state=42` (already default) to maximize consistency
- Ensure all representations are computed for the same molecules

**Future Fix**: Planned refactor to split train/test indices once globally, then apply to all representations.

### ~~Limitation 2: CV Random State~~ ✅ FIXED

**Previous Issue**: sklearn's `cross_val_score` didn't fix `random_state` for KFold splitting.

**Status**: ✅ **FIXED** - MolBlender now creates `KFold`/`StratifiedKFold` objects with `random_state=42` before passing to `cross_val_score`.

**Implementation**: `evaluator.py:301-323`

```python
# Create KFold with fixed random_state
if task_type == CLASSIFICATION:
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
else:
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

# Use splitter object instead of integer
cv_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)
```

**Result**: Both CV scores and test scores are now fully reproducible across different runs.

## Code Reference Index

Quick reference to implementation details:

| Functionality | File | Lines | Description |
|--------------|------|-------|-------------|
| **Configuration** | `base.py` | 215-217 | `cv_folds=5, test_size=0.2, random_state=42` |
| **Train/Test Split** | `data_handler.py` | 120-125 | `train_test_split(test_size, random_state, shuffle=True)` |
| **Stratified Split** | `data_handler.py` | 108-117 | `StratifiedShuffleSplit` for classification |
| **CV Call** | `evaluator.py` | 139 | `cv_scores = self._cross_validate(model, X_train, y_train)` |
| **CV Implementation** | `evaluator.py` | 285-327 | `cross_val_score(model, X, y, cv=cv_splitter, scoring=...)` |
| **Final Training** | `evaluator.py` | 167-183 | `model.fit(X_train, y_train)` on full training set |
| **Test Prediction** | `evaluator.py` | 192-210 | `test_pred = model.predict(X_test)` |
| **Metric Computation** | `evaluator.py` | 313-400 | Compute R², RMSE, MAE, etc. |
| **Per-Repr Splitting** | `standard.py` | 84-90 | Each representation split independently |

## Further Reading

- {doc}`screening` - Complete API reference for screening functions
- {doc}`results` - Understanding and exporting screening results
- {doc}`../dashboard/metrics` - Dashboard metric interpretation
- [sklearn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html) - sklearn documentation

## Summary

```{admonition} Key Takeaways
:class: tip

1. **Train/Test Split**: 80/20 by default, fixed with `random_state=42`
2. **Cross-Validation**: 5-fold CV performed **only on training set** with fixed `random_state=42`
3. **Final Model**: Trained on **full training set** (not just one fold)
4. **Test Evaluation**: Metrics computed on **held-out test set**
5. **Dashboard Shows**: **Test scores** as primary metrics, CV scores as auxiliary
6. **Reproducibility**: ✅ Both CV scores and test scores are **fully reproducible**
7. **Known Issues**: Per-representation splitting (planned fix in future release)
```

For questions or issues about the evaluation methodology, please [open an issue on GitHub](https://github.com/gxf1212/MolBlender/issues).
