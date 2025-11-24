# Best Practices & Guidelines

Comprehensive guidelines for choosing and using splitting strategies effectively.

```{admonition} Overview
:class: note

This page provides:
- **Decision framework** for choosing the right splitting strategy
- **Reproducibility guidelines** for reliable experiments
- **Performance optimization** tips
- **Common pitfalls** to avoid
- **Recommended configurations** by dataset size and use case
```

**Navigation**: {doc}`index` > Best Practices

---

## Choosing the Right Strategy

### Decision Framework

```{mermaid}
graph TD
    A[Start: Dataset Characterization] --> B{Dataset Size?}

    B -->|< 100| C[cv_only]
    B -->|100-500| D{Need HPO?}
    B -->|500-5000| E{Application Type?}
    B -->|> 5000| F{Application Type?}

    D -->|Yes| G[nested_cv]
    D -->|No| H[train_test<br/>test_size=0.3]

    E -->|Drug Discovery| I{What aspect?}
    E -->|Standard ML| J[train_test<br/>test_size=0.2]

    F -->|Drug Discovery| K{What aspect?}
    F -->|Standard ML| L[train_test<br/>test_size=0.15]

    I -->|Novel Scaffolds| M[scaffold]
    I -->|Diverse Libraries| N[perimeter]
    I -->|Lead Optimization| O[lead_opt]
    I -->|Activity Cliffs| P[dnr]

    K -->|Novel Scaffolds| Q[scaffold]
    K -->|Deployment-Aware| R[mood]
    K -->|Fragment-to-Lead| S[molecular_weight]

    style C fill:#90EE90
    style G fill:#FFB6C1
    style H fill:#87CEEB
    style J fill:#87CEEB
    style L fill:#87CEEB
    style M fill:#FFD700
    style N fill:#FFA500
    style O fill:#FF6347
    style P fill:#9370DB
    style Q fill:#FFD700
    style R fill:#FFA500
    style S fill:#FF6347
```

### By Primary Goal

#### Goal: Standard ML Baseline
```python
# Most common scenario
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_test",  # Simple 80/20
    test_size=0.2,
    cv_folds=5,
    random_state=42
)
```

**Rationale**: Fast, reliable, industry-standard evaluation

#### Goal: Novel Scaffold Generalization
```python
# Drug discovery: Test on novel structures
train, test = dataset.train_test_split(
    method='scaffold',
    scaffold_func='bemis_murcko',
    test_size=0.2
)
```

**Rationale**: Tests generalization to chemically distinct scaffolds

#### Goal: Challenging OOD Validation
```python
# Research: Most conservative estimate
train, test = dataset.train_test_split(
    method='perimeter',
    n_clusters=25,
    test_size=0.2
)
```

**Rationale**: Test on chemical space perimeter (extrapolation)

#### Goal: Unbiased HPO Evaluation
```python
# Academic research: Publication-grade
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="nested_cv",
    outer_cv_folds=5,
    inner_cv_folds=3,
    enable_hpo=True
)
```

**Rationale**: Avoids optimistic bias from hyperparameter tuning

---

## Recommended Configurations

### By Dataset Size

| Dataset Size | Strategy | test_size | cv_folds | Rationale |
|--------------|----------|-----------|----------|-----------|
| < 50 | `cv_only` | — | 10 (LOOCV if < 30) | Maximize data usage |
| 50-100 | `cv_only` | — | 5 | Small but not tiny |
| 100-500 | `train_test` | 0.3 | 5 | Sufficient test set |
| 500-1000 | `train_test` | 0.25 | 5 | Balanced split |
| 1000-5000 | `train_test` or `train_val_test` | 0.2 or 0.15 | 5 | Standard config |
| 5000-10000 | `train_test` | 0.15 | 3-5 | Large enough for small test |
| > 10000 | `train_test` | 0.1 | 3 | Efficient validation |

### By Task Type

#### Regression Tasks
```python
# Standard configuration
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    task_type="regression",
    split_strategy="train_test",
    test_size=0.2,
    cv_folds=5,      # KFold (no stratification)
    random_state=42
)
```

#### Classification Tasks
```python
# Automatic stratification
results = universal_screen(
    dataset=dataset,
    target_column="class_label",
    task_type="classification",
    split_strategy="train_test",
    test_size=0.2,
    cv_folds=5,      # StratifiedKFold (maintains class balance)
    random_state=42
)
```

**Key Difference**: Classification uses StratifiedKFold automatically

### By Use Case

#### Virtual Screening
```python
# Conservative estimate for hit discovery
train, test = dataset.train_test_split(
    method='perimeter',  # Diverse test set
    n_clusters=25,
    test_size=0.2
)
```

#### Lead Optimization
```python
# SAR analysis within chemical series
train, clusters, y_train, y_clusters = dataset.train_test_split(
    method='lead_opt',
    lo_threshold=0.4,
    lo_min_cluster_size=5
)
```

#### Fragment-to-Lead
```python
# Generalization across molecular weight
train, test = dataset.train_test_split(
    method='molecular_weight',
    generalize_to_larger=True,  # Train small, test large
    test_size=0.25
)
```

---

## Reproducibility

### Fixed Random Seeds

**Always use fixed `random_state` for reproducibility:**

```python
# ✅ Good: Reproducible
results1 = universal_screen(dataset, "activity", random_state=42)
results2 = universal_screen(dataset, "activity", random_state=42)
assert (results1['test_indices'] == results2['test_indices']).all()

# ❌ Bad: Non-reproducible
results1 = universal_screen(dataset, "activity")  # random_state=None (default)
results2 = universal_screen(dataset, "activity")  # Different splits!
```

### What's Reproducible

✅ **Guaranteed reproducible** (with same `random_state`):
- Train/test split indices
- Cross-validation fold assignments
- Scaffold assignments (deterministic)
- Clustering results (with fixed random_state)
- Model training (if model has fixed seed)
- Final performance metrics

⚠️ **May vary slightly**:
- Training time (system load)
- Memory usage (Python GC)
- Floating-point rounding (hardware)

### Reproducibility Checklist

```{admonition} Reproducibility Checklist
:class: tip

**For Reproducible Experiments:**
- ✅ Fix `random_state` in splitting
- ✅ Fix random seeds in model training
- ✅ Document dataset version and preprocessing
- ✅ Record splitting strategy and parameters
- ✅ Save split indices for exact reproduction
- ✅ Report Python/package versions

**Example Documentation:**
```python
# Experiment configuration
config = {
    'dataset': 'molecules_v2.csv',
    'split_strategy': 'scaffold',
    'scaffold_func': 'bemis_murcko',
    'test_size': 0.2,
    'random_state': 42,
    'python_version': '3.9',
    'molblender_version': '0.3.0'
}
```
```

---

## Performance Optimization

### Memory Efficiency

**Dataset Size Impact:**

| Strategy | Memory Usage | Speed | Suitable Dataset Size |
|----------|--------------|-------|----------------------|
| `train_test` | Low (single split) | Fast | Any |
| `cv_only` | Medium (K models) | Medium | < 10K |
| `nested_cv` | High (K×K models) | Slow | 500-5000 |
| `scaffold` | Low (computation) | Fast | Any |
| `butina` | High (similarity matrix) | Slow | < 10K |
| `perimeter` | Medium (clustering) | Medium | < 50K |

**Tips for Large Datasets (>10K molecules):**

```python
# Use smaller test set
results = universal_screen(
    dataset=large_dataset,
    split_strategy="train_test",
    test_size=0.1,  # 10% instead of 20%
    cv_folds=3,     # 3 instead of 5
    random_state=42
)

# Or use clustering-based methods with reduced clusters
train, test = large_dataset.train_test_split(
    method='perimeter',
    n_clusters=50,  # More clusters for large dataset
    test_size=0.1
)
```

### Computational Cost

**Fastest Configurations:**
```python
# Minimal validation (quick baseline)
results = universal_screen(
    dataset=dataset,
    split_strategy="train_test",
    test_size=0.2,
    cv_folds=3,  # ~40% faster than 5-fold
    random_state=42
)
```

**Slowest Configurations:**
```python
# Comprehensive validation (research-grade)
results = universal_screen(
    dataset=dataset,
    split_strategy="nested_cv",
    outer_cv_folds=5,
    inner_cv_folds=5,  # 25 total model fits per model type
    enable_hpo=True
)
```

**Optimization Strategies:**
1. **Reduce CV folds**: 3-fold vs 5-fold saves ~40% time
2. **Smaller test set**: 0.15 vs 0.2 gives more training data, fewer test evaluations
3. **Skip HPO**: Stage 1 only is much faster
4. **Use clustering**: Butina/feature_clustering with fewer clusters
5. **Parallel processing**: Increase `max_cpu_cores` for fingerprint-based models

---

## Common Pitfalls

### 1. Data Leakage

❌ **Wrong: Using test set for hyperparameter tuning**
```python
# ❌ BAD: Tune on test set
model.set_params(alpha=best_alpha_from_test_set)  # LEAKAGE!
final_score = model.fit(X_train, y_train).score(X_test, y_test)
```

✅ **Correct: Use validation set or nested CV**
```python
# ✅ GOOD: Tune on validation set
results = universal_screen(
    dataset=dataset,
    split_strategy="train_val_test",  # Dedicated validation
    enable_hpo=True
)
```

### 2. Scaffold Leakage

❌ **Wrong: Random split with similar scaffolds**
```python
# ❌ BAD: Similar scaffolds in train/test
train, test = dataset.train_test_split(method='random', test_size=0.2)
# Many benzene derivatives in both train and test → optimistic R²
```

✅ **Correct: Scaffold split**
```python
# ✅ GOOD: Scaffold split ensures no overlap
train, test = dataset.train_test_split(method='scaffold', test_size=0.2)
# Benzene derivatives ONLY in train or ONLY in test → realistic R²
```

### 3. Small Test Sets

❌ **Wrong: Too small test set for meaningful evaluation**
```python
# ❌ BAD: 50-molecule dataset, test_size=0.2 → 10 test molecules!
results = universal_screen(
    dataset=small_dataset,  # n=50
    split_strategy="train_test",
    test_size=0.2,  # Only 10 test samples!
    random_state=42
)
# High variance in test score
```

✅ **Correct: Use CV-only for small datasets**
```python
# ✅ GOOD: CV-only for small datasets
results = universal_screen(
    dataset=small_dataset,  # n=50
    split_strategy="cv_only",
    cv_folds=5,  # Every sample used for validation
    random_state=42
)
```

### 4. Ignoring Stratification

❌ **Wrong: Imbalanced classification without stratification**
```python
# ❌ BAD: Manual split without stratification
# Class distribution: 90% class 0, 10% class 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Test set might have 0 samples of class 1!
```

✅ **Correct: Automatic stratification in MolBlender**
```python
# ✅ GOOD: Automatic StratifiedKFold for classification
results = universal_screen(
    dataset=imbalanced_dataset,
    task_type="classification",  # Auto-stratification
    split_strategy="train_test",
    test_size=0.2
)
# Both train and test maintain 90/10 class ratio
```

### 5. Inconsistent Splits for Comparison

❌ **Wrong: Different splits for comparing models**
```python
# ❌ BAD: Different random splits
results1 = universal_screen(dataset, "activity", random_state=42)
results2 = universal_screen(dataset, "activity", random_state=99)  # Different split!
# Can't fairly compare model performance!
```

✅ **Correct: Same random_state for all comparisons**
```python
# ✅ GOOD: Fixed random_state
results1 = universal_screen(dataset, "activity", random_state=42)
results2 = universal_screen(dataset, "activity", random_state=42)  # Same split
# Fair comparison
```

---

## Strategy Comparison Summary

### Performance Expectations

Typical R² values for the same dataset with different splits:

| Strategy | Expected R² | Interpretation |
|----------|-------------|----------------|
| Random | 0.78 | Optimistic (similar molecules in train/test) |
| Scaffold | 0.65 | Realistic (novel scaffolds) |
| Butina | 0.68 | Moderate (similar molecule clusters) |
| DNR (threshold) | 0.55 | Conservative (activity cliffs) |
| Perimeter | 0.62 | Conservative (diverse extrapolation) |
| MaxMin (unfriendly) | 0.50 | Very conservative (worst-case diversity) |

**Interpretation:**
- Higher R² doesn't mean better model, just easier test set
- Lower R² from challenging splits = more realistic estimate
- Use appropriate split for your deployment scenario

### When to Use Each Strategy

```{list-table}
:header-rows: 1
:widths: 20 30 50

* - Strategy
  - Best For
  - Avoid When
* - `train_test`
  - Standard ML baseline, quick screening
  - Dataset < 100 samples
* - `scaffold`
  - Novel scaffold generalization, drug discovery
  - Few unique scaffolds
* - `butina`
  - Similar molecule validation, cluster-based
  - Dataset < 100 or no similarity structure
* - `perimeter`
  - Diverse virtual screening, extrapolation
  - Dataset already very diverse
* - `dnr`
  - Activity cliff robustness, rough SAR
  - Smooth SAR landscape
* - `maxmin`
  - Chemical space coverage testing
  - No diversity in dataset
* - `nested_cv`
  - Unbiased HPO, academic research
  - Dataset < 500 (too slow)
* - `lead_opt`
  - SAR analysis, medicinal chemistry
  - No clear chemical series
```

---

## Recommended Workflow

### Standard Project Workflow

```python
# 1. Quick baseline (train_test)
baseline_results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_test",
    test_size=0.2,
    random_state=42
)
print(f"Baseline R²: {baseline_results['best_model']['test_r2']:.3f}")

# 2. Conservative validation (scaffold or perimeter)
if drug_discovery_context:
    train, test = dataset.train_test_split(method='scaffold', test_size=0.2)
else:
    train, test = dataset.train_test_split(method='perimeter', test_size=0.2)

conservative_results = universal_screen(
    dataset=train,  # Use split train set
    target_column="activity",
    random_state=42
)
print(f"Conservative R²: {conservative_results['best_model']['test_r2']:.3f}")

# 3. Report both
print(f"Performance range: {conservative_r2:.3f} - {baseline_r2:.3f}")
```

### Publication-Grade Workflow

```python
# For academic papers: Nested CV + conservative split
results_nestedcv = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="nested_cv",
    outer_cv_folds=5,
    inner_cv_folds=3,
    enable_hpo=True,
    random_state=42
)

# Report unbiased performance
print(f"Nested CV R² (unbiased): {results_nestedcv['nested_cv_r2_mean']:.3f} ± {results_nestedcv['nested_cv_r2_std']:.3f}")
```

---

## Related Documentation

- {doc}`index` - Overview of all splitting strategies
- {doc}`basic_strategies` - Standard ML splits
- {doc}`chemical_strategies` - Structure-based splits
- {doc}`property_strategies` - Property & diversity splits
- {doc}`advanced_strategies` - Advanced OOD splits
- {doc}`custom_splits` - User-provided splits

---

**Navigation**: {doc}`index` | Previous: {doc}`custom_splits`
