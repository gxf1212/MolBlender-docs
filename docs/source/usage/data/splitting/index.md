# Dataset Splitting Strategies

Comprehensive guide to data splitting and cross-validation strategies in MolBlender.

## Overview

MolBlender provides **14 professional-grade splitting strategies** designed for molecular machine learning, ranging from standard ML practices to advanced drug discovery scenarios.

```{admonition} Key Features
:class: tip

- **14 splitting strategies** covering all common use cases
- **Fixed random seeds** for complete reproducibility
- **Automatic stratification** for classification tasks
- **Chemical structure-aware** splitting for realistic validation
- **Integration with universal_screen** for seamless model evaluation
```

## Quick Start

```python
from molblender.models import universal_screen
from molblender.data import MolecularDataset

# Load dataset
dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    label_columns=["activity"]
)

# Use default splitting (train_test with 80/20 split)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    task_type="regression",
    test_size=0.2,
    cv_folds=5,
    random_state=42
)
```

## Available Strategies

### Basic Strategies (Standard ML)
Standard machine learning splitting strategies for most use cases.

| Strategy | Use Case | Train/Val/Test | Documentation |
|----------|----------|----------------|---------------|
| `train_test` | Standard screening | 80% / — / 20% | {doc}`basic_strategies` |
| `train_val_test` | With HPO | 70% / 15% / 15% | {doc}`basic_strategies` |
| `nested_cv` | Unbiased HPO | Nested CV | {doc}`basic_strategies` |
| `cv_only` | Small datasets | CV only | {doc}`basic_strategies` |

**→ See {doc}`basic_strategies` for details**

### Chemical Structure-Based Strategies
Splitting based on molecular scaffolds and structural similarity.

| Strategy | Use Case | Best For | Documentation |
|----------|----------|----------|---------------|
| `scaffold` | Drug discovery | Novel structure generalization | {doc}`chemical_strategies` |
| `butina` | Cluster validation | Similar molecule generalization | {doc}`chemical_strategies` |
| `feature_clustering` | Custom representations | 3D embeddings, language models | {doc}`chemical_strategies` |

**→ See {doc}`chemical_strategies` for details**

### Property & Diversity Strategies
Splitting based on molecular properties and chemical space diversity.

| Strategy | Use Case | Best For | Documentation |
|----------|----------|----------|---------------|
| `dnr` | Rough SAR analysis | Testing on challenging molecules | {doc}`property_strategies` |
| `maxmin` | Diversity testing | Chemical space extrapolation | {doc}`property_strategies` |

**→ See {doc}`property_strategies` for details**

### Advanced Strategies (Splito-based)
Research-grade splitting for rigorous out-of-distribution evaluation.

| Strategy | Use Case | OOD Challenge | Documentation |
|----------|----------|---------------|---------------|
| `perimeter` | Virtual screening | Test on chemical space perimeter | {doc}`advanced_strategies` |
| `molecular_weight` | Fragment-to-lead | Generalization across MW ranges | {doc}`advanced_strategies` |
| `mood` | Deployment-aware | Optimized for target space | {doc}`advanced_strategies` |
| `lead_opt` | SAR exploration | Test on similar molecule clusters | {doc}`advanced_strategies` |

**→ See {doc}`advanced_strategies` for details**

### Custom Splits
User-defined splitting for specialized scenarios.

| Method | Use Case | Documentation |
|--------|----------|---------------|
| `custom` | User-provided | Temporal splits, external validation | {doc}`custom_splits` |

**→ See {doc}`custom_splits` for details**

## Choosing the Right Strategy

```{mermaid}
graph TD
    A[How much data?] --> B{< 100 samples}
    A --> C{100-500 samples}
    A --> D{500-5000 samples}
    A --> E{> 5000 samples}

    B --> F[cv_only]

    C --> G{Need HPO?}
    G -->|Yes| H[nested_cv]
    G -->|No| I[train_test<br/>test_size=0.3]

    D --> J{Need HPO?}
    J -->|Yes| K[train_val_test]
    J -->|No| L[train_test<br/>test_size=0.2]

    E --> M{Need HPO?}
    M -->|Yes| N[train_val_test]
    M -->|No| O[train_test<br/>test_size=0.15]

    style F fill:#90EE90
    style H fill:#FFB6C1
    style I fill:#87CEEB
    style K fill:#FFB6C1
    style L fill:#87CEEB
    style N fill:#FFB6C1
    style O fill:#87CEEB
```

**For drug discovery scenarios**, consider:
- **Novel scaffolds** → {doc}`chemical_strategies` (scaffold split)
- **Diverse libraries** → {doc}`advanced_strategies` (perimeter split)
- **Lead optimization** → {doc}`advanced_strategies` (lead_opt split)
- **Fragment-to-lead** → {doc}`advanced_strategies` (molecular_weight split)

**→ See {doc}`best_practices` for comprehensive decision guide**

## Quick Reference

### By Task Type

**Classification Tasks:**
- Automatic StratifiedKFold for balanced folds
- Stratified train/test split
- See: {doc}`basic_strategies`

**Regression Tasks:**
- Standard KFold cross-validation
- Regular train/test split
- See: {doc}`basic_strategies`

**Drug Discovery:**
- Scaffold-based splitting for novel structures
- Butina clustering for similar molecule validation
- See: {doc}`chemical_strategies`

**Research/Publications:**
- Nested CV for unbiased HPO evaluation
- Advanced splito methods for rigorous OOD testing
- See: {doc}`basic_strategies`, {doc}`advanced_strategies`

### By Dataset Size

| Dataset Size | Recommended Strategy | Split Ratio | CV Folds |
|--------------|---------------------|-------------|----------|
| < 100 | `cv_only` | — | 5 |
| 100-500 | `train_test` | 70/30 | 5 |
| 500-5000 | `train_test` or `train_val_test` | 80/20 or 70/15/15 | 5 |
| > 5000 | `train_test` | 85/15 or 90/10 | 3 |

**→ See {doc}`best_practices` for detailed recommendations**

## Documentation Structure

```{toctree}
:maxdepth: 2

basic_strategies
chemical_strategies
property_strategies
advanced_strategies
custom_splits
best_practices
```

## Key Concepts

### Reproducibility
All splitting strategies use **fixed random seeds** (default: `random_state=42`) to ensure complete reproducibility across different runs.

```python
# Run 1
results1 = universal_screen(dataset, "activity", random_state=42)

# Run 2 (different session)
results2 = universal_screen(dataset, "activity", random_state=42)

# Guarantee: Identical splits
assert (results1['test_indices'] == results2['test_indices']).all()
```

**→ See {doc}`best_practices` for reproducibility details**

### Stratification
Classification tasks automatically use **StratifiedKFold** to maintain class balance across folds, preventing fold-to-fold variance from class imbalance.

**→ See {doc}`basic_strategies` for stratification details**

### Chemical Awareness
Structure-based strategies ensure **no information leakage** from structural similarity:
- **Scaffold split**: No scaffold overlap between train/test
- **Butina split**: Leave-cluster-out validation
- **Perimeter split**: Test on chemical space perimeter

**→ See {doc}`chemical_strategies` and {doc}`advanced_strategies`**

## Integration with Universal Screening

All splitting strategies integrate seamlessly with `universal_screen`:

```python
from molblender.models import universal_screen

# Basic split
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_test",  # or any strategy
    test_size=0.2,
    random_state=42
)

# Advanced split
results = universal_screen(
    dataset=dataset,
    target_column="pIC50",
    split_strategy="scaffold",
    scaffold_func='bemis_murcko',
    test_size=0.2
)
```

## Getting Help

```{admonition} Need Help Choosing?
:class: tip

1. **Most users**: Start with {doc}`basic_strategies` (train_test split)
2. **Drug discovery**: Use {doc}`chemical_strategies` (scaffold split)
3. **Research validation**: Use {doc}`advanced_strategies` (perimeter or MOOD split)
4. **Custom needs**: See {doc}`custom_splits`

Still unsure? Check the {doc}`best_practices` decision tree.
```

## Related Documentation

- {doc}`../dataset` - Dataset management and loading
- {doc}`../../models/screening` - Model screening API
- {doc}`../../models/methodology` - Evaluation methodology

## References

- [Scikit-learn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Nested Cross-Validation Explained](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Splito Package](https://github.com/datamol-io/splito) - Advanced splitting methods
