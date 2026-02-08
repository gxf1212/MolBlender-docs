# Screening Functions

Complete reference for MolBlender's screening functions - from quick evaluation to comprehensive multimodal screening.

## Overview

MolBlender provides multiple screening functions optimized for different use cases:

| **Function** | **Purpose** | **Models** | **Time** | **Use Case** |
|-------------|------------|-----------|---------|-------------|
| `simple_evaluate()` | Single model test | 1 model | <1 min | Quick baseline |
| `quick_screen()` | Fast essential screening | 5-10 | 2-5 min | Initial exploration |
| `universal_screen()` | Multimodal comprehensive | 15-30 | 10-60 min | **Recommended default** |
| `compare_models()` | Model comparison | Custom | 5-15 min | Model selection |
| `compare_representations()` | Representation comparison | Custom | 5-15 min | Feature selection |

## universal_screen()

**The primary screening function** - automatically detects data modalities and selects compatible models.

### Basic Usage

```python
from molblender.models.api import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="activity",
    task_type="regression"
)
```

### Complete API Reference

```python
def universal_screen(
    dataset,                              # Required
    target_column: str,                   # Required
    task_type: str = "regression",        # "regression" or "classification"
    modality_categories: List[str] = None,  # Auto-detect if None
    combinations: str = "auto",           # "auto", "comprehensive", or custom list
    primary_metric: str = None,           # Auto-select based on task_type
    cv_folds: int = 5,                    # Cross-validation folds
    test_size: float = 0.2,               # Test set proportion
    random_state: int = 42,               # Random seed
    verbose: int = 1,                     # 0=quiet, 1=normal, 2=detailed
    enable_feature_selection: bool = True,  # Zero-variance filtering
    enable_db_storage: bool = False,      # SQLite database storage
    db_path: str = None,                  # Database file path
    max_cpu_cores: int = -1,              # -1 = use all
    max_workers_per_model: int = 1,       # Parallelism per model
    execution_preference: str = "balanced"  # "speed", "memory", "balanced"
) -> Dict[str, Any]
```

### Parameters Explained

**Required Parameters**

`dataset`: `MolecularDataset`
: Input dataset with molecules and labels

`target_column`: `str`
: Name of the target variable column

**Task Configuration**

`task_type`: `str`, default=`"regression"`
: Type of ML task. Options:
  - `"regression"` - Continuous predictions (logP, solubility, binding affinity)
  - `"classification"` - Categorical predictions (active/inactive, ADMET classes)

`primary_metric`: `str`, optional
: Evaluation metric for model comparison. Auto-selected based on task_type if not specified.
  - Regression: `"r2"`, `"rmse"`, `"mae"`, `"pearson_r"`, `"spearman_rho"`, `"kendall_tau"`
  - Classification: `"f1"`, `"accuracy"`, `"roc_auc"`, `"precision"`, `"recall"`

**Modality Selection**

`modality_categories`: `List[str]`, optional
: Hierarchical category paths for representation selection. Auto-detected if `None`.
  ```python
  modality_categories=[
      "fingerprints/molecular",      # Molecular fingerprints
      "descriptors/physicochemical", # RDKit descriptors
      "sequential/language_model",   # Pre-trained embeddings
      "sequential/string",           # Raw SMILES for transformers
      "spatial/matrix",              # Adjacency/Coulomb matrices
      "image/2d"                     # 2D molecular images
  ]
  ```

`combinations`: `str` or `List`, default=`"auto"`
: Model selection strategy. Options:
  - `"auto"` **(recommended)** - Automatic selection based on modality
  - `"comprehensive"` - All compatible models including backup paths
  - `List[str]` - Specific model names: `["random_forest", "xgboost", "svm_rbf"]`
  - `List[Combination]` - Custom model-representation pairs

**Cross-Validation Settings**

`cv_folds`: `int`, default=`5`
: Number of cross-validation folds (3-10 recommended). See {doc}`methodology` for detailed explanation of the cross-validation protocol.

`test_size`: `float`, default=`0.2`
: Proportion of data for test set (0.1-0.3 recommended). This defines the train/test split ratio.

`random_state`: `int`, default=`42`
: Random seed for reproducibility. Ensures consistent train/test splits across runs.

```{seealso}
For a complete explanation of data splitting, cross-validation, and evaluation metrics, see {doc}`methodology`.
```

**Performance Options**

`max_cpu_cores`: `int`, default=`-1`
: Maximum CPU cores to use
  - `-1` = Use all available cores
  - `n` = Use n cores (e.g., `-2` leaves 2 cores free)

`max_workers_per_model`: `int`, default=`1`
: Parallelism within individual models (for sklearn models)

`execution_preference`: `str`, default=`"balanced"`
: Resource allocation strategy
  - `"speed"` - Maximize parallel execution
  - `"memory"` - Minimize memory usage
  - `"balanced"` - Balance speed and memory

**CUDA Fallback Behavior**

If a CUDA‑enabled model (VAE/CNN/Transformer/XGBoost) hits a CUDA error during training,
MolBlender will **fall back to CPU only when effective CPU cores ≥ 32**. If fewer
cores are available, the error is raised as usual. This keeps GPU failures from
spamming logs on large CPU machines while preserving strict failure on smaller nodes.

**Log Noise Suppression**

Repeated CUDA/CV/XGBoost failures and near‑constant prediction warnings are deduplicated
to avoid extremely long `screening.out` logs. The first occurrence is kept, subsequent
repeats are suppressed.

**Storage Options**

`enable_db_storage`: `bool`, default=`False`
: Enable SQLite database storage for incremental saving and caching

`db_path`: `str`, optional
: Path to SQLite database file (default: `"screening_results.db"`)

**Hyperparameter Optimization (HPO)**

`enable_hpo`: `bool`, default=`False`
: Enable two-stage hyperparameter optimization for top performers

`hpo_stage`: `str`, default=`"coarse"`
: HPO granularity level
  - `"coarse"` - Fast grid search (3-5 values per parameter)
  - `"fine"` - Detailed grid search (5-10 values per parameter)
  - `"custom"` - User-defined grid from parameter_grids.py

`hpo_method`: `str`, default=`"grid"`
: Search algorithm for hyperparameter optimization
  - `"grid"` - Exhaustive grid search (recommended)
  - `"random"` - Random search (faster for large grids)

`top_n_for_hpo`: `int`, default=`5`
: Number of top-performing models to optimize in Stage 2

`hpo_cv_folds`: `int`, default=`None`
: Cross-validation folds for HPO (defaults to `cv_folds` if not specified)

**Other Options**

`enable_feature_selection`: `bool`, default=`True`
: Remove zero-variance features (recommended)

`verbose`: `int`, default=`1`
: Logging verbosity
  - `0` = Errors only
  - `1` = Normal progress
  - `2` = Detailed debugging

### Return Value Structure

```python
{
    'success': True,                    # Overall success status
    'best_score': 0.852,                # Best primary metric value
    'best_model': {                     # Best performing model
        'model_name': 'random_forest',
        'representation_name': 'morgan_fp_r2_1024',
        'metrics': {
            'r2': 0.852,
            'rmse': 0.543,
            'mae': 0.421,
            'pearson_r': 0.924
        },
        'cv_scores': [0.831, 0.867, 0.849, 0.856, 0.857],
        'training_time': 12.34,
        'estimator': <trained_model>,   # Trained sklearn-compatible model
    },
    'results': [                        # All model results sorted by performance
        {...}, {...}, ...
    ],
    'summary': {                        # Statistical summary
        'n_models_evaluated': 18,
        'n_representations': 6,
        'mean_score': 0.764,
        'std_score': 0.089,
        'best_modality': 'VECTOR'
    },
    'database_path': './screening_results.db',  # If enable_db_storage=True
    'timestamp': '2025-01-15T10:30:45'
}
```

### Common Usage Patterns

#### Pattern 1: Default Screening

```python
# Automatic modality detection and model selection
results = universal_screen(
    dataset=dataset,
    target_column="logP"
)
```

#### Pattern 2: Custom Modality Selection

```python
# Screen only fingerprints and descriptors
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    modality_categories=[
        "fingerprints/molecular",
        "descriptors/physicochemical"
    ]
)
```

#### Pattern 3: Comprehensive Screening with Storage

```python
# All models with SQLite storage
results = universal_screen(
    dataset=dataset,
    target_column="bioactivity",
    combinations="comprehensive",
    enable_db_storage=True,
    db_path="./comprehensive_screen.db"
)
```

#### Pattern 4: Memory-Constrained Environment

```python
# Optimize for limited memory
results = universal_screen(
    dataset=large_dataset,
    target_column="solubility",
    execution_preference="memory",
    max_cpu_cores=4,
    cv_folds=3  # Fewer folds = less memory
)
```

#### Pattern 5: Classification Task

```python
# Binary classification with F1 optimization
results = universal_screen(
    dataset=dataset,
    target_column="is_drug_like",
    task_type="classification",
    primary_metric="f1"
)
```

#### Pattern 6: Two-Stage HPO for Top Performers

```python
# Automatic hyperparameter optimization for best models
results = universal_screen(
    dataset=dataset,
    target_column="binding_affinity",
    enable_hpo=True,
    hpo_stage="coarse",     # Fast grid search
    top_n_for_hpo=3,        # Optimize top 3 models
    hpo_cv_folds=3,         # Use 3-fold CV for HPO
    enable_db_storage=True  # Track optimization progress
)
```

### Modality Auto-Detection

MolBlender automatically detects available modalities from your dataset:

```python
# Dataset with SMILES only → AUTO detects STRING modality
dataset = MolecularDataset.from_csv("data.csv", input_column="SMILES")
# Will use: Transformers for raw strings + fingerprints/descriptors (computed on-the-fly)

# Dataset with pre-computed features → AUTO detects VECTOR modality
dataset.add_representation("morgan_fp_r2_1024")
dataset.add_representation("rdkit_descriptors_2d")
# Will use: Traditional ML models (RF, XGBoost, SVM, etc.)

# Dataset with adjacency matrices → AUTO detects MATRIX modality
dataset.add_representation("adjacency_matrix")
# Will use: CNN models + flattened vectors for ML
```

## quick_screen()

Fast screening with essential models for initial exploration.

### Basic Usage

```python
from molblender.models.api import quick_screen

results = quick_screen(
    dataset=dataset,
    target_column="activity",
    representations=["morgan_fp_r2_1024", "rdkit_descriptors_2d"]
)
```

### API Reference

```python
def quick_screen(
    dataset,
    target_column: str,
    representations: List[str] = None,  # Auto-select if None
    task_type: str = "regression",
    primary_metric: str = None,
    cv_folds: int = 3,                  # Fewer folds for speed
    test_size: float = 0.2,
    max_cpu_cores: int = -1
) -> Dict[str, Any]
```

**Key Differences from `universal_screen()`:**
- Tests only 5-10 essential models (RF, XGBoost, Ridge, Lasso, KNN)
- Uses 3 CV folds instead of 5 for speed
- No deep learning models (CNN, Transformers)
- Optimized for datasets < 10K molecules

### When to Use

✅ Initial data exploration
✅ Baseline performance assessment
✅ Small datasets (<1K molecules)
✅ Time-constrained scenarios
✅ Rapid prototyping

❌ Final model selection
❌ Large datasets
❌ Publication-quality results

## simple_evaluate()

Test a single model quickly without comprehensive screening.

### Basic Usage

```python
from molblender.models.api import simple_evaluate

result = simple_evaluate(
    dataset=dataset,
    target_column="activity",
    model_name="random_forest",
    representation_name="morgan_fp_r2_1024"
)
```

### API Reference

```python
def simple_evaluate(
    dataset,
    target_column: str,
    model_name: str = "random_forest",
    representation_name: str = None,    # Auto-select if None
    task_type: str = "regression",
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]
```

### When to Use

✅ Testing specific model hypothesis
✅ Benchmarking against literature
✅ Quick sanity checks
✅ Debugging

## compare_models()

Compare multiple models on the same representation.

### Basic Usage

```python
from molblender.models.api import compare_models

results = compare_models(
    dataset=dataset,
    target_column="activity",
    representation_name="morgan_fp_r2_1024",
    models=["random_forest", "xgboost", "svm_rbf", "ridge"]
)
```

### API Reference

```python
def compare_models(
    dataset,
    target_column: str,
    representation_name: str,
    models: List[str] = None,           # All compatible if None
    task_type: str = "regression",
    primary_metric: str = None,
    cv_folds: int = 5,
    statistical_tests: bool = True      # Significance testing
) -> Dict[str, Any]
```

**Additional Return Values:**
```python
{
    ...
    'statistical_comparison': {
        'friedman_test': {'statistic': 12.34, 'p_value': 0.002},
        'pairwise_comparisons': [
            {'models': ['rf', 'xgb'], 'p_value': 0.045, 'significant': True},
            ...
        ]
    }
}
```

## compare_representations()

Compare multiple representations on the same model.

### Basic Usage

```python
from molblender.models.api import compare_representations

results = compare_representations(
    dataset=dataset,
    target_column="activity",
    model_name="random_forest",
    representations=[
        "morgan_fp_r2_1024",
        "morgan_fp_r3_2048",
        "rdkit_descriptors_2d",
        "maccs_keys"
    ]
)
```

### API Reference

```python
def compare_representations(
    dataset,
    target_column: str,
    model_name: str,
    representations: List[str],
    task_type: str = "regression",
    primary_metric: str = None,
    cv_folds: int = 5,
    statistical_tests: bool = True
) -> Dict[str, Any]
```

## Advanced: Custom Combinations

For fine-grained control, use `Combination` objects:

```python
from molblender.models.api import universal_screen
from molblender.models.api.core import Combination

custom_combinations = [
    Combination(
        model_name="random_forest",
        representation_name="morgan_fp_r2_1024",
        model_params={'n_estimators': 200, 'max_depth': 10}
    ),
    Combination(
        model_name="xgboost",
        representation_name="rdkit_descriptors_2d",
        model_params={'learning_rate': 0.01, 'n_estimators': 500}
    )
]

results = universal_screen(
    dataset=dataset,
    target_column="activity",
    combinations=custom_combinations
)
```

## Error Handling

All screening functions handle errors gracefully:

```python
results = universal_screen(dataset, target_column="activity")

if not results.get('success', False):
    print(f"Screening failed: {results.get('error')}")
else:
    print(f"Screening succeeded: {results['n_models_evaluated']} models tested")
```

**Common Errors:**
- `KeyError: target_column` - Target column not found in dataset
- `ValueError: Invalid task_type` - Use "regression" or "classification"
- `MemoryError` - Reduce cv_folds, use execution_preference="memory"
- `TimeoutError` - Individual model timeout (auto-skipped, doesn't halt screening)

## Hyperparameter Optimization (HPO)

MolBlender implements an **intelligent two-stage HPO system** for efficient model tuning.

### Quick Overview

**Stage 1**: Screen all models with default parameters (~10-30 minutes)
**Stage 2**: Optimize top performers with GridSearchCV (~10-60 minutes)

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,           # Enable Stage 2 optimization
    hpo_stage="coarse",        # "coarse" | "fine" | "custom"
    hpo_method="grid",         # "grid" | "random"
    top_n_for_hpo=5,           # Optimize top 5 models
    hpo_selection_strategy="global",  # "global" | "per_type" | "per_subtype"
    hpo_cv_folds=3,            # Use 3-fold CV for HPO (faster)
    enable_db_storage=True     # Track all stages in database
)
```

### HPO Selection Strategies

`hpo_selection_strategy` controls which models are selected for Stage 2 optimization:

- **`"global"`** (default) - Top N models overall by primary metric
- **`"per_type"`** - Top N Traditional ML + Top N Deep Learning
- **`"per_subtype"`** - Top N from each model family (LINEAR, TREE, BOOSTING, KERNEL, VAE, TRANSFORMER, CNN)

```{admonition} For Complete HPO Documentation
:class: seealso

See {doc}`hpo` for comprehensive HPO guide including:
- Selection strategies and model subtypes
- Parameter grid configurations
- Custom grid definitions
- Database integration and resuming
- Performance tips and troubleshooting
```

## Performance Tips

```{admonition} Best Practices
:class: tip

1. **Start with `universal_screen()` default settings** - balanced performance
2. **Enable database storage** for runs > 10 minutes - `enable_db_storage=True`
3. **Use 3 CV folds for large datasets** (>10K molecules) - faster with minimal accuracy loss
4. **Monitor with `verbose=2`** when debugging - detailed progress logging
5. **Leave 2 cores free** for system - `max_cpu_cores=-2`
6. **Enable HPO for final models** - `enable_hpo=True` after initial screening
```

## Next Steps

- **See available models**: {doc}`models` - Complete model catalog
- **Access results**: {doc}`results` - SQLite database and exports
- **Visualize performance**: {doc}`../dashboard/index` - Interactive dashboard
