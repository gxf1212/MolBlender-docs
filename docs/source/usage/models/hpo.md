# Hyperparameter Optimization (HPO)

Complete guide to MolBlender's two-stage hyperparameter optimization system for automated model tuning.

## Overview

MolBlender implements an **intelligent two-stage HPO workflow** that balances exploration speed with optimization quality:

- **Stage 1**: Screen all model-representation combinations with default parameters (~10-30 minutes)
- **Stage 2**: Optimize top performers with GridSearchCV or RandomizedSearchCV (~10-60 minutes)

This approach is **far more efficient** than optimizing every model upfront, especially when screening 20+ model-representation combinations.

## Quick Start

### Basic HPO Usage

```python
from molblender.models.api import universal_screen

# Enable HPO for top 5 models
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,            # Enable Stage 2 HPO
    top_n_for_hpo=5,            # Optimize top 5 models
    hpo_stage="coarse",         # Fast grid search
    enable_db_storage=True      # Track optimization progress
)
```

### View HPO Results

```python
# All results include both Stage 1 and Stage 2
print(f"Total models evaluated: {len(results['results'])}")
print(f"Stage 1 (default params): {sum(1 for r in results['results'] if r.get('stage') == 1)}")
print(f"Stage 2 (optimized): {sum(1 for r in results['results'] if r.get('stage') == 2)}")

# Best model (automatically from Stage 2 if HPO ran)
best = results['best_model']
print(f"Best: {best['model_name']} + {best['representation_name']}")
print(f"Optimized params: {best.get('best_params', {})}")
print(f"HPO CV score: {best.get('hpo_cv_score', 'N/A')}")
```

## Two-Stage Workflow

### Stage 1: Fast Screening with Defaults

**Purpose**: Identify promising model-representation combinations quickly

**Parameters**: Uses default/recommended hyperparameters for each model:
- `RandomForest`: `n_estimators=100`, `max_depth=None`
- `XGBoost`: `n_estimators=100`, `learning_rate=0.1`
- `SVM`: `C=1.0`, `gamma='scale'`
- etc.

**Output**: Ranked list of all tested combinations with baseline performance

**Duration**: 10-30 minutes (varies by dataset size and number of combinations)

```python
# Stage 1 only (no HPO)
results_stage1 = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=False,  # Default
    cv_folds=5
)

# Check Stage 1 results
for r in sorted(results_stage1['results'], key=lambda x: x['primary_metric'], reverse=True)[:5]:
    print(f"{r['model_name']:20s} + {r['representation_name']:20s} = {r['primary_metric']:.4f}")
```

### Stage 2: Targeted Optimization

**Purpose**: Fine-tune hyperparameters for top-performing models only

**Selection Strategies**: Three methods to choose which models to optimize:
1. **`"global"`**: Top N models overall (default, recommended)
2. **`"per_type"`**: Top N Traditional ML + Top N Deep Learning
3. **`"per_subtype"`**: Top N from each model family (LINEAR, TREE, BOOSTING, etc.)

**Search Methods**:
- **Grid Search** (default): Exhaustive search over parameter grid
- **Randomized Search**: Random sampling from parameter distributions (faster for large grids)

**Output**: Optimized models with best parameters and CV scores

**Duration**: 10-60 minutes (depends on grid size and `hpo_cv_folds`)

```python
# Stage 2: Optimize top 5 models
results_stage2 = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,
    hpo_selection_strategy="global",  # Top 5 overall
    top_n_for_hpo=5,
    hpo_stage="coarse",
    hpo_cv_folds=3,  # Faster HPO with 3-fold CV
    enable_db_storage=True
)
```

## HPO Configuration Parameters

### Selection Strategy

`hpo_selection_strategy`: `str`, default=`"global"`
: How to select models for Stage 2 optimization

  - **`"global"`** - Select top N models overall by primary metric (recommended)
    ```python
    # Example: Top 5 models regardless of type
    # Could be: 4 XGBoost + 1 Random Forest
    top_n_for_hpo=5, hpo_selection_strategy="global"
    ```

  - **`"per_type"`** - Select top N Traditional ML + top N Deep Learning
    ```python
    # Example: Top 3 Traditional ML + Top 3 Deep Learning = 6 total
    # Ensures both categories are optimized
    top_n_for_hpo=3, hpo_selection_strategy="per_type"
    ```

  - **`"per_subtype"`** - Select top N from each model family
    ```python
    # Example: Top 2 from each subtype (LINEAR, TREE, BOOSTING, etc.)
    # Could result in 2×7 = 14 models optimized
    top_n_for_hpo=2, hpo_selection_strategy="per_subtype"
    ```

**Model Subtypes**:
- **LINEAR**: Ridge, Lasso, ElasticNet, BayesianRidge
- **TREE**: RandomForest, ExtraTrees, DecisionTree
- **BOOSTING**: XGBoost, LightGBM, GradientBoosting, AdaBoost, CatBoost
- **KERNEL**: SVM (RBF/Linear/Poly), KNN
- **VAE**: VAE models (latent=64/128/256, compact, deep)
- **TRANSFORMER**: Transformer models (small/medium)
- **CNN**: Matrix CNN, Image CNN models
- **OTHER**: Neural networks (MLP, etc.)

```{admonition} Which Strategy to Use?
:class: tip

- **`"global"`**: Best for most cases - optimizes the absolute best performers
- **`"per_type"`**: Use when you want balanced coverage of Traditional ML and Deep Learning
- **`"per_subtype"`**: Use when exploring model diversity (e.g., comparing tree vs boosting vs linear)
```

### HPO Granularity

`hpo_stage`: `str`, default=`"coarse"`
: Hyperparameter grid resolution

  - **`"coarse"`** - Fast grid search (3-5 values per parameter)
    - Example: `n_estimators: [50, 100, 200]`
    - Duration: ~10-20 minutes for 5 models
    - **Recommended for initial optimization**

  - **`"fine"`** - Detailed grid search (5-10 values per parameter)
    - Example: `n_estimators: [50, 100, 150, 200, 300]`
    - Duration: ~30-60 minutes for 5 models
    - Use after identifying promising models with coarse search

  - **`"custom"`** - User-defined grids (advanced)
    - Edit `src/molblender/models/api/core/hpo/parameter_grids.py`
    - Full control over parameter ranges

### Search Method

`hpo_method`: `str`, default=`"grid"`
: Hyperparameter search algorithm

  - **`"grid"`** - GridSearchCV (exhaustive search, recommended)
    - Tests all parameter combinations
    - Guaranteed to find best combination in grid
    - Slower but thorough

  - **`"random"`** - RandomizedSearchCV (sampling-based)
    - Tests random subset of combinations
    - Faster for very large grids
    - Set `n_iter` to control number of samples

  - **`"optuna"`** - Optuna Bayesian optimization ⭐ NEW
    - Intelligent sampling using Tree-structured Parzen Estimator (TPE)
    - Efficient search focused on promising regions
    - MedianPruner for early stopping (aborts unpromising trials)
    - Warm-start from Grid Search best parameters (±50% range)
    - Ideal for:
      - Fine-tuning models with many hyperparameters
      - Slow-training models (Transformer, CNN) where Grid Search is expensive
      - Top 3 models from coarse grid that deserve focused optimization
    - **Usage**: Requires Optuna installation (`pip install optuna`)

```python
# Grid search (exhaustive)
enable_hpo=True, hpo_method="grid", hpo_stage="coarse"

# Random search (faster for large grids)
enable_hpo=True, hpo_method="random", hpo_stage="fine", n_iter=50

# Optuna (Bayesian optimization with pruning)
enable_hpo=True, hpo_method="optuna", hpo_stage="fine"
# Note: Falls back to Grid Search if Optuna not installed
```

#### Optuna Configuration (Advanced)

When using `hpo_method="optuna"`, Optuna-specific parameters apply:

- **`n_trials`**: Number of optimization trials (default: 50)
- **`timeout`**: Maximum optimization time in seconds (default: None)
- **`pruning`**: Enable MedianPruner for early stopping (default: True)

**Integration with Grid Search**:

Optuna is designed as a **Stage 3** fine-tuning step:

```python
# Recommended workflow: Grid (coarse) → Grid (fine) → Optuna (focused)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,
    hpo_method="grid",     # Stage 2
    hpo_stage="coarse",
    top_n_for_hpo=5,
    enable_db_storage=True,
    db_path="optimization.db"
)

# Then manually run Optuna on top 3 models using OptunaOptimizer
from molblender.models.api.core.hpo.optuna_optimizer import create_optimizer

optuna = create_optimizer(
    config=screening_config,
    n_trials=100,              # More trials for focused optimization
    pruning=True               # Early stopping enabled
)
```

**When to Use Optuna**:
- ✅ **Top models** (top 3) from coarse grid deserve focused optimization
- ✅ **Slow models** (Transformer, CNN) where each trial is expensive
- ✅ **Many hyperparameters** (10+) where Grid Search is impractical
- ❌ **Small datasets** (<500 molecules) - insufficient data for reliable Bayesian optimization

**Optuna vs Grid Search**:

| Aspect | Grid Search | Optuna |
|--------|-------------|--------|
| Speed | Fast for small grids, slow for large grids | Intelligent sampling, fewer trials needed |
| Completeness | Tests all combinations | Samples promising regions, may miss edge cases |
| Best Guarantee | Finds best in grid | Usually finds best, but not guaranteed |
| Ideal For | Small/medium grids, exhaustive search needed | Large grids, slow training models |

### Model Selection

`top_n_for_hpo`: `int`, default=`5`
: Number of models to optimize in Stage 2

  - **Recommended**: 3-5 for most cases
  - **Small datasets** (<1K molecules): 3 models sufficient
  - **Large datasets** (>10K molecules): 5-10 models for better coverage
  - **Very large grids** (`hpo_stage="fine"`): Reduce to 3 to save time

`hpo_models_per_type`: `int`, optional
: Override `top_n_for_hpo` for `per_type` and `per_subtype` strategies

  ```python
  # Select top 3 from each subtype (could be 3×7 = 21 models)
  hpo_selection_strategy="per_subtype",
  hpo_models_per_type=3
  ```

### Cross-Validation

`hpo_cv_folds`: `int`, optional (defaults to `cv_folds`)
: Number of CV folds for HPO grid search

  - **Recommended**: 3 (faster with minimal accuracy loss)
  - Uses same folds as Stage 1 if not specified
  - Reduce to 3 for Stage 2 even if Stage 1 uses 5

```python
# Stage 1 with 5-fold CV, Stage 2 with 3-fold CV
universal_screen(
    dataset=dataset,
    target_column="activity",
    cv_folds=5,          # Stage 1
    enable_hpo=True,
    hpo_cv_folds=3       # Stage 2 (faster)
)
```

## Parameter Grids

MolBlender provides pre-defined parameter grids for all supported models. Grids are automatically selected based on `hpo_stage`.

### Example Grids

#### Random Forest (Coarse)

```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'bootstrap': [True, False]
}
# Total combinations: 3 × 3 × 2 × 2 = 36
```

#### XGBoost (Coarse)

```python
{
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
# Total combinations: 3 × 3 × 3 × 2 × 2 = 108
```

#### SVM (Coarse)

```python
{
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf']
}
# Total combinations: 3 × 4 × 1 = 12
```

#### LightGBM (Coarse)

```python
{
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'num_leaves': [31, 63, 127],
    'max_depth': [5, 10, -1],
    'min_child_samples': [20, 50]
}
# Total combinations: 3 × 3 × 3 × 3 × 2 = 162
```

### Custom Parameter Grids

To define custom grids, edit:
```
src/molblender/models/api/core/hpo/parameter_grids.py
```

Example custom grid:

```python
# In parameter_grids.py
CUSTOM_GRIDS = {
    'random_forest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
}

# Then use in screening
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,
    hpo_stage="custom"  # Uses CUSTOM_GRIDS
)
```

## Database Integration

When `enable_db_storage=True`, HPO progress is tracked in SQLite:

### Schema Structure

```sql
-- Stage 1 results
SELECT model_name, representation_name, primary_metric, stage
FROM model_results
WHERE session_id = 'session_xyz' AND stage = 1
ORDER BY primary_metric DESC;

-- Stage 2 results (optimized)
SELECT model_name, representation_name, primary_metric, stage, best_params
FROM model_results
WHERE session_id = 'session_xyz' AND stage = 2
ORDER BY primary_metric DESC;
```

### Resume Interrupted HPO

When using an existing database, MolBlender automatically **skips combinations that already
have Stage 2 results** and **saves Stage 2 output after each model**. To resume, simply
re-run with the same `db_path` and `enable_db_storage=True`.

```python
from molblender.models.api.utils.results_db import ScreeningResultsDB

# Load previous results
db = ScreeningResultsDB("screening_results.db")
previous_results = db.load_comprehensive_results(session_id="session_xyz")

# Check what was already optimized
stage2_models = [r for r in previous_results if r.get('stage') == 2]
print(f"Already optimized: {len(stage2_models)} models")

# Continue optimization with different strategy
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,
    hpo_selection_strategy="per_subtype",  # Try different strategy
    db_path="screening_results.db",  # Reuse same database
    session_id="session_xyz_extended"  # New session ID
)
```

## Common Usage Patterns

### Pattern 1: Fast Initial Optimization

```python
# Quick exploration with coarse grid
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,
    hpo_stage="coarse",
    top_n_for_hpo=3,
    hpo_cv_folds=3,
    enable_db_storage=True
)
```

**Use Case**: Initial screening, small-medium datasets (<10K molecules)
**Duration**: ~20-30 minutes total

### Pattern 2: Comprehensive Fine-Tuning

```python
# Step 1: Coarse optimization
results_coarse = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,
    hpo_stage="coarse",
    top_n_for_hpo=5,
    enable_db_storage=True,
    db_path="optimization.db"
)

# Step 2: Fine-tune best model
best_model_name = results_coarse['best_model']['model_name']
best_repr_name = results_coarse['best_model']['representation_name']

results_fine = universal_screen(
    dataset=dataset,
    target_column="activity",
    combinations=[{
        'model_name': best_model_name,
        'representation_name': best_repr_name
    }],
    enable_hpo=True,
    hpo_stage="fine",
    db_path="optimization.db"
)
```

**Use Case**: Publication-quality results, final model deployment
**Duration**: ~60-90 minutes total

### Pattern 3: Balanced Model Coverage

```python
# Optimize top performers from each model family
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,
    hpo_selection_strategy="per_subtype",
    hpo_models_per_type=2,  # Top 2 from each subtype
    hpo_stage="coarse",
    hpo_cv_folds=3
)
```

**Use Case**: Model comparison studies, exploring algorithm diversity
**Duration**: ~40-60 minutes (optimizing 10-14 models)

### Pattern 4: Deep Learning Focus

```python
# Optimize Traditional ML and Deep Learning separately
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,
    hpo_selection_strategy="per_type",
    top_n_for_hpo=5,  # Top 5 Traditional ML + Top 5 DL = 10 total
    hpo_stage="coarse"
)
```

**Use Case**: Comparing traditional ML vs deep learning approaches
**Duration**: ~30-45 minutes

### Pattern 5: Large Dataset Optimization

```python
# Optimize with minimal CV folds for speed
results = universal_screen(
    dataset=large_dataset,  # >50K molecules
    target_column="activity",
    cv_folds=3,  # Stage 1
    enable_hpo=True,
    hpo_stage="coarse",
    top_n_for_hpo=3,
    hpo_cv_folds=2,  # Stage 2 - very fast
    enable_db_storage=True
)
```

**Use Case**: Very large datasets where time is critical
**Duration**: ~60-120 minutes

## Performance Considerations

### Time Estimates

| Configuration | Stage 1 | Stage 2 | Total | Dataset Size |
|--------------|---------|---------|-------|--------------|
| Fast (coarse, N=3, CV=3) | 10 min | 10 min | **20 min** | <1K molecules |
| Standard (coarse, N=5, CV=3) | 15 min | 20 min | **35 min** | 1-10K molecules |
| Comprehensive (fine, N=5, CV=5) | 30 min | 60 min | **90 min** | 10-50K molecules |
| Large dataset (coarse, N=3, CV=2) | 45 min | 30 min | **75 min** | >50K molecules |

### Memory Usage

HPO memory usage scales with:
- **Dataset size**: Larger datasets require more memory for CV folds
- **Grid size**: More parameter combinations = more parallel workers
- **Number of CV folds**: Each fold creates a copy of training data
 - **Representation caching**: HPO pre-computes required representations once, then reuses them

```{admonition} Memory Tips
:class: tip

- Use `hpo_cv_folds=3` instead of 5 (saves ~40% memory)
- Reduce `top_n_for_hpo` if running out of memory
- Use `hpo_stage="coarse"` for initial runs (smaller grids)
```

### CPU Utilization

GridSearchCV automatically parallelizes across CV folds:

```python
# Default: Uses all CPU cores for CV fold parallelism
n_jobs = -1  # Automatically set by sklearn

# To limit CPU usage:
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
```

### GPU Support

Some models automatically use GPU when available:
- **XGBoost**: Set `tree_method='gpu_hist'` in custom grids
- **LightGBM**: Set `device='gpu'` in custom grids
- **Deep Learning** (CNN, Transformers, VAE): Automatically use GPU via PyTorch

## Best Practices

```{admonition} HPO Best Practices
:class: tip

1. **Start with `hpo_stage="coarse"`** - Fast exploration before fine-tuning
2. **Use `hpo_cv_folds=3`** for Stage 2 - 30-40% faster with minimal accuracy loss
3. **Set `top_n_for_hpo=3-5`** - Good balance of coverage and time
4. **Enable database storage** - `enable_db_storage=True` for progress tracking
5. **Use `"global"` strategy first** - Optimize absolute best performers
6. **Monitor verbose output** - Watch for Stage 2 progression messages
7. **Save intermediate results** - Database stores all optimization attempts
8. **Fine-tune later** - Use coarse → fine workflow for best models
```

## Troubleshooting

### Problem: HPO Takes Too Long

**Solutions**:
1. Reduce `hpo_cv_folds` to 3 or 2
2. Use `hpo_stage="coarse"` instead of "fine"
3. Reduce `top_n_for_hpo` to 3
4. Use `hpo_method="random"` with `n_iter=50`

### Problem: Out of Memory During HPO

**Solutions**:
1. Reduce `hpo_cv_folds` (fewer parallel folds)
2. Reduce `top_n_for_hpo` (fewer models in parallel)
3. Use smaller parameter grids (custom grids)
4. Run Stage 2 with sequential execution instead of parallel

### Problem: No Improvement from HPO

**Possible Causes**:
1. Stage 1 already used near-optimal default parameters
2. Dataset too small for reliable hyperparameter tuning
3. Parameter grid doesn't include optimal values

**Solutions**:
1. Check if default parameters are already good (compare Stage 1 vs Stage 2)
2. Use larger datasets (>1K molecules recommended for HPO)
3. Expand parameter grids with custom grids

### Problem: NaNs in Descriptor Features

**Symptoms**:
- Warnings about NaN values in features
- Instability when using `rdkit_all_descriptors` or large descriptor sets

**Behavior**:
- HPO automatically **imputes NaNs with mean values** for sklearn compatibility

**Solutions**:
1. Clean or filter descriptors with high NaN rates
2. Use smaller descriptor sets when possible

### Problem: GridSearchCV Warnings

**Common Warnings**:
```
ConvergenceWarning: Maximum iterations reached
```
**Solution**: Add `max_iter` to parameter grid or increase default value

```
DataConversionWarning: A column-vector y was passed
```
**Solution**: Ignore - handled internally by MolBlender

## Next Steps

- **See screening functions**: {doc}`screening` - Complete API reference
- **View results**: {doc}`results` - Database access and exports
- **Interactive dashboard**: {doc}`../dashboard/index` - Visualize HPO results
- **Model catalog**: {doc}`models` - All supported models with default parameters
