# Lead Sensitivity Analysis

Systematically evaluate how different lead molecule selections impact LOGO cross-validation performance.

## Overview

Lead Sensitivity Analysis helps answer the critical question: **"How does the choice of lead molecules affect model generalization?"**

This module provides comprehensive tools to:
- **Enumerate all possible lead combinations** (exhaustive or smart sampling)
- **Train and evaluate** models with different lead selections
- **Compare performance** across lead counts and combinations
- **Visualize results** with publication-quality plots
- **Store results** in structured SQLite database

```{admonition} Use Case
:class: tip
When developing predictive models for new molecular classes, you often have a few "lead molecules" from that class. This analysis quantifies how lead selection impacts your ability to predict the rest of that class.
```

## Quick Start

### Basic Usage

```python
from molblender.data import MolecularDataset, InputType
from molblender.models.api.lead_sensitivity import run_lead_sensitivity_analysis

# Load dataset
dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    mol_input_type=InputType.SMILES,
    label_columns=["activity"]
)

# Run analysis
results = run_lead_sensitivity_analysis(
    logo_results_dir="./results_logo_improved",
    dataset=dataset,
    target_column="activity",
    fold_names=["fold_A"],  # Or None for all folds
    max_leads=3,
    strategy="exhaustive",  # Test all combinations
    n_jobs=8
)

# Inspect results
for fold_name, fold_df in results.items():
    print(f"{fold_name}: {len(fold_df)} combinations tested")
    print(fold_df.groupby('lead_count')['pearson_r'].describe())
```

### With Visualization

```python
from molblender.models.api.lead_sensitivity_viz import (
    plot_performance_vs_lead_count,
    plot_top_bottom_combinations,
    plot_lead_heatmap
)

# Load results from database
import pandas as pd
import sqlite3

conn = sqlite3.connect("lead_sensitivity_results/lead_sensitivity.db")
df = pd.read_sql("SELECT * FROM lead_sensitivity_results WHERE fold_name='fold_A'", conn)

# Generate plots
plot_performance_vs_lead_count(
    df,
    metric='pearson_r',
    output_path="performance_vs_leads.png"
)

plot_top_bottom_combinations(
    df,
    metric='pearson_r',
    top_n=10,
    output_path="best_worst_leads.png"
)

plot_lead_heatmap(
    df,
    metric='pearson_r',
    output_path="lead_heatmap.png"
)
```

## Workflow Overview

```mermaid
graph TD
    A[LOGO Results] --> B[Extract Best Config]
    B --> C{Select Strategy}
    C -->|Exhaustive| D[Generate All C(n,k)]
    C -->|Random| E[Random Sample]
    C -->|Extreme| F[Highest/Lowest Target]
    C -->|Diverse| G[Greedy Distance]
    C -->|Representative| H[K-means Centroids]
    D --> I[Parallel Train & Evaluate]
    E --> I
    F --> I
    G --> I
    H --> I
    I --> J[Save to SQLite]
    J --> K[Generate Plots]
    K --> L[Analysis Complete]
```

**Process Details:**
1. **Extract Configuration**: Read best model/representation from LOGO `screening_results.db`
2. **Load Metadata**: Get train/test indices from `fold_metadata.json`
3. **Generate Combinations**: Create lead combinations using selected strategy
4. **Train Models**: For each combination, train model with selected leads + other groups
5. **Evaluate**: Test on remaining molecules from target group
6. **Store Results**: Save all metrics to SQLite with full provenance
7. **Visualize**: Generate publication-ready plots

## Selection Strategies

### Exhaustive (Default)

Test **all possible combinations** C(n,k) for k=min_leads to max_leads.

```python
results = run_lead_sensitivity_analysis(
    logo_results_dir="./results_logo_improved",
    dataset=dataset,
    target_column="activity",
    strategy="exhaustive",
    min_leads=1,
    max_leads=3  # Tests C(n,1), C(n,2), C(n,3)
)
```

**When to use:** Small groups (<20 molecules), comprehensive analysis needed

**Combinations generated:**
- n=16, max_leads=2 → 16 + 120 = **136 combinations**
- n=32, max_leads=3 → 32 + 496 + 4,960 = **5,488 combinations**

```{warning}
For n≥20 and max_leads≥4, exhaustive search becomes computationally expensive. Consider using random or diverse strategies.
```

### Random Sampling

Randomly sample a fixed number of combinations.

```python
results = run_lead_sensitivity_analysis(
    logo_results_dir="./results_logo_improved",
    dataset=dataset,
    target_column="activity",
    strategy="random",
    max_leads=3,
    sample_size=100,  # Sample 100 combinations per lead_count
    random_state=42
)
```

**When to use:** Large groups, exploratory analysis, time-constrained screening

### Extreme Values

Select leads with highest/lowest target values.

```python
results = run_lead_sensitivity_analysis(
    logo_results_dir="./results_logo_improved",
    dataset=dataset,
    target_column="activity",
    strategy="extreme",
    max_leads=3
)
```

**When to use:** Hypothesis testing (do extreme-value leads generalize better?)

**Logic:** For k leads, select k//2 with highest target and k//2 with lowest target

### Diverse Selection

Greedily select leads that maximize pairwise distances in feature space.

```python
results = run_lead_sensitivity_analysis(
    logo_results_dir="./results_logo_improved",
    dataset=dataset,
    target_column="activity",
    strategy="diverse",
    max_leads=3,
    sample_size=50  # Generate 50 diverse lead sets
)
```

**When to use:** Ensure chemical diversity in lead selection

**Algorithm:**
1. Select first lead randomly
2. For each subsequent lead, pick molecule with maximum minimum distance to existing leads
3. Repeat with different starting molecules

### Representative Selection

Use k-means clustering to identify representative molecules.

```python
results = run_lead_sensitivity_analysis(
    logo_results_dir="./results_logo_improved",
    dataset=dataset,
    target_column="activity",
    strategy="representative",
    max_leads=3
)
```

**When to use:** Identify structurally central molecules as leads

**Algorithm:** Run k-means with k clusters, select closest molecule to each centroid

## Function Reference

### run_lead_sensitivity_analysis()

Main entry point for lead sensitivity analysis.

```python
def run_lead_sensitivity_analysis(
    logo_results_dir: Union[str, Path],
    dataset: Union[MolecularDataset, str, Path],
    target_column: str,
    fold_names: Optional[List[str]] = None,
    strategy: str = "exhaustive",
    min_leads: int = 1,
    max_leads: int = 3,
    sample_size: Optional[int] = None,
    output_dir: Union[str, Path] = "./lead_sensitivity_results",
    n_jobs: int = -1,
    random_state: int = 42,
    force_recompute: bool = False,
    verbose: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Run lead sensitivity analysis on LOGO cross-validation folds.

    Parameters
    ----------
    logo_results_dir : str or Path
        Directory containing LOGO results (fold_* subdirectories with screening_results.db)
    dataset : MolecularDataset, str, or Path
        Dataset used in LOGO validation (needed to recompute features)
    target_column : str
        Name of target column for prediction
    fold_names : list of str, optional
        Specific folds to analyze (e.g., ['A_N', 'C']). If None, analyzes all folds.
    strategy : {'exhaustive', 'random', 'extreme', 'diverse', 'representative'}
        Lead combination selection strategy
    min_leads : int, default=1
        Minimum number of leads to test
    max_leads : int, default=3
        Maximum number of leads to test
    sample_size : int, optional
        For 'random' and 'diverse' strategies, number of combinations to sample per lead_count
    output_dir : str or Path, default='./lead_sensitivity_results'
        Output directory for database and visualizations
    n_jobs : int, default=-1
        Number of parallel workers (-1 uses all cores)
    random_state : int, default=42
        Random seed for reproducibility
    force_recompute : bool, default=False
        If True, recompute even if results exist in database
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=detailed)

    Returns
    -------
    results : dict[str, pd.DataFrame]
        Dictionary mapping fold_name to DataFrame of results with columns:
        - lead_count, lead_indices, lead_names
        - test_size, train_size
        - representation_name, model_name
        - pearson_r, rmse, mae, r2_score
        - predictions, true_values (JSON)
        - error_message (if failed)
    """
```

### Database Schema

Results are stored in `lead_sensitivity_results/lead_sensitivity.db`:

```sql
CREATE TABLE lead_sensitivity_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fold_name TEXT NOT NULL,
    lead_count INTEGER NOT NULL,
    lead_indices TEXT NOT NULL,  -- JSON: [0, 5, 12]
    lead_names TEXT NOT NULL,    -- "mol_A, mol_B, mol_C"
    test_size INTEGER,
    train_size INTEGER,
    representation_name TEXT,
    model_name TEXT,
    pearson_r REAL,
    rmse REAL,
    mae REAL,
    r2_score REAL,
    predictions TEXT,            -- JSON: [2.3, 1.8, ...]
    true_values TEXT,            -- JSON: [2.1, 1.9, ...]
    error_message TEXT,          -- NULL if success
    timestamp TEXT,
    UNIQUE(fold_name, lead_indices)
);
```

### Visualization Functions

All visualization functions follow publication standards (Times New Roman, 300 DPI).

#### plot_performance_vs_lead_count()

Box plot showing performance distribution across lead counts.

```python
from molblender.models.api.lead_sensitivity_viz import plot_performance_vs_lead_count

plot_performance_vs_lead_count(
    df: pd.DataFrame,
    metric: str = 'pearson_r',
    fold_name: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    show_scatter: bool = True,
    show_mean: bool = True
)
```

**Output:** Box plot with overlaid scatter points and annotated means

#### plot_top_bottom_combinations()

Horizontal bar chart comparing best and worst lead combinations.

```python
from molblender.models.api.lead_sensitivity_viz import plot_top_bottom_combinations

plot_top_bottom_combinations(
    df: pd.DataFrame,
    metric: str = 'pearson_r',
    top_n: int = 10,
    fold_name: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 8)
)
```

**Output:** Two-panel bar chart (top 10 best, top 10 worst)

#### plot_lead_heatmap()

2D heatmap for 2-lead combination performance matrix.

```python
from molblender.models.api.lead_sensitivity_viz import plot_lead_heatmap

plot_lead_heatmap(
    df: pd.DataFrame,
    metric: str = 'pearson_r',
    fold_name: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 9),
    cmap: str = 'RdYlGn'
)
```

**Output:** NxN heatmap showing all pairwise lead performance

#### plot_cross_fold_summary()

Grouped box plots comparing all folds side-by-side.

```python
from molblender.models.api.lead_sensitivity_viz import plot_cross_fold_summary

plot_cross_fold_summary(
    results_dict: Dict[str, pd.DataFrame],
    metric: str = 'pearson_r',
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6)
)
```

**Output:** Multi-fold comparison grouped by lead_count

## Command-Line Usage

The example script `run_lead_sensitivity_analysis.py` provides a full CLI:

```bash
python run_lead_sensitivity_analysis.py \
    --logo_dir ./results_logo_improved \
    --dataset ../data/molecules.csv \
    --target deltaG \
    --fold A_N C H \
    --strategy exhaustive \
    --max_leads 2 \
    --n_jobs 8 \
    --verbose 1
```

**Arguments:**
- `--logo_dir`: Path to LOGO results directory
- `--dataset`: Path to CSV dataset file
- `--target`: Target column name
- `--fold`: Fold names to analyze (space-separated, or omit for all)
- `--strategy`: {exhaustive, random, extreme, diverse, representative}
- `--min_leads`, `--max_leads`: Lead count range
- `--sample_size`: For random/diverse strategies
- `--n_jobs`: Parallel workers
- `--output_dir`: Output directory path
- `--verbose`: 0 (silent), 1 (progress), 2 (detailed)

## Performance Considerations

### Computational Cost

**Per Combination:**
- Feature extraction: 10-60s (depends on representation)
- Model training: 1-30s (depends on model/dataset size)
- Total per combination: **15-90 seconds**

**Total Time:**
- 100 combinations, 8 workers → **2-12 minutes**
- 500 combinations, 8 workers → **10-60 minutes**
- 5000 combinations, 8 workers → **2-8 hours**

```{tip}
Use `strategy='random'` with `sample_size=100-200` for initial exploration, then `strategy='exhaustive'` on promising folds.
```

### Parallelization

- Uses `joblib` with Loky backend
- Each worker handles one lead combination
- Recommended: `n_jobs = n_cores - 2` (leave 2 for OS)

### Caching

- Representation features are computed on-the-fly (not cached)
- Database uses UNIQUE constraint to prevent duplicate work
- If interrupted, rerun with same parameters to resume

## Example Workflow

### 1. Run Improved LOGO

```python
from molblender.models.api.multimodal import improved_logo_validation

results = improved_logo_validation(
    dataset=dataset,
    target_column="activity",
    fold_config={
        'A_N': {'test_group': 'A_N', 'lead_indices': [3]},
        'C': {'test_group': 'C', 'lead_indices': [20, 35]}
    },
    output_dir="./results_logo_improved"
)
```

### 2. Run Lead Sensitivity Analysis

```python
from molblender.models.api.lead_sensitivity import run_lead_sensitivity_analysis

sens_results = run_lead_sensitivity_analysis(
    logo_results_dir="./results_logo_improved",
    dataset=dataset,
    target_column="activity",
    fold_names=["A_N"],  # Start with one fold
    strategy="exhaustive",
    max_leads=2,
    n_jobs=8
)
```

### 3. Analyze Results

```python
import pandas as pd

fold_df = sens_results['A_N']

# Performance by lead count
print(fold_df.groupby('lead_count')['pearson_r'].describe())

# Best combination
best_row = fold_df.loc[fold_df['pearson_r'].idxmax()]
print(f"Best leads: {best_row['lead_names']}")
print(f"Pearson r: {best_row['pearson_r']:.3f}")

# Failed combinations
failed = fold_df[fold_df['error_message'].notna()]
print(f"Failed: {len(failed)}/{len(fold_df)}")
```

### 4. Generate Visualizations

```python
from molblender.models.api.lead_sensitivity_viz import (
    plot_performance_vs_lead_count,
    plot_top_bottom_combinations,
    plot_lead_heatmap
)

# Performance distribution
plot_performance_vs_lead_count(
    fold_df,
    metric='pearson_r',
    output_path="A_N_performance.png"
)

# Best/worst combinations
plot_top_bottom_combinations(
    fold_df,
    metric='pearson_r',
    top_n=10,
    output_path="A_N_combinations.png"
)

# Heatmap (for 2-lead analysis)
if fold_df['lead_count'].max() >= 2:
    plot_lead_heatmap(
        fold_df,
        metric='pearson_r',
        output_path="A_N_heatmap.png"
    )
```

## Interpreting Results

### Key Questions

1. **Does performance improve with more leads?**
   - Look at mean Pearson r across lead_count
   - Check if increase is statistically significant

2. **Which specific molecules make the best leads?**
   - Identify molecules appearing in top 10 combinations
   - Analyze their structural features

3. **Is there high variance?**
   - Large std → lead choice is critical
   - Small std → robust to lead selection

4. **Are some leads universally bad?**
   - Check molecules in bottom 10 combinations
   - May indicate outliers or annotation errors

### Statistical Analysis

```python
from scipy import stats

# Test if 2 leads significantly better than 1 lead
group1 = fold_df[fold_df['lead_count'] == 1]['pearson_r']
group2 = fold_df[fold_df['lead_count'] == 2]['pearson_r']
t_stat, p_val = stats.ttest_ind(group2, group1)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")

# Effect size (Cohen's d)
cohens_d = (group2.mean() - group1.mean()) / fold_df['pearson_r'].std()
print(f"Cohen's d: {cohens_d:.3f}")
```

## Common Patterns

### Pattern 1: Linear Improvement

**Observation:** Mean Pearson r increases linearly with lead_count

**Interpretation:** Each additional lead provides independent information

**Action:** Use as many leads as available

### Pattern 2: Diminishing Returns

**Observation:** Large jump from 1→2 leads, plateau after 2

**Interpretation:** First lead captures major signal, additional leads redundant

**Action:** 2 leads is optimal, save computational cost

### Pattern 3: High Variance

**Observation:** Wide box plots, many outliers

**Interpretation:** Lead choice is critical, some combinations are much better

**Action:** Use diverse/representative strategy to avoid bad combinations

### Pattern 4: No Improvement

**Observation:** Flat performance across lead counts

**Interpretation:** Either (1) leads don't help, or (2) test group too different

**Action:** Check fold definition, consider different grouping

## Troubleshooting

### All Combinations Fail

**Symptoms:** All error_message fields non-null

**Common Causes:**
1. Wrong dataset path or file format
2. Mismatched target_column name
3. Missing dependencies (sklearn, joblib)

**Solution:** Run with `verbose=2` to see detailed error logs

### Performance Worse Than Expected

**Symptoms:** Pearson r < 0.3 even with leads

**Common Causes:**
1. Test group is structurally very different from training groups
2. Target values are noisy or annotation errors
3. Best model from LOGO is overfitted

**Solution:** Check data quality, try different model/representation

### Memory Issues

**Symptoms:** Process killed, "Out of memory" errors

**Common Causes:**
1. Too many parallel workers on large dataset
2. Predictions/true_values arrays stored for all combinations

**Solution:** Reduce `n_jobs`, use `sample_size` to limit combinations

### Slow Execution

**Symptoms:** Taking >10 minutes per combination

**Common Causes:**
1. Heavy representation (e.g., ECIF3D, ChemBERTa embeddings)
2. Large dataset (>50K molecules)

**Solution:** Use lighter representation (e.g., Morgan fingerprints), reduce n_jobs

## API Reference

See {doc}`/api/models/lead_sensitivity` for complete API documentation.

## Next Steps

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card}
**→ LOGO Validation**
{doc}`screening` - Learn about LOGO cross-validation setup
:::

:::{grid-item-card}
**→ Model Selection**
{doc}`models` - Understand available models and compatibility
:::

:::{grid-item-card}
**→ Result Storage**
{doc}`results` - Work with SQLite database and exports
:::

:::{grid-item-card}
**→ Visualization**
{doc}`../dashboard/index` - Interactive result exploration
:::

::::

## Citation

If you use Lead Sensitivity Analysis in your research, please cite:

```bibtex
@software{molblender2024,
  title={MolBlender: Multi-Modal Molecular Representation Learning Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/MolBlender}
}
```
