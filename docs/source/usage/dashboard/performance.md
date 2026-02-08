# Performance Analysis

Comprehensive guide to the Performance Analysis tab - understanding model comparisons, representation effectiveness, and performance distributions.

## Overview

The Performance Analysis tab is your first stop for understanding screening results. It provides:

- **Overview Cards** - Quick summary of dataset and best results
- **Model Comparison** - Average performance across representations
- **Representation Analysis** - Effectiveness grouped by modality
- **Performance Distributions** - Box plots, violin plots, histograms

**Best For:** Initial exploration, identifying top performers, comparing model families

## Tab Location

**Navigation:** Dashboard → Performance Analysis (Tab 1)

This is the default tab when launching the dashboard.

## Section 1: Overview Cards

Three information cards summarizing your screening results.

### Dataset Information Card

**What It Shows:**
```
┌─────────────────────────────────┐
│ Dataset Information             │
├─────────────────────────────────┤
│ Molecules: 1,247                │
│ Features: 1,024 (avg)           │
│ Task: Regression                │
│ Target: logP                    │
│ Primary Metric: R²              │
└─────────────────────────────────┘
```

**Key Information:**
- **Molecules** - Total dataset size
- **Features** - Average feature count across representations
- **Task** - Regression or Classification
- **Target** - Target variable name
- **Primary Metric** - Default comparison metric

**What to Look For:**
- Is the dataset size appropriate for your models?
- Are there enough molecules for reliable evaluation?
- Is the feature count consistent with your representations?

### Performance Summary Card

**What It Shows:**
```
┌─────────────────────────────────┐
│ Performance Summary             │
├─────────────────────────────────┤
│ Models Evaluated: 18            │
│ Representations: 6              │
│ Best Score: R² = 0.856          │
│ Mean Score: R² = 0.764          │
│ Std Dev: 0.089                  │
└─────────────────────────────────┘
```

**Key Metrics:**
- **Models Evaluated** - Total model-representation combinations tested
- **Representations** - Number of different molecular representations
- **Best Score** - Top performing result
- **Mean Score** - Average across all combinations
- **Std Dev** - Performance variability (lower = more consistent)

**Interpretation:**
- **Low std dev (<0.05)** - Consistent performance, most models work
- **Medium std dev (0.05-0.15)** - Some variation, model selection matters
- **High std dev (>0.15)** - Large differences, careful selection crucial

### Configuration Details Card

**What It Shows:**
```
┌─────────────────────────────────┐
│ Configuration                   │
├─────────────────────────────────┤
│ CV Folds: 5                     │
│ Test Size: 20%                  │
│ Random State: 42                │
│ Timestamp: 2025-01-15 10:30     │
└─────────────────────────────────┘
```

**Details:**
- **CV Folds** - Cross-validation folds used
- **Test Size** - Percentage held out for testing
- **Random State** - Seed for reproducibility
- **Timestamp** - When screening was completed

## Section 2: Model Comparison

Horizontal bar charts comparing average model performance across all representations.

### Chart Features

**Visual Elements:**
- **Bars** - Average performance across representations
- **Error Bars** - Standard deviation (performance variability)
- **Color** - Professional blue (#6BAED6)
- **Sorting** - Descending by average score

**Example Chart:**
```
RandomForest    ████████████████████░░ 0.842 ± 0.034
XGBoost         ███████████████████░░░ 0.831 ± 0.041
Ridge           ████████████████░░░░░░ 0.798 ± 0.028
Lasso           ███████████████░░░░░░░ 0.782 ± 0.031
KNN             ███████████░░░░░░░░░░░ 0.654 ± 0.087
                0.0                0.9
```

### Interpreting the Charts

**Long Bars = Better Performance**
- Models further right perform better on average
- Compare bar lengths to identify clear winners

**Short Error Bars = Consistent Performance**
- Narrow error bars mean the model works well across different representations
- Wide error bars suggest the model is sensitive to representation choice

**Example Interpretations:**

**Good Model:**
```
XGBoost  ████████████████████░ 0.851 ± 0.012
         ^                     ^       ^
         Long bar             High    Small error
         = Good avg           score   = Consistent
```

**Inconsistent Model:**
```
SVM      ████████████░░░░░░░░ 0.687 ± 0.154
         ^                    ^       ^
         Medium bar          Moderate Large error
         = Average            score   = Inconsistent
```

### Interactive Features

**Hover:**
- Move mouse over bars to see:
  - Exact mean score
  - Standard deviation
  - Number of representations tested
  - Min/max scores

**Switch Metrics:**
- Use sidebar dropdown to change from R² to RMSE, MAE, etc.
- Chart updates instantly with new rankings
- Error bars recalculate automatically

**Example Metric Switching:**
```python
# R² (higher is better)
RandomForest: 0.842
XGBoost: 0.831

# RMSE (lower is better)
XGBoost: 0.423
RandomForest: 0.456

# Rankings can change!
```

### What to Look For

✅ **Clear Winner** - One model significantly outperforms others
```
Best model bar is visibly longer than second-best
```

⚠️ **Close Competition** - Multiple models perform similarly
```
Top 3-5 models have overlapping error bars
Consider other factors: speed, interpretability
```

❌ **High Variance** - Large error bars across all models
```
May indicate:
- Dataset too small
- Representations not suitable
- Task difficulty
```

### Specific Model Analysis (Category Filter)

When you select a **model category** (anything other than "All Categories"), the dashboard
adds a **Specific Model Analysis** chart that breaks down **individual models within that
category**. Use this view to compare model distributions inside one family.

## Section 3: Representation Analysis

Horizontal bar charts showing representation effectiveness, grouped by modality.

### Chart Structure

**Grouping by Modality:**
```
VECTOR Representations:
  morgan_fp_r2_1024     ████████████████████░ 0.856
  rdkit_descriptors     ███████████████████░░ 0.842
  maccs_keys            ██████████████░░░░░░░ 0.731

STRING Representations:
  canonical_smiles      ████████████░░░░░░░░░ 0.678

MATRIX Representations:
  adjacency_matrix      ███████████████░░░░░░ 0.789
```

### Interpreting Representation Performance

**Best Representation Families:**
- Compare average performance within each modality
- VECTOR often performs best for traditional ML
- STRING competitive with large datasets + Transformers

**Representation-Specific Insights:**

**Fingerprints (VECTOR):**
- Morgan fingerprints often outperform other fingerprints
- Larger radius (r=3) may capture more structure
- Longer bit vectors (2048) vs shorter (1024) - check trade-offs

**Descriptors (VECTOR):**
- RDKit descriptors provide interpretable features
- Performance often close to fingerprints
- Faster to compute than deep learning embeddings

**Language Model Embeddings (VECTOR):**
- ChemBERTa, MolFormer pre-trained representations
- Excellent for transfer learning
- Slow to compute but high quality

**Raw Strings (STRING):**
- Only work with Transformer models
- Require large datasets (>1000 molecules)
- Computationally expensive

**Matrices (MATRIX):**
- Adjacency, Coulomb matrices
- Work with CNN models
- Good for capturing 3D structure

### Interactive Features

**Hover Details:**
- Representation name
- Average score across models
- Best model for this representation
- Feature count

**Filtering:**
- Click modality labels to show/hide groups
- Useful when many representations tested

**Metric Switching:**
- Change metric to see if representation rankings change
- Consistent top performers are more reliable

### What to Look For

**Within-Modality Comparison:**
```
Which fingerprint works best?
morgan_fp_r2_1024: 0.856
morgan_fp_r3_2048: 0.842
ecfp4: 0.831
```

**Cross-Modality Comparison:**
```
Is VECTOR better than STRING?
Best VECTOR: 0.856
Best STRING: 0.678
→ Yes, use fingerprints/descriptors
```

**Representation Diversity:**
```
Are results consistent across representation types?
If yes → Robust signal in data
If no → Task-specific representation needed
```

## Section 4: Performance Distribution Charts

Visualize the distribution of performance scores across all model-representation combinations.

### Available Chart Types

The dashboard provides multiple visualization options in this section.

### Box Plot

**What It Shows:**
```
      │                           ╭─────╮
      │              ┌────────────┤     ├─────┐
      │              │            ╰─────╯     │
      │          whisker        quartiles  whisker
      │              │            │  │  │     │
     ─┼──────────────┴────────────┴──┴──┴─────┴───
    0.0           0.5          0.7 0.8 0.9      1.0
                              Q1 Med Q3

Legend:
- Box: 25th to 75th percentile (IQR)
- Line in box: Median
- Whiskers: Min/max within 1.5×IQR
- Dots: Outliers beyond whiskers
```

**Use For:**
- Identifying median performance
- Detecting outliers
- Understanding quartile distribution

**Interpretation:**
- **Narrow box** - Consistent performance
- **Wide box** - High variability
- **Outliers below** - Some models perform very poorly
- **Outliers above** - Exceptional models

### Violin Plot

**What It Shows:**
```
Combines box plot with kernel density estimation:

      │         ╱╲
      │        ╱  ╲
      │       ╱    ╲      ← Density (width)
      │      ╱ ┌──┐ ╲
      │     ╱  │▓▓│  ╲    ← Box plot
      │    ╱   └──┘   ╲
      │   ╱           ╲
      │  ╱             ╲
     ─┼─┴───────────────┴─
    0.0               1.0
```

**Use For:**
- Understanding distribution shape
- Identifying multimodal distributions (multiple peaks)
- Comparing density across ranges

**Interpretation:**
- **Wide sections** - Many results in that range
- **Narrow sections** - Few results in that range
- **Multiple bulges** - Distinct performance clusters

### Histogram

**What It Shows:**
```
Binned frequency counts:

    12 ┤        ████
    10 ┤        ████
     8 ┤   ████ ████
     6 ┤   ████ ████ ████
     4 ┤   ████ ████ ████
     2 ┤████████ ████ ████ ████
     0 ┴────────────────────────
      0.5  0.6  0.7  0.8  0.9

X-axis: Performance bins
Y-axis: Count of models in each bin
```

**Use For:**
- Quick distribution overview
- Identifying most common performance ranges
- Spotting gaps in performance

**Interpretation:**
- **Tall bars** - Common performance level
- **Gaps** - No models in that range
- **Skewed distribution** - Performance concentrated on one side

### Scatter Plot (Many Models)

**What It Shows:**
```
For 100+ models, scatter plot with jitter:

    1.0 ┤              • •
    0.9 ┤          • • • • •
    0.8 ┤      • • • • • • • •
    0.7 ┤    • • • • • •
    0.6 ┤  • • •
    0.5 ┴───────────────────
        Model Index

Each dot: One model-representation combination
```

**Use For:**
- Overview of many results
- Spotting overall trends
- Identifying dense clusters

## Using Performance Analysis Effectively

### Workflow 1: Quick Assessment

**Goal:** Understand if screening was successful

1. **Check Performance Summary card**
   - Is best score acceptable? (e.g., R² > 0.7)
   - Is mean score reasonable?
   - Is std dev low enough (consistency)?

2. **Glance at Model Comparison**
   - Is there a clear winner?
   - Or multiple competitive models?

3. **Decision:**
   - Good results → Proceed to Model Inspection
   - Poor results → Check Representation Analysis for insights

### Workflow 2: Model Selection

**Goal:** Choose best model for deployment

1. **Sort by metric relevant to your application**
   - Accuracy → R², Pearson R
   - Error magnitude → RMSE, MAE
   - Rank correlation → Spearman ρ, Kendall τ

2. **Compare top 3 models in Model Comparison**
   - Check error bars (consistency)
   - Consider training time (if important)

3. **Cross-check with different metrics**
   - Switch to RMSE if using R²
   - Ensure top model is robust across metrics

4. **Inspect best model in Model Inspection tab**
   - Verify prediction quality visually
   - Check for systematic errors

### Workflow 3: Representation Optimization

**Goal:** Find best molecular representation for your dataset

1. **Review Representation Analysis chart**
   - Identify top-performing representations
   - Note modality (VECTOR vs STRING vs MATRIX)

2. **Check if top representations are from same family**
   - All Morgan fingerprints → Use fingerprints
   - Mixed results → Try ensemble of representations

3. **Consider computational cost**
   - Language model embeddings: High quality, slow
   - Fingerprints: Good quality, fast
   - Balance based on your needs

4. **Test hypothesis with new screening**
   - Re-run with only top representation family
   - Add more variants (different parameters)

### Workflow 4: Diagnosing Poor Performance

**Goal:** Understand why results are disappointing

1. **Check Distribution Charts**
   - Are all models poor? (Dataset issue)
   - Or only some models? (Model-representation mismatch)

2. **Review Model Comparison error bars**
   - Large error bars → Representation-sensitive models
   - Try more consistent models (low error bar)

3. **Examine Representation Analysis**
   - Is one modality much worse?
   - May need better representations

4. **Switch to Distribution Analysis tab**
   - Group by modality or model type
   - Identify patterns in failures

## Tips and Best Practices

```{admonition} Performance Analysis Tips
:class: tip

1. **Always start here** - Best tab for initial screening assessment
2. **Use metric switching** - Check consistency across R², RMSE, MAE
3. **Pay attention to error bars** - Consistency matters as much as mean
4. **Compare modalities** - VECTOR often best for small datasets
5. **Don't ignore distribution** - Outliers may indicate data issues
```

```{admonition} Common Pitfalls
:class: warning

- **Focusing only on best score** - Check consistency (error bars)
- **Ignoring representation families** - Some families consistently better
- **Not switching metrics** - Rankings may change significantly
- **Overlooking distribution shape** - May reveal data quality issues
```

## Exporting Performance Data

### Export Charts

**Method 1: Interactive Download**
1. Hover over chart
2. Click camera icon (top right of chart)
3. Saves as PNG (high resolution)

**Method 2: Browser Right-Click**
1. Right-click on chart
2. "Save image as..."
3. Choose location and filename

### Export Summary Statistics

Navigate to **Detailed Results tab** to export full data including:
- Model names and representations
- All performance metrics
- Training times
- Feature counts

Download as CSV for offline analysis.

## Next Steps

- **Understand distributions in depth**: {doc}`distributions` - 5 chart types with flexible grouping
- **Inspect specific models**: {doc}`model_inspection` - Prediction scatter plots and residuals
- **Get started**: {doc}`quickstart` - Launch the dashboard
