# Practical Workflows

Real-world decision-making workflows using the MolBlender dashboard - from initial exploration to production model selection.

## Overview

This guide provides step-by-step workflows for common molecular ML tasks, demonstrating how to use the dashboard effectively for practical decisions.

**Workflows Covered:**
1. Finding the best production model
2. Comparing representation families
3. Debugging poor performance
4. Optimizing for deployment constraints
5. Virtual screening model selection
6. Iterative model improvement

## Workflow 1: Finding the Best Production Model

**Scenario:** You've completed screening and need to select a model for production deployment.

**Requirements:**
- Highest accuracy possible
- Reasonable training time (<10 minutes)
- Consistent performance across datasets

### Step-by-Step

**Step 1: Initial Assessment (Performance Analysis Tab)**

1. **Launch dashboard** → Performance Analysis tab
2. **Check Overview Cards:**
   ```
   Best Score: R² = 0.856
   Mean Score: R² = 0.764
   Std Dev: 0.089
   ```
3. **Quick Decision:**
   - Best score acceptable? (R² > 0.7 = Yes)
   - Std dev reasonable? (0.089 = Yes, good consistency)
   - Proceed to detailed analysis

**Step 2: Identify Top Candidates (Model Comparison)**

1. **Scroll to Model Comparison chart**
2. **Note top 3 models by average performance:**
   ```
   1. XGBoost:      0.842 ± 0.034
   2. RandomForest: 0.839 ± 0.028
   3. Ridge:        0.798 ± 0.021
   ```

3. **Consider error bars:**
   - RandomForest: Smallest error bar (0.028) = Most consistent
   - XGBoost: Slightly higher mean but more variable
   - Ridge: Lower mean but very consistent

**Step 3: Check Robustness (Switch Metrics)**

1. **Change metric to RMSE** (sidebar dropdown)
2. **Rankings update:**
   ```
   1. RandomForest: 0.421 ± 0.052
   2. XGBoost:      0.435 ± 0.068
   3. Ridge:        0.498 ± 0.034
   ```

3. **Change to Spearman ρ:**
   ```
   1. XGBoost:      0.912 ± 0.031
   2. RandomForest: 0.908 ± 0.025
   3. Ridge:        0.887 ± 0.018
   ```

4. **Conclusion:** XGBoost and RandomForest trade places, both robust

**Step 4: Detailed Inspection (Model Inspection Tab)**

1. **Switch to Model Inspection tab**
2. **Search for "xgboost morgan"** (best combination)
3. **Check prediction scatter plot:**
   - R² = 0.856 (matches overview)
   - Points cluster near y=x line ✓
   - Few outliers ✓
   - No systematic bias (residual plot centered at zero) ✓

4. **Repeat for "random_forest morgan":**
   - R² = 0.842 (slightly lower)
   - Similar scatter pattern
   - Also no systematic bias

**Step 5: Consider Training Time**

1. **Check model details:**
   ```
   XGBoost:      Training Time: 28.4 seconds
   RandomForest: Training Time: 12.1 seconds
   ```

2. **Decision factor:**
   - XGBoost: +0.014 R² but +16 seconds
   - RandomForest: Faster, nearly as accurate

**Step 6: Final Decision**

**Analysis:**
- **Accuracy:** XGBoost slightly better (0.856 vs 0.842)
- **Consistency:** RandomForest more stable (lower CV std)
- **Speed:** RandomForest 2.3× faster
- **Robustness:** Both rank high across metrics

**Decision: Choose RandomForest**
- Difference in R² (0.014) not meaningful
- 2× faster training valuable for retraining
- Better consistency (lower error bars)
- Simpler hyperparameters

**Step 7: Export and Deploy**

1. **In Model Inspection, click "Generate Reproduction Code"**
2. **Copy code to production script**
3. **Download model data CSV** for validation
4. **Document decision** in project notes

## Workflow 2: Comparing Representation Families

**Scenario:** You want to know which molecular representation type works best for your dataset.

### Step-by-Step

**Step 1: Overview (Performance Analysis → Representation Analysis)**

1. **Scroll to Representation Analysis chart**
2. **Note grouping by modality:**
   ```
   VECTOR Representations:
     morgan_fp_r2_1024:    0.856 (best overall)
     morgan_fp_r3_2048:    0.842
     rdkit_descriptors:    0.798
     maccs_keys:           0.731

   STRING Representations:
     canonical_smiles:     0.678

   MATRIX Representations:
     adjacency_matrix:     0.789
   ```

3. **Initial Observation:**
   - VECTOR modality dominates
   - Morgan fingerprints best within VECTOR
   - STRING (Transformers) underperforms

**Step 2: Detailed Comparison (Distribution Analysis Tab)**

1. **Switch to Distribution Analysis tab**
2. **Select Box Plot chart type**
3. **Group by: Modality**

4. **Observe distributions:**
   ```
   VECTOR:  Median = 0.82, IQR = 0.10 (consistent)
   STRING:  Median = 0.65, IQR = 0.08 (lower but consistent)
   MATRIX:  Median = 0.75, IQR = 0.15 (variable)
   ```

5. **Conclusion:** VECTOR most reliable

**Step 3: Within-VECTOR Comparison**

1. **Switch grouping to: Representation**
2. **Filter to show only VECTOR representations** (use search)

3. **Compare fingerprint types:**
   ```
   Morgan FP r=2: Best median, narrow IQR
   Morgan FP r=3: Slightly lower, wider IQR
   RDKit Desc:    Lower median, narrow IQR
   MACCS Keys:    Lowest, widest IQR
   ```

4. **Insight:** Morgan r=2 optimal balance of performance and consistency

**Step 4: Check Across Models**

1. **Return to Performance Analysis tab**
2. **Export Detailed Results to CSV**
3. **In Python:**
   ```python
   import pandas as pd

   df = pd.read_csv('results.csv')

   # Average performance by representation
   repr_perf = df.groupby('representation_name')['r2'].agg(['mean', 'std', 'count'])
   print(repr_perf.sort_values('mean', ascending=False))

   # Output:
   #                      mean   std  count
   # morgan_fp_r2_1024   0.812  0.052    6
   # morgan_fp_r3_2048   0.798  0.068    6
   # rdkit_descriptors   0.756  0.048    6
   # maccs_keys          0.698  0.089    6
   # canonical_smiles    0.632  0.076    2
   ```

5. **Conclusion:** Morgan FP r=2 1024-bit best across all models

**Step 5: Cost-Benefit Analysis**

```
Morgan r=2 1024:
  Performance: R² = 0.812 (avg)
  Compute time: <1s per molecule
  Memory: 1024 features
  → RECOMMENDED

Morgan r=3 2048:
  Performance: R² = 0.798 (avg, -0.014)
  Compute time: <2s per molecule
  Memory: 2048 features (2× more)
  → Not worth 2× cost for 1.4% loss

RDKit Descriptors:
  Performance: R² = 0.756 (avg, -0.056)
  Compute time: ~1s per molecule
  Memory: ~200 features
  → Consider if interpretability critical

Transformers:
  Performance: R² = 0.632 (avg, -0.180)
  Compute time: ~30s per molecule (30× slower!)
  Memory: Large model (GB)
  → NOT recommended for this dataset size
```

**Final Recommendation:** Use Morgan Fingerprint radius=2, 1024 bits

## Workflow 3: Debugging Poor Performance

**Scenario:** Screening results are disappointing (best R² < 0.5). You need to diagnose the problem.

### Step-by-Step

**Step 1: Assess Severity (Performance Analysis → Overview Cards)**

```
Best Score: R² = 0.423
Mean Score: R² = 0.287
Std Dev: 0.145
```

**Red flags:**
- Best score very low (< 0.5)
- High standard deviation (0.145) = Inconsistent
- Some models may be failing completely

**Step 2: Check Distribution (Distribution Analysis Tab)**

1. **Select Histogram chart type**
2. **No grouping**

3. **Observe distribution:**
   ```
   Count
    8 ┤           ████
    6 ┤      ████ ████
    4 ┤ ████ ████ ████ ████
    2 ┤ ████ ████ ████ ████ ████
    0 ┴────────────────────────────
     0.0  0.1  0.2  0.3  0.4  0.5
   ```

4. **Interpretation:**
   - Wide spread (0.0 to 0.5)
   - Some models near zero (complete failure)
   - Best models only moderate (0.4-0.5)

**Step 3: Identify Failure Patterns (Group by Model Type)**

1. **Switch to Box Plot**
2. **Group by: Model Type**

3. **Results:**
   ```
   Transformers:  Median = 0.08 (failed)
   CNNs:          Median = 0.12 (failed)
   RandomForest:  Median = 0.38 (poor)
   XGBoost:       Median = 0.42 (best, but still poor)
   Ridge:         Median = 0.35 (poor)
   ```

4. **Diagnosis:**
   - Deep learning models failed completely
   - Traditional ML models poor but not failed
   - Problem likely: dataset or representation issue

**Step 4: Check Representations (Group by Representation)**

1. **Change grouping to: Representation**

2. **Results:**
   ```
   canonical_smiles:   Median = 0.09 (Transformers failed)
   adjacency_matrix:   Median = 0.11 (CNNs failed)
   morgan_fp_r2_1024:  Median = 0.38
   rdkit_descriptors:  Median = 0.41 (best)
   maccs_keys:         Median = 0.32
   ```

3. **Diagnosis:**
   - STRING and MATRIX modalities failed
   - VECTOR modality works but performance poor
   - Confirms: Deep learning not suitable for this data

**Step 5: Inspect Best Model (Model Inspection Tab)**

1. **Search for best combination:** "xgboost rdkit_descriptors"
2. **Load prediction scatter plot**

3. **Observations:**
   ```
   R² = 0.423 (matches overview)

   Scatter plot shows:
   - Random scatter, weak pattern
   - Many points far from y=x line
   - Heteroscedastic (funnel pattern)
   ```

4. **Residual plot:**
   - Non-random pattern
   - Curved residual vs predicted
   - Suggests non-linear relationship not captured

**Step 6: Root Cause Analysis**

**Possible causes:**
1. **Insufficient data** - Check dataset size
2. **Poor quality data** - Measurement errors, inconsistent labels
3. **Intrinsic difficulty** - Task may be very hard
4. **Missing features** - Important structural information not captured
5. **Wrong task framing** - Maybe classification instead of regression?

**Diagnostic checks:**
```python
# Load original dataset
import pandas as pd
df = pd.read_csv('molecules.csv')

# Check size
print(f"Dataset size: {len(df)}")
# Output: 127 molecules → TOO SMALL!

# Check target distribution
print(df['activity'].describe())
# Output:
#   count    127
#   mean     2.34
#   std      3.87  → High variance
#   min     -1.23
#   max     15.67  → Possible outliers

# Check for duplicates
print(f"Duplicates: {df.duplicated().sum()}")
# Output: 23 duplicates → DATA QUALITY ISSUE
```

**Step 7: Action Plan**

**Based on diagnosis:**

**Problem 1: Dataset Too Small (127 molecules)**
- **Action:** Collect more data (target: >500 molecules)
- **Alternative:** Use simpler models (Ridge), avoid overfitting

**Problem 2: Data Quality Issues (duplicates, outliers)**
- **Action:** Clean dataset, remove duplicates and outliers
- **Expected improvement:** +10-20% R²

**Problem 3: High Target Variance**
- **Action:** Consider log transformation of target
- **Or:** Use robust metrics (MAE, Spearman ρ)

**Problem 4: Non-linear Relationship**
- **Action:** Try kernel-based models (SVM, KNN)
- **Or:** Add interaction features

**Step 8: Rerun and Compare**

```python
# After data cleaning and increasing dataset to 543 molecules
results_new = universal_screen(dataset_cleaned, target_column='activity')
```

**New results:**
```
Best Score: R² = 0.782 (+0.359!)
Mean Score: R² = 0.721 (+0.434!)
Std Dev: 0.067 (-0.078, more consistent!)

→ Problem solved: Data quality + dataset size
```

## Workflow 4: Optimizing for Deployment Constraints

**Scenario:** You need a model that trains in <30 seconds and predicts in <1ms per molecule for real-time web application.

### Step-by-Step

**Step 1: Check Timing Data (Advanced Features → Efficiency Analysis)**

1. **Locate Training Time vs Performance scatter plot**
2. **Identify models in top-left quadrant** (fast + accurate)

**Step 2: Filter by Time Constraint**

1. **Export results to CSV**
2. **Filter in Python:**
   ```python
   import pandas as pd

   df = pd.read_csv('results.csv')

   # Filter by training time <30s
   fast_models = df[df['training_time'] < 30]

   # Sort by R² descending
   fast_models = fast_models.sort_values('r2', ascending=False)

   print(fast_models[['model_name', 'representation_name', 'r2', 'training_time']].head(10))
   ```

3. **Top fast models:**
   ```
   Model             Representation      R²     Time(s)
   RandomForest      morgan_fp_r2       0.842   12.1
   Ridge             rdkit_desc         0.798    0.8
   Lasso             morgan_fp_r2       0.782    1.2
   ExtraTrees        morgan_fp_r2       0.839   8.3
   KNN               rdkit_desc         0.754    0.3
   ```

**Step 3: Check Prediction Speed**

Prediction speed depends on:
- **Model complexity** - Linear models fastest
- **Feature count** - Fewer features faster

**Estimate prediction times:**
```
Ridge (200 features):     <0.1ms per molecule ✓
RandomForest (1024 feat): ~0.5ms per molecule ✓
XGBoost (1024 feat):      ~0.8ms per molecule ✓
Transformer:              ~50ms per molecule ✗
```

**Step 4: Balance Accuracy vs Speed**

**Option A: Maximum Speed (Ridge)**
- Training: 0.8s
- Prediction: <0.1ms
- R²: 0.798
- **Use if:** Speed critical, 0.8 R² acceptable

**Option B: Balanced (RandomForest)**
- Training: 12.1s
- Prediction: ~0.5ms
- R²: 0.842
- **Use if:** Can tolerate 0.5ms, want better accuracy

**Option C: Best Accuracy (within constraint, ExtraTrees)**
- Training: 8.3s
- Prediction: ~0.4ms
- R²: 0.839
- **Use if:** Similar to RandomForest, slightly faster training

**Decision: Choose RandomForest**
- Best accuracy within constraints
- Prediction speed acceptable (0.5ms << 1ms requirement)
- Training time comfortable (12s << 30s limit)

**Step 5: Verify and Optimize**

```python
# Load and test best model
import joblib
from molblender.representations import get_featurizer
import time

model = results['best_estimator']  # RandomForest

# Prepare test molecules
featurizer = get_featurizer('morgan_fp_r2_1024')
X_test = featurizer.transform(test_smiles)

# Benchmark prediction speed
start = time.time()
predictions = model.predict(X_test)
end = time.time()

per_molecule = (end - start) / len(test_smiles) * 1000  # ms
print(f"Prediction time: {per_molecule:.2f}ms per molecule")
# Output: 0.43ms per molecule ✓

# Verify meets requirement
assert per_molecule < 1.0, "Too slow!"
assert results['best_score'] > 0.8, "Not accurate enough!"

# Save for production
joblib.dump(model, 'production_model.pkl')
```

## Workflow 5: Virtual Screening Model Selection

**Scenario:** You're doing virtual screening where ranking molecules correctly is more important than absolute predictions.

### Step-by-Step

**Step 1: Switch to Ranking Metrics (Performance Analysis)**

1. **Sidebar → Select metric: Spearman ρ** (rank correlation)
2. **Charts update to show ranking performance**

3. **Top models by Spearman:**
   ```
   XGBoost + morgan_fp:     ρ = 0.912
   RandomForest + morgan:   ρ = 0.908
   SVM_RBF + rdkit_desc:    ρ = 0.898
   ```

**Step 2: Verify with Kendall τ**

1. **Switch to Kendall τ** (more conservative)
2. **Top models:**
   ```
   XGBoost + morgan_fp:     τ = 0.765
   RandomForest + morgan:   τ = 0.758
   SVM_RBF + rdkit_desc:    τ = 0.742
   ```

3. **Consistent:** Same top 3 models

**Step 3: Compare with Absolute Metrics**

1. **Switch to R²:**
   ```
   XGBoost:      R² = 0.856
   RandomForest: R² = 0.842
   SVM:          R² = 0.798
   ```

2. **Analysis:**
   - XGBoost best for both ranking and absolute prediction
   - SVM better at ranking (ρ=0.898) than absolute (R²=0.798)
   - For virtual screening, ranking matters most

**Step 4: Enrichment Analysis (Model Inspection)**

1. **Search for "xgboost morgan"**
2. **Load prediction scatter plot**
3. **Focus on high-value region** (top 10% true values)

4. **Check:** Are high-true-value molecules ranked high by model?
   - Zoom into top-right corner
   - Look for points near y=x line at high values
   - Few points in top-right but predicted low = Poor enrichment

5. **Conclusion:** XGBoost successfully ranks high-value molecules high

**Step 5: Top-N Accuracy**

**Calculate:** Of top 100 predictions, how many are truly active?

```python
import pandas as pd
import numpy as np

# Load predictions
model_data = load_model_predictions('xgboost_morgan')
y_true = model_data['test_true']
y_pred = model_data['test_predictions']

# Top 100 by prediction
top_100_idx = np.argsort(y_pred)[-100:]

# Of those, how many are truly in top 100?
true_top_100 = set(np.argsort(y_true)[-100:])
pred_top_100 = set(top_100_idx)

overlap = len(true_top_100 & pred_top_100)
print(f"Enrichment: {overlap}/100 = {overlap}%")
# Output: 87/100 = 87% enrichment

# Random would be: 10% (100/1000)
# Enrichment factor: 87/10 = 8.7×
```

**Decision: Choose XGBoost**
- Highest Spearman ρ and Kendall τ
- 87% top-100 enrichment (8.7× vs random)
- Also good absolute predictions (R²=0.856)

## Common Decision Patterns

### Pattern 1: Clear Winner

**Situation:**
- One model significantly better across all metrics
- Low variance, high performance
- No deployment constraints

**Action:** Choose the clear winner, proceed to deployment

### Pattern 2: Close Competition

**Situation:**
- Top 2-3 models within 2% R²
- Different strengths (speed vs accuracy)

**Action:**
1. Check consistency (CV std, residual plots)
2. Consider deployment requirements
3. Choose fastest/simplest if accuracy similar

### Pattern 3: Metric-Dependent Rankings

**Situation:**
- Different models best for different metrics
- R² winner ≠ RMSE winner ≠ Spearman winner

**Action:**
1. Identify application priority (ranking? absolute accuracy?)
2. Choose model that excels at priority metric
3. Verify acceptable on other metrics

### Pattern 4: All Models Poor

**Situation:**
- Best R² < 0.5
- High variance across models
- Deep learning failed

**Action:**
1. Diagnose: Data quality? Size? Representation?
2. Fix root cause (see Workflow 3)
3. Rerun screening
4. If still poor: Task may be intrinsically difficult

## Summary Checklist

Before finalizing model selection, verify:

✅ **Performance:**
- R² > 0.7 (or task-appropriate threshold)
- Consistent across multiple metrics
- Low CV standard deviation (<0.10)

✅ **Robustness:**
- No systematic bias (residual plot)
- Few outliers or outliers explainable
- Performs well on validation set

✅ **Practical:**
- Training time acceptable
- Prediction speed meets requirements
- Model complexity manageable

✅ **Documentation:**
- Reproduction code exported
- Model data saved
- Decision rationale documented
- Hyperparameters recorded

## Next Steps

- **Get started with dashboard**: {doc}`quickstart` - Installation and first launch
- **Understand visualizations**: {doc}`performance` - Performance analysis guide
- **Learn about metrics**: {doc}`metrics` - Complete metric reference
- **Master distributions**: {doc}`distributions` - Distribution analysis techniques
- **Inspect models**: {doc}`model_inspection` - Prediction scatter plots and residuals
- **Advanced techniques**: {doc}`advanced` - Statistical analysis and multi-session comparison
