# Metrics Guide

Comprehensive guide to all 42 evaluation metrics supported by the dashboard - when to use each, how to interpret, and what they tell you about model performance.

## Overview

The MolBlender dashboard supports **42 performance metrics** across regression, classification, and cross-validation tasks.

**Metric Categories:**
- **Regression** (15 metrics) - Continuous value prediction
- **Classification** (18 metrics) - Categorical prediction
- **Cross-Validation** (9 metrics) - Training stability metrics

**Dynamic Switching:** Change metrics instantly from sidebar dropdown - all charts update in real-time.

## Metric Selector

**Location:** Dashboard sidebar (top section, sticky)

```
┌────────────────────────────────┐
│ Primary Metric:                │
│ [R² (Coefficient of Determ▼]   │
└────────────────────────────────┘
```

**All charts update when you select a new metric** - no page reload needed.

## Regression Metrics

For predicting continuous molecular properties (logP, solubility, binding affinity, etc.)

### R² (Coefficient of Determination)

**Display:** R²
**Range:** (-∞, 1.0], typically [0, 1]
**Best:** Higher (closer to 1.0)

**What It Measures:**
Proportion of variance in true values explained by model predictions.

**Formula:**
$$R^2 = 1 - \frac{\sum(y_{true} - y_{pred})^2}{\sum(y_{true} - \bar{y})^2}$$

**Interpretation:**
- **R² = 1.0** - Perfect predictions
- **R² = 0.9** - Excellent (90% variance explained)
- **R² = 0.7** - Good (70% variance explained)
- **R² = 0.5** - Moderate (50% variance explained)
- **R² = 0.0** - No better than mean
- **R² < 0** - Worse than predicting mean

**When to Use:**
✅ Standard metric for regression tasks
✅ Comparing models on same dataset
✅ Publication and reporting

**Limitations:**
⚠️ Sensitive to outliers
⚠️ Can be misleading with small datasets
⚠️ Doesn't distinguish over/under-prediction

### Pearson R (Pearson Correlation)

**Display:** Pearson R
**Range:** [-1, 1]
**Best:** Higher (closer to 1.0)

**What It Measures:**
Linear correlation between predictions and true values.

**Formula:**
$$r = \frac{\sum(x - \bar{x})(y - \bar{y})}{\sqrt{\sum(x - \bar{x})^2 \sum(y - \bar{y})^2}}$$

**Interpretation:**
- **r = 1.0** - Perfect positive linear relationship
- **r = 0.9** - Strong positive correlation
- **r = 0.7** - Moderate positive correlation
- **r = 0.5** - Weak positive correlation
- **r = 0.0** - No linear correlation
- **r = -1.0** - Perfect negative correlation

**When to Use:**
✅ Assessing linear relationship strength
✅ When absolute values less important than ranking
✅ Comparing with literature (common in cheminformatics)

**Note:** Pearson R² = (Pearson R)² ≈ R² for linear relationships

### Spearman ρ (Spearman Rank Correlation)

**Display:** Spearman ρ
**Range:** [-1, 1]
**Best:** Higher (closer to 1.0)

**What It Measures:**
Monotonic relationship between predictions and true values using ranks.

**How It Works:**
Converts values to ranks, then calculates correlation of ranks.

**Interpretation:**
- **ρ = 1.0** - Perfect monotonic relationship (predictions rank perfectly)
- **ρ = 0.8** - Strong rank agreement
- **ρ = 0.5** - Moderate rank agreement
- **ρ = 0.0** - No rank correlation

**When to Use:**
✅ Ranking tasks (e.g., virtual screening)
✅ When outliers present
✅ Non-linear but monotonic relationships
✅ Ordinal data

**Advantage over Pearson:**
Robust to outliers and non-linear monotonic relationships.

### Kendall τ (Kendall Tau)

**Display:** Kendall's τ
**Range:** [-1, 1]
**Best:** Higher (closer to 1.0)

**What It Measures:**
Rank concordance - probability that pairs are ordered correctly.

**How It Works:**
Counts concordant and discordant pairs.

**Interpretation:**
- **τ = 1.0** - All pairs correctly ordered
- **τ = 0.7** - 70% of pairs concordant
- **τ = 0.5** - Moderate ranking ability
- **τ = 0.0** - Random ranking

**When to Use:**
✅ Small datasets
✅ Ties in data
✅ More robust statistical interpretation than Spearman
✅ Virtual screening prioritization

**Note:** More conservative than Spearman (typically lower values).

### RMSE (Root Mean Squared Error)

**Display:** RMSE
**Range:** [0, ∞)
**Best:** Lower (closer to 0)

**What It Measures:**
Average magnitude of prediction errors, penalizing large errors more.

**Formula:**
$$RMSE = \sqrt{\frac{1}{n}\sum(y_{true} - y_{pred})^2}$$

**Interpretation:**
- **RMSE = 0** - Perfect predictions
- **RMSE = 0.5** - Average error of 0.5 units
- **RMSE = 2.0** - Average error of 2.0 units

**When to Use:**
✅ When large errors are particularly bad
✅ Same scale as target variable
✅ Comparing models on same dataset

**Limitations:**
⚠️ Not normalized (depends on target scale)
⚠️ Sensitive to outliers
⚠️ Can't compare across different datasets

### MAE (Mean Absolute Error)

**Display:** MAE
**Range:** [0, ∞)
**Best:** Lower (closer to 0)

**What It Measures:**
Average absolute difference between predictions and true values.

**Formula:**
$$MAE = \frac{1}{n}\sum|y_{true} - y_{pred}|$$

**Interpretation:**
- **MAE = 0** - Perfect predictions
- **MAE = 0.3** - On average, predictions off by 0.3 units

**When to Use:**
✅ More interpretable than RMSE
✅ When all errors equally important
✅ Robust to outliers

**MAE vs RMSE:**
- **RMSE > MAE** - Large errors present
- **RMSE ≈ MAE** - Errors evenly distributed

### MSE (Mean Squared Error)

**Display:** MSE
**Range:** [0, ∞)
**Best:** Lower (closer to 0)

**What It Measures:**
Average squared error, heavily penalizing large errors.

**Formula:**
$$MSE = \frac{1}{n}\sum(y_{true} - y_{pred})^2$$

**Note:** MSE = RMSE²

**When to Use:**
✅ Loss function for model training
✅ Mathematical optimization

**Limitation:** Less interpretable than RMSE (squared units).

### Other Regression Metrics

**MedAE (Median Absolute Error)**
- Median instead of mean of absolute errors
- Very robust to outliers
- Use when outliers should be ignored

**Max Error**
- Maximum absolute error across all predictions
- Identifies worst-case performance
- Critical for safety-critical applications

**MAPE (Mean Absolute Percentage Error)**
- Average percentage error
- Scale-independent
- Undefined when true values are zero

**MSLE (Mean Squared Log Error)**
- Log-transformed MSE
- Penalizes under-prediction more than over-prediction
- Use for exponential relationships

**Explained Variance**
- Similar to R² but doesn't account for bias
- Range: [0, 1]
- Use when bias is acceptable

## Classification Metrics

For predicting categorical outcomes (active/inactive, drug-like/non-drug-like, etc.)

### Accuracy

**Display:** Accuracy
**Range:** [0, 1]
**Best:** Higher (closer to 1.0)

**What It Measures:**
Proportion of correct predictions.

**Formula:**
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Interpretation:**
- **Acc = 1.0** - All predictions correct
- **Acc = 0.9** - 90% correct
- **Acc = 0.5** - 50% correct (coin flip for binary)

**When to Use:**
✅ Balanced datasets
✅ All classes equally important
✅ Simple baseline metric

**Limitations:**
⚠️ Misleading with imbalanced data
⚠️ Doesn't show which errors (FP vs FN)

**Example Imbalanced Issue:**
```
Dataset: 95% class 0, 5% class 1
Model predicts all class 0
Accuracy: 95% (looks good!)
But: Completely fails on minority class
```

### F1 Score

**Display:** F1 Score
**Range:** [0, 1]
**Best:** Higher (closer to 1.0)

**What It Measures:**
Harmonic mean of precision and recall.

**Formula:**
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Interpretation:**
- **F1 = 1.0** - Perfect precision and recall
- **F1 = 0.8** - Good balance
- **F1 = 0.5** - Moderate performance

**When to Use:**
✅ **Recommended default for classification**
✅ Imbalanced datasets
✅ Both precision and recall matter

**F1 Variants:**
- **F1 (binary)** - Single class focus
- **F1 (macro)** - Average F1 per class (unweighted)
- **F1 (micro)** - Global F1 across all classes
- **F1 (weighted)** - Weighted average by class frequency

### Precision

**Display:** Precision
**Range:** [0, 1]
**Best:** Higher

**What It Measures:**
Of predictions marked positive, how many are correct?

**Formula:**
$$Precision = \frac{TP}{TP + FP}$$

**Interpretation:**
- **Prec = 1.0** - No false positives
- **Prec = 0.8** - 80% of positive predictions are correct

**When to Use:**
✅ When false positives are costly
✅ Drug candidate selection (avoid wasting resources on false leads)
✅ Toxicity prediction (false alarm acceptable)

### Recall (Sensitivity)

**Display:** Recall
**Range:** [0, 1]
**Best:** Higher

**What It Measures:**
Of actual positives, how many did we find?

**Formula:**
$$Recall = \frac{TP}{TP + FN}$$

**Interpretation:**
- **Rec = 1.0** - Found all positives
- **Rec = 0.8** - Found 80% of positives

**When to Use:**
✅ When false negatives are costly
✅ Toxicity screening (don't miss toxic compounds)
✅ Active compound identification

**Precision vs Recall Trade-off:**
- **High Precision, Low Recall** - Conservative, few false positives
- **Low Precision, High Recall** - Liberal, few false negatives
- **F1 balances both**

### ROC-AUC (Area Under ROC Curve)

**Display:** ROC-AUC
**Range:** [0, 1]
**Best:** Higher (closer to 1.0)

**What It Measures:**
Probability model ranks random positive higher than random negative.

**Interpretation:**
- **AUC = 1.0** - Perfect discrimination
- **AUC = 0.9** - Excellent (90% chance correct ranking)
- **AUC = 0.7** - Acceptable
- **AUC = 0.5** - Random (useless)

**When to Use:**
✅ Comparing model discrimination ability
✅ Threshold-independent evaluation
✅ Imbalanced datasets

**Advantage:** Independent of classification threshold.

### Matthews Correlation Coefficient (MCC)

**Display:** Matthews Correlation
**Range:** [-1, 1]
**Best:** Higher (closer to 1.0)

**What It Measures:**
Quality of binary classification, accounting for all confusion matrix elements.

**Formula:**
$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

**Interpretation:**
- **MCC = 1.0** - Perfect prediction
- **MCC = 0** - Random prediction
- **MCC = -1.0** - Perfect inverse prediction

**When to Use:**
✅ **Best single metric for imbalanced data**
✅ When all confusion matrix cells matter
✅ Comparing classifiers fairly

### Cohen's κ (Kappa)

**Display:** Cohen's κ
**Range:** [-1, 1]
**Best:** Higher (closer to 1.0)

**What It Measures:**
Agreement beyond chance.

**Interpretation:**
- **κ = 1.0** - Perfect agreement
- **κ = 0.8** - Strong agreement
- **κ = 0.5** - Moderate agreement
- **κ = 0** - Agreement by chance

**When to Use:**
✅ Inter-rater reliability
✅ Accounting for chance agreement

### Balanced Accuracy

**Display:** Balanced Accuracy
**Range:** [0, 1]
**Best:** Higher

**What It Measures:**
Average of recall for each class.

**Formula:**
$$Balanced\ Acc = \frac{1}{2}(Sensitivity + Specificity)$$

**When to Use:**
✅ Imbalanced datasets
✅ When both classes equally important

## Cross-Validation Metrics

Metrics describing model stability across training folds.

### CV Mean

**Display:** CV Mean
**Range:** Same as primary metric
**Best:** Depends on metric

**What It Measures:**
Average performance across cross-validation folds.

**When to Use:**
✅ Assessing expected performance
✅ Comparing with test set performance

### CV Std (Standard Deviation)

**Display:** CV Std
**Range:** [0, ∞)
**Best:** Lower (more stable)

**What It Measures:**
Variability of performance across folds.

**Interpretation:**
- **Low std (<0.05)** - Consistent, stable model
- **High std (>0.15)** - Unstable, sensitive to data splits

**When to Use:**
✅ Assessing model robustness
✅ Comparing consistency between models

**Red Flag:** If CV std is high, model may not generalize well.

### Individual Fold Scores

**Display:** Fold 1, Fold 2, ..., Fold 5
**Range:** Same as primary metric

**What It Measures:**
Performance on each individual cross-validation fold.

**When to Use:**
✅ Debugging high CV std
✅ Identifying problematic folds
✅ Understanding variance sources

## Metric Selection Guide

### By Task Type

**Regression (Continuous Prediction):**
1. **Start with:** R² or Pearson R
2. **For ranking:** Spearman ρ or Kendall τ
3. **For error magnitude:** RMSE or MAE
4. **For robustness:** Spearman ρ (outlier-resistant)

**Binary Classification:**
1. **Balanced data:** Accuracy or F1
2. **Imbalanced data:** F1, MCC, or ROC-AUC
3. **Precision matters:** Precision + F1
4. **Recall matters:** Recall + F1

**Multi-class Classification:**
1. **Start with:** F1 (macro) or Accuracy
2. **Imbalanced:** F1 (weighted) or MCC
3. **Per-class detail:** F1 (macro) + confusion matrix

### By Application

**Virtual Screening (Ranking):**
- Primary: Spearman ρ or Kendall τ
- Secondary: Pearson R

**Drug Discovery (Accuracy):**
- Primary: R² or RMSE
- Secondary: MAE

**Toxicity Prediction (Avoid False Negatives):**
- Primary: Recall
- Secondary: F1, MCC

**Lead Optimization (Avoid False Positives):**
- Primary: Precision
- Secondary: F1

**ADMET Classification:**
- Primary: F1 or MCC
- Secondary: ROC-AUC

### By Data Characteristics

**Small Dataset (<100 samples):**
- Use: Kendall τ (more robust than Spearman)
- Avoid: Accuracy (unreliable)

**Large Dataset (>10,000 samples):**
- Use: Any metric appropriate for task
- Note: R² and accuracy more reliable

**Outliers Present:**
- Use: Spearman ρ, MAE, MedAE
- Avoid: RMSE, Pearson R (sensitive)

**Imbalanced Classes:**
- Use: F1, MCC, ROC-AUC, Balanced Accuracy
- Avoid: Accuracy (misleading)

## Understanding Metric Changes

When you switch metrics in the dashboard, rankings may change significantly.

### Why Rankings Change

**Example:**
```
Model A: R² = 0.85, RMSE = 0.42
Model B: R² = 0.82, RMSE = 0.38

By R²:  Model A better
By RMSE: Model B better
```

**Reasons:**
1. **Different aspects measured** - R² (variance explained) vs RMSE (error magnitude)
2. **Outlier sensitivity** - RMSE penalizes large errors more
3. **Scale effects** - Normalized vs absolute metrics
4. **Linear vs non-linear** - Pearson (linear) vs Spearman (monotonic)

### Consistent Performers

**Look for models that rank high across multiple metrics:**

```
Model: XGBoost + morgan_fp

Rank by R²:         #1 (0.856)
Rank by RMSE:       #2 (0.421)
Rank by Spearman:   #1 (0.912)
Rank by Kendall:    #1 (0.765)

→ Robust model, consistently good
```

**Red Flag - Inconsistent Rankings:**
```
Model: SVM + rdkit_desc

Rank by R²:         #3 (0.782)
Rank by RMSE:       #15 (0.892)
Rank by Spearman:   #8 (0.801)

→ May have outliers or bias issues
→ Investigate residual plots
```

## Best Practices

```{admonition} Metric Selection Tips
:class: tip

1. **Use 2-3 complementary metrics** - R² + RMSE + Spearman
2. **Match metric to goal** - Ranking? Use Spearman/Kendall
3. **Check consistency** - Switch metrics, see if top models change
4. **Consider downstream use** - Virtual screening → rank correlation
5. **Report primary + alternatives** - R² (primary), RMSE and MAE (supporting)
```

```{admonition} Common Mistakes
:class: warning

- **Using only R²** - Doesn't show error magnitude or outliers
- **Using accuracy for imbalanced data** - Misleading, use F1 or MCC
- **Ignoring CV std** - High variance = unreliable model
- **Not matching metric to task** - Classification task with RMSE
- **Comparing metrics across datasets** - RMSE not comparable across different scales
```

## Metric Formulas Reference

### Regression

| Metric | Formula | Range | Best |
|--------|---------|-------|------|
| R² | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | (-∞, 1] | High |
| RMSE | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | [0, ∞) | Low |
| MAE | $\frac{1}{n}\sum\|y - \hat{y}\|$ | [0, ∞) | Low |

### Classification

| Metric | Formula | Range | Best |
|--------|---------|-------|------|
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | [0, 1] | High |
| Precision | $\frac{TP}{TP + FP}$ | [0, 1] | High |
| Recall | $\frac{TP}{TP + FN}$ | [0, 1] | High |
| F1 | $2 \times \frac{P \times R}{P + R}$ | [0, 1] | High |

## Next Steps

- **Apply metrics in practice**: the main dashboard guide - Real decision-making examples
- **Visualize distributions**: {doc}`distributions` - See how metrics distribute
- **Compare models**: {doc}`performance` - Model comparison charts
- **Inspect predictions**: {doc}`model_inspection` - Scatter plots by metric
