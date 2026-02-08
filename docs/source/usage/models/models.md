# Available Models

MolBlender provides 28 machine learning models spanning traditional ML to deep learning, automatically matched to your data modality.

## Model Categories

Models are organized by type, performance characteristics, and computational requirements:

| **Category** | **Models** | **Training Speed** | **When to Use** |
|-------------|-----------|-------------------|----------------|
| **Linear** (4) | Ridge, Logistic, LinearSVR, Bayesian | ⚡️ Fastest | Baseline, interpretability, large datasets |

```{note} Lasso and ElasticNet are currently disabled due to poor performance on small datasets (<500 samples). They may be re-enabled for larger datasets in future versions.
```
| **Tree** (3) | RandomForest, ExtraTrees, DecisionTree | ⚡️⚡️ Fast | Good accuracy-speed balance, feature importance |
| **Boosting** (4) | GradientBoosting, XGBoost, LightGBM, AdaBoost | ⚡️⚡️ Fast | Highest accuracy on tabular data |
| **Kernel** (2) | SVM_RBF, KNN | ⚡️⚡️⚡️ Medium | Non-linear patterns, small-medium datasets |
| **Neural** (1) | MLP | ⚡️⚡️⚡️ Medium | Complex non-linear relationships |
| **Deep Learning** (9) | Transformer, CNN, VAE variants | ⚡️⚡️⚡️⚡️ Slow | Raw strings, images, matrices; large datasets |
| **Ensemble** (3) | Bagging, Voting, Stacking | ⚡️⚡️⚡️ Medium | Maximum performance, combining predictions |

## Model-Modality Compatibility Matrix

Different data modalities require specific model types:

| **Data Modality** | **Compatible Models** | **Example Representations** |
|------------------|----------------------|----------------------------|
| **VECTOR** | All traditional ML + VAE | `morgan_fp`, `rdkit_descriptors`, `chemberta_embeddings`, `unimol_cls` |
| **STRING** | Transformers only | `canonical_smiles`, `selfies`, `inchikey` |
| **MATRIX** | CNN + flattened→ML backup | `adjacency_matrix`, `coulomb_matrix` |
| **IMAGE** | CNN + flattened→ML backup | `2d_image`, `3d_conformer_image` |

```{admonition} Automatic Compatibility Checking
:class: tip
MolBlender automatically prevents invalid combinations (e.g., Ridge regression + raw SMILES strings). You don't need to manually check compatibility.
```

## Complete Model Catalog

### Linear Models (6 models)

Fast, interpretable models with L1/L2 regularization. Best for baseline performance and large datasets.

**Ridge Regression**
- **Use:** L2 regularization, correlated features
- **Parameters:** `alpha=[0.01, 0.1, 1.0, 10.0, 100.0]`
- **Tasks:** Regression
- **Scaling:** Required
- **Time:** <1 second per fold

**Lasso Regression** ⚠️ *Disabled*
- **Status:** Currently disabled due to poor performance on small datasets (Pearson R = 0.00)
- **Use:** L1 regularization, feature selection
- **When to Re-enable:** Large datasets (>1000 samples) where feature selection is beneficial
- **Parameters:** `alpha=[0.01, 0.1, 1.0, 10.0]`
- **Tasks:** Regression
- **Scaling:** Required
- **Time:** <1 second per fold

**ElasticNet** ⚠️ *Disabled*
- **Status:** Currently disabled due to poor performance on small datasets (Pearson R ≈ 0.00)
- **Use:** Combined L1+L2, balanced feature selection
- **When to Re-enable:** Large datasets (>1000 samples) where elastic net regularization is beneficial
- **Parameters:** `alpha=[0.01, 0.1, 1.0]`, `l1_ratio=[0.1, 0.5, 0.9]`
- **Tasks:** Regression
- **Scaling:** Required
- **Time:** <1 second per fold

**Logistic Regression**
- **Use:** Binary/multiclass classification baseline
- **Parameters:** `C=[0.01, 0.1, 1.0, 10.0]`, `penalty=['l1', 'l2']`
- **Tasks:** Classification
- **Scaling:** Required
- **Time:** <2 seconds per fold

**LinearSVR**
- **Use:** Support vector regression, linear kernel
- **Parameters:** `C=[0.1, 1.0, 10.0]`, `epsilon=[0.01, 0.1, 1.0]`
- **Tasks:** Regression
- **Scaling:** Required
- **Time:** 2-5 seconds per fold

**Bayesian Ridge**
- **Use:** Probabilistic predictions with uncertainty
- **Parameters:** `alpha_1=[1e-6, 1e-5]`, `alpha_2=[1e-6, 1e-5]`
- **Tasks:** Regression
- **Scaling:** Required
- **Time:** <2 seconds per fold

### Tree-Based Models (3 models)

Excellent accuracy-speed balance with built-in feature importance. No scaling required.

**Random Forest** ⭐ *Recommended*
- **Use:** General-purpose, robust to outliers
- **Parameters:** `n_estimators=[50, 100, 200]`, `max_depth=[5, 10, None]`, `min_samples_split=[2, 5]`
- **Grid combinations:** 18 variants
- **Tasks:** Regression, Classification
- **Scaling:** Not required
- **Feature importance:** Yes
- **Time:** 5-15 seconds per fold
- **Memory:** 50-200 MB

**Extra Trees**
- **Use:** Faster than RF, good for large datasets
- **Parameters:** `n_estimators=[50, 100, 200]`, `max_depth=[5, 10, None]`
- **Tasks:** Regression, Classification
- **Scaling:** Not required
- **Time:** 3-10 seconds per fold

**Decision Tree**
- **Use:** Maximum interpretability, visualization
- **Parameters:** `max_depth=[3, 5, 10, 20]`, `min_samples_split=[2, 5, 10]`
- **Tasks:** Regression, Classification
- **Scaling:** Not required
- **Time:** <2 seconds per fold

### Boosting Models (4 models)

Highest accuracy on structured data. Sequential ensemble methods.

**XGBoost** ⭐ *Recommended*
- **Use:** Best overall performance on tabular data
- **Parameters:** `n_estimators=[100, 200]`, `learning_rate=[0.01, 0.1]`, `max_depth=[3, 6]`, `subsample=[0.8, 1.0]`
- **Grid combinations:** 16 variants
- **Tasks:** Regression, Classification
- **Scaling:** Not required
- **Feature importance:** Yes
- **Time:** 10-30 seconds per fold
- **Memory:** 100-400 MB

**LightGBM**
- **Use:** Fast gradient boosting, large datasets
- **Parameters:** `n_estimators=[100, 200]`, `learning_rate=[0.01, 0.1]`, `num_leaves=[31, 63]`
- **Tasks:** Regression, Classification
- **Scaling:** Not required
- **Time:** 5-20 seconds per fold

**Gradient Boosting**
- **Use:** sklearn implementation, stable
- **Parameters:** `n_estimators=[100, 200]`, `learning_rate=[0.01, 0.1]`, `max_depth=[3, 5]`
- **Tasks:** Regression, Classification
- **Scaling:** Not required
- **Time:** 15-45 seconds per fold

**AdaBoost**
- **Use:** Boosting weak learners
- **Parameters:** `n_estimators=[50, 100, 200]`, `learning_rate=[0.1, 0.5, 1.0]`
- **Tasks:** Regression, Classification
- **Scaling:** Not required
- **Time:** 10-30 seconds per fold

### Kernel Methods (2 models)

Non-linear pattern recognition with kernel tricks.

**SVM (RBF Kernel)**
- **Use:** Non-linear relationships, medium datasets
- **Parameters:** `C=[0.1, 1.0, 10.0]`, `gamma=['scale', 'auto', 0.001, 0.01]`
- **Grid combinations:** 12 variants
- **Tasks:** Regression, Classification
- **Scaling:** Required
- **Time:** 30-120 seconds per fold
- **Memory:** Scales with dataset size (O(n²))

**K-Nearest Neighbors (KNN)**
- **Use:** Instance-based learning, simple baseline
- **Parameters:** `n_neighbors=[3, 5, 7, 10]`, `weights=['uniform', 'distance']`, `p=[1, 2]`
- **Tasks:** Regression, Classification
- **Scaling:** Required
- **Time:** Fast training, slow prediction
- **Memory:** Stores entire dataset

### Neural Networks (1 model)

Multi-layer perceptron for complex non-linear patterns.

**MLP (Multi-Layer Perceptron)**
- **Use:** Complex non-linear relationships on vectors
- **Parameters:** `hidden_layer_sizes=[(100,), (100, 50), (200,)]`, `learning_rate_init=[0.001, 0.01]`, `alpha=[0.0001, 0.001]`
- **Grid combinations:** 18 variants
- **Tasks:** Regression, Classification
- **Scaling:** Required
- **Time:** 60-180 seconds per fold
- **Memory:** 200-500 MB

### Deep Learning Models (9 models)

Advanced neural architectures for raw strings, images, and matrices.

**CUDA fallback policy**
- If a CUDA error occurs during training, MolBlender falls back to CPU only when effective CPU cores ≥ 32.
- If fewer cores are available, the error is raised (no silent fallback).

**Transformer (Small)**
- **Use:** Raw SMILES/SELFIES strings, pre-training
- **Parameters:** `max_length=[256, 512]`, `learning_rate=[1e-4, 5e-5]`, `num_layers=[4, 6]`
- **Input:** STRING modality only
- **Tasks:** Regression, Classification
- **Time:** 10-30 minutes per fold
- **Memory:** 2-4 GB
- **GPU:** Highly recommended

**Transformer (Medium)**
- **Use:** Larger model capacity for complex patterns
- **Parameters:** `max_length=[512, 1024]`, `learning_rate=[1e-4, 5e-5]`, `num_layers=[8, 12]`
- **Input:** STRING modality only
- **Time:** 20-60 minutes per fold
- **Memory:** 4-8 GB
- **GPU:** Required

**Matrix CNN** ⭐ *For adjacency/Coulomb matrices*
- **Use:** 2D molecular matrices (adjacency, Coulomb)
- **Parameters:** `learning_rate=[0.001, 0.01]`, `dropout=[0.2, 0.3, 0.5]`, `filters=[32, 64]`
- **Input:** MATRIX modality only
- **Tasks:** Regression, Classification
- **Time:** 5-15 minutes per fold
- **Memory:** 1-3 GB
- **GPU:** Recommended

**Matrix CNN (Small)**
- **Use:** Faster CNN for matrices, fewer parameters
- **Parameters:** `learning_rate=[0.001, 0.01]`, `dropout=[0.2, 0.5]`
- **Input:** MATRIX modality only
- **Time:** 3-10 minutes per fold

**Image CNN** ⭐ *For 2D/3D molecular images*
- **Use:** Molecular structure images
- **Parameters:** `learning_rate=[0.001, 0.01]`, `dropout=[0.2, 0.3, 0.5]`
- **Input:** IMAGE modality only
- **Tasks:** Regression, Classification
- **Time:** 5-15 minutes per fold
- **Memory:** 1-3 GB
- **GPU:** Recommended

**Image CNN (Small)**
- **Use:** Faster CNN for images
- **Parameters:** `learning_rate=[0.001, 0.01]`
- **Input:** IMAGE modality only
- **Time:** 3-10 minutes per fold

**VAE (Variational Autoencoder)** - 3 variants
- **Use:** Deep learning backup for fingerprints, unsupervised feature learning
- **Variants:**
  - `vae_latent_64`: 64-dim latent space
  - `vae_latent_128`: 128-dim latent space (default)
  - `vae_latent_256`: 256-dim latent space
- **Input:** VECTOR modality (fingerprints, descriptors)
- **Parameters:** `learning_rate=[0.001, 0.01]`, `beta=[0.5, 1.0]` (KL divergence weight)
- **Tasks:** Regression (via latent space)
- **Time:** 10-20 minutes per fold
- **Memory:** 500 MB - 2 GB
- **GPU:** Optional but faster

### Ensemble Methods (3 models)

Combine multiple models for improved performance.

**Bagging**
- **Use:** Reduce variance by bootstrap aggregation
- **Parameters:** `n_estimators=[10, 50, 100]`, `max_samples=[0.5, 1.0]`
- **Tasks:** Regression, Classification
- **Time:** Depends on base estimator

**Voting Regressor/Classifier**
- **Use:** Average predictions from multiple models
- **Parameters:** Custom base estimators, `weights`
- **Tasks:** Regression, Classification
- **Time:** Sum of all base estimators

**Stacking**
- **Use:** Meta-learning over base models
- **Parameters:** Base estimators, `final_estimator`
- **Tasks:** Regression, Classification
- **Time:** Sum of base + meta-estimator training

## Resource Requirements

### LIGHT Tasks (Parallel Combination Execution)

Models execute in parallel across different representation combinations:

**Models:** Ridge, RandomForest, XGBoost, LightGBM, SVM, KNN, Logistic, LinearSVR, Bayesian Ridge

**Resource Profile:**
- CPU: Uses all available cores (16 cores = 16 parallel combinations)
- Memory: 50-400 MB per combination
- Time: 1-60 seconds per combination
- GPU: Not used

**Example:** With 16 cores and 20 fingerprint+ML combinations, all 16 cores process different combinations simultaneously.

### HEAVY Tasks (Sequential + Internal Parallelism)

Models execute sequentially with internal multi-core/GPU parallelism:

**Models:** Transformer, Matrix CNN, Image CNN, MLP, VAE

**Resource Profile:**
- CPU: Fewer parallel jobs (16 cores = 2-4 jobs × 4-8 cores each)
- Memory: 1-8 GB per model
- Time: 3-60 minutes per model
- GPU: 4-12 GB VRAM (highly recommended)

**Example:** With 1 GPU, transformers run sequentially, each using the full GPU.

## Model Selection Guide

### By Use Case

**Quick Exploration (<5 minutes)**
- RandomForest, XGBoost, Ridge

**Production Deployment**
- XGBoost, LightGBM, RandomForest (fast prediction)

**Maximum Accuracy (unlimited time)**
- XGBoost, Transformer, Ensemble methods

**Interpretability Priority**
- Ridge, Lasso, DecisionTree, RandomForest (feature importance)

**Limited Memory (<2GB)**
- Ridge, DecisionTree, Bayesian Ridge

**Large Datasets (>100K molecules)**
- LightGBM, ExtraTrees, SGDRegressor

**Raw SMILES Strings**
- Transformer (only option for STRING modality)

### By Data Type

**Fingerprints (VECTOR):** XGBoost > RandomForest > Ridge
**Descriptors (VECTOR):** XGBoost > Ridge > Bayesian Ridge
**Pre-computed Embeddings (VECTOR):** RandomForest > XGBoost > SVM
**Raw SMILES (STRING):** Transformer Small/Medium
**Adjacency Matrices (MATRIX):** Matrix CNN > Flattened + XGBoost
**Molecular Images (IMAGE):** Image CNN > Flattened + RandomForest

## Parameter Grid Philosophy

All models include scientifically-informed parameter grids:

- **Coarse grids** with 2-5 values per parameter
- **Logarithmic scales** for alpha, C, learning_rate
- **Linear scales** for tree depth, n_estimators
- **Total combinations** typically 5-20 per model

```python
# Example: XGBoost parameter grid
{
    'n_estimators': [100, 200],          # 2 values
    'learning_rate': [0.01, 0.1],        # 2 values
    'max_depth': [3, 6],                 # 2 values
    'subsample': [0.8, 1.0]              # 2 values
}
# Total: 2×2×2×2 = 16 combinations tested via cross-validation
```

## Next Steps

- **Run screening**: {doc}`screening` - Use these models with `universal_screen()`
- **View results**: {doc}`results` - Access model performance and predictions
- **Dashboard**: {doc}`../dashboard/index` - Compare model performance interactively
