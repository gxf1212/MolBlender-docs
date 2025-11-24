# Dataset Splitting Strategies

Comprehensive guide to data splitting and cross-validation strategies in MolBlender.

## Overview

MolBlender provides flexible, professional-grade data splitting strategies for molecular machine learning. The splitting system is designed to:

- **Ensure fair model comparison** through consistent random seeds
- **Support multiple splitting strategies** for different use cases
- **Handle both classification and regression** with appropriate techniques
- **Maintain reproducibility** across different runs

```{admonition} Key Feature
:class: tip

All splitting strategies use **fixed random seeds** by default, ensuring complete reproducibility of model evaluations across different runs and users.
```

## Quick Start

### Basic Usage

```python
from molblender.models import universal_screen
from molblender.data import MolecularDataset

# Load your dataset
dataset = MolecularDataset.from_csv("molecules.csv",
                                   input_column="SMILES",
                                   label_columns=["activity"])

# Use default splitting (train_test with 80/20 split)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    task_type="regression",
    test_size=0.2,          # 20% for testing
    cv_folds=5,             # 5-fold cross-validation
    random_state=42         # Fixed random seed
)
```

## Supported Splitting Strategies

MolBlender supports 10 different splitting strategies, each suited for specific scenarios:

| Strategy | Use Case | Train/Val/Test | Best For |
|----------|----------|----------------|----------|
| `train_test` | Standard screening | 80% / — / 20% | Most common scenarios |
| `train_val_test` | With hyperparameter optimization | 70% / 15% / 15% | Large datasets with HPO |
| `nested_cv` | Unbiased HPO performance | Nested CV | Academic research |
| `cv_only` | Small datasets | CV only | < 100 samples |
| `scaffold` | Drug discovery | Scaffold-based | Novel structure generalization |
| `dnr` | Rough SAR analysis | DNR-based | Testing on challenging molecules |
| `maxmin` | Diversity testing | MaxMin-based | Chemical space extrapolation |
| `butina` | Cluster-based validation | Butina clustering | Similar molecule generalization |
| `feature_clustering` | General clustering split | K-means/Hierarchical/DBSCAN | Custom representations (3D, embeddings) |
| `user_provided` | Custom splits | User-defined | Temporal, custom logic |

### 1. Train/Test Split (Default)

The standard two-way split used for most screening tasks.

#### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_test",  # Default
    test_size=0.2,                # 20% test set
    cv_folds=5,                   # 5-fold CV on training set
    random_state=42
)
```

#### How It Works

```
Dataset (1000 samples)
    ↓
train_test_split(test_size=0.2, random_state=42)
    ├─→ Training Set: 800 samples (80%)
    │      ↓
    │   5-Fold Cross-Validation
    │   ├─→ Fold 1: train 640, val 160
    │   ├─→ Fold 2: train 640, val 160
    │   ├─→ Fold 3: train 640, val 160
    │   ├─→ Fold 4: train 640, val 160
    │   └─→ Fold 5: train 640, val 160
    │      ↓
    │   Final model: trained on all 800 samples
    │
    └─→ Test Set: 200 samples (20%)
           ↓
        Final evaluation
```

#### Implementation Details

**Code Location**: `src/molblender/models/api/core/splitting/strategies.py:26-84`

```python
def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Standard train/test split.

    For classification, uses StratifiedShuffleSplit to maintain class balance.
    For regression, uses regular train_test_split with shuffling.
    """
    if stratify is not None:
        # Stratified split for classification
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, stratify))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        # Regular split for regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'split_type': 'train_test'
    }
```

### 2. Train/Val/Test Split

Three-way split for scenarios involving hyperparameter optimization.

#### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_val_test",
    test_size=0.15,      # 15% test set
    val_size=0.15,       # 15% validation set
    random_state=42
)
```

#### How It Works

```
Dataset (1000 samples)
    ↓
First split: (train+val) vs test
    ├─→ Temp Set: 850 samples (85%)
    │      ↓
    │   Second split: train vs val
    │   ├─→ Training Set: 700 samples (70%)
    │   └─→ Validation Set: 150 samples (15%)
    │
    └─→ Test Set: 150 samples (15%)
```

#### Use Cases

- **Hyperparameter optimization**: Use validation set to tune hyperparameters
- **Model selection**: Choose best model architecture on validation set
- **Large datasets**: When you have enough data (>5000 samples) for three-way split

#### Implementation

**Code Location**: `splitting/strategies.py:86-169`

```python
def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Three-way split: train / validation / test.

    Best for HPO when you have sufficient data.
    """
    # First split: (train+val) vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=True
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp if stratify is not None else None,
        shuffle=True
    )

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'split_type': 'train_val_test'
    }
```

### 3. Nested Cross-Validation

Provides unbiased performance estimates for hyperparameter optimization.

#### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="nested_cv",
    outer_cv_folds=5,     # Outer CV for performance estimation
    inner_cv_folds=3,     # Inner CV for hyperparameter tuning
    random_state=42
)
```

#### How It Works

```
Dataset (1000 samples)
    ↓
Outer CV (5 folds) - for performance estimation
    ├─→ Fold 1: dev 800, test 200
    │      ↓
    │   Inner CV (3 folds on dev set) - for HPO
    │   ├─→ Inner Fold 1: train 533, val 267
    │   ├─→ Inner Fold 2: train 533, val 267
    │   └─→ Inner Fold 3: train 534, val 266
    │      ↓
    │   Best hyperparameters → test on outer test (200)
    │
    ├─→ Fold 2: dev 800, test 200
    │   (repeat inner CV...)
    ...
    └─→ Fold 5: dev 800, test 200
           ↓
        Average performance across 5 outer folds
```

#### Use Cases

- **Academic research**: Unbiased performance estimation for publications
- **Model comparison**: Fair comparison when HPO is involved
- **Small-to-medium datasets**: Maximize data utilization

#### Implementation

**Code Location**: `splitting/strategies.py:215-284`

```python
def get_nested_cv_splitter(
    n_samples: int,
    outer_cv_folds: int = 5,
    inner_cv_folds: int = 3,
    random_state: int = 42,
    is_classification: bool = False
) -> Dict[str, Any]:
    """
    Get nested cross-validation splitters.

    Nested CV provides unbiased performance estimates when doing HPO.
    - Outer CV: For performance estimation
    - Inner CV: For hyperparameter tuning
    """
    if is_classification:
        outer_cv = StratifiedKFold(
            n_splits=outer_cv_folds,
            shuffle=True,
            random_state=random_state
        )
        inner_cv = StratifiedKFold(
            n_splits=inner_cv_folds,
            shuffle=True,
            random_state=random_state + 1  # Different seed
        )
    else:
        outer_cv = KFold(
            n_splits=outer_cv_folds,
            shuffle=True,
            random_state=random_state
        )
        inner_cv = KFold(
            n_splits=inner_cv_folds,
            shuffle=True,
            random_state=random_state + 1
        )

    return {
        'outer_cv': outer_cv,
        'inner_cv': inner_cv,
        'split_type': 'nested_cv'
    }
```

### 4. CV-Only Strategy

Pure cross-validation without a separate test set, for small datasets.

#### Configuration

```python
results = universal_screen(
    dataset=small_dataset,  # < 100 samples
    target_column="activity",
    split_strategy="cv_only",
    cv_folds=5,
    random_state=42
)
```

#### How It Works

```
Dataset (80 samples)
    ↓
5-Fold Cross-Validation (no separate test set)
    ├─→ Fold 1: train 64, val 16
    ├─→ Fold 2: train 64, val 16
    ├─→ Fold 3: train 64, val 16
    ├─→ Fold 4: train 64, val 16
    └─→ Fold 5: train 64, val 16
           ↓
        Average CV score as final metric
```

#### Use Cases

- **Small datasets**: < 100 samples where test set would be too small
- **Maximum data utilization**: Every sample used for both training and validation
- **Exploratory analysis**: Quick performance estimates

```{warning}
CV-only strategy doesn't provide a truly independent test set. Performance estimates may be optimistically biased. Use only when dataset size prohibits train/test split.
```

### 5. Scaffold Split

Scaffold-based splitting for drug discovery and medicinal chemistry applications.

#### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="scaffold",
    test_size=0.2,
    scaffold_func='bemis_murcko',  # or 'generic'
    split_method='balanced',       # or 'random'
    random_state=42
)
```

#### How It Works

Scaffold split ensures that molecules in the test set have **different chemical scaffolds** from those in the training set, providing a more realistic evaluation of model generalization to novel structures.

```
Dataset (1000 molecules)
    ↓
Compute Bemis-Murcko scaffolds
    ↓
Group molecules by scaffold
    Scaffold A: 400 molecules (benzene derivatives)
    Scaffold B: 350 molecules (pyridine derivatives)
    Scaffold C: 150 molecules (furan derivatives)
    Scaffold D: 100 molecules (thiophene derivatives)
    ↓
Assign entire scaffold groups to train or test
    ├─→ Training Set: Scaffolds A, B (750 molecules, 75%)
    └─→ Test Set: Scaffolds C, D (250 molecules, 25%)
           ↓
        ✓ No scaffold leakage between train and test
```

#### Scaffold Generation Methods

**1. Bemis-Murcko Scaffolds (default)**

Extracts the core ring system and linker atoms, removing all side chains.

```python
from molblender.models.api.core.splitting import compute_bemis_murcko_scaffolds

smiles = [
    "c1ccccc1CCO",      # Benzene + ethanol side chain
    "c1ccccc1CCN",      # Benzene + ethylamine side chain
    "c1ccccc1C(=O)O",   # Benzene + carboxylic acid
]

scaffolds = compute_bemis_murcko_scaffolds(smiles)
# All three have the same scaffold: "c1ccccc1" (benzene)
```

**2. Generic Scaffolds**

Further abstracts by replacing all atoms with carbons and all bonds with single bonds, focusing purely on topology.

```python
from molblender.models.api.core.splitting import compute_generic_scaffolds

smiles = [
    "c1ccccc1CCO",    # Aromatic benzene ring
    "C1CCCCC1CCO",    # Aliphatic cyclohexane ring
]

generic_scaffolds = compute_generic_scaffolds(smiles)
# Both have the same generic scaffold: "C1CCCCC1" (6-membered ring)
```

#### Split Methods

**1. Balanced Split (default)**

Greedily assigns scaffold groups to achieve the desired train/test ratio.

```python
# Balanced method - aims for exact test_size
result = scaffold_split(
    X, y, smiles,
    split_method='balanced',  # Balances dataset sizes
    test_size=0.2
)
```

**Strategy:**
1. Sort scaffold groups by size (largest first)
2. Greedily assign to train/test to match target ratio
3. Result: test set close to desired size

**2. Random Split**

Randomly assigns each scaffold group to either train or test.

```python
# Random method - more stochastic
result = scaffold_split(
    X, y, smiles,
    split_method='random',  # Random scaffold assignment
    test_size=0.2
)
```

**Strategy:**
1. Randomly shuffle unique scaffolds
2. Assign scaffolds to train/test until target ratio met
3. Result: more variability across runs (with different seeds)

#### Use Cases

**Drug Discovery:**
- **Generalization to novel scaffolds**: Test model's ability to predict activity for chemically distinct structures
- **Lead optimization**: Evaluate performance on scaffold hops
- **Virtual screening**: More realistic estimate of hit rate on diverse libraries

**Example:**
```python
from molblender.data import MolecularDataset
from molblender.models import universal_screen

# Load drug-like molecules
dataset = MolecularDataset.from_csv(
    "kinase_inhibitors.csv",
    input_column="SMILES",
    label_columns=["pIC50"]
)

# Scaffold-based screening
results = universal_screen(
    dataset=dataset,
    target_column="pIC50",
    split_strategy="scaffold",      # Use scaffold split
    test_size=0.2,
    scaffold_func='bemis_murcko',   # Standard BM scaffolds
    split_method='balanced',
    random_state=42
)

# Results reflect generalization to novel scaffolds
print(f"Test R²: {results['best_model']['test_r2']:.3f}")
```

#### Direct Usage

For more control, use the scaffold split function directly:

```python
from molblender.models.api.core.splitting import scaffold_split
import numpy as np

# Your data
smiles = ["c1ccccc1CCO", "c1ccccc1CCN", ...]  # List of SMILES
X = np.array([...])  # Feature matrix
y = np.array([...])  # Target values

# Perform scaffold split
result = scaffold_split(
    X, y, smiles,
    test_size=0.2,
    scaffold_func='bemis_murcko',
    split_method='balanced',
    random_state=42
)

# Access results
X_train = result['X_train']
X_test = result['X_test']
y_train = result['y_train']
y_test = result['y_test']
scaffolds = result['scaffolds']  # Scaffold for each molecule

# Verify no scaffold leakage
train_scaffolds = set(scaffolds[i] for i in result['train_indices'])
test_scaffolds = set(scaffolds[i] for i in result['test_indices'])
overlap = train_scaffolds & test_scaffolds
print(f"Scaffold overlap: {len(overlap)} scaffolds")  # Should be 0
```

#### Implementation Details

**Code Location**: `src/molblender/models/api/core/splitting/scaffold.py`

```python
def scaffold_split(
    X: np.ndarray,
    y: np.ndarray,
    smiles: List[str],
    test_size: float = 0.2,
    scaffold_func: str = 'bemis_murcko',
    split_method: str = 'balanced',
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Perform scaffold-based train/test split.

    Returns:
        Dictionary with keys: X_train, X_test, y_train, y_test,
        train_indices, test_indices, scaffolds, split_type
    """
```

#### Advantages

```{admonition} Why Use Scaffold Split?
:class: tip

**1. Realistic Evaluation**: Mimics real-world scenarios where you predict activity for novel chemical matter

**2. Prevents Data Leakage**: Ensures no similar structures in both train and test sets

**3. Conservative Estimates**: Usually gives lower performance than random split, providing a more honest assessment

**4. Industry Standard**: Widely used in pharmaceutical industry for virtual screening validation
```

#### Comparison with Random Split

```python
# Random split (optimistic)
random_results = universal_screen(
    dataset, "pIC50",
    split_strategy="train_test",  # Random split
    test_size=0.2,
    random_state=42
)
print(f"Random split R²: {random_results['best_model']['test_r2']:.3f}")

# Scaffold split (realistic)
scaffold_results = universal_screen(
    dataset, "pIC50",
    split_strategy="scaffold",  # Scaffold split
    test_size=0.2,
    random_state=42
)
print(f"Scaffold split R²: {scaffold_results['best_model']['test_r2']:.3f}")

# Typical result: Scaffold split R² < Random split R²
# Example: 0.65 vs 0.78
```

```{note}
Scaffold split typically yields **lower performance metrics** than random split because it tests generalization to truly novel structures. This is expected and provides a more realistic estimate of real-world performance.
```

### 6. Butina Clustering Split

Leave-cluster-out validation based on Tanimoto similarity clustering, preventing information leakage from chemically similar molecules.

#### What is Butina Clustering?

Butina clustering is a sphere exclusion algorithm that groups molecules based on structural similarity:
- **Automatic clustering**: Self-adaptive cluster count based on similarity threshold (no need to specify K)
- **Leave-cluster-out**: Entire clusters move as units to train or test set
- **Similarity-based**: Uses Tanimoto similarity on molecular fingerprints

```{admonition} Use Case
:class: tip

Butina split is ideal when you want to:
- Test generalization to **similar but unseen** chemical combinations
- Avoid information leakage from structural similarity
- Implement MolAgent-style clustering validation
- Evaluate on OP-like structured design spaces (R group + Y functional group combinations)
```

#### Configuration

```python
from molblender.models import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="dG",
    split_strategy="butina",           # Butina clustering
    test_size=0.2,
    butina_similarity_threshold=0.6,   # Tanimoto threshold
    fingerprint_type='morgan',         # Morgan/RDKit/MACCS
    fp_radius=2,
    fp_nbits=2048,
    random_state=42
)

# Check cluster statistics
print(f"Number of clusters: {results['split_info']['n_clusters']}")
print(f"Actual test size: {results['split_info']['test_size_actual']:.1%}")
print(f"Train intra-similarity: {results['split_info']['train_avg_intra_similarity']:.3f}")
print(f"Test intra-similarity: {results['split_info']['test_avg_intra_similarity']:.3f}")
print(f"Train-test similarity: {results['split_info']['train_test_similarity']:.3f}")
```

#### How It Works

```
Dataset (120 molecules)
    ↓
Compute Tanimoto similarity matrix (Morgan fingerprints)
    ↓
Butina clustering (similarity_threshold=0.6)
    ├─→ Cluster 1: 15 molecules (all Tanimoto > 0.6)
    ├─→ Cluster 2: 12 molecules
    ├─→ Cluster 3: 8 molecules
    ├─→ ...
    └─→ Cluster N: 5 molecules
       ↓
Greedy balanced assignment (largest clusters first)
    ├─→ Test set: Clusters 1, 3, 5 (24 molecules, 20%)
    └─→ Train set: Clusters 2, 4, 6, ... (96 molecules, 80%)
       ↓
Leave-cluster-out validation
(No molecules from same cluster split across train/test)
```

#### Direct Usage

```python
from molblender.models.api.core.splitting import butina_split

# Perform Butina clustering split
result = butina_split(
    dataset=dataset,
    test_size=0.2,
    similarity_threshold=0.6,  # Molecules with Tanimoto > 0.6 cluster together
    fingerprint_type='morgan',
    radius=2,
    nbits=2048,
    random_state=42
)

# Access results
train_idx = result['train_indices']
test_idx = result['test_indices']
cluster_info = result['split_info']

print(f"Clusters: {cluster_info['n_clusters']}")
print(f"Largest cluster: {cluster_info['largest_cluster_size']} molecules")
print(f"Smallest cluster: {cluster_info['smallest_cluster_size']} molecules")
```

#### Implementation Details

**Code Location**: `src/molblender/models/api/core/splitting/butina.py`

```python
def butina_split(
    dataset: MolecularDataset,
    test_size: float = 0.2,
    similarity_threshold: float = 0.6,
    fingerprint_type: str = 'morgan',
    radius: int = 2,
    nbits: int = 2048,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Butina clustering + leave-cluster-out splitting.

    Uses Butina's sphere exclusion algorithm to cluster molecules,
    then assigns entire clusters to train or test sets.
    """
```

**Algorithm**: Butina, D. J. Chem. Inf. Comput. Sci. 1999, 39, 747-750

#### Butina vs Scaffold Split

| Feature | Butina | Scaffold |
|---------|--------|----------|
| **Granularity** | Fine-grained (full molecule topology) | Coarse-grained (core structure only) |
| **Considers** | All atoms + substituents + functional groups | Only ring systems + linkers |
| **Use Case** | Similar molecule generalization | Novel scaffold generalization |
| **Example** | Benzene ↔ Toluene (similar) | Benzene ↔ Pyridine (different scaffolds) |

```python
# Example: OP-based dataset (R groups + Y functional groups)
# Scaffold: Only considers backbone structure (too coarse)
# Butina: Considers R + Y combinations (appropriate granularity)

# Molecules with same scaffold but different substituents
mol1 = "Polymer-N+(cyclohexyl)-O-...-CO2-"  # R=cyclohexyl, Y=carboxyl
mol2 = "Polymer-N+(cycloheptyl)-O-...-SO3-"  # R=cycloheptyl, Y=sulfonate

# Scaffold split: Same scaffold → likely both in train or both in test
# Butina split: Different Tanimoto → may be in different clusters
```

#### Advantages

```{admonition} Why Use Butina Split?
:class: tip

**1. Prevents Similarity Bias**: Avoids over-optimistic evaluation from similar molecules in train/test

**2. Self-Adaptive**: Automatic cluster count based on similarity threshold

**3. Cluster Integrity**: Entire clusters move as units (true leave-cluster-out)

**4. Fine-Grained Control**: More granular than scaffold (considers full molecular structure)

**5. MolAgent Alignment**: Implements cluster-based validation from MolAgent literature
```

### 7. Feature Clustering Split

General-purpose clustering split supporting arbitrary molecular representations (not limited to fingerprints).

#### What is Feature Clustering?

Feature clustering split uses general machine learning clustering algorithms (K-means, Hierarchical, DBSCAN) to partition molecules based on **user-defined feature spaces**, then performs leave-cluster-out splitting:

- **Flexible Feature Sources**: User representations, RDKit descriptors, or fingerprints
- **Multiple Algorithms**: K-means (fast, spherical), Hierarchical (tree-based), DBSCAN (density-based)
- **Auto K-Selection**: Silhouette score optimization for optimal cluster count
- **Quality Metrics**: Silhouette, Calinski-Harabasz, Davies-Bouldin indices

Feature clustering split is ideal when:

- **Non-Fingerprint Representations**: Using 3D embeddings (Boltz-2), language models (ChemBERTa), quantum features
- **Custom Feature Spaces**: Physicochemical descriptors, docking scores, or domain-specific features
- **Flexible Clustering**: Need control over clustering algorithm (K-means vs DBSCAN)
- **Representation Evaluation**: Testing model generalization across different representation space regions

#### Configuration

```python
from molblender.models import universal_screen
from molblender.data import MolecularDataset

# Using RDKit descriptors with K-means
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="feature_clustering",

    # Clustering parameters
    clustering_algorithm="kmeans",   # 'kmeans', 'hierarchical', 'dbscan'
    n_clusters=5,                    # Manual k (None for auto-selection)
    auto_select_k=False,             # Set True to auto-optimize k

    # Feature source (priority: features > descriptors > fingerprints)
    use_descriptors=True,            # Use RDKit descriptors (~20 features)
    fingerprint_type="morgan",       # Fallback if use_descriptors=False

    # DBSCAN-specific (only if algorithm='dbscan')
    dbscan_eps=0.5,                  # Max distance between neighbors
    dbscan_min_samples=5,            # Min cluster size

    test_size=0.2,
    random_state=42
)
```

#### How It Works

```
1. Feature Extraction
   ├─ User-provided features (if provided) → Use directly
   ├─ RDKit descriptors (if use_descriptors=True) → Compute MW, LogP, TPSA, etc.
   └─ Fingerprints (fallback) → Morgan/RDKit/MACCS

2. Feature Standardization
   └─ StandardScaler normalization (zero mean, unit variance)

3. Clustering
   ├─ K-means: Spherical clusters, need k specification
   ├─ Hierarchical: Ward linkage, tree-based structure
   └─ DBSCAN: Density-based, auto cluster count

4. Optimal K Selection (if auto_select_k=True)
   └─ Test k ∈ [2, min(10, √n)]
   └─ Select k maximizing Silhouette score

5. Leave-Cluster-Out Assignment
   ├─ Sort clusters by size (descending)
   ├─ Greedy assignment: Largest clusters → test set first
   ├─ Target test_size proportion
   └─ DBSCAN: Noise points (-1 label) → train set

6. Quality Metrics
   ├─ Silhouette score: [-1, 1], >0.5 = good clustering
   ├─ Calinski-Harabasz: Higher = better-defined clusters
   └─ Davies-Bouldin: Lower = better separation
```

#### Direct Usage

```python
from molblender.models.api.core.splitting import feature_clustering_split
from molblender.data import MolecularDataset

# Example 1: User-provided features (3D embeddings)
from molblender.representations import SomeEmbeddingGenerator

generator = SomeEmbeddingGenerator()
features = generator.generate(dataset)  # Shape: (n_molecules, n_features)

splits = feature_clustering_split(
    dataset=dataset,
    features=features,              # Provide custom features
    clustering_algorithm='kmeans',
    n_clusters=None,                # Auto-select optimal k
    auto_select_k=True,
    test_size=0.2,
    random_state=42
)

# Example 2: RDKit descriptors with hierarchical clustering
splits = feature_clustering_split(
    dataset=dataset,
    use_descriptors=True,           # Use ~20 RDKit descriptors
    clustering_algorithm='hierarchical',
    auto_select_k=True,
    test_size=0.25,
    random_state=42
)

# Example 3: DBSCAN with fingerprints
splits = feature_clustering_split(
    dataset=dataset,
    clustering_algorithm='dbscan',
    eps=0.5,                        # DBSCAN epsilon
    min_samples=5,                  # Min samples per cluster
    fingerprint_type='morgan',
    radius=2,
    nbits=2048,
    test_size=0.2,
    random_state=42
)

# Access results
train_idx = splits['train_indices']
test_idx = splits['test_indices']
info = splits['split_info']

print(f"Clusters: {info['n_clusters']}")
print(f"Silhouette score: {info['silhouette_score']:.3f}")
print(f"Feature source: {info['feature_source']}")
```

#### Implementation Details

```python
def feature_clustering_split(
    dataset: MolecularDataset,
    test_size: float = 0.2,
    clustering_algorithm: Literal['kmeans', 'hierarchical', 'dbscan'] = 'kmeans',
    n_clusters: Optional[int] = None,
    auto_select_k: bool = True,
    features: Optional[np.ndarray] = None,
    use_descriptors: bool = False,
    fingerprint_type: str = 'morgan',
    radius: int = 2,
    nbits: int = 2048,
    random_state: int = 42,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Feature-based clustering + leave-cluster-out splitting.

    Supports arbitrary molecular representations beyond fingerprints.
    """
```

**Key RDKit Descriptors Computed** (when `use_descriptors=True`):
- MW, LogP, TPSA, NumRotatableBonds
- NumHBondDonors, NumHBondAcceptors
- NumAromaticRings, FractionCSP3
- MolMR, BalabanJ, BertzCT, Chi0v
- HallKierAlpha, Kappa1, LabuteASA
- PEOE_VSA1, SMR_VSA1

#### Feature Clustering vs Butina

| Feature | Feature Clustering | Butina |
|---------|-------------------|--------|
| **Feature Input** | Arbitrary (3D, embeddings, descriptors) | Fingerprints only |
| **Similarity Metric** | Euclidean distance (after standardization) | Tanimoto similarity |
| **Clustering Algorithm** | K-means / Hierarchical / DBSCAN | Sphere exclusion (Butina) |
| **Cluster Count** | Manual or auto-optimized (Silhouette) | Auto (threshold-based) |
| **Use Case** | Non-fingerprint representations | Fingerprint-based validation |
| **Flexibility** | High (algorithm choice, features) | Moderate (fingerprint + threshold) |

**When to Use Feature Clustering**:
- ✅ Using 3D molecular embeddings (e.g., Boltz-2 structure predictions)
- ✅ Language model representations (ChemBERTa, MolFormer)
- ✅ Quantum chemical descriptors or docking features
- ✅ Need control over clustering algorithm

**When to Use Butina**:
- ✅ Standard fingerprint-based validation
- ✅ Need Tanimoto similarity specifically
- ✅ Simpler, domain-specific clustering

```python
# Typical workflow: Boltz-2 3D embeddings + Feature Clustering
from molblender.representations import Boltz2Embedder

# Generate 3D structure embeddings
embedder = Boltz2Embedder()
embeddings = embedder.generate(dataset)  # Shape: (n, 768)

# Cluster-based split in embedding space
splits = feature_clustering_split(
    dataset=dataset,
    features=embeddings,
    clustering_algorithm='kmeans',
    auto_select_k=True,
    test_size=0.2
)

# Result: Test set contains molecules from distinct 3D structure clusters
```

#### Advantages

```{admonition} Why Use Feature Clustering Split?
:class: tip

**1. Representation Flexibility**: Works with any molecular representation (not limited to fingerprints)

**2. Algorithm Control**: Choose clustering algorithm based on data distribution

**3. Auto K-Selection**: Silhouette score optimization removes manual tuning

**4. Quality Assurance**: Three independent metrics (Silhouette, CH, DB) validate clustering

**5. Standardization**: Auto feature scaling ensures fair distance computation

**6. Advanced Representations**: Ideal for 3D, language models, quantum features
```

### 8. User-Provided Splits

Custom splitting for specialized scenarios like scaffold or temporal splits.

#### Configuration

```python
# Example: Scaffold-based split
from rdkit.Chem.Scaffolds import MurckoScaffold

# Compute scaffolds for your molecules
scaffolds = [
    MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    for mol in dataset.molecules
]

# Create custom train/test indices
# (Example implementation - you would implement scaffold-based logic)
train_indices, test_indices = custom_scaffold_split(scaffolds, test_size=0.2)

# Pass to universal_screen
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="user_provided",
    user_splits={
        'train_indices': train_indices,
        'test_indices': test_indices
    }
)
```

#### Use Cases

- **Scaffold splits**: Ensure test set contains novel chemical scaffolds
- **Temporal splits**: Time-based train/test division
- **Stratified splits**: Custom stratification logic
- **External test sets**: Pre-defined validation datasets

## Cross-Validation Details

### Adaptive CV Configuration

MolBlender automatically adjusts cross-validation based on dataset size and task type.

#### Automatic Fold Adjustment

**Code Location**: `evaluation/evaluator.py:288-299`

```python
def _cross_validate(self, model, X, y):
    """Perform cross-validation with automatic fold adjustment."""

    # Validate cv_folds parameter
    cv_folds = self.config.cv_folds
    n_samples = len(y)

    # Ensure we don't have more folds than samples
    if cv_folds > n_samples:
        logger.warning(f"cv_folds={cv_folds} > n_samples={n_samples}, "
                      f"using cv_folds={n_samples}")
        cv_folds = n_samples

    # Minimum 2 folds required
    if cv_folds < 2:
        logger.warning(f"cv_folds={cv_folds} invalid, using cv_folds=2")
        cv_folds = 2
```

#### Classification vs Regression

**Code Location**: `evaluation/evaluator.py:302-315`

```python
# Create CV splitter based on task type
if self.config.task_type in [TaskType.CLASSIFICATION,
                             TaskType.BINARY_CLASSIFICATION]:
    # Use StratifiedKFold for classification
    cv_splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=self.config.random_state
    )
else:
    # Use KFold for regression
    cv_splitter = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=self.config.random_state
    )
```

### Stratified Sampling

For classification tasks, MolBlender automatically uses **stratified sampling** to maintain class balance across folds.

#### Benefits

- **Maintains class distribution** in each fold
- **Prevents fold-to-fold variance** due to class imbalance
- **More reliable CV scores** for imbalanced datasets

#### Example

```python
# Classification dataset with imbalanced classes
# Class 0: 800 samples, Class 1: 200 samples

# Without stratification (bad):
# Fold 1 might have: Class 0: 195, Class 1: 5 (95% vs 5%)
# Fold 2 might have: Class 0: 165, Class 1: 35 (83% vs 17%)

# With StratifiedKFold (good):
# All folds maintain: Class 0: 160, Class 1: 40 (80% vs 20%)
```

## Choosing the Right Strategy

### Decision Tree

```
How much data do you have?
    │
    ├─ < 100 samples
    │      └─→ Use cv_only (maximize data usage)
    │
    ├─ 100-500 samples
    │      ├─ Need HPO? → nested_cv
    │      └─ Otherwise → train_test (test_size=0.3)
    │
    ├─ 500-5000 samples
    │      ├─ Need HPO? → train_val_test
    │      └─ Otherwise → train_test (test_size=0.2)
    │
    └─ > 5000 samples
           ├─ Need HPO? → train_val_test
           ├─ Academic study? → nested_cv
           └─ Otherwise → train_test (test_size=0.15)
```

### Recommended Configurations

| Scenario | Strategy | test_size | cv_folds | Rationale |
|----------|----------|-----------|----------|-----------|
| Quick screening (any size) | `train_test` | 0.2 | 3 | Fast, reasonable estimates |
| Standard screening (>500) | `train_test` | 0.2 | 5 | Balanced, industry standard |
| Small dataset (<100) | `cv_only` | — | 5 | Maximum data utilization |
| Large dataset (>10K) | `train_test` | 0.1 | 3 | Efficient, large test set |
| HPO required (>1K) | `train_val_test` | 0.15 | 0.15 | Dedicated validation set |
| Research/publication | `nested_cv` | — | 5/3 | Unbiased performance |
| Scaffold split needed | `user_provided` | Custom | 5 | Domain-specific |

## Reproducibility

### Fixed Random Seeds

All splitting strategies use fixed random seeds by default to ensure reproducibility.

**Code Location**: `models/api/core/base.py:215-217`

```python
@dataclass
class ScreeningConfig:
    """Configuration for model screening."""

    cv_folds: int = 5            # Cross-validation folds
    test_size: float = 0.2       # Test set proportion
    random_state: int = 42       # Fixed random seed
```

### Guaranteed Reproducibility

```python
# Run 1
results1 = universal_screen(
    dataset=dataset,
    target_column="activity",
    random_state=42  # Fixed
)

# Run 2 (different session)
results2 = universal_screen(
    dataset=dataset,
    target_column="activity",
    random_state=42  # Same seed
)

# Guarantee: results1 and results2 will have identical splits
assert (results1['test_indices'] == results2['test_indices']).all()
```

### What's Reproducible

✅ **Guaranteed reproducible** (with same `random_state`):
- Train/test split indices
- Cross-validation fold assignments
- Model training (if model uses same seed)
- Final test scores

⚠️ **May vary slightly**:
- Training time (system load dependent)
- Memory usage (Python GC behavior)

## Advanced Usage

### Custom Splitting Logic

If you need specialized splitting logic not covered by the built-in strategies, use the `user_provided` strategy:

```python
from molblender.models.api.core.splitting import validate_user_splits

# Your custom splitting logic
def my_custom_split(dataset, test_ratio=0.2):
    """Custom split based on molecular properties."""

    # Example: Split by molecular weight
    mol_weights = [mol.GetDescriptors()['MolWt']
                   for mol in dataset.molecules]

    # Sort by molecular weight
    sorted_indices = np.argsort(mol_weights)
    n_test = int(len(dataset) * test_ratio)

    # Heaviest molecules in test set
    test_indices = sorted_indices[-n_test:]
    train_indices = sorted_indices[:-n_test]

    return train_indices, test_indices

# Create splits
train_idx, test_idx = my_custom_split(dataset, test_ratio=0.2)

# Validate splits (checks for overlaps, coverage, etc.)
validated_splits = validate_user_splits(
    X=dataset.features.values,
    y=dataset.labels.values,
    user_splits={
        'train_indices': train_idx,
        'test_indices': test_idx
    }
)

# Use in screening
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="user_provided",
    user_splits=validated_splits
)
```

### Combining Multiple Strategies

For comprehensive model evaluation, you can run multiple splitting strategies and compare:

```python
strategies = ['train_test', 'nested_cv', 'cv_only']
results_by_strategy = {}

for strategy in strategies:
    results_by_strategy[strategy] = universal_screen(
        dataset=dataset,
        target_column="activity",
        split_strategy=strategy,
        random_state=42
    )

# Compare performance across strategies
for strategy, results in results_by_strategy.items():
    print(f"{strategy}: R² = {results['best_model']['test_r2']:.3f}")
```

## Best Practices

```{admonition} Splitting Best Practices
:class: tip

**Always:**
- Use fixed `random_state` for reproducibility
- Choose `test_size` based on dataset size (see table above)
- Use stratification for classification tasks (automatic)
- Validate custom splits before use

**Never:**
- Use test set for hyperparameter tuning
- Peek at test set during model development
- Use different splits for comparing models
- Ignore warnings about insufficient samples per fold
```

## Performance Considerations

### Memory Efficiency

Different strategies have different memory footprints:

| Strategy | Memory Usage | Speed | Best For |
|----------|--------------|-------|----------|
| `train_test` | Low | Fast | Large datasets |
| `cv_only` | Medium | Medium | Small datasets |
| `train_val_test` | Low | Fast | Large datasets with HPO |
| `nested_cv` | High | Slow | Medium datasets, research |

### Computational Cost

```python
# Fastest: Simple train/test with 3-fold CV
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_test",
    cv_folds=3  # ~2x faster than 5-fold
)

# Slowest: Nested CV with many folds
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="nested_cv",
    outer_cv_folds=5,
    inner_cv_folds=5  # 25 total model fits per model type
)
```

## Related Topics

- {doc}`../models/methodology` - Complete evaluation methodology documentation
- {doc}`../models/screening` - Model screening API reference
- {doc}`dataset` - Dataset management guide

## References

- [Scikit-learn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Nested Cross-Validation Explained](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Stratified Sampling](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

## Summary

```{admonition} Key Takeaways
:class: tip

1. **5 Splitting Strategies**: train_test (default), train_val_test, nested_cv, cv_only, user_provided
2. **Automatic Stratification**: Classification tasks use StratifiedKFold automatically
3. **Fixed Random Seeds**: `random_state=42` ensures complete reproducibility
4. **Adaptive Configuration**: CV folds automatically adjusted based on dataset size
5. **Flexible Integration**: Easy to plug in custom splitting logic via `user_provided`
```

## Advanced Molecular Splitting Strategies

MolBlender provides advanced molecular-aware splitting strategies adapted from the [splito package](https://github.com/datamol-io/splito), designed specifically for rigorous out-of-distribution (OOD) evaluation in drug discovery and molecular machine learning.

```{admonition} New in Version 0.3.0
:class: note

Advanced splitting methods from splito package: **PerimeterSplit**, **MolecularWeightSplit**, **MOODSplitter**, and **LoSplitter** for lead optimization.
```

### Why Advanced Splitting Matters

Standard random splitting often produces **overly optimistic** performance estimates because:
- Test molecules are too similar to training molecules
- Model memorizes chemical patterns rather than learning generalizable rules
- Real-world deployment involves truly novel structures

Advanced splitting strategies address this by creating **realistic OOD scenarios**:

| Method | OOD Challenge | Real-World Scenario |
|--------|---------------|---------------------|
| **Perimeter** | Test on chemical space perimeter | Virtual screening on diverse libraries |
| **Molecular Weight** | Test on different size molecules | Generalization across MW ranges |
| **MOOD** | Deployment-aware selection | Optimized for specific target space |
| **Lead Optimization** | Test on similar molecule clusters | SAR exploration within chemical series |

### 9. Perimeter Split (Extrapolation-Oriented)

Places the most dissimilar molecule pairs in the test set, forcing the model to extrapolate to the perimeter of the chemical space.

#### What is Perimeter Split?

Perimeter split identifies pairs of molecules with **maximum pairwise distance** in fingerprint space and assigns them to the test set. This creates a challenging OOD scenario where the test set lies on the "edge" of the training data distribution.

**Algorithm**:
1. Compute molecular fingerprints (ECFP4 by default)
2. Reduce to k-means cluster centers (default: 25 clusters)
3. Compute pairwise distances between clusters
4. Iteratively select furthest pairs for test set
5. Assign remaining molecules to maintain test_size ratio

#### MolecularDataset Integration

```python
from molblender.data import MolecularDataset

# Load dataset
dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    label_columns=["activity"]
)

# Perimeter split - most dissimilar molecules in test
train, test = dataset.train_test_split(
    method='perimeter',
    test_size=0.2,
    n_clusters=25,          # K-means clusters for speed
    metric='euclidean',     # Distance metric
    random_state=42
)

print(f"Train: {len(train)}, Test: {len(test)}")
```

#### Standalone Functional API

```python
from molblender.data.dataset.splitting import train_test_split
import numpy as np

# Your data
smiles = ["CCO", "c1ccccc1", "CC(=O)O", ...]
X = np.array([...])  # Feature matrix
y = np.array([...])  # Target values

# Perimeter split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    molecules=smiles,
    method='perimeter',
    test_size=0.2,
    n_clusters=25,
    random_state=42
)
```

#### Class-based API

```python
from molblender.data.dataset.splitting import PerimeterSplit

# Create splitter
splitter = PerimeterSplit(
    test_size=0.2,
    n_clusters=25,
    metric='euclidean',
    random_state=42
)

# Generate splits
train_idx, test_idx = next(splitter.split(smiles))

# Apply to data
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
```

#### Use Cases

**Virtual Screening:**
- Test model's ability to predict activity for **structurally diverse** compounds
- Evaluate extrapolation beyond the training chemical space
- More realistic estimate for hit discovery on diverse libraries

**Example:**
```python
# Drug discovery scenario
from molblender.models import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="pIC50",
    split_strategy="perimeter",  # Use via models API
    test_size=0.2,
    n_clusters=25
)

# Test set contains most dissimilar molecules
# → More conservative performance estimate
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_size` | float | 0.2 | Fraction of data for test set |
| `n_clusters` | int | 25 | K-means clusters (reduces computation) |
| `metric` | str | 'euclidean' | Distance metric |
| `random_state` | int | None | Random seed for reproducibility |

```{tip}
**Choosing n_clusters**: 
- Larger values (50-100): More accurate distance calculation, slower
- Smaller values (10-25): Faster, approximation via cluster centers
- For datasets < 1000 molecules, use n_clusters = sqrt(n_samples)
```

### 10. Molecular Weight Split

Splits molecules by molecular weight, testing generalization across different molecular sizes.

#### How It Works

```
Dataset (1000 molecules)
    ↓
Compute molecular weights
    ↓
Sort by MW
    ├─→ Train set: MW 100-300 Da (small molecules)
    └─→ Test set: MW 400-600 Da (large molecules)
       ↓
    Test generalization to larger/smaller molecules
```

#### Configuration

```python
# Generalize from small to large molecules
train, test = dataset.train_test_split(
    method='molecular_weight',
    test_size=0.2,
    generalize_to_larger=True,    # Train on small, test on large
    random_state=42
)

# Or vice versa (train on large, test on small)
train, test = dataset.train_test_split(
    method='molecular_weight',
    generalize_to_larger=False,   # Train on large, test on small
    test_size=0.2
)
```

#### Functional API

```python
from molblender.data.dataset.splitting import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    molecules=smiles,
    method='molecular_weight',
    generalize_to_larger=True,
    test_size=0.2
)

# Verify MW distribution
from molblender.data.dataset.splitting import compute_molecular_weights
train_mws = compute_molecular_weights([smiles[i] for i in range(len(X_train))])
test_mws = compute_molecular_weights([smiles[i] for i in range(len(X_test))])

print(f"Train MW range: {train_mws.min():.1f} - {train_mws.max():.1f}")
print(f"Test MW range: {test_mws.min():.1f} - {test_mws.max():.1f}")
```

#### Use Cases

**Peptide Drug Discovery:**
- Train on small molecule fragments
- Test on larger peptide-like structures

**Fragment-to-Lead Optimization:**
- Evaluate model's ability to predict activity as molecules grow in size
- Test generalization from fragments to drug-like molecules

**Example:**
```python
# Fragment-based drug discovery
dataset = MolecularDataset.from_csv(
    "fragments_and_leads.csv",
    input_column="SMILES",
    label_columns=["binding_affinity"]
)

# Train on fragments (MW < 300), test on leads (MW > 300)
train, test = dataset.train_test_split(
    method='molecular_weight',
    generalize_to_larger=True,
    test_size=0.3
)

# Results reflect generalization from fragments to leads
```

### 11. MOOD Split (Model-Optimized Out-of-Distribution)

Automatically selects the best splitting strategy based on similarity to deployment data.

#### What is MOOD?

MOOD (Model-Optimized Out-of-Distribution) splitter evaluates multiple candidate splitting strategies and selects the one whose **test set is most similar** to your expected deployment distribution.

**Algorithm**:
1. Provide deployment molecules (expected real-world data)
2. MOOD evaluates candidate splitters (e.g., perimeter + MW)
3. Compute test set similarity to deployment set
4. Select splitter with highest similarity
5. Use prescribed splitter for final train/test split

#### Configuration

```python
# Deployment molecules (what you'll actually predict on)
deployment_smiles = [
    "c1ccc(O)cc1",    # Phenolic compounds
    "CCCCCO",         # Aliphatic alcohols
    "CC(=O)OC"        # Esters
]

# MOOD automatically chooses best split strategy
train, test = dataset.train_test_split(
    method='mood',
    deployment_smiles=deployment_smiles,
    test_size=0.2,
    n_clusters=25,
    random_state=42
)
```

#### Functional API with Custom Candidates

```python
from molblender.data.dataset.splitting import (
    train_test_split,
    PerimeterSplit,
    MolecularWeightSplit
)

# Define candidate splitters
candidates = {
    'perimeter': PerimeterSplit(test_size=0.2, n_clusters=25),
    'molecular_weight': MolecularWeightSplit(
        test_size=0.2,
        generalize_to_larger=True,
        smiles=smiles
    ),
}

# MOOD selects best one
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    molecules=smiles,
    deployment_molecules=deployment_smiles,
    method='mood',
    test_size=0.2,
    candidate_methods=candidates,  # Custom candidates
    random_state=42
)
```

#### Class-based API

```python
from molblender.data.dataset.splitting import MOODSplitter

# Create splitter with candidates
splitter = MOODSplitter(
    candidate_splitters=candidates,
    metric='euclidean'
)

# Fit to determine best strategy
splitter.fit(training_smiles, deployment_smiles)

# Check which was selected
print(f"Prescribed splitter: {splitter._prescribed_splitter_label}")

# Generate splits using best strategy
train_idx, test_idx = next(splitter.split(smiles))
```

#### Use Cases

**Deployment-Aware Validation:**
- You have a specific target chemical space for deployment
- Want test set to mimic real-world distribution
- Optimize split strategy for your use case

**Example:**
```python
# Virtual screening for kinase inhibitors
# Deployment: ATP-competitive inhibitors

# Fetch representative ATP-competitive structures
deployment_smiles = [
    "c1ccc2c(c1)nc(nc2N)N",  # Representative kinase inhibitor
    # ... more deployment structures
]

train, test = dataset.train_test_split(
    method='mood',
    deployment_smiles=deployment_smiles,
    test_size=0.2
)

# Test set now resembles ATP-competitive inhibitor space
# → More realistic validation for deployment
```

### 12. Lead Optimization Split (LoSplitter)

Creates test clusters of structurally similar molecules, ideal for SAR (Structure-Activity Relationship) exploration.

#### What is Lead Optimization Split?

Unlike standard splits, LoSplitter returns:
- **Training set**: Diverse molecules
- **Test clusters**: Multiple clusters of similar molecules (not a single test set)

Each test cluster represents a **chemical series** for SAR analysis.

**Algorithm**:
1. Cluster molecules by Tanimoto similarity
2. Filter clusters by:
   - Minimum size (e.g., ≥5 molecules)
   - Activity variance (std > threshold)
3. Select top N diverse clusters
4. Return training set + list of test clusters

#### Configuration

```python
# Lead optimization split
train_dataset, test_clusters, y_train, y_test_clusters = dataset.train_test_split(
    method='lead_opt',
    lo_threshold=0.4,           # Tanimoto similarity threshold
    lo_min_cluster_size=5,      # Minimum molecules per cluster
    lo_max_clusters=10,         # Maximum test clusters
    lo_std_threshold=0.60       # Activity variance threshold
)

# test_clusters is a list of MolecularDataset objects
print(f"Number of test clusters: {len(test_clusters)}")
for i, cluster in enumerate(test_clusters):
    print(f"Cluster {i+1}: {len(cluster)} molecules")
```

#### Functional API

```python
from molblender.data.dataset.splitting import train_test_split

X_train, X_clusters, y_train, y_clusters = train_test_split(
    X, y,
    molecules=smiles,
    method='lead_opt',
    lo_threshold=0.5,
    lo_min_cluster_size=5,
    lo_max_clusters=10,
    lo_std_threshold=0.60
)

# X_clusters and y_clusters are lists of numpy arrays
print(f"Training set: {len(X_train)} molecules")
print(f"Test clusters: {len(X_clusters)}")
for i, cluster_X in enumerate(X_clusters):
    print(f"  Cluster {i+1}: {len(cluster_X)} molecules")
```

#### Class-based API

```python
from molblender.data.dataset.splitting import LoSplitter

splitter = LoSplitter(
    threshold=0.5,
    min_cluster_size=5,
    max_clusters=10,
    std_threshold=0.60
)

train_idx, cluster_idx_list = splitter.split(
    smiles=smiles,
    values=activity_values.tolist()
)

# train_idx: training set indices
# cluster_idx_list: list of test cluster indices
print(f"Train: {len(train_idx)}, Clusters: {len(cluster_idx_list)}")
```

#### Use Cases

**SAR Exploration:**
- Evaluate model's ability to predict activity within chemical series
- Test interpolation within local chemical space
- Identify SAR-consistent vs SAR-inconsistent predictions

**Lead Optimization Campaigns:**
- Each test cluster = potential lead series
- Predict activity for new analogs within series
- Prioritize series with confident predictions

**Scaffold Hopping Scenarios:**
A real-world pharmaceutical use case from the meeting discussion:

```{admonition} Scaffold Hopping Example
:class: note

A medicinal chemistry team exhaustively explores **Scaffold 1** with many functional group variations but hits a ceiling (e.g., can't achieve <100 nM potency). They then perform **scaffold hopping** to explore Scaffolds 2, 3, 4 with limited compounds each.

One of the new scaffolds shows promising results. The team wants to predict activity for additional untested compounds in that scaffold series.

**LoSplitter addresses this scenario:**
1. **Training set**: Scaffold 1 data + 1-2 compounds from each new scaffold (known leads)
2. **Test clusters**: Remaining compounds from each new scaffold series
3. **Evaluation**: Can the model generalize within a scaffold family based on functional group variations learned from other scaffolds?

This validates whether a model can successfully:
- Learn from extensive data on one scaffold
- Transfer that knowledge to predict activity in related but different scaffolds
- Guide which new scaffold series to prioritize for further exploration
```

**Example:**
```python
# Medicinal chemistry campaign
dataset = MolecularDataset.from_csv(
    "lead_series.csv",
    input_column="SMILES",
    label_columns=["IC50"]
)

# Get test clusters
train, test_clusters, y_train, y_clusters = dataset.train_test_split(
    method='lead_opt',
    lo_threshold=0.4,
    lo_min_cluster_size=5
)

# Evaluate model on each cluster independently
for i, cluster in enumerate(test_clusters):
    # Train model
    model.fit(train.features.values, y_train)
    
    # Predict on cluster
    cluster_preds = model.predict(cluster.features.values)
    cluster_true = y_clusters[i]
    
    # Compute cluster-specific metrics
    from sklearn.metrics import r2_score
    r2 = r2_score(cluster_true, cluster_preds)
    print(f"Cluster {i+1} R²: {r2:.3f}")
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lo_threshold` | float | 0.4 | Tanimoto similarity for clustering |
| `lo_min_cluster_size` | int | 5 | Minimum molecules per cluster |
| `lo_max_clusters` | int | 50 | Maximum test clusters to return |
| `lo_std_threshold` | float | 0.60 | Activity std deviation threshold |

```{admonition} Why Cluster-Based Validation?
:class: tip

**Traditional Split**: One test set, average performance
**LoSplitter**: Multiple test clusters, series-specific performance

Benefits:
- Identify which chemical series are predictable
- Detect series where model fails (SAR cliffs)
- Guide medicinal chemistry decisions per series
```

### 13. Custom User-Provided Split

When you have predefined train/test assignments from external sources or need full control over the splitting process.

#### What is Custom Split?

Custom split allows you to:
- Use existing train/test assignments from benchmark datasets
- Apply pre-computed temporal or experimental splits
- Integrate splits from external pipelines or publications

#### Configuration

**Option 1: Using a split column from metadata**

```python
# Dataset with a pre-existing split column
dataset = MolecularDataset.from_csv(
    "benchmark_data.csv",
    input_column="SMILES",
    label_columns=["pIC50"]
)

# Assuming metadata has a 'split' column with 'train'/'test' values
dataset.metadata['split'] = ['train', 'train', 'test', 'train', 'test', ...]

train_ds, test_ds = dataset.train_test_split(
    method='custom',
    split_column='split'
)
```

**Option 2: Using explicit indices**

```python
# Predefined indices (e.g., from temporal split, experimental batches)
train_indices = [0, 1, 2, 5, 6, 7, 10, 11, 12]
test_indices = [3, 4, 8, 9, 13, 14]

train_ds, test_ds = dataset.train_test_split(
    method='custom',
    train_indices=train_indices,
    test_indices=test_indices
)
```

#### Functional API

```python
from molblender.data.dataset.splitting import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    method='custom',
    train_indices=train_idx,
    test_indices=test_idx
)
```

#### Use Cases

**Benchmark Reproduction:**
- Use exact same train/test splits from published papers
- Ensure fair comparison with literature results

**Temporal Splits:**
- Train on older data, test on newer data
- Simulate real-world deployment scenarios

**Experimental Design:**
- Group experimental batches together
- Account for batch effects in validation

**External Annotations:**
- Use expert-curated splits
- Apply domain-specific splitting criteria

#### Supported Split Column Formats

| Format | Train Value | Test Value |
|--------|-------------|------------|
| String | 'train' | 'test' |
| Numeric | 0 | 1 |
| Boolean | False | True |

```{admonition} When to Use Custom Split?
:class: tip

Use custom split when:
- You need to reproduce specific benchmark results
- Your data has inherent temporal or experimental structure
- External experts have defined optimal splits for your domain
- You want to use the same split across multiple experiments
```

### Advanced Splitting Summary

```{admonition} Choosing Advanced Splitting Methods
:class: tip

| Method | Use When | Performance Expectation |
|--------|----------|------------------------|
| **Perimeter** | Testing extrapolation to diverse molecules | Most conservative (lowest R²) |
| **Molecular Weight** | Generalizing across MW ranges | Conservative for large MW gap |
| **MOOD** | Have deployment data, want optimized split | Deployment-realistic |
| **Lead Optimization** | SAR exploration, multiple series | Cluster-dependent (variable R²) |

**General Guidance:**
- Use **Perimeter** for virtual screening validation
- Use **Molecular Weight** for fragment-to-lead pipelines
- Use **MOOD** when you know deployment distribution
- Use **Lead Optimization** for medicinal chemistry campaigns
```

### Integration with Universal Screening

All advanced splitting methods integrate seamlessly with the `universal_screen` API:

```python
from molblender.models import universal_screen

# Example: MOOD split in model screening
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="mood",           # Advanced split
    deployment_smiles=deploy_smiles,
    test_size=0.2,
    random_state=42
)
```

```{note}
Currently, advanced splitting methods (`perimeter`, `molecular_weight`, `mood`, `lead_opt`) are available via the `MolecularDataset.train_test_split()` method and functional API, but not yet fully integrated into `universal_screen`. Full integration planned for v0.4.0.
```

### Comparison: Standard vs Advanced Splitting

```python
# Standard random split (optimistic)
train_random, test_random = dataset.train_test_split(
    method='random',
    test_size=0.2
)

# Advanced perimeter split (realistic)
train_perimeter, test_perimeter = dataset.train_test_split(
    method='perimeter',
    test_size=0.2,
    n_clusters=25
)

# Typical outcome:
# Random split R²: 0.78 (optimistic)
# Perimeter split R²: 0.62 (realistic)
```

### API Reference

See {doc}`../../api/data/splitting` for complete API documentation.

### References

```{admonition} Attribution
:class: note

Advanced splitting strategies adapted from the **splito package**:
- Repository: https://github.com/datamol-io/splito
- License: Apache 2.0
- Copyright (c) 2024 Datamol.io

All `datamol` dependencies have been replaced with direct RDKit calls for compatibility.
```

