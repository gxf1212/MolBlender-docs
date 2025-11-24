# Advanced Splitting Strategies

Research-grade splitting methods adapted from the **splito package** for rigorous out-of-distribution (OOD) evaluation.

```{admonition} Overview
:class: note

Advanced splitting strategies from [splito](https://github.com/datamol-io/splito) create realistic OOD scenarios for drug discovery:
- **perimeter**: Test on chemical space perimeter (extrapolation-oriented)
- **molecular_weight**: Test on different molecular sizes
- **mood**: Model-Optimized Out-of-Distribution split (deployment-aware)
- **lead_opt**: Lead optimization split (returns test clusters for SAR exploration)

These methods provide more conservative performance estimates than random splitting.
```

**Navigation**: {doc}`index` > Advanced Strategies

---

## Why Advanced Splitting Matters

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

---

## 1. Perimeter Split (Extrapolation-Oriented)

Places the most dissimilar molecule pairs in the test set, forcing extrapolation to the perimeter of chemical space.

### What is Perimeter Split?

Perimeter split identifies pairs of molecules with **maximum pairwise distance** in fingerprint space and assigns them to the test set. This creates a challenging OOD scenario where the test set lies on the "edge" of the training data distribution.

**Algorithm:**
1. Compute molecular fingerprints (ECFP4 by default)
2. Reduce to k-means cluster centers (default: 25 clusters)
3. Compute pairwise distances between clusters
4. Iteratively select furthest pairs for test set
5. Assign remaining molecules to maintain test_size ratio

### When to Use

✅ **Use perimeter split when:**
- Virtual screening validation on diverse libraries
- Testing extrapolation beyond training chemical space
- Need most conservative performance estimate
- Evaluating hit discovery on diverse compound collections

❌ **Don't use when:**
- Dataset is already very diverse (perimeter = most molecules)
- Need scaffold-level generalization (use scaffold split)
- Dataset < 100 molecules (insufficient for clustering)

### MolecularDataset Integration

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

### Standalone Functional API

```python
from molblender.data.dataset.splitting import train_test_split
import numpy as np

# Your data
smiles = ["CCO", "c1ccccc1", "CC(=O)O", ...]
X = np.array([...])
y = np.array([...])

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

### Class-Based API

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

### Use Cases

**Virtual Screening:**
```python
from molblender.models import universal_screen

# Drug discovery scenario
results = universal_screen(
    dataset=screening_library,
    target_column="pIC50",
    split_strategy="perimeter",  # Use via models API (future)
    test_size=0.2,
    n_clusters=25
)

# Test set contains most dissimilar molecules
# → More conservative performance estimate
```

### Parameters

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

### Implementation Details

**Code Location**: `src/molblender/data/dataset/splitting/advanced.py`

```python
class PerimeterSplit:
    """
    Extrapolation-oriented split placing most dissimilar molecules in test set.

    Adapted from splito package (Apache 2.0 License).
    """
    def __init__(
        self,
        test_size: float = 0.2,
        n_clusters: int = 25,
        metric: str = 'euclidean',
        random_state: Optional[int] = None
    ):
        ...
```

---

## 2. Molecular Weight Split

Splits molecules by molecular weight, testing generalization across different molecular sizes.

### How It Works

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

### When to Use

✅ **Use molecular weight split when:**
- Fragment-to-lead optimization
- Testing generalization across MW ranges
- Peptide drug discovery (fragments → peptides)
- Size-dependent property prediction

❌ **Don't use when:**
- Dataset has narrow MW range (<100 Da spread)
- MW not relevant to property prediction
- Need structural generalization (use scaffold split)

### Configuration

```python
from molblender.data import MolecularDataset

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

### Functional API

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
train_smiles = [smiles[i] for i in range(len(X_train))]
test_smiles = [smiles[i] for i in range(len(X_test))]

train_mws = compute_molecular_weights(train_smiles)
test_mws = compute_molecular_weights(test_smiles)

print(f"Train MW range: {train_mws.min():.1f} - {train_mws.max():.1f}")
print(f"Test MW range: {test_mws.min():.1f} - {test_mws.max():.1f}")
```

### Use Cases

**Peptide Drug Discovery:**
```python
# Train on small molecule fragments
# Test on larger peptide-like structures

dataset = MolecularDataset.from_csv(
    "fragments_and_peptides.csv",
    input_column="SMILES",
    label_columns=["binding_affinity"]
)

train, test = dataset.train_test_split(
    method='molecular_weight',
    generalize_to_larger=True,
    test_size=0.3
)

# Train MW < 300 Da (fragments)
# Test MW > 300 Da (peptides)
```

**Fragment-Based Drug Discovery:**
```python
# Evaluate model's ability to predict activity as molecules grow

train, test = dataset.train_test_split(
    method='molecular_weight',
    generalize_to_larger=True,
    test_size=0.2
)

# Train on fragments (MW < 300)
# Test on leads (MW > 300)
```

### Implementation Details

**Code Location**: `src/molblender/data/dataset/splitting/advanced.py`

```python
class MolecularWeightSplit:
    """
    Split by molecular weight for size-dependent generalization testing.

    Adapted from splito package (Apache 2.0 License).
    """
    def __init__(
        self,
        test_size: float = 0.2,
        generalize_to_larger: bool = True,
        smiles: Optional[List[str]] = None
    ):
        ...
```

---

## 3. MOOD Split (Model-Optimized Out-of-Distribution)

Automatically selects the best splitting strategy based on similarity to deployment data.

### What is MOOD?

MOOD (Model-Optimized Out-of-Distribution) splitter evaluates multiple candidate splitting strategies and selects the one whose **test set is most similar** to your expected deployment distribution.

**Algorithm:**
1. Provide deployment molecules (expected real-world data)
2. MOOD evaluates candidate splitters (e.g., perimeter + MW)
3. Compute test set similarity to deployment set
4. Select splitter with highest similarity
5. Use prescribed splitter for final train/test split

### When to Use

✅ **Use MOOD when:**
- You have specific target chemical space for deployment
- Want test set to mimic real-world distribution
- Optimize split strategy for your use case
- Need deployment-aware validation

❌ **Don't use when:**
- No deployment data available
- Deployment space unknown
- Standard splits sufficient

### Configuration

```python
from molblender.data import MolecularDataset

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

### Functional API with Custom Candidates

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

### Class-Based API

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

### Use Cases

**Deployment-Aware Validation:**
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

### Implementation Details

**Code Location**: `src/molblender/data/dataset/splitting/advanced.py`

```python
class MOODSplitter:
    """
    Model-Optimized Out-of-Distribution splitter.

    Selects best splitting strategy based on deployment data similarity.
    Adapted from splito package (Apache 2.0 License).
    """
    def __init__(
        self,
        candidate_splitters: Dict[str, Any],
        metric: str = 'euclidean'
    ):
        ...

    def fit(
        self,
        training_smiles: List[str],
        deployment_smiles: List[str]
    ):
        """Determine best splitter based on deployment similarity."""
        ...
```

---

## 4. Lead Optimization Split (LoSplitter)

Creates test clusters of structurally similar molecules for SAR (Structure-Activity Relationship) exploration.

### What is Lead Optimization Split?

Unlike standard splits, LoSplitter returns:
- **Training set**: Diverse molecules
- **Test clusters**: Multiple clusters of similar molecules (not a single test set)

Each test cluster represents a **chemical series** for SAR analysis.

**Algorithm:**
1. Cluster molecules by Tanimoto similarity
2. Filter clusters by:
   - Minimum size (e.g., ≥5 molecules)
   - Activity variance (std > threshold)
3. Select top N diverse clusters
4. Return training set + list of test clusters

### When to Use

✅ **Use LoSplitter when:**
- SAR exploration within chemical series
- Lead optimization campaigns
- Evaluating predictions within local chemical space
- Multiple test clusters needed for series-specific analysis

❌ **Don't use when:**
- Need single test set (use other splits)
- Dataset lacks clear chemical series
- No activity variance within series

### Configuration

```python
from molblender.data import MolecularDataset

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

### Functional API

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

### Class-Based API

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

### Use Cases

#### SAR Exploration

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

#### Scaffold Hopping Scenarios

A real-world pharmaceutical use case:

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

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lo_threshold` | float | 0.4 | Tanimoto similarity for clustering |
| `lo_min_cluster_size` | int | 5 | Minimum molecules per cluster |
| `lo_max_clusters` | int | 50 | Maximum test clusters to return |
| `lo_std_threshold` | float | 0.60 | Activity std deviation threshold |

### Benefits

```{admonition} Why Use LoSplitter?
:class: tip

**Traditional Split**: One test set, average performance

**LoSplitter**: Multiple test clusters, series-specific performance

**Benefits:**
- Identify which chemical series are predictable
- Detect series where model fails (SAR cliffs)
- Guide medicinal chemistry decisions per series
- Series-by-series performance analysis
```

### Implementation Details

**Code Location**: `src/molblender/data/dataset/splitting/advanced.py`

```python
class LoSplitter:
    """
    Lead optimization splitter returning test clusters for SAR exploration.

    Adapted from splito package (Apache 2.0 License).
    """
    def __init__(
        self,
        threshold: float = 0.4,
        min_cluster_size: int = 5,
        max_clusters: int = 50,
        std_threshold: float = 0.60
    ):
        ...
```

---

## Advanced Splitting Summary

```{admonition} Choosing Advanced Methods
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

---

## Integration with Universal Screening

```{note}
Currently, advanced splitting methods (`perimeter`, `molecular_weight`, `mood`, `lead_opt`) are available via the `MolecularDataset.train_test_split()` method and functional API, but not yet fully integrated into `universal_screen`. Full integration planned for future release.
```

**Current Usage:**
```python
from molblender.data import MolecularDataset

# Split first
train, test = dataset.train_test_split(
    method='perimeter',
    test_size=0.2
)

# Then screen
from molblender.models import universal_screen

results = universal_screen(
    dataset=train,  # Train on perimeter split
    target_column="activity",
    # ... screening parameters
)
```

---

## Comparison: Standard vs Advanced Splitting

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

---

## Attribution

```{admonition} License & Attribution
:class: note

Advanced splitting strategies adapted from the **splito package**:
- Repository: https://github.com/datamol-io/splito
- License: Apache 2.0
- Copyright (c) 2024 Datamol.io

All `datamol` dependencies have been replaced with direct RDKit calls for compatibility with MolBlender.
```

---

## Related Strategies

- **For basic splits**: See {doc}`basic_strategies`
- **For chemical structure**: See {doc}`chemical_strategies` (scaffold, butina, feature_clustering)
- **For property-based**: See {doc}`property_strategies` (dnr, maxmin)
- **For custom needs**: See {doc}`custom_splits`

---

**Navigation**: {doc}`index` | Previous: {doc}`property_strategies` | Next: {doc}`custom_splits`
