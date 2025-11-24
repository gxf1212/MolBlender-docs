# Property & Diversity-Based Splitting

Splitting strategies based on molecular properties and chemical space diversity for challenging validation scenarios.

```{admonition} Overview
:class: note

Property and diversity-based splitting strategies create challenging OOD scenarios:
- **dnr**: Split by Different Neighbor Ratio (tests generalization on rough SAR regions)
- **maxmin**: Split by MaxMinPicker diversity (tests extrapolation to diverse chemical space)

These strategies identify and isolate challenging molecules for rigorous model evaluation.
```

**Navigation**: {doc}`index` > Property & Diversity Strategies

---

## 1. DNR-Based Splitting

Systematically tests model performance on rough SAR regions and challenging molecules.

### What is DNR?

**DNR (Different Neighbor Ratio)** quantifies how much a molecule's property differs from its structurally similar neighbors:
- **Low DNR**: Smooth SAR region (easy to predict)
- **High DNR**: Rough SAR region / activity cliffs (hard to predict)

DNR-based splitting places high-DNR molecules (challenging cases) in the test set.

**Definition** (from "Upgrading Reliability in Molecular Property Prediction"):
```
DNR(molecule) = (# neighbors with property diff > threshold) / (total # neighbors)

Where:
- Neighbors: Molecules with Tanimoto similarity > 0.5
- Property diff: |property_i - property_neighbor| > 1.0 log unit
```

### When to Use

✅ **Use DNR split when:**
- Testing model robustness on activity cliffs
- Evaluating performance in rough SAR regions
- Identifying challenging molecules for model failure analysis
- Need conservative performance estimates

❌ **Don't use when:**
- Dataset has smooth SAR (uniform property distribution)
- Dataset too small to compute meaningful neighbors (<100 molecules)
- Need scaffold-level generalization (use scaffold split)

### Configuration

```python
from molblender.models import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="pIC50",
    split_strategy="dnr",

    # DNR parameters
    dnr_mode="threshold",           # 'threshold', 'quantile', 'neighbor'
    dnr_threshold=0.3,              # High DNR threshold
    similarity_threshold=0.5,       # Tanimoto threshold for neighbors
    property_diff_threshold=1.0,    # Property difference threshold (log units)

    test_size=0.2,
    random_state=42
)
```

### Split Modes

#### 1. Threshold Mode

Split by DNR threshold: high DNR (challenging) → test, low DNR (smooth) → train.

```python
splits = dnr_split(
    dataset=dataset,
    mode="threshold",
    dnr_threshold=0.3,      # Molecules with DNR > 0.3 → test
    test_size=0.2
)
```

**How it works:**
1. Compute DNR for all molecules
2. Molecules with DNR > threshold → test set
3. Remaining molecules → train set
4. Adjust threshold to achieve target test_size

#### 2. Quantile Mode

Split by DNR quantiles: top X% highest DNR → test.

```python
splits = dnr_split(
    dataset=dataset,
    mode="quantile",
    dnr_quantile=0.2,       # Top 20% highest DNR → test
    test_size=0.2
)
```

**How it works:**
1. Compute DNR for all molecules
2. Sort by DNR (descending)
3. Top 20% highest DNR → test set
4. Remaining 80% → train set

#### 3. Neighbor Mode

Split by neighbor presence: isolated molecules (no neighbors) → test.

```python
splits = dnr_split(
    dataset=dataset,
    mode="neighbor",
    similarity_threshold=0.5,   # Tanimoto threshold
    test_size=0.2
)
```

**How it works:**
1. Compute Tanimoto similarity matrix
2. Identify molecules with no neighbors (similarity < threshold to all others)
3. Isolated molecules → test set
4. Connected molecules → train set

### Direct Usage

```python
from molblender.models.api.core.splitting import dnr_split

# Threshold mode
splits = dnr_split(
    dataset=dataset,
    mode="threshold",
    dnr_threshold=0.3,
    similarity_threshold=0.5,
    property_diff_threshold=1.0,
    test_size=0.2,
    random_state=42
)

# Access results
train_idx = splits['train_indices']
test_idx = splits['test_indices']
dnr_info = splits['split_info']

print(f"Train mean DNR: {dnr_info['train_mean_dnr']:.3f}")
print(f"Test mean DNR: {dnr_info['test_mean_dnr']:.3f}")
print(f"High-DNR molecules in test: {dnr_info['test_high_dnr_count']}")
```

### Implementation Details

**Code Location**: `src/molblender/models/api/core/splitting/dnr.py`

```python
def dnr_split(
    dataset: MolecularDataset,
    mode: str = 'threshold',
    dnr_threshold: float = 0.3,
    dnr_quantile: float = 0.2,
    similarity_threshold: float = 0.5,
    property_diff_threshold: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    DNR-based splitting for challenging molecule validation.

    Three modes:
    - threshold: Split by DNR threshold (high vs low)
    - quantile: Split by DNR quantiles (top X% vs rest)
    - neighbor: Split by neighbor presence (isolated vs connected)
    """
```

### Use Cases

**Activity Cliff Detection:**
```python
# Identify molecules at activity cliffs
splits = dnr_split(
    dataset=kinase_dataset,
    mode="threshold",
    dnr_threshold=0.5,      # Very high DNR = activity cliff
    property_diff_threshold=2.0  # 2 log unit difference
)

# Test set contains challenging activity cliff cases
```

**Model Robustness Testing:**
```python
# Test model on rough SAR regions
results = universal_screen(
    dataset=dataset,
    target_column="pIC50",
    split_strategy="dnr",
    dnr_mode="quantile",
    dnr_quantile=0.2  # Top 20% most challenging molecules
)

# Conservative performance estimate
print(f"Test R² (rough SAR): {results['best_model']['test_r2']:.3f}")
```

### Benefits

```{admonition} Why Use DNR Split?
:class: tip

**1. Identifies Challenging Cases**: Systematically isolates hard-to-predict molecules

**2. Tests SAR Robustness**: Evaluates model performance on activity cliffs

**3. Conservative Estimates**: More realistic than random split for rough SARs

**4. Diagnostic Value**: Reveals where model fails (rough vs smooth regions)

**5. Literature Validated**: Based on "Upgrading Reliability" paper methodology
```

### Interpretation

**DNR Values:**
- DNR = 0.0: Smooth SAR (all neighbors have similar properties)
- DNR = 0.5: Mixed SAR (half of neighbors differ significantly)
- DNR = 1.0: Rough SAR / activity cliff (all neighbors differ)

**Performance Expectations:**
- Random split: R² ≈ 0.75 (optimistic)
- DNR split (threshold mode): R² ≈ 0.55 (realistic for rough SAR)
- DNR split (quantile mode): R² ≈ 0.45 (very conservative, hardest cases)

---

## 2. MaxMin Diversity Split

Diversity-based splitting using RDKit's MaxMinPicker algorithm for chemical space exploration.

### What is MaxMin?

**MaxMinPicker** is a greedy algorithm that selects maximally diverse molecules:
1. Pick random starting molecule
2. Iteratively add molecule with **maximum minimum distance** to already selected set
3. Result: Diverse subset covering chemical space

MaxMin split creates either:
- **Friendly mode**: Diverse training set (broad coverage for learning)
- **Unfriendly mode**: Diverse test set (challenging extrapolation)

### When to Use

✅ **Use MaxMin split when:**
- Testing extrapolation to diverse chemical space
- Virtual screening validation on diverse libraries
- Evaluating chemical space coverage
- Need challenging OOD scenario (unfriendly mode)

❌ **Don't use when:**
- Dataset is already very diverse (MaxMin won't help)
- Need scaffold-level generalization (use scaffold split)
- Dataset too small (<100 molecules)

### Configuration

```python
from molblender.models import universal_screen

# Unfriendly mode: Diverse test set (challenging)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="maxmin",

    # MaxMin parameters
    maxmin_mode="unfriendly",       # 'friendly' or 'unfriendly'
    fingerprint_type="morgan",
    fp_radius=2,
    fp_nbits=2048,

    test_size=0.2,
    random_state=42
)
```

### Split Modes

#### Friendly Mode

Diverse training set ensures broad chemical space coverage for learning.

```python
splits = maxmin_split(
    dataset=dataset,
    mode="friendly",        # Diverse train set
    test_size=0.2
)
```

**How it works:**
1. Use MaxMinPicker to select diverse molecules for **training set** (80%)
2. Remaining molecules → test set (20%)
3. Result: Training set covers diverse chemical space, test set is more clustered

**Use for**: Learning from diverse chemical space

#### Unfriendly Mode

Diverse test set creates most challenging generalization scenario.

```python
splits = maxmin_split(
    dataset=dataset,
    mode="unfriendly",      # Diverse test set
    test_size=0.2
)
```

**How it works:**
1. Use MaxMinPicker to select diverse molecules for **test set** (20%)
2. Remaining molecules → train set (80%)
3. Result: Test set spans diverse chemical space, training set is more clustered

**Use for**: Challenging extrapolation validation

### Direct Usage

```python
from molblender.models.api.core.splitting import maxmin_split

# Unfriendly mode (diverse test set)
splits = maxmin_split(
    dataset=dataset,
    mode="unfriendly",
    fingerprint_type="morgan",
    radius=2,
    nbits=2048,
    test_size=0.2,
    random_state=42
)

# Access results
train_idx = splits['train_indices']
test_idx = splits['test_indices']
stats = splits['split_info']

print(f"Train avg similarity: {stats['train_avg_similarity']:.3f}")
print(f"Test avg similarity: {stats['test_avg_similarity']:.3f}")
print(f"Train-test similarity: {stats['train_test_similarity']:.3f}")
```

### Implementation Details

**Code Location**: `src/molblender/models/api/core/splitting/diversity.py`

```python
def maxmin_split(
    dataset: MolecularDataset,
    mode: str = 'unfriendly',
    fingerprint_type: str = 'morgan',
    radius: int = 2,
    nbits: int = 2048,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    MaxMinPicker diversity-based splitting.

    Two modes:
    - friendly: Diverse training set (broad coverage)
    - unfriendly: Diverse test set (challenging extrapolation)
    """
```

### Fingerprint Options

**Morgan (default):**
```python
splits = maxmin_split(dataset, fingerprint_type="morgan", radius=2, nbits=2048)
```

**RDKit Topological:**
```python
splits = maxmin_split(dataset, fingerprint_type="rdkit", nbits=2048)
```

**MACCS Keys:**
```python
splits = maxmin_split(dataset, fingerprint_type="maccs")  # Fixed 166 bits
```

### Use Cases

**Virtual Screening Validation (Unfriendly Mode):**
```python
# Test on diverse molecules
results = universal_screen(
    dataset=screening_library,
    target_column="pIC50",
    split_strategy="maxmin",
    maxmin_mode="unfriendly",  # Diverse test set
    test_size=0.2
)

# Performance estimate for diverse hit discovery
print(f"Diverse test R²: {results['best_model']['test_r2']:.3f}")
```

**Broad Training Coverage (Friendly Mode):**
```python
# Train on diverse chemical space
results = universal_screen(
    dataset=training_data,
    target_column="activity",
    split_strategy="maxmin",
    maxmin_mode="friendly",    # Diverse train set
    test_size=0.2
)

# Model learns from broad chemical space
```

### Benefits

```{admonition} Why Use MaxMin Split?
:class: tip

**1. Diversity Quantification**: Reports similarity statistics for validation

**2. Challenging Extrapolation**: Unfriendly mode tests worst-case scenario

**3. Broad Coverage**: Friendly mode ensures diverse training set

**4. Flexible Fingerprints**: Supports Morgan, RDKit, MACCS

**5. Algorithm Simplicity**: Well-established MaxMinPicker algorithm
```

### Interpretation

**Similarity Statistics:**
- **Train avg similarity**: Average Tanimoto within training set
  - Low (0.2-0.4): Very diverse training set (friendly mode)
  - High (0.6-0.8): Clustered training set (unfriendly mode)

- **Test avg similarity**: Average Tanimoto within test set
  - Low (0.2-0.4): Very diverse test set (unfriendly mode)
  - High (0.6-0.8): Clustered test set (friendly mode)

- **Train-test similarity**: Average Tanimoto between train and test
  - Low (0.2-0.4): Large chemical space gap (challenging)
  - High (0.6-0.8): Small gap (easier generalization)

**Performance Expectations:**
- **Friendly mode**: R² ≈ 0.70 (similar to random split)
- **Unfriendly mode**: R² ≈ 0.50 (more conservative, diverse test)

---

## Quick Comparison

| Strategy | What It Tests | Difficulty | Use Case | Performance |
|----------|---------------|------------|----------|-------------|
| **dnr (threshold)** | Rough SAR regions | High | Activity cliffs | R² ≈ 0.55 |
| **dnr (quantile)** | Top X% hardest | Very High | Robustness testing | R² ≈ 0.45 |
| **maxmin (friendly)** | Standard generalization | Medium | Broad coverage | R² ≈ 0.70 |
| **maxmin (unfriendly)** | Diverse extrapolation | High | Diverse libraries | R² ≈ 0.50 |

---

## Combining with Other Strategies

### DNR + Scaffold

```python
# First: Scaffold split (novel structures)
scaffold_splits = scaffold_split(dataset, test_size=0.3)

# Then: DNR split within scaffold test set (challenging molecules)
dnr_splits = dnr_split(
    dataset.subset(scaffold_splits['test_indices']),
    mode="threshold",
    dnr_threshold=0.4
)

# Result: Novel scaffolds + activity cliffs in test set
```

### MaxMin + Feature Clustering

```python
# First: MaxMin (diverse selection)
maxmin_splits = maxmin_split(dataset, mode="unfriendly", test_size=0.3)

# Then: Feature clustering within diverse test set
clustering_splits = feature_clustering_split(
    dataset.subset(maxmin_splits['test_indices']),
    clustering_algorithm='kmeans',
    n_clusters=5
)

# Result: Multiple diverse test clusters
```

---

## Related Strategies

- **For basic splits**: See {doc}`basic_strategies`
- **For chemical structure**: See {doc}`chemical_strategies` (scaffold, butina)
- **For advanced OOD**: See {doc}`advanced_strategies` (perimeter, MOOD)
- **For custom needs**: See {doc}`custom_splits`

---

**Navigation**: {doc}`index` | Previous: {doc}`chemical_strategies` | Next: {doc}`advanced_strategies`
