# Chemical Structure-Based Splitting

Splitting strategies based on molecular scaffolds and structural similarity for drug discovery applications.

```{admonition} Overview
:class: note

Chemical structure-based splitting ensures **realistic validation** by preventing information leakage from structural similarity:
- **scaffold**: Split by Bemis-Murcko or generic scaffolds (novel structure generalization)
- **butina**: Split by Butina clustering (similar molecule generalization)
- **feature_clustering**: Split by K-means/Hierarchical/DBSCAN clustering (custom representations)

These strategies provide more conservative performance estimates than random splitting, better reflecting real-world deployment scenarios.
```

**Navigation**: {doc}`index` > Chemical Strategies

---

## 1. Scaffold Split

Scaffold-based splitting for drug discovery and medicinal chemistry applications.

### When to Use

✅ **Use scaffold split when:**
- Drug discovery applications
- Testing generalization to novel chemical scaffolds
- Virtual screening validation
- Lead optimization campaigns
- Need realistic performance estimates

❌ **Don't use when:**
- Dataset has very few unique scaffolds (<10)
- All molecules share the same scaffold
- Random performance baseline needed for comparison

### Configuration

```python
from molblender.models import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="pIC50",
    split_strategy="scaffold",
    test_size=0.2,
    scaffold_func='bemis_murcko',  # or 'generic'
    split_method='balanced',       # or 'random'
    random_state=42
)
```

### How It Works

Scaffold split ensures that molecules in the test set have **different chemical scaffolds** from those in the training set.

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

### Scaffold Generation Methods

#### 1. Bemis-Murcko Scaffolds (Default)

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

**Use for**: Standard drug discovery (most common)

#### 2. Generic Scaffolds

Further abstracts by replacing all atoms with carbons and all bonds with single bonds.

```python
from molblender.models.api.core.splitting import compute_generic_scaffolds

smiles = [
    "c1ccccc1CCO",    # Aromatic benzene ring
    "C1CCCCC1CCO",    # Aliphatic cyclohexane ring
]

generic_scaffolds = compute_generic_scaffolds(smiles)
# Both have the same generic scaffold: "C1CCCCC1" (6-membered ring)
```

**Use for**: Topological generalization (ignores atom/bond types)

### Split Methods

#### Balanced Split (Default)

Greedily assigns scaffold groups to achieve the desired train/test ratio.

```python
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

#### Random Split

Randomly assigns each scaffold group to either train or test.

```python
result = scaffold_split(
    X, y, smiles,
    split_method='random',  # Random scaffold assignment
    test_size=0.2
)
```

**Strategy:**
1. Randomly shuffle unique scaffolds
2. Assign scaffolds to train/test until target ratio met
3. Result: more variability across runs

### Direct Usage

```python
from molblender.models.api.core.splitting import scaffold_split
import numpy as np

# Your data
smiles = ["c1ccccc1CCO", "c1ccccc1CCN", ...]
X = np.array([...])
y = np.array([...])

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
scaffolds = result['scaffolds']

# Verify no scaffold leakage
train_scaffolds = set(scaffolds[i] for i in result['train_indices'])
test_scaffolds = set(scaffolds[i] for i in result['test_indices'])
overlap = train_scaffolds & test_scaffolds
print(f"Scaffold overlap: {len(overlap)} scaffolds")  # Should be 0
```

### Implementation Details

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

### Why Use Scaffold Split?

```{admonition} Benefits
:class: tip

**1. Realistic Evaluation**: Mimics real-world scenarios where you predict activity for novel chemical matter

**2. Prevents Data Leakage**: Ensures no similar structures in both train and test sets

**3. Conservative Estimates**: Usually gives lower performance than random split, providing honest assessment

**4. Industry Standard**: Widely used in pharmaceutical industry for virtual screening validation
```

### Comparison with Random Split

```python
# Random split (optimistic)
random_results = universal_screen(
    dataset, "pIC50",
    split_strategy="train_test",
    test_size=0.2
)
print(f"Random split R²: {random_results['best_model']['test_r2']:.3f}")

# Scaffold split (realistic)
scaffold_results = universal_screen(
    dataset, "pIC50",
    split_strategy="scaffold",
    test_size=0.2
)
print(f"Scaffold split R²: {scaffold_results['best_model']['test_r2']:.3f}")

# Typical result: Scaffold split R² < Random split R²
# Example: 0.65 vs 0.78
```

```{note}
Scaffold split typically yields **lower performance metrics** than random split because it tests generalization to truly novel structures. This is expected and provides a more realistic estimate of real-world performance.
```

---

## 2. Butina Clustering Split

Leave-cluster-out validation based on Tanimoto similarity clustering.

### When to Use

✅ **Use Butina split when:**
- Testing generalization to **similar but unseen** chemical combinations
- Avoiding information leakage from structural similarity
- Implementing MolAgent-style clustering validation
- Evaluating on OP-like structured design spaces (R group + Y functional group)

❌ **Don't use when:**
- Need scaffold-level generalization (use scaffold split)
- Dataset < 100 molecules (insufficient for clustering)
- All molecules are very dissimilar (similarity threshold too high)

### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="dG",
    split_strategy="butina",
    test_size=0.2,
    butina_similarity_threshold=0.6,   # Tanimoto threshold
    fingerprint_type='morgan',
    fp_radius=2,
    fp_nbits=2048,
    random_state=42
)

# Check cluster statistics
print(f"Number of clusters: {results['split_info']['n_clusters']}")
print(f"Train intra-similarity: {results['split_info']['train_avg_intra_similarity']:.3f}")
```

### How It Works

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

**Key Point**: Entire clusters move as units to train or test set.

### Direct Usage

```python
from molblender.models.api.core.splitting import butina_split

# Perform Butina clustering split
result = butina_split(
    dataset=dataset,
    test_size=0.2,
    similarity_threshold=0.6,
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
```

### Implementation Details

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

### Butina vs Scaffold Split

| Feature | Butina | Scaffold |
|---------|--------|----------|
| **Granularity** | Fine-grained (full molecule) | Coarse-grained (core only) |
| **Considers** | All atoms + substituents | Only ring systems + linkers |
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

### Benefits

```{admonition} Why Use Butina Split?
:class: tip

**1. Prevents Similarity Bias**: Avoids over-optimistic evaluation from similar molecules in train/test

**2. Self-Adaptive**: Automatic cluster count based on similarity threshold

**3. Cluster Integrity**: Entire clusters move as units (true leave-cluster-out)

**4. Fine-Grained Control**: More granular than scaffold split

**5. MolAgent Alignment**: Implements cluster-based validation from MolAgent literature
```

---

## 3. Feature Clustering Split

General-purpose clustering split supporting arbitrary molecular representations.

### When to Use

✅ **Use feature clustering when:**
- Using non-fingerprint representations (3D embeddings, language models)
- Custom feature spaces (quantum features, docking scores)
- Need flexible clustering algorithm choice (K-means vs DBSCAN)
- Testing model generalization across representation space regions

❌ **Don't use when:**
- Standard fingerprint-based validation is sufficient (use Butina instead)
- Dataset < 100 molecules (insufficient for clustering)
- No clear cluster structure in data

### Configuration

```python
from molblender.models import universal_screen

# Using RDKit descriptors with K-means
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="feature_clustering",

    # Clustering parameters
    clustering_algorithm="kmeans",   # 'kmeans', 'hierarchical', 'dbscan'
    n_clusters=5,                    # Manual k (None for auto-selection)
    auto_select_k=False,

    # Feature source
    use_descriptors=True,            # Use RDKit descriptors
    fingerprint_type="morgan",       # Fallback

    test_size=0.2,
    random_state=42
)
```

### How It Works

```
1. Feature Extraction
   ├─ User-provided features → Use directly
   ├─ RDKit descriptors → Compute MW, LogP, TPSA, etc.
   └─ Fingerprints (fallback) → Morgan/RDKit/MACCS

2. Feature Standardization
   └─ StandardScaler (zero mean, unit variance)

3. Clustering
   ├─ K-means: Spherical clusters, need k
   ├─ Hierarchical: Ward linkage, tree-based
   └─ DBSCAN: Density-based, auto cluster count

4. Optimal K Selection (if auto_select_k=True)
   └─ Test k ∈ [2, min(10, √n)]
   └─ Select k maximizing Silhouette score

5. Leave-Cluster-Out Assignment
   ├─ Greedy assignment: Largest clusters → test
   └─ DBSCAN: Noise points (-1 label) → train

6. Quality Metrics
   ├─ Silhouette score: [-1, 1], >0.5 = good
   ├─ Calinski-Harabasz: Higher = better-defined
   └─ Davies-Bouldin: Lower = better separation
```

### Direct Usage

#### Example 1: User-Provided Features (3D Embeddings)

```python
from molblender.models.api.core.splitting import feature_clustering_split
from molblender.representations import SomeEmbeddingGenerator

# Generate custom embeddings
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
```

#### Example 2: RDKit Descriptors with Hierarchical Clustering

```python
splits = feature_clustering_split(
    dataset=dataset,
    use_descriptors=True,           # Use ~20 RDKit descriptors
    clustering_algorithm='hierarchical',
    auto_select_k=True,
    test_size=0.25,
    random_state=42
)
```

#### Example 3: DBSCAN with Fingerprints

```python
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

### Implementation Details

**Code Location**: `src/molblender/models/api/core/splitting/feature_clustering.py`

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

### Feature Clustering vs Butina

| Feature | Feature Clustering | Butina |
|---------|-------------------|--------|
| **Feature Input** | Arbitrary (3D, embeddings, descriptors) | Fingerprints only |
| **Similarity Metric** | Euclidean distance (standardized) | Tanimoto similarity |
| **Clustering Algorithm** | K-means/Hierarchical/DBSCAN | Sphere exclusion (Butina) |
| **Cluster Count** | Manual or auto-optimized | Auto (threshold-based) |
| **Use Case** | Non-fingerprint representations | Fingerprint-based validation |
| **Flexibility** | High (algorithm choice, features) | Moderate (fingerprint + threshold) |

**When to Use Feature Clustering**:
- ✅ Using 3D molecular embeddings (e.g., Boltz-2)
- ✅ Language model representations (ChemBERTa, MolFormer)
- ✅ Quantum chemical descriptors
- ✅ Need control over clustering algorithm

**When to Use Butina**:
- ✅ Standard fingerprint-based validation
- ✅ Need Tanimoto similarity specifically
- ✅ Simpler, domain-specific clustering

### Typical Workflow: Boltz-2 Embeddings

```python
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

### Benefits

```{admonition} Why Use Feature Clustering Split?
:class: tip

**1. Representation Flexibility**: Works with any molecular representation

**2. Algorithm Control**: Choose clustering algorithm based on data distribution

**3. Auto K-Selection**: Silhouette score optimization removes manual tuning

**4. Quality Assurance**: Three metrics (Silhouette, CH, DB) validate clustering

**5. Standardization**: Auto feature scaling ensures fair distance computation

**6. Advanced Representations**: Ideal for 3D, language models, quantum features
```

---

## Quick Comparison

| Strategy | Granularity | Similarity Metric | Use Case | Flexibility |
|----------|-------------|-------------------|----------|-------------|
| **scaffold** | Coarse (core structure) | Scaffold match | Novel scaffolds | Low |
| **butina** | Fine (full molecule) | Tanimoto | Similar molecules | Moderate |
| **feature_clustering** | Custom | Euclidean (standardized) | Custom features | High |

---

## Related Strategies

- **For basic splits**: See {doc}`basic_strategies`
- **For diversity testing**: See {doc}`property_strategies` (maxmin split)
- **For advanced OOD**: See {doc}`advanced_strategies` (perimeter, MOOD)
- **For custom needs**: See {doc}`custom_splits`

---

**Navigation**: {doc}`index` | Previous: {doc}`basic_strategies` | Next: {doc}`property_strategies`
