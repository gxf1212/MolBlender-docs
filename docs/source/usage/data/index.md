# Data Management

Efficient molecular data handling and dataset management with integrated featurization, caching, and multi-format I/O support for machine learning workflows.

## Introduction

The data module provides the foundational classes for managing molecular datasets throughout the MolBlender workflow. It handles the complexity of different input formats, molecular representations, and feature computation while providing a clean, intuitive API.

Key capabilities include:
- **Unified Data Loading**: Support for SMILES, SDF, CSV, Excel, and more
- **Lazy Computation**: Features computed on-demand with intelligent caching
- **Error Resilience**: Graceful handling of invalid molecules and failed computations
- **Memory Efficiency**: Optimized for large datasets with streaming and chunking
- **ML Integration**: Direct compatibility with scikit-learn and other ML libraries

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🧪 **Molecule Objects**
:link: molecule
:link-type: doc
Individual molecular representations with lazy loading
:::

:::{grid-item-card} 📊 **Dataset**
:link: dataset
:link-type: doc
Efficient collection handling with batch operations
:::

:::{grid-item-card} 🔀 **Data Splitting**
:link: splitting/index
:link-type: doc
14 professional splitting strategies and cross-validation methods
:::

:::{grid-item-card} 🧬 **Protein Handling**
:link: protein
:link-type: doc
Comprehensive protein data management and database retrieval
:::

:::{grid-item-card} 💾 **I/O Operations**
Multi-format loading and saving capabilities
:::

:::{grid-item-card} ⚡ **Performance**
Memory-efficient processing for large datasets
:::

:::{grid-item-card} 🔄 **Caching**
Intelligent feature caching and storage
:::
::::

## Quick Start

### Creating Datasets

```python
import molblender as mbl
from molblender.data import MolecularDataset

# From SMILES list
molecules = ["CCO", "CCN", "CCC", "c1ccccc1"]
dataset = MolecularDataset.from_smiles(molecules)

# From CSV file
dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    label_columns=["activity", "toxicity"]
)

# From SDF file
dataset = MolecularDataset.from_sdf("compounds.sdf", 
                                   label_columns=["IC50"])

print(f"Dataset size: {len(dataset)} molecules")
print(f"Properties: {list(dataset.properties.columns)}")
```

### Adding Features

```python
# Single featurizer
dataset.add_features("morgan_fp_r2_1024", n_workers=4)

# Multiple featurizers
dataset.add_features([
    "morgan_fp_r2_1024", 
    "rdkit_descriptors",
    "maccs_keys"
], n_workers=4)

# Access features
print(f"Available features: {list(dataset.features.columns)}")
print(f"Morgan fingerprint shape: {dataset.features['morgan_fp_r2_1024'][0].shape}")
```

### Basic Operations

```python
# Dataset info
print(dataset.info())

# Iterate over molecules
for i, molecule in enumerate(dataset):
    if i >= 3:  # Show first 3
        break
    print(f"Molecule {i}: {molecule.smiles}")

# Filtering
active_dataset = dataset.filter(lambda mol: mol.properties.get('activity', 0) > 5.0)
print(f"Active molecules: {len(active_dataset)}")

# Splitting
train_set, test_set = dataset.split(test_size=0.2, random_state=42)
```

## Working with Different File Formats

### CSV Files

```python
# Complex CSV with multiple columns
dataset = MolecularDataset.from_csv(
    "drug_data.csv",
    input_column="canonical_smiles",
    label_columns=["logP", "solubility", "toxicity"],
    id_column="compound_id",
    dropna=True,                    # Remove invalid molecules
    validate_molecules=True         # Check molecule validity
)

# Handle missing values
dataset = MolecularDataset.from_csv(
    "messy_data.csv",
    input_column="SMILES",
    label_columns=["activity"],
    na_values=["NA", "NULL", ""],   # Additional NA markers
    drop_invalid=True               # Drop invalid SMILES
)
```

### SDF Files

```python
# SDF with properties
dataset = MolecularDataset.from_sdf(
    "pubchem_compounds.sdf",
    label_columns=["MW", "LogP"],       # Extract these properties
    max_molecules=10000,                # Limit for large files
    skip_invalid=True                   # Skip problematic structures
)

# Multiple SDF files
import glob
sdf_files = glob.glob("data/*.sdf")
datasets = [MolecularDataset.from_sdf(f) for f in sdf_files]
combined = MolecularDataset.concat(datasets)
```

### Excel Files

```python
# Excel with multiple sheets
dataset = MolecularDataset.from_excel(
    "compounds.xlsx",
    sheet_name="training_data",
    input_column="SMILES",
    label_columns=["IC50", "selectivity"]
)

# Multiple sheets
datasets = {}
for sheet in ["train", "test", "validation"]:
    datasets[sheet] = MolecularDataset.from_excel(
        "data.xlsx", 
        sheet_name=sheet,
        input_column="SMILES", 
        label_columns=["target"]
    )
```

## Advanced Dataset Operations

### Filtering and Selection

```python
# Property-based filtering
high_activity = dataset.filter(
    lambda mol: mol.properties.get('activity', 0) > 7.0
)

# SMILES pattern filtering
aromatic_compounds = dataset.filter(
    lambda mol: 'c1ccccc1' in mol.smiles  # Contains benzene
)

# Complex filtering with multiple conditions
druglike = dataset.filter(
    lambda mol: (
        mol.properties.get('MW', 0) < 500 and
        mol.properties.get('LogP', 0) < 5 and
        mol.properties.get('HBD', 0) <= 5
    )
)

print(f"Drug-like molecules: {len(druglike)}/{len(dataset)}")
```

### Dataset Splitting

```python
# Simple random split
train, test = dataset.split(test_size=0.2, random_state=42)

# Stratified split (by activity bins)
train, test = dataset.split(
    test_size=0.2, 
    stratify='activity',  # Column to stratify by
    bins=5,              # Number of bins
    random_state=42
)

# Time-based split (if molecules have dates)
train, test = dataset.split_by_time(
    time_column='date_added',
    cutoff_date='2023-01-01'
)

# Scaffold split (by molecular scaffold)
train, test = dataset.scaffold_split(
    test_size=0.2,
    scaffold_func='bemis_murcko'  # or custom function
)
```

### Data Validation and Cleaning

```python
# Comprehensive data validation
validation_report = dataset.validate()
print(f"Invalid molecules: {validation_report['invalid_count']}")
print(f"Duplicate molecules: {validation_report['duplicate_count']}")
print(f"Missing properties: {validation_report['missing_properties']}")

# Clean the dataset
cleaned = dataset.clean(
    remove_invalid=True,      # Remove invalid SMILES
    remove_duplicates=True,   # Remove duplicate molecules
    standardize=True,         # Standardize SMILES representation
    remove_salts=True,        # Remove salt components
    neutralize=True           # Neutralize charges
)

print(f"Cleaned: {len(dataset)} → {len(cleaned)} molecules")
```

### Feature Management

```python
# Compute features with error handling
success_count = dataset.add_features(
    "mordred_descriptors_2d", 
    n_workers=8,
    handle_errors=True,       # Continue on individual failures
    error_value=None          # Value for failed computations
)

print(f"Successfully computed: {success_count}/{len(dataset)}")

# Feature statistics
feature_stats = dataset.feature_statistics()
print("Feature summary:")
for feature_name, stats in feature_stats.items():
    print(f"  {feature_name}: {stats['shape']}, "
          f"{stats['missing_count']} missing")

# Remove features with too many missing values
dataset.remove_features_with_missing(threshold=0.1)  # >10% missing

# Feature correlation analysis
corr_matrix = dataset.compute_feature_correlations()
highly_correlated = dataset.find_correlated_features(threshold=0.95)
dataset.remove_features(highly_correlated)
```

## Memory-Efficient Processing

### Large Dataset Handling

```python
# Stream processing for very large files
def process_large_sdf(filename, chunk_size=10000):
    """Process large SDF files in chunks"""
    results = []
    
    for chunk in MolecularDataset.iter_sdf_chunks(filename, chunk_size):
        # Process each chunk
        chunk.add_features("morgan_fp_r2_1024", n_workers=4)
        
        # Extract what you need
        features = chunk.get_feature_matrix("morgan_fp_r2_1024")
        properties = chunk.properties.values
        
        results.append((features, properties))
        
        # Clear memory
        del chunk
    
    return results

# Memory monitoring
import psutil

def monitor_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

print(f"Memory usage: {monitor_memory_usage():.1f} MB")
```

### Parallel Processing

```python
# Parallel feature computation
from concurrent.futures import ProcessPoolExecutor

def parallel_featurization(dataset, featurizers, n_workers=4):
    """Compute multiple featurizers in parallel"""
    
    def compute_single_featurizer(args):
        dataset_subset, featurizer_name = args
        dataset_subset.add_features(featurizer_name)
        return dataset_subset.features[featurizer_name].tolist()
    
    # Split dataset into chunks
    chunk_size = len(dataset) // n_workers
    chunks = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]
    
    # Prepare tasks
    tasks = [(chunk, feat) for chunk in chunks for feat in featurizers]
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(compute_single_featurizer, tasks)
    
    return list(results)
```

## Caching Strategies

### Feature Caching

```python
# Enable persistent caching
dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    cache_dir="./feature_cache",    # Cache directory
    cache_enabled=True              # Enable caching
)

# Features are cached automatically
dataset.add_features("morgan_fp_r2_1024")  # Computed and cached
dataset.add_features("morgan_fp_r2_1024")  # Loaded from cache (fast!)

# Cache management
cache_info = dataset.cache_info()
print(f"Cache size: {cache_info['total_size_mb']:.1f} MB")
print(f"Cache hit rate: {cache_info['hit_rate']:.2%}")

# Clear cache
dataset.clear_cache()              # Clear all cached features
dataset.clear_cache("morgan_fp")   # Clear specific feature cache
```

### Custom Caching

```python
# Custom cache configuration
cache_config = {
    'backend': 'disk',              # disk, memory, or redis
    'compression': 'lz4',           # Compression algorithm
    'max_size_gb': 10,              # Maximum cache size
    'ttl_hours': 24                 # Cache expiration
}

dataset = MolecularDataset.from_smiles(
    molecules,
    cache_config=cache_config
)
```

## Integration with ML Libraries

### Scikit-learn Integration

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Prepare data for ML
X = dataset.get_feature_matrix("morgan_fp_r2_1024")
y = dataset.properties["activity"].values

# Remove any NaN values
mask = ~np.isnan(y)
X, y = X[mask], y[mask]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"Cross-validation R²: {scores.mean():.3f} ± {scores.std():.3f}")
```

### PyTorch Integration

```python
import torch
from torch.utils.data import DataLoader

# Convert to PyTorch dataset
class MolecularDatasetPyTorch(torch.utils.data.Dataset):
    def __init__(self, polyglot_dataset, feature_name, target_name):
        self.features = polyglot_dataset.get_feature_matrix(feature_name)
        self.targets = polyglot_dataset.properties[target_name].values
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])

# Create PyTorch dataset and loader
torch_dataset = MolecularDatasetPyTorch(dataset, "morgan_fp_r2_1024", "activity")
dataloader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

# Use in training loop
for features, targets in dataloader:
    # Training code here
    pass
```

## Performance Tips

```{admonition} Optimization Guidelines
:class: tip

**Loading Data:**
- Use `validate_molecules=False` for trusted data sources
- Set `max_molecules` to limit memory usage for exploration
- Use streaming for files >1GB

**Feature Computation:**
- Start with fast featurizers for initial exploration
- Use `n_workers=-1` only on dedicated compute nodes
- Enable caching for repeated feature access

**Memory Management:**
- Process large datasets in chunks
- Clear unnecessary features with `remove_features()`
- Use generators instead of loading full datasets

**Performance Monitoring:**
- Monitor memory with `psutil` or `memory_profiler`
- Profile feature computation times
- Use `dataset.info()` to check memory usage
```

## Error Handling

### Robust Data Loading

```python
# Handle various data quality issues
try:
    dataset = MolecularDataset.from_csv(
        "real_world_data.csv",
        input_column="SMILES",
        label_columns=["activity"],
        
        # Error handling options
        skip_invalid=True,          # Skip invalid SMILES
        validate_molecules=True,    # Validate each molecule
        max_errors=100,            # Stop if too many errors
        error_log="errors.log"     # Log errors to file
    )
    
except Exception as e:
    print(f"Data loading failed: {e}")
    # Fallback strategy
    dataset = MolecularDataset.from_csv(
        "real_world_data.csv",
        input_column="SMILES",
        skip_invalid=True,
        validate_molecules=False    # Less strict validation
    )

# Check data quality
quality_report = dataset.data_quality_report()
print(f"Data quality score: {quality_report['score']:.2f}")
```

### Feature Computation Errors

```python
# Robust featurization with detailed error reporting
def robust_featurization(dataset, featurizer_names):
    """Compute features with comprehensive error handling"""
    
    results = {}
    for featurizer in featurizer_names:
        try:
            success_count = dataset.add_features(
                featurizer,
                handle_errors='warn',  # Options: 'raise', 'warn', 'ignore'
                n_workers=4
            )
            
            success_rate = success_count / len(dataset)
            results[featurizer] = {
                'success_count': success_count,
                'success_rate': success_rate,
                'status': 'completed'
            }
            
            if success_rate < 0.8:  # Less than 80% success
                print(f"Warning: {featurizer} had low success rate: {success_rate:.1%}")
                
        except Exception as e:
            results[featurizer] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"Failed to compute {featurizer}: {e}")
    
    return results

# Usage
feature_results = robust_featurization(
    dataset, 
    ["morgan_fp_r2_1024", "rdkit_descriptors", "mordred_descriptors_2d"]
)
```

## References

- [RDKit Molecule Handling](https://www.rdkit.org/docs/GettingStartedInPython.html) - RDKit molecule objects and operations
- [Pandas DataFrame Operations](https://pandas.pydata.org/docs/user_guide/index.html) - DataFrame manipulation techniques used internally
- [SDF Format Specification](http://help.accelryds.com/ulm/onelab/1.0/content/ulm_pdfs/direct/reference/ctfileformats2016.pdf) - Chemical file format details

```{toctree}
:maxdepth: 2
:hidden:

molecule
protein
dataset
splitting/index
splitting/basic_strategies
splitting/chemical_strategies
splitting/property_strategies
splitting/advanced_strategies
splitting/custom_splits
splitting/best_practices
```