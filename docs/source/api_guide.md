# MolBlender Unified API Guide

## Overview

MolBlender provides a unified API layer (`molblender.api`), integrating all core functionalities including representation generation, model screening, Dashboard, and more.

Since the architecture consolidation in 2026-03, the recommended usage is:

- `molblender.api`: Unified convenience entry point, suitable for new code and tutorial examples
- `molblender.models` / `molblender.representations`: Domain APIs, suitable for scenarios requiring more complete control
- `molblender.drawings`: Static plotting tools
- `molblender.dashboard`: Interactive analysis UI

The top-level `import molblender` and `import molblender.api` now both use lazy facades, which do not immediately pull up entire subpackages upon import. This makes interactive exploration and script startup lighter while maintaining backward compatibility with old entry points.

## API Layers

MolBlender provides multi-layered APIs, and users can choose the appropriate entry point based on their needs:

### Recommended Entry: molblender.api (Unified Facade)

**Use Case**: New code, quick start, common functionalities

`molblender.api` is a unified convenience entry point (Facade pattern) that provides concise interfaces for the most commonly used functionalities:

```python
from molblender.api import (
    # Representations
    get_featurizer,
    list_featurizers,
    # Model screening
    screen_models,
    load_results,
    # Visualization
    run_dashboard,
)
```

**Advantages**:
- Simple and easy to use
- Unified entry point
- Backward compatibility guaranteed
- Lighter imports (lazy facade)

### Domain APIs: Richer Functionality

For scenarios requiring more control or advanced features, you can directly use domain APIs:

#### molblender.models (ML Screening Domain API)

**Use Case**: Need complete ML screening functionality

```python
from molblender.models import (
    # Basic screening (same as molblender.api)
    screen_models,
    quick_screen,
    thorough_screen,
    # Analysis functions (richer API)
    analyze_results,
    compare_models,
    compare_representations,
    # Visualization
    plot_screening_results,
    create_performance_dashboard,
)
```

**Additional Features**:
- Multiple screening strategies (quick/thorough/interpretable)
- Result analysis and comparison
- Statistical tests
- Performance visualization

#### molblender.representations (Representation Domain API)

**Use Case**: Need detailed representation information

```python
from molblender.representations import (
    # Basic functions (same as molblender.api)
    get_featurizer,
    list_available_featurizers,
    # Detailed information (richer API)
    get_featurizer_info,
    print_available_featurizers,
    # Category selection
    get_category_info,
    select_featurizers_by_category,
    # Submodule access
    fingerprints,
    descriptors,
    graph,
)
```

**Additional Features**:
- Detailed representation metadata
- Category browsing and filtering
- Direct submodule access

#### molblender.drawings (Static Plotting Tools)

**Use Case**: Generate publication-quality static plots

```python
from molblender.drawings import (
    # Core configuration
    PlotConfig,
    set_plot_style,
    # Plotting functions
    plot_histogram,
    plot_scatter_fit,
    plot_heatmap,
    # Themes
    set_scientific_publication_style,
    set_presentation_style,
)
```

**Position**: Static plotting tools (matplotlib/seaborn), not equivalent to interactive dashboard

#### molblender.dashboard (Interactive Exploration)

**Use Case**: Interactive data exploration and result analysis

```python
from molblender.dashboard import run_dashboard

# Launch interactive Dashboard
run_dashboard("screening_results.db")
```

**Position**: Streamlit-based interactive web UI launch layer

**Notes**:
- `molblender.dashboard.run_dashboard(...)` is the low-level launch entry point of the UI package itself
- For regular workflow code, `molblender.api.run_dashboard(...)` is recommended

### Top-level molblender (Most Common Features)

The top-level `import molblender` exposes the most commonly used functions:

```python
import molblender

# Recommended (unified facade)
molblender.screen_models(...)
molblender.get_featurizer(...)
molblender.run_dashboard(...)

# Compatibility entries (direct import from submodules)
molblender.list_available_featurizers(...)
molblender.analyze_results(...)  # Need richer API
```

**Notes**:
- The top-level `molblender` is suitable for notebooks and quick scripts
- For new code that needs to clearly express intent, `molblender.api` or corresponding domain subpackages are still recommended
- The top-level facade does not expose runtime-policy internals or legacy execution helpers

### Selection Guide

| Need | Recommended Entry | Alternative |
|------|----------|----------|
| **New code/Quick start** | `molblender.api` | Top-level `molblender` |
| **Complete ML screening** | `molblender.models` | `molblender.api.screen_models` |
| **Detailed representation info** | `molblender.representations` | `molblender.api.get_featurizer_info` |
| **Static plots** | `molblender.drawings` | - |
| **Interactive exploration** | `molblender.dashboard` | - |

## Dashboard Tools Comparison

MolBlender provides two independent Dashboard tools for different purposes:

### Top-Level Dashboard (`molblender view`)

**Purpose**: Screening results analysis and visualization

**When to use**: After model training

**Launch**:
```bash
# CLI
molblender view /path/to/results

# Or specify database file
molblender view screening_results.db

# Python API
from molblender.api import run_dashboard
run_dashboard("screening_results.db")
```

**Key Features**:
- Model performance comparison visualization
- Hyperparameter analysis
- Experiment comparison and ranking
- Results export
- Interactive data exploration

**Typical Use Cases**:
- View best models after screening
- Compare performance across representations
- Analyze hyperparameter effects
- Select optimal model for production

### Diagnostics Dashboard (`molblender-diagnostics`)

**Purpose**: Dataset quality diagnostics

**When to use**: Before model training

**Launch**:
```bash
# CLI (upload dataset in UI)
molblender-diagnostics

# Or specify dataset file
molblender-diagnostics dataset.csv

# Custom column names
molblender-diagnostics data.csv --input-column SMILES --label-column pIC50

# Python module
python -m molblender.data.diagnostics.dashboard data.csv
```

**Key Features**:
- DNR (Different Neighbor Ratio) analysis
- Activity cliffs detection
- Molecular diversity analysis
- Data quality report generation
- Dataset statistics and visualization

**Python entrypoint**:
```python
from molblender.data import diagnostics

diagnostics.dashboard.run_diagnostics_dashboard("dataset.csv")
```

**Typical Use Cases**:
- Assess dataset quality before modeling
- Detect problematic regions in dataset
- Identify chemical space gaps
- Understand potential model failure reasons

### Dashboard Selection Guide

| Need | Tool | Input Format |
|------|------|--------------|
| Analyze screening results | Top-Level Dashboard | SQLite database (.db) |
| Diagnose data quality | Diagnostics Dashboard | CSV dataset |
| Visualize model performance | Top-Level Dashboard | SQLite database (.db) |
| Detect activity cliffs | Diagnostics Dashboard | CSV dataset |
| Compare experiments | Top-Level Dashboard | SQLite database (.db) |
| Evaluate molecular diversity | Diagnostics Dashboard | CSV dataset |

### Workflow Example

Typical workflow uses both Dashboards sequentially:

```bash
# 1. Before modeling: diagnose data quality
molblender-diagnostics my_dataset.csv
# → Identify high-DNR regions, decide to add more data

# 2. Run model screening
from molblender.api import screen_models
results = screen_models(...)

# 3. After modeling: analyze screening results
molblender view screening_results.db
# → View best models, analyze performance
```

**Note**: The two Dashboards are completely independent tools in different modules:
- Top-Level Dashboard: `molblender/dashboard/`
- Diagnostics Dashboard: `molblender/data/diagnostics/dashboard/`

Diagnostics Dashboard belongs to the `molblender.data` domain and is intentionally
separate from the results dashboard. Importing one should not be required to use
the other.

### Execution Layer Guide

Most users do not need to import execution layers directly. To avoid confusion, the current positioning of execution-related layers is as follows:

| Layer | Current Position | Target Audience |
|------|------------------|-----------------|
| `molblender.models.api.infrastructure` | Primary screening runtime layer | Package internal, advanced developers |
| `molblender.representations.utils` | Generic batching/caching helpers | Developers needing direct control over representation batching |
| `molblender.models.execution` | Compatibility layer (legacy) | Legacy code migration |

Most users should prioritize using workflow entry points like `screen_models()`, `universal_screen()`, `molblender.api.run_dashboard()`, rather than directly assembling execution/runtime components.

### Data Subdomain Guide

`molblender.data` currently contains three main tiers and one auxiliary subdomain:

| Subdomain | Position | Recommendation Level |
|-----------|----------|---------------------|
| `molblender.data.dataset` | Dataset structures and public splitting helpers | Recommended |
| `molblender.data.io` | Shared input types and compatible parsing helpers | Supported |
| `molblender.data.molecule` | Single-molecule objects and file I/O | Supported |
| `molblender.data.protein` | Protein objects and sequence/PDB I/O | Supported |
| `molblender.data.diagnostics` | Data quality analysis | Specialized |
| `molblender.data.cache` | Cache implementation | Supported |
| `molblender.data.cache.multimodal` | Multimodal-specific caching and special storage backends | Supported |
| `molblender.data.preprocessing` | Feature preprocessing, balancing, time splitting, and other helper tools | Supported |

All these subdomains now use lazy facades—importing the top-level `molblender.data` will not automatically pull them all into memory.
Among them, `molblender.data.cache` lazy exposes the `multimodal` submodule, and `molblender.data.diagnostics`
also lazy exposes the specialized `dashboard` submodule.

### Programmatic Architecture Snapshot

If you need to check current recommended entry points, execution layer roles, or migration recommendations, you can read directly from the code:

```python
from molblender.architecture_roles import (
    get_package_role_catalog,
    get_recommended_entrypoints,
    get_execution_layer_decisions,
)

print(get_recommended_entrypoints())
print(get_execution_layer_decisions())
```

You can also output a JSON snapshot in the terminal:

```bash
python -m molblender.architecture_roles
```

## Quick Start

### Installation

```bash
pip install molblender
```

### Basic Usage

```python
from molblender.api import (
    get_featurizer,
    list_featurizers,
    screen_models,
    create_screener,
    load_results,
    run_dashboard,
    load_dashboard_data,
)
```

## API Modules

### 1. Representation API

#### list_featurizers()

List all available molecular representation generators.

```python
from molblender.api import list_featurizers

# List all representations
all_featurizers = list_featurizers()
print(f"Available representations: {len(all_featurizers)}")

# List representations in a specific category
fingerprint_featurizers = list_featurizers(category="fingerprints")
protein_featurizers = list_featurizers(category="protein")
```

**Returns**: `list[str]` - List of representation names

#### get_featurizer()

Get a molecular representation generator instance.

```python
from molblender.api import get_featurizer

# Get fingerprint representation
morgan_featurizer = get_featurizer("morgan_fp")
rdkit_featurizer = get_featurizer("rdkit_fp")

# Get protein representation
prot_featurizer = get_featurizer("prot_bert")

# Generate representations
smiles = ["CCO", "c1ccccc1", "CC(C)CC"]
features = morgan_featurizer.featurize(smiles)
print(f"Feature shape: {features.shape}")
```

**Parameters**:
- `name` (`str`): representation name

**Returns**: `BaseFeaturizer` - Featurizer instance

#### get_featurizer_info()

Get detailed information about the representation.

```python
from molblender.api import get_featurizer_info

# Get representation information
info = get_featurizer_info("morgan_fp")
print(f"Name: {info['name']}")
print(f"Category: {info['category']}")
print(f"Output shape: {info['output_shape']}")
print(f"Description: {info['description']}")
```

**Returns**: `dict` - Dictionary containing detailed representation information

### 2. Model API

#### screen_models()

Execute ML model screening (recommended usage).

```python
from molblender.api import screen_models

# Simple screening
results = screen_models(
    smiles_data=["CCO", "c1ccccc1", "CC(C)CC", ...],
    representations=["morgan_fp", "rdkit_fp"],
    task_type="regression",
    target_values=[0.5, 1.2, 0.8, ...],
)

# Advanced screening
results = screen_models(
    smiles_data=smiles_list,
    representations=["morgan_fp", "chemberta"],
    models=["random_forest", "xgboost"],
    task_type="regression",
    target_values=activity_values,
    split_strategy="random",
    test_size=0.2,
    max_cpu_cores=-1,
    max_workers_per_model=1,
    enable_hpo=False,
    combinations="auto",
)

# Save results to database
results = screen_models(
    ...,
    save_path="screening_results.db",
    session_name="my_screening_session",
)
```

**Parameters**:
- `smiles_data` (`list[str]`): SMILES molecule list
- `representations` (`list[str]`): molecular representation list
- `models` (`list[str]` | `None`): model list（Nonemeans auto-select）
- `task_type` (`str`): task type（"classification" or "regression"）
- `target_values` (`list[float]`): target value
- `save_path` (`str | None`): result save path
- `session_name` (`str | None`): session name
- Other parameters see `ScreeningConfig`

Parallel parameter conventions:
- `max_cpu_cores`: Total CPU budget available for the entire screening workflow
- `max_workers_per_model`: Maximum number of workers that can be used within a single model
- `n_jobs`: Legacy interface compatibility alias, not recommended for new code

**Returns**: `dict` - Screening results dictionary

#### create_screener()

Create screener instance (advanced usage).

```python
from molblender.api import create_screener
from molblender.models import ScreeningConfig

# Create configuration
config = ScreeningConfig(
    task_type="regression",
    representations=["morgan_fp"],
    models=["random_forest"],
    max_cpu_cores=-1,
    max_workers_per_model=1,
)

# Create screener
screener = create_screener(config)

# Prepare data
screener.prepare_data(
    smiles_data=["CCO", "c1ccccc1"],
    target_values=[0.5, 1.2],
)

# Run screening
results = screener.run_screening()
```

**Parameters**:
- `config` (`ScreeningConfig`): screening configuration

**Returns**: `BaseScreener` - Screener instance

#### load_results()

Load screening results from database.

```python
from molblender.api import load_results

# Load results
results = load_results(
    db_path="screening_results.db",
    session_name="my_screening_session",
)

# Get best results
best_result = results.get_best_result(metric="r2_score")
print(f"Best model: {best_result['model_name']}")
print(f"Best representation: {best_result['representation_name']}")
print(f"Best score: {best_result['primary_metric']}")
```

**Parameters**:
- `db_path` (`str`): database path
- `session_name` (`str | None`): session name（Nonemeans load all sessions）

**Returns**: `ResultsDatabase` - Results database object

### 3. Dashboard API

#### run_dashboard()

Launch interactive Dashboard.

```python
from molblender.api import run_dashboard

# Launch Dashboard (default port 8501)
run_dashboard("screening_results.db")

# Specify database and port
run_dashboard(
    "screening_results.db",
    port=8502,
    no_browser=True,
)
```

**Parameters**:
- `results_path` (`str | Path`): results database or results directory path
- `port` (`int`): Web service port
- `no_browser` (`bool`): do not auto-open browser

#### load_dashboard_data()

Load Dashboard data (for custom analysis).

```python
from molblender.api import load_dashboard_data

# Load data
data = load_dashboard_data("screening_results.db")

# Access data
sessions = data['sessions']
results = data['results']
models = data['models']
representations = data['representations']
```

**Returns**: `dict` - Dictionary containing all Dashboard data

## Complete Examples

### Example 1: Simple Regression Task

```python
from molblender.api import screen_models, load_results, run_dashboard

# 1. Prepare data
smiles = [
    "CCO",  # Ethanol
    "c1ccccc1",  # benzene
    "CC(C)CC",  # Isobutane
    # ... more molecules
]
activities = [0.5, 1.2, 0.8, ...]  # bioactivity value

# 2. run screening
results = screen_models(
    smiles_data=smiles,
    representations=["morgan_fp", "rdkit_fp"],
    task_type="regression",
    target_values=activities,
    save_path="results.db",
    session_name="regression_screening",
)

# 3. view results
print(f"completed {len(results)} models-representation combination screening")

# 4. launch Dashboard visualize
run_dashboard("results.db")
```

### example2: classification task

```python
from molblender.api import screen_models

# Classification data
smiles = [...]
labels = [0, 1, 0, 1, ...]  # 0: inactive, 1: active

# Run classification screening
results = screen_models(
    smiles_data=smiles,
    representations=["morgan_fp", "chemberta"],
    models=["random_forest", "xgboost", "logistic_regression"],
    task_type="classification",
    target_values=labels,
    save_path="classification_results.db",
    session_name="classification_screening",
    metric_type="roc_auc",  # Use AUC as evaluation metric
)
```

### Example 3: Advanced screening configuration

```python
from molblender.api import create_screener
from molblender.models import ScreeningConfig

# Create advanced configuration
config = ScreeningConfig(
    task_type="regression",
    representations=["morgan_fp", "chemberta", "unimol_cls"],
    models=None,  # Auto-select compatible models
    cv_folds=5,
    test_size=0.2,
    split_strategy="scaffold_split",  # Use scaffold split
    max_cpu_cores=-1,
    max_workers_per_model=1,
    enable_hpo=False,
    combinations="auto",  # Use primary path + backup path
    auto_resource_optimization=True,  # automatic resource optimization
)

# Create and run screener
screener = create_screener(config)
screener.prepare_data(
    smiles_data=smiles,
    target_values=activities,
)
results = screener.run_screening(
    save_path="advanced_results.db",
    session_name="advanced_screening",
)
```

### Example 4: Results analysis

```python
from molblender.api import load_results

# Load results
db = load_results("results.db", session_name="my_screening")

# Get all results
all_results = db.get_all_results()
print(f"Total results: {len(all_results)}")

# Get best results
best = db.get_best_result()
print(f"Best combination: {best['model_name']} + {best['representation_name']}")
print(f"Best score: {best['primary_metric']:.3f}")

# Filter specific results
rf_results = db.get_results(
    model_name="random_forest",
    representation_name="morgan_fp",
)

# Get statistics
stats = db.get_statistics()
print(f"Mean R²: {stats['mean_primary_metric']:.3f}")
print(f"Std dev: {stats['std_primary_metric']:.3f}")
```

## API Design Principles

### 1. Simplicity

- Minimal parameters, reasonable defaults
- One function completes common tasks
- Auto-detection and validation

### 2. Consistency

- Unified naming conventions
- Consistent parameter order
- Unified return types

### 3. Backward Compatibility

- Old import paths still work
- Gradual migration, no immediate rewrite needed
- Smooth upgrade path

### 4. Extensibility

- Support advanced configuration
- Support custom extensions
- Support plugin mechanism

## error handling

### Common Errors

```python
# 1. Incompatible model-representation combinations
try:
    results = screen_models(
        smiles_data=smiles,
        representations=["canonical_smiles"],  # STRING type
        models=["random_forest"],  # Cannot handle STRING
        task_type="regression",
        target_values=activities,
    )
except ValueError as e:
    print(f"Compatibility error: {e}")

# 2. Invalid representation name
try:
    featurizer = get_featurizer("invalid_featurizer")
except ValueError as e:
    print(f"Representation error: {e}")

# 3. Data format error
try:
    results = screen_models(
        smiles_data=["INVALID_SMILES"],
        representations=["morgan_fp"],
        task_type="regression",
        target_values=[0.5],
    )
except Exception as e:
    print(f"Data error: {e}")
```

### error handlingBest Practices

```python
from molblender.api import screen_models
import logging

logger = logging.getLogger(__name__)

def safe_screening(smiles, targets, reprs, models):
    """Screening with error handling"""
    try:
        results = screen_models(
            smiles_data=smiles,
            representations=reprs,
            models=models,
            task_type="regression",
            target_values=targets,
        )
        return results
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return None
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        return None
```

## Performance Optimization

### 1. Parallel Processing

```python
# Auto parallel (recommended)
results = screen_models(
    smiles_data=large_smiles_list,
    representations=["morgan_fp"],
    models=["random_forest"],
    max_cpu_cores=-1,          # Total CPU budget
    max_workers_per_model=1,   # Max parallel workers per model
)
```

### 2. Cache Optimization

```python
import os
from molblender.config import config_manager

# Set cache directory
config_manager.set_cache_dir("representations", "./cache")

# Enable cache
os.environ["MOLBLENDER_CACHE_ENABLED"] = "true"

# Screening automatically uses cache
results = screen_models(
    smiles_data=smiles,
    representations=["morgan_fp"],
    task_type="regression",
    target_values=activities,
)
```

### 3. Resource Optimization

```python
from molblender.models import ScreeningConfig

# automatic resource optimization
config = ScreeningConfig(
    task_type="regression",
    representations=["morgan_fp"],
    auto_resource_optimization=True,  # Enable auto optimization
)

# Create screener
screener = create_screener(config)
```

## Related Documentation

- [Migration Guide](migration_guide.md) - Migrate from old API
- [ConfigManager Guide](config_manager_guide.md) - Configuration management
- [Quick Start](quickstart.md) - Getting started
- [Architecture Overview](development/architecture.md) - Current public layers with internal boundaries

---

**Last updated**: 2026-03-10
**Version**: 1.0.0

## Advanced: Tool Registry API

MolBlender provides a unified representation registry system for advanced metadata query and filtering.

### ToolRegistry

`ToolRegistry` provides unified representation discovery and query interfaces.

```python
from molblender.representations.tool_registry import ToolRegistry, ToolInfo

# Get registry instance
registry = ToolRegistry()

# Filter by category
molecular_featurizers = registry.list(category="molecular")
protein_featurizers = registry.list(category="protein")

# Filter by tag
gpu_featurizers = registry.list(tags=["gpu"])
experimental_featurizers = registry.list(tags=["experimental"])

# Get detailed metadata
info: ToolInfo = registry.get("ecfp")
print(f"Description: {info.description}")
print(f"Dependencies: {info.dependencies}")
print(f"Output shape: {info.output_shape}")

# Search representations
results = registry.search("fingerprint")
for info in results:
    print(f"{info.name}: {info.description}")
```

### ToolInfo

`ToolInfo` contains complete metadata for representations:

```python
@dataclass
class ToolInfo:
    name: str                      # Representation name
    category: str                  # Category (molecular, protein, etc.)
    description: str               # Description
    source: str                    # Source (rdkit, deepchem, etc.)
    tags: list[str]                # Tags (gpu, experimental, etc.)
    input_type: str                # Input type (smiles, sdf, etc.)
    output_type: str               # Output type (vector, matrix, etc.)
    output_shape: tuple            # Output shape
    default_kwargs: dict           # Default parameters
    is_available: bool             # Is available (dependency check)
    dependencies: list[str]        # Dependencies
```

### Usage Example

```python
from molblender.representations.tool_registry import get_tool_registry

# Get global registry instance
registry = get_tool_registry()

# List all available GPU representations
gpu_tools = registry.list(tags=["gpu"])
for tool in gpu_tools:
    if tool.is_available:
        print(f"✅ {tool.name}: {tool.description}")
    else:
        print(f"❌ {tool.name}: Missing dependencies {tool.dependencies}")

# Find specific representation
info = registry.get("morgan_fp")
if info and info.is_available:
    print(f"Morgan fingerprint is available!")
    print(f"Default radius: {info.default_kwargs.get('radius', 'N/A')}")
    print(f"Default n_bits: {info.default_kwargs.get('n_bits', 'N/A')}")
```
