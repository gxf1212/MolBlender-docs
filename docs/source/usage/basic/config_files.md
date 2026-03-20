# Configuration Files

MolBlender supports loading screening configurations from YAML and TOML files, making it easy to manage, version control, and reuse complex screening setups.

## Quick Start

### 1. Create a Configuration Template

```python
from molblender.config import create_config_template

# Create a basic template
create_config_template("basic", "my_config.yaml")
```

This creates a YAML file with default settings that you can customize.

### 2. Edit Your Configuration

```yaml
# my_config.yaml
split_config:
  strategy: "train_test"
  test_size: 0.2
  cv_folds: 5

resource_config:
  max_cpu_cores: 8
  execution_preference: "speed"

hpo_config:
  enabled: true
  stage: "coarse"
  method: "grid"

combinations: "auto"
verbose: 2
```

### 3. Run Screening with Config File

```python
from molblender.api import universal_screen

results = universal_screen(
    dataset=your_dataset,
    target_column="activity",
    config_file="my_config.yaml"
)
```

## Available Templates

MolBlender provides three pre-configured templates:

- **basic** - Minimal configuration for quick start
- **advanced** - Full-featured template with all available options
- **hpo** - Optimized for hyperparameter optimization workflows

```python
# Create different templates
create_config_template("basic", "basic_config.yaml")
create_config_template("advanced", "advanced_config.yaml")
create_config_template("hpo", "hpo_config.yaml")
```

## Configuration Sections

### split_config

Data splitting configuration:

```yaml
split_config:
  strategy: "train_test"  # Options: train_test, train_val_test, cv_only, nested_cv
  test_size: 0.2
  cv_folds: 5
  # Molecular splitting parameters
  fingerprint_type: "morgan"
  butina_similarity_threshold: 0.6
```

### resource_config

Resource management:

```yaml
resource_config:
  max_cpu_cores: 8  # -1 = use all available
  execution_preference: "speed"  # Options: speed, memory, balanced
  auto_optimization: true
```

### hpo_config

Hyperparameter optimization:

```yaml
hpo_config:
  enabled: true
  stage: "coarse"  # Options: coarse (grid search), fine (Optuna)
  method: "grid"  # Options: grid, random
  top_n_for_hpo: 10
```

### database_config

Database storage:

```yaml
database_config:
  db_path: "screening_results.db"
  session_name: null  # Auto-generated if null
  enable_caching: true
```

### core_config

Core screening settings:

```yaml
core_config:
  task_type: "regression"  # Options: regression, classification
  primary_metric: "pearson_r"
  random_state: 42
  n_jobs: -1
```

## Environment Variables

Configuration files support environment variable substitution:

```yaml
resource_config:
  max_cpu_cores: ${MAX_CORES:-8}  # Use MAX_CORES env var, default to 8

database_config:
  db_path: "${PROJECT_DIR}/results_${USER}.db"
  session_name: "experiment_${TIMESTAMP}"
```

Set environment variables before running:

```bash
export MAX_CORES=16
export PROJECT_DIR=/data/experiments
python my_screening.py
```

## Supported Formats

### YAML Format

```yaml
split_config:
  strategy: "nested_cv"
  outer_cv_folds: 5

hpo_config:
  enabled: true
```

### TOML Format

```toml
[split_config]
strategy = "nested_cv"
outer_cv_folds = 5

[hpo_config]
enabled = true
```

## Configuration Priority

When multiple configuration sources are provided, MolBlender uses this priority order:

1. **Explicit parameters** passed to `universal_screen()` (highest priority)
2. **Configuration file** values
3. **Default values** from dataclass definitions (lowest priority)

Example:

```python
# test_size=0.3 from explicit parameter overrides config file
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    config_file="config.yaml",  # Has test_size=0.2
    split_config=SplitConfig(test_size=0.3)  # This wins
)
```

## Exporting Configurations

Save your current configuration for reuse:

```python
from molblender.config import export_screening_config
from molblender.models.api.multimodal.api import SplitConfig, HPOConfig

export_screening_config(
    split_config=SplitConfig(strategy="nested_cv", outer_cv_folds=10),
    hpo_config=HPOConfig(enabled=True, stage="fine"),
    combinations="comprehensive",
    verbose=2,
    output_file="my_exported_config.yaml"
)
```

## Best Practices

### Version Control

Store configuration files in git to track experiment settings:

```bash
git add experiments/config_v1.yaml
git commit -m "Add baseline screening configuration"
```

### Organize by Experiment

Use descriptive filenames:

```
configs/
├── baseline_regression.yaml
├── scaffold_split_hpo.yaml
├── nested_cv_comprehensive.yaml
└── production_fast.yaml
```

### Environment-Specific Configs

Use environment variables for deployment-specific settings:

```yaml
# config.yaml - works in dev/test/prod
resource_config:
  max_cpu_cores: ${CORES:-4}

database_config:
  db_path: "${DATA_DIR:-./data}/results.db"
```

```bash
# Development
export CORES=4
export DATA_DIR=./dev_data

# Production
export CORES=32
export DATA_DIR=/mnt/storage/prod_data
```

## Complete Example

```python
from molblender.api import universal_screen
from molblender.data import MolecularDataset, InputType

# Load dataset
dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    mol_input_type=InputType.SMILES,
    label_columns=["activity"]
)

# Run screening with config file
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    config_file="screening_config.yaml"
)

# Results are saved to database specified in config
print(f"Best model: {results['best_model']}")
print(f"Best score: {results['best_score']}")
```

## API Reference

- {func}`~molblender.config.create_config_template` - Create configuration template
- {func}`~molblender.config.load_config_file` - Load configuration from file
- {func}`~molblender.config.export_screening_config` - Export current configuration
- {func}`~molblender.config.save_yaml` - Save configuration to YAML

## See Also

- {doc}`/usage/models/index` - Model screening guide
- {doc}`/usage/data/index` - Data handling guide
- {doc}`/api/config/index` - Configuration API reference
