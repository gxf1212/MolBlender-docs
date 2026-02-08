# Working with Results

Learn how to access, analyze, and export screening results from MolBlender's SQLite database storage.

## Result Storage

MolBlender uses **SQLite databases** as the primary storage format for screening results, replacing older JSON-only approaches.

###Benefits of SQLite Storage

✅ **Incremental Saving** - Results saved after each model completes
✅ **Crash Recovery** - Resume interrupted screenings automatically
✅ **Efficient Queries** - Fast filtering and aggregation
✅ **Caching** - Skip already-completed combinations
✅ **Portable** - Single `.db` file contains all results

### Enabling Database Storage

```python
from molblender.models.api import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_db_storage=True,  # Enable SQLite storage
    db_path="./my_screening.db"  # Optional: custom path
)
```

## Database Schema

MolBlender creates three tables to store screening results:

### 1. screening_sessions

Stores metadata about each screening run:

```sql
CREATE TABLE screening_sessions (
    session_id TEXT PRIMARY KEY,
    timestamp TEXT,
    task_type TEXT,  -- 'regression' or 'classification'
    primary_metric TEXT,  -- 'pearson_r', 'f1', etc.
    dataset_name TEXT,
    cv_folds INTEGER,
    test_size REAL,
    random_state INTEGER,
    n_models_evaluated INTEGER,
    n_representations INTEGER,
    best_score REAL,
    mean_score REAL,
    std_score REAL
)
```

### 2. model_results

Individual model evaluation results:

```sql
CREATE TABLE model_results (
    id INTEGER PRIMARY KEY,
    session_id TEXT,  -- Links to screening_sessions
    model_name TEXT,
    representation_name TEXT,
    representation_config TEXT,  -- JSON representation config
    model_config TEXT,  -- JSON full model config
    score REAL,  -- Primary metric value
    rank INTEGER,
    cv_scores TEXT,  -- JSON array of fold scores
    training_time REAL,
    n_features INTEGER,
    model_params TEXT,  -- JSON model configuration
    predictions TEXT,  -- JSON predictions
    feature_importance TEXT,  -- JSON feature importance
    stage INTEGER,  -- 1 = screening, 2 = HPO
    hpo_stage TEXT,  -- 'coarse' or 'fine'
    best_params TEXT,  -- JSON best params (Stage 2)
    hpo_cv_score REAL,  -- Stage 2 CV score
    train_indices TEXT,  -- JSON indices
    test_indices TEXT,  -- JSON indices
    created_at TIMESTAMP
)
```

Additional columns may be present for dashboard compatibility (e.g., `primary_metric`,
`primary_metric_name`, `all_metrics`).

### 3. dataset_info

Train/test split information:

```sql
CREATE TABLE dataset_info (
    session_id TEXT PRIMARY KEY,
    target_column TEXT,
    dataset_n_train_samples INTEGER,
    dataset_n_test_samples INTEGER,
    train_true_values TEXT,  -- JSON array
    test_true_values TEXT,   -- JSON array
    train_input_data TEXT,   -- JSON SMILES/identifiers
    test_input_data TEXT     -- JSON SMILES/identifiers
)
```

## Accessing Results from Python

### Quick Access from Return Value

```python
results = universal_screen(dataset, target_column="activity")

# Best model information
best = results['best_model']
print(f"Model: {best['model_name']}")
print(f"Representation: {best['representation_name']}")
print(f"R²: {best['metrics']['r2']:.3f}")

# Access trained model
trained_model = best['estimator']
predictions = trained_model.predict(new_data)

# All results sorted by performance
for result in results['results'][:5]:  # Top 5
    print(f"{result['model_name']} + {result['representation_name']}: {result['score']:.3f}")
```

### Loading from Database

```python
from molblender.models.api.utils import load_results_from_database

# Load all sessions from database
results = load_results_from_database("./my_screening.db")

# Load specific session
results = load_results_from_database(
    "./my_screening.db",
    session_id="screening_20250115_103045"
)

# Access results
print(f"Best score: {results['summary']['best_score']}")
for model in results['results']:
    print(f"{model['model_name']}: {model['score']}")
```

### Direct SQL Queries

For advanced analysis, query the database directly:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("./my_screening.db")

# Get all results sorted by score
df = pd.read_sql_query("""
    SELECT
        model_name,
        representation_name,
        score,
        training_time,
        n_features
    FROM model_results
    WHERE session_id = ?
    ORDER BY score DESC
""", conn, params=("screening_20250115_103045",))

# Get summary statistics by model type
summary = pd.read_sql_query("""
    SELECT
        model_name,
        COUNT(*) as n_representations,
        AVG(score) as mean_score,
        MAX(score) as best_score,
        AVG(training_time) as avg_time
    FROM model_results
    WHERE session_id = ?
    GROUP BY model_name
    ORDER BY mean_score DESC
""", conn, params=("screening_20250115_103045",))

conn.close()
```

## Exporting Results

### Export to JSON

```python
from molblender.models.api.utils import export_to_json

# Export database results to JSON format
export_to_json(
    db_path="./my_screening.db",
    output_path="./results.json",
    session_id="screening_20250115_103045"  # Optional
)
```

### Export to CSV

```python
from molblender.models.api.utils import export_results_csv

# Export results table to CSV
export_results_csv(
    results=results,
    output_path="./screening_results.csv"
)

# Or use pandas for custom exports
import sqlite3
import pandas as pd

conn = sqlite3.connect("./my_screening.db")
df = pd.read_sql_query("SELECT * FROM model_results", conn)
df.to_csv("./all_results.csv", index=False)
conn.close()
```

### Save Best Model

```python
import joblib

# Get best model from results
best_estimator = results['best_estimator']

# Save using joblib (recommended for sklearn models)
joblib.dump(best_estimator, './best_model.pkl')

# Load later
loaded_model = joblib.load('./best_model.pkl')
predictions = loaded_model.predict(new_data)
```

## Result Structure

### Complete Result Dictionary

```python
{
    'success': True,
    'timestamp': '2025-01-15T10:30:45',
    'session_id': 'screening_20250115_103045',

    # Best model
    'best_model': {
        'model_name': 'xgboost',
        'representation_name': 'morgan_fp_r2_1024',
        'score': 0.852,
        'rank': 1,
        'metrics': {
            'r2': 0.852,
            'rmse': 0.543,
            'mae': 0.421,
            'pearson_r': 0.924,
            'spearman_rho': 0.912,
            'kendall_tau': 0.765
        },
        'cv_scores': [0.831, 0.867, 0.849, 0.856, 0.857],
        'cv_mean': 0.852,
        'cv_std': 0.012,
        'training_time': 12.34,
        'n_features': 1024,
        'estimator': <XGBRegressor object>,
        'model_params': {'n_estimators': 200, 'learning_rate': 0.1, ...},
        'predictions': {
            'test_predictions': [2.1, 3.4, ...],
            'test_true': [2.3, 3.2, ...]
        }
    },

    # Best estimator (trained model)
    'best_estimator': <XGBRegressor object>,
    'best_score': 0.852,

    # All results sorted by performance
    'results': [
        {'model_name': 'xgboost', 'score': 0.852, ...},
        {'model_name': 'random_forest', 'score': 0.841, ...},
        ...
    ],

    # Statistical summary
    'summary': {
        'n_models_evaluated': 18,
        'n_representations': 6,
        'n_unique_models': 3,
        'mean_score': 0.764,
        'std_score': 0.089,
        'best_modality': 'VECTOR',
        'worst_modality': None,
        'task_type': 'regression',
        'primary_metric': 'pearson_r'
    },

    # Configuration
    'config': {
        'task_type': 'regression',
        'primary_metric': 'pearson_r',
        'cv_folds': 5,
        'test_size': 0.2,
        'random_state': 42
    },

    # Storage
    'database_path': './my_screening.db',
    'output_dir': './screening_results_20250115_103045'
}
```

## Caching and Resuming

### Automatic Caching

With `enable_db_storage=True`, MolBlender automatically:

1. **Checks existing results** before training
2. **Skips completed combinations** (same model + representation)
3. **Saves incrementally** after each successful evaluation
4. **Resumes from interruption** (Ctrl+C, crashes, timeouts)

```python
# First run - trains 20 models
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_db_storage=True,
    db_path="./cache.db"
)
# Training completes 15/20 models before interruption...

# Second run - automatically skips 15 completed models
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_db_storage=True,
    db_path="./cache.db"  # Same database
)
# Only trains remaining 5 models!
```

### Manual Cache Management

```python
from molblender.models.api.utils import get_cache_info, clear_cache

# Check cache contents
cache_info = get_cache_info("./cache.db")
print(f"Cached sessions: {cache_info['n_sessions']}")
print(f"Cached results: {cache_info['n_results']}")

# Clear specific session
clear_cache("./cache.db", session_id="screening_20250115_103045")

# Clear entire database
clear_cache("./cache.db", clear_all=True)
```

## Launching the Dashboard

The easiest way to explore results is through the interactive dashboard:

```bash
# View results from database
molblender view ./my_screening.db

# View results from output directory
molblender view ./screening_results_20250115_103045

# Custom port
molblender view ./my_screening.db --port 8503
```

The dashboard automatically:
- Loads all sessions from the database
- Provides interactive charts and tables
- Supports metric switching
- Enables filtering and sorting
- Allows CSV export of filtered results

See {doc}`../dashboard/index` for complete dashboard documentation.

## Programmatic Analysis

### Compare Multiple Runs

```python
import sqlite3
import pandas as pd

# Load results from multiple databases
dbs = ["run1.db", "run2.db", "run3.db"]
all_results = []

for db_path in dbs:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            model_name,
            representation_name,
            score,
            training_time
        FROM model_results
    """, conn)
    df['run'] = db_path
    all_results.append(df)
    conn.close()

combined = pd.concat(all_results)
print(combined.groupby('model_name')['score'].agg(['mean', 'std', 'max']))
```

### Extract Predictions

```python
import json

# Get predictions from best model
best_model = results['best_model']
predictions_json = best_model['predictions']

# Parse JSON if stored as string
if isinstance(predictions_json, str):
    predictions = json.loads(predictions_json)
else:
    predictions = predictions_json

test_pred = predictions['test_predictions']
test_true = predictions['test_true']

# Calculate custom metrics
from sklearn.metrics import r2_score, mean_absolute_error
print(f"R²: {r2_score(test_true, test_pred):.3f}")
print(f"MAE: {mean_absolute_error(test_true, test_pred):.3f}")
```

### Feature Importance Analysis

```python
# Extract feature importance from tree-based models
best_model = results['best_model']

if 'feature_importance' in best_model:
    importance = best_model['feature_importance']

    # Create DataFrame for analysis
    import pandas as pd
    importance_df = pd.DataFrame({
        'feature': range(len(importance)),
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(importance_df.head(10))  # Top 10 features
```

## Best Practices

```{admonition} Recommendations
:class: tip

1. **Always enable `enable_db_storage=True`** for runs > 5 minutes
2. **Use descriptive `db_path` names** with timestamps or experiment IDs
3. **Export to JSON** for archiving and sharing
4. **Save best models** with joblib for production deployment
5. **Use the dashboard** for initial exploration before programmatic analysis
6. **Query database directly** for custom statistics and aggregations
```

```{admonition} Common Pitfalls
:class: warning

- **Don't delete `.db` files** until you've exported results
- **Check `success` field** before accessing results
- **Handle missing predictions** - not all models save prediction details
- **Close database connections** when done with direct SQL queries
```

## Troubleshooting

**Q: Database file is locked**
```python
# Close any open connections
conn.close()

# Or use context manager (auto-closes)
with sqlite3.connect("results.db") as conn:
    df = pd.read_sql_query("SELECT * FROM model_results", conn)
```

**Q: Can't find session_id**
```python
# List all sessions
conn = sqlite3.connect("results.db")
sessions = pd.read_sql_query("SELECT session_id, timestamp FROM screening_sessions", conn)
print(sessions)
conn.close()
```

**Q: Results dictionary missing keys**
```python
# Always check success status
if not results.get('success', False):
    print(f"Screening failed: {results.get('error')}")
    exit(1)

# Use .get() with defaults for optional keys
best_score = results.get('best_score', float('-inf'))
```

## Next Steps

- **Visualize results**: {doc}`../dashboard/index` - Interactive exploration
- **Run more screenings**: {doc}`screening` - Function reference
- **Understanding models**: {doc}`models` - Model catalog and selection
