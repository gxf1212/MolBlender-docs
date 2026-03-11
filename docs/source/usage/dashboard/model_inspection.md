# Model Inspection

Detailed analysis of individual model predictions, performance visualization, and export capabilities for production deployment.

## Overview

The Model Inspection tab allows you to examine specific models in detail, visualize predictions, and export model information.

**Key Features:**
- **Search & Filter** - Find models by keyword with pagination
- **Prediction Scatter Plots** - True vs predicted values with regression lines
- **Residual Analysis** - Identify systematic prediction errors
- **Model Details** - Parameters, training time, feature counts
- **Export Tools** - Download data and reproduction code

**Best For:** Model validation, selecting production models, understanding prediction quality, debugging failures

## Tab Location

**Navigation:** Dashboard → Model Inspection (Tab 3)

## Section 1: Search and Filter

Efficient navigation through large result sets.

### Search Interface

**Keyword Search:**
```
┌──────────────────────────────────────┐
│ 🔍 Search models...                 │
│ [random forest morgan_fp________]   │
└──────────────────────────────────────┘
```

**Search Capabilities:**
- **Model names** - "random", "xgboost", "ridge"
- **Representations** - "morgan", "rdkit", "descriptors"
- **Combinations** - "xgboost morgan_fp"
- **Partial matches** - "rand" finds "RandomForest"

**Case-insensitive** - "XGBoost" = "xgboost" = "XGBOOST"

### Filtering Options

**Performance Threshold Filter:**
```
┌─────────────────────────────────────┐
│ Show only: R² > [0.8___________]    │
│ Checkbox: ☑ Apply filter           │
└─────────────────────────────────────┘
```

Displays only models above specified performance threshold.

**Top-N Selection:**
```
┌─────────────────────────────────────┐
│ Display top: [10___] models         │
└─────────────────────────────────────┘
```

Shows only the N best-performing models.

### Pagination

For large result sets (100+ models):

```
┌─────────────────────────────────────┐
│ Showing 1-100 of 345 results        │
│                                     │
│ [< Previous]  Page 1 of 4  [Next >]│
└─────────────────────────────────────┘
```

**Features:**
- **100 models per page** - Optimal balance of detail and performance
- **Page navigation** - Previous/Next buttons
- **Jump to page** - Direct page selection
- **Result count** - Total matching models displayed

### Model Selection List

After filtering, models appear in a sortable list:

```
┌─────────────────────────────────────────────────────────┐
│ Model                    | Representation  | R²    | ▼  │
├─────────────────────────────────────────────────────────┤
│ XGBoost                  | morgan_fp_r2    | 0.856 | 🔍 │
│ RandomForest             | morgan_fp_r2    | 0.842 | 🔍 │
│ Ridge                    | rdkit_desc      | 0.798 | 🔍 │
│ ...                                                      │
└─────────────────────────────────────────────────────────┘
```

**Click 🔍 icon** to load model details and visualizations below.

## Section 2: Prediction Scatter Plot

Visualize how well the model's predictions match true experimental values.

### Plot Structure

```
Predicted
  1.0 ┤            •
      │          • •
  0.8 ┤        •••
      │      ••••
  0.6 ┤    ••••
      │  •••
  0.4 ┤••
      │
  0.2 ┤
      └────────────────
       0.2  0.4  0.6  0.8  1.0
                True Values

• Test data points
─ Perfect prediction line (y=x)
─ Fitted regression line
```

### Components

**Perfect Prediction Line (y = x):**
- Diagonal line from (0,0) to (1,1)
- If all predictions were perfect, points would fall on this line
- Gray dashed line

**Fitted Regression Line:**
- Linear fit through actual data points
- Shows systematic over/under-prediction
- Blue solid line
- Equation displayed: y = mx + b, R²

**Data Points:**
- **Blue circles** - Test set predictions (default)
- **Orange circles** - Training set predictions (if "Show Train Data" enabled)
- **Size** - Scaled for visibility
- **Transparency** - Overlapping points visible

**Annotations:**
- **R² value** - Correlation between predictions and true values
- **RMSE** - Root mean squared error
- **MAE** - Mean absolute error
- **N points** - Number of test samples

### Interpretation Guide

**Perfect Predictions:**
```
Predicted
  │      •
  │    ••
  │  ••
  │••
  └────────
   True
```
All points on y=x line. R² = 1.0. Model captures all variance.

**Good Predictions:**
```
Predicted
  │     •••
  │   •••
  │ •••
  │•
  └────────
   True
```
Points cluster near y=x line. R² > 0.8. Reliable model.

**Systematic Overestimation:**
```
Predicted
  │       •
  │      ••
  │    ••     ← Points above
  │  ••         y=x line
  └────────
   True
```
Fitted line above y=x. Model consistently overestimates. Check calibration.

**Systematic Underestimation:**
```
Predicted
  │  •
  │ ••         ← Points below
  │••            y=x line
  │•
  └────────
   True
```
Fitted line below y=x. Model consistently underestimates.

**Heteroscedastic Errors:**
```
Predicted
  │        •
  │      •••    ← Spread increases
  │    ••••••
  │  •••
  └────────
   True
```
Error variance increases with value. Consider log transformation or different model.

**Outliers:**
```
Predicted
  │          •  ← Far from line
  │    •••
  │  •••
  │ •
  └────────
   True
```
Few points far from regression line. Investigate: data errors? difficult molecules?

**Random Scatter (Poor Model):**
```
Predicted
  │  • •  •
  │ •   •
  │  •    •  ← No pattern
  │ •  •
  └────────
   True
```
No correlation between predicted and true. R² ≈ 0. Model failed.

### Interactive Features

**Hover Over Points:**
```
┌────────────────────────┐
│ SMILES: CCO            │
│ True: 0.72             │
│ Predicted: 0.68        │
│ Error: -0.04           │
│ Residual: -0.04        │
└────────────────────────┘
```
Shows molecule identifier, true value, prediction, and error.

**Zoom:**
- Click and drag to zoom into region
- Double-click to reset zoom
- Useful for examining dense clusters

**Show/Hide Training Data:**
```
Checkbox: ☐ Show Training Data
```
- **Enabled** - Displays both train (orange) and test (blue) points
- **Disabled** - Shows only test data (default)

**Why Compare Train vs Test:**
- **Train points much better than test** → Overfitting
- **Train and test similar** → Good generalization
- **Both poor** → Underfitting or difficult task

### Use Cases

✅ **Model validation** - Is R² consistent with cross-validation?
✅ **Outlier identification** - Which molecules are poorly predicted?
✅ **Systematic bias** - Is there over/under-prediction pattern?
✅ **Heteroscedasticity** - Does error variance change with magnitude?
✅ **Publication figures** - High-quality scatter plots

## Section 3: Residual Analysis

Examine prediction errors to identify systematic patterns.

### Residual Plot

```
Residual
  0.2 ┤    •
      │  • •
  0.0 ┤•••••••  ← Ideally random scatter
      │  • •      around zero
 -0.2 ┤    •
      └────────
       0.2  0.8
     Predicted Value
```

**Residual = True Value - Predicted Value**

### Components

**Zero Line:**
- Horizontal line at y=0
- Perfect predictions would all be on this line

**Residual Points:**
- **Above zero** - Model underestimated
- **Below zero** - Model overestimated
- **Distance from zero** - Magnitude of error

### Interpretation Guide

**Random Pattern (Good):**
```
Residual
  │  • •
  │•  •  •
  ├─────────  ← No pattern
  │ •  •
  │  •
```
Points randomly scattered around zero. No systematic bias. Good model.

**Systematic Underestimation (Poor):**
```
Residual
  │ •••
  │•••••
  ├─────────  ← All positive
  │
  │
```
All residuals positive. Model consistently underestimates. Needs calibration.

**Systematic Overestimation (Poor):**
```
Residual
  │
  │
  ├─────────  ← All negative
  │•••••
  │ •••
```
All residuals negative. Model consistently overestimates.

**Funnel Pattern (Heteroscedasticity):**
```
Residual
  │      •
  │    •••    ← Spread increases
  │  •••••      (funnel shape)
  ├───••──
  │  •••
```
Error variance increases with predicted value. Consider transformation or different model.

**Non-linear Pattern (Model Misspecification):**
```
Residual
  │ •      •
  │  •••••    ← Curved pattern
  ├────────
  │  •••
  │ •    •
```
Curved residual pattern. Linear model insufficient. Try non-linear models.

### Use Cases

✅ **Bias detection** - Identify systematic over/under-prediction
✅ **Heteroscedasticity check** - Is error variance constant?
✅ **Model assumptions** - Validate linear model appropriateness
✅ **Error patterns** - Find non-random error structure

## Section 4: Model Details

Comprehensive information about model configuration and performance.

### Model Information Panel

```
┌─────────────────────────────────────┐
│ Model Configuration                 │
├─────────────────────────────────────┤
│ Model: XGBoost                      │
│ Representation: morgan_fp_r2_1024   │
│ Features: 1,024                     │
│ Training Time: 12.34 seconds        │
│                                     │
│ Performance Metrics:                │
│  R²: 0.856                          │
│  RMSE: 0.423                        │
│  MAE: 0.312                         │
│  Pearson R: 0.925                   │
│                                     │
│ Cross-Validation:                   │
│  Mean: 0.852 ± 0.012               │
│  Folds: [0.849, 0.867, 0.851,       │
│          0.849, 0.844]              │
└─────────────────────────────────────┘
```

### Model Parameters

```
┌─────────────────────────────────────┐
│ Hyperparameters                     │
├─────────────────────────────────────┤
│ n_estimators: 200                   │
│ learning_rate: 0.1                  │
│ max_depth: 6                        │
│ subsample: 0.8                      │
│ colsample_bytree: 0.8              │
│ random_state: 42                    │
└─────────────────────────────────────┘
```

Shows exact configuration for reproducibility.

### Representation Configuration

```
┌─────────────────────────────────────┐
│ Representation Details              │
├─────────────────────────────────────┤
│ Type: Fingerprint                   │
│ Modality: VECTOR                    │
│ Algorithm: Morgan (Circular)        │
│ Radius: 2                           │
│ Bits: 1024                          │
│ Use Features: No                    │
└─────────────────────────────────────┘
```

## Section 5: Export Tools

Download model data and generate reproduction code via Dashboard or CLI.

### CLI Export Commands

Export models directly from database without opening Dashboard:

**Export Best Model:**
```bash
molblender export best screening_results.db -o best_model.py
```

**Export Specific Model:**
```bash
molblender export model screening_results.db \
  --model xgboost \
  --representation datamol_avalon \
  -o model.py
```

**Export Top N Models:**
```bash
molblender export top screening_results.db \
  --output-dir models/ \
  --number 5
```

**CLI Features:**
- Reads all metadata from database (no guessing)
- Uses exact train/test indices from screening
- Generates complete reproduction scripts
- Supports absolute CSV paths

### Export Model Data as CSV

**Button:** "Download Model Data (CSV)"

**Contents:**
```csv
molecule_id,true_value,predicted_value,residual
mol_001,0.72,0.68,-0.04
mol_002,0.85,0.89,0.04
mol_003,0.54,0.51,-0.03
...
```

**Use For:**
- Offline analysis in Excel/Python
- Sharing results with collaborators
- Custom visualization tools
- Quality control checks

### Generate Reproduction Code (Dashboard)

**Button:** "💾 Download Python Script"

**Generated Script Features:**
- **Configuration section** - All parameters at top
- **Absolute paths** - CSV filepath from database
- **Smart split handling** - Uses pre-defined indices from database
- **Exact parameters** - Model hyperparameters from screening
- **Multiple metrics** - Pearson R, R² score, RMSE, MAE

**Example Output:**
```python
# Model Recreation Code: xgboost + datamol_avalon
# Generated by MolBlender Dashboard

# Configuration
CSV_FILEPATH = '/path/to/dataset.csv'
INPUT_COLUMN = 'standardized_smiles'
TARGET_COLUMN = 'pX'
REPRESENTATION = 'datamol_avalon'

# Load dataset
dataset = MolecularDataset.from_csv(
    filepath=CSV_FILEPATH,
    input_column=INPUT_COLUMN,
    mol_input_type=InputType.SMILES,
    label_columns=[TARGET_COLUMN]
)

# Generate representations (uses cache automatically)
featurizer = get_featurizer(REPRESENTATION)
X = featurizer.featurize(dataset.molecules)

# Data split using pre-defined indices from database
import json
train_indices = np.array(json.loads('[0, 1, 2, ...]'))
test_indices = np.array(json.loads('[10, 11, 12, ...]'))
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Train model with screening-optimized parameters
from xgboost import XGBRegressor
model = XGBRegressor(
    colsample_bytree=0.8,
    gamma=0.1,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=50,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
from scipy.stats import pearsonr
pearson_r, _ = pearsonr(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Pearson R: {pearson_r:.4f}")
print(f"R² score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
```

**Running Exported Script:**
```bash
python best_model.py
```

**Expected Output:**
```
Using pre-defined split from database
Train size: 321, Test size: 32
Pearson R: 0.8617
R² score: 0.5752
RMSE: 0.2411
MAE: 0.1684
```

### Save Best Model

After identifying your production model:

```python
# From your Python environment after screening
import joblib

# Get best model from screening results
best_model = results['best_estimator']

# Save to disk
joblib.dump(best_model, 'my_production_model.pkl')

# Later: Load and use
loaded_model = joblib.load('my_production_model.pkl')
predictions = loaded_model.predict(new_features)
```

## Practical Workflows

### Workflow 1: Finding Production Model

**Goal:** Select best model for deployment

1. **Review Performance Analysis tab** → Note top 3 models
2. **Switch to Model Inspection tab**
3. **Search for each top model**
4. **For each model:**
   - Check prediction scatter plot R²
   - Examine residual plot for bias
   - Review training time (if deployment speed matters)
   - Note parameter complexity

5. **Decision criteria:**
   - **Highest R²** → Best accuracy
   - **Lowest residual bias** → Most calibrated
   - **Fastest training** → Easier to retrain
   - **Simplest parameters** → Easier to maintain

6. **Export winner:**
   - Generate reproduction code
   - Download model data CSV
   - Document model details

### Workflow 2: Debugging Poor Performance

**Goal:** Understand why a model fails

1. **Search for poorly performing model**
2. **Load prediction scatter plot**
3. **Diagnose issue:**

   **If R² very low (<0.3):**
   - Random scatter → Wrong representation/model
   - Check if different modality needed

   **If systematic bias (all above/below line):**
   - Model miscalibrated
   - Consider calibration techniques

   **If many outliers:**
   - Hover over outliers → Note molecule IDs
   - Check if outliers share structural features
   - May need feature engineering

   **If heteroscedastic (funnel pattern):**
   - Error variance not constant
   - Try log-transforming target variable
   - Or use quantile regression

4. **Action:**
   - Switch to better representation/model
   - Or refine dataset (remove outliers)
   - Or apply transformations

### Workflow 3: Model Comparison

**Goal:** Compare two similar-performing models

1. **Search first model** (e.g., "xgboost morgan")
2. **Note:**
   - R² and RMSE
   - Residual pattern
   - Training time

3. **Search second model** (e.g., "random_forest morgan")
4. **Compare:**
   - Which has higher R²?
   - Which has less bias (residual plot)?
   - Which trains faster?
   - Which has simpler parameters?

5. **Decision:**
   - If similar accuracy: Choose faster/simpler
   - If accuracy differs: Choose more accurate
   - Consider ensemble of both

### Workflow 4: Outlier Investigation

**Goal:** Understand why certain molecules are poorly predicted

1. **Load best model in Model Inspection**
2. **Zoom into outlier region** of scatter plot
3. **Hover over outlier points** → Note SMILES
4. **Record outlier molecules:**
   ```
   SMILES: CC(C)C1=CC=C(C=C1)C(C)C(O)=O
   True: 2.34
   Predicted: 0.87
   Error: -1.47
   ```

5. **Analyze outlier structures:**
   - Do they share common substructures?
   - Are they chemically unusual?
   - Data quality issues (measurement errors)?

6. **Actions:**
   - If systematic: Add relevant features
   - If rare structures: Collect more similar data
   - If data errors: Correct or remove

## Tips and Best Practices

```{admonition} Model Inspection Tips
:class: tip

1. **Always check residual plot** - Scatter R² can be misleading
2. **Use keyword search** - Faster than scrolling through lists
3. **Compare train vs test** - Detect overfitting early
4. **Export reproduction code** - Document for future reference
5. **Hover for details** - Every point has molecule-level information
6. **Pagination helps** - Don't be overwhelmed by 100+ models
```

```{admonition} Red Flags to Watch For
:class: warning

- **Train R² >> Test R²** - Overfitting, model won't generalize
- **Funnel residuals** - Heteroscedasticity, model assumptions violated
- **Systematic bias** - All residuals positive/negative, needs calibration
- **Many outliers** - Data quality or representation issues
- **Curved residual pattern** - Non-linear relationship, wrong model type
```

## Performance Notes

**For Large Result Sets (1000+ models):**
- Pagination automatically enabled (100/page)
- Search and filters reduce displayed models
- Scatter plots cached for speed
- Residual plots computed on-demand

**Optimization Tips:**
- Use keyword search to narrow results
- Apply performance threshold filters
- Close other browser tabs
- Clear browser cache if slow

## Next Steps

- **Compare distributions**: {doc}`distributions` - See how this model fits overall distribution
- **Performance overview**: {doc}`performance` - Compare with other models
- **Get started**: {doc}`quickstart` - Launch the dashboard
