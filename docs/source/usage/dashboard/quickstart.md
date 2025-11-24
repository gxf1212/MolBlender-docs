# Getting Started

Install, launch, and navigate the MolBlender dashboard with this comprehensive getting started guide.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4 GB minimum, 8 GB recommended
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **CPU**: 2 cores minimum

### Recommended for Large Results (>1000 models)
- **RAM**: 16 GB or more
- **CPU**: 4+ cores recommended
- **Browser**: Chrome or Firefox (best performance)

### Optional Enhancements
- **GPU**: Not required but improves model training (not dashboard)
- **SSD**: Faster SQLite database loading

## Installation

### Method 1: Install with Dashboard Extras (Recommended)

```bash
# Install MolBlender with dashboard dependencies
pip install molblender[dashboard]
```

This installs:
- `streamlit>=1.28.0` - Dashboard framework
- `plotly>=5.17.0` - Interactive charts
- `scipy>=1.9.0` - Statistical functions
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical operations

### Method 2: Manual Installation

```bash
# Install core MolBlender
pip install molblender

# Install dashboard dependencies manually
pip install streamlit plotly scipy

# Optional: additional statistical packages
pip install scikit-learn matplotlib seaborn
```

### Method 3: Development Installation

```bash
# Clone repository (if developing)
git clone https://github.com/your-org/molblender.git
cd molblender

# Install in development mode
pip install -e ".[dashboard]"
```

### Verify Installation

```bash
# Check if molblender command is available
molblender --help

# Expected output:
usage: molblender view [-h] [--port PORT] [--verbose] path
positional arguments:
  path                  Path to results file or directory
options:
  -h, --help           show this help message and exit
  --port PORT            Port to run dashboard on (default: 8502)
  --verbose             Enable verbose logging
```

## First Screening Run

Create your first screening results to explore in the dashboard.

### Example: Quick Screening

```python
from molblender.data import MolecularDataset
from molblender.models.api import quick_screen

# 1. Prepare sample dataset (or use your own)
sample_data = """SMILES,logP
CCO,-0.32
CCOCC,-0.41
CCOCCC,-0.86
CCN,-0.57
CC(C)O,-0.17
"""

# Save to CSV
import pandas as pd
df = pd.read_csv(pd.StringIO(sample_data))
df.to_csv("molecules.csv", index=False)

# 2. Load dataset
dataset = MolecularDataset.from_csv(
    "molecules.csv",
    input_column="SMILES",
    label_columns=["logP"]
)

# 3. Run quick screening
results = quick_screen(
    dataset=dataset,
    target_column="logP"
)

print(f"Screening completed! Best R²: {results['best_score']:.3f}")
```

### Example: Universal Screening (Recommended)

```python
from molblender.models.api import universal_screen

# Run comprehensive screening
results = universal_screen(
    dataset=dataset,
    target_column="logP",
    enable_db_storage=True,  # Creates SQLite database
    max_cpu_cores=-1       # Use all available cores
)

print(f"Results saved to: {results['database_path']}")
```

## Launching Dashboard

### Basic Launch

```bash
# From SQLite database (recommended)
molblender view ./screening_results.db

# From JSON file (backward compatibility)
molblender view ./screening_results.json

# From directory (auto-detects .db or .json)
molblender view ./results_folder
```

### Custom Port Configuration

```bash
# Use custom port (when default 8502 is occupied)
molblender view ./screening_results.db --port 8503

# Try multiple ports if needed
molblender view ./results --port 8504 --verbose
```

### Verbose Mode

```bash
# Enable detailed logging for troubleshooting
molblender view ./results.db --verbose
```

## What to Expect

### Browser Opening

The dashboard automatically opens in your default browser:

```
URL: http://localhost:8502

Example view in browser:

🧪 MolBlender Results Dashboard
========================================

[ Performance Analysis ] [ Distribution Analysis ] [ Model Inspection ] [ Detailed Results ]

Data Source: screening_results.db
Models Evaluated: 8
Best Score: R² = 0.856
Loading: ███████████████████████████████ 100%

┌─────────────────────────────────────────────┐
│ Dataset Information                        │
├─────────────────────────────────────────────┤
│ Molecules: 5                            │
│ Features: 644                             │
│ Task: Regression (logP)                   │
│ Primary Metric: R²                         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Model Performance                         │
├─────────────────────────────────────────────┤
│ [Bar Chart: Random Forest: 0.856]     │
│ [Bar Chart: XGBoost: 0.842]          │
│ [Bar Chart: Ridge: 0.798]             │
└─────────────────────────────────────────────┘
```

### Dashboard Interface Components

**Sidebar (Left Side):**
- **Metric Selector** - Choose evaluation metric (R², RMSE, etc.)
- **File Information** - Current data source and statistics
- **Quick Statistics** - Models tested, best score, etc.
- **Filter Controls** - Performance thresholds, model types

**Main Area:**
- **Tab Navigation** - 4 main analysis tabs
- **Charts Container** - Interactive visualizations
- **Export Buttons** - Download charts and data
- **Status Indicators** - Loading spinners, error messages

## Navigation Basics

### Understanding the Layout

```
┌─────────────────┬──────────────────────────────────────────┐
│                 │                                          │
│  S I D E B A R  │              M A I N   A R E A               │
│                 │                                          │
│ Metric Selector │ [ Tab 1 ] [ Tab 2 ] [ Tab 3 ] [ Tab 4 ] │
│ File Info       │                                          │
│ Statistics     │           Chart Area                    │
│ Filters        │                                          │
│                 │         [ Export Button ]                │
└─────────────────┴──────────────────────────────────────────┘
```

### Interactive Elements

**Charts:**
- **Hover** - Move mouse over chart elements to see values
- **Zoom** - Click and drag to zoom in on chart regions
- **Reset Zoom** - Double-click chart or use Reset axes button
- **Download** - Right-click chart → Save image as

**Tables:**
- **Sort** - Click column headers to sort ascending/descending
- **Filter** - Use filter controls to show only subsets
- **Search** - Use browser search (Ctrl+F) to find text
- **Pagination** - Navigate through large tables

**Tabs:**
- Click any tab to switch analysis perspectives
- Content updates instantly without page reload
- Each tab has specialized tools for different analyses

### Keyboard Shortcuts

| **Key** | **Action** |
|---------|------------|
| `Ctrl+F` | Search within tables |
| `Ctrl++` | Zoom in (browser) |
| `Ctrl+-` | Zoom out (browser) |
| `Ctrl+0` | Reset zoom (browser) |
| `F5` | Refresh dashboard |
| `Esc` | Cancel current operation |

## Your First Dashboard Exploration

### Step 1: Overview Cards (Performance Analysis Tab)

1. **Launch dashboard** with your results
2. **Look at Overview Cards** in the first section:
   - **Dataset Information** - Molecules, features, task type
   - **Performance Summary** - Best score, mean, standard deviation
   - **Configuration** - CV folds, test size, timestamp

3. **Note key information:**
   - How many models were evaluated?
   - What's the best performing score?
   - Are the results consistent (low standard deviation)?

### Step 2: Model Comparison

1. **Scroll down** to Model Comparison chart
2. **Hover over bars** to see exact performance values
3. **Compare different models** across representations
4. **Switch metrics** using the sidebar dropdown (try R², then RMSE)

**Key Questions to Ask:**
- Which model performs best overall?
- Is there consistent performance across metrics?
- Are there clear winners and losers?

### Step 3: Representation Analysis

1. **Scroll further** to Representation Analysis chart
2. **Grouped by modality** - see VECTOR vs other types
3. **Identify best representations** for your dataset

**Look For:**
- Which molecular representation works best?
- Are certain representation families consistently better?
- How much performance difference between top and bottom?

### Step 4: Model Inspection (Tab 3)

1. **Click "Model Inspection"** tab
2. **Search for best model** from previous analysis
3. **View prediction scatter plot**
4. **Check for patterns** in residuals

**To Look For:**
- How well do predictions match true values?
- Are there systematic errors or outliers?
- Is the model overfitting or underfitting?

### Step 5: Export Results

1. **Click "Download CSV"** in Detailed Results tab
2. **Open spreadsheet** to examine data offline
3. **Save best model** for future use:

```python
# From Python, after screening
best_model = results['best_estimator']
import joblib
joblib.dump(best_model, 'my_best_model.pkl')
```

## Common First-Time Issues

### Dashboard Won't Start

**Problem:**
```bash
molblender view results.db
# Error: command not found
```

**Solutions:**
```bash
# Check if molblender is installed
pip show molblender

# If not installed
pip install molblender[dashboard]

# If installed but not in PATH
python -m molblender.view results.db
```

**Problem:**
```bash
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install streamlit plotly scipy
```

### Port Already in Use

**Problem:**
```bash
Error: Port 8502 is already in use
```

**Solutions:**
```bash
# Use different port
molblender view results.db --port 8503

# Kill existing process (Linux/Mac)
lsof -ti:8502 | xargs kill -9

# Kill existing process (Windows PowerShell)
netstat -ano | findstr :8502
# Then kill the PID
taskkill /PID <PID> /F
```

### Results Not Loading

**Problem:** Dashboard shows "No results found" or empty charts

**Solutions:**
```bash
# Check file exists and has data
ls -lh results.db

# Verify SQLite database contains results
sqlite3 results.db "SELECT COUNT(*) FROM model_results;"

# If file is empty, run screening first:
from molblender.models.api import quick_screen
# ... your screening code ...
```

### Browser Issues

**Problem:** Dashboard loads but charts are blank or unresponsive

**Solutions:**
1. **Check browser console** (F12 → Console tab) for errors
2. **Clear browser cache** and refresh
3. **Try different browser** (Chrome works best)
4. **Disable ad blockers** or add localhost to exceptions

**Problem:** Dashboard very slow

**Solutions:**
1. **Close other browser tabs** and applications
2. **Reduce dataset size** or use filters
3. **Clear browser cache**
4. **Increase memory** if possible

### Performance Tips for Large Results

If you have 1000+ models:

1. **Use filters** to reduce displayed results
2. **Enable pagination** in tables
3. **Clear browser cache** before session
4. **Close unnecessary applications** to free memory
5. **Consider using Chrome** for best performance

## Tips for Better Experience

### Browser Optimization

1. **Use Chrome or Firefox** for best performance
2. **Maximize browser window** for better chart visibility
3. **Bookmark the dashboard URL** for easy access
4. **Enable hardware acceleration** in browser settings

### Workflow Optimization

1. **Start with Performance Analysis tab** for overview
2. **Use consistent metrics** when comparing results
3. **Export CSV** for offline analysis
4. **Save best models** for deployment
5. **Document your findings** using dashboard insights

### Data Organization

1. **Use descriptive database filenames** like `protein_binding_screening.db`
2. **Keep results organized** in project folders
3. **Archive old results** to avoid confusion
4. **Backup important databases** regularly

## Next Steps

### Immediate Actions
1. ✅ **Complete this guide** - You should now have a working dashboard
2. 🔍 **Explore your results** - Spend 15 minutes clicking around the interface
3. 📊 **Try different metrics** - See how rankings change with R² vs RMSE
4. 📥 **Export some data** - Download CSV for your records
5. 💾 **Save best model** - Keep your top performer

### Learning Resources
- **Deep dive into analysis**: {doc}`performance` - Master performance analysis
- **Distribution analysis**: {doc}`distributions` - Explore performance distributions
- **Model inspection**: {doc}`model_inspection` - Examine individual predictions

### Common Tasks to Try
1. **Compare representation families** - Which molecular features work best?
2. **Identify consistent performers** - Which models work across metrics?
3. **Find outliers** - Are there any unexpected poor/great performers?
4. **Export production model** - Get code to reproduce best model
5. **Document your findings** - Create summary of dashboard insights

## Getting Help

### Documentation
- This guide covers installation and first steps
- **Full dashboard guide**: {doc}`index` - Complete feature overview
- **API reference**: {doc}`/api/models/index` - Detailed API documentation
- **Examples directory**: Check `examples/` in the repository

### Troubleshooting
- **Common issues**: See "Common First-Time Issues" above
- **Verbose mode**: Use `--verbose` flag for detailed logs
- **Browser console**: Press F12 for error diagnostics
- **Community**: Check GitHub issues and discussions

### Quick Reference

```bash
# Quick commands
molblender view ./results.db                    # Launch dashboard
molblender view ./results --port 8503        # Custom port
molblender view ./results --verbose             # Debug mode

# Python essentials
from molblender.models.api import quick_screen
results = quick_screen(dataset, target_column="logP")
best_model = results['best_estimator']
```

---

**Congratulations!** You now have the MolBlender dashboard running and can explore your screening results interactively. 🎉

Next: Dive deeper into {doc}`performance` to understand your results in detail.