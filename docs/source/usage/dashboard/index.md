# Interactive Dashboard

Professional web-based visualization for exploring MolBlender screening results with interactive charts, dynamic filtering, and comprehensive analysis tools.

## Overview

The MolBlender dashboard replaces static PDF reports with a modern, interactive interface built on Streamlit and Plotly. Launch with a single command and explore your results through dynamic visualizations, sortable tables, and real-time metric switching.

```bash
# Launch dashboard
molblender view /path/to/results

# Opens at http://localhost:8502
```

## Key Features

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🎯 **Dynamic Metrics**
Switch between 42 evaluation metrics instantly. All charts update in real-time.
:::

:::{grid-item-card} 📊 **Rich Visualizations**
Performance comparisons, distributions, scatter plots, heatmaps, and more.
:::

:::{grid-item-card} 🔍 **Interactive Tables**
Sort, filter, search through 1000+ models with pagination and export.
:::

:::{grid-item-card} 📈 **Statistical Analysis**
Correlation heatmaps, hierarchical clustering, significance testing.
:::

:::{grid-item-card} ⚡ **Fast Performance**
Optimized caching and vectorized operations for large result sets.
:::

:::{grid-item-card} 💾 **Export Ready**
Download filtered CSV data, high-res charts, and reproduction code.
:::

::::

## Quick Start

### 1. Install Dashboard Dependencies

```bash
# Install with dashboard extras
pip install molblender[dashboard]

# Or install manually
pip install streamlit plotly scipy
```

### 2. Run Screening

```python
from molblender.models.api import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_db_storage=True  # Creates SQLite database
)
# Results saved to: screening_results.db
```

### 3. Launch Dashboard

```bash
# From database file
molblender view ./screening_results.db

# From results directory
molblender view ./screening_results_20250115

# Custom port
molblender view ./results --port 8503
```

The dashboard automatically opens in your browser at `http://localhost:8502`.

## Dashboard Sections

The dashboard is organized into multiple tabs for different analysis perspectives:

### Tab 1: Performance Analysis

Comprehensive overview of model and representation performance.

**What You'll Find:**
- **Overview Cards** - Dataset statistics, best model, key metrics at a glance
- **Model Comparison** - Bar charts comparing average performance across representations
- **Representation Analysis** - Effectiveness of different molecular representations
- **Performance Distributions** - Box plots, violin plots, histograms showing metric distributions

**Use For:** Understanding overall results, identifying top performers, comparing model families

→ See {doc}`performance` for detailed guide

### Tab 2: Distribution Analysis

Deep dive into performance distributions with flexible grouping options.

**What You'll Find:**
- **5 Chart Types** - Box, violin, raincloud, histogram, density plots
- **Flexible Grouping** - By model type, representation, modality, or performance quartile
- **Statistical Summaries** - Quartiles, outliers, distribution shapes

**Use For:** Identifying outliers, understanding variance, comparing distributions

→ See {doc}`distributions` for detailed guide

### Tab 3: Model Inspection

Detailed analysis of individual model predictions and performance.

**What You'll Find:**
- **Search & Filter** - Find specific models by keyword with pagination
- **Prediction Scatter Plots** - True vs predicted values with regression lines
- **Residual Analysis** - Identify systematic prediction errors
- **Parameter Details** - Inspect model configurations
- **Export Tools** - Download data and generate reproduction code

**Use For:** Debugging models, understanding predictions, selecting production models

→ See {doc}`model_inspection` for detailed guide

### Tab 4: Detailed Results

Sortable, filterable table of all screening results.

**What You'll Find:**
- **Interactive Table** - Sort by any column, filter by thresholds
- **Multi-column Display** - Model name, representation, test metric, CV mean/std, best fold score, training time
- **Export** - Download filtered results as CSV

**Use For:** Finding specific combinations, exporting subsets, ranking by custom criteria

## Supported Metrics

The dashboard supports **42 evaluation metrics** across regression, classification, and cross-validation:

**Regression Metrics:**
- R², Pearson R, Spearman ρ, Kendall τ
- RMSE, MAE, MSE, MedAE, Max Error
- MAPE, MSLE, Explained Variance

**Classification Metrics:**
- Accuracy, Balanced Accuracy
- F1 Score (macro, micro, weighted, binary)
- Precision, Recall (macro, micro, weighted)
- ROC-AUC, Matthews Correlation, Cohen's κ

**Cross-Validation Metrics:**
- CV Mean, CV Std
- Individual Fold Scores

## Data Sources

The dashboard loads results from multiple sources:

**SQLite Database** (Recommended)
```bash
molblender view ./screening_results.db
```
- Fastest loading
- Supports multiple sessions
- Incremental updates

**JSON Files**
```bash
molblender view ./screening_results.json
```
- Backward compatibility
- Easy sharing
- Human-readable

**Results Directory**
```bash
molblender view ./results_folder
```
- Auto-detects .db or .json files
- Loads first found

## Key Features in Detail

### Dynamic Metric Switching

Switch evaluation metrics on-the-fly without reloading:

1. Select metric from sidebar dropdown
2. All charts update instantly
3. Rankings automatically adjust
4. Tables resort by new metric

**Supported:** All 42 metrics for any chart or table

### Interactive Charts

All visualizations are built with Plotly for full interactivity:

- **Hover** - See exact values and details
- **Zoom** - Click and drag to zoom regions
- **Pan** - Shift+drag to pan view
- **Select** - Box or lasso select data points
- **Export** - Download as PNG (high resolution)

### Flexible Filtering

Filter results by multiple criteria:

- **Performance Thresholds** - Show only R² > 0.8
- **Model Types** - Filter to tree-based models only
- **Modalities** - Compare VECTOR vs STRING results
- **Top-N Selection** - Display top 10 performers

### Search and Pagination

Handle large result sets efficiently:

- **Keyword Search** - Find models by name or representation
- **Pagination** - 100 models per page
- **Jump to Page** - Navigate directly to specific pages
- **Total Count** - Always see total results matching filters

### Export Capabilities

Download data and visualizations:

- **CSV Export** - Filtered results as spreadsheet
- **Chart Images** - High-resolution PNG downloads
- **Reproduction Code** - Python code to recreate best model
- **Model Parameters** - Full configuration details

## Why Dashboard > PDF Reports

| **Feature** | **PDF Reports** | **Dashboard** |
|------------|----------------|--------------|
| Interactivity | ❌ Static | ✅ Fully interactive |
| Metric Switching | ❌ Fixed metrics | ✅ 42 metrics on-the-fly |
| File Size | ❌ 5-50 MB | ✅ <1 MB (SQLite) |
| Filtering | ❌ Pre-generated views | ✅ Custom filters |
| Updates | ❌ Regenerate entire PDF | ✅ Live data loading |
| Sharing | ❌ Email attachment | ✅ Single command launch |
| Accessibility | ❌ Limited | ✅ Screen reader support |
| Mobile | ❌ Poor rendering | ✅ Responsive design |

## Performance Optimizations

The dashboard is optimized for large result sets (1000+ models):

**Caching Strategy:**
- `@st.cache_data` with 5-minute TTL
- Cached: Data loading, metric calculations, predictions
- Invalidation: Automatic on file changes

**Vectorized Operations:**
- Pandas vectorization for display formatting
- NumPy for metric calculations
- 100x faster than Python loops

**Lazy Loading:**
- Pagination for large tables
- Progressive chart rendering
- Spinners for user feedback

**Memory Management:**
- JSON serialization for cache keys
- Garbage collection after tab switches
- Efficient DataFrame operations

## Navigation Guide

**Sidebar:**
- Metric selector (sticky at top)
- File information
- Quick statistics
- Filter controls

**Main Area:**
- Tab navigation (large, clear labels)
- Chart containers with titles
- Interactive visualizations
- Export buttons

**Tips:**
- Use Ctrl+F to search within tables
- Shift+Click to select multiple chart elements
- Right-click charts for save options
- Refresh browser to reload data

## Common Use Cases

### Finding the Best Model

1. Launch dashboard: `molblender view results.db`
2. Check **Performance Analysis** tab → Overview cards
3. Look at **Model Comparison** chart → Identify top models
4. Switch to **Model Inspection** tab → Search for best model
5. Review prediction scatter plot and residuals
6. Export model parameters and reproduction code

### Comparing Representation Families

1. Go to **Performance Analysis** tab
2. View **Representation Analysis** chart (grouped by modality)
3. Switch metrics (R², RMSE, Kendall τ) to see consistency
4. Note which modalities perform best
5. Export filtered results for top representations

### Debugging Poor Performance

1. Go to **Distribution Analysis** tab
2. Select box plot chart type
3. Group by model type
4. Identify models with high variance or low medians
5. Switch to **Model Inspection** → Search failing models
6. Check residual plots for systematic bias

## Troubleshooting

**Dashboard won't start:**
```bash
# Check dependencies
pip install streamlit plotly scipy

# Try specific port
molblender view results.db --port 8503

# Check file exists
ls -lh results.db
```

**No results displayed:**
- Verify file contains data: `sqlite3 results.db "SELECT COUNT(*) FROM model_results;"`
- Check browser console for errors (F12)
- Try loading from JSON: `molblender view results.json`

**Slow performance:**
- Reduce displayed results with filters
- Clear browser cache
- Use pagination for large tables
- Close other Streamlit apps

**Port already in use:**
```bash
# Kill existing process
lsof -ti:8502 | xargs kill -9

# Or use different port
molblender view results.db --port 8504
```

## CLI Reference

```bash
# Basic usage
molblender view <path>

# Options
molblender view <path> --port 8503  # Custom port
molblender info                      # Package info
molblender --help                    # Full help

# Examples
molblender view ./results.db
molblender view ./screening_results_20250115
molblender view ../experiments/run1/results.json --port 8505
```

## Detailed Guides

Explore comprehensive guides for each dashboard section:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🚀 **Getting Started**
:link: quickstart
:link-type: doc
Installation, first launch, interface overview
:::

:::{grid-item-card} 📊 **Performance Analysis**
:link: performance
:link-type: doc
Overview cards, model comparisons, distributions
:::

:::{grid-item-card} 📈 **Distribution Analysis**
:link: distributions
:link-type: doc
5 chart types, flexible grouping, statistics
:::

:::{grid-item-card} 🔍 **Model Inspection**
:link: model_inspection
:link-type: doc
Search, predictions, residuals, exports
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

quickstart
performance
distributions
model_inspection
metrics
advanced
workflows
```

## Next Steps

- **Install and launch**: {doc}`quickstart` - Get started in 5 minutes
- **Explore visualizations**: {doc}`performance` - Understand your results
- **Analyze distributions**: {doc}`distributions` - Deep dive into performance patterns
- **Inspect models**: {doc}`model_inspection` - Examine individual predictions
