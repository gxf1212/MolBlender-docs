# Dashboard Troubleshooting Guide

This guide covers common issues and solutions when using the MolBlender Dashboard.

## Table of Contents

1. {ref}`🚀 Quick Troubleshooting Path <quick-troubleshooting-path>` - NEW
2. [Merged Sessions Loading](#merged-sessions-loading)
3. [Common Error Messages](#common-error-messages)
4. [Diagnostic Commands](#diagnostic-commands)
5. [Performance Tips](#performance-tips)

---

(quick-troubleshooting-path)=
## 🚀 Quick Troubleshooting Path

**Get your Dashboard working in 3 steps or less.**

### Step 1: Verify Database

```bash
# Check database has results
sqlite3 your_results.db "SELECT COUNT(*) FROM model_results;"

# Expected: > 0
# If 0: Your screening didn't produce results. Re-run screening.
```

### Step 2: Start Dashboard Correctly

```bash
# ✅ CORRECT - Use molblender view
molblender view path/to/results_folder/

# ❌ WRONG - Don't do this
python -m molblender.dashboard.app  # Will fail!
```

**Expected output**:
```
✓ Found results file: path/to/results_folder/all_classification_final.db
🚀 Starting dashboard on port 8502...
💻 Opening browser at http://localhost:8502
```

### Step 3: Common Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| **No results shown** | Check database has results (Step 1) |
| **"File not found" error** | Use `molblender view` with folder path, not file path |
| **Empty plots/charts** | Check if `predictions` column has data (not NULL) |
| **Slow loading** | Normal for >1000 results. Cache helps on 2nd load. |
| **Import errors** | Reinstall: `pip install -e .` |

### Still Stuck?

Jump to:
- [Merged Sessions Loading](#merged-sessions-loading) - Multi-session issues
- [Common Error Messages](#common-error-messages) - Specific errors
- [Diagnostic Commands](#diagnostic-commands) - Deep debugging
- [Performance Tips](#performance-tips) - Optimization

---

## Merged Sessions Loading

### Understanding Merged Sessions

When you run multiple screening sessions across different datasets or configurations, MolBlender can merge the results into a single view. The **merged session workflow** allows you to:

- View results from multiple datasets in one dashboard
- Compare performance across different screening configurations
- Analyze merged session data without manual aggregation

### Loading Merged Sessions

#### Method 1: Automatic Detection

The dashboard automatically detects merged session databases:

```bash
# Dashboard will detect merged_session.db automatically
molblender view results/
```

**How it works**:
1. Dashboard scans for `.db` files in the target directory
2. Prioritizes `*_merged.db` or `merged_session.db` files
3. Loads all sessions from the merged database

#### Method 2: Manual Selection

If you have multiple database files, use the file selector:

```bash
molblender view  # Opens file selector dialog
```

**In the UI**:
- Click "Browse" to select a specific database file
- Merged sessions will show all session IDs in the sidebar

### Merged Session Structure

A merged session database contains:

- **Multiple session_ids**: Each original screening session is preserved
- **Unified results**: All model results across sessions in one table
- **Session metadata**: Original dataset info and screening config per session

**Data flow**:
```
Original Session 1 (session_id="screening_20250308_001")
  ↓
Original Session 2 (session_id="screening_20250308_002")
  ↓
[Merge Process]
  ↓
Merged Session Database
  ├─ session_id="merged_session"
  ├─ Contains all results from Session 1 + Session 2
  └─ Preserves original session metadata
```

### Common Merged Session Issues

#### Issue: "No results found in merged session"

**Cause**: Database was created but no model results were written

**Solution**:
```bash
# Verify database has results
sqlite3 merged_session.db "SELECT COUNT(*) FROM model_results;"

# If 0, re-run the screening tasks that fed into this merge
```

#### Issue: "Session metadata missing for session_id XYZ"

**Cause**: Incomplete merge process or corrupted session

**Solution**:
```bash
# Check which sessions exist in the database
sqlite3 merged_session.db "SELECT DISTINCT session_id FROM model_results;"

# Verify screening_configs table has matching entries
sqlite3 merged_session.db "SELECT session_id FROM screening_configs;"
```

#### Issue: Overview tab shows "N/A" for merged session

**Cause**: Merged session doesn't have unified dataset_info

**Solution**:
- Merged sessions aggregate results but may not have unified dataset metadata
- This is expected behavior - use Detailed Results tab for merged sessions
- Overview tab works best with single-session databases

---

## Common Error Messages

### `TypeError: 'NoneType' object is not subscriptable`

**Symptom**:
```
TypeError: 'NoneType' object is not subscriptable
  File "dashboard/data/__init__.py", line 123, in load_results
```

**Cause**: Missing predictions or model_data in database

**Solutions**:

1. **Check database integrity**:
```bash
sqlite3 results.db "SELECT COUNT(*) FROM model_results WHERE predictions IS NULL;"
```

2. **Apply migrations** (if using old database format):
```python
from molblender.dashboard.data import ResultsDataLoader
loader = ResultsDataLoader()
results = loader.load_results("results.db")  # Auto-applies migrations
```

3. **Re-run failed models**:
```bash
# Identify which models have missing predictions
sqlite3 results.db \
  "SELECT model_name, representation_name FROM model_results WHERE predictions IS NULL;"
```

### `AttributeError: 'sqlite3.Row' object has no attribute 'get'`

**Symptom**:
```
AttributeError: 'sqlite3.Row' object has no attribute 'get'
```

**Cause**: Code trying to use dict-like `.get()` method on sqlite3.Row object

**Solution**:
- **This bug was fixed** in dashboard refactoring (Round 7)
- Update to latest MolBlender version:
```bash
pip install --upgrade molblender
```

- **If issue persists**, clear cache:
```bash
rm -rf .mbl_cache/
molblender view results/  # Fresh start
```

### `KeyError: 'primary_metric'`

**Symptom**:
```
KeyError: 'primary_metric'
  File "dashboard/metrics/central.py", line 45
```

**Cause**: Database schema mismatch (old format)

**Solution**:
```bash
# Run database migration
python -c "
from molblender.dashboard.data import ResultsDataLoader
loader = ResultsDataLoader()
results = loader.load_results('results.db')  # Auto-migrates
print('Migration complete')
"
```

### `StreamlitAPIException: This script can only be run from the Streamlit CLI`

**Symptom**:
```
StreamlitAPIException: This script can only be run from the Streamlit CLI
```

**Cause**: Importing dashboard module directly (not using `molblender view`)

**Solution**:
```bash
# ❌ WRONG - Don't do this
python -m molblender.dashboard.app

# ✅ CORRECT - Use CLI
molblender view results/
```

### `RuntimeWarning: Mean of empty slice` (NumPy warnings)

**Symptom**:
```
RuntimeWarning: Mean of empty slice
  RuntimeWarning: invalid value encountered in scalar divide
```

**Cause**: Empty prediction arrays or NaN values in results

**Impact**: **Not a critical error** - calculations handle NaN gracefully

**Solution** (if warnings clutter logs):
```python
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
```

---

## Diagnostic Commands

### Quick Health Check

```bash
# 1. Check database exists and is readable
sqlite3 results.db "SELECT COUNT(*) FROM model_results;"

# 2. Verify database schema
sqlite3 results.db ".schema model_results"

# 3. Check for NULL predictions
sqlite3 results.db \
  "SELECT COUNT(*) FROM model_results WHERE predictions IS NULL;"

# 4. List all sessions in database
sqlite3 results.db "SELECT DISTINCT session_id FROM model_results;"

# 5. Check primary_metric values
sqlite3 results.db \
  "SELECT primary_metric_name, AVG(primary_metric) as avg_score \
   FROM model_results GROUP BY primary_metric_name;"
```

### Minimal Diagnostic Script

Save as `diagnose_db.py`:

```python
#!/usr/bin/env python3
"""Minimal diagnostic script for MolBlender databases."""

import sqlite3
import sys
from pathlib import Path


def diagnose_database(db_path: str) -> None:
    """Run diagnostics on MolBlender database."""
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"📊 Diagnosing: {db_path}")
    print("=" * 50)

    # 1. Total results
    cursor.execute("SELECT COUNT(*) FROM model_results")
    total = cursor.fetchone()[0]
    print(f"✅ Total model results: {total}")

    # 2. Sessions
    cursor.execute("SELECT DISTINCT session_id FROM model_results")
    sessions = cursor.fetchall()
    print(f"✅ Sessions: {len(sessions)}")
    for session in sessions:
        print(f"   - {session[0]}")

    # 3. Missing predictions
    cursor.execute("SELECT COUNT(*) FROM model_results WHERE predictions IS NULL")
    missing = cursor.fetchone()[0]
    if missing > 0:
        print(f"⚠️  Results missing predictions: {missing}")
    else:
        print(f"✅ All results have predictions")

    # 4. Metric types
    cursor.execute(
        "SELECT primary_metric_name, COUNT(*) FROM model_results "
        "GROUP BY primary_metric_name"
    )
    metrics = cursor.fetchall()
    print(f"✅ Metric types: {len(metrics)}")
    for metric, count in metrics:
        print(f"   - {metric}: {count} results")

    # 5. Performance range
    cursor.execute(
        "SELECT MIN(primary_metric), MAX(primary_metric), AVG(primary_metric) "
        "FROM model_results WHERE primary_metric IS NOT NULL"
    )
    min_perf, max_perf, avg_perf = cursor.fetchone()
    print(f"✅ Performance range:")
    print(f"   - Min: {min_perf:.4f}")
    print(f"   - Max: {max_perf:.4f}")
    print(f"   - Avg: {avg_perf:.4f}")

    conn.close()
    print("=" * 50)
    print("✅ Diagnostics complete")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_db.py <database_path>")
        sys.exit(1)

    diagnose_database(sys.argv[1])
```

**Usage**:
```bash
python diagnose_db.py results/active_complete.db
```

### Dashboard Startup Diagnostics

The dashboard has built-in diagnostics (run before dashboard starts):

```bash
# Diagnostics run automatically with -v flag
molblender view results/ -v

# Output includes:
# ✓ Found results database: results/active_complete.db
# ✓ Loaded 150 model results
# ✓ Detected 3 available metrics: pearson_r2, rmse, mae
# ✓ Dashboard ready at http://localhost:8501
```

---

## Performance Tips

### Large Databases (>1000 results)

**Symptom**: Dashboard loads slowly

**Solutions**:

1. **Use metrics cache** (automatic):
```python
# Cache is created automatically on first load
# Subsequent loads are faster
# Cache location: .mbl_cache/<dataset_hash>/
```

2. **Filter early**:
- Use sidebar filters to reduce data before rendering
- Enable "Show top N only" in Model Analysis tab

3. **Limit loaded sessions**:
```bash
# Load specific session instead of all merged sessions
# (Feature coming in v1.1)
```

### Memory Issues

**Symptom**: Out of memory errors with large datasets

**Solutions**:

1. **Increase Streamlit memory limit**:
```bash
streamlit run dashboard/app.py --server.maxUploadSize=200
```

2. **Clear cache**:
```bash
rm -rf .mbl_cache/
```

3. **Use SQLite mode** (default):
- Results are loaded on-demand from database
- Not all data kept in memory

### Slow Rendering

**Symptom**: Charts take long to render

**Solutions**:

1. **Reduce chart complexity**:
- Use "Box Plot" instead of "Violin Plot" for large datasets
- Enable "Show top N only" in charts

2. **Disable auto-refresh**:
```python
# In dashboard/config.py (advanced)
AUTO_REFRESH_INTERVAL = 0  # Disable auto-refresh
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check diagnostics**:
```bash
python diagnose_db.py results/your_database.db
```

2. **Verify installation**:
```bash
molblender --version
python -c "import molblender; print(molblender.__version__)"
```

3. **Enable verbose logging**:
```bash
molblender view results/ -v --log-level=DEBUG
```

4. **Report bugs**:
   - Include diagnostic output
   - Attach database schema (`.schema` command)
   - MolBlender version number

---

## Appendix: Database Schema Reference

### Key Tables

**model_results**:
```sql
CREATE TABLE model_results (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    model_name TEXT,
    representation_name TEXT,
    primary_metric REAL,
    primary_metric_name TEXT,
    predictions BLOB,  -- JSON serialized
    model_params BLOB, -- JSON serialized
    -- ... more columns
);
```

**screening_configs**:
```sql
CREATE TABLE screening_configs (
    session_id TEXT PRIMARY KEY,
    config BLOB  -- JSON serialized ScreeningConfig
);
```

**dataset_info**:
```sql
CREATE TABLE dataset_info (
    session_id TEXT PRIMARY KEY,
    info BLOB  -- JSON serialized dataset metadata
);
```

---

**Last Updated**: 2026-03-08 (Round 7 Documentation)
**MolBlender Version**: 1.0.x
