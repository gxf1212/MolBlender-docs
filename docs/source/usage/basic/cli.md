# Command Line Interface (CLI)

MolBlender provides a powerful command-line interface for common operations like launching dashboards, managing screening results, and merging databases.

## Installation

Make sure MolBlender is installed in your environment:

```bash
cd /path/to/MolBlender
pip install -e .
```

## Available Commands

### 1. `molblender view` - Launch Interactive Dashboard

Start a web-based dashboard to explore screening results.

```bash
# Basic usage
molblender view /path/to/screening_results

# Custom port
molblender view ./screening_output --port 8503
```

**Features**:
- Interactive charts for performance analysis
- Model ranking and comparison
- Hyperparameter visualization
- Download results as CSV

---

### 2. `molblender merge_session` - Merge Database Sessions

Merge multiple sessions **within a single database file** into one unified result.

```bash
# Analyze database
molblender merge_session ./screening_results.db

# Merge sessions in-place
molblender merge_session ./screening_results.db --in-place

# Export merged results to JSON
molblender merge_session ./screening_results.db --output merged.json

# Export to new database
molblender merge_session ./screening_results.db --output screening_merged.db
```

**Options**:
- `--in-place`: Merge sessions directly in the original database
- `--output, -o`: Output file path (.json, .db, .sqlite, .sqlite3)
- `--remove-duplicates`: Remove duplicates by model + representation + params (default: True)
- `--keep-duplicates`: Keep all records including duplicates

**When to use**:
- Clean up duplicate entries from multiple runs
- Consolidate results from different sessions in the same database file
- Prepare database for dashboard analysis

---

### 3. `molblender merge_databases` - Merge Multiple Database Files ⭐ NEW

Merge **multiple different database files** into one unified database.

```bash
# Merge two databases
molblender merge_databases \
    screening_results_old.db \
    screening_results_new.db \
    --output screening_merged.db

# Merge multiple databases
molblender merge_databases \
    db_vector.db \
    db_string.db \
    db_matrix.db \
    db_image.db \
    --output screening_complete.db
```

**Options**:
- `--output, -o`: Output database file path (required)
- `--keep-duplicates`: Keep all entries including duplicates (default: remove duplicates, keep best score)
- `--keep-invalid`: Keep entries with NaN/invalid scores (default: remove them)

**Features**:
- **Automatic Deduplication**: Removes duplicates by `model_name + representation_name + params` (preserves HPO variants)
- **Best Score Retention**: When duplicates exist, keeps the result with best score
- **Invalid Data Filtering**: Removes entries with NaN/invalid scores automatically
- **Complete Schema Preservation**: Copies all 4 required tables (`screening_sessions`, `model_results`, `dataset_info`, `schema_version`) and preserves original session metadata

```{note}
**Recent Improvements**: Previous versions had bugs that could cause Dashboard loading failures. The latest version properly preserves all database tables and session IDs. Update to the latest version if you experienced issues with merged databases.
```

**When to use**:
- **Merge interrupted + resumed screenings**: Combine results from a failed run and a resumed run
- **Consolidate modality-specific screenings**: Merge separate VECTOR, STRING, MATRIX, IMAGE screening databases
- **Clean up test runs**: Remove duplicates from multiple test screenings
- **Prepare for dashboard**: Create one clean database for visualization

**Example Workflow**:
```bash
# Scenario: You ran screening 3 times with different settings
# Run 1: Only fingerprints (interrupted at 50%)
# Run 2: Only fingerprints (completed)
# Run 3: All modalities (completed)

# Merge all three databases
molblender merge_databases \
    run1_screening_partial.db \
    run2_fingerprints_complete.db \
    run3_all_modalities.db \
    --output final_merged.db

# Now view merged results
molblender view ./final_merged.db
```

---

### 4. `molblender info` - Show Package Information

Display version and installation information.

```bash
molblender info
```

---

## Common Workflows

### Workflow 1: Resume Interrupted Screening

```bash
# First run (interrupted)
python run_screening.py --disable-gpu
# → Creates screening_results.db with 500 results

# Resume and merge
python run_screening.py --disable-gpu --skip-existing
# → Creates screening_results_new.db with 800 results (500 old + 300 new)

# Merge databases
molblender merge_databases \
    screening_results.db \
    screening_results_new.db \
    --output screening_final.db

# View final results
molblender view ./screening_final.db
```

### Workflow 2: Combine Multi-Modality Screenings

```bash
# Run each modality separately
python screen_vector.py --output db_vector.db
python screen_string.py --output db_string.db
python screen_matrix.py --output db_matrix.db
python screen_image.py --output db_image.db

# Merge all databases
molblender merge_databases \
    db_vector.db \
    db_string.db \
    db_matrix.db \
    db_image.db \
    --output db_complete.db

# Analyze complete results
molblender view ./db_complete.db
```

### Workflow 3: Clean Duplicate Entries

```bash
# Check database before cleaning
molblender merge_session ./screening_results.db

# Clean duplicates in-place
molblender merge_session ./screening_results.db --in-place

# Or export cleaned version
molblender merge_session ./screening_results.db \
    --output screening_cleaned.db
```

---

## Troubleshooting

### Error: "Database file not found"
```bash
# Check if database exists
ls -lh screening_results.db

# Use correct path
molblender merge_databases screening_results.db --output merged.db
```

### Error: "No results found"
```bash
# Verify database has results
sqlite3 screening_results.db "SELECT COUNT(*) FROM model_results;"

# If empty, screening may have failed - check logs
tail -100 screening.log
```

### Error: "Table screening_sessions already exists"
```bash
# Delete old merged database and retry
rm screening_merged.db
molblender merge_databases db1.db db2.db --output screening_merged.db
```

---

## Best Practices

1. **Backup Before Merging**: Always backup original databases before merge operations
   ```bash
   cp screening_results.db screening_results_backup.db
   ```

2. **Verify Merged Results**: Check the merged database before using it
   ```bash
   molblender merge_session screening_merged.db
   ```

3. **Keep Good Records**: Maintain clear naming conventions for databases
   - `screening_results_YYYYMMDD.db`: Date-stamped results
   - `screening_VECTOR.db`, `screening_STRING.db`: Modality-specific
   - `screening_partial.db`, `screening_complete.db`: Status indicators

4. **Use `--skip-existing` for Resumes**: When re-running screening, use `--skip-existing` to avoid recomputing
   ```bash
   python run_screening.py --skip-existing
   ```

5. **Monitor Database Size**: Large databases (>500MB) may slow down merge operations
   ```bash
   # Check database size
   ls -lh screening_results.db
   ```

---

## See Also

- [Dashboard Documentation](../dashboard/index) - Visualize merged results
- [Model Screening Guide](../models/index) - Learn about screening workflows
- [Data Handling Guide](../data/index) - Dataset preparation tips
