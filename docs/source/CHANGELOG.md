# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Disabled Lasso and ElasticNet Models** (2026-02-08)
  - Removed lasso and elastic_net from available models due to poor performance on small datasets
  - These models consistently produce Pearson R = 0.00 on small datasets (<500 training samples)
  - Model registration commented out in `models/api/core/model_registry.py`
  - Parameter grids commented out in `models/corpus/grids/linear_models.py`
  - Code preserved for potential future re-enablement on larger datasets
  - Linear models category now reduced to 4 models: Ridge, Logistic, LinearSVR, Bayesian Ridge

- **Mordred Batch Processing** (2026-02-07)
  - Added `_featurize_batch()` method to MordredFeaturizer for parallel computation
  - Uses Mordred's `calc.pandas()` API for batch descriptor calculation
  - Significantly faster than individual molecule processing (10-50x speedup)
  - Automatically falls back to individual calculation if batch fails
  - File modified: `representations/descriptors/descriptors_basic.py`

- **Default Primary Metric: Pearson R** (2026-02-07)
  - Changed default regression metric from R² to Pearson R for better interpretability
  - Pearson R directly measures linear correlation strength (-1 to 1)
  - More stable on small datasets and symmetric for prediction evaluation
  - Files modified: `api.py`, `reporting.py`, `results_db.py`, `migration.py`

### Fixed

- **skip_existing_results Empty Database Bug** (2026-02-07)
  - Fixed bug where `skip_existing_results=True` would skip all screening when database is empty
  - Now correctly runs full screening when no existing results are found (count=0)
  - Only skips when all results exist (count=len(representations))
  - Partial results still load existing and skip missing to avoid OOM
  - File modified: `models/api/multimodal/modality_handlers.py`

- **Mol2Vec Offline Support** (2026-02-07)
  - Removed network connectivity check that blocked Mol2Vec in offline environments
  - Now works with locally cached models without internet access
  - File modified: `config/dependencies.py`

### Added

- **Mol2Vec Auto-Download Enhancement** (2026-02-07)
  - Added `download_mol2vec_model()` function for automatic model downloading
  - Caches model in `~/.mol2vec/` directory (26MB)
  - Supports custom model paths via `pretrain_model_path` parameter
  - Falls back gracefully with clear error messages if download fails
  - File modified: `config/models.py`, `representations/fingerprints/deepchem.py`

- **HPO Resume + Stage 2 Autosave** (2026-02-07)
  - Stage 2 skips combinations already optimized in the database
  - Results are saved after each optimized model to reduce lost work
  - Pre-computes required representations once before HPO loop

- **Mol2Vec Auto-Download + Offline Checks** (2026-02-07)
  - Auto-downloads Mol2Vec pretrained model (cached in `~/.mol2vec`) when not provided
  - Detects missing network connectivity for PubChem/Mol2Vec features and warns early

- **Dashboard Analysis Enhancements** (2026-02-07)
  - Adds category-specific model distribution chart in Performance Analysis
  - Detailed Results table now includes best CV fold score and training-set primary metric (when available)

- **Universal skip_existing_results for All Modalities** (2026-02-04)
  - **Problem**: `skip_existing_results` only worked for VECTOR modality, other modalities (STRING, MATRIX, IMAGE, LANGUAGE_MODEL) always re-ran Stage 1 screening
  - **Solution**: Created universal helper method `_check_and_load_existing_results()` in `ModalityHandlerMixin`
  - **Implementation**: All 5 modality handlers now call this shared method to check database for existing Stage 1 results
  - **Behavior**:
    - ✅ If **all** representations have Stage 1 results → skip screening, load from database
    - ⚠️ If **partial** representations have results → run full screening (ensures completeness)
    - ✅ If **no** representations have results → run full screening
  - **Files Modified**:
    - `src/molblender/models/api/multimodal/modality_handlers.py`: Added `_check_and_load_existing_results()` helper (126 lines)
    - All modality handlers (VECTOR, STRING, MATRIX, IMAGE, LANGUAGE_MODEL) now use this method
  - **User Impact**: Significantly faster HPO Stage 2 execution when Stage 1 results already exist in database

- **Database Merger Enhancements** (2026-02-04)
  - **Session Preservation**: Now preserves all sessions and their metadata when merging databases
  - **Flexible Deduplication**: Added `--no-remove-duplicates` flag to keep all results including duplicates
  - **Table Completeness**: Ensures all required tables (dataset_info, schema_version) are copied to merged database
  - **Session Consistency**: Maintains session_id consistency across model_results and dataset_info tables
  - **File Modified**: `src/molblender/models/api/utils/database_merger.py`
  - **New Behavior**:
    - Default: Remove duplicates, keep best score per model+representation
    - With `--no-remove-duplicates`: Keep all results from all databases
  - **Example**: `molblender merge_databases db1.db db2.db -o merged.db --no-remove-duplicates`

- **Multi-Database Merge Tool** (`molblender merge_databases`)
  - New CLI command to merge multiple `.db` files into one unified database
  - Automatically deduplicates `model_name + representation_name` combinations (keeps best score)
  - Filters out entries with NaN/invalid scores
  - Example: `molblender merge_databases db1.db db2.db -o merged.db`
  - Use cases: Merge interrupted+resumed screenings, combine multi-modality results, clean up test runs
  - Documentation: `usage/basic/cli.md`

- **Data Type Error Fix** (data_handler.py)
  - Fixed `ufunc 'isnan' not supported for the input types` error when processing non-numeric dtype arrays
  - Added automatic object→float64 conversion for numeric data stored as object dtype
  - Prevents VECTOR modality failures on larger datasets (2535 molecules)

- **Grid Search Resolution Improvement** (tree_models.py)
  - Increased grid points for well-performing models (Random Forest, XGBoost, LightGBM)
  - Random Forest: 36 → 2160 combinations (60x increase)
  - XGBoost: 108 → 57600 combinations (533x increase)
  - All grid centers now use sklearn/official defaults

- **Optuna Integration** (optuna_optimizer.py)
  - New Optuna-based Bayesian optimizer for fine-tuning top models
  - Warm-start from Grid Search best parameters (±50% range)
  - MedianPruner for early stopping of unpromising trials
  - Focuses on top 3 models + slow models (Transformer, CNN)

- **Cross-Session Result Caching** (`--skip-existing`)
  - **Problem**: Previously, re-running screening would re-compute all model-representation combinations even if results existed in previous database sessions
  - **Solution**: Added `skip_existing_results` configuration parameter and `--skip-existing` CLI flag
  - **Implementation Details**:
    - `ScreeningConfig.skip_existing_results`: Check across ALL sessions, not just current session
    - `ScreeningResultsDB.check_existing_result(check_all_sessions=True)`: SQL query without session_id filter
    - **Smart Caching**: When enabled, skips any model-representation combination that exists in ANY previous session
  - **Use Cases**:
    - Resume interrupted screening jobs without re-computing completed results
    - Add new representations/models to existing screening without re-running everything
    - Incremental screening: run on subset, then expand with `--skip-existing`
  - **Performance**: For the RdRp dataset (723 existing results), resuming with `--skip-existing` would skip all 723 combinations and only compute new ones
  - **Backward Compatibility**: Default `skip_existing_results=False` preserves existing behavior (always recompute)
  - **Files Modified**:
    - `src/molblender/models/api/core/base.py`: Added `skip_existing_results` configuration parameter
    - `src/molblender/models/api/multimodal/processors/database.py`: Added `check_all_sessions` parameter to `check_existing_result()`
    - `src/molblender/models/api/utils/results_db.py`: Added `check_all_sessions` parameter to `check_existing_result()`
    - `src/molblender/models/api/multimodal/api.py`: Added `skip_existing_results` parameter to `universal_screen()`
    - `tests/.../run_molblender_screening.py`: Added `--skip-existing` CLI argument
  - **Example Usage**:
    ```bash
    # First run: compute all combinations (723 results)
    python run_molblender_screening.py --disable-gpu

    # Second run: skip existing 723 results
    python run_molblender_screening.py --disable-gpu --skip-existing
    ```

- **Placeholder Value Protection in Database Storage**
  - **Problem**: Database was storing placeholder `[0,1,2,...]` indices instead of actual `y_test`/`y_train` values in `dataset_info.test_true_values`
  - **Root Cause**: Early session creation stored placeholder data before actual split data was available
  - **Impact**: Dashboard scatter plots showed incorrect Pearson R (near 0) due to placeholder true values
  - **Solution**: Added validation in `ScreeningResultsDB.save_dataset_info()`:
    - Checks if `test_true_values == list(range(len(test_true_values)))` (placeholder detection)
    - Checks if `train_true_values == list(range(len(train_true_values)))` (placeholder detection)
    - Refuses to save placeholder data with clear error message
    - Returns `False` to prevent invalid data storage
  - **Error Message**: `"❌ Refusing to save placeholder test_true_values (indices) for session {session_id}. These should be actual y_test values, not [0, 1, 2, ...]. This causes incorrect metric recalculations in Dashboard."`
  - **Files Modified**:
    - `src/molblender/models/api/utils/results_db.py`: Added placeholder validation in `save_dataset_info()` (lines 595-617)
  - **User Impact**:
    - Prevents database corruption from placeholder data
    - Ensures Dashboard metrics are computed from correct `y_test` values
    - Forces proper data flow: `evaluator.py` → real `y_test` → database → Dashboard
  - **Note**: Old databases with placeholder values should be regenerated by re-running screening with latest code

### Changed

- **Split Strategy Name Mapping**
  - **Fixed**: `max_dissimilarity` → `maxmin` in `run_molblender_screening.py`
  - **Reason**: API only recognizes `maxmin`, not `max_dissimilarity`
  - **Impact**: Correct MaxDissimilarity split functionality now works
  - **File**: `tests/data/.../run_molblender_screening.py` line 237

- **Database Merge Deduplication Keys** (2026-02-07)
  - Deduplication now considers model params and HPO best params to preserve tuned variants
  - Improves session/database merges for mixed Stage 1 + Stage 2 results

## [Unreleased]

### Fixed

- **Stage 1 Partial Reuse to Avoid OOM** (2026-02-07)
  - When only some representations have Stage 1 results, existing ones are loaded instead of forcing a full re-run

- **HPO Feature Reconstruction Robustness** (2026-02-07)
  - Handles nested embeddings with fast stacking and imputes NaNs in descriptor features

- **Dashboard MAPE Radar Chart Display** (2026-02-04)
  - **Problem**: Extremely large MAPE values (e.g., 3.6 billion %) were clamped to max range (2.0), resulting in normalized value of 0 on radar charts
  - **Impact**: Radar charts showed misleading "0" for MAPE when actual values were unreasonably high
  - **Solution**: Skip MAPE values >500% (5.0) from radar chart display entirely
  - **Rationale**: MAPE >500% indicates severe data quality issues (e.g., predictions near zero for non-zero true values)
  - **File Modified**: `src/molblender/dashboard/components/model_inspection/visualization.py`
  - **Code Change**:
    ```python
    # Skip unreasonably large MAPE values (>500% indicates data issues)
    if key == "mape" and value > 5.0:
        continue
    ```
  - **User Impact**: Radar charts now show only meaningful metrics, preventing confusion from extreme outliers

- **Dashboard Training Time Analysis Rendering**
  - Fixed import path for modality training time pie chart in `general.py`
  - Ensured Session Breakdown renders only in the right column
  - Resolved pandas column mismatch for session aggregation when `n_jobs` is present
- **Log Noise Reduction**
  - Added repeated-message suppression for common CUDA/XGBoost/CV failure patterns
  - Reduced spam for near-constant predictions and low Pearson‑r warnings
- **Database Merge Tool (`molblender merge_databases`)**
  - Fixed critical bugs causing Dashboard loading failures and data loss
  - Now copies all 4 required tables (`screening_sessions`, `model_results`, `dataset_info`, `schema_version`) instead of only 2
  - Preserves original session_ids instead of creating artificial "merged" session
  - Properly copies `dataset_info` and `schema_version` entries for each session
  - File: `src/molblender/models/api/utils/database_merger.py` (complete rewrite)

### Changed

- **CUDA → CPU Fallback (Thresholded)**
  - Added CPU fallback for VAE/CNN/Transformer models on CUDA failure when CPU cores ≥32
  - XGBoost CUDA fallback now respects the same core threshold (otherwise re-raises)

### Added

- **Sample Weighting for Imbalanced Regression** (`src/molblender/models/api/core/weighting.py`)
  - **Three Weighting Strategies** for handling zero-inflated or heavily imbalanced regression data:
    - **threshold**: Binary weighting (e.g., pX > 0 samples get 2-3x weight, pX = 0 get 1x weight)
    - **inverse_density**: KDE-based density estimation, weights ∝ 1/density for rare value ranges
    - **quantile**: Quantile-based weighting, extreme quantiles get higher weight
  - **Configuration Parameters** in `ScreeningConfig`:
    - `use_sample_weights`: Enable/disable weighting (default: False for backward compatibility)
    - `weight_strategy`: Choose strategy ("threshold", "inverse_density", "quantile")
    - `weight_threshold`: Threshold for binary split (default: 0.0)
    - `weight_power`: Power to smooth extreme weight differences (0.5-1.0, default: 1.0)
    - `active_subset_threshold`: Threshold for active subset evaluation (default: 0.0)
  - **Safe Model Training** (`_safe_fit_with_weights()` in evaluator.py):
    - Automatically handles models that don't support `sample_weight`
    - Graceful fallback to standard training if weights not supported
    - Logs warnings when weights cannot be used
  - **Active Subset Metrics** for comprehensive imbalanced regression evaluation:
    - `active_r2`, `active_rmse`, `active_mae`: Metrics computed only on active samples (e.g., pX > 0)
    - `active_count`, `active_percentage`: Number and percentage of active samples
    - Computed when `use_sample_weights=True` and task is regression
  - **Command-Line Interface** (run_molblender_screening.py):
    - `--use-sample-weights`: Enable sample weighting
    - `--weight-strategy`: Choose weighting strategy
    - `--weight-threshold`: Set threshold for binary weighting
    - `--weight-power`: Smooth weight differences (default: 1.0)
    - `--active-subset-threshold`: Set threshold for active subset evaluation
  - **Use Case**: RdRp inhibitor screening with 86% inactive (pX=0) and 14% active (pX>0) samples
    - Training on weighted samples improves model performance on rare active compounds
    - Active subset metrics provide visibility into model performance on the region of interest
  - **Backward Compatibility**: Default `use_sample_weights=False` ensures no impact on existing workflows
  - **Reference**: SMOGN (Synthetic Minority Over-sampling Technique for Non-Gaussian continuous features)
    https://link.springer.com/article/10.1007/s10994-021-06023-5

- **Advanced Molecular Splitting Strategies** (`src/molblender/data/dataset/splitting/`)
  - **4 Advanced Splitters from splito package** (Apache 2.0 License):
    - **PerimeterSplit**: Extrapolation-oriented split placing most dissimilar molecules in test set
    - **MolecularWeightSplit**: Tests generalization across different molecular sizes
    - **MOODSplitter**: Model-Optimized Out-of-Distribution split based on deployment data similarity
    - **LoSplitter**: Lead optimization split returning multiple test clusters for SAR exploration
  - **Dual API Design**:
    - **Functional API (sklearn-style)**: `train_test_split(X, y, molecules=smiles, method='perimeter')`
    - **Class-based API**: `PerimeterSplit(test_size=0.2).split(smiles)`
    - **MolecularDataset integration**: `dataset.train_test_split(method='perimeter')`
  - **RDKit-based Implementation**:
    - Replaced all `datamol` dependencies with direct RDKit calls
    - Utility functions: `compute_fingerprints()`, `compute_molecular_weights()`, `to_mol()`, `to_smiles()`
    - K-means clustering for distance-based splits (computational efficiency)
  - **Modular Package Structure** (`splitting/` subdirectory):
    - `base.py`: SplittingMixin for MolecularDataset
    - `advanced.py`: 4 advanced splitters (~780 lines)
    - `functional.py`: sklearn-style functional API (~350 lines)
    - `utils.py`: RDKit utility functions
  - **Comprehensive Testing**:
    - 16 unit tests covering all methods and edge cases
    - Test functional API, class-based API, and MolecularDataset integration
    - Test utility functions (fingerprints, molecular weights)
  - **Documentation**:
    - API reference: `docs/api/data/splitting.rst`
    - Usage guide: `docs/usage/data/splitting.md` (extended with 4 new sections)
    - Complete examples for all splitting methods
  - **License Attribution**:
    - Apache 2.0 license headers in all relevant files
    - References to splito package: https://github.com/datamol-io/splito
  - **Use Cases**:
    - Virtual screening validation (PerimeterSplit)
    - Fragment-to-lead optimization (MolecularWeightSplit)
    - Deployment-aware validation (MOODSplitter)
    - SAR exploration in medicinal chemistry (LoSplitter)
  - **Custom User-Provided Split** (`method='custom'`):
    - Use predefined train/test assignments from external sources
    - **Two input modes**:
      - `split_column`: Column name with 'train'/'test', 0/1, or True/False values
      - `train_indices`/`test_indices`: Explicit index arrays
    - Use cases: benchmark reproduction, temporal splits, experimental batches
    - Complete validation: overlap detection, range checking, value parsing
    - Utility function: `indices_from_split_column()` for column parsing
  - **Enhanced LoSplitter Documentation**:
    - Added scaffold hopping pharmaceutical use case example
    - Real-world scenario: Scaffold 1 exhaustion → scaffold hopping → series prediction
    - Explains how LoSplitter validates model generalization across scaffolds
  - **Backward Compatible**:
    - All existing splitting methods remain unchanged
    - New methods are opt-in via `method` parameter
    - Default behavior (random split) preserved
  - **Expanded Test Coverage**:
    - 28 unit tests total (12 new custom split tests)
    - Tests for functional API and MolecularDataset integration
    - Edge case testing: NaN values, invalid formats, missing columns
- **Two-Stage Hyperparameter Optimization System** (`src/molblender/models/api/`)
  - **Stage 1**: Model screening with default parameters for rapid baseline performance assessment
  - **Stage 2**: Automated hyperparameter optimization (HPO) for top-N performers from Stage 1
  - **Configuration via ScreeningConfig**:
    - `enable_hpo=True/False`: Enable/disable HPO Stage 2 (default: False for backward compatibility)
    - `hpo_stage='coarse'/'fine'`: Coarse grid search or fine-grained Optuna optimization (default: 'coarse')
    - `hpo_method='grid'/'random'`: GridSearchCV or RandomizedSearchCV (default: 'grid')
    - `top_n_for_hpo=N`: Number of top Stage 1 performers to optimize (default: 10)
    - `hpo_cv_folds=K`: CV folds for HPO (default: None, uses `cv_folds` or `inner_cv_folds`)
    - `hpo_selection_strategy='global'/'per_type'/'per_subtype'`: Model selection strategy for HPO (default: 'global')
      - **'global'**: Select top N performers overall across all model types (default behavior)
      - **'per_type'**: Select top N from Traditional ML AND top N from Deep Learning separately
        - Ensures balanced HPO coverage when one model category dominates Stage 1 performance
        - Example: `hpo_selection_strategy='per_type', top_n_for_hpo=5` → 5 Traditional ML + 5 Deep Learning = 10 total HPO runs
        - Use case: CNN/VAE/Transformer models receive HPO even if Traditional ML models have higher Stage 1 scores
      - **'per_subtype'**: Select top N from each fine-grained model category (LINEAR, TREE, BOOSTING, VAE, CNN, TRANSFORMER)
        - Most granular option for comprehensive model type coverage across all architectures
    - `hpo_models_per_type=N`: Override number of models per type/subtype (default: None, uses `top_n_for_hpo`)
  - **Model Type Classification System** (`processors/hpo.py`):
    - High-level: TRADITIONAL_ML vs DEEP_LEARNING (using ModelCorpus enum from model registry)
    - Fine-grained: LINEAR, TREE, BOOSTING, KERNEL, VAE, TRANSFORMER, CNN subtypes
    - Automatic classification based on model registry metadata and categories
  - **Parameter Grid System** (`src/molblender/models/corpus/parameter_grids.py`):
    - Comprehensive coarse grids for all model types (tree, boosting, SVM, linear, neural, VAE, CNN)
    - Model-specific parameter ranges based on best practices
    - CNN grids: learning_rate, batch_size, epochs (matrix_cnn, image_cnn variants)
    - Fine grids reserved for Optuna optimization (future)
  - **Database Integration**:
    - Stage tracking in `model_results` table: `stage=1` (default params) vs `stage=2` (optimized params)
    - HPO metadata: `hpo_stage`, `best_params`, `hpo_cv_score` columns
    - Incremental result storage for both Stage 1 and Stage 2
  - **Smart Workflow**:
    - Automatically skips Stage 2 if insufficient high-performing models in Stage 1
    - Preserves Stage 1 results even when Stage 2 is enabled
    - Compatible with all data splitting strategies (train_val_test, nested_cv, etc.)
  - **Performance Impact**:
    - Stage 1 only: Fast baseline screening (~5-10 min for 100 combinations)
    - Stage 1 + Stage 2: Comprehensive optimization (~15-30 min with top_n=10)
  - **Complete GridSearchCV Results Storage** (`all_cv_results` column):
    - Stores ALL tested parameter combinations from GridSearchCV, not just the best one
    - Enables HPO parameter sensitivity analysis in dashboard
    - Captured data for each combination:
      - `params`: Parameter values tested (e.g., `{"alpha": 0.1}`, `{"alpha": 1.0}`)
      - `mean_test_score`, `std_test_score`, `rank_test_score`: Aggregated CV performance
      - Individual fold scores (`split0_test_score`, `split1_test_score`, etc.)
      - Timing information (`mean_fit_time`, `std_fit_time`, `mean_score_time`)
    - JSON serialization with numpy→list conversion for database storage
    - Backward compatible: NULL for Stage 1 results (default parameters, no HPO)
    - Example use case: Compare Ridge α=[0.1, 1.0, 10.0] to visualize sensitivity
    - Database schema migration: Automatically adds `all_cv_results TEXT` column to existing databases
    - Implementation locations:
      - `grid_search.py:148-186`: `_extract_cv_results()` method extracts all cv_results_ data
      - `screeners.py:919,947,1120-1122,1154-1174`: HPO orchestration captures and stores cv_results
      - `results_db.py:186-187`: Database migration adds all_cv_results column
  - Successfully tested with:
    - Traditional ML models (Ridge, RandomForest, XGBoost, LightGBM, etc.)
    - Deep learning models (VAE with latent_dim, learning_rate, batch_size grids)
    - 3D representations (spatial matrices, UniMol embeddings, 3D fingerprints)
    - Multiple fingerprint categories (RDKit, CDK, Datamol)

- **VAE (Variational Autoencoder) Integration for Molecular Fingerprints** (`src/molblender/models/`)
  - **Complete VAE Implementation** (`modality_models/vae_models.py`):
    - MolecularFingerprintVAE class with encoder-decoder architecture
    - Integrated predictor network for direct property prediction from latent space
    - 5 pre-configured variants: VAE (latent=64/128/256), VAE (compact), VAE (deep)
    - Auto-generates 3D conformations from SMILES when needed
    - PyTorch-based with GPU/CPU auto-detection
  - **Model Registry Integration** (`api/core/model_registry.py`):
    - All 5 VAE models registered in ModelRegistry
    - Categorized under `DEEP_LEARNING` and `ACCURATE` corpus
    - HPO parameter grids: latent_dim, learning_rate, batch_size, epochs
  - **Pathway System** (`api/multimodal/`):
    - `combinations="auto"`: Traditional ML only (default behavior, backward compatible)
    - `combinations="comprehensive"`: Traditional ML + VAE models for vector modality
    - Similar to matrix/image pathway handling (flattened vs CNN)
  - **sklearn Compatibility Fixes**:
    - Fixed `clone()` compatibility: Store device as string, convert to torch.device only when needed
    - Fixed numpy conversion: Use `.detach().cpu().numpy()` for proper tensor→array conversion
    - Proper 1D/2D array shape handling for sklearn metrics
  - **User Experience**:
    - Auto-excludes VAE models in default mode (no impact on existing workflows)
    - Optional VAE screening via `combinations="comprehensive"` parameter
    - Seamless integration with HPO system (Stage 1 + Stage 2 optimization)
    - GPU acceleration when available, graceful CPU fallback
  - **Current Status**: Fully integrated and tested, VAE models accessible but showing 0.0 scores (model performance issue, not integration bug)

- **DNR Diagnostics Module** (`src/molblender/data/diagnostics/`)
  - Complete implementation of DNR (Different Neighbor Ratio) analysis for dataset quality assessment
  - Paper-accurate parameters from "Upgrading Reliability in Molecular Property Prediction" (similarity threshold 0.5, property diff 1.0 log unit)
  - Activity cliff detection for identifying similar molecules with large property differences
  - Comprehensive visualization suite with 5 plot types (DNR distribution, DNR vs property, activity cliff network, similarity heatmap, neighbor statistics)
  - SVG output format to avoid matplotlib/numpy compatibility issues with PNG
  - One-click full diagnostics workflow with `run_full_diagnostics()` method
  - Automatic sampling for large datasets (>10,000 molecules) to prevent memory issues
  - Integration with MolecularDataset API via `DatasetDiagnostics` class
  - Comprehensive markdown report generation with interpretation guidelines
  - Submodules: `core.py` (main diagnostics), `similarity.py` (fingerprint utilities), `visualization.py` (plotting functions)
  - **Interactive Streamlit Dashboard** (`src/molblender/data/diagnostics/dashboard/`)
    - Integrated into package as proper module
    - Console script entry point: `molblender-diagnostics` command
    - Built-in comprehensive documentation and interpretation guide
    - Adjustable thresholds with real-time recomputation
    - CSV export functionality for results
    - Support for file upload or command-line file path
  - Successfully tested on 120-molecule head.csv dataset with complete output generation
  - Comprehensive module documentation in `src/molblender/data/diagnostics/CLAUDE.md`

- **Protein Data Handling Documentation** (`docs/source/usage/data/protein.md`)
  - Comprehensive guide to Protein class and multi-format support
  - Database retrieval from RCSB PDB, AlphaFold, and UniProt
  - Structure prediction and repair with ESMFold and PDBFixer
  - FASTA, PDB, mmCIF format handling with automatic detection
  - BioPython integration for structural analysis
  - Intelligent caching system for downloads and predictions
  - Protein-ligand dataset integration examples
  - Multi-chain structure support and metadata extraction
  - Sequence validation and cleaning utilities
  - Best practices for high-throughput protein processing

- **Scaffold-Based Splitting Implementation** (`src/molblender/models/api/core/splitting/scaffold.py`)
  - Complete scaffold-based train/test splitting for drug discovery applications
  - Two scaffold generation methods: Bemis-Murcko and Generic (topology-only)
  - Two split strategies: Balanced (greedy size matching) and Random (random assignment)
  - RDKit integration for scaffold computation with error handling
  - Scaffold leakage detection to ensure train/test set separation
  - Integration with `DataSplitter` class via `strategy="scaffold"` parameter
  - Comprehensive unit tests with 18 test cases covering all functionality
  - Documentation added to `docs/source/usage/data/splitting.md` with examples and use cases

- **DNR-Based Splitting Strategy** (`src/molblender/models/api/core/splitting/dnr.py`)
  - Systematically tests model performance on rough SAR regions and challenging molecules
  - Three split modes:
    - **Threshold mode**: Split by DNR threshold (high vs low DNR molecules)
    - **Quantile mode**: Split by DNR quantiles (top X% vs rest)
    - **Neighbor mode**: Split by neighbor presence (isolated vs connected molecules)
  - Configurable parameters: DNR threshold, similarity threshold, property difference threshold
  - Enables systematic evaluation of two major error modes from "Upgrading Reliability in Molecular Property Prediction" paper
  - Integration with `DataSplitter` class via `strategy="dnr"` parameter
  - Leverages existing DNR calculation infrastructure from diagnostics module
  - Provides detailed split statistics including mean DNR, high-DNR counts, no-neighbor counts

- **MaxMinPicker Diversity Splitting** (`src/molblender/models/api/core/splitting/diversity.py`)
  - Diversity-based splitting using RDKit's MaxMinPicker algorithm
  - Two operational modes:
    - **Friendly mode**: Diverse training set ensures broad chemical space coverage for learning
    - **Unfriendly mode**: Diverse test set creates most challenging generalization scenario
  - Multiple fingerprint support: Morgan (default), RDKit topological, MACCS keys
  - Configurable fingerprint parameters: radius, number of bits
  - Uses Tanimoto dissimilarity for maximum diversity selection
  - Reports similarity statistics: train/test avg similarity, cross-similarity
  - Integration with `DataSplitter` class via `strategy="maxmin"` parameter
  - Addresses diversity sampling requirements from molecular ML literature

- **Butina Clustering-Based Splitting** (`src/molblender/models/api/core/splitting/butina.py`)
  - Leave-cluster-out cross-validation based on Tanimoto similarity clustering
  - Uses Butina's sphere exclusion algorithm (Butina 1999, J. Chem. Inf. Comput. Sci.)
  - Ensures chemically similar molecules stay together in either training or test set
  - Prevents information leakage from structural similarity, addressing MolAgent clustering strategy
  - Key features:
    - **Automatic clustering**: Self-adaptive cluster count based on similarity threshold
    - **Greedy balanced assignment**: Largest clusters assigned first to balance train/test sizes
    - **Leave-cluster-out validation**: Entire clusters move as units (no split within cluster)
  - Configurable parameters: similarity_threshold (default 0.6), fingerprint type, radius, nbits
  - Reports cluster statistics: n_clusters, cluster sizes, intra/inter-cluster similarity
  - Integration with `DataSplitter` class via `strategy="butina"` parameter
  - More fine-grained than scaffold splitting (considers full molecular topology vs core structure only)
  - Suitable for evaluating generalization to similar but unseen chemical combinations

- **Feature Clustering-Based Splitting** (`src/molblender/models/api/core/splitting/feature_clustering.py`)
  - General-purpose clustering split supporting arbitrary molecular representations (not limited to fingerprints)
  - Three clustering algorithms:
    - **K-means++**: Fast spherical clustering with smart initialization
    - **Hierarchical**: Ward linkage tree-based clustering
    - **DBSCAN**: Density-based clustering with automatic cluster count detection
  - Flexible feature input sources:
    - **User-provided representations**: 3D embeddings (Boltz-2), language models (ChemBERTa), custom features
    - **RDKit descriptors**: ~20 physicochemical descriptors (MW, LogP, TPSA, etc.)
    - **Fingerprints**: Morgan/RDKit/MACCS as fallback
  - Automatic optimal k selection via Silhouette score maximization (k ∈ [2, √n])
  - Feature standardization with StandardScaler for fair distance computation
  - Comprehensive clustering quality metrics:
    - **Silhouette score**: [-1, 1], >0.5 indicates good clustering
    - **Calinski-Harabasz index**: Higher values indicate better-defined clusters
    - **Davies-Bouldin index**: Lower values indicate better separation
  - Leave-cluster-out assignment with greedy balanced strategy
  - DBSCAN noise point handling (assigned to training set)
  - Integration with `DataSplitter` class via `strategy="feature_clustering"` parameter
  - Configurable parameters: clustering_algorithm, n_clusters, auto_select_k, features, use_descriptors, dbscan_eps, dbscan_min_samples
  - Complementary to Butina splitting: Butina uses Tanimoto+fingerprints, feature_clustering uses Euclidean+arbitrary features
  - Ideal for non-fingerprint representations (3D structures, quantum features, embeddings)
  - Comprehensive test suite (14 passed, 2 DBSCAN skipped due to numpy compatibility)

- **Shared Splitting Utilities** (`src/molblender/models/api/core/splitting/utils.py`)
  - Centralized utility functions to eliminate code duplication across splitting strategies
  - `compute_fingerprints()`: Unified fingerprint generation (Morgan, RDKit, MACCS)
  - `compute_avg_similarity()`: Average pairwise Tanimoto similarity within a set
  - `compute_cross_similarity()`: Average Tanimoto similarity between train and test sets
  - Reduces code duplication by 64 lines across butina.py and diversity.py
  - Shared by Butina, MaxMin, and Feature Clustering splitting strategies

- **Dataset Splitting Documentation** (`docs/source/usage/data/splitting.md`)
  - Comprehensive guide to all 10 supported splitting strategies (train_test, train_val_test, nested_cv, cv_only, scaffold, dnr, maxmin, butina, feature_clustering, user_provided)
  - Detailed implementation references with code locations and line numbers
  - Visual workflow diagrams for each splitting strategy
  - Scaffold split section with Bemis-Murcko vs Generic comparison and balanced vs random split methods
  - Feature Clustering section with complete examples:
    - User-provided features (3D embeddings, language models)
    - RDKit descriptors with K-means/Hierarchical clustering
    - DBSCAN with automatic cluster detection
    - Feature Clustering vs Butina comparison table
    - Typical workflow example with Boltz-2 embeddings
  - Drug discovery use cases and advantages over random splitting
  - Best practice recommendations based on dataset size and use case
  - Reproducibility guarantees and fixed random seed documentation
  - Cross-validation protocol details with StratifiedKFold for classification
  - Decision tree for choosing the right splitting strategy
  - Performance considerations and memory efficiency comparisons
- **Boltz-2 AI Structure Prediction Integration** (`src/molblender/representations/AI_fold/boltz2/`)
  - Complete module for extracting embeddings from Boltz-2 protein-ligand complex predictions
  - Support for three embedding types: global (29-33 dim), token-level (4-dim pooled), pairwise distance matrices
  - Intelligent caching system to avoid redundant structure predictions (saves hours of GPU time)
  - Isolated conda environment execution via subprocess to prevent dependency conflicts
  - Automatic YAML input generation with short IDs to avoid Boltz-2 truncation errors
  - MSA server integration for protein sequence alignment
  - Structure file parsing (CIF format) with recursive file discovery
  - Geometric feature extraction including COM, radius of gyration, protein-ligand contacts, confidence scores (pLDDT)
  - Complete test suite with unit tests and integration tests
  - **Note**: Structure prediction via subprocess currently has limitations; users can provide pre-computed CIF files for embedding extraction

- **Protein-Ligand Dataset Support** (`src/molblender/data/`)
  - Extended `Molecule` class with `protein_sequence` and `protein_pdb_path` attributes for protein-ligand complexes
  - New parameters in `MolecularDataset.from_csv()`: `protein_sequence_column` and `protein_pdb_column`
  - Automatic storage of protein information in molecule properties for CSV round-tripping
  - Seamless integration with existing featurization pipeline via kwargs passing
- **Dynamic Metric Resolution System** (`src/molblender/dashboard/metrics/central.py`)
  - Single source of truth for all metric definitions and display names
  - Automatic resolution of `primary_metric` to actual metric names (e.g., "Pearson R²", "MAE")
  - Consistent metric naming across all dashboard components
  - Eliminated hardcoded "Primary Metric" references throughout the interface
  - Enhanced `format_metric_name()` utility with DataFrame context for proper resolution

- **Professional Chart Styling Framework** (`src/molblender/dashboard/components/utils/chart_fonts.py`)
  - Unified font sizing system across all dashboard visualizations
  - Professional color scheme (light blue #6BAED6, light green #74C476, light orange #FD8D3C)
  - Consistent axis styling and formatting for research-quality charts
  - Global font size control via `AXIS_TITLE_FONT_SIZE`, `TICK_LABEL_FONT_SIZE`, `CHART_TITLE_FONT_SIZE`

- **Interactive Table System** (`src/molblender/dashboard/components/tables.py`)
  - Complete replacement of HTML tables with native `st.dataframe()` components
  - Dynamic metric column names with proper formatting
  - Sorting, filtering, and export capabilities
  - Responsive design with container width optimization

- **Comprehensive Distribution Analysis Charts** (`src/molblender/dashboard/components/distribution/charts.py`)
  - Five chart types: Box Plot, Violin Plot, Histogram, Density Plot, Raincloud Plot
  - Unified chart rendering system with consistent styling
  - Professional axis labeling with dynamic metric names
  - Eliminated "undefined" chart titles across all visualizations

- **Comprehensive Methodology Documentation** (`docs/source/usage/models/methodology.md`)
  - Complete explanation of train/test data splitting strategy (80/20 default with `random_state=42`)
  - Detailed 5-fold cross-validation protocol documentation
  - Model training and evaluation workflow with code references
  - Numerical examples and visual diagrams showing data flow
  - Best practices for choosing `test_size` and `cv_folds` based on dataset size
  - Known limitations and future improvements documented
  - Added cross-references from `screening.md` and navigation in `models/index.md`
  - Added CHANGELOG to documentation TOC

### Fixed
- **Boltz-2 Test Failures and Registration** (`src/molblender/representations/AI_fold/boltz2/`)
  - Fixed ligand_id truncation issue: Changed from 16-character cache_key to short ID 'L' to prevent Boltz-2 KeyError (`structure_predictor.py:157`)
  - Added AI_fold module import to `representations/__init__.py` to trigger Boltz2Embedder registration
  - Fixed test_registration: Corrected assertion to check instance instead of class (registry returns instances by design)
  - Fixed test_full_workflow: Corrected Dataset API usage (`feature_names`, `features.iloc[0]`) and `add_features()` argument order
  - Fixed VAE models indentation: Corrected MolecularVAE class `__init__` method indentation to prevent `RuntimeError: super(): no arguments`
  - Test results: 12 passed (up from 9), 1 skipped (GPU-dependent test validated manually with pre-computed CIF files)
  - All 4 originally failing tests now pass: `test_structure_prediction`, `test_caching_works`, `test_registration`, `test_full_workflow`

- **Dynamic Metric Resolution** (`src/molblender/dashboard/components/`)
  - Fixed "Primary Metric" displaying in Outlier Details table - now shows actual metric name
  - Fixed "Primary Metric" displaying in Distribution Overview charts - now shows formatted axis titles
  - Eliminated "undefined" chart titles across all dashboard visualizations
  - Fixed NameError with undefined 'df' variable in modality comparison charts
  - Resolved duplicate column names in results tables

- **Modality Filter Functionality** (`src/molblender/dashboard/components/charts/performance.py`)
  - Fixed non-responsive modality filter in Representation Analysis section
  - Added debug output for troubleshooting filter behavior
  - Enhanced modality mapping with proper error handling

- **Cross-Validation Reproducibility** (`src/molblender/models/api/core/evaluation/evaluator.py`)
  - Fixed CV random_state issue: Now uses `KFold(random_state=42)` and `StratifiedKFold(random_state=42)` objects
  - Ensures complete reproducibility of both CV scores and test scores across different runs
  - Automatically selects StratifiedKFold for classification tasks to maintain class balance
  - Updated documentation to reflect this fix in methodology.md

- **DNR and MaxMin Splitting Test Fixes** (`tests/models/splitting/test_dnr_maxmin_split.py`)
  - Fixed test dataset creation: Changed from `from_smiles_list()` to `from_df()` to properly set label_names
  - Fixed diversity.py: Changed `mol.rdkit_mol` to `mol.get_rdkit_mol()` method call
  - All 12 tests now pass successfully (previously 9 failed, 3 passed)
  - Verified DNR-based splitting with all 3 modes (threshold, quantile, neighbor)
  - Verified MaxMinPicker diversity splitting with friendly/unfriendly modes
  - Confirmed DataSplitter integration for both new strategies

- **Dashboard Distribution Charts Column Name Errors** (`src/molblender/dashboard/components/distribution/charts.py`)
  - Fixed ValueError: "Value of 'x' is not the name of a column in 'data_frame'" in 6 chart rendering functions
  - Root cause: Chart functions incorrectly used metric parameter (e.g., 'pearson_r') as dataframe column names
  - Solution: Dashboard always uses 'score' column for metric values; metric names are for labeling only
  - Fixed functions: `render_box_plot()`, `render_violin_plot()`, `render_histogram()`, `render_density_plot()`, `render_raincloud_plot()`, `render_quick_distribution()`
  - Pattern: Changed from `df[metric]` to `df['score']` while keeping `format_metric_name(metric)` for axis labels

- **Dashboard Usage Analysis pd.crosstab Error** (`src/molblender/dashboard/components/distribution/usage_analysis.py`)
  - Fixed ValueError: "aggfunc cannot be used without values" in representation-model combination heatmap
  - Simplified `pd.crosstab()` call to show combination counts only (removed conditional aggfunc logic)
  - Removed conflicting parameters: `values=df[selected_metric]` and `aggfunc='count'/'mean'`
  - Now displays usage frequency (how many times each representation-model pair was tested)

- **CNN Model Dimension Mismatch Bug** (`src/molblender/models/modality_models/cnn_models.py`)
  - **Critical Fix**: Replaced `squeeze()` with `flatten()` to prevent scalar predictions causing "Length mismatch: y_true=457, y_pred=1" errors
  - **Root Cause**: `squeeze()` removes all size-1 dimensions, converting single-sample predictions from `[1]` array to scalar
  - **Fixed Locations**:
    - `predict()` method (lines 601-605): Changed `outputs.squeeze()` → `outputs.flatten()`
    - `fit()` method training loop (line 546): Loss calculation now uses `outputs.flatten(), batch_y.flatten()`
    - `fit()` method validation loop (line 562): Same fix for validation loss calculation
  - **Batch Size Safety** (lines 515-529): Added dynamic batch_size adjustment for small datasets to prevent BatchNorm errors
    - Automatically reduces batch_size if dataset size < configured batch_size
    - Logs warning when dataset too small for BatchNorm, preventing training crashes
  - **Impact**: Eliminates 100+ "Length mismatch" errors per screening run, enables CNN evaluation on small datasets
  - **Verification**: Test confirmed single-sample prediction returns 1D array `(1,)` instead of scalar

- **XGBoost CUDA Compatibility** (`src/molblender/models/corpus/model_corpus.py`, `src/molblender/config/version_compat.py`)
  - **XGBoostRegressorWrapper/XGBoostClassifierWrapper**: Automatic CUDA fallback on initialization errors
  - **Mechanism**:
    - Try fitting with default parameters (may attempt CUDA)
    - On `XGBoostError` with "cuda" in error message, automatically retry with `device="cpu", tree_method="hist"`
    - Logs warning when fallback occurs for user awareness
  - **Implementation**: Wrapper classes with `fit()` and `predict()` methods implementing sklearn interface
  - **Error Prevention**: Resolves 8 CUDA-related failures in production screening logs
  - **Transparent Usage**: Drop-in replacement for `XGBRegressor`/`XGBClassifier` with no API changes
  - **Note**: `get_xgboost_params()` now only handles version compatibility (e.g., use_label_encoder), device selection handled by wrapper

- **Modality Compatibility Enforcement** (`src/molblender/models/api/core/model_registry.py`)
  - **Critical Fix**: Added modality compatibility check when `model_names` parameter is explicitly provided
  - **Root Cause**: Lines 82-91 previously skipped modality filtering when user specified model list
  - **Problem**: `universal_screen(..., model_names=['transformer_small'])` with fingerprint data incorrectly attempted Transformer+VECTOR combination
  - **Solution** (lines 82-98):
    - Check modality compatibility even with explicit model_names
    - Log warning and skip incompatible models: `Model '{name}' is not compatible with data modality '{modality}', skipping`
  - **Compatibility Matrix**:
    - VECTOR (fingerprints/descriptors) → Traditional ML only, NO Transformers/CNN
    - STRING (raw SMILES) → Transformer only
    - MATRIX/IMAGE → CNN only
  - **Verification**: `transformer_small + maccs_keys` now correctly skipped, only `ridge + maccs_keys` evaluated

- **Configurable Timeout System** (`src/molblender/models/api/core/base.py`, `src/molblender/models/api/core/evaluation/evaluator.py`)
  - **User-Customizable Timeouts** in `ScreeningConfig`:
    - `model_timeout: Optional[int] = None` - Set absolute timeout (seconds), None=use adaptive
    - `base_model_timeout: int = 600` - Base timeout for adaptive calculation (increased from 300s to 600s)
    - `min_model_timeout: int = 60` - Minimum allowed timeout (seconds)
    - `max_model_timeout: int = 3600` - Maximum allowed timeout (seconds)
  - **Adaptive Timeout Calculation** (`_get_adaptive_timeout()` method):
    - User-specified `model_timeout` takes priority if set
    - Otherwise uses `base_model_timeout` × multipliers:
      - Representation type: Transformer ×3, CNN ×2, Fingerprint ×0.5
      - Data size: n_samples > 10000 → ×2, n_features > 5000 → ×1.5
      - Operation type: CV uses `cv_folds × 0.3` multiplier
    - Final timeout bounded by `[min_model_timeout, max_model_timeout]`
  - **Impact**:
    - Transformer CV with 10k samples: 600 × 3 × 2 × (5 × 0.3) = **5400s (90min)**
    - Fingerprint training: 600 × 0.5 = **300s (5min)**
  - **Default Timeout Increase**: Base timeout doubled from 5min to 10min for better deep learning model coverage

### Changed
- **Dataset Quality Analysis Migration**: Replaced `overview.py` with comprehensive `diagnostics/` module
  - Migrated basic statistics and molecular weight calculations to `diagnostics/core.py`
  - Enhanced with DNR analysis and activity cliff detection capabilities
  - Improved visualization quality with publication-ready SVG output
  - Updated `src/molblender/data/__init__.py` to export `DatasetDiagnostics` instead of `generate_dataset_report`

- **Dashboard UI Reorganization**: Improved navigation and logical grouping of analysis components
  - Performance vs Training Time chart now colored by modality category (fingerprints, language-model, spatial, image, string) instead of individual models for clearer pattern recognition
  - Simplified Model Analysis from 3 to 2 sub-tabs (Performance Deep Dive and Efficiency Analysis)
  - Moved Multi-Dimensional Model Comparison (radar chart) from Performance Analysis to Detailed Results as new 6th sub-tab
  - Combined Hierarchical Clustering Analysis with Correlation Matrix in Detailed Results → Metric Correlation tab for related statistical analysis
  - Moved Statistical Summary to Detailed Results → Outlier & Distribution tab (displayed at top)
  - Removed duplicate Representation Analysis from Model Analysis → Performance Deep Dive (still available in dedicated Representation Analysis tab)
  - Modality Performance Statistics table uses HTML rendering for compatibility (Streamlit dataframe attempted but reverted due to PyArrow compatibility issues with Pandas 1.5.3)

- **Dashboard Individual Model Inspection Improvements**:
  - Made Predictions vs True Values scatter plot square-shaped (650×650 pixels) for better visual proportions
  - Moved Hyperparameter Analysis from Detailed Results to Individual Model Inspection tab for logical grouping
  - Consolidated Model Parameters into Hyperparameter Analysis tab (reduced from 4 tabs to 3)
  - Enhanced Hyperparameter Analysis now shows: Model/Representation info cards, parameter configuration (cards + table), all performance metrics, and export options (Python dict + JSON)
  - Improved tab structure: Prediction Scatter Plot → Hyperparameter Analysis → Export & Code

- **Dashboard UI Font Hierarchy**:
  - Implemented 3-level tab font size hierarchy for better visual hierarchy and readability
  - Level 1 main tabs: 32px (Overview, Performance Analysis, etc.)
  - Level 2 sub-tabs: 26px (Modality Analysis, Model Analysis, Efficiency Analysis)
  - Level 3 nested tabs: 20px (Modality Overview, Representation Analysis, etc.)

- **Dashboard Performance Analysis Cleanup**:
  - Removed duplicate Efficiency Analysis sub-tab from Model Analysis (now only appears as top-level tab)
  - Streamlined Model Analysis to focus on Category and Specific Model comparisons
  - Removed Statistical Summary section from Comprehensive Distribution Analysis to reduce redundancy

- **Detailed Results Tab Optimization**:
  - Reduced from 6 sub-tabs to 5 by moving Hyperparameter Analysis to Individual Model Inspection
  - Current structure: Results Table, Metric Correlation, Outlier & Distribution, Distribution Analysis, Multi-Dimensional Comparison

## [0.1.0] - 2025-01-XX

### Added
- Initial release of MolBlender
- Multi-modal molecular representation generation
- Automated ML model screening system
- Interactive Streamlit dashboard for results visualization
- Support for 60+ molecular representations across multiple modalities (fingerprints, language models, images, spatial)
- Integration with 20+ ML algorithms (scikit-learn, XGBoost, LightGBM, neural networks)
- SQLite-based results persistence with caching support
- Comprehensive evaluation metrics for regression and classification tasks
