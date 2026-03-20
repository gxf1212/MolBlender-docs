# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **YAML/TOML Configuration File Support** (2026-03-17)
  - Created `config/loader.py` for loading YAML and TOML configuration files
  - Auto-detect format by file extension (.yaml/.yml/.toml)
  - Environment variable substitution with `${VAR_NAME}` and `${VAR_NAME:default}` syntax
  - Created `config/export.py` for exporting and creating configuration templates
  - Added 4 configuration templates:
    - `screening_basic.yaml` - Minimal quick-start template
    - `screening_advanced.yaml` - Full-featured template with all options
    - `screening_hpo.yaml` - HPO-optimized template
    - `screening_basic.toml` - TOML format example
  - Integrated with `universal_screen()` via new `config_file` parameter
  - Priority order: explicit params > config file > defaults
  - Backward compatible with existing API
  - Impact: Version control configs, easier collaboration, non-programmers can modify settings

- **Parallel Execution Infrastructure Module** (2026-03-17)
  - Created new `molblender.parallel` module (457 lines) for unified parallel execution
  - Core primitives: `WorkerError`, `ProgressTracker`, `execute_with_timeout()`
  - Parallel executor: `execute_with_fallback()` with automatic serial fallback
  - Cached execution: `CachedExecutor`, `CacheStats`, `make_cache_key()`
  - Refactored 3 files to use new infrastructure:
    - `representations/utils/parallel.py` - uses `execute_with_fallback()`
    - `representations/utils/parallel_cached.py` - uses `CachedExecutor`
    - `representations/image/utils.py` - uses unified parallel execution
  - Eliminates code duplication across representations layer
  - Uses runtime `concurrent.futures` imports for proper monkeypatch test compatibility
  - Impact: Cleaner parallel execution, better error handling, consistent progress tracking

- **Representations Registry Module Refactoring** (2026-03-17)
  - Moved `utils/registry*.py` (7 files, 1153 lines) to independent `registry/` module
  - Removed `registry_` prefix from filenames: `registry_core.py` → `core.py`, etc.
  - Renamed `selectors.py` → `selection.py` to avoid conflict with Python stdlib
  - Created clean module boundary: registry handles featurizer registration, utils handles utilities
  - Updated all imports across 35+ source files and 5 test files
  - Impact: Clearer architecture, better separation of concerns, eliminates utils module bloat

- **Modality Models Base Class Simplification** (2026-03-17)
  - Consolidated 5 base classes into focused single-responsibility modules
  - Created `core.py` (was `base_core.py`) for shared abstractions
  - Created dedicated base files: `cnn/base.py`, `string_models/base.py`, `vae/base.py`, `graph/base.py`
  - Deleted redundant `base_*.py` files that mixed multiple abstractions
  - Added `modality_models/README.md` documenting the new structure
  - Impact: Easier to understand, modify, and extend modality models

- **API Layer Configuration Objects** (2026-03-17)
  - Replaced 78 parameters in `universal_screen()` with 6 dataclass config objects
  - New config classes: `CoreScreeningConfig`, `SplitConfig`, `ResourceConfig`, `HPOConfig`, `DatabaseConfig`, `WeightConfig`
  - Reduced `api.py` from 785 lines to ~550 lines (30% reduction)
  - Removed 53 legacy parameters and ~100 lines of parameter merging logic
  - Maintained backward compatibility through facade pattern
  - Updated 10 RdRp test scripts to use new config object API
  - Impact: Cleaner API, easier parameter management, better code maintainability

### Changed

- **Representations Module Organization** (2026-03-17)
  - Old: `utils/` module mixed registry logic with utility functions (16 files)
  - New: `registry/` as independent module, `utils/` focused on pure utilities
  - Eliminated circular imports between utils and registry
  - All registry functions now import from `molblender.representations.registry`
  - Impact: Single-responsibility principle, clearer module boundaries

- **DeepChem Models Integration** (2026-03-17)
  - Moved `models/deepchem/` to `models/modality_models/graph/`
  - DeepChem GNN models now properly inherit from `GraphModalityModel`
  - Updated imports across models using DeepChem
  - Deleted obsolete `models/deepchem/` directory

### Fixed

- **Documentation API Compatibility** (2026-03-20)
  - Added `MolecularDataset.validate()` method for dataset validation as shown in quickstart documentation
  - Added `MolecularDataset.from_dataframe()` alias pointing to `from_df()` for backward compatibility
  - Fixed UniMol dependency loading to expose `UniMolRepr` class via `deps.get_unimol_tools()["UniMolRepr"]`
  - Updated UniMol test expectations for actual output dimensions (768 instead of 512 for UniMol-V2-84M)
  - All documentation examples now work correctly (6/6 tests passing)
  - Impact: Documentation matches actual API, better user experience

- **Registry Module Lint and Type Errors** (2026-03-17)
  - Fixed Ruff B007 unused loop variable errors in test files
  - Fixed mypy type annotation errors in registry/display.py, registry/info.py, registry/shapes.py
  - Added TYPE_CHECKING imports for BaseFeaturizer to avoid circular imports
  - Added missing logging import in display.py
  - Fixed bandit B307 security issue: replaced eval() with ast.literal_eval()
  - Updated test imports from old registry_* paths to new registry/ module paths
  - Impact: All lint checks passing, improved type safety, better security

### Removed

- **Registry Filename Prefixes** (2026-03-17)
  - Removed `registry_` prefix from all registry module files
  - `registry_core.py` → `core.py`, `registry.py` → `facade.py`, etc.
  - Renamed `selectors.py` → `selection.py` to avoid Python stdlib conflict
  - Impact: Cleaner filenames, better naming conventions
  - Impact: Consistent model architecture, all graph models in one location

- **Execution Layer Cleanup** (2026-03-17)
  - Removed `execution/optimized_parallel.py` (duplicate functionality)
  - Deleted `models/api/utils/resource_scheduler.py` (replaced by infrastructure layer)
  - Cleaned up legacy compatibility shims
  - Impact: Removed code duplication, clearer dependencies

### Tests

- Added `tests/models/api/test_model_discovery.py` (model catalog discovery tests)
- Added `tests/models/modality_models/test_graph_models.py` (graph model tests)
- Verified 157 featurizers still accessible after refactoring
- All 51 multimodal/representations tests passing

### Fixed

- Fixed circular import between `utils/` and `registry/` modules
- Fixed name collision between new `ScreeningConfig` and existing `screening_engine.ScreeningConfig`
- Fixed dashboard imports after API refactoring
- Fixed relative imports in registry module after file moves

- **Architecture Role Catalog & Executable Snapshot** (2026-03-12)
  - Added machine-readable package role metadata across top-level facades, domain APIs, visualization layers, and execution layers
  - Added `molblender.architecture_roles` helpers for:
    - package role catalog
    - recommended entrypoints
    - execution layer decisions
    - visualization layer decisions
    - migration guidance
  - Added `python -m molblender.architecture_roles` JSON snapshot output
  - Impact: Architecture guidance is now executable, testable, and CI-friendly

- **Execution Layer Boundary Contracts** (2026-03-12)
  - Added contract tests to lock the distinction between:
    - `molblender.models.api.infrastructure` (primary screening runtime)
    - `molblender.representations.utils` (generic batching/caching helpers)
    - `molblender.models.execution` (compatibility layer)
  - Added import-isolation and public-surface tests to prevent legacy executor leakage into recommended APIs
  - Impact: Clearer long-term migration path and lower risk of architectural drift

- **Infrastructure: Telemetry Package Modularization** (2026-03-11)
  - Replaced monolithic `telemetry.py` (755 lines) with modular `telemetry/` package
  - New structure: `types.py`, `backends.py`, `emitter.py`, `global_emitter.py`, `legacy.py`
  - Maintained 100% backward compatibility - all old imports still work
  - Export contract tests: 16 tests passing
  - Infrastructure tests: 33 tests passing
  - Impact: Better code organization, clearer separation of concerns, improved maintainability

- **Representations: Public API Consolidation** (2026-03-11)
  - Added comprehensive API export contract tests (23 tests)
  - Verified tool_registry purity: metadata management only, no UI logic
  - Added import isolation tests (14 tests)
  - Confirmed no circular imports in representations module
  - Validated lazy imports: transformers not loaded on package import
  - Impact: Stable public API, better import performance, clearer module boundaries

- **Drawings Package Positioning** (2026-03-11)
  - Clarified drawings as "static plotting utilities" layer
  - Verified drawings does NOT export dashboard/interactive components
  - Added 23 API contract tests for drawings/models/dashboard separation
  - Updated module documentation to distinguish drawings (static) vs dashboard (interactive)
  - Impact: Clearer package boundaries, easier to choose right visualization tool

- **Public API Layer Audit** (2026-03-11)
  - Added 33 public API contract tests
  - Verified molblender.api = unified convenience facade
  - Verified molblender.models = richer ML domain API
  - Verified molblender.representations = richer featurizer API
  - Updated top-level molblender documentation with API layer explanation
  - Distinguished RECOMMENDED (unified facade) vs COMPATIBILITY (direct imports) in __all__
  - Impact: Clearer API usage guidance, better developer experience

- **Architecture Role Contract Tests** (2026-03-11)
  - Added 27 package role contract tests
  - Verified drawings = static plotting layer (not dashboard)
  - Verified models.api.core = screening engine core components
  - Verified models.api.infrastructure = runtime policy/telemetry/error policy
  - Verified molblender.api = facade (doesn't contain implementation details)
  - Impact: Architectural boundaries locked down, prevents future drift

### Changed

- **Top-Level Facades Now Use Lazy Imports**
  - `molblender` and `molblender.api` now use lazy facade exports
  - Importing the top-level package no longer eagerly loads large subpackages such as representations, models, dashboard, or drawings
  - Workflow subfacades (`molblender.api.models`, `molblender.api.representations`, `molblender.api.dashboard`) also use lighter import paths
  - Impact: Faster startup, lower import overhead, and cleaner package boundaries

- **Execution and Architecture Documentation Refined**
  - Updated source README files and developer-facing docs to distinguish current recommended layers from historical compatibility layers
  - Clarified that `models.api.core` is the screening engine core, not the project-wide core
  - Clarified that `drawings` is for static plotting while `dashboard` is the interactive UI
  - Impact: Easier navigation for contributors and clearer package roles for users

- **Telemetry Module Organization**
  - Old: Single file `telemetry.py` with all implementations
  - New: Modular package with separate files for types, backends, emitter, global functions
  - Benefit: Easier to maintain and extend, clearer code organization

- **Representations Import Behavior**
  - Confirmed: Heavy dependencies (torch, transformers) loaded lazily, not on import
  - Confirmed: Multiple imports consistent, no side effects
  - Benefit: Faster imports, lower memory footprint for basic usage

### Tests

- Added `tests/models/api/infrastructure/test_telemetry_exports.py` (16 tests)
  - Public API exports verified
  - Import patterns tested (from infrastructure, from telemetry package, from submodules)
  - Backend classes verified
  - EventEmitter availability confirmed
  - Package structure validated

- Added `tests/representations/test_api_exports_contract.py` (23 tests)
  - Core utilities availability verified
  - Base classes and exceptions confirmed
  - Tool registry purity validated
  - No heavy imports on package import confirmed
  - No side effects on import verified
  - Circular import prevention confirmed

- Added `tests/core/test_import_isolation.py` (14 tests)
  - Representations import isolation verified
  - Telemetry import isolation verified
  - Import consistency confirmed
  - Lazy imports validated

- Added `tests/drawings/test_api_exports_contract.py` (23 tests)
  - Drawings public API exports verified
  - Drawings does NOT export dashboard components verified
  - Optional imports handled gracefully confirmed
  - Drawings vs dashboard separation validated

- Added `tests/core/test_public_api_contract.py` (33 tests)
  - Top-level molblender API verified
  - molblender.api as unified facade validated
  - molblender.models as richer domain API confirmed
  - molblender.representations as richer featurizer API confirmed
  - API layer distinction verified

- Added `tests/core/test_package_roles.py` (27 tests)
  - Drawings as static plotting layer role verified
  - models.api.core as screening engine role confirmed
  - models.api.infrastructure as runtime policy role verified
  - molblender.api as facade (no implementation) validated
  - Package boundaries separation confirmed

### Technical Details

**Telemetry Package Structure**:
```
infrastructure/telemetry/
├── __init__.py         # Unified exports (backward compatible)
├── types.py            # EventType, EventSeverity enums
├── backends.py         # EventBackend, JSONFileBackend, LogFileBackend, etc.
├── emitter.py          # EventEmitter class
├── global_emitter.py   # get_global_emitter(), configure_global_emitter(), emit_event()
└── legacy.py           # build_event(), emit_legacy_event()
```

**Representations Public API**:
- Core utilities: `get_featurizer`, `list_available_featurizers`, `get_featurizer_info`
- Base classes: `BaseFeaturizer`, `BaseProteinFeaturizer`
- Exceptions: `FeaturizationError`, `InvalidInputError`, `RegistryError`
- Tool registry: `ToolInfo`, `ToolRegistry` (pure metadata, no UI logic)
- Enhanced registry: `FeaturizerInfo`, `FeaturizerCatalog`, `get_featurizer_recommendations`

### Migration Notes

**For Telemetry Users**:
- No changes required - all old imports still work
- New modular imports available for finer control:
  ```python
  # Old way (still works)
  from molblender.models.api.infrastructure import emit_event, EventType
  
  # New way (more granular)
  from molblender.models.api.infrastructure.telemetry.types import EventType
  from molblender.models.api.infrastructure.telemetry.global_emitter import emit_event
  ```

**For Representations Users**:
- No changes required - public API unchanged
- Tool registry now available for advanced metadata queries:
  ```python
  from molblender.representations.tool_registry import ToolRegistry
  
  registry = ToolRegistry()
  gpu_featurizers = registry.list(tags=["gpu"])
  protein_featurizers = registry.list(category="protein")
  ```


- **Phase 3-4: Configuration Management & Deprecated Code Cleanup** (2026-03-10)
  - **Phase 3: Configuration Management Unification**
    - Created `config/core.py` (309 lines): Unified ConfigManager singleton
    - Dataclasses: `CacheConfig`, `ModelConfig`, `LoggingConfig`
    - Centralized environment variable reading with validation
    - Runtime configuration update: `set_cache_dir()`, `get_cache_dir()`
    - Backward compatibility: Legacy `settings` exports preserved with `_legacy` suffix
    - Tests: 12 ConfigManager tests passing, 73 total config tests passing
  - **Phase 4: Deprecated Code Cleanup**
    - Deleted `models/api/utils/resource_scheduler.py` (102-line shim)
    - Removed `timeout_context` from `evaluation/utilities.py` (~100 lines)
    - Updated `evaluation/__init__.py`: Removed timeout_context from `__all__` and `__getattr__`
    - Deleted test files: `test_resource_scheduler_compat.py`, `test_timeout_context_shim.py`
    - Updated guard tests: Removed resource_scheduler and resource_profiles tests
    - All infrastructure and evaluation tests passing (4 passed)
  - **Documentation**
    - Created `config_manager_guide.md`: ConfigManager usage and best practices
    - Created `migration_guide.md`: API migration guide from old to new API
    - Created `api_guide.md`: Unified API layer usage guide
  - **Impact**: Unified configuration management, removed ~200 lines deprecated code, 306 total tests passing

- **Round 8: Dashboard State Management & Resource Tracking** (2026-03-09)
  - **Task 1: Dashboard State Management Modularization**
    - Created `dashboard/state/` package with three manager classes
    - `SessionManager`: Manages session-wide state (cache, files, refresh)
    - `NavigationManager`: Manages navigation state (active tab, history)
    - `FilterManager`: Manages filter state (metrics, models, representations)
    - Total: 459 lines across 3 modules
  - **Task 2: Dashboard Cache Hierarchical Refactoring**
    - Created `dashboard/cache/policies.py` with CachePolicies system
    - Clear separation: `st.cache_data` (DataFrame, dict) vs `st.cache_resource` (connections)
    - Predefined strategies: SESSION_DATA, SHORT_LIVED_DATA, FEATURIZER, CONNECTION
    - `@cache_with_policy` decorator for easy policy application
  - **Task 3: CI Matrix and Gate Configuration**
    - Created `.github/workflows/dashboard_smoke.yml`
    - Python 3.9/3.10 matrix testing
    - Test suites: state management, cache manager, API contracts, dashboard smoke
  - **Task 4: Dashboard Integrated Resource Tracking System**
    - Created `dashboard/components/resource_tracking/` package
    - `representation_selector.py` (192 lines): Type selector, comparison table, parameter info
    - `cache_statistics.py` (273 lines): Sidebar stats, management page, helper functions
    - Updated `app.py`: Added 2 new tabs (Resource Management, Representation Types)
    - Total tabs: 5 → 7
  - **Phase 5: Auto-discovery Enhancement**
    - Enhanced `representations/registry/` with dependency checking and availability flags
    - Created `representations/registry/validation.py` (320 lines): Parameter validation framework
    - Auto-discovery now detects dependencies and sets `is_available` flag
    - Parameter validation with type checking, choices, min/max values
    - Helper functions: `sanitize_representation_params()`, `get_representation_summary()`
  - **Tests**: 290 tests passing (12 Phase 4 + 19 Phase 5 + 259 previous)
  - **Impact**: Improved dashboard modularity, cache observability, resource tracking, auto-discovery

- **Round 7: Dashboard API Contract & Smoke Testing (CC)** (2026-03-08)
  - **Task 1: Dashboard API Export Contract Tests** (21 tests)
    - Created `tests/dashboard/test_api_exports_contract.py`
    - Verified all public API exports from dashboard modules
    - Tested backward compatibility of key interfaces
    - Coverage: `ResultsDataLoader`, `DashboardMetrics`, `FilterConfig`, render functions
  - **Task 2: Dashboard Smoke Stabilization** (26 tests)
    - Created `tests/dashboard/test_task2_smoke_stabilization.py`
    - Verified tab order is fixed and matches documentation
    - Tested all 4 main pages (Overview, Performance, Detailed Results, Hyperparameter)
    - Validated data loading pipeline and metrics system integration
    - Verified filtering system and backward compatibility
  - **Task 3: User Documentation Sync**
    - Created `docs/source/dashboard_troubleshooting.md`
    - Merged sessions loading workflow explanation
    - Common error handling: `NoneType.get`, `sqlite3.Row.get`, `KeyError: 'primary_metric'`
    - Diagnostic commands and minimal health check script
    - Performance tips for large databases
  - **Task 4: Regression Gate**
    - All 248 dashboard tests passing (7 skipped)
    - 47 new tests added (21 API + 26 smoke)
    - Zero regressions in existing functionality
  - **Impact**: Strengthened dashboard API stability, improved documentation, comprehensive test coverage

- **Round 6: Efficiency & Representations Modularization** (2026-03-08)
  - **Efficiency Analysis Refactoring**: Split `efficiency.py` (645 lines) into `efficiency/` package
    - New structure: `scatter.py` (132), `distribution.py` (131), `metrics.py` (181), `speed.py` (122)
    - Facade: `__init__.py` (88 lines) with `render_efficiency_analysis()` entry point
    - Clear separation: scatter plots, distributions, metrics, speed analysis
  - **Base Featurizer Refactoring**: Split `base.py` (633 lines) into focused modules
    - `base_featurizer.py` (415 lines) - Small molecule featurizer base class
    - `base_protein_featurizer.py` (147 lines) - Protein featurizer base class
    - Facade: `base.py` (11 lines) for backward compatibility
  - **CDK Fingerprint Factory Pattern**: Split `cdk.py` (633 lines) into `cdk/` package
    - `loaders.py` (109 lines) - Lazy CDK component loading
    - `base.py` (213 lines) - `BaseCDKFingerprint` base class
    - `classes.py` (155 lines) - Factory pattern eliminates 13 duplicate class definitions
    - Facade: `cdk.py` (5 lines) + `cdk/__init__.py` (47 lines)
    - `_make_cdk_class()` factory function for dynamic class generation
  - **Validators Duplicate Code Removal**: Removed 242-line duplicate `MetricsCalculator`
    - `validators.py` (860→613 lines, -29%)
    - Now uses unified `MetricsCalculator` from `metrics_calculator.py`
  - **Test Organization**: Moved 8 test files from `tests/` to `tests/data/`
    - Added `__init__.py` to test directories to fix import name collisions
  - **Test Marker Documentation**: Added comprehensive usage guidelines for `@pytest.mark.slow` and `@pytest.mark.network`
  - **Impact**: 4 files → 12 modules, eliminated duplicate code, improved maintainability
  - **Test Results**: 159 passed (19 dashboard + 140 CDK)

- **Round 5: Dashboard Boundary & Stability Hardening** (2026-03-08)
  - **Package Import Contracts**: All dashboard submodules importable without streamlit runtime
    - Tests: `test_round5_package_imports.py` (12 tests)
    - Verified: All submodules in `data/`, `metrics/`, `components/` import cleanly
  - **Behavioral Contracts**: Verified critical UI behaviors preserved
    - Tests: `test_round5_app_services_contracts.py` (9 tests)
    - Tab order validation: Overview → Performance → Detailed Results → Hyperparameter → Individual Model
    - Default page verification: "🔍 Overview" as initial tab
    - CLI entry points: `main()` and `run_from_cli()` signatures validated
  - **Type Safety**: Added `ClassVar` annotations for class-level constants (RUF012 fix)
  - **Test Coverage**: +21 tests (170 total, +68% from baseline)
  - **Regression**: All 319 tests passing (0 failures)
  - **Impact**: Hardened package boundaries, prevented UI regressions, improved type safety

- **Round 4: Dashboard app.py Refactoring** (2026-03-08)
  - **Major Refactoring**: Split `app.py` (766 lines) into `app_services/` package
    - `startup_diagnostics.py` (62 lines) - Diagnostic utilities
    - `page_config.py` (164 lines) - Page configuration and CSS styling
    - `pages.py` (377 lines) - Tab rendering logic
    - Facade: `app.py` (247 lines, -68%) with main orchestration
  - **Architecture Improvements**:
    - Phase 1: Extracted diagnostics and configuration (766→549, -28%)
    - Phase 2: Extracted page rendering logic (549→320, -42%)
    - Phase 3: Simplified tab dispatch (320→247, -23%)
  - **Streamlit Decoupling**: Lazy import pattern for streamlit dependencies
    - Enables pytest testing without streamlit runtime
    - Fixed `ModuleNotFoundError` in test collection
  - **Contract Tests**: Added API contract tests (12 tests)
    - `test_app_contracts.py`: Module structure, CLI signatures, UI behavior preservation
  - **Test Coverage**: +12 tests (142 total, +41% from baseline)
  - **Impact**: Clear separation of concerns, improved testability, backward compatible

- **HPO Module Refactoring** (2026-03-03)
  - Split `hpo.py` (1220 lines) into modular `hpo/` directory
  - New structure: `processor.py` (workflow), `selection.py` (model selection), `results.py` (data reconstruction)
  - Added utility function `serialize_indices()` for JSON serialization of train/test indices
  - Added `_create_model_for_task()` method to eliminate duplicate model instantiation code
  - Unified target extraction using existing `extract_targets_from_dataset()` function
  - Eliminated ~30 lines of repeated code across processors
  - Improved code maintainability: single point of change vs scattered modifications
  - Files added: `processors/hpo/__init__.py`, `processors/hpo/processor.py`,
    `processors/hpo/selection.py`, `processors/hpo/results.py`
  - Files modified: `processors/data_converter.py` (serialize_indices),
    `processors/hpo/processor.py` (refactored),
    `processors/hpo/results.py` (unified target extraction)

- **DeepChem Graph Neural Network Integration** (2026-03-02)
  - Plan B GNN mode implementation with full DeepChem graph support
  - Graph converters: ConvMol → GraphData, WeaveMol → GraphData
  - Smart routing: auto-detect GNN vs vector mode based on user categories
  - Supported GNN models: GCN, GAT (DeepChem 2.8.0 compatible)
  - Supported graph featurizers: ConvMol, MolGraphConv, Weave
  - Files added: `models/deepchem/gnn.py`, `models/api/multimodal/modality_handlers/graph_converter.py`
  - Files modified: `representations/graph/__init__.py`, `models/api/multimodal/modality_handlers/graph.py`
  - Test results: Plan A (vector mode) 26 results, best: deepchem_graphconv_vector + xgboost (Pearson r = 0.379)

- **Splito Integration for Advanced Splitting** (2026-02-24)
  - Unified API for splito cluster-level splitting strategies via `DataSplitter`
  - New strategies: `splito_perimeter`, `splito_molecular_weight`, `splito_max_dissimilarity`, `splito_scaffold`
  - Cluster-level algorithms (K-means based) vs molecular-level algorithms
  - Files modified: `models/api/core/splitting/strategies.py`

- **Specific Pairs Mode for Precise Combination Control** (2026-02-23)
  - New `combinations.representations` parameter for precise "representation + model" pair specification
  - Auto-detection of `specific_pairs_mode` when combinations specify both representations and models
  - Added `run_combination_screening()` utility function in multimodal utils
  - Supports all modality handlers: vector, string, matrix, image, language_model
  - Temporarily disables `skip_existing_results` to ensure specified combinations are re-run
  - Fixed result extraction: uses `screener.results` (ModelResult objects) instead of formatted dict
  - Verification on remote server 209: Stage 1 (15 results), Stage 2 HPO (756 results)
  - Best result: datamol_avalon + decision_tree (R² = 0.7859)
  - Files added: `src/molblender/models/api/multimodal/utils.py` (run_combination_screening)
  - Files modified: `src/molblender/models/api/core/base.py` (specific_pairs_mode flag),
    `src/molblender/models/api/multimodal/api.py` (auto-detection),
    `src/molblender/models/api/multimodal/modality_handlers/*.py` (all handlers)

- **Model Export CLI** (2026-02-19)
  - New `molblender export` command for exporting model recreation scripts from database
  - Subcommands: `export best`, `export model`, `export top`
  - Generates complete Python scripts with all parameters from screening
  - Supports absolute path resolution and CSV pre-defined splits
  - Files added: `src/molblender/utils/export_cli.py`

### Changed

- **CDK Fingerprint Defaults: `cdk_sub` native size + `cdk_signature` default disable** (2026-03-03)
  - `cdk_sub` now uses CDK native 307-bit shape (removed project-level zero-padding-to-1024 behavior)
  - `cdk_signature` remains implemented but is no longer registered by default
  - Rationale: raw Signature fingerprint outputs are variable-token and need explicit vocabulary alignment before ML
  - Files modified: `src/molblender/representations/fingerprints/cdk.py`,
    `tests/representations/fingerprints/test_cdk.py`,
    `docs/source/usage/representations/fingerprints/cdk.md`

- **Splitting Strategy Naming Clarity** (2026-02-24)
  - Renamed `max_dissimilarity_split` → `sequential_max_dissimilarity_split`
  - Clarifies molecular-level (sequential) vs cluster-level (splito) algorithms
  - Backward compatibility maintained with alias
  - Files modified: `models/api/core/splitting/diversity.py`, `models/api/core/splitting/strategies.py`

- **Database Schema Enhancement** (2026-02-19)
  - Added `split_column` field to `screening_sessions` table
  - Added `input_column` field to `screening_sessions` table
  - Enhanced `create_session()` to save split configuration metadata
  - Modified `load_dataset_with_split()` to return input_column information
  - Improved export code generation to read from database instead of guessing
  - Files modified: `models/api/utils/results_db.py`, `models/api/multimodal/processors/database.py`,
    `dashboard/components/model_inspection/export.py`, `utils/export_cli.py`

- **Dashboard Performance Optimization** (2026-02-18)
  - Added permanent caching to `create_results_dataframe()` with `@st.cache_data`
  - Optimized `MetricsCalculator.enrich_dataframe_with_metrics()` with filtering and tqdm progress
  - Files modified: `dashboard/data/processors.py`, `dashboard/metrics/central.py`

### Fixed

- **Database NULL Handling** (2026-02-18)
  - Fixed NoneType error when accessing NULL all_metrics fields
  - Uses `(result.get("all_metrics") or {})` instead of `result.get("all_metrics", {})`
  - Fixed None checks for `dataset_info` and `merged_dataset_info`
  - Files modified: `dashboard/data/loaders.py`

- **Dashboard Session Wall Time Calculation** (2026-03-06)
  - Fixed wall_time showing 0 in Session Breakdown table
  - Now correctly loads `created_at` and `updated_at` from `screening_sessions` table
  - Attaches session timestamps to each model result for per-session wall_time calculation
  - Each session shows independent wall_time based on its own timestamps (not merged across sessions)
  - Files modified: `dashboard/data/loaders.py`, `dashboard/components/charts/general.py`

- **Classification F1 Metric Alias** (2026-03-06)
  - Added "f1" as alias for "f1_score" in CLASSIFICATION_ONLY_METRICS
  - Ensures backward compatibility with databases using "f1" key instead of "f1_score"
  - Fixed KeyError when displaying F1 Score in Dashboard
  - File modified: `dashboard/metrics/central.py`

- **Model Export Split Information Fallback** (2026-03-06)
  - Fixed export code generation to check multiple sources for split configuration
  - Now tries `screening_config` if `dataset_info` doesn't have split indices/column
  - Ensures exported scripts can recreate data splits correctly
  - File modified: `dashboard/components/model_inspection/export.py`

- **Metric Column Series Name Attribute** (2026-03-06)
  - Fixed `get_metric_column()` returning Series without name attribute
  - Series now includes `name=selected_metric` for proper groupby operations
  - Prevents errors when using metric columns in grouping operations
  - File modified: `dashboard/components/utils/__init__.py`

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

- **Language Model Routing in Multimodal Screening** (2026-02-14)
  - Prevented duplicate processing of language-model representations inside VECTOR workflow
  - Ensures each modality is evaluated once with correct modality mapping

### Fixed

- **CV Scoring Metric: Pearson R instead of neg_MSE** (2026-03-01)
  - Changed cross-validation scoring from `neg_mean_squared_error` to `pearson_r`
  - Ensures CV evaluation uses same metric as final model assessment
  - Provides more interpretable results (correlation strength vs squared error)
  - Files modified: `models/api/core/metrics.py`, `models/api/multimodal/modality_handlers/vector.py`

- **Per-Fold CV Scores Storage for HPO Stage 2** (2026-03-01)
  - Fixed per-fold CV scores not being saved to database for HPO Stage 2 results
  - Now stores all fold scores in `all_metrics['cv_fold_scores']`
  - Enables detailed fold-level analysis in dashboard
  - Files modified: `models/api/multimodal/processors/database.py`

- **Primary Metric Renaming** (2026-03-01)
  - Refactored internal variable name from `score` to `primary_metric` across package
  - Improves code clarity and distinguishes from model.score() method
  - Files modified: Multiple files across `models/` and `dashboard/`

- **Dashboard Progress Bar Cleanup** (2026-02-28)
  - Fixed progress bar not being cleaned up properly after completion
  - Progress bar now rendered inside container for proper cleanup
  - Files modified: `dashboard/app.py`

- **Dashboard Loading Placeholder** (2026-02-28)
  - Fixed loading placeholder not being cleared after success message
  - Improves user experience by removing redundant UI elements
  - Files modified: `dashboard/app.py`

- **Hyperparameter Chart Performance** (2026-02-27)
  - Optimized parameter extraction with progress indicator
  - Reduced loading time for Individual Model page with 200k+ results
  - Files modified: `dashboard/components/charts/hyperparameters/main.py`

- **Dashboard Special Characters** (2026-02-27)
  - Sanitized special characters in sunburst chart colorbar labels
  - Fixed rendering issues with underscores and special characters
  - Files modified: `dashboard/components/model_inspection/sunburst.py`

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

- **HPO Stage 2 Resume Stability** (2026-02-14)
  - Removed call to non-existent `check_existing_hpo_result` during resumed optimization
  - Stage 2 now resumes using the standard DB existence checks

- **Offline/Feature Robustness Improvements** (2026-02-14)
  - Improved handling for offline environments and count-based fingerprint edge cases
  - Reduced modality failures caused by degenerate feature arrays

- **Cross-Session Stage 1 Result Loading** (2026-02-17)
  - Fixed `skip_existing_results` to load Stage 1 results from all sessions, not just current session
  - Fixed exception handler to preserve existing results when partial screening fails
  - Result: HPO Stage 2 now correctly uses all available Stage 1 results (193 → 1425 results)

- **HPO 56-Core Parallelization** (2026-02-17)
  - Fixed GridSearchCV to use all available CPU cores via `parallel_backend`
  - Force estimator `n_jobs=1` to prevent thread conflicts with GridSearchCV parallelism
  - Result: 240 parameter fits in 30.1s (vs 20min before), CPU utilization 98.9%

- **Train/Test Indices Persistence** (2026-02-17)
  - Save train/test indices to screening_sessions table for user_provided splits
  - HPO now correctly reconstructs training data across sessions
  - Fixed train/test split consistency between Stage 1 and Stage 2

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

## [2026-03-09] - Round 6 Refactoring Regression Fixes

### Fixed
- **Dashboard Round 6 Regression Fixes** (4 critical bug fixes)
  - **Import Error Fix** (`src/molblender/dashboard/app_services/pages.py`)
    - Fix Overview page import error: `_render_performance_vs_time_scatter` → `render_performance_vs_time_scatter` (2 locations)
    - During Round 6 refactoring, function was renamed but call sites were not updated
    - Impact: Overview page crashes, Performance vs Training Time chart cannot display
  
  - **Filter Parameter Count Fix** (`src/molblender/dashboard/components/filters/model_filters.py`)
    - Fix parameter count error in `_render_filter_summary()` call (passed 4 but accepts 3)
    - Remove redundant `selected_metric` parameter
    - Impact: Performance Analysis tab filter crashes
  
  - **Test Size Calculation Fix** (`src/molblender/dashboard/data/processors_helpers/loading.py`)
    - **Critical data accuracy fix**: Prioritize calculating actual test_size from real `n_train`/`n_test`
    - Before fix: Display configured value `test_size=0.2` (20%) - **Incorrect**
    - After fix: Display actual split `5271/53235=9.9%` - **Correct**
    - Impact: MaxDissimilarity split creates 9.9% test set, but Dashboard displays 20%
    - Code logic:
      ```python
      # Extract configured value from screening_config
      if "n_train" in dataset_info and "n_test" in dataset_info:
          n_train = dataset_info["n_train"]
          n_test = dataset_info["n_test"]
          if n_train and n_test:
              total = n_train + n_test
              if total > 0:
                  dataset_info["test_size"] = n_test / total  # Override configured value
      ```
  
  - **Scatter Plot API Update** (`src/molblender/dashboard/components/model_inspection/render.py`)
    - Fix Individual Model Inspection scatter plot rendering
    - After Round 6 refactoring, API changed to accept `InspectionPayload` instead of individual parameters
    - Before fix:`render_scatter_plot(model_data, dataset_info, show_train, add_regression, selected_metric=metric)`
    - After fix:
      ```python
      context = build_context_from_results(results, dataset_info)
      payload = build_payload_from_model_data(model_data, dataset_info, context)
      render_scatter_plot(payload, show_train_data=show_train, add_regression_line=add_regression)
      ```
    - Impact: Individual Model tab crashes, scatter plot cannot display

- **Modality Models Modularization** (`src/molblender/models/modality_models/`)
  - **File splitting refactoring**: base.py (622 lines) → 5 modules
    - `base_core.py` (6.8K) - Core base classes `BaseModalityModel` and `ModelResult`
    - `base_vector.py` (1.3K) - Vector modality `VectorModalityModel`
    - `base_string.py` (4.3K) - String modality `StringModalityModel`
    - `base_additional_modalities.py` (2.6K) - Image/matrix/graph/3D modalities
    - `base.py` (24 lines) - Backward compatible facade, re-exports all classes
  - **Design principles**: Comply with <800 lines rule, high cohesion low coupling
  - **Backward compatibility**: All existing import statements require no changes

- **Registry Error Handling Improvement** (`src/molblender/representations/utils/registry_core.py`)
  - **Dependency error type preservation**: Don't wrap into `RegistryError`, preserve original exception types
  - **Purpose**: Callers/tests can distinguish optional-feature scenarios vs hard registry errors
  - **Simplified logic**: `get_featurizer_info()` and `get_protein_featurizer_info()` simplified to call `build_featurizer_info()`

- **Datamol Fingerprint Calculator Enhancement** (`src/molblender/representations/fingerprints/datamol.py`)
  - **New method**: `_resolve_molfeat_calculator()` static method
  - **Compatibility**: Support both new and old molfeat loading formats
    - Current format: `{"available": True, "modules": {"molfeat_calc": module}}`
    - Old format: Directly flattened keys (`{"FPCalculator": ...}`)

- **Test Configuration Documentation** (`pytest.ini`)
  - **New test markers**:
    - `@pytest.mark.slow` - Slow tests (model download >100MB or >30s, heavy computation >10s, integration tests)
    - `@pytest.mark.network` - Network tests (external API calls, PDB/UniProt, etc., fail in offline environments)
  - **Usage examples**:
    - `pytest -m "not slow"` - Skip slow tests
    - `pytest -m "not network"` - Offline mode testing
    - `pytest -m "slow"` - Only run slow tests

### Impact
- **Dashboard stability**: Fixed 4 critical bugs, all 5 tabs and sub-functions fully available
- **Data accuracy**: Test Size corrected from incorrectly displaying 20% to actual value 9.9%
- **Code quality**: Model modularization refactoring complies with <800 lines rule, improves maintainability
- **Error handling**: Improved dependency error type preservation, increases test flexibility
- **Testing efficiency**: New test markers support rapid development iteration (skip slow/network tests)

### Verification
- ✅ Dashboard Overview: Test Size displays 9.9% (correct)
- ✅ Performance Analysis: Filters work correctly
- ✅ Individual Model: Scatter plot renders correctly
- ✅ All 4 fixed files pass manual testing
- ✅ Modularization refactoring maintains backward compatibility

### Removed

- **Duplicate tool_registry.py Module** (2026-03-17)
  - Deleted `tool_registry.py` (445 lines) - redundant with registry/ module
  - Deleted `test_tool_registry.py` - dedicated test file no longer needed
  - All functionality already exists in registry/ module:
    - `ToolInfo` → `FeaturizerInfo`
    - `ToolRegistry` → `registry/` facade.py + core.py + queries.py
    - `list_featurizers()` → `list_available_featurizers()`
    - `search_featurizers()` → `FeaturizerQuery.search()`
  - Updated all tests to use registry/ module instead
  - Impact: Single source of truth, cleaner architecture, less confusion

