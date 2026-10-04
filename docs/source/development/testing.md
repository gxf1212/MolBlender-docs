# Testing Guidelines

Comprehensive testing ensures MolBlender reliability and maintainability.

## Test Structure

```
tests/
├── test_representations/
│   ├── test_fingerprints.py
│   ├── test_descriptors.py
│   └── test_spatial.py
├── test_data/
│   ├── test_molecule.py
│   └── test_dataset.py
├── test_models/
│   └── test_screening.py
└── conftest.py  # Shared fixtures
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_representations/test_fingerprints.py

# Run with coverage
pytest --cov=molblender --cov-report=html

# Run specific test
pytest tests/test_molecule.py::test_smiles_validation

# Run in parallel
pytest -n auto
```

## Writing Tests

### Basic Test

```python
def test_featurizer_basic():
    """Test basic featurizer functionality."""
    featurizer = MorganFingerprint(n_bits=1024)
    result = featurizer.featurize("CCO")
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (1024,)
    assert not np.isnan(result).any()
```

### Parametric Test

```python
@pytest.mark.parametrize("smiles,expected_valid", [
    ("CCO", True),
    ("invalid", False),
    ("C1CCCCC1", True),
    ("", False),
])
def test_molecule_validation(smiles, expected_valid):
    """Test molecule validation with various inputs."""
    result = is_valid_smiles(smiles)
    assert result == expected_valid
```

### Fixtures

```python
# conftest.py
import pytest

@pytest.fixture
def sample_molecules():
    """Provide sample molecules for testing."""
    return ["CCO", "CCN", "CCC"]

@pytest.fixture
def test_dataset(sample_molecules, tmp_path):
    """Create test dataset."""
    df = pd.DataFrame({"SMILES": sample_molecules, "y": [1, 2, 3]})
    return MolecularDataset.from_dataframe(df)

# Usage in tests
def test_screening(test_dataset):
    """Test screening with fixture."""
    results = quick_screen(test_dataset, target_column="y")
    assert "best_score" in results
```

### Error Testing

```python
def test_invalid_input_raises():
    """Test that invalid input raises appropriate error."""
    featurizer = MyFeaturizer()
    
    with pytest.raises(InvalidInputError, match="Empty molecule"):
        featurizer.featurize("")
```

## Coverage Requirements

- **New features**: Minimum 80% coverage
- **Bug fixes**: Add test reproducing the bug
- **Refactoring**: Maintain existing coverage

Check coverage:
```bash
pytest --cov=molblender --cov-report=term-missing
```

## Best Practices

1. **Test behavior, not implementation**
2. **One assert per test** (when possible)
3. **Clear test names** describing what is tested
4. **Use fixtures** for common setup
5. **Mock expensive operations** (network, GPU)

## Maintainer: Test Layers

MolBlender marks tests by purpose. Markers are declared in `pyproject.toml`
under `[tool.pytest.ini_options] markers`, and resource-sensitive layers are
gated in `tests/conftest.py`.

**Markers are a declared vocabulary, not an automatic classification.** Declaring
a marker in `pyproject.toml` does not attach it to any test: a test only belongs
to a marker when it carries the matching `@pytest.mark.*` decorator. The commands
in this page therefore run tests by explicit file path, which is what the
verification in this pass actually used.

| Layer | Marker | Purpose | How to run it in practice | Dependencies |
| --- | --- | --- | --- | --- |
| Unit | `unit` | Fast isolated behaviour of one module | `pytest -m unit` selects only tests already decorated with `@pytest.mark.unit`; otherwise run the file or directory directly | core install |
| Contract | `contract` | Public interfaces, API boundaries, import rules, documentation contracts | `pytest -m contract` currently selects nothing, because no test in the repository carries `@pytest.mark.contract`. Run the contract tests by path, plus `python tests/ci/check_layer_dependencies.py` | core install |
| Integration | `integration` | Several components working together | `pytest -m integration` for decorated tests; otherwise run the owning directory | often `dependency` / `asset` |
| Real screen | `real_screen` | Executes a real screening run end to end | `pytest -m real_screen --run-real-screen` | opt in; ≥ 8 GB headroom by default |
| GPU / heavy torch | `gpu`, `torch_heavy` | GPU scheduling and heavyweight torch paths | `pytest -m gpu`, `pytest -m torch_heavy --run-heavy-torch` | GPU node; ≥ 4 GB headroom by default |
| Dashboard smoke | `smoke` | Broad quick-feedback smoke tests; a separate, smaller set from the Dashboard scope tests | `pytest -m smoke` runs the tests already decorated with `@pytest.mark.smoke`. The Dashboard scope and selector tests are *not* decorated, so use the two groups in {doc}`dashboard-smoke` | `dashboard` extra for the UI group |

Two gates are easy to miss:

- `real_screen` and `torch_heavy` are skipped unless `--run-real-screen` /
  `--run-heavy-torch` is passed or `MOLBLENDER_ALLOW_REAL_SCREEN_TESTS=1` /
  `MOLBLENDER_ALLOW_HEAVY_TORCH_TESTS=1` is set. Both also refuse to run under
  `pytest-xdist`, because memory accounting per worker is unreliable.
- Free-memory headroom is checked before those tests start; override the default
  with `MOLBLENDER_TEST_MIN_HEADROOM_GB`.

### Reproducible commands (work environment)

All commands below assume the conda `work` environment (Python 3.9) and are run
from the repository root.

```bash
# Group A — scope and provenance contract tests, no Streamlit required
python -m pytest tests/dashboard/test_split_comparison.py \
    tests/dashboard/test_scope_evaluation_modes.py -q

# Group B — selector / render / AppTest smoke tests, requires the dashboard extra
python -m pytest tests/dashboard/test_scope_selector.py \
    tests/dashboard/test_render_results_table.py \
    tests/dashboard/test_scope_end_to_end.py -q

# Group A + B together: 76 passed in this pass (needs the dashboard extra)

# Documentation contract tests (this pass: 27 passed)
python -m pytest tests/screening/test_configuration_doc_contract.py \
    tests/persistence/test_database_schema_doc_contract.py \
    tests/test_compatibility_doc_contract.py -q

# Layer and import boundary check
python tests/ci/check_layer_dependencies.py

# Lint (add --preview --select RUF059 for the unused-unpack rule)
python -m ruff check src/molblender/dashboard/metrics/scope.py \
    tests/dashboard/test_split_comparison.py

# Documentation build
cd docs && python -m sphinx -b html source build/html
```

### Not verified in this pass

The following are **declared** but were **not executed** in the pass that
produced this page. Do not cite them as passing results:

- the full Dashboard test suite (`pytest tests/dashboard`);
- any re-validation against a real benchmark database;
- Python 3.10 and 3.11 runs — verification for this pass used Python 3.9 only.

Result-shape risks in this pass were covered by small synthetic fixtures that
exercise the same code paths. A synthetic fixture is not a benchmark-scale
verification.

## See Also

- {doc}`contributing` - Contribution workflow
- {doc}`style` - Code style guidelines
- {doc}`dashboard-smoke` - Dashboard smoke behaviour and coverage
- {doc}`compatibility` - Verification scope, declared versus executed
