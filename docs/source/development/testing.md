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

## See Also

- {doc}`contributing` - Contribution workflow
- {doc}`style` - Code style guidelines
