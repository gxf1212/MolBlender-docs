# Development Setup

Guide for setting up your development environment for MolBlender.

## Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

## Installation Steps

### 1. Fork and Clone

```bash
# Fork on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/MolBlender.git
cd MolBlender
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n molblender python=3.9
conda activate molblender
```

### 3. Install in Editable Mode

```bash
# Core dependencies + development tools
pip install -e ".[dev]"

# With all optional dependencies
pip install -e ".[all,dev]"
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

## IDE Configuration

### VSCode

Recommended extensions:
- Python
- Pylance
- autoDocstring
- GitLens

Settings (`.vscode/settings.json`):
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

### PyCharm

- Enable pytest as test runner
- Configure Black as formatter
- Enable type checking

## Verify Installation

```bash
# Run tests
pytest

# Check code style
black --check src/

# Run type checker
mypy src/molblender
```

## Next Steps

- Read {doc}`contributing` for workflow
- Check {doc}`style` for coding standards
- See {doc}`testing` for test guidelines
