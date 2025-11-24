# Contributing to MolBlender

Thank you for your interest in contributing to MolBlender! This guide will help you get started.

## Code of Conduct

By participating in this project, you agree to maintain a respectful, inclusive, and collaborative environment. Be kind, constructive, and professional in all interactions.

## Ways to Contribute

### 1. Report Issues

Found a bug or have a feature request?

**Bug Reports:**
- Check if the issue already exists
- Use the bug report template
- Include: OS, Python version, MolBlender version
- Provide minimal reproducible example
- Include error messages and stack traces

**Feature Requests:**
- Explain the use case
- Describe expected behavior
- Suggest potential implementation (optional)
- Link to relevant papers/resources

### 2. Improve Documentation

Documentation contributions are highly valued:
- Fix typos, unclear explanations
- Add examples and tutorials
- Improve API docstrings
- Create how-to guides

### 3. Contribute Code

See detailed workflow below.

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/MolBlender.git
cd MolBlender

# Add upstream remote
git remote add upstream https://github.com/gxf1212/MolBlender.git
```

### 2. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feat/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch Naming:**
- `feat/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/fixes

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### 4. Make Changes

**Before coding:**
- Read relevant architecture documentation
- Check style guidelines
- Look at similar existing code

**While coding:**
- Write clean, documented code
- Follow PEP 8 and type hint guidelines
- Add docstrings (Google style)
- Handle errors gracefully

### 5. Write Tests

**Every code change must include tests.**

Run tests locally:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=molblender --cov-report=html
```

### 6. Commit Changes

**Commit Message Format** (Conventional Commits):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Example:**
```
feat(representations): add MACCS fingerprint featurizer

Implements MACCS keys fingerprint using RDKit. Supports both
166-bit and 167-bit versions.

Closes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### 7. Push and Create Pull Request

```bash
# Push to your fork
git push origin feat/your-feature-name
```

**PR Checklist:**
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Branch is up-to-date with main

Thank you for making MolBlender better! 🚀
