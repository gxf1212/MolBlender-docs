# Release Process

Guidelines for maintainers releasing new MolBlender versions.

## Version Numbering

Follow **Semantic Versioning** (SemVer):

- `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Examples:
- `1.0.0` → `1.0.1`: Bug fix
- `1.0.1` → `1.1.0`: New feature
- `1.1.0` → `2.0.0`: Breaking change

## Pre-Release Checklist

- [ ] All tests passing on CI
- [ ] Documentation builds without warnings
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml`
- [ ] All PRs merged and main branch stable

## Release Steps

### 1. Update Version

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md with release date
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
```

### 2. Create Git Tag

```bash
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin main
git push origin vX.Y.Z
```

### 3. Build Distribution

```bash
python -m build
# Creates dist/molblender-X.Y.Z.tar.gz and .whl
```

### 4. Upload to PyPI

```bash
twine upload dist/*
```

### 5. Create GitHub Release

- Go to GitHub Releases
- Select the tag
- Add release notes from CHANGELOG
- Attach built distributions

## Post-Release

- Update documentation if needed
- Announce on relevant channels
- Monitor for issues

## See Also

- {doc}`contributing` - Contribution guidelines
