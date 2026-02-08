# Installation

MolBlender is designed to be a modular toolkit. You can install a lightweight core and then add optional dependencies to enable specific functionalities like protein analysis, advanced machine learning models, or molecular dynamics.

The package requires **Python \>= 3.9**.

-----

## Installation Methods

We recommend using `conda` to manage complex cheminformatics dependencies like RDKit, especially on servers without GPU. The `pip` installation is also fully supported but may encounter compilation issues for some packages.

### Method 1: Conda (Recommended for Servers)

Using `conda` is the most reliable way to install the core cheminformatics backends, as `conda-forge provides pre-compiled binaries for all major operating systems.

```bash
# 1. Create and activate a new conda environment
conda create -n molblender python=3.9 -y
conda activate molblender

# 2. Install cheminformatics dependencies from conda-forge
conda install -c conda-forge rdkit deepchem -y

# 3. Install MolBlender in editable mode
cd /path/to/MolBlender
pip install -e .

# 4. Install optional functionalities as needed
# For ML models (recommended):
pip install torch torchvision scikit-learn xgboost lightgbm catboost
```

### Method 2: pip (Simplest for Testing)

You can install MolBlender directly from PyPI.

#### Basic Installation

This installs the core package with its minimal dependencies (`numpy`, `pandas`, `rdkit-pypi`).

```bash
pip install molblender
```

#### Installation with Extras

To use advanced features, you need to install "extras." This is done by adding brackets `[]` with the name of the feature group to the pip install command. You can install multiple groups at once.

```bash
# Install with a single optional group (e.g., for ML models)
pip install molblender[models]

# Install multiple groups at once
pip install molblender[protein,models]

# Install all optional dependencies
pip install molblender[all]
```

### Method 3: Using environment.yml (Production)

For production environments or servers, use the provided `environment.yml` file:

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate molblender

# Install MolBlender
cd /path/to/MolBlender
pip install -e .
```

This method ensures all dependencies are installed from conda-forge, avoiding compilation issues.

-----

## Known Installation Issues

### Issue 1: h5py/pyarrow Compilation Failure

**Symptoms**:
```
error: subprocess-exited-with-error
× Building wheel for h5py (pyproject.toml) did not run successfully
× Building wheel for pyarrow (pyproject.toml) did not run successfully
```

**Cause**: These packages require CMake 3.25+ and system libraries (HDF5 for h5py).

**Solution**:
```bash
# Install via conda instead
conda install h5py pyarrow -c conda-forge
```

**Note**: h5py and pyarrow are only required for specific features (e.g., molfeat with Google Cloud storage). Most MolBlender functionality works without them.

### Issue 2: PyTorch Network Unreachable

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement torch
ERROR: No matching distribution found for torch
```

**Cause**: Attempting to install from PyTorch official wheel server which may be blocked.

**Solution**: Use the configured mirror (already set in your environment):
```bash
# For CPU-only servers (already configured)
pip install torch torchvision

# Or explicitly from mirrors.zju.edu.cn
pip install torch torchvision -i https://mirrors.zju.edu.cn/pypi/web/simple
```

### Issue 3: RDKit Installation Slow or Fails

**Symptoms**: RDKit installation takes a very long time or fails with compilation errors.

**Solution**: Always install RDKit via conda:
```bash
conda install -c conda-forge rdkit
```

Then install MolBlender:
```bash
pip install molblender
```

**Note**: When RDKit is already installed via conda, pip will skip it during MolBlender installation.

### Issue 4: Database Has Placeholder Values

**Symptoms**: Dashboard scatter plots show incorrect Pearson R, Overview shows correct score.

**Cause**: Old databases created with placeholder `[0,1,2,...]` instead of actual `y_test` values.

**Solution**: Regenerate the database by re-running screening with the latest code:
```bash
# Backup old database
mv screening_results.db screening_results_old.db

# Re-run screening (this will save correct y_test values)
python run_molblender_screening.py --disable-gpu
```

The new code includes validation that prevents placeholder values from being saved.

-----

### Method 2: pip

You can install MolBlender directly from PyPI.

#### Basic Installation

This installs the core package with its minimal dependencies (`numpy`, `pandas`, `rdkit-pypi`).

```bash
pip install molblender
```

#### Installation with Extras

To use advanced features, you need to install "extras." This is done by adding brackets `[]` with the name of the feature group to the pip install command. You can install multiple groups at once.

```bash
# Install with a single optional group (e.g., for ML models)
pip install molblender[models]

# Install multiple groups at once
pip install molblender[deepchem,md]

# Install a convenience group that includes several related packages
pip install molblender[cheminformatics_base]

# Install all optional dependencies
pip install molblender[all]
```

-----

## Core and Optional Dependencies

### Core Dependencies

These are installed automatically with `pip install molblender`:

  - [NumPy](https://numpy.org/)
  - [Pandas](https://pandas.pydata.org/)
  - [RDKit](https://www.rdkit.org/)

### Optional Dependency Groups (`extras`)

The following table lists the main optional groups and the key functionalities they enable.

| Installation Command                     | Key Libraries Installed                               | Functionality Enabled                                      |
| :--------------------------------------- | :---------------------------------------------------- | :--------------------------------------------------------- |
| `pip install molblender[models]`        | `scikit-learn`, `xgboost`, `lightgbm`, `catboost`       | Standard and gradient-boosted models for QSAR/ML tasks.    |
| `pip install molblender[protein]`       | `biopython`, `fair-esm`, `transformers`, `tokenizers` | Protein featurizers (PLMs) and structural biology tools.   |
| `pip install molblender[cheminformatics_ml]` | `chemprop`, `mol2vec`                            | Advanced ML models like Chemprop and Mol2Vec embeddings.   |
| `pip install molblender[deepchem]`      | `deepchem`                                            | Featurizers and models from the DeepChem library.          |
| `pip install molblender[mordred]`       | `mordred`                                             | Calculation of a large set of 2D & 3D molecular descriptors. |
| `pip install molblender[spatial]`       | `unimol-tools`, `dscribe`, `ase`                      | 3D spatial representations like Coulomb Matrices and Uni-Mol. |
| `pip install molblender[md]`            | `mdanalysis`                                          | Analysis of molecular dynamics trajectories.               |
| `pip install molblender[cdk]`           | `CDK-pywrapper`                                       | CDK-based fingerprints (requires **Java \>= 11**).          |
| `pip install molblender[molfeat]`       | `molfeat`                                             | Various fingerprints and representations via Molfeat.      |
| `pip install molblender[drawing]`       | `matplotlib`, `seaborn`                               | Molecule and data visualization utilities.                 |
| `pip install molblender[openbabel]`     | `openbabel-wheel`                                     | Support for Open Babel via Pybel.                          |

-----

## Developer Installation

If you want to contribute to MolBlender, you should clone the repository and install it in editable mode with the development dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/gxf1212/molblender.git
cd molblender

# 2. Create and activate a conda environment (recommended)
conda create -n molblender_dev python=3.9 -y
conda activate molblender_dev
conda install -c conda-forge rdkit -y # Install base dependencies

# 3. Install the package in editable mode with all development extras
pip install -e .[dev,all]

# 4. Set up pre-commit hooks (optional, but recommended)
pre-commit install
```

This will install the package in a way that your changes to the source code are immediately reflected, and it will also install all testing and linting tools like `pytest` and `ruff`.