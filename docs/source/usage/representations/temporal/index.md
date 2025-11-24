# Temporal Representations

Extract features and representations from molecular dynamics (MD) trajectories and conformational ensembles to capture dynamic molecular behavior over time.

## Introduction

Temporal representations capture the dynamic aspects of molecular systems that static representations cannot reveal. These are crucial for understanding:

- Conformational flexibility and dynamics
- Protein folding and unfolding pathways  
- Drug-target binding kinetics
- Ensemble-averaged molecular properties
- Time-dependent chemical reactions

MolBlender provides tools to analyze both molecular dynamics trajectories and multi-conformer ensembles, extracting statistical features that characterize dynamic behavior.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🎬 **MD Trajectories**
Statistical features from molecular dynamics simulations
:::

:::{grid-item-card} 🔄 **Conformer Ensembles**
Multi-conformer analysis and shape diversity
:::

:::{grid-item-card} 📊 **Time Series**
Temporal statistics and autocorrelation analysis
:::

:::{grid-item-card} ⚡ **Streaming Processing**
Handle large trajectories with memory efficiency
:::
::::

## Quick Start

```python
import molblender as pm

# Conformational ensemble analysis
ensemble = pm.get_featurizer("conformer_ensemble")
flexibility_features = ensemble.featurize("CCCCCCCC")  # Flexible alkane chain

print(f"Ensemble features shape: {flexibility_features.shape}")
print(f"Conformational diversity score: {flexibility_features[10]:.3f}")

# MD trajectory analysis (requires trajectory files)
try:
    md_analyzer = pm.get_featurizer("md_trajectory_features")
    trajectory_features = md_analyzer.featurize("trajectory.dcd", "topology.pdb")
    print(f"Trajectory features: {trajectory_features.shape}")
except FileNotFoundError:
    print("MD trajectory files not found - using ensemble analysis instead")
```

## MD Trajectory Analysis

### Loading Trajectories

MolBlender integrates with MDAnalysis for trajectory processing:

```bash
# Install MD analysis dependencies
pip install MDAnalysis
```

```python
import molblender as pm

# Initialize trajectory analyzer
md_analyzer = pm.get_featurizer("md_trajectory_features", 
                               selection="protein",  # Atom selection
                               frame_step=10)        # Every 10th frame

# Analyze single trajectory
trajectory_file = "md_simulation.dcd"
topology_file = "system.pdb"

features = md_analyzer.featurize(trajectory_file, topology_file)
print(f"Trajectory features shape: {features.shape}")

# Available atom selections:
selections = [
    "all",           # All atoms
    "protein",       # Protein atoms only
    "backbone",      # Protein backbone
    "sidechain",     # Side chain atoms
    "resname LIG",   # Ligand molecules
    "within 5 of resname LIG"  # Atoms near ligand
]
```

### Extracted Features

The MD trajectory analyzer extracts comprehensive time-dependent features:

| Feature Category | Count | Description |
|---|---|---|
| **RMSF Statistics** | 10 | Root mean square fluctuation metrics |
| **Distance Statistics** | 10 | Inter-atomic distance variations |
| **Angle Statistics** | 10 | Bond/dihedral angle fluctuations |
| **Shape Statistics** | 10 | Radius of gyration and shape changes |
| **Correlation Features** | 10 | Temporal autocorrelation measures |

```python
# Get detailed trajectory analysis
md_analyzer = pm.get_featurizer("md_trajectory_features")
features = md_analyzer.featurize("trajectory.dcd", "topology.pdb")

# Feature interpretation (example indices)
print("Trajectory Analysis:")
print(f"Mean RMSF: {features[0]:.3f} Å")
print(f"RMSF standard deviation: {features[1]:.3f} Å")
print(f"Average radius of gyration: {features[30]:.3f} Å")
print(f"RoG fluctuation: {features[31]:.3f} Å")
print(f"Autocorrelation time: {features[40]:.1f} frames")
```

### Time Series Analysis

Extract and analyze specific properties over time:

```python
# Custom time series extraction
class CustomTrajectoryAnalyzer:
    def __init__(self, trajectory_file, topology_file):
        import MDAnalysis as mda
        self.universe = mda.Universe(topology_file, trajectory_file)
        
    def extract_properties(self, selection="protein"):
        """Extract time series of molecular properties"""
        atoms = self.universe.select_atoms(selection)
        
        properties = {
            'radius_gyration': [],
            'end_to_end_distance': [],
            'secondary_structure': []
        }
        
        for frame in self.universe.trajectory:
            # Radius of gyration
            rog = atoms.radius_of_gyration()
            properties['radius_gyration'].append(rog)
            
            # End-to-end distance (if applicable)
            if len(atoms) > 1:
                distance = np.linalg.norm(atoms[0].position - atoms[-1].position)
                properties['end_to_end_distance'].append(distance)
        
        return properties
    
    def compute_statistics(self, time_series):
        """Compute statistical features from time series"""
        stats = {}
        for prop_name, values in time_series.items():
            values = np.array(values)
            stats[prop_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'autocorr_time': self.autocorrelation_time(values)
            }
        return stats

# Usage
analyzer = CustomTrajectoryAnalyzer("trajectory.dcd", "topology.pdb")
time_series = analyzer.extract_properties()
statistics = analyzer.compute_statistics(time_series)

print("Time series statistics:")
for prop, stats in statistics.items():
    print(f"{prop}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
```

## Conformational Ensembles

### Multi-Conformer Generation

Analyze conformational flexibility by generating multiple conformers:

```python
# Initialize ensemble analyzer
ensemble = pm.get_featurizer("conformer_ensemble", 
                           n_conformers=50,      # Generate 50 conformers
                           energy_window=10.0)   # Within 10 kcal/mol

# Analyze flexible molecules
flexible_mol = "CCCCCCCC"  # Octane - highly flexible
rigid_mol = "c1ccccc1"     # Benzene - rigid ring

flex_features = ensemble.featurize(flexible_mol)
rigid_features = ensemble.featurize(rigid_mol)

print(f"Flexible molecule diversity: {flex_features[10]:.3f}")
print(f"Rigid molecule diversity: {rigid_features[10]:.3f}")

# Expected: Flexible molecules have higher diversity scores
```

### Conformational Diversity Metrics

The ensemble analyzer computes various diversity measures:

```python
ensemble = pm.get_featurizer("conformer_ensemble")
features = ensemble.featurize("CC(C)C(=O)O")  # Flexible carboxylic acid

# Feature interpretation
print("Conformational Analysis:")
print(f"Energy range: {features[1] - features[0]:.1f} kcal/mol")
print(f"Mean energy: {features[2]:.1f} kcal/mol")
print(f"RMSD diversity: {features[4]:.2f} Å")
print(f"Shape diversity (PMI): {features[7]:.3f}")
print(f"Radius of gyration spread: {features[15]:.2f} Å")

# Compare different molecule types
molecules = {
    "Flexible chain": "CCCCCCCC",
    "Rigid ring": "c1ccccc1", 
    "Semi-flexible": "CCc1ccccc1",
    "Branched": "CC(C)(C)C"
}

for name, smiles in molecules.items():
    features = ensemble.featurize(smiles)
    diversity = features[4]  # RMSD diversity index
    print(f"{name:15}: diversity = {diversity:.3f}")
```

### Energy Landscape Analysis

Analyze the conformational energy landscape:

```python
# Custom ensemble analysis with energy details
import numpy as np

def analyze_conformational_landscape(smiles, n_conformers=100):
    """Detailed conformational landscape analysis"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate conformers
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=42)
    
    # Calculate energies
    energies = []
    for conf_id in conf_ids:
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        energies.append(ff.CalcEnergy())
    
    energies = np.array(energies)
    min_energy = np.min(energies)
    relative_energies = energies - min_energy
    
    return {
        'n_conformers': len(conf_ids),
        'energy_range': np.ptp(relative_energies),
        'low_energy_conformers': np.sum(relative_energies < 2.0),  # < 2 kcal/mol
        'energy_gaps': np.diff(np.sort(relative_energies)[:5]),  # Gaps between lowest 5
        'mean_energy': np.mean(relative_energies),
        'energy_distribution': np.histogram(relative_energies, bins=10)[0]
    }

# Analyze different molecules
results = analyze_conformational_landscape("CCCC(C)C(=O)O")
print(f"Energy range: {results['energy_range']:.1f} kcal/mol")
print(f"Low energy conformers: {results['low_energy_conformers']}/100")
print(f"Energy gaps: {results['energy_gaps']}")
```

## Time Series Features

### Autocorrelation Analysis

Analyze temporal correlations in molecular properties:

```python
def compute_autocorrelation(time_series, max_lag=50):
    """Compute autocorrelation function"""
    time_series = np.array(time_series)
    time_series = time_series - np.mean(time_series)
    
    autocorr = np.correlate(time_series, time_series, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    return autocorr[:max_lag]

def autocorr_decay_time(autocorr):
    """Find decay time (1/e point)"""
    try:
        decay_idx = np.where(autocorr < 1/np.e)[0][0]
        return decay_idx
    except IndexError:
        return len(autocorr)

# Example with simulated data
import numpy as np

# Simulate RMSD time series with correlation
t = np.linspace(0, 100, 1000)
rmsd_series = 2.0 + 0.5 * np.sin(0.1 * t) + np.random.normal(0, 0.2, 1000)

autocorr = compute_autocorrelation(rmsd_series)
decay_time = autocorr_decay_time(autocorr)

print(f"Autocorrelation decay time: {decay_time} frames")
print(f"Correlation at lag 10: {autocorr[10]:.3f}")

# Visualize if matplotlib available
try:
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(rmsd_series[:200])
    ax1.set_title("RMSD Time Series")
    ax1.set_ylabel("RMSD (Å)")
    
    ax2.plot(autocorr)
    ax2.axhline(1/np.e, color='r', linestyle='--', label='1/e decay')
    ax2.set_title("Autocorrelation Function")
    ax2.set_xlabel("Lag (frames)")
    ax2.set_ylabel("Autocorrelation")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("temporal_analysis.png")
    print("Analysis plot saved as temporal_analysis.png")
    
except ImportError:
    print("Matplotlib not available - skipping visualization")
```

### Drift and Trend Analysis

Detect systematic changes over time:

```python
def detect_drift(time_series):
    """Detect linear drift/trend in time series"""
    from scipy.stats import linregress
    
    time_points = np.arange(len(time_series))
    slope, intercept, r_value, p_value, std_err = linregress(time_points, time_series)
    
    return {
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
    }

# Example: Detect protein unfolding
# Simulated radius of gyration increase (unfolding)
t = np.linspace(0, 100, 500)
rog_series = 15 + 0.02 * t + np.random.normal(0, 0.5, 500)  # Gradual increase

drift_analysis = detect_drift(rog_series)
print(f"Trend: {drift_analysis['trend']}")
print(f"Slope: {drift_analysis['slope']:.4f} per frame")
print(f"R-squared: {drift_analysis['r_squared']:.3f}")
print(f"Significant trend: {drift_analysis['significant']}")
```

## Memory-Efficient Processing

### Streaming Large Trajectories

Handle large trajectory files without loading everything into memory:

```python
def stream_trajectory_analysis(trajectory_file, topology_file, chunk_size=1000):
    """Analyze large trajectories in chunks"""
    import MDAnalysis as mda
    
    universe = mda.Universe(topology_file, trajectory_file)
    selection = universe.select_atoms("protein")
    
    # Initialize streaming statistics
    running_stats = {
        'count': 0,
        'mean_rog': 0,
        'var_rog': 0,
        'min_rog': float('inf'),
        'max_rog': float('-inf')
    }
    
    print(f"Processing {len(universe.trajectory)} frames in chunks of {chunk_size}")
    
    for chunk_start in range(0, len(universe.trajectory), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(universe.trajectory))
        
        # Process chunk
        chunk_rogs = []
        for frame_idx in range(chunk_start, chunk_end):
            universe.trajectory[frame_idx]
            rog = selection.radius_of_gyration()
            chunk_rogs.append(rog)
        
        # Update running statistics
        update_running_stats(running_stats, chunk_rogs)
        
        print(f"Processed frames {chunk_start}-{chunk_end}")
    
    return finalize_stats(running_stats)

def update_running_stats(stats, new_data):
    """Update running statistics with new chunk"""
    new_data = np.array(new_data)
    
    # Update count
    old_count = stats['count']
    stats['count'] += len(new_data)
    
    # Update mean and variance (Welford's algorithm)
    if old_count == 0:
        stats['mean_rog'] = np.mean(new_data)
        stats['var_rog'] = np.var(new_data)
    else:
        old_mean = stats['mean_rog']
        new_mean = old_mean + np.sum(new_data - old_mean) / stats['count']
        
        # Update variance
        old_var = stats['var_rog']
        new_var = (old_count * old_var + np.sum((new_data - new_mean) ** 2)) / stats['count']
        
        stats['mean_rog'] = new_mean
        stats['var_rog'] = new_var
    
    # Update min/max
    stats['min_rog'] = min(stats['min_rog'], np.min(new_data))
    stats['max_rog'] = max(stats['max_rog'], np.max(new_data))

# Usage
try:
    streaming_stats = stream_trajectory_analysis("large_trajectory.dcd", "topology.pdb")
    print("Streaming analysis complete:")
    print(f"Mean RoG: {streaming_stats['mean_rog']:.2f} Å")
    print(f"Std RoG: {np.sqrt(streaming_stats['var_rog']):.2f} Å")
    print(f"Range: {streaming_stats['min_rog']:.1f} - {streaming_stats['max_rog']:.1f} Å")
except FileNotFoundError:
    print("Trajectory files not found - this is a demonstration")
```

### Parallel Ensemble Processing

Process multiple conformers or trajectories in parallel:

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def parallel_conformer_analysis(smiles_list, n_conformers=50):
    """Analyze conformational ensembles in parallel"""
    
    def analyze_single_molecule(smiles):
        ensemble = pm.get_featurizer("conformer_ensemble", n_conformers=n_conformers)
        return ensemble.featurize(smiles)
    
    # Use all available CPUs
    n_workers = mp.cpu_count()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(analyze_single_molecule, smiles_list))
    
    return results

# Example usage
molecules = ["CCCCCCCC", "c1ccccc1", "CCc1ccccc1", "CC(C)(C)C"]

print(f"Analyzing {len(molecules)} molecules in parallel...")
parallel_results = parallel_conformer_analysis(molecules)

for i, (smiles, features) in enumerate(zip(molecules, parallel_results)):
    diversity = features[4]  # RMSD diversity
    print(f"Molecule {i+1} ({smiles}): diversity = {diversity:.3f}")
```

## Integration with Dataset

```python
from molblender.data import MolecularDataset

# Create dataset with temporal features
molecules = ["CCCCCCCC", "c1ccccc1", "CCc1ccccc1"] 
dataset = MolecularDataset.from_smiles(molecules)

# Add conformational ensemble features
dataset.add_features("conformer_ensemble", n_workers=4)

# Access temporal features
print("Dataset with temporal features:")
print(dataset.features.columns.tolist())

# Analyze conformational flexibility across dataset
flexibility_scores = []
for features in dataset.features.iloc[:, 0]:  # First feature column
    diversity = features[4]  # RMSD diversity index
    flexibility_scores.append(diversity)

print(f"Dataset flexibility range: {min(flexibility_scores):.3f} - {max(flexibility_scores):.3f}")
```

## Troubleshooting

### Memory Management

```python
# Monitor memory usage during processing
import psutil
import os

def monitor_memory_usage(func, *args, **kwargs):
    """Monitor memory usage of a function"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024**2  # MB
    
    result = func(*args, **kwargs)
    
    final_memory = process.memory_info().rss / 1024**2  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"Memory usage: {initial_memory:.1f} → {final_memory:.1f} MB (Δ{memory_increase:.1f} MB)")
    
    return result

# Example
ensemble = pm.get_featurizer("conformer_ensemble")
features = monitor_memory_usage(ensemble.featurize, "CCCCCCCC")
```

### Performance Optimization

```python
# Optimize conformer generation parameters
def benchmark_conformer_settings():
    """Benchmark different conformer generation settings"""
    import time
    
    smiles = "CC(C)C(=O)NCCCC"  # Medium complexity molecule
    settings = [
        {'n_conformers': 25, 'energy_window': 15.0},
        {'n_conformers': 50, 'energy_window': 10.0},
        {'n_conformers': 100, 'energy_window': 5.0}
    ]
    
    for setting in settings:
        ensemble = pm.get_featurizer("conformer_ensemble", **setting)
        
        start_time = time.time()
        features = ensemble.featurize(smiles)
        elapsed = time.time() - start_time
        
        diversity = features[4]
        print(f"n_conf={setting['n_conformers']}, "
              f"E_window={setting['energy_window']}: "
              f"time={elapsed:.1f}s, diversity={diversity:.3f}")

benchmark_conformer_settings()
```

## References

- [MDAnalysis](https://www.mdanalysis.org/) - Python library for MD trajectory analysis
- [RDKit Conformers](https://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules) - Conformer generation methods
- [Conformational Analysis Review](https://doi.org/10.1021/acs.jcim.7b00221) - Methods for conformational ensemble analysis

```{toctree}
:maxdepth: 1
:hidden:
```