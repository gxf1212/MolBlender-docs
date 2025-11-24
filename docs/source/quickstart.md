# Quick Start

A comprehensive end-to-end example demonstrating how to use MolBlender for molecular property prediction, from data loading through model deployment.

## Overview

This tutorial walks through a complete molecular machine learning workflow:

1. **Data Loading & Validation** - Import molecular datasets from various formats
2. **Representation Generation** - Create diverse molecular features  
3. **Automated Model Screening** - Find optimal model+representation combinations
4. **Results Visualization** - Generate publication-quality plots
5. **Model Deployment** - Save and use trained models for predictions

## Dataset: Predicting Molecular Solubility

We'll predict aqueous solubility using a public dataset of drug-like molecules.

### Step 1: Data Loading and Exploration

```python
import pandas as pd
import numpy as np
import molblender as pm
from molblender.data import MolecularDataset
from molblender.models.api import thorough_screen
from molblender.drawings import plot_publication_regression

# Load solubility dataset (example data)
# In practice, download from: https://www.moleculenet.org/datasets-1
solubility_data = pd.DataFrame({
    'SMILES': [
        'CCO',                                    # Ethanol
        'c1ccccc1O',                             # Phenol  
        'CC(=O)OC1=CC=CC=C1C(=O)O',             # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',        # Caffeine
        'CC1=CC=C(C=C1)C(=O)O',                 # p-Toluic acid
        'CCCCCCCCCC(=O)O',                       # Decanoic acid
        'c1ccc2c(c1)ccc3c2ccc4c3cccc4',         # Anthracene
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'        # Ibuprofen
    ],
    'solubility': [-0.77, -0.04, -2.23, -0.07, -2.25, -4.09, -7.56, -3.97]  # LogS values
})

# Create MolBlender dataset
dataset = MolecularDataset.from_dataframe(
    solubility_data,
    input_column='SMILES', 
    label_columns=['solubility']
)

print(f"Dataset loaded: {len(dataset)} molecules")
print(f"Solubility range: {dataset.properties['solubility'].min():.2f} to {dataset.properties['solubility'].max():.2f}")

# Data validation
validation_report = dataset.validate()
print(f"All molecules valid: {validation_report['all_valid']}")
```

### Step 2: Generate Diverse Molecular Representations

```python
# Select diverse representation types for comprehensive screening
representations = [
    # Fingerprints - structural patterns
    "morgan_fp_r2_1024",           # Circular fingerprints
    "rdkit_fp_1024",               # RDKit path-based fingerprints  
    "maccs_keys",                  # 166-bit pharmacophore keys
    
    # Descriptors - physicochemical properties
    "rdkit_descriptors",           # ~200 2D/3D descriptors
    "rdkit_essential_descriptors", # Key drug-like properties
    
    # Advanced representations
    "coulomb_matrix",              # 3D spatial representation
    "mordred_descriptors_2d"       # Comprehensive descriptors (1613)
]

# Add representations with parallel processing
print("Computing molecular representations...")
for repr_name in representations:
    try:
        success_count = dataset.add_features(repr_name, n_workers=4)
        success_rate = success_count / len(dataset) * 100
        print(f"  {repr_name}: {success_rate:.0f}% success")
    except Exception as e:
        print(f"  {repr_name}: FAILED - {e}")

# Show feature summary
print(f"\nFeatures computed: {list(dataset.features.columns)}")
for col in dataset.features.columns:
    shape = dataset.features[col].iloc[0].shape if dataset.features[col].iloc[0] is not None else "None"
    print(f"  {col}: {shape}")
```

### Step 3: Automated Model Screening

```python
# Comprehensive model screening with multi-stage optimization
print("Starting automated model screening...")

screening_results = thorough_screen(
    dataset=dataset,
    target_column='solubility',
    task_type='regression',
    
    # Model selection
    model_corpus='accurate',        # Use high-performance models
    
    # Representation selection (use subset for speed in this example)
    representations=[
        "morgan_fp_r2_1024",
        "rdkit_essential_descriptors", 
        "maccs_keys"
    ],
    
    # Multi-stage settings
    enable_stage3=True,             # Enable adaptive refinement
    stage1_top_percent=0.3,         # Keep top 30% for stage 2
    stage2_top_n=3,                 # Report top 3 models
    
    # Performance settings
    n_jobs=4,                       # Parallel execution
    cv_folds=5,                     # 5-fold cross-validation
    
    # Output settings  
    project_name="solubility_screening",
    save_models=True
)

# Display results
print(f"\nScreening Results:")
print(f"Success: {screening_results['success']}")
print(f"Best Model: {screening_results['best_model']['model_name']}")
print(f"Best Representation: {screening_results['best_model']['representation_name']}")

best_metrics = screening_results['best_model']['metrics']
print(f"Performance: R² = {best_metrics['r2']:.3f}, RMSE = {best_metrics['rmse']:.3f}")

# Show top 3 models
print(f"\nTop 3 Models:")
for i, model_info in enumerate(screening_results['top_models'][:3]):
    print(f"  {i+1}. {model_info['model_name']} + {model_info['representation_name']}")
    print(f"     R² = {model_info['metrics']['r2']:.3f}")
```

### Step 4: Results Visualization

```python
# Get the best trained model
best_estimator = screening_results['best_estimator']
best_repr_name = screening_results['best_model']['representation_name']

# Make predictions for visualization
X = dataset.get_feature_matrix(best_repr_name)
y_true = dataset.properties['solubility'].values
y_pred = best_estimator.predict(X)

# Create publication-quality plot
from molblender.drawings import plot_publication_regression

fig, ax, metrics = plot_publication_regression(
    y_true=y_true,
    y_pred=y_pred,
    title=f"Solubility Prediction\n{screening_results['best_model']['model_name']} + {best_repr_name}",
    xlabel="Experimental LogS",
    ylabel="Predicted LogS", 
    save_path="solubility_prediction.png"
)

print(f"Visualization saved: solubility_prediction.png")
print(f"Final metrics: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}")

# Model comparison plot
from molblender.drawings import plot_model_comparison

comparison_data = []
for model_info in screening_results['top_models'][:5]:
    comparison_data.append({
        'name': f"{model_info['model_name']}\n{model_info['representation_name'][:20]}",
        'r2': model_info['metrics']['r2'],
        'rmse': model_info['metrics']['rmse']
    })

plot_model_comparison(
    comparison_data,
    metric='r2',
    title="Model Performance Comparison",
    save_path="model_comparison.png"
)
```

### Step 5: Model Deployment

```python
import joblib
from datetime import datetime

# Save the complete model pipeline
model_info = {
    'model': best_estimator,
    'representation_name': best_repr_name, 
    'scaler': None,  # Add if you used scaling
    'feature_names': dataset.get_feature_names(best_repr_name),
    'training_date': datetime.now(),
    'performance_metrics': best_metrics,
    'dataset_size': len(dataset)
}

# Save model
model_filename = "solubility_model.joblib"
joblib.dump(model_info, model_filename)
print(f"Model saved: {model_filename}")

# Create prediction function
def predict_solubility(smiles_list):
    """Predict solubility for new molecules"""
    
    # Load model
    model_info = joblib.load(model_filename)
    
    # Create temporary dataset
    temp_dataset = MolecularDataset.from_smiles(smiles_list)
    
    # Generate same representations as training
    temp_dataset.add_features(model_info['representation_name'])
    
    # Get features and predict
    X = temp_dataset.get_feature_matrix(model_info['representation_name'])
    predictions = model_info['model'].predict(X)
    
    return predictions

# Test prediction function
test_molecules = ['CCO', 'CCCO', 'CCCCO']  # Alcohols of increasing chain length
predicted_solubility = predict_solubility(test_molecules)

print(f"\nPredictions for new molecules:")
for smiles, pred in zip(test_molecules, predicted_solubility):
    print(f"  {smiles}: LogS = {pred:.2f}")
```

### Step 6: Advanced Analysis

```python
# Feature importance analysis (for tree-based models)
if hasattr(best_estimator, 'feature_importances_'):
    import matplotlib.pyplot as plt
    
    feature_names = dataset.get_feature_names(best_repr_name)
    importances = best_estimator.feature_importances_
    
    # Get top 10 most important features
    indices = np.argsort(importances)[-10:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    print("Feature importance plot saved: feature_importance.png")

# Cross-validation analysis
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    best_estimator, X, y_true, 
    cv=5, scoring='r2'
)

print(f"\nCross-validation results:")
print(f"  Mean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  Individual folds: {[f'{score:.3f}' for score in cv_scores]}")

# Error analysis
residuals = y_true - y_pred
print(f"\nError analysis:")
print(f"  Mean absolute error: {np.mean(np.abs(residuals)):.3f}")
print(f"  Max error: {np.max(np.abs(residuals)):.3f}")
print(f"  Error std dev: {np.std(residuals):.3f}")
```

## Summary and Best Practices

This workflow demonstrates MolBlender's key capabilities:

```{admonition} Key Takeaways
:class: tip

**Data Management:**
- Always validate your dataset with `dataset.validate()`
- Handle invalid molecules gracefully with appropriate error strategies
- Use multiple file format support for flexible data loading

**Representation Selection:**
- Start with diverse representation types (fingerprints, descriptors, 3D)
- Use `thorough_screen()` to automatically find optimal combinations
- Consider computational cost vs. accuracy trade-offs

**Model Screening:**
- Enable multi-stage optimization for best results
- Use parallel processing (`n_jobs`) to speed up screening
- Save screening results for reproducibility

**Results Analysis:**
- Generate publication-quality visualizations
- Analyze feature importance for interpretability
- Validate performance with proper cross-validation

**Deployment:**
- Save complete model pipelines with metadata
- Create reusable prediction functions
- Document model performance and limitations
```

### Scaling to Larger Datasets

For datasets with >10,000 molecules:

```python
# Memory-efficient processing
large_dataset = MolecularDataset.from_csv(
    "large_dataset.csv",
    input_column="SMILES",
    label_columns=["target"],
    chunk_size=10000,          # Process in chunks
    cache_dir="./cache"        # Enable caching
)

# Use quick screening first
quick_results = pm.models.api.quick_screen(
    large_dataset, 
    target_column="target",
    representations=["morgan_fp_r2_1024", "rdkit_essential_descriptors"]
)

# Then detailed screening on promising approaches
detailed_results = pm.models.api.thorough_screen(
    large_dataset,
    target_column="target", 
    representations=[quick_results['best_model']['representation_name']],
    model_corpus='accurate'
)
```

### Next Steps

- Explore specialized representations for your domain
- Implement custom featurizers for novel descriptors
- Use the visualization module for advanced plotting
- Deploy models in production environments
- Extend to multi-task and multi-modal learning

This complete workflow showcases MolBlender's power in automating molecular machine learning while maintaining flexibility and interpretability.