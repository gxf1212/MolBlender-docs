Model Corpus
============

Pre-defined model configurations and parameter grids.

.. currentmodule:: molblender.models.corpus

Overview
--------

The model corpus contains definitions for 28+ machine learning models with optimized hyperparameter grids.

Model Corpus
------------

.. automodule:: molblender.models.corpus.model_corpus
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Grids
---------------

.. automodule:: molblender.models.corpus.parameter_grids
   :members:
   :undoc-members:
   :show-inheritance:

Available Models
----------------

**Traditional ML:**

* RandomForest
* XGBoost
* LightGBM  
* GradientBoosting
* SVM
* ElasticNet
* Ridge
* Lasso
* KNN
* DecisionTree

**Neural Networks:**

* VAE (Variational Autoencoder)
* CNN (Convolutional Neural Network)
* Transformer
* MLP (Multi-Layer Perceptron)

**Ensemble Methods:**

* AdaBoost
* BaggingRegressor
* ExtraTreesRegressor
* StackingRegressor

See Also
--------

- :doc:`../../usage/models/models` - Model overview
- :doc:`screening` - Screening functions
- :doc:`modality_models` - Modality-specific wrappers
