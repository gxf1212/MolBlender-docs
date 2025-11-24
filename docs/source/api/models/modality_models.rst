Modality Models
===============

Modality-specific model wrappers and detection.

.. currentmodule:: molblender.models.modality_models

Overview
--------

This module provides wrappers for models that handle specific data modalities (STRING, MATRIX, IMAGE).

Modality Detector
-----------------

.. automodule:: molblender.models.modality_models.modality_detector
   :members:
   :undoc-members:
   :show-inheritance:

CNN Models
----------

.. automodule:: molblender.models.modality_models.cnn_models
   :members:
   :undoc-members:
   :show-inheritance:

VAE Models
----------

.. automodule:: molblender.models.modality_models.vae_models
   :members:
   :undoc-members:
   :show-inheritance:

String Models
-------------

.. automodule:: molblender.models.modality_models.string_models
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
------------

.. automodule:: molblender.models.modality_models.base
   :members:
   :undoc-members:
   :show-inheritance:

Supported Modalities
--------------------

**VECTOR**
  Traditional fingerprints, descriptors, pre-computed embeddings.
  Compatible with: Traditional ML, VAE

**STRING**
  Raw SMILES/SELFIES strings.
  Compatible with: Transformer models

**MATRIX**
  2D arrays (adjacency, Coulomb matrices).
  Compatible with: CNN models

**IMAGE**
  Molecular images (2D drawings, 3D renders).
  Compatible with: CNN models

**LANGUAGE_MODEL**
  Pre-computed embeddings from ChemBERTa, MolFormer, etc.
  Compatible with: Traditional ML

See Also
--------

- :doc:`../../usage/models/index` - Usage guide
- :doc:`corpus` - Model definitions
- :doc:`screening` - Screening functions
