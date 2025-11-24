Dependencies
============

Unified dependency management system.

.. currentmodule:: molblender.config.dependencies

Overview
--------

The dependencies module provides a unified interface for checking and loading optional dependencies.

Functions
---------

RDKit
~~~~~

.. autofunction:: is_rdkit_available

.. autofunction:: get_rdkit

DeepChem
~~~~~~~~

.. autofunction:: is_deepchem_available

.. autofunction:: get_deepchem

PyTorch
~~~~~~~

.. autofunction:: is_torch_available

.. autofunction:: get_torch

Transformers
~~~~~~~~~~~~

.. autofunction:: is_transformers_available

.. autofunction:: get_transformers

Usage Pattern
-------------

All dependency functions follow the same pattern:

.. code-block:: python

   from molblender.config import dependencies as deps
   
   # Check if available (doesn't raise error)
   if deps.is_rdkit_available():
       print("RDKit is installed")
   
   # Get dependency (raises DependencyNotFoundError if missing)
   rdkit = deps.get_rdkit()
   Chem = rdkit['Chem']
   AllChem = rdkit['AllChem']

Benefits
--------

* **Graceful degradation**: Features fail gracefully when deps missing
* **Clear error messages**: Helpful installation instructions
* **Easy mocking**: Simplified testing
* **Centralized**: Single place to manage all dependencies

See Also
--------

* :doc:`settings` - Global settings
* :doc:`../../development/setup` - Installing dependencies
