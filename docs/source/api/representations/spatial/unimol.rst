.. default-role:: py:obj
.. automodule:: molblender.representations.spatial.unimol
   :members: UniMolCLSFeaturizer, UniMolAtomicCoordsFeaturizer, UniMolAtomicReprsFeaturizer, UNIMOL_BASE_CONFIGS
   :undoc-members:
   :show-inheritance:
   :inherited-members: BaseFeaturizer

Uni-Mol Representations
=======================

This module provides an interface to Uni-Mol models for generating molecule-level (CLS token) and atom-level representations. These models are designed to capture 3D spatial information.

This featurizer relies on the `unimol_tools` library. For installation and setup, refer to the :doc:`/usage/representations/spatial/unimol` guide.

Featurizer Classes
------------------

The module provides three specialized featurizers:

- **UniMolCLSFeaturizer**: Returns molecule-level CLS token representation
- **UniMolAtomicCoordsFeaturizer**: Returns 3D coordinates of each atom
- **UniMolAtomicReprsFeaturizer**: Returns representation for each atom

All featurizers handle SMILES strings and RDKit Mol objects as input.

Available Configurations
------------------------

.. py:data:: UNIMOL_BASE_CONFIGS
   :noindex:

   A dictionary defining pre-registered Uni-Mol model configurations. Each key is a registered name
   (e.g., ``"UniMol-V2-84M"``) that can be used with
   :py:func:`~molblender.representations.get_featurizer`. The values are dictionaries specifying
   parameters like `data_type`, `unimol_tools_model_name`, `remove_hs`, and `output_dim`.

   **Example structure:**

   .. code-block:: python

      UNIMOL_BASE_CONFIGS = {
          "UniMol-V2-84M": {
              "data_type": "molecule",
              "unimol_tools_model_name": "unimolv2",
              "unimol_tools_model_size": "84m",
              "remove_hs": False,
              "output_dim": 512
          },
          # ... other configurations ...
      }

Helper Functions
----------------

The module contains an internal helper function ``_ensure_unimol_loaded`` for dependency checking, which is not part of the public API for featurization.
