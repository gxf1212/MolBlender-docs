.. _api-representations-fingerprints-rdkit:

RDKit Fingerprints (`rdkit`)
============================

This module provides featurizer classes that wrap various fingerprint generation algorithms from the RDKit library. Specific configurations (e.g., fingerprint size, radius) are typically handled during instantiation or via predefined keys in the central registry.

.. automodule:: molblender.representations.fingerprints.rdkit
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: BaseFeaturizer
   :exclude-members: BaseRDKitFingerprint, RDKIT_FP_CONFIG, DEFAULT_SPARSE_FP_SIZE, _convert_fp_to_numpy, _RDKIT_AVAILABLE, _UTILS_AVAILABLE, _EXPECTED_CONFIG_NAMES

   .. rubric:: Core Featurizer Classes

   The following classes provide the core implementation for different RDKit fingerprint types. You typically instantiate these classes directly if you need non-default parameters, or use :func:`~molblender.get_featurizer` with a registered key (like ``"morgan_fp_r2_1024"``) which handles instantiation with default parameters.

   .. autoclass:: RDKitTopologicalFP
      :members: __init__
      :noindex:

   .. autoclass:: MorganBitFP
      :members: __init__
      :noindex:

   .. autoclass:: MorganCountFP
      :members: __init__
      :noindex:

   .. autoclass:: MACCSKeysFP
      :members: __init__
      :noindex:

   .. autoclass:: AtomPairBitFP
      :members: __init__
      :noindex:

   .. autoclass:: AtomPairCountFP
      :members: __init__
      :noindex:

   .. autoclass:: TorsionBitFP
      :members: __init__
      :noindex:

   .. autoclass:: TorsionCountFP
      :members: __init__
      :noindex:

   .. # The RDKIT_FP_CONFIG dictionary might be an internal detail now,
   .. # consider removing it from the public API documentation unless intended for users.
   .. # .. rubric:: Configuration (Internal Reference)
   .. # .. autodata:: RDKIT_FP_CONFIG
   .. #    :annotation:

