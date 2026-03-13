###############################################################
Advanced Splitting API (`molblender.data.dataset.splitting`)
###############################################################

.. automodule:: molblender.data.dataset.splitting
   :members:
   :no-index:
   :undoc-members:
   :show-inheritance:

==========
Overview
==========

The splitting subpackage provides advanced molecular-aware splitting strategies adapted from the `splito package <https://github.com/datamol-io/splito>`_.

Available Methods
=================

Advanced Splitters (Class-based API)
-------------------------------------

.. autoclass:: molblender.data.dataset.splitting.PerimeterSplit
   :members:
   :no-index:
   :show-inheritance:

.. autoclass:: molblender.data.dataset.splitting.MolecularWeightSplit
   :members:
   :no-index:
   :show-inheritance:

.. autoclass:: molblender.data.dataset.splitting.MOODSplitter
   :members:
   :no-index:
   :show-inheritance:

.. autoclass:: molblender.data.dataset.splitting.LoSplitter
   :members:
   :no-index:
   :show-inheritance:

Functional API (sklearn-style)
-------------------------------

.. autofunction:: molblender.data.dataset.splitting.train_test_split
   :no-index:

.. autofunction:: molblender.data.dataset.splitting.train_test_split_indices
   :no-index:

Utility Functions
-----------------

.. autofunction:: molblender.data.dataset.splitting.compute_fingerprints
   :no-index:

.. autofunction:: molblender.data.dataset.splitting.compute_molecular_weights
   :no-index:

.. autofunction:: molblender.data.dataset.splitting.to_mol
   :no-index:

.. autofunction:: molblender.data.dataset.splitting.to_smiles
   :no-index:

.. autofunction:: molblender.data.dataset.splitting.get_kmeans_clusters
   :no-index:

.. autofunction:: molblender.data.dataset.splitting.get_iqr_outlier_bounds
   :no-index:

MolecularDataset Integration
-----------------------------

.. autoclass:: molblender.data.dataset.splitting.SplittingMixin
   :members:
   :no-index:
   :show-inheritance:

==========
References
==========

.. note::
   Advanced splitting strategies adapted from splito package:
   https://github.com/datamol-io/splito

   Copyright (c) 2024 Datamol.io - Apache 2.0 License
