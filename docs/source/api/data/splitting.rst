###############################################################
Advanced Splitting API (`molblender.data.dataset.splitting`)
###############################################################

.. automodule:: molblender.data.dataset.splitting
   :members:
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
   :show-inheritance:

.. autoclass:: molblender.data.dataset.splitting.MolecularWeightSplit
   :members:
   :show-inheritance:

.. autoclass:: molblender.data.dataset.splitting.MOODSplitter
   :members:
   :show-inheritance:

.. autoclass:: molblender.data.dataset.splitting.LoSplitter
   :members:
   :show-inheritance:

Functional API (sklearn-style)
-------------------------------

.. autofunction:: molblender.data.dataset.splitting.train_test_split

.. autofunction:: molblender.data.dataset.splitting.train_test_split_indices

Utility Functions
-----------------

.. autofunction:: molblender.data.dataset.splitting.compute_fingerprints

.. autofunction:: molblender.data.dataset.splitting.compute_molecular_weights

.. autofunction:: molblender.data.dataset.splitting.to_mol

.. autofunction:: molblender.data.dataset.splitting.to_smiles

.. autofunction:: molblender.data.dataset.splitting.get_kmeans_clusters

.. autofunction:: molblender.data.dataset.splitting.get_iqr_outlier_bounds

MolecularDataset Integration
-----------------------------

.. autoclass:: molblender.data.dataset.splitting.SplittingMixin
   :members:
   :show-inheritance:

==========
References
==========

.. note::
   Advanced splitting strategies adapted from splito package:
   https://github.com/datamol-io/splito

   Copyright (c) 2024 Datamol.io - Apache 2.0 License
