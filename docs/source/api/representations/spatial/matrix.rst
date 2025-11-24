.. default-role:: py:obj
.. automodule:: molblender.representations.spatial.matrix
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: BaseFeaturizer 
   :exclude-members: BOND_ORDER_MAP 

Matrix-Based Spatial Representations
====================================

This module provides featurizers that encode molecular structures as various types of matrices. These are essential for capturing spatial relationships, electrostatic interactions, and connectivity in a format suitable for machine learning.

For usage examples, please refer to the :doc:`/usage/representations/spatial/matrix` guide.

Featurizer Classes
------------------

The following classes are available for generating matrix-based representations:

.. py:class:: CoulombMatrix(max_atoms: int = 30, remove_hydrogens: bool = False, permutation: str = 'sorted_l2', upper_tri: bool = False, sigma: Optional[float] = None, seed: Optional[int] = None, flatten: bool = True, backend: Optional[str] = None, **kwargs)
   :noindex:

   Computes the Coulomb matrix for a molecule, representing electrostatic interactions.
   Requires 3D coordinates. Can use "deepchem" or "dscribe" as a backend.
   Inherits from :py:class:`~molblender.representations.utils.base.BaseFeaturizer`.

   .. automethod:: _featurize
   .. automethod:: get_feature_names
      *Overridden to reflect that feature names are typically not applicable or are highly numerous and non-standard for raw matrices.*
   .. automethod:: get_output_info


.. py:class:: CoulombMatrixEig(max_atoms: int = 30, remove_hydrogens: bool = False, backend: Optional[str] = None, seed: Optional[int] = None, **kwargs)
   :noindex:

   Computes the eigenvalues of the Coulomb matrix, providing a permutation-invariant descriptor.
   Requires 3D coordinates.
   Inherits from :py:class:`CoulombMatrix`.

   .. automethod:: _featurize 
      *(Inherited and specialized via constructor parameters passed to CoulombMatrix)*
   .. automethod:: get_output_info


.. py:class:: AdjacencyMatrix(max_atoms: int = 30, remove_hydrogens: bool = True, flatten: bool = True, **kwargs)
   :noindex:

   Generates the adjacency matrix, where an element (i,j) is 1 if atoms i and j are bonded, 0 otherwise.
   Inherits from :py:class:`~molblender.representations.utils.base.BaseFeaturizer`.

   .. automethod:: _featurize
   .. automethod:: get_output_info


.. py:class:: EdgeMatrix(max_atoms: int = 30, remove_hydrogens: bool = True, flatten: bool = True, **kwargs)
   :noindex:

   Generates the edge matrix (bond order matrix), where element (i,j) represents the bond order between atoms i and j.
   Inherits from :py:class:`~molblender.representations.utils.base.BaseFeaturizer`.

   .. automethod:: _featurize
   .. automethod:: get_output_info

Module Constants
----------------

.. py:data:: BOND_ORDER_MAP
   :noindex:

   A dictionary mapping RDKit bond types (as numeric values) to their corresponding bond order values used in the EdgeMatrix.
   Example: `{1: 1.0, 2: 2.0, 3: 3.0, 1.5: 1.5}`

Helper Functions
----------------

The module also contains internal helper functions for dependency checking (e.g., `_ensure_rdkit_loaded`, `_ensure_deepchem_loaded`, `_ensure_dscribe_loaded`), which are not part of the public API for featurization.

