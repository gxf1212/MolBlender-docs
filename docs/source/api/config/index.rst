Configuration
=============

Global configuration and dependency management.

.. toctree::
   :maxdepth: 2

   dependencies
   settings

Overview
--------

The config module manages:

* **Dependencies**: Unified system for optional packages (RDKit, DeepChem, PyTorch, etc.)
* **Settings**: Global paths, cache directories, environment variables

Example
-------

.. code-block:: python

   from molblender.config import dependencies as deps
   
   # Check availability
   if deps.is_rdkit_available():
       rdkit = deps.get_rdkit()
       Chem = rdkit['Chem']
   
   # Get settings
   from molblender.config import settings
   print(settings.CACHE_DIR)

See Also
--------

* :doc:`../../development/architecture` - System design
* :doc:`../../development/setup` - Development setup
