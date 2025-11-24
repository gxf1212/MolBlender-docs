Settings
========

Global settings and paths configuration.

.. currentmodule:: molblender.config.settings

Overview
--------

The settings module manages global configuration including:

* Cache directories
* Model download paths
* Logging configuration
* Environment variables

Settings Variables
------------------

.. automodule:: molblender.config.settings
   :members:
   :undoc-members:

Key Settings
------------

**Cache Directory**
  Location for cached representations::
  
    from molblender.config import settings
    print(settings.CACHE_DIR)  # ~/.molblender/cache

**Model Paths**
  Paths for downloading pre-trained models::
  
    TORCH_HOME  # PyTorch models
    HF_HOME     # HuggingFace models

**Logging**
  Global logging configuration::
  
    LOG_LEVEL    # DEBUG, INFO, WARNING, ERROR
    LOG_FORMAT   # Log message format

Environment Variables
---------------------

MolBlender respects these environment variables:

* ``MOLBLENDER_CACHE_DIR`` - Override default cache location
* ``TORCH_HOME`` - PyTorch model cache
* ``HF_HOME`` - HuggingFace model cache
* ``NUMEXPR_MAX_THREADS`` - NumExpr thread limit

Example
-------

.. code-block:: python

   from molblender.config import settings
   
   # Access settings
   print(f"Cache dir: {settings.CACHE_DIR}")
   print(f"Log level: {settings.LOG_LEVEL}")
   
   # Set via environment
   import os
   os.environ['MOLBLENDER_CACHE_DIR'] = '/custom/cache/path'

See Also
--------

* :doc:`dependencies` - Dependency management
* :doc:`../../development/setup` - Development setup
