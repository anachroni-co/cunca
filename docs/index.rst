.. CapibaraGPT documentation master file

CapibaraGPT Documentation
=========================

Welcome to CapibaraGPT's documentation! CapibaraGPT is a modular, high-performance
language model framework designed for research and production use.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   guides/getting-started
   guides/configuration
   guides/adding-layers

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/overview
   architecture/backends
   architecture/layers
   architecture/module-gate

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/core
   api/layers
   api/training
   api/utils
   api/benchmarks

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing/code-standards
   contributing/factorization-rules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Quick Links
-----------

* **GitHub Repository**: https://github.com/capibara/capibaraGPT
* **Issue Tracker**: https://github.com/capibara/capibaraGPT/issues


Features
--------

* **Multi-Backend Support**: PyTorch, JAX, and CPU fallback
* **Modular Architecture**: Easy to extend and customize
* **High Performance**: Optimized for TPU and GPU training
* **Memory Efficient**: Built-in memory profiling and optimization
* **Comprehensive Testing**: Extensive test coverage


Installation
------------

.. code-block:: bash

   pip install capibaraGPT

Or install from source:

.. code-block:: bash

   git clone https://github.com/capibara/capibaraGPT.git
   cd capibaraGPT
   pip install -e .


Quick Example
-------------

.. code-block:: python

   from core.backends import get_backend
   from core.layers import TransformerBlock

   # Get the appropriate backend
   backend = get_backend()

   # Create a transformer block
   block = TransformerBlock(
       hidden_size=768,
       num_heads=12,
       mlp_ratio=4.0
   )

   # Process input
   output = block(input_tensor)


Getting Help
------------

If you need help:

1. Check the :doc:`guides/getting-started` guide
2. Browse the :doc:`api/index` reference
3. Open an issue on GitHub


License
-------

CapibaraGPT is released under the MIT License.
