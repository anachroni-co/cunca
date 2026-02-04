Utilities API
=============

This module provides utility functions and classes for various tasks including
caching, checkpointing, monitoring, memory profiling, and more.

utils
-----

.. automodule:: utils
   :members:
   :undoc-members:
   :show-inheritance:


Memory Profiler
---------------

The memory profiler module provides advanced memory profiling utilities for
detecting memory leaks, tracking allocations, and generating reports.

.. automodule:: utils.memory_profiler
   :members:
   :undoc-members:
   :show-inheritance:

Classes
~~~~~~~

.. autoclass:: utils.memory_profiler.MemoryProfiler
   :members:
   :special-members: __init__

.. autoclass:: utils.memory_profiler.MemorySnapshot
   :members:

.. autoclass:: utils.memory_profiler.TrainingMemoryTracker
   :members:
   :special-members: __init__

Functions
~~~~~~~~~

.. autofunction:: utils.memory_profiler.get_profiler

.. autofunction:: utils.memory_profiler.profile_memory

.. autofunction:: utils.memory_profiler.memory_profile_block

.. autofunction:: utils.memory_profiler.check_for_leaks

.. autofunction:: utils.memory_profiler.print_memory_summary


Monitoring
----------

Real-time resource monitoring utilities.

.. automodule:: utils.monitoring
   :members:
   :undoc-members:
   :show-inheritance:


Checkpoint Manager
------------------

.. automodule:: utils.checkpoint_manager
   :members:
   :undoc-members:
   :show-inheritance:


Cache
-----

.. automodule:: utils.cache_standalone
   :members:
   :undoc-members:
   :show-inheritance:


Logging Utilities
-----------------

.. automodule:: utils.logging_utils
   :members:
   :undoc-members:
   :show-inheritance:


Validation Utilities
--------------------

.. automodule:: utils.validation_utils
   :members:
   :undoc-members:
   :show-inheritance:
