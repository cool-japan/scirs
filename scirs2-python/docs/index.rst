SciRS2 Python Bindings
======================

**SciRS2** is a comprehensive scientific computing library written in pure Rust,
exposed to Python through high-performance PyO3 bindings.  It provides a
SciPy-compatible API with Rust-level performance and zero-copy NumPy integration
via ``scirs2-numpy``.

.. code-block:: python

   import numpy as np
   import scirs2

   data = np.random.randn(1000)

   # Statistics — up to 23x faster than SciPy on small datasets
   print(scirs2.skew_py(data))
   print(scirs2.kurtosis_py(data))

   # DLPack interop with PyTorch / JAX
   import torch
   t = torch.ones(3, 4)
   arr = scirs2.from_dlpack(t.__dlpack__())

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
