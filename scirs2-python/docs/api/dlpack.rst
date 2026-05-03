DLPack Interop (``scirs2.from_dlpack`` / ``scirs2.to_dlpack``)
===============================================================

SciRS2 exposes a DLPack 1.0 compatible interface for zero-copy tensor
sharing with PyTorch, JAX, CuPy, TensorFlow, and other DLPack-aware
frameworks.

Protocol Overview
-----------------

A *DLPack capsule* is a ``PyCapsule`` object whose name is ``"dltensor"``.
Any framework that supports `PEP 3118 <https://peps.python.org/pep-3118/>`_
or DLPack 1.0 can produce and consume such capsules.

Functions
---------

.. function:: scirs2.from_dlpack(capsule) -> numpy.ndarray

   Convert a DLPack capsule to a scirs2 (NumPy-compatible) array.

   :param capsule: A ``PyCapsule`` named ``"dltensor"``.
                   Obtain one by calling ``tensor.__dlpack__()``.
   :type capsule: PyCapsule
   :returns: A zero-copy CPU array view.
   :rtype: numpy.ndarray
   :raises NotImplementedError: GPU tensors are not yet supported.

.. function:: scirs2.to_dlpack(array) -> PyCapsule

   Export a scirs2 array as a DLPack ``PyCapsule``.

   :param array: A NumPy-compatible array.
   :type array: numpy.ndarray
   :returns: A capsule named ``"dltensor"``.
   :rtype: PyCapsule

Examples
--------

PyTorch round-trip
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch, numpy as np
   import scirs2

   # PyTorch → scirs2
   t = torch.randn(3, 4)
   arr = scirs2.from_dlpack(t.__dlpack__())   # zero-copy CPU view

   # Process with scirs2 / NumPy
   result = np.linalg.norm(arr)

   # scirs2 → PyTorch
   arr2 = np.ones((5, 5), dtype=np.float32)
   cap  = scirs2.to_dlpack(arr2)
   t2   = torch.from_dlpack(cap)

JAX round-trip
~~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp, jax.dlpack as jdlp
   import scirs2

   x_jax = jnp.ones((4, 4))
   arr   = scirs2.from_dlpack(jdlp.to_dlpack(x_jax))

Notes
-----

* GPU tensors (CUDA device) raise ``NotImplementedError`` until the
  optional ``gpu`` feature is enabled.
* The array must remain alive for the lifetime of the capsule.
* After a consumer calls ``from_dlpack``, the capsule is renamed to
  ``"used_dltensor"`` to prevent double-frees.
