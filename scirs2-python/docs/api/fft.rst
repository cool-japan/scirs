FFT (``scirs2.fft``)
====================

Fast Fourier Transform functions powered by OxiFFT — a pure-Rust FFT
engine.  All functions operate on NumPy arrays.

.. automodule:: scirs2.fft
   :members:
   :undoc-members:
   :show-inheritance:

Function Reference
------------------

Core Transforms
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``fft_py(x, n)``
     - 1-D discrete Fourier transform (complex input)
   * - ``ifft_py(x, n)``
     - 1-D inverse discrete Fourier transform
   * - ``rfft_py(x, n)``
     - 1-D FFT of a real-valued signal (half spectrum)
   * - ``irfft_py(x, n)``
     - 1-D inverse of ``rfft_py``
   * - ``dct_py(x, type)``
     - Discrete Cosine Transform (type I–IV)
   * - ``idct_py(x, type)``
     - Inverse Discrete Cosine Transform

Utilities
~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``fftfreq_py(n, d)``
     - Frequencies for ``fft_py`` output
   * - ``rfftfreq_py(n, d)``
     - Frequencies for ``rfft_py`` output
   * - ``fftshift_py(x)``
     - Shift zero-frequency component to centre
   * - ``ifftshift_py(x)``
     - Inverse of ``fftshift_py``
   * - ``next_fast_len_py(n, real)``
     - Next efficient FFT input length

Examples
--------

.. code-block:: python

   import numpy as np
   import scirs2

   # Real FFT
   signal = np.sin(2 * np.pi * np.linspace(0, 1, 256))
   spectrum = scirs2.rfft_py(signal)
   freqs   = scirs2.rfftfreq_py(len(signal), d=1/256)

   # Reconstruct
   recovered = scirs2.irfft_py(spectrum.real, spectrum.imag, len(signal))
