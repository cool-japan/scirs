Signal Processing (``scirs2.signal``)
=======================================

Digital filter design, filtering, spectral analysis, and adaptive filters.

.. automodule:: scirs2.signal
   :members:
   :undoc-members:
   :show-inheritance:

Filter Design
-------------

.. code-block:: python

   import scirs2, numpy as np

   # Butterworth low-pass at 0.3 * Nyquist, order 5
   b, a = scirs2.butter_py(5, 0.3, btype="low")

   # Second-order sections (more numerically stable)
   sos = scirs2.butter_py(5, 0.3, btype="low", output="sos")

Available designs: ``butter_py``, ``cheby1_py``, ``cheby2_py``

Filter Application
------------------

.. code-block:: python

   # Direct-form II transposed
   y = scirs2.lfilter_py(b, a, x)

   # SOS form (preferred)
   y = scirs2.sosfilt_py(sos, x)

Spectral Analysis
-----------------

.. code-block:: python

   f, t, Sxx = scirs2.spectrogram_py(x, fs=1000.0)
   f, Pxx   = scirs2.periodogram_py(x, fs=1000.0)
   f, t, Z  = scirs2.stft_py(x, fs=1000.0, nperseg=256)

State Estimation (Extended Module)
-----------------------------------

- ``KalmanFilter`` — Linear Kalman filter
- ``ExtendedKalmanFilter`` — EKF for nonlinear systems
- ``UnscentedKalmanFilter`` — UKF with sigma-point propagation

Adaptive Filters
----------------

- ``LMSFilter(mu, order)`` — Least-mean-squares adaptive filter
- ``RLSFilter(lam, order)`` — Recursive least-squares filter
