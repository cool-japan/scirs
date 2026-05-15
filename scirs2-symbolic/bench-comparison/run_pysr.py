#!/usr/bin/env python3
"""
PySR benchmark runner for FSReD equations.

Run manually — requires Python 3.11+, Julia 1.10+, and PySR installed::

    pip install pysr

This script runs PySR on the same 20 FSReD equations as
``scirs2-symbolic/benches/fsred_bench.rs`` and writes results to
``bench-comparison/pysr_results.json``.

NOT run in CI — Julia toolchain not available in standard CI environment.

Suggested usage::

    python bench-comparison/run_pysr.py
    python bench-comparison/compare.py target/fsred_results.json \\
        bench-comparison/pysr_results.json
"""

# Suggested structure for manual implementation
# (requires Python 3.11+, Julia 1.10+, and PySR — not run in CI):
#
#   import math, json, time
#   import numpy as np
#   from pysr import PySRRegressor
#
#   def generate_data(f, n_features, n_samples=1000, seed=42):
#       rng = np.random.default_rng(seed)
#       X = rng.uniform(1.0, 5.0, size=(n_samples, n_features))
#       y = np.array([f(*row) for row in X])
#       mask = np.isfinite(y)
#       return X[mask], y[mask]
#
#   EQUATIONS = [
#       {"name": "I.6.2a",            "n_features": 1,
#        "f": lambda x: math.exp(-x**2/2) / math.sqrt(2*math.pi)},
#       {"name": "sum_squares",       "n_features": 2,
#        "f": lambda x, y: x**2 + y**2},
#       # ... (mirror the 20 equations in fsred_bench.rs)
#   ]
#
#   results = []
#   for eq in EQUATIONS:
#       X, y = generate_data(eq["f"], eq["n_features"])
#       model = PySRRegressor(
#           niterations=40,
#           binary_operators=["+", "-", "*", "/"],
#           unary_operators=["sin", "cos", "exp", "log", "sqrt"],
#           verbosity=0,
#       )
#       t0 = time.perf_counter()
#       model.fit(X, y)
#       elapsed_ms = (time.perf_counter() - t0) * 1000.0
#       best = model.get_best()
#       results.append({
#           "name": eq["name"],
#           "best_expr": str(best["sympy_format"]),
#           "mse": float(best["loss"]),
#           "elapsed_ms": elapsed_ms,
#       })
#       print(f"  {eq['name']:25s}  mse={best['loss']:.3e}  t={elapsed_ms:.0f}ms")
#
#   with open("bench-comparison/pysr_results.json", "w") as fp:
#       json.dump(results, fp, indent=2)
#   print("Wrote bench-comparison/pysr_results.json")


if __name__ == "__main__":
    raise NotImplementedError(
        "Run manually with Julia 1.10+ and PySR installed.  See README.md."
    )
