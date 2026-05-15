#!/usr/bin/env python3
"""
Compare Rust (scirs2-symbolic) vs PySR FSReD benchmark results.

Usage::

    python bench-comparison/compare.py \\
        target/fsred_results.json \\
        bench-comparison/pysr_results.json

Both files must exist.  Run ``cargo bench --bench fsred_bench`` to produce
the Rust results, and ``python bench-comparison/run_pysr.py`` for PySR.

NOT run in CI.
"""

import sys
import json


def load(path: str) -> dict:
    with open(path) as fp:
        rows = json.load(fp)
    return {r["name"]: r for r in rows}


def compare(rust_path: str, pysr_path: str) -> None:
    rust = load(rust_path)
    pysr = load(pysr_path)

    names = sorted(set(rust) | set(pysr))
    header = f"{'Name':<25}  {'Rust MSE':>12}  {'PySR MSE':>12}  {'Rust Rec':>9}  {'PySR Rec':>9}"
    print(header)
    print("-" * len(header))

    rust_rec = 0
    pysr_rec = 0
    for name in names:
        r = rust.get(name, {})
        p = pysr.get(name, {})
        rust_mse = r.get("mse", float("inf"))
        pysr_mse = p.get("mse", float("inf"))
        rr = r.get("recovered", False)
        pr = p.get("mse", float("inf")) < 1e-3
        if rr:
            rust_rec += 1
        if pr:
            pysr_rec += 1
        print(
            f"{name:<25}  {rust_mse:>12.3e}  {pysr_mse:>12.3e}"
            f"  {str(rr):>9}  {str(pr):>9}"
        )

    print()
    print(f"Recovered: Rust {rust_rec}/{len(names)},  PySR {pysr_rec}/{len(names)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
