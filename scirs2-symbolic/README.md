# scirs2-symbolic

[![crates.io](https://img.shields.io/crates/v/scirs2-symbolic)](https://crates.io/crates/scirs2-symbolic)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-symbolic)](https://docs.rs/scirs2-symbolic)

**Symbolic mathematics for the SciRS2 scientific computing ecosystem.**

`scirs2-symbolic` provides symbolic expression trees, automatic symbolic differentiation,
algebraic simplification, and numeric evaluation of symbolic expressions — a computer algebra
system (CAS) component designed to complement the numeric capabilities of SciRS2.

## Installation

```toml
[dependencies]
scirs2-symbolic = "0.4.4"
```

## Quick Start

```rust
use scirs2_symbolic::{Expr, diff, simplify, eval};
use std::collections::HashMap;

// Build the expression  f(x) = x² + 3x
let x = Expr::var("x");
let f = x.clone().pow(Expr::from(2.0)) + Expr::from(3.0) * x.clone();

// Differentiate symbolically:  f'(x) = 2x + 3
let df = simplify(&diff(&f, "x"));
println!("f'(x) = {}", df);  // 2*x + 3

// Evaluate at x = 2:  2*2 + 3 = 7
let mut vars = HashMap::new();
vars.insert("x", 2.0_f64);
let result = eval(&df, &vars).unwrap();
assert!((result - 7.0).abs() < 1e-10);
```

## Features

### Expression Trees

Build symbolic expressions using the `Expr` enum and standard arithmetic operators:

```rust
use scirs2_symbolic::Expr;

let x = Expr::var("x");
let y = Expr::var("y");

// Arithmetic
let sum = x.clone() + y.clone();
let product = x.clone() * Expr::from(2.0);
let quotient = y.clone() / Expr::from(3.0);
let power = x.clone().pow(Expr::from(3.0));

// Transcendental functions
let s = x.clone().sin();
let c = x.clone().cos();
let e = x.clone().exp();
let l = x.clone().ln();
```

### Symbolic Differentiation

Compute exact symbolic derivatives with `diff` and higher-order derivatives with `diff_n`:

```rust
use scirs2_symbolic::{Expr, diff, diff_n, simplify};

let x = Expr::var("x");

// f(x) = sin(x)
let f = x.clone().sin();

// f'(x) = cos(x)
let df = simplify(&diff(&f, "x"));

// f''(x) = -sin(x)
let d2f = simplify(&diff_n(&f, "x", 2));
```

### Algebraic Simplification

Reduce expressions with constant folding and identity rules:

```rust
use scirs2_symbolic::{Expr, simplify, simplify_full};

let x = Expr::var("x");

// Constant folding: 2 + 3 → 5
let e1 = Expr::from(2.0) + Expr::from(3.0);
assert_eq!(format!("{}", simplify(&e1)), "5");

// Identity rules: x * 1 → x, x + 0 → x
let e2 = x.clone() * Expr::from(1.0);
assert_eq!(format!("{}", simplify(&e2)), "x");

// Full simplification (multiple passes)
let e3 = simplify_full(&(x.clone().pow(Expr::from(1.0))));
assert_eq!(format!("{}", e3), "x");
```

### Numeric Evaluation

Evaluate symbolic expressions numerically by substituting variable bindings:

```rust
use scirs2_symbolic::{Expr, eval};
use std::collections::HashMap;

let x = Expr::var("x");
let expr = x.clone().sin() + x.clone().cos();

let mut vars = HashMap::new();
vars.insert("x", std::f64::consts::PI / 4.0);

let result = eval(&expr, &vars).expect("evaluation failed");
// sin(π/4) + cos(π/4) ≈ √2
assert!((result - std::f64::consts::SQRT_2).abs() < 1e-10);
```

### LaTeX Export

Render any `LoweredOp` expression as a LaTeX math string with `to_latex`:

```rust
use scirs2_symbolic::eml::{to_latex, LoweredOp};
use std::f64::consts::PI;

// π / (x₀² + 1)
let op = LoweredOp::Div(
    Box::new(LoweredOp::Const(PI)),
    Box::new(LoweredOp::Add(
        Box::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        )),
        Box::new(LoweredOp::Const(1.0)),
    )),
);
assert_eq!(to_latex(&op), "\\frac{\\pi}{\\left(x_{0}^{2} + 1\\right)}");
```

Recognised constants: `\pi`, `e`. Operators: `\frac`, `\cdot`, `a^{b}`, `\sqrt`, `\left|\right|`, `\operatorname{arcsinh}` etc. All traversals are iterative — no stack overflow on deeply-nested expressions.

## Module Overview

| Module | Description |
|--------|-------------|
| [`expr`] | `Expr` enum, arithmetic operator overloads, `var`/`from` constructors |
| [`diff`] | Symbolic differentiation: `diff(expr, var)`, `diff_n(expr, var, n)` |
| [`simplify`] | Constant folding + identity rules: `simplify`, `simplify_full` |
| [`eval`] | Numeric evaluation: `eval(expr, bindings)` |
| [`display`] | `Display` impl for human-readable infix notation |
| [`eml::display`] | `Display` for EML IR + [`eml::to_latex`] — render any expression as LaTeX |
| [`error`] | `SymbolicError` error type variants |

## Design Notes

- **Pure Rust, no external dependencies** beyond `thiserror` for error derives.
- **No `unwrap()`** in production code — all fallible paths return `Result<_, SymbolicError>`.
- Expression trees are **immutable** `Clone`-able values with no shared mutable state,
  making them thread-safe by construction.
- Implements `Display` for human-readable infix notation.

## Optional Features

| Feature | Pulls in | Purpose |
|---------|----------|---------|
| `serde` | `serde`, `serde_json`, `oxicode` | Round-trip serialization of `EmlTree`, `LoweredOp`, `Interval` (JSON + binary) |
| `smt`   | `oxiz` | SMT-pruned symbolic regression and certified rewrite engine (v0.4.5+) |

Default features are empty — the crate is 100% Pure Rust with zero C/Fortran dependencies in the default build.

## Dependency-Cycle Rule

`scirs2-symbolic` MUST NOT appear in the dependency tree of `scirs2-core` (or any of its direct deps). This is enforced by `scripts/check-no-symbolic-in-core.sh` in CI. The reason: `scirs2-core` is the universal substrate; if it grew a dep on `scirs2-symbolic`, every CAS bug would become a workspace-wide compile failure. `scirs2-symbolic` may depend on `scirs2-core`, never the reverse.

## Part of SciRS2

`scirs2-symbolic` is part of the [SciRS2](https://github.com/cool-japan/scirs) ecosystem — a Rust port of SciPy with AI/ML extensions.

- [SciRS2 main documentation](https://docs.rs/scirs2)
- [GitHub repository](https://github.com/cool-japan/scirs)
- [Changelog](../CHANGELOG.md)
- [CAS Tutorial](docs/cas_tutorial.md) — end-to-end: SR → canonicalize → differentiate → JIT → deploy

## Cycle-Prevention CI Gate

The script `scripts/check-no-symbolic-in-core.sh` enforces two ADR-0001 rules:
1. `scirs2-core` MUST NOT depend on `scirs2-symbolic` (would create a workspace-wide cycle).
2. `oxieml` MUST NOT appear as a production dep of any workspace crate (it is `[dev-dependencies]` only).

Run locally before submitting PRs:

```bash
bash scripts/check-no-symbolic-in-core.sh
```

Exit codes: 0 = pass; 1 = rule 1 violated; 2 = rule 2 violated; 3 = tool missing.

## License

Apache-2.0 — see [LICENSE](../LICENSE).
