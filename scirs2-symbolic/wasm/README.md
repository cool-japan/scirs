# scirs2-symbolic-wasm

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](../../LICENSE)
[![crates.io](https://img.shields.io/badge/crates.io-not%20published-lightgrey)]()

WebAssembly playground bindings for the [`scirs2-symbolic`](../) computer algebra system (CAS), part of the [SciRS2](https://github.com/cool-japan/scirs) ecosystem.

## Overview

`scirs2-symbolic-wasm` exposes a small, string-in / string-out API over `wasm-bindgen` so a browser (or Node, or any other JS environment) can drive the `scirs2-symbolic` CAS — canonicalization, simplification, symbolic differentiation, numeric evaluation, and structural-equality checks — without needing to understand any Rust types.

It is its own standalone Cargo workspace (it declares its own `[workspace]` table in `Cargo.toml`), so it is never absorbed into the parent `scirs2` workspace build, and it is intentionally **not published** to crates.io (`publish = false`). It exists purely as a demo/playground crate, not a library other crates depend on.

A minimal static demo (`playground/index.html` + `playground/main.js`) exercises the whole API in a browser.

## Expression syntax

Expressions are parsed by a small hand-rolled, iterative (non-recursive) Pratt parser:

| Category | Syntax |
|---|---|
| Variables | `x0`, `x1`, ..., `xN`; `x` is an alias for `x0`; `y` is an alias for `x1` |
| Constants | decimal literals: `3.14`, `2`, `-1` |
| Unary functions | `sin cos tan exp ln sqrt abs sinh cosh tanh arcsin arccos arctan` |
| Binary operators (decreasing precedence) | `^` (right-associative), `* /`, `+ -` |
| Grouping | parentheses `( expr )` |

Canonicalized/simplified output always displays variables in their canonical `x0`/`x1`/... form, even if the input used the `x`/`y` alias.

## WASM API

All five exported functions take and return plain strings; on failure the returned string starts with `"Error: "`.

| Function | Signature | Behavior |
|---|---|---|
| `wasm_canonicalize` | `(expr: &str) -> String` | Canonicalizes the expression, returns its infix display |
| `wasm_simplify` | `(expr: &str) -> String` | Applies constant folding + algebraic rewrites, returns the simplified infix display |
| `wasm_grad` | `(expr: &str, wrt_var: usize) -> String` | Symbolic derivative `d(expr)/d(x_wrt_var)`, returned simplified and in infix form |
| `wasm_eval` | `(expr: &str, bindings_json: &str) -> String` | Numerically evaluates `expr` given a JSON array of `f64` bindings (`Var(0)` ↔ index 0, etc.) |
| `wasm_is_identity` | `(expr1: &str, expr2: &str) -> String` | `"true"` / `"false"` depending on whether the two expressions canonicalize to the same form |

### Usage example

```javascript
import init, * as symbolic from './pkg/scirs2_symbolic_wasm.js';

await init();

symbolic.wasm_canonicalize("ln(exp(x))");           // "x0"
symbolic.wasm_simplify("x + 0");                    // "x0"
symbolic.wasm_grad("x^2 + y^2", 0);                 // "(2 * x0)"   (d/dx0)
symbolic.wasm_eval("x + y", "[3.0, 4.0]");          // "7"
symbolic.wasm_eval("sin(x)^2 + cos(x)^2", "[1.0]"); // "1"
symbolic.wasm_is_identity("x + y", "y + x");        // "true"
symbolic.wasm_is_identity("ln(exp(x))", "x");       // "true"
symbolic.wasm_is_identity("x + 1", "x - 1");        // "false"
```

(All outputs above were checked against the current implementation.)

## Building from source

### Prerequisites

- Rust with the `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/): `cargo install wasm-pack`

### Build

This directory is a standalone Cargo workspace — run commands from inside `scirs2-symbolic/wasm/`, not from the parent `scirs2` workspace root:

```bash
cd scirs2-symbolic/wasm
wasm-pack build --target web
```

This produces a `pkg/` directory with the JS glue (`scirs2_symbolic_wasm.js`), the compiled `.wasm` binary, and type declarations.

`Cargo.toml` pins a direct `getrandom = { version = "0.3", features = ["wasm_js"] }` dependency purely for feature unification: `ahash` (pulled in transitively through `scirs2-symbolic`) depends on `getrandom 0.3`, which needs its `wasm_js` feature enabled to compile for `wasm32-unknown-unknown`, and declaring the dependency directly here (with that feature on) turns it on for the transitive copy too. This is the same conflict, fixed the same way, as the sibling [`scirs2-wasm`](../../scirs2-wasm/) crate's `getrandom_v3` workspace dependency.

### Native checks and tests

Since this is its own workspace, `cargo check` and `cargo test` compile for your native host target and exercise the parser and all five `wasm_*` functions directly as ordinary Rust — no browser or `wasm-pack test` needed:

```bash
cd scirs2-symbolic/wasm
cargo check
cargo test --lib   # 15 unit tests covering the parser and the wasm_* API
```

## Try the playground demo

`playground/index.html` is a small, dependency-free static page with five panels — Canonicalize, Simplify, Symbolic Gradient, Numeric Evaluation, and Canonical Identity Check — wired up by `playground/main.js`. It loads the compiled module from `./pkg/scirs2_symbolic_wasm.js` (a path relative to `main.js` itself), so:

1. Build the wasm package so its output lands under `playground/pkg/`:
   ```bash
   wasm-pack build --target web --out-dir playground/pkg
   ```
2. Serve `scirs2-symbolic/wasm/` with any local HTTP server — the page uses an ES module `import`, which browsers refuse to resolve under `file://` — e.g.:
   ```bash
   python3 -m http.server 8080
   ```
3. Open `http://localhost:8080/playground/index.html`.

## Project layout

```
scirs2-symbolic/wasm/
├── Cargo.toml       - standalone workspace manifest, publish = false
├── src/
│   └── lib.rs        - wasm-bindgen exports, Pratt parser, native unit tests
└── playground/
    ├── index.html     - static demo UI (5 panels)
    └── main.js        - wires the UI to the wasm_* API, loads ./pkg/
```

## Related projects

- [`scirs2-symbolic`](../) — the CAS this crate binds to
- [`scirs2-wasm`](../../scirs2-wasm/) — general-purpose SciRS2 WASM bindings (linear algebra, stats, FFT, signal processing, ...)
- [`wasm-bindgen`](https://github.com/rustwasm/wasm-bindgen) — the WASM/JS interop framework used here

## License

Licensed under the Apache License 2.0. See [LICENSE](../../LICENSE) for details.

## Authors

COOLJAPAN OU (Team Kitasan)
