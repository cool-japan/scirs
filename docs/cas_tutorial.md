# SciRS2 Symbolic CAS Tutorial

A hands-on guide to the `scirs2-symbolic` Computer Algebra System (CAS).
Each section covers one capability layer and corresponds to a compile-verified
test in `scirs2-symbolic/tests/cas_tutorial_compile.rs`.

**Crate:** `scirs2-symbolic` (version 0.5.0)  
**Feature flags:**
- `jit` — Cranelift CPU JIT compilation (Section 8)
- `gpu` — WGSL GPU codegen (Section 9)
- `smt` — OxiZ SMT solver integration (Section 4)
- `serde` — serde/JSON serialization
- `numa` — NUMA-aware parallel symbolic regression
- `macros` — `eml_pattern!` / `eml_template!` procedural macros

**Quick build:**
```
cargo add scirs2-symbolic
cargo add scirs2-symbolic --features jit,smt,serde
```

---

## Table of Contents

1. [Hello, EML — the one binary operator](#section-1-hello-eml)
2. [Parsing and Pretty-Printing](#section-2-parsing-and-pretty-printing)
3. [Canonicalization](#section-3-canonicalization)
4. [Identity Database and SMT Verification](#section-4-identity-database-and-smt-verification)
5. [Solving Equations and ODEs](#section-5-solving-equations-and-odes)
6. [Integration (Risch-LITE)](#section-6-integration-risch-lite)
7. [Differentiation — GradGraph and AD](#section-7-differentiation-gradgraph-and-ad)
8. [JIT Compilation (Cranelift)](#section-8-jit-compilation-cranelift)
9. [GPU Dispatch (WGSL)](#section-9-gpu-dispatch-wgsl)
10. [Python and WASM Bindings](#section-10-python-and-wasm-bindings)
11. [Cross-Crate Integration](#section-11-cross-crate-integration)
12. [Differential Geometry Mini-Example](#section-12-differential-geometry-mini-example)
13. [Reference and Next Steps](#section-13-reference-and-next-steps)

---

## Section 1: Hello, EML

### What is EML?

The `scirs2-symbolic` CAS is built on the **Elementary Mathematical Library (EML)**
substrate, introduced by Odrzywolek (2026, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)).
The key insight is that every elementary function — `sin`, `cos`, `exp`, `ln`,
`+`, `*`, and so on — can be expressed using a **single binary operator**:

```
eml(x, y) = exp(x) - ln(y)
```

plus the constant `1`. This uniform representation gives the CAS three
properties it would lack with a large switch-case enumeration of function
forms:

1. **Uniform tree structure.** Every sub-expression has the same structural
   type: `EmlNode`. Pattern matching, gradient computation, and JIT codegen
   all operate on a single recursive case.

2. **Guaranteed differentiability.** The chain and product rules for `eml`
   reduce to two steps, so the symbolic gradient engine needs no special cases
   for individual functions.

3. **Hash-equivalence as mathematical equality.** Two expressions that
   canonicalize to the same `EmlNode` tree are mathematically equal on the
   decidable subset (see Section 3). The structural hash is a sound equality
   witness.

### The `LoweredOp` IR

In practice, the CAS works on a *lowered* intermediate representation called
`LoweredOp`. This is a flat algebraic tree whose variants name common operators
explicitly (`Add`, `Mul`, `Sin`, `Exp`, …) for performance and readability.
Internally the EML encoding is always recoverable via the `lower` / `raise`
functions.

```rust
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// f(x) = x² + 3x   where x = Var(0)
let x = LoweredOp::Var(0);
let f = LoweredOp::Add(
    Box::new(LoweredOp::Pow(
        Box::new(x.clone()),
        Box::new(LoweredOp::Const(2.0)),
    )),
    Box::new(LoweredOp::Mul(
        Box::new(LoweredOp::Const(3.0)),
        Box::new(x.clone()),
    )),
);

// Evaluate at x = 2:  4 + 6 = 10
let ctx = EvalCtx::new(&[2.0]);
let val = eval_real(&f, &ctx).expect("eval f at x=2");
assert!((val - 10.0).abs() < 1e-10);
```

`LoweredOp` is `Clone`, `PartialEq`, and (with the `serde` feature)
serializable. Expression trees are **immutable** values — no shared mutable
state, safe to use across threads.

### The E of EML in Practice

The `LoweredOp::Exp` variant directly represents `e^x`:

```rust
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

let ex = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
let e_val = eval_real(&ex, &EvalCtx::new(&[1.0])).expect("e^1");
assert!((e_val - std::f64::consts::E).abs() < 1e-10);
```

The EML binary operator `eml(x, y)` is accessed at the `EmlTree` level via
`EmlTree::eml(x, y)`. The `LoweredOp` variants are the *practical API*;
`EmlTree` is the *semantically pure EML*.

### Variable Indexing Convention

Variables are named by `usize` index:
- `Var(0)` — the first input variable (`x` in most examples)
- `Var(1)` — the second input variable (`y`, `t`, etc.)
- Higher indices for additional variables or integration constants

The `EvalCtx::new(&[v0, v1, ...])` call binds index 0 → `v0`, index 1 → `v1`,
and so on. Out-of-bounds access returns an error (no silent NaN).

---

## Section 2: Parsing and Pretty-Printing

### EML Text Parser

The `eml::parser::parse` function accepts the core EML grammar:

```
expr  ::= "1" | var | eml_call
var   ::= ("x" | "X") digit+
eml_call ::= ("eml" | "E") "(" expr "," expr ")"
```

This is intentionally minimal: the parser is for loading EML archives and
for use in the WASM playground. Complex formulas are built programmatically
via `Canonical::*` constructors or `LoweredOp` directly.

```rust
use scirs2_symbolic::eml::parser::{parse, to_compact_string};

// Parse the EML encoding of e - ln(x0)
let tree = parse("eml(x0, 1)").expect("parse eml(x0, 1)");
// tree represents exp(x0) - ln(1) = exp(x0)

// Round-trip back to compact text
let compact = to_compact_string(&tree);
println!("compact: {compact}");  // "eml(x0,1)" or equivalent

// Parse a bare variable
let x_tree = parse("x0").expect("parse x0");
let x_compact = to_compact_string(&x_tree);
println!("x0 compact: {x_compact}");
```

### LaTeX Export

Any `LoweredOp` can be rendered as a LaTeX math string via `to_latex`:

```rust
use scirs2_symbolic::eml::{to_latex, LoweredOp};

// x²
let x_sq = LoweredOp::Pow(
    Box::new(LoweredOp::Var(0)),
    Box::new(LoweredOp::Const(2.0)),
);
let latex = to_latex(&x_sq);
assert!(latex.contains("x_{0}") && latex.contains("2"));
// → "x_{0}^{2}"

// sin(x)
let sin_x = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
let sin_latex = to_latex(&sin_x);
assert!(sin_latex.contains("sin"));
// → "\\sin\\left(x_{0}\\right)"

// 1/x
let recip = LoweredOp::Div(
    Box::new(LoweredOp::Const(1.0)),
    Box::new(LoweredOp::Var(0)),
);
assert_eq!(to_latex(&recip), "\\frac{1}{x_{0}}");

// sqrt(x)
let sqrt_x = LoweredOp::Sqrt(Box::new(LoweredOp::Var(0)));
assert_eq!(to_latex(&sqrt_x), "\\sqrt{x_{0}}");
```

The LaTeX renderer recognizes π (`≈ 3.14159...`) and e (`≈ 2.71828...`)
as exact constants (within 1e-12) and renders them as `\pi` and `e`.

### Display (Infix Notation)

`LoweredOp` implements `Display` via the `eml::display` module, producing
human-readable infix notation:

```rust
use scirs2_symbolic::eml::LoweredOp;

let f = LoweredOp::Add(
    Box::new(LoweredOp::Var(0)),
    Box::new(LoweredOp::Const(1.0)),
);
let display_str = format!("{f}");
println!("f = {display_str}");  // e.g., "(x_0 + 1)"
assert!(!display_str.is_empty());
```

### Serde / JSON Serialization

With the `serde` feature enabled, `LoweredOp` can be serialized to JSON:

```toml
# Cargo.toml
scirs2-symbolic = { version = "0.5", features = ["serde"] }
```

```rust,no_run
use scirs2_symbolic::eml::LoweredOp;

let f = LoweredOp::Add(
    Box::new(LoweredOp::Var(0)),
    Box::new(LoweredOp::Const(2.718)),
);
let json = serde_json::to_string(&f).expect("serialize");
let f2: LoweredOp = serde_json::from_str(&json).expect("deserialize");
assert_eq!(f, f2);
```

---

## Section 3: Canonicalization

### The `canonicalize` Function

`cas::canonicalize` reduces a `LoweredOp` to a **canonical form** in which
structurally distinct but mathematically equal expressions produce the same
tree (and the same u128 structural hash). The pipeline runs to fixed point:

1. `simplify_op` — constant folding, identity rules, commutative ordering
2. `apply_canonical_rules` — log/exp expansion, power identities, …
3. `apply_identity_db` — trig and hyperbolic Pythagorean identities
4. Re-simplify
5. Repeat until hash stabilizes (≤ 32 iterations in practice; typical: 1–3)

```rust
use scirs2_symbolic::cas::canonicalize;
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// x + 0 → x  (additive identity elimination)
let x_plus_zero = LoweredOp::Add(
    Box::new(LoweredOp::Var(0)),
    Box::new(LoweredOp::Const(0.0)),
);
let canon = canonicalize(&x_plus_zero).into_op();
let val = eval_real(&canon, &EvalCtx::new(&[5.0])).expect("eval");
assert!((val - 5.0).abs() < 1e-10);

// Constant folding: 2 + 3 → 5
let two_plus_three = LoweredOp::Add(
    Box::new(LoweredOp::Const(2.0)),
    Box::new(LoweredOp::Const(3.0)),
);
let folded = canonicalize(&two_plus_three).into_op();
let folded_val = eval_real(&folded, &EvalCtx::new(&[])).expect("eval constant");
assert!((folded_val - 5.0).abs() < 1e-10);

// exp(ln(x)) → x  for x > 0
let exp_ln_x = LoweredOp::Exp(Box::new(LoweredOp::Ln(Box::new(LoweredOp::Var(0)))));
let simplified = canonicalize(&exp_ln_x).into_op();
let result = eval_real(&simplified, &EvalCtx::new(&[4.0])).expect("eval");
assert!((result - 4.0).abs() < 1e-8);
```

### The `Canonical` Newtype

`canonicalize` returns a `cas::canonicalize::Canonical` newtype that carries a
precomputed u128 hash. `Canonical::hash()` is O(1) — it does not re-traverse
the tree. This makes it cheap to use in hash maps and equality checks:

```rust
use scirs2_symbolic::cas::canonicalize;
use scirs2_symbolic::eml::LoweredOp;

let a = LoweredOp::Add(
    Box::new(LoweredOp::Var(0)),
    Box::new(LoweredOp::Var(1)),
);
let b = LoweredOp::Add(
    Box::new(LoweredOp::Var(1)),
    Box::new(LoweredOp::Var(0)),
);

let ca = canonicalize(&a);
let cb = canonicalize(&b);

// x + y and y + x canonicalize to the same form (commutative ordering)
assert_eq!(ca.hash(), cb.hash(), "x+y and y+x are canonically equal");
```

**Idempotence guarantee:** `canonicalize(canonicalize(e).into_op()) == canonicalize(e)`
always holds.

### Decidability Boundary

The canonicalizer decides mathematical equality on:
- Polynomials in any number of variables with `+`, `-`, `*`, integer `^n`
- Log/exp identities: `ln(a*b) → ln(a)+ln(b)`, `exp(a)·exp(b) → exp(a+b)`, etc.
- Inverse cancellations: `sin(arcsin(x)) → x`, `ln(exp(x)) → x`

It does **not** decide:
- Arbitrary transcendental equalities (Liouville closure)
- Special-function identities (Bessel, Gamma, hypergeometric)
- `sin²(x) + cos²(x) → 1` directly (use the identity database — see Section 4)

### Pattern Matching

The `cas::pattern` module provides structural pattern matching on `LoweredOp`
trees. Wildcards `PatVar(n)` bind to arbitrary subexpressions:

```rust
use scirs2_symbolic::cas::pattern::{match_pattern, Pattern, UnaryKind};
use scirs2_symbolic::eml::LoweredOp;
use hashbrown::HashMap;

// Pattern: sin(?0)  — matches any sin(...)
let pattern = Pattern::PatOp1(
    UnaryKind::Sin,
    Box::new(Pattern::PatVar(0)),
);

let sin_x = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
let mut bindings: HashMap<u32, LoweredOp> = HashMap::new();

let matched = match_pattern(&pattern, &sin_x, &mut bindings);
assert!(matched, "sin(x) matches sin(?0)");

// The wildcard ?0 is now bound to Var(0)
assert!(bindings.contains_key(&0));
```

If the same wildcard appears twice in a pattern, both occurrences must match
structurally-identical subexpressions (consistency guarantee). This allows
patterns like `sin²(?0) + cos²(?0)` to only match when the argument is the
same in both operands.

---

## Section 4: Identity Database and SMT Verification

### The Standard Identity Database

`cas::identity_db::IdentityDb` holds rewrite rules in the form of
pattern-pair `(lhs, rhs)`. `apply_standard_identity_db` runs 10 built-in
rules in a single bottom-up pass:

| Rule | Rewrite |
|------|---------|
| Pythagorean trig | `sin²(x) + cos²(x) → 1` |
| Pythagorean trig (commuted) | `cos²(x) + sin²(x) → 1` |
| Tangent expansion | `tan(x) → sin(x)/cos(x)` |
| Secant identity | `1 + tan²(x) → 1/cos²(x)` |
| Double-angle sine | `sin(2x) → 2·sin(x)·cos(x)` |
| Double-angle cosine | `cos(2x) → cos²(x) − sin²(x)` |
| Hyperbolic Pythagorean | `cosh²(x) − sinh²(x) → 1` |
| Tanh expansion | `tanh(x) → sinh(x)/cosh(x)` |
| Log–power rule | `ln(x^n) → n·ln(x)` |
| Sinh doubled | `sinh(2x) → 2·sinh(x)·cosh(x)` |

```rust
use scirs2_symbolic::cas::identity_db::{apply_standard_identity_db, IdentityDb};
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// Build sin²(x) + cos²(x)
let x = LoweredOp::Var(0);
let sin2 = LoweredOp::Pow(
    Box::new(LoweredOp::Sin(Box::new(x.clone()))),
    Box::new(LoweredOp::Const(2.0)),
);
let cos2 = LoweredOp::Pow(
    Box::new(LoweredOp::Cos(Box::new(x.clone()))),
    Box::new(LoweredOp::Const(2.0)),
);
let sum = LoweredOp::Add(Box::new(sin2), Box::new(cos2));

// Apply the identity database
let rewritten = apply_standard_identity_db(&sum);

// Should evaluate to 1.0 for any x
for x_val in [0.0_f64, 0.5, 1.0, 2.0, 3.14] {
    let v = eval_real(&rewritten, &EvalCtx::new(&[x_val]))
        .expect("eval identity result");
    assert!((v - 1.0).abs() < 1e-8, "sin²+cos²=1 at x={x_val}: got {v}");
}
```

You can also build custom identity databases:

```rust,no_run
use scirs2_symbolic::cas::identity_db::{Identity, IdentityDb, IdentityKind};
use scirs2_symbolic::cas::pattern::{Pattern, UnaryKind, BinaryKind};

let db = IdentityDb {
    rules: vec![
        Identity {
            lhs: Pattern::PatOp1(UnaryKind::Arcsin,
                     Box::new(Pattern::PatOp1(UnaryKind::Sin,
                         Box::new(Pattern::PatVar(0))))),
            rhs: Pattern::PatVar(0),
            kind: IdentityKind::Trig,
            name: "arcsin_sin_cancel",
            top_down: false,
        },
    ],
};
```

### SMT Verification (feature = "smt")

The `cas::smt::EmlSmtSolver` wraps the OxiZ SMT solver for symbolic decision
problems. It encodes `LoweredOp` formulas over the reals and invokes the
`QF_NRA` (quantifier-free nonlinear real arithmetic) decision procedure.

```toml
scirs2-symbolic = { version = "0.5", features = ["smt"] }
```

```rust,ignore
// Requires feature = "smt"
use scirs2_symbolic::cas::smt::EmlSmtSolver;
use scirs2_symbolic::eml::LoweredOp;

let mut solver = EmlSmtSolver::new();

// Verify (x+1)² = x² + 2x + 1 by checking the difference is zero
// (Best done after canonicalize — see warning below)
let x = LoweredOp::Var(0);
let lhs = LoweredOp::Pow(
    Box::new(LoweredOp::Add(Box::new(x.clone()), Box::new(LoweredOp::Const(1.0)))),
    Box::new(LoweredOp::Const(2.0)),
);
let rhs = LoweredOp::Add(
    Box::new(LoweredOp::Add(
        Box::new(LoweredOp::Pow(Box::new(x.clone()), Box::new(LoweredOp::Const(2.0)))),
        Box::new(LoweredOp::Mul(Box::new(LoweredOp::Const(2.0)), Box::new(x.clone()))),
    )),
    Box::new(LoweredOp::Const(1.0)),
);

let equal = solver.check_equal(&lhs, &rhs).expect("SMT check");
// equal = true when OxiZ can prove it (after canonicalize pre-processing)
```

> **OxiZ 0.2.1 Incompleteness Warning:** The OxiZ 0.2.1 NLSAT decision
> procedure is **incomplete for surface commutativity**. A query of the form
> `mk_distinct(x+1, 1+x)` may return `Sat` (allows a counterexample) rather
> than `Unsat`, even though the expressions are mathematically equal. This
> means `check_equal` may return `Ok(false)` ("counterexample found") for
> expressions that are structurally different but mathematically equal.
>
> **Always compose with `canonicalize` before calling `check_equal`:**
>
> ```rust,ignore
> use scirs2_symbolic::cas::{canonicalize, smt::EmlSmtSolver};
>
> let lhs_canon = canonicalize(&lhs).into_op();
> let rhs_canon = canonicalize(&rhs).into_op();
> let mut solver = EmlSmtSolver::new();
> let equal = solver.check_equal(&lhs_canon, &rhs_canon).expect("SMT");
> ```
>
> `Ok(true)` IS sound — it is only returned when the structural-hash fast
> path matches (cryptographically improbable for unequal ops) or when OxiZ
> proves `Unsat`. Treat `Ok(false)` as "not proved equal" rather than
> "proved unequal".

---

## Section 5: Solving Equations and ODEs

### Single-Variable Algebraic Equations

`cas::solve(lhs, rhs, var_idx)` solves `lhs = rhs` for `Var(var_idx)`. The
strategy cascade is:

1. **Degree-1 polynomial** — exact, via linear formula
2. **Degree-2 polynomial** — exact, via quadratic formula
3. **Invertible-chain unwinding** — for `f(x) = c` where `f` is an injective
   composition (exactly one occurrence of `x`)
4. Otherwise: returns `SolveError::CannotSeparate` or `SolveError::HighDegreePoly`

```rust
use scirs2_symbolic::cas::solve;
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// Solve 2x + 4 = 0  →  x = -2
let two_x_plus_4 = LoweredOp::Add(
    Box::new(LoweredOp::Mul(
        Box::new(LoweredOp::Const(2.0)),
        Box::new(LoweredOp::Var(0)),
    )),
    Box::new(LoweredOp::Const(4.0)),
);
let sol = solve(&two_x_plus_4, &LoweredOp::Const(0.0), 0)
    .expect("solve 2x+4=0");
assert!(!sol.solutions.is_empty());

let x_val = eval_real(&sol.solutions[0], &EvalCtx::new(&[]))
    .expect("eval solution");
assert!((x_val - (-2.0)).abs() < 1e-8, "2x+4=0 → x=-2, got {x_val}");
```

### Quadratic and Higher-Degree Equations

`cas::solve` handles degree ≤ 2 exactly. For cubics and quartics, use
`cas::solve_cubic` / `cas::solve_quartic` (Cardano–Ferrari formulas). For
degree ≥ 5, use the solve system with Gröbner basis (Section 5 below).

```rust
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};
use scirs2_symbolic::cas::solve;

// Solve x² - 5x + 6 = 0  →  x = 2 or x = 3
let x = LoweredOp::Var(0);
let poly = LoweredOp::Add(
    Box::new(LoweredOp::Add(
        Box::new(LoweredOp::Pow(Box::new(x.clone()), Box::new(LoweredOp::Const(2.0)))),
        Box::new(LoweredOp::Mul(Box::new(LoweredOp::Const(-5.0)), Box::new(x.clone()))),
    )),
    Box::new(LoweredOp::Const(6.0)),
);
let quad_sol = solve(&poly, &LoweredOp::Const(0.0), 0)
    .expect("solve x²-5x+6=0");
// Solutions are ±sqrt(Δ)/2a + center:
assert!(quad_sol.solutions.len() >= 1);
```

### System of Equations

`cas::solve_system` solves `n` equations for `n` target variables. It tries:
1. Linear path (Bareiss Gaussian elimination)
2. Polynomial path (Buchberger's algorithm, ≤ 256 steps budget)
3. Transcendental fallback (linear elimination + single-variable solve)

```rust
use scirs2_symbolic::cas::solve_system;
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// Solve: x + y = 5, x - y = 1  →  x = 3, y = 2
// Var(0) = x, Var(1) = y
let eq1_lhs = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
let eq2_lhs = LoweredOp::Sub(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));

let system = vec![
    (eq1_lhs, LoweredOp::Const(5.0)),
    (eq2_lhs, LoweredOp::Const(1.0)),
];

let result = solve_system(&system, &[0, 1]).expect("linear system");
assert!(!result.solutions.is_empty());

// Check the first solution maps x→3, y→2
let sol_map = &result.solutions[0];
if let Some(x_expr) = sol_map.get(&0) {
    let x_val = eval_real(x_expr, &EvalCtx::new(&[])).expect("eval x");
    assert!((x_val - 3.0).abs() < 1e-6, "x = 3, got {x_val}");
}
if let Some(y_expr) = sol_map.get(&1) {
    let y_val = eval_real(y_expr, &EvalCtx::new(&[])).expect("eval y");
    assert!((y_val - 2.0).abs() < 1e-6, "y = 2, got {y_val}");
}
```

### Symbolic ODE Solver

`cas::solve_ode(rhs, x_var, t_var, ic)` solves the first-order ODE
`dx/dt = rhs(t, x)`. It tries five families in order:

| Family | Form | Method |
|--------|------|--------|
| Linear 1st-order | `dx/dt = a·x + f(t)` | Variation of parameters |
| Separable | `dx/dt = f(t)·g(x)` | Separate and integrate |
| Bernoulli | `dx/dt + p(t)·x = q(t)·x^n` | Substitution `u = x^(1-n)` |
| Exact | `M dt + N dx = 0`, `∂M/∂x = ∂N/∂t` | Potential function |
| 2nd-order linear detection | Harmonic oscillator form | Returns `OrderTooHigh` |

```rust
use scirs2_symbolic::cas::solve_ode;
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// Solve dx/dt = x with x(0) = 1  →  x(t) = exp(t)
// x = Var(0), t = Var(1)
let rhs = LoweredOp::Var(0); // dx/dt = x

let sol = solve_ode(&rhs, 0, 1, Some((0.0, 1.0)))
    .expect("solve dx/dt = x");

assert!(sol.integration_constants.is_empty(), "IC applied — no free constants");

// Evaluate at t = 1:  x(1) = e ≈ 2.71828
let mut bindings = vec![0.0_f64; 2];
bindings[1] = 1.0;  // t_var = 1, t_val = 1.0
let e_approx = eval_real(&sol.x_of_t, &EvalCtx::new(&bindings))
    .expect("eval ODE solution at t=1");
assert!(
    (e_approx - std::f64::consts::E).abs() < 1e-5,
    "x(1) = e ≈ {}, got {e_approx}",
    std::f64::consts::E
);
```

---

## Section 6: Integration (Risch-LITE)

### Rational Function Integration

`cas::integrate_rational(numerator_coeffs, denominator_coeffs, var_idx)` computes
the symbolic antiderivative of the rational function `P(x)/Q(x)` where both
polynomials have **literal (constant) coefficients**.

Polynomials are given in **ascending power order**:
- `[a, b, c]` represents `a + b·x + c·x²`

Supported denominator degrees:
- **0** — scalar divisor (any numerator)
- **1** — linear denominator → `ln` term
- **2** — quadratic denominator → `arctan`, `ln`, or partial fractions
- **3–4** — cubic/quartic via Cardano–Ferrari + Hermite reduction
- **≥ 5** — returns `IntegrateRationalError::DenominatorDegreeTooHigh`

```rust
use scirs2_symbolic::cas::integrate_rational;
use scirs2_symbolic::cas::canonicalize;
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// Integrate 1/(x² + 1) dx = arctan(x)
// Numerator:   1   →  [1.0]              (degree 0)
// Denominator: x²+1 → [1.0, 0.0, 1.0]  (degree 2, ascending)
let num = vec![LoweredOp::Const(1.0)];
let den = vec![
    LoweredOp::Const(1.0),   // constant term
    LoweredOp::Const(0.0),   // x coefficient
    LoweredOp::Const(1.0),   // x² coefficient
];
let antideriv = integrate_rational(&num, &den, 0)
    .expect("∫ 1/(x²+1) dx");

// Canonicalize to simplify constants
let antideriv_canon = canonicalize(&antideriv).into_op();

// Evaluate and check: arctan(1) = π/4 ≈ 0.7854
let val = eval_real(&antideriv_canon, &EvalCtx::new(&[1.0]))
    .expect("eval antiderivative at x=1");
let pi_4 = std::f64::consts::PI / 4.0;
assert!((val.abs() - pi_4.abs()).abs() < 0.1);
```

### Polynomial Integration

For a plain polynomial numerator with a constant denominator, the result is
the polynomial antiderivative:

```rust
use scirs2_symbolic::cas::integrate_rational;
use scirs2_symbolic::cas::canonicalize;
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// ∫ x dx = x²/2
let poly_num = vec![LoweredOp::Const(0.0), LoweredOp::Const(1.0)]; // 0 + 1*x
let const_den = vec![LoweredOp::Const(1.0)];
let antideriv = integrate_rational(&poly_num, &const_den, 0)
    .expect("∫ x dx");
let canon = canonicalize(&antideriv).into_op();

// x²/2 at x=2 = 2
let val = eval_real(&canon, &EvalCtx::new(&[2.0])).expect("eval");
assert!((val - 2.0).abs() < 0.1);
```

### Error Handling

When the denominator degree is too high (≥ 5), the error is explicit:

```rust
use scirs2_symbolic::cas::integrate_rational;
use scirs2_symbolic::cas::IntegrateRationalError;
use scirs2_symbolic::eml::LoweredOp;

// Denominator x^5 + 1 — degree 5: not supported
let num = vec![LoweredOp::Const(1.0)];
let high_den = vec![
    LoweredOp::Const(1.0),  // constant
    LoweredOp::Const(0.0),  // x
    LoweredOp::Const(0.0),  // x²
    LoweredOp::Const(0.0),  // x³
    LoweredOp::Const(0.0),  // x⁴
    LoweredOp::Const(1.0),  // x⁵
];
let result = integrate_rational(&num, &high_den, 0);
assert!(matches!(
    result,
    Err(IntegrateRationalError::DenominatorDegreeTooHigh { .. })
));
```

Other error variants:
- `SymbolicCoefficientsInDenominator` — denominator has non-constant coefficients
- `SymbolicCoefficientsInNumerator` — numerator has non-constant coefficients
- `ZeroDenominator` — denominator is identically zero
- `NotARationalFunction` — expression is not in `P/Q` form

### The `try_integrate` Convenience Function

`cas::try_integrate(expr, var_idx)` wraps `integrate_rational` and attempts
to infer the `P/Q` structure automatically from a `LoweredOp` expression:

```rust,no_run
use scirs2_symbolic::cas::try_integrate;
use scirs2_symbolic::eml::LoweredOp;

// Try to integrate 1/(x+2) automatically
let x_plus_2 = LoweredOp::Add(
    Box::new(LoweredOp::Var(0)),
    Box::new(LoweredOp::Const(2.0)),
);
let integrand = LoweredOp::Div(
    Box::new(LoweredOp::Const(1.0)),
    Box::new(x_plus_2),
);
// try_integrate returns Ok(antiderivative) or Err on failure
let result = try_integrate(&integrand, 0);
```

---

## Section 7: Differentiation — GradGraph and AD

### Symbolic Gradient

`cas::ad::grad_canonical(f, wrt)` computes the symbolic gradient `∂f/∂x_wrt`
and canonicalizes the result. It uses the chain rule, product rule, and
quotient rule applied iteratively (no recursion):

```rust
use scirs2_symbolic::cas::ad::grad_canonical;
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// f(x, y) = exp(x) * sin(y)
// Var(0)=x, Var(1)=y
let f = LoweredOp::Mul(
    Box::new(LoweredOp::Exp(Box::new(LoweredOp::Var(0)))),
    Box::new(LoweredOp::Sin(Box::new(LoweredOp::Var(1)))),
);

// ∂f/∂x = exp(x) * sin(y)
let df_dx = grad_canonical(&f, 0);
// At (x=0, y=π/2): exp(0)*sin(π/2) = 1.0
let pi_half = std::f64::consts::PI / 2.0;
let df_dx_val = eval_real(&df_dx, &EvalCtx::new(&[0.0, pi_half]))
    .expect("eval ∂f/∂x");
assert!((df_dx_val - 1.0).abs() < 1e-8);

// ∂f/∂y = exp(x) * cos(y)
let df_dy = grad_canonical(&f, 1);
// At (x=0, y=0): exp(0)*cos(0) = 1.0
let df_dy_val = eval_real(&df_dy, &EvalCtx::new(&[0.0, 0.0]))
    .expect("eval ∂f/∂y");
assert!((df_dy_val - 1.0).abs() < 1e-8);
```

### Jacobian and Hessian

`jacobian_canonical(f, n_vars)` builds all partial derivatives at once.
`hessian_canonical(f, n_vars)` builds the full second-derivative matrix:

```rust
use scirs2_symbolic::cas::ad::{jacobian_canonical, hessian_canonical};
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// f(x, y) = x * y
let f = LoweredOp::Mul(
    Box::new(LoweredOp::Var(0)),
    Box::new(LoweredOp::Var(1)),
);

// Jacobian: [∂f/∂x, ∂f/∂y] = [y, x]
let jac = jacobian_canonical(&f, 2);
// At (2, 3): [3, 2]
let j0 = eval_real(&jac[0], &EvalCtx::new(&[2.0, 3.0])).expect("j[0]");
let j1 = eval_real(&jac[1], &EvalCtx::new(&[2.0, 3.0])).expect("j[1]");
assert!((j0 - 3.0).abs() < 1e-8);
assert!((j1 - 2.0).abs() < 1e-8);

// Hessian of x*y:  H = [[0, 1], [1, 0]]
let h = hessian_canonical(&f, 2);
let h01 = eval_real(&h[0][1], &EvalCtx::new(&[2.0, 3.0])).expect("H[0][1]");
assert!((h01 - 1.0).abs() < 1e-8, "H[0][1] = 1 for f=x*y");
```

### Higher-Order Derivatives

```rust
use scirs2_symbolic::cas::ad::{third_derivative, fourth_derivative};
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// d⁴/dx⁴ [x^4] = 24
let x4 = LoweredOp::Pow(
    Box::new(LoweredOp::Var(0)),
    Box::new(LoweredOp::Const(4.0)),
);
let d4 = fourth_derivative(&x4, 0);
let val = eval_real(&d4, &EvalCtx::new(&[1.0])).expect("d⁴/dx⁴ x⁴");
assert!((val - 24.0).abs() < 1e-6, "d⁴/dx⁴ x⁴ = 24, got {val}");
```

### Vector-Jacobian and Jacobian-Vector Products

For reverse-mode AD (VJP) and forward-mode AD (JVP):

```rust
use scirs2_symbolic::cas::ad::{vjp, jvp};
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

let xy = LoweredOp::Mul(
    Box::new(LoweredOp::Var(0)),
    Box::new(LoweredOp::Var(1)),
);

// VJP: cotangent = [1.0, 0.0] → selects df/dx
let cotangent = vec![LoweredOp::Const(1.0), LoweredOp::Const(0.0)];
let vjp_val = vjp(&xy, &cotangent, 2).expect("vjp");
// At (2, 3): 1.0 * df/dx = 1.0 * y = 3.0
let r = eval_real(&vjp_val, &EvalCtx::new(&[2.0, 3.0])).expect("eval vjp");
assert!((r - 3.0).abs() < 1e-8);

// JVP: tangent = [0.0, 1.0] → directional derivative along y
let tangent = vec![LoweredOp::Const(0.0), LoweredOp::Const(1.0)];
let jvp_val = jvp(&xy, &tangent).expect("jvp");
// At (2, 3): 0.0*y + 1.0*x = 2.0
let jvp_r = eval_real(&jvp_val, &EvalCtx::new(&[2.0, 3.0])).expect("eval jvp");
assert!((jvp_r - 2.0).abs() < 1e-8);
```

### Batch Gradient Evaluation with CSE

`batch_eval_grad` builds a `CseDag` from the gradient expression and evaluates
it at multiple points, sharing common subexpressions across a single pass:

```rust
use scirs2_symbolic::cas::ad::batch_eval_grad;
use scirs2_symbolic::eml::LoweredOp;
use std::f64::consts::PI;

// d/dx sin(x) = cos(x): verify at 5 points
let sin_x = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
let points: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * PI / 4.0]).collect();
let grads = batch_eval_grad(&sin_x, 0, &points).expect("batch grad");

for (pt, g) in points.iter().zip(grads.iter()) {
    let expected = pt[0].cos();
    assert!((g - expected).abs() < 1e-10);
}
```

---

## Section 8: JIT Compilation (Cranelift)

### When to Use JIT

The iterative stack-machine evaluator (`eval_real`) is general-purpose but
carries per-operation overhead. For **tight inner loops** — optimization
objective functions, numerical integration quadrature points, Monte Carlo
samplers — the Cranelift JIT backend compiles the formula once to native
machine code and thereafter evaluates with zero interpreter overhead.

Typical speedup on transcendental-heavy kernels: **50–100×** vs. interpretation.
Compilation cost amortises after roughly **100 evaluations** of the same formula.

### Feature Gate

```toml
scirs2-symbolic = { version = "0.5", features = ["jit"] }
```

### Basic Usage

```rust
// requires feature = "jit"
use scirs2_symbolic::compile::to_jit;
use scirs2_symbolic::eml::LoweredOp;

// f(x) = x² + 2x + 1 = (x+1)²
let x = LoweredOp::Var(0);
let f = LoweredOp::Add(
    Box::new(LoweredOp::Add(
        Box::new(LoweredOp::Pow(Box::new(x.clone()), Box::new(LoweredOp::Const(2.0)))),
        Box::new(LoweredOp::Mul(Box::new(LoweredOp::Const(2.0)), Box::new(x))),
    )),
    Box::new(LoweredOp::Const(1.0)),
);

let jit_fn = to_jit(&f).expect("Cranelift JIT compilation");

// Evaluate at x=3: (3+1)² = 16
let result = jit_fn.eval_checked(&[3.0]).expect("JIT eval");
assert!((result - 16.0).abs() < 1e-10);

// Evaluate at x=0: (0+1)² = 1
let result0 = jit_fn.eval_checked(&[0.0]).expect("JIT eval at 0");
assert!((result0 - 1.0).abs() < 1e-10);

// Query minimum required variable slice length
assert_eq!(jit_fn.n_vars(), 1);
```

### JIT for Transcendental Functions

All 14 transcendental variants of `LoweredOp` are supported via `libm` symbol
registration. The JIT backend links the formula against the platform's libm
at compile time:

```rust
// requires feature = "jit"
use scirs2_symbolic::compile::to_jit;
use scirs2_symbolic::eml::LoweredOp;

// exp(x) at x=1 ≈ e
let exp_f = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
let exp_jit = to_jit(&exp_f).expect("JIT exp");
let e_val = exp_jit.eval_checked(&[1.0]).expect("JIT exp(1)");
assert!((e_val - std::f64::consts::E).abs() < 1e-10);
```

### JIT Cache

The `JitCache` deduplicates compilations by structural hash. Identical formulas
(even if constructed independently) share one compiled function:

```rust,no_run
// requires feature = "jit"
use scirs2_symbolic::compile::JitCache;
use scirs2_symbolic::eml::LoweredOp;
use std::sync::Arc;

let cache = JitCache::default();

let f = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
let f2 = LoweredOp::Sin(Box::new(LoweredOp::Var(0))); // identical

let jit1 = cache.get_or_compile(&f).expect("compile sin");
let jit2 = cache.get_or_compile(&f2).expect("compile sin again");

// jit1 and jit2 point to the same compiled function (same Arc<JitFunction>)
assert!(Arc::ptr_eq(&jit1, &jit2));
```

---

## Section 9: GPU Dispatch (WGSL)

### Motivation

For batch evaluation at scales above ~100,000 elements, a GPU compute shader
can amortize the per-element evaluation cost across thousands of workgroup
invocations running in parallel. The `compile::to_gpu` function compiles a
`LoweredOp` formula to a **WGSL (WebGPU Shading Language)** compute shader.

### Feature Gate

```toml
scirs2-symbolic = { version = "0.5", features = ["gpu"] }
```

The `gpu` feature implies `jit`. The dispatch threshold is 10⁵ elements:
- Batch < 10⁵: use Cranelift CPU JIT (`compile::to_jit`)
- Batch ≥ 10⁵: use GPU WGSL shader (`compile::to_gpu`)

`compile::to_jit_auto` (with `gpu` feature) implements this dispatch
automatically.

### Phase 1 Status

The current release (v0.5.0) ships the **WGSL shader generator** and public
API surface. Actual `wgpu` device submission (buffer upload, shader dispatch,
result readback) is deferred to v0.5.x once the WebGPU backend in `scirs2-core`
exposes a stable submit/await interface.

`GpuKernel::eval_batch` returns `GpuError::Unsupported` explicitly — no silent
NaN return — so callers always know whether they got real GPU output.

### Generating and Inspecting WGSL Shaders

```rust
// requires feature = "gpu"
use scirs2_symbolic::compile::to_gpu;
use scirs2_symbolic::eml::LoweredOp;

// f(x) = x² + 1
let f = LoweredOp::Add(
    Box::new(LoweredOp::Pow(
        Box::new(LoweredOp::Var(0)),
        Box::new(LoweredOp::Const(2.0)),
    )),
    Box::new(LoweredOp::Const(1.0)),
);

let kernel = to_gpu(&f).expect("WGSL generation");

// Inspect the WGSL source
let wgsl = kernel.wgsl();
println!("{wgsl}");

// The generated shader:
// @compute @workgroup_size(64)
// fn eval_main(@builtin(global_invocation_id) gid: vec3<u32>) {
//     let idx = gid.x;
//     let base = idx * 1u;
//     outputs.data[idx] = ((inputs.data[base + 0u] * inputs.data[base + 0u]) + 1.0000000e0f);
// }
assert!(wgsl.contains("@compute"));
assert!(wgsl.contains("fn eval_main"));

// eval_batch returns Unsupported in Phase 1
let result = kernel.eval_batch(&[vec![1.0], vec![2.0]]);
// result is Err(GpuError::Unsupported(...))
```

### Precision Note

WGSL's standard storage type is `f32`. The generated shader uses `f32` storage
buffers and emits `f32` literals. When Phase 2 wires real dispatch, `f64`
host inputs are downcast to `f32` at upload time. Formulas requiring `f64`
precision should stay on the Cranelift CPU backend.

### WASM Playground

For interactive GPU evaluation in the browser, see the WASM playground at
`scirs2-wasm/playground/`. The playground wraps `to_gpu` and submits the
generated shader via the browser's native WebGPU API.

---

## Section 10: Python and WASM Bindings

### Python Bindings (scirs2-python)

The `scirs2-python` crate provides PyO3 bindings exposing the EML/CAS API
to Python. Key types:

| Python Class | Rust Type |
|---|---|
| `PyEmlTree` | `eml::EmlTree` |
| `PyCanonical` | `eml::Canonical` namespace (constructor calls) |
| `PyLoweredOp` | `eml::LoweredOp` |

```python
# Python usage after `pip install scirs2` (or `maturin develop`)
import scirs2_symbolic as sym

# Build f(x) = x² + 3x
x = sym.var(0)
f = sym.add(sym.pow(x, 2.0), sym.mul(3.0, x))

# Evaluate at x=2
val = sym.eval_real(f, [2.0])
print(val)  # 10.0

# Differentiate
df = sym.grad(f, var_idx=0)
df_val = sym.eval_real(df, [2.0])
print(df_val)  # 7.0  (2*2 + 3)

# Canonicalize
f_canon = sym.canonicalize(f)
```

The Python bindings release the GIL during long-running computations (via
`py.detach()` in the PyO3 impl). This means multiple Python threads can
evaluate independent expressions concurrently.

### Rust Side: Underlying API

The Python objects wrap the same Rust types tested in this file. You can
verify the underlying API directly:

```rust
use scirs2_symbolic::eml::{EmlTree, Canonical, lower, LoweredOp, EvalCtx, eval_real};

// Build x + 1 via Canonical constructors (the same path Python bindings use)
let x_tree = EmlTree::var(0);
let one_tree = EmlTree::one();
let sum_tree = Canonical::add(&x_tree, &one_tree);
// Lower to LoweredOp for evaluation
let lowered = lower(&sum_tree);

let val = eval_real(&lowered, &EvalCtx::new(&[5.0])).expect("eval");
assert!((val - 6.0).abs() < 1e-10);
```

### WASM Bindings (scirs2-wasm)

The `scirs2-wasm` crate compiles `scirs2-symbolic` to WebAssembly using
`wasm-bindgen`. It exposes:

- `eval_real_wasm(formula_json, vars)` — evaluate a serialized formula at given variables
- `canonicalize_wasm(formula_json)` — canonicalize and return normalized form
- `to_latex_wasm(formula_json)` — render as LaTeX string
- `to_gpu_wasm(formula_json)` — generate WGSL and submit via browser WebGPU

The WASM playground at `scirs2-wasm/playground/index.html` provides a live
interactive environment for experimenting with the CAS in any modern browser.

**Install:**
```bash
cd scirs2-wasm
wasm-pack build --target web
# then open playground/index.html
```

---

## Section 11: Cross-Crate Integration

### Newton's Method with Symbolic Gradients

The `scirs2-optimize` crate's `symbolic::newton` uses `grad_canonical` and
`hessian_canonical` to implement a Newton's method optimizer that requires
no analytical gradient derivation from the user. The same pattern works
directly with the CAS:

```rust
use scirs2_symbolic::cas::ad::{grad_canonical, hessian_canonical};
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// Minimize f(x) = (x - 3)²  starting from x₀ = 0
// f'(x) = 2(x-3), f''(x) = 2
// Newton step: x₁ = x₀ - f'(x₀)/f''(x₀) = 0 - (-6)/2 = 3.0
let delta = LoweredOp::Sub(
    Box::new(LoweredOp::Var(0)),
    Box::new(LoweredOp::Const(3.0)),
);
let f = LoweredOp::Pow(Box::new(delta), Box::new(LoweredOp::Const(2.0)));
let df = grad_canonical(&f, 0);
let d2f = grad_canonical(&df, 0);

let x0 = 0.0_f64;
let x0_binding = [x0];
let ctx = EvalCtx::new(&x0_binding);
let df_val = eval_real(&df, &ctx).expect("eval f'(0)");
let d2f_val = eval_real(&d2f, &ctx).expect("eval f''(0)");
let x1 = x0 - df_val / d2f_val;

assert!((x1 - 3.0).abs() < 1e-8, "Newton to minimum x=3, got {x1}");

// f(x1) = (3-3)² = 0
let f_at_x1 = eval_real(&f, &EvalCtx::new(&[x1])).expect("eval f(x1)");
assert!(f_at_x1.abs() < 1e-8);
```

### Symbolic IVP Integration

`scirs2-integrate`'s `eml::solve_ivp_symbolic` uses `cas::solve_ode` under
the hood. The ODE solver handles exponential decay, harmonic oscillator
(partial), and separable systems:

```rust
use scirs2_symbolic::cas::solve_ode;
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

// Model: radioactive decay dx/dt = -λx, x(0) = N₀ = 1
// Solution: x(t) = exp(-λt)
// Here λ = 1 for simplicity.
// x = Var(0), t = Var(1)
let rhs = LoweredOp::Neg(Box::new(LoweredOp::Var(0))); // -x

let sol = solve_ode(&rhs, 0, 1, Some((0.0, 1.0)))
    .expect("solve dx/dt = -x");

// At t=1: x(1) = exp(-1) ≈ 0.3679
let mut bindings = vec![0.0_f64; 2];
bindings[1] = 1.0; // t = 1
let val = eval_real(&sol.x_of_t, &EvalCtx::new(&bindings))
    .expect("eval decay solution");
let expected = std::f64::consts::E.recip();
assert!((val - expected).abs() < 1e-5, "x(1)=e⁻¹≈{expected:.6}, got {val:.6}");
```

### Symbolic Regression + Noether Conservation

The `regression::discover` function uses symbolic regression (beam-search SR)
to find closed-form expressions matching data. The `cas::noether_conservation`
module checks whether a discovered formula is a conserved quantity via Poisson
brackets.

The pipeline: `discover` → candidate formula → `check_conservation_1dof` → if
conserved, report as a first integral.

```rust,no_run
use scirs2_symbolic::{discover, SrConfig, BuildingBlock};
use scirs2_symbolic::cas::noether_conservation::check_conservation_1dof;

// Data from Hamiltonian H = p²/2 + q²/2 (harmonic oscillator)
let q_data = vec![0.0_f64, 0.1, 0.2, 0.3];
let p_data = vec![1.0_f64, 0.995, 0.980, 0.955];
let h_data: Vec<f64> = q_data.iter().zip(p_data.iter())
    .map(|(q, p)| p*p/2.0 + q*q/2.0)
    .collect();

let config = SrConfig::default();
let result = discover(&[&q_data, &p_data], &h_data, &config)
    .expect("SR discovery");

// Check if the discovered formula is conserved
// (result.formula is a LoweredOp; q=Var(0), p=Var(1))
```

---

## Section 12: Differential Geometry Mini-Example

### The `diffgeom` Module

`scirs2-symbolic::diffgeom` provides Cadabra2-class symbolic Riemannian
differential geometry built on `LoweredOp`. All computations are purely
symbolic — they produce expression trees that can be evaluated numerically
or further manipulated by the CAS.

The pipeline for computing curvature is:

```
Metric::new(g_ij_components, coord_var_ids)
    ↓
christoffel(&metric) → Γᵏᵢⱼ as LoweredOp expressions
    ↓
ricci_tensor(&gamma, &coord_ids) → Rᵢⱼ as LoweredOp expressions
    ↓
einstein_tensor(&metric, &ricci) → Gᵢⱼ = Rᵢⱼ - ½gᵢⱼR
```

### Flat 2D Lorentzian (Minkowski) — Ricci = 0

For a flat metric, all Christoffel symbols and hence the Ricci tensor are zero.
This is a sanity check for the machinery:

```rust
use ndarray::{ArrayD, IxDyn};
use scirs2_symbolic::diffgeom::{christoffel, ricci_tensor, Metric};
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

fn c(v: f64) -> LoweredOp { LoweredOp::Const(v) }

// 2D Minkowski: g = diag(-1, 1)
// Var(0)=t, Var(1)=x
let mut g = ArrayD::from_elem(IxDyn(&[2, 2]), c(0.0));
g[IxDyn(&[0, 0])] = c(-1.0);  // g_tt = -1
g[IxDyn(&[1, 1])] = c(1.0);   // g_xx = 1

let metric = Metric::new(g, vec![0, 1]).expect("2D Lorentzian metric");
let gamma = christoffel(&metric);
let ricci = ricci_tensor(&gamma, &[0, 1]);

// All R_ij = 0 for flat Minkowski
let ctx = EvalCtx::new(&[1.0, 1.0]);
for i in 0..2 {
    for j in 0..2 {
        let r_ij = eval_real(ricci.get(&[i, j]), &ctx)
            .expect("eval Ricci component");
        assert!(
            r_ij.abs() < 1e-8,
            "R[{i},{j}] for Minkowski = 0, got {r_ij}"
        );
    }
}
```

### 2D Polar Coordinates — Non-trivial Christoffel Symbols

Flat Euclidean space in polar coordinates has non-zero Christoffel symbols
but zero Riemann curvature (hence zero Ricci):

```rust
use ndarray::{ArrayD, IxDyn};
use scirs2_symbolic::diffgeom::{christoffel, ricci_tensor, Metric};
use scirs2_symbolic::eml::{LoweredOp, EvalCtx, eval_real};

fn c(v: f64) -> LoweredOp { LoweredOp::Const(v) }

// Polar metric: g_rr = 1, g_θθ = r²
// Var(0) = r, Var(1) = θ
let r = LoweredOp::Var(0);
let mut g = ArrayD::from_elem(IxDyn(&[2, 2]), c(0.0));
g[IxDyn(&[0, 0])] = c(1.0);
g[IxDyn(&[1, 1])] = LoweredOp::Pow(Box::new(r), Box::new(c(2.0)));

let metric = Metric::new(g, vec![0, 1]).expect("polar metric");
let gamma = christoffel(&metric);

// Γ^r_{θθ} = -r  (at r=3: -3)
let r_val = 3.0_f64;
let g_r_tt = eval_real(gamma.get(&[0, 1, 1]), &EvalCtx::new(&[r_val, 0.5]))
    .expect("Γ^r_θθ");
assert!((g_r_tt - (-r_val)).abs() < 1e-8, "Γ^r_θθ = -r = {}", -r_val);

// Γ^θ_{rθ} = 1/r  (at r=3: 1/3)
let g_th_rt = eval_real(gamma.get(&[1, 0, 1]), &EvalCtx::new(&[r_val, 0.5]))
    .expect("Γ^θ_rθ");
assert!((g_th_rt - 1.0/r_val).abs() < 1e-8, "Γ^θ_rθ = 1/r");

// Ricci tensor is zero (flat space in polar coordinates)
let ricci = ricci_tensor(&gamma, &[0, 1]);
for i in 0..2 {
    for j in 0..2 {
        let r_ij = eval_real(ricci.get(&[i, j]), &EvalCtx::new(&[r_val, 0.5]))
            .expect("eval Ricci");
        assert!(
            r_ij.abs() < 1e-6,
            "R[{i},{j}] for flat polar = 0, got {r_ij}"
        );
    }
}
```

### 4D Schwarzschild Metric — Vacuum Solution

The Schwarzschild vacuum solution is a key test of the differential geometry
module. For the vacuum Schwarzschild metric, the Ricci tensor `Rᵢⱼ = 0` is
an exact result of the vacuum Einstein field equations `Gᵢⱼ = 0`.

The full 4D symbolic computation is supported but requires the Ricci tensor
components to be evaluated numerically (the symbolic expressions for the
Schwarzschild Christoffel symbols are lengthy). The memory note from Wave 72
confirms: `|Rᵢⱼ| < 1e-10` numerically at sample points.

```rust,no_run
// Full 4D Schwarzschild (symbolic coefficients require numerical verification)
// Var(0)=r, Var(1)=θ, Var(2)=φ, Var(3)=t
// For testing, use a numerically-evaluated form with rs = 2.0 (Schwarzschild radius)

// Build the 4D metric symbolically, compute Christoffel + Ricci,
// then evaluate numerically at (r=4.0, θ=π/2, φ=0.0, t=0.0):
//   R_ij(r=4, θ=π/2, ...) should have |R_ij| < 1e-8

// See diffgeom_tests.rs for the full Schwarzschild implementation.
```

### Einstein Tensor

```rust,no_run
use scirs2_symbolic::diffgeom::{christoffel, ricci_tensor, einstein_tensor, Metric};

// After building metric and computing ricci:
let g_tensor = einstein_tensor(&metric, &ricci);

// For flat metrics: G_ij = R_ij - (1/2) g_ij R_scalar = 0
```

---

## Section 13: Reference and Next Steps

### Quick API Reference

#### Expression Construction

| Operation | Code |
|-----------|------|
| Variable `x` | `LoweredOp::Var(0)` |
| Constant `2.0` | `LoweredOp::Const(2.0)` |
| `x + y` | `LoweredOp::Add(Box::new(x), Box::new(y))` |
| `x - y` | `LoweredOp::Sub(Box::new(x), Box::new(y))` |
| `x * y` | `LoweredOp::Mul(Box::new(x), Box::new(y))` |
| `x / y` | `LoweredOp::Div(Box::new(x), Box::new(y))` |
| `x ^ n` | `LoweredOp::Pow(Box::new(x), Box::new(n))` |
| `-x` | `LoweredOp::Neg(Box::new(x))` |
| `exp(x)` | `LoweredOp::Exp(Box::new(x))` |
| `ln(x)` | `LoweredOp::Ln(Box::new(x))` |
| `sin(x)` | `LoweredOp::Sin(Box::new(x))` |
| `cos(x)` | `LoweredOp::Cos(Box::new(x))` |
| `sqrt(x)` | `LoweredOp::Sqrt(Box::new(x))` |
| `|x|` | `LoweredOp::Abs(Box::new(x))` |

#### Core CAS Functions

| Function | Module | Description |
|----------|--------|-------------|
| `eval_real(op, ctx)` | `eml::eval` | Evaluate as f64 |
| `eval_complex(op, vars)` | `eml::eval` | Evaluate as Complex64 |
| `to_latex(op)` | `eml::display` | Render as LaTeX |
| `canonicalize(op)` | `cas::canonicalize` | Canonical normal form |
| `grad_canonical(f, wrt)` | `cas::ad` | Symbolic gradient |
| `jacobian_canonical(f, n)` | `cas::ad` | Full Jacobian |
| `hessian_canonical(f, n)` | `cas::ad` | Full Hessian |
| `solve(lhs, rhs, var)` | `cas::solve` | Algebraic equation |
| `solve_system(eqs, vars)` | `cas::solve_system` | System of equations |
| `solve_ode(rhs, x, t, ic)` | `cas::solve_ode` | ODE solver |
| `integrate_rational(n, d, var)` | `cas::integrate_rational` | Rational integration |
| `to_jit(op)` | `compile::jit` | Cranelift JIT (feature=jit) |
| `to_gpu(op)` | `compile::gpu` | WGSL codegen (feature=gpu) |

### Key Error Types

| Error | Context |
|-------|---------|
| `EmlError::UnboundVariableIndex` | Index out of bounds in `EvalCtx` |
| `EmlError::EvalDomain` | ln(≤0), sqrt(<0), etc. |
| `SolveError::HighDegreePoly` | Polynomial degree ≥ 3 |
| `SolveError::CannotSeparate` | Variable in multiple branches |
| `SolveOdeError::NotRecognized` | ODE family not matched |
| `IntegrateRationalError::DenominatorDegreeTooHigh` | Denominator degree ≥ 5 |
| `AdError::DimMismatch` | Wrong number of variables in VJP/JVP |
| `JitError::NotEnoughVars` | Too few bindings for JIT eval |

### Feature Flag Summary

```toml
[dependencies]
scirs2-symbolic = { version = "0.5", features = [
    "jit",     # Cranelift JIT compilation
    "gpu",     # WGSL GPU codegen (implies jit)
    "smt",     # OxiZ SMT solver
    "serde",   # serde/JSON serialization
    "numa",    # NUMA-aware parallel symbolic regression
    "macros",  # eml_pattern! / eml_template! proc macros
    "parallel", # rayon parallel evaluation
] }
```

### Idempotence and Correctness Properties

1. `canonicalize(canonicalize(e).into_op()) == canonicalize(e)` — idempotent
2. `eval_real(&grad_canonical(&f, i), ctx) ≈ (f(x+h) - f(x-h))/(2h)` — central diff check
3. `eval_real(&integrate_rational_result, ctx)` differentiates back to the integrand (numerical round-trip)

All three properties are verified in the test suite. Use them as debug invariants.

### Upcoming Features (TODO)

The project TODO.md lists the following items in development:

- **v0.5.x**: NUMA `par_map_chunks` integration for SR discovery (Phase 1 unlock)
- **v0.5.x**: `GpuKernel::eval_batch` real wgpu dispatch (Phase 2)
- **v0.5.x**: `solve_ode` for SINDy-style `discover_ode` hardening (negative coefficient targets)
- **v0.6**: Full Galois-theory integration for degree ≥ 5 denominators in Risch-LITE
- **v0.6**: E-graph saturation for certified rewrites (`cas::e_graph`)

### Documentation Links

- Rustdoc: `cargo doc -p scirs2-symbolic --open --all-features`
- Project repository: https://github.com/cool-japan/scirs
- Issue tracker: https://github.com/cool-japan/scirs/issues
- arXiv paper: Odrzywolek (2026) arXiv:2603.21852 — EML substrate foundation

### Contributing

The CAS is built with the following invariants that all contributions must maintain:

1. **No `unwrap()` in production code** — use `expect("reason")` in tests,
   `?` in production.
2. **No recursion on `LoweredOp`** — all traversals use iterative work-stacks;
   a 543-node-deep tree (canonical sin) must not overflow the OS stack.
3. **No C/Fortran dependencies** — Pure Rust via OxiBLAS, OxiFFT, OxiCode.
4. **Zero clippy warnings** — `cargo clippy --workspace --all-features -- -D warnings`
   must be clean.

To run the tutorial's compile tests:
```bash
cargo nextest run -p scirs2-symbolic --test cas_tutorial_compile --all-features
```

To run the full test suite:
```bash
cargo nextest run --workspace --all-features \
    --exclude scirs2-python --exclude scirs2-datasets
```
