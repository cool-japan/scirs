# scirs2-symbolic-macros

Proc-macro DSL for `scirs2-symbolic` pattern-matching rewrite rules.

## Package metadata

| Field      | Value                                       |
|------------|---------------------------------------------|
| Version    | 0.6.5                                       |
| License    | Apache-2.0                                  |
| Repository | <https://github.com/cool-japan/scirs>       |
| Authors    | COOLJAPAN OU (Team KitaSan)                 |
| Docs       | <https://docs.rs/scirs2-symbolic-macros>    |

## What this crate provides

| Macro           | Purpose                                                                          |
|-----------------|----------------------------------------------------------------------------------|
| `eml_pattern!`  | Build a `Pattern` from a concise DSL expression (left-hand side of a rule).     |
| `eml_template!` | Same syntax; conventionally marks the *right-hand side* of a rewrite rule.      |

Both macros emit fully-qualified `scirs2_symbolic::cas::pattern` code — no `use` imports
are required at the call site.

## Mini-DSL reference

```text
?0, ?1, ?2          → PatVar(0), PatVar(1), PatVar(2)          (wildcard captures)
var(0), var(1)      → PatGroundVar(0), PatGroundVar(1)          (ground variable)
const(f)            → PatConst(f as f64)                         (float/int literal)
int(n)              → PatConstInt(n as u32)                      (integer constant)
add(A, B)           → PatOp2(BinaryKind::Add, A, B)
sub(A, B)           → PatOp2(BinaryKind::Sub, A, B)
mul(A, B)           → PatOp2(BinaryKind::Mul, A, B)
div(A, B)           → PatOp2(BinaryKind::Div, A, B)
pow(A, B)           → PatOp2(BinaryKind::Pow, A, B)
neg(A)              → PatOp1(UnaryKind::Neg, A)
sin(A) … tanh(A)    → PatOp1(UnaryKind::Sin …)
exp(A), ln(A)       → PatOp1(UnaryKind::Exp / Ln)
sqrt(A), abs(A)     → PatOp1(UnaryKind::Sqrt / Abs)
arcsin … arctanh    → PatOp1(UnaryKind::Arcsin …)
```

## Usage example

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
scirs2-symbolic = "0.6.5"
scirs2-symbolic-macros = "0.6.5"
```

Define a rewrite rule with the DSL macros:

```rust
use scirs2_symbolic_macros::{eml_pattern, eml_template};

// Rule: exp(ln(?0)) → ?0
let lhs = eml_pattern!(exp(ln(?0)));
let rhs = eml_template!(?0);
```

The macros expand at compile time into `Pattern` enum constructors from
`scirs2_symbolic::cas::pattern`, so there is zero runtime overhead from the DSL syntax.

## Part of the SciRS2 ecosystem

This crate is a helper for [scirs2-symbolic](https://github.com/cool-japan/scirs/tree/master/scirs2-symbolic)
and is part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem —
a pure-Rust port of SciPy with AI/ML extensions.
