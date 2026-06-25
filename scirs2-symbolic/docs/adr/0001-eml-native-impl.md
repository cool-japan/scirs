# ADR-0001: Clean-Room Native EML Implementation in scirs2-symbolic

## Status

Accepted — 2026-05-03

## Context

`scirs2-symbolic` was originally a minimal CAS (14-variant `Expr` enum, recursive diff/simplify/eval, `thiserror` as the only dep). For v0.4.4 we re-anchored the crate around the **EML construction** (Odrzywolek, 2026; [arXiv:2603.21852](https://arxiv.org/abs/2603.21852), v2 published 2026-04-04), which proves that every elementary function reduces to a single binary operator `eml(x, y) = exp(x) - ln(y)` plus the constant `1`.

The reference implementation is [`oxieml`](https://crates.io/crates/oxieml) v0.1 — a battle-tested Pure Rust crate by COOLJAPAN OU that ships symbolic regression, lowering, JIT, SMT, dimensional analysis, interval arithmetic, multi-output discovery, ODE/PDE discovery, and PyO3+WASM bindings. `oxieml` validates the substrate; `scirs2-symbolic` could either:

1. **Depend on `oxieml` at runtime** and add a thin SciRS2-flavored adapter layer
2. **Implement EML clean-room** inside `scirs2-symbolic` and use `oxieml` only as a `[dev-dependencies]` parity reference

## Decision

We choose option 2 — **clean-room native EML implementation** — with `oxieml` strictly as a `[dev-dependencies]` parity test harness.

Clean-room applies to **IR design and API surface**. Already-validated numerical kernels (Adam state update, MCTS rollouts, Lentz continued fractions, the iterative post-order eval loop, etc.) MAY be ported with an attribution comment of the form:

```rust
// Adapted from oxieml v0.1.0, src/<path>.rs
```

## Consequences

### Positive

- **No circular dependency risk.** `scirs2-symbolic` may depend on any SciRS2 crate (subject to ADR-0001's cycle-prevention rule against `scirs2-core` as a consumer); `oxieml` never enters the production dependency graph.
- **Free use of SciRS2 substrate.** EML symbolic regression can use `scirs2-core::parallel::numa` directly (Phase 1 design-freedom unlock), the GPU pipeline for batched JIT eval, structured tracing, and the unified memory pool — none of which `oxieml` exposes.
- **API surface independence.** scirs2-symbolic can offer ndarray-first `regression::discover(features: &Array2<f64>, targets: &Array1<f64>, config: &SrConfig)` ergonomics tailored to the SciRS2 ecosystem, rather than tracking `oxieml`'s DataFrame-style entry points.
- **Native `LoweredOp::Sqrt` and `LoweredOp::Abs` variants** — these would be cumbersome to thread through an `oxieml`-bridge layer. With a clean-room IR we add them as first-class variants and the gradient pass uses the closed-form `d/dx √f = f' / (2·√f)` rule, avoiding the `(0.5) · x^(-0.5) · dx` blow-up at `x = 0` that `oxieml`'s `Pow(_, 0.5)` lowering produces.
- **Compatibility with no-`unwrap()` policy.** `oxieml` uses `assert!(n > 0)` in `Canonical::nat`; the SciRS2 version returns `Result<EmlTree, EmlError>` with an `EmlError::InvalidConstant` variant.

### Negative

- **Maintenance burden of two implementations.** Bug fixes in `oxieml` must be tracked and (where applicable) ported to `scirs2-symbolic`. Mitigation: the parity test harness (`tests/oxieml_parity.rs`) catches numerical divergence at 1e-12 tolerance on every PR.
- **Cost of porting kernels.** Adam, MCTS, Lentz CF, etc. take development time even with attribution. Mitigation: kernels are well-isolated functions; porting is mechanical translation, not redesign.
- **Risk of intentional divergence drift.** Three intentional divergences are documented (native Sqrt/Abs, outward 1-ULP widening on intervals, `nat(0)` returns Result not panic). Each must be preserved across `oxieml` version bumps. Mitigation: pinned `divergence_*` tests in the parity harness.

## Alternatives Considered

### Alternative A — Depend on `oxieml` at runtime

Rejected because:

1. **Cycle risk.** `oxieml` does not currently depend on any `scirs2-*` crate, but if it ever did (e.g. for SciRS2 substrate features), `scirs2-symbolic → oxieml → scirs2-X → ... → scirs2-symbolic` would deadlock the build.
2. **Substrate access.** `oxieml` uses `rayon` for parallelism; `scirs2-symbolic`'s Phase 1 design-freedom plan requires `scirs2-core::parallel::numa` for NUMA-aware SR. A bridge layer can't unlock substrate features `oxieml` doesn't expose.
3. **API ergonomics.** `oxieml`'s primary entry points are designed for general use; SciRS2 wants ndarray-first signatures consistent with its ecosystem.
4. **Independent evolution.** SciRS2 wants the freedom to add SciRS2-specific features (e.g. `units::SiUnit` interop with `scirs2-stats` distributions) without coordinating with upstream `oxieml` releases.

### Alternative B — Vendor `oxieml` as a path = "../oxieml" sibling

Rejected because vendoring without attribution is incompatible with `oxieml`'s license expectations and creates a worse maintenance burden than a clean-room port (every upstream change requires a manual sync).

## Attribution Policy

Already-validated `oxieml` numerical kernels MAY be ported into `scirs2-symbolic` with an attribution comment of the form:

```rust
// Adapted from oxieml v0.1.0, src/<path>.rs
//
// <One-sentence note on what was adapted and any divergences>
```

Examples of kernels appropriate for porting:
- The iterative post-order eval loop pattern (oxieml/src/lower.rs:538-633)
- The simplify pattern recognisers (oxieml/src/lower_simplify.rs:127-169)
- The sin/cos critical-point interval splits (oxieml/src/lower_interval.rs:347-392)
- Lentz continued fraction kernels (oxieml/src/eval.rs)
- Adam optimizer state update (when Phase 1 SR engine is built)

Examples of items that should be CLEAN-ROOM (no port):
- The IR layout (`EmlNode`, `EmlTree`, `LoweredOp`)
- The public API trait surface (`ToLowered`, `FromLowered`, `Canonical`)
- The hash-cons strategy (clean-room — `oxieml` does not hash-cons)
- All `pub fn` signatures
- Documentation and examples

## Cycle-Prevention Rule (encoded in ADR)

`scirs2-symbolic` MUST NOT appear in the dependency tree of `scirs2-core`. Enforcement: `scripts/check-no-symbolic-in-core.sh` runs in CI. The script also verifies `oxieml` does NOT appear in the production dependency graph (only in `[dev-dependencies]`). See Phase 0 item 13.

## References

- Odrzywolek, A. (2026). EML Construction. arXiv:[2603.21852](https://arxiv.org/abs/2603.21852), v2 published 2026-04-04.
- `oxieml` v0.1 source: https://github.com/cool-japan/oxieml
- `scirs2-symbolic` TODO.md: `scirs2-symbolic/TODO.md` (619 lines, 84 items, Phases 0-4)
- v0.4.4 implementation plan: `$HOME/.claude/plans/indexed-plotting-lampson.md`

---
*ADR authored 2026-05-03 by Claude (Opus 4.7) under direction of COOLJAPAN OU (Team KitaSan).*
