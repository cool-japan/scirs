# scirs2-symbolic TODO — v0.5.0 and Beyond

**Subtitle.** Position `scirs2-symbolic` as the world's first **general-purpose computer algebra system whose canonical form is the EML uniform binary tree** for SciRS2, guided by the [oxieml](https://github.com/cool-japan/oxieml) reference implementation (Odrzywolek 2026, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852), v2 published 2026-04-04). oxieml proved the substrate; `scirs2-symbolic` builds the computer algebra system on top.

The EML construction shows — constructively — that every elementary function reduces to a single binary operator `eml(x, y) = exp(x) − ln(y)` plus the constant `1`. As of 2026-05-03, oxieml v0.1.1 ships the canonical phylogenetic tree (~30 functions), Adam + beam + MCTS symbolic regression, an OxiZ-backed SMT layer, a Cranelift JIT, SI dimensional analysis, interval arithmetic, multi-output and ODE/PDE discovery, serde + JSON + oxicode round-trip, and PyO3 + WASM bindings — that is, oxieml is the reference *symbolic-regression engine* that uses EML. `scirs2-symbolic` is a different category of artifact: the first general-purpose CAS to adopt this single-operator construction as its *foundational IR*. The uniformity of the EML grammar — alphabet of size 2, depth-bounded by construction — collapses canonical-form simplification to structural hashing, makes every algebraic rewrite a candidate for SMT-certification, narrows neural-guided topology search to a finite leaf-set, and (for the first time) gives a CAS where every elementary identity reduces to tree equality. Cross-pollinated across the rest of SciRS2: symbolic gradients feed `scirs2-optimize` Newton solvers; ODE-discovery feeds `scirs2-integrate`; symbolic priors regularize `scirs2-neural`; closed-form simplification accelerates `scirs2-linalg` matrix-function evaluation. The strategic bet is that an ecosystem-wide EML-IR-native CAS produces a category of capability — provably-sound rewriting, dimensional consistency by construction, neural search amortized over a tiny grammar, and seamless symbolic↔numerical fusion — that no existing CAS (SymPy, Mathematica, Maple, Maxima, GiNaC, Cadabra) can deliver.

This file tracks every work item from v0.4.4 onward. The item template (Why / Design / Files / Tests / Risk / size tag) mirrors `$HOME/work/oxieml/TODO.md`. The architectural decision is final: `scirs2-symbolic` ships a **clean-room, SciRS2-native EML implementation** built directly on `scirs2-core`, `scirs2-autograd`, and `scirs2-linalg`, with `oxieml` appearing **only as a `[dev-dependencies]` entry** for a parity test harness. This eliminates path/cycle concerns and frees the crate to use SciRS2 substrate features (NUMA-aware scheduler, GPU pipeline, structured tracing) directly. Clean-room applies to IR design and API surface — already-validated numerical kernels (Adam state update, MCTS rollouts, Lentz continued fractions, etc.) MAY be ported with an attribution comment of the form `// Adapted from oxieml v0.1.1, src/<path>.rs`. The existing `scirs2_symbolic::Expr` API stays as a deprecation-target shim through the v0.4.x series; new EML IR lives natively in `scirs2_symbolic::eml::*`; `Expr` becomes a `#[deprecated]` shim no earlier than v0.6.x.

---

## Test status as of 2026-07-15 (v0.6.1)

scirs2-symbolic's own test suite (freshly run 2026-07-15): 947 tests pass, 0 failed, 0 skipped (default
features — the crate's default feature set is empty, so this is the pure-Rust core with no `smt`/`jit`/
`gpu`/`cuda`/`macros`/`numa`/`serde`/`parallel` extras); 1058 tests pass, 0 failed, 0 skipped
(`--all-features`, exercising all of the above). See "Correctness hardening — e-graph extraction
operand-swap bug (2026-07-15)" near the end of this file for the one real bug found and fixed this
session.

---

## Status as of 2026-05-15 (v0.5.0, post Waves 59–67)

- **Phase 0** (substrate): COMPLETE (13/13 items) — Wave 53
- **Phase 1** (native API): 11/13 items complete — Waves 54–56
  - Done: `regression::discover`, `discover_multi`, `discover_ode`, `units::UnitAware`,
    `regression::with_constraints`, `compile::to_jit`, `interval::eval_interval`,
    LaTeX/pretty/JSON/oxicode round-trip, `scirs2-python::symbolic` bindings,
    physics examples, integration test suite (759 `#[test]` markers as of Wave 67)
  - Pending: FSReD benchmark suite vs PySR
- **Phase 1 Design-freedom unlocks**: 3/4 complete + 1 partial
  - Done: `compile::to_gpu` (WGSL JIT, `to_jit_auto`); seamless `scirs2-autograd`
    integration via `EmlOp` and `eml_scalar_op` (provenance-dispatched);
    SMT calls OxiZ directly (Wave 57 real QF_NRA integration)
  - Partial: SR engine NUMA-aware scheduling — rayon path shipped, `scirs2-core`
    NUMA worker pinning deferred to v0.4.5
- **Phase 2** (EML-IR-native CAS centerpiece): 15/15 complete — Waves 57 + 59 + 61 + 62 + 63 + 64 + 65 + 66 + 67
  - Done: `cas::canonicalize` (32 tests, Wave 57); `cas::pattern` (712 LoC, Wave 59
    prerequisite); `cas::identity_db` (11 standard identities, 73 tests, Wave 59);
    `cas::smt` Ackermann encoding (16 transcendental ops, Pythagorean axiom, 10 tests,
    Wave 59); `cas::certified_rewrite` (RAII push/pop, MAX_CERT_ITER=8, 7 tests, Wave 59);
    `cas/e_graph/` (6-file egg-style engine, 1,983 LoC, 16 tests, Wave 59);
    `units::infer_dimension` + `Constraint::DimensionMatch` (22 tests, Wave 59)
  - Note: `cas::pattern` is counted as a support item under e-graphs to keep denominator at 15;
    `DimensionMatch` is a `units` Phase-1 extension, not a standalone Phase-2 slot
  - Done (Wave 61): `cas::cse_dag::CseDag` (O(unique-nodes) Kahn topo eval, 11 tests);
    `cas::series` (taylor + pade via iterated grad + Gaussian elimination, 8 tests);
    proptest EML-rewrite suite (3 properties × 1024 cases, tests/cas_rewrite_proptest.rs)
  - Done (Wave 62): `cas::certified_value::CertifiedValue` (certified [lo,hi] interval;
    `certify`, `certify_const`, `tighten_to`; MAX_TIGHTEN_ITERS=64; 9 tests)
  - Done (Wave 63): `cas::solve` — invertible-chain unwinding + polynomial detection (degree 1-2
    exact, degree≥3 HighDegreePoly); `SolveResult`, `SolveError`; 1,086 LoC; 10 tests
  - Done (Wave 67): `cas::ad::GradGraph` — CSE-shared canonical AD (736 LoC, 16 tests);
    `scirs2-symbolic-wasm` Pratt-parser playground (622 LoC, 15 tests, standalone crate)
- **Phase 3** (cross-crate integration): 15/12+ complete — Waves 57 + 60 + 62 + 63 + 67 + 70
  - Done (Wave 57): `scirs2-optimize::symbolic::newton` (6 tests);
    `scirs2-autograd::symbolic_backend::EmlOp` + `eml_scalar_op` (8 integration tests)
  - Done (Wave 60): `scirs2-integrate::eml` (solve_ivp_symbolic + quad_gauss_legendre_symbolic,
    15 tests); `scirs2-stats::mle_symbolic` (fit_mle_symbolic, 8 tests);
    `scirs2-neural::{activations,losses}::symbolic` (SymbolicActivation + SymbolicLoss, 10 tests);
    `scirs2-linalg::symbolic` (det_symbolic, eigenvalues_symbolic_2x2,
    condition_number_symbolic, 12 tests)
  - Done (Wave 62): scirs2-autograd: float-tape vs EML gradient parity suite (12 ops × 100 points)
  - Done (Wave 62): scirs2-optimize: L-BFGS + trust-region with exact symbolic gradient; 8 tests
  - Done (Wave 63): scirs2-optimize: Lagrangian + KKT — `build_kkt` + `solve_lagrangian_symbolic`;
    Newton on N×N KKT system; `KktSystem`, `LagrangianError`; 6 tests (x=y=0.5 verified)
  - Done (Wave 67): `scirs2-autograd` EML tape backend — `tape/eml_tape.rs` (EmlElementWiseOp,
    EmlJacobianOp, EmlHessianOp, 326 LoC) + `tape/dispatch.rs` (74 LoC); 10 tests in
    `tests/eml_tape_tests.rs`
  - Done (Wave 68): Benchmark vs float-tape on MLP/transformer (`scirs2-autograd/benches/eml_vs_tape.rs`, 392 LoC, complete)
  - Done (Wave 70): `cas::moments_catalog` (8 tests, scirs2-stats hook); `cas::expected_fisher_catalog` (4 tests, scirs2-stats hook); `cas::noether_conservation` (10 tests, scirs2-integrate hook); `scirs2-neural::symbolic::rope_attention` (9 tests, closed-form RoPE attention logit)
- **Phase 4** (research): 9/N complete — Wave 68 (7 items above) + Wave 69 `cas::reversible` + Wave 70 `cas::integrate_rational` (Risch-LITE: 16 tests)

Cross-references:
- `scirs2-autograd::symbolic_backend::EmlOp` — `scirs2-autograd/src/symbolic_backend.rs`
- `scirs2-optimize::symbolic` — `scirs2-optimize/src/symbolic.rs`
- `scirs2-python::symbolic` — `scirs2-python/src/symbolic.rs`
- Workspace CHANGELOG.md `[0.4.4]` covers the full release scope

---

## Phase 0 — Foundations (must land first, blocks everything else)

The clean-room, SciRS2-native EML substrate. Every item below is implemented inside `scirs2-symbolic`; `oxieml` is consulted only as the reference implementation. Numerical kernels MAY be ported with an attribution comment of the form `// Adapted from oxieml v0.1.1, src/<path>.rs`; IR design and API surface are clean-room.

- [x] **`scirs2_symbolic::eml::tree` — `EmlNode` and `EmlTree`** — Arc-shared uniform binary tree. `[small]` (completed 2026-05-03)
  - **Why:** This is the foundational data structure for the entire EML IR. Leaves `One` and inner nodes `Eml(left, right)` plus `Var(usize)` for parameters; `Arc`-shared so structural-hash CSE costs nothing in memory. Without this primitive, no other Phase 0 item compiles.
  - **Design:** Mirror oxieml's `src/tree.rs` shape but use `Arc<EmlNode>` from `std::sync` (or a `scirs2-core` shared-pointer abstraction if one is exposed). `enum EmlNode { One, Var(usize), Eml(Arc<EmlNode>, Arc<EmlNode>) }`. `struct EmlTree { root: Arc<EmlNode> }`. Provide `depth()`, `structural_hash() -> u128`, `Eq`/`Hash`/`Clone` (Clone is `Arc::clone`, deep-clone-safe). All construction goes through smart constructors that hash-cons via a thread-local `WeakValueHashMap` so identical subtrees share `Arc` instances.
  - **Files:** `scirs2-symbolic/src/eml/tree.rs`.
  - **Tests:** `depth(eml(eml(1,1), 1)) == 2`; structural hash agrees on equivalent shapes; equality matches hash; deep-clone safety (cloning across thread boundaries does not corrupt the hash-cons table).
  - **Risk:** Hash-cons table memory growth on long-running processes — mitigation: weak-pointer eviction; document the size invariant.

- [x] **`Canonical` phylogenetic-tree constructors (~30 functions)** — every elementary function in the paper's Figure 1. `[medium]` (completed 2026-05-03)
  - **Why:** This is the bridge from human-readable math to the EML IR. Without canonical constructors for `exp`, `ln`, `sin`, `cos`, `sqrt`, `pow`, `add`, `sub`, `mul`, `div`, etc., every downstream item has to re-derive the EML shapes by hand.
  - **Design:** Module `Canonical` exposes one function per elementary op: `exp(t)`, `ln(t)`, `e()`, `sin(t)`, `cos(t)`, `tan(t)`, `arcsin(t)`, `arccos(t)`, `arctan(t)`, `sinh(t)`, `cosh(t)`, `tanh(t)`, `arcsinh(t)`, `arccosh(t)`, `arctanh(t)`, `sqrt(t)`, `abs(t)`, `pow(b, e)`, `add(a, b)`, `sub(a, b)`, `mul(a, b)`, `div(a, b)`, `neg(t)`, `reciprocal(t)`, `pi()`, `zero()`, `nat(n: u64)`, `imag_unit()`. Each returns the canonical `EmlTree` from the paper. Reference: `oxieml/src/canonical.rs`. Doc-comments cite the paper section and the oxieml reference function.
  - **Files:** `scirs2-symbolic/src/eml/canonical.rs`.
  - **Tests:** Golden depth-counts matching the paper's Tables 1–7; numerical evaluation matches `f64::sin`/`f64::cos`/etc. to 1e-12 at 100 random points; constructors are deterministic (two calls produce structurally equal `EmlTree`s).
  - **Risk:** Constructor count (~30 items) — a single missing function blocks everything depending on it. Mitigation: enumerate all ~30 in a single PR; CI golden-table test catches drift.

- [x] **`LoweredOp` IR + lowering rules** — post-order operator IR. `[medium]` (completed 2026-05-03)
  - **Why:** Many CAS algorithms (gradient, simplify, JIT, eval) are easier to express on a flat operator IR than on the recursive `EmlTree`. Lowering preserves canonical-shape recognition: `eml(ln(x), eml(y, 1)) → x − y` etc.
  - **Design:** `enum LoweredOp { Add(Box<L>, Box<L>), Sub, Mul, Div, Pow, Exp(Box<L>), Ln, Neg, Sin, Cos, Tan, Sinh, Cosh, Tanh, Arcsin, Arccos, Arctan, Arcsinh, Arccosh, Arctanh, Sqrt, Abs, Const(f64), Var(usize) }`. Lowering pass walks the `EmlTree` post-order and recognises canonical shapes, emitting the matching `LoweredOp`. Reference: `oxieml/src/lower.rs`. Inverse pass (`LoweredOp` → `EmlTree`) goes through `Canonical::*`.
  - **Files:** `scirs2-symbolic/src/eml/op.rs`, `scirs2-symbolic/src/eml/lower.rs`.
  - **Tests:** Round-trip lowering preserves evaluation to 1e-12 on 50 random expressions; canonical shape recognisers fire on hand-crafted EML trees (e.g. `eml(ln(x), eml(y, 1))` lowers to `Sub(Var(x), Var(y))`).
  - **Risk:** Recogniser misses an admissible EML shape, leading to silently sub-optimal lowering. Mitigation: maintain a registry of recognised shapes; CI test enumerates and asserts each.

- [x] **Parser + pretty-print + LaTeX export** — recursive descent over `E(x, y)` / `eml(x, y)` notation. `[small]` (completed 2026-05-03)
  - **Why:** Notebooks, papers, and command-line tools need a textual round-trip. LaTeX is the standard publication format; `to_compact_string()` is the standard debug format; the parser closes the loop for round-trip serialisation.
  - **Design:** `parse(s: &str) -> Result<EmlTree>` accepts both `E(x, y)` and `eml(x, y)` notation. `to_compact_string(&EmlTree) -> String` round-trips with the parser. `to_latex(&LoweredOp) -> String` emits LaTeX with π/e detection (a node whose canonical hash matches `Canonical::pi()` renders as `\pi`). Reference: `oxieml/src/parser.rs`, the `to_latex` impl in `oxieml/src/lower.rs`.
  - **Files:** `scirs2-symbolic/src/eml/parser.rs`, `scirs2-symbolic/src/eml/display.rs`.
  - **Tests:** `parse(format(t)) == t` for 100 random trees; LaTeX golden strings for `sin(x)`, `cos(x)`, `exp(x)`, `ln(x)`, and the Pythagorean identity.
  - **Risk:** Parser ambiguity on `eml(eml(x, y), z)` vs `eml(x, eml(y, z))` — left-associative by default, documented.

- [x] **Stack-machine evaluator (real + complex)** — post-order traversal, no recursion. `[medium]` (completed 2026-05-03)
  - **Why:** Sin/cos canonical trees alone are 543 nodes deep; any recursive evaluator overflows the OS stack on canonical inputs. A stack-machine traversal scales linearly without stack pressure. Complex evaluation is required because `pi = ln(-1)`.
  - **Design:** `EvalCtx` carries `bindings: Vec<f64>` (or `Vec<Complex64>`) indexed by `Var(usize)`, and an internal evaluation stack. Real path uses `f64`; complex path uses `num_complex::Complex64`. `pub fn eval(op: &LoweredOp, ctx: &EvalCtx) -> Result<f64>` and `eval_complex(...)`. Reference: `oxieml/src/eval.rs`.
  - **Files:** `scirs2-symbolic/src/eml/eval.rs`.
  - **Tests:** Scalar parity with `f64::sin`/`f64::cos`/`f64::exp`/`f64::ln` on 100 random inputs to 1e-12; complex parity for `sin`/`cos`/`exp` on 50 random complex inputs to 1e-10; deep-tree (depth 1000) evaluation does not overflow.
  - **Risk:** Performance gap vs JIT — interpreter is the correctness baseline; JIT comes later (Phase 1 design-freedom item). Document the order.

- [x] **Basic `simplify` pass** — constant folding, exp/ln inverse cancellation, structural-hash CSE, identity elimination. `[medium]` (completed 2026-05-03)
  - **Why:** Even the most basic CAS surface (gradient, eval, parser round-trip) accumulates redundant subtrees that explode tree depth. Without a simplify pass, `grad(grad(grad(x)))` is unusable.
  - **Design:** Apply rewrite rules to fixed point: `ln(exp(x)) → x`, `exp(ln(x)) → x`, `x + 0 → x`, `x * 1 → x`, `x / 1 → x`, `x ^ 0 → 1`, `x ^ 1 → x`, constant folding when both operands are `Const`, structural-hash CSE (any two subtrees with equal canonical hash share an `Arc`). Reference: `oxieml/src/simplify.rs` and `lower_simplify.rs`. Returns a new `LoweredOp` (input unchanged).
  - **Files:** `scirs2-symbolic/src/eml/simplify.rs`.
  - **Tests:** Idempotence (`simplify(simplify(x)) == simplify(x)`); golden strings for each rule; structural-hash agrees on equivalent forms; numerical equality (1e-12) of `simplify(x)` and `x` at 100 random points.
  - **Risk:** Non-confluent rule set causes oscillation. Mitigation: priority ordering on rules; max-iteration cap with a warn (not panic) on cap-hit.

- [x] **Symbolic gradient on `LoweredOp`** — chain/product/quotient/Pow-via-exp-log rules. `[medium]` (completed 2026-05-03)
  - **Why:** This is the centerpiece of the symbolic↔numerical fusion thesis. A user calls `grad(f, x)` and gets a *symbolic* expression — inspectable, simplifiable, JIT-compilable — not a tape-bound float. Every Phase 3 cross-crate item depends on this primitive.
  - **Design:** `pub fn grad(f: &LoweredOp, wrt: usize) -> LoweredOp` returns the symbolic derivative w.r.t. `Var(wrt)`. Implement chain rule for transcendentals, product rule for `Mul`, quotient rule for `Div`, `Pow` via `exp(b * ln(a))` lowered. Result is run through `simplify` before return so downstream consumers see the canonical form. Reference: `oxieml/src/lower_grad.rs`.
  - **Files:** `scirs2-symbolic/src/eml/grad.rs`.
  - **Tests:** 20 hand-picked closed-form derivatives (`grad(sin(x), x) = cos(x)`, `grad(x*y, x) = y`, etc.) verified by structural equality after `canonicalize`; proptest with 1024 random expressions cross-checked against central differences with tolerance `max(1e-5·|expected|, 1e-7)`.
  - **Risk:** Branch-cut handling on `Pow` with non-integer exponents — document the convention; emit `Sqrt` as a special case to avoid the `exp(0.5 * ln(x))` blow-up at `x = 0`.

- [x] **Interval arithmetic on `LoweredOp`** — `eval_interval(&[Interval]) -> Interval` for range analysis. `[small]` (completed 2026-05-03)
  - **Why:** Adaptive integrators (Phase 3 `scirs2-integrate`) and trust-region optimisers (Phase 3 `scirs2-optimize`) consume range bounds. Pure-symbolic interval evaluation gives them tight, rigorous bounds without sampling.
  - **Design:** `pub fn eval_interval(op: &LoweredOp, boxes: &[Interval]) -> Interval` evaluates the `LoweredOp` post-order with rounding-aware interval rules. Transcendentals split at monotone-region boundaries (e.g. `sin` over an interval that crosses `π/2` is split into the rising and falling parts and merged). `Interval { lo: f64, hi: f64 }` with constructors `point(x)`, `wide(lo, hi)`. Reference: `oxieml/src/lower_interval.rs`.
  - **Files:** `scirs2-symbolic/src/eml/interval.rs`.
  - **Tests:** Containment property — point eval ∈ interval eval on 1000 random points × 100 random formulas; tight bound on monotone functions (e.g. `eval_interval(exp, [0, 1])` equals `[1, e]` to 1e-15).
  - **Risk:** Interval blow-up on long expressions due to dependency loss. Mitigation: document; recommend the structural-hash-CSE form before interval eval.

- [x] **`Expr ↔ scirs2_symbolic::eml::LoweredOp` adapter** — backward-compat bridge between the two same-crate IRs. `[small]` (completed 2026-05-03)
  - **Why:** The existing public `Expr` API is consumed by 29 SciRS2 crates today. Without an adapter, every consumer breaks the moment we introduce the EML IR. Keeping the adapter inside `scirs2-symbolic` (no cross-crate path) makes it dramatically simpler than the original cross-crate bridge it replaces.
  - **Design:** Implement on both sides. `Expr::to_lowered(&self) -> Result<LoweredOp>` walks the existing `Expr` enum (`Const`, `Var`, `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Neg`, plus the transcendentals `Exp`, `Ln`, `Sin`, `Cos`, `Sqrt`, `Abs`) and emits the matching `LoweredOp` variant via `Canonical::*` constructors. `LoweredOp::to_expr(&self) -> Result<Expr>` is the structural inverse. A `VarMap { names: Vec<String> }` thread the `Var(usize)` ↔ named-variable mapping. New traits `ToLowered` and `FromLowered` so future `Expr` variants need only two methods.
  - **Files:** `scirs2-symbolic/src/eml/bridge.rs`, `scirs2-symbolic/src/expr.rs` (add the trait impl), `scirs2-symbolic/src/eml/mod.rs` (export `bridge`).
  - **Tests:** Golden round-trip on every `Expr` variant individually; 50-expression random-tree property test asserting `to_expr(to_lowered(e)) == simplify(e)` (round-trip equality after canonicalisation, not bit-equality, because `Mul(2, x)` and `Mul(x, 2)` lower to the same EML tree); cross-evaluation on 100 random points within `[-10, 10]^n` to 1e-12.
  - **Risk:** Loss of operator metadata — `Expr` carries variable names as `&'static str`; `LoweredOp` uses `usize` indices. Mitigation: `VarMap` is a first-class type owned by the conversion call.

- [x] **`oxieml` parity test harness** — load-bearing item. `[medium]` (completed 2026-05-03)
  - **Why:** This is how we mechanically check that the clean-room `scirs2-symbolic` implementation produces the same numerical answers as the reference. Without it, we are flying blind on every implementation choice. With it, every divergence is caught at PR time.
  - **Design:** Add `oxieml = { version = "0.1.1", default-features = false }` to `[dev-dependencies]` in `scirs2-symbolic/Cargo.toml`. Create `tests/oxieml_parity.rs` that, for every public `scirs2-symbolic` API surface item (canonical constructors, gradient, evaluation, simplify), constructs the equivalent `oxieml` expression and asserts numerical agreement to 1e-12 at 100 random points. Pin the oxieml version explicitly. Document intentional divergence points (e.g., `scirs2-symbolic` uses `scirs2-core` parallel primitives, oxieml uses `rayon` directly). Bump cadence is quarterly (TODO: revisit if oxieml ships parity-affecting fixes).
  - **Files:** `scirs2-symbolic/Cargo.toml`, `scirs2-symbolic/tests/oxieml_parity.rs`.
  - **Tests:** The harness itself is the test; CI gate.
  - **Risk:** **Unresolved tension.** Quarterly bump cadence may be insufficient if oxieml ships a fix that `scirs2-symbolic` depends on for parity validation; faster bump cadence trades drift-stability for fix-availability. Documented as an open question for the maintainer.

- [x] **Add `oxiz` to SciRS2 workspace dependencies** — register OxiZ so Phase 1 `cas::smt` and Phase 2 certified-rewrite engine can pull it in via `oxiz = { workspace = true }`. `[trivial]` (completed 2026-05-03)
  - **Why:** As of 2026-05-03 the SciRS2 workspace `Cargo.toml` has zero `oxiz` references (verified by `grep -n "oxiz" Cargo.toml`). Phase 1's "SMT calls OxiZ directly" item and Phase 2's SMT-certified rewrite engine both need OxiZ at runtime; without a workspace-level registration each consuming crate has to pin the version itself, violating workspace policy. OxiZ is itself a leaf in the dependency DAG (its workspace at `~/work/oxiz/` declares no `scirs2-*` or `oxieml` deps), so no cycle risk.
  - **Design:** Add `oxiz = { version = "0.2.1", default-features = false }` to `Cargo.toml [workspace.dependencies]`. In `scirs2-symbolic/Cargo.toml`, add `oxiz = { workspace = true, optional = true }` under `[dependencies]` and a `smt = ["dep:oxiz"]` entry under `[features]`. Default features stay empty (`default = []`) so consumers without SMT pay zero compile cost. Document in `scirs2-symbolic/README.md` that the `smt` feature pulls in OxiZ.
  - **Files:** `Cargo.toml`, `scirs2-symbolic/Cargo.toml`, `scirs2-symbolic/README.md`.
  - **Tests:** `cargo metadata --format-version 1 | jq '.workspace_metadata'` shows the new dep; `cargo check -p scirs2-symbolic` (no features) and `cargo check -p scirs2-symbolic --features smt` both succeed.
  - **Risk:** OxiZ minor-version bumps may break the `oxiz::Context` API consumed by Phase 1's `EmlSmtSolver`. Mitigation: pin to `0.2.x`; bump explicitly via the same quarterly cadence as the oxieml parity pin; document the sync point in the parity test harness's README.

- [x] **Architecture decision record (ADR)** — write `docs/adr/0001-eml-native-impl.md` capturing the clean-room-with-reference decision. `[trivial]` (completed 2026-05-03)
  - **Why:** Future maintainers will ask "why didn't we just depend on `oxieml` at runtime?" The answer (cycle concerns + design freedom for GPU/NUMA via `scirs2-core` + IR/API independence) needs to live somewhere durable. Also documents the attribution policy for ported numerical kernels.
  - **Design:** Single markdown file under `scirs2-symbolic/docs/adr/0001-eml-native-impl.md`. Sections: Status, Context, Decision (clean-room native impl, oxieml as dev-dep parity reference), Consequences, Alternatives Considered. Attribution policy: kernels MAY be ported with comment `// Adapted from oxieml v0.1.1, src/<path>.rs`. Reference: this TODO.md, oxieml v0.1.1, the 29-crate dep graph.
  - **Files:** `scirs2-symbolic/docs/adr/0001-eml-native-impl.md`.
  - **Tests:** None (documentation).
  - **Risk:** None.

- [x] **Compile-time non-goal: `scirs2-symbolic` MUST NOT be a dep of `scirs2-core`** — assert this with a CI check. `[trivial]` (completed 2026-05-03)
  - **Why:** Cycle prevention. `scirs2-core` is the universal substrate; if it grows a dep on `scirs2-symbolic`, every CAS bug becomes a workspace-wide compile failure. Future agents may forget this — encode it mechanically. Independent of the oxieml decision; the rule still applies.
  - **Design:** New script `scripts/check-no-symbolic-in-core.sh` that runs `cargo metadata --format-version 1 | jq -e '.packages[] | select(.name == "scirs2-core") | .dependencies[] | select(.name == "scirs2-symbolic") | error("scirs2-core depends on scirs2-symbolic — cycle!")'`. Wire into existing CI matrix as `lint-cycle-prevention` job. Document the rule in `scirs2-symbolic`'s `README.md`. The same script also asserts `oxieml` does NOT appear in the production dep graph of `scirs2-symbolic` (only in `[dev-dependencies]`).
  - **Files:** `scripts/check-no-symbolic-in-core.sh`, `scirs2-symbolic/README.md`.
  - **Tests:** Script exits 0 on current workspace; injecting a fake dep in a sandbox makes it exit non-zero; `oxieml` in `[dependencies]` (vs `[dev-dependencies]`) trips the script.
  - **Risk:** None.

---

## Phase 1 — First-class SciRS2-native EML API surface (v0.4.4 / v0.4.5)

The goal of this phase: a SciRS2 user with `scirs2 = { features = ["symbolic"] }` should reach symbolic regression, JIT, SMT, units, intervals, and serde-round-trip through native `scirs2_symbolic::*` modules — never importing `oxieml::*`. Every API is ndarray-first to match SciRS2 conventions, and every implementation lives natively in this crate (oxieml is consulted as the reference, not invoked at runtime).

- [x] **`scirs2_symbolic::regression::discover` ndarray API** — primary symbolic-regression entry point. `[small]` (completed 2026-05-04)
  - **Why:** SciRS2 users expect `scirs2_symbolic::regression::*` as a native ndarray-first API. `Config` is owned by this crate so we can evolve the public surface independently of any reference implementation.
  - **Design:** New module `scirs2-symbolic/src/regression/mod.rs`. Public function:
    ```rust
    pub fn discover(
        features: ArrayView2<'_, f64>,
        targets: ArrayView1<'_, f64>,
        config: &Config,
    ) -> Result<Pareto>
    ```
    `Config` is a SciRS2-native builder exposing `max_depth`, `learning_rate`, `tolerance`, `cv_folds`, `loss`, `strategy`, `seed`, `parallel`. Internally driven by the native `SymRegEngine` (see design-freedom item below) over `LoweredOp` topologies. `Pareto { entries: Vec<DiscoveredFormula> }` exposes SciRS2-style accessors (`best_by_mse`, `best_by_complexity`, `iter_pareto`).
  - **Files:** `scirs2-symbolic/src/regression/mod.rs`, `scirs2-symbolic/src/regression/config.rs`, `scirs2-symbolic/src/regression/pareto.rs`.
  - **Tests:** Parity with the oxieml reference on three canonical datasets (pendulum, harmonic oscillator, projectile motion) via the parity harness — Pareto fronts equal under topology hashing; numerical assertions to 1e-12.
  - **Risk:** Native engine performance regression vs oxieml reference on tuned cases. Mitigation: criterion benchmark item (later in this Phase) tracks the gap.

- [x] **`scirs2_symbolic::regression::discover_multi` for vector-valued outputs** — multi-output SR API. `[small]` (completed 2026-05-04)
  - **Why:** Vector physics (Lorenz, double pendulum, Navier-Stokes-on-grid) needs co-evolved formulas. The native `SymRegEngine` exposes `discover_multi` directly through the ndarray-first SciRS2 surface.
  - **Design:** Signature `pub fn discover_multi(features: ArrayView2<f64>, targets: ArrayView2<f64>, config: &Config) -> Result<Vec<Pareto>>`. Internally calls the native `SymRegEngine::discover_multi` with `MultiOutputStrategy::{Independent, SharedTopology}` taken from `Config.multi_output`.
  - **Files:** `scirs2-symbolic/src/regression/mod.rs`.
  - **Tests:** Lorenz attractor — three-component target; both strategies recover topology correctly within budget.
  - **Risk:** `SharedTopology` cost blows up on > 5 outputs. Document the recommendation; default to `Independent`.

- [x] **`scirs2_symbolic::regression::discover_ode` SINDy-style API** — discover ODE right-hand sides from trajectories. `[small]` (completed 2026-05-04)
  - **Why:** `scirs2-integrate` consumes trajectories; the EML symbolic-regression engine inverts trajectories into ODE right-hand sides natively. Wrapper integrates with `scirs2-integrate` trajectory types.
  - **Design:** Signature `pub fn discover_ode(trajectory: ArrayView2<f64>, time: ArrayView1<f64>, config: &OdeConfig) -> Result<Vec<DiscoveredFormula>>`. `OdeConfig` extends `Config` with `derivative_method: DerivativeMethod::{CentralDiff, SavitzkyGolay { window, order }}`.
  - **Files:** `scirs2-symbolic/src/regression/ode.rs`.
  - **Tests:** Lorenz trajectory generated by `scirs2-integrate` RK45; recovered RHS coefficients match σ=10, ρ=28, β=8/3 within 5%.
  - **Risk:** Numerical differentiation noise dominates fit at low sampling rates. Document recommended dt < 0.05 for Lorenz.

- [x] **`scirs2_symbolic::units::UnitAware` SI-unit-aware regression** — dimensional analysis pruning. `[small]` (completed 2026-05-04)
  - **Why:** Physics users routinely have unit metadata for every column. The native `Units` algebra enforces dimensional consistency at topology-construction time; this exposes it through a builder so users tag features once and get unit-correct formulas back.
  - **Design:** New `UnitAware { features, targets, feature_units, target_units, config }` builder; `.discover()` invokes the native SR engine with `Config.units = Some(...)`. Units constructed via `Units::si(meter: i8, kg: i8, sec: i8, ampere: i8, kelvin: i8, mol: i8, candela: i8)`.
  - **Files:** `scirs2-symbolic/src/units/mod.rs`.
  - **Tests:** Pendulum with feature units `{L: m, g: m/s²}`, target unit `s` — solver rejects `T = L + g` (unit mismatch) and returns `T = 2π√(L/g)`.
  - **Risk:** Unit metadata representation drift across sub-modules — keep the SI-vector as the canonical form.

- [x] **`scirs2_symbolic::regression::with_constraints` SMT-pruned search** — invoke `EmlSmtSolver` for topology pruning. `[medium]` (completed 2026-05-04)
  - **Why:** Users with extra-data constraints (monotonicity, range-bounded outputs, oddness, evenness) deserve to skip the topologies that violate the constraints before Adam fitting burns cycles on them.
  - **Design:** Builder `with_constraints(self, constraints: Vec<EmlConstraint>) -> Self`; constraints expressible via convenience constructors (`monotone_increasing(var)`, `range_bounded(var, lo, hi)`, `odd_function(var)`). Internally pre-pass: for each candidate topology, query `EmlSmtSolver::check_sat` against the conjunction; reject UNSAT topologies before Adam fitting. SMT layer is the native `cas::smt` module (Phase 1 design-freedom item) that talks to OxiZ directly.
  - **Files:** `scirs2-symbolic/src/regression/constraints.rs`.
  - **Tests:** Synthetic data generated from `f(x) = exp(-x²)` (positive-valued, even, monotone-decreasing on x > 0); monotonicity constraint reduces fit-time by ≥ 30% with no quality loss.
  - **Risk:** SMT call overhead exceeds Adam fit cost on small topologies — gate behind a depth threshold (default: SMT only for depth ≥ 5).

- [x] **`scirs2_symbolic::compile::to_jit` Cranelift JIT API** — `LoweredOp` to native machine code. `[trivial]` (completed 2026-05-04)
  - **Why:** Hot-loop integration in `scirs2-optimize` and `scirs2-integrate` needs near-native eval speed. Native Cranelift backend lowers `LoweredOp` to machine code with a hash-keyed cache.
  - **Design:** `pub fn to_jit(op: &LoweredOp) -> Result<JitFn>` and `pub fn to_jit_batch(ops: &[LoweredOp]) -> Result<Vec<JitFn>>`. `JitCache` keyed by structural hash for reuse. Behind `jit` feature.
  - **Files:** `scirs2-symbolic/src/compile/mod.rs`.
  - **Tests:** Compile `sin(x) + cos(x)`, eval at 10⁶ random points, assert ≥ 5× speedup over `LoweredOp::eval` interpreter.
  - **Risk:** Cranelift compile time can dominate on tiny ops; cache-friendly API (`to_jit_batch`) recommended.

- [x] **`scirs2_symbolic::interval::eval_interval` range analysis** — public surface for the Phase 0 `eval_interval` primitive. `[trivial]` (completed 2026-05-03)
  - **Why:** `scirs2-integrate` adaptive solvers and `scirs2-optimize` trust-region methods both consume range bounds. The native interval IR is the right primitive.
  - **Design:** `pub fn eval_interval(op: &LoweredOp, boxes: &[Interval]) -> Interval`. Re-exports `Interval { lo: f64, hi: f64 }` from `scirs2_symbolic::eml::interval`. Provides convenience constructors (`Interval::point(x)`, `Interval::wide(lo, hi)`).
  - **Files:** `scirs2-symbolic/src/interval/mod.rs`.
  - **Tests:** Containment property (point eval ∈ interval eval) on 1000 random points × 100 random formulas.
  - **Risk:** None significant.

- [x] **LaTeX / pretty / JSON / oxicode round-trip serialization** — public surface for the Phase 0 IO primitives. `[small]` (completed 2026-05-03)
  - **Why:** Notebook and paper workflows need LaTeX; pipeline workflows need JSON; checkpointing needs oxicode binary. Native exporters cover all three; we standardise the API.
  - **Design:** `scirs2_symbolic::io::{to_latex, to_pretty, to_json, from_json, to_oxicode, from_oxicode}` — generic over `LoweredOp`, `EmlTree`, `DiscoveredFormula`. Behind `serde` feature for the binary forms. JSON uses `serde_json`; binary uses `oxicode`.
  - **Files:** `scirs2-symbolic/src/io/mod.rs`.
  - **Tests:** Round-trip property: `from_X(to_X(t)) == t` for each format on the canonical-tree corpus (~30 functions).
  - **Risk:** None.

- [x] **Python bindings — native PyO3 surface** — wrap `scirs2-symbolic` directly so Python users get SR + JIT + SMT immediately. `[medium]` (completed 2026-05-04)
  - **Why:** `scirs2-python` is the SciRS2 Python crate; it should expose symbolic capabilities through native PyO3 wrappers, not by piggybacking on a separate crate's bindings.
  - **Design:** New crate slice `scirs2-symbolic-python` (or extend `scirs2-python` with a `symbolic` feature) that depends on `scirs2-symbolic` and exposes native PyO3 classes (`PyEmlTree`, `PyLoweredOp`, `PySymRegEngine`, etc.) under module path `scirs2.symbolic.*`. Maturin pyproject.toml in `python/` directory; CI via existing `pypi-publish.yml`.
  - **Files:** `scirs2-symbolic/python/scirs2_symbolic/__init__.py`, `scirs2-symbolic/python/pyproject.toml`, `scirs2-symbolic/src/python.rs`.
  - **Tests:** pytest suite, importing `from scirs2.symbolic import discover, eval_interval, to_jit`.
  - **Risk:** ABI churn on `pyo3` major bumps. Mitigation: pin via workspace dep.

- [x] **Examples directory: physics applications** — `examples/{physics_pipeline.rs, pendulum.rs, harmonic_oscillator.rs, lorenz.rs}` using the native SciRS2 API. `[small]` (completed 2026-05-03)
  - **Why:** Discoverability. Users who land on `scirs2-symbolic` README should see runnable examples that demonstrate the native EML surface.
  - **Design:** Each example reads or generates synthetic data, calls `scirs2_symbolic::regression::discover`, prints LaTeX, optionally JIT-compiles for batch eval. New `lorenz.rs` exercises `discover_multi`.
  - **Files:** `scirs2-symbolic/examples/physics_pipeline.rs`, `scirs2-symbolic/examples/pendulum.rs`, `scirs2-symbolic/examples/harmonic_oscillator.rs`, `scirs2-symbolic/examples/lorenz.rs`.
  - **Tests:** `cargo run --example pendulum --features parallel,simd` produces output matching a golden snapshot.
  - **Risk:** None.

- [x] **Integration test suite (30+ tests)** — exercise the native API against the v0.4.3 `Expr` API and against the oxieml dev-dep parity reference. `[medium]` (completed 2026-05-03)
  - **Why:** This is the primary safety net. Every gap between `Expr` semantics and `LoweredOp` semantics — and every divergence from the reference implementation — should be caught here, not in production downstream.
  - **Design:** New file `tests/eml_facade_test.rs`. Test classes: (a) round-trip `Expr ↔ LoweredOp` for every existing `Expr` variant; (b) eval parity at 100 random points to 1e-12; (c) parity vs the oxieml dev-dep reference for `discover`, `discover_multi`, `discover_ode`; (d) JIT parity vs interpreter; (e) interval containment; (f) units rejection of dimensionally-inadmissible candidates; (g) SMT pruning correctness; (h) serde round-trip in JSON and oxicode.
  - **Files:** `scirs2-symbolic/tests/eml_facade_test.rs`.
  - **Tests:** 30+ `#[test]` items, all passing under `cargo nextest run --features smt,simd,parallel,jit,serde`.
  - **Risk:** Test runtime — keep individual tests under 1 s; mark slow ones `#[ignore]`.

- [x] **Criterion benchmark suite vs PySR on FSReD (Rust-only baseline)** (completed 2026-05-07)
  - **Goal:** Land the in-process half of the FSReD bench. PySR side is documented external (Julia toolchain in CI is the documented blocker; the existing plan accepts this).
  - **Design:** New `scirs2-symbolic/benches/fsred_bench.rs` using criterion. Hardcode the first 20 FSReD equations (Feynman Lectures volume I — I.6.2, I.11.19, I.12.4, I.27.6, II.2.42, etc. — the standard "easy" subset). For each: (1) Synthesize 1000 sample points from Uniform([1,5]^k). (2) Compute y=equation(x_i). (3) Run `scirs2_symbolic::regression::discover` with default config. (4) Record (recovery_success: bool, mse: f64, time_ms: f64). (5) Output JSON to `scirs2-symbolic/target/fsred_results.json`. PySR side: ship `bench-comparison/run_pysr.py` and `compare.py` as STUBS with explicit manual-only comments. README.md in `bench-comparison/` documents the manual workflow.
  - **Files:** `scirs2-symbolic/benches/fsred_bench.rs` (new); `scirs2-symbolic/bench-comparison/run_pysr.py` (new, stub); `scirs2-symbolic/bench-comparison/compare.py` (new, stub); `scirs2-symbolic/bench-comparison/README.md` (new); `scirs2-symbolic/Cargo.toml` (criterion dev-dep already present).
  - **Prerequisites:** none beyond existing `regression::discover`.
  - **Tests:** `cargo bench --bench fsred_bench -p scirs2-symbolic` runs to completion; output JSON well-formed; per-equation runtime < 30s on default 4 cores.
  - **Risk:** CI time. Mitigation: 20 equations × 30s = 10 min manageable for nightly. Only run when explicitly invoked.

- [x] **`docs/cas_tutorial.md` walkthrough** — SR → simplify → JIT → deploy in one document. `[small]` (completed 2026-05-05 — Wave 68 Track 1; five-section end-to-end walkthrough covering SR discovery, canonical form, differentiation via `cas::ad::GradGraph`, JIT compilation, and serde round-trip deploy; README link added)
  - **Why:** Tutorials drive adoption. A 1500-word walkthrough that produces a closed-form formula from data, simplifies it, JIT-compiles it, and embeds it in a `scirs2-optimize` Newton step covers the canonical user journey.
  - **Design:** Markdown file with progressive code blocks, each runnable as a doctest. Sections: (1) Generate data; (2) Discover formula; (3) Inspect Pareto front; (4) Simplify and verify; (5) JIT compile; (6) Use in optimization loop. Cross-links to API docs.
  - **Files:** `scirs2-symbolic/docs/cas_tutorial.md`.
  - **Tests:** All code blocks execute via `cargo test --doc -p scirs2-symbolic`.
  - **Risk:** Tutorial drift as APIs evolve. Mitigation: doctest harness catches drift mechanically.

### Design-freedom unlocks (native impl only)

These four items are *only* possible because `scirs2-symbolic` is native — they exploit substrate features (`scirs2-core`, `scirs2-autograd`, OxiZ) directly, with no oxieml layer to route through.

- [x] **SR engine on `scirs2-core` NUMA-aware parallel scheduler** — replace generic rayon with the workspace's NUMA-aware work-stealing scheduler. `[medium]` (completed 2026-05-07 — wired to scirs2-core::par_map_chunks)
  - **Why:** On multi-socket systems, rayon's locality-blind scheduling causes 30–50% throughput loss on data-heavy SR fitness evaluation. The `scirs2-core` scheduler pins workers per NUMA node and routes feature-matrix slices accordingly.
  - **Design:** `predict_parallel` in `src/regression/discover.rs` now calls `scirs2_core::par_map_chunks` (chunk size 64) instead of `parallel_ops::parallel_map`. On Linux, worker threads are pinned to NUMA nodes via `pthread_setaffinity_np`; on Darwin/WASM the plain-rayon fallback fires transparently. The `NUMA_DISPATCH_THRESHOLD = 1024` constant controls the serial/parallel dispatch boundary.
  - **Files:** `scirs2-symbolic/src/regression/discover.rs`, `scirs2-symbolic/tests/regression_numa_tests.rs`.
  - **Tests:** 3 integration tests in `regression_numa_tests.rs` under `#[cfg(feature = "numa")]`: above-threshold correctness (4096 samples, MSE < 1e-12 vs serial), below-threshold correctness (256 samples), and constant assertion (`NUMA_DISPATCH_THRESHOLD == 1024`).
  - **Risk:** NUMA topology unavailable in CI VMs — `par_map_chunks` falls back to rayon automatically; no CI gate needed.

- [x] **JIT routes through `scirs2-core` GPU pipeline for batch eval at scale** — Cranelift handles CPU; for batches > 10⁵ points, lower the same `LoweredOp` sequence to the workspace GPU runtime. `[medium]` (completed 2026-05-04 — `compile::to_gpu` + `to_jit_auto`)
  - **Why:** Symbolic regression fitness evaluation on 10⁶+ points is the bottleneck. The workspace already ships a WebGPU + WGSL runtime with a memory pool; emitting WGSL from `LoweredOp` reuses all of it.
  - **Design:** New `compile::to_gpu(op: &LoweredOp) -> Result<GpuKernel>` lowers `LoweredOp` to WGSL via a per-op shader-template table; uses `scirs2-core::gpu` for buffer management and dispatch. `to_jit_auto` heuristic dispatches CPU JIT for batch < 10⁵ and GPU for batch ≥ 10⁵. Reuses the existing GPU memory pool.
  - **Files:** `scirs2-symbolic/src/compile/gpu.rs`, `scirs2-symbolic/src/compile/wgsl_templates.rs`.
  - **Tests:** Numerical parity (1e-10) between CPU JIT and GPU paths on `sin(x) + cos(x)` at 10⁶ random points; benchmark shows GPU wins above 10⁵ points on M-series Mac and discrete GPU.
  - **Risk:** WGSL transcendental support varies by adapter — feature-detect at runtime; fall back to CPU JIT on unsupported paths.

- [x] **Symbolic gradient as the *native* AD tape backend for `scirs2-autograd`** — no feature flag, no opt-in. `[medium]` (completed 2026-05-04 — see `scirs2-autograd::symbolic_backend::EmlOp`)
  - **Why:** When an `scirs2-autograd::Tensor` is constructed from a `scirs2_symbolic::eml::LoweredOp`, gradients should flow through the EML kernel by default — symbolic structure is preserved end-to-end, simplifications fire, JIT can compile the result. This is impossible with a separate-package autograd; requires both crates to share the substrate.
  - **Design:** New module `scirs2-symbolic/src/eml/autograd_bridge.rs` exposes a `LoweredOp::to_tape_node()` constructor for `scirs2-autograd`. Cross-crate file `scirs2-autograd/src/tape/eml_tape.rs` adds an `EmlTape` variant to the tape backend that defers to `scirs2_symbolic::cas::ad::grad`. Tape-based path remains for non-symbolic tensors. No feature flag — it just *is* a backend, dispatched by tensor provenance.
  - **Files:** `scirs2-symbolic/src/eml/autograd_bridge.rs`, `scirs2-autograd/src/tape/eml_tape.rs`.
  - **Tests:** Numerical parity (1e-10) between EML-tape `grad` and float-tape `grad` on 50 random expressions; speed parity (within 2×) of EML-tape-via-JIT vs float-tape on a 3-layer MLP forward+backward.
  - **Risk:** Performance regression on tape-friendly workloads (large reverse-mode neural-net training) — dispatch-by-provenance keeps the float-tape default for non-symbolic tensors so this case is unaffected.

- [x] **SMT calls OxiZ directly** — no oxieml wrapper layer. `[medium]` (completed 2026-05-04 — Wave 57 real QF_NRA integration)
  - **Why:** The Phase 2 SMT-certified rewrite engine and the Phase 1 constrained SR both need a fast SMT layer. Calling OxiZ natively (rather than through an oxieml shim) gives us the whole OxiZ feature surface, including LRA push/pop interactivity that is not exposed in oxieml's abstraction.
  - **Design:** New `scirs2_symbolic::cas::smt` module talks to `oxiz` 0.2 (workspace dep) directly. EML-aware interval propagation built on the native `eval_interval` (Phase 0) feeds OxiZ as initial bounds. `EmlSmtSolver { ctx: oxiz::Context, ... }` exposes `check_sat`, `push`, `pop`, `assert_constraint`, `model`. `EmlConstraint` is the native constraint AST.
  - **Files:** `scirs2-symbolic/src/eml/smt.rs`, `scirs2-symbolic/src/eml/smt_constraint.rs`.
  - **Tests:** SAT/UNSAT classification on the standard QF-NRA benchmark subset; `push`/`pop` interactivity preserved across queries; performance within 1.5× of raw OxiZ on the same problems.
  - **Risk:** OxiZ API churn on minor bumps — pin to a workspace minor; CI has a job that runs the parity tests against the latest OxiZ.

---

## Phase 2 — World-First: EML-IR-Native CAS (v0.4.5 / v0.4.6) — THE CENTERPIECE

This is the novelty. First general-purpose CAS whose canonical form is the EML uniform binary tree (oxieml proved the substrate; `scirs2-symbolic` builds the algebra system on top). No other CAS — SymPy, Mathematica, Maple, Maxima, GiNaC, Cadabra2, FriCAS — operates over Odrzywolek's single-binary-operator IR. Every item below describes a capability that is either impossible or infeasible without the EML uniformity.

- [x] **EML canonical form: `scirs2_symbolic::cas::canonicalize`** — every expression rewrites to a unique minimal-depth EML tree. `[large]` (completed 2026-05-04 — 7 algebraic rewrite rules; fixed-point idempotent; 32 tests)
  - **Why:** Two expressions are mathematically equal iff their canonical EML hashes match. This is the foundation of every other Phase 2 capability. The killer demo: prove `sin²(x) + cos²(x) = 1` by structural hash equality after canonicalization — something no traditional CAS can do *purely structurally*. Today, CAS systems detect this identity via hand-coded rules; here it falls out of the algebra.
  - **Design:** Pipeline: (a) lower input `Expr`/`LoweredOp` to `EmlTree`; (b) apply confluent rewriting via the native `scirs2_symbolic::eml::simplify` plus a new `scirs2_symbolic::cas::canonical_rules` set; (c) sort commutative-equivalent subtrees by structural hash so `Add(x, y)` and `Add(y, x)` produce identical canonical trees; (d) re-lower and minimize depth via DAG common-subexpression elimination using `EmlNode` `Arc` sharing. Output is a `Canonical` newtype around `EmlTree` with a `pub fn hash(&self) -> u128` whose equality on two values implies mathematical equality on a documented decidable subset of the language. Decidability boundary documented explicitly: trivially decidable for the polynomial subring, decidable-with-exceptions for analytic identities involving `sin`, `cos`, `exp`, `ln`, `sqrt` (cf. Richardson's theorem).
  - **Files:** `scirs2-symbolic/src/cas/canonical.rs`, `scirs2-symbolic/src/cas/mod.rs`.
  - **Tests:** Golden table of 500 known-equal expression pairs (Pythagorean identities, log-product rules, double-angle, hyperbolic-identities, exp-ln inverses); `canonicalize(lhs).hash() == canonicalize(rhs).hash()` for every pair. Negative tests: 100 known-NOT-equal pairs hash differently. Property test: `canonicalize(canonicalize(x)) == canonicalize(x)`.
  - **Risk:** Richardson's theorem says general elementary-function equality is undecidable. Mitigation: document decidable subset; provide `canonicalize_or_warn` returning `Result<Canonical, AmbiguityWarning>` for the boundary cases.

- [x] **EML structural-hash CSE across pipeline** — lift gradients, Hessians, Jacobians from many call sites into a single shared subexpression DAG keyed by EML hash. `[medium]` (completed 2026-05-04 — CseDag with O(unique-nodes) Kahn topo eval; 11 tests)
  - **Why:** A typical `scirs2-optimize` Newton step computes f, ∇f, ∇²f at the same point — these share nearly all subexpressions, but today each is computed independently. Hashing every subexpression in the lowered IR and reusing computed values across calls yields O(1) lookup of any previously-computed sub-result. This is impossible in CAS that lack a uniform IR.
  - **Design:** New `CseDag { nodes: HashMap<u128, LoweredOp>, parent_of: HashMap<u128, Vec<u128>> }`. Public API: `dag.add(op)` returns the canonical hash; `dag.get(hash)` returns the op; `dag.eval_all(point) -> HashMap<u128, f64>` evaluates every node once with topological order. Used by Newton solvers to eval f, ∇f, ∇²f in a single pass.
  - **Files:** `scirs2-symbolic/src/cas/cse_dag.rs`.
  - **Tests:** Construct a 3-variable function; assert `eval_all` performs n unique node evaluations (n = unique-node-count), not 3·n.
  - **Risk:** Hash collision — use 128-bit hash, document collision probability < 2⁻⁶⁴ per pipeline.

- [x] **SMT-certified rewrite engine** — every algebraic rewrite is paired with an OxiZ proof obligation; rules whose obligations cannot be discharged are rejected at registration time. `[large]` (completed 2026-05-04 — `CertifiedRule` trait, `rewrite_certified`, `rewrite_certified_fixpoint`, RAII push/pop safety, `MAX_CERT_ITER=8`, `smt` feature; 7 tests)
  - **Why:** **World-first.** Every CAS in production today has buggy simplification rules — Mathematica's `Simplify`, SymPy's `simplify`, Maple's `simplify` all have documented incorrectness on tricky domains (branch cuts, near-zero denominators, complex domain). A CAS where the rewrite system is mechanically sound by construction is a different category of artifact.
  - **Design:** New trait:
    ```rust
    pub trait CertifiedRule {
        fn lhs_pattern(&self) -> &EmlPattern;
        fn rhs_template(&self) -> &EmlTemplate;
        fn proof_obligation(&self, lhs: &LoweredOp) -> EmlConstraint;
    }
    ```
    Rule registration: `register_rule<R: CertifiedRule>(r: R) -> Result<RuleId, ProofFailure>`. The registration step calls `EmlSmtSolver::check_sat(&!proof_obligation)`; if SAT, the rule is rejected at registration time (counterexample reported). Rules that pass are added to the active set. The rewrite engine itself uses standard term-rewriting on the EML grammar; certificate proofs accumulated in a `RewriteTrace` for downstream consumers (Lean export, see Phase 4).
  - **Files:** `scirs2-symbolic/src/cas/certified.rs`, `scirs2-symbolic/src/cas/rule_db.rs`.
  - **Tests:** Plant a deliberately-buggy rule (`exp(x) = 1 + x` for all x — only true in a narrow neighborhood); registration rejects with counterexample `x = 1`. The standard rule set (Pythagorean, log-product, exp-of-ln) registers cleanly. End-to-end: simplifying `sin²(x) + cos²(x)` produces `1` *and* a valid certificate.
  - **Risk:** SMT timeouts on complicated proof obligations — provide `register_rule_unchecked` for known-good rules with a manual proof reference; document the trust boundary.

- [x] **EML-native equation solver: `scirs2_symbolic::cas::solve`** — completed 2026-05-04 — invertible-chain + degree-1/2 polynomial; 10 tests; degree≥3 deferred. Original design: Risch-style integration over the EML grammar; native single-variable kernel extended to systems via Gröbner-style elimination on EML structural equality. `[large]`
  - **Why:** Integration and equation solving are the two hardest CAS problems. The native single-variable solver handles invertible cases; we extend to polynomial-system solving via elimination on the structural hash, and to Risch-style integration via classification of EML topologies into Liouville-extension categories.
  - **Design:** Module `cas::solve` exposes:
    ```rust
    pub fn solve_equation(eq: &Equation, var: &str) -> SolveResult;
    pub fn solve_system(eqs: &[Equation], vars: &[&str]) -> SystemSolveResult;
    pub fn integrate(integrand: &LoweredOp, var: &str) -> IntegrateResult;
    ```
    For single-variable solving, delegate to the native `cas::solve` kernel (invertibility table over `LoweredOp`). For systems, build a Gröbner-like rewrite system on the EML structural-hash equivalence (treating equations as rewrite rules `lhs → rhs` keyed by canonical hash). For integration, classify the integrand against the Risch decision algorithm projected onto EML grammar — if the integrand lies in a Liouville extension over its variables, attempt the Risch tower; otherwise return `IntegrateResult::NoElementaryAntiderivative` with a witness.
  - **Files:** `scirs2-symbolic/src/cas/solve.rs`, `scirs2-symbolic/src/cas/integrate.rs`, `scirs2-symbolic/src/cas/equation.rs`.
  - **Tests:** Solve `x² + 2x + 1 = 0` → `x = -1`; solve system `{x + y = 3, x - y = 1}` → `{x = 2, y = 1}`; integrate `sin(x)` → `-cos(x) + C`; integrate `exp(x²)` → `NoElementaryAntiderivative` with Liouville witness.
  - **Risk:** Risch is famously deep (~50 papers worth of edge cases). Aim for the elementary-functions subset that EML covers natively; document gaps explicitly.

- [x] **Symbolic identity discovery from data** — given `(x, f(x))` pairs, *prove* that f matches a known closed form by combining native EML symbolic regression with canonical-form matching and OxiZ SAT certificates. `[research, large]` (completed 2026-05-04 — cas::identity_proof; SR + canonicalize + hash-match + ProofCertificate; 8 tests)
  - **Why:** Bridges discovery (Phase 1 SR) and verification (Phase 2 CAS). Result is a machine-checkable proof certificate that a numerical observation matches a known identity. No CAS does this end-to-end today.
  - **Design:** Pipeline: (a) run `discover` on the data to find candidate closed forms; (b) for each candidate, canonicalize and compare its hash against a precomputed identity-database hash table; (c) if match, query SMT for `∀x ∈ domain, |candidate(x) − target(x)| ≤ ε`; (d) if SAT-checked, emit a `ProofCertificate { candidate, matched_identity, smt_witness }`. Output certificates serializable to JSON for downstream tooling (and to Lean/Coq, see Phase 4).
  - **Files:** `scirs2-symbolic/src/cas/identity_proof.rs`.
  - **Tests:** Generate noisy `sin(x)` data; pipeline returns a certificate matching the `sin(x)` identity in the database; tampering with the data (replacing `sin` with `tan`) yields no match.
  - **Risk:** Database completeness — we never claim to identify identities not in the DB. Be explicit in error messages.

- [x] **EML-native automatic differentiation kernel** — replace `scirs2-autograd`'s float-tape backend (where applicable) with EML-IR symbolic adjoint that lowers to identical numerical code via Cranelift JIT. `[large]` (completed 2026-05-04 — cas::ad; GradGraph + grad_canonical + jacobian_canonical + hessian_canonical + vjp + jvp + batch_eval_grad + numerical_grad; 16 tests)
  - **Why:** **World-first.** A unified symbolic+numerical AD: you write `let g = grad(f, x);`, you get back a *symbolic* expression that you can simplify, inspect, and JIT-compile to native machine code via the same pipeline. No autograd library today has this property — they all materialize tape values immediately and lose the symbolic structure.
  - **Design:** New `scirs2_symbolic::cas::ad` module. Function `grad(f: &LoweredOp, wrt: &str) -> LoweredOp` builds on the Phase 0 native gradient. Function `jacobian` is the column-stack of `grad`. Provides the *native* AD tape backend for `scirs2-autograd` (no opt-in feature flag — see Phase 1 design-freedom item): when an `scirs2-autograd::Tensor` is constructed from a `LoweredOp`, gradients flow through the EML kernel by default, lowering symbolic adjoints with `to_jit` for fast eval. Benchmark vs the float-tape backend on standard MLP / transformer microbenchmarks.
  - **Files:** `scirs2-symbolic/src/cas/ad.rs`; cross-crate: `scirs2-autograd/src/tape/eml_tape.rs`.
  - **Tests:** Numerical parity (1e-10) between EML-AD `grad` and finite-difference on 50 random expressions; speed parity (within 2×) of EML-AD-via-JIT vs float-tape on a 3-layer MLP.
  - **Risk:** Performance regression on tape-friendly workloads (large neural nets) where reverse-mode tape outperforms symbolic. Mitigation: keep EML-backend as opt-in; route only when symbolic structure is preserved end-to-end.

- [x] **EML rewriter benchmarks vs SymPy / Mathematica `Simplify`** — AXIOM benchmark suite. `[medium]` (completed 2026-05-04 — criterion baseline in cas_bench.rs; SymPy subprocess comparison deferred to v0.4.5)
  - **Why:** We need credibility on the simplification axis. Target: ≤ 5× SymPy on the slow cases, with deterministic results (SymPy's `simplify` is famously non-deterministic across versions).
  - **Design:** Adopt the AXIOM CAS benchmark suite (or an equivalent: ECRH, the SymPy `slow` test suite). For each benchmark expression: time `scirs2_symbolic::cas::canonicalize` vs `sympy.simplify` (Python harness) vs `Simplify[...]` (Mathematica subprocess if available). Report wall-time and structural-hash equality of outputs.
  - **Files:** `scirs2-symbolic/benches/axiom_bench.rs`, `scirs2-symbolic/bench-comparison/run_sympy.py`.
  - **Tests:** Bench runs to completion; results stored as JSON for trend analysis.
  - **Risk:** Mathematica availability in CI (commercial license). Mitigation: SymPy comparison is mandatory; Mathematica is best-effort.

- [x] **Term rewriting via e-graphs with EML equivalence classes** — Tate-style equality saturation, but with EML's tiny alphabet drastically reducing rewrite explosion. `[large]` (completed 2026-05-04 — 6-file engine: `union_find.rs`, `enode.rs`, `egraph.rs`, `pattern.rs`, `budget.rs`, `extract.rs`; 1,983 LoC total; `cas::pattern` prerequisite module (712 LoC) added as support; `canonicalize_egraph` public entry; `SaturationBudget`; DP extraction; 16 tests)
  - **Why:** Equality saturation (`egg`-style) is the state of the art in compiler rewriting. EML's grammar — alphabet of size 2 — means the e-class explosion that plagues general egg applications is bounded by the tree-depth distribution. This is structural luck we should exploit.
  - **Design:** New `cas/e_graph/` module directory. `EClass { id: EClassId, nodes: Vec<EmlNode>, canonical_hash: u128 }`; `EGraph { classes: HashMap<EClassId, EClass>, hashcons: HashMap<EmlNode, EClassId> }`. Apply rewrite rules from the certified rule DB until saturation or step budget. Extract the minimal-cost representative from each e-class via dynamic programming. Native `scirs2_symbolic::eml::simplify` rules are lifted into the e-graph rewrite format directly. `cas::pattern` (prerequisite) provides the shared `EmlPattern` AST and `match_pattern` matcher.
  - **Files:** `scirs2-symbolic/src/cas/e_graph/union_find.rs`, `enode.rs`, `egraph.rs`, `pattern.rs`, `budget.rs`, `extract.rs`; `scirs2-symbolic/src/cas/pattern.rs` (prerequisite).
  - **Tests:** Egraph saturation on `(x + 1)·(x − 1)` produces an e-class containing both `(x + 1)·(x − 1)` and `x² − 1`; extraction by node count returns `x² − 1`; 16 tests total.
  - **Risk:** Saturation budget tuning — too low yields suboptimal rewrites, too high explodes memory. Default 1000 steps; document.

- [x] **Identity database** — serde-frozen, EML-hash-indexed table of ~10⁴ classical identities (trig, hyperbolic, log, beta/gamma/special-function relations). `[medium]` (completed 2026-05-04 — 11 standard trig/hyperbolic/log rules via `IdentityDb::standard()`; hooked into `cas::canonicalize` fixed-point loop; `IdentityRecord { name, lhs_canonical_hash, rhs_canonical_hash, conditions }` type; O(1) lookup via `HashMap<u128, Vec<IdentityRecord>>`; 73 tests)
  - **Why:** Looking up an arbitrary expression's identities becomes O(1) via canonical hash. This unlocks both fast simplification (try every applicable identity at zero search cost) and identity-discovery-from-data (Phase 2 item above).
  - **Design:** Curate identities from the CAS literature (Abramowitz & Stegun, NIST DLMF, SymPy `assumptions` test corpus). Each entry: `IdentityRecord { name: String, lhs_canonical_hash: u128, rhs_canonical_hash: u128, conditions: Option<EmlConstraint>, source_citation: String }`. Storage: oxicode-serialized in `data/identities.oc`. Loaded lazily into a `HashMap<u128, Vec<IdentityRecord>>` at first use. Build-time generation script ensures identities round-trip through canonicalize → hash before being committed.
  - **Files:** `scirs2-symbolic/src/cas/identity_db.rs`, `scirs2-symbolic/data/identities.oc`, `scirs2-symbolic/build/gen_identities.rs`.
  - **Tests:** 73 tests covering identity round-trips, canonical equality, negative tests, and lookup correctness.
  - **Risk:** Database curation cost. Mitigation: start with ~100 entries and grow; accept community contributions via PR.

- [x] **Transcendental closure decision (Ackermann encoding)** — given an expression, map transcendental ops to fresh OxiZ uninterpreted-function terms for SMT-based reasoning. `[research, large]` (completed 2026-05-04 — Ackermann reduction for 16 transcendental ops in `cas::smt`; `encode_transcendental` + `trans_cache: HashMap<u128, oxiz::TermId>`; Pythagorean axiom `sin²(x)+cos²(x)=1` auto-asserted on first sin/cos pair; cache keyed on canonical structural hash; 10 tests; `smt` feature)
  - **Why:** Combines the native EML lowering with a custom Liouville-extension tracker. Useful both as a math tool (proving non-elementarity of `∫ exp(x²) dx`) and as a CAS-internal short-circuit (when `solve` is called with a target that is provably non-elementary, return immediately with the obstruction).
  - **Design:** Ackermann reduction: each distinct transcendental sub-expression (Sin, Cos, Exp, Ln, Sqrt, Abs, and hyperbolic/inverse variants — 16 ops total) is mapped to a fresh uninterpreted OxiZ function term; functional-consistency axioms assert that equal inputs yield equal outputs; the Pythagorean axiom `sin²(x)+cos²(x)=1` is injected automatically. Full Liouville-tower tracker (algebraic/exponential/logarithmic field extensions) is deferred to v0.4.5.
  - **Files:** `scirs2-symbolic/src/cas/smt.rs` (extended), `scirs2-symbolic/src/cas/mod.rs`.
  - **Tests:** 10 tests on OxiZ SMT reasoning with transcendental encoding.
  - **Risk:** Decidability boundary again — Ackermann encoding is complete for function-consistency but does not decide general elementary-function identities. Full Liouville tracker deferred. Document.

- [x] **Verified numerical bounds: every closed-form value carries an interval certificate** — computed via `IntervalLO` + OxiZ. `[medium]` (completed 2026-05-04 — CertifiedValue; certified [lo,hi] interval; 9 tests)
  - **Why:** Reproducible numerical computing. When `cas::solve` returns `x = π`, the user often needs a certified numerical interval (e.g. `x ∈ [3.14159265358..., 3.14159265359...]`) for downstream rigor.
  - **Design:** New `CertifiedValue { closed_form: LoweredOp, certified_interval: IntervalLO, certificate: SmtCertificate }`. `cas::solve` returns `Vec<CertifiedValue>`. Interval computed via `LoweredOp::eval_interval` then sharpened via OxiZ SAT bounds.
  - **Files:** `scirs2-symbolic/src/cas/certified_value.rs`.
  - **Tests:** Certificate width for `π` < 1e-10; for `sqrt(2)` < 1e-10; certificates always contain the true value.
  - **Risk:** Performance — sharpening to 1e-10 width can be slow. Default to 1e-6, expose `tighten_to(eps)` for users who need more.

- [x] **Symbolic series expansions: Taylor + Padé in EML-canonical form** — series-truncation and rational-approximation under canonical hashing. `[medium]` (completed 2026-05-04 — cas::series; taylor + pade via iterated grad + Gaussian elimination; 8 tests)
  - **Why:** Numerical analysis (asymptotic expansions, ODE perturbation theory) consumes Taylor and Padé approximants. Computing them in EML form means subsequent simplification/canonicalization "just works".
  - **Design:** `cas::series::taylor(f: &LoweredOp, var: &str, around: f64, order: usize) -> LoweredOp` returns the truncated Taylor polynomial. `cas::series::pade(f: &LoweredOp, var: &str, around: f64, num_order: usize, den_order: usize) -> LoweredOp` returns the Padé approximant via the standard linear-algebra construction (delegated to `scirs2-linalg` for the `solve` step). Canonicalize result via `cas::canonicalize` so equivalent expansions hash equally.
  - **Files:** `scirs2-symbolic/src/cas/series.rs`.
  - **Tests:** `taylor(exp(x), "x", 0.0, 4)` produces `1 + x + x²/2 + x³/6 + x⁴/24` (modulo canonical form); `pade(exp(x), "x", 0.0, 2, 2)` produces the standard `[2,2]` Padé approximant of exp.
  - **Risk:** Round-off in the linear-algebra step for high-order Padé. Mitigation: warn at orders > 20; document numerical limitations.

- [x] **Property-based EML-rewrite testing via proptest** — random EML trees rewrite to canonical form; `canonical(rewrite(x)) == canonical(x)` for any rewrite ∈ rule database. `[small]` (completed 2026-05-04 — 3 properties × 1024 cases)
  - **Why:** The rewrite system has many rules; exhaustive testing is infeasible. Property-based testing on randomly-generated EML trees is the right safety net.
  - **Design:** proptest strategy generating `LoweredOp` trees up to depth 6. For each tree `t` and each rule `r ∈ rule_db`: assert `canonicalize(apply_rule(r, t)) == canonicalize(t)`. Run 1024 cases per CI run; persist failed seeds.
  - **Files:** `scirs2-symbolic/tests/cas_rewrite_proptest.rs`.
  - **Tests:** Implicit (this *is* the test). Acceptance: 1024 cases pass, no failed seeds in regression file.
  - **Risk:** Flaky tests on numerical edge cases (NaN, Inf). Mitigation: filter pathological points in the strategy.

- [x] **Browser playground (WASM)** — interactive notebook demonstrating EML canonicalization, SMT-certified rewrites, and SR live in a 1-page HTML app via native wasm bindings. `[medium]` (completed 2026-05-04 — scirs2-symbolic/wasm/; Pratt parser for EML expressions; wasm_canonicalize/wasm_grad/wasm_simplify/wasm_eval/wasm_is_identity; playground/index.html + main.js; 15 tests native)
  - **Why:** Discoverability + education. A `cas.scirs.dev/playground.html` page where a user types `sin(x)^2 + cos(x)^2` and sees `1` (with a clickable certificate) is the strongest possible demo.
  - **Design:** New `scirs2-symbolic-wasm` crate exposing `canonicalize`, `discover`, and `rewrite` to JS via native `wasm-bindgen` wrappers around `scirs2-symbolic`. Static-served HTML + a Monaco editor + a results pane. Hosted via GitHub Pages.
  - **Files:** `scirs2-symbolic/wasm/Cargo.toml`, `scirs2-symbolic/wasm/src/lib.rs`, `scirs2-symbolic/wasm/playground/index.html`, `scirs2-symbolic/wasm/playground/main.js`, `.github/workflows/npm-publish.yml` (extend for the playground build step).
  - **Tests:** Playground builds; canonicalize on `sin(x)^2 + cos(x)^2` returns `1` in browser.
  - **Risk:** WASM bundle size — keep minimal feature set (`eml + smt + serde`); document as "interactive demo" not production performance.

- [x] **EML pattern matching DSL** — `eml_pattern!` proc-macro for writing rewrite-rule patterns in source. `[medium]` (completed 2026-05-04 — scirs2-symbolic-macros crate; eml_pattern!/eml_template!; 13 tests)
  - **Why:** Phase 2 will accumulate hundreds of rewrite rules. Hand-coding each one as `EmlPattern::Eml(Box::new(EmlPattern::Var(0)), ...)` is unwieldy. A proc-macro that lets rule authors write `eml_pattern!(eml(?x, 1))` produces dramatically cleaner code.
  - **Design:** New crate slice `scirs2-symbolic-macros` (proc-macro crate). Macro `eml_pattern!(...)` parses a mini-DSL (`?x` = variable, `1` = constant one, `eml(a, b)` = binary node) into the corresponding `EmlPattern` value. Pair with `eml_template!(...)` for the right-hand side.
  - **Files:** `scirs2-symbolic-macros/Cargo.toml`, `scirs2-symbolic-macros/src/lib.rs`.
  - **Tests:** 13 integration tests in `scirs2-symbolic/tests/eml_pattern_macro_tests.rs` covering all operator kinds, wildcard binding, consistency checks, and round-trip instantiate.
  - **Risk:** Proc-macro compile time. Mitigation: keep the macro implementation under 500 lines.

- [x] **`cas::solve_system` — multivariate algebraic system solver** (completed 2026-05-07)
  - **Goal:** A native primitive that takes `&[(LoweredOp, LoweredOp)]` (lhs=rhs equations) and a target variable list `&[usize]`, returns `SystemSolveResult { solutions: Vec<HashMap<usize, LoweredOp>>, complete: bool, kind: SystemKind }`. Each solution is one branch of the variety. Powers `mle::derive` and `solve_ode`.
  - **Design:** Three-tier dispatch on canonical-form structure of residuals lhsᵢ − rhsᵢ: (1) **Linear path** (degree-1): build augmented matrix A|b over LoweredOp, run Gaussian elimination with structural pivot, back-substitute; (2) **Polynomial path** (degree≥2): Buchberger's algorithm in lex order, reusing `cas::solve::as_polynomial`; step budget MAX_BUCHBERGER_STEPS=256, overrun returns SystemKind::PartialGroebner; triangulated basis solved bottom-up via existing `cas::solve`; (3) **Transcendental fallback**: linear elimination first, then one-equation-one-unknown to `cas::solve`, else SystemSolveError::CannotEliminateTranscendental. Exposed as `pub fn solve_system(eqs: &[(LoweredOp, LoweredOp)], vars: &[usize]) -> Result<SystemSolveResult, SystemSolveError>`. Iterative — no recursion.
  - **Files:** `scirs2-symbolic/src/cas/solve_system.rs` (new), `scirs2-symbolic/src/cas/mod.rs` (re-export).
  - **Prerequisites:** `cas::solve::as_polynomial` (pub(crate) ✓); `cas::canonicalize` (✓); `cas::matrix_ops` (✓).
  - **Tests:** ≥15 in `tests/cas_solve_system_tests.rs`: linear 2×2, linear 3×3, circle∩line, two conics 4 solutions, inconsistent, underdetermined, 4×4 KKT, degree-3 HighDegreePoly bail, Buchberger overrun no panic, transcendental fallback success, transcendental bail.
  - **Risk:** Buchberger exponential worst case. Mitigation: MAX_BUCHBERGER_STEPS=256; float Gröbner uses Rational64::approximate_float for inner loop.
- [x] **`cas::solve_ode` — symbolic ODE solver** (completed 2026-05-07)
  - **Goal:** Native primitive recognising five closed-form ODE families; returns `OdeSolution { x_of_t: LoweredOp, integration_constants: Vec<usize>, kind: OdeKind }`. Powers `scirs2_integrate::symbolic_first`.
  - **Design:** Entry: `pub fn solve_ode(rhs: &LoweredOp, x_var: usize, t_var: usize, ic: Option<(f64,f64)>) -> Result<OdeSolution, SolveOdeError>`. Dispatch: (1) Linear 1st-order dx/dt=a·x+f(t): variation of parameters via cas::integrate_rational::try_integrate; (2) Linear 2nd-order: characteristic polynomial via cas::solve_system, basis {e^(λt), t·e^(λt)} for repeated roots, n>2 → OrderTooHigh; (3) Separable dx/dt=f(t)·g(x): canonicalised-tree detection, ∫dx/g(x)=∫f(t)dt+C; (4) Exact M dt+N dx=0 with ∂M/∂x=∂N/∂t: potential function via path integral; (5) Bernoulli dx/dt+p(t)x=q(t)x^n: u=x^(1-n) substitution → linear. IC via cas::solve_system. Iterative.
  - **Files:** `scirs2-symbolic/src/cas/solve_ode.rs` (new), `scirs2-symbolic/src/cas/mod.rs` (re-exports).
  - **Prerequisites:** `cas::solve_system` (above ✓); `cas::integrate_rational::try_integrate` (✓); `cas::canonicalize` (✓); `eml::grad` (✓).
  - **Tests:** ≥15 in `tests/cas_solve_ode_tests.rs`: dx/dt=x with x(0)=1→exp(t), dx/dt=-2x→3exp(-2t), dx/dt=x²+1→tan(t+C), dx/dt+2x=sin(t), 2nd-order d²x/dt²+ω²x=0, exact (2tx)dt+(t²+1)dx=0, Bernoulli dx/dt+x=x², Painlevé→OrderTooHigh, dx/dt=e^(x²)→IntegralNotElementary, IVP uniqueness, OdeKind classification, canonical invariance.
  - **Risk:** ODE detection brittle on near-canonical forms. Mitigation: pre-canonicalise input; symbolic IVP via cas::solve_system may bail → keep symbolic constants.

---

## Phase 3 — Cross-crate integration (4–6 high-leverage targets, v0.4.5 / v0.4.6 / v0.5.0)

This phase wires the EML-IR-native CAS into the rest of SciRS2. Each sub-section targets one downstream crate; each item identifies the highest-leverage symbolic-computation injection point in that crate.

**Status as of 2026-05-04 (Waves 57 + 60 + 62)**: Eight Phase-3 integrations are live.

*Wave 57 (original):* `scirs2-optimize::symbolic::newton` consumes a `LoweredOp` objective
and uses `eml::grad` to produce exact gradient and Hessian, with Gaussian-elimination linear
solve. `scirs2-autograd::symbolic_backend::EmlOp` and `eml_scalar_op` provide seamless
symbolic-tensor integration: forward routes through `eval_real`, backward through `sym_grad`,
both composable with stock autograd ops via provenance dispatch.

*Wave 60 (new):* Four additional Phase-3 integrations landed.
- `scirs2-integrate::eml`: `solve_ivp_symbolic` (BDF1 stiff ODE + symbolic Jacobian JIT-compiled once at entry), `quad_gauss_legendre_symbolic` (symbolic integrand lowered and JIT-compiled before quadrature); 15 tests
- `scirs2-stats::mle_symbolic`: `fit_mle_symbolic` — gradient descent with backtracking Armijo line search on symbolic NLL; `MleResult { params, log_likelihood, grad_norm }`; 8 tests
- `scirs2-neural::{activations,losses}::symbolic`: `SymbolicActivation` (Activation + Layer traits via `eval_real` + symbolic grad); `SymbolicLoss` (Loss trait); 10 tests
- `scirs2-linalg::symbolic`: `det_symbolic` (Leibniz formula, n ≤ 4), `eigenvalues_symbolic_2x2` (closed-form quadratic), `condition_number_symbolic`; 12 tests

*Wave 62 (new):* Two additional Phase-3 integrations landed.
- scirs2-autograd: float-tape vs EML gradient parity suite (12 ops × 100 points)
- scirs2-optimize: L-BFGS + trust-region with exact symbolic gradient; 8 tests

### 3.1 scirs2-autograd

- [x] **Symbolic gradient parity test suite** — for every `Tensor` op in `scirs2-autograd`, assert that the float-tape gradient matches the EML symbolic gradient at 100 random points. `[medium]` (completed 2026-05-04 — 12 ops × 100 deterministic points, |float_tape_grad − sym_grad| < 1e-10; ops: x², sin, cos, exp, ln, sqrt, x³, 1/x, tan, sinh, cosh, arctan; 1165 autograd tests total)
  - **Why:** Establishes correctness baseline before any backend swap. Catches divergences early.
  - **Design:** New `tests/eml_parity_test.rs` in `scirs2-autograd`. For each elementary op (`add`, `mul`, `exp`, `ln`, `sin`, ...): build a `Tensor` graph, call `.gradients()`, compare against `scirs2_symbolic::cas::ad::grad` of the equivalent `LoweredOp`.
  - **Files:** `scirs2-autograd/tests/eml_parity_test.rs`.
  - **Tests:** 100+ parity assertions, all green.
  - **Risk:** Float-tape uses different floating-point order — tolerance 1e-12, justify per-op.

- [x] **EML symbolic Jacobian/Hessian as the *native* tape backend** — provenance-dispatched, no feature flag. `[large]` (completed 2026-05-04 — eml_tape.rs: EmlElementWiseOp + EmlJacobianOp + EmlHessianOp + eml_elementwise/eml_jacobian/eml_hessian constructors; dispatch.rs: is_eml_backed + extract_lowered_op + try_build_symbolic_jacobian; gradient.rs dispatch for EmlElementWiseOp backward; 10 tests)
  - **Why:** Continues the Phase 2 AD item and the Phase 1 design-freedom item. End-to-end: any `scirs2-autograd::Tensor` constructed from a `scirs2_symbolic::eml::LoweredOp` flows gradients through the EML kernel by default — symbolic structure preserved, simplifications fire, JIT compiles the result. Float-tape remains the default for non-symbolic tensors. Dispatch is by tensor provenance, not by feature flag.
  - **Design:** Provenance-tagged `Tensor::gradients()` in `scirs2-autograd` checks for an `EmlTape` variant on the backing tape; when present, routes through `scirs2_symbolic::cas::ad` and JIT-compiles the resulting `LoweredOp`. Behavior is byte-identical for elementary ops; numerical-precision tests required. No feature flag — the EML path *is* a backend, lit up by tensor construction.
  - **Files:** `scirs2-autograd/src/tape/eml_tape.rs`, `scirs2-autograd/src/tape/dispatch.rs`.
  - **Tests:** Same as parity test but with the EML backend active; benchmark vs float-tape.
  - **Risk:** Reverse-mode performance gap on large graphs — document, recommend float-tape (default for non-symbolic tensors) for deep-NN training.

- [x] **Benchmark vs autograd float-tape on MLP / transformer microbenchmarks** — establish performance envelope. `[small]` (completed 2026-05-05 — Wave 68; `scirs2-autograd/benches/eml_vs_tape.rs` 392 LoC; criterion workloads: 3-layer MLP forward+backward, attention block forward+backward, scalar-loss optimization step; time/memory/symbolic-graph-size reported)
  - **Why:** Marketing + roadmap planning. Need numbers.
  - **Design:** Criterion benchmarks in `scirs2-autograd/benches/eml_vs_tape.rs`. Workloads: 3-layer MLP forward+backward, attention block forward+backward, scalar-loss optimization step. Report time, memory, and (for the symbolic backend) the size of the resulting symbolic graph.
  - **Files:** `scirs2-autograd/benches/eml_vs_tape.rs`.
  - **Tests:** Benchmarks complete; results stored to Bencher.dev.
  - **Risk:** None.

### 3.2 scirs2-optimize

- [x] **Symbolic-gradient feed for Newton / L-BFGS / trust-region** — skip finite-difference fallback when input is symbolic. `[medium]` (completed 2026-05-04 — Newton via scirs2_optimize::symbolic::newton (6 tests); L-BFGS via lbfgs_symbolic two-loop recursion + strong Wolfe (c1=1e-4, c2=0.9); trust-region via trust_region_symbolic dogleg + ρ-based radius update; SymbolicOptResult + SymbolicOptError; 8 new tests; 14 symbolic tests total)
  - **Why:** Today, when `scirs2-optimize` receives a closed-form objective, it numerically differentiates. With EML-IR-native gradients available, we can hand the solver an exact gradient (and Hessian) for free, eliminating finite-difference noise and step-size tuning.
  - **Design:** Extend `scirs2_optimize::OptimizationProblem` with an `Option<&LoweredOp>` field for the symbolic objective. Each solver (`newton`, `l_bfgs`, `trust_region_dogleg`) checks this field; if present, uses `cas::ad::grad` and `cas::ad::hessian` (JIT-compiled) instead of finite-difference. Document `Problem::from_symbolic(op: &LoweredOp)` constructor.
  - **Files:** `scirs2-optimize/src/problem.rs`, `scirs2-optimize/src/newton.rs`, `scirs2-optimize/src/l_bfgs.rs`.
  - **Tests:** Rosenbrock convergence in fewer steps with symbolic gradient than with finite-difference.
  - **Risk:** Gradient evaluation cost can exceed FD on small problems — provide a heuristic to fall back to FD when the symbolic graph is small.

- [x] **Symbolic-aware Lagrangian for constrained problems** — completed 2026-05-04 — `KktSystem` + `solve_lagrangian_symbolic` + Newton on KKT; 6 tests. Original design: express equality constraints in EML form, derive KKT conditions symbolically. `[medium]`
  - **Why:** Constrained optimization (interior-point, augmented Lagrangian) requires Jacobian-of-constraints. With EML, we get this for free in symbolic form, including correct handling of nonlinear constraints with branch cuts.
  - **Design:** New `scirs2_optimize::constrained::SymbolicConstrainedProblem`. KKT conditions assembled symbolically and JIT-compiled. Multiplier updates via standard augmented-Lagrangian recurrence.
  - **Files:** `scirs2-optimize/src/constrained/symbolic.rs`.
  - **Tests:** Solve `min x²+y² s.t. x+y=1` exactly; recover `x = y = 1/2`.
  - **Risk:** Constraint dependencies between variables — careful index-management on `VarMap`.

- [x] **Closed-form line-search step where derivable** — when the line-search subproblem (`min φ(α) s.t. α > 0` along a direction) is symbolically solvable, skip the iterative solver and apply the closed form. `[small]` (completed 2026-05-05 — Wave 68 Track 7; `cas::quadratic_line_search::closed_form_step` computes α* = −(∇f·d)/(dᵀHd) symbolically; `scirs2-optimize::symbolic::line_search::SymbolicLineSearch` wrapper for per-step evaluation; 7+2 tests; 0 clippy warnings)
  - **Why:** Quadratic models (e.g. Newton step on a quadratic) admit a closed-form line-search; this is widely known but rarely implemented because traditional optimizers don't have CAS access. With EML, we can detect quadraticity (degree analysis on the canonical form) and shortcut.
  - **Design:** New `scirs2_optimize::line_search::symbolic_step` that takes the objective `LoweredOp`, the current point, and the step direction; if the restricted function `φ(α)` canonicalizes to a low-degree polynomial, solve via `cas::solve` and return the optimal α directly.
  - **Files:** `scirs2-optimize/src/line_search/symbolic.rs`.
  - **Tests:** On a quadratic, symbolic line-search finds the exact minimum in one step.
  - **Risk:** Detection brittleness on near-quadratic functions; default to iterative line-search.

### 3.3 scirs2-integrate

- [x] **`scirs2_integrate::eml` — `solve_ivp_symbolic` + `quad_gauss_legendre_symbolic`** — stiff ODE solver and Gauss-Legendre quadrature using symbolic JIT-compiled kernels. `[medium]` (completed 2026-05-04 — BDF1 stiff ODE, symbolic Jacobian via `eml::grad` JIT-compiled once at entry; Gauss-Legendre quadrature with JIT-compiled integrand; `symbolic` feature; 15 tests)
  - **Files:** `scirs2-integrate/src/eml.rs`.
  - **Tests:** Stiff-system convergence, quadrature accuracy vs analytic values, 15 tests total.

- [x] **`scirs2_integrate::discover_ode_from_trajectory` first-class API** — wraps `scirs2_symbolic::regression::discover_ode`. `[small]` (completed 2026-05-05 — Wave 68 Track 8; `OdeDiscoveryConfig` builder with sensible defaults; thin facade over `regression::discover_ode`; `symbolic` feature; 4 integration tests)
  - **Why:** SINDy is a `scirs2-integrate` use-case (it discovers ODEs); the API should live there for discoverability.
  - **Design:** Thin facade; `discover_ode_from_trajectory(trajectory: ArrayView2<f64>, time: ArrayView1<f64>) -> Result<DiscoveredOde>`. `DiscoveredOde { rhs: Vec<LoweredOp>, jit_fn: Option<JitFn> }` for direct integration.
  - **Files:** `scirs2-integrate/src/discover/mod.rs`.
  - **Tests:** Lorenz system: discover → integrate via `scirs2-integrate`'s own RK45 → trajectory matches synthetic to 1e-3.
  - **Risk:** Coupling between integrate and symbolic — gate behind `symbolic` feature in `scirs2-integrate`.

- [x] **Closed-form ODE branches when `cas::solve_ode` succeeds** (completed 2026-05-07; supersedes original plan)
  - **Goal:** `scirs2_integrate::symbolic_first::solve_ode_symbolic_or_numerical` — attempt `cas::solve_ode` first; on success return LoweredOp-typed x(t); on failure fall back to existing numerical RK45/BDF.
  - **Design:** New `scirs2-integrate/src/symbolic_first.rs` behind `symbolic` feature. `SymbolicOrNumericalResult::{Symbolic{x_of_t, kind, integration_constants}, Numerical{trajectory, time}}`. Fallback via existing `solve_ivp`. ForceNumerical opt bypasses symbolic attempt.
  - **Files:** `scirs2-integrate/src/symbolic_first.rs` (new); `scirs2-integrate/src/lib.rs` (export under symbolic feature); `scirs2-integrate/Cargo.toml` (scirs2-symbolic already optional under symbolic feature).
  - **Prerequisites:** `cas::solve_ode` (above).
  - **Tests:** ≥6 in `tests/symbolic_first_tests.rs`: dx/dt=x→Symbolic JIT-eval matches exp(t) to 1e-12; Lorenz→Numerical fallback; Painlevé→numerical fallback; ForceNumerical bypass; dimension mismatch error.
  - **Cross-crate:** Plan block also in `scirs2-integrate/TODO.md`.

- [x] **Symbolic conservation-law detection (Noether-style)** — feeds adaptive integrators with invariant constraints. `[research, large]` (completed 2026-05-05 — Wave 70 Track 4; `cas::noether_conservation` with `poisson_bracket_1dof`, `poisson_bracket_ndof`, `check_conservation_1dof`, `check_conservation_ndof`, `first_integrals_1dof`; `ConservationCheck { poisson_bracket, is_conserved }`; conservation detected when bracket canonicalizes to `Const(c)` with `|c| < 1e-15`; verified on harmonic oscillator H = ½(p² + q²), free particle, 2-DOF angular momentum; 10 tests; 480 LoC)
  - **Why:** Many ODEs have conserved quantities (Hamiltonian, angular momentum) that should be preserved exactly during integration. Detecting them symbolically lets adaptive integrators *enforce* the conservation, not just hope for it.
  - **Design:** Given a vector ODE `dx/dt = f(x)`, search for first integrals `H(x)` such that `∇H · f = 0`. Use `cas::solve` to attempt the PDE for `H`; if successful, integrate with `H(x_n+1) = H(x_n)` as a constraint (projected RK methods).
  - **Files:** `scirs2-integrate/src/conservation.rs`.
  - **Tests:** Pendulum has `H = (1/2)θ̇² + (g/L)(1−cos θ)`; detector recovers it; symplectic integrator with detected H drifts < 1e-8 over 10⁴ steps.
  - **Risk:** Conserved-quantity discovery is undecidable in general; we cover the polynomial and linear-trig cases.

### 3.4 scirs2-stats

- [x] **Closed-form moments / characteristic functions / pdf-cdf relationships in EML form** — stored per parametric distribution. `[medium]` (completed 2026-05-05 — Wave 70 Track 2a; `cas::moments_catalog::symbolic_moments_catalog(family) -> MomentsCatalog { mean, variance, mgf }`; supports Normal, Exponential, Bernoulli, Geometric (k = #failures convention), Uniform (mgf = None due to t=0 case split); MGF returned as `Option<LoweredOp>` since not every distribution's MGF is real-EML-expressible; 8 tests; 308 LoC)
  - **Why:** `scirs2-stats` distributions today have hand-coded `mean`, `variance`, `cdf`. With EML representations, these are all derivable from a single `pdf` via `cas::integrate` + `cas::solve`. Adding a new distribution requires only its `pdf`.
  - **Design:** New trait `SymbolicDistribution { fn pdf(&self) -> &LoweredOp; fn variable(&self) -> &str; ... }`. Default-method implementations of `mean`, `variance`, `cdf` via `cas::integrate(self.pdf() * x, "x", -∞, ∞)` etc. Existing closed-form implementations stay as overrides for performance.
  - **Files:** `scirs2-stats/src/distributions/symbolic.rs`.
  - **Tests:** Symbolic Normal pdf → derived mean = μ, variance = σ²; symbolic Exponential pdf → derived mean = 1/λ, variance = 1/λ².
  - **Risk:** Symbolic integration can fail on heavy-tail distributions; fallback to numerical quadrature.

- [x] **`scirs2_stats::mle_symbolic` — `fit_mle_symbolic` gradient descent MLE** — backtracking gradient descent for symbolic NLL. `[medium]` (completed 2026-05-04 — forms log-likelihood from user-supplied `LoweredOp` PDF, differentiates via `eml::grad`, runs gradient descent with backtracking Armijo line search; `MleResult { params, log_likelihood, grad_norm }`; `symbolic` feature; 8 tests covering Gaussian MLE, Exponential MLE, convergence, dimension errors)
  - **Files:** `scirs2-stats/src/mle_symbolic.rs`.
  - **Tests:** Normal `(μ, σ)` MLE derived; matches sample mean and sample SD on synthetic data; 8 tests total.

- [x] **Symbolic MLE / method-of-moments — pdf-driven** (completed 2026-05-07; see scirs2-stats/src/mle/derive.rs)
  - **Goal:** `scirs2_stats::mle::derive` takes parametric pdf as LoweredOp, param Var indices, data Var index; returns Estimator callable on data. Closes last open Phase 3 cross-crate item.
  - **Design:** New `scirs2-stats/src/mle/derive.rs` behind `symbolic` feature. Builds log-likelihood ℓ(θ)=Σln(pdf) via balanced add-tree, differentiates w.r.t. each θ via cas::ad::grad, calls cas::solve_system on score equations. closed_form populated on success; falls_back_to_numeric=true on CannotEliminateTranscendental. Estimator::fit: closed-form JIT-eval or Newton fallback.
  - **Files:** `scirs2-stats/src/mle/derive.rs` (new); `scirs2-stats/src/mle/mod.rs` (new; mle_symbolic.rs becomes mle/symbolic.rs); `scirs2-stats/src/lib.rs` (export under symbolic feature).
  - **Prerequisites:** `cas::solve_system` (above); `cas::mle_catalog` (✓); `cas::ad::grad` (✓).
  - **Tests:** ≥8 in `tests/mle_derive_tests.rs`: Normal→μ̂=x̄, σ̂²=sample var; Exponential→λ̂=1/x̄; Bernoulli→p̂=x̄; Geometric; Cauchy→numeric fallback; dim mismatch error; n_samples=0 rejected; canonical invariance.
  - **Cross-crate:** Plan block also in `scirs2-stats/TODO.md`.

- [x] **Symbolic Fisher information matrix** — `cas`-derived expected information for any symbolic distribution. `[small]` (completed 2026-05-05 — Wave 70 Track 2b; `cas::expected_fisher_catalog::expected_fisher_catalog(family, n_samples) -> Vec<Vec<LoweredOp>>`; per-sample I(θ) × n returned as 2-D matrix of canonicalized `LoweredOp` entries; supports Normal (diag(n/σ², 2n/σ²)), Exponential (n/λ²), Bernoulli (n/(p(1-p))), Geometric; rejects Uniform with `UnsupportedFamily` due to boundary-determined support violating regularity conditions; 4 tests; 209 LoC)
  - **Why:** Information geometry, asymptotic confidence intervals, model selection (AIC/BIC) all require the Fisher information matrix. Today users compute it by hand; symbolic derivation eliminates the bookkeeping.
  - **Design:** `scirs2_stats::fisher::information(pdf: &LoweredOp, params: &[&str]) -> Matrix<LoweredOp>`. Computes `E[(∂log p / ∂θ_i)(∂log p / ∂θ_j)]` symbolically via `cas::integrate`.
  - **Files:** `scirs2-stats/src/fisher/symbolic.rs`.
  - **Tests:** Normal Fisher information matrix has classical form `diag(1/σ², 2/σ²)` after derivation.
  - **Risk:** Same as MLE — integration may fail; fallback to Monte Carlo.

### 3.5 scirs2-neural

- [x] **`scirs2_neural::{activations,losses}::symbolic` — `SymbolicActivation` + `SymbolicLoss`** — element-wise symbolic activation and loss functions via `eval_real`. `[medium]` (completed 2026-05-04 — `SymbolicActivation` implements `Activation` + `Layer` traits; forward via `eval_real` per element, backward via symbolic gradient JIT-compiled; `SymbolicLoss` implements `Loss` trait; `symbolic` feature; 10 tests covering forward eval, gradient parity vs finite-difference, composition with stock layers, chain-rule correctness)
  - **Files:** `scirs2-neural/src/activations/symbolic.rs`, `scirs2-neural/src/losses/symbolic.rs`.
  - **Tests:** 10 tests; gradient parity to 1e-5 vs finite-difference.

- [x] **Symbolic-regression-as-prior** — initialize NN weights from a discovered formula. `[medium]` (completed 2026-05-05 — Wave 68 Track 8; `scirs2-neural::symbolic::init_weights_from_formula` uses `scirs2-linalg::lstsq` on a sample grid to project formula outputs into weight space; `symbolic` feature; 4 tests)
  - **Why:** When the user has a candidate functional form, initializing the NN at that form (rather than random) accelerates convergence dramatically. Today nobody does this because it requires symbolic→neural translation.
  - **Design:** `scirs2_neural::init::from_symbolic(formula: &LoweredOp, target_arch: &MlpArch) -> Mlp`. Heuristic: each `LoweredOp::Add` becomes a sum-pool, each `LoweredOp::Mul` becomes a product (via the `log+exp` identity since standard MLPs lack a multiplication unit), each transcendental becomes the matching activation. Initialize weights so the network exactly represents the formula at construction; subsequent training fine-tunes.
  - **Files:** `scirs2-neural/src/init/from_symbolic.rs`.
  - **Tests:** Initialize MLP from `f(x) = sin(x) + 0.1·x`; pre-training loss is identically zero on synthetic data; post-training loss decreases on noisy data.
  - **Risk:** Architecture mismatch — not every formula maps cleanly. Document supported subset.

- [x] **Formula extraction from trained MLP via SR over network outputs** — close the loop: train a network, extract a closed form. `[medium]` (completed 2026-05-05 — Wave 68 Track 8; `scirs2-neural::symbolic::extract_formula_from_mlp` queries model on input grid then calls `regression::discover` on outputs; `symbolic` feature; 3 tests)
  - **Why:** Interpretability. A trained MLP is a black box; running SR over its outputs (treating the trained net as an oracle) yields a candidate closed form.
  - **Design:** `scirs2_neural::interpret::extract_formula(model: &Mlp, sample_grid: ArrayView2<f64>) -> Vec<DiscoveredFormula>`. Internally: query the model on the grid, feed `(grid, outputs)` into `regression::discover`, return Pareto front.
  - **Files:** `scirs2-neural/src/interpret/extract.rs`.
  - **Tests:** Train MLP on `f(x) = sin(x)`; extracted formula is `sin(x)` within tolerance.
  - **Risk:** Sample-grid coverage — extraction quality depends on grid density.

- [x] **Closed-form attention computation for known-structure transformers** — when attention weights have detectable algebraic structure (e.g. position-only sinusoidal), compute the result symbolically. `[research, medium]` (completed 2026-05-05 — Wave 70 Track 5; `scirs2-neural::symbolic::rope_attention::rope_attention_logit(d_head, theta_base) -> RopeAttentionSymbolic`; produces a `LoweredOp` for `RoPE(q,m)·RoPE(k,n) = Σᵢ [(q_{2i}·k_{2i} + q_{2i+1}·k_{2i+1})·cos((m-n)·θᵢ) + (q_{2i+1}·k_{2i} − q_{2i}·k_{2i+1})·sin((m-n)·θᵢ)]` proving the dot-product depends only on the relative position; result canonicalized; `OddDimension`/`DimensionTooLarge`/`InvalidBase` errors; 9 tests verifying d=2,4 structural shape, numerical equivalence to dense RoPE attention, canonicalize idempotency, and relative-position-only dependence; 419 LoC)
  - **Why:** Many positional encodings have closed-form attention patterns; computing them symbolically eliminates the per-step matrix multiply.
  - **Design:** Pattern-detect known structures (sinusoidal-RoPE, ALiBi linear bias) in the attention weights; if matched, replace the dense `softmax(QKᵀ)V` with the closed-form expression in EML; JIT-compile.
  - **Files:** `scirs2-neural/src/attention/closed_form.rs`.
  - **Tests:** Pure-RoPE attention block; symbolic version produces identical outputs to dense computation but in O(seq_len) time vs O(seq_len²).
  - **Risk:** Pattern brittleness — research-grade.

### 3.6 scirs2-linalg

- [x] **`scirs2_linalg::symbolic` — `det_symbolic`, `eigenvalues_symbolic_2x2`, `condition_number_symbolic`** — closed-form symbolic linear algebra for small matrices. `[medium]` (completed 2026-05-04 — `det_symbolic` Leibniz formula for n ≤ 4 returning `LoweredOp`; `eigenvalues_symbolic_2x2` closed-form quadratic formula in EML; `condition_number_symbolic` as `max_eigenvalue / min_eigenvalue` symbolic form, evaluated via `eval_real`; `symbolic` feature; 12 tests verifying determinant correctness, eigenvalue agreement with numerical solver to 1e-10, condition-number bounds)
  - **Files:** `scirs2-linalg/src/symbolic.rs`.
  - **Tests:** 12 tests; eigenvalue agreement to 1e-10 vs numerical solver.

- [x] **Symbolic matrix simplification** — recognize structured matrices at compile time (e.g. `(I + uvᵀ)`). `[medium]` (completed 2026-05-05 — Wave 69 Track 1; `scirs2-linalg::symbolic::recognize` with StructureKind {Scalar,Diagonal,LowRankUpdate,Circulant,General}; Sherman-Morrison inverse for rank-1 updates; 8 tests; 615 LoC)
  - **Why:** Sherman-Morrison, Woodbury, and friends apply only when matrix structure is detectable. Today users hand-detect; with EML representations of matrix entries, the structure can be recognized via canonical-hash matching on the entry expressions.
  - **Design:** `scirs2_linalg::symbolic::recognize(m: &SymbolicMatrix) -> StructureKind`. `StructureKind::{LowRankUpdate(I, u, v), Diagonal, Toeplitz, Circulant, ...}`. Once recognized, dispatch to the matching specialized solver.
  - **Files:** `scirs2-linalg/src/symbolic/recognize.rs`.
  - **Tests:** Construct `I + uvᵀ` symbolically; recognized as `LowRankUpdate`; inverse via Sherman-Morrison matches dense inverse to 1e-12.
  - **Risk:** Recognition is incomplete; document supported structure list.

- [x] **Closed-form matrix exponential for nilpotent / diagonal / 2×2 / 3×3 cases** — Cayley-Hamilton + EML symbolic. `[medium]` (completed 2026-05-05 — Wave 69 Track 2; `expm_symbolic_2x2`, `expm_symbolic_3x3` wrapping `cas::matrix_exp`; diagonal fast path; 26 tests; 538 LoC)
  - **Why:** `expm` for small matrices admits closed forms (Cayley-Hamilton gives a polynomial in the matrix of degree ≤ n−1). Today's `scirs2-linalg::expm` uses Padé universally; closed forms are 5-10× faster for small matrices.
  - **Design:** `scirs2_linalg::expm::closed_form_2x2`, `closed_form_3x3` derive their entries via the symbolic CAS path: build the Cayley-Hamilton polynomial in EML, solve for coefficients via `cas::solve_system`, JIT-compile. Auto-dispatch from `expm` for `n ≤ 3`.
  - **Files:** `scirs2-linalg/src/expm/closed_form.rs`.
  - **Tests:** Closed-form 2×2 matches Padé to 1e-14 on random matrices; 5× faster.
  - **Risk:** Branch-cut handling for matrices with complex eigenvalues — careful.

- [x] **Symbolic spectral decomposition for special matrix families** — circulant, Toeplitz triangular, Vandermonde. `[medium]` (completed 2026-05-05 — Wave 69 Track 3; `eigenvalues_circulant` DFT formula; `eigenpairs_symmetric_2x2`; `structured_eigenvalues` dispatch; 7 tests; 430 LoC)
  - **Why:** These families have known closed-form eigendecompositions (circulant via DFT, etc.). Today's general eigensolvers ignore the structure.
  - **Design:** `scirs2_linalg::eigh::structured` dispatch path: detect structure via the symbolic-recognize hook above; if matched, return the closed-form decomposition (eigenvalues + eigenvectors as `LoweredOp`s).
  - **Files:** `scirs2-linalg/src/eigh/structured.rs`.
  - **Tests:** Random circulant matrix; symbolic eigvals match dense eigh to 1e-12; 10× faster.
  - **Risk:** Same as recognize — incomplete coverage.

---

## Phase 4 — Research / speculative (v0.5.0 / v0.6.0+, clearly labeled as research)

These items require novel research and may not pan out. Each is tagged `[research]` and treated as exploratory.

- [x] **`cas::inverse_symbolic` — Inverse-Symbolic Calculator (lite)** — recover rational and π/e/ln2/√2/γ forms from an `f64` via continued fractions + integer-relation detection. `[research]` (completed 2026-05-05 — Wave 68 Track 2; `recover(x, &RecoverOpts) -> Vec<Candidate>`; Stern–Brocot CF expansion + PSLQ-lite over 7-element constants table; scoring by −log10(residual) − 0.5·tree_size; 13 tests; 579 LoC)

- [x] **`cas::matrix_ops` — small symbolic matrix simplification** — det/trace/cofactor/adjugate/inverse for 2×2/3×3/4×4 `[[LoweredOp; N]; N]` arrays. `[research]` (completed 2026-05-05 — Wave 68 Track 3; cofactor expansion, adjugate transpose, `InverseResult::Singular` when det canonicalizes to zero; 14 tests; 538 LoC)

- [x] **`cas::matrix_exp` — closed-form matrix exponential** — expm for diagonal, nilpotent, 2×2 (Cayley–Hamilton mean-shift via cosh/sinh), and 3×3 (numeric path) matrices. `[research]` (completed 2026-05-05 — Wave 68 Track 4; `expm_2x2` uses `cosh(δ)·I + sinh(δ)/δ·M'`; nilpotent via iterative Taylor truncation; 10 tests; 704 LoC)

- [x] **`cas::spectral_2x2` — symmetric 2×2 spectral decomposition** — closed-form eigenvalues (quadratic formula) + eigenvectors `[b, λ−a]` for real symmetric input. `[research]` (completed 2026-05-05 — Wave 68 Track 5; `eig_symmetric_2x2` returns `SymmetricEig2 { eigenvalues, eigenvectors }`; eigenvector orthogonality proved via `<v1,v2> = b²+(λ₁-a)(λ₂-a) = 0`; 9 tests; 306 LoC)

- [x] **`cas::mle_catalog` — symbolic MLE estimators for catalog distributions** — Normal, Exponential, Bernoulli, Geometric closed-form estimators as `LoweredOp` expressions over sample `Var(i)`. `[research]` (completed 2026-05-05 — Wave 68 Track 6; balanced binary Add-tree for depth-O(log n) sums; `symbolic_mle_catalog(family, n_samples)`; 5 tests; ~250 LoC)

- [x] **`cas::observed_fisher` — observed Fisher information matrix** — `−∂²ℓ/∂θᵢ∂θⱼ` via `eml::hessian`, canonicalized. `[research]` (completed 2026-05-05 — Wave 68 Track 6; `observed_fisher_matrix(log_lik, param_indices)`; 4 tests + 1 doctest; ~120 LoC)

- [x] **`cas::integrate_rational` — Risch-LITE rational integration** — symbolic integration of P(x)/Q(x) with literal coefficients; degree-2 denominators via partial fractions (real distinct, repeated, complex conjugate); polynomial division; `try_integrate(op, var_idx)` + `integrate_polynomial`; `IntegrateRationalError::{DenominatorDegreeTooHigh, SymbolicCoefficientsInDenominator, NumeratorDegreeTooHigh, ZeroDenominator, NotARationalFunction}`; iterative traversal (no recursion); requires `as_polynomial` from `cas::solve` (now `pub(crate)`); 16 tests covering ∫1/x = ln|x|, ∫1/(x²+1) = atan(x), ∫1/(x²-1) partial fractions, repeated roots, polynomial+rational mixed; 860 LoC. `[research]` (completed 2026-05-05 — Wave 70 Track 1)

- [ ] **Neural-guided EML topology search** — train a small transformer (via `scirs2-neural`) on `(data → topology)` pairs to amortize MCTS expansion. `[research]`
  - **Why:** AlphaSymPy, DSO, NeSymReS all use neural guidance for symbolic regression. The unique angle here: EML's depth-2 alphabet means the topology output space is dramatically smaller than tree-structured predictions over a wide alphabet — a small transformer can learn the search policy faster.
  - **Design:** Pre-training: synthesize 10⁵ `(features, targets, topology)` triples by sampling random EML topologies, evaluating, and recording. Train an encoder-decoder transformer to predict topology from `(features, targets)`. At inference: use the predicted topology distribution as the prior in MCTS rollouts.
  - **Files:** `scirs2-symbolic/src/neural_guided/mod.rs`, `scirs2-symbolic/src/neural_guided/transformer.rs`, `scirs2-symbolic/src/neural_guided/training.rs`.
  - **Tests:** Pre-trained model; MCTS-with-prior outperforms uniform-prior MCTS on FSReD.
  - **Risk:** Training data quality, transformer-arch tuning. Mitigation: ship pre-trained checkpoint.

- [ ] **Coq/Lean proof export via OxiLean bridge** — every SMT-certified rewrite emits a Lean 4 tactic; CAS sessions are mechanically replayable as proofs. `[research]`
  - **Why:** **World-first.** Bridges CAS and proof assistants. A `cas::canonicalize` invocation produces both the simplified expression *and* a Lean 4 proof script that a Lean kernel can replay.
  - **Design:** Each certified rewrite rule has an associated Lean tactic template (`rw [...]`, `simp only [...]`). The `RewriteTrace` accumulator records the sequence of rules applied; export converts the trace to a Lean script. Use `OxiLean` (the COOLJAPAN Lean-4-inspired prover) as the verification target.
  - **Files:** `scirs2-symbolic/src/proof_export/lean.rs`, `scirs2-symbolic/src/proof_export/oxilean.rs`.
  - **Tests:** Canonicalize `sin²(x) + cos²(x) → 1`; export Lean script; OxiLean accepts the proof.
  - **Risk:** Lean 4 syntax churn; OxiLean is itself early-stage. Coordinate via `~/work/oxilean`.

- [ ] **Differentially-private symbolic regression** — output a discovered formula whose coefficients satisfy (ε, δ)-DP w.r.t. the input data. `[research]`
  - **Why:** Symbolic regression on private datasets (medical, financial) needs formal privacy guarantees. The discrete topology-selection step is the harder part — coefficient noising via Gaussian/Laplace is straightforward; but the topology choice itself can leak.
  - **Design:** Two-stage: (a) Topology selection via the *exponential mechanism* (Dwork-McSherry-Talwar): sample topology ∝ `exp(−ε·loss/sensitivity)`; (b) Adam fitting on the chosen topology with DP-SGD (clip gradients, add Gaussian noise). Sensitivity analysis: bound how much the loss can change with one record changed.
  - **Files:** `scirs2-symbolic/src/private/dp_regression.rs`.
  - **Tests:** Privacy budget tracking; output formula's coefficients are within DP bounds; recovery quality on synthetic data is strictly worse than non-private baseline (sanity).
  - **Risk:** Privacy proofs are easy to get wrong; submit for external review before publishing.

- [ ] **Quantum-symbolic algebra (QuantRS2 partnership)** — symbolic Pauli-string algebra, symbolic Hamiltonian simplification, symbolic VQE ansatz manipulation. `[research]`
  - **Why:** Quantum software stacks (Qiskit, Cirq) handle symbolic Hamiltonians as a separate code path from classical CAS. With EML as a unified IR, we can express both classical and Pauli-string algebra in the same simplification framework.
  - **Design:** Extend the EML grammar with a Pauli-string variant `EmlNode::Pauli { qubits, ops }`. New canonical-form rules: Pauli commutation (`X·Y = iZ`, etc.), commutator/anticommutator simplification, BCH expansion truncation. VQE ansatz simplification: detect equivalent gate sequences, collapse them.
  - **Files:** `scirs2-symbolic/src/quantum/pauli.rs`, `scirs2-symbolic/src/quantum/hamiltonian.rs`. Cross-crate: `~/work/quantrs/quantrs2-core/src/symbolic/eml.rs`.
  - **Tests:** Pauli `X·X = I`; commutator `[X, Y] = 2iZ`; H₂ Hamiltonian simplification reduces operator count.
  - **Risk:** Grammar extension may break canonicalization properties — careful.

- [x] **Symbolic differential geometry (Cadabra2-class)** (completed 2026-05-07)
  - **Goal:** Index-aware tensor extension on LoweredOp. Christoffel, Ricci, Einstein end-to-end. Verification: Schwarzschild metric → Christoffels → Ricci tensor → vacuum Einstein Rᵢⱼ=0 numerically at sample points.
  - **Design:** New module `scirs2-symbolic/src/diffgeom/`. Files: tensor.rs (Tensor{rank_up,rank_down,dim,components:ArrayD<LoweredOp>}); metric.rs (Metric{g,g_inv,coords}, inverse via matrix_ops; extend with inverse_4x4+adjugate_4x4); contraction.rs (contract_indices balanced Add-tree); christoffel.rs (Γᵏᵢⱼ = ½gᵏˡ(∂ᵢgⱼˡ+∂ⱼgᵢˡ−∂ˡgᵢⱼ) via eml::grad); ricci.rs (Rᵢⱼ = ∂ₖΓᵏᵢⱼ−∂ⱼΓᵏᵢₖ+ΓΓ−ΓΓ); einstein.rs (Gᵢⱼ=Rᵢⱼ−½gᵢⱼR); covariant_derivative.rs (Γ-corrections for up/down indices, iterative).
  - **Files:** `scirs2-symbolic/src/diffgeom/{mod,tensor,metric,contraction,christoffel,ricci,einstein,covariant_derivative}.rs`; `scirs2-symbolic/src/lib.rs` (re-export diffgeom); `scirs2-symbolic/src/cas/matrix_ops.rs` (add adjugate_4x4, inverse_4x4).
  - **Prerequisites:** `eml::grad` (✓); `cas::canonicalize` (✓); `cas::matrix_ops` (✓ + 4×4 extension).
  - **Tests:** ≥12 in `tests/diffgeom_tests.rs`: flat Euclidean Christoffels=0; 2D polar Γʳθθ=-r; sphere S² Ricci R=2; 4×4 inverse; contract g_up·g_down=identity; Schwarzschild Rᵢⱼ<1e-10 numerically at (rs=2,r=10,θ=π/2) and (r=5,θ=π/4); vacuum Gᵢⱼ<1e-10; metric compatibility; scalar curvature R=0.
  - **Risk:** Structural symbolic-zero on Schwarzschild Ricci not achievable without trig-identity pass. Document as "numerical-at-sample-points to 1e-10"; full structural zero deferred.

- [ ] **Inverse-Symbolic Calculator at scale** — given a numerical value to N digits, search a multi-billion-entry EML-canonical-hash database for closed forms (PSLQ + canonical-form matching). `[research]`
  - **Why:** OEIS-for-real-numbers. A user types `2.6651441426902251886502972498...` and gets back `Catalan's constant`. Unique angle: pre-compute the canonical-hash → numerical-value table for all EML topologies up to depth 6; lookup is O(log n) on the value, O(1) on the hash. Combined with PSLQ for integer-relation detection.
  - **Design:** Offline: enumerate every depth-≤-6 EML topology, evaluate each at high precision (use `rug` MPFR via the special-functions feature), record `(value, canonical_hash)`. Online: given a query value, run PSLQ to find candidate integer relations; lookup each candidate's canonical hash in the precomputed table.
  - **Files:** `scirs2-symbolic/src/inverse_calc/db.rs`, `scirs2-symbolic/src/inverse_calc/pslq.rs`, `scirs2-symbolic/data/inverse_db.oc` (large file, gitignored; download via build script).
  - **Tests:** Recover `Catalan = G`, `Apéry = ζ(3)`, `Khinchin's constant`, `Glaisher-Kinkelin` from their numerical values to 50 digits.
  - **Risk:** Database size (potentially > 1 GB) — host externally; build script downloads on demand.

- [ ] **EML-program-synthesis from natural language** — couple with a small LM in `scirs2-neural` that emits EML topologies from problem descriptions; the SMT layer rejects ill-formed outputs. `[research]`
  - **Why:** "Find an equation of motion for a damped pendulum" → EML topology candidate → SMT-verified solution. Restores soundness lost by the LM via the SMT post-check.
  - **Design:** Fine-tune a small (1B param) language model on `(description, eml_topology)` pairs synthesized from textbook problems. At inference, sample N candidates; for each, run `cas::canonicalize` and SMT-verify against any provided constraints; return the first verified candidate.
  - **Files:** `scirs2-symbolic/src/lm_synth/mod.rs`, `scirs2-symbolic/src/lm_synth/training.rs`.
  - **Tests:** Hand-curated test set of 50 physics problems; ≥ 30% correct topology recovery.
  - **Risk:** LM hallucination; SMT post-check is the safety net but slow.

- [x] **Reversible CAS — every simplification step is undoable via a recorded EML proof trace** — useful for didactic / interpretability tooling. `[research]` (completed 2026-05-05 — Wave 69 Track 4; `RewriteStep`, `RewriteTrace`, `canonicalize_traced`; batch-pass tracing; `is_fully_reversible()` + `reverse()`; 8 tests; 413 LoC)
  - **Why:** Educational tools (Wolfram Alpha "Step-by-step solution") need bidirectional traversal of the simplification path. With EML certified-rewrite traces, every rewrite has a recorded inverse, enabling true reversibility.
  - **Design:** Each `CertifiedRule` exposes `inverse(&self) -> Option<&dyn CertifiedRule>` (returning `None` for irreversible steps like constant folding). `RewriteTrace::reverse(trace) -> Option<LoweredOp>` reconstructs the original expression from the simplified form. Pair with a notebook UI that lets users step forward/backward through a simplification.
  - **Files:** `scirs2-symbolic/src/reversible/trace.rs`, `scirs2-symbolic/src/reversible/inverse.rs`.
  - **Tests:** `reverse(canonicalize(x))` recovers the original `x` for reversible rule sets.
  - **Risk:** Not every simplification has a clean inverse (e.g. `0 * x → 0` loses `x`). Document.

---

## Constraints & policies (must appear at the end as a checklist for future agents)

Every item in this TODO must be implemented under the following COOLJAPAN policies. Violations block release.

- [ ] **No-unwrap policy.** Production code uses `Result` everywhere. `unwrap()` is permitted only in `tests/`, `examples/`, and `benches/` (and even there, prefer `expect("explicit reason")`).
- [ ] **No-warnings policy.** `cargo clippy --workspace --all-features -- -D warnings` exits 0; `cargo doc --no-deps -p scirs2-symbolic` with `RUSTDOCFLAGS="-D warnings"` exits 0.
- [ ] **Pure Rust default features.** No C/Fortran in the default feature set; any C/Fortran-bearing dependency is gated behind an explicit `unsafe-c` or `unsafe-fortran` feature with a documented fallback.
- [ ] **OxiBLAS / OxiFFT / OxiARC / oxicode / OxiZ.** Never `bincode`, `flate2`, `zstd`, `bzip2`, `lz4`, `tar`, `snap`, `brotli`, `miniz_oxide`, `zip`, `rustfft`, `Z3`, `openblas`. Any compression/decompression in this crate routes through `oxiarc-*`. Any FFT routes through OxiFFT. Any SMT routes through OxiZ.
- [ ] **Workspace policy.** Every dep declared via `*.workspace = true` in `Cargo.toml`; no per-crate version pins. Workspace-level `[workspace.dependencies]` is the source of truth.
- [ ] **Refactoring policy.** No source file > 2000 lines. Use `splitrs` (installed at `~/work/splitrs`) when a file approaches the limit. Run `rslines 50 src/` to find candidates.
- [ ] **Naming convention.** `snake_case` for variables and functions; `CamelCase` for types and traits; `SCREAMING_SNAKE_CASE` for constants. No exceptions.
- [ ] **f64 default precision.** No `f32` in scope unless a downstream user explicitly requests it (out-of-scope for this TODO).
- [ ] **Branch-name-driven version bumps.** Cargo.toml version follows the active git branch (e.g. `0.5.0`). Never publish without explicit user permission. `cargo publish` is forbidden in automation; use `--dry-run` only.
- [ ] **No git commits without permission.** Every commit must be explicitly requested by the user. CI/local testing uses uncommitted working tree.
- [ ] **No new GitHub Actions workflows** beyond `pypi-publish.yml` and `npm-publish.yml`. Existing CI is the only allowed CI surface.
- [ ] **Use `rtk` for development commands.** All `cargo`, `git`, `rustup` invocations route through the `rtk` proxy for token analytics (handled by hook).
- [ ] **Latest crates always.** `oxieml` (dev-dep), `ndarray`, `rayon`, etc. pin to the latest crates.io release at the time of each phase. Re-check before starting a new phase.
- [ ] **Parity with oxieml.** `tests/oxieml_parity.rs` MUST pass on every PR. Numerical tolerance 1e-12. Intentional divergence allowed when documented in an ADR; bug-for-bug compatibility is NOT a goal.

---

## Non-goals

The following items are explicitly out of scope for `scirs2-symbolic`. Future agents tempted to expand into these areas should add a separate TODO file or open an upstream issue first.

- **Replacing SymPy as a *general-purpose* CAS for all domains.** Focus is the EML-expressible math domain (still covers ~95% of physics, engineering, and applied math). Number theory beyond elementary diophantine, abstract algebra over arbitrary rings, computational group theory, polynomial-quotient-ring computations, free-algebra word problems — out of scope.
- **Symbolic algebra over arbitrary algebraic data types beyond ℝ / ℂ / unit-bearing reals.** No general rings, no polynomial quotient rings, no algebraic-extension towers beyond what the native complex / units machinery provides — until a later major version with explicit user demand.
- **Replicating Mathematica's pattern-matching DSL surface (`/. f[x_] :> g[x]`).** We provide e-graph rewrites instead; users wanting Mathematica-style pattern dispatch should preprocess their patterns into `EmlPattern` form via the proc-macro (Phase 2 item).
- **C / Fortran in default features.** Pure Rust Policy. C/Fortran-bearing deps allowed only behind explicit feature gates.
- **`scirs2-symbolic` as a dependency of `scirs2-core`.** Cycle prevention; enforced by CI lint.
- **Symbolic profiler / symbolic privacy outside Phase 4 research.** These are explicitly research items; no production commitment until Phase 4 graduates.
- **GPU acceleration of symbolic operations.** EML rewriting is inherently irregular and pointer-chasing; GPU offload offers no speedup. Numerical evaluation can be GPU-accelerated via the existing `scirs2-core` GPU pipeline when JIT-compiled artifacts are dispatched.
- **Real-time interactive REPL.** Out of scope; the WASM playground (Phase 2) is the closest approximation.
- **Custom proof assistant (beyond OxiLean export).** OxiLean is the COOLJAPAN proof target; no separate kernel.
- **Symbolic computation over `f32`.** f64 is the floor; lower-precision symbolic work is undefined.
- **Compatibility shim for SymPy `Expr` API.** Users wanting SymPy compatibility should use SymPy. The Phase 1 facade is ndarray-first, not SymPy-first.
- **`scirs2-symbolic` depending on `oxieml` at runtime.** `oxieml` is a `[dev-dependencies]` entry only, used by the parity harness. Any production-code import of `oxieml::*` is a CI failure (enforced by the cycle-prevention script).
- **Bug-for-bug compatibility with oxieml.** Where oxieml has a known bug or sub-optimal API, `scirs2-symbolic` is allowed to diverge; the divergence is documented in an ADR. Parity is a numerical correctness check, not a behavioural mirror.

---

*Last updated: 2026-07-15 (post-0.6.1 e-graph extraction correctness fix; prior update 2026-05-15, Waves 59–70 + plan blocks 2026-05-06). Branch: `0.6.1`. Maintainer: COOLJAPAN OU (Team Kitasan). Architecture: clean-room, SciRS2-native EML implementation. Substrate guidance: oxieml v0.1.1 (pinned in `[dev-dependencies]` for parity testing only); paper reference arXiv:2603.21852 v2 (2026-04-04). Phase 2: 15/15. Phase 3: 15/12+. Phase 4: 9/N.*

*Note: as of 2026-05-03, oxieml's `Cargo.toml` has hardcoded absolute paths for `tensorlogic-ir`, `scirs2-core`, `oxicode` — this is an upstream oxieml issue, separate from this crate.*

---

## Wave 74 — CAS depth (2026-05-08)

- [x] **Trigonometric identity closure for Schwarzschild structural zero** (completed 2026-05-08, Wave 74)
  - **Goal:** Lift `canonicalize` so that `sin²(x) + cos²(x)` returns `Canonical::one()`, `sin(2x) − 2·sin(x)·cos(x)` returns `Canonical::zero()`, and the Ricci tensor of the Schwarzschild metric (Wave 72 diffgeom) achieves **structural zero** rather than the current 1e-10 numerical zero.
  - **Design:** Extended `cas::identity_db` with 8 new identities (Rules 11a/11b/12a/12b for `1−2sin²/2cos²−1` double-angle inverses with commuted-Mul variants; Rules 13/14 sin-sum collapsing recognizers; Rules 17a/17b product-to-sum recognizers). Added `outer_first` flag to `Identity` and a top-down rule pre-pass to `apply_db_once`. Extended `cas::canonical_rules` with `rule_sub` distributing `a − (b − c) → (a + c) − b`. Added oscillation detection (hash-trail) to `cas::canonicalize::canonicalize`.
  - **Deviation:** The Schwarzschild Ricci centerpiece is verified numerically at **four** evaluation points (vs Wave 72's two). Structural zero would require rational-function GCD/cancellation outside the trig-closure scope; the residue is rational, not trig. Documented in test `schwarzschild_ricci_structural_zero`.
  - **Files:** `scirs2-symbolic/src/cas/identity_db.rs` (1648 lines), `scirs2-symbolic/src/cas/canonical_rules.rs`, `scirs2-symbolic/src/cas/canonicalize.rs`.
  - **Tests:** 12 tests in `scirs2-symbolic/tests/cas_trig_identity_tests.rs`, all pass. Includes Pythagorean, three-form double-angle, sum-difference, product-to-sum direction, complex-argument Pythagorean, exp-log identity, Schwarzschild Ricci numerical at 4 points, oscillation detector, MAX_ITER safety, constant args, combined trig-exp (real Euler-proxy).

- [x] **Degree-4 polynomial solver via depressed-quartic Ferrari method** (completed 2026-05-08, Wave 74)
  - **Goal:** Lift `solve_polynomial` from "linear / quadratic" to "linear / quadratic / cubic / quartic" via Cardano + Ferrari closed-form. Unblocks closed-form MLE for Weibull, Pareto, etc.
  - **Design:** New module `cas::cardano_ferrari` (484 lines) with `solve_cubic` (depressed cubic + trigonometric/Cardano dispatch on Δ sign) and `solve_quartic` (Ferrari resolvent cubic + biquadratic edge case + sign-aware √(2z−p) factorisation). `solve_polynomial` in `cas::solve` now dispatches to these for degrees 3 and 4 via a coefficient folder.
  - **Files:** `scirs2-symbolic/src/cas/cardano_ferrari.rs`, `scirs2-symbolic/src/cas/solve.rs`, `scirs2-symbolic/src/cas/mod.rs`.
  - **Tests:** 11 in `scirs2-symbolic/tests/cas_solve_quartic_tests.rs` plus 7 in `cardano_ferrari` lib tests, all pass. Includes `quartic_real_distinct_roots`, `quartic_complex_conjugate_pair`, `quartic_double_root`, `quartic_biquadratic_special_case`, `quartic_irreducible_over_rationals`, `quartic_high_precision_recovery`, `weibull_mle_quartic_system_recovers_closed_form`, `pareto_mle_quartic_returns_real_root_only`, `quartic_resolvent_cubic_zero_root_handled`, `quartic_too_high_degree_raises_clean_error` (degree 5 → `HighDegreePoly`), `buchberger_quartic_via_ferrari` multivariate.

- [x] **Hermite reduction for higher-degree rational denominators** (completed 2026-05-08, Wave 74)
  - **Goal:** Lift `integrate_rational` (Wave 70 Risch-LITE) from `deg(Q) ≤ 2` to `deg(Q) ≤ 4` via Yun squarefree + Cardano/Ferrari root extraction + cover-up partial fractions + Hermite reduction step.
  - **Design:** New module `cas::hermite_reduction` (~580 lines) implementing `Poly`, `poly_add/sub/mul/scale/derivative/divmod/gcd`, `yun_squarefree`, `real_roots_low_degree` (dispatches to Cardano/Ferrari), `partial_fractions_simple` (Heaviside cover-up), `hermite_reduce_step`, `poly_extended_gcd`. `integrate_rational` adds an `integrate_proper_fraction_high_degree` path with: (a) all-real-root cover-up; (b) factor split into linear-roots-plus-residual-quadratic; (c) repeated linear factor handling via `(x−a)^k` shift expansion; (d) repeated quadratic factor handling via Hermite step iteration.
  - **Files:** `scirs2-symbolic/src/cas/hermite_reduction.rs`, `scirs2-symbolic/src/cas/integrate_rational.rs`, `scirs2-symbolic/src/cas/mod.rs`.
  - **Tests:** 9 in `scirs2-symbolic/tests/cas_integrate_rational_hermite_tests.rs`, all pass. Includes simple repeated factor, cubic distinct real roots, cubic with complex conjugate pair, quartic via Ferrari, repeated quadratic, Yun squarefree decomposition of `(x−1)²(x²+1)³`, polynomial quotient first then remainder, degree-5 unhandled, Risch-LITE chain with canonicalize.

- [x] **Third- and fourth-order symbolic derivatives** (completed 2026-05-08, Wave 74)
  - **Goal:** Extend `cas::ad` with `higher_order_grad`, `third_derivative`, `fourth_derivative`, `taylor_higher_order`. Riemann curvature (diffgeom Phase 4) needs third-order; Newton-Krylov / trust-region need fourth-order.
  - **Design:** Module-scope `RwLock<HashMap<(u128, usize), LoweredOp>>` cache keyed on canonical hash (Wave 53 stable u128). `cached_grad` uses read-then-fallback-to-compute-then-write pattern. `higher_order_grad` iterates `cached_grad` order times. `third_derivative`/`fourth_derivative` chain three/four `cached_grad` calls. `taylor_higher_order` uses `substitute_var_with_const` (iterative work-stack) to evaluate at `x₀` then scales by `1/k!`.
  - **Files:** `scirs2-symbolic/src/cas/ad.rs` (1122 lines), `scirs2-symbolic/src/cas/mod.rs`.
  - **Tests:** 8 in `scirs2-symbolic/tests/cas_ad_higher_order_tests.rs`, all pass. Includes `third_derivative_cubic_polynomial`, `fourth_derivative_quartic`, `higher_order_mixed_partial_quadratic`, `higher_order_grad_iteration_matches_chained`, `taylor_higher_order_around_zero` (sin(x) Taylor), `riemann_tensor_3d_flat_space_returns_zero`, `higher_order_with_constants`, `cache_reuses_partial_derivatives`.

---

## Proposed follow-ups

These items remain `- [ ]` after this wave. Each has an explicit reason and target wave.

- ~~**`phase1.numa_engine`**~~ — completed 2026-05-07: `par_map_chunks` now exported from scirs2-core root; `predict_parallel` in `regression/discover.rs` uses it directly.
- **`phase4.neural_guided_search`** — Encoder-decoder transformer pre-training on 10⁵ synthesized triples + checkpoint hosting. Training infrastructure substantial; dedicated wave needed. Defer.
- **`phase4.coq_lean_export`** — Lean-4 proof script export. Blocked on OxiLean maturity + cross-org coordination via ~/work/oxilean. Defer.
- **`phase4.dp_regression`** — Differentially-private SR. Privacy proof requires external review before implementation. Do not freelance.
- **`phase4.quantum_pauli`** — Pauli-string algebra. Requires new EML grammar variant that invalidates current canonicalize hash invariants. Re-evaluate after diffgeom (#5) validates index-extension pattern.
- **`phase4.inverse_symbolic_at_scale`** — Multi-billion-entry hash DB. >1 GB data file with hosting/build-script questions. `cas::inverse_symbolic` (lite) covers small-table case. Defer; gather demand first.
- **`phase4.lm_synth`** — Natural-language → EML topology via LM. 1B-param fine-tuning + 50 hand-curated physics problems is a research wave. Defer.

### Wave 74 deferred (defer to v0.4.5+)

- **Phase 4 research items** (6 entries: neural-guided topology search, Coq/Lean export via OxiLean, DP-SR, quantum-Pauli algebra, inverse-symbolic at scale, NL-LM topology synthesis). Each is a research wave on its own; explicitly deferred per Wave 73 plan.
- **scirs2-fft AVX-512 + NEON / SVE butterflies** (TODO L128-141). Pure performance with no correctness gap; defer to a dedicated SIMD wave.
- **scirs2-interpolate LLE eigendecomposition** (high_dimensional.rs:475). Currently random-projection fallback; full LLE needs sparse-eigendecomp Lanczos. Defer to Wave 75.
- **scirs2-cluster subspace performance refactor** (32 workspace timeouts on kPCA / LLE / diffusion-maps). Needs Nyström low-rank approximation + data-point sampling fallback. Defer to Wave 75.
- **scirs2-signal elliptic filter Jacobi functions** (filter/iir.rs:738-744). Currently Chebyshev approximation. Needs `sn`/`cn`/`dn` from scirs2-special; defer until those are stress-tested.
- **scirs2-linalg distributed::mpi::collective::distributed_matmul SUMMA stub** (collective.rs:291). Wave 1-37 already implemented SUMMA in `distributed/algorithms/gemm.rs` — collective-level stub may be intentional layering. Verify before scheduling.
- **scirs2-autograd source-to-source AD** (v0.4.0 Roadmap line). Requires proc-macro framework; defer.
- **scirs2-symbolic IGA multi-patch coupling + trimmed NURBS** (existing Known Issue per Wave 73 plan).

---

## Correctness hardening — e-graph extraction operand-swap bug (2026-07-15)

- [x] **Fixed a real soundness bug in e-graph DP term-extraction** (found and fixed 2026-07-15)
  - **Bug:** `reconstruct` in `scirs2-symbolic/src/cas/e_graph/extract.rs` popped a binary node's
    two already-reconstructed child results off the `results` stack in the wrong order. Children
    are pushed onto the work stack via `children.iter().rev()` (right first, then left), so the
    LEFT child is processed first and its result lands on `results` *before* the right child's —
    meaning the right child's result is on top. The pre-fix code popped `left` before `right` (a
    comment claiming "left was pushed last -> left pops first" was itself backwards), silently
    swapping the operands of every binary node during extraction.
  - **Impact:** Invisible for commutative operators (`Add`, `Mul`), but a genuine correctness
    violation for non-commutative ones (`Pow`, `Sub`, `Div`): `Pow(sin(x), 2)` (i.e. `sin²(x)`)
    could be reconstructed as `Pow(2, sin(x))` (i.e. `2^sin(x)`), which is a different function.
    This means any CAS simplification pipeline that routed a non-commutative binary expression
    through e-graph extraction (`canonicalize_egraph` and friends) could silently return a
    mathematically wrong result rather than erroring — the worst class of CAS bug. This is why the
    module-level "sound by construction" language elsewhere in this file refers specifically to the
    SMT-certified-rewrite-rule *registration* mechanism (`cas::certified_rewrite`), not to unrelated
    engine-internal code such as term extraction; the two are different subsystems with different
    correctness arguments, and this bug lived in the latter.
  - **Fix:** Swapped the pop order — pop `right` first, then `left` — matching the (already
    correct) pop order used in `pattern::instantiate`. Added an extensive code comment at the fix
    site tracing the push/pop order explicitly so the invariant cannot silently regress again.
  - **Verification:** Fixed and verified with 120 consecutive passing `cargo nextest` runs of the
    `scirs2-symbolic` test suite (catching any residual iteration-order flakiness), plus a new
    permanent regression test `test_extract_preserves_noncommutative_operand_order` in
    `cas/e_graph/extract.rs` covering `Pow`, `Sub`, `Div`, and a nested non-commutative case at
    multiple tree depths simultaneously.
  - **Secondary hardening:** `cas/e_graph/mod.rs::test_saturation_with_identity_db` had a latent
    test-quality bug of its own — the documented "numeric fallback, defense-in-depth" check was
    written as a second unconditional `assert!` placed *after* the primary structural-hash
    `assert_eq!`, so it could never actually execute as a fallback (the first assertion already
    panics on mismatch before control reaches the second). Restructured so the numeric check is
    only — but always — reached when the structural check misses, matching the documented intent.
  - **Files:** `scirs2-symbolic/src/cas/e_graph/extract.rs`, `scirs2-symbolic/src/cas/e_graph/mod.rs`.
  - **Takeaway for future agents:** treat "N tests pass" and "the engine is sound" as separate
    claims. This bug shipped with passing tests for a long time because no existing test exercised
    extraction of a non-commutative binary node through a multi-node e-class. When adding new
    e-graph rewrite rules or extraction paths, add an explicit operand-order assertion, not just a
    numeric-equality assertion — numeric checks on `Pow`/`Sub`/`Div` test cases can coincidentally
    pass for symmetric inputs even with swapped operands.
