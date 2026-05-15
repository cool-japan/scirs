# CAS Tutorial: From Data to Deployed Formula

This tutorial walks through the full pipeline of `scirs2-symbolic`: discovering
a closed-form formula from data, reducing it to canonical form, computing exact
symbolic gradients, compiling it to native machine code with Cranelift JIT, and
serialising it for deployment.

Every code block uses the **actual public API** as it exists in v0.4.4. Feature
requirements are noted inline. Snippets are complete and can be copy-pasted into
a `main.rs` with the corresponding `Cargo.toml` additions shown at the end of
each section.

---

## 1. Symbolic Regression — Discovering a Formula

Symbolic regression searches a grammar of operators for a closed-form expression
that fits your data. The entry point is `discover`, which accepts an `ndarray`
feature matrix and a target vector.

```rust
// No extra feature flags required.
use ndarray::{Array1, Array2};
use scirs2_symbolic::{
    discover, DiscoveredFormula, SrConfig, BuildingBlock,
};
use scirs2_symbolic::eml::to_latex;

fn main() {
    // ── 1. Build toy dataset: y = x₀² + 2·x₀ ─────────────────────────
    let n: usize = 40;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1 - 2.0).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| x * x + 2.0 * x).collect();

    // Array2 shape (n_samples, n_features); Array1 shape (n_samples,).
    let features = Array2::from_shape_vec((n, 1), xs)
        .expect("feature array shape");
    let targets = Array1::from_vec(ys);

    // ── 2. Configure the search ────────────────────────────────────────
    let config = SrConfig::default()
        .with_max_iter(60)
        .with_beam_width(48)
        .with_top_n(3)
        .with_tolerance(1e-6)
        .with_building_blocks(vec![
            BuildingBlock::Arithmetic,
            BuildingBlock::Pow,
        ]);

    // ── 3. Run discovery ───────────────────────────────────────────────
    let results: Vec<DiscoveredFormula> =
        discover(features.view(), targets.view(), &config);

    if results.is_empty() {
        println!("No formula found.");
        return;
    }

    // ── 4. Inspect the top result ──────────────────────────────────────
    let best: &DiscoveredFormula = &results[0];

    // `best.op` is the LoweredOp representation; Display renders infix notation.
    println!("Best formula:  {}", best.op);
    println!("  MSE          = {:.2e}", best.fitness.mse);
    println!("  R²           = {:.6}", best.fitness.r2);
    println!("  Node count   = {}", best.node_count);
    println!("  LaTeX        = {}", to_latex(&best.op));

    println!("\nTop {} formulas:", results.len());
    for (rank, f) in results.iter().enumerate() {
        println!("  [{}] {} (mse={:.2e})", rank + 1, f.op, f.fitness.mse);
    }
}
```

`Cargo.toml` additions:

```toml
[dependencies]
scirs2-symbolic = "0.4.4"
ndarray = "0.16"
```

**Key types:**

| Symbol | Path | Notes |
|--------|------|-------|
| `SrConfig` | `scirs2_symbolic::SrConfig` | Builder pattern; `Default` is reasonable |
| `BuildingBlock` | `scirs2_symbolic::BuildingBlock` | Enum of operator families |
| `discover` | `scirs2_symbolic::discover` | Returns `Vec<DiscoveredFormula>` sorted by fitness |
| `DiscoveredFormula` | `scirs2_symbolic::DiscoveredFormula` | Fields: `.op`, `.fitness`, `.node_count`, `.n_vars` |

---

## 2. Canonical Form — Algebraic Equivalence

`cas::canonicalize` reduces a `LoweredOp` to a hash-unique canonical form.
Two expressions with the same `Canonical::hash()` are algebraically equal on
the decidable subset (polynomial subring + analytic log/exp/power identities).

```rust
// No extra feature flags required.
use scirs2_symbolic::LoweredOp;
use scirs2_symbolic::cas::{canonicalize, Canonical};

fn main() {
    // ── Build x + y and y + x as LoweredOp trees ──────────────────────
    // Var(0) = x,  Var(1) = y
    let x_plus_y = LoweredOp::Add(
        Box::new(LoweredOp::Var(0)),
        Box::new(LoweredOp::Var(1)),
    );
    let y_plus_x = LoweredOp::Add(
        Box::new(LoweredOp::Var(1)),
        Box::new(LoweredOp::Var(0)),
    );

    let canon_xy: Canonical = canonicalize(&x_plus_y);
    let canon_yx: Canonical = canonicalize(&y_plus_x);

    println!("x+y canonical hash: {:032x}", canon_xy.hash());
    println!("y+x canonical hash: {:032x}", canon_yx.hash());
    println!(
        "Hash equal (=> algebraically equal): {}",
        canon_xy.hash() == canon_yx.hash()
    );

    // ── Log/exp cancellation: ln(exp(x)) → x ─────────────────────────
    let ln_exp_x = LoweredOp::Ln(Box::new(LoweredOp::Exp(Box::new(
        LoweredOp::Var(0),
    ))));
    let just_x = LoweredOp::Var(0);

    let c_ln_exp = canonicalize(&ln_exp_x);
    let c_x      = canonicalize(&just_x);

    println!(
        "ln(exp(x)) == x by canonical hash: {}",
        c_ln_exp.hash() == c_x.hash()
    );

    // ── Access the underlying LoweredOp ──────────────────────────────
    // `into_op()` moves out; `op()` borrows.
    let reduced: LoweredOp = c_ln_exp.into_op();
    println!("Reduced form: {}", reduced);

    // ── Canonicalize is idempotent ────────────────────────────────────
    let c2 = canonicalize(&canonicalize(&x_plus_y).into_op());
    let c1 = canonicalize(&x_plus_y);
    assert_eq!(c1.hash(), c2.hash(), "canonicalize must be idempotent");
    println!("Idempotence check passed.");
}
```

**Important scope note:**

`scirs2_symbolic::Canonical` (re-exported from `eml::canonical`) is the
*EML constructor namespace* (static methods for building canonical EML tree
nodes like `Canonical::sin`, `Canonical::cos`).

`scirs2_symbolic::cas::Canonical` (from `cas::canonicalize`) is the *canonical
form newtype* with a `.hash()` method for algebraic equality testing.

They are different types — use the full path `cas::Canonical` when you need the
canonical-form newtype.

---

## 3. Symbolic Differentiation — Exact Gradients

`cas::ad` provides symbolic differentiation with automatic canonicalization of
each derivative, plus a `GradGraph` for computing value + full gradient in one
CSE pass.

```rust
// No extra feature flags required.
use scirs2_symbolic::LoweredOp;
use scirs2_symbolic::cas::ad::{grad_canonical, GradGraph};
use scirs2_symbolic::eml::{eval_real, EvalCtx};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Simple scalar function: f(x) = x³ ────────────────────────────
    let x_cubed = LoweredOp::Pow(
        Box::new(LoweredOp::Var(0)),
        Box::new(LoweredOp::Const(3.0)),
    );

    // Symbolic gradient df/dx (canonicalized automatically).
    let df_dx: LoweredOp = grad_canonical(&x_cubed, 0);
    println!("f(x)   = {}", x_cubed);
    println!("f'(x)  = {}", df_dx);

    // Evaluate the gradient at x = 2.0: expected 3·x² = 12.
    let ctx = EvalCtx::new(&[2.0_f64]);
    let grad_val = eval_real(&df_dx, &ctx)?;
    println!("f'(2)  = {} (expected 12.0)", grad_val);
    assert!((grad_val - 12.0).abs() < 1e-10, "gradient mismatch");

    // ── Vector function with GradGraph: f(x,y) = x·sin(y) ─────────────
    let x_sin_y = LoweredOp::Mul(
        Box::new(LoweredOp::Var(0)),
        Box::new(LoweredOp::Sin(Box::new(LoweredOp::Var(1)))),
    );

    // GradGraph::new(op, n_vars) — n_vars is 2 because we have x and y.
    let gg = GradGraph::new(&x_sin_y, 2);
    println!("\nf(x,y) = {}", x_sin_y);
    println!("df/dx  = {}", gg.grad_op(0).expect("grad 0"));
    println!("df/dy  = {}", gg.grad_op(1).expect("grad 1"));

    // Evaluate value + gradient in one CSE pass at point (1.0, π/2).
    let point = [1.0_f64, std::f64::consts::FRAC_PI_2];
    let (val, grad) = gg.eval_with_grad(&point)?;

    // f(1, π/2) = 1·sin(π/2) = 1.0
    println!(
        "\nf(1, π/2)        = {:.8} (expected 1.0)", val
    );
    // df/dx at (1, π/2) = sin(π/2) = 1.0
    println!(
        "df/dx at (1,π/2) = {:.8} (expected 1.0)", grad[0]
    );
    // df/dy at (1, π/2) = x·cos(y) = 1·cos(π/2) ≈ 0.0
    println!(
        "df/dy at (1,π/2) = {:.8} (expected ≈ 0.0)", grad[1]
    );

    assert!((val - 1.0).abs() < 1e-10);
    assert!((grad[0] - 1.0).abs() < 1e-10);
    assert!(grad[1].abs() < 1e-8);

    Ok(())
}
```

**Key API summary:**

| Function | Signature | Returns |
|----------|-----------|---------|
| `grad_canonical` | `(&LoweredOp, wrt: usize) -> LoweredOp` | Symbolic gradient, canonicalized |
| `GradGraph::new` | `(&LoweredOp, n_vars: usize) -> GradGraph` | Pre-computed canonical gradient graph |
| `GradGraph::eval_with_grad` | `(&[f64]) -> Result<(f64, Vec<f64>), AdError>` | Value + all gradients, CSE pass |
| `eval_real` | `(&LoweredOp, &EvalCtx<'_>) -> Result<f64, EmlError>` | Scalar evaluation |

---

## 4. JIT Compilation — High-Performance Evaluation

`compile::to_jit` compiles a `LoweredOp` to native machine code via Cranelift.
The JIT-compiled function has the same ABI as `eval_real` but runs without any
interpreter overhead.

```rust
// requires feature = "jit"
use scirs2_symbolic::LoweredOp;
use scirs2_symbolic::compile::to_jit;
use scirs2_symbolic::eml::{eval_real, EvalCtx};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // f(x, y) = x² + sin(y) + x·y
    let formula = LoweredOp::Add(
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Pow(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(2.0)),
            )),
            Box::new(LoweredOp::Sin(Box::new(LoweredOp::Var(1)))),
        )),
        Box::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        )),
    );

    // ── Compile once; evaluate many times ────────────────────────────
    let jit_fn = to_jit(&formula).expect("JIT compilation");

    // Warm-up pass (forces any lazy mapping to complete).
    let _ = jit_fn.eval_checked(&[1.0, 1.0])
        .expect("JIT eval warm-up");

    // ── Benchmark: 100 000 evaluations via JIT ────────────────────────
    let vars = [2.5_f64, 0.8_f64];
    let n_evals: u32 = 100_000;

    let t0 = Instant::now();
    let mut jit_acc = 0.0_f64;
    for _ in 0..n_evals {
        jit_acc += jit_fn.eval_checked(&vars).expect("JIT eval");
    }
    let jit_ns = t0.elapsed().as_nanos() as f64 / f64::from(n_evals);

    // ── Benchmark: same count via interpreter ─────────────────────────
    let ctx = EvalCtx::new(&vars);
    let t1 = Instant::now();
    let mut interp_acc = 0.0_f64;
    for _ in 0..n_evals {
        interp_acc += eval_real(&formula, &ctx).expect("interp eval");
    }
    let interp_ns = t1.elapsed().as_nanos() as f64 / f64::from(n_evals);

    // Verify numerical parity.
    assert!(
        (jit_acc - interp_acc).abs() / (interp_acc.abs() + 1.0) < 1e-10,
        "JIT and interpreter diverged: {jit_acc} vs {interp_acc}"
    );

    println!("JIT       : {jit_ns:.1} ns / eval");
    println!("Interpreter: {interp_ns:.1} ns / eval");
    println!(
        "Speedup   : {:.1}×",
        interp_ns / jit_ns.max(0.001)
    );
    println!("Result (sum over {n_evals} evals): {jit_acc:.6}");

    // ── eval_checked vs eval ──────────────────────────────────────────
    // eval_checked returns Result; eval panics in debug on short slices.
    let checked = jit_fn.eval_checked(&vars)?;
    let unchecked = jit_fn.eval(&vars);
    assert!((checked - unchecked).abs() < 1e-15);

    println!("n_vars required by JIT: {}", jit_fn.n_vars());

    Ok(())
}
```

`Cargo.toml` additions:

```toml
[dependencies]
scirs2-symbolic = { version = "0.4.4", features = ["jit"] }
```

**`JitFunction` API:**

| Method | Returns | Notes |
|--------|---------|-------|
| `eval_checked(&[f64])` | `Result<f64, JitError>` | Bounds-checks before calling native code |
| `eval(&[f64])` | `f64` | Debug-asserts length; use in hot loops |
| `n_vars()` | `usize` | Minimum `vars` slice length required |

---

## 5. Serialization and Deployment

`LoweredOp` implements `serde::Serialize` / `serde::Deserialize` when the
`serde` feature is enabled. This lets you persist discovered formulas to disk,
send them over the wire, or embed them in configuration files.

```rust
// requires feature = "serde"
use scirs2_symbolic::LoweredOp;
use scirs2_symbolic::eml::{eval_real, EvalCtx};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Build a formula: f(x) = exp(-x²) (Gaussian kernel) ───────────
    let x_sq = LoweredOp::Pow(
        Box::new(LoweredOp::Var(0)),
        Box::new(LoweredOp::Const(2.0)),
    );
    let neg_x_sq = LoweredOp::Neg(Box::new(x_sq));
    let gaussian = LoweredOp::Exp(Box::new(neg_x_sq));

    // ── Serialize to JSON ─────────────────────────────────────────────
    let json_str: String = serde_json::to_string_pretty(&gaussian)
        .expect("serialization");
    println!("Serialized ({} bytes):", json_str.len());
    println!("{}", &json_str[..json_str.len().min(200)]);

    // ── Deserialize back ──────────────────────────────────────────────
    let restored: LoweredOp = serde_json::from_str(&json_str)
        .expect("deserialization");
    println!("\nRestored formula: {}", restored);

    // ── Verify numerical equivalence ──────────────────────────────────
    let test_points = [-1.0_f64, -0.5, 0.0, 0.5, 1.0];
    for &x in &test_points {
        let ctx = EvalCtx::new(&[x]);
        let original = eval_real(&gaussian, &ctx)?;
        let from_json = eval_real(&restored, &ctx)?;
        assert!(
            (original - from_json).abs() < 1e-15,
            "round-trip mismatch at x={x}: {original} vs {from_json}"
        );
    }
    println!("Round-trip numerical parity verified at 5 test points.");

    // ── Persist to a file ─────────────────────────────────────────────
    // (Writes to the system temp directory so the example is self-contained.)
    let formula_path = std::env::temp_dir().join("gaussian_kernel.json");
    std::fs::write(&formula_path, &json_str)
        .expect("write formula");
    println!("\nFormula saved to: {}", formula_path.display());

    // ── Load from file and re-evaluate ────────────────────────────────
    let loaded_json = std::fs::read_to_string(&formula_path)
        .expect("read formula");
    let loaded: LoweredOp = serde_json::from_str(&loaded_json)
        .expect("parse loaded formula");

    let ctx = EvalCtx::new(&[0.0_f64]);
    let val = eval_real(&loaded, &ctx)?;
    println!("f(0) from loaded formula: {} (expected 1.0)", val);
    assert!((val - 1.0).abs() < 1e-15);

    Ok(())
}
```

`Cargo.toml` additions:

```toml
[dependencies]
scirs2-symbolic = { version = "0.4.4", features = ["serde"] }
serde_json = "1"
```

---

## End-to-End: Chaining All Five Steps

The pipeline naturally composes: discover a formula, canonicalize it, build the
gradient graph, JIT-compile for deployment, and save the formula to disk.

```rust
// requires features = ["jit", "serde"]
use ndarray::{Array1, Array2};
use scirs2_symbolic::{
    discover, SrConfig, BuildingBlock,
};
use scirs2_symbolic::cas::{canonicalize};
use scirs2_symbolic::cas::ad::GradGraph;
use scirs2_symbolic::compile::to_jit;
use scirs2_symbolic::eml::{eval_real, EvalCtx};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. DISCOVER ─────────────────────────────────────────────────────
    let xs: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();
    let features = Array2::from_shape_vec((50, 1), xs).expect("shape");
    let targets = Array1::from_vec(ys);

    let config = SrConfig::default()
        .with_max_iter(80)
        .with_tolerance(1e-7)
        .with_building_blocks(vec![
            BuildingBlock::Arithmetic,
            BuildingBlock::Pow,
        ]);
    let results = discover(features.view(), targets.view(), &config);
    let best_op = results
        .into_iter()
        .next()
        .expect("discovery returned no results")
        .op;
    println!("Discovered: {}", best_op);

    // 2. CANONICALIZE ─────────────────────────────────────────────────
    let canonical = canonicalize(&best_op);
    let canon_op = canonical.into_op();
    println!("Canonical:  {}", canon_op);

    // 3. GRADIENT ─────────────────────────────────────────────────────
    let gg = GradGraph::new(&canon_op, 1);
    let (val, grad) = gg.eval_with_grad(&[3.0])?;
    println!("f(3) = {val:.4}  f'(3) = {:.4}", grad[0]);

    // 4. JIT ──────────────────────────────────────────────────────────
    let jit_fn = to_jit(&canon_op).expect("JIT");
    let jit_val = jit_fn.eval_checked(&[3.0])?;
    assert!((jit_val - val).abs() < 1e-10, "JIT/interp parity");
    println!("JIT f(3) = {jit_val:.4}");

    // 5. SERIALIZE ────────────────────────────────────────────────────
    let json = serde_json::to_string(&canon_op).expect("serialize");
    let path = std::env::temp_dir().join("discovered_formula.json");
    std::fs::write(&path, &json).expect("write");
    println!("Saved to {}", path.display());

    Ok(())
}
```

---

## Reference: Feature Flags

| Feature | Enables |
|---------|---------|
| *(none)* | `discover`, `canonicalize`, `grad_canonical`, `GradGraph`, `eval_real`, `simplify_op`, `EvalCtx` |
| `jit` | `compile::to_jit`, `JitFunction`, `JitError` |
| `serde` | `serde::Serialize`/`Deserialize` on `LoweredOp`, `OxiOp`, `EmlTree` |
| `smt` | `cas::EmlSmtSolver`, `cas::smt`, `cas::certified_rewrite` |
| `parallel` | Rayon-backed parallel map primitives |
| `numa` | NUMA-aware scheduler for `regression::discover` on large datasets |
| `gpu` | GPU-WGSL JIT via `wgpu` (implies `jit`); actual shader dispatch lands in v0.4.5 |
| `all-features` | All of the above |

`Cargo.toml` for all features:

```toml
[dependencies]
scirs2-symbolic = { version = "0.4.4", features = ["jit", "serde"] }
ndarray = "0.16"
serde_json = "1"
```
