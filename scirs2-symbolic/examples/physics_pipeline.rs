//! Physics Pipeline — `scirs2-symbolic` end-to-end example.
//!
//! Demonstrates the complete pipeline:
//!
//! 1. Build a formula directly via [`scirs2_symbolic::eml::LoweredOp`]
//!    constructors.
//! 2. Simplify symbolically via [`scirs2_symbolic::eml::simplify_op`].
//! 3. Compute symbolic gradients via [`scirs2_symbolic::eml::grad`].
//! 4. Evaluate at numerical points via
//!    [`scirs2_symbolic::eml::eval::eval_real`].
//! 5. Export to LaTeX via [`scirs2_symbolic::eml::to_latex`].
//!
//! Formula: kinetic energy `KE = ½·m·v²` and its partial derivatives.
//! Sanity check: `∂KE/∂v = m·v` is the linear-momentum magnitude.
//!
//! Run with: `cargo run -p scirs2-symbolic --example physics_pipeline`

use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::{grad, simplify_op, to_latex, LoweredOp};

fn main() {
    println!("=== Physics Pipeline: Kinetic Energy ===\n");

    // KE = ½ · m · v² with m = x_0, v = x_1.
    let m = LoweredOp::Var(0);
    let v = LoweredOp::Var(1);

    let v_squared = LoweredOp::Mul(Box::new(v.clone()), Box::new(v.clone()));
    let mv_squared = LoweredOp::Mul(Box::new(m.clone()), Box::new(v_squared));
    let ke = LoweredOp::Mul(Box::new(LoweredOp::Const(0.5)), Box::new(mv_squared));

    // 1. Show the original LoweredOp via Display + LaTeX.
    println!("Original LoweredOp (infix): {}", ke);
    println!("Original LaTeX            : {}", to_latex(&ke));

    // 2. Simplify (KE is already canonical-ish, but exercise the pass).
    let ke_simplified = simplify_op(&ke);
    println!("\nSimplified (infix)        : {}", ke_simplified);
    println!("Simplified (LaTeX)        : {}", to_latex(&ke_simplified));

    // 3. Symbolic partial derivatives.
    let dke_dm = grad(&ke, 0);
    let dke_dv = grad(&ke, 1);
    println!("\n∂KE/∂m (infix)            : {}", dke_dm);
    println!("∂KE/∂m (LaTeX)            : {}", to_latex(&dke_dm));
    println!("∂KE/∂v (infix)            : {}", dke_dv);
    println!("∂KE/∂v (LaTeX)            : {}", to_latex(&dke_dv));

    // 4. Evaluate at m = 2 kg, v = 10 m/s.
    let bindings = [2.0_f64, 10.0];
    let ctx = EvalCtx::new(&bindings);

    let ke_val = eval_real(&ke, &ctx).expect("eval KE");
    let dke_dm_val = eval_real(&dke_dm, &ctx).expect("eval ∂KE/∂m");
    let dke_dv_val = eval_real(&dke_dv, &ctx).expect("eval ∂KE/∂v");

    println!("\nAt m = 2 kg, v = 10 m/s:");
    println!("  KE         = {:>8.3} J          (expected 100)", ke_val);
    println!(
        "  ∂KE/∂m     = {:>8.3} J/kg        (expected 50  = ½·v²)",
        dke_dm_val
    );
    println!(
        "  ∂KE/∂v     = {:>8.3} kg·m/s      (expected 20  = m·v)",
        dke_dv_val
    );

    // 5. Closed-form physics cross-checks.
    let momentum = bindings[0] * bindings[1];
    let half_v_sq = 0.5 * bindings[1] * bindings[1];
    let ke_closed = 0.5 * bindings[0] * bindings[1] * bindings[1];

    println!("\nCross-checks:");
    println!(
        "  KE  closed-form = {:>10.6}, EML = {:>10.6}, |Δ| = {:.3e}",
        ke_closed,
        ke_val,
        (ke_val - ke_closed).abs()
    );
    println!(
        "  m·v closed-form = {:>10.6}, ∂KE/∂v EML = {:>10.6}, |Δ| = {:.3e}",
        momentum,
        dke_dv_val,
        (dke_dv_val - momentum).abs()
    );
    println!(
        "  ½v² closed-form = {:>10.6}, ∂KE/∂m EML = {:>10.6}, |Δ| = {:.3e}",
        half_v_sq,
        dke_dm_val,
        (dke_dm_val - half_v_sq).abs()
    );

    // Mini parameter sweep for KE(v) at fixed m = 1 kg.
    println!("\nKE sweep at m = 1 kg:");
    println!("  {:>6}  {:>10}  {:>10}", "v", "KE", "p = m·v");
    for vv in [0.0_f64, 1.0, 2.5, 5.0, 7.5, 10.0] {
        let local = [1.0, vv];
        let lctx = EvalCtx::new(&local);
        let ke_v = eval_real(&ke, &lctx).expect("eval KE in sweep");
        let p_v = eval_real(&dke_dv, &lctx).expect("eval ∂KE/∂v in sweep");
        println!("  {:>6.2}  {:>10.4}  {:>10.4}", vv, ke_v, p_v);
    }
}
