//! Lorenz Attractor — `scirs2-symbolic` example.
//!
//! Builds the Lorenz system symbolically and computes the 3×3 Jacobian
//! matrix at a chosen state via [`scirs2_symbolic::eml::grad`].
//!
//! The system:
//! - `dx/dt = σ·(y − x)`
//! - `dy/dt = x·(ρ − z) − y`
//! - `dz/dt = x·y − β·z`
//!
//! Closed-form Jacobian (verified by the printed numerical values):
//! ```text
//!     ⎡ -σ      σ      0  ⎤
//! J = ⎢ ρ-z   -1     -x  ⎥
//!     ⎣  y     x    -β   ⎦
//! ```
//!
//! Variables: `x = x_0`, `y = x_1`, `z = x_2`, `σ = x_3`, `ρ = x_4`, `β = x_5`.
//!
//! Run with: `cargo run -p scirs2-symbolic --example lorenz`

use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::{grad, LoweredOp};

fn main() {
    println!("=== Lorenz Attractor — Jacobian (scirs2-symbolic) ===\n");

    // State variables.
    let x = LoweredOp::Var(0);
    let y = LoweredOp::Var(1);
    let z = LoweredOp::Var(2);
    // Parameters.
    let sigma = LoweredOp::Var(3);
    let rho = LoweredOp::Var(4);
    let beta = LoweredOp::Var(5);

    // dx/dt = σ·(y − x)
    let dx_dt = LoweredOp::Mul(
        Box::new(sigma.clone()),
        Box::new(LoweredOp::Sub(Box::new(y.clone()), Box::new(x.clone()))),
    );

    // dy/dt = x·(ρ − z) − y
    let dy_dt = LoweredOp::Sub(
        Box::new(LoweredOp::Mul(
            Box::new(x.clone()),
            Box::new(LoweredOp::Sub(Box::new(rho.clone()), Box::new(z.clone()))),
        )),
        Box::new(y.clone()),
    );

    // dz/dt = x·y − β·z
    let dz_dt = LoweredOp::Sub(
        Box::new(LoweredOp::Mul(Box::new(x.clone()), Box::new(y.clone()))),
        Box::new(LoweredOp::Mul(Box::new(beta.clone()), Box::new(z.clone()))),
    );

    // Symbolic Jacobian: ∂(dx/dt, dy/dt, dz/dt) / ∂(x, y, z).
    let jacobian: Vec<Vec<LoweredOp>> = vec![
        vec![grad(&dx_dt, 0), grad(&dx_dt, 1), grad(&dx_dt, 2)],
        vec![grad(&dy_dt, 0), grad(&dy_dt, 1), grad(&dy_dt, 2)],
        vec![grad(&dz_dt, 0), grad(&dz_dt, 1), grad(&dz_dt, 2)],
    ];

    // Standard Lorenz parameters in the chaotic regime: σ=10, ρ=28, β=8/3.
    // Initial probe point (x, y, z) = (1, 1, 1).
    let bindings = [1.0_f64, 1.0, 1.0, 10.0, 28.0, 8.0 / 3.0];
    let ctx = EvalCtx::new(&bindings);

    let var_names = ["x", "y", "z"];
    let row_names = ["dx/dt", "dy/dt", "dz/dt"];

    println!(
        "Jacobian at (x={}, y={}, z={}) with σ={}, ρ={}, β={:.6}:",
        bindings[0], bindings[1], bindings[2], bindings[3], bindings[4], bindings[5]
    );
    println!(
        "  {:>10}  {:>10}  {:>10}  {:>10}",
        "", "∂/∂x", "∂/∂y", "∂/∂z"
    );
    for (i, row) in jacobian.iter().enumerate() {
        print!("  {:>10}", row_names[i]);
        for cell in row.iter() {
            let r = eval_real(cell, &ctx).expect("eval Jacobian cell");
            print!("  {:>10.4}", r);
        }
        println!();
    }

    // Independently reconstructed analytic Jacobian for comparison.
    let (sx, sy, sz, ssigma, srho, sbeta) = (
        bindings[0],
        bindings[1],
        bindings[2],
        bindings[3],
        bindings[4],
        bindings[5],
    );
    let analytic = [
        [-ssigma, ssigma, 0.0],
        [srho - sz, -1.0, -sx],
        [sy, sx, -sbeta],
    ];

    println!("\nAnalytic Jacobian (closed form):");
    println!(
        "  {:>10}  {:>10}  {:>10}  {:>10}",
        "", "∂/∂x", "∂/∂y", "∂/∂z"
    );
    for (i, row) in analytic.iter().enumerate() {
        print!("  {:>10}", row_names[i]);
        for cell in row.iter() {
            print!("  {:>10.4}", cell);
        }
        println!();
    }

    // Maximum |EML − analytic| element across the Jacobian.
    let mut max_err = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let r_eml = eval_real(&jacobian[i][j], &ctx).expect("eval Jacobian for diff");
            let r_an = analytic[i][j];
            let e = (r_eml - r_an).abs();
            if e > max_err {
                max_err = e;
            }
            // Suppress noise on var index unused warning.
            let _ = var_names[j];
        }
    }
    println!(
        "\nMax |EML − analytic| over the 3×3 Jacobian: {:.3e}",
        max_err
    );
}
