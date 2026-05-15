//! Simple Pendulum — `scirs2-symbolic` example.
//!
//! Builds the equation of motion `d²θ/dt² = -(g/L)·sin(θ)` symbolically,
//! computes the gradient `d/dθ sin(θ) = cos(θ)`, and demonstrates the
//! small-angle approximation `sin(θ) ≈ θ`.
//!
//! # Note on evaluation path
//!
//! `Canonical::sin(x)` encodes Euler's formula `(exp(ix) − exp(−ix)) / (2i)`
//! and contains `ln(−1)` in the lowered tree. It must be evaluated through
//! [`scirs2_symbolic::eml::eval_complex`]; the real-only path raises a
//! domain violation by construction (see `eml/eval.rs:614-622` and the doc
//! comment on `Canonical::sin` in `eml/canonical.rs:160-166`). We bind the
//! real-valued angle as a `Complex64` with zero imaginary part and read the
//! real component back.
//!
//! Run with: `cargo run -p scirs2-symbolic --example pendulum`

use num_complex::Complex64;
use scirs2_symbolic::eml::{eval_complex, grad, lower, Canonical, EmlTree};

fn main() {
    println!("=== Simple Pendulum (scirs2-symbolic) ===\n");

    // Variables: θ = x_0
    let theta = EmlTree::var(0);

    // Canonical sin(θ) — Euler-formula encoding.
    let sin_theta = Canonical::sin(&theta);
    println!("sin(θ) canonical depth: {}", sin_theta.depth());

    // Lower to flat IR (literal `Sub(Exp, Ln)` fallback for canonical sin).
    let lowered = lower(&sin_theta);

    // Symbolic gradient: d/dθ sin(θ) = cos(θ).
    // Printed via `Display` (parenthesised infix); the literal form is large
    // because Canonical::sin's 543-deep encoding survives differentiation.
    let g = grad(&lowered, 0);
    println!(
        "d/dθ sin(θ) — simplified LoweredOp size (chars): {}",
        g.to_string().len()
    );

    // Evaluate at small angles via the complex path.
    println!("\nSmall-angle behaviour (θ in radians):");
    println!(
        "  {:>8}  {:>15}  {:>15}  {:>12}",
        "θ", "sin(θ) [EML]", "linear approx θ", "rel error"
    );
    for theta_v in [0.0_f64, 0.01, 0.1, 0.5, 1.0, std::f64::consts::FRAC_PI_4] {
        let z = Complex64::new(theta_v, 0.0);
        match eval_complex(&lowered, &[z]) {
            Ok(c) => {
                let s = c.re;
                let rel_err = if theta_v.abs() > 1e-12 {
                    ((s - theta_v) / s).abs()
                } else {
                    0.0
                };
                println!(
                    "  {:>8.4}  {:>15.9}  {:>15.9}  {:>12.3e}",
                    theta_v, s, theta_v, rel_err
                );
            }
            Err(e) => println!("  sin({:.4}) failed: {}", theta_v, e),
        }
    }

    // Cross-check against the std `f64::sin` (the EML path is what's
    // load-bearing; this is just a tiny independent sanity check).
    let theta_v = std::f64::consts::FRAC_PI_4;
    let std_sin = theta_v.sin();
    let z = Complex64::new(theta_v, 0.0);
    let eml_sin = eval_complex(&lowered, &[z])
        .map(|c| c.re)
        .unwrap_or(f64::NAN);
    println!(
        "\nCross-check at θ = π/4: std::sin = {:.12}, EML sin = {:.12}, |Δ| = {:.3e}",
        std_sin,
        eml_sin,
        (std_sin - eml_sin).abs()
    );
}
