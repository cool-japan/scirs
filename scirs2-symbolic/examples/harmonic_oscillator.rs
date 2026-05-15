//! Harmonic Oscillator — `scirs2-symbolic` example.
//!
//! Builds the position `x(t) = A·cos(ωt + φ)`, computes velocity
//! `dx/dt = −A·ω·sin(ωt + φ)` and acceleration
//! `d²x/dt² = −A·ω²·cos(ωt + φ) = −ω²·x` symbolically, and verifies the
//! second-order ODE `ẍ + ω²·x = 0` numerically at a representative point.
//!
//! Variables: `t = x_0`, `A = x_1`, `ω = x_2`, `φ = x_3`.
//!
//! Run with: `cargo run -p scirs2-symbolic --example harmonic_oscillator`

use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::{grad, LoweredOp};

fn main() {
    println!("=== Harmonic Oscillator (scirs2-symbolic) ===\n");

    // x(t) = A · cos(ω·t + φ).
    let t = LoweredOp::Var(0);
    let amp = LoweredOp::Var(1);
    let omega = LoweredOp::Var(2);
    let phi = LoweredOp::Var(3);

    let inner = LoweredOp::Add(
        Box::new(LoweredOp::Mul(Box::new(omega.clone()), Box::new(t.clone()))),
        Box::new(phi.clone()),
    );
    let x_pos = LoweredOp::Mul(
        Box::new(amp.clone()),
        Box::new(LoweredOp::Cos(Box::new(inner))),
    );

    // Velocity: dx/dt = −A·ω·sin(ω·t + φ).
    let velocity = grad(&x_pos, 0);

    // Acceleration: d²x/dt² = −A·ω²·cos(ω·t + φ) = −ω²·x.
    let acceleration = grad(&velocity, 0);

    // Evaluate at t=0, A=1, ω=2, φ=0 → expect x=1, v=0, a=−4.
    let bindings = [0.0_f64, 1.0, 2.0, 0.0];
    let ctx = EvalCtx::new(&bindings);

    let x_val = eval_real(&x_pos, &ctx).expect("eval x_pos at origin");
    let v_val = eval_real(&velocity, &ctx).expect("eval velocity at origin");
    let a_val = eval_real(&acceleration, &ctx).expect("eval acceleration at origin");

    println!("At t=0, A=1, ω=2, φ=0:");
    println!("  x   = {:>10.6}", x_val);
    println!("  v   = {:>10.6}", v_val);
    println!("  a   = {:>10.6}", a_val);

    let omega_v = bindings[2];
    let omega_sq = omega_v * omega_v;
    let restoring = omega_sq * x_val;
    let ode_residual = a_val + restoring;
    println!("  ω²·x = {:>10.6}", restoring);
    println!(
        "  ODE residual (a + ω²·x) = {:>10.3e} (expected 0)",
        ode_residual
    );

    // Sweep over a quarter-period (T = 2π/ω) to confirm the ODE holds
    // identically along the trajectory — a tighter test than the single
    // origin point above.
    println!("\nODE residual sweep across one quarter-period:");
    println!(
        "  {:>8}  {:>12}  {:>12}  {:>12}",
        "t", "x(t)", "a(t)", "a + ω²·x"
    );
    let n_steps = 6_usize;
    let quarter = std::f64::consts::FRAC_PI_2 / omega_v;
    for k in 0..=n_steps {
        let t_v = quarter * (k as f64) / (n_steps as f64);
        let local_bindings = [t_v, bindings[1], omega_v, bindings[3]];
        let local_ctx = EvalCtx::new(&local_bindings);
        let xk = eval_real(&x_pos, &local_ctx).expect("eval x_pos in sweep");
        let ak = eval_real(&acceleration, &local_ctx).expect("eval acceleration in sweep");
        let res = ak + omega_sq * xk;
        println!("  {:>8.4}  {:>12.6}  {:>12.6}  {:>12.3e}", t_v, xk, ak, res);
    }
}
