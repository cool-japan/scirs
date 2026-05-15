//! Symbolic gradient parity tests for `scirs2-autograd`.
//!
//! For each elementary operation, the float-tape gradient produced by `EmlOp`
//! is compared against the exact EML symbolic gradient at 100 evenly-spaced
//! test points.  All tests are gated on the `symbolic` feature.
//!
//! The tolerance `PARITY_TOL = 1e-10` is strict: both sides evaluate
//! *exactly the same* expression (the symbolic grad), so any gap beyond
//! floating-point round-off indicates a real bug in one of the backends.

#[cfg(feature = "symbolic")]
mod tests {
    use scirs2_autograd as ag;
    use scirs2_autograd::tensor_ops as T;
    use scirs2_symbolic::eml::{eval_real, grad as sym_grad, EvalCtx, LoweredOp};
    use std::sync::Arc;

    const PARITY_TOL: f64 = 1e-10;

    /// Generate `n` evenly-spaced points in `[lo, hi]`.
    fn test_points(n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n)
            .map(|i| lo + (hi - lo) * (i as f64) / ((n - 1) as f64))
            .collect()
    }

    /// Evaluate the symbolic gradient of `op` with respect to `var_idx` at `x`.
    fn eval_sym_grad(op: &LoweredOp, var_idx: usize, x: f64) -> f64 {
        let gop = sym_grad(op, var_idx);
        let mut bindings = vec![0.0_f64; var_idx + 1];
        bindings[var_idx] = x;
        let ctx = EvalCtx::new(&bindings);
        eval_real(&gop, &ctx).unwrap_or(f64::NAN)
    }

    /// Evaluate the float-tape gradient of `op` (treating it as an `EmlOp`) at `x`.
    fn eval_tape_grad(op: Arc<LoweredOp>, x: f64) -> f64 {
        let mut result = f64::NAN;
        ag::run(|g: &mut ag::Context<f64>| {
            let x_ph = g.placeholder("x", &[]);
            let y = ag::eml_scalar_op(Arc::clone(&op), &[x_ph], g);
            let dy_dx = &T::grad(&[y], &[x_ph])[0];

            let x_val = scirs2_core::ndarray::arr0(x).into_dyn();
            let out = g.evaluator().push(dy_dx).feed(x_ph, x_val.view()).run();
            result = out[0]
                .as_ref()
                .ok()
                .and_then(|a| a.iter().next().copied())
                .unwrap_or(f64::NAN);
        });
        result
    }

    // ------------------------------------------------------------------
    // 1. x^2  — symbolic grad = 2*x,  x in [0.1, 2.0]
    // ------------------------------------------------------------------
    #[test]
    fn parity_x_squared() {
        let op = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        for x in test_points(100, 0.1, 2.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "x^2 at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 2. sin(x)  — symbolic grad = cos(x),  x in [0.1, 2.0]
    // ------------------------------------------------------------------
    #[test]
    fn parity_sin() {
        let op = Arc::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0))));
        for x in test_points(100, 0.1, 2.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "sin(x) at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 3. cos(x)  — symbolic grad = -sin(x),  x in [0.1, 2.0]
    // ------------------------------------------------------------------
    #[test]
    fn parity_cos() {
        let op = Arc::new(LoweredOp::Cos(Box::new(LoweredOp::Var(0))));
        for x in test_points(100, 0.1, 2.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "cos(x) at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 4. exp(x)  — symbolic grad = exp(x),  x in [0.1, 2.0]
    // ------------------------------------------------------------------
    #[test]
    fn parity_exp() {
        let op = Arc::new(LoweredOp::Exp(Box::new(LoweredOp::Var(0))));
        for x in test_points(100, 0.1, 2.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "exp(x) at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 5. ln(x)  — symbolic grad = 1/x,  x in [0.5, 3.0]
    // ------------------------------------------------------------------
    #[test]
    fn parity_ln() {
        let op = Arc::new(LoweredOp::Ln(Box::new(LoweredOp::Var(0))));
        for x in test_points(100, 0.5, 3.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "ln(x) at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 6. sqrt(x)  — symbolic grad = 1 / (2*sqrt(x)),  x in [0.5, 3.0]
    // ------------------------------------------------------------------
    #[test]
    fn parity_sqrt() {
        let op = Arc::new(LoweredOp::Sqrt(Box::new(LoweredOp::Var(0))));
        for x in test_points(100, 0.5, 3.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "sqrt(x) at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 7. x^3  — symbolic grad = 3*x^2,  x in [0.1, 2.0]
    // ------------------------------------------------------------------
    #[test]
    fn parity_x_cubed() {
        let op = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(3.0)),
        ));
        for x in test_points(100, 0.1, 2.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "x^3 at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 8. 1/x  — symbolic grad = -1/x^2,  x in [0.5, 3.0]
    //
    // Expressed as Div(Const(1.0), Var(0)) to avoid negative-exponent
    // ambiguity in Pow.
    // ------------------------------------------------------------------
    #[test]
    fn parity_reciprocal() {
        let op = Arc::new(LoweredOp::Div(
            Box::new(LoweredOp::Const(1.0)),
            Box::new(LoweredOp::Var(0)),
        ));
        for x in test_points(100, 0.5, 3.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "1/x at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 9. tan(x)  — symbolic grad = 1/cos²(x),  x in [0.1, 1.4]
    //    (avoids the singularity at π/2 ≈ 1.5708)
    // ------------------------------------------------------------------
    #[test]
    fn parity_tan() {
        let op = Arc::new(LoweredOp::Tan(Box::new(LoweredOp::Var(0))));
        for x in test_points(100, 0.1, 1.4) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "tan(x) at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 10. sinh(x)  — symbolic grad = cosh(x),  x in [0.1, 2.0]
    // ------------------------------------------------------------------
    #[test]
    fn parity_sinh() {
        let op = Arc::new(LoweredOp::Sinh(Box::new(LoweredOp::Var(0))));
        for x in test_points(100, 0.1, 2.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "sinh(x) at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 11. cosh(x)  — symbolic grad = sinh(x),  x in [0.1, 2.0]
    // ------------------------------------------------------------------
    #[test]
    fn parity_cosh() {
        let op = Arc::new(LoweredOp::Cosh(Box::new(LoweredOp::Var(0))));
        for x in test_points(100, 0.1, 2.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "cosh(x) at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }

    // ------------------------------------------------------------------
    // 12. arctan(x)  — symbolic grad = 1/(1+x²),  x in [0.1, 2.0]
    // ------------------------------------------------------------------
    #[test]
    fn parity_arctan() {
        let op = Arc::new(LoweredOp::Arctan(Box::new(LoweredOp::Var(0))));
        for x in test_points(100, 0.1, 2.0) {
            let sym = eval_sym_grad(&op, 0, x);
            let tape = eval_tape_grad(Arc::clone(&op), x);
            assert!(
                (sym - tape).abs() < PARITY_TOL,
                "arctan(x) at x={x:.6}: sym={sym}, tape={tape}, diff={}",
                (sym - tape).abs()
            );
        }
    }
}
