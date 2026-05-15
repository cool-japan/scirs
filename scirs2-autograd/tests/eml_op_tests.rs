//! Integration tests for EmlOp symbolic backend.
//!
//! All tests require the `symbolic` feature to be enabled. They verify that
//! forward evaluation and backward (gradient) computation through `EmlOp` match
//! the exact symbolic results from `scirs2_symbolic::eml`.

#[cfg(feature = "symbolic")]
mod tests {
    use scirs2_autograd as ag;
    use scirs2_autograd::tensor_ops as T;
    use scirs2_symbolic::eml::LoweredOp;
    use std::sync::Arc;

    const TOL: f64 = 1e-10;

    // ------------------------------------------------------------------
    // Test 1: x^2 forward = 9.0 at x = 3.0
    // ------------------------------------------------------------------
    #[test]
    fn test_forward_x_squared_at_3() {
        let op = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = ag::eml_scalar_op(Arc::clone(&op), &[x], g);

            let x_val = scirs2_core::ndarray::arr0(3.0_f64).into_dyn();
            let result = g.evaluator().push(&y).feed(x, x_val.view()).run();
            let val = result[0].clone().expect("eval should succeed");
            let got = val.iter().next().copied().unwrap_or(f64::NAN);
            assert!((got - 9.0).abs() < TOL, "x^2 at 3 = {}, expected 9", got);
        });
    }

    // ------------------------------------------------------------------
    // Test 2: x^2 backward = 6.0 at x = 3.0 (exact: d/dx x^2 = 2x)
    // ------------------------------------------------------------------
    #[test]
    fn test_backward_x_squared_grad_at_3() {
        let op = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = ag::eml_scalar_op(Arc::clone(&op), &[x], g);
            let dy_dx = &T::grad(&[y], &[x])[0];

            let x_val = scirs2_core::ndarray::arr0(3.0_f64).into_dyn();
            let result = g.evaluator().push(dy_dx).feed(x, x_val.view()).run();
            let grad = result[0].clone().expect("grad eval should succeed");
            let got = grad.iter().next().copied().unwrap_or(f64::NAN);
            // 2 * 3 = 6
            assert!(
                (got - 6.0).abs() < TOL,
                "d/dx x^2 at 3 = {}, expected 6",
                got
            );
        });
    }

    // ------------------------------------------------------------------
    // Test 3: sin(x) backward = cos(0) = 1.0 at x = 0
    // ------------------------------------------------------------------
    #[test]
    fn test_backward_sin_at_zero() {
        let op = Arc::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0))));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = ag::eml_scalar_op(Arc::clone(&op), &[x], g);
            let dy_dx = &T::grad(&[y], &[x])[0];

            let x_val = scirs2_core::ndarray::arr0(0.0_f64).into_dyn();
            let result = g.evaluator().push(dy_dx).feed(x, x_val.view()).run();
            let grad = result[0].clone().expect("grad eval should succeed");
            let got = grad.iter().next().copied().unwrap_or(f64::NAN);
            // cos(0) = 1.0
            assert!(
                (got - 1.0).abs() < TOL,
                "d/dx sin(x) at 0 = {}, expected 1",
                got
            );
        });
    }

    // ------------------------------------------------------------------
    // Test 4: f(x,y) = x*y — grad w.r.t. x = y = 3 at (2, 3)
    // ------------------------------------------------------------------
    #[test]
    fn test_backward_xy_grad_x() {
        let op = Arc::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        ));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y_ph = g.placeholder("y", &[]);
            let result_tensor = ag::eml_scalar_op(Arc::clone(&op), &[x, y_ph], g);
            let d_dx = &T::grad(&[result_tensor], &[x])[0];

            let x_val = scirs2_core::ndarray::arr0(2.0_f64).into_dyn();
            let y_val = scirs2_core::ndarray::arr0(3.0_f64).into_dyn();
            let out = g
                .evaluator()
                .push(d_dx)
                .feed(x, x_val.view())
                .feed(y_ph, y_val.view())
                .run();
            let grad = out[0].clone().expect("grad eval should succeed");
            let got = grad.iter().next().copied().unwrap_or(f64::NAN);
            // d/dx (x*y) = y = 3
            assert!(
                (got - 3.0).abs() < TOL,
                "d/dx x*y at (2,3) = {}, expected 3",
                got
            );
        });
    }

    // ------------------------------------------------------------------
    // Test 5: f(x,y) = x*y — grad w.r.t. y = x = 2 at (2, 3)
    // ------------------------------------------------------------------
    #[test]
    fn test_backward_xy_grad_y() {
        let op = Arc::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        ));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y_ph = g.placeholder("y", &[]);
            let result_tensor = ag::eml_scalar_op(Arc::clone(&op), &[x, y_ph], g);
            let d_dy = &T::grad(&[result_tensor], &[y_ph])[0];

            let x_val = scirs2_core::ndarray::arr0(2.0_f64).into_dyn();
            let y_val = scirs2_core::ndarray::arr0(3.0_f64).into_dyn();
            let out = g
                .evaluator()
                .push(d_dy)
                .feed(x, x_val.view())
                .feed(y_ph, y_val.view())
                .run();
            let grad = out[0].clone().expect("grad eval should succeed");
            let got = grad.iter().next().copied().unwrap_or(f64::NAN);
            // d/dy (x*y) = x = 2
            assert!(
                (got - 2.0).abs() < TOL,
                "d/dy x*y at (2,3) = {}, expected 2",
                got
            );
        });
    }

    // ------------------------------------------------------------------
    // Test 6: ln(x) gradient = 1/x = 0.5 at x = 2
    // ------------------------------------------------------------------
    #[test]
    fn test_backward_ln_at_2() {
        let op = Arc::new(LoweredOp::Ln(Box::new(LoweredOp::Var(0))));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = ag::eml_scalar_op(Arc::clone(&op), &[x], g);
            let dy_dx = &T::grad(&[y], &[x])[0];

            let x_val = scirs2_core::ndarray::arr0(2.0_f64).into_dyn();
            let result = g.evaluator().push(dy_dx).feed(x, x_val.view()).run();
            let grad = result[0].clone().expect("grad eval should succeed");
            let got = grad.iter().next().copied().unwrap_or(f64::NAN);
            // d/dx ln(x) = 1/x = 0.5 at x=2
            assert!(
                (got - 0.5).abs() < TOL,
                "d/dx ln(x) at 2 = {}, expected 0.5",
                got
            );
        });
    }

    // ------------------------------------------------------------------
    // Test 7: constant expression — gradient w.r.t. any variable is 0
    //
    // We use LoweredOp::Const(5.0) with one dummy variable input.
    // sym_grad(Const(5.0), 0) = Const(0.0), so gx = 0 * gy = 0.
    // ------------------------------------------------------------------
    #[test]
    fn test_backward_const_has_zero_grad() {
        // f = 5.0 (a constant, ignores Var(0))
        let op = Arc::new(LoweredOp::Const(5.0));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = ag::eml_scalar_op(Arc::clone(&op), &[x], g);
            let dy_dx = &T::grad(&[y], &[x])[0];

            let x_val = scirs2_core::ndarray::arr0(42.0_f64).into_dyn();
            let result = g.evaluator().push(dy_dx).feed(x, x_val.view()).run();
            let grad = result[0].clone().expect("grad eval should succeed");
            let got = grad.iter().next().copied().unwrap_or(f64::NAN);
            assert!(got.abs() < TOL, "d/dx const at any x = {}, expected 0", got);
        });
    }

    // ------------------------------------------------------------------
    // Test 8: EmlOp composable with regular autograd ops
    //
    // f(x) = x^2, then multiply by 2.0 using normal autograd `* 2.0`.
    // Result: h(x) = 2 * x^2, dh/dx = 4*x.
    // At x=3: dh/dx = 12.
    // ------------------------------------------------------------------
    #[test]
    fn test_composable_with_regular_ops() {
        let op = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let x_sq = ag::eml_scalar_op(Arc::clone(&op), &[x], g);
            // Compose with regular autograd multiplication
            let h = x_sq * 2.0_f64;
            let dh_dx = &T::grad(&[h], &[x])[0];

            let x_val = scirs2_core::ndarray::arr0(3.0_f64).into_dyn();
            let result = g.evaluator().push(dh_dx).feed(x, x_val.view()).run();
            let grad = result[0].clone().expect("grad eval should succeed");
            let got = grad.iter().next().copied().unwrap_or(f64::NAN);
            // dh/dx = 2 * 2x = 4x; at x=3 → 12
            assert!(
                (got - 12.0).abs() < TOL,
                "d/dx (2*x^2) at 3 = {}, expected 12",
                got
            );
        });
    }
}
