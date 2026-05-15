//! Integration tests for the EML tape backend:
//! `EmlElementWiseOp`, `EmlJacobianOp`, `EmlHessianOp`, and dispatch utilities.
//!
//! All tests require the `symbolic` feature.

#[cfg(feature = "symbolic")]
mod tests {
    use scirs2_autograd as ag;
    use scirs2_symbolic::eml::LoweredOp;
    use std::sync::Arc;

    const TOL: f64 = 1e-9;

    // eval_scalar removed — inline the evaluator pattern in each test to avoid lifetime issues.

    // ------------------------------------------------------------------
    // Test 1: eml_elementwise forward — sin(x) on [0, π/2, π]
    // Expected result ≈ [0, 1, 0]  (within 1e-9)
    // ------------------------------------------------------------------
    #[test]
    fn test_elementwise_sin_forward() {
        let sin_op = Arc::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0))));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[3]);
            let y = ag::eml_elementwise(Arc::clone(&sin_op), x, g);

            let x_arr = scirs2_core::ndarray::Array::from_vec(vec![
                0.0_f64,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::PI,
            ])
            .into_dyn();

            let result = g.evaluator().push(&y).feed(x, x_arr.view()).run();
            let arr = result[0].clone().expect("eval ok");
            let vals: Vec<f64> = arr.iter().copied().collect();
            assert_eq!(vals.len(), 3);
            assert!(vals[0].abs() < TOL, "sin(0) = {}", vals[0]);
            assert!((vals[1] - 1.0).abs() < 1e-9, "sin(π/2) = {}", vals[1]);
            assert!(vals[2].abs() < 1e-9, "sin(π) = {}", vals[2]);
        });
    }

    // ------------------------------------------------------------------
    // Test 2: eml_elementwise backward — gradient of x^2 element-wise
    // f(x) = x^2 element-wise, df/dx = 2x; at x=[1,2,3] → [2,4,6]
    // ------------------------------------------------------------------
    #[test]
    fn test_elementwise_square_backward() {
        let sq_op = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[3]);
            let y = ag::eml_elementwise(Arc::clone(&sq_op), x, g);
            // Sum so we can get a scalar gradient (dy_sum/dx[i] = 2*x[i])
            let y_sum = ag::tensor_ops::reduce_sum(y, &[0], false);
            let grads = ag::tensor_ops::grad(&[y_sum], &[x]);
            let gx = &grads[0];

            let x_arr = scirs2_core::ndarray::Array::from_vec(vec![1.0_f64, 2.0, 3.0]).into_dyn();
            let result = g.evaluator().push(gx).feed(x, x_arr.view()).run();
            let arr = result[0].clone().expect("grad eval ok");
            let vals: Vec<f64> = arr.iter().copied().collect();
            assert_eq!(vals.len(), 3);
            assert!((vals[0] - 2.0).abs() < TOL, "d/dx x^2 at 1 = {}", vals[0]);
            assert!((vals[1] - 4.0).abs() < TOL, "d/dx x^2 at 2 = {}", vals[1]);
            assert!((vals[2] - 6.0).abs() < TOL, "d/dx x^2 at 3 = {}", vals[2]);
        });
    }

    // ------------------------------------------------------------------
    // Test 3: eml_jacobian forward — f=[x*y, x+y] at (2, 3)
    // J = [[y, x], [1, 1]] = [[3, 2], [1, 1]]
    // ------------------------------------------------------------------
    #[test]
    fn test_jacobian_forward_xy() {
        // f0 = x * y = Var(0) * Var(1)
        let f0 = Arc::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        ));
        // f1 = x + y = Var(0) + Var(1)
        let f1 = Arc::new(LoweredOp::Add(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        ));

        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = g.placeholder("y", &[]);
            let jac = ag::eml_jacobian(vec![Arc::clone(&f0), Arc::clone(&f1)], &[x, y], g);

            let x_arr = scirs2_core::ndarray::arr0(2.0_f64).into_dyn();
            let y_arr = scirs2_core::ndarray::arr0(3.0_f64).into_dyn();
            let result = g
                .evaluator()
                .push(&jac)
                .feed(x, x_arr.view())
                .feed(y, y_arr.view())
                .run();
            let arr = result[0].clone().expect("jac eval ok");
            let vals: Vec<f64> = arr.iter().copied().collect();
            // Row-major: [[3, 2], [1, 1]]
            assert_eq!(vals.len(), 4);
            assert!((vals[0] - 3.0).abs() < TOL, "J[0,0]=df0/dx={}", vals[0]);
            assert!((vals[1] - 2.0).abs() < TOL, "J[0,1]=df0/dy={}", vals[1]);
            assert!((vals[2] - 1.0).abs() < TOL, "J[1,0]=df1/dx={}", vals[2]);
            assert!((vals[3] - 1.0).abs() < TOL, "J[1,1]=df1/dy={}", vals[3]);
        });
    }

    // ------------------------------------------------------------------
    // Test 4: eml_jacobian shape — result is 2-D with correct shape
    // ------------------------------------------------------------------
    #[test]
    fn test_jacobian_shape() {
        let f0 = Arc::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        ));
        let f1 = Arc::new(LoweredOp::Add(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        ));

        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = g.placeholder("y", &[]);
            let jac = ag::eml_jacobian(vec![Arc::clone(&f0), Arc::clone(&f1)], &[x, y], g);

            let x_arr = scirs2_core::ndarray::arr0(1.0_f64).into_dyn();
            let y_arr = scirs2_core::ndarray::arr0(1.0_f64).into_dyn();
            let result = g
                .evaluator()
                .push(&jac)
                .feed(x, x_arr.view())
                .feed(y, y_arr.view())
                .run();
            let arr = result[0].clone().expect("jac shape eval ok");
            assert_eq!(arr.ndim(), 2, "Jacobian must be 2-D");
            assert_eq!(arr.shape(), &[2, 2], "Jacobian shape must be [n_out, n_in]");
        });
    }

    // ------------------------------------------------------------------
    // Test 5: eml_hessian forward — f=x²+y², H=[[2,0],[0,2]]
    // ------------------------------------------------------------------
    #[test]
    fn test_hessian_sum_of_squares() {
        // f = x^2 + y^2 = Var(0)^2 + Var(1)^2
        let f = Arc::new(LoweredOp::Add(
            Box::new(LoweredOp::Pow(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(2.0)),
            )),
            Box::new(LoweredOp::Pow(
                Box::new(LoweredOp::Var(1)),
                Box::new(LoweredOp::Const(2.0)),
            )),
        ));

        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = g.placeholder("y", &[]);
            let hess = ag::eml_hessian(Arc::clone(&f), &[x, y], g);

            let x_arr = scirs2_core::ndarray::arr0(5.0_f64).into_dyn();
            let y_arr = scirs2_core::ndarray::arr0(7.0_f64).into_dyn();
            let result = g
                .evaluator()
                .push(&hess)
                .feed(x, x_arr.view())
                .feed(y, y_arr.view())
                .run();
            let arr = result[0].clone().expect("hessian eval ok");
            let vals: Vec<f64> = arr.iter().copied().collect();
            // H = [[2, 0], [0, 2]]
            assert_eq!(vals.len(), 4);
            assert!((vals[0] - 2.0).abs() < TOL, "H[0,0]={}", vals[0]);
            assert!(vals[1].abs() < TOL, "H[0,1]={}", vals[1]);
            assert!(vals[2].abs() < TOL, "H[1,0]={}", vals[2]);
            assert!((vals[3] - 2.0).abs() < TOL, "H[1,1]={}", vals[3]);
        });
    }

    // ------------------------------------------------------------------
    // Test 6: eml_hessian forward — f=x*y, H=[[0,1],[1,0]]
    // ------------------------------------------------------------------
    #[test]
    fn test_hessian_product() {
        // f = x * y
        let f = Arc::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        ));

        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = g.placeholder("y", &[]);
            let hess = ag::eml_hessian(Arc::clone(&f), &[x, y], g);

            let x_arr = scirs2_core::ndarray::arr0(3.0_f64).into_dyn();
            let y_arr = scirs2_core::ndarray::arr0(4.0_f64).into_dyn();
            let result = g
                .evaluator()
                .push(&hess)
                .feed(x, x_arr.view())
                .feed(y, y_arr.view())
                .run();
            let arr = result[0].clone().expect("hessian product eval ok");
            let vals: Vec<f64> = arr.iter().copied().collect();
            // H = [[0, 1], [1, 0]]
            assert_eq!(vals.len(), 4);
            assert!(vals[0].abs() < TOL, "H[0,0]={}", vals[0]);
            assert!((vals[1] - 1.0).abs() < TOL, "H[0,1]={}", vals[1]);
            assert!((vals[2] - 1.0).abs() < TOL, "H[1,0]={}", vals[2]);
            assert!(vals[3].abs() < TOL, "H[1,1]={}", vals[3]);
        });
    }

    // ------------------------------------------------------------------
    // Test 7: is_eml_backed returns true for eml_scalar_op tensor
    // ------------------------------------------------------------------
    #[test]
    fn test_is_eml_backed_true() {
        let op = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = ag::eml_scalar_op(Arc::clone(&op), &[x], g);
            assert!(
                ag::is_eml_backed(&y),
                "eml_scalar_op tensor must be EML-backed"
            );
        });
    }

    // ------------------------------------------------------------------
    // Test 8: is_eml_backed returns false for regular tensor
    // ------------------------------------------------------------------
    #[test]
    fn test_is_eml_backed_false() {
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            // A plain autograd multiplication — not EML-backed
            let y = x * x;
            assert!(
                !ag::is_eml_backed(&y),
                "regular autograd tensor must not be EML-backed"
            );
        });
    }

    // ------------------------------------------------------------------
    // Test 9: extract_lowered_op returns Some for EML tensor
    // ------------------------------------------------------------------
    #[test]
    fn test_extract_lowered_op_some() {
        let op = Arc::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0))));
        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = ag::eml_scalar_op(Arc::clone(&op), &[x], g);
            let extracted = ag::extract_lowered_op(&y);
            assert!(
                extracted.is_some(),
                "extract_lowered_op must return Some for EML-backed tensor"
            );
        });
    }

    // ------------------------------------------------------------------
    // Test 10: try_build_symbolic_jacobian — 2 EML outputs × 2 inputs → 2×2 correct
    // ------------------------------------------------------------------
    #[test]
    fn test_try_build_symbolic_jacobian_2x2() {
        // f0 = x^2, f1 = x * y  at (x=2, y=3)
        // J = [[df0/dx, df0/dy], [df1/dx, df1/dy]]
        //   = [[2x,     0     ], [y,      x     ]]
        //   = [[4,      0     ], [3,      2     ]]
        let f0_op = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        let f1_op = Arc::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        ));

        ag::run(|g: &mut ag::Context<f64>| {
            let x = g.placeholder("x", &[]);
            let y = g.placeholder("y", &[]);

            // Build two EML-backed scalar tensors
            let out0 = ag::eml_scalar_op(Arc::clone(&f0_op), &[x, y], g);
            let out1 = ag::eml_scalar_op(Arc::clone(&f1_op), &[x, y], g);

            let jac_opt = ag::try_build_symbolic_jacobian(&[out0, out1], &[x, y], g);
            assert!(
                jac_opt.is_some(),
                "should return Some when all outputs are EML-backed"
            );

            let jac = jac_opt.expect("jacobian tensor");
            let x_arr = scirs2_core::ndarray::arr0(2.0_f64).into_dyn();
            let y_arr = scirs2_core::ndarray::arr0(3.0_f64).into_dyn();
            let result = g
                .evaluator()
                .push(&jac)
                .feed(x, x_arr.view())
                .feed(y, y_arr.view())
                .run();
            let arr = result[0].clone().expect("jac eval ok");
            let vals: Vec<f64> = arr.iter().copied().collect();
            // [[4, 0], [3, 2]]
            assert_eq!(vals.len(), 4);
            assert!((vals[0] - 4.0).abs() < TOL, "J[0,0]=2x={}", vals[0]);
            assert!(vals[1].abs() < TOL, "J[0,1]=0={}", vals[1]);
            assert!((vals[2] - 3.0).abs() < TOL, "J[1,0]=y={}", vals[2]);
            assert!((vals[3] - 2.0).abs() < TOL, "J[1,1]=x={}", vals[3]);
        });
    }
}
