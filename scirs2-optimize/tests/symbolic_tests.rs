//! Integration tests for symbolic-gradient optimizers.
//!
//! All tests require the `symbolic` feature.

#[cfg(feature = "symbolic")]
mod symbolic {
    use std::sync::Arc;

    use scirs2_core::ndarray::array;
    use scirs2_optimize::symbolic::{lbfgs_symbolic, trust_region_symbolic, SymbolicOptError};
    use scirs2_symbolic::eml::LoweredOp;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }
    fn sq(op: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Box::new(op.clone()), Box::new(op))
    }
    fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Sub(Box::new(a), Box::new(b))
    }
    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Box::new(a), Box::new(b))
    }
    fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Box::new(a), Box::new(b))
    }

    // ── L-BFGS tests ─────────────────────────────────────────────────────────

    /// f(x) = x² → minimum at x = 0.
    #[test]
    fn test_lbfgs_x_squared_1d() {
        let obj = Arc::new(sq(var(0)));
        let result = lbfgs_symbolic(&obj, array![3.0].view(), 200, 1e-8, 10).expect("converge");
        assert!(result.converged, "L-BFGS on x² did not converge");
        assert!(
            result.x[0].abs() < 1e-6,
            "x = {}, expected ≈ 0",
            result.x[0]
        );
    }

    /// Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)² → minimum at (1,1).
    #[test]
    fn test_lbfgs_rosenbrock() {
        // f = (1-x)^2 + 100*(y - x^2)^2
        let one_minus_x = sub(c(1.0), var(0)); // 1 - x
        let y_minus_x2 = sub(var(1), sq(var(0))); // y - x²
        let term1 = sq(one_minus_x); // (1-x)²
        let term2 = mul(c(100.0), sq(y_minus_x2)); // 100(y-x²)²
        let obj = Arc::new(add(term1, term2));

        let result =
            lbfgs_symbolic(&obj, array![0.0, 0.0].view(), 500, 1e-4, 20).expect("converge");
        assert!(result.converged, "L-BFGS on Rosenbrock did not converge");
        assert!(
            (result.x[0] - 1.0).abs() < 1e-2,
            "x = {}, expected ≈ 1",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-2,
            "y = {}, expected ≈ 1",
            result.x[1]
        );
    }

    /// f(x,y) = x² + y² → minimum at (0, 0).
    #[test]
    fn test_lbfgs_xy_squared() {
        let obj = Arc::new(add(sq(var(0)), sq(var(1))));
        let result =
            lbfgs_symbolic(&obj, array![3.0, 4.0].view(), 200, 1e-8, 10).expect("converge");
        assert!(result.converged);
        assert!(
            result.x[0].abs() < 1e-6,
            "x = {}, expected ≈ 0",
            result.x[0]
        );
        assert!(
            result.x[1].abs() < 1e-6,
            "y = {}, expected ≈ 0",
            result.x[1]
        );
    }

    /// 1-D objective supplied with 2-D x0 → DimMismatch.
    #[test]
    fn test_lbfgs_dim_mismatch() {
        let obj = Arc::new(sq(var(0))); // 1 variable
        let err = lbfgs_symbolic(&obj, array![1.0, 2.0].view(), 10, 1e-8, 5);
        assert!(
            matches!(
                err,
                Err(SymbolicOptError::DimMismatch {
                    expected: 1,
                    got: 2
                })
            ),
            "expected DimMismatch, got {:?}",
            err
        );
    }

    /// max_iter = 0 → NotConverged immediately.
    #[test]
    fn test_lbfgs_max_iter_returns_not_converged() {
        let obj = Arc::new(sq(var(0)));
        let err = lbfgs_symbolic(&obj, array![5.0].view(), 0, 1e-8, 10);
        assert!(
            matches!(err, Err(SymbolicOptError::NotConverged { iters: 0, .. })),
            "expected NotConverged{{iters:0}}, got {:?}",
            err
        );
    }

    // ── Trust-region tests ────────────────────────────────────────────────────

    /// f(x) = x² → minimum at x = 0.
    #[test]
    fn test_trust_region_x_squared_1d() {
        let obj = Arc::new(sq(var(0)));
        let result =
            trust_region_symbolic(&obj, array![3.0].view(), 200, 1e-8, 1.0).expect("converge");
        assert!(result.converged, "TR on x² did not converge");
        assert!(
            result.x[0].abs() < 1e-6,
            "x = {}, expected ≈ 0",
            result.x[0]
        );
    }

    /// f(x,y) = x² + y² → minimum at (0, 0).
    #[test]
    fn test_trust_region_xy_squared() {
        let obj = Arc::new(add(sq(var(0)), sq(var(1))));
        let result =
            trust_region_symbolic(&obj, array![3.0, 4.0].view(), 200, 1e-8, 1.0).expect("converge");
        assert!(result.converged);
        assert!(
            result.x[0].abs() < 1e-6,
            "x = {}, expected ≈ 0",
            result.x[0]
        );
        assert!(
            result.x[1].abs() < 1e-6,
            "y = {}, expected ≈ 0",
            result.x[1]
        );
    }

    /// f(x) = x³ - 3x has local minima at x = 1 (f = -2) and local max at x = -1.
    /// Starting from x0 = 0.5 (between 0 and 1), TR should converge to x = 1.
    #[test]
    fn test_trust_region_cubic() {
        // f(x) = x³ - 3x  →  f'(x) = 3x² - 3  →  critical points at x = ±1
        // f''(x) = 6x: at x=1, f''=6 > 0 (local min).  at x=-1, f''=-6 < 0 (local max).
        let x3 = LoweredOp::Pow(Box::new(var(0)), Box::new(c(3.0)));
        let three_x = mul(c(3.0), var(0));
        let obj = Arc::new(sub(x3, three_x));

        let result =
            trust_region_symbolic(&obj, array![0.5].view(), 300, 1e-4, 0.5).expect("converge");
        assert!(result.converged, "TR on cubic did not converge");
        assert!(
            (result.x[0] - 1.0).abs() < 1e-3,
            "x = {}, expected ≈ 1",
            result.x[0]
        );
    }

    // ── KKT / Lagrangian tests ────────────────────────────────────────────────

    /// Build a KKT system for min x² s.t. x - 3 = 0 and verify structure.
    #[test]
    fn test_kkt_build_1d() {
        use scirs2_optimize::{build_kkt, KktSystem};

        // min x^2 s.t. x - 3 = 0 → x = 3
        let f = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        let g = Arc::new(LoweredOp::Sub(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(3.0)),
        ));
        let kkt: KktSystem = build_kkt(&f, &[g], 1).expect("kkt build");
        assert_eq!(kkt.n_vars, 1);
        assert_eq!(kkt.n_constraints, 1);
        assert_eq!(kkt.stationarity.len(), 1);
    }

    /// Solve min x² s.t. x - 3 = 0 via Lagrangian Newton; expect x ≈ 3.
    #[test]
    fn test_solve_lagrangian_1d() {
        use scirs2_optimize::solve_lagrangian_symbolic;

        let f = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        let g = Arc::new(LoweredOp::Sub(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(3.0)),
        ));
        let x0 = array![0.0f64];
        let lam0 = array![0.0f64];
        let result = solve_lagrangian_symbolic(&f, &[g], x0.view(), lam0.view(), 50, 1e-8)
            .expect("should converge");
        assert!(
            (result.x[0] - 3.0).abs() < 1e-4,
            "x = {} (expected 3)",
            result.x[0]
        );
    }

    /// Solve min x² + y² s.t. x + y = 1; solution is x = y = 0.5.
    #[test]
    fn test_solve_lagrangian_xy_on_line() {
        use scirs2_optimize::solve_lagrangian_symbolic;

        let x2 = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(2.0)));
        let y2 = LoweredOp::Pow(Box::new(LoweredOp::Var(1)), Box::new(LoweredOp::Const(2.0)));
        let f = Arc::new(LoweredOp::Add(Box::new(x2), Box::new(y2)));
        let sum = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let g = Arc::new(LoweredOp::Sub(
            Box::new(sum),
            Box::new(LoweredOp::Const(1.0)),
        ));
        let x0 = array![0.7f64, 0.3f64];
        let lam0 = array![0.0f64];
        let result = solve_lagrangian_symbolic(&f, &[g], x0.view(), lam0.view(), 100, 1e-8)
            .expect("should converge");
        assert!((result.x[0] - 0.5).abs() < 1e-3, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 0.5).abs() < 1e-3, "x[1] = {}", result.x[1]);
    }

    /// max_iter = 0 must return NotConverged immediately.
    #[test]
    fn test_solve_lagrangian_max_iter_zero() {
        use scirs2_optimize::solve_lagrangian_symbolic;

        let f = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        let g = Arc::new(LoweredOp::Sub(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(1.0)),
        ));
        let x0 = array![0.0f64];
        let lam0 = array![0.0f64];
        let result = solve_lagrangian_symbolic(&f, &[g], x0.view(), lam0.view(), 0, 1e-8);
        assert!(
            matches!(result, Err(SymbolicOptError::NotConverged { .. })),
            "expected NotConverged, got {:?}",
            result
        );
    }

    /// Verify KktSystem dimensions for n=2 primal vars, m=2 constraints.
    #[test]
    fn test_build_kkt_dim() {
        use scirs2_optimize::build_kkt;

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
        let g1 = Arc::new(LoweredOp::Sub(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(1.0)),
        ));
        let g2 = Arc::new(LoweredOp::Sub(
            Box::new(LoweredOp::Var(1)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        let kkt = build_kkt(&f, &[g1, g2], 2).expect("kkt build");
        assert_eq!(kkt.n_vars, 2);
        assert_eq!(kkt.n_constraints, 2);
        assert_eq!(kkt.stationarity.len(), 2);
        assert_eq!(kkt.constraint_residuals.len(), 2);
    }

    /// x0 has wrong length → DimMismatch.
    #[test]
    fn test_solve_lagrangian_dim_mismatch() {
        use scirs2_optimize::solve_lagrangian_symbolic;

        let f = Arc::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        ));
        let g = Arc::new(LoweredOp::Sub(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(1.0)),
        ));
        // x0 has 2 elements but objective only references Var(0) → n_vars = 1 → DimMismatch
        let x0 = array![0.0f64, 0.0f64];
        let lam0 = array![0.0f64];
        let result = solve_lagrangian_symbolic(&f, &[g], x0.view(), lam0.view(), 10, 1e-8);
        assert!(
            matches!(result, Err(SymbolicOptError::DimMismatch { .. })),
            "expected DimMismatch, got {:?}",
            result
        );
    }
}
