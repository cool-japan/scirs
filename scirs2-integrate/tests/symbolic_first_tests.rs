//! Integration tests for `scirs2_integrate::symbolic_first`.
//!
//! All tests require the `symbolic` feature (gated at the module level).

#[cfg(feature = "symbolic")]
mod tests {
    use scirs2_integrate::symbolic_first::{
        rhs_from_symbolic_only, solve_ode_symbolic_or_numerical, OdeOpts, SolvePreference,
        SymbolicOrNumericalResult,
    };
    use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};

    // -----------------------------------------------------------------------
    // Helper: evaluate a LoweredOp at a set of (var_id, value) bindings.
    // Builds a positional `Vec<f64>` sized to the maximum var id present.
    // -----------------------------------------------------------------------
    fn eval_at(op: &LoweredOp, vars: &[(usize, f64)]) -> f64 {
        let max_id = vars.iter().map(|(id, _)| *id).max().unwrap_or(0);
        let mut bindings = vec![0.0_f64; max_id + 1];
        for (id, val) in vars {
            bindings[*id] = *val;
        }
        let ctx = EvalCtx::new(&bindings);
        eval_real(op, &ctx).unwrap_or(f64::NAN)
    }

    // -----------------------------------------------------------------------
    // Test 1: dx/dt = x, x(0) = 1 → Symbolic branch → x(t) = exp(t)
    //
    // `try_linear_1st_order` recognises rhs = Var(x_var) as a=1, f(t)=0.
    // With IC (0, 1) the constant is determined to 1, yielding exp(t).
    // We accept Numerical as fallback (the important invariant is: if Symbolic
    // is returned, it is numerically correct).
    // -----------------------------------------------------------------------
    #[test]
    fn test_linear_exponential_symbolic_or_numerical() {
        let x_var: usize = 0;
        let t_var: usize = 1;

        // rhs = x  (dx/dt = x)
        let rhs_sym = LoweredOp::Var(x_var);
        let rhs_num = |_t: f64, x: f64| x;

        let opts = OdeOpts::default();
        let result = solve_ode_symbolic_or_numerical(
            Some(&rhs_sym),
            rhs_num,
            x_var,
            t_var,
            (0.0, 1.0),
            2.0,
            &opts,
        )
        .expect("solver should not return Err");

        match result {
            SymbolicOrNumericalResult::Symbolic { x_of_t, .. } => {
                // x(1) should be exp(1) ≈ 2.71828…
                let val = eval_at(&x_of_t, &[(t_var, 1.0)]);
                assert!(
                    (val - std::f64::consts::E).abs() < 1e-6,
                    "Symbolic x(1) expected exp(1)≈{:.6}, got {val:.6}",
                    std::f64::consts::E
                );
                // x(2) should be exp(2) ≈ 7.38906…
                let val2 = eval_at(&x_of_t, &[(t_var, 2.0)]);
                assert!(
                    (val2 - std::f64::consts::E.powi(2)).abs() < 1e-4,
                    "Symbolic x(2) expected exp(2)≈{:.4}, got {val2:.4}",
                    std::f64::consts::E.powi(2)
                );
            }
            SymbolicOrNumericalResult::Numerical { trajectory, .. } => {
                // Fallback is acceptable — verify the numerical answer is close to exp(t)
                // The trajectory has n_steps+1 rows; the last row is at t=2.
                let n = trajectory.nrows();
                let x_end = trajectory[[n - 1, 1]];
                assert!(
                    (x_end - std::f64::consts::E.powi(2)).abs() < 1e-3,
                    "Numerical fallback x(2) ≈ exp(2), got {x_end:.6}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: ForceNumerical bypasses the symbolic path even when a symbolic
    // RHS is supplied.
    // -----------------------------------------------------------------------
    #[test]
    fn test_force_numerical_bypasses_symbolic() {
        let x_var: usize = 0;
        let t_var: usize = 1;
        let rhs_sym = LoweredOp::Var(x_var);
        let rhs_num = |_t: f64, x: f64| x;

        let opts = OdeOpts {
            preferred: SolvePreference::ForceNumerical,
            max_steps: 100,
            ..Default::default()
        };
        let result = solve_ode_symbolic_or_numerical(
            Some(&rhs_sym),
            rhs_num,
            x_var,
            t_var,
            (0.0, 1.0),
            1.0,
            &opts,
        )
        .expect("solver should succeed");

        assert!(
            matches!(result, SymbolicOrNumericalResult::Numerical { .. }),
            "ForceNumerical must return Numerical variant"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: No symbolic RHS supplied → always falls back to numerical.
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_symbolic_rhs_falls_back() {
        let x_var: usize = 0;
        let t_var: usize = 1;
        let rhs_num = |t: f64, x: f64| -x + t.sin();

        let opts = OdeOpts::default();
        let result =
            solve_ode_symbolic_or_numerical(None, rhs_num, x_var, t_var, (0.0, 0.0), 1.0, &opts)
                .expect("solver should succeed");

        assert!(
            matches!(result, SymbolicOrNumericalResult::Numerical { .. }),
            "None rhs_symbolic must give Numerical"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: Invalid interval (t_end <= t0) → Err(InvalidInterval).
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalid_interval_returns_error() {
        let x_var: usize = 0;
        let t_var: usize = 1;
        let rhs_num = |_t: f64, x: f64| x;

        let opts = OdeOpts::default();

        // t_end == t0 → invalid
        let result =
            solve_ode_symbolic_or_numerical(None, rhs_num, x_var, t_var, (1.0, 1.0), 1.0, &opts);
        assert!(result.is_err(), "t_end == t0 should be an error");

        // t_end < t0 → invalid
        let result2 =
            solve_ode_symbolic_or_numerical(None, rhs_num, x_var, t_var, (2.0, 1.0), 0.5, &opts);
        assert!(result2.is_err(), "t_end < t0 should be an error");
    }

    // -----------------------------------------------------------------------
    // Test 5: Numerical trajectory has correct shape.
    //
    // With max_steps = 50 and ForceNumerical, trajectory must be (51, 2)
    // and time must have length 51.
    // -----------------------------------------------------------------------
    #[test]
    fn test_numerical_trajectory_shape() {
        let x_var: usize = 0;
        let t_var: usize = 1;
        let rhs_num = |_t: f64, x: f64| -x;

        let opts = OdeOpts {
            max_steps: 50,
            preferred: SolvePreference::ForceNumerical,
            ..Default::default()
        };
        let result =
            solve_ode_symbolic_or_numerical(None, rhs_num, x_var, t_var, (0.0, 1.0), 1.0, &opts)
                .expect("should succeed");

        match result {
            SymbolicOrNumericalResult::Numerical { trajectory, time } => {
                assert_eq!(
                    trajectory.nrows(),
                    51,
                    "trajectory rows should be n_steps+1=51"
                );
                assert_eq!(
                    trajectory.ncols(),
                    2,
                    "trajectory must have 2 columns [t, x]"
                );
                assert_eq!(time.len(), 51, "time length should be 51");
                // First time point is t0
                assert!((time[0] - 0.0).abs() < 1e-15, "time[0] should be t0=0");
                // Last time point is t_end
                assert!((time[50] - 1.0).abs() < 1e-10, "time[50] should be t_end=1");
            }
            _ => panic!("Expected Numerical result"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: rhs_from_symbolic_only evaluates correctly.
    //
    // `rhs = Var(x_var)` encodes `dx/dt = x`.  The derived numeric closure
    // should return `x` for any `(t, x)` pair.
    // -----------------------------------------------------------------------
    #[test]
    fn test_rhs_from_symbolic_only() {
        let x_var: usize = 0;
        let t_var: usize = 1;

        // sym_rhs = Var(x_var) → evaluates to x
        let sym_rhs = LoweredOp::Var(x_var);
        let f = rhs_from_symbolic_only(sym_rhs, x_var, t_var);

        let val_a = f(0.0, 3.0);
        assert!(
            (val_a - 3.0).abs() < 1e-12,
            "f(0, 3) expected 3.0, got {val_a}"
        );

        let val_b = f(5.0, 2.0);
        assert!(
            (val_b - 2.0).abs() < 1e-12,
            "f(5, 2) expected 2.0, got {val_b}"
        );

        let val_c = f(100.0, -7.5);
        assert!(
            (val_c - (-7.5)).abs() < 1e-12,
            "f(100, -7.5) expected -7.5, got {val_c}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: Numerical accuracy for dx/dt = -x, x(0) = 1.
    //
    // Exact solution: x(t) = exp(-t).  With 10 000 steps over [0, 5] the
    // fixed-step RK4 error should be < 1e-6.
    // -----------------------------------------------------------------------
    #[test]
    fn test_numerical_accuracy_decay() {
        let x_var: usize = 0;
        let t_var: usize = 1;
        let rhs_num = |_t: f64, x: f64| -x;

        let opts = OdeOpts {
            max_steps: 10_000,
            preferred: SolvePreference::ForceNumerical,
            ..Default::default()
        };
        let result =
            solve_ode_symbolic_or_numerical(None, rhs_num, x_var, t_var, (0.0, 1.0), 5.0, &opts)
                .expect("should succeed");

        match result {
            SymbolicOrNumericalResult::Numerical { trajectory, .. } => {
                let n = trajectory.nrows();
                let x_end = trajectory[[n - 1, 1]];
                let expected = (-5.0_f64).exp();
                assert!(
                    (x_end - expected).abs() < 1e-6,
                    "x(5) ≈ exp(-5)={expected:.8}, got {x_end:.8}"
                );
            }
            _ => panic!("Expected Numerical result"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: OdeOpts default values are sensible.
    // -----------------------------------------------------------------------
    #[test]
    fn test_opts_defaults() {
        let opts = OdeOpts::default();
        assert_eq!(opts.max_steps, 10_000);
        assert_eq!(opts.preferred, SolvePreference::SymbolicFirst);
        assert!(opts.rtol < 1e-4);
        assert!(opts.atol < 1e-6);
    }
}
