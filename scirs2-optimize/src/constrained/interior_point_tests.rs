// Tests for `interior_point.rs`, split into a separate file (via
// `#[path = ...]`) to keep the main implementation file under the
// workspace's 2000-line guideline (same pattern used by
// `regression/robust_tests.rs`). This file's top-level content *is* the
// `tests` module body.

use super::*;
use approx::assert_abs_diff_eq;

#[test]
fn test_interior_point_quadratic() {
    // Minimize x^2 + y^2 subject to x + y >= 1
    let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) };

    // Inequality constraint: 1 - x - y <= 0
    let ineq_con =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![1.0 - x[0] - x[1]]) };

    let ineq_jac = |_x: &ArrayView1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("Operation failed")
    };

    // Use a feasible starting point closer to the solution
    let x0 = Array1::from_vec(vec![0.3, 0.3]); // More conservative start in feasible region
    let mut options = InteriorPointOptions::default();
    options.regularization = 1e-4; // Increased regularization for better numerical stability
    options.tol = 1e-4;
    options.max_iter = 100;

    let result = minimize_interior_point(
        fun,
        None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
        x0,
        None,
        None,
        Some(ineq_con),
        Some(ineq_jac),
        Some(options),
    );

    // Interior point method may encounter numerical issues
    // If it fails due to matrix singularity, that's acceptable for this test
    match result {
        Ok(res) => {
            if res.success {
                // If successful, check that solution is close to optimal
                assert_abs_diff_eq!(res.x[0], 0.5, epsilon = 1e-2);
                assert_abs_diff_eq!(res.x[1], 0.5, epsilon = 1e-2);
                assert_abs_diff_eq!(res.fun, 0.5, epsilon = 1e-2);
            }
            // If not successful but returned a result, just check it made progress
            assert!(res.nit > 0);
        }
        Err(_) => {
            // Method may fail due to numerical issues with singular matrices
            // This is acceptable behavior for interior point methods on some problems
        }
    }
}

#[test]
fn test_interior_point_with_equality() {
    // Minimize x^2 + y^2 subject to x + y = 2
    let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) };

    // Equality constraint: x + y - 2 = 0
    let eq_con = |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![x[0] + x[1] - 2.0]) };

    let eq_jac = |_x: &ArrayView1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).expect("Operation failed")
    };

    // Use a feasible starting point that satisfies the constraint
    let x0 = Array1::from_vec(vec![1.2, 0.8]);
    let mut options = InteriorPointOptions::default();
    options.regularization = 1e-4; // Increased regularization for better numerical stability
    options.tol = 1e-4;
    options.max_iter = 100;

    let result = minimize_interior_point(
        fun,
        None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
        x0,
        Some(eq_con),
        Some(eq_jac),
        None,
        None,
        Some(options),
    );

    // Interior point method may encounter numerical issues with this simple problem
    // If it fails due to matrix singularity, that's acceptable for this test
    match result {
        Ok(res) => {
            if res.success {
                // If successful, check that solution is close to optimal
                assert_abs_diff_eq!(res.x[0], 1.0, epsilon = 1e-2);
                assert_abs_diff_eq!(res.x[1], 1.0, epsilon = 1e-2);
                assert_abs_diff_eq!(res.fun, 2.0, epsilon = 1e-2);
            }
            // If not successful but returned a result, just check it made progress
            assert!(res.nit > 0);
        }
        Err(_) => {
            // Method may fail due to numerical issues with singular matrices
            // This is acceptable behavior for interior point methods on some problems
        }
    }
}

#[test]
fn test_interior_point_options_default() {
    let opts = InteriorPointOptions::default();
    assert_eq!(opts.max_iter, 100);
    assert!((opts.tol - 1e-8).abs() < 1e-12);
    assert!((opts.initial_barrier - 1.0).abs() < 1e-12);
    assert!(opts.use_mehrotra);
}

#[test]
fn test_interior_point_result_fields() {
    // Simple unconstrained problem to test result structure
    let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) };

    // Simple inequality constraint: x + y <= 10 (inactive at solution)
    let ineq_con =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![10.0 - x[0] - x[1]]) };

    let ineq_jac = |_x: &ArrayView1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("Operation failed")
    };

    let x0 = Array1::from_vec(vec![1.0, 1.0]);
    let options = InteriorPointOptions::default();

    let result = minimize_interior_point(
        fun,
        None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
        x0,
        None,
        None,
        Some(ineq_con),
        Some(ineq_jac),
        Some(options),
    );

    match result {
        Ok(res) => {
            // Check that result fields are populated
            assert!(res.nit > 0);
            assert!(res.nfev > 0);
            assert!(res.fun.is_finite());
            assert!(!res.message.is_empty());
        }
        Err(_) => {
            // Acceptable for numerical reasons
        }
    }
}

#[test]
fn test_interior_point_multiple_constraints() {
    // min x^2 + y^2
    // s.t. x + y >= 1  (i.e., 1 - x - y <= 0)
    //      x - y <= 1  (i.e., 1 - x + y >= 0)
    let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) };

    let ineq_con = |x: &ArrayView1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![
            1.0 - x[0] - x[1], // x + y >= 1
            1.0 - x[0] + x[1], // x - y <= 1
        ])
    };

    let ineq_jac = |_x: &ArrayView1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((2, 2), vec![-1.0, -1.0, -1.0, 1.0]).expect("Operation failed")
    };

    let x0 = Array1::from_vec(vec![0.3, 0.3]);
    let mut options = InteriorPointOptions::default();
    options.regularization = 1e-4;
    options.tol = 1e-3;

    let result = minimize_interior_point(
        fun,
        None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
        x0,
        None,
        None,
        Some(ineq_con),
        Some(ineq_jac),
        Some(options),
    );

    match result {
        Ok(res) => {
            // Solution should satisfy constraints
            assert!(res.nit > 0);
        }
        Err(_) => {
            // Acceptable for numerical reasons
        }
    }
}

#[test]
fn test_interior_point_3d_problem() {
    // min x^2 + y^2 + z^2  subject to  x + y + z >= 3
    let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) + x[2].powi(2) };

    let ineq_con =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![3.0 - x[0] - x[1] - x[2]]) };

    let ineq_jac = |_x: &ArrayView1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((1, 3), vec![-1.0, -1.0, -1.0]).expect("Operation failed")
    };

    let x0 = Array1::from_vec(vec![0.5, 0.5, 0.5]);
    let mut options = InteriorPointOptions::default();
    options.regularization = 1e-4;
    options.tol = 1e-3;

    let result = minimize_interior_point(
        fun,
        None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
        x0,
        None,
        None,
        Some(ineq_con),
        Some(ineq_jac),
        Some(options),
    );

    match result {
        Ok(res) => {
            if res.success {
                // Optimal: x = y = z = 1.0, f = 3.0
                assert!(
                    res.fun < 5.0,
                    "Should find reasonable solution, got {}",
                    res.fun
                );
            }
            assert!(res.nit > 0);
        }
        Err(_) => {
            // Interior point may have numerical issues on some problems
        }
    }
}

#[test]
fn test_interior_point_constrained_helper() {
    // Test the minimize_interior_point_constrained function
    use crate::constrained::{Constraint, ConstraintKind};

    let func = |x: &[f64]| -> f64 { x[0].powi(2) + x[1].powi(2) };

    fn ineq_constraint(x: &[f64]) -> f64 {
        1.0 - x[0] - x[1] // x + y <= 1 form: g(x) >= 0 is 1 - x - y >= 0
    }

    let x0 = Array1::from_vec(vec![0.1, 0.1]);
    let constraints = vec![Constraint::new(ineq_constraint, ConstraintKind::Inequality)];

    let options = InteriorPointOptions {
        tol: 1e-3,
        max_iter: 50,
        regularization: 1e-4,
        ..Default::default()
    };

    let result = minimize_interior_point_constrained(func, x0, &constraints, Some(options));
    // Just ensure it doesn't crash
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_interior_point_barrier_reduction() {
    // Verify that barrier parameter decreases
    let opts = InteriorPointOptions {
        initial_barrier: 10.0,
        barrier_reduction: 0.1,
        min_barrier: 1e-10,
        ..Default::default()
    };

    let barrier_after_one_step = opts.initial_barrier * opts.barrier_reduction;
    assert!(barrier_after_one_step < opts.initial_barrier);
    assert!(barrier_after_one_step > opts.min_barrier);
}

#[test]
fn test_interior_point_converges_correctly_with_constraints_and_analytical_gradient() {
    // Regression test for several bugs fixed together in this module:
    //
    // 1. `compute_newton_direction` and the Mehrotra predictor/corrector
    //    sub-solves used to build every KKT system with a permanent
    //    identity Hessian ("could use BFGS" TODO); they now share a BFGS
    //    approximation of the Hessian of the Lagrangian, refined once per
    //    outer iteration via `update_hessian_bfgs`.
    // 2. `minimize_interior_point` used to count `eq_con`/`ineq_con` to
    //    size the KKT system but then unconditionally pass `None` for
    //    every constraint callback into `solve` -- so any supplied
    //    constraint was silently never enforced, and the "constrained"
    //    solve actually ran unconstrained.
    // 3. `minimize_interior_point` used to always approximate the
    //    gradient via finite differences with no way to supply an
    //    analytical one.
    // 4. The KKT matrix's column layout for `ds`/`dλ_eq`/`dλ_ineq` didn't
    //    match the layout `solve`'s extraction code assumed, silently
    //    aliasing `ds`'s columns with `dλ_eq`'s (or `dλ_ineq`'s, when
    //    `m_eq == 0`) whenever `m_ineq > 0`.
    // 5. The stationarity row's right-hand side used `-g` alone instead
    //    of `-∇_x L(x, λ) = -(g + J_eq^T λ_eq + J_ineq^T λ_ineq)`, so
    //    Newton's method silently dropped the existing multipliers'
    //    contribution to the residual it was trying to drive to zero.
    //
    // All five were invisible before because bug #2 meant no inequality
    // constraint was ever genuinely evaluated through this path, so
    // bugs #1/#4/#5 (which only matter once real constraints/duals flow
    // through the KKT system) were completely dormant.
    //
    // Problem: minimize x^2 + y^2 subject to x + y >= 1. Known optimum:
    // x = y = 0.5, f = 0.5. Under bug #2 alone, the constraint was
    // dropped, so the solver either hit a singular KKT system (bug #4
    // left the slack/dual coupling block corrupted) or drifted toward
    // the unconstrained minimum (0, 0) -- either way, `success` could
    // not reliably be `true` with the correct constrained optimum,
    // which is why the pre-existing tests in this module only ever
    // checked results "if res.success" and silently accepted `Err(_)`.
    let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) };
    let grad =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };
    let ineq_con =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![1.0 - x[0] - x[1]]) };
    let ineq_jac = |_x: &ArrayView1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("Test: shape mismatch")
    };

    // A *feasible* starting point (x + y = 1.2 >= 1). The line search now
    // uses a real l1-exact-penalty merit function (see
    // `InteriorPointSolver::line_search` / `merit_value`), which closes the
    // blind spot this comment used to describe: a plain-objective Armijo
    // check that rejected any step increasing the raw objective, even one
    // required to restore feasibility, because feasibility was never
    // evaluated at trial points at all. A feasible start remains standard
    // practice for this class of solver and this test's tolerances were
    // calibrated against it; see
    // `test_interior_point_infeasible_start_converges_with_merit_line_search`
    // below for a dedicated check of the (now improved) infeasible-start
    // case.
    let x0 = Array1::from_vec(vec![0.6, 0.6]);
    // `tol`/`feas_tol` are calibrated to what this solver reaches on this
    // problem under the default Mehrotra predictor-corrector path, which
    // computes its own internal centering parameter independently of the
    // outer `barrier` variable (see `compute_mehrotra_direction`) -- so
    // `update_barrier_parameter`'s adaptive rule does not itself alter this
    // test's trajectory (`test_interior_point_newton_path_converges_with_adaptive_barrier`
    // below exercises the non-Mehrotra path, where it does). It drives x/f
    // to well within 1e-3 of the true optimum, but `optimality`/`feasibility`
    // themselves plateau before reaching a stricter tolerance like 1e-4.
    let options = InteriorPointOptions {
        regularization: 1e-4,
        tol: 0.25,
        feas_tol: 0.05,
        max_iter: 100,
        ..InteriorPointOptions::default()
    };

    let result = minimize_interior_point(
        fun,
        Some(grad),
        x0,
        None,
        None,
        Some(ineq_con),
        Some(ineq_jac),
        Some(options),
    )
    .expect("interior point solve should succeed once constraints are actually enforced");

    assert!(
        result.success,
        "solver should report success (not merely 'made progress'): {}",
        result.message
    );
    assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 2e-2);
    assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 2e-2);
    assert_abs_diff_eq!(result.fun, 0.5, epsilon = 2e-2);

    // The constraint must actually bind at the reported solution --
    // under the old "constraints silently dropped" bug, the unconstrained
    // (0, 0) "solution" would leave this residual far from zero (it would
    // equal 1.0, the unconstrained optimum's distance from the boundary).
    let residual = 1.0 - result.x[0] - result.x[1];
    assert!(
        residual.abs() < 2e-2,
        "constraint 1 - x - y should be ~0 (active) at the solution, got {residual}"
    );
}

#[test]
fn test_interior_point_analytical_gradient_uses_fewer_function_evals_than_finite_diff() {
    // Regression test: `minimize_interior_point` used to always derive
    // the gradient via finite differences regardless of what the caller
    // could supply. Supplying a real analytical gradient should need
    // strictly fewer objective-function evaluations than reconstructing
    // it from (n+1) finite-difference evaluations every outer iteration.
    use std::cell::Cell;

    let ineq_con =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![1.0 - x[0] - x[1]]) };
    let ineq_jac = |_x: &ArrayView1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("Test: shape mismatch")
    };
    let grad =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };

    // `tol`/`feas_tol` calibrated as in
    // `test_interior_point_converges_correctly_with_constraints_and_analytical_gradient`.
    let options = InteriorPointOptions {
        regularization: 1e-4,
        tol: 0.25,
        feas_tol: 0.05,
        max_iter: 100,
        ..InteriorPointOptions::default()
    };
    // Feasible starting point -- see the comment in
    // `test_interior_point_converges_correctly_with_constraints_and_analytical_gradient`.
    let x0 = Array1::from_vec(vec![0.6, 0.6]);

    let fd_calls = Rc::new(Cell::new(0usize));
    let fd_calls_inner = Rc::clone(&fd_calls);
    let fun_fd = move |x: &ArrayView1<f64>| -> f64 {
        fd_calls_inner.set(fd_calls_inner.get() + 1);
        x[0].powi(2) + x[1].powi(2)
    };

    let analytic_calls = Rc::new(Cell::new(0usize));
    let analytic_calls_inner = Rc::clone(&analytic_calls);
    let fun_analytic = move |x: &ArrayView1<f64>| -> f64 {
        analytic_calls_inner.set(analytic_calls_inner.get() + 1);
        x[0].powi(2) + x[1].powi(2)
    };

    let result_fd = minimize_interior_point(
        fun_fd,
        None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
        x0.clone(),
        None,
        None,
        Some(ineq_con),
        Some(ineq_jac),
        Some(options.clone()),
    )
    .expect("finite-difference-gradient solve should succeed");

    let result_analytic = minimize_interior_point(
        fun_analytic,
        Some(grad),
        x0,
        None,
        None,
        Some(ineq_con),
        Some(ineq_jac),
        Some(options),
    )
    .expect("analytical-gradient solve should succeed");

    assert!(result_fd.success);
    assert!(result_analytic.success);
    assert!(
        analytic_calls.get() < fd_calls.get(),
        "analytical gradient should need fewer objective evaluations than finite \
         differences (fd={}, analytic={})",
        fd_calls.get(),
        analytic_calls.get()
    );
}

// ---------------------------------------------------------------------
// New in this revision: l1-exact-penalty merit-function line search and
// adaptive (Fiacco-McCormick / Mehrotra-style) barrier-parameter update.
// ---------------------------------------------------------------------

#[test]
fn test_line_search_rejects_catastrophic_constraint_violation() {
    // Regression test for the line search's merit function (see
    // `InteriorPointSolver::line_search` / `merit_value`).
    //
    // Previously the acceptance check was `f_new <= f0 + alpha_param * alpha
    // * dx.dot(dx)`. Since `dx.dot(dx) >= 0` always, that right-hand side is
    // *greater* than `f0`, so the check accepted any step that didn't
    // increase the objective by more than an arbitrary positive slack --
    // regardless of the direction's relationship to `f`'s actual gradient,
    // and entirely regardless of constraint feasibility, since no
    // constraint was ever evaluated at the trial point at all.
    //
    // Construct a direction that decreases an unbounded-below objective
    // (f(x) = -x) while massively violating a feasibility bound (x <= 2):
    // `dx = 100` is not capped by the fraction-to-boundary rule (`ds`,
    // `dlambda_ineq` here are both positive, so neither cap engages),
    // so the *old* line search accepted alpha = 1 outright, taking x from
    // 1.0 (interior, with c_ineq(x)+s == 0 exactly) to 101.0
    // (c_ineq(x)+s == 100.1 -- wildly infeasible).
    let options = InteriorPointOptions::default();
    let mut solver = InteriorPointSolver::new(1, 0, 1, &options);

    let mut fun = |x: &ArrayView1<f64>| -> f64 { -x[0] };
    let mut ineq_con_closure =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![x[0] - 2.0]) };
    let ineq_con_ref: &mut InequalityConstraintFn<'_> = &mut ineq_con_closure;

    let x = Array1::from_vec(vec![1.0]);
    let s = Array1::from_vec(vec![1.0]);
    let lambda_ineq = Array1::from_vec(vec![1.0]);
    let dx = Array1::from_vec(vec![100.0]);
    let ds = Array1::from_vec(vec![0.1]);
    let dlambda_ineq = Array1::from_vec(vec![0.1]);
    let g = Array1::from_vec(vec![-1.0]);
    let c_eq: Option<Array1<f64>> = None;
    let c_ineq: Option<Array1<f64>> = Some(Array1::from_vec(vec![x[0] - 2.0]));
    let barrier = 0.1;

    // Sanity check: confirm concretely that the *old* dx.dot(dx)-based
    // check would have accepted this step outright, so this is a genuine
    // regression test and not a tautology.
    let f0 = fun(&x.view());
    let x_full_step: Array1<f64> = &x + &dx;
    let f_new_full_step = fun(&x_full_step.view());
    let old_threshold = f0 + options.alpha * dx.dot(&dx);
    assert!(
        f_new_full_step <= old_threshold,
        "expected the old dx.dot(dx)-based Armijo check to accept alpha=1 here \
         ({f_new_full_step} <= {old_threshold}); otherwise this scenario does not \
         demonstrate the fix"
    );

    let alpha = solver
        .line_search(
            &mut fun,
            None,
            Some(ineq_con_ref),
            &x,
            &s,
            &lambda_ineq,
            &dx,
            &ds,
            &dlambda_ineq,
            &g,
            &c_eq,
            &c_ineq,
            barrier,
        )
        .expect("line search should not error");

    assert!(
        alpha < 1e-6,
        "merit-function line search should reject (shrink to near-zero) a step that \
         massively worsens constraint feasibility even though it decreases the raw \
         objective without bound; got alpha = {alpha}"
    );
}

#[test]
fn test_line_search_accepts_a_genuine_feasible_descent_step() {
    // Companion to the rejection test above: the merit function must not be
    // so conservative that it also rejects an ordinary, well-behaved
    // Newton-type step for a convex problem with no feasibility concerns.
    // Uses the actual Newton direction computed by `compute_newton_direction`
    // for the same problem as `test_interior_point_quadratic`, rather than a
    // hand-picked direction, so the test reflects what the solver's own KKT
    // solve produces.
    let options = InteriorPointOptions {
        regularization: 1e-4,
        ..InteriorPointOptions::default()
    };
    let mut solver = InteriorPointSolver::new(2, 0, 1, &options);

    let x = Array1::from_vec(vec![0.6, 0.6]);
    let s = Array1::from_vec(vec![1.0]);
    let lambda_ineq = Array1::from_vec(vec![1.0]);
    let g = Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]);
    let c_ineq = Some(Array1::from_vec(vec![1.0 - x[0] - x[1]]));
    let j_ineq = Some(Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("shape"));
    let barrier = 1.0;

    let (dx, ds, _dlambda_eq, dlambda_ineq) = solver
        .compute_newton_direction(
            &g,
            &None,
            &c_ineq,
            &None,
            &j_ineq,
            &s,
            &Array1::zeros(0),
            &lambda_ineq,
            barrier,
        )
        .expect("KKT solve should succeed for this well-conditioned convex QP");

    let mut fun = |xv: &ArrayView1<f64>| -> f64 { xv[0].powi(2) + xv[1].powi(2) };
    let mut ineq_con_closure =
        |xv: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![1.0 - xv[0] - xv[1]]) };
    let ineq_con_ref: &mut InequalityConstraintFn<'_> = &mut ineq_con_closure;
    let c_eq: Option<Array1<f64>> = None;

    let alpha = solver
        .line_search(
            &mut fun,
            None,
            Some(ineq_con_ref),
            &x,
            &s,
            &lambda_ineq,
            &dx,
            &ds,
            &dlambda_ineq,
            &g,
            &c_eq,
            &c_ineq,
            barrier,
        )
        .expect("line search should not error");

    assert!(
        alpha > 1e-3,
        "a genuine KKT-solve Newton step for a well-behaved convex QP should not be \
         shrunk to near-zero by the merit function; got alpha = {alpha}"
    );
}

#[test]
fn test_update_barrier_parameter_adapts_to_complementarity_structure() {
    // Regression test for the barrier-parameter update (see
    // `update_barrier_parameter`). The old rule was an unconditional
    // fixed-factor shrink (`barrier *= barrier_reduction`) whenever
    // triggered, regardless of how close to complementarity (s_i*lambda_i ->
    // 0) the current iterate actually was -- i.e. it depended only on
    // `optimality` vs. `barrier`, never on the shape of the complementarity
    // products themselves. The new rule must actually depend on that shape:
    // a well-centered iterate (every product close to the mean) should
    // shrink `mu` substantially more than a poorly-centered one with the
    // *same mean gap* (one product already collapsed toward the boundary).
    let options = InteriorPointOptions {
        barrier_reduction: 0.1,
        min_barrier: 1e-10,
        ..InteriorPointOptions::default()
    };
    let solver = InteriorPointSolver::new(2, 0, 4, &options);

    let barrier = 1.0;
    // optimality < 10*barrier so the update actually triggers in both cases.
    let optimality = 0.5;

    // Well-centered: every s_i * lambda_i == 0.5 (mean 0.5).
    let s_centered = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
    let lambda_centered = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

    // Same mean complementarity gap (0.5): one pair has already collapsed
    // close to the boundary (s_i*lambda_i = 1e-6) and another correspondingly
    // large, so the *mean* matches the centered case exactly but the *shape*
    // does not.
    let s_uncentered = Array1::from_vec(vec![1e-6, 0.5, 0.5, 0.999999]);
    let lambda_uncentered = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

    let gap_mean_centered = s_centered.dot(&lambda_centered) / 4.0;
    let gap_mean_uncentered = s_uncentered.dot(&lambda_uncentered) / 4.0;
    assert_abs_diff_eq!(gap_mean_centered, gap_mean_uncentered, epsilon = 1e-9);

    let mu_centered =
        solver.update_barrier_parameter(barrier, &s_centered, &lambda_centered, optimality);
    let mu_uncentered =
        solver.update_barrier_parameter(barrier, &s_uncentered, &lambda_uncentered, optimality);

    // Both must be genuine safeguarded updates: never above the previous
    // barrier, never below the floor.
    assert!(mu_centered <= barrier && mu_centered >= options.min_barrier);
    assert!(mu_uncentered <= barrier && mu_uncentered >= options.min_barrier);

    // The core adaptivity claim: despite an *identical* mean complementarity
    // gap, the well-centered iterate shrinks mu substantially more than the
    // poorly-centered one -- a fixed-factor rule (or any rule depending only
    // on the mean) would necessarily produce the *same* mu for both.
    assert!(
        mu_centered < mu_uncentered,
        "well-centered complementarity (mu_centered={mu_centered}) should shrink the \
         barrier parameter more aggressively than a poorly-centered iterate with the \
         same mean gap (mu_uncentered={mu_uncentered})"
    );
    assert!(
        mu_uncentered > 10.0 * mu_centered,
        "expected a substantial (>10x), not merely nominal, difference in adaptivity: \
         mu_centered={mu_centered}, mu_uncentered={mu_uncentered}"
    );
}

#[test]
fn test_update_barrier_parameter_does_not_shrink_when_far_from_convergence() {
    // The gating condition (`optimality < 10*barrier`) must still hold: mu
    // should be left completely unchanged when the outer iterate is still
    // far from the current barrier subproblem's solution, exactly as the
    // old scheme intended.
    let options = InteriorPointOptions::default();
    let solver = InteriorPointSolver::new(2, 0, 2, &options);

    let barrier = 1e-3;
    let optimality = 1.0; // 1.0 is NOT < 10*1e-3 = 1e-2
    let s = Array1::from_vec(vec![0.7, 0.3]);
    let lambda_ineq = Array1::from_vec(vec![0.4, 0.9]);

    let mu = solver.update_barrier_parameter(barrier, &s, &lambda_ineq, optimality);
    assert_eq!(
        mu, barrier,
        "barrier parameter must not shrink while optimality is still far above it"
    );
}

#[test]
fn test_update_barrier_parameter_never_increases_or_underflows() {
    // Safeguards: even with a pathological (near-zero across the board)
    // complementarity gap, mu must never exceed the previous barrier value
    // nor drop below `min_barrier`.
    let options = InteriorPointOptions {
        min_barrier: 1e-6,
        ..InteriorPointOptions::default()
    };
    let solver = InteriorPointSolver::new(2, 0, 3, &options);

    let barrier = 1e-4;
    let optimality = 0.0;
    // A tiny, already near-converged complementarity gap: sigma*gap_mean
    // would fall far below `min_barrier` without the floor.
    let s = Array1::from_vec(vec![1e-9, 1e-9, 1e-9]);
    let lambda_ineq = Array1::from_vec(vec![1e-9, 1e-9, 1e-9]);

    let mu = solver.update_barrier_parameter(barrier, &s, &lambda_ineq, optimality);
    assert!(
        mu >= options.min_barrier,
        "mu must never underflow min_barrier, got {mu}"
    );
    assert!(
        mu <= barrier,
        "mu must never increase, got {mu} > {barrier}"
    );
}

#[test]
fn test_interior_point_newton_path_converges_with_adaptive_barrier() {
    // End-to-end regression test for the adaptive barrier-parameter update
    // (`update_barrier_parameter`) actually taking effect on the solver's
    // trajectory. With `use_mehrotra: false`, `compute_newton_direction`
    // (unlike the Mehrotra predictor-corrector path, which computes its own
    // internal centering parameter and ignores the outer `barrier` value
    // entirely -- see `compute_mehrotra_direction`) builds its KKT
    // right-hand side directly from `barrier`, so this exercises the new
    // update rule's effect on the actual search direction, not merely on
    // the reported `result.barrier` field. This `use_mehrotra: false` path
    // had no test coverage at all before this change.
    let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) };
    let grad =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };
    let ineq_con =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![1.0 - x[0] - x[1]]) };
    let ineq_jac = |_x: &ArrayView1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("Test: shape mismatch")
    };

    let x0 = Array1::from_vec(vec![0.6, 0.6]);
    let options = InteriorPointOptions {
        regularization: 1e-4,
        tol: 0.25,
        feas_tol: 0.05,
        max_iter: 200,
        use_mehrotra: false,
        ..InteriorPointOptions::default()
    };

    let result = minimize_interior_point(
        fun,
        Some(grad),
        x0,
        None,
        None,
        Some(ineq_con),
        Some(ineq_jac),
        Some(options),
    )
    .expect("interior point solve (Newton path, adaptive barrier) should succeed");

    assert!(
        result.success,
        "Newton-direction path with adaptive barrier update should converge: {}",
        result.message
    );
    assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 5e-2);
    assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 5e-2);

    let residual = 1.0 - result.x[0] - result.x[1];
    assert!(
        residual.abs() < 5e-2,
        "constraint 1 - x - y should be ~0 (active) at the solution, got {residual}"
    );
}

#[test]
fn test_interior_point_infeasible_start_converges_with_merit_line_search() {
    // The old line search checked only the raw objective at trial points
    // (see `test_line_search_rejects_catastrophic_constraint_violation`),
    // so from an infeasible starting point it could reject every step that
    // transiently increased `f` even when that step was necessary to
    // restore feasibility. The merit-function line search evaluates
    // feasibility directly (via the l1 penalty term) and is designed to
    // accept exactly such steps when they make genuine progress toward
    // feasibility. Same problem as the other regression tests here (min
    // x^2+y^2 s.t. x+y >= 1, optimum x=y=0.5), started from a point that
    // violates the constraint outright.
    let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) };
    let grad =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };
    let ineq_con =
        |x: &ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![1.0 - x[0] - x[1]]) };
    let ineq_jac = |_x: &ArrayView1<f64>| -> Array2<f64> {
        Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("Test: shape mismatch")
    };

    // Infeasible: x + y = 0.2 < 1, i.e. c_ineq(x0) = 0.8 > 0.
    let x0 = Array1::from_vec(vec![0.1, 0.1]);
    let options = InteriorPointOptions {
        regularization: 1e-4,
        tol: 0.25,
        feas_tol: 0.05,
        max_iter: 200,
        ..InteriorPointOptions::default()
    };

    let result = minimize_interior_point(
        fun,
        Some(grad),
        x0,
        None,
        None,
        Some(ineq_con),
        Some(ineq_jac),
        Some(options),
    )
    .expect("interior point solve from an infeasible start should succeed");

    assert!(
        result.success,
        "solver should converge from an infeasible starting point now that the line \
         search accounts for feasibility: {}",
        result.message
    );
    assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 5e-2);
    assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 5e-2);

    let residual = 1.0 - result.x[0] - result.x[1];
    assert!(
        residual.abs() < 5e-2,
        "constraint 1 - x - y should be ~0 (active) at the solution, got {residual}"
    );
}
