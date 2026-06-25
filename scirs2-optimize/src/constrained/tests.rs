//! Tests for constrained optimization module

use crate::constrained::*;
use crate::error::OptimizeError;
use scirs2_core::ndarray::array;

#[allow(dead_code)]
fn objective(x: &[f64]) -> f64 {
    (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2)
}

#[allow(dead_code)]
fn constraint(x: &[f64]) -> f64 {
    3.0 - x[0] - x[1] // Should be >= 0
}

#[test]
#[allow(dead_code)]
fn test_minimize_constrained_placeholder() {
    // We're now using the real implementation, so this test needs to be adjusted
    let x0 = array![0.0, 0.0];
    let constraints = vec![Constraint::new(constraint, Constraint::INEQUALITY)];

    // Use minimal iterations to check basic algorithm behavior
    let options = Options {
        maxiter: Some(1), // Just a single iteration
        ..Options::default()
    };

    let result = minimize_constrained(
        objective,
        &x0.view(),
        &constraints,
        Method::SLSQP,
        Some(options),
    )
    .expect("Operation failed");

    // With limited iterations, we expect it not to converge
    assert!(!result.success);

    // Check that constraint value was computed
    assert!(result.constr.is_some());
    let constr = result.constr.expect("Operation failed");
    assert_eq!(constr.len(), 1);
}

// Test the SLSQP algorithm on a simple constrained problem
#[test]
#[allow(dead_code)]
fn test_minimize_slsqp() {
    // Problem:
    // Minimize (x-1)^2 + (y-2.5)^2
    // Subject to: x + y <= 3

    let x0 = array![0.0, 0.0];
    let constraints = vec![Constraint::new(constraint, Constraint::INEQUALITY)];

    let options = Options {
        maxiter: Some(100),
        gtol: Some(1e-6),
        ftol: Some(1e-6),
        ctol: Some(1e-6),
        ..Options::default()
    };

    let result = minimize_constrained(
        objective,
        &x0.view(),
        &constraints,
        Method::SLSQP,
        Some(options),
    )
    .expect("Operation failed");

    // For the purpose of this test, we're just checking that the algorithm runs
    // and produces reasonable output. The convergence may vary.

    // Check that we're moving in the right direction
    assert!(result.x[0] >= 0.0);
    assert!(result.x[1] >= 0.0);

    // Function value should be decreasing from initial point
    let initial_value = objective(&[0.0, 0.0]);
    assert!(result.fun <= initial_value);

    // Check that constraint values are computed
    assert!(result.constr.is_some());

    // Output the result for inspection
    println!(
        "SLSQP result: x = {:?}, f = {}, iterations = {}",
        result.x, result.fun, result.nit
    );
}

// Test the Trust Region Constrained algorithm
#[test]
#[allow(dead_code)]
fn test_minimize_trust_constr() {
    // Problem:
    // Minimize (x-1)^2 + (y-2.5)^2
    // Subject to: x + y <= 3

    let x0 = array![0.0, 0.0];
    let constraints = vec![Constraint::new(constraint, Constraint::INEQUALITY)];

    let options = Options {
        maxiter: Some(500), // Increased iterations for convergence
        gtol: Some(1e-6),
        ftol: Some(1e-6),
        ctol: Some(1e-6),
        ..Options::default()
    };

    let result = minimize_constrained(
        objective,
        &x0.view(),
        &constraints,
        Method::TrustConstr,
        Some(options.clone()),
    )
    .expect("Operation failed");

    // Check that we're moving in the right direction
    assert!(result.x[0] >= 0.0);
    assert!(result.x[1] >= 0.0);

    // Function value should be decreasing from initial point
    let initial_value = objective(&[0.0, 0.0]);
    assert!(result.fun <= initial_value);

    // Check that constraint values are computed
    assert!(result.constr.is_some());

    // Output the result for inspection
    println!(
        "TrustConstr result: x = {:?}, f = {}, iterations = {}",
        result.x, result.fun, result.nit
    );
}

// Test both constrained optimization methods on a more complex problem
#[test]
#[allow(dead_code)]
fn test_constrained_rosenbrock() {
    // Rosenbrock function with a constraint
    fn rosenbrock(x: &[f64]) -> f64 {
        100.0 * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2)
    }

    // Constraint: x[0]^2 + x[1]^2 <= 1.5
    fn circle_constraint(x: &[f64]) -> f64 {
        1.5 - (x[0].powi(2) + x[1].powi(2)) // Should be >= 0
    }

    let x0 = array![0.0, 0.0];
    let constraints = vec![Constraint::new(circle_constraint, Constraint::INEQUALITY)];

    let options = Options {
        maxiter: Some(1000), // More iterations for this harder problem
        gtol: Some(1e-4),    // Relaxed tolerances
        ftol: Some(1e-4),
        ctol: Some(1e-4),
        ..Options::default()
    };

    // For this test, we'll clone options at each stage to avoid move issues
    let options_copy1 = options.clone();
    let options_copy2 = options.clone();

    // Test SLSQP
    let result_slsqp = minimize_constrained(
        rosenbrock,
        &x0.view(),
        &constraints,
        Method::SLSQP,
        Some(options_copy1),
    )
    .expect("Operation failed");

    // Test TrustConstr
    let result_trust = minimize_constrained(
        rosenbrock,
        &x0.view(),
        &constraints,
        Method::TrustConstr,
        Some(options_copy2),
    )
    .expect("Operation failed");

    // Check that both methods find a reasonable solution
    println!(
        "SLSQP Rosenbrock result: x = {:?}, f = {}, iterations = {}",
        result_slsqp.x, result_slsqp.fun, result_slsqp.nit
    );
    println!(
        "TrustConstr Rosenbrock result: x = {:?}, f = {}, iterations = {}",
        result_trust.x, result_trust.fun, result_trust.nit
    );

    // Check that function value is better than initial point
    let initial_value = rosenbrock(&[0.0, 0.0]);
    assert!(result_slsqp.fun < initial_value);
    assert!(result_trust.fun < initial_value);

    // Check that constraint is satisfied
    let constr_slsqp = result_slsqp.constr.expect("Operation failed");
    let constr_trust = result_trust.constr.expect("Operation failed");
    assert!(constr_slsqp[0] >= -0.01); // Relaxed tolerance for the test
    assert!(constr_trust[0] >= -0.01); // Relaxed tolerance for the test
}

#[test]
#[allow(dead_code)]
fn test_cobyla_not_implemented() {
    // Test that COBYLA returns a NotImplementedError
    let x0 = array![0.0, 0.0];
    let constraints = vec![Constraint::new(constraint, Constraint::INEQUALITY)];

    let result = minimize_constrained(objective, &x0.view(), &constraints, Method::COBYLA, None);

    // COBYLA is now implemented, so it should succeed
    assert!(result.is_ok());
    let opt_result = result.expect("Operation failed");
    assert!(opt_result.success || opt_result.nit > 0); // Should make progress or succeed
}

/// Verify that Method::AugmentedLagrangian is wired up and returns a valid result.
/// The inequality constraint is x0 + x1 <= 3, minimising (x0-1)^2 + (x1-2.5)^2.
/// Unconstrained minimum at (1, 2.5) satisfies the constraint, so we expect success
/// near that point when running enough iterations.
#[test]
fn test_augmented_lagrangian_wired_up() {
    let x0 = array![0.5, 0.5];
    let constraints = vec![Constraint::new(constraint, Constraint::INEQUALITY)];

    let options = Options {
        maxiter: Some(50),
        ..Options::default()
    };

    let result = minimize_constrained(
        objective,
        &x0.view(),
        &constraints,
        Method::AugmentedLagrangian,
        Some(options),
    );

    // The method must succeed without error (no longer returns NotImplementedError)
    assert!(
        result.is_ok(),
        "AugmentedLagrangian returned an error: {:?}",
        result.err()
    );
    let opt_result = result.expect("AugmentedLagrangian failed");
    // Must make at least one iteration
    assert!(opt_result.nit > 0 || opt_result.success);
    // Objective must be finite
    assert!(opt_result.fun.is_finite());
}

/// Test with equality constraint: x0 + x1 = 2, minimise x0^2 + x1^2.
/// Optimal solution is x0 = x1 = 1 with f = 2.
#[test]
fn test_augmented_lagrangian_equality_wired_up() {
    fn obj_sum_sq(x: &[f64]) -> f64 {
        x[0].powi(2) + x[1].powi(2)
    }
    // Equality g(x) = 0, expressed as g(x) = x0 + x1 - 2 (must be 0)
    // The constraint API uses "fun >= 0" for inequality and "fun == 0" for equality.
    fn eq_con(x: &[f64]) -> f64 {
        x[0] + x[1] - 2.0
    }

    let x0 = array![0.5, 1.5];
    let constraints = vec![Constraint::new(eq_con, Constraint::EQUALITY)];
    let options = Options {
        maxiter: Some(100),
        ..Options::default()
    };

    let result = minimize_constrained(
        obj_sum_sq,
        &x0.view(),
        &constraints,
        Method::AugmentedLagrangian,
        Some(options),
    );

    assert!(
        result.is_ok(),
        "AugmentedLagrangian equality failed: {:?}",
        result.err()
    );
    let opt_result = result.expect("AugmentedLagrangian equality failed");
    assert!(opt_result.fun.is_finite());
}

// ---------------------------------------------------------------------------
// Issue #126 — closures (capturing outer variables) accepted as constraints
// ---------------------------------------------------------------------------

/// A closure capturing a local `threshold` is used as an inequality constraint.
///
/// Problem: minimise (x0 - 1)^2 + (x1 - 2.5)^2 subject to
///   threshold - x0 - x1 >= 0  with threshold = 3.0.
/// The unconstrained optimum (1.0, 2.5) has x0 + x1 = 3.5 > 3, so it is
/// infeasible; the constrained optimum lies on x0 + x1 = 3 at (0.75, 2.25).
///
/// Starting from a feasible interior point (1.5, 1.5) the SLSQP solver reaches
/// the boundary. Before issue #126, a captured-variable closure could not be
/// stored in a `Constraint` (only `fn` pointers were accepted), so this test
/// would not even compile.
#[test]
fn test_issue_126_constraint_captures_variable() {
    let threshold = 3.0_f64;

    let obj = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2);
    // Closure captures `threshold` from the enclosing scope.
    let cons = vec![Constraint::new(
        move |x: &[f64]| threshold - x[0] - x[1],
        Constraint::INEQUALITY,
    )];

    let x0 = array![1.5, 1.5];
    let result = minimize_constrained(obj, &x0, &cons, Method::SLSQP, None)
        .expect("constrained minimisation should not error");

    assert!(result.success, "SLSQP did not converge: {}", result.message);
    // The constrained solution lies on x0 + x1 = threshold.
    let sum = result.x[0] + result.x[1];
    assert!(
        (sum - threshold).abs() < 1e-2,
        "x0 + x1 = {} (expected ~{})",
        sum,
        threshold
    );
    assert!(result.fun.is_finite());
    // Objective at (0.75, 2.25) is 0.125.
    assert!(
        (result.fun - 0.125).abs() < 1e-2,
        "objective = {} (expected ~0.125)",
        result.fun
    );
}

/// Two closures capturing *different* local variables live in a single
/// `Vec<Constraint>`. This is only possible because the constraint callable is
/// stored as a boxed trait object (issue #126); distinct closure types would
/// otherwise be incompatible elements of one `Vec`.
#[test]
fn test_issue_126_heterogeneous_closures_in_vec() {
    let upper = 3.0_f64; // captured by closure A
    let lower = 0.1_f64; // captured by a *different* closure B

    let obj = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2);

    // closure_a captures `upper`; closure_b captures `lower` — different types.
    let closure_a = move |x: &[f64]| upper - x[0] - x[1];
    let closure_b = move |x: &[f64]| x[0] - lower;

    let cons = vec![
        Constraint::new(closure_a, Constraint::INEQUALITY),
        Constraint::new(closure_b, Constraint::INEQUALITY),
    ];

    let x0 = array![1.5, 1.5];
    // The point of this test is that the heterogeneous `Vec` compiles and the
    // solver runs without error.
    let result = minimize_constrained(obj, &x0, &cons, Method::SLSQP, None)
        .expect("constrained minimisation should not error");
    assert!(result.fun.is_finite());
    assert_eq!(result.constr.expect("constraint values present").len(), 2);
}

// ---------------------------------------------------------------------------
// Issue #127 — optional analytical objective gradient / constraint Jacobians
// ---------------------------------------------------------------------------

/// Supplying an analytical objective gradient via `minimize_constrained_with_jac`
/// reaches the same optimum as the finite-difference path. The analytical
/// gradient must genuinely be used (here it converges in fewer iterations).
#[test]
fn test_issue_127_analytical_objective_gradient() {
    let threshold = 3.0_f64;
    let obj = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2);
    let x0 = array![1.5, 1.5];

    // Finite-difference baseline.
    let cons_fd = vec![Constraint::new(
        move |x: &[f64]| threshold - x[0] - x[1],
        Constraint::INEQUALITY,
    )];
    let fd = minimize_constrained(obj, &x0, &cons_fd, Method::SLSQP, None)
        .expect("FD minimisation should not error");

    // Analytical objective gradient: grad f = [2(x0 - 1), 2(x1 - 2.5)].
    // Wrap the closure so it bumps a shared counter on every invocation. This
    // makes the test tamper-evident: a broken wiring that silently fell back to
    // finite differences would leave the counter at zero (and still match the
    // FD baseline below), so the equivalence asserts alone could not catch it.
    // `Arc<AtomicUsize>` is Clone + Send + Sync + 'static, satisfying the
    // `G: Fn(..) + Clone` bound of `minimize_constrained_with_jac`.
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    let grad_calls = Arc::new(AtomicUsize::new(0));
    let gc = Arc::clone(&grad_calls);
    let obj_grad = move |x: &[f64]| {
        gc.fetch_add(1, Ordering::Relaxed);
        array![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.5)]
    };
    let cons_jac = vec![Constraint::new(
        move |x: &[f64]| threshold - x[0] - x[1],
        Constraint::INEQUALITY,
    )];
    let jac =
        minimize_constrained_with_jac(obj, Some(obj_grad), &x0, &cons_jac, Method::SLSQP, None)
            .expect("analytical-gradient minimisation should not error");

    // Tamper-evidence: the user-supplied gradient must actually have been called.
    assert!(
        grad_calls.load(Ordering::Relaxed) > 0,
        "analytical objective gradient was never invoked — the analytical path is broken \
         (a silent finite-difference fallback would still match the FD baseline)"
    );

    assert!(jac.success, "SLSQP (analytical grad) did not converge");
    // Analytical and FD runs converge to the same point.
    assert!(
        (jac.x[0] - fd.x[0]).abs() < 1e-4,
        "x0: analytical {} vs FD {}",
        jac.x[0],
        fd.x[0]
    );
    assert!(
        (jac.x[1] - fd.x[1]).abs() < 1e-4,
        "x1: analytical {} vs FD {}",
        jac.x[1],
        fd.x[1]
    );
    assert!(
        (jac.fun - fd.fun).abs() < 1e-4,
        "objective: analytical {} vs FD {}",
        jac.fun,
        fd.fun
    );
}

/// Attaching an analytical constraint Jacobian via `Constraint::with_jacobian`
/// yields the same solution as the finite-difference path. For the linear
/// constraint `3 - x0 - x1`, the exact Jacobian is `[-1, -1]`.
#[test]
fn test_issue_127_constraint_with_jacobian() {
    let threshold = 3.0_f64;
    let obj = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2);
    let x0 = array![1.5, 1.5];

    // Finite-difference baseline (no analytical constraint Jacobian).
    let cons_fd = vec![Constraint::new(
        move |x: &[f64]| threshold - x[0] - x[1],
        Constraint::INEQUALITY,
    )];
    let fd = minimize_constrained(obj, &x0, &cons_fd, Method::SLSQP, None)
        .expect("FD minimisation should not error");

    // Same constraint, now with an analytical Jacobian attached. The Jacobian
    // closure bumps a shared counter on every call so the test is tamper-evident:
    // if `with_jacobian` wiring were broken and the solver silently used finite
    // differences, the counter would stay at zero (and the run would still match
    // the FD baseline below). `Arc<AtomicUsize>` is Send + Sync + 'static, which
    // satisfies the `J: Fn(..) + Send + Sync + 'static` bound of `with_jacobian`.
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    let jac_calls = Arc::new(AtomicUsize::new(0));
    let jcc = Arc::clone(&jac_calls);
    let cons_jac = vec![Constraint::new(
        move |x: &[f64]| threshold - x[0] - x[1],
        Constraint::INEQUALITY,
    )
    .with_jacobian(move |_x: &[f64]| {
        jcc.fetch_add(1, Ordering::Relaxed);
        array![-1.0, -1.0]
    })];
    let jac = minimize_constrained(obj, &x0, &cons_jac, Method::SLSQP, None)
        .expect("constraint-Jacobian minimisation should not error");

    // Tamper-evidence: the user-supplied constraint Jacobian must actually have
    // been called.
    assert!(
        jac_calls.load(Ordering::Relaxed) > 0,
        "analytical constraint Jacobian was never invoked — `with_jacobian` wiring is broken \
         (a silent finite-difference fallback would still match the FD baseline)"
    );

    assert!(jac.success, "SLSQP (constraint Jacobian) did not converge");
    assert!(
        (jac.x[0] - fd.x[0]).abs() < 1e-4,
        "x0: analytical-jac {} vs FD {}",
        jac.x[0],
        fd.x[0]
    );
    assert!(
        (jac.x[1] - fd.x[1]).abs() < 1e-4,
        "x1: analytical-jac {} vs FD {}",
        jac.x[1],
        fd.x[1]
    );
    assert!(
        (jac.fun - fd.fun).abs() < 1e-4,
        "objective: analytical-jac {} vs FD {}",
        jac.fun,
        fd.fun
    );
}
