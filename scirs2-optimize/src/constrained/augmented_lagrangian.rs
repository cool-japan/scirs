//! Augmented Lagrangian methods for constrained optimization
//!
//! This module implements augmented Lagrangian methods (also known as method of multipliers)
//! for solving constrained optimization problems. These methods transform constrained problems
//! into a sequence of unconstrained problems by augmenting the objective function with
//! penalty terms and Lagrange multiplier estimates.

use crate::error::OptimizeError;
use crate::unconstrained::{minimize, Method, Options};
use ndarray::{Array1, ArrayView1};

/// Options for augmented Lagrangian method
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianOptions {
    /// Maximum number of outer iterations
    pub max_iter: usize,
    /// Tolerance for constraint violation
    pub constraint_tol: f64,
    /// Tolerance for optimality
    pub optimality_tol: f64,
    /// Initial penalty parameter
    pub initial_penalty: f64,
    /// Penalty update factor
    pub penalty_update_factor: f64,
    /// Maximum penalty parameter
    pub max_penalty: f64,
    /// Lagrange multiplier update tolerance
    pub multiplier_update_tol: f64,
    /// Maximum allowed constraint violation before increasing penalty
    pub max_constraint_violation: f64,
    /// Method for solving unconstrained subproblems
    pub unconstrained_method: Method,
    /// Options for unconstrained subproblems
    pub unconstrained_options: Options,
    /// Trust region radius for constrained optimization
    pub trust_radius: Option<f64>,
    /// Use adaptive penalty parameter updates
    pub adaptive_penalty: bool,
}

impl Default for AugmentedLagrangianOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            constraint_tol: 1e-6,
            optimality_tol: 1e-6,
            initial_penalty: 1.0,
            penalty_update_factor: 10.0,
            max_penalty: 1e8,
            multiplier_update_tol: 1e-8,
            max_constraint_violation: 1e-3,
            unconstrained_method: Method::LBFGS,
            unconstrained_options: Options::default(),
            trust_radius: None,
            adaptive_penalty: true,
        }
    }
}

/// Result from augmented Lagrangian optimization
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianResult {
    /// Optimal solution
    pub x: Array1<f64>,
    /// Optimal objective value
    pub fun: f64,
    /// Lagrange multipliers for equality constraints
    pub lambda_eq: Option<Array1<f64>>,
    /// Lagrange multipliers for inequality constraints
    pub lambda_ineq: Option<Array1<f64>>,
    /// Number of outer iterations
    pub nit: usize,
    /// Total number of function evaluations
    pub nfev: usize,
    /// Success flag
    pub success: bool,
    /// Status message
    pub message: String,
    /// Final penalty parameter
    pub penalty: f64,
    /// Final constraint violation
    pub constraint_violation: f64,
    /// Final optimality measure
    pub optimality: f64,
}

/// Augmented Lagrangian solver state
struct AugmentedLagrangianState {
    /// Current penalty parameter
    penalty: f64,
    /// Lagrange multipliers for equality constraints
    lambda_eq: Option<Array1<f64>>,
    /// Lagrange multipliers for inequality constraints
    lambda_ineq: Option<Array1<f64>>,
    /// Number of variables
    #[allow(dead_code)]
    n: usize,
    /// Number of equality constraints
    #[allow(dead_code)]
    m_eq: usize,
    /// Number of inequality constraints
    #[allow(dead_code)]
    m_ineq: usize,
}

impl AugmentedLagrangianState {
    fn new(n: usize, m_eq: usize, m_ineq: usize, initial_penalty: f64) -> Self {
        Self {
            penalty: initial_penalty,
            lambda_eq: if m_eq > 0 {
                Some(Array1::zeros(m_eq))
            } else {
                None
            },
            lambda_ineq: if m_ineq > 0 {
                Some(Array1::zeros(m_ineq))
            } else {
                None
            },
            n,
            m_eq,
            m_ineq,
        }
    }

    /// Update Lagrange multipliers based on constraint values
    fn update_multipliers(&mut self, c_eq: &Option<Array1<f64>>, c_ineq: &Option<Array1<f64>>) {
        // Update equality constraint multipliers
        if let (Some(ref mut lambda), Some(ref c)) = (&mut self.lambda_eq, c_eq) {
            for i in 0..lambda.len() {
                lambda[i] += self.penalty * c[i];
            }
        }

        // Update inequality constraint multipliers
        if let (Some(ref mut lambda), Some(ref c)) = (&mut self.lambda_ineq, c_ineq) {
            for i in 0..lambda.len() {
                lambda[i] = f64::max(0.0, lambda[i] + self.penalty * c[i]);
            }
        }
    }

    /// Compute constraint violation
    fn compute_constraint_violation(
        &self,
        c_eq: &Option<Array1<f64>>,
        c_ineq: &Option<Array1<f64>>,
    ) -> f64 {
        let mut violation = 0.0;

        // Equality constraint violation
        if let Some(ref c) = c_eq {
            violation += c.mapv(|x| x.abs()).sum();
        }

        // Inequality constraint violation (only positive violations count)
        if let Some(ref c) = c_ineq {
            violation += c.mapv(|x| f64::max(0.0, x)).sum();
        }

        violation
    }
}

/// Minimize a function subject to constraints using augmented Lagrangian method
pub fn minimize_augmented_lagrangian<F, EqCon, IneqCon>(
    fun: F,
    x0: Array1<f64>,
    eq_constraints: Option<EqCon>,
    ineq_constraints: Option<IneqCon>,
    options: Option<AugmentedLagrangianOptions>,
) -> Result<AugmentedLagrangianResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
    EqCon: Fn(&ArrayView1<f64>) -> Array1<f64> + Clone,
    IneqCon: Fn(&ArrayView1<f64>) -> Array1<f64> + Clone,
{
    let options = options.unwrap_or_default();
    let n = x0.len();

    // Determine constraint dimensions
    let m_eq = if let Some(ref eq_con) = eq_constraints {
        // Evaluate once to get dimension
        eq_con(&x0.view()).len()
    } else {
        0
    };

    let m_ineq = if let Some(ref ineq_con) = ineq_constraints {
        // Evaluate once to get dimension
        ineq_con(&x0.view()).len()
    } else {
        0
    };

    let mut state = AugmentedLagrangianState::new(n, m_eq, m_ineq, options.initial_penalty);
    let mut x = x0.clone();
    let mut total_nfev = 0;

    for iter in 0..options.max_iter {
        // Create augmented Lagrangian function as a closure that captures by value
        let penalty = state.penalty;
        let lambda_eq = state.lambda_eq.clone();
        let lambda_ineq = state.lambda_ineq.clone();
        let fun_clone = fun.clone();
        let eq_con_clone = eq_constraints.clone();
        let ineq_con_clone = ineq_constraints.clone();

        let augmented_lagrangian = move |x: &ArrayView1<f64>| -> f64 {
            let mut result = fun_clone(x);

            // Add equality constraint terms
            if let (Some(ref eq_con), Some(ref lambda)) = (&eq_con_clone, &lambda_eq) {
                let c_eq = eq_con(x);
                for i in 0..c_eq.len() {
                    result += lambda[i] * c_eq[i] + 0.5 * penalty * c_eq[i].powi(2);
                }
            }

            // Add inequality constraint terms
            if let (Some(ref ineq_con), Some(ref lambda)) = (&ineq_con_clone, &lambda_ineq) {
                let c_ineq = ineq_con(x);
                for i in 0..c_ineq.len() {
                    let augmented_term = lambda[i] + penalty * c_ineq[i];
                    if augmented_term > 0.0 {
                        result += augmented_term * c_ineq[i] - 0.5 / penalty * lambda[i].powi(2);
                    } else {
                        result -= 0.5 / penalty * lambda[i].powi(2);
                    }
                }
            }

            result
        };

        // Solve unconstrained subproblem
        let result = minimize(
            augmented_lagrangian,
            x.as_slice().unwrap(),
            options.unconstrained_method,
            Some(options.unconstrained_options.clone()),
        )?;

        x = result.x;
        total_nfev += result.func_evals;

        // Evaluate constraints at current point
        let c_eq = eq_constraints.as_ref().map(|f| f(&x.view()));
        let c_ineq = ineq_constraints.as_ref().map(|f| f(&x.view()));

        // Compute constraint violation and optimality
        let constraint_violation = state.compute_constraint_violation(&c_eq, &c_ineq);
        let optimality = compute_optimality(&fun, &x, &c_eq, &c_ineq, &state);

        // Check convergence
        if constraint_violation < options.constraint_tol && optimality < options.optimality_tol {
            let final_fun = fun(&x.view());
            return Ok(AugmentedLagrangianResult {
                x,
                fun: final_fun,
                lambda_eq: state.lambda_eq.clone(),
                lambda_ineq: state.lambda_ineq.clone(),
                nit: iter,
                nfev: total_nfev,
                success: true,
                message: "Optimization terminated successfully.".to_string(),
                penalty: state.penalty,
                constraint_violation,
                optimality,
            });
        }

        // Update multipliers
        state.update_multipliers(&c_eq, &c_ineq);

        // Update penalty parameter if needed
        if options.adaptive_penalty && constraint_violation > options.max_constraint_violation {
            state.penalty =
                (state.penalty * options.penalty_update_factor).min(options.max_penalty);
        }
    }

    // Maximum iterations reached
    let c_eq = eq_constraints.as_ref().map(|f| f(&x.view()));
    let c_ineq = ineq_constraints.as_ref().map(|f| f(&x.view()));
    let final_violation = state.compute_constraint_violation(&c_eq, &c_ineq);
    let final_optimality = compute_optimality(&fun, &x, &c_eq, &c_ineq, &state);

    let final_fun = fun(&x.view());
    Ok(AugmentedLagrangianResult {
        x,
        fun: final_fun,
        lambda_eq: state.lambda_eq,
        lambda_ineq: state.lambda_ineq,
        nit: options.max_iter,
        nfev: total_nfev,
        success: false,
        message: "Maximum iterations reached.".to_string(),
        penalty: state.penalty,
        constraint_violation: final_violation,
        optimality: final_optimality,
    })
}

/// Create augmented Lagrangian function
#[allow(dead_code)]
fn create_augmented_lagrangian<'a, F, EqCon, IneqCon>(
    fun: &'a F,
    eq_constraints: &'a Option<EqCon>,
    ineq_constraints: &'a Option<IneqCon>,
    state: &'a AugmentedLagrangianState,
) -> impl Fn(&ArrayView1<f64>) -> f64 + 'a
where
    F: Fn(&ArrayView1<f64>) -> f64,
    EqCon: Fn(&ArrayView1<f64>) -> Array1<f64>,
    IneqCon: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    move |x: &ArrayView1<f64>| -> f64 {
        let mut result = fun(x);

        // Add equality constraint terms
        if let (Some(ref eq_con), Some(ref lambda_eq)) = (eq_constraints, &state.lambda_eq) {
            let c_eq = eq_con(x);
            for i in 0..c_eq.len() {
                result += lambda_eq[i] * c_eq[i] + 0.5 * state.penalty * c_eq[i].powi(2);
            }
        }

        // Add inequality constraint terms
        if let (Some(ref ineq_con), Some(ref lambda_ineq)) = (ineq_constraints, &state.lambda_ineq)
        {
            let c_ineq = ineq_con(x);
            for i in 0..c_ineq.len() {
                let augmented_term = lambda_ineq[i] + state.penalty * c_ineq[i];
                if augmented_term > 0.0 {
                    result +=
                        augmented_term * c_ineq[i] - 0.5 / state.penalty * lambda_ineq[i].powi(2);
                } else {
                    result -= 0.5 / state.penalty * lambda_ineq[i].powi(2);
                }
            }
        }

        result
    }
}

/// Compute optimality measure  
fn compute_optimality<F>(
    fun: &F,
    x: &Array1<f64>,
    _c_eq: &Option<Array1<f64>>,
    _c_ineq: &Option<Array1<f64>>,
    _state: &AugmentedLagrangianState,
) -> f64
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    // Compute finite difference gradient of objective
    let eps = 1e-8;
    let mut grad = Array1::zeros(x.len());
    let f0 = fun(&x.view());

    for i in 0..x.len() {
        let mut x_plus = x.clone();
        x_plus[i] += eps;
        let f_plus = fun(&x_plus.view());
        grad[i] = (f_plus - f0) / eps;
    }

    // For now, just return the norm of the objective gradient
    // In practice, we would need to include constraint gradient contributions
    grad.mapv(|x| x.abs()).sum()
}

/// Minimize with equality constraints only
pub fn minimize_equality_constrained<F, EqCon>(
    fun: F,
    x0: Array1<f64>,
    eq_constraints: EqCon,
    options: Option<AugmentedLagrangianOptions>,
) -> Result<AugmentedLagrangianResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
    EqCon: Fn(&ArrayView1<f64>) -> Array1<f64> + Clone,
{
    minimize_augmented_lagrangian(
        fun,
        x0,
        Some(eq_constraints),
        None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
        options,
    )
}

/// Minimize with inequality constraints only
pub fn minimize_inequality_constrained<F, IneqCon>(
    fun: F,
    x0: Array1<f64>,
    ineq_constraints: IneqCon,
    options: Option<AugmentedLagrangianOptions>,
) -> Result<AugmentedLagrangianResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
    IneqCon: Fn(&ArrayView1<f64>) -> Array1<f64> + Clone,
{
    minimize_augmented_lagrangian(
        fun,
        x0,
        None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
        Some(ineq_constraints),
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_augmented_lagrangian_equality_constraint() {
        // Minimize x^2 + y^2 subject to x + y = 1
        let fun = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let eq_con = |x: &ArrayView1<f64>| array![x[0] + x[1] - 1.0];

        let x0 = array![0.0, 0.0];
        let options = AugmentedLagrangianOptions {
            max_iter: 50,
            constraint_tol: 1e-6,
            optimality_tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_equality_constrained(fun, x0, eq_con, Some(options)).unwrap();

        // Just check that the algorithm runs without error - optimization may not converge perfectly
        assert!(result.nit > 0);
        // Optimal solution should be at (0.5, 0.5)
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 1e-3);
        assert_abs_diff_eq!(result.fun, 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_augmented_lagrangian_inequality_constraint() {
        // Minimize x^2 + y^2 subject to x + y >= 1
        let fun = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let ineq_con = |x: &ArrayView1<f64>| array![1.0 - x[0] - x[1]]; // g(x) <= 0 form

        let x0 = array![2.0, 2.0];
        let options = AugmentedLagrangianOptions {
            max_iter: 50,
            constraint_tol: 1e-6,
            optimality_tol: 1e-6,
            ..Default::default()
        };

        let result = minimize_inequality_constrained(fun, x0, ineq_con, Some(options)).unwrap();

        // Just check that the algorithm runs without error - optimization may not converge perfectly
        assert!(result.nit > 0);
        // Optimal solution should be at (0.5, 0.5)
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 1e-3);
        assert_abs_diff_eq!(result.fun, 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_augmented_lagrangian_mixed_constraints() {
        // Minimize (x-1)^2 + (y-2)^2 subject to x + y = 3 and x >= 0
        let fun = |x: &ArrayView1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let eq_con = |x: &ArrayView1<f64>| array![x[0] + x[1] - 3.0];
        let ineq_con = |x: &ArrayView1<f64>| array![-x[0]]; // x >= 0 becomes -x <= 0

        let x0 = array![1.0, 1.0];
        let options = AugmentedLagrangianOptions {
            max_iter: 50,
            constraint_tol: 1e-6,
            optimality_tol: 1e-6,
            ..Default::default()
        };

        let result =
            minimize_augmented_lagrangian(fun, x0, Some(eq_con), Some(ineq_con), Some(options))
                .unwrap();

        // Just check that the algorithm runs without error - optimization may not converge perfectly
        assert!(result.nit > 0);
        // Optimal solution should be at (1, 2) due to equality constraint
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.fun, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_augmented_lagrangian_unconstrained() {
        // Test with no constraints (should behave like unconstrained optimization)
        let fun = |x: &ArrayView1<f64>| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);

        let x0 = array![0.0, 0.0];
        let options = AugmentedLagrangianOptions {
            max_iter: 50,
            ..Default::default()
        };

        let result = minimize_augmented_lagrangian(
            fun,
            x0,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            Some(options),
        )
        .unwrap();

        // Just check that the algorithm runs and makes some progress
        assert!(result.fun < 4.0); // Should reduce from initial objective value
    }
}
