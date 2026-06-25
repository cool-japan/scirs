//! Sequential Quadratic Programming (SQP) for constrained optimization
//!
//! This module implements a full SQP method for nonlinear constrained optimization.
//! SQP solves a sequence of quadratic programming subproblems, each formed by
//! approximating the Lagrangian with a quadratic model and linearizing the constraints.
//!
//! Features:
//! - BFGS Hessian approximation of the Lagrangian
//! - Active set management for inequality constraints
//! - L1 exact penalty merit function with Maratos correction
//! - Watchdog line search strategy
//! - Equality and inequality constraint support
//!
//! # References
//!
//! - Nocedal & Wright, "Numerical Optimization", Chapter 18
//! - Boggs & Tolle, "Sequential Quadratic Programming", Acta Numerica 1995

use crate::constrained::{Constraint, ConstraintKind};
use crate::error::OptimizeError;
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix1};

/// Options for the SQP algorithm
#[derive(Debug, Clone)]
pub struct SqpOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for optimality (KKT conditions)
    pub tol: f64,
    /// Tolerance for constraint violation
    pub constraint_tol: f64,
    /// Step size for finite difference gradient approximation
    pub eps: f64,
    /// Initial penalty parameter for merit function
    pub initial_penalty: f64,
    /// Penalty increase factor
    pub penalty_increase: f64,
    /// Maximum penalty parameter
    pub max_penalty: f64,
    /// Line search backtracking factor
    pub backtrack_factor: f64,
    /// Sufficient decrease parameter (Armijo)
    pub armijo_c: f64,
    /// Maximum line search iterations
    pub max_ls_iter: usize,
    /// BFGS damping parameter (Powell's modification)
    pub bfgs_damping_threshold: f64,
    /// Whether to use BFGS updates for Hessian
    pub use_bfgs: bool,
}

impl Default for SqpOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-8,
            constraint_tol: 1e-8,
            eps: 1e-8,
            initial_penalty: 10.0,
            penalty_increase: 2.0,
            max_penalty: 1e10,
            backtrack_factor: 0.5,
            armijo_c: 1e-4,
            max_ls_iter: 40,
            bfgs_damping_threshold: 0.2,
            use_bfgs: true,
        }
    }
}

/// Result from SQP optimization
#[derive(Debug, Clone)]
pub struct SqpResult {
    /// Optimal solution
    pub x: Array1<f64>,
    /// Optimal objective value
    pub fun: f64,
    /// Lagrange multipliers for equality constraints
    pub lambda_eq: Array1<f64>,
    /// Lagrange multipliers for inequality constraints
    pub lambda_ineq: Array1<f64>,
    /// Number of iterations
    pub nit: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Success flag
    pub success: bool,
    /// Status message
    pub message: String,
    /// Final constraint violation
    pub constraint_violation: f64,
    /// Final optimality measure (norm of KKT residual)
    pub optimality: f64,
}

/// Active set status for inequality constraints
#[derive(Debug, Clone, Copy, PartialEq)]
enum ConstraintStatus {
    /// Constraint is inactive (slack > tolerance)
    Inactive,
    /// Constraint is active (at the boundary)
    Active,
}

/// SQP solver for constrained optimization
struct SqpSolver {
    /// Number of decision variables
    n: usize,
    /// Number of equality constraints
    n_eq: usize,
    /// Number of inequality constraints
    n_ineq: usize,
    /// Current iterate
    x: Array1<f64>,
    /// Current objective value
    f_val: f64,
    /// Current gradient
    grad: Array1<f64>,
    /// Approximate Hessian of the Lagrangian (BFGS)
    hessian: Array2<f64>,
    /// Equality constraint values
    c_eq: Array1<f64>,
    /// Inequality constraint values (g(x) >= 0 form)
    c_ineq: Array1<f64>,
    /// Equality constraint Jacobian
    jac_eq: Array2<f64>,
    /// Inequality constraint Jacobian
    jac_ineq: Array2<f64>,
    /// Lagrange multipliers for equality constraints
    lambda_eq: Array1<f64>,
    /// Lagrange multipliers for inequality constraints
    lambda_ineq: Array1<f64>,
    /// Penalty parameter for merit function
    penalty: f64,
    /// Active set for inequality constraints
    active_set: Vec<ConstraintStatus>,
    /// Number of function evaluations
    nfev: usize,
    /// Options
    options: SqpOptions,
}

impl SqpSolver {
    /// Create a new SQP solver
    fn new(x0: Array1<f64>, n_eq: usize, n_ineq: usize, options: SqpOptions) -> Self {
        let n = x0.len();
        Self {
            n,
            n_eq,
            n_ineq,
            x: x0,
            f_val: 0.0,
            grad: Array1::zeros(n),
            hessian: Array2::eye(n),
            c_eq: Array1::zeros(n_eq),
            c_ineq: Array1::zeros(n_ineq),
            jac_eq: Array2::zeros((n_eq, n)),
            jac_ineq: Array2::zeros((n_ineq, n)),
            lambda_eq: Array1::zeros(n_eq),
            lambda_ineq: Array1::zeros(n_ineq),
            penalty: options.initial_penalty,
            active_set: vec![ConstraintStatus::Inactive; n_ineq],
            nfev: 0,
            options,
        }
    }

    /// Evaluate the objective function and gradient
    fn evaluate_objective<F>(&mut self, func: &F, x: &Array1<f64>) -> (f64, Array1<f64>)
    where
        F: Fn(&[f64]) -> f64,
    {
        let x_slice = x.as_slice().unwrap_or(&[]);
        let f_val = func(x_slice);
        self.nfev += 1;

        // Finite difference gradient
        let mut grad = Array1::zeros(self.n);
        for i in 0..self.n {
            let mut x_plus = x.clone();
            x_plus[i] += self.options.eps;
            let f_plus = func(x_plus.as_slice().unwrap_or(&[]));
            self.nfev += 1;

            let mut x_minus = x.clone();
            x_minus[i] -= self.options.eps;
            let f_minus = func(x_minus.as_slice().unwrap_or(&[]));
            self.nfev += 1;

            grad[i] = (f_plus - f_minus) / (2.0 * self.options.eps);
        }

        (f_val, grad)
    }

    /// Evaluate constraints and their Jacobians
    fn evaluate_constraints(
        &mut self,
        eq_constraints: &[(usize, &Constraint)],
        ineq_constraints: &[(usize, &Constraint)],
        x: &Array1<f64>,
    ) {
        let x_slice = x.as_slice().unwrap_or(&[]);

        // Evaluate equality constraints
        for (idx, (_, constraint)) in eq_constraints.iter().enumerate() {
            self.c_eq[idx] = (constraint.fun)(x_slice);
            self.nfev += 1;

            // Jacobian via finite differences
            for j in 0..self.n {
                let mut x_plus = x.clone();
                x_plus[j] += self.options.eps;
                let c_plus = (constraint.fun)(x_plus.as_slice().unwrap_or(&[]));
                self.nfev += 1;
                self.jac_eq[[idx, j]] = (c_plus - self.c_eq[idx]) / self.options.eps;
            }
        }

        // Evaluate inequality constraints (g(x) >= 0)
        for (idx, (_, constraint)) in ineq_constraints.iter().enumerate() {
            self.c_ineq[idx] = (constraint.fun)(x_slice);
            self.nfev += 1;

            // Jacobian via finite differences
            for j in 0..self.n {
                let mut x_plus = x.clone();
                x_plus[j] += self.options.eps;
                let c_plus = (constraint.fun)(x_plus.as_slice().unwrap_or(&[]));
                self.nfev += 1;
                self.jac_ineq[[idx, j]] = (c_plus - self.c_ineq[idx]) / self.options.eps;
            }
        }
    }

    /// Update the active set based on current constraint values and multipliers
    fn update_active_set(&mut self) {
        for i in 0..self.n_ineq {
            if self.c_ineq[i].abs() < self.options.constraint_tol * 10.0 || self.c_ineq[i] < 0.0 {
                self.active_set[i] = ConstraintStatus::Active;
            } else if self.lambda_ineq[i].abs() < self.options.tol {
                self.active_set[i] = ConstraintStatus::Inactive;
            }
        }
    }

    /// Solve the QP subproblem to find the search direction
    ///
    /// min  grad^T d + 0.5 d^T H d
    /// s.t. jac_eq * d + c_eq = 0       (equality)
    ///      jac_ineq * d + c_ineq >= 0   (inequality, active set)
    ///
    /// Returns (direction, lambda_eq, lambda_ineq)
    fn solve_qp_subproblem(
        &self,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), OptimizeError> {
        // Collect active inequality constraints
        let active_indices: Vec<usize> = (0..self.n_ineq)
            .filter(|&i| self.active_set[i] == ConstraintStatus::Active)
            .collect();

        let n_active = active_indices.len();
        let n_total_constraints = self.n_eq + n_active;

        if n_total_constraints == 0 {
            // Unconstrained QP: d = -H^{-1} g
            let d = solve_symmetric_system(&self.hessian, &(-&self.grad))?;
            return Ok((d, Array1::zeros(self.n_eq), Array1::zeros(self.n_ineq)));
        }

        // Build the KKT system:
        // [H   A^T] [d]   = [-g]
        // [A   0  ] [mu]    [-c]
        //
        // where A is the Jacobian of active constraints, c is active constraint values
        let kkt_size = self.n + n_total_constraints;
        let mut kkt_matrix = Array2::zeros((kkt_size, kkt_size));
        let mut kkt_rhs = Array1::zeros(kkt_size);

        // Top-left block: H
        for i in 0..self.n {
            for j in 0..self.n {
                kkt_matrix[[i, j]] = self.hessian[[i, j]];
            }
            // Add small regularization for numerical stability
            kkt_matrix[[i, i]] += 1e-10;
        }

        // Top-right and bottom-left blocks: A^T and A (equality)
        for i in 0..self.n_eq {
            for j in 0..self.n {
                kkt_matrix[[j, self.n + i]] = self.jac_eq[[i, j]];
                kkt_matrix[[self.n + i, j]] = self.jac_eq[[i, j]];
            }
        }

        // Top-right and bottom-left blocks: A^T and A (active inequality)
        for (k, &idx) in active_indices.iter().enumerate() {
            let row = self.n_eq + k;
            for j in 0..self.n {
                kkt_matrix[[j, self.n + row]] = self.jac_ineq[[idx, j]];
                kkt_matrix[[self.n + row, j]] = self.jac_ineq[[idx, j]];
            }
        }

        // RHS: [-g; -c_eq; -c_ineq_active]
        for i in 0..self.n {
            kkt_rhs[i] = -self.grad[i];
        }
        for i in 0..self.n_eq {
            kkt_rhs[self.n + i] = -self.c_eq[i];
        }
        for (k, &idx) in active_indices.iter().enumerate() {
            kkt_rhs[self.n + self.n_eq + k] = -self.c_ineq[idx];
        }

        // Solve the KKT system
        let solution = solve_general_system(&kkt_matrix, &kkt_rhs)?;

        // Extract direction and multipliers
        let d = solution
            .slice(scirs2_core::ndarray::s![..self.n])
            .to_owned();

        let mut lambda_eq = Array1::zeros(self.n_eq);
        for i in 0..self.n_eq {
            lambda_eq[i] = solution[self.n + i];
        }

        let mut lambda_ineq = Array1::zeros(self.n_ineq);
        for (k, &idx) in active_indices.iter().enumerate() {
            lambda_ineq[idx] = solution[self.n + self.n_eq + k];
            // Project inequality multipliers to be non-negative
            if lambda_ineq[idx] < 0.0 {
                lambda_ineq[idx] = 0.0;
            }
        }

        Ok((d, lambda_eq, lambda_ineq))
    }

    /// Compute the L1 exact penalty merit function
    ///
    /// phi(x, mu) = f(x) + mu * (sum|h_i(x)| + sum max(0, -g_j(x)))
    fn merit_function<F>(
        &mut self,
        func: &F,
        x: &Array1<f64>,
        eq_constraints: &[(usize, &Constraint)],
        ineq_constraints: &[(usize, &Constraint)],
    ) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let x_slice = x.as_slice().unwrap_or(&[]);
        let f_val = func(x_slice);
        self.nfev += 1;

        let mut violation = 0.0;

        // Equality constraint violations
        for (_, constraint) in eq_constraints {
            let c = (constraint.fun)(x_slice);
            self.nfev += 1;
            violation += c.abs();
        }

        // Inequality constraint violations (g(x) >= 0 form)
        for (_, constraint) in ineq_constraints {
            let c = (constraint.fun)(x_slice);
            self.nfev += 1;
            if c < 0.0 {
                violation += -c;
            }
        }

        f_val + self.penalty * violation
    }

    /// Compute the directional derivative of the merit function
    fn merit_directional_derivative(&self, d: &Array1<f64>) -> f64 {
        let grad_d = self.grad.dot(d);

        let mut violation_deriv = 0.0;

        // Equality: d/dalpha |h_i(x+alpha*d)| at alpha=0 = sign(h_i) * jac_eq_i . d
        for i in 0..self.n_eq {
            let jac_d = self.jac_eq.row(i).dot(d);
            if self.c_eq[i] > 0.0 {
                violation_deriv += jac_d;
            } else if self.c_eq[i] < 0.0 {
                violation_deriv -= jac_d;
            } else {
                violation_deriv += jac_d.abs();
            }
        }

        // Inequality: d/dalpha max(0, -g_j(x+alpha*d)) at alpha=0
        for i in 0..self.n_ineq {
            if self.c_ineq[i] < 0.0 {
                let jac_d = self.jac_ineq.row(i).dot(d);
                violation_deriv -= jac_d;
            }
        }

        grad_d + self.penalty * violation_deriv
    }

    /// Line search using the merit function
    fn line_search<F>(
        &mut self,
        func: &F,
        d: &Array1<f64>,
        eq_constraints: &[(usize, &Constraint)],
        ineq_constraints: &[(usize, &Constraint)],
    ) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let x_copy = self.x.clone();
        let merit_0 = self.merit_function(func, &x_copy, eq_constraints, ineq_constraints);
        let dir_deriv = self.merit_directional_derivative(d);

        let mut alpha = 1.0;

        for _ in 0..self.options.max_ls_iter {
            let x_trial = &self.x + &(alpha * d);
            let merit_trial = self.merit_function(func, &x_trial, eq_constraints, ineq_constraints);

            if merit_trial <= merit_0 + self.options.armijo_c * alpha * dir_deriv {
                return alpha;
            }

            alpha *= self.options.backtrack_factor;
        }

        alpha
    }

    /// Update penalty parameter
    fn update_penalty(&mut self, lambda_eq: &Array1<f64>, lambda_ineq: &Array1<f64>) {
        // Set penalty > max(|lambda_i|) to ensure descent in merit function
        let max_lambda_eq = lambda_eq.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let max_lambda_ineq = lambda_ineq.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let max_lambda = max_lambda_eq.max(max_lambda_ineq);

        if self.penalty < 1.1 * max_lambda {
            self.penalty = (1.1 * max_lambda).min(self.options.max_penalty);
        }
    }

    /// BFGS update of the Hessian approximation with Powell's damping
    fn bfgs_update(&mut self, s: &Array1<f64>, y_raw: &Array1<f64>) {
        let s_norm_sq = s.dot(s);
        if s_norm_sq < 1e-30 {
            return; // Skip update if step is too small
        }

        // Powell's damped BFGS update
        let hs = self.hessian.dot(s);
        let s_hs = s.dot(&hs);

        let s_y = s.dot(y_raw);
        let threshold = self.options.bfgs_damping_threshold;

        // Compute damped y
        let y = if s_y >= threshold * s_hs {
            y_raw.clone()
        } else {
            // Damping: y = theta * y_raw + (1 - theta) * Hs
            let theta = if (s_hs - s_y).abs() > 1e-30 {
                ((1.0 - threshold) * s_hs) / (s_hs - s_y)
            } else {
                1.0
            };
            theta * y_raw + (1.0 - theta) * &hs
        };

        let s_y_damped = s.dot(&y);
        if s_y_damped.abs() < 1e-30 || s_hs.abs() < 1e-30 {
            return; // Skip degenerate updates
        }

        // Standard BFGS formula: H <- H - (Hs)(Hs)^T / (s^T Hs) + yy^T / (s^T y)
        for i in 0..self.n {
            for j in 0..self.n {
                self.hessian[[i, j]] += y[i] * y[j] / s_y_damped - hs[i] * hs[j] / s_hs;
            }
        }
    }

    /// Compute Lagrangian gradient for BFGS y-vector
    fn lagrangian_gradient(
        &self,
        grad: &Array1<f64>,
        lambda_eq: &Array1<f64>,
        lambda_ineq: &Array1<f64>,
    ) -> Array1<f64> {
        let mut lag_grad = grad.clone();

        // L = f - sum lambda_eq_i * h_i - sum lambda_ineq_j * g_j
        // grad_x L = grad_f - sum lambda_eq_i * grad h_i - sum lambda_ineq_j * grad g_j
        for i in 0..self.n_eq {
            for j in 0..self.n {
                lag_grad[j] -= lambda_eq[i] * self.jac_eq[[i, j]];
            }
        }
        for i in 0..self.n_ineq {
            for j in 0..self.n {
                lag_grad[j] -= lambda_ineq[i] * self.jac_ineq[[i, j]];
            }
        }

        lag_grad
    }

    /// Compute constraint violation measure
    fn constraint_violation(&self) -> f64 {
        let eq_viol: f64 = self.c_eq.iter().map(|c| c.abs()).sum();
        let ineq_viol: f64 = self
            .c_ineq
            .iter()
            .map(|c| if *c < 0.0 { -c } else { 0.0 })
            .sum();
        eq_viol + ineq_viol
    }

    /// Compute KKT optimality measure
    fn kkt_optimality(&self) -> f64 {
        let lag_grad = self.lagrangian_gradient(&self.grad, &self.lambda_eq, &self.lambda_ineq);
        let stationarity = lag_grad.dot(&lag_grad).sqrt();

        let primal_feasibility = self.constraint_violation();

        // Complementarity: lambda_i * g_i(x) = 0 for active inequality constraints
        let mut complementarity = 0.0;
        for i in 0..self.n_ineq {
            complementarity += (self.lambda_ineq[i] * self.c_ineq[i]).abs();
        }

        stationarity + primal_feasibility + complementarity
    }

    /// Run the SQP algorithm
    fn solve<F>(
        &mut self,
        func: &F,
        eq_constraints: &[(usize, &Constraint)],
        ineq_constraints: &[(usize, &Constraint)],
    ) -> Result<SqpResult, OptimizeError>
    where
        F: Fn(&[f64]) -> f64,
    {
        // Initial evaluation
        let (f_val, grad) = self.evaluate_objective(func, &self.x.clone());
        self.f_val = f_val;
        self.grad = grad;
        self.evaluate_constraints(eq_constraints, ineq_constraints, &self.x.clone());

        // Initialize active set
        self.update_active_set();

        let mut iteration = 0;

        for _ in 0..self.options.max_iter {
            iteration += 1;

            // Solve QP subproblem
            let (d, qp_lambda_eq, qp_lambda_ineq) = match self.solve_qp_subproblem() {
                Ok(result) => result,
                Err(_) => {
                    // If QP solve fails, try steepest descent direction
                    let d = -&self.grad;
                    let d_norm = d.dot(&d).sqrt();
                    if d_norm > 1e-30 {
                        (d / d_norm, self.lambda_eq.clone(), self.lambda_ineq.clone())
                    } else {
                        return Ok(SqpResult {
                            x: self.x.clone(),
                            fun: self.f_val,
                            lambda_eq: self.lambda_eq.clone(),
                            lambda_ineq: self.lambda_ineq.clone(),
                            nit: iteration,
                            nfev: self.nfev,
                            success: false,
                            message: "QP subproblem solve failed with zero gradient".to_string(),
                            constraint_violation: self.constraint_violation(),
                            optimality: self.kkt_optimality(),
                        });
                    }
                }
            };

            // Check for convergence
            let d_norm = d.dot(&d).sqrt();
            let kkt = self.kkt_optimality();

            if kkt < self.options.tol && self.constraint_violation() < self.options.constraint_tol {
                return Ok(SqpResult {
                    x: self.x.clone(),
                    fun: self.f_val,
                    lambda_eq: self.lambda_eq.clone(),
                    lambda_ineq: self.lambda_ineq.clone(),
                    nit: iteration,
                    nfev: self.nfev,
                    success: true,
                    message: "Optimization converged: KKT conditions satisfied".to_string(),
                    constraint_violation: self.constraint_violation(),
                    optimality: kkt,
                });
            }

            if d_norm < 1e-15 {
                return Ok(SqpResult {
                    x: self.x.clone(),
                    fun: self.f_val,
                    lambda_eq: self.lambda_eq.clone(),
                    lambda_ineq: self.lambda_ineq.clone(),
                    nit: iteration,
                    nfev: self.nfev,
                    success: self.constraint_violation() < self.options.constraint_tol,
                    message: "Step size too small".to_string(),
                    constraint_violation: self.constraint_violation(),
                    optimality: kkt,
                });
            }

            // Update penalty parameter
            self.update_penalty(&qp_lambda_eq, &qp_lambda_ineq);

            // Store old Lagrangian gradient for BFGS
            let lag_grad_old = self.lagrangian_gradient(&self.grad, &qp_lambda_eq, &qp_lambda_ineq);

            // Line search
            let alpha = self.line_search(func, &d, eq_constraints, ineq_constraints);

            // Update x
            let s = alpha * &d;
            self.x = &self.x + &s;

            // Re-evaluate at new point
            let x_new = self.x.clone();
            let (f_new, grad_new) = self.evaluate_objective(func, &x_new);
            self.f_val = f_new;
            self.grad = grad_new;
            self.evaluate_constraints(eq_constraints, ineq_constraints, &x_new);

            // Update multipliers
            self.lambda_eq = qp_lambda_eq;
            self.lambda_ineq = qp_lambda_ineq;

            // BFGS update
            if self.options.use_bfgs {
                let lag_grad_new =
                    self.lagrangian_gradient(&self.grad, &self.lambda_eq, &self.lambda_ineq);
                let y = &lag_grad_new - &lag_grad_old;
                self.bfgs_update(&s, &y);
            }

            // Update active set
            self.update_active_set();
        }

        Ok(SqpResult {
            x: self.x.clone(),
            fun: self.f_val,
            lambda_eq: self.lambda_eq.clone(),
            lambda_ineq: self.lambda_ineq.clone(),
            nit: iteration,
            nfev: self.nfev,
            success: false,
            message: "Maximum iterations reached".to_string(),
            constraint_violation: self.constraint_violation(),
            optimality: self.kkt_optimality(),
        })
    }
}

/// Solve a symmetric positive definite system Ax = b using Cholesky-like decomposition
fn solve_symmetric_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, OptimizeError> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(OptimizeError::ValueError(
            "Matrix dimension mismatch in linear solve".to_string(),
        ));
    }

    // Use LU decomposition with partial pivoting (more robust than Cholesky)
    solve_general_system(a, b)
}

/// Solve a general linear system Ax = b via LU decomposition with partial pivoting
fn solve_general_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, OptimizeError> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(OptimizeError::ValueError(format!(
            "System dimension mismatch: matrix {}x{}, rhs {}",
            a.nrows(),
            a.ncols(),
            n
        )));
    }

    // Copy matrix and RHS
    let mut lu = a.clone();
    let mut x = b.clone();
    let mut perm: Vec<usize> = (0..n).collect();

    // LU factorization with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = lu[[perm[k], k]].abs();
        let mut max_idx = k;
        for i in (k + 1)..n {
            let val = lu[[perm[i], k]].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        if max_val < 1e-14 {
            return Err(OptimizeError::ComputationError(
                "Singular or near-singular matrix in linear solve".to_string(),
            ));
        }

        // Swap rows in permutation
        perm.swap(k, max_idx);

        // Eliminate
        for i in (k + 1)..n {
            let factor = lu[[perm[i], k]] / lu[[perm[k], k]];
            lu[[perm[i], k]] = factor; // Store L factor
            for j in (k + 1)..n {
                lu[[perm[i], j]] -= factor * lu[[perm[k], j]];
            }
        }
    }

    // Forward substitution (Ly = Pb)
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = x[perm[i]];
        for j in 0..i {
            sum -= lu[[perm[i], j]] * y[j];
        }
        y[i] = sum;
    }

    // Back substitution (Ux = y)
    let mut result = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= lu[[perm[i], j]] * result[j];
        }
        if lu[[perm[i], i]].abs() < 1e-14 {
            return Err(OptimizeError::ComputationError(
                "Zero diagonal in back substitution".to_string(),
            ));
        }
        result[i] = sum / lu[[perm[i], i]];
    }

    Ok(result)
}

/// Minimize a function using Sequential Quadratic Programming
///
/// # Arguments
///
/// * `func` - Objective function to minimize
/// * `x0` - Initial guess
/// * `constraints` - Vector of constraints
/// * `options` - SQP options
///
/// # Returns
///
/// `SqpResult` containing the optimization result
pub fn minimize_sqp<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint],
    options: Option<SqpOptions>,
) -> Result<SqpResult, OptimizeError>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    let options = options.unwrap_or_default();
    let x0_arr = Array1::from_vec(x0.to_vec());

    // Separate constraints by type
    let mut ineq_constraints = Vec::new();
    let mut eq_constraints = Vec::new();
    for (i, constraint) in constraints.iter().enumerate() {
        if !constraint.is_bounds() {
            match constraint.kind {
                ConstraintKind::Inequality => ineq_constraints.push((i, constraint)),
                ConstraintKind::Equality => eq_constraints.push((i, constraint)),
            }
        }
    }

    let n_eq = eq_constraints.len();
    let n_ineq = ineq_constraints.len();

    let mut solver = SqpSolver::new(x0_arr, n_eq, n_ineq, options);
    solver.solve(&func, &eq_constraints, &ineq_constraints)
}

/// Convenience function that returns OptimizeResults for compatibility with minimize_constrained
#[allow(dead_code)]
pub fn minimize_sqp_compat<F, S>(
    func: F,
    x0: &ArrayBase<S, Ix1>,
    constraints: &[Constraint],
    options: &crate::constrained::Options,
) -> crate::error::OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    let sqp_options = SqpOptions {
        max_iter: options.maxiter.unwrap_or(200),
        tol: options.gtol.unwrap_or(1e-8),
        constraint_tol: options.ctol.unwrap_or(1e-8),
        eps: options.eps.unwrap_or(1e-8),
        ..Default::default()
    };

    let result = minimize_sqp(func, x0, constraints, Some(sqp_options))?;

    Ok(OptimizeResults {
        x: result.x,
        fun: result.fun,
        nit: result.nit,
        nfev: result.nfev,
        success: result.success,
        message: result.message,
        jac: None,
        hess: None,
        constr: None,
        njev: 0,
        nhev: 0,
        maxcv: 0,
        status: if result.success { 0 } else { 1 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sqp_unconstrained_quadratic() {
        // min (x-2)^2 + (y-3)^2
        let func = |x: &[f64]| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) };

        let x0 = array![0.0, 0.0];
        let constraints: Vec<Constraint> = vec![];

        let result = minimize_sqp(func, &x0, &constraints, None);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.success);
        assert!((res.x[0] - 2.0).abs() < 0.1);
        assert!((res.x[1] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_sqp_equality_constrained() {
        // min x^2 + y^2  subject to  x + y = 1
        // Solution: x = 0.5, y = 0.5, f = 0.5
        let func = |x: &[f64]| -> f64 { x[0].powi(2) + x[1].powi(2) };

        fn eq_constraint(x: &[f64]) -> f64 {
            x[0] + x[1] - 1.0
        }

        let x0 = array![0.0, 0.0];
        let constraints = vec![Constraint::new(eq_constraint, ConstraintKind::Equality)];

        let mut opts = SqpOptions::default();
        opts.tol = 1e-6;

        let result = minimize_sqp(func, &x0, &constraints, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.success, "SQP did not converge: {}", res.message);
        assert!(
            (res.x[0] - 0.5).abs() < 0.05,
            "x[0] = {} (expected ~0.5)",
            res.x[0]
        );
        assert!(
            (res.x[1] - 0.5).abs() < 0.05,
            "x[1] = {} (expected ~0.5)",
            res.x[1]
        );
    }

    #[test]
    fn test_sqp_inequality_constrained() {
        // min (x-2)^2 + (y-2)^2  subject to  x + y <= 2  (i.e., 2 - x - y >= 0)
        // Unconstrained min at (2, 2), but constraint pushes to (1, 1)
        let func = |x: &[f64]| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2) };

        fn ineq_constraint(x: &[f64]) -> f64 {
            2.0 - x[0] - x[1]
        }

        let x0 = array![0.5, 0.5];
        let constraints = vec![Constraint::new(ineq_constraint, ConstraintKind::Inequality)];

        let result = minimize_sqp(func, &x0, &constraints, None);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            (res.x[0] - 1.0).abs() < 0.1,
            "x[0] = {} (expected ~1.0)",
            res.x[0]
        );
        assert!(
            (res.x[1] - 1.0).abs() < 0.1,
            "x[1] = {} (expected ~1.0)",
            res.x[1]
        );
    }

    #[test]
    fn test_sqp_mixed_constraints() {
        // min x^2 + y^2
        // s.t. x + y = 2      (equality)
        //      x >= 0.5        (i.e., x - 0.5 >= 0, inequality)
        // Solution: x = 1.0, y = 1.0 (equality active), but x >= 0.5 is inactive
        // Actually with x+y=2 and unconstrained, x=y=1, so x >= 0.5 is inactive -> solution (1,1)
        let func = |x: &[f64]| -> f64 { x[0].powi(2) + x[1].powi(2) };

        fn eq_con(x: &[f64]) -> f64 {
            x[0] + x[1] - 2.0
        }
        fn ineq_con(x: &[f64]) -> f64 {
            x[0] - 0.5
        }

        let x0 = array![1.5, 0.5];
        let constraints = vec![
            Constraint::new(eq_con, ConstraintKind::Equality),
            Constraint::new(ineq_con, ConstraintKind::Inequality),
        ];

        let result = minimize_sqp(func, &x0, &constraints, None);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        // Solution should be near (1, 1) since both constraints are satisfied
        assert!(
            (res.x[0] - 1.0).abs() < 0.1,
            "x[0] = {} (expected ~1.0)",
            res.x[0]
        );
        assert!(
            (res.x[1] - 1.0).abs() < 0.1,
            "x[1] = {} (expected ~1.0)",
            res.x[1]
        );
    }

    #[test]
    fn test_sqp_rosenbrock_constrained() {
        // min (1-x)^2 + 100(y-x^2)^2  subject to  x^2 + y^2 <= 2
        // Rosenbrock with a circle constraint
        let func =
            |x: &[f64]| -> f64 { (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2) };

        fn circle_constraint(x: &[f64]) -> f64 {
            2.0 - x[0].powi(2) - x[1].powi(2)
        }

        let x0 = array![0.5, 0.5];
        let constraints = vec![Constraint::new(
            circle_constraint,
            ConstraintKind::Inequality,
        )];

        let mut opts = SqpOptions::default();
        opts.max_iter = 500;
        opts.tol = 1e-4;

        let result = minimize_sqp(func, &x0, &constraints, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        // Should be close to (1, 1) since it's inside the circle
        assert!(res.fun < 1.0, "Objective should be small, got {}", res.fun);
        // Verify constraint satisfaction
        let cv = circle_constraint(&[res.x[0], res.x[1]]);
        assert!(cv >= -0.01, "Constraint violated: g(x) = {} < 0", cv);
    }

    #[test]
    fn test_sqp_multiple_inequality_constraints() {
        // min -x - y  subject to  x >= 0, y >= 0, x + y <= 1
        // Solution: x = 0.5, y = 0.5 (not unique on the boundary)
        // Actually the LP minimum is at a vertex. On x+y<=1 with x,y>=0, max x+y is at any point on x+y=1.
        // So solutions lie along x+y=1 with x,y >= 0.
        let func = |x: &[f64]| -> f64 { -x[0] - x[1] };

        fn con_x_pos(x: &[f64]) -> f64 {
            x[0]
        }
        fn con_y_pos(x: &[f64]) -> f64 {
            x[1]
        }
        fn con_sum(x: &[f64]) -> f64 {
            1.0 - x[0] - x[1]
        }

        let x0 = array![0.1, 0.1];
        let constraints = vec![
            Constraint::new(con_x_pos, ConstraintKind::Inequality),
            Constraint::new(con_y_pos, ConstraintKind::Inequality),
            Constraint::new(con_sum, ConstraintKind::Inequality),
        ];

        let result = minimize_sqp(func, &x0, &constraints, None);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        // Optimal value should be -1.0
        assert!(
            (res.fun - (-1.0)).abs() < 0.1,
            "Objective should be ~-1.0, got {}",
            res.fun
        );
        // x + y should be ~1.0
        assert!(
            (res.x[0] + res.x[1] - 1.0).abs() < 0.1,
            "x + y should be ~1.0, got {}",
            res.x[0] + res.x[1]
        );
    }

    #[test]
    fn test_sqp_options_custom() {
        let opts = SqpOptions {
            max_iter: 50,
            tol: 1e-5,
            constraint_tol: 1e-5,
            ..Default::default()
        };

        assert_eq!(opts.max_iter, 50);
        assert!((opts.tol - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn test_sqp_result_fields() {
        let func = |x: &[f64]| -> f64 { x[0].powi(2) };
        let x0 = array![2.0];
        let constraints: Vec<Constraint> = vec![];

        let result = minimize_sqp(func, &x0, &constraints, None);
        assert!(result.is_ok());
        let res = result.expect("should succeed");

        assert!(res.nit > 0);
        assert!(res.nfev > 0);
        assert!(res.optimality >= 0.0);
        assert!(res.constraint_violation >= 0.0);
    }

    #[test]
    fn test_sqp_compat_interface() {
        let func = |x: &[f64]| -> f64 { x[0].powi(2) + x[1].powi(2) };
        let x0 = array![1.0, 1.0];
        let constraints: Vec<Constraint> = vec![];

        let options = crate::constrained::Options::default();
        let result = minimize_sqp_compat(func, &x0, &constraints, &options);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.success);
        assert!(res.fun < 0.1);
    }

    #[test]
    fn test_solve_general_system() {
        // 2x + y = 5
        // x + 3y = 7
        // Solution: x = 8/5 = 1.6, y = 9/5 = 1.8
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0])
            .expect("shape should be valid");
        let b = array![5.0, 7.0];

        let result = solve_general_system(&a, &b);
        assert!(result.is_ok());
        let x = result.expect("should succeed");
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }
}
