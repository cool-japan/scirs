//! Interior point methods for constrained optimization
//!
//! This module implements primal-dual interior point methods for solving
//! constrained optimization problems with equality and inequality constraints.

use super::{Constraint, ConstraintKind};
use crate::error::OptimizeError;
use crate::unconstrained::OptimizeResult;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::cell::RefCell;
use std::rc::Rc;

/// Type alias for equality constraint function.
///
/// Carries an explicit lifetime so that callers may pass closures that borrow
/// from the (non-`'static`) `Constraint` slice; the boxed constraint callables
/// (issue #126) cannot be copied out, so they are evaluated by reference.
type EqualityConstraintFn<'a> = dyn FnMut(&ArrayView1<f64>) -> Array1<f64> + 'a;

/// Type alias for equality constraint jacobian function
type EqualityJacobianFn<'a> = dyn FnMut(&ArrayView1<f64>) -> Array2<f64> + 'a;

/// Type alias for inequality constraint function
type InequalityConstraintFn<'a> = dyn FnMut(&ArrayView1<f64>) -> Array1<f64> + 'a;

/// Type alias for inequality constraint jacobian function
type InequalityJacobianFn<'a> = dyn FnMut(&ArrayView1<f64>) -> Array2<f64> + 'a;

/// Type alias for Newton direction result to reduce type complexity
type NewtonDirectionResult = (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>);

/// Interior point method options
#[derive(Debug, Clone)]
pub struct InteriorPointOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for optimality conditions
    pub tol: f64,
    /// Initial barrier parameter
    pub initial_barrier: f64,
    /// Barrier reduction factor
    pub barrier_reduction: f64,
    /// Minimum barrier parameter
    pub min_barrier: f64,
    /// Maximum number of line search iterations
    pub max_ls_iter: usize,
    /// Line search backtracking factor
    pub alpha: f64,
    /// Line search shrinkage factor
    pub beta: f64,
    /// Tolerance for feasibility
    pub feas_tol: f64,
    /// Use Mehrotra's predictor-corrector method
    pub use_mehrotra: bool,
    /// Regularization parameter for KKT system
    pub regularization: f64,
}

impl Default for InteriorPointOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
            initial_barrier: 1.0,
            barrier_reduction: 0.1,
            min_barrier: 1e-10,
            max_ls_iter: 50,
            alpha: 0.3,
            beta: 0.5,
            feas_tol: 1e-8,
            use_mehrotra: true,
            regularization: 1e-8,
        }
    }
}

/// Result from interior point optimization
#[derive(Debug, Clone)]
pub struct InteriorPointResult {
    /// Optimal solution
    pub x: Array1<f64>,
    /// Optimal objective value
    pub fun: f64,
    /// Lagrange multipliers for equality constraints
    pub lambda_eq: Option<Array1<f64>>,
    /// Lagrange multipliers for inequality constraints
    pub lambda_ineq: Option<Array1<f64>>,
    /// Number of iterations
    pub nit: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Success flag
    pub success: bool,
    /// Status message
    pub message: String,
    /// Final barrier parameter
    pub barrier: f64,
    /// Final optimality measure
    pub optimality: f64,
}

/// Interior point solver for constrained optimization
pub struct InteriorPointSolver<'a> {
    /// Number of variables
    n: usize,
    /// Number of equality constraints
    m_eq: usize,
    /// Number of inequality constraints
    m_ineq: usize,
    /// Options
    options: &'a InteriorPointOptions,
    /// Function evaluation counter
    nfev: usize,
    /// BFGS approximation of the Hessian of the Lagrangian, shared by every
    /// KKT-system build (Newton, affine-scaling predictor, and corrector).
    /// Initialized to the identity and refined via [`Self::update_hessian_bfgs`]
    /// once per outer iteration, replacing what used to be a permanent
    /// identity ("could use BFGS" TODO).
    hessian_approx: Array2<f64>,
    /// Penalty weight `nu` for the l1-exact-penalty line-search merit
    /// function (see [`Self::line_search`] / [`Self::merit_value`]).
    /// Monotonically nondecreasing across outer iterations (raised only when
    /// needed so the current step is a guaranteed descent direction of the
    /// merit function, per the standard exact-penalty safeguard: Nocedal &
    /// Wright, *Numerical Optimization*, 2nd ed., eq. 18.36), to avoid
    /// cycling.
    merit_penalty: f64,
}

impl<'a> InteriorPointSolver<'a> {
    /// Create new interior point solver
    pub fn new(n: usize, m_eq: usize, m_ineq: usize, options: &'a InteriorPointOptions) -> Self {
        Self {
            n,
            m_eq,
            m_ineq,
            options,
            nfev: 0,
            hessian_approx: Array2::eye(n),
            merit_penalty: 1.0,
        }
    }

    /// Gradient of the Lagrangian `L(x, λ) = f(x) + λ_eq · c_eq(x) + λ_ineq ·
    /// c_ineq(x)` with respect to `x`, given the objective gradient and
    /// whatever constraint Jacobians/multipliers are active.
    fn lagrangian_gradient(
        &self,
        g: &Array1<f64>,
        j_eq: &Option<Array2<f64>>,
        lambda_eq: &Array1<f64>,
        j_ineq: &Option<Array2<f64>>,
        lambda_ineq: &Array1<f64>,
    ) -> Array1<f64> {
        let mut lag_grad = g.clone();

        if let (Some(j_eq), true) = (j_eq, self.m_eq > 0) {
            lag_grad = &lag_grad + &j_eq.t().dot(lambda_eq);
        }

        if let (Some(j_ineq), true) = (j_ineq, self.m_ineq > 0) {
            lag_grad = &lag_grad + &j_ineq.t().dot(lambda_ineq);
        }

        lag_grad
    }

    /// Update the shared BFGS Hessian-of-the-Lagrangian approximation given
    /// the most recent primal step `s = x_{k+1} - x_k` and the corresponding
    /// change in the Lagrangian gradient `y = ∇_x L(x_{k+1}) - ∇_x L(x_k)`.
    ///
    /// Uses Powell's damped BFGS update (Nocedal & Wright, *Numerical
    /// Optimization*, 2nd ed., Procedure 18.2), which keeps the
    /// approximation symmetric positive definite even when the raw curvature
    /// condition `s^T y > 0` fails -- routine for a Lagrangian Hessian (unlike
    /// an objective Hessian for a convex problem, its sign is not guaranteed).
    fn update_hessian_bfgs(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        let n = self.n;
        if n == 0 {
            return;
        }

        let hs = self.hessian_approx.dot(s);
        let shs = s.dot(&hs);
        if !(shs > 0.0) || !shs.is_finite() {
            // Degenerate curvature in the current approximation: skip this
            // update rather than risk dividing by (near-)zero.
            return;
        }

        let sty = s.dot(y);
        let theta = if sty >= 0.2 * shs {
            1.0
        } else {
            (0.8 * shs) / (shs - sty)
        }
        .clamp(0.0, 1.0);

        let y_damped = if theta >= 1.0 {
            y.clone()
        } else {
            theta * y + (1.0 - theta) * &hs
        };

        let sty_damped = s.dot(&y_damped);
        if sty_damped.abs() < 1e-12 || !sty_damped.is_finite() {
            return;
        }

        for i in 0..n {
            for j in 0..n {
                let updated = self.hessian_approx[[i, j]] - hs[i] * hs[j] / shs
                    + y_damped[i] * y_damped[j] / sty_damped;
                self.hessian_approx[[i, j]] = updated;
            }
        }
    }

    /// Writes the shared BFGS Hessian approximation (plus regularization on
    /// the diagonal) into the leading `n x n` block of a KKT matrix.
    fn write_hessian_block(&self, kkt_matrix: &mut Array2<f64>, reg: f64) {
        for i in 0..self.n {
            for j in 0..self.n {
                kkt_matrix[[i, j]] = self.hessian_approx[[i, j]];
            }
            kkt_matrix[[i, i]] += reg;
        }
    }

    /// Solve the constrained optimization problem
    #[allow(clippy::many_single_char_names)]
    pub fn solve<F, G>(
        &mut self,
        fun: &mut F,
        grad: &mut G,
        mut eq_con: Option<&mut EqualityConstraintFn<'_>>,
        mut eq_jac: Option<&mut EqualityJacobianFn<'_>>,
        mut ineq_con: Option<&mut InequalityConstraintFn<'_>>,
        mut ineq_jac: Option<&mut InequalityJacobianFn<'_>>,
        x0: &Array1<f64>,
    ) -> Result<InteriorPointResult, OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> f64,
        G: FnMut(&ArrayView1<f64>) -> Array1<f64> + ?Sized,
    {
        // Initialize variables
        let mut x = x0.clone();
        let mut s = Array1::ones(self.m_ineq); // Slack variables
        let mut lambda_eq = Array1::zeros(self.m_eq);
        let mut lambda_ineq = Array1::ones(self.m_ineq);
        let mut barrier = self.options.initial_barrier;

        // Initialize iteration counter
        let mut iter = 0;

        // Tracks the point and Lagrangian gradient from the previous
        // iteration so the shared BFGS Hessian approximation can be updated
        // from the actual (step, gradient-change) pair once one is available.
        let mut prev_lag_grad: Option<Array1<f64>> = None;
        let mut prev_x: Option<Array1<f64>> = None;

        // Main interior point loop
        while iter < self.options.max_iter {
            // Evaluate functions and gradients
            let f = fun(&x.view());
            let g = grad(&x.view());
            self.nfev += 2;

            // Evaluate constraints and Jacobians
            let (c_eq, j_eq) = if self.m_eq > 0 && eq_con.is_some() && eq_jac.is_some() {
                let c = eq_con.as_mut().expect("Operation failed")(&x.view());
                let j = eq_jac.as_mut().expect("Operation failed")(&x.view());
                self.nfev += 2;
                (Some(c), Some(j))
            } else {
                (None, None)
            };

            let (c_ineq, j_ineq) = if self.m_ineq > 0 && ineq_con.is_some() && ineq_jac.is_some() {
                let c = ineq_con.as_mut().expect("Operation failed")(&x.view());
                let j = ineq_jac.as_mut().expect("Operation failed")(&x.view());
                self.nfev += 2;
                (Some(c), Some(j))
            } else {
                (None, None)
            };

            // Refine the shared BFGS Hessian-of-the-Lagrangian approximation
            // from the step and gradient change since the previous iterate
            // (nothing to update from on the very first iteration).
            let lag_grad = self.lagrangian_gradient(&g, &j_eq, &lambda_eq, &j_ineq, &lambda_ineq);
            if let (Some(prev_grad), Some(prev_x)) = (prev_lag_grad.as_ref(), prev_x.as_ref()) {
                let y = &lag_grad - prev_grad;
                let step = &x - prev_x;
                self.update_hessian_bfgs(&step, &y);
            }
            prev_lag_grad = Some(lag_grad);
            prev_x = Some(x.clone());

            // Check convergence
            let (optimality, feasibility) = self.compute_convergence_measures(
                &g,
                &c_eq,
                &c_ineq,
                &j_eq,
                &j_ineq,
                &lambda_eq,
                &lambda_ineq,
                &s,
            );

            if optimality < self.options.tol && feasibility < self.options.feas_tol {
                return Ok(InteriorPointResult {
                    x,
                    fun: f,
                    lambda_eq: if self.m_eq > 0 { Some(lambda_eq) } else { None },
                    lambda_ineq: if self.m_ineq > 0 {
                        Some(lambda_ineq)
                    } else {
                        None
                    },
                    nit: iter,
                    nfev: self.nfev,
                    success: true,
                    message: "Optimization terminated successfully.".to_string(),
                    barrier,
                    optimality,
                });
            }

            // Compute search direction
            let (dx, ds, dlambda_eq, dlambda_ineq) = if self.options.use_mehrotra {
                self.compute_mehrotra_direction(
                    &g,
                    &c_eq,
                    &c_ineq,
                    &j_eq,
                    &j_ineq,
                    &s,
                    &lambda_eq,
                    &lambda_ineq,
                    barrier,
                )?
            } else {
                self.compute_newton_direction(
                    &g,
                    &c_eq,
                    &c_ineq,
                    &j_eq,
                    &j_ineq,
                    &s,
                    &lambda_eq,
                    &lambda_ineq,
                    barrier,
                )?
            };

            // Line search: fraction-to-boundary rule + Armijo backtracking on
            // the l1-exact-penalty merit function (see `line_search` /
            // `merit_value`), which rejects steps that reduce the objective
            // while catastrophically worsening constraint feasibility --
            // something a plain-objective Armijo check cannot detect.
            let step_size = self.line_search(
                fun,
                eq_con.as_deref_mut(),
                ineq_con.as_deref_mut(),
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
            )?;

            // Update variables
            x = &x + step_size * &dx;
            if self.m_ineq > 0 {
                s = &s + step_size * &ds;
                lambda_ineq = &lambda_ineq + step_size * &dlambda_ineq;
            }
            if self.m_eq > 0 {
                lambda_eq = &lambda_eq + step_size * &dlambda_eq;
            }

            // Update barrier parameter: adaptive Fiacco-McCormick /
            // Mehrotra-style rule (see `update_barrier_parameter`), replacing
            // the previous unconditional fixed-factor shrink.
            barrier = self.update_barrier_parameter(barrier, &s, &lambda_ineq, optimality);

            iter += 1;
        }

        let final_f = fun(&x.view());
        self.nfev += 1;
        let (final_optimality, final_feasibility) = self.compute_convergence_measures(
            &grad(&x.view()),
            &None,
            &None,
            &None,
            &None,
            &lambda_eq,
            &lambda_ineq,
            &s,
        );
        self.nfev += 1;

        Ok(InteriorPointResult {
            x,
            fun: final_f,
            lambda_eq: if self.m_eq > 0 { Some(lambda_eq) } else { None },
            lambda_ineq: if self.m_ineq > 0 {
                Some(lambda_ineq)
            } else {
                None
            },
            nit: iter,
            nfev: self.nfev,
            success: false,
            message: "Maximum iterations reached.".to_string(),
            barrier,
            optimality: final_optimality,
        })
    }

    /// Compute convergence measures
    fn compute_convergence_measures(
        &self,
        g: &Array1<f64>,
        c_eq: &Option<Array1<f64>>,
        c_ineq: &Option<Array1<f64>>,
        j_eq: &Option<Array2<f64>>,
        j_ineq: &Option<Array2<f64>>,
        lambda_eq: &Array1<f64>,
        lambda_ineq: &Array1<f64>,
        s: &Array1<f64>,
    ) -> (f64, f64) {
        let lag_grad = self.lagrangian_gradient(g, j_eq, lambda_eq, j_ineq, lambda_ineq);
        let optimality = lag_grad.mapv(|x| x.abs()).sum();

        // Feasibility
        let mut feasibility = 0.0;

        if let Some(c_eq) = c_eq {
            feasibility += c_eq.mapv(|x| x.abs()).sum();
        }

        if let (Some(c_ineq), true) = (c_ineq, self.m_ineq > 0) {
            feasibility += (c_ineq + s).mapv(|x| x.abs()).sum();
        }

        // Complementarity: the true (mu -> 0) KKT condition is s_i * lambda_i
        // = 0, not s_i * lambda_i = barrier. Comparing against the *current*
        // barrier is only meaningful as an inner-loop "have we converged
        // this barrier subproblem" check; it is unsound as the overall
        // termination criterion, and actively wrong whenever the search
        // direction doesn't track the outer `barrier` value at all (as with
        // Mehrotra's predictor-corrector, which adapts its own centering
        // parameter internally every iteration and ignores the `barrier`
        // argument entirely -- see `compute_mehrotra_direction`). Comparing
        // against the stale outer `barrier` previously meant the solver
        // could converge to the exact constrained optimum yet still report
        // "maximum iterations reached" forever, because the barrier
        // reduction schedule had frozen (its own trigger condition depends
        // on `optimality`, which the unbounded growth of `lambda_ineq` at a
        // tightly-active constraint could keep permanently large).
        if self.m_ineq > 0 {
            let complementarity = s
                .iter()
                .zip(lambda_ineq.iter())
                .map(|(&si, &li)| (si * li).abs())
                .sum::<f64>();
            feasibility += complementarity;
        }

        (optimality, feasibility)
    }

    /// Adaptive barrier-parameter (`mu`) update.
    ///
    /// Replaces the previous unconditional fixed-factor shrink (`if
    /// optimality < 10*barrier { barrier *= barrier_reduction }`, applied
    /// every time it triggered regardless of how far the iterate actually
    /// was from complementarity) with the standard Fiacco-McCormick /
    /// Mehrotra-style rule `mu <- sigma * (s^T lambda_ineq) / m_ineq`: the
    /// *average complementarity gap* at the current iterate, scaled by a
    /// centering parameter `sigma in (0,1)`.
    ///
    /// `sigma` itself is adapted from how well-centered the current iterate
    /// is, via the minimum-to-mean complementarity ratio `min_i(s_i*lambda_i)
    /// / mean_i(s_i*lambda_i)` -- the same idea Mehrotra's predictor-corrector
    /// uses to set its own centering parameter (`(mu_aff/mu)^3` in
    /// `compute_mehrotra_direction`): a well-centered iterate (ratio near 1,
    /// every pair close to the mean) can afford an aggressive (small) sigma,
    /// while a poorly centered one (ratio near 0, some pair already collapsed
    /// toward the boundary) is throttled toward `sigma -> 1` (mu left almost
    /// unchanged) so the *next* iteration re-centers instead of racing mu
    /// down while some complementarity pair is still badly out of balance.
    ///
    /// Safeguards (standard for this class of scheme; Nocedal & Wright,
    /// *Numerical Optimization*, 2nd ed., §19.3; Wright, *Primal-Dual
    /// Interior-Point Methods*, §5.4): `mu` never increases and never drops
    /// below `options.min_barrier`; a degenerate complementarity gap (zero,
    /// negative from floating-point noise, or non-finite) falls back to the
    /// previous fixed-factor shrink rather than propagating NaN/garbage into
    /// `mu`. As before, `mu` is only touched at all once the outer iterate's
    /// optimality residual is already within an order of magnitude of it --
    /// shrinking `mu` while the Newton step itself is still far from
    /// converged just makes the *next* KKT system pathologically
    /// ill-conditioned for no benefit.
    fn update_barrier_parameter(
        &self,
        barrier: f64,
        s: &Array1<f64>,
        lambda_ineq: &Array1<f64>,
        optimality: f64,
    ) -> f64 {
        if !(optimality < 10.0 * barrier) {
            return barrier;
        }

        if self.m_ineq == 0 {
            // No slacks/complementarity to track (equality-only or fully
            // unconstrained problem): fall back to the previous fixed-factor
            // shrink, gated the same way, purely so `barrier` still moves
            // toward `min_barrier` at a bounded rate for reporting purposes.
            return (barrier * self.options.barrier_reduction).max(self.options.min_barrier);
        }

        let m = self.m_ineq as f64;
        let products: Vec<f64> = s
            .iter()
            .zip(lambda_ineq.iter())
            .map(|(&si, &li)| si * li)
            .collect();
        let gap_mean = products.iter().sum::<f64>() / m;

        if !(gap_mean > 0.0) || !gap_mean.is_finite() {
            return (barrier * self.options.barrier_reduction).max(self.options.min_barrier);
        }

        let min_gap = products.iter().copied().fold(f64::INFINITY, f64::min);
        let centering_ratio = (min_gap / gap_mean).clamp(0.0, 1.0);

        // Mehrotra-style adaptive centering exponent (cubic damping, as in
        // the classical `sigma = (mu_aff/mu)^3` heuristic): centered ->
        // sigma near 0 (aggressive shrink); uncentered -> sigma near 1 (mu
        // barely moves). Clamped away from the extremes so a single
        // iteration can neither collapse mu numerically nor stall it
        // completely.
        let sigma = (1.0 - centering_ratio).powi(3).clamp(1e-3, 0.9);

        let mu_candidate = sigma * gap_mean;

        mu_candidate.clamp(self.options.min_barrier, barrier)
    }

    /// Compute Newton direction for the KKT system
    fn compute_newton_direction(
        &self,
        g: &Array1<f64>,
        c_eq: &Option<Array1<f64>>,
        c_ineq: &Option<Array1<f64>>,
        j_eq: &Option<Array2<f64>>,
        j_ineq: &Option<Array2<f64>>,
        s: &Array1<f64>,
        lambda_eq: &Array1<f64>,
        lambda_ineq: &Array1<f64>,
        barrier: f64,
    ) -> Result<NewtonDirectionResult, OptimizeError> {
        // Build KKT system
        let n_total = self.n + self.m_eq + 2 * self.m_ineq;
        let mut kkt_matrix = Array2::zeros((n_total, n_total));
        let mut rhs = Array1::zeros(n_total);

        // Add regularization to ensure positive definiteness
        let reg = self.options.regularization.max(1e-8);

        // Hessian of the Lagrangian: the shared BFGS approximation refined
        // across outer iterations (see `update_hessian_bfgs`), regularized
        // on the diagonal for conditioning.
        self.write_hessian_block(&mut kkt_matrix, reg);

        // Stationarity residual: -∇_x L(x, λ) = -(g + J_eq^T λ_eq + J_ineq^T
        // λ_ineq), NOT just -g. A previous version of this function used
        // `-g` alone, silently dropping the existing multipliers'
        // contribution to the Newton residual (correct only in the
        // degenerate case λ = 0); since λ moves away from its initial value
        // within the very first iteration, this made every subsequent
        // Newton step solve a subtly wrong linear system and converge to a
        // spurious fixed point instead of the true KKT point.
        let lag_grad = self.lagrangian_gradient(g, j_eq, lambda_eq, j_ineq, lambda_ineq);
        for i in 0..self.n {
            rhs[i] = -lag_grad[i];
        }

        // Column/row layout for the remaining blocks must match how the
        // solution vector is sliced below: [dx (n) | ds (m_ineq) | dλ_eq
        // (m_eq) | dλ_ineq (m_ineq)]. `ds` occupies `[n, n+m_ineq)`
        // (referenced directly as `self.n + i` below), so the dλ_eq/dλ_ineq
        // row_offset must start *after* that block, not right after `dx` --
        // starting it at `self.n` (as a previous version of this function
        // did) made `ds`'s columns silently alias with `dλ_eq`'s (or, when
        // `m_eq == 0`, with `dλ_ineq`'s), corrupting the KKT system for any
        // problem with inequality constraints.
        let mut row_offset = self.n + self.m_ineq;

        // Equality constraints
        if let (Some(j_eq), Some(c_eq), true) = (j_eq, c_eq, self.m_eq > 0) {
            // J_eq^T in upper right
            for i in 0..self.m_eq {
                for j in 0..self.n {
                    kkt_matrix[[j, row_offset + i]] = j_eq[[i, j]];
                    kkt_matrix[[row_offset + i, j]] = j_eq[[i, j]];
                }
            }

            // RHS for equality constraints
            for i in 0..self.m_eq {
                rhs[row_offset + i] = -c_eq[i];
            }

            row_offset += self.m_eq;
        }

        // Inequality constraints
        if let (Some(j_ineq), Some(c_ineq), true) = (j_ineq, c_ineq, self.m_ineq > 0) {
            // J_ineq^T in upper right (stationarity <-> dλ_ineq coupling)
            for i in 0..self.m_ineq {
                for j in 0..self.n {
                    kkt_matrix[[j, row_offset + i]] = j_ineq[[i, j]];
                    kkt_matrix[[row_offset + i, j]] = j_ineq[[i, j]];
                }
            }

            // RHS for inequality constraints (unscaled: c_ineq(x) + s = 0)
            for i in 0..self.m_ineq {
                rhs[row_offset + i] = -(c_ineq[i] + s[i]);
            }

            // Complementarity conditions (row-scaled by 1/s_i for numerical
            // stability: dividing `λ_i·ds_i + s_i·dλ_ineq_i = μ - s_iλ_i`
            // through by s_i gives diagonal coefficient λ_i/s_i on ds_i and
            // exactly 1 on dλ_ineq_i). The ds<->dλ_ineq coupling is
            // symmetric identity in both this (scaled) complementarity row
            // and the (unscaled) "+ds" term of the inequality-feasibility
            // row above -- NOT s_i/λ_i, which would conflate the two rows'
            // independent scalings.
            for i in 0..self.m_ineq {
                // Avoid division by very small slack variables
                let s_i = s[i].max(1e-10);
                let lambda_i = lambda_ineq[i].max(0.0);

                kkt_matrix[[self.n + i, self.n + i]] = lambda_i / s_i + reg;
                kkt_matrix[[self.n + i, row_offset + i]] = 1.0;
                kkt_matrix[[row_offset + i, self.n + i]] = 1.0;
                rhs[self.n + i] = barrier / s_i - lambda_i;
            }
        }

        // Solve KKT system
        let solution = solve(&kkt_matrix, &rhs)?;

        // Extract components
        let dx = solution
            .slice(scirs2_core::ndarray::s![0..self.n])
            .to_owned();
        let ds = if self.m_ineq > 0 {
            solution
                .slice(scirs2_core::ndarray::s![self.n..self.n + self.m_ineq])
                .to_owned()
        } else {
            Array1::zeros(0)
        };

        let mut offset = self.n + self.m_ineq;
        let dlambda_eq = if self.m_eq > 0 {
            solution
                .slice(scirs2_core::ndarray::s![offset..offset + self.m_eq])
                .to_owned()
        } else {
            Array1::zeros(0)
        };

        offset += self.m_eq;
        let dlambda_ineq = if self.m_ineq > 0 {
            solution
                .slice(scirs2_core::ndarray::s![offset..offset + self.m_ineq])
                .to_owned()
        } else {
            Array1::zeros(0)
        };

        Ok((dx, ds, dlambda_eq, dlambda_ineq))
    }

    /// Compute Mehrotra's predictor-corrector direction
    ///
    /// This implements the full Mehrotra algorithm with predictor and corrector steps:
    /// 1. Compute predictor step (affine scaling direction)
    /// 2. Estimate complementarity gap after predictor step
    /// 3. Compute centering parameter based on gap reduction
    /// 4. Compute corrector step combining predictor and centering
    fn compute_mehrotra_direction(
        &self,
        g: &Array1<f64>,
        c_eq: &Option<Array1<f64>>,
        c_ineq: &Option<Array1<f64>>,
        j_eq: &Option<Array2<f64>>,
        j_ineq: &Option<Array2<f64>>,
        s: &Array1<f64>,
        lambda_eq: &Array1<f64>,
        lambda_ineq: &Array1<f64>,
        _barrier: f64,
    ) -> Result<NewtonDirectionResult, OptimizeError> {
        if self.m_ineq == 0 {
            // No inequality constraints, use standard Newton direction
            return self.compute_newton_direction(
                g,
                c_eq,
                c_ineq,
                j_eq,
                j_ineq,
                s,
                lambda_eq,
                lambda_ineq,
                0.0,
            );
        }

        // Step 1: Compute predictor step (affine scaling direction)
        // This is the Newton step with zero _barrier parameter (affine scaling)
        let (dx_aff, ds_aff, dlambda_eq_aff, dlambda_ineq_aff) = self
            .compute_affine_scaling_direction(
                g,
                c_eq,
                c_ineq,
                j_eq,
                j_ineq,
                s,
                lambda_eq,
                lambda_ineq,
            )?;

        // Step 2: Compute maximum step lengths for predictor step
        let alpha_primal_max = self.compute_max_step_primal(s, &ds_aff);
        let alpha_dual_max = self.compute_max_step_dual(lambda_ineq, &dlambda_ineq_aff);

        // Step 3: Estimate complementarity gap after predictor step
        let current_gap = s
            .iter()
            .zip(lambda_ineq.iter())
            .map(|(&si, &li)| si * li)
            .sum::<f64>();
        let mu = current_gap / (self.m_ineq as f64);

        // Predict gap after affine step
        let mut predicted_gap = 0.0;
        for i in 0..self.m_ineq {
            let s_new = s[i] + alpha_primal_max * ds_aff[i];
            let lambda_new = lambda_ineq[i] + alpha_dual_max * dlambda_ineq_aff[i];
            predicted_gap += s_new * lambda_new;
        }

        let mu_aff = predicted_gap / (self.m_ineq as f64);

        // Step 4: Compute centering parameter using Mehrotra's heuristic
        let sigma = if mu > 0.0 {
            (mu_aff / mu).powi(3)
        } else {
            0.1 // Default centering when current gap is zero
        };

        // Ensure sigma is in reasonable bounds
        let sigma = sigma.max(0.0).min(1.0);

        // Step 5: Compute target _barrier parameter for corrector step
        let sigma_mu = sigma * mu;

        // Step 6: Compute corrector step
        // This combines the predictor direction with centering and second-order corrections
        self.compute_corrector_direction(
            g,
            c_eq,
            c_ineq,
            j_eq,
            j_ineq,
            s,
            lambda_ineq,
            &dx_aff,
            &ds_aff,
            &dlambda_ineq_aff,
            sigma_mu,
        )
    }

    /// Compute affine scaling direction (predictor step)
    fn compute_affine_scaling_direction(
        &self,
        g: &Array1<f64>,
        c_eq: &Option<Array1<f64>>,
        c_ineq: &Option<Array1<f64>>,
        j_eq: &Option<Array2<f64>>,
        j_ineq: &Option<Array2<f64>>,
        s: &Array1<f64>,
        lambda_eq: &Array1<f64>,
        lambda_ineq: &Array1<f64>,
    ) -> Result<NewtonDirectionResult, OptimizeError> {
        // Build KKT system for affine scaling (barrier = 0)
        let n_total = self.n + self.m_eq + 2 * self.m_ineq;
        let mut kkt_matrix = Array2::zeros((n_total, n_total));
        let mut rhs = Array1::zeros(n_total);

        let reg = self.options.regularization.max(1e-8);

        // Hessian of the Lagrangian: shared BFGS approximation + regularization.
        self.write_hessian_block(&mut kkt_matrix, reg);

        // Stationarity residual: -∇_x L(x, λ), not just -g (see the matching
        // comment in `compute_newton_direction`).
        let lag_grad = self.lagrangian_gradient(g, j_eq, lambda_eq, j_ineq, lambda_ineq);
        for i in 0..self.n {
            rhs[i] = -lag_grad[i];
        }

        // See the matching comment in `compute_newton_direction`: this
        // offset must start *after* the `ds` block at `[n, n+m_ineq)` (used
        // directly as `self.n + i` below) to match how the solution vector
        // is sliced in `extract_direction_components`.
        let mut row_offset = self.n + self.m_ineq;

        // Equality constraints
        if let (Some(j_eq), Some(c_eq), true) = (j_eq, c_eq, self.m_eq > 0) {
            for i in 0..self.m_eq {
                for j in 0..self.n {
                    kkt_matrix[[j, row_offset + i]] = j_eq[[i, j]];
                    kkt_matrix[[row_offset + i, j]] = j_eq[[i, j]];
                }
            }

            for i in 0..self.m_eq {
                rhs[row_offset + i] = -c_eq[i];
            }

            row_offset += self.m_eq;
        }

        // Inequality constraints
        if let (Some(j_ineq), Some(c_ineq), true) = (j_ineq, c_ineq, self.m_ineq > 0) {
            for i in 0..self.m_ineq {
                for j in 0..self.n {
                    kkt_matrix[[j, row_offset + i]] = j_ineq[[i, j]];
                    kkt_matrix[[row_offset + i, j]] = j_ineq[[i, j]];
                }
            }

            for i in 0..self.m_ineq {
                rhs[row_offset + i] = -(c_ineq[i] + s[i]);
            }

            // Complementarity conditions for affine scaling (no barrier
            // term); see `compute_newton_direction` for why the ds<->dλ_ineq
            // coupling must be the identity 1.0, not s_i/λ_i.
            for i in 0..self.m_ineq {
                let s_i = s[i].max(1e-10);
                let lambda_i = lambda_ineq[i].max(0.0);

                kkt_matrix[[self.n + i, self.n + i]] = lambda_i / s_i + reg;
                kkt_matrix[[self.n + i, row_offset + i]] = 1.0;
                kkt_matrix[[row_offset + i, self.n + i]] = 1.0;

                // RHS for affine scaling: -s_i * lambda_i (no barrier term),
                // row-scaled by 1/s_i like the diagonal above.
                rhs[self.n + i] = -lambda_i;
            }
        }

        // Solve KKT system
        let solution = solve(&kkt_matrix, &rhs)?;

        // Extract components
        self.extract_direction_components(&solution)
    }

    /// Compute corrector direction combining predictor and centering
    fn compute_corrector_direction(
        &self,
        self_g: &Array1<f64>,
        _c_eq: &Option<Array1<f64>>,
        _c_ineq: &Option<Array1<f64>>,
        j_eq: &Option<Array2<f64>>,
        j_ineq: &Option<Array2<f64>>,
        s: &Array1<f64>,
        lambda_ineq: &Array1<f64>,
        dx_aff: &Array1<f64>,
        ds_aff: &Array1<f64>,
        dlambda_ineq_aff: &Array1<f64>,
        sigma_mu: f64,
    ) -> Result<NewtonDirectionResult, OptimizeError> {
        // Build KKT system for corrector step
        let n_total = self.n + self.m_eq + 2 * self.m_ineq;
        let mut kkt_matrix = Array2::zeros((n_total, n_total));
        let mut rhs = Array1::zeros(n_total);

        let reg = self.options.regularization.max(1e-8);

        // Hessian of the Lagrangian: shared BFGS approximation + regularization.
        self.write_hessian_block(&mut kkt_matrix, reg);

        // Gradient of Lagrangian (zero for corrector)
        for i in 0..self.n {
            rhs[i] = 0.0;
        }

        // See the matching comment in `compute_newton_direction`: this
        // offset must start *after* the `ds` block at `[n, n+m_ineq)` (used
        // directly as `self.n + i` below) to match how the solution vector
        // is sliced in `extract_direction_components`.
        let mut row_offset = self.n + self.m_ineq;

        // Equality constraints (zero RHS for corrector)
        if let (Some(j_eq), true) = (j_eq, self.m_eq > 0) {
            for i in 0..self.m_eq {
                for j in 0..self.n {
                    kkt_matrix[[j, row_offset + i]] = j_eq[[i, j]];
                    kkt_matrix[[row_offset + i, j]] = j_eq[[i, j]];
                }
            }

            for i in 0..self.m_eq {
                rhs[row_offset + i] = 0.0;
            }

            row_offset += self.m_eq;
        }

        // Inequality constraints (zero RHS for corrector)
        if let (Some(j_ineq), true) = (j_ineq, self.m_ineq > 0) {
            for i in 0..self.m_ineq {
                for j in 0..self.n {
                    kkt_matrix[[j, row_offset + i]] = j_ineq[[i, j]];
                    kkt_matrix[[row_offset + i, j]] = j_ineq[[i, j]];
                }
            }

            for i in 0..self.m_ineq {
                rhs[row_offset + i] = 0.0;
            }

            // Complementarity conditions with centering and second-order
            // corrections; see `compute_newton_direction` for why the
            // ds<->dλ_ineq coupling must be the identity 1.0, not s_i/λ_i.
            for i in 0..self.m_ineq {
                let s_i = s[i].max(1e-10);
                let lambda_i = lambda_ineq[i].max(0.0);

                kkt_matrix[[self.n + i, self.n + i]] = lambda_i / s_i + reg;
                kkt_matrix[[self.n + i, row_offset + i]] = 1.0;
                kkt_matrix[[row_offset + i, self.n + i]] = 1.0;

                // RHS includes centering term and second-order correction
                // sigma_mu - ds_aff[i] * dlambda_ineq_aff[i]
                let correction = sigma_mu - ds_aff[i] * dlambda_ineq_aff[i];
                rhs[self.n + i] = correction / s_i;
            }
        }

        // Solve KKT system
        let solution = solve(&kkt_matrix, &rhs)?;

        // Extract components and combine with predictor step
        let (dx_cor, ds_cor, dlambda_eq_cor, dlambda_ineq_cor) =
            self.extract_direction_components(&solution)?;

        // Combine predictor and corrector steps
        let dx_final = dx_aff + &dx_cor;
        let ds_final = ds_aff + &ds_cor;
        let dlambda_eq_final = &Array1::zeros(self.m_eq) + &dlambda_eq_cor;
        let dlambda_ineq_final = dlambda_ineq_aff + &dlambda_ineq_cor;

        Ok((dx_final, ds_final, dlambda_eq_final, dlambda_ineq_final))
    }

    /// Extract direction components from KKT solution
    fn extract_direction_components(
        &self,
        solution: &Array1<f64>,
    ) -> Result<NewtonDirectionResult, OptimizeError> {
        let dx = solution
            .slice(scirs2_core::ndarray::s![0..self.n])
            .to_owned();
        let ds = if self.m_ineq > 0 {
            solution
                .slice(scirs2_core::ndarray::s![self.n..self.n + self.m_ineq])
                .to_owned()
        } else {
            Array1::zeros(0)
        };

        let mut offset = self.n + self.m_ineq;
        let dlambda_eq = if self.m_eq > 0 {
            solution
                .slice(scirs2_core::ndarray::s![offset..offset + self.m_eq])
                .to_owned()
        } else {
            Array1::zeros(0)
        };

        offset += self.m_eq;
        let dlambda_ineq = if self.m_ineq > 0 {
            solution
                .slice(scirs2_core::ndarray::s![offset..offset + self.m_ineq])
                .to_owned()
        } else {
            Array1::zeros(0)
        };

        Ok((dx, ds, dlambda_eq, dlambda_ineq))
    }

    /// Compute maximum step length for primal variables
    fn compute_max_step_primal(&self, s: &Array1<f64>, ds: &Array1<f64>) -> f64 {
        if self.m_ineq == 0 {
            return 1.0;
        }

        let tau = 0.995; // Fraction to boundary parameter
        let mut alpha = 1.0;

        for i in 0..self.m_ineq {
            if ds[i] < 0.0 {
                alpha = f64::min(alpha, -tau * s[i] / ds[i]);
            }
        }

        alpha.max(0.0).min(1.0)
    }

    /// Compute maximum step length for dual variables
    fn compute_max_step_dual(&self, lambda_ineq: &Array1<f64>, dlambda_ineq: &Array1<f64>) -> f64 {
        if self.m_ineq == 0 {
            return 1.0;
        }

        let tau = 0.995; // Fraction to boundary parameter
        let mut alpha = 1.0;

        for i in 0..self.m_ineq {
            if dlambda_ineq[i] < 0.0 {
                alpha = f64::min(alpha, -tau * lambda_ineq[i] / dlambda_ineq[i]);
            }
        }

        alpha.max(0.0).min(1.0)
    }

    /// L1 norm of the constraint violation at a point: `||c_eq||_1 +
    /// ||c_ineq + s||_1`. Shared by [`Self::merit_value`] and the
    /// penalty-parameter safeguard in [`Self::line_search`].
    fn l1_infeasibility(c_eq: &Option<Array1<f64>>, c_ineq_plus_s: &Option<Array1<f64>>) -> f64 {
        let mut infeas = 0.0;
        if let Some(c_eq) = c_eq {
            infeas += c_eq.iter().map(|v| v.abs()).sum::<f64>();
        }
        if let Some(c) = c_ineq_plus_s {
            infeas += c.iter().map(|v| v.abs()).sum::<f64>();
        }
        infeas
    }

    /// The interior-point line-search merit function `phi_mu,nu(x, s) = f(x)
    /// - mu * sum_i ln(s_i) + nu * (||c_eq(x)||_1 + ||c_ineq(x) + s||_1)`:
    /// the standard l1-exact-penalty merit function for primal-dual
    /// interior-point line searches (Nocedal & Wright, *Numerical
    /// Optimization*, 2nd ed., §19.6), blending the log-barrier subproblem
    /// objective with an exact l1 penalty on constraint violation.
    fn merit_value(
        &self,
        f: f64,
        s: &Array1<f64>,
        barrier: f64,
        c_eq: &Option<Array1<f64>>,
        c_ineq_plus_s: &Option<Array1<f64>>,
        nu: f64,
    ) -> f64 {
        let mut phi = f;

        if self.m_ineq > 0 {
            phi -= barrier * s.iter().map(|&si| si.max(1e-300).ln()).sum::<f64>();
        }

        phi + nu * Self::l1_infeasibility(c_eq, c_ineq_plus_s)
    }

    /// Line search combining the fraction-to-boundary rule with Armijo
    /// backtracking on an l1-exact-penalty merit function.
    ///
    /// `alpha` is first capped, as before, so that `s + alpha*ds` and
    /// `lambda_ineq + alpha*dlambda_ineq` stay strictly positive (the
    /// fraction-to-boundary rule). Within that cap, `alpha` is chosen by
    /// Armijo backtracking on the merit function `phi_mu,nu(x, s)`
    /// ([`Self::merit_value`]) rather than on the raw objective alone.
    ///
    /// Previously this method accepted any step satisfying `f_new <= f0 +
    /// alpha_param * alpha * dx.dot(dx)`. Since `dx.dot(dx) >= 0` always, that
    /// right-hand side is *greater* than `f0`, so the check accepted any step
    /// whose objective merely didn't increase by more than an arbitrary
    /// positive slack -- not a real sufficient-decrease test at all -- and
    /// entirely regardless of constraint feasibility (the objective was the
    /// only thing ever evaluated at the trial point). A step that drove the
    /// objective down while catastrophically violating the constraints was
    /// therefore always accepted outright, at the largest fraction-to-
    /// boundary step available. The merit-function check rejects such steps:
    /// growing the l1 constraint-violation term is enough to fail the Armijo
    /// condition on `phi`, forcing backtracking instead (see
    /// `test_line_search_rejects_catastrophic_constraint_violation`).
    ///
    /// The penalty weight `nu` (`self.merit_penalty`) is the standard
    /// exact-penalty safeguard (Nocedal & Wright eq. 18.36): raised, when
    /// necessary, so that the supplied `(dx, ds)` -- which, as a genuine
    /// Newton/Mehrotra step from this module's KKT solves, (approximately)
    /// satisfies the linearized feasibility equations `J_eq dx = -c_eq` and
    /// `J_ineq dx + ds = -(c_ineq + s)` -- is guaranteed to be a descent
    /// direction of `phi`. `nu` is kept monotonically nondecreasing across
    /// calls (i.e. across outer iterations) to avoid cycling.
    #[allow(clippy::too_many_arguments)]
    fn line_search<F>(
        &mut self,
        fun: &mut F,
        mut eq_con: Option<&mut EqualityConstraintFn<'_>>,
        mut ineq_con: Option<&mut InequalityConstraintFn<'_>>,
        x: &Array1<f64>,
        s: &Array1<f64>,
        lambda_ineq: &Array1<f64>,
        dx: &Array1<f64>,
        ds: &Array1<f64>,
        dlambda_ineq: &Array1<f64>,
        g: &Array1<f64>,
        c_eq: &Option<Array1<f64>>,
        c_ineq: &Option<Array1<f64>>,
        barrier: f64,
    ) -> Result<f64, OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> f64,
    {
        // Fraction to boundary rule
        let tau = 0.995;
        let mut alpha_primal = 1.0;
        let mut alpha_dual = 1.0;

        // Maximum step to maintain positivity of slack variables
        if self.m_ineq > 0 {
            for i in 0..self.m_ineq {
                if ds[i] < 0.0 {
                    alpha_primal = f64::min(alpha_primal, -tau * s[i] / ds[i]);
                }
                if dlambda_ineq[i] < 0.0 {
                    alpha_dual = f64::min(alpha_dual, -tau * lambda_ineq[i] / dlambda_ineq[i]);
                }
            }
        }

        let alpha_max = f64::min(alpha_primal, alpha_dual).clamp(0.0, 1.0);

        // Merit function value and its directional derivative at the current
        // point, along (dx, ds).
        let f0 = fun(&x.view());
        self.nfev += 1;

        let c_ineq_plus_s0 = c_ineq.as_ref().map(|c| c + s);
        let infeas0 = Self::l1_infeasibility(c_eq, &c_ineq_plus_s0);

        // D(f - mu*sum(ln s); (dx,ds)) = g.dot(dx) - mu*sum(ds_i/s_i).
        let barrier_dir_deriv = g.dot(dx)
            - if self.m_ineq > 0 {
                barrier
                    * s.iter()
                        .zip(ds.iter())
                        .map(|(&si, &dsi)| dsi / si.max(1e-300))
                        .sum::<f64>()
            } else {
                0.0
            };

        // Exact-penalty safeguard (Nocedal & Wright eq. 18.36): raise `nu`
        // just enough that `(dx, ds)` is a descent direction of `phi`, given
        // that the l1 term's directional derivative is `-infeas0` whenever
        // the linearized feasibility equations are (approximately) satisfied
        // -- true for the Newton/Mehrotra steps this method is called with.
        let margin = 0.1;
        if infeas0 > 1e-12 {
            let nu_required = (barrier_dir_deriv / ((1.0 - margin) * infeas0)).max(0.0);
            if self.merit_penalty < nu_required {
                self.merit_penalty = nu_required + 1.0;
            }
        }
        let nu = self.merit_penalty;

        // Defensive fallback: a legitimate descent direction always makes
        // this negative (see above), but guard against surprises (e.g.
        // severe regularization-induced KKT-solve error) forcing a spurious
        // non-negative value back into an overly permissive Armijo
        // threshold -- exactly the failure mode this rewrite fixes.
        let dir_deriv = {
            let d = barrier_dir_deriv - nu * infeas0;
            if d.is_finite() && d < 0.0 {
                d
            } else {
                -1e-10 * (1.0 + f0.abs())
            }
        };

        let phi0 = self.merit_value(f0, s, barrier, c_eq, &c_ineq_plus_s0, nu);

        // Only re-evaluate a constraint at trial points if it was actually
        // supplied *and* treated as active this iteration (i.e. `c_eq`/
        // `c_ineq` at the current point is `Some`, matching how the KKT
        // system that produced `dx`/`ds` was built).
        let eval_eq = c_eq.is_some();
        let eval_ineq = c_ineq.is_some();

        let mut alpha = alpha_max;

        for _ in 0..self.options.max_ls_iter {
            let x_new = x + alpha * dx;
            let f_new = fun(&x_new.view());
            self.nfev += 1;

            let s_new = if self.m_ineq > 0 {
                s + alpha * ds
            } else {
                s.clone()
            };

            let c_eq_new: Option<Array1<f64>> = if eval_eq {
                if let Some(f) = eq_con.as_mut() {
                    self.nfev += 1;
                    Some(f(&x_new.view()))
                } else {
                    None
                }
            } else {
                None
            };

            let c_ineq_new: Option<Array1<f64>> = if eval_ineq {
                if let Some(f) = ineq_con.as_mut() {
                    self.nfev += 1;
                    Some(f(&x_new.view()))
                } else {
                    None
                }
            } else {
                None
            };
            let c_ineq_plus_s_new = c_ineq_new.map(|c| c + &s_new);

            let phi_new =
                self.merit_value(f_new, &s_new, barrier, &c_eq_new, &c_ineq_plus_s_new, nu);

            if phi_new <= phi0 + self.options.alpha * alpha * dir_deriv {
                return Ok(alpha);
            }

            alpha *= self.options.beta;
        }

        Ok(alpha)
    }
}

/// Solve linear system using LU decomposition from scirs2-linalg
#[allow(dead_code)]
fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, OptimizeError> {
    use scirs2_linalg::solve;

    solve(&a.view(), &b.view(), None)
        .map_err(|e| OptimizeError::ComputationError(format!("Linear system solve failed: {}", e)))
}

/// Minimize a function subject to constraints using interior point method
///
/// Both the equality and inequality constraint callbacks (and their
/// Jacobians) are wired all the way through to the solver: previously this
/// convenience wrapper used `eq_con`/`ineq_con` only to *count* constraints
/// (to size the KKT system) but then unconditionally passed `None` for every
/// constraint callback into [`InteriorPointSolver::solve`], so any supplied
/// constraint was silently never enforced. When a constraint is supplied
/// without its own Jacobian, a forward finite-difference approximation of
/// that specific constraint is used instead of dropping it.
///
/// If `grad_fn` is `None`, the gradient is estimated via forward finite
/// differences (as before); when supplied, the analytical gradient is used
/// directly instead of unconditionally re-deriving it via finite
/// differences.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn minimize_interior_point<F, G, H, J>(
    fun: F,
    grad_fn: Option<G>,
    x0: Array1<f64>,
    eq_con: Option<H>,
    eq_jac: Option<J>,
    ineq_con: Option<H>,
    ineq_jac: Option<J>,
    options: Option<InteriorPointOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64 + Clone,
    G: FnMut(&ArrayView1<f64>) -> Array1<f64>,
    H: FnMut(&ArrayView1<f64>) -> Array1<f64>,
    J: FnMut(&ArrayView1<f64>) -> Array2<f64>,
{
    let options = options.unwrap_or_default();
    let n = x0.len();

    // Each constraint callback may return any number of constraint rows (it
    // is `Array1`-valued, not scalar); determine the real dimensionality by
    // evaluating it once at `x0` instead of assuming exactly one row
    // whenever a callback is present. Getting this wrong silently
    // undersizes `s`/`lambda_ineq`/the KKT system versus what `j_ineq`
    // actually is, which previously went undetected only because the
    // constraint callbacks were never even passed to the solver (see below).
    let mut eq_con = eq_con;
    let mut ineq_con = ineq_con;
    let m_eq = match eq_con.as_mut() {
        Some(f) => f(&x0.view()).len(),
        None => 0,
    };
    let m_ineq = match ineq_con.as_mut() {
        Some(f) => f(&x0.view()).len(),
        None => 0,
    };

    // Create solver
    let mut solver = InteriorPointSolver::new(n, m_eq, m_ineq, &options);

    // Prepare function and gradient: use the caller's analytical gradient
    // when supplied, falling back to forward finite differences otherwise.
    let mut fun_mut = fun.clone();
    let eps = 1e-8;
    let mut fd_fun = fun.clone();
    let mut grad_owned: Box<dyn FnMut(&ArrayView1<f64>) -> Array1<f64>> = match grad_fn {
        Some(g) => Box::new(g),
        None => Box::new(move |x: &ArrayView1<f64>| finite_diff_gradient(&mut fd_fun, x, eps)),
    };

    // Each constraint callback is shared behind `Rc<RefCell<_>>` so it can be
    // evaluated both for its own value and, when no analytical Jacobian is
    // supplied, again at perturbed points for a finite-difference Jacobian --
    // without requiring `H: Clone`.
    let eq_con_shared = eq_con.map(|f| Rc::new(RefCell::new(f)));
    let ineq_con_shared = ineq_con.map(|f| Rc::new(RefCell::new(f)));

    let mut eq_con_owned: Option<Box<dyn FnMut(&ArrayView1<f64>) -> Array1<f64>>> =
        eq_con_shared.as_ref().map(|shared| {
            let shared = Rc::clone(shared);
            Box::new(move |x: &ArrayView1<f64>| shared.borrow_mut()(x))
                as Box<dyn FnMut(&ArrayView1<f64>) -> Array1<f64>>
        });
    let mut eq_jac_owned: Option<Box<dyn FnMut(&ArrayView1<f64>) -> Array2<f64>>> =
        match (eq_jac, eq_con_shared.as_ref()) {
            (Some(j), _) => Some(Box::new(j)),
            (None, Some(shared)) => {
                let shared = Rc::clone(shared);
                Some(Box::new(move |x: &ArrayView1<f64>| {
                    let mut con = |xv: &ArrayView1<f64>| shared.borrow_mut()(xv);
                    finite_diff_jacobian_fn(&mut con, x, eps)
                }))
            }
            (None, None) => None,
        };

    let mut ineq_con_owned: Option<Box<dyn FnMut(&ArrayView1<f64>) -> Array1<f64>>> =
        ineq_con_shared.as_ref().map(|shared| {
            let shared = Rc::clone(shared);
            Box::new(move |x: &ArrayView1<f64>| shared.borrow_mut()(x))
                as Box<dyn FnMut(&ArrayView1<f64>) -> Array1<f64>>
        });
    let mut ineq_jac_owned: Option<Box<dyn FnMut(&ArrayView1<f64>) -> Array2<f64>>> =
        match (ineq_jac, ineq_con_shared.as_ref()) {
            (Some(j), _) => Some(Box::new(j)),
            (None, Some(shared)) => {
                let shared = Rc::clone(shared);
                Some(Box::new(move |x: &ArrayView1<f64>| {
                    let mut con = |xv: &ArrayView1<f64>| shared.borrow_mut()(xv);
                    finite_diff_jacobian_fn(&mut con, x, eps)
                }))
            }
            (None, None) => None,
        };

    let result: InteriorPointResult = solver.solve(
        &mut fun_mut,
        &mut *grad_owned,
        eq_con_owned.as_deref_mut(),
        eq_jac_owned.as_deref_mut(),
        ineq_con_owned.as_deref_mut(),
        ineq_jac_owned.as_deref_mut(),
        &x0,
    )?;

    Ok(OptimizeResult {
        x: result.x,
        fun: result.fun,
        nit: result.nit,
        func_evals: result.nfev,
        nfev: result.nfev,
        success: result.success,
        message: result.message,
        jacobian: None,
        hessian: None,
    })
}

/// Compute gradient using finite differences
#[allow(dead_code)]
fn finite_diff_gradient<F>(fun: &mut F, x: &ArrayView1<f64>, eps: f64) -> Array1<f64>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut grad = Array1::zeros(n);
    let f0 = fun(x);
    let mut x_pert = x.to_owned();

    for i in 0..n {
        let h = eps * (1.0 + x[i].abs());
        x_pert[i] = x[i] + h;
        let f_plus = fun(&x_pert.view());
        grad[i] = (f_plus - f0) / h;
        x_pert[i] = x[i];
    }

    grad
}

/// Compute the Jacobian of a single vector-valued constraint function via
/// forward finite differences (one column per input dimension).
#[allow(dead_code)]
fn finite_diff_jacobian_fn<C>(con: &mut C, x: &ArrayView1<f64>, eps: f64) -> Array2<f64>
where
    C: FnMut(&ArrayView1<f64>) -> Array1<f64>,
{
    let n = x.len();
    let f0 = con(x);
    let m = f0.len();
    let mut jac = Array2::zeros((m, n));
    let mut x_pert = x.to_owned();

    for j in 0..n {
        let h = eps * (1.0 + x[j].abs());
        x_pert[j] = x[j] + h;
        let f_plus = con(&x_pert.view());
        for i in 0..m {
            jac[[i, j]] = (f_plus[i] - f0[i]) / h;
        }
        x_pert[j] = x[j];
    }

    jac
}

/// Compute the Jacobian of multiple constraints.
///
/// For each constraint, the analytical Jacobian attached via
/// [`Constraint::with_jacobian`] (issue #127) is used when present; otherwise a
/// forward finite-difference approximation is computed. An analytical Jacobian
/// of the wrong length falls back to finite differences (no panic, no unwrap).
#[allow(dead_code)]
fn finite_diff_jacobian_constraints(
    constraints: &[&Constraint],
    x: &ArrayView1<f64>,
    eps: f64,
) -> Array2<f64> {
    let n = x.len();
    let m = constraints.len();
    let mut jac = Array2::zeros((m, n));
    let x_slice = x.as_slice().expect("Operation failed");

    // Evaluate constraints at current point (reused by the finite-difference path)
    let f0: Vec<f64> = constraints.iter().map(|c| (c.fun)(x_slice)).collect();

    // Track which rows still need finite differences (analytical not available)
    let mut needs_fd = vec![true; m];
    for (i, c) in constraints.iter().enumerate() {
        if let Some(ref jac_fn) = c.jac {
            let grad = jac_fn(x_slice);
            if grad.len() == n {
                for j in 0..n {
                    jac[[i, j]] = grad[j];
                }
                needs_fd[i] = false;
            }
        }
    }

    if needs_fd.iter().any(|&b| b) {
        let mut x_pert = x.to_owned();

        for j in 0..n {
            let h = eps * (1.0 + x[j].abs());
            x_pert[j] = x[j] + h;
            let x_pert_slice = x_pert.as_slice().expect("Operation failed");

            // Evaluate constraints at perturbed point (FD rows only)
            for i in 0..m {
                if needs_fd[i] {
                    let f_plus = (constraints[i].fun)(x_pert_slice);
                    jac[[i, j]] = (f_plus - f0[i]) / h;
                }
            }

            x_pert[j] = x[j]; // Reset
        }
    }

    jac
}

/// Minimize a function subject to constraints using interior point method
/// with constraint conversion from general format
#[allow(dead_code)]
pub fn minimize_interior_point_constrained<F>(
    func: F,
    x0: Array1<f64>,
    constraints: &[Constraint],
    options: Option<InteriorPointOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&[f64]) -> f64 + Clone,
{
    let options = options.unwrap_or_default();
    let n = x0.len();

    // Separate constraints by type
    let eq_constraints: Vec<_> = constraints
        .iter()
        .filter(|c| c.kind == ConstraintKind::Equality && !c.is_bounds())
        .collect();
    let ineq_constraints: Vec<_> = constraints
        .iter()
        .filter(|c| c.kind == ConstraintKind::Inequality && !c.is_bounds())
        .collect();

    let m_eq = eq_constraints.len();
    let m_ineq = ineq_constraints.len();

    // Create solver with proper constraint counts
    let mut solver = InteriorPointSolver::new(n, m_eq, m_ineq, &options);

    // Prepare function and gradient
    let func_clone = func.clone();
    let mut fun_mut =
        move |x: &ArrayView1<f64>| -> f64 { func(x.as_slice().expect("Operation failed")) };
    let mut grad_mut = move |x: &ArrayView1<f64>| -> Array1<f64> {
        let mut fun_fd =
            |x: &ArrayView1<f64>| -> f64 { func_clone(x.as_slice().expect("Operation failed")) };
        finite_diff_gradient(&mut fun_fd, x, 1e-8)
    };

    // Prepare constraint functions and Jacobians if needed.
    //
    // The constraint callables are boxed trait objects (issue #126) and cannot
    // be copied out of the `Constraint` slice. Instead, each closure captures a
    // shared reference to the partitioned constraint list and evaluates the
    // constraints in place, preserving the original numerical behaviour. The
    // closures borrow `eq_constraints` / `ineq_constraints` (and through them
    // `constraints`), so they are scoped to this function via `+ '_`.
    #[allow(clippy::type_complexity)]
    let mut eq_con_mut: Option<Box<dyn FnMut(&ArrayView1<f64>) -> Array1<f64> + '_>> = if m_eq > 0 {
        Some(Box::new(|x: &ArrayView1<f64>| -> Array1<f64> {
            let x_slice = x.as_slice().expect("Operation failed");
            Array1::from_vec(eq_constraints.iter().map(|c| (c.fun)(x_slice)).collect())
        }))
    } else {
        None
    };

    #[allow(clippy::type_complexity)]
    let mut eq_jac_mut: Option<Box<dyn FnMut(&ArrayView1<f64>) -> Array2<f64> + '_>> = if m_eq > 0 {
        Some(Box::new(|x: &ArrayView1<f64>| -> Array2<f64> {
            let eps = 1e-8;
            finite_diff_jacobian_constraints(&eq_constraints, x, eps)
        }))
    } else {
        None
    };

    #[allow(clippy::type_complexity)]
    let mut ineq_con_mut: Option<Box<dyn FnMut(&ArrayView1<f64>) -> Array1<f64> + '_>> =
        if m_ineq > 0 {
            Some(Box::new(|x: &ArrayView1<f64>| -> Array1<f64> {
                let x_slice = x.as_slice().expect("Operation failed");
                Array1::from_vec(ineq_constraints.iter().map(|c| (c.fun)(x_slice)).collect())
            }))
        } else {
            None
        };

    #[allow(clippy::type_complexity)]
    let mut ineq_jac_mut: Option<Box<dyn FnMut(&ArrayView1<f64>) -> Array2<f64> + '_>> =
        if m_ineq > 0 {
            Some(Box::new(|x: &ArrayView1<f64>| -> Array2<f64> {
                let eps = 1e-8;
                finite_diff_jacobian_constraints(&ineq_constraints, x, eps)
            }))
        } else {
            None
        };

    // Solve with constraints
    let result = solver.solve(
        &mut fun_mut,
        &mut grad_mut,
        eq_con_mut.as_mut().map(|f| f.as_mut()),
        eq_jac_mut.as_mut().map(|f| f.as_mut()),
        ineq_con_mut.as_mut().map(|f| f.as_mut()),
        ineq_jac_mut.as_mut().map(|f| f.as_mut()),
        &x0,
    )?;

    // Handle bounds constraints separately if present
    let bounds_constraints: Vec<_> = constraints.iter().filter(|c| c.is_bounds()).collect();

    if !bounds_constraints.is_empty() {
        eprintln!("Warning: Box constraints (bounds) are not yet fully integrated with interior point method");
    }

    Ok(OptimizeResult {
        x: result.x,
        fun: result.fun,
        nit: result.nit,
        func_evals: result.nfev,
        nfev: result.nfev,
        success: result.success,
        message: result.message,
        jacobian: None,
        hessian: None,
    })
}

// Tests live in `interior_point_tests.rs` (split out to keep this
// implementation file under the workspace's 2000-line guideline).
#[cfg(test)]
#[path = "interior_point_tests.rs"]
mod tests;
