//! Mixed Integer Programming (MIP) — unified top-level interface.
//!
//! This module exposes a clean, high-level API for Mixed-Integer Linear
//! Programming (MILP) problems.  It re-exports the core solver types from
//! [`crate::integer`] and adds:
//!
//! - [`MilpProblem`] — unified problem description with a fluent builder.
//! - [`MilpSolverConfig`] — algorithm selection and tuning.
//! - [`MilpSolver`] — single entry-point that dispatches B&B or cutting planes.
//! - [`IntegerConstraint`] — per-variable integrality annotations.
//!
//! # Problem form
//!
//! ```text
//! minimise   c^T x
//! subject to A  x ≤ b            (linear inequalities, optional)
//!            Aeq x = beq          (linear equalities, optional)
//!            lb ≤ x ≤ ub          (variable bounds)
//!            x_i ∈ ℤ   ∀ i ∈ I   (integrality)
//!            x_i ∈ {0,1} ∀ i ∈ B (binary subset)
//! ```
//!
//! # Example
//!
//! ```rust
//! use scirs2_optimize::mip::{MilpProblem, MilpSolverConfig, MilpSolver, IntegerConstraint};
//! use scirs2_core::ndarray::{array, Array2};
//!
//! // Simple knapsack: max 4x₀ + 5x₁ + 3x₂  s.t. 2x₀+3x₁+x₂ ≤ 5, x ∈ {0,1}
//! let c = array![-4.0, -5.0, -3.0]; // negate for min
//! let a = Array2::from_shape_vec((1, 3), vec![2.0, 3.0, 1.0]).expect("valid input");
//! let b = array![5.0];
//!
//! let prob = MilpProblem::builder(c)
//!     .inequalities(a, b)
//!     .bounds(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0])
//!     .all_binary()
//!     .build()
//!     .expect("valid input");
//!
//! let result = MilpSolver::default().solve(&prob).expect("valid input");
//! assert!(result.success);
//! assert!(result.objective <= -7.0 + 1e-4);
//! ```

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};

// ─── Re-exports from integer submodule ───────────────────────────────────────

pub use crate::integer::milp_branch_and_bound::{
    branch_and_bound, BnbConfig, BranchingStrategy, MilpProblem as MilpProblemInner,
    MilpResult as MilpResultInner,
};
pub use crate::integer::{
    is_integer_valued, BranchAndBoundOptions, BranchAndBoundSolver, CuttingPlaneOptions,
    CuttingPlaneSolver, IntegerKind, IntegerVariableSet, LinearProgram, MipResult,
};

// ─── IntegerConstraint ───────────────────────────────────────────────────────

/// Per-variable integrality specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegerConstraint {
    /// Variable is continuous (real-valued).
    Continuous,
    /// Variable must take an integer value.
    Integer,
    /// Variable is binary: must be 0 or 1.
    Binary,
    /// Variable must be a non-negative integer.
    NonNegativeInteger,
}

impl IntegerConstraint {
    /// Convert to the lower-level [`IntegerKind`].
    pub fn to_kind(self) -> IntegerKind {
        match self {
            IntegerConstraint::Continuous => IntegerKind::Continuous,
            IntegerConstraint::Integer | IntegerConstraint::NonNegativeInteger => {
                IntegerKind::Integer
            }
            IntegerConstraint::Binary => IntegerKind::Binary,
        }
    }
}

// ─── MilpProblem ─────────────────────────────────────────────────────────────

/// A Mixed-Integer Linear Program.
///
/// Prefer constructing via [`MilpProblem::builder`].
#[derive(Debug, Clone)]
pub struct MilpProblem {
    /// Objective coefficients (length n).
    pub c: Array1<f64>,
    /// Inequality LHS (m_ub × n), optional.
    pub a_ub: Option<Array2<f64>>,
    /// Inequality RHS (length m_ub), optional.
    pub b_ub: Option<Array1<f64>>,
    /// Equality LHS (m_eq × n), optional.
    pub a_eq: Option<Array2<f64>>,
    /// Equality RHS (length m_eq), optional.
    pub b_eq: Option<Array1<f64>>,
    /// Lower bounds (length n).
    pub lb: Array1<f64>,
    /// Upper bounds (length n).
    pub ub: Array1<f64>,
    /// Per-variable integrality specification (length n).
    pub constraints: Vec<IntegerConstraint>,
}

impl MilpProblem {
    /// Begin building a [`MilpProblem`].
    pub fn builder(c: Array1<f64>) -> MilpProblemBuilder {
        let n = c.len();
        MilpProblemBuilder {
            c,
            a_ub: None,
            b_ub: None,
            a_eq: None,
            b_eq: None,
            lb: Array1::zeros(n),
            ub: Array1::from_elem(n, f64::INFINITY),
            constraints: vec![IntegerConstraint::Continuous; n],
        }
    }

    /// Number of decision variables.
    #[inline]
    pub fn n_vars(&self) -> usize {
        self.c.len()
    }

    /// Number of inequality constraints.
    #[inline]
    pub fn n_ineq(&self) -> usize {
        self.a_ub.as_ref().map_or(0, |a| a.nrows())
    }

    /// Number of equality constraints.
    #[inline]
    pub fn n_eq(&self) -> usize {
        self.a_eq.as_ref().map_or(0, |a| a.nrows())
    }

    /// Indices of integer / binary variables.
    pub fn integer_indices(&self) -> Vec<usize> {
        self.constraints
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != IntegerConstraint::Continuous)
            .map(|(i, _)| i)
            .collect()
    }

    /// Convert to the lower-level [`LinearProgram`] (ignoring integrality).
    pub fn to_linear_program(&self) -> LinearProgram {
        LinearProgram {
            c: self.c.clone(),
            a_ub: self.a_ub.clone(),
            b_ub: self.b_ub.clone(),
            a_eq: self.a_eq.clone(),
            b_eq: self.b_eq.clone(),
            lower: Some(self.lb.clone()),
            upper: Some(self.ub.clone()),
        }
    }

    /// Convert to the lower-level [`IntegerVariableSet`].
    pub fn to_integer_variable_set(&self) -> IntegerVariableSet {
        IntegerVariableSet {
            kinds: self.constraints.iter().map(|c| c.to_kind()).collect(),
        }
    }

    /// Convert to [`MilpProblemInner`] for use with [`branch_and_bound`].
    ///
    /// Equality constraints are expanded into two inequalities:
    /// `Aeq x ≤ beq` and `-Aeq x ≤ -beq`.
    pub fn to_inner(&self) -> OptimizeResult<MilpProblemInner> {
        let (a_combined, b_combined) = self.build_combined_ineq()?;
        MilpProblemInner::new(
            self.c.clone(),
            a_combined,
            b_combined,
            self.lb.clone(),
            self.ub.clone(),
            self.integer_indices(),
        )
    }

    fn build_combined_ineq(&self) -> OptimizeResult<(Array2<f64>, Array1<f64>)> {
        let n = self.n_vars();
        let m_ub = self.n_ineq();
        let m_eq = self.n_eq();
        let total = m_ub + 2 * m_eq;

        if total == 0 {
            return Ok((Array2::zeros((0, n)), Array1::zeros(0)));
        }

        let mut a = Array2::<f64>::zeros((total, n));
        let mut b = Array1::<f64>::zeros(total);

        if let (Some(aub), Some(bub)) = (&self.a_ub, &self.b_ub) {
            for i in 0..m_ub {
                for j in 0..n {
                    a[[i, j]] = aub[[i, j]];
                }
                b[i] = bub[i];
            }
        }

        if let (Some(aeq), Some(beq)) = (&self.a_eq, &self.b_eq) {
            for k in 0..m_eq {
                let rp = m_ub + k;
                let rm = m_ub + m_eq + k;
                for j in 0..n {
                    a[[rp, j]] = aeq[[k, j]];
                    a[[rm, j]] = -aeq[[k, j]];
                }
                b[rp] = beq[k];
                b[rm] = -beq[k];
            }
        }

        Ok((a, b))
    }
}

// ─── MilpProblemBuilder ───────────────────────────────────────────────────────

/// Fluent builder for [`MilpProblem`].
pub struct MilpProblemBuilder {
    c: Array1<f64>,
    a_ub: Option<Array2<f64>>,
    b_ub: Option<Array1<f64>>,
    a_eq: Option<Array2<f64>>,
    b_eq: Option<Array1<f64>>,
    lb: Array1<f64>,
    ub: Array1<f64>,
    constraints: Vec<IntegerConstraint>,
}

impl MilpProblemBuilder {
    /// Add inequality constraints `A x ≤ b`.
    pub fn inequalities(mut self, a: Array2<f64>, b: Array1<f64>) -> Self {
        self.a_ub = Some(a);
        self.b_ub = Some(b);
        self
    }

    /// Add equality constraints `Aeq x = beq`.
    pub fn equalities(mut self, a: Array2<f64>, b: Array1<f64>) -> Self {
        self.a_eq = Some(a);
        self.b_eq = Some(b);
        self
    }

    /// Set variable bounds.
    pub fn bounds(mut self, lb: Array1<f64>, ub: Array1<f64>) -> Self {
        self.lb = lb;
        self.ub = ub;
        self
    }

    /// Set per-variable integrality.
    pub fn integer_constraints(mut self, c: Vec<IntegerConstraint>) -> Self {
        self.constraints = c;
        self
    }

    /// Mark all variables as binary (sets ub=1, lb=0).
    pub fn all_binary(mut self) -> Self {
        let n = self.c.len();
        self.constraints = vec![IntegerConstraint::Binary; n];
        self.lb = Array1::zeros(n);
        self.ub = Array1::ones(n);
        self
    }

    /// Mark all variables as integer.
    pub fn all_integer(mut self) -> Self {
        let n = self.c.len();
        self.constraints = vec![IntegerConstraint::Integer; n];
        self
    }

    /// Validate and finalise the problem.
    pub fn build(self) -> OptimizeResult<MilpProblem> {
        let n = self.c.len();

        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Objective vector c is empty".to_string(),
            ));
        }
        if self.lb.len() != n || self.ub.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "lb/ub length ({}/{}) must equal n={}",
                self.lb.len(),
                self.ub.len(),
                n
            )));
        }
        if self.constraints.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "constraints length ({}) must equal n={}",
                self.constraints.len(),
                n
            )));
        }
        if let (Some(a), Some(b)) = (&self.a_ub, &self.b_ub) {
            if a.ncols() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "A_ub has {} columns but n={}",
                    a.ncols(),
                    n
                )));
            }
            if a.nrows() != b.len() {
                return Err(OptimizeError::InvalidInput(format!(
                    "A_ub rows ({}) ≠ b_ub len ({})",
                    a.nrows(),
                    b.len()
                )));
            }
        }
        if let (Some(a), Some(b)) = (&self.a_eq, &self.b_eq) {
            if a.ncols() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "A_eq has {} columns but n={}",
                    a.ncols(),
                    n
                )));
            }
            if a.nrows() != b.len() {
                return Err(OptimizeError::InvalidInput(format!(
                    "A_eq rows ({}) ≠ b_eq len ({})",
                    a.nrows(),
                    b.len()
                )));
            }
        }

        Ok(MilpProblem {
            c: self.c,
            a_ub: self.a_ub,
            b_ub: self.b_ub,
            a_eq: self.a_eq,
            b_eq: self.b_eq,
            lb: self.lb,
            ub: self.ub,
            constraints: self.constraints,
        })
    }
}

// ─── MilpSolveResult ─────────────────────────────────────────────────────────

/// Result of solving a MILP.
#[derive(Debug, Clone)]
pub struct MilpSolveResult {
    /// Optimal solution vector (length n).
    pub x: Array1<f64>,
    /// Optimal objective value (minimisation sense).
    pub objective: f64,
    /// `true` if an integer-feasible solution was found.
    pub success: bool,
    /// Human-readable status.
    pub message: String,
    /// Number of B&B nodes explored.
    pub nodes_explored: usize,
    /// Lower bound on the optimal objective at termination.
    pub lower_bound: f64,
    /// Total LP sub-problems solved.
    pub lp_solves: usize,
    /// Relative optimality gap: `|obj − lb| / (1 + |obj|)`.
    pub gap: f64,
}

impl MilpSolveResult {
    fn from_bnb(r: MilpResultInner) -> Self {
        let gap = if r.success {
            (r.obj - r.lower_bound).abs() / (1.0 + r.obj.abs())
        } else {
            f64::INFINITY
        };
        MilpSolveResult {
            x: r.x,
            objective: r.obj,
            success: r.success,
            message: r.message,
            nodes_explored: r.nodes_explored,
            lower_bound: r.lower_bound,
            lp_solves: 0,
            gap,
        }
    }

    fn from_mip(r: MipResult) -> Self {
        let gap = if r.success {
            (r.fun - r.lower_bound).abs() / (1.0 + r.fun.abs())
        } else {
            f64::INFINITY
        };
        MilpSolveResult {
            x: r.x,
            objective: r.fun,
            success: r.success,
            message: r.message,
            nodes_explored: r.nodes_explored,
            lower_bound: r.lower_bound,
            lp_solves: r.lp_solves,
            gap,
        }
    }
}

// ─── MilpAlgorithm ────────────────────────────────────────────────────────────

/// Algorithm selection for [`MilpSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MilpAlgorithm {
    /// Branch-and-bound with LP relaxation at each node (default).
    BranchAndBound,
    /// Gomory cutting planes followed by B&B fallback.
    CuttingPlanes,
}

// ─── MilpSolverConfig ─────────────────────────────────────────────────────────

/// Configuration for [`MilpSolver`].
#[derive(Debug, Clone)]
pub struct MilpSolverConfig {
    /// Algorithm to use.
    pub algorithm: MilpAlgorithm,
    /// Maximum B&B nodes.
    pub max_nodes: usize,
    /// Integrality tolerance.
    pub int_tol: f64,
    /// Optimality gap tolerance (relative).
    pub opt_tol: f64,
    /// Branching strategy (B&B).
    pub branching: BranchingStrategy,
    /// Maximum Gomory cut rounds (`CuttingPlanes` only).
    pub max_cut_rounds: usize,
    /// Emit progress messages to stderr.
    pub verbose: bool,
}

impl Default for MilpSolverConfig {
    fn default() -> Self {
        MilpSolverConfig {
            algorithm: MilpAlgorithm::BranchAndBound,
            max_nodes: 50_000,
            int_tol: 1e-6,
            opt_tol: 1e-6,
            branching: BranchingStrategy::MostFractional,
            max_cut_rounds: 50,
            verbose: false,
        }
    }
}

// ─── MilpSolver ──────────────────────────────────────────────────────────────

/// Unified MILP solver.
///
/// Dispatches to branch-and-bound or cutting-planes based on
/// [`MilpSolverConfig::algorithm`].
pub struct MilpSolver {
    config: MilpSolverConfig,
}

impl MilpSolver {
    /// Create with the given configuration.
    pub fn new(config: MilpSolverConfig) -> Self {
        MilpSolver { config }
    }

    /// Solve a [`MilpProblem`].
    pub fn solve(&self, problem: &MilpProblem) -> OptimizeResult<MilpSolveResult> {
        match self.config.algorithm {
            MilpAlgorithm::BranchAndBound => self.solve_bnb(problem),
            MilpAlgorithm::CuttingPlanes => self.solve_cutting_planes(problem),
        }
    }

    fn bnb_config(&self) -> BnbConfig {
        BnbConfig {
            max_nodes: self.config.max_nodes,
            int_tol: self.config.int_tol,
            abs_gap: self.config.opt_tol,
            rel_gap: self.config.opt_tol,
            branching: self.config.branching,
            ..Default::default()
        }
    }

    fn solve_bnb(&self, problem: &MilpProblem) -> OptimizeResult<MilpSolveResult> {
        let inner = problem.to_inner()?;
        let r = branch_and_bound(&inner, &self.bnb_config())?;
        Ok(MilpSolveResult::from_bnb(r))
    }

    fn solve_cutting_planes(&self, problem: &MilpProblem) -> OptimizeResult<MilpSolveResult> {
        let lp = problem.to_linear_program();
        let ivs = problem.to_integer_variable_set();
        let opts = CuttingPlaneOptions {
            max_iter: self.config.max_cut_rounds,
            int_tol: self.config.int_tol,
            fallback_bb: true,
            ..Default::default()
        };
        let r = CuttingPlaneSolver::with_options(opts).solve(&lp, &ivs)?;
        Ok(MilpSolveResult::from_mip(r))
    }
}

impl Default for MilpSolver {
    fn default() -> Self {
        MilpSolver::new(MilpSolverConfig::default())
    }
}

// ─── Convenience function ─────────────────────────────────────────────────────

/// Solve a [`MilpProblem`] with default solver settings.
pub fn solve_milp(problem: &MilpProblem) -> OptimizeResult<MilpSolveResult> {
    MilpSolver::default().solve(problem)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    fn knapsack_problem(values: &[f64], weights: &[f64], cap: f64) -> MilpProblem {
        let n = values.len();
        let c = Array1::from_vec(values.iter().map(|&v| -v).collect());
        let a = Array2::from_shape_vec((1, n), weights.to_vec()).expect("shape");
        let b = Array1::from_vec(vec![cap]);
        MilpProblem::builder(c)
            .inequalities(a, b)
            .bounds(Array1::zeros(n), Array1::ones(n))
            .all_binary()
            .build()
            .expect("valid problem")
    }

    #[test]
    fn test_builder_basic_shape() {
        let c = array![-4.0, -5.0, -3.0];
        let a = Array2::from_shape_vec((1, 3), vec![2.0, 3.0, 1.0]).expect("shape");
        let b = array![5.0];
        let p = MilpProblem::builder(c)
            .inequalities(a, b)
            .all_binary()
            .build()
            .expect("valid");
        assert_eq!(p.n_vars(), 3);
        assert_eq!(p.n_ineq(), 1);
        assert_eq!(p.n_eq(), 0);
    }

    #[test]
    fn test_builder_dimension_mismatch() {
        let c = array![1.0, 2.0];
        // A has 3 cols but c has 2
        let a = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).expect("shape");
        let b = array![5.0];
        let result = MilpProblem::builder(c)
            .inequalities(a, b)
            .all_binary()
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_binary_knapsack() {
        // max 4x+5y s.t. 2x+3y<=5, x,y in {0,1}
        // opt: x=1,y=1 weight=5<=5, obj=-9
        let p = knapsack_problem(&[4.0, 5.0], &[2.0, 3.0], 5.0);
        let r = solve_milp(&p).expect("solve ok");
        assert!(r.success);
        assert!(r.objective <= -8.0 + 1e-4, "obj={}", r.objective);
    }

    #[test]
    fn test_solve_pure_integer() {
        // min x+y s.t. x+y >= 3.5, x,y>=0, integer -> opt=4
        let c = array![1.0, 1.0];
        let a = Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("shape");
        let b = array![-3.5];
        let p = MilpProblem::builder(c)
            .inequalities(a, b)
            .bounds(array![0.0, 0.0], array![10.0, 10.0])
            .all_integer()
            .build()
            .expect("valid");
        let r = solve_milp(&p).expect("solve ok");
        assert!(r.success);
        assert_abs_diff_eq!(r.objective, 4.0, epsilon = 1e-3);
    }

    #[test]
    fn test_solve_with_equalities() {
        // min x+y s.t. x+y=5, x,y>=0, integer -> opt=5
        let c = array![1.0, 1.0];
        let aeq = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).expect("shape");
        let beq = array![5.0];
        let p = MilpProblem::builder(c)
            .equalities(aeq, beq)
            .bounds(array![0.0, 0.0], array![10.0, 10.0])
            .all_integer()
            .build()
            .expect("valid");
        let r = solve_milp(&p).expect("solve ok");
        assert!(r.success);
        assert_abs_diff_eq!(r.objective, 5.0, epsilon = 1e-3);
    }

    #[test]
    fn test_solve_infeasible() {
        // x in {0,1}, x >= 2 -> infeasible
        let c = array![-1.0];
        let a = Array2::from_shape_vec((1, 1), vec![-1.0]).expect("shape");
        let b = array![-2.0];
        let p = MilpProblem::builder(c)
            .inequalities(a, b)
            .bounds(array![0.0], array![1.0])
            .all_binary()
            .build()
            .expect("valid");
        let r = solve_milp(&p).expect("runs");
        assert!(!r.success);
    }

    #[test]
    fn test_solve_cutting_planes() {
        let c = array![1.0, 1.0];
        let a = Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).expect("shape");
        let b = array![-3.5];
        let p = MilpProblem::builder(c)
            .inequalities(a, b)
            .bounds(array![0.0, 0.0], array![10.0, 10.0])
            .all_integer()
            .build()
            .expect("valid");
        let cfg = MilpSolverConfig {
            algorithm: MilpAlgorithm::CuttingPlanes,
            ..Default::default()
        };
        let r = MilpSolver::new(cfg).solve(&p).expect("ok");
        assert!(r.success);
        assert_abs_diff_eq!(r.objective, 4.0, epsilon = 1e-3);
    }

    #[test]
    fn test_integer_constraint_to_kind() {
        assert_eq!(
            IntegerConstraint::Continuous.to_kind(),
            IntegerKind::Continuous
        );
        assert_eq!(IntegerConstraint::Binary.to_kind(), IntegerKind::Binary);
        assert_eq!(IntegerConstraint::Integer.to_kind(), IntegerKind::Integer);
        assert_eq!(
            IntegerConstraint::NonNegativeInteger.to_kind(),
            IntegerKind::Integer
        );
    }

    #[test]
    fn test_gap_computation() {
        let p = knapsack_problem(&[5.0, 3.0], &[3.0, 2.0], 4.0);
        let r = solve_milp(&p).expect("ok");
        if r.success {
            assert!(r.gap >= 0.0);
            assert!(r.gap < 1.0 + 1e-10);
        }
    }
}
