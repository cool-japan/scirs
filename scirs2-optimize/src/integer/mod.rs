//! Integer Programming and Mixed-Integer Optimization
//!
//! This module provides algorithms for optimization problems where some or all
//! variables must take integer values. It implements:
//!
//! - [`BranchAndBoundSolver`]: Branch-and-bound for mixed-integer programs
//! - [`IntegerVariableSet`]: Specification of integer/binary/general integer constraints
//! - [`CuttingPlaneSolver`]: Gomory cutting plane method for pure integer programs
//!
//! # Mixed-Integer Programming (MIP)
//!
//! A mixed-integer program has the form:
//! ```text
//! minimize    c^T x
//! subject to  A x <= b      (linear inequality constraints)
//!             Aeq x = beq   (linear equality constraints)
//!             lb <= x <= ub (bounds)
//!             x_I in Z      (integrality constraints for a subset I)
//! ```
//!
//! # References
//! - Land, A.H. & Doig, A.G. (1960). "An automatic method of solving discrete
//!   programming problems." Econometrica, 28(3), 497-520.
//! - Gomory, R.E. (1958). "Outline of an algorithm for integer solutions to
//!   linear programs." Bulletin of the American Mathematical Society, 64(5), 275-278.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};

pub mod branch_and_bound;
pub mod cdcl_branching;
pub mod column_generation;
pub mod cutting_plane;
pub mod knapsack;
pub mod lift_project;
pub mod lift_project_mip;
pub mod lp_relaxation;
pub mod milp_branch_and_bound;

pub use branch_and_bound::{BranchAndBoundOptions, BranchAndBoundSolver};
pub use cutting_plane::{CuttingPlaneOptions, CuttingPlaneSolver};
pub use lp_relaxation::LpRelaxationSolver;

/// Type of integer variable constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerKind {
    /// Continuous variable (no integrality constraint)
    Continuous,
    /// Binary variable: must be 0 or 1
    Binary,
    /// General integer variable: must be an integer
    Integer,
}

/// Set of integer variable constraints for a MIP problem
#[derive(Debug, Clone)]
pub struct IntegerVariableSet {
    /// Variable types (indexed by variable position)
    pub kinds: Vec<IntegerKind>,
}

impl IntegerVariableSet {
    /// Create a new set with all continuous variables
    pub fn new(n: usize) -> Self {
        IntegerVariableSet {
            kinds: vec![IntegerKind::Continuous; n],
        }
    }

    /// Create a set with all integer variables
    pub fn all_integer(n: usize) -> Self {
        IntegerVariableSet {
            kinds: vec![IntegerKind::Integer; n],
        }
    }

    /// Create a set with all binary variables
    pub fn all_binary(n: usize) -> Self {
        IntegerVariableSet {
            kinds: vec![IntegerKind::Binary; n],
        }
    }

    /// Set variable i to be of given kind
    pub fn set_kind(&mut self, i: usize, kind: IntegerKind) {
        if i < self.kinds.len() {
            self.kinds[i] = kind;
        }
    }

    /// Return true if variable i has an integrality constraint
    pub fn is_integer(&self, i: usize) -> bool {
        match self.kinds.get(i) {
            Some(IntegerKind::Integer) | Some(IntegerKind::Binary) => true,
            _ => false,
        }
    }

    /// Number of variables
    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    /// True if no integer variables are defined
    pub fn is_empty(&self) -> bool {
        self.kinds.iter().all(|k| *k == IntegerKind::Continuous)
    }
}

/// Result from a MIP solve
#[derive(Debug, Clone)]
pub struct MipResult {
    /// Optimal solution
    pub x: Array1<f64>,
    /// Optimal objective value
    pub fun: f64,
    /// Whether an optimal solution was found
    pub success: bool,
    /// Status message
    pub message: String,
    /// Number of nodes explored (B&B)
    pub nodes_explored: usize,
    /// Number of LP solves
    pub lp_solves: usize,
    /// Lower bound on optimal value
    pub lower_bound: f64,
}

/// Linear program specification
#[derive(Debug, Clone)]
pub struct LinearProgram {
    /// Objective coefficients: minimize c^T x
    pub c: Array1<f64>,
    /// Inequality constraint matrix (A x <= b)
    pub a_ub: Option<Array2<f64>>,
    /// Inequality constraint RHS
    pub b_ub: Option<Array1<f64>>,
    /// Equality constraint matrix (Aeq x = beq)
    pub a_eq: Option<Array2<f64>>,
    /// Equality constraint RHS
    pub b_eq: Option<Array1<f64>>,
    /// Lower bounds (default: 0)
    pub lower: Option<Array1<f64>>,
    /// Upper bounds (default: infinity)
    pub upper: Option<Array1<f64>>,
}

impl LinearProgram {
    /// Create a new LP with just an objective
    pub fn new(c: Array1<f64>) -> Self {
        LinearProgram {
            c,
            a_ub: None,
            b_ub: None,
            a_eq: None,
            b_eq: None,
            lower: None,
            upper: None,
        }
    }

    /// Number of variables
    pub fn n_vars(&self) -> usize {
        self.c.len()
    }
}

/// Check if a value is integer (within tolerance)
#[inline]
pub fn is_integer_valued(v: f64, tol: f64) -> bool {
    (v - v.round()).abs() <= tol
}

/// Find the most fractional variable index
pub fn most_fractional_variable(x: &[f64], var_set: &IntegerVariableSet) -> Option<usize> {
    let mut best_idx = None;
    let mut best_frac = -1.0_f64;
    for (i, &xi) in x.iter().enumerate() {
        if var_set.is_integer(i) {
            let frac = (xi - xi.floor()).abs();
            let frac = frac.min(1.0 - frac); // distance from nearest integer
            if frac > 1e-8 && frac > best_frac {
                best_frac = frac;
                best_idx = Some(i);
            }
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_variable_set_creation() {
        let ivs = IntegerVariableSet::new(5);
        assert_eq!(ivs.len(), 5);
        assert!(ivs.is_empty());
        for i in 0..5 {
            assert!(!ivs.is_integer(i));
        }
    }

    #[test]
    fn test_integer_variable_set_all_integer() {
        let ivs = IntegerVariableSet::all_integer(3);
        assert!(!ivs.is_empty());
        for i in 0..3 {
            assert!(ivs.is_integer(i));
        }
    }

    #[test]
    fn test_integer_variable_set_mixed() {
        let mut ivs = IntegerVariableSet::new(4);
        ivs.set_kind(1, IntegerKind::Integer);
        ivs.set_kind(3, IntegerKind::Binary);
        assert!(!ivs.is_integer(0));
        assert!(ivs.is_integer(1));
        assert!(!ivs.is_integer(2));
        assert!(ivs.is_integer(3));
    }

    #[test]
    fn test_is_integer_valued() {
        assert!(is_integer_valued(3.0, 1e-8));
        assert!(is_integer_valued(3.0000000001, 1e-7));
        assert!(!is_integer_valued(3.5, 1e-8));
        assert!(!is_integer_valued(3.1, 1e-8));
    }

    #[test]
    fn test_most_fractional_variable() {
        let x = vec![1.0, 2.6, 3.1, 0.5];
        let ivs = IntegerVariableSet::all_integer(4);
        let idx = most_fractional_variable(&x, &ivs);
        // x[3] = 0.5 is most fractional (distance 0.5 from nearest integer)
        assert_eq!(idx, Some(3));
    }

    #[test]
    fn test_linear_program_construction() {
        use scirs2_core::ndarray::array;
        let c = array![1.0, -2.0, 3.0];
        let lp = LinearProgram::new(c);
        assert_eq!(lp.n_vars(), 3);
    }
}
