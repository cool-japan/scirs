//! COBYLA (Constrained Optimization BY Linear Approximations) algorithm

use crate::constrained::{Constraint, ConstraintFn, Options};
use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use ndarray::{ArrayBase, Data, Ix1};

/// COBYLA optimizer for constrained optimization
///
/// COBYLA is a numerical optimization method for constrained problems where
/// the objective function and the constraints are not required to be differentiable.
///
/// This is currently a placeholder implementation that returns a NotImplementedError.
///
/// # References
///
/// Powell, M. J. D. (1994). "A Direct Search Optimization Method That Models
/// the Objective and Constraint Functions by Linear Interpolation."
/// In Advances in Optimization and Numerical Analysis, pp. 51-67.
pub fn minimize_cobyla<F, S>(
    _func: F,
    _x0: &ArrayBase<S, Ix1>,
    _constraints: &[Constraint<ConstraintFn>],
    _options: &Options,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Data<Elem = f64>,
{
    Err(OptimizeError::NotImplementedError(
        "COBYLA algorithm is not yet implemented".to_string(),
    ))
}
