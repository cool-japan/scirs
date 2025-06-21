//! Backward compatibility wrappers for functions that gained workers parameters
//!
//! This module provides wrapper functions with the old signatures to maintain
//! backward compatibility while transitioning to the new worker-aware API.

use crate::error::LinalgResult;
use crate::{decomposition, solve};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign};
use std::iter::Sum;

/// Backward compatibility wrapper for cholesky decomposition
pub fn cholesky_compat<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    decomposition::cholesky(a, None)
}

/// Backward compatibility wrapper for LU decomposition  
pub fn lu_compat<F>(a: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + 'static,
{
    decomposition::lu(a, None)
}

/// Backward compatibility wrapper for QR decomposition
pub fn qr_compat<F>(a: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + 'static,
{
    decomposition::qr(a, None)
}

/// Backward compatibility wrapper for SVD
pub fn svd_compat<F>(
    a: &ArrayView2<F>,
    full_matrices: bool,
) -> LinalgResult<(Array2<F>, Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + 'static + ndarray::ScalarOperand,
{
    decomposition::svd(a, full_matrices, None)
}

/// Backward compatibility wrapper for solve
pub fn solve_compat<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    solve::solve(a, b, None)
}

/// Backward compatibility wrapper for lstsq
pub fn lstsq_compat<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> LinalgResult<solve::LstsqResult<F>>
where
    F: Float + NumAssign + Sum + 'static + ndarray::ScalarOperand,
{
    solve::lstsq(a, b, None)
}

/// Backward compatibility wrapper for solve_multiple
pub fn solve_multiple_compat<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    solve::solve_multiple(a, b, None)
}
