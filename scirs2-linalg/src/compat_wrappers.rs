//! Backward compatibility wrappers for functions that gained workers parameters
//!
//! This module provides wrapper functions with the old signatures to maintain
//! backward compatibility while transitioning to the new worker-aware API.

use crate::error::LinalgResult;
use crate::{decomposition, solve};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

/// Backward compatibility wrapper for cholesky decomposition
#[allow(dead_code)]
pub fn cholesky_compat<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    decomposition::cholesky(a, None)
}

/// Backward compatibility wrapper for LU decomposition  
#[allow(dead_code)]
pub fn lu_compat<F>(a: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    decomposition::lu(a, None)
}

/// Backward compatibility wrapper for QR decomposition
#[allow(dead_code)]
pub fn qr_compat<F>(a: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    decomposition::qr(a, None)
}

/// Backward compatibility wrapper for SVD
#[allow(dead_code)]
pub fn svd_compat<F>(
    a: &ArrayView2<F>,
    full_matrices: bool,
) -> LinalgResult<(Array2<F>, Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    decomposition::svd(a, full_matrices, None)
}

/// Backward compatibility wrapper for solve
#[allow(dead_code)]
pub fn solve_compat<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    solve::solve(a, b, None)
}

/// Backward compatibility wrapper for lstsq
#[allow(dead_code)]
pub fn lstsq_compat<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> LinalgResult<solve::LstsqResult<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    solve::lstsq(a, b, None)
}

/// Backward compatibility wrapper for solve_multiple
#[allow(dead_code)]
pub fn solve_multiple_compat<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    solve::solve_multiple(a, b, None)
}
