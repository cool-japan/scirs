//! Utility functions for cubic spline interpolation
//!
//! This module contains helper functions used by spline interpolation algorithms.
//! These utilities provide common operations like numerical integration of polynomial
//! segments and filtering of root candidates in root-finding operations.
//!
//! ## Functions
//!
//! - **Integration utilities**: Analytical integration of cubic polynomial segments
//! - **Root filtering**: Distance-based filtering for duplicate root detection
//!
//! ## Usage
//!
//! These functions are typically used internally by spline interpolation algorithms
//! and root-finding routines, but may also be useful for custom spline operations.

use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

/// Integrate a cubic polynomial segment from a to b
///
/// The polynomial is defined as: p(x) = a + b*(x-x0) + c*(x-x0)^2 + d*(x-x0)^3
/// where x0 is the left endpoint of the segment.
///
/// # Arguments
///
/// * `coeffs` - Array containing the polynomial coefficients [a, b, c, d]
/// * `x0` - The left endpoint of the segment (reference point)
/// * `a` - Lower integration bound
/// * `b` - Upper integration bound
///
/// # Returns
///
/// The definite integral of the polynomial from a to b.
///
/// # Mathematical Details
///
/// The integral is computed analytically:
/// ∫(a + b*x + c*x^2 + d*x^3) dx = a*x + b*x^2/2 + c*x^3/3 + d*x^4/4
///
/// The function shifts coordinates to (x-x0) and evaluates the antiderivative
/// at the bounds, returning the difference.
#[allow(dead_code)]
pub fn integrate_segment<F: InterpolationFloat>(
    coeffs: &Array1<F>,
    x0: F,
    a: F,
    b: F,
) -> F {
    // Shift to x-x0 coordinates
    let a_shifted = a - x0;
    let b_shifted = b - x0;

    // Extract coefficients
    let coef_a = coeffs[0];
    let coef_b = coeffs[1];
    let coef_c = coeffs[2];
    let coef_d = coeffs[3];

    // Integrate the polynomial:
    // ∫(a + b*x + c*x^2 + d*x^3) dx = a*x + b*x^2/2 + c*x^3/3 + d*x^4/4
    let two = F::from_f64(2.0).unwrap_or_else(|| F::from(2).unwrap_or(F::zero()));
    let three = F::from_f64(3.0).unwrap_or_else(|| F::from(3).unwrap_or(F::zero()));
    let four = F::from_f64(4.0).unwrap_or_else(|| F::from(4).unwrap_or(F::zero()));

    // Evaluate at the bounds
    let int_a = coef_a * a_shifted
        + coef_b * a_shifted * a_shifted / two
        + coef_c * a_shifted * a_shifted * a_shifted / three
        + coef_d * a_shifted * a_shifted * a_shifted * a_shifted / four;

    let int_b = coef_a * b_shifted
        + coef_b * b_shifted * b_shifted / two
        + coef_c * b_shifted * b_shifted * b_shifted / three
        + coef_d * b_shifted * b_shifted * b_shifted * b_shifted / four;

    // Return the difference
    int_b - int_a
}

/// Check if a root candidate is far enough from existing roots
///
/// This function implements a simple distance-based filter to avoid
/// reporting duplicate or very close roots that may arise from
/// numerical noise in root-finding algorithms.
///
/// # Arguments
///
/// * `roots` - Slice of existing roots found so far
/// * `candidate` - The candidate root to check
/// * `tolerance` - Minimum distance required between roots
///
/// # Returns
///
/// `true` if the candidate is at least `tolerance` distance away from
/// all existing roots, `false` otherwise.
///
/// # Usage
///
/// This is typically used in root-finding algorithms to filter out
/// duplicate roots that may be found due to:
/// - Numerical precision issues
/// - Multiple iterations converging to the same root
/// - Overlapping search intervals
#[allow(dead_code)]
pub fn root_far_enough<F: Float>(roots: &[F], candidate: F, tolerance: F) -> bool {
    for &existing_root in roots {
        if (candidate - existing_root).abs() < tolerance {
            return false;
        }
    }
    true
}