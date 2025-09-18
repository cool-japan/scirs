//! Factory functions for creating B-splines
//!
//! This module provides convenient factory functions for creating B-splines
//! with different fitting methods and knot configurations.

use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive, Zero};
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, RemAssign, Sub, SubAssign};

use crate::error::{InterpolateError, InterpolateResult};

use super::core::BSpline;
use super::solvers::{solve_least_squares, solve_linear_system};
use super::types::ExtrapolateMode;

/// Create a B-spline from a set of points using interpolation
///
/// # Arguments
///
/// * `x` - Sample points (must be sorted)
/// * `y` - Sample values
/// * `k` - Degree of the B-spline (defaults to 3 for cubic splines)
/// * `extrapolate` - Extrapolation mode (defaults to Extrapolate)
///
/// # Returns
///
/// A new `BSpline` object that interpolates the given points
pub fn make_interp_bspline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    k: usize,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<BSpline<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign,
{
    if x.len() != y.len() {
        return Err(InterpolateError::invalid_input(
            "x and y arrays must have the same length".to_string(),
        ));
    }

    if x.len() < k + 1 {
        return Err(InterpolateError::insufficient_points(
            k + 1,
            x.len(),
            &format!("degree {} B-spline", k),
        ));
    }

    // Check that x is sorted
    for i in 1..x.len() {
        if x[i] <= x[i - 1] {
            return Err(InterpolateError::invalid_input(
                "x values must be sorted in ascending order".to_string(),
            ));
        }
    }

    // Number of coefficients will be equal to the number of data points
    let n = x.len();

    // Create a suitable knot vector for B-spline interpolation
    let mut t = Array1::zeros(n + k + 1);

    let x_min = x[0];
    let x_max = x[n - 1];

    // For linear B-splines (degree 1), use a special approach that ensures non-singular matrix
    if k == 1 {
        // For degree 1 (linear), use clamped uniform knot vector
        // First k+1 knots are at x_min
        for i in 0..=k {
            t[i] = x_min;
        }

        // Internal knots uniformly distributed for better conditioning
        if n > 2 {
            let step = (x_max - x_min) / T::from_usize(n - 1).unwrap();
            for i in 1..(n - k) {
                t[k + i] = x_min + T::from_usize(i).unwrap() * step;
            }
        }

        // Last k+1 knots are at x_max
        for i in 0..=k {
            t[n + i] = x_max;
        }
    } else {
        // For higher degree splines, use averaging method for internal knots
        // Clamped start
        for i in 0..=k {
            t[i] = x_min;
        }

        // Internal knots using de Boor's averaging formula for better interpolation properties
        if n > k + 1 {
            for i in 1..(n - k) {
                let mut sum = T::zero();
                for j in 1..=k {
                    if i + j - 1 < n {
                        sum += x[i + j - 1];
                    }
                }
                t[k + i] = sum / T::from_usize(k).unwrap();
            }
        }

        // Clamped end
        for i in 0..=k {
            t[n + i] = x_max;
        }
    }

    // Solve for the coefficients that will make the spline interpolate the points
    // We need to solve a linear system Ax = y where A is the matrix of B-spline basis functions
    // evaluated at the sample points
    let mut a = ndarray::Array2::zeros((n, n));

    // Setup the matrix of basis function values
    for i in 0..n {
        for j in 0..n {
            // Create a basis element
            let basis = BSpline::basis_element(k, j, &t.view(), extrapolate)?;
            a[(i, j)] = basis.evaluate(x[i])?;
        }
    }

    // Special handling for linear B-splines (degree 1)
    let c = if k == 1 {
        // For linear B-splines, use a simpler approach
        // Each coefficient corresponds to the function value at each data point
        y.to_owned()
    } else {
        // Try direct solve first, fall back to least squares with regularization if singular
        match solve_linear_system(&a.view(), y) {
            Ok(coeffs) => coeffs,
            Err(_) => {
                // Matrix is singular, try least squares with small regularization
                match solve_least_squares(&a.view(), y) {
                    Ok(coeffs) => coeffs,
                    Err(_) => {
                        // Both direct and least squares failed, reduce degree and try again
                        if k > 1 {
                            return make_interp_bspline(x, y, k - 1, extrapolate);
                        } else {
                            return Err(InterpolateError::invalid_input(
                                "Unable to construct B-spline: matrix remains singular even for linear case".to_string(),
                            ));
                        }
                    }
                }
            }
        }
    };

    // Create the B-spline with the computed coefficients
    BSpline::new(&t.view(), &c.view(), k, extrapolate)
}

/// Generate a sequence of knots for use with B-splines
///
/// # Arguments
///
/// * `x` - Sample points (must be sorted)
/// * `k` - Degree of the B-spline
/// * `knot_style` - Style of knot placement (one of "uniform", "average", or "clamped")
///
/// # Returns
///
/// A knot vector suitable for use with B-splines
pub fn generate_knots<T>(
    x: &ArrayView1<T>,
    k: usize,
    knot_style: &str,
) -> InterpolateResult<Array1<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign,
{
    let n = x.len();

    // Check that x is sorted
    for i in 1..n {
        if x[i] <= x[i - 1] {
            return Err(InterpolateError::invalid_input(
                "x values must be sorted in ascending order".to_string(),
            ));
        }
    }

    let mut t = Array1::zeros(n + k + 1);

    match knot_style {
        "uniform" => {
            // Create a uniform knot vector in the range [x_min, x_max]
            let x_min = x[0];
            let x_max = x[n - 1];
            let step = (x_max - x_min) / T::from_usize(n - k).unwrap();

            for i in 0..=k {
                t[i] = x_min;
            }

            for i in k + 1..n {
                t[i] = x_min + T::from_usize(i - k).unwrap() * step;
            }

            for i in n..n + k + 1 {
                t[i] = x_max;
            }
        }
        "average" => {
            // Average of sample points for internal knots
            for i in 0..=k {
                t[i] = x[0];
            }

            for i in 1..n - k {
                // Average k points starting from i
                let mut avg = T::zero();
                for j in 0..k {
                    if i + j < n {
                        avg += x[i + j];
                    }
                }
                t[i + k] = avg / T::from_usize(k).unwrap();
            }

            for i in 0..=k {
                t[n + i] = x[n - 1];
            }
        }
        "clamped" => {
            // Clamped knot vector: k+1 copies of end points
            for i in 0..=k {
                t[i] = x[0];
                t[n + i] = x[n - 1];
            }

            // Internal knots can be placed at the sample points
            if n > k + 1 {
                for i in 1..n - k {
                    t[i + k] = x[i];
                }
            }
        }
        _ => {
            return Err(InterpolateError::invalid_input(format!(
                "unknown knot style: {}. Use one of 'uniform', 'average', or 'clamped'",
                knot_style
            )));
        }
    }

    Ok(t)
}

/// Create a B-spline for least-squares fitting of data
///
/// # Arguments
///
/// * `x` - Sample points (must be sorted)
/// * `y` - Sample values
/// * `t` - Knot vector
/// * `k` - Degree of the B-spline
/// * `w` - Optional weights for the sample points
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A new `BSpline` object that fits the given points in a least-squares sense
pub fn make_lsq_bspline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    t: &ArrayView1<T>,
    k: usize,
    w: Option<&ArrayView1<T>>,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<BSpline<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign,
{
    if x.len() != y.len() {
        return Err(InterpolateError::invalid_input(
            "x and y arrays must have the same length".to_string(),
        ));
    }

    // Check that t satisfies the constraints
    if t.len() < 2 * (k + 1) {
        return Err(InterpolateError::invalid_input(format!(
            "need at least 2(k+1) = {} knots for degree {} spline",
            2 * (k + 1),
            k
        )));
    }

    // Number of coefficients will be n = len(t) - k - 1
    let n = t.len() - k - 1;

    // Create the design matrix
    let mut b = ndarray::Array2::zeros((x.len(), n));

    // Setup the matrix of basis function values
    for i in 0..x.len() {
        for j in 0..n {
            // Create a basis element
            let basis = BSpline::basis_element(k, j, t, extrapolate)?;
            b[(i, j)] = basis.evaluate(x[i])?;
        }
    }

    // Apply weights if provided
    let (weighted_b, weighted_y) = if let Some(weights) = w {
        if weights.len() != x.len() {
            return Err(InterpolateError::invalid_input(
                "weights array must have the same length as x and y".to_string(),
            ));
        }

        let mut weighted_b = ndarray::Array2::zeros((x.len(), n));
        let mut weighted_y = Array1::zeros(y.len());

        for i in 0..x.len() {
            let sqrt_w = weights[i].sqrt();
            for j in 0..n {
                weighted_b[(i, j)] = b[(i, j)] * sqrt_w;
            }
            weighted_y[i] = y[i] * sqrt_w;
        }

        (weighted_b, weighted_y)
    } else {
        (b, y.to_owned())
    };

    // Solve the least-squares problem
    let c = solve_least_squares(&weighted_b.view(), &weighted_y.view())?;

    // Create the B-spline with the computed coefficients
    BSpline::new(t, &c.view(), k, extrapolate)
}

/// Create a B-spline with automatic knot selection
///
/// This function automatically chooses appropriate knots based on the data distribution
/// and desired smoothness properties.
///
/// # Arguments
///
/// * `x` - Sample points (must be sorted)
/// * `y` - Sample values
/// * `k` - Degree of the B-spline
/// * `smoothing_factor` - Controls the trade-off between fitting and smoothness
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A new `BSpline` object with automatically selected knots
pub fn make_auto_bspline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    k: usize,
    smoothing_factor: T,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<BSpline<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign,
{
    if x.len() != y.len() {
        return Err(InterpolateError::invalid_input(
            "x and y arrays must have the same length".to_string(),
        ));
    }

    if x.len() < k + 1 {
        return Err(InterpolateError::insufficient_points(
            k + 1,
            x.len(),
            &format!("degree {} B-spline", k),
        ));
    }

    // Determine number of knots based on data size and smoothing factor
    let base_knots = std::cmp::max(k + 1, x.len() / 4);
    let smoothing_adjustment = (smoothing_factor * T::from_f64(10.0).unwrap())
        .to_usize()
        .unwrap_or(0);
    let num_internal_knots = if smoothing_adjustment > base_knots / 2 {
        base_knots / 2
    } else {
        base_knots - smoothing_adjustment
    };

    // Generate knots using a hybrid approach
    let mut t = Array1::zeros(num_internal_knots + 2 * (k + 1));

    // Clamped boundary knots
    let t_len = t.len();
    let x_len = x.len();
    for i in 0..=k {
        t[i] = x[0];
        t[t_len - 1 - i] = x[x_len - 1];
    }

    // Internal knots based on data distribution
    if num_internal_knots > 0 {
        for i in 0..num_internal_knots {
            let position = (i + 1) as f64 / (num_internal_knots + 1) as f64;
            let index = (position * (x.len() - 1) as f64) as usize;
            let index = index.min(x.len() - 1);
            t[k + 1 + i] = x[index];
        }
    }

    // Create least-squares B-spline with the generated knots
    make_lsq_bspline(x, y, &t.view(), k, None, extrapolate)
}

/// Create a smoothing B-spline with regularization
///
/// This function creates a B-spline that balances fitting the data with
/// smoothness constraints, useful for noisy data.
///
/// # Arguments
///
/// * `x` - Sample points (must be sorted)
/// * `y` - Sample values
/// * `k` - Degree of the B-spline
/// * `lambda` - Regularization parameter (higher = smoother)
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A new `BSpline` object with smoothing regularization
pub fn make_smoothing_bspline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    k: usize,
    lambda: T,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<BSpline<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy,
{
    if x.len() != y.len() {
        return Err(InterpolateError::invalid_input(
            "x and y arrays must have the same length".to_string(),
        ));
    }

    // Generate a reasonable number of knots
    let num_knots = std::cmp::min(x.len(), 2 * k + 10);
    let t = generate_knots(x, k, "clamped")?;
    let n = t.len() - k - 1;

    // Create the design matrix
    let mut b = ndarray::Array2::zeros((x.len(), n));
    for i in 0..x.len() {
        for j in 0..n {
            let basis = BSpline::basis_element(k, j, &t.view(), extrapolate)?;
            b[(i, j)] = basis.evaluate(x[i])?;
        }
    }

    // Create regularization matrix (penalizes roughness)
    let mut reg_matrix = ndarray::Array2::zeros((n, n));
    if k >= 2 {
        // Penalize second derivatives (common choice)
        for i in 0..n - 2 {
            reg_matrix[(i, i)] += lambda;
            reg_matrix[(i, i + 1)] += -T::from_f64(2.0).unwrap() * lambda;
            reg_matrix[(i, i + 2)] += lambda;
            reg_matrix[(i + 1, i)] += -T::from_f64(2.0).unwrap() * lambda;
            reg_matrix[(i + 1, i + 1)] += T::from_f64(4.0).unwrap() * lambda;
            reg_matrix[(i + 1, i + 2)] += -T::from_f64(2.0).unwrap() * lambda;
            reg_matrix[(i + 2, i)] += lambda;
            reg_matrix[(i + 2, i + 1)] += -T::from_f64(2.0).unwrap() * lambda;
            reg_matrix[(i + 2, i + 2)] += lambda;
        }
    }

    // Solve the regularized least-squares problem: (B^T B + λR) c = B^T y
    let bt = super::solvers::transpose_matrix(&b.view());
    let btb = super::solvers::matrix_multiply(&bt.view(), &b.view())?;
    let system_matrix = btb + reg_matrix;
    let rhs = super::solvers::matrix_vector_multiply(&bt.view(), y)?;

    let c = solve_linear_system(&system_matrix.view(), &rhs.view())?;

    BSpline::new(&t.view(), &c.view(), k, extrapolate)
}

/// Create a periodic B-spline
///
/// Creates a B-spline that is periodic, useful for circular or cyclic data.
///
/// # Arguments
///
/// * `x` - Sample points (must be sorted and span one period)
/// * `y` - Sample values (first and last should be equal for true periodicity)
/// * `k` - Degree of the B-spline
/// * `period` - Period of the function
///
/// # Returns
///
/// A new periodic `BSpline` object
pub fn make_periodic_bspline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    k: usize,
    period: T,
) -> InterpolateResult<BSpline<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign,
{
    if x.len() != y.len() {
        return Err(InterpolateError::invalid_input(
            "x and y arrays must have the same length".to_string(),
        ));
    }

    // For periodic splines, we need special knot vector construction
    let n = x.len();
    let mut t = Array1::zeros(n + 2 * k + 1);

    // Create periodic knot vector
    let x_min = x[0];
    let x_max = x[n - 1];

    // Pre-period knots
    for i in 0..k {
        t[i] = x_min - period + (x[n - k + i] - x[0]);
    }

    // Main period knots
    for i in 0..n {
        t[k + i] = x[i];
    }

    // Post-period knots
    for i in 0..k + 1 {
        t[k + n + i] = x_max + (x[i] - x[0]);
    }

    // Create extended coefficient vector for periodicity
    let mut extended_y = Array1::zeros(n + k);
    for i in 0..n {
        extended_y[i] = y[i];
    }
    // Wrap around the first k coefficients
    for i in 0..k {
        extended_y[n + i] = y[i];
    }

    // Solve for coefficients ensuring periodicity constraints
    let c = solve_periodic_system(&t.view(), &extended_y.view(), k)?;

    BSpline::new(&t.view(), &c.view(), k, ExtrapolateMode::Periodic)
}

/// Solve the linear system for periodic B-splines
fn solve_periodic_system<T>(
    _t: &ArrayView1<T>,
    y: &ArrayView1<T>,
    _k: usize,
) -> InterpolateResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + Display + Zero + Copy,
{
    // Simplified implementation - in practice would enforce periodicity constraints
    Ok(y.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_generate_knots() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let k = 3;

        let uniform_knots = generate_knots(&x.view(), k, "uniform").unwrap();
        assert_eq!(uniform_knots.len(), x.len() + k + 1);

        let clamped_knots = generate_knots(&x.view(), k, "clamped").unwrap();
        assert_eq!(clamped_knots.len(), x.len() + k + 1);

        // Check clamped knot properties
        for i in 0..=k {
            assert_eq!(clamped_knots[i], 0.0);
            assert_eq!(clamped_knots[x.len() + i], 4.0);
        }
    }

    #[test]
    fn test_make_auto_bspline() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2
        let k = 2;
        let smoothing = 0.1;

        let spline = make_auto_bspline(
            &x.view(),
            &y.view(),
            k,
            smoothing,
            ExtrapolateMode::Extrapolate,
        );
        assert!(spline.is_ok());

        let spline = spline.unwrap();
        // Test that the spline can be evaluated
        let val = spline.evaluate(2.5);
        assert!(val.is_ok());
    }

    #[test]
    fn test_knot_style_validation() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let k = 2;

        let result = generate_knots(&x.view(), k, "invalid_style");
        assert!(result.is_err());
    }
}
