//! Spline-based interpolation functions

use scirs2_core::ndarray::{Array, Array1, Axis, Dimension};
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};

/// B-spline poles for different orders
/// Based on the theory of B-spline interpolation
#[allow(dead_code)]
fn get_spline_poles<T: Float + FromPrimitive>(order: usize) -> Vec<T> {
    match order {
        0 | 1 => vec![], // No poles for constant or linear
        2 => {
            // Quadratic B-spline has one pole at sqrt(8) - 3
            let sqrt8 = T::from_f64(8.0).expect("Operation failed").sqrt();
            let three = T::from_f64(3.0).expect("Operation failed");
            vec![sqrt8 - three]
        }
        3 => {
            // Cubic B-spline has one pole at sqrt(3) - 2
            let sqrt3 = T::from_f64(3.0).expect("Operation failed").sqrt();
            let two = T::from_f64(2.0).expect("Operation failed");
            vec![sqrt3 - two]
        }
        4 => {
            // Quartic B-spline has two poles
            let val1 = T::from_f64(0.361341225285).expect("Operation failed"); // sqrt(664 - sqrt(438976)) / 8 - 13
            let val2 = T::from_f64(0.013725429297).expect("Operation failed"); // sqrt(664 + sqrt(438976)) / 8 - 13
            vec![val1, val2]
        }
        5 => {
            // Quintic B-spline has two poles
            let val1 = T::from_f64(0.430575347099).expect("Operation failed");
            let val2 = T::from_f64(0.043096288203).expect("Operation failed");
            vec![val1, val2]
        }
        _ => vec![], // Higher orders not supported
    }
}

/// Compute initial causal coefficient for B-spline filtering
#[allow(dead_code)]
fn get_initial_causal_coefficient<T: Float + FromPrimitive>(
    coeffs: &[T],
    pole: T,
    tolerance: T,
) -> T {
    let mut sum = T::zero();
    let mut z_power = T::one();
    let _abs_pole = pole.abs();

    for &coeff in coeffs {
        sum = sum + coeff * z_power;
        z_power = z_power * pole;
        if z_power.abs() < tolerance {
            break;
        }
    }

    sum
}

/// Compute initial anti-causal coefficient for B-spline filtering
#[allow(dead_code)]
fn get_initial_anti_causal_coefficient<T: Float + FromPrimitive>(coeffs: &[T], pole: T) -> T {
    let n = coeffs.len();
    if n < 2 {
        return T::zero();
    }

    let last_idx = n - 1;
    (pole / (pole * pole - T::one())) * (pole * coeffs[last_idx] + coeffs[last_idx - 1])
}

/// Apply causal filtering (forward pass)
#[allow(dead_code)]
fn apply_causal_filter<T: Float + FromPrimitive>(coeffs: &mut [T], pole: T, initialcoeff: T) {
    if coeffs.is_empty() {
        return;
    }

    coeffs[0] = initialcoeff;

    for i in 1..coeffs.len() {
        coeffs[i] = coeffs[i] + pole * coeffs[i - 1];
    }
}

/// Apply anti-causal filtering (backward pass)
#[allow(dead_code)]
fn apply_anti_causal_filter<T: Float + FromPrimitive>(coeffs: &mut [T], pole: T, initialcoeff: T) {
    if coeffs.is_empty() {
        return;
    }

    let last_idx = coeffs.len() - 1;
    coeffs[last_idx] = initialcoeff;

    for i in (0..last_idx).rev() {
        coeffs[i] = pole * (coeffs[i + 1] - coeffs[i]);
    }
}

/// Spline filter for use in interpolation
///
/// # Arguments
///
/// * `input` - Input array
/// * `order` - Spline order (default: 3)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn spline_filter<T, D>(input: &Array<T, D>, order: Option<usize>) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static,
    D: Dimension + scirs2_core::ndarray::RemoveAxis + 'static,
    usize: scirs2_core::ndarray::NdIndex<<D as scirs2_core::ndarray::Dimension>::Smaller>,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    let spline_order = order.unwrap_or(3);

    if spline_order == 0 || spline_order > 5 {
        return Err(NdimageError::InvalidInput(format!(
            "Spline order must be between 1 and 5, got {}",
            spline_order
        )));
    }

    // For orders 0 and 1, no filtering is needed
    if spline_order <= 1 {
        return Ok(input.to_owned());
    }

    // Create output array
    let mut output = input.to_owned();

    // Apply spline filtering along each axis
    for axis in 0..input.ndim() {
        spline_filter_axis(&mut output, spline_order, axis)?;
    }

    Ok(output)
}

/// Spline filter 1D for use in separable interpolation
///
/// # Arguments
///
/// * `input` - Input 1D array
/// * `order` - Spline order (default: 3)
/// * `axis` - Axis along which to filter (default: 0)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn spline_filter1d<T, D>(
    input: &Array<T, D>,
    order: Option<usize>,
    axis: Option<usize>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static,
    D: Dimension + scirs2_core::ndarray::RemoveAxis + 'static,
    usize: scirs2_core::ndarray::NdIndex<<D as scirs2_core::ndarray::Dimension>::Smaller>,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    let spline_order = order.unwrap_or(3);
    let axis_val = axis.unwrap_or(0);

    if spline_order == 0 || spline_order > 5 {
        return Err(NdimageError::InvalidInput(format!(
            "Spline order must be between 1 and 5, got {}",
            spline_order
        )));
    }

    if axis_val >= input.ndim() {
        return Err(NdimageError::InvalidInput(format!(
            "Axis {} is out of bounds for array of dimension {}",
            axis_val,
            input.ndim()
        )));
    }

    // For orders 0 and 1, no filtering is needed
    if spline_order <= 1 {
        return Ok(input.to_owned());
    }

    // Create output array
    let mut output = input.to_owned();

    // Apply spline filtering along the specified axis
    spline_filter_axis(&mut output, spline_order, axis_val)?;

    Ok(output)
}

/// Evaluate a B-spline at given positions
///
/// # Arguments
///
/// * `positions` - Positions at which to evaluate the spline
/// * `order` - Spline order (default: 3)
/// * `derivative` - Order of the derivative to evaluate (default: 0)
///
/// # Returns
///
/// * `Result<Array<T, scirs2_core::ndarray::Ix1>>` - B-spline values
#[allow(dead_code)]
pub fn bspline<T>(
    positions: &Array<T, scirs2_core::ndarray::Ix1>,
    order: Option<usize>,
    derivative: Option<usize>,
) -> NdimageResult<Array<T, scirs2_core::ndarray::Ix1>>
where
    T: Float + FromPrimitive + Debug,
{
    // Validate inputs
    let spline_order = order.unwrap_or(3);
    let deriv = derivative.unwrap_or(0);

    if spline_order == 0 || spline_order > 5 {
        return Err(NdimageError::InvalidInput(format!(
            "Spline order must be between 1 and 5, got {}",
            spline_order
        )));
    }

    if deriv > spline_order {
        return Err(NdimageError::InvalidInput(format!(
            "Derivative order must be less than or equal to spline order (got {} for order {})",
            deriv, spline_order
        )));
    }

    // Evaluate B-spline basis function at given positions
    let mut result = Array1::<T>::zeros(positions.len());

    for (i, &pos) in positions.iter().enumerate() {
        result[i] = evaluate_bspline_basis(pos, spline_order, deriv);
    }

    Ok(result)
}

/// Apply B-spline filtering along a specific axis
#[allow(dead_code)]
fn spline_filter_axis<T, D>(data: &mut Array<T, D>, order: usize, axis: usize) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Clone,
    D: Dimension + scirs2_core::ndarray::RemoveAxis,
    usize: scirs2_core::ndarray::NdIndex<<D as scirs2_core::ndarray::Dimension>::Smaller>,
{
    let poles = get_spline_poles::<T>(order);
    if poles.is_empty() {
        return Ok(());
    }

    let tolerance = T::from_f64(1e-10).expect("Operation failed");
    let axis_len = data.shape()[axis];

    // Process each 1D line along the specified axis
    for mut lane in data.axis_iter_mut(Axis(axis)) {
        let mut coeffs: Vec<T> = lane.iter().cloned().collect();

        // Apply filtering for each pole
        for &pole in &poles {
            // Forward pass (causal)
            let initial_causal = get_initial_causal_coefficient(&coeffs, pole, tolerance);
            apply_causal_filter(&mut coeffs, pole, initial_causal);

            // Backward pass (anti-causal)
            let initial_anti_causal = get_initial_anti_causal_coefficient(&coeffs, pole);
            apply_anti_causal_filter(&mut coeffs, pole, initial_anti_causal);
        }

        // Copy filtered coefficients back
        for (i, &coeff) in coeffs.iter().enumerate() {
            lane[i] = coeff;
        }
    }

    Ok(())
}

/// Build the local, uniformly-spaced knot vector for the centered, symmetric
/// cardinal B-spline basis function of the given polynomial `order`
/// (degree `n = order`): `order + 2` knots
/// `-(n+1)/2, -(n+1)/2 + 1, ..., (n+1)/2`, giving a basis function supported
/// on `[-(n+1)/2, (n+1)/2]`.
#[allow(dead_code)]
fn bspline_knots<T: Float + FromPrimitive>(order: usize) -> Vec<T> {
    let half_span = T::from_f64((order + 1) as f64 / 2.0).expect("Operation failed");
    (0..=order + 1)
        .map(|i| T::from_usize(i).expect("Operation failed") - half_span)
        .collect()
}

/// Evaluate the Cox-de Boor recursion for the degree-`k` B-spline segment
/// `B_{i,k}` defined by the local knot vector `knots`, at position `x`.
#[allow(dead_code)]
fn cox_de_boor<T: Float + FromPrimitive>(i: usize, k: usize, x: T, knots: &[T]) -> T {
    if k == 0 {
        return if knots[i] <= x && x < knots[i + 1] {
            T::one()
        } else {
            T::zero()
        };
    }

    let mut result = T::zero();

    let denom_left = knots[i + k] - knots[i];
    if denom_left != T::zero() {
        result = result + (x - knots[i]) / denom_left * cox_de_boor(i, k - 1, x, knots);
    }

    let denom_right = knots[i + k + 1] - knots[i + 1];
    if denom_right != T::zero() {
        result =
            result + (knots[i + k + 1] - x) / denom_right * cox_de_boor(i + 1, k - 1, x, knots);
    }

    result
}

/// Evaluate the `d`-th derivative of the Cox-de Boor basis function
/// `B_{i,k}` at position `x`, using the standard B-spline derivative
/// recursion applied `d` times:
/// `B'_{i,k}(x) = k * (B_{i,k-1}(x)/(t[i+k]-t[i]) - B_{i+1,k-1}(x)/(t[i+k+1]-t[i+1]))`.
#[allow(dead_code)]
fn cox_de_boor_derivative<T: Float + FromPrimitive>(
    i: usize,
    k: usize,
    x: T,
    knots: &[T],
    d: usize,
) -> T {
    if d == 0 {
        return cox_de_boor(i, k, x, knots);
    }
    if k == 0 {
        // The derivative of a piecewise-constant segment is zero wherever
        // it is defined (ignoring the measure-zero jump discontinuities).
        return T::zero();
    }

    let k_t = T::from_usize(k).expect("Operation failed");
    let mut result = T::zero();

    let denom_left = knots[i + k] - knots[i];
    if denom_left != T::zero() {
        result = result + cox_de_boor_derivative(i, k - 1, x, knots, d - 1) / denom_left;
    }

    let denom_right = knots[i + k + 1] - knots[i + 1];
    if denom_right != T::zero() {
        result = result - cox_de_boor_derivative(i + 1, k - 1, x, knots, d - 1) / denom_right;
    }

    k_t * result
}

/// Evaluate B-spline basis function at a given position
///
/// Implements the general Cox-de Boor recursion, which covers every spline
/// order and every derivative order uniformly (unlike hand-coded piecewise
/// polynomials, which would need a distinct closed form per order/derivative
/// combination and were previously only worked out for orders 0-3 at
/// derivative 0).
#[allow(dead_code)]
fn evaluate_bspline_basis<T: Float + FromPrimitive>(x: T, order: usize, derivative: usize) -> T {
    if derivative > order {
        return T::zero();
    }

    let knots: Vec<T> = bspline_knots(order);
    cox_de_boor_derivative(0, order, x, &knots, derivative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_spline_filter() {
        let input: Array2<f64> = Array2::eye(3);
        let result = spline_filter(&input, None).expect("Operation failed");
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_spline_filter1d() {
        let input: Array2<f64> = Array2::eye(3);
        let result = spline_filter1d(&input, None, None).expect("Operation failed");
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_bspline() {
        let positions = Array1::linspace(0.0, 2.0, 5);
        let result = bspline(&positions, None, None).expect("Operation failed");
        assert_eq!(result.len(), positions.len());
    }

    #[test]
    fn test_bspline_orders_and_derivatives_vs_analytic() {
        // Reference values independently computed via exact rational
        // Cox-de Boor evaluation plus symbolic differentiation (see
        // scratchpad/bspline_ref.py). Covers every advertised order 1..=5
        // and every derivative 0..=order at two representative interior,
        // non-breakpoint positions -- the pre-fix implementation only
        // handled orders 0-3 at derivative 0 and silently fell back to 0.0
        // for everything else (order 4, order 5, or any nonzero
        // derivative), which would fail essentially every case below.
        let cases: &[(usize, usize, f64, f64)] = &[
            // order, derivative, x, expected
            (1, 0, 0.3, 0.7),
            (1, 0, 1.3, 0.0),
            (1, 1, 0.3, -1.0),
            (1, 1, 1.3, 0.0),
            (2, 0, 0.3, 0.66),
            (2, 0, 1.3, 0.02),
            (2, 1, 0.3, -0.6),
            (2, 1, 1.3, -0.2),
            (2, 2, 0.3, -2.0),
            (2, 2, 1.3, 1.0),
            (3, 0, 0.3, 0.5901666667),
            (3, 0, 1.3, 0.0571666667),
            (3, 1, 0.3, -0.465),
            (3, 1, 1.3, -0.245),
            (3, 2, 0.3, -1.1),
            (3, 2, 1.3, 0.7),
            (3, 3, 0.3, 3.0),
            (3, 3, 1.3, -1.0),
            (4, 0, 0.3, 0.5447333333),
            (4, 0, 1.3, 0.0860666667),
            (4, 1, 0.3, -0.348),
            (4, 1, 1.3, -0.2813333333),
            (4, 2, 0.3, -0.98),
            (4, 2, 1.3, 0.62),
            (4, 3, 0.3, 1.8),
            (4, 3, 1.3, -0.2),
            (4, 4, 0.3, 6.0),
            (4, 4, 1.3, -4.0),
            (5, 0, 0.3, 0.5068225),
            (5, 0, 1.3, 0.1099179167),
            (5, 1, 0.3, -0.276375),
            (5, 1, 1.3, -0.2879791667),
            (5, 2, 0.3, -0.775),
            (5, 2, 1.3, 0.4758333333),
            (5, 3, 0.3, 1.35),
            (5, 3, 1.3, 0.025),
            (5, 4, 0.3, 3.0),
            (5, 4, 1.3, -2.5),
            (5, 5, 0.3, -10.0),
            (5, 5, 1.3, 5.0),
        ];

        for &(order, deriv, x, expected) in cases {
            let positions = Array1::from_vec(vec![x]);
            let result = bspline(&positions, Some(order), Some(deriv)).unwrap_or_else(|e| {
                panic!("bspline(order={order}, deriv={deriv}, x={x}) failed: {e}")
            });
            assert!(
                (result[0] - expected).abs() < 1e-9,
                "order={order} deriv={deriv} x={x}: got {} expected {expected}",
                result[0]
            );
        }
    }

    #[test]
    fn test_bspline_is_symmetric_about_zero() {
        // A B-spline basis function of any order is an even function
        // (symmetric about 0) at derivative 0.
        for order in 1..=5usize {
            let positions = Array1::from_vec(vec![0.3, -0.3, 1.3, -1.3]);
            let result = bspline(&positions, Some(order), Some(0)).expect("Operation failed");
            assert!((result[0] - result[1]).abs() < 1e-9, "order {order}");
            assert!((result[2] - result[3]).abs() < 1e-9, "order {order}");
        }
    }

    #[test]
    fn test_bspline_rejects_invalid_order() {
        let positions = Array1::from_vec(vec![0.0]);
        assert!(bspline(&positions, Some(0), None).is_err());
        assert!(bspline(&positions, Some(6), None).is_err());
    }

    #[test]
    fn test_bspline_rejects_derivative_above_order() {
        let positions = Array1::from_vec(vec![0.0]);
        assert!(bspline(&positions, Some(2), Some(3)).is_err());
    }
}
