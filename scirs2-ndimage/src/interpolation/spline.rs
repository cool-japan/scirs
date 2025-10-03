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
            let sqrt8 = T::from_f64(8.0).unwrap().sqrt();
            let three = T::from_f64(3.0).unwrap();
            vec![sqrt8 - three]
        }
        3 => {
            // Cubic B-spline has one pole at sqrt(3) - 2
            let sqrt3 = T::from_f64(3.0).unwrap().sqrt();
            let two = T::from_f64(2.0).unwrap();
            vec![sqrt3 - two]
        }
        4 => {
            // Quartic B-spline has two poles
            let val1 = T::from_f64(0.361341225285).unwrap(); // sqrt(664 - sqrt(438976)) / 8 - 13
            let val2 = T::from_f64(0.013725429297).unwrap(); // sqrt(664 + sqrt(438976)) / 8 - 13
            vec![val1, val2]
        }
        5 => {
            // Quintic B-spline has two poles
            let val1 = T::from_f64(0.430575347099).unwrap();
            let val2 = T::from_f64(0.043096288203).unwrap();
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

    let tolerance = T::from_f64(1e-10).unwrap();
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

/// Evaluate B-spline basis function at a given position
#[allow(dead_code)]
fn evaluate_bspline_basis<T: Float + FromPrimitive>(x: T, order: usize, derivative: usize) -> T {
    if derivative > order {
        return T::zero();
    }

    // For simplicity, we implement only the basic cases
    // More sophisticated implementations would use the Cox-de Boor recursion
    match order {
        0 => {
            if derivative == 0 {
                if x >= T::zero() && x < T::one() {
                    T::one()
                } else {
                    T::zero()
                }
            } else {
                T::zero()
            }
        }
        1 => {
            if derivative == 0 {
                let abs_x = x.abs();
                if abs_x < T::one() {
                    T::one() - abs_x
                } else {
                    T::zero()
                }
            } else if derivative == 1 {
                if x > T::zero() && x < T::one() {
                    -T::one()
                } else if x > -T::one() && x < T::zero() {
                    T::one()
                } else {
                    T::zero()
                }
            } else {
                T::zero()
            }
        }
        2 => {
            // Quadratic B-spline
            let abs_x = x.abs();
            if derivative == 0 {
                if abs_x < T::from_f64(0.5).unwrap() {
                    let _half = T::from_f64(0.5).unwrap();
                    let three_quarters = T::from_f64(0.75).unwrap();
                    three_quarters - x * x
                } else if abs_x < T::from_f64(1.5).unwrap() {
                    let half = T::from_f64(0.5).unwrap();
                    let val = abs_x - T::from_f64(1.5).unwrap();
                    half * val * val
                } else {
                    T::zero()
                }
            } else {
                // Derivatives for higher orders are more complex
                T::zero()
            }
        }
        3 => {
            // Cubic B-spline (most common)
            let abs_x = x.abs();
            if derivative == 0 {
                if abs_x < T::one() {
                    let two_thirds = T::from_f64(2.0 / 3.0).unwrap();
                    let half = T::from_f64(0.5).unwrap();
                    two_thirds - abs_x * abs_x + half * abs_x * abs_x * abs_x
                } else if abs_x < T::from_f64(2.0).unwrap() {
                    let one_sixth = T::from_f64(1.0 / 6.0).unwrap();
                    let val = T::from_f64(2.0).unwrap() - abs_x;
                    one_sixth * val * val * val
                } else {
                    T::zero()
                }
            } else {
                // Derivatives for cubic are more complex
                T::zero()
            }
        }
        _ => T::zero(), // Higher orders not implemented
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_spline_filter() {
        let input: Array2<f64> = Array2::eye(3);
        let result = spline_filter(&input, None).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_spline_filter1d() {
        let input: Array2<f64> = Array2::eye(3);
        let result = spline_filter1d(&input, None, None).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_bspline() {
        let positions = Array1::linspace(0.0, 2.0, 5);
        let result = bspline(&positions, None, None).unwrap();
        assert_eq!(result.len(), positions.len());
    }
}
