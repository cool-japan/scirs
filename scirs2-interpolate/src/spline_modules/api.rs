//! Public API functions for cubic spline interpolation
//!
//! This module provides SciPy-compatible public functions for creating
//! and using cubic splines. These functions offer convenient interfaces
//! that match the behavior of SciPy's interpolation functions.

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{ArrayView1, Array1};
use super::core::CubicSpline;
use super::types::SplineBoundaryCondition;

/// Create a cubic spline with SciPy-compatible interface
///
/// This function provides the same interface as SciPy's `CubicSpline` constructor,
/// allowing for easy migration from Python-based workflows.
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
/// * `bc_type` - Boundary condition type as string
///   - "natural": Zero second derivative at endpoints
///   - "not-a-knot": Continuous third derivative at second and second-to-last points
///   - "clamped": Specified first derivatives at endpoints (requires bc_values)
///   - "periodic": Function and derivatives match at endpoints
/// * `bc_values` - Boundary condition values for clamped splines (left_deriv, right_deriv)
/// * `_extrapolate` - Whether to allow extrapolation (currently unused for compatibility)
///
/// # Returns
///
/// A new `CubicSpline` object with the specified boundary conditions
///
/// # Errors
///
/// Returns an error if:
/// - Arrays have different lengths
/// - Insufficient points for the boundary condition
/// - x coordinates are not sorted
/// - Invalid boundary condition type
/// - Missing boundary condition values for clamped splines
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_interpolate::spline::cubic_spline_scipy;
///
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![0.0, 1.0, 4.0, 9.0];
///
/// // Natural boundary conditions
/// let spline1 = cubic_spline_scipy(&x.view(), &y.view(), "natural", None, false).unwrap();
///
/// // Clamped boundary conditions
/// let spline2 = cubic_spline_scipy(&x.view(), &y.view(), "clamped", Some((0.0, 6.0)), false).unwrap();
///
/// // Not-a-knot boundary conditions
/// let spline3 = cubic_spline_scipy(&x.view(), &y.view(), "not-a-knot", None, false).unwrap();
/// ```
pub fn cubic_spline_scipy<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    bc_type: &str,
    bc_values: Option<(F, F)>,
    _extrapolate: bool,
) -> InterpolateResult<CubicSpline<F>> {
    match bc_type {
        "natural" => CubicSpline::new(x, y),
        "not-a-knot" => CubicSpline::new_not_a_knot(x, y),
        "clamped" => {
            if let Some((left_deriv, right_deriv)) = bc_values {
                CubicSpline::with_boundary_condition(
                    x,
                    y,
                    SplineBoundaryCondition::Clamped(left_deriv, right_deriv),
                )
            } else {
                Err(InterpolateError::invalid_input(
                    "Clamped boundary conditions require derivative values".to_string(),
                ))
            }
        }
        "periodic" => CubicSpline::with_boundary_condition(x, y, SplineBoundaryCondition::Periodic),
        _ => Err(InterpolateError::invalid_input(format!(
            "Unknown boundary condition type: {}. Use 'natural', 'not-a-knot', 'clamped', or 'periodic'",
            bc_type
        ))),
    }
}

/// Create an interpolating spline with flexible boundary conditions
///
/// This function provides a more flexible interface for creating cubic splines
/// with various boundary conditions, similar to SciPy's `make_interp_spline`.
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
/// * `bc_type` - Boundary condition type as string
///   - "natural": Zero second derivative at endpoints
///   - "not-a-knot": Continuous third derivative at second and second-to-last points
///   - "clamped": Specified first derivatives at endpoints (requires bc_params)
///   - "periodic": Function and derivatives match at endpoints
/// * `bc_params` - Optional boundary condition parameters
///   - For "clamped": Array of [left_derivative, right_derivative]
///   - For other types: unused
///
/// # Returns
///
/// A new `CubicSpline` object with the specified boundary conditions
///
/// # Errors
///
/// Returns an error if:
/// - Arrays have different lengths
/// - Insufficient points for the boundary condition
/// - x coordinates are not sorted
/// - Invalid boundary condition type
/// - Invalid boundary condition parameters
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_interpolate::spline::make_interp_spline;
///
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![0.0, 1.0, 4.0, 9.0];
///
/// // Natural boundary conditions
/// let spline1 = make_interp_spline(&x.view(), &y.view(), "natural", None).unwrap();
///
/// // Clamped boundary conditions with derivatives
/// let bc_params = array![0.0, 6.0];  // [left_deriv, right_deriv]
/// let spline2 = make_interp_spline(&x.view(), &y.view(), "clamped", Some(&bc_params.view())).unwrap();
/// ```
pub fn make_interp_spline<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    bc_type: &str,
    bc_params: Option<&ArrayView1<F>>,
) -> InterpolateResult<CubicSpline<F>> {
    match bc_type {
        "natural" => CubicSpline::new(x, y),
        "not-a-knot" => CubicSpline::new_not_a_knot(x, y),
        "clamped" => {
            if let Some(params) = bc_params {
                if params.len() != 2 {
                    return Err(InterpolateError::invalid_input(
                        "clamped boundary conditions require 2 parameters: [first_deriv_start, first_deriv_end]".to_string(),
                    ));
                }
                CubicSpline::with_boundary_condition(
                    x,
                    y,
                    SplineBoundaryCondition::Clamped(params[0], params[1]),
                )
            } else {
                Err(InterpolateError::invalid_input(
                    "clamped boundary conditions require bc_params: [first_deriv_start, first_deriv_end]".to_string(),
                ))
            }
        }
        "periodic" => CubicSpline::with_boundary_condition(x, y, SplineBoundaryCondition::Periodic),
        _ => Err(InterpolateError::invalid_input(format!(
            "Unknown boundary condition type: {}. Use 'natural', 'not-a-knot', 'clamped', or 'periodic'",
            bc_type
        ))),
    }
}

/// Create a SciPy-compatible interpolation function
///
/// This function provides a simplified interface similar to SciPy's `interp1d`
/// with cubic spline interpolation. It returns a closure that can be used
/// to interpolate values.
///
/// # Arguments
///
/// * `x` - Known x values (must be sorted)
/// * `y` - Known y values (must have same length as x)
/// * `kind` - Interpolation kind ("cubic" for cubic spline, others may be added)
/// * `bounds_error` - Whether to raise error for out-of-bounds points
/// * `fill_value` - Value to use for out-of-bounds points if bounds_error=false
///
/// # Returns
///
/// A closure that takes an array of x values and returns interpolated y values
///
/// # Errors
///
/// Returns an error if:
/// - Input validation fails
/// - Unsupported interpolation kind
/// - Spline construction fails
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_interpolate::spline::interp1d_scipy;
///
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![0.0, 1.0, 4.0, 9.0];
///
/// let interp_fn = interp1d_scipy(&x.view(), &y.view(), "cubic", true, None).unwrap();
///
/// let x_new = array![0.5, 1.5, 2.5];
/// let y_interp = interp_fn(&x_new.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn interp1d_scipy<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    kind: &str,
    bounds_error: bool,
    fill_value: Option<F>,
) -> InterpolateResult<impl Fn(&ArrayView1<F>) -> InterpolateResult<Array1<F>>> {
    if kind != "cubic" {
        return Err(InterpolateError::invalid_input(format!(
            "Unsupported interpolation kind: {}. Only 'cubic' is supported",
            kind
        )));
    }

    let spline = CubicSpline::new(x, y)?;

    Ok(move |x_new: &ArrayView1<F>| -> InterpolateResult<Array1<F>> {
        let mut result = Array1::zeros(x_new.len());

        for (i, &xi) in x_new.iter().enumerate() {
            match spline.evaluate(xi) {
                Ok(yi) => result[i] = yi,
                Err(_) if !bounds_error => {
                    if let Some(fill) = fill_value {
                        result[i] = fill;
                    } else {
                        result[i] = F::nan();
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Ok(result)
    })
}

/// Create a spline with specified second derivatives at endpoints
///
/// This function creates a cubic spline with specified second derivative
/// boundary conditions, which is useful for certain physical constraints.
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
/// * `d2y0` - Second derivative at the left endpoint
/// * `d2yn` - Second derivative at the right endpoint
///
/// # Returns
///
/// A new `CubicSpline` object with specified second derivative boundary conditions
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_interpolate::spline::cubic_spline_second_derivative;
///
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![0.0, 1.0, 4.0, 9.0];
///
/// // Specify curvature at endpoints
/// let spline = cubic_spline_second_derivative(&x.view(), &y.view(), 1.0, -1.0).unwrap();
/// ```
pub fn cubic_spline_second_derivative<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    d2y0: F,
    d2yn: F,
) -> InterpolateResult<CubicSpline<F>> {
    CubicSpline::with_boundary_condition(
        x,
        y,
        SplineBoundaryCondition::SecondDerivative(d2y0, d2yn),
    )
}

/// Create a spline with parabolic runout boundary conditions
///
/// This creates a cubic spline with parabolic runout boundary conditions,
/// which is a specialized condition for certain applications.
///
/// # Arguments
///
/// * `x` - The x coordinates (must be sorted in ascending order)
/// * `y` - The y coordinates (must have the same length as x)
///
/// # Returns
///
/// A new `CubicSpline` object with parabolic runout boundary conditions
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_interpolate::spline::cubic_spline_parabolic_runout;
///
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![0.0, 1.0, 4.0, 9.0];
///
/// let spline = cubic_spline_parabolic_runout(&x.view(), &y.view()).unwrap();
/// ```
pub fn cubic_spline_parabolic_runout<F: InterpolationFloat>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
) -> InterpolateResult<CubicSpline<F>> {
    CubicSpline::with_boundary_condition(x, y, SplineBoundaryCondition::ParabolicRunout)
}

/// Utility function to validate boundary condition parameters
///
/// This internal helper function validates that boundary condition parameters
/// are appropriate for the chosen boundary condition type.
///
/// # Arguments
///
/// * `bc_type` - The boundary condition type
/// * `bc_params` - The boundary condition parameters
/// * `n_points` - Number of data points
///
/// # Returns
///
/// `Ok(())` if parameters are valid, error otherwise
fn validate_boundary_conditions<F: InterpolationFloat>(
    bc_type: &str,
    bc_params: Option<&ArrayView1<F>>,
    n_points: usize,
) -> InterpolateResult<()> {
    match bc_type {
        "natural" | "periodic" | "not-a-knot" => {
            if bc_params.is_some() {
                return Err(InterpolateError::invalid_input(format!(
                    "Boundary condition '{}' does not accept parameters",
                    bc_type
                )));
            }
        }
        "clamped" => {
            match bc_params {
                Some(params) => {
                    if params.len() != 2 {
                        return Err(InterpolateError::invalid_input(
                            "Clamped boundary conditions require exactly 2 parameters".to_string(),
                        ));
                    }
                }
                None => {
                    return Err(InterpolateError::invalid_input(
                        "Clamped boundary conditions require parameters".to_string(),
                    ));
                }
            }
        }
        _ => {
            return Err(InterpolateError::invalid_input(format!(
                "Unknown boundary condition type: {}",
                bc_type
            )));
        }
    }

    // Check minimum points for specific boundary conditions
    let min_points = match bc_type {
        "not-a-knot" => 4,
        _ => 3,
    };

    if n_points < min_points {
        return Err(InterpolateError::insufficient_points(
            min_points,
            n_points,
            &format!("cubic spline with {} boundary conditions", bc_type),
        ));
    }

    Ok(())
}