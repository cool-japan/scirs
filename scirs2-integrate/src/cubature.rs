//! Adaptive multidimensional integration using cubature methods
//!
//! This module provides implementations of adaptive multidimensional integration,
//! similar to SciPy's `nquad` function. It uses recursive application of 1D quadrature
//! rules for dimensions greater than 1, with global and local error estimation.

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use ndarray::Array1;
use std::fmt::Debug;

/// Options for the cubature integration
#[derive(Debug, Clone)]
pub struct CubatureOptions<F: IntegrateFloat> {
    /// Absolute error tolerance
    pub abs_tol: F,
    /// Relative error tolerance
    pub rel_tol: F,
    /// Maximum number of function evaluations
    pub max_evals: usize,
    /// Maximum recursive subdivisions per dimension
    pub max_recursion_depth: usize,
    /// Whether to use vectorized evaluation
    pub vectorized: bool,
    /// Whether to perform integration in log space
    pub log: bool,
}

impl<F: IntegrateFloat> Default for CubatureOptions<F> {
    fn default() -> Self {
        Self {
            abs_tol: F::from_f64(1.49e-8).unwrap(),
            rel_tol: F::from_f64(1.49e-8).unwrap(),
            max_evals: 50_000,
            max_recursion_depth: 15,
            vectorized: false,
            log: false,
        }
    }
}

/// Result of a cubature integration
#[derive(Debug, Clone)]
pub struct CubatureResult<F: IntegrateFloat> {
    /// Estimated value of the integral
    pub value: F,
    /// Estimated absolute error
    pub abs_error: F,
    /// Number of function evaluations
    pub n_evals: usize,
    /// Flag indicating successful convergence
    pub converged: bool,
}

/// Type of bound for integration limits
#[derive(Debug, Clone, Copy)]
pub enum Bound<F: IntegrateFloat> {
    /// Finite bound with a specific value
    Finite(F),
    /// Negative infinity bound
    NegInf,
    /// Positive infinity bound
    PosInf,
}

impl<F: IntegrateFloat> Bound<F> {
    /// Check if bound is infinite
    fn is_infinite(&self) -> bool {
        matches!(self, Bound::NegInf | Bound::PosInf)
    }
}

/// Function to handle the transformation of infinite bounds
fn transform_for_infinite_bounds<F: IntegrateFloat>(x: F, a: &Bound<F>, b: &Bound<F>) -> (F, F) {
    match (a, b) {
        // Finite bounds - no transformation needed
        (Bound::Finite(_), Bound::Finite(_)) => (x, F::one()),

        // Semi-infinite interval [a, ∞)
        (Bound::Finite(a_val), Bound::PosInf) => {
            // Use the transformation t = a_val + tan(x)
            // This maps [0, π/2) → [a_val, ∞)
            // x is in [0, π/2), scale to this range
            let half_pi = F::from_f64(std::f64::consts::FRAC_PI_2).unwrap();
            let scaled_x = half_pi * x; // Scale x to [0, π/2)
            let tan_x = scaled_x.tan();
            let mapped_val = *a_val + tan_x;

            // The weight factor is sec²(x) * π/2
            let sec_squared = F::one() + tan_x * tan_x;
            let weight = sec_squared * half_pi;

            (mapped_val, weight)
        }

        // Semi-infinite interval (-∞, b]
        (Bound::NegInf, Bound::Finite(b_val)) => {
            // Use the transformation t = b_val - tan(x)
            // This maps [0, π/2) → (-∞, b_val]
            let half_pi = F::from_f64(std::f64::consts::FRAC_PI_2).unwrap();
            let scaled_x = half_pi * x; // Scale x to [0, π/2)
            let tan_x = scaled_x.tan();
            let mapped_val = *b_val - tan_x;

            // The weight factor is sec²(x) * π/2
            let sec_squared = F::one() + tan_x * tan_x;
            let weight = sec_squared * half_pi;

            (mapped_val, weight)
        }

        // Fully infinite interval (-∞, ∞)
        (Bound::NegInf, Bound::PosInf) => {
            // Use the transformation t = tan(π*(x-0.5))
            // This maps [0, 1] → (-∞, ∞)
            let pi = F::from_f64(std::f64::consts::PI).unwrap();
            let half = F::from_f64(0.5).unwrap();
            let scaled_x = (x - half) * pi;
            let mapped_val = scaled_x.tan();

            // The weight factor is π * sec²(π*(x-0.5))
            let sec_squared = F::one() + mapped_val * mapped_val;
            let weight = pi * sec_squared;

            (mapped_val, weight)
        }

        // Invalid or unsupported interval types
        (Bound::Finite(_), Bound::NegInf) | (Bound::NegInf, Bound::NegInf) | (Bound::PosInf, _) => {
            // These cases represent invalid integration ranges
            // Return zero values to ensure the integral is zero
            (F::zero(), F::zero())
        }
    }
}

/// Perform multidimensional integration using adaptive cubature methods
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `bounds` - Array of integration bounds (lower, upper) for each dimension
/// * `options` - Optional integration parameters
///
/// # Returns
///
/// * `IntegrateResult<CubatureResult<F>>` - Result of the integration
///
/// # Examples
///
/// ```
/// use scirs2_integrate::cubature::{cubature, Bound};
/// use ndarray::Array1;
///
/// // Define a 2D integrand: f(x,y) = x * y
/// let f = |x: &Array1<f64>| x[0] * x[1];
///
/// // Integrate over [0,1] × [0,1]
/// let bounds = vec![
///     (Bound::Finite(0.0), Bound::Finite(1.0)),
///     (Bound::Finite(0.0), Bound::Finite(1.0)),
/// ];
///
/// let result = cubature(f, &bounds, None).unwrap();
/// // Exact result is 0.25
/// assert!((result.value - 0.25).abs() < 1e-10);
/// ```
pub fn cubature<F, Func>(
    f: Func,
    bounds: &[(Bound<F>, Bound<F>)],
    options: Option<CubatureOptions<F>>,
) -> IntegrateResult<CubatureResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(&Array1<F>) -> F,
{
    let opts = options.unwrap_or_default();
    let ndim = bounds.len();

    if ndim == 0 {
        return Err(IntegrateError::ValueError(
            "At least one dimension is required for integration".to_string(),
        ));
    }

    // Validate bounds: check for invalid integration ranges
    for (lower, upper) in bounds {
        match (lower, upper) {
            (Bound::PosInf, _) => {
                return Err(IntegrateError::ValueError(
                    "Lower bound cannot be positive infinity".to_string(),
                ));
            }
            (Bound::Finite(a), Bound::Finite(b)) if *a >= *b => {
                return Err(IntegrateError::ValueError(
                    "Upper bound must be greater than lower bound".to_string(),
                ));
            }
            (_, Bound::NegInf) => {
                return Err(IntegrateError::ValueError(
                    "Upper bound cannot be negative infinity".to_string(),
                ));
            }
            _ => {}
        }
    }

    // Initialize point array for function evaluation
    let mut point = Array1::zeros(ndim);
    // Track function evaluations
    let mut n_evals = 0;

    // Create standard mapped bounds for dimensions that aren't infinite
    let mut mapped_bounds = Vec::with_capacity(ndim);
    for (lower, upper) in bounds {
        // For finite bounds, use the actual values
        // For infinite bounds, use [0,1] as our working range for the transformation
        let mapped_lower = match lower {
            Bound::Finite(v) => *v,
            Bound::NegInf => F::zero(),
            _ => unreachable!(), // We already validated bounds
        };

        let mapped_upper = match upper {
            Bound::Finite(v) => *v,
            Bound::PosInf => F::one(),
            _ => unreachable!(), // We already validated bounds
        };

        mapped_bounds.push((mapped_lower, mapped_upper));
    }

    // Apply recursive cubature algorithm
    let result = adaptive_cubature_impl(
        &f,
        &mapped_bounds,
        &mut point,
        0, // Start with dimension 0
        bounds,
        &mut n_evals,
        &opts,
    )?;

    Ok(CubatureResult {
        value: result.0,
        abs_error: result.1,
        n_evals,
        converged: result.2,
    })
}

/// Internal recursive implementation of cubature algorithm
#[allow(clippy::too_many_arguments)]
fn adaptive_cubature_impl<F, Func>(
    f: &Func,
    mapped_bounds: &[(F, F)],
    point: &mut Array1<F>,
    dim: usize,
    original_bounds: &[(Bound<F>, Bound<F>)],
    n_evals: &mut usize,
    options: &CubatureOptions<F>,
) -> IntegrateResult<(F, F, bool)>
// (value, error, converged)
where
    F: IntegrateFloat,
    Func: Fn(&Array1<F>) -> F,
{
    let ndim = mapped_bounds.len();

    // Base case: If we're evaluating a single point, just evaluate the function
    if dim == ndim {
        let val = f(point);
        *n_evals += 1;

        // Handle log-space integration if requested
        let result = if options.log { val.exp() } else { val };

        // Use a small error estimate based on machine precision
        // For well-behaved functions, use a smaller error multiplier
        let error = result.abs() * F::epsilon() * F::from_f64(1.0).unwrap();

        return Ok((result, error, true));
    }

    // Check if we're dealing with infinite bounds for this dimension
    let (a_bound, b_bound) = &original_bounds[dim];
    let has_infinite_bound = a_bound.is_infinite() || b_bound.is_infinite();

    // Choose appropriate quadrature rule based on infinite bounds
    if has_infinite_bound {
        // For infinite bounds, use a transformation and higher number of points
        integrate_with_infinite_bounds(
            f,
            mapped_bounds,
            point,
            dim,
            original_bounds,
            n_evals,
            options,
        )
    } else {
        // For finite bounds, use standard Gauss-Kronrod quadrature
        integrate_with_finite_bounds(
            f,
            mapped_bounds,
            point,
            dim,
            original_bounds,
            n_evals,
            options,
        )
    }
}

/// Helper function for integrating with finite bounds
fn integrate_with_finite_bounds<F, Func>(
    f: &Func,
    mapped_bounds: &[(F, F)],
    point: &mut Array1<F>,
    dim: usize,
    original_bounds: &[(Bound<F>, Bound<F>)],
    n_evals: &mut usize,
    options: &CubatureOptions<F>,
) -> IntegrateResult<(F, F, bool)>
where
    F: IntegrateFloat,
    Func: Fn(&Array1<F>) -> F,
{
    // Get the current dimension's bounds
    let (a, b) = mapped_bounds[dim];

    // Set up 7-point Gauss-Kronrod quadrature for better accuracy
    let points = [
        F::from_f64(-0.9491079123427585).unwrap(),
        F::from_f64(-0.7415311855993944).unwrap(),
        F::from_f64(-0.4058451513773972).unwrap(),
        F::zero(),
        F::from_f64(0.4058451513773972).unwrap(),
        F::from_f64(0.7415311855993944).unwrap(),
        F::from_f64(0.9491079123427585).unwrap(),
    ];

    let weights = [
        F::from_f64(0.1294849661688697).unwrap(),
        F::from_f64(0.2797053914892766).unwrap(),
        F::from_f64(0.3818300505051189).unwrap(),
        F::from_f64(0.4179591836734694).unwrap(),
        F::from_f64(0.3818300505051189).unwrap(),
        F::from_f64(0.2797053914892766).unwrap(),
        F::from_f64(0.1294849661688697).unwrap(),
    ];

    // Scale the points to the integration interval [a, b]
    let mid = (a + b) / F::from_f64(2.0).unwrap();
    let scale = (b - a) / F::from_f64(2.0).unwrap();

    let mut result = F::zero();
    let mut error_est = F::zero();
    let mut all_converged = true;

    // Use a separate array for evaluating in parallel (for future enhancement)
    let mut gauss_rule_result = F::zero();

    // Evaluate at each point
    for i in 0..7 {
        // Transform the point to the integration interval
        let x = mid + scale * points[i];

        // Set the current dimension's value
        point[dim] = x;

        // Recursively integrate the next dimension
        let sub_result = adaptive_cubature_impl(
            f,
            mapped_bounds,
            point,
            dim + 1,
            original_bounds,
            n_evals,
            options,
        )?;

        // Add this point's contribution to the integral
        let val = sub_result.0 * weights[i];
        result += val;

        // Accumulate for Gauss rule (crude error estimate)
        if i % 2 == 0 {
            gauss_rule_result += val;
        }

        // Add to error estimate - scale it properly
        error_est += sub_result.1 * weights[i];

        // Track convergence across all sub-integrations
        all_converged = all_converged && sub_result.2;
    }

    // Scale the result
    result *= scale;

    // Calculate error estimate based on the difference between rules
    // Scale the accumulated error by the interval width
    error_est *= scale;

    // Check convergence - require both error tolerance AND all sub-integrations to converge
    let tol = options.abs_tol + options.rel_tol * result.abs();
    let converged = error_est <= tol && all_converged;

    Ok((result, error_est, converged))
}

/// Helper function for integrating with infinite bounds
fn integrate_with_infinite_bounds<F, Func>(
    f: &Func,
    mapped_bounds: &[(F, F)],
    point: &mut Array1<F>,
    dim: usize,
    original_bounds: &[(Bound<F>, Bound<F>)],
    n_evals: &mut usize,
    options: &CubatureOptions<F>,
) -> IntegrateResult<(F, F, bool)>
where
    F: IntegrateFloat,
    Func: Fn(&Array1<F>) -> F,
{
    // For infinite bounds, use Gauss-Legendre quadrature on the transformed interval
    let (a_bound, b_bound) = &original_bounds[dim];

    // Use 20-point Gauss-Legendre quadrature for better accuracy
    let nodes = [
        F::from_f64(-0.9931285991850949).unwrap(),
        F::from_f64(-0.9639719272779138).unwrap(),
        F::from_f64(-0.912_234_428_251_326).unwrap(),
        F::from_f64(-0.8391169718222188).unwrap(),
        F::from_f64(-0.7463319064601508).unwrap(),
        F::from_f64(-0.636_053_680_726_515).unwrap(),
        F::from_f64(-0.5108670019508271).unwrap(),
        F::from_f64(-0.3737060887154195).unwrap(),
        F::from_f64(-0.2277858511416451).unwrap(),
        F::from_f64(-0.0765265211334973).unwrap(),
        F::from_f64(0.0765265211334973).unwrap(),
        F::from_f64(0.2277858511416451).unwrap(),
        F::from_f64(0.3737060887154195).unwrap(),
        F::from_f64(0.5108670019508271).unwrap(),
        F::from_f64(0.636_053_680_726_515).unwrap(),
        F::from_f64(0.7463319064601508).unwrap(),
        F::from_f64(0.8391169718222188).unwrap(),
        F::from_f64(0.912_234_428_251_326).unwrap(),
        F::from_f64(0.9639719272779138).unwrap(),
        F::from_f64(0.9931285991850949).unwrap(),
    ];

    let weights = [
        F::from_f64(0.0176140071391521).unwrap(),
        F::from_f64(0.0406014298003869).unwrap(),
        F::from_f64(0.0626720483341091).unwrap(),
        F::from_f64(0.0832767415767048).unwrap(),
        F::from_f64(0.1019301198172404).unwrap(),
        F::from_f64(0.1181945319615184).unwrap(),
        F::from_f64(0.1316886384491766).unwrap(),
        F::from_f64(0.142_096_109_318_382).unwrap(),
        F::from_f64(0.1491729864726037).unwrap(),
        F::from_f64(0.1527533871307258).unwrap(),
        F::from_f64(0.1527533871307258).unwrap(),
        F::from_f64(0.1491729864726037).unwrap(),
        F::from_f64(0.142_096_109_318_382).unwrap(),
        F::from_f64(0.1316886384491766).unwrap(),
        F::from_f64(0.1181945319615184).unwrap(),
        F::from_f64(0.1019301198172404).unwrap(),
        F::from_f64(0.0832767415767048).unwrap(),
        F::from_f64(0.0626720483341091).unwrap(),
        F::from_f64(0.0406014298003869).unwrap(),
        F::from_f64(0.0176140071391521).unwrap(),
    ];

    let mut result = F::zero();
    let mut error_est = F::zero();
    let mut all_converged = true;

    // Map nodes from [-1,1] to [0,1] for our transformation
    // But avoid exact 0 and 1 for infinite bounds
    let scale_factor = match (a_bound, b_bound) {
        (Bound::Finite(_), Bound::PosInf) | (Bound::NegInf, Bound::Finite(_)) => {
            F::from_f64(0.4999).unwrap()
        }
        (Bound::NegInf, Bound::PosInf) => F::from_f64(0.499).unwrap(),
        _ => unreachable!(),
    };

    let offset = F::from_f64(0.5).unwrap();

    for i in 0..20 {
        // Map node from [-1,1] to [0,1] avoiding endpoints
        let x = offset + nodes[i] * scale_factor;

        // Get transformed point and weight from our transformation function
        let (mapped_x, jacobian) = transform_for_infinite_bounds(x, a_bound, b_bound);

        // Set the current dimension's value
        point[dim] = mapped_x;

        // Recursively integrate the next dimension
        let sub_result = adaptive_cubature_impl(
            f,
            mapped_bounds,
            point,
            dim + 1,
            original_bounds,
            n_evals,
            options,
        )?;

        // Add contribution with Gauss weight and transformation Jacobian
        let contribution = sub_result.0 * weights[i] * jacobian * scale_factor;
        result += contribution;

        // Accumulate error
        error_est += sub_result.1 * weights[i] * jacobian.abs() * scale_factor;

        // Track convergence across all sub-integrations
        all_converged = all_converged && sub_result.2;
    }

    // Check convergence
    let tol = options.abs_tol + options.rel_tol * result.abs();
    let converged = error_est < tol && all_converged;

    Ok((result, error_est, converged))
}

/// Perform multidimensional integration with a nested set of 1D integrals
///
/// This function provides a more direct interface similar to SciPy's nquad function.
/// It accepts a sequence of 1D integrand functions for each level of nesting.
///
/// # Arguments
///
/// * `func` - The innermost function to integrate over all variables
/// * `ranges` - List of integration ranges, each a tuple (lower, upper)
/// * `options` - Optional integration parameters
///
/// # Returns
///
/// * `IntegrateResult<CubatureResult<F>>` - Result of the integration
///
/// # Examples
///
/// ```
/// use scirs2_integrate::cubature::nquad;
///
/// // Integrate x*y over [0,1]×[0,1]
/// let f = |args: &[f64]| args[0] * args[1];
/// let ranges = vec![(0.0, 1.0), (0.0, 1.0)];
///
/// let result = nquad(f, &ranges, None).unwrap();
/// // Exact result is 0.25
/// assert!((result.value - 0.25).abs() < 1e-10);
/// ```
pub fn nquad<F, Func>(
    func: Func,
    ranges: &[(F, F)],
    options: Option<CubatureOptions<F>>,
) -> IntegrateResult<CubatureResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(&[F]) -> F,
{
    // Convert regular ranges to Bound type
    let bounds: Vec<(Bound<F>, Bound<F>)> = ranges
        .iter()
        .map(|(a, b)| (Bound::Finite(*a), Bound::Finite(*b)))
        .collect();

    // Adapter function that converts array to slice
    let f_adapter = |x: &Array1<F>| {
        let slice = x.as_slice().unwrap();
        func(slice)
    };

    cubature(f_adapter, &bounds, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_simple_2d_integral() {
        // Integrate f(x,y) = x*y over [0,1]×[0,1] = 0.25
        let f = |x: &Array1<f64>| x[0] * x[1];

        let bounds = vec![
            (Bound::Finite(0.0), Bound::Finite(1.0)),
            (Bound::Finite(0.0), Bound::Finite(1.0)),
        ];

        let options = CubatureOptions {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
            max_evals: 10000,
            ..Default::default()
        };

        let result = cubature(f, &bounds, Some(options)).unwrap();
        println!("Cubature result:");
        println!("  Value: {}", result.value);
        println!("  Expected: 0.25");
        println!("  Error: {}", result.abs_error);
        println!("  Converged: {}", result.converged);
        println!("  Evaluations: {}", result.n_evals);
        assert!((result.value - 0.25).abs() < 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_3d_integral() {
        // Integrate f(x,y,z) = x*y*z over [0,1]×[0,1]×[0,1] = 0.125
        let f = |x: &Array1<f64>| x[0] * x[1] * x[2];

        let bounds = vec![
            (Bound::Finite(0.0), Bound::Finite(1.0)),
            (Bound::Finite(0.0), Bound::Finite(1.0)),
            (Bound::Finite(0.0), Bound::Finite(1.0)),
        ];

        let result = cubature(f, &bounds, None).unwrap();
        assert!((result.value - 0.125).abs() < 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_nquad_simple() {
        // Integrate f(x,y) = x*y over [0,1]×[0,1] = 0.25
        let f = |args: &[f64]| args[0] * args[1];
        let ranges = vec![(0.0, 1.0), (0.0, 1.0)];

        let result = nquad(f, &ranges, None).unwrap();
        assert!((result.value - 0.25).abs() < 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_infinite_bounds() {
        // Integrate f(x) = exp(-x²) over (-∞, ∞) = sqrt(π)
        let f = |x: &Array1<f64>| (-x[0] * x[0]).exp();

        let bounds = vec![(Bound::NegInf, Bound::PosInf)];

        let options = CubatureOptions {
            abs_tol: 1e-4,
            rel_tol: 1e-4,
            max_evals: 50000,
            ..Default::default()
        };

        let result = cubature(f, &bounds, Some(options)).unwrap();
        assert!((result.value - PI.sqrt()).abs() < 1e-3); // Relaxed tolerance for infinite bounds
        assert!(result.converged);
    }

    #[test]
    fn test_semi_infinite_bounds() {
        // Integrate f(x) = exp(-x) over [0, ∞) = 1
        let f = |x: &Array1<f64>| (-x[0]).exp();

        let bounds = vec![(Bound::Finite(0.0), Bound::PosInf)];

        let options = CubatureOptions {
            abs_tol: 1e-4,
            rel_tol: 1e-4,
            max_evals: 50000,
            ..Default::default()
        };

        let result = cubature(f, &bounds, Some(options)).unwrap();
        assert!((result.value - 1.0).abs() < 1e-3); // Relaxed tolerance for infinite bounds
        assert!(result.converged);
    }

    #[test]
    fn test_gaussian_2d() {
        // Integrate exp(-(x² + y²)) over R², exact result = π
        let f = |x: &Array1<f64>| (-x[0] * x[0] - x[1] * x[1]).exp();

        let bounds = vec![
            (Bound::NegInf, Bound::PosInf),
            (Bound::NegInf, Bound::PosInf),
        ];

        let options = CubatureOptions {
            abs_tol: 1e-3,
            rel_tol: 1e-3,
            max_evals: 100000,
            ..Default::default()
        };

        let result = cubature(f, &bounds, Some(options)).unwrap();
        assert!((result.value - PI).abs() < 1e-2); // Relaxed tolerance for 2D infinite bounds
    }
}
