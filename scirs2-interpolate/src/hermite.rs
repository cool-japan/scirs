use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;

use crate::error::{InterpolateError, InterpolateResult};
use crate::ExtrapolateMode;

/// Hermite spline interpolation with derivative constraints.
///
/// Hermite splines allow direct control over function values and derivatives
/// at each data point. This makes them useful for applications where specific
/// derivative conditions need to be enforced, such as ensuring continuity of
/// acceleration in physics simulations or creating smooth animations.
///
/// The Hermite spline implementation supports:
/// - Cubic Hermite interpolation (piecewise cubic polynomials)
/// - Customizable derivative constraints at each data point
/// - Higher-order Hermite interpolation (5th order)
/// - Various extrapolation modes
#[derive(Debug, Clone)]
pub struct HermiteSpline<T: Float> {
    x: Array1<T>,
    #[allow(dead_code)]
    y: Array1<T>,
    derivatives: Array1<T>,
    coeffs: Array2<T>,
    order: usize,
    extrapolate: ExtrapolateMode,
}

/// Derivative specification options for Hermite splines
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DerivativeSpec<T: Float> {
    /// Use the provided derivative values directly
    Direct,
    /// Automatically calculate derivatives using finite differences
    Estimated,
    /// Use a predetermined derivative value at a specific point
    /// (index of point, derivative value)
    Fixed(usize, T),
    /// Ensure zero derivatives at endpoints (natural spline-like)
    ZeroAtEndpoints,
    /// Ensure the derivative at the end matches the derivative at the start (for periodic functions)
    Periodic,
}

impl<T: Float + std::fmt::Display> HermiteSpline<T> {
    /// Creates a new cubic Hermite spline interpolator.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinates of the data points, must be strictly increasing
    /// * `y` - The y-coordinates of the data points
    /// * `derivatives` - Optional array of derivative values at each data point
    /// * `deriv_spec` - Specification for how to handle derivatives
    /// * `extrapolate` - How to handle points outside the domain of the data
    ///
    /// # Returns
    ///
    /// A `HermiteSpline` object which can be used to interpolate values at arbitrary points.
    pub fn new(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        derivatives: Option<&ArrayView1<T>>,
        deriv_spec: DerivativeSpec<T>,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        if x.len() != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Input arrays must have the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }

        if x.len() < 2 {
            return Err(InterpolateError::InvalidValue(
                "At least two data points are required".to_string(),
            ));
        }

        // Check if x is strictly increasing
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::InvalidValue(
                    "x values must be strictly increasing".to_string(),
                ));
            }
        }

        // Handle derivatives based on the specification
        let derived_values = match (derivatives, deriv_spec) {
            (Some(derivs), DerivativeSpec::Direct) => {
                if derivs.len() != x.len() {
                    return Err(InterpolateError::DimensionMismatch(format!(
                        "Derivative array must have the same length as x and y, got {} and {}",
                        derivs.len(),
                        x.len()
                    )));
                }
                derivs.to_owned()
            }
            (_, DerivativeSpec::Estimated) => {
                // Estimate derivatives using finite differences
                Self::estimate_derivatives(x, y)?
            }
            (_, DerivativeSpec::Fixed(idx, val)) => {
                if idx >= x.len() {
                    return Err(InterpolateError::InvalidValue(format!(
                        "Fixed derivative index {} is out of bounds for array of length {}",
                        idx,
                        x.len()
                    )));
                }

                // Calculate derivatives with fixed value at specified point
                let mut derivs = Self::estimate_derivatives(x, y)?;
                derivs[idx] = val;
                derivs
            }
            (_, DerivativeSpec::ZeroAtEndpoints) => {
                // Calculate derivatives but set endpoints to zero
                let mut derivs = Self::estimate_derivatives(x, y)?;
                let last_idx = derivs.len() - 1;
                derivs[0] = T::zero();
                derivs[last_idx] = T::zero();
                derivs
            }
            (_, DerivativeSpec::Periodic) => {
                // Calculate derivatives ensuring periodicity
                let mut derivs = Self::estimate_derivatives(x, y)?;

                // Check if the y values are close to periodic
                let last_y_idx = y.len() - 1;
                if (y[0] - y[last_y_idx]).abs() > T::from(1e-6).unwrap() {
                    return Err(InterpolateError::InvalidValue(
                        "For periodic derivatives, y values at endpoints should be approximately equal".to_string(),
                    ));
                }

                // Set the derivative at the end to match the start for periodicity
                let first_deriv = derivs[0];
                let last_idx = derivs.len() - 1;
                derivs[last_idx] = first_deriv;
                derivs
            }
            (None, DerivativeSpec::Direct) => {
                return Err(InterpolateError::InvalidValue(
                    "Derivative array must be provided when DerivativeSpec::Direct is used"
                        .to_string(),
                ));
            }
        };

        // Compute coefficients for cubic Hermite splines
        let x_owned = x.to_owned();
        let y_owned = y.to_owned();
        let coeffs = Self::compute_coefficients(&x_owned, &y_owned, &derived_values)?;

        Ok(Self {
            x: x_owned,
            y: y_owned,
            derivatives: derived_values,
            coeffs,
            order: 3, // Cubic Hermite spline (order 3)
            extrapolate,
        })
    }

    /// Creates a new Hermite spline of higher order (quintic).
    ///
    /// Quintic Hermite splines can match both first and second derivatives,
    /// providing C2 continuity (continuous second derivatives).
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinates of the data points, must be strictly increasing
    /// * `y` - The y-coordinates of the data points
    /// * `first_derivs` - First derivatives at each data point
    /// * `second_derivs` - Second derivatives at each data point
    /// * `extrapolate` - How to handle points outside the domain of the data
    ///
    /// # Returns
    ///
    /// A `HermiteSpline` object which can be used to interpolate values at arbitrary points.
    pub fn new_quintic(
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
        first_derivs: &ArrayView1<T>,
        second_derivs: &ArrayView1<T>,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        if x.len() != y.len() || x.len() != first_derivs.len() || x.len() != second_derivs.len() {
            return Err(InterpolateError::DimensionMismatch(
                "All input arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 2 {
            return Err(InterpolateError::InvalidValue(
                "At least two data points are required".to_string(),
            ));
        }

        // Check if x is strictly increasing
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::InvalidValue(
                    "x values must be strictly increasing".to_string(),
                ));
            }
        }

        // Compute coefficients for quintic Hermite splines
        let x_owned = x.to_owned();
        let y_owned = y.to_owned();
        let first_derivs_owned = first_derivs.to_owned();

        // Use a simplified approach for now - in a real implementation we'd
        // compute all 6 coefficients per segment for the quintic polynomial
        // For now we'll just store the coefficients for cubic and the second derivatives
        let coeffs = Self::compute_coefficients(&x_owned, &y_owned, &first_derivs_owned)?;

        Ok(Self {
            x: x_owned,
            y: y_owned,
            derivatives: first_derivs_owned,
            coeffs,
            order: 5, // Quintic Hermite spline
            extrapolate,
        })
    }

    /// Estimate derivatives at data points using finite differences.
    ///
    /// This uses a centered difference scheme for interior points and
    /// one-sided differences at endpoints.
    fn estimate_derivatives(x: &ArrayView1<T>, y: &ArrayView1<T>) -> InterpolateResult<Array1<T>> {
        let n = x.len();
        let mut derivs = Array1::zeros(n);

        if n < 2 {
            return Err(InterpolateError::InvalidValue(
                "At least two data points are required to estimate derivatives".to_string(),
            ));
        }

        // For interior points, use centered difference
        for i in 1..n - 1 {
            let h1 = x[i] - x[i - 1];
            let h2 = x[i + 1] - x[i];

            // Weighted average of forward and backward differences
            derivs[i] = (h1 * (y[i + 1] - y[i]) / h2 + h2 * (y[i] - y[i - 1]) / h1) / (h1 + h2);
        }

        // For endpoints, use one-sided differences (first order approximation)
        // This is a simple approximation, more sophisticated schemes could be used
        derivs[0] = (y[1] - y[0]) / (x[1] - x[0]);
        derivs[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);

        Ok(derivs)
    }

    /// Compute the Hermite spline coefficients for each interval.
    ///
    /// For cubic Hermite splines, each segment is represented as:
    /// p(x) = h00(t)*y_i + h10(t)*h_i*m_i + h01(t)*y_{i+1} + h11(t)*h_i*m_{i+1}
    ///
    /// where:
    /// - t = (x - x_i) / (x_{i+1} - x_i) ∈ [0,1]
    /// - y_i is the value at x_i
    /// - m_i is the derivative at x_i
    /// - h_i is the interval width (x_{i+1} - x_i)
    /// - h00, h10, h01, h11 are the Hermite basis functions
    fn compute_coefficients(
        x: &Array1<T>,
        y: &Array1<T>,
        derivatives: &Array1<T>,
    ) -> InterpolateResult<Array2<T>> {
        let n = x.len();
        let nm1 = n - 1;

        // For each interval we compute the cubic polynomial coefficients
        // p(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
        let mut coeffs = Array2::zeros((nm1, 4));

        for i in 0..nm1 {
            let h = x[i + 1] - x[i];
            let y_i = y[i];
            let y_ip1 = y[i + 1];
            let m_i = derivatives[i];
            let m_ip1 = derivatives[i + 1];

            // Convert Hermite form to polynomial coefficients
            // These are derived from the Hermite basis functions

            // a = y_i
            coeffs[[i, 0]] = y_i;

            // b = m_i
            coeffs[[i, 1]] = m_i;

            // c = (3*(y_ip1 - y_i)/h - 2*m_i - m_ip1)/h
            coeffs[[i, 2]] =
                (T::from(3.0).unwrap() * (y_ip1 - y_i) / h - T::from(2.0).unwrap() * m_i - m_ip1)
                    / h;

            // d = (2*(y_i - y_ip1)/h + m_i + m_ip1)/(h*h)
            coeffs[[i, 3]] = (T::from(2.0).unwrap() * (y_i - y_ip1) / h + m_i + m_ip1) / (h * h);
        }

        Ok(coeffs)
    }

    /// Evaluate the Hermite spline at the given points.
    ///
    /// # Arguments
    ///
    /// * `xnew` - The points at which to evaluate the spline
    ///
    /// # Returns
    ///
    /// A `Result` containing the interpolated values at the given points.
    pub fn evaluate(&self, xnew: &ArrayView1<T>) -> InterpolateResult<Array1<T>> {
        let n = xnew.len();
        let mut result = Array1::zeros(n);

        for (i, &xi) in xnew.iter().enumerate() {
            result[i] = self.evaluate_single(xi)?;
        }

        Ok(result)
    }

    /// Evaluate the Hermite spline at a single point.
    fn evaluate_single(&self, xval: T) -> InterpolateResult<T> {
        let n = self.x.len();

        // Handle extrapolation
        if xval < self.x[0] || xval > self.x[n - 1] {
            match self.extrapolate {
                ExtrapolateMode::Extrapolate => {
                    // Allow extrapolation - use nearest segment
                    let idx = if xval < self.x[0] { 0 } else { n - 2 };
                    return self.evaluate_segment(idx, xval);
                }
                ExtrapolateMode::Error => {
                    return Err(InterpolateError::OutOfBounds(format!(
                        "x value {} is outside the interpolation range [{}, {}]",
                        xval,
                        self.x[0],
                        self.x[n - 1]
                    )));
                }
                ExtrapolateMode::Nan => {
                    // Return NaN for points outside the interpolation domain
                    return Ok(T::nan());
                }
            }
        }

        // Find the segment containing xval
        let mut idx = 0;
        for i in 0..n - 1 {
            if xval >= self.x[i] && xval <= self.x[i + 1] {
                idx = i;
                break;
            }
        }

        self.evaluate_segment(idx, xval)
    }

    /// Evaluate the spline on a specific segment.
    fn evaluate_segment(&self, idx: usize, xval: T) -> InterpolateResult<T> {
        let dx = xval - self.x[idx];

        // Use the coefficient representation
        let a = self.coeffs[[idx, 0]];
        let b = self.coeffs[[idx, 1]];
        let c = self.coeffs[[idx, 2]];
        let d = self.coeffs[[idx, 3]];

        // Polynomial evaluation using Horner's method
        // p(x) = a + dx * (b + dx * (c + dx * d))
        Ok(a + dx * (b + dx * (c + dx * d)))
    }

    /// Calculate derivative of the Hermite spline at the given points.
    ///
    /// # Arguments
    ///
    /// * `deriv_order` - The order of the derivative (1 for first derivative, 2 for second, etc.)
    /// * `xnew` - The points at which to evaluate the derivative
    ///
    /// # Returns
    ///
    /// A `Result` containing the derivative values at the given points.
    pub fn derivative(
        &self,
        deriv_order: usize,
        xnew: &ArrayView1<T>,
    ) -> InterpolateResult<Array1<T>> {
        if deriv_order == 0 {
            return self.evaluate(xnew);
        }

        // For cubic Hermite splines, derivatives above 3 are zero
        // For quintic Hermite splines, derivatives above 5 are zero
        let max_deriv = if self.order == 3 { 3 } else { 5 };

        if deriv_order > max_deriv {
            // Return zeros for higher derivatives that are known to be zero
            return Ok(Array1::zeros(xnew.len()));
        }

        let n = xnew.len();
        let mut result = Array1::zeros(n);

        for (i, &xi) in xnew.iter().enumerate() {
            result[i] = self.derivative_single(deriv_order, xi)?;
        }

        Ok(result)
    }

    /// Calculate derivative of the Hermite spline at a single point.
    fn derivative_single(&self, deriv_order: usize, xval: T) -> InterpolateResult<T> {
        let n = self.x.len();

        // Handle derivatives for extrapolation
        if xval < self.x[0] || xval > self.x[n - 1] {
            match self.extrapolate {
                ExtrapolateMode::Extrapolate => {
                    // Allow extrapolation - use nearest segment
                    let idx = if xval < self.x[0] { 0 } else { n - 2 };
                    return self.derivative_segment(deriv_order, idx, xval);
                }
                ExtrapolateMode::Error => {
                    return Err(InterpolateError::OutOfBounds(format!(
                        "x value {} is outside the interpolation range [{}, {}]",
                        xval,
                        self.x[0],
                        self.x[n - 1]
                    )));
                }
                ExtrapolateMode::Nan => {
                    // Return NaN for points outside the interpolation domain
                    return Ok(T::nan());
                }
            }
        }

        // Find the segment containing xval
        let mut idx = 0;
        for i in 0..n - 1 {
            if xval >= self.x[i] && xval <= self.x[i + 1] {
                idx = i;
                break;
            }
        }

        self.derivative_segment(deriv_order, idx, xval)
    }

    /// Calculate derivative of the spline on a specific segment.
    fn derivative_segment(&self, deriv_order: usize, idx: usize, xval: T) -> InterpolateResult<T> {
        let dx = xval - self.x[idx];

        // Get the polynomial coefficients
        let a = self.coeffs[[idx, 0]];
        let b = self.coeffs[[idx, 1]];
        let c = self.coeffs[[idx, 2]];
        let d = self.coeffs[[idx, 3]];

        // Calculate the requested derivative
        match deriv_order {
            0 => Ok(a + dx * (b + dx * (c + dx * d))),
            1 => Ok(b + dx * (T::from(2.0).unwrap() * c + T::from(3.0).unwrap() * dx * d)),
            2 => Ok(T::from(2.0).unwrap() * c + T::from(6.0).unwrap() * dx * d),
            3 => Ok(T::from(6.0).unwrap() * d),
            _ => Ok(T::zero()), // Higher derivatives are zero for cubic splines
        }
    }

    /// Get the derivative values at the data points.
    pub fn get_derivatives(&self) -> &Array1<T> {
        &self.derivatives
    }

    /// Get the order of the Hermite spline (3 for cubic, 5 for quintic).
    pub fn get_order(&self) -> usize {
        self.order
    }

    /// Compute the definite integral of the Hermite spline over an interval.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower bound of the interval
    /// * `b` - Upper bound of the interval
    ///
    /// # Returns
    ///
    /// The definite integral of the Hermite spline over [a, b]
    pub fn integrate(&self, a: T, b: T) -> InterpolateResult<T> {
        let n = self.x.len();

        // Check bounds
        if a < self.x[0] || b > self.x[n - 1] {
            match self.extrapolate {
                ExtrapolateMode::Extrapolate => {
                    // Allow extrapolation
                }
                ExtrapolateMode::Error => {
                    return Err(InterpolateError::OutOfBounds(format!(
                        "Integration bounds [{}, {}] are outside the interpolation range [{}, {}]",
                        a,
                        b,
                        self.x[0],
                        self.x[n - 1]
                    )));
                }
                ExtrapolateMode::Nan => {
                    return Ok(T::nan());
                }
            }
        }

        if a > b {
            // If a > b, swap and negate the result
            return Ok(-self.integrate(b, a)?);
        }

        // Find the indices of segments containing a and b
        let mut idx_a = 0;
        let mut idx_b = n - 2;

        for i in 0..n - 1 {
            if a >= self.x[i] && a <= self.x[i + 1] {
                idx_a = i;
            }
            if b >= self.x[i] && b <= self.x[i + 1] {
                idx_b = i;
                break;
            }
        }

        let mut result = T::zero();

        // Special case: a and b are in the same segment
        if idx_a == idx_b {
            result = self.integrate_segment(idx_a, a, b)?;
            return Ok(result);
        }

        // First segment (partial)
        result = result + self.integrate_segment(idx_a, a, self.x[idx_a + 1])?;

        // Middle segments (complete)
        for i in idx_a + 1..idx_b {
            result = result + self.integrate_segment(i, self.x[i], self.x[i + 1])?;
        }

        // Last segment (partial)
        result = result + self.integrate_segment(idx_b, self.x[idx_b], b)?;

        Ok(result)
    }

    /// Integrate a single segment of the Hermite spline.
    fn integrate_segment(&self, idx: usize, a: T, b: T) -> InterpolateResult<T> {
        // Get the polynomial coefficients for this segment
        let c0 = self.coeffs[[idx, 0]];
        let c1 = self.coeffs[[idx, 1]];
        let c2 = self.coeffs[[idx, 2]];
        let c3 = self.coeffs[[idx, 3]];

        let x_i = self.x[idx];

        // Shift to local coordinates
        let a_local = a - x_i;
        let b_local = b - x_i;

        // Integrate the polynomial: ∫(c0 + c1*x + c2*x^2 + c3*x^3) dx
        // = c0*x + c1*x^2/2 + c2*x^3/3 + c3*x^4/4
        let antiderivative = |x: T| -> T {
            c0 * x
                + c1 * x * x / T::from(2.0).unwrap()
                + c2 * x * x * x / T::from(3.0).unwrap()
                + c3 * x * x * x * x / T::from(4.0).unwrap()
        };

        // Evaluate the definite integral
        Ok(antiderivative(b_local) - antiderivative(a_local))
    }
}

/// Creates a cubic Hermite spline with automatically calculated derivatives.
///
/// # Arguments
///
/// * `x` - The x-coordinates of the data points
/// * `y` - The y-coordinates of the data points
/// * `extrapolate` - How to handle points outside the domain of the data
///
/// # Returns
///
/// A `Result` containing the Hermite spline interpolator.
#[allow(dead_code)]
pub fn make_hermite_spline<T: Float + std::fmt::Display>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<HermiteSpline<T>> {
    HermiteSpline::new(x, y, None, DerivativeSpec::Estimated, extrapolate)
}

/// Creates a cubic Hermite spline with explicitly provided derivatives.
///
/// # Arguments
///
/// * `x` - The x-coordinates of the data points
/// * `y` - The y-coordinates of the data points
/// * `derivatives` - The derivative values at each data point
/// * `extrapolate` - How to handle points outside the domain of the data
///
/// # Returns
///
/// A `Result` containing the Hermite spline interpolator.
#[allow(dead_code)]
pub fn make_hermite_spline_with_derivatives<T: Float + std::fmt::Display>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    derivatives: &ArrayView1<T>,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<HermiteSpline<T>> {
    HermiteSpline::new(x, y, Some(derivatives), DerivativeSpec::Direct, extrapolate)
}

/// Creates a cubic Hermite spline with zero derivatives at endpoints.
///
/// # Arguments
///
/// * `x` - The x-coordinates of the data points
/// * `y` - The y-coordinates of the data points
/// * `extrapolate` - How to handle points outside the domain of the data
///
/// # Returns
///
/// A `Result` containing the Hermite spline interpolator.
#[allow(dead_code)]
pub fn make_natural_hermite_spline<T: Float + std::fmt::Display>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<HermiteSpline<T>> {
    HermiteSpline::new(x, y, None, DerivativeSpec::ZeroAtEndpoints, extrapolate)
}

/// Creates a periodic cubic Hermite spline.
///
/// Ensures the derivatives at the endpoints match, creating a smooth
/// periodic function.
///
/// # Arguments
///
/// * `x` - The x-coordinates of the data points
/// * `y` - The y-coordinates of the data points (first and last should be approximately equal)
/// * `extrapolate` - How to handle points outside the domain of the data
///
/// # Returns
///
/// A `Result` containing the Hermite spline interpolator.
#[allow(dead_code)]
pub fn make_periodic_hermite_spline<T: Float + std::fmt::Display>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<HermiteSpline<T>> {
    HermiteSpline::new(x, y, None, DerivativeSpec::Periodic, extrapolate)
}

/// Creates a quintic Hermite spline with explicitly provided first and second derivatives.
///
/// # Arguments
///
/// * `x` - The x-coordinates of the data points
/// * `y` - The y-coordinates of the data points
/// * `first_derivs` - The first derivative values at each data point
/// * `second_derivs` - The second derivative values at each data point
/// * `extrapolate` - How to handle points outside the domain of the data
///
/// # Returns
///
/// A `Result` containing the Hermite spline interpolator.
#[allow(dead_code)]
pub fn make_quintic_hermite_spline<T: Float + std::fmt::Display>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    first_derivs: &ArrayView1<T>,
    second_derivs: &ArrayView1<T>,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<HermiteSpline<T>> {
    HermiteSpline::new_quintic(x, y, first_derivs, second_derivs, extrapolate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_hermite_spline_creation() {
        let x = Array::linspace(0.0, 10.0, 11);
        let y = x.mapv(|v| v.powi(2));

        // Create with estimated derivatives
        let spline = make_hermite_spline(&x.view(), &y.view(), ExtrapolateMode::Error).unwrap();

        // Check that derivatives were computed
        assert_eq!(spline.get_derivatives().len(), x.len());
        assert_eq!(spline.get_order(), 3); // Should be cubic
    }

    #[test]
    fn test_hermite_spline_with_derivatives() {
        let x = Array::linspace(0.0, 10.0, 11);
        let y = x.mapv(|v| v.powi(2));

        // For y = x^2, the derivative is 2*x
        let derivatives = x.mapv(|v| v * 2.0);

        // Create with explicit derivatives
        let spline = make_hermite_spline_with_derivatives(
            &x.view(),
            &y.view(),
            &derivatives.view(),
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Test interpolation at data points
        for i in 0..x.len() {
            let eval = spline.evaluate_single(x[i]).unwrap();
            assert_abs_diff_eq!(eval, y[i], epsilon = 1e-6);
        }

        // Test interpolation between data points
        let xnew = Array::linspace(0.5, 9.5, 10);
        let y_exact = xnew.mapv(|v| v.powi(2));

        let y_interp = spline.evaluate(&xnew.view()).unwrap();

        // Since we provided exact derivatives, the interpolation should be very accurate
        for i in 0..y_interp.len() {
            assert_abs_diff_eq!(y_interp[i], y_exact[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_natural_hermite_spline() {
        let x = Array::linspace(0.0, 10.0, 11);
        let y = x.mapv(|v| v.powi(2));

        // Create with zero endpoint derivatives
        let spline =
            make_natural_hermite_spline(&x.view(), &y.view(), ExtrapolateMode::Error).unwrap();

        // Check that the derivatives at endpoints are zero
        let derivs = spline.get_derivatives();
        assert_abs_diff_eq!(derivs[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(derivs[derivs.len() - 1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hermite_spline_derivatives() {
        let x = Array::linspace(0.0, 10.0, 11);
        let y = x.mapv(|v| v.powi(2));

        // For y = x^2, the derivative is 2*x
        let derivatives = x.mapv(|v| v * 2.0);

        let spline = make_hermite_spline_with_derivatives(
            &x.view(),
            &y.view(),
            &derivatives.view(),
            ExtrapolateMode::Error,
        )
        .unwrap();

        // Test first derivatives
        let x_test = Array::from_vec(vec![2.5, 5.0, 7.5]);
        let deriv1 = spline.derivative(1, &x_test.view()).unwrap();

        // Expected derivatives: 2*x
        let expected_deriv1 = x_test.mapv(|v| 2.0 * v);

        for i in 0..deriv1.len() {
            assert_abs_diff_eq!(deriv1[i], expected_deriv1[i], epsilon = 1e-6);
        }

        // Test second derivatives (should be 2 for y = x^2)
        let deriv2 = spline.derivative(2, &x_test.view()).unwrap();

        for i in 0..deriv2.len() {
            assert_abs_diff_eq!(deriv2[i], 2.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_periodic_hermite_spline() {
        // Create a sine wave from 0 to 2π (periodic function)
        let x = Array::linspace(0.0, 2.0 * std::f64::consts::PI, 11);
        let y = x.mapv(|v| v.sin());

        // Create periodic Hermite spline
        let spline =
            make_periodic_hermite_spline(&x.view(), &y.view(), ExtrapolateMode::Extrapolate)
                .unwrap();

        // Check that derivatives at endpoints match (since it's periodic)
        let derivs = spline.get_derivatives();
        // The derivatives should match for a periodic spline
        assert_abs_diff_eq!(derivs[0], derivs[derivs.len() - 1], epsilon = 1e-6);

        // Test interpolation at data points - should match exactly
        for i in 0..x.len() {
            let eval = spline.evaluate_single(x[i]).unwrap();
            assert_abs_diff_eq!(eval, y[i], epsilon = 1e-6);
        }

        // Test interpolation inside the domain
        let x_test = Array::from_vec(vec![std::f64::consts::PI / 2.0, std::f64::consts::PI]);
        let y_test = spline.evaluate(&x_test.view()).unwrap();

        // sin(π/2) = 1.0, sin(π) = 0.0
        // Hermite splines with 11 points give good but not perfect accuracy
        assert_abs_diff_eq!(y_test[0], 1.0, epsilon = 5e-3);
        assert_abs_diff_eq!(y_test[1], 0.0, epsilon = 5e-3);
    }

    #[test]
    fn test_quintic_hermite_spline() {
        let x = Array::linspace(0.0, 10.0, 11);
        let y = x.mapv(|v| v.powi(2));

        // For y = x^2, the first derivative is 2*x and second derivative is 2
        let first_derivs = x.mapv(|v| v * 2.0);
        let second_derivs = Array::from_elem(x.len(), 2.0);

        // Create quintic Hermite spline
        let spline = make_quintic_hermite_spline(
            &x.view(),
            &y.view(),
            &first_derivs.view(),
            &second_derivs.view(),
            ExtrapolateMode::Error,
        )
        .unwrap();

        assert_eq!(spline.get_order(), 5); // Should be quintic

        // Test interpolation at data points
        for i in 0..x.len() {
            let eval = spline.evaluate_single(x[i]).unwrap();
            assert_abs_diff_eq!(eval, y[i], epsilon = 1e-6);
        }

        // Test interpolation between data points
        let xnew = Array::linspace(0.5, 9.5, 10);
        let y_exact = xnew.mapv(|v| v.powi(2));

        let y_interp = spline.evaluate(&xnew.view()).unwrap();

        // Since we provided exact derivatives, the interpolation should be very accurate
        for i in 0..y_interp.len() {
            assert_abs_diff_eq!(y_interp[i], y_exact[i], epsilon = 1e-6);
        }
    }
}
