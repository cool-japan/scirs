//! Gaussian quadrature integration methods
//!
//! This module provides implementations of numerical integration using
//! Gaussian quadrature methods, which are generally more accurate than
//! simpler methods like the trapezoid rule or Simpson's rule for
//! functions that can be well-approximated by polynomials.

use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Gauss-Legendre quadrature nodes and weights
#[derive(Debug, Clone)]
pub struct GaussLegendreQuadrature<F: Float> {
    /// Quadrature nodes (points) on the interval [-1, 1]
    pub nodes: Array1<F>,
    /// Quadrature weights
    pub weights: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> GaussLegendreQuadrature<F> {
    /// Create a new Gauss-Legendre quadrature with the given number of points
    ///
    /// # Arguments
    ///
    /// * `n` - Number of quadrature points (must be at least 1)
    ///
    /// # Returns
    ///
    /// * `IntegrateResult<GaussLegendreQuadrature<F>>` - The quadrature nodes and weights
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_integrate::gaussian::GaussLegendreQuadrature;
    ///
    /// let quad = GaussLegendreQuadrature::<f64>::new(5).unwrap();
    /// assert_eq!(quad.nodes.len(), 5);
    /// assert_eq!(quad.weights.len(), 5);
    /// ```
    pub fn new(n: usize) -> IntegrateResult<Self> {
        if n == 0 {
            return Err(IntegrateError::ValueError(
                "Number of quadrature points must be at least 1".to_string(),
            ));
        }

        // For common small orders, use pre-computed values for efficiency and accuracy
        match n {
            1 => Ok(Self::gauss_legendre_1()),
            2 => Ok(Self::gauss_legendre_2()),
            3 => Ok(Self::gauss_legendre_3()),
            4 => Ok(Self::gauss_legendre_4()),
            5 => Ok(Self::gauss_legendre_5()),
            // Add more pre-computed cases as needed
            10 => Ok(Self::gauss_legendre_10()),
            // For larger n, we could implement a more general algorithm
            // but for this implementation we'll restrict to known cases
            _ => Err(IntegrateError::NotImplementedError(format!(
                "Gauss-Legendre quadrature with {} points is not implemented",
                n
            ))),
        }
    }

    // Pre-computed nodes and weights for n=1
    fn gauss_legendre_1() -> Self {
        GaussLegendreQuadrature {
            nodes: Array1::from_vec(vec![F::zero()]),
            weights: Array1::from_vec(vec![F::from_f64(2.0).unwrap()]),
        }
    }

    // Pre-computed nodes and weights for n=2
    fn gauss_legendre_2() -> Self {
        // Correct values for 2-point Gauss-Legendre quadrature
        let nodes = vec![
            F::from_f64(-0.5773502691896257).unwrap(), // -1/sqrt(3)
            F::from_f64(0.5773502691896257).unwrap(),  // 1/sqrt(3)
        ];

        let weights = vec![F::from_f64(1.0).unwrap(), F::from_f64(1.0).unwrap()];

        GaussLegendreQuadrature {
            nodes: Array1::from_vec(nodes),
            weights: Array1::from_vec(weights),
        }
    }

    // Pre-computed nodes and weights for n=3
    fn gauss_legendre_3() -> Self {
        // Correct values for 3-point Gauss-Legendre quadrature
        let nodes = vec![
            F::from_f64(-0.7745966692414834).unwrap(), // -sqrt(3/5)
            F::zero(),
            F::from_f64(0.7745966692414834).unwrap(), // sqrt(3/5)
        ];

        let weights = vec![
            F::from_f64(5.0 / 9.0).unwrap(),
            F::from_f64(8.0 / 9.0).unwrap(),
            F::from_f64(5.0 / 9.0).unwrap(),
        ];

        GaussLegendreQuadrature {
            nodes: Array1::from_vec(nodes),
            weights: Array1::from_vec(weights),
        }
    }

    // Pre-computed nodes and weights for n=4
    fn gauss_legendre_4() -> Self {
        // Correct values for 4-point Gauss-Legendre quadrature
        let nodes = vec![
            F::from_f64(-0.8611363115940526).unwrap(),
            F::from_f64(-0.3399810435848563).unwrap(),
            F::from_f64(0.3399810435848563).unwrap(),
            F::from_f64(0.8611363115940526).unwrap(),
        ];

        let weights = vec![
            F::from_f64(0.3478548451374538).unwrap(),
            F::from_f64(0.6521451548625461).unwrap(),
            F::from_f64(0.6521451548625461).unwrap(),
            F::from_f64(0.3478548451374538).unwrap(),
        ];

        GaussLegendreQuadrature {
            nodes: Array1::from_vec(nodes),
            weights: Array1::from_vec(weights),
        }
    }

    // Pre-computed nodes and weights for n=5
    fn gauss_legendre_5() -> Self {
        // Correct values for 5-point Gauss-Legendre quadrature
        let nodes = vec![
            F::from_f64(-0.906_179_845_938_664).unwrap(),
            F::from_f64(-0.538_469_310_105_683).unwrap(),
            F::zero(),
            F::from_f64(0.538_469_310_105_683).unwrap(),
            F::from_f64(0.906_179_845_938_664).unwrap(),
        ];

        let weights = vec![
            F::from_f64(0.2369268850561891).unwrap(),
            F::from_f64(0.4786286704993665).unwrap(),
            F::from_f64(0.5688888888888889).unwrap(),
            F::from_f64(0.4786286704993665).unwrap(),
            F::from_f64(0.2369268850561891).unwrap(),
        ];

        GaussLegendreQuadrature {
            nodes: Array1::from_vec(nodes),
            weights: Array1::from_vec(weights),
        }
    }

    // Pre-computed nodes and weights for n=10
    fn gauss_legendre_10() -> Self {
        // These values are pre-computed high-precision values for n=10
        let nodes = vec![
            F::from_f64(-0.9739065285171717).unwrap(),
            F::from_f64(-0.8650633666889845).unwrap(),
            F::from_f64(-0.6794095682990244).unwrap(),
            F::from_f64(-0.4333953941292472).unwrap(),
            F::from_f64(-0.1488743389816312).unwrap(),
            F::from_f64(0.1488743389816312).unwrap(),
            F::from_f64(0.4333953941292472).unwrap(),
            F::from_f64(0.6794095682990244).unwrap(),
            F::from_f64(0.8650633666889845).unwrap(),
            F::from_f64(0.9739065285171717).unwrap(),
        ];

        let weights = vec![
            F::from_f64(0.0666713443086881).unwrap(),
            F::from_f64(0.1494513491505806).unwrap(),
            F::from_f64(0.219_086_362_515_982).unwrap(),
            F::from_f64(0.2692667193099963).unwrap(),
            F::from_f64(0.2955242247147529).unwrap(),
            F::from_f64(0.2955242247147529).unwrap(),
            F::from_f64(0.2692667193099963).unwrap(),
            F::from_f64(0.219_086_362_515_982).unwrap(),
            F::from_f64(0.1494513491505806).unwrap(),
            F::from_f64(0.0666713443086881).unwrap(),
        ];

        GaussLegendreQuadrature {
            nodes: Array1::from_vec(nodes),
            weights: Array1::from_vec(weights),
        }
    }

    /// Apply the quadrature rule to integrate a function over [a, b]
    ///
    /// # Arguments
    ///
    /// * `f` - The function to integrate
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// * The approximate value of the integral
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_integrate::gaussian::GaussLegendreQuadrature;
    ///
    /// // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
    /// let quad = GaussLegendreQuadrature::<f64>::new(5).unwrap();
    /// let result = quad.integrate(|x| x * x, 0.0, 1.0);
    /// assert!((result - 1.0/3.0).abs() < 1e-10);
    /// ```
    pub fn integrate<Func>(&self, f: Func, a: F, b: F) -> F
    where
        Func: Fn(F) -> F,
    {
        // Change of variables from [-1, 1] to [a, b]
        let mid = (a + b) / F::from_f64(2.0).unwrap();
        let half_length = (b - a) / F::from_f64(2.0).unwrap();

        // Apply quadrature rule
        let mut result = F::zero();
        for (i, &x) in self.nodes.iter().enumerate() {
            let transformed_x = mid + half_length * x;
            result = result + self.weights[i] * f(transformed_x);
        }

        // Scale by half-length due to the change of variables
        result * half_length
    }
}

/// Integrate a function using Gauss-Legendre quadrature
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `n` - Number of quadrature points (more points generally give higher accuracy)
///
/// # Returns
///
/// * `IntegrateResult<F>` - The approximate value of the integral
///
/// # Examples
///
/// ```ignore
/// use scirs2_integrate::gaussian::gauss_legendre;
///
/// // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
/// let result = gauss_legendre(|x: f64| x * x, 0.0, 1.0, 5).unwrap();
/// assert!((result - 1.0/3.0).abs() < 1e-10);
/// ```
pub fn gauss_legendre<F, Func>(f: Func, a: F, b: F, n: usize) -> IntegrateResult<F>
where
    F: Float + FromPrimitive + Debug,
    Func: Fn(F) -> F,
{
    let quadrature = GaussLegendreQuadrature::new(n)?;
    Ok(quadrature.integrate(f, a, b))
}

/// Integrate a function over multiple dimensions using Gauss-Legendre quadrature
///
/// # Arguments
///
/// * `f` - The multidimensional function to integrate
/// * `ranges` - Array of integration ranges (a, b) for each dimension
/// * `n_points` - Number of quadrature points to use for each dimension
///
/// # Returns
///
/// * `IntegrateResult<F>` - The approximate value of the integral
///
/// # Examples
///
/// ```ignore
/// use scirs2_integrate::gaussian::multi_gauss_legendre;
/// use ndarray::Array1;
///
/// // Integrate f(x,y) = x²+y² over [0,1]×[0,1] (exact result: 2/3)
/// let result = multi_gauss_legendre(
///     |x| x.iter().map(|&xi| xi*xi).sum::<f64>(),
///     &[(0.0, 1.0), (0.0, 1.0)],
///     5
/// ).unwrap();
/// assert!((result - 2.0/3.0).abs() < 1e-10);
/// ```
pub fn multi_gauss_legendre<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    n_points: usize,
) -> IntegrateResult<F>
where
    F: Float + FromPrimitive + Debug,
    Func: Fn(ArrayView1<F>) -> F,
{
    if ranges.is_empty() {
        return Err(IntegrateError::ValueError(
            "Integration ranges cannot be empty".to_string(),
        ));
    }

    let quadrature = GaussLegendreQuadrature::new(n_points)?;
    let n_dims = ranges.len();

    // Inner function to perform recursive multidimensional integration
    fn integrate_recursive<F, Func>(
        f: &Func,
        ranges: &[(F, F)],
        quadrature: &GaussLegendreQuadrature<F>,
        dim: usize,
        point: &mut Array1<F>,
        n_dims: usize,
    ) -> F
    where
        F: Float + FromPrimitive + Debug,
        Func: Fn(ArrayView1<F>) -> F,
    {
        if dim == n_dims {
            // Base case: Evaluate the function at the current point
            return f(point.view());
        }

        // Change of variables from [-1, 1] to [a, b]
        let (a, b) = ranges[dim];
        let mid = (a + b) / F::from_f64(2.0).unwrap();
        let half_length = (b - a) / F::from_f64(2.0).unwrap();

        // Apply quadrature rule for the current dimension
        let mut result = F::zero();
        for (i, &x) in quadrature.nodes.iter().enumerate() {
            let transformed_x = mid + half_length * x;
            point[dim] = transformed_x;

            // Recursively integrate remaining dimensions
            let inner_result = integrate_recursive(f, ranges, quadrature, dim + 1, point, n_dims);
            result = result + quadrature.weights[i] * inner_result;
        }

        // Scale by half-length due to the change of variables
        result * half_length
    }

    // Initialize a point array to store coordinates during integration
    let mut point = Array1::zeros(n_dims);

    // Start the recursive integration from the first dimension
    Ok(integrate_recursive(
        &f,
        ranges,
        &quadrature,
        0,
        &mut point,
        n_dims,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use std::f64::consts::PI;

    #[test]
    fn test_gauss_legendre_quadrature() {
        // Test integrating x² from 0 to 1 (exact result: 1/3)
        let quad5 = GaussLegendreQuadrature::<f64>::new(5).unwrap();
        let result = quad5.integrate(|x| x * x, 0.0, 1.0);

        // With the corrected nodes and weights, the result should be accurate
        assert_relative_eq!(result, 1.0 / 3.0, epsilon = 1e-10);

        // Test integrating sin(x) from 0 to π (exact result: 2)
        let quad10 = GaussLegendreQuadrature::<f64>::new(10).unwrap();
        let result = quad10.integrate(|x| x.sin(), 0.0, PI);
        assert_relative_eq!(result, 2.0, epsilon = 1e-10);

        // Test integrating exp(-x²) from -1 to 1
        // This is related to the error function, with exact result: sqrt(π)·erf(1)
        let quad10 = GaussLegendreQuadrature::<f64>::new(10).unwrap();
        let result = quad10.integrate(|x| (-x * x).exp(), -1.0, 1.0);
        let exact = PI.sqrt() * libm::erf(1.0);
        assert_relative_eq!(result, exact, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_legendre_helper() {
        // Test the high-level helper function
        let result = gauss_legendre(|x| x * x, 0.0, 1.0, 5).unwrap();

        // With the corrected nodes and weights, the result should be accurate
        assert_relative_eq!(result, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_dimensional_integration() {
        // Test 2D integration: f(x,y) = x² + y² over [0,1]×[0,1]
        // Exact result: 2/3 (1/3 for x² + 1/3 for y²)
        let result =
            multi_gauss_legendre(|x| x[0] * x[0] + x[1] * x[1], &[(0.0, 1.0), (0.0, 1.0)], 5)
                .unwrap();

        // With the corrected nodes and weights, the result should be accurate
        assert_relative_eq!(result, 2.0 / 3.0, epsilon = 1e-10);

        // Test 3D integration: f(x,y,z) = x²y²z² over [0,1]³
        // Exact result: (1/3)³ = 1/27
        let result = multi_gauss_legendre(
            |x| x[0] * x[0] * x[1] * x[1] * x[2] * x[2],
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            5,
        )
        .unwrap();

        // With the corrected nodes and weights, the result should be accurate
        assert_relative_eq!(result, 1.0 / 27.0, epsilon = 1e-10);
    }
}
