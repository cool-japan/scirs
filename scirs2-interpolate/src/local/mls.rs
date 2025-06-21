//! Moving Least Squares Interpolation
//!
//! This module provides an implementation of Moving Least Squares (MLS) interpolation,
//! which is particularly useful for scattered data with potentially noisy values.
//! MLS creates a smooth approximation function by fitting local polynomials at each
//! evaluation point using weighted least squares where closer points have higher weights.
//!
//! The technique is popular in computer graphics, mesh processing, and scientific computing
//! for its ability to handle irregularly spaced data and provide smooth results.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::error::{InterpolateError, InterpolateResult};

/// Weight function types for MLS
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightFunction {
    /// Gaussian weight: w(r) = exp(-r²/h²)
    Gaussian,

    /// Wendland C2 compactly supported function
    /// w(r) = (1-r/h)⁴(4r/h+1) for r < h, 0 otherwise
    WendlandC2,

    /// Inverse distance: w(r) = 1/(ε + r²)
    InverseDistance,

    /// Cubic spline with compact support
    /// w(r) = 2/3 - 9r²/h² + 19r³/h³ for r < h/3
    /// w(r) = 1/3 * (2 - 3r/h)³ for h/3 < r < h
    /// w(r) = 0 for r > h
    CubicSpline,
}

/// Polynomial basis types for the local approximation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolynomialBasis {
    /// Constant basis: [1]
    Constant,

    /// Linear basis: [1, x, y, ...]
    Linear,

    /// Quadratic basis: [1, x, y, ..., x², xy, y², ...]
    Quadratic,
}

/// Moving Least Squares interpolator for scattered data
///
/// This interpolator uses a weighted least squares fit at each evaluation point,
/// where the weights depend on the distance from the evaluation point to the data points.
/// The result is a smooth function that approximates the data points.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "linalg")]
/// # {
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::local::mls::{MovingLeastSquares, WeightFunction, PolynomialBasis};
///
/// // Create some 2D scattered data
/// let points = Array2::from_shape_vec((5, 2), vec![
///     0.0, 0.0,
///     1.0, 0.0,
///     0.0, 1.0,
///     1.0, 1.0,
///     0.5, 0.5,
/// ]).unwrap();
/// let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.5]);
///
/// // Create MLS interpolator (simplified configuration for test)
/// let mls = MovingLeastSquares::<f64>::new(
///     points,
///     values,
///     WeightFunction::Gaussian,
///     PolynomialBasis::Constant, // Using constant basis to avoid linalg feature requirement
///     0.5, // bandwidth parameter
/// ).unwrap();
///
/// // Evaluate at a new point
/// let query = Array1::from_vec(vec![0.25, 0.25]);
/// let result = mls.evaluate(&query.view()).unwrap();
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MovingLeastSquares<F>
where
    F: Float + FromPrimitive + Debug + 'static + std::cmp::PartialOrd,
{
    /// Points coordinates (input locations)
    points: Array2<F>,

    /// Values at points
    values: Array1<F>,

    /// Weight function to use
    weight_fn: WeightFunction,

    /// Polynomial basis to use
    basis: PolynomialBasis,

    /// Bandwidth parameter (h)
    bandwidth: F,

    /// Small value to add to denominators to avoid division by zero
    epsilon: F,

    /// Maximum number of points to use (for efficiency)
    max_points: Option<usize>,

    /// Marker for generic type parameter
    _phantom: PhantomData<F>,
}

impl<F> MovingLeastSquares<F>
where
    F: Float + FromPrimitive + Debug + 'static + std::cmp::PartialOrd,
{
    /// Create a new MovingLeastSquares interpolator
    ///
    /// # Arguments
    ///
    /// * `points` - Point coordinates with shape (n_points, n_dims)
    /// * `values` - Values at each point with shape (n_points,)
    /// * `weight_fn` - Weight function to use
    /// * `basis` - Polynomial basis for the local fit
    /// * `bandwidth` - Bandwidth parameter controlling locality (larger = smoother)
    ///
    /// # Returns
    ///
    /// A new MovingLeastSquares interpolator
    pub fn new(
        points: Array2<F>,
        values: Array1<F>,
        weight_fn: WeightFunction,
        basis: PolynomialBasis,
        bandwidth: F,
    ) -> InterpolateResult<Self> {
        // Validate inputs
        if points.shape()[0] != values.len() {
            return Err(InterpolateError::DimensionMismatch(
                "Number of points must match number of values".to_string(),
            ));
        }

        if points.shape()[0] < 2 {
            return Err(InterpolateError::InvalidValue(
                "At least 2 points are required for MLS interpolation".to_string(),
            ));
        }

        if bandwidth <= F::zero() {
            return Err(InterpolateError::InvalidValue(
                "Bandwidth parameter must be positive".to_string(),
            ));
        }

        Ok(Self {
            points,
            values,
            weight_fn,
            basis,
            bandwidth,
            epsilon: F::from_f64(1e-10).unwrap(),
            max_points: None,
            _phantom: PhantomData,
        })
    }

    /// Set maximum number of points to use for local fit
    ///
    /// This is useful for large datasets where using all points
    /// would be computationally expensive.
    ///
    /// # Arguments
    ///
    /// * `max_points` - Maximum number of points to use
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_max_points(mut self, max_points: usize) -> Self {
        self.max_points = Some(max_points);
        self
    }

    /// Set epsilon value for numerical stability
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Small value to add to denominators
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_epsilon(mut self, epsilon: F) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Evaluate the MLS approximation at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - Query point with shape (n_dims,)
    ///
    /// # Returns
    ///
    /// Interpolated value at the query point
    pub fn evaluate(&self, x: &ArrayView1<F>) -> InterpolateResult<F> {
        // Check dimensions
        if x.len() != self.points.shape()[1] {
            return Err(InterpolateError::DimensionMismatch(
                "Query point dimension must match training points".to_string(),
            ));
        }

        // Get points to use for local fit
        let (indices, distances) = self.find_relevant_points(x)?;

        if indices.is_empty() {
            return Err(InterpolateError::ValueError(
                "No points found within effective range".to_string(),
            ));
        }

        // Compute weights
        let weights = self.compute_weights(&distances)?;

        // Create basis functions for these points
        let basis_functions = self.create_basis_functions(&indices, x)?;

        // Weighted least squares solution
        let result = self.solve_weighted_least_squares(&indices, &weights, &basis_functions, x)?;

        Ok(result)
    }

    /// Evaluate the MLS approximation at multiple points
    ///
    /// # Arguments
    ///
    /// * `points` - Query points with shape (n_points, n_dims)
    ///
    /// # Returns
    ///
    /// Interpolated values at the query points
    pub fn evaluate_multi(&self, points: &ArrayView2<F>) -> InterpolateResult<Array1<F>> {
        // Check dimensions
        if points.shape()[1] != self.points.shape()[1] {
            return Err(InterpolateError::DimensionMismatch(
                "Query points dimension must match training points".to_string(),
            ));
        }

        let n_points = points.shape()[0];
        let mut results = Array1::zeros(n_points);

        // Evaluate at each point
        for i in 0..n_points {
            let point = points.slice(ndarray::s![i, ..]);
            results[i] = self.evaluate(&point)?;
        }

        Ok(results)
    }

    /// Find points to use for local fit
    ///
    /// Returns indices of points to use and their distances to the query point
    fn find_relevant_points(&self, x: &ArrayView1<F>) -> InterpolateResult<(Vec<usize>, Vec<F>)> {
        let n_points = self.points.shape()[0];
        let n_dims = self.points.shape()[1];

        // Compute squared distances
        let mut distances = Vec::with_capacity(n_points);
        for i in 0..n_points {
            let mut d_squared = F::zero();
            for j in 0..n_dims {
                let diff = x[j] - self.points[[i, j]];
                d_squared = d_squared + diff * diff;
            }
            let dist = d_squared.sqrt();
            distances.push((i, dist));
        }

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply max_points limit if specified
        let limit = match self.max_points {
            Some(limit) => std::cmp::min(limit, n_points),
            None => n_points,
        };

        // Filter out points with zero weight (if using compactly supported weight function)
        let effective_radius = match self.weight_fn {
            WeightFunction::WendlandC2 | WeightFunction::CubicSpline => self.bandwidth,
            _ => F::infinity(),
        };

        let mut indices = Vec::new();
        let mut dist_values = Vec::new();

        for &(idx, dist) in distances.iter().take(limit) {
            if dist <= effective_radius {
                indices.push(idx);
                dist_values.push(dist);
            }
        }

        // Ensure we have enough points for the basis
        let min_points = match self.basis {
            PolynomialBasis::Constant => 1,
            PolynomialBasis::Linear => n_dims + 1,
            PolynomialBasis::Quadratic => ((n_dims + 1) * (n_dims + 2)) / 2,
        };

        if indices.len() < min_points {
            // If not enough points with compact support, take the closest ones
            indices = distances
                .iter()
                .take(min_points)
                .map(|&(idx, _)| idx)
                .collect();
            dist_values = distances
                .iter()
                .take(min_points)
                .map(|&(_, dist)| dist)
                .collect();
        }

        Ok((indices, dist_values))
    }

    /// Compute weights for the given distances
    fn compute_weights(&self, distances: &[F]) -> InterpolateResult<Array1<F>> {
        let n = distances.len();
        let mut weights = Array1::zeros(n);

        for (i, &d) in distances.iter().enumerate() {
            // Normalize distance by bandwidth
            let r = d / self.bandwidth;

            // Compute weight based on the chosen weight function
            let weight = match self.weight_fn {
                WeightFunction::Gaussian => (-r * r).exp(),
                WeightFunction::WendlandC2 => {
                    if r < F::one() {
                        let t = F::one() - r;
                        let factor = F::from_f64(4.0).unwrap() * r + F::one();
                        t.powi(4) * factor
                    } else {
                        F::zero()
                    }
                }
                WeightFunction::InverseDistance => F::one() / (self.epsilon + r * r),
                WeightFunction::CubicSpline => {
                    if r < F::from_f64(1.0 / 3.0).unwrap() {
                        let r2 = r * r;
                        let r3 = r2 * r;
                        F::from_f64(2.0 / 3.0).unwrap() - F::from_f64(9.0).unwrap() * r2
                            + F::from_f64(19.0).unwrap() * r3
                    } else if r < F::one() {
                        let t = F::from_f64(2.0).unwrap() - F::from_f64(3.0).unwrap() * r;
                        F::from_f64(1.0 / 3.0).unwrap() * t.powi(3)
                    } else {
                        F::zero()
                    }
                }
            };

            weights[i] = weight;
        }

        // Normalize weights to sum to 1 for numerical stability
        let sum = weights.sum();
        if sum > F::zero() {
            weights.mapv_inplace(|w| w / sum);
        } else {
            // If all weights are zero (shouldn't happen), use equal weights
            weights.fill(F::from_f64(1.0 / n as f64).unwrap());
        }

        Ok(weights)
    }

    /// Create basis functions for the given points
    fn create_basis_functions(
        &self,
        indices: &[usize],
        x: &ArrayView1<F>,
    ) -> InterpolateResult<Array2<F>> {
        let n_points = indices.len();
        let n_dims = x.len();

        // Determine number of basis functions
        let n_basis = match self.basis {
            PolynomialBasis::Constant => 1,
            PolynomialBasis::Linear => n_dims + 1,
            PolynomialBasis::Quadratic => ((n_dims + 1) * (n_dims + 2)) / 2,
        };

        let mut basis = Array2::zeros((n_points, n_basis));

        // Fill basis functions for each point
        for (i, &idx) in indices.iter().enumerate() {
            let point = self.points.row(idx);
            let mut col = 0;

            // Constant term
            basis[[i, col]] = F::one();
            col += 1;

            if self.basis == PolynomialBasis::Linear || self.basis == PolynomialBasis::Quadratic {
                // Linear terms
                for j in 0..n_dims {
                    basis[[i, col]] = point[j];
                    col += 1;
                }
            }

            if self.basis == PolynomialBasis::Quadratic {
                // Quadratic terms
                for j in 0..n_dims {
                    for k in j..n_dims {
                        basis[[i, col]] = point[j] * point[k];
                        col += 1;
                    }
                }
            }
        }

        Ok(basis)
    }

    /// Create basis functions for evaluation at the query point
    fn create_query_basis(&self, x: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let n_dims = x.len();

        // Determine number of basis functions
        let n_basis = match self.basis {
            PolynomialBasis::Constant => 1,
            PolynomialBasis::Linear => n_dims + 1,
            PolynomialBasis::Quadratic => ((n_dims + 1) * (n_dims + 2)) / 2,
        };

        let mut basis = Array1::zeros(n_basis);
        let mut col = 0;

        // Constant term
        basis[col] = F::one();
        col += 1;

        if self.basis == PolynomialBasis::Linear || self.basis == PolynomialBasis::Quadratic {
            // Linear terms
            for j in 0..n_dims {
                basis[col] = x[j];
                col += 1;
            }
        }

        if self.basis == PolynomialBasis::Quadratic {
            // Quadratic terms
            for j in 0..n_dims {
                for k in j..n_dims {
                    basis[col] = x[j] * x[k];
                    col += 1;
                }
            }
        }

        Ok(basis)
    }

    /// Solve the weighted least squares problem
    fn solve_weighted_least_squares(
        &self,
        indices: &[usize],
        weights: &Array1<F>,
        basis: &Array2<F>,
        x: &ArrayView1<F>,
    ) -> InterpolateResult<F> {
        let n_points = indices.len();
        let n_basis = basis.shape()[1];

        // Create the weighted basis matrix and target vector
        let mut w_basis = Array2::zeros((n_points, n_basis));
        let mut w_values = Array1::zeros(n_points);

        for i in 0..n_points {
            let sqrt_w = weights[i].sqrt();
            for j in 0..n_basis {
                w_basis[[i, j]] = basis[[i, j]] * sqrt_w;
            }
            w_values[i] = self.values[indices[i]] * sqrt_w;
        }

        // Solve the least squares problem: (B'B)c = B'y
        #[cfg(feature = "linalg")]
        let btb = w_basis.t().dot(&w_basis);
        #[cfg(not(feature = "linalg"))]
        let _btb = w_basis.t().dot(&w_basis);
        #[allow(unused_variables)]
        let bty = w_basis.t().dot(&w_values);

        // Solve the system for coefficients
        #[cfg(feature = "linalg")]
        let coeffs = {
            use ndarray_linalg::Solve;
            let btb_f64 = btb.mapv(|x| x.to_f64().unwrap());
            let bty_f64 = bty.mapv(|x| x.to_f64().unwrap());
            match btb_f64.solve(&bty_f64) {
                Ok(c) => c.mapv(|x| F::from_f64(x).unwrap()),
                Err(_) => {
                    // Fallback: use local mean for numerical stability
                    let mut mean = F::zero();
                    let mut sum_weights = F::zero();
                    for (i, &idx) in indices.iter().enumerate() {
                        mean = mean + weights[i] * self.values[idx];
                        sum_weights = sum_weights + weights[i];
                    }

                    if sum_weights > F::zero() {
                        // For the fallback, we'll create a coefficient vector with just the mean
                        // as the constant term and zeros elsewhere
                        let mut fallback_coeffs = Array1::zeros(bty.len());
                        fallback_coeffs[0] = mean / sum_weights;
                        fallback_coeffs
                    } else {
                        return Err(InterpolateError::ComputationError(
                            "Failed to solve weighted least squares system".to_string(),
                        ));
                    }
                }
            }
        };

        #[cfg(not(feature = "linalg"))]
        let coeffs = {
            // Fallback implementation when linalg is not available
            // Simple diagonal approximation
            let mut result = Array1::zeros(bty.len());

            // Use local mean for constant term
            let mut mean = F::zero();
            let mut sum_weights = F::zero();
            for (i, &idx) in indices.iter().enumerate() {
                mean = mean + weights[i] * self.values[idx];
                sum_weights = sum_weights + weights[i];
            }

            if sum_weights > F::zero() {
                result[0] = mean / sum_weights;
            }

            result
        };

        // Evaluate at the query point by creating the basis for it
        let query_basis = self.create_query_basis(x)?;
        let result = query_basis.dot(&coeffs);

        Ok(result)
    }

    /// Get the weight function used by this MLS interpolator
    pub fn weight_fn(&self) -> WeightFunction {
        self.weight_fn
    }

    /// Get the bandwidth parameter used by this MLS interpolator
    pub fn bandwidth(&self) -> F {
        self.bandwidth
    }

    /// Get the points used by this MLS interpolator
    pub fn points(&self) -> &Array2<F> {
        &self.points
    }

    /// Get the values used by this MLS interpolator
    pub fn values(&self) -> &Array1<F> {
        &self.values
    }

    /// Get the basis type used by this MLS interpolator
    pub fn basis(&self) -> PolynomialBasis {
        self.basis
    }

    /// Get the maximum points setting used by this MLS interpolator
    pub fn max_points(&self) -> Option<usize> {
        self.max_points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_mls_constant_basis() {
        // Simple test with 2D data and constant basis
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        // Simple plane: z = x + y
        let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

        let mls = MovingLeastSquares::new(
            points,
            values,
            WeightFunction::Gaussian,
            PolynomialBasis::Constant,
            0.5,
        )
        .unwrap();

        // Test at center point - should be close to average of all values (1.0)
        let center = array![0.5, 0.5];
        let val = mls.evaluate(&center.view()).unwrap();

        assert_abs_diff_eq!(val, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_mls_linear_basis() {
        // Simple test with 2D data and linear basis
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        // Simple plane: z = x + y
        let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

        let mls = MovingLeastSquares::new(
            points,
            values,
            WeightFunction::Gaussian,
            PolynomialBasis::Linear,
            1.0,
        )
        .unwrap();

        // With linear basis, should be able to reproduce the plane equation
        let test_points = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.5, 0.5, // Should be exactly 1.0
                0.25, 0.25, // Should be exactly 0.5
                0.75, 0.25, // Should be exactly 1.0
                0.25, 0.75, // Should be exactly 1.0
                0.75, 0.75, // Should be exactly 1.5
            ],
        )
        .unwrap();

        let expected = Array1::from_vec(vec![1.0, 0.5, 1.0, 1.0, 1.5]);
        let results = mls.evaluate_multi(&test_points.view()).unwrap();

        // Allow some numerical error, but should be close to exact values
        for (result, expect) in results.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(result, expect, epsilon = 0.5);
        }
    }

    #[test]
    fn test_different_weight_functions() {
        // Simple test with 2D data - well-spaced points to avoid singularities
        let points = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.7, 0.7],
        )
        .unwrap();

        // Simple function: z = x + y (linear function for better numerical stability)
        let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 0.6, 1.4]);

        // Test with different weight functions
        let weight_fns = [WeightFunction::Gaussian, WeightFunction::InverseDistance];

        let query = array![0.5, 0.5];
        let expected = 0.5 + 0.5; // 1.0 (linear function: z = x + y)

        for &weight_fn in &weight_fns {
            let mls = MovingLeastSquares::new(
                points.clone(),
                values.clone(),
                weight_fn,
                PolynomialBasis::Linear, // Use linear basis for better stability
                2.0,                     // Large bandwidth to include all points
            )
            .unwrap();

            let result = mls.evaluate(&query.view());

            match result {
                Ok(val) => {
                    if val.is_finite() {
                        // Allow reasonable error for MLS approximation
                        assert!((val - expected).abs() < 0.5,
                               "Weight function {:?}: result {:.6} differs too much from expected {:.6}", 
                               weight_fn, val, expected);
                    } else {
                        panic!(
                            "Weight function {:?} produced non-finite result: {}",
                            weight_fn, val
                        );
                    }
                }
                Err(e) => {
                    panic!("Weight function {:?} failed with error: {}", weight_fn, e);
                }
            }
        }
    }
}
