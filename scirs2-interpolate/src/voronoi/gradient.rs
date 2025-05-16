//! Gradient estimation for Voronoi-based interpolation methods
//!
//! This module provides implementations for computing gradients of functions
//! interpolated using natural neighbor interpolation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::natural::{InterpolationMethod, NaturalNeighborInterpolator};
use crate::error::{InterpolateError, InterpolateResult};

/// Trait for interpolators that can calculate values at query points
pub trait Interpolator<F: Float + FromPrimitive + Debug> {
    /// Interpolate at a single query point
    fn interpolate(&self, query: &ArrayView1<F>) -> InterpolateResult<F>;
}

impl<
        F: Float + FromPrimitive + Debug + ndarray::ScalarOperand + 'static + std::cmp::PartialOrd,
    > Interpolator<F> for NaturalNeighborInterpolator<F>
{
    fn interpolate(&self, query: &ArrayView1<F>) -> InterpolateResult<F> {
        // Simply forward to the NaturalNeighborInterpolator's interpolate method
        NaturalNeighborInterpolator::interpolate(self, query)
    }
}

/// Trait for interpolators that can compute gradients
pub trait GradientEstimation<F: Float + FromPrimitive + Debug> {
    /// Computes the gradient of the interpolated function at a query point
    ///
    /// # Arguments
    /// * `query` - The point at which to compute the gradient
    ///
    /// # Returns
    /// A vector of partial derivatives with respect to each coordinate
    fn gradient(&self, query: &ArrayView1<F>) -> InterpolateResult<Array1<F>>;

    /// Computes the gradients of the interpolated function at multiple query points
    ///
    /// # Arguments
    /// * `queries` - The points at which to compute gradients
    ///
    /// # Returns
    /// A matrix where each row is the gradient at the corresponding query point
    fn gradient_multi(&self, queries: &ArrayView2<F>) -> InterpolateResult<Array2<F>>;
}

/// Extends NaturalNeighborInterpolator with gradient estimation
impl<
        F: Float
            + FromPrimitive
            + Debug
            + ndarray::ScalarOperand
            + 'static
            + for<'a> std::iter::Sum<&'a F>
            + std::cmp::PartialOrd,
    > GradientEstimation<F> for NaturalNeighborInterpolator<F>
{
    fn gradient(&self, query: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        let dim = query.len();

        if dim != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query point dimension ({}) does not match data dimension ({})",
                dim,
                self.points.ncols()
            )));
        }

        // For natural neighbor interpolation, we can compute the gradient directly
        // from the interpolation weights and the values at data points

        // Get the natural neighbor weights
        let neighbor_weights = self.voronoi_diagram().natural_neighbors(query)?;

        if neighbor_weights.is_empty() {
            // If no natural neighbors found, use finite difference approximation
            return finite_difference_gradient(self, query);
        }

        // Compute the gradient based on the interpolation method
        match self.method() {
            InterpolationMethod::Sibson => {
                // For Sibson's method, the gradient is computed as a weighted sum of
                // value differences and point differences
                let mut gradient = Array1::zeros(dim);

                for (idx, weight) in neighbor_weights.iter() {
                    let neighbor_point = self.points.row(*idx);
                    let neighbor_value = self.values[*idx];

                    // Compute contribution to gradient from this neighbor
                    for d in 0..dim {
                        let coordinate_diff = neighbor_point[d] - query[d];
                        gradient[d] = gradient[d] + *weight * neighbor_value * coordinate_diff;
                    }
                }

                // Normalize the gradient if necessary
                let weight_sum: F = neighbor_weights.values().sum();
                if weight_sum > F::zero() {
                    gradient = gradient / weight_sum;
                }

                Ok(gradient)
            }
            InterpolationMethod::Laplace => {
                // For Laplace's method, the gradient is approximated by a finite difference
                // approach using the natural neighbors
                let mut gradient = Array1::zeros(dim);

                let center_value = self.interpolate(query)?;

                // Compute a weighted average of finite differences
                let mut total_weight = F::zero();

                for (idx, weight) in neighbor_weights.iter() {
                    let neighbor_point = self.points.row(*idx);
                    let neighbor_value = self.values[*idx];

                    // Compute distance from query to neighbor
                    let mut distance = F::zero();
                    for d in 0..dim {
                        distance = distance + (neighbor_point[d] - query[d]).powi(2);
                    }
                    distance = distance.sqrt();

                    // Skip very close points to avoid numerical issues
                    if distance < F::epsilon() {
                        continue;
                    }

                    // Compute value difference
                    let value_diff = neighbor_value - center_value;

                    // Contribute to gradient
                    for d in 0..dim {
                        let coordinate_diff = neighbor_point[d] - query[d];
                        // Directional derivative along the vector from query to neighbor
                        let dir_deriv = value_diff / distance;
                        // Project onto coordinate axis
                        gradient[d] =
                            gradient[d] + *weight * dir_deriv * coordinate_diff / distance;
                    }

                    total_weight = total_weight + *weight;
                }

                // Normalize the gradient
                if total_weight > F::zero() {
                    gradient = gradient / total_weight;
                }

                Ok(gradient)
            }
        }
    }

    fn gradient_multi(&self, queries: &ArrayView2<F>) -> InterpolateResult<Array2<F>> {
        let n_queries = queries.nrows();
        let dim = queries.ncols();

        if dim != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query points dimension ({}) does not match data dimension ({})",
                dim,
                self.points.ncols()
            )));
        }

        let mut gradients = Array2::zeros((n_queries, dim));

        for i in 0..n_queries {
            let query = queries.row(i);
            let gradient = self.gradient(&query)?;

            gradients.row_mut(i).assign(&gradient);
        }

        Ok(gradients)
    }
}

/// Computes a gradient using finite difference approximation
///
/// This is a fallback method when natural neighbor weights are not available.
///
/// # Arguments
/// * `interpolator` - The interpolator to use for function evaluations
/// * `query` - The point at which to compute the gradient
///
/// # Returns
/// The estimated gradient vector
fn finite_difference_gradient<F, T>(
    interpolator: &T,
    query: &ArrayView1<F>,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
    T: GradientEstimation<F> + Interpolator<F>,
{
    let dim = query.len();
    let mut gradient = Array1::zeros(dim);

    // Use central differences for better accuracy
    let h = F::from(1e-6).unwrap(); // Step size

    // Compute the center value
    let center_value = match interpolator.interpolate(query) {
        Ok(v) => v,
        Err(_) => {
            // If interpolation at the center fails, use a one-sided difference
            // by evaluating at nearby points only
            for d in 0..dim {
                let mut forward_query = query.to_owned();
                forward_query[d] = forward_query[d] + h;

                if let Ok(forward_value) = interpolator.interpolate(&forward_query.view()) {
                    gradient[d] = forward_value / h; // Approximate slope
                }
            }
            return Ok(gradient);
        }
    };

    // Use central differences for each dimension
    for d in 0..dim {
        let mut forward_query = query.to_owned();
        forward_query[d] = forward_query[d] + h;

        let mut backward_query = query.to_owned();
        backward_query[d] = backward_query[d] - h;

        // Try to compute the forward and backward values
        let forward_result = interpolator.interpolate(&forward_query.view());
        let backward_result = interpolator.interpolate(&backward_query.view());

        match (forward_result, backward_result) {
            (Ok(forward_value), Ok(backward_value)) => {
                // Central difference
                gradient[d] = (forward_value - backward_value) / (h + h);
            }
            (Ok(forward_value), Err(_)) => {
                // Forward difference
                gradient[d] = (forward_value - center_value) / h;
            }
            (Err(_), Ok(backward_value)) => {
                // Backward difference
                gradient[d] = (center_value - backward_value) / h;
            }
            (Err(_), Err(_)) => {
                // Can't compute gradient in this direction
                gradient[d] = F::zero();
            }
        }
    }

    Ok(gradient)
}

/// Information returned by interpolation with gradient
pub struct InterpolateWithGradientResult<F: Float + FromPrimitive + Debug> {
    /// The interpolated value
    pub value: F,

    /// The gradient vector
    pub gradient: Array1<F>,
}

/// Extension trait for interpolators to compute interpolated values with gradients
pub trait InterpolateWithGradient<F: Float + FromPrimitive + Debug> {
    /// Interpolates a value and computes its gradient at a query point
    ///
    /// # Arguments
    /// * `query` - The point at which to interpolate and compute the gradient
    ///
    /// # Returns
    /// A struct containing the interpolated value and gradient
    fn interpolate_with_gradient(
        &self,
        query: &ArrayView1<F>,
    ) -> InterpolateResult<InterpolateWithGradientResult<F>>;

    /// Interpolates values and computes gradients at multiple query points
    ///
    /// # Arguments
    /// * `queries` - The points at which to interpolate and compute gradients
    ///
    /// # Returns
    /// A vector of structs containing the interpolated values and gradients
    fn interpolate_with_gradient_multi(
        &self,
        queries: &ArrayView2<F>,
    ) -> InterpolateResult<Vec<InterpolateWithGradientResult<F>>>;
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + ndarray::ScalarOperand
            + 'static
            + for<'a> std::iter::Sum<&'a F>
            + std::cmp::PartialOrd,
    > InterpolateWithGradient<F> for NaturalNeighborInterpolator<F>
{
    fn interpolate_with_gradient(
        &self,
        query: &ArrayView1<F>,
    ) -> InterpolateResult<InterpolateWithGradientResult<F>> {
        let value = self.interpolate(query)?;
        let gradient = self.gradient(query)?;

        Ok(InterpolateWithGradientResult { value, gradient })
    }

    fn interpolate_with_gradient_multi(
        &self,
        queries: &ArrayView2<F>,
    ) -> InterpolateResult<Vec<InterpolateWithGradientResult<F>>> {
        let n_queries = queries.nrows();
        let mut results = Vec::with_capacity(n_queries);

        let values = self.interpolate_multi(queries)?;
        let gradients = self.gradient_multi(queries)?;

        for i in 0..n_queries {
            results.push(InterpolateWithGradientResult {
                value: values[i],
                gradient: gradients.row(i).to_owned(),
            });
        }

        Ok(results)
    }
}
