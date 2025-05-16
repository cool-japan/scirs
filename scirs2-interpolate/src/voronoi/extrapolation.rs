//! Extrapolation methods for Voronoi-based interpolation
//!
//! This module provides methods for extrapolating values outside the convex hull
//! of the input data points when using Voronoi-based interpolation methods.

use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::natural::NaturalNeighborInterpolator;
use crate::error::{InterpolateError, InterpolateResult};

/// Defines the method used for extrapolation outside the data domain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrapolationMethod {
    /// Uses the value of the nearest data point
    NearestNeighbor,

    /// Uses inverse distance weighting with a specified number of nearest neighbors
    InverseDistanceWeighting,

    /// Uses a linear extrapolation based on the nearest data points and their gradients
    LinearGradient,

    /// Uses a constant value for all points outside the domain
    ConstantValue,
}

/// Parameters for extrapolation
#[derive(Debug, Clone)]
pub struct ExtrapolationParams<F: Float + FromPrimitive + Debug> {
    /// The method to use for extrapolation
    pub method: ExtrapolationMethod,

    /// The number of nearest neighbors to use (for methods that support it)
    pub n_neighbors: usize,

    /// The power parameter for inverse distance weighting
    pub idw_power: F,

    /// The constant value to use for ConstantValue extrapolation
    pub constant_value: F,
}

impl<F: Float + FromPrimitive + Debug> Default for ExtrapolationParams<F> {
    fn default() -> Self {
        ExtrapolationParams {
            method: ExtrapolationMethod::NearestNeighbor,
            n_neighbors: 3,
            idw_power: F::from(2.0).unwrap(), // Default to inverse squared distance
            constant_value: F::zero(),
        }
    }
}

/// Extension trait for handling extrapolation
pub trait Extrapolation<F: Float + FromPrimitive + Debug> {
    /// Extrapolate a value at a query point outside the domain
    ///
    /// # Arguments
    /// * `query` - The query point
    /// * `params` - Parameters for extrapolation
    ///
    /// # Returns
    /// The extrapolated value
    fn extrapolate(
        &self,
        query: &ArrayView1<F>,
        params: &ExtrapolationParams<F>,
    ) -> InterpolateResult<F>;

    /// Extrapolate values at multiple query points outside the domain
    ///
    /// # Arguments
    /// * `queries` - The query points
    /// * `params` - Parameters for extrapolation
    ///
    /// # Returns
    /// An array of extrapolated values
    fn extrapolate_multi(
        &self,
        queries: &ArrayView2<F>,
        params: &ExtrapolationParams<F>,
    ) -> InterpolateResult<Array1<F>>;

    /// Interpolate or extrapolate a value at a query point
    ///
    /// This method first tries to interpolate the value. If the query point is outside
    /// the domain and interpolation fails, it falls back to extrapolation.
    ///
    /// # Arguments
    /// * `query` - The query point
    /// * `params` - Parameters for extrapolation
    ///
    /// # Returns
    /// The interpolated or extrapolated value
    fn interpolate_or_extrapolate(
        &self,
        query: &ArrayView1<F>,
        params: &ExtrapolationParams<F>,
    ) -> InterpolateResult<F>;

    /// Interpolate or extrapolate values at multiple query points
    ///
    /// This method first tries to interpolate each value. If a query point is outside
    /// the domain and interpolation fails, it falls back to extrapolation.
    ///
    /// # Arguments
    /// * `queries` - The query points
    /// * `params` - Parameters for extrapolation
    ///
    /// # Returns
    /// An array of interpolated or extrapolated values
    fn interpolate_or_extrapolate_multi(
        &self,
        queries: &ArrayView2<F>,
        params: &ExtrapolationParams<F>,
    ) -> InterpolateResult<Array1<F>>;
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + ndarray::ScalarOperand
            + 'static
            + for<'a> std::iter::Sum<&'a F>
            + std::cmp::PartialOrd,
    > Extrapolation<F> for NaturalNeighborInterpolator<F>
{
    fn extrapolate(
        &self,
        query: &ArrayView1<F>,
        params: &ExtrapolationParams<F>,
    ) -> InterpolateResult<F> {
        let dim = query.len();

        if dim != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query point dimension ({}) does not match data dimension ({})",
                dim,
                self.points.ncols()
            )));
        }

        match params.method {
            ExtrapolationMethod::NearestNeighbor => {
                // Find the nearest data point
                let indices = self.kdtree.query_nearest(query, 1)?;

                if indices.is_empty() {
                    return Err(InterpolateError::InterpolationFailed(
                        "Nearest neighbor search failed".to_string(),
                    ));
                }

                // Return the value at the nearest point
                Ok(self.values[indices[0]])
            }

            ExtrapolationMethod::InverseDistanceWeighting => {
                // Use inverse distance weighting with the k nearest neighbors
                let k = params.n_neighbors.min(self.points.nrows());
                let indices = self.kdtree.query_nearest(query, k)?;

                if indices.is_empty() {
                    return Err(InterpolateError::InterpolationFailed(
                        "Nearest neighbor search failed".to_string(),
                    ));
                }

                let mut weighted_sum = F::zero();
                let mut weight_sum = F::zero();

                for &idx in &indices {
                    let point = self.points.row(idx);

                    // Compute distance
                    let mut dist_sq = F::zero();
                    for j in 0..dim {
                        dist_sq = dist_sq + (point[j] - query[j]).powi(2);
                    }

                    // Avoid division by zero
                    if dist_sq < F::epsilon() {
                        return Ok(self.values[idx]);
                    }

                    // Compute weight as inverse distance to the power p
                    let weight = F::one() / dist_sq.powf(params.idw_power / F::from(2.0).unwrap());

                    weighted_sum = weighted_sum + weight * self.values[idx];
                    weight_sum = weight_sum + weight;
                }

                if weight_sum > F::zero() {
                    Ok(weighted_sum / weight_sum)
                } else {
                    Err(InterpolateError::InterpolationFailed(
                        "All weights are zero in inverse distance weighting".to_string(),
                    ))
                }
            }

            ExtrapolationMethod::LinearGradient => {
                // Find the nearest data point
                let indices = self.kdtree.query_nearest(query, 1)?;

                if indices.is_empty() {
                    return Err(InterpolateError::InterpolationFailed(
                        "Nearest neighbor search failed".to_string(),
                    ));
                }

                let nearest_idx = indices[0];
                let nearest_point = self.points.row(nearest_idx);
                let nearest_value = self.values[nearest_idx];

                // Create a query for the nearest point to compute its gradient
                let nearest_query = nearest_point.to_owned();

                // Get the gradient at the nearest point
                let gradient = match super::gradient::GradientEstimation::gradient(
                    self,
                    &nearest_query.view(),
                ) {
                    Ok(grad) => grad,
                    Err(_) => {
                        // If gradient estimation fails, fall back to nearest neighbor
                        return Ok(nearest_value);
                    }
                };

                // Compute linear extrapolation: f(query) = f(nearest) + gradient · (query - nearest)
                let mut extrapolated_value = nearest_value;

                for j in 0..dim {
                    extrapolated_value =
                        extrapolated_value + gradient[j] * (query[j] - nearest_point[j]);
                }

                Ok(extrapolated_value)
            }

            ExtrapolationMethod::ConstantValue => {
                // Simply return the specified constant value
                Ok(params.constant_value)
            }
        }
    }

    fn extrapolate_multi(
        &self,
        queries: &ArrayView2<F>,
        params: &ExtrapolationParams<F>,
    ) -> InterpolateResult<Array1<F>> {
        let n_queries = queries.nrows();
        let dim = queries.ncols();

        if dim != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query points dimension ({}) does not match data dimension ({})",
                dim,
                self.points.ncols()
            )));
        }

        let mut results = Array1::zeros(n_queries);

        for i in 0..n_queries {
            let query = queries.row(i);
            results[i] = self.extrapolate(&query, params)?;
        }

        Ok(results)
    }

    fn interpolate_or_extrapolate(
        &self,
        query: &ArrayView1<F>,
        params: &ExtrapolationParams<F>,
    ) -> InterpolateResult<F> {
        // First try to interpolate
        match self.interpolate(query) {
            Ok(value) => Ok(value),
            Err(_) => {
                // If interpolation fails, try extrapolation
                self.extrapolate(query, params)
            }
        }
    }

    fn interpolate_or_extrapolate_multi(
        &self,
        queries: &ArrayView2<F>,
        params: &ExtrapolationParams<F>,
    ) -> InterpolateResult<Array1<F>> {
        let n_queries = queries.nrows();
        let dim = queries.ncols();

        if dim != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query points dimension ({}) does not match data dimension ({})",
                dim,
                self.points.ncols()
            )));
        }

        let mut results = Array1::zeros(n_queries);

        for i in 0..n_queries {
            let query = queries.row(i);
            results[i] = self.interpolate_or_extrapolate(&query, params)?;
        }

        Ok(results)
    }
}

/// Creates extrapolation parameters for nearest neighbor extrapolation
pub fn nearest_neighbor_extrapolation<F: Float + FromPrimitive + Debug>() -> ExtrapolationParams<F>
{
    ExtrapolationParams {
        method: ExtrapolationMethod::NearestNeighbor,
        ..Default::default()
    }
}

/// Creates extrapolation parameters for inverse distance weighting
///
/// # Arguments
/// * `n_neighbors` - The number of nearest neighbors to use
/// * `power` - The power parameter for inverse distance weighting
pub fn inverse_distance_extrapolation<F: Float + FromPrimitive + Debug>(
    n_neighbors: usize,
    power: F,
) -> ExtrapolationParams<F> {
    ExtrapolationParams {
        method: ExtrapolationMethod::InverseDistanceWeighting,
        n_neighbors,
        idw_power: power,
        ..Default::default()
    }
}

/// Creates extrapolation parameters for linear gradient extrapolation
pub fn linear_gradient_extrapolation<F: Float + FromPrimitive + Debug>() -> ExtrapolationParams<F> {
    ExtrapolationParams {
        method: ExtrapolationMethod::LinearGradient,
        ..Default::default()
    }
}

/// Creates extrapolation parameters for constant value extrapolation
///
/// # Arguments
/// * `value` - The constant value to use for extrapolation
pub fn constant_value_extrapolation<F: Float + FromPrimitive + Debug>(
    value: F,
) -> ExtrapolationParams<F> {
    ExtrapolationParams {
        method: ExtrapolationMethod::ConstantValue,
        constant_value: value,
        ..Default::default()
    }
}
