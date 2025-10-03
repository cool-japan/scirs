//! Natural Neighbor interpolation
//!
//! This module provides implementations of Natural Neighbor interpolation methods,
//! including Sibson and non-Sibsonian (Laplace) variants.
//!
//! Natural Neighbor interpolation is a spatial interpolation method that uses
//! the concept of natural neighbors in a Voronoi diagram to determine weights
//! for interpolation. It provides C¹ continuity except at the data points.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use super::voronoi_cell::VoronoiDiagram;
use crate::error::{InterpolateError, InterpolateResult};
use crate::spatial::kdtree::KdTree;

/// Defines the method used for Natural Neighbor interpolation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Sibson's original natural neighbor interpolation algorithm
    Sibson,

    /// Non-Sibsonian (Laplace) interpolation
    Laplace,
}

/// Natural Neighbor interpolator
///
/// Provides natural neighbor interpolation for scattered data points.
/// Supports both Sibson and non-Sibsonian (Laplace) interpolation methods.
#[derive(Debug, Clone)]
pub struct NaturalNeighborInterpolator<
    F: Float
        + FromPrimitive
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + 'static
        + std::cmp::PartialOrd
        + ordered_float::FloatCore,
> {
    /// The Voronoi diagram of the input points
    voronoi_diagram: VoronoiDiagram<F>,

    /// The input data points
    pub points: Array2<F>,

    /// The values at the data points
    pub values: Array1<F>,

    /// The method used for interpolation
    method: InterpolationMethod,

    /// KD-Tree for nearest neighbor search (used for extrapolation)
    pub kdtree: KdTree<F>,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + scirs2_core::ndarray::ScalarOperand
            + 'static
            + std::cmp::PartialOrd
            + ordered_float::FloatCore,
    > NaturalNeighborInterpolator<F>
{
    /// Creates a new Natural Neighbor interpolator
    ///
    /// # Arguments
    /// * `points` - The input data points (rows = points, columns = dimensions)
    /// * `values` - The values at the data points
    /// * `method` - The interpolation method to use
    ///
    /// # Returns
    /// A new Natural Neighbor interpolator
    pub fn new(
        points: Array2<F>,
        values: Array1<F>,
        method: InterpolationMethod,
    ) -> InterpolateResult<Self> {
        let n_points = points.nrows();
        let dim = points.ncols();

        if n_points != values.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Number of points ({}) does not match number of values ({})",
                n_points,
                values.len()
            )));
        }

        if n_points < 3 {
            return Err(InterpolateError::InsufficientData(
                "At least 3 data points are required for Natural Neighbor interpolation"
                    .to_string(),
            ));
        }

        if dim != 2 && dim != 3 {
            return Err(InterpolateError::UnsupportedOperation(format!(
                "Natural Neighbor interpolation for {dim}-dimensional data not yet implemented"
            )));
        }

        // Create the Voronoi diagram
        let voronoi_diagram = VoronoiDiagram::new(points.view(), values.view(), None)?;

        // Create KD-Tree for nearest neighbor search (used for extrapolation)
        let kdtree = KdTree::new(points.view())?;

        Ok(NaturalNeighborInterpolator {
            voronoi_diagram,
            points,
            values,
            method,
            kdtree,
        })
    }

    /// Interpolates the value at a query point
    ///
    /// # Arguments
    /// * `query` - The query point
    ///
    /// # Returns
    /// The interpolated value at the query point
    pub fn interpolate(&self, query: &ArrayView1<F>) -> InterpolateResult<F> {
        let dim = query.len();

        if dim != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query point dimension ({}) does not match data dimension ({})",
                dim,
                self.points.ncols()
            )));
        }

        // Check if the query point is one of the data points
        for i in 0..self.points.nrows() {
            let point = self.points.row(i);
            let mut is_same = true;

            for j in 0..dim {
                if scirs2_core::numeric::Float::abs(point[j] - query[j])
                    > <F as scirs2_core::numeric::Float>::epsilon()
                {
                    is_same = false;
                    break;
                }
            }

            if is_same {
                return Ok(self.values[i]);
            }
        }

        // Find the natural neighbors of the query point
        let neighbor_weights = match self.voronoi_diagram.natural_neighbors(query) {
            Ok(weights) => weights,
            Err(_) => {
                // Fallback to nearest neighbor for extrapolation
                let (idx, _) = self.kdtree.nearest_neighbor(&query.to_vec())?;
                let mut weights = HashMap::new();
                weights.insert(idx, F::one());
                weights
            }
        };

        if neighbor_weights.is_empty() {
            // If no natural neighbors found, use nearest neighbor
            let (idx, _) = self.kdtree.nearest_neighbor(&query.to_vec())?;
            return Ok(self.values[idx]);
        }

        // Apply the appropriate interpolation method
        match self.method {
            InterpolationMethod::Sibson => {
                // Sibson's natural neighbor interpolation
                // This is a weighted average of the values at the natural neighbors,
                // where the weights are the areas of the stolen Voronoi cells

                let mut interpolated_value = F::zero();
                let mut total_weight = F::zero();

                for (idx, weight) in neighbor_weights.iter() {
                    interpolated_value = interpolated_value + self.values[*idx] * *weight;
                    total_weight = total_weight + *weight;
                }

                if total_weight > F::zero() {
                    interpolated_value = interpolated_value / total_weight;
                } else {
                    return Err(InterpolateError::InterpolationFailed(
                        "Total weight is zero in Sibson interpolation".to_string(),
                    ));
                }

                Ok(interpolated_value)
            }
            InterpolationMethod::Laplace => {
                // Non-Sibsonian (Laplace) natural neighbor interpolation
                // This uses the ratio of distances to neighbors as weights

                let mut interpolated_value = F::zero();
                let mut total_weight = F::zero();

                for (idx_, _) in neighbor_weights.iter() {
                    // For Laplace method, we need to compute a different weight
                    // based on the distances and Voronoi cell edge lengths
                    let site = &self.voronoi_diagram.cells[*idx_].site;

                    // Compute distance to the neighbor
                    let mut distance = F::zero();
                    for j in 0..dim {
                        distance =
                            distance + scirs2_core::numeric::Float::powi(site[j] - query[j], 2);
                    }
                    distance = distance.sqrt();

                    if distance < <F as scirs2_core::numeric::Float>::epsilon() {
                        // If the query point is very close to this site,
                        // just return the value at this site
                        return Ok(self.values[*idx_]);
                    }

                    // In the Laplace method, weight is inversely proportional to distance
                    let weight = F::one() / distance;

                    interpolated_value = interpolated_value + self.values[*idx_] * weight;
                    total_weight = total_weight + weight;
                }

                if total_weight > F::zero() {
                    interpolated_value = interpolated_value / total_weight;
                } else {
                    return Err(InterpolateError::InterpolationFailed(
                        "Total weight is zero in Laplace interpolation".to_string(),
                    ));
                }

                Ok(interpolated_value)
            }
        }
    }

    /// Interpolates values at multiple query points
    ///
    /// # Arguments
    /// * `queries` - The query points (rows = points, columns = dimensions)
    ///
    /// # Returns
    /// An array of interpolated values
    pub fn interpolate_multi(&self, queries: &ArrayView2<F>) -> InterpolateResult<Array1<F>> {
        let n_queries = queries.nrows();
        let dim = queries.ncols();

        if dim != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query point dimension ({}) does not match data dimension ({})",
                dim,
                self.points.ncols()
            )));
        }

        let mut results = Array1::zeros(n_queries);

        for i in 0..n_queries {
            let query = queries.row(i);
            results[i] = self.interpolate(&query)?;
        }

        Ok(results)
    }

    /// Returns the underlying Voronoi diagram
    pub fn voronoi_diagram(&self) -> &VoronoiDiagram<F> {
        &self.voronoi_diagram
    }

    /// Sets the interpolation method
    pub fn set_method(&mut self, method: InterpolationMethod) {
        self.method = method;
    }

    /// Returns the interpolation method
    pub fn method(&self) -> InterpolationMethod {
        self.method
    }
}

/// Creates a new Natural Neighbor interpolator
///
/// # Arguments
/// * `points` - The input data points (rows = points, columns = dimensions)
/// * `values` - The values at the data points
/// * `method` - The interpolation method to use
///
/// # Returns
/// A new Natural Neighbor interpolator
#[allow(dead_code)]
pub fn make_natural_neighbor_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + 'static
        + std::cmp::PartialOrd
        + ordered_float::FloatCore,
>(
    points: Array2<F>,
    values: Array1<F>,
    method: InterpolationMethod,
) -> InterpolateResult<NaturalNeighborInterpolator<F>> {
    NaturalNeighborInterpolator::new(points, values, method)
}

/// Creates a new Sibson interpolator
///
/// # Arguments
/// * `points` - The input data points (rows = points, columns = dimensions)
/// * `values` - The values at the data points
///
/// # Returns
/// A new Natural Neighbor interpolator using Sibson's method
#[allow(dead_code)]
pub fn make_sibson_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + 'static
        + std::cmp::PartialOrd
        + ordered_float::FloatCore,
>(
    points: Array2<F>,
    values: Array1<F>,
) -> InterpolateResult<NaturalNeighborInterpolator<F>> {
    NaturalNeighborInterpolator::new(points, values, InterpolationMethod::Sibson)
}

/// Creates a new Laplace interpolator
///
/// # Arguments
/// * `points` - The input data points (rows = points, columns = dimensions)
/// * `values` - The values at the data points
///
/// # Returns
/// A new Natural Neighbor interpolator using the non-Sibsonian (Laplace) method
#[allow(dead_code)]
pub fn make_laplace_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + 'static
        + std::cmp::PartialOrd
        + ordered_float::FloatCore,
>(
    points: Array2<F>,
    values: Array1<F>,
) -> InterpolateResult<NaturalNeighborInterpolator<F>> {
    NaturalNeighborInterpolator::new(points, values, InterpolationMethod::Laplace)
}
