//! Voronoi-based interpolation methods
//!
//! This module provides interpolation methods based on Voronoi diagrams, which
//! partition space into regions where each region contains all points closest
//! to a particular input site.
//!
//! Voronoi-based interpolation is particularly useful for:
//! - Scattered data with irregular spacing
//! - Data where sharp transitions need to be preserved
//! - Applications requiring natural neighbor interpolation
//! - Problems with piecewise-constant or locally varying behavior
//!
//! The module includes:
//! - Natural Neighbor interpolation
//! - Sibson interpolation (a specific type of natural neighbor)
//! - Laplace interpolation (non-Sibsonian natural neighbor)
//! - Voronoi-based gradient estimation
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::ndarray::{Array1, Array2};
//! use scirs2_interpolate::voronoi::{
//!     NaturalNeighborInterpolator, InterpolationMethod
//! };
//!
//! // Create some scattered 2D data
//! let points = Array2::from_shape_vec((5, 2), vec![
//!     0.0, 0.0,
//!     1.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 1.0,
//!     0.5, 0.5,
//! ]).unwrap();
//! let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.5]);
//!
//! // Create natural neighbor interpolator
//! let interpolator = NaturalNeighborInterpolator::new(
//!     points,
//!     values,
//!     InterpolationMethod::Sibson,
//! ).unwrap();
//!
//! // Interpolate at a query point
//! let query = Array1::from_vec(vec![0.25, 0.25]);
//! let result = interpolator.interpolate(&query.view()).unwrap();
//! ```

// Simply re-export the modules since they're already defined below

// Re-export submodules
pub mod extrapolation;
pub mod gradient;
pub mod natural;
pub mod parallel;
pub mod voronoi_cell;

#[cfg(test)]
mod tests;

// Re-export main types
pub use extrapolation::{
    constant_value_extrapolation, inverse_distance_extrapolation, linear_gradient_extrapolation,
    nearest_neighbor_extrapolation, Extrapolation, ExtrapolationMethod, ExtrapolationParams,
};
pub use gradient::{GradientEstimation, InterpolateWithGradient, InterpolateWithGradientResult};
pub use natural::{
    make_laplace_interpolator, make_natural_neighbor_interpolator, make_sibson_interpolator,
    InterpolationMethod, NaturalNeighborInterpolator,
};
pub use parallel::{
    make_parallel_laplace_interpolator, make_parallel_natural_neighbor_interpolator,
    make_parallel_sibson_interpolator, ParallelNaturalNeighborInterpolator,
};
pub use voronoi_cell::{VoronoiCell, VoronoiDiagram};
