//! Filtering functions for n-dimensional arrays
//!
//! This module provides functions for filtering n-dimensional arrays,
//! including Gaussian filters, median filters, uniform filters, and various convolution operations.

use std::fmt::Debug;

mod convolve;
mod edge;
mod extrema;
mod gaussian;
mod median;
mod rank;
mod tests;
mod uniform;
mod utils;

// Convolve module exports
pub use convolve::{
    convolve,
    // Re-export the uniform_filter from convolve with a different name
    uniform_filter as convolve_uniform_filter,
};

// Edge module exports
pub use edge::{gradient_magnitude, laplace, prewitt, roberts, scharr, sobel};

// Extrema module exports (new implementation)
pub use extrema::{maximum_filter, minimum_filter};

// Gaussian module exports
pub use gaussian::{gaussian_filter, gaussian_filter_f32, gaussian_filter_f64};

// Median module exports
pub use median::*;

// Rank module exports
// Add minimum_filter and maximum_filter from rank module with different names
// to avoid conflicts with extrema module
pub use rank::{
    maximum_filter as rank_maximum_filter, minimum_filter as rank_minimum_filter,
    percentile_filter, rank_filter,
};

// Uniform module exports
pub use uniform::{uniform_filter, uniform_filter_separable};

// Utils module exports
pub use utils::*;

/// Border handling modes for filters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    /// Pad with zeros
    Constant,

    /// Reflect values across the edges
    Reflect,

    /// Mirror and reflect values across the edges
    Mirror,

    /// Wrap around to the opposite edge (periodic)
    Wrap,

    /// Repeat edge values
    Nearest,
}
