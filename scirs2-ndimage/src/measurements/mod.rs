//! Measurement functions for n-dimensional arrays
//!
//! This module provides functions for measuring properties of labeled arrays.

use scirs2_core::numeric::Float;
use std::fmt::Debug;

mod extrema;
mod moments;
mod region;
mod statistics;

pub use extrema::*;
pub use moments::*;
pub use region::*;
pub use statistics::*;

/// Properties structure for labeled regions
#[derive(Debug, Clone)]
pub struct RegionProperties<T: Float> {
    /// Label value
    pub label: usize,
    /// Area (number of pixels) of the region
    pub area: usize,
    /// Center of mass (centroid) of the region
    pub centroid: Vec<T>,
    /// Bounding box of the region (min_row, min_col, max_row, max_col)
    pub bbox: Vec<usize>,
}
