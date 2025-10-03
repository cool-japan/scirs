//! Morphological operations on n-dimensional arrays
//!
//! This module provides functions for performing morphological operations on n-dimensional
//! arrays, such as erosion, dilation, opening, closing, gradient, tophat and more.
//!
//! Morphological operations are nonlinear operations that process images based on shapes.
//! They can be used for many purposes: removing noise, extracting features, detecting edges, etc.
//!
//! # Usage Recommendations
//!
//! For best results, use the following guidelines:
//!
//! 1. For 1D and 2D arrays, use the specific implementations:
//!    - For 1D arrays, the generic `binary_*` functions will automatically use optimized 1D implementations
//!    - For 2D arrays, prefer the functions in the `simple_morph` module, like `binary_erosion_2d` and `grey_dilation_2d`
//!
//! 2. For n-dimensional arrays of arbitrary dimension:
//!    - Use the generic functions (`binary_erosion`, `grey_dilation`, etc.), but be aware of limitations
//!    - Convert your arrays to `IxDyn` dimension type before passing to these functions
//!    - Some operations for dimensions greater than 2 are limited in functionality
//!
//! # Example
//!
//! ```
//! use scirs2_core::ndarray::{Array2, array};
//! use scirs2_ndimage::morphology::simple_morph::binary_erosion_2d;
//!
//! // Create a binary image
//! let input = array![[false, true, true],
//!                    [false, true, true],
//!                    [false, false, false]];
//!
//! // Apply binary erosion
//! let result = binary_erosion_2d(&input, None, None, None, None).unwrap();
//! assert_eq!(result[[1, 1]], false); // Corner pixels are eroded
//! ```

use std::fmt::Debug;

mod connected;
mod distance_transform;
mod distance_transform_optimized;
pub mod morphology_optimized;
pub mod simple_morph;
mod structuring;
mod utils;

// These modules have issues, not exporting directly
mod binary;
mod grayscale;

// Public re-exports
pub use binary::*;
pub use connected::*;
pub use distance_transform::*;
pub use distance_transform_optimized::*;
pub use grayscale::*;
pub use morphology_optimized::*;
pub use structuring::*;
pub use utils::*;

/// Border handling modes for morphological operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphBorderMode {
    /// Pad with constant value (default is 0 for binary morphology)
    Constant,

    /// Reflect values at border
    Reflect,

    /// Mirror values at border (reflects and repeats edge values)
    Mirror,

    /// Wrap around to opposite edge (periodic)
    Wrap,

    /// Use nearest edge values
    Nearest,
}

/// Structuring element connectivity for binary morphology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// Only face neighbors (4-connectivity in 2D)
    Face,

    /// Face and edge neighbors (8-connectivity in 2D)
    FaceEdge,

    /// Face, edge, and vertex neighbors (full connectivity)
    Full,
}
