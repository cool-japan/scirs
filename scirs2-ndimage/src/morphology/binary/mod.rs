//! Binary morphological operations on arrays
//!
//! This module provides binary erosion, dilation, opening, closing, fill holes,
//! and hit-or-miss transform for 1D, 2D, and n-dimensional arrays.
//!
//! # Important Implementation Notes
//!
//! 1. Dimensions and handling:
//!    - 1D arrays: Fully supported with optimized implementation
//!    - 2D arrays: Fully supported with optimized implementation
//!    - nD arrays (n > 2): Support is limited and uses dynamic dimension handling
//!
//! 2. Function signatures:
//!    - All functions accept generic dimension parameter D
//!    - When D is Ix1 or Ix2, specific implementation is used
//!    - When D is IxDyn, the function checks the dimensionality and routes to the right implementation
//!
//! # Recommended Usage
//!
//! - For 1D and 2D arrays: Both the generic functions here and the functions in simple_morph work well
//! - For higher dimensional arrays: Convert to IxDyn first, but be aware of limitations
//! - For production code: Prefer the simple_morph module when working with 2D arrays

pub mod dilation;
pub mod erosion;
pub mod operations;

pub use dilation::binary_dilation;
pub use erosion::binary_erosion;
pub use operations::{binary_closing, binary_fill_holes, binary_hit_or_miss, binary_opening};
