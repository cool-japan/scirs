//! Perspective transformation modules
//!
//! This module has been refactored into focused sub-modules for better organization:
//! - `core`: Core perspective transformation structures and basic operations
//! - `estimation`: RANSAC-based homography estimation for robust transformation
//! - `warping`: Image warping and interpolation operations with SIMD optimization
//! - `rectification`: Perspective correction and automatic rectification

pub mod core;
pub mod estimation;
pub mod warping;
pub mod rectification;

// Re-export main public API from core module
pub use core::{
    BorderMode, PerspectiveTransform, RansacParams, RansacResult,
};

// Re-export estimation functionality
pub use estimation::{
    find_homography_ransac, find_homography_adaptive_ransac,
    evaluate_homography_quality, validate_homography_geometry,
};

// Re-export warping functionality
pub use warping::{
    warp_perspective, warp_perspective_simd, bilinear_interpolate,
    bilinear_interpolate_simd, modulo,
};

// Re-export rectification functionality
pub use rectification::{
    auto_perspective_correction, extract_rectangle,
};