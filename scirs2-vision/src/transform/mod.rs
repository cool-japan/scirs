//! Image transformation module
//!
//! This module provides functionality for geometric transformations
//! like affine, perspective, and non-rigid transformations, as well as various
//! interpolation methods for image resampling.

pub mod affine;
pub mod interpolation;
pub mod non_rigid;
/// Perspective transformation module
///
/// Contains functions and structs for applying perspective transformations
/// to images, including homographies and perspective warping.
pub mod perspective;

pub use affine::{estimate_affine_transform, warp_affine, AffineTransform};
pub use interpolation::{
    convolve_1d, resize, resize_bicubic, resize_convolution, resize_edge_preserving,
    resize_lanczos, InterpolationMethod,
};
pub use non_rigid::{
    generate_grid_points, warp_elastic, warp_non_rigid, warp_thin_plate_spline, ElasticDeformation,
    NonRigidTransform, ThinPlateSpline,
};
pub use perspective::{
    correct_perspective, detect_quad, warp_perspective, BorderMode, PerspectiveTransform,
};
