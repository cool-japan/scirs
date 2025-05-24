//! Computer vision module for SciRS2
//!
//! This module provides computer vision functionality that builds on top of the
//! scirs2-ndimage module, including image processing, feature detection, and segmentation.

#![warn(missing_docs)]

// Re-export image crate with the expected name
extern crate image as image;

pub mod color;
pub mod error;
pub mod feature;
/// Image preprocessing functionality
///
/// Includes operations like filtering, histogram manipulation,
/// and morphological operations.
pub mod preprocessing;
pub mod quality;
pub mod segmentation;
pub mod transform;

// Comment out problematic modules during tests to focus on fixing other issues
#[cfg(not(test))]
/// Private transform module for compatibility
///
/// Contains placeholder modules that help maintain compatibility
/// with external code that might reference these modules directly.
pub mod _transform {
    /// Non-rigid transformation compatibility module
    pub mod non_rigid {}
    /// Perspective transformation compatibility module
    pub mod perspective {}
}

// Re-export commonly used items
pub use error::{Result, VisionError};

// Re-export feature functionality (select items to avoid conflicts)
pub use feature::{
    array_to_image,
    descriptor::{detect_and_compute, match_descriptors, Descriptor, KeyPoint},
    harris_corners, image_to_array,
    log_blob::{log_blob_detect, log_blobs_to_image, LogBlob, LogBlobConfig},
    orb::{detect_and_compute_orb, match_orb_descriptors, OrbConfig, OrbDescriptor},
    sobel_edges,
};
// Re-export with unique name to avoid ambiguity
pub use feature::homography::warp_perspective as feature_warp_perspective;

// Re-export segmentation functionality
pub use segmentation::*;

// Re-export preprocessing functionality
pub use preprocessing::*;

// Re-export color functionality
pub use color::*;

// Re-export transform functionality (select items to avoid conflicts)
pub use transform::{
    affine::{estimate_affine_transform, warp_affine, AffineTransform, BorderMode},
    non_rigid::{
        warp_elastic, warp_non_rigid, warp_thin_plate_spline, ElasticDeformation, ThinPlateSpline,
    },
    perspective::{correct_perspective, BorderMode as PerspectiveBorderMode, PerspectiveTransform},
};
// Re-export with unique name to avoid ambiguity
pub use transform::perspective::warp_perspective as transform_warp_perspective;
