//! N-dimensional image processing module
//!
//! This module provides functions for processing and analyzing n-dimensional arrays as images.
//! It includes filters, interpolation, measurements, morphology, feature detection, and segmentation functions.

// Public modules
pub mod error;
pub mod features;
pub mod filters;
pub mod interpolation;
pub mod measurements;
pub mod morphology;
pub mod segmentation;

// Re-exports
pub use self::error::*;

// Feature detection module exports
pub use self::features::{
    canny, edge_detector, edge_detector_simple, fast_corners, gradient_edges, harris_corners,
    laplacian_edges, sobel_edges, EdgeDetectionAlgorithm, EdgeDetectionConfig, GradientMethod,
};

// Filters module exports
pub use self::filters::{
    bilateral_filter, convolve, filter_functions, gaussian_filter, gaussian_filter_f32,
    gaussian_filter_f64, generic_filter, laplace, maximum_filter, median_filter, minimum_filter,
    percentile_filter, rank_filter, sobel, uniform_filter, BorderMode,
};

#[cfg(feature = "simd")]
pub use self::filters::{bilateral_filter_simd_f32, bilateral_filter_simd_f64};

// Segmentation module exports
pub use self::segmentation::{
    adaptive_threshold, marker_watershed, otsu_threshold, threshold_binary, watershed,
    AdaptiveMethod,
};

// Interpolation module exports
pub use self::interpolation::{
    affine_transform, bspline, geometric_transform, map_coordinates, rotate, shift, spline_filter,
    spline_filter1d, value_at_coordinates, zoom, BoundaryMode, InterpolationOrder,
};

// Measurements module exports
pub use self::measurements::{
    center_of_mass, count_labels, extrema, find_objects, local_extrema, mean_labels, moments,
    moments_inertia_tensor, peak_prominences, peak_widths, region_properties, sum_labels,
    variance_labels, RegionProperties,
};

// Morphology module exports
pub use self::morphology::{
    binary_closing, binary_dilation, binary_erosion, binary_fill_holes, binary_hit_or_miss,
    binary_opening, black_tophat, box_structure, disk_structure, find_boundaries,
    generate_binary_structure, grey_closing, grey_dilation, grey_erosion, grey_opening,
    iterate_structure, label, morphological_gradient, morphological_laplace, remove_small_holes,
    remove_small_objects, white_tophat, Connectivity, MorphBorderMode,
};
