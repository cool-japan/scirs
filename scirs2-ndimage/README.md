# SciRS2 NDImage

[![crates.io](https://img.shields.io/crates/v/scirs2-ndimage.svg)](https://crates.io/crates/scirs2-ndimage)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-ndimage)](https://docs.rs/scirs2-ndimage)

Multidimensional image processing functionality for the SciRS2 scientific computing library. This module provides a comprehensive set of tools for image processing in n-dimensional arrays, including filtering, morphology, measurements, segmentation, and interpolation.

## Features

- **Filters**: Full n-dimensional filtering support including Gaussian, median, rank (min/max/percentile), and edge filters (Sobel, Prewitt, Laplace)
- **Morphology**: Binary and grayscale morphological operations with optimized distance transforms and hit-or-miss operations
- **Measurements**: Region properties, moments (raw, central, normalized, Hu), extrema detection, comprehensive statistics
- **Segmentation**: Advanced thresholding (Otsu, adaptive) and watershed algorithms
- **Feature Detection**: Corner detection (Harris, FAST) and edge detection (Canny, unified edge detector)
- **Interpolation**: Comprehensive spline and geometric interpolation with affine transforms, rotation, and zoom
- **Performance**: Optimized implementations with SIMD acceleration and parallel processing support

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-ndimage = "0.1.0-alpha.5"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-ndimage = { version = "0.1.0-alpha.5", features = ["parallel", "simd"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_ndimage::{filters, morphology, measurements, interpolation};
use ndarray::{Array2, Array3};

// Create a sample 2D image
let image = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| {
    if (i > 3 && i < 7) && (j > 3 && j < 7) {
        1.0
    } else {
        0.0
    }
});

// Apply Gaussian filter
let sigma = 1.0;
let filtered = filters::gaussian::gaussian_filter(&image, sigma, None, None).unwrap();

// Apply rank filters (works on any dimension)
let max_filtered = filters::rank::maximum_filter(&image, &[3, 3], None).unwrap();
let min_filtered = filters::rank::minimum_filter(&image, &[3, 3], None).unwrap();

// Apply binary dilation
let struct_elem = morphology::structuring::generate_disk(2).unwrap();
let dilated = morphology::binary::binary_dilation(&image, &struct_elem, None, None).unwrap();

// Hit-or-miss transform for pattern detection
let pattern = Array2::from_shape_vec((3, 3), vec![0, 1, 0, 1, 1, 1, 0, 1, 0]).unwrap();
let hit_miss = morphology::binary::binary_hit_or_miss(&image, &pattern, None, None).unwrap();

// Distance transform with multiple metrics
use ndarray::IxDyn;
let image_dyn = image.clone().into_dimensionality::<IxDyn>().unwrap();
let (distances, _) = morphology::distance_transform_edt(&image_dyn, None, true, false);

// Measure region properties
let labels = measurements::region::label(&image, None).unwrap();
let props = measurements::region::regionprops(&labels, Some(&image), None).unwrap();
for region in props {
    println!("Region: area={}, centroid={:?}", region.area, region.centroid);
}

// 3D example - rank filters work on any dimension
let volume = Array3::<f64>::zeros((20, 20, 20));
let filtered_3d = filters::rank::median_filter(&volume, &[3, 3, 3], None).unwrap();

// Rotate image using spline interpolation
let rotated = interpolation::geometric::rotate(&image, 45.0, None, None, None, None).unwrap();
```

## Components

### Filters

Image filtering functionality:

```rust
use scirs2_ndimage::filters::{
    // Gaussian filters
    gaussian::gaussian_filter,         // Apply Gaussian filter to an n-dimensional array
    gaussian::gaussian_filter1d,       // Apply Gaussian filter along a single axis
    gaussian::gaussian_gradient_magnitude, // Compute gradient magnitude using Gaussian derivatives
    gaussian::gaussian_laplace,        // Compute Laplace filter using Gaussian 2nd derivatives
    
    // Median filters
    median::median_filter,           // Apply median filter (n-dimensional)
    
    // Rank filters (full n-dimensional support)
    rank::rank_filter,             // Generic rank filter (any dimension)
    rank::percentile_filter,       // Percentile filter (nth_percentile)
    rank::minimum_filter,          // Minimum filter
    rank::maximum_filter,          // Maximum filter
    
    // Edge filters
    edge::prewitt,                 // Apply Prewitt filter
    edge::sobel,                   // Apply Sobel filter (n-dimensional)
    edge::laplace,                 // Apply Laplace filter
    edge::scharr,                  // Apply Scharr filter
    edge::roberts,                 // Apply Roberts cross-gradient filter
    
    // Generic and specialized filters
    generic::generic_filter,       // Apply custom function over sliding window
    uniform::uniform_filter,       // Apply uniform/box filter
    bilateral::bilateral_filter,   // Edge-preserving bilateral filter
    
    // Convolution
    convolve::convolve,            // N-dimensional convolution
    convolve::convolve1d,          // 1-dimensional convolution
};
```

### Morphology

Morphological operations:

```rust
use scirs2_ndimage::morphology::{
    // Binary morphology
    binary::binary_erosion,          // Binary erosion
    binary::binary_dilation,         // Binary dilation
    binary::binary_opening,          // Binary opening
    binary::binary_closing,          // Binary closing
    binary::binary_hit_or_miss,      // Binary hit-or-miss transform
    binary::binary_propagation,      // Binary propagation
    binary::binary_fill_holes,       // Fill holes in binary objects
    
    // Grayscale morphology
    grayscale::grey_erosion,         // Grayscale erosion
    grayscale::grey_dilation,        // Grayscale dilation
    grayscale::grey_opening,         // Grayscale opening
    grayscale::grey_closing,         // Grayscale closing
    grayscale::white_tophat,         // White top-hat transform
    grayscale::black_tophat,         // Black top-hat transform
    grayscale::morphological_gradient, // Morphological gradient
    grayscale::morphological_laplace, // Morphological Laplace
    
    // Distance transforms (optimized)
    distance_transform_edt,          // Euclidean distance transform
    distance_transform_cdt,          // City-block (Manhattan) distance transform
    distance_transform_bf,           // Brute-force distance transform (multiple metrics)
    
    // Connected components
    connected::label,                // Label connected components
    connected::find_objects,         // Find objects in labeled array
    connected::remove_small_objects, // Remove small connected components
    
    // Structuring elements
    structuring::generate_binary_structure, // Generate binary structuring element
    structuring::iterate_structure,  // Iterate structure by successive dilations
};
```

### Measurements

Measurement functions:

```rust
use scirs2_ndimage::measurements::{
    // Statistics
    statistics::sum_labels,            // Sum of array elements over labeled regions
    statistics::mean_labels,           // Mean of array elements over labeled regions
    statistics::count_labels,          // Count elements in labeled regions
    
    // Extrema
    extrema::extrema,                  // Min, max, min position, max position
    extrema::local_extrema,            // Find local extrema in an array
    
    // Moments
    moments::moments,                  // Calculate all raw moments
    moments::moments_central,          // Calculate central moments
    moments::moments_normalized,       // Calculate normalized moments
    moments::moments_hu,               // Calculate Hu moments (rotation invariant)
    moments::center_of_mass,           // Calculate center of mass
    moments::inertia_tensor,           // Calculate inertia tensor
    
    // Region properties
    region::find_objects,              // Find objects in a labeled array
    region::region_properties,         // Measure properties of labeled regions
};
```

### Segmentation

Image segmentation functions:

```rust
use scirs2_ndimage::segmentation::{
    // Thresholding
    thresholding::otsu_threshold,      // Otsu's thresholding method
    thresholding::threshold_binary,    // Basic binary thresholding
    thresholding::adaptive_threshold,  // Adaptive thresholding
    
    // Watershed
    watershed::watershed,              // Watershed algorithm
    watershed::marker_watershed,       // Marker-controlled watershed
};
```

### Features

Feature detection:

```rust
use scirs2_ndimage::features::{
    // Corner detection
    corners::harris_corners,           // Harris corner detector
    corners::fast_corners,             // FAST corner detector
    
    // Edge detection
    edges::canny,                      // Canny edge detector
    edges::edge_detector,              // Unified edge detector with multiple methods
    edges::edge_detector_simple,       // Simple edge detector
    edges::gradient_edges,             // Gradient-based edge detection
    edges::laplacian_edges,            // Laplacian-based edge detection
};
```

### Interpolation

Interpolation functions:

```rust
use scirs2_ndimage::interpolation::{
    // Coordinate mapping
    coordinates::map_coordinates,      // Map input array to new coordinates using interpolation
    coordinates::interpn,              // N-dimensional interpolation
    coordinates::value_at_coordinates, // Interpolate value at specific coordinates
    
    // Spline interpolation
    spline::spline_filter,             // Multi-dimensional spline filter
    spline::spline_filter1d,           // Spline filter along a single axis
    spline::bspline,                   // B-spline interpolation
    
    // Geometric transformations
    geometric::shift,                  // Shift an array
    geometric::rotate,                 // Rotate an array
    geometric::zoom,                   // Zoom an array
    
    // Advanced transforms
    transform::affine_transform,       // Apply an affine transformation
    transform::geometric_transform,    // General geometric transformation
};
```

## Benchmarks

The module includes comprehensive benchmarks for performance-critical operations:

- **Filter benchmarks**: Rank filters, generic filters, edge filters, boundary mode comparisons
- **Morphological operations benchmarks**: Binary/grayscale erosion, dilation, hit-or-miss transforms
- **Distance transform benchmarks**: Optimized vs. brute force, 2D vs. 3D, different metrics
- **Interpolation benchmarks**: Affine transforms, map coordinates, different interpolation orders
- **Multi-dimensional scaling analysis**: Performance across 1D, 2D, 3D, and higher dimensions

Run benchmarks with:
```bash
cargo bench                          # Run all benchmarks
cargo bench --bench filters_bench    # Run specific benchmark suite
cargo bench --bench distance_transform_bench
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
