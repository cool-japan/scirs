# SciRS2 NDImage

[![crates.io](https://img.shields.io/crates/v/scirs2-ndimage.svg)](https://crates.io/crates/scirs2-ndimage)
[![License](https://img.shields.io/crates/l/scirs2-ndimage.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-ndimage)](https://docs.rs/scirs2-ndimage)

Multidimensional image processing functionality for the SciRS2 scientific computing library. This module provides a comprehensive set of tools for image processing in n-dimensional arrays, including filtering, morphology, measurements, segmentation, and interpolation.

## Features

- **Filters**: Various filters including Gaussian, median, rank, and edge filters
- **Morphology**: Binary and grayscale morphological operations
- **Measurements**: Region properties, moments, extrema detection
- **Segmentation**: Thresholding and watershed algorithms
- **Feature Detection**: Corner and edge detection
- **Interpolation**: Spline and geometric interpolation algorithms

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-ndimage = "0.1.0-alpha.1"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-ndimage = { version = "0.1.0-alpha.1", features = ["parallel"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_ndimage::{filters, morphology, measurements, interpolation};
use ndarray::{Array2, Array, Ix2};

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

// Apply binary dilation
let struct_elem = morphology::structuring::generate_disk(2).unwrap();
let dilated = morphology::binary::binary_dilation(&image, &struct_elem, None, None).unwrap();

// Measure region properties
let labels = measurements::region::label(&image, None).unwrap();
let props = measurements::region::regionprops(&labels, Some(&image), None).unwrap();
for region in props {
    println!("Region: area={}, centroid={:?}", region.area, region.centroid);
}

// Rotate image using spline interpolation
let rotated = interpolation::geometric::rotate(&image, 45.0, None, None, None, None).unwrap();
```

## Components

### Filters

Image filtering functionality:

```rust
use scirs2_ndimage::filters::{
    // Gaussian filters
    gaussian_filter,         // Apply Gaussian filter to an n-dimensional array
    gaussian_filter1d,       // Apply Gaussian filter along a single axis
    gaussian_gradient_magnitude, // Compute gradient magnitude using Gaussian derivatives
    gaussian_laplace,        // Compute Laplace filter using Gaussian 2nd derivatives
    
    // Median filters
    median_filter,           // Apply median filter
    
    // Rank filters
    rank_filter,             // Generic rank filter
    percentile_filter,       // Percentile filter (nth_percentile)
    minimum_filter,          // Minimum filter
    maximum_filter,          // Maximum filter
    
    // Edge filters
    prewitt,                 // Apply Prewitt filter
    sobel,                   // Apply Sobel filter
    laplace,                 // Apply Laplace filter
    
    // Convolution
    convolve,                // N-dimensional convolution
    convolve1d,              // 1-dimensional convolution
};
```

### Morphology

Morphological operations:

```rust
use scirs2_ndimage::morphology::{
    // Binary morphology
    binary_erosion,          // Binary erosion
    binary_dilation,         // Binary dilation
    binary_opening,          // Binary opening
    binary_closing,          // Binary closing
    binary_hit_or_miss,      // Binary hit-or-miss transform
    binary_propagation,      // Binary propagation
    binary_fill_holes,       // Fill holes in binary objects
    
    // Grayscale morphology
    grey_erosion,            // Grayscale erosion
    grey_dilation,           // Grayscale dilation
    grey_opening,            // Grayscale opening
    grey_closing,            // Grayscale closing
    
    // Connected components
    label,                   // Label connected components
    find_objects,            // Find objects in labeled array
    
    // Structuring elements
    generate_disk,           // Generate disk-shaped structuring element
    generate_rectangle,      // Generate rectangle-shaped structuring element
    generate_cross,          // Generate cross-shaped structuring element
    iterate_structure,       // Iterate structure by successive dilations
};
```

### Measurements

Measurement functions:

```rust
use scirs2_ndimage::measurements::{
    // Statistics
    sum,                     // Sum of array elements over a labeled region
    mean,                    // Mean of array elements over a labeled region
    variance,                // Variance over a labeled region
    standard_deviation,      // Standard deviation over a labeled region
    
    // Extrema
    minimum,                 // Minimum of array elements over a labeled region
    maximum,                 // Maximum of array elements over a labeled region
    minimum_position,        // Position of the minimum
    maximum_position,        // Position of the maximum
    extrema,                 // Min, max, min position, max position
    
    // Moments
    moments,                 // Calculate all raw moments
    moments_central,         // Calculate central moments
    moments_normalized,      // Calculate normalized moments
    moments_hu,              // Calculate Hu moments
    
    // Region properties
    label,                   // Label features in an array
    regionprops,             // Measure properties of labeled regions
    find_objects,            // Find objects in a labeled array
};
```

### Segmentation

Image segmentation functions:

```rust
use scirs2_ndimage::segmentation::{
    // Thresholding
    threshold_otsu,          // Otsu's thresholding method
    threshold_isodata,       // ISODATA thresholding
    threshold_li,            // Li's minimum cross entropy thresholding
    threshold_yen,           // Yen's thresholding method
    threshold_adaptive,      // Adaptive thresholding
    
    // Watershed
    watershed,               // Watershed algorithm
    distance_transform_edt,  // Euclidean distance transform
};
```

### Features

Feature detection:

```rust
use scirs2_ndimage::features::{
    // Corner detection
    corner_harris,           // Harris corner detector
    corner_kitchen_rosenfeld, // Kitchen and Rosenfeld corner detector
    corner_shi_tomasi,       // Shi-Tomasi corner detector
    corner_foerstner,        // Foerstner corner detector
    
    // Edge detection
    canny,                   // Canny edge detector
    roberts,                 // Roberts edge detector
    prewitt,                 // Prewitt edge detector
    sobel,                   // Sobel edge detector
};
```

### Interpolation

Interpolation functions:

```rust
use scirs2_ndimage::interpolation::{
    // Spline interpolation
    map_coordinates,         // Map input array to new coordinates using interpolation
    spline_filter,           // Multi-dimensional spline filter
    spline_filter1d,         // Spline filter along a single axis
    
    // Geometric transformations
    shift,                   // Shift an array
    rotate,                  // Rotate an array
    zoom,                    // Zoom an array
    affine_transform,        // Apply an affine transformation
};
```

## Benchmarks

The module includes benchmarks for performance-critical operations:

- Rank filter benchmarks
- Convolution benchmarks
- Morphological operations benchmarks

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.