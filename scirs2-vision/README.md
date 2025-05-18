# SciRS2 Vision

[![crates.io](https://img.shields.io/crates/v/scirs2-vision.svg)](https://crates.io/crates/scirs2-vision)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-vision)](https://docs.rs/scirs2-vision)

Computer vision module for SciRS2, providing functionality for image processing, feature detection, segmentation, and color transformations.

## Overview

The `scirs2-vision` module is designed to provide comprehensive computer vision capabilities for scientific computing applications. It builds upon the foundation of the `scirs2-ndimage` module but provides additional specialized functionality specifically for computer vision tasks.

## Features

### Feature Detection and Description
- Edge detection with Sobel operator
- Canny edge detection
- Prewitt edge detection
- Laplacian edge detection (including Laplacian of Gaussian)
- Zero-crossing detection for edge finding
- Corner detection with Harris detector
- Shi-Tomasi corner detection (Good Features to Track)
- FAST corner detection for real-time applications
- Blob detection (Difference of Gaussians - DoG)
- Blob detection (Laplacian of Gaussian - LoG)
- Region detection (Maximally Stable Extremal Regions - MSER)
- Hough Circle Transform for circle detection
- Feature point extraction with sub-pixel accuracy
- SIFT-like feature descriptors and matching

### Image Segmentation
- Binary thresholding
- Otsu's automatic thresholding
- Adaptive thresholding (mean, Gaussian)
- Connected component labeling

### Preprocessing
- Brightness and contrast normalization
- Histogram equalization
- Gaussian blur
- Unsharp masking

### Color Processing
- RGB ↔ HSV conversion
- RGB ↔ LAB conversion
- Channel splitting and merging
- Weighted grayscale conversion

### Morphological Operations
- Erosion and dilation
- Opening and closing
- Morphological gradient
- Top-hat and black-hat transforms

## Examples

### Feature Detection

```rust
use scirs2_vision::feature::{
    sobel_edges, canny_simple, prewitt_edges, laplacian_edges, 
    laplacian_of_gaussian, harris_corners
};
use scirs2_vision::preprocessing::gaussian_blur;

// Load an image
let img = image::open("input.jpg")?;

// Preprocess
let blurred = gaussian_blur(&img, 1.0)?;

// Detect edges using Sobel
let sobel = sobel_edges(&blurred, 0.1)?;

// Detect edges using Canny
let canny = canny_simple(&blurred, 1.0)?;

// Detect edges using Prewitt
let prewitt = prewitt_edges(&blurred, 0.1)?;

// Detect edges using Laplacian
let laplacian = laplacian_edges(&blurred, 0.05, true)?;

// Detect edges using Laplacian of Gaussian
let log = laplacian_of_gaussian(&img, 2.0, 0.05)?;

// Detect corners
let corners = harris_corners(&blurred, 3, 0.04, 0.01)?;
```

### Color Transformations

```rust
use scirs2_vision::color::{rgb_to_hsv, rgb_to_lab, split_channels};

// Load an image
let img = image::open("input.jpg")?;

// Convert to HSV
let hsv = rgb_to_hsv(&img)?;

// Convert to LAB
let lab = rgb_to_lab(&img)?;

// Split into channels
let (r_channel, g_channel, b_channel) = split_channels(&img)?;
```

### Image Segmentation

```rust
use scirs2_vision::segmentation::{
    threshold_binary, otsu_threshold, adaptive_threshold, connected_components,
    AdaptiveMethod,
};

// Load an image
let img = image::open("input.jpg")?;

// Apply Otsu's thresholding
let (binary, threshold) = otsu_threshold(&img)?;

// Apply adaptive thresholding
let adaptive = adaptive_threshold(&img, 11, 0.02, AdaptiveMethod::Gaussian)?;

// Perform connected component labeling
let (labeled, num_components) = connected_components(&binary)?;
```

### Morphological Operations

```rust
use scirs2_vision::preprocessing::{
    erode, dilate, opening, closing, morphological_gradient, StructuringElement,
};

// Load an image
let img = image::open("input.jpg")?;

// Define a structuring element
let se = StructuringElement::Ellipse(5, 5);

// Apply opening (erosion followed by dilation)
let opened = opening(&img, se)?;

// Calculate morphological gradient
let gradient = morphological_gradient(&img, se)?;
```

### Blob and Region Detection

```rust
use scirs2_vision::feature::{
    dog_detect, log_blob_detect, mser_detect, hough_circles,
    DogConfig, LogBlobConfig, MserConfig, HoughCircleConfig,
};

// Load an image
let img = image::open("input.jpg")?;

// Detect blobs using Difference of Gaussians
let dog_config = DogConfig::default();
let dog_blobs = dog_detect(&img, dog_config)?;

// Detect blobs using Laplacian of Gaussian
let log_config = LogBlobConfig::default();
let log_blobs = log_blob_detect(&img, log_config)?;

// Detect stable regions using MSER
let mser_config = MserConfig::default();
let mser_regions = mser_detect(&img, mser_config)?;

// Detect circles using Hough Transform
let hough_config = HoughCircleConfig::default();
let circles = hough_circles(&img, hough_config)?;
```

## Installation

Add `scirs2-vision` to your dependencies in `Cargo.toml`:

```toml
[dependencies]
scirs2-vision = "0.1.0-alpha.3"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-vision = { version = "0.1.0-alpha.3", features = ["parallel"] }
```

## Documentation

For detailed documentation, examples, and API reference, please refer to the [API documentation](https://docs.rs/scirs2-vision) and the examples directory.

## Testing

The module includes a comprehensive test suite to ensure functionality works as expected:

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_grayscale_conversion
```

## Dependencies

- `scirs2-core`: Core functionality for SciRS2
- `scirs2-ndimage`: N-dimensional image processing
- `ndarray`: N-dimensional array manipulation
- `image`: Basic image processing in Rust
- `num-traits` and `num-complex`: Numerical type traits

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
## Contributing

Contributions are welcome! Please see the project's [CONTRIBUTING.md](../CONTRIBUTING.md) file for guidelines.
