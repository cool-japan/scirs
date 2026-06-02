# SciRS2 NDImage

[![crates.io](https://img.shields.io/crates/v/scirs2-ndimage.svg)](https://crates.io/crates/scirs2-ndimage)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-ndimage)](https://docs.rs/scirs2-ndimage)
[![Version](https://img.shields.io/badge/version-0.5.0-green)]()

**scirs2-ndimage** is the N-dimensional image processing crate for the [SciRS2](https://github.com/cool-japan/scirs) scientific computing library. It provides a comprehensive toolkit for filtering, morphology, interpolation, measurements, segmentation, and feature detection on arrays of arbitrary dimensionality, modeled after SciPy's `ndimage` module.

## What scirs2-ndimage Provides

Use scirs2-ndimage when you need to:

- Filter N-dimensional arrays (Gaussian, median, rank, bilateral, edge detection)
- Apply morphological operations to binary or grayscale images in any dimension
- Measure region properties (area, centroid, moments, Hu moments) after labeling
- Segment images with watershed, active contours, or graph cut methods
- Transform arrays geometrically (rotate, zoom, shift, affine transform)
- Analyze 3D volumetric data (medical images, electron microscopy)
- Process hyperspectral imagery
- Compute co-occurrence matrices and texture features
- Detect features (corners, edges, SIFT descriptors, HOG)
- Perform atlas-based segmentation

## Features (v0.5.0)

### Image Filtering
- **Gaussian Filters**: `gaussian_filter`, `gaussian_filter1d`, `gaussian_gradient_magnitude`, `gaussian_laplace`
- **Median Filter**: N-dimensional median filter with configurable footprint
- **Rank Filters**: Minimum, maximum, percentile, generic rank filter (full n-dimensional support)
- **Edge Detection**: Sobel, Prewitt, Laplacian, Scharr, Roberts cross-gradient
- **Bilateral Filter**: Edge-preserving bilateral filtering
- **Uniform Filter**: Box/uniform convolution filter
- **Generic Filter**: Apply any custom function over a sliding window
- **Convolution**: N-dimensional `convolve` and `convolve1d`
- **Boundary Modes**: `reflect`, `nearest`, `wrap`, `mirror`, `constant`
- **Fourier Filters**: Fourier Gaussian, uniform, ellipsoid, shift operations

### Morphological Operations
- **Binary Morphology**: Erosion, dilation, opening, closing, hit-or-miss transform, propagation, hole filling
- **Grayscale Morphology**: Erosion, dilation, opening, closing, top-hat (white/black), morphological gradient, Laplace
- **Distance Transforms**: Euclidean (EDT via Felzenszwalb-Huttenlocher O(n) algorithm), city-block, chessboard
- **Connected Components**: Labeling, find objects, remove small objects
- **Structuring Elements**: Generate disk, square, diamond, and arbitrary structuring elements
- **Skeletonization**: Topological thinning to medial axis

### Image Measurements
- **Region Statistics**: Sum, mean, variance, standard deviation, min, max per label
- **Moments**: Raw moments, central moments, normalized moments, Hu moments (rotation-invariant)
- **Region Properties**: Area, perimeter, centroid, bounding box, eccentricity, orientation, principal axes
- **Center of Mass**: N-dimensional center of mass computation
- **Extrema**: Local and global minima/maxima with positions
- **Histograms**: Per-label histogram computation
- **Inertia Tensor**: Region inertia tensor for orientation analysis

### Image Segmentation
- **Thresholding**: Binary, Otsu's automatic, adaptive (mean/Gaussian)
- **Watershed**: Standard watershed and marker-controlled watershed
- **Active Contours**: Snakes with gradient vector flow (GVF)
- **Level Set Methods**: Chan-Vese segmentation (single and multi-phase)
- **Graph Cuts**: Max-flow/min-cut segmentation with interactive refinement
- **SLIC Superpixels**: Simple Linear Iterative Clustering (2D and 3D)
- **Atlas-Based Segmentation**: Label transfer via registration atlas

### Feature Detection
- **Edge Detection**: Canny edge detector, unified edge detection API
- **Corner Detection**: Harris corners, FAST corners
- **SIFT Descriptor Computation**: Scale-space keypoint detection and description
- **HOG (Histogram of Oriented Gradients)**: Cell-based gradient histogram features
- **Template Matching**: Normalized cross-correlation, zero-mean NCC
- **Gabor Filters**: 2D Gabor filter bank for texture analysis
- **Shape Analysis**: Moments-based shape descriptors, shape matching

### Geometric Interpolation
- **Map Coordinates**: Interpolate array at arbitrary coordinates (0th-5th order splines)
- **Affine Transform**: Apply an affine transformation matrix
- **Geometric Transform**: General geometric transformation with custom mapping
- **Shift**: Sub-pixel shift with spline interpolation
- **Rotate**: Array rotation about any axis
- **Zoom**: Uniform and anisotropic zooming
- **Spline Filter**: Pre-filter for spline interpolation (`spline_filter`, `spline_filter1d`)

### 3D Volume Analysis
- **Volumetric Operations**: 3D morphology, filtering, distance transforms
- **3D Filters**: 3D Gaussian, Sobel, Laplacian, bilateral
- **Volume Measurements**: 3D region properties, surface area, Euler characteristic
- **Slice Processing**: Per-slice operations on 3D stacks

### Medical Image Processing
- **Frangi Vesselness**: Multi-scale vessel enhancement filter
- **Bone Enhancement**: Bone structure enhancement for CT data
- **Lung Nodule Detection**: Basic nodule candidate generation
- **DICOM-Compatible Arrays**: Works natively with 3D medical arrays

### Hyperspectral Image Analysis
- **Band Processing**: Per-band filtering and morphology
- **Spectral Indices**: NDVI, NDWI, and custom spectral index computation
- **Spectral Unmixing**: Linear unmixing of spectral signatures
- **Cloud Detection**: Cloud and shadow masking for satellite imagery
- **Pan-Sharpening**: Fusion of panchromatic and multispectral bands

### Co-occurrence Matrices and Texture
- **GLCM**: Gray-level co-occurrence matrix computation (2D and 3D)
- **Texture Features**: Contrast, correlation, energy, homogeneity from GLCM
- **LBP**: Local binary patterns
- **Gabor Feature Maps**: Multi-scale multi-orientation Gabor responses

### Deep Feature Extraction Interface
- Interface for forwarding arrays through external feature extractors
- Hooks for deep learning model integration (via scirs2-neural)

## Installation

```toml
[dependencies]
scirs2-ndimage = "0.5.0"
```

For parallel processing and SIMD:

```toml
[dependencies]
scirs2-ndimage = { version = "0.5.0", features = ["parallel", "simd"] }
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `parallel` | Enable Rayon-based multi-core parallel processing (recommended for arrays >10K elements) |
| `simd` | Enable SIMD vectorization for filters and morphological operations |

## Quick Start

```rust
use scirs2_ndimage::{filters, morphology};
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Array2::<f64>::from_shape_fn((100, 100), |(i, j)| {
        if (i > 30 && i < 70) && (j > 30 && j < 70) { 1.0 } else { 0.0 }
    });

    // Gaussian smoothing
    let smoothed = filters::gaussian_filter(&image, 2.0, None, None)?;

    // Morphological dilation
    let struct_elem = morphology::structuring::generate_disk(3)?;
    let dilated = morphology::binary_dilation(&image, &struct_elem, None, None)?;

    println!("Image processed: {:?}", smoothed.shape());
    Ok(())
}
```

## Comprehensive Examples

### Filtering

```rust
use scirs2_ndimage::filters;
use scirs2_core::ndarray::Array2;

fn filtering_example() -> Result<(), Box<dyn std::error::Error>> {
    let image = Array2::<f64>::from_shape_fn((256, 256), |(i, j)| {
        (i as f64 * 0.1).sin() * (j as f64 * 0.1).cos()
    });

    // Gaussian filter
    let gaussian = filters::gaussian_filter(&image, 2.0, None, None)?;

    // Median filter (rank-based, N-dimensional)
    let median = filters::median_filter(&image, &[5, 5], None)?;

    // Maximum filter
    let dilated = filters::maximum_filter(&image, &[3, 3], None)?;

    // Sobel edge detection
    let edges_x = filters::sobel(&image, 0, None)?;
    let edges_y = filters::sobel(&image, 1, None)?;

    // Custom generic filter (mean over 5x5 window)
    let mean_filtered = filters::generic_filter(
        &image, |window| window.iter().sum::<f64>() / window.len() as f64,
        &[5, 5], None,
    )?;

    println!("All filters applied");
    Ok(())
}
```

### Morphological Operations

```rust
use scirs2_ndimage::morphology;
use scirs2_core::ndarray::Array2;

fn morphology_example() -> Result<(), Box<dyn std::error::Error>> {
    let binary = Array2::<f64>::from_shape_fn((100, 100), |(i, j)| {
        if i > 30 && i < 70 && j > 30 && j < 70 { 1.0 } else { 0.0 }
    });

    let disk = morphology::structuring::generate_disk(5)?;

    // Binary erosion and dilation
    let eroded = morphology::binary_erosion(&binary, &disk, None, None)?;
    let dilated = morphology::binary_dilation(&binary, &disk, None, None)?;

    // Opening removes small bright regions
    let opened = morphology::binary_opening(&binary, &disk, None, None)?;

    // Distance transform (Euclidean, O(n) algorithm)
    use scirs2_core::ndarray::IxDyn;
    let dyn_img = binary.into_dimensionality::<IxDyn>().unwrap();
    let (distances, _indices) = morphology::distance_transform_edt(&dyn_img, None, true, false);

    // Hit-or-miss for pattern detection
    let pattern = Array2::from_shape_vec((3, 3), vec![0i32, 1, 0, 1, 1, 1, 0, 1, 0]).unwrap();
    // let hit_miss = morphology::binary_hit_or_miss(&binary, &pattern, None, None)?;

    Ok(())
}
```

### Region Measurements

```rust
use scirs2_ndimage::{measurements, morphology};
use scirs2_core::ndarray::Array2;

fn measurement_example() -> Result<(), Box<dyn std::error::Error>> {
    let image = Array2::<f64>::from_shape_fn((100, 100), |(i, j)| {
        if (i as f64 - 50.0).hypot(j as f64 - 50.0) < 20.0 { 1.0 } else { 0.0 }
    });

    // Label connected components
    let labels = measurements::label(&image, None)?;

    // Region properties
    let props = measurements::regionprops(&labels, Some(&image), None)?;
    for region in &props {
        println!("Region {}: area={}, centroid={:?}",
            region.label, region.area, region.centroid);
    }

    // Hu moments (rotation-invariant descriptors)
    let hu = measurements::moments_hu(&image)?;
    println!("Hu moments: {:?}", hu);

    Ok(())
}
```

### Watershed Segmentation

```rust
use scirs2_ndimage::segmentation::watershed;
use scirs2_ndimage::filters;
use scirs2_core::ndarray::Array2;

fn watershed_example() -> Result<(), Box<dyn std::error::Error>> {
    let image = Array2::<f64>::zeros((200, 200));
    // ... populate image ...

    // Compute gradient magnitude as elevation map
    let grad_x = filters::sobel(&image, 0, None)?;
    let grad_y = filters::sobel(&image, 1, None)?;
    let gradient = grad_x.mapv(|v| v * v) + grad_y.mapv(|v| v * v);
    let gradient = gradient.mapv(f64::sqrt);

    // Watershed (seeds placed at local minima of gradient)
    // let labels = watershed(&gradient, &markers, None)?;

    Ok(())
}
```

### 3D Volume Processing

```rust
use scirs2_ndimage::filters;
use scirs2_core::ndarray::Array3;

fn volume_example() -> Result<(), Box<dyn std::error::Error>> {
    let volume = Array3::<f64>::zeros((64, 256, 256));

    // 3D Gaussian smoothing
    let smoothed = filters::gaussian_filter(&volume, 1.5, None, None)?;

    // 3D rank filter
    let max_filtered = filters::maximum_filter(&volume, &[3, 3, 3], None)?;

    // 3D median filter
    let median = filters::median_filter(&volume, &[3, 3, 3], None)?;

    println!("3D volume processed: {:?}", smoothed.shape());
    Ok(())
}
```

### SLIC Superpixels

```rust
use scirs2_ndimage::segmentation::slic_superpixels;
use scirs2_core::ndarray::Array3;

fn slic_example() -> Result<(), Box<dyn std::error::Error>> {
    // let image: Array3<f64> = ...;  // H x W x C color image
    // let labels = slic_superpixels(&image, 100, 10.0, None)?;
    // println!("Superpixel labels shape: {:?}", labels.shape());
    Ok(())
}
```

### Atlas-Based Segmentation

```rust
use scirs2_ndimage::segmentation::atlas::atlas_based_segment;
use scirs2_core::error::CoreResult;

fn atlas_example() -> CoreResult<()> {
    // Register atlas to subject, then transfer labels
    // let result = atlas_based_segment(&subject, &atlas_image, &atlas_labels, None)?;
    // println!("Segmented regions: {:?}", result.unique_labels());
    Ok(())
}
```

## Performance

- **SIMD acceleration**: 2-4x speedup on supported filter and morphology operations
- **Parallel processing**: Linear scaling with CPU cores for large arrays (`parallel` feature)
- **O(n) distance transform**: Felzenszwalb-Huttenlocher separable EDT algorithm
- **Memory-efficient**: Chunked processing for images larger than available RAM
- **N-dimensional**: Consistent API and performance across 1D, 2D, 3D, and higher dimensions

## Compatibility with SciPy ndimage

API is modeled after `scipy.ndimage`. Key equivalents:

| SciRS2 | SciPy |
|--------|-------|
| `filters::gaussian_filter()` | `scipy.ndimage.gaussian_filter()` |
| `filters::median_filter()` | `scipy.ndimage.median_filter()` |
| `filters::sobel()` | `scipy.ndimage.sobel()` |
| `morphology::binary_erosion()` | `scipy.ndimage.binary_erosion()` |
| `morphology::distance_transform_edt()` | `scipy.ndimage.distance_transform_edt()` |
| `measurements::label()` | `scipy.ndimage.label()` |
| `measurements::center_of_mass()` | `scipy.ndimage.center_of_mass()` |
| `interpolation::affine_transform()` | `scipy.ndimage.affine_transform()` |
| `interpolation::map_coordinates()` | `scipy.ndimage.map_coordinates()` |
| `interpolation::rotate()` | `scipy.ndimage.rotate()` |
| `interpolation::zoom()` | `scipy.ndimage.zoom()` |

## Documentation

Full API reference: [docs.rs/scirs2-ndimage](https://docs.rs/scirs2-ndimage)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
