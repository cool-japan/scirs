# SciRS2 NDImage

[![crates.io](https://img.shields.io/crates/v/scirs2-ndimage.svg)](https://crates.io/crates/scirs2-ndimage)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-ndimage)](https://docs.rs/scirs2-ndimage)
[![Version](https://img.shields.io/badge/version-0.6.6-green)]()
[![Status](https://img.shields.io/badge/status-partial-yellow)]()
[![Tests](https://img.shields.io/badge/tests-1199%20passing-brightgreen)]()

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

## Features (v0.6.6)

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
- **Binary Morphology**: Erosion, dilation, opening, closing, hit-or-miss transform, propagation
- **Hole Filling**: `fill_holes_2d` (2D, via morphological reconstruction) is real and working; the general N-D `binary_fill_holes` is currently a no-op stub (returns its input unchanged — see Known Issues)
- **Grayscale Morphology**: Erosion, dilation, opening, closing, top-hat (white/black), morphological gradient, Laplace
- **Distance Transforms**: Euclidean (EDT via Felzenszwalb-Huttenlocher O(n) algorithm), city-block, chessboard
- **Connected Components**: Labeling, find objects, remove small objects
- **Structuring Elements**: Generate disk (exact Euclidean disk in 2D; N > 2 dimensions fall back to a box shape), square, diamond, and arbitrary structuring elements (`iterate_structure`, which is meant to grow a structuring element over N iterations, is currently a no-op stub — see Known Issues)
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
- **SLIC Superpixels**: Simple Linear Iterative Clustering (2D grayscale; `segmentation_advanced::superpixels_slic`)
- **Atlas-Based Segmentation**: Label fusion (majority voting, STAPLE, joint label fusion) over pre-registered atlas label volumes; no built-in registration

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
- **Affine Transform**: Apply an affine transformation matrix (exact 2x2 inversion in 2D; N-D beyond 2D uses a simplified diagonal-only approximation)
- **Shift**: Sub-pixel shift with spline interpolation
- **Rotate**: Array rotation about any axis (output keeps the input shape; the `reshape` option to grow the canvas is accepted but not yet honored)
- **Zoom**: Uniform zooming (N-dimensional, `zoom`); anisotropic per-axis zoom for 2D via `interpolation::zoom_optimized`
- **Spline Filter**: Pre-filter for spline interpolation (`spline_filter`, `spline_filter1d`)

> **Known stub**: `interpolation::geometric_transform` (general transform with a custom
> coordinate-mapping closure) is currently a no-op — it validates its arguments but its body
> just returns a copy of the input, ignoring the supplied mapping function entirely
> (`src/interpolation/transform.rs`). Use `map_coordinates`, `affine_transform`, `shift`,
> `rotate`, or `zoom` instead, which are genuinely implemented.

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

### CNN-Inspired Feature Extraction (no neural-network training required)
- Gabor filter bank, HOG (Histogram of Oriented Gradients), simplified SIFT keypoint detection and descriptors (`deep_features` module)
- Heuristic ML-assisted detection utilities: `SemanticFeatureExtractor`, `ObjectProposalGenerator`, learned edge/keypoint descriptor configs
- Note: this crate does not depend on scirs2-neural; there is no wired-up hook for external deep-learning model integration today

## Installation

```toml
[dependencies]
scirs2-ndimage = "0.6.6"
```

For parallel processing and SIMD:

```toml
[dependencies]
scirs2-ndimage = { version = "0.6.6", features = ["parallel", "simd"] }
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `parallel` | Enable Rayon-based multi-core parallel processing (recommended for arrays >10K elements) |
| `simd` | Enable SIMD vectorization for filters and morphological operations |
| `gpu` | GPU backend abstraction layer (device detection, buffer management). Does not auto-enable a specific backend |
| `cuda` | CUDA backend (`backend/cuda.rs`); kernel JIT compiles CUDA-C to PTX via the pure-Rust `oxicuda-nvrtc` crate (runtime `dlopen` of `libnvrtc`, zero build-time CUDA SDK dependency); experimental |
| `opencl` | OpenCL backend scaffolding; experimental |
| `metal` | Metal backend scaffolding (macOS only); experimental |
| `compression` | Pure-Rust compression for streaming/out-of-core I/O via `oxiarc-deflate`/`oxiarc-zstd`/`oxiarc-lz4` |

Note: the `gpu`/`cuda`/`opencl`/`metal` flags expose device-detection and buffer-management
scaffolding; generic GPU kernel dispatch (`backend::gpu_acceleration_framework`) currently
returns an explicit "not yet implemented" error rather than executing on real hardware, so
CPU code paths remain the primary, production-ready path.

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

    // Morphological dilation (binary morphology operates on bool arrays)
    let binary = image.mapv(|v| v > 0.5).into_dyn();
    let struct_elem = morphology::disk_structure(3.0, None)?;
    let dilated =
        morphology::binary_dilation(&binary, Some(&struct_elem), None, None, None, None, None)?;

    println!("Image processed: {:?}, dilated: {:?}", smoothed.shape(), dilated.shape());
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
        &[5, 5], None, None,
    )?;

    println!("All filters applied");
    Ok(())
}
```

### Morphological Operations

```rust
use scirs2_ndimage::morphology;
use scirs2_core::ndarray::{Array, IxDyn};

fn morphology_example() -> Result<(), Box<dyn std::error::Error>> {
    // Binary morphology operates on `Array<bool, D>`; build the image directly
    // in dynamic-dimension (`IxDyn`) form so it matches the structuring element.
    let binary = Array::from_shape_fn(IxDyn(&[100, 100]), |idx| {
        idx[0] > 30 && idx[0] < 70 && idx[1] > 30 && idx[1] < 70
    });

    let disk = morphology::disk_structure(5.0, None)?;

    // Binary erosion, dilation, and opening take:
    // (input, structure, iterations, mask, border_value, origin, brute_force)
    let eroded = morphology::binary_erosion(&binary, Some(&disk), None, None, None, None, None)?;
    let dilated = morphology::binary_dilation(&binary, Some(&disk), None, None, None, None, None)?;

    // Opening removes small bright regions
    let opened = morphology::binary_opening(&binary, Some(&disk), None, None, None, None, None)?;

    // Distance transform (Euclidean, O(n) algorithm)
    let (distances, _indices) = morphology::distance_transform_edt(&binary, None, true, false)?;

    // Hit-or-miss for pattern detection (structure1, structure2, mask, border_value, origin1, origin2)
    let pattern = Array::from_shape_vec(IxDyn(&[3, 3]), vec![false, true, false, true, true, true, false, true, false])?;
    let hit_miss = morphology::binary_hit_or_miss(&binary, Some(&pattern), None, None, None, None, None)?;

    println!(
        "eroded={:?} dilated={:?} opened={:?} distances_present={} hit_miss={:?}",
        eroded.shape(), dilated.shape(), opened.shape(), distances.is_some(), hit_miss.shape()
    );

    Ok(())
}
```

### Region Measurements

```rust
use scirs2_ndimage::{measurements, moment_invariants, morphology};
use scirs2_core::ndarray::Array2;

fn measurement_example() -> Result<(), Box<dyn std::error::Error>> {
    let image = Array2::<f64>::from_shape_fn((100, 100), |(i, j)| {
        if (i as f64 - 50.0).hypot(j as f64 - 50.0) < 20.0 { 1.0 } else { 0.0 }
    });
    let binary = image.mapv(|v| v > 0.5);

    // Label connected components (label lives in `morphology`, not `measurements`)
    let (labels, num_labels) = morphology::label(&binary, None, None, None)?;
    println!("Found {} labeled region(s)", num_labels);

    // Region properties (2D: area, centroid, perimeter, eccentricity, orientation, ...)
    let props = measurements::regionprops_2d(&image, &labels)?;
    for region in &props {
        println!("Region {}: area={}, centroid={:?}",
            region.label, region.area, region.centroid);
    }

    // Hu moments (rotation-invariant descriptors) - lives in `moment_invariants`, is infallible
    let hu = moment_invariants::hu_moments(&image.view());
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

    // Markers: 0 = unknown region, unique positive integers = seed regions
    let mut markers = Array2::<i32>::zeros((200, 200));
    markers[[50, 50]] = 1;
    markers[[150, 150]] = 2;

    let labels = watershed(&gradient, &markers)?;
    println!("Watershed labels shape: {:?}", labels.shape());

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
use scirs2_ndimage::segmentation_advanced::superpixels_slic;
use scirs2_core::ndarray::Array2;

fn slic_example() -> Result<(), Box<dyn std::error::Error>> {
    let image = Array2::<f64>::from_shape_fn((100, 100), |(i, j)| {
        ((i + j) as f64 / 200.0).sin()
    });

    // (image, n_segments, compactness) -> label array in [0, n_segments)
    let labels = superpixels_slic(&image, 100, 10.0)?;
    println!("Superpixel labels shape: {:?}", labels.shape());
    Ok(())
}
```

### Atlas-Based Segmentation

```rust
use scirs2_ndimage::segmentation::atlas::AtlasSegmentation;
use scirs2_core::ndarray::Array3;

fn atlas_example() -> Result<(), Box<dyn std::error::Error>> {
    // Pre-registered atlas label volumes (registration itself is not performed by this crate)
    let atlas_label_1 = Array3::<u32>::zeros((32, 32, 32));
    let atlas_label_2 = Array3::<u32>::zeros((32, 32, 32));

    // Default configuration fuses via majority voting; STAPLE and joint label
    // fusion are also available through `AtlasSegmentation::with_config`.
    let result = AtlasSegmentation::new().segment(&[atlas_label_1, atlas_label_2], None, None)?;
    println!("Fused label volume shape: {:?}", result.label.shape());
    Ok(())
}
```

## Performance

- **SIMD acceleration**: 2-4x speedup on supported filter and morphology operations
- **Parallel processing**: Linear scaling with CPU cores for large arrays (`parallel` feature)
- **O(n) distance transform**: Felzenszwalb-Huttenlocher separable EDT algorithm
- **Memory-efficient**: Chunked processing for images larger than available RAM
- **N-dimensional**: Consistent API and performance across 1D, 2D, 3D, and higher dimensions

## Test Coverage

Freshly measured via `cargo nextest run -p scirs2-ndimage` (2026-07-15):

| Mode | Result |
|------|--------|
| Default features | 1170 tests run: 1170 passed, 1 skipped, 0 failed |
| `--all-features` | 1199 tests run: 1199 passed, 3 skipped, 0 failed |

## Compatibility with SciPy ndimage

API is modeled after `scipy.ndimage`. Key equivalents:

| SciRS2 | SciPy |
|--------|-------|
| `filters::gaussian_filter()` | `scipy.ndimage.gaussian_filter()` |
| `filters::median_filter()` | `scipy.ndimage.median_filter()` |
| `filters::sobel()` | `scipy.ndimage.sobel()` |
| `morphology::binary_erosion()` | `scipy.ndimage.binary_erosion()` |
| `morphology::distance_transform_edt()` | `scipy.ndimage.distance_transform_edt()` |
| `morphology::label()` | `scipy.ndimage.label()` |
| `measurements::center_of_mass()` | `scipy.ndimage.center_of_mass()` |
| `interpolation::affine_transform()` | `scipy.ndimage.affine_transform()` |
| `interpolation::map_coordinates()` | `scipy.ndimage.map_coordinates()` |
| `interpolation::rotate()` | `scipy.ndimage.rotate()` |
| `interpolation::zoom()` | `scipy.ndimage.zoom()` |

## Known Issues

A handful of functions validate their arguments correctly but have a body that does not perform
the documented computation (no `todo!()`/`unimplemented!()` panic — they just silently return an
unchanged copy of the input or a hardcoded placeholder value), so they will not show up in a
naive stub scan. Confirmed as of 2026-07-15 (see `TODO.md` for detail and source locations):

- `interpolation::geometric_transform` — ignores its custom coordinate-mapping closure entirely
- `morphology::iterate_structure` — does not actually grow/iterate the structuring element
- `morphology::binary_fill_holes` (general N-D) — returns the input unchanged; use `fill_holes_2d` for 2D instead, which is a real, working implementation
- `peak_prominences` / `peak_widths` — return hardcoded placeholder values, not computed ones

Other known limitations:
- `interpolation::rotate`'s `reshape` option is accepted but not honored (output always keeps the input's shape)
- `interpolation::affine_transform` inverts the transformation matrix exactly in 2D; for N > 2 dimensions it falls back to a simplified diagonal-only approximation
- GPU feature flags (`gpu`/`cuda`/`opencl`/`metal`) provide device-detection/buffer scaffolding only; generic kernel dispatch returns an explicit "not yet implemented" error rather than running on hardware
- Bilateral filter performance degrades significantly for large kernel sizes (>21x21)
- Chan-Vese segmentation convergence depends strongly on `mu`/`lambda1`/`lambda2`; automatic initialization is not implemented
- SLIC superpixels (`segmentation_advanced::superpixels_slic`) are 2D grayscale only
- Atlas-based segmentation requires pre-registered atlas label volumes; no built-in registration is performed

## Documentation

Full API reference: [docs.rs/scirs2-ndimage](https://docs.rs/scirs2-ndimage)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
