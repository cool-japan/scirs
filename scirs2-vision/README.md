# SciRS2 Vision

[![crates.io](https://img.shields.io/crates/v/scirs2-vision.svg)](https://crates.io/crates/scirs2-vision)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-vision)](https://docs.rs/scirs2-vision)
[![Version](https://img.shields.io/badge/version-0.6.3-green)]()
[![Status](https://img.shields.io/badge/status-partial-yellow)]()

**scirs2-vision** is the computer vision crate for the [SciRS2](https://github.com/cool-japan/scirs) scientific computing library. It provides comprehensive tools for feature detection, image segmentation, geometric transformations, stereo vision, 3D reconstruction, object detection, video processing, and camera calibration with APIs familiar to users of OpenCV and scikit-image. Recent waves added Neural Radiance Fields (NeRF) with Instant-NGP hash-grid acceleration, monocular and sparse-LiDAR depth completion, and temporal action segmentation primitives.

## What scirs2-vision Provides

Use scirs2-vision when you need to:

- Detect and match features in images (Harris corners, SIFT, ORB, FAST, HOG)
- Segment images with watershed, SLIC superpixels, instance segmentation, or panoptic segmentation
- Estimate camera pose (PnP), calibrate cameras, or work with stereo depth
- Process video frames: optical flow, video stabilization, frame extraction
- Build a 3D reconstruction pipeline (ICP, RANSAC-based registration)
- Perform object detection and face detection
- Apply style transfer or image quality enhancement

## Features (v0.6.3)

### Feature Detection and Description
- **Edge Detection**: Sobel, Canny, Prewitt, Laplacian, Laplacian of Gaussian (LoG)
- **Corner Detection**: Harris corners, Shi-Tomasi (Good Features to Track), FAST corners
- **Blob Detection**: Difference of Gaussians (DoG), LoG, MSER (Maximally Stable Extremal Regions)
- **Keypoint Descriptors**: SIFT (Scale-Invariant Feature Transform), ORB, BRIEF, HOG (Histogram of Oriented Gradients)
- **Feature Matching**: RANSAC-based robust matching, homography estimation
- **Hough Transforms**: Hough circle transform, Hough line transform
- **Sub-pixel refinement**: Corner refinement to sub-pixel accuracy

### Image Segmentation
- **Thresholding**: Binary, Otsu's automatic, adaptive (mean/Gaussian)
- **Region-Based**: SLIC superpixels, watershed algorithm, region growing
- **Instance Segmentation**: Mask generation, per-instance labeling
- **Panoptic Segmentation**: Combined semantic and instance segmentation
- **Interactive Segmentation**: GrabCut-style foreground/background separation
- **Advanced**: Mean shift clustering, connected component analysis

### Camera and 3D Vision
- **Camera Calibration**: Intrinsic parameter estimation, lens distortion correction
- **Camera Models**: Pinhole, fisheye, and generic camera models
- **Stereo Depth Estimation**: Disparity map computation, depth from stereo pairs
- **PnP Pose Estimation**: Perspective-n-Point solver for 6-DOF pose from 2D-3D correspondences
- **SLAM Foundations**: Feature tracking, map point management, loop closure detection

### Point Cloud Processing
- **ICP (Iterative Closest Point)**: Point cloud registration and alignment
- **RANSAC Registration**: Robust point cloud alignment with outlier rejection
- **Point Cloud I/O**: Load/save PLY, XYZ formats
- **3D Reconstruction Pipeline**: Multi-view stereo foundations; `structure_from_motion()` currently returns an empty point cloud unconditionally (real feature extraction/matching/pose estimation/triangulation is not yet wired in) — see Known Issues in `TODO.md`

### Video Processing
- **Frame Extraction**: Extract frames from image-sequence directories (`VideoStreamReader::from_source(VideoSource::ImageSequence(...))`); direct video-container decoding (mp4/avi) and live camera capture are not yet implemented — see Known Issues in `TODO.md`
- **Dense Optical Flow**: Farneback algorithm, Lucas-Kanade dense flow
- **Video Stabilization**: Feature-based and mesh-based stabilization
- **Motion Detection**: Frame differencing, background subtraction

### Object Detection
- **Detection Framework**: Bounding box prediction, Non-Maximum Suppression (NMS)
- **Sliding Window**: Multi-scale sliding window detector
- **HOG+SVM Pedestrian Detection**: Classical HOG-based detection pipeline

### Face Detection
- **Viola-Jones Foundation**: Haar cascade evaluation
- **Face Detection Pipeline**: Multi-scale face candidate generation

### Image Enhancement and Preprocessing
- **Noise Reduction**: Non-local means denoising, bilateral filtering, guided filtering
- **Enhancement**: Histogram equalization, CLAHE, gamma correction
- **Filtering**: Gaussian blur, median, unsharp masking
- **Super-Resolution**: Single-image super-resolution algorithms

### Color Processing
- **Color Space Conversions**: RGB to/from HSV, LAB, YCbCr, grayscale
- **Channel Operations**: Splitting, merging, per-channel processing
- **Color Quantization**: K-means, median cut, octree quantization
- **Color Normalization**: Histogram matching, color transfer

### Geometric Transformations
- **Affine**: Translation, rotation, scaling, shearing with multiple interpolation modes
- **Perspective**: Homography-based warping with robust estimation
- **Non-Rigid**: Thin-plate spline deformation, elastic transformations
- **Interpolation Methods**: Bilinear, bicubic, Lanczos, edge-preserving

### Image Registration
- **Feature-Based Registration**: Using detected keypoints and RANSAC
- **Intensity-Based Registration**: Normalized cross-correlation, mutual information
- **Supported Transforms**: Rigid, similarity, affine, homography

### Morphological Operations
- Erosion, dilation with customizable structuring elements
- Opening, closing, morphological gradient
- Top-hat, black-hat transforms

### Style Transfer
- Neural style transfer interface
- Artistic stylization using statistical feature matching

### Image Quality
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Blind image quality assessment

### Texture Analysis
- Gray-level co-occurrence matrix (GLCM)
- Local binary patterns (LBP)
- Gabor filters
- Tamura texture features

### Medical Imaging
- DICOM-compatible array handling
- Frangi vesselness filter
- Bone enhancement filters
- Basic segmentation for medical images

## Installation

```toml
[dependencies]
scirs2-vision = "0.6.4"
```

Parallel (Rayon-based) processing is provided unconditionally via `scirs2-core` — there is no separate `parallel` feature to enable. For GPU-accelerated kernels:

```toml
[dependencies]
scirs2-vision = { version = "0.6.4", features = ["wgpu"] }
```

## Quick Start

### Feature Detection

```rust
use scirs2_vision::{harris_corners, sobel_edges, image_to_array};
use scirs2_vision::feature::{canny_simple, fast_corners};
use image::open;

fn feature_example() -> Result<(), Box<dyn std::error::Error>> {
    let img = open("photo.jpg")?;
    let arr = image_to_array(&img)?;

    // Harris corners
    let corners = harris_corners(&img, 3, 0.04, 0.01)?;
    println!("Harris corners: {}", corners.len());

    // FAST corners (faster, less accurate)
    let fast = fast_corners(&arr, 9, 0.05)?;
    println!("FAST corners: {}", fast.len());

    // Canny edge detection
    let edges = canny_simple(&arr, 1.0)?;
    println!("Edge map computed");

    Ok(())
}
```

### Stereo Depth Estimation

```rust
use scirs2_vision::stereo::{compute_disparity, stereo_rectify};
use scirs2_core::error::CoreResult;

fn stereo_example() -> CoreResult<()> {
    // let left = image_to_array(&open("left.jpg")?)?;
    // let right = image_to_array(&open("right.jpg")?)?;

    // Compute disparity map
    // let disparity = compute_disparity(&left, &right, 64, 11)?;

    // Convert disparity to depth (requires calibrated baseline and focal length)
    // let depth = disparity_to_depth(&disparity, focal_length, baseline)?;

    Ok(())
}
```

### Camera Pose Estimation (PnP)

```rust
use scirs2_vision::pose::solve_pnp;
use scirs2_core::error::CoreResult;

fn pose_example() -> CoreResult<()> {
    // 3D world points and corresponding 2D image points
    // let world_points: Vec<[f64; 3]> = vec![...];
    // let image_points: Vec<[f64; 2]> = vec![...];
    // let camera_matrix = ...;

    // Solve for rotation and translation
    // let (rvec, tvec) = solve_pnp(&world_points, &image_points, &camera_matrix, None)?;
    // println!("Rotation: {:?}", rvec);
    // println!("Translation: {:?}", tvec);

    Ok(())
}
```

### ICP Point Cloud Registration

```rust
use scirs2_vision::point_cloud::icp_registration;
use scirs2_core::error::CoreResult;

fn icp_example() -> CoreResult<()> {
    // let source: Vec<[f64; 3]> = vec![...];  // source point cloud
    // let target: Vec<[f64; 3]> = vec![...];  // target point cloud

    // Align source to target
    // let result = icp_registration(&source, &target, 50, 1e-6)?;
    // println!("ICP converged: {}", result.converged);
    // println!("Final RMSE: {:.4}", result.rmse);
    // println!("Transform:\n{:?}", result.transform);

    Ok(())
}
```

### Dense Optical Flow

```rust
use scirs2_vision::optical_flow_dense::farneback_flow;
use scirs2_vision::image_to_array;
use image::open;
use scirs2_core::error::CoreResult;

fn flow_example() -> CoreResult<()> {
    // let frame1 = image_to_array(&open("frame001.jpg")?)?;
    // let frame2 = image_to_array(&open("frame002.jpg")?)?;

    // Compute dense optical flow
    // let (flow_x, flow_y) = farneback_flow(&frame1, &frame2, None)?;
    // println!("Flow computed: {:?}", flow_x.shape());

    Ok(())
}
```

### Instance Segmentation

```rust
use scirs2_vision::instance_segmentation::{InstanceSegmenter, InstanceSegConfig};
use scirs2_vision::image_to_array;
use image::open;
use scirs2_core::error::CoreResult;

fn instance_seg_example() -> CoreResult<()> {
    // let img = image_to_array(&open("scene.jpg")?)?;
    // let config = InstanceSegConfig::default();
    // let segmenter = InstanceSegmenter::new(config)?;
    // let instances = segmenter.segment(&img)?;
    // println!("Found {} instances", instances.len());
    // for inst in &instances {
    //     println!("  Class: {}, Score: {:.3}", inst.class_id, inst.score);
    // }
    Ok(())
}
```

### Image Segmentation

```rust
use scirs2_vision::{
    otsu_threshold, adaptive_threshold, connected_components,
    image_to_array, AdaptiveMethod,
};
use image::open;

fn segmentation_example() -> Result<(), Box<dyn std::error::Error>> {
    let img = open("image.png")?;
    let arr = image_to_array(&img)?;

    // Otsu's automatic threshold
    let (binary, threshold) = otsu_threshold(&arr)?;
    println!("Otsu threshold: {}", threshold);

    // Adaptive thresholding for uneven illumination
    let adaptive = adaptive_threshold(&arr, 11, 0.02, AdaptiveMethod::Gaussian)?;

    // Connected components labeling
    let (labeled, count) = connected_components(&binary)?;
    println!("Found {} objects", count);

    Ok(())
}
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `wgpu` | Real GPU (WebGPU/wgpu) execution for accelerated vision kernels; without it, kernel dispatch transparently falls back to CPU/SIMD |
| `cuda` | Direct oxicuda (COOLJAPAN Pure Rust CUDA) backend |

Rayon-based parallel processing is always compiled in via `scirs2-core` and is not a separate opt-in feature.

## Performance

- Parallel processing via Rayon for CPU-intensive operations
- SIMD-accelerated convolution kernels for filtering
- Memory-efficient streaming for video and large image sequences
- Benchmarked against OpenCV and scikit-image reference implementations

## Testing

1,341 tests passing (default features) / 1,345 tests passing (all-features, `cargo nextest run -p scirs2-vision --all-features`), plus doctests. 0 `todo!()`/`unimplemented!()` markers in `src/` — see Known Issues in `TODO.md` for the silent-stub findings that keep this crate at "Partial" status despite the clean test run.

## Documentation

Full API reference: [docs.rs/scirs2-vision](https://docs.rs/scirs2-vision)

## Dependencies

- `scirs2-core`: Core SciRS2 abstractions (error handling, array types, parallel, random, GPU)
- `scirs2-linalg`: Linear algebra (used by pose/stereo/registration solvers)
- `image`: Rust image loading and format support
- `statrs`, `ordered-float`: Statistical distributions and total-order float wrappers
- `oxicuda-driver` / `oxicuda-memory` / `oxicuda-dnn` (optional, `cuda` feature): COOLJAPAN Pure Rust CUDA backend

Note: `scirs2-ndimage` is not currently a dependency (commented out in `Cargo.toml`; not referenced anywhere in `src/`), and `num-traits`/`num-complex` are not direct dependencies either — numeric traits are used via `scirs2-core` re-exports.

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Authors

COOLJAPAN OU (Team KitaSan)
