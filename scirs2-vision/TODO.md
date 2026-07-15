# scirs2-vision TODO

## Status: v0.6.1 (current, 2026-07-15) — reassessed Stable → Partial

Untouched by this release's fix work (no vision-specific changes shipped in 0.6.1); this is a fresh
implementation-status survey. 0 `todo!()`/`unimplemented!()` markers in `src/` — but a targeted sweep
for the *silent*-stub pattern (code that compiles, looks real, and returns a plausible-looking value
without actually computing it — the same pattern `scirs2-ndimage`/`scirs2-integrate` are marked
"Partial" for) turned up three confirmed, publicly-reachable instances, so the status badge is
downgraded from Stable to Partial pending fixes. One additional honest gap (clear `Err`, not a
silent no-op) is also tracked below and does **not** by itself justify "Partial" — it is listed
for completeness.

**Confirmed silent stubs (justify the Partial rating):**
1. `vision_3d::structure_from_motion()` (`src/vision_3d.rs`, re-exported at the crate root) — validates
   its `images.len() >= 2` precondition, then unconditionally returns an **empty** `PointCloud`
   (`Array2::zeros((0, 3))`, `colors: None`). The doc comment's own inline plan ("1. Extract features
   ... 5. Bundle adjustment") is not executed at all — no feature extraction, matching, pose
   estimation, or triangulation happens. Every caller silently gets zero points back, with no error.
2. `segmentation::semantic::legacy::FCN::segment()` / `.forward()` (`src/segmentation/semantic/legacy.rs`)
   — `forward()`'s own comment says "Simulate network output with random predictions" / "For
   demonstration: create a simple pattern"; `segment()` returns an all-zeros `class_map` and
   all-zeros `confidence` array. No neural inference occurs. Mitigating context: this lives in a
   module literally named `legacy`, and `segmentation::semantic::deeplab` (ASPP / atrous convolution
   forward pass) alongside `segmentation::panoptic` appear to be the real, current implementations —
   `panoptic.rs` does not reference `legacy::FCN`. So the crate's advertised "Panoptic Segmentation"
   feature is not affected by this particular stub, but `legacy::FCN` is still `pub use`-exported at
   the `segmentation::semantic` level and callable directly by any user who reaches for it.
3. `performance_benchmark::AdvancedBenchmarkSuite` (`src/performance_benchmark.rs`, re-exported at the
   crate root) — of its 8 internal `benchmark_*` methods backing `run_comprehensive_benchmark()`, 6 are
   explicitly commented "// Placeholder implementation" and return hardcoded/default metrics
   (`benchmark_neuromorphic_processing`, `benchmark_ai_optimization`,
   `benchmark_cross_module_integration`, `benchmark_scalability`, `benchmark_quality_accuracy`,
   `benchmark_resource_efficiency`); even the two that do real timing
   (`benchmark_baseline_performance`, `benchmark_quantum_processing`) mix in hardcoded
   `ComparisonMetrics` fields (e.g. `quantum_advantage: 2.3, // Estimated quantum advantage`). This
   benchmarking harness is not mentioned in `README.md`'s feature list, but it is public API.

**Honest gaps (return a clear `Err`, not a silent no-op — do not by themselves justify "Partial"):**
- `VideoSource::VideoFile` / `VideoSource::Camera` in `streaming_modules/video_io.rs` are not
  implemented (only `VideoSource::ImageSequence` and `VideoSource::Dummy` currently work).

None of the above were implemented as part of this documentation pass (out of scope for a README/TODO
accuracy sweep); they are recorded here so the next implementation pass has a precise starting list.
This sweep (grep for "Placeholder implementation" / "simulate" / "for demonstration" style comments)
was targeted, not exhaustive — `src/` has 1475+ public items across ~140 files, and dozens more
"simulate"/"for demonstration" comments exist that were not individually triaged (some are almost
certainly benign, e.g. doctest example values; others, especially in `feature/neural_features.rs`
and `feature/advanced_tracking.rs`, use "synthetic weights for demonstration" language that would
benefit from a dedicated follow-up stub-check pass).

Fresh test counts (2026-07-15, `cargo nextest run -p scirs2-vision` / `--all-features`): **1,341
passed, 4 skipped** (default features) / **1,345 passed, 4 skipped** (all-features).

## Status: v0.4.3 Released (May 3, 2026)

All v0.4.3 features are complete and production-ready. NeRF / Instant-NGP scene representation,
sparse-LiDAR depth completion, and temporal action segmentation are now stable across the
quality gate (cargo check + clippy clean).

## Status: v0.3.4 Released (March 18, 2026)

## v0.3.3 Completed

### Feature Detection and Description
- Edge detection: Sobel, Canny, Prewitt, Laplacian, LoG
- Corner detection: Harris, FAST, Shi-Tomasi
- Blob detection: DoG, LoG, MSER
- Keypoint descriptors: SIFT, ORB, BRIEF, HOG
- Feature matching: RANSAC, homography estimation
- Hough circle and line transforms
- Sub-pixel corner refinement

### Image Segmentation
- Thresholding: binary, Otsu, adaptive (mean/Gaussian)
- Region-based: SLIC superpixels, watershed, region growing
- Instance segmentation: mask generation, per-instance labeling
- Panoptic segmentation: combined semantic and instance
- GrabCut-style interactive segmentation
- Connected component analysis

### Camera and 3D Vision
- Camera calibration (intrinsic parameters, lens distortion)
- Pinhole, fisheye, and generic camera models
- Stereo depth estimation (disparity maps, depth conversion)
- PnP pose estimation (Perspective-n-Point, 6-DOF)
- SLAM foundations: feature tracking, loop closure

### Point Cloud Processing
- ICP (Iterative Closest Point) registration
- RANSAC-based robust point cloud alignment
- Point cloud loading (PLY, XYZ)

### Video Processing
- Frame extraction from video streams
- Dense optical flow (Farneback, Lucas-Kanade)
- Video stabilization (feature-based, mesh-based)
- Background subtraction and motion detection

### Object Detection
- Sliding window multi-scale detector
- HOG+SVM pedestrian detection pipeline
- Non-Maximum Suppression (NMS)
- Bounding box utilities

### Face Detection
- Viola-Jones foundation (Haar cascade evaluation)
- Multi-scale face candidate generation

### 3D Reconstruction
- Multi-view stereo foundations
- Essential and fundamental matrix estimation
- Triangulation of 3D points from stereo pairs

### Image Enhancement and Preprocessing
- Non-local means, bilateral, guided filtering
- Histogram equalization, CLAHE, gamma correction
- Gaussian blur, median filtering, unsharp masking

### Color Processing
- RGB to/from HSV, LAB, YCbCr, grayscale
- Color quantization: K-means, median cut, octree
- Histogram matching, color transfer

### Geometric Transformations
- Affine, perspective, non-rigid (thin-plate spline, elastic)
- Bilinear, bicubic, Lanczos interpolation
- Feature-based and intensity-based image registration

### Morphological Operations
- Erosion, dilation, opening, closing, morphological gradient
- Top-hat, black-hat transforms

### Style Transfer
- Neural style transfer interface
- Statistical feature matching stylization

### Image Quality
- PSNR, SSIM metrics
- Blind image quality assessment

### Texture Analysis
- GLCM, LBP, Gabor filters, Tamura features

### Medical Imaging
- Frangi vesselness filter
- Bone enhancement, basic segmentation

## v0.4.0 Roadmap

### NeRF (Neural Radiance Fields) — Implemented in v0.4.0
- [x] Implicit neural scene representation
- [x] Volume rendering with ray marching
- [x] Training pipeline for novel view synthesis
- [x] Integration with scirs2-neural for MLP backbone

### 3D Object Detection
- Point cloud object detection (PointNet++ backbone)
- Frustum-based 3D detection from RGB-D
- Lidar object detection foundations
- 3D bounding box estimation and NMS

### Foundation Model Integration
- CLIP-based image and text feature extraction
- SAM (Segment Anything Model) interface wrapper
- DINOv2 feature extraction API
- Prompt-based segmentation pipeline

### Advanced Video Understanding
- Temporal action recognition foundations
- Video object segmentation (VOS)
- Dense video captioning interfaces
- Multi-object tracking (MOT) evaluation metrics

### Advanced Depth Estimation — Implemented in v0.4.0
- [x] Monocular depth estimation (MiDaS-style interface)
- [x] Depth completion from sparse LiDAR
- [x] Confidence-weighted depth fusion

### Camera Network Calibration
- Multi-camera extrinsic calibration
- Rolling shutter camera models
- Omnidirectional camera calibration

## Known Issues

- SIFT descriptor computation is approximate; results may differ slightly from OpenCV reference
- Farneback optical flow requires grayscale input; color flow not yet supported
- ICP convergence is sensitive to initial alignment; RANSAC pre-alignment recommended for large misalignments
- Video stabilization requires sufficient texture in scene; textureless scenes may produce artifacts
- Panoptic segmentation API stabilized in v0.4.1; interface is now current and should not change without a deprecation cycle
- `streaming_modules::video_io::VideoStreamReader::from_source` returns a clear error for `VideoSource::VideoFile` ("Video file reading not yet implemented. Use image sequences instead.") and `VideoSource::Camera` ("Camera reading not yet implemented."). Only `VideoSource::ImageSequence` (a directory of image files) and `VideoSource::Dummy` (synthetic test source) are functional today. Real video-container decoding and camera capture would require a codec/camera integration (e.g. ffmpeg/gstreamer or a platform camera API), which is out of scope for the Pure Rust default build; frame-array-level processing (`video_processing.rs`, optical flow, stabilization, motion detection) works fine once frames are supplied as arrays.
