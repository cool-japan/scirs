# scirs2-vision TODO

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
