# scirs2-ndimage TODO

## Status: v0.4.3 Released (May 3, 2026)

All v0.4.3 features are complete and production-ready. Wave 3 stub-check resolved the
`fusion_processing` zero-cascade issue via consciousness-amplitude superposition initialisation,
restoring all multi-channel fusion paths.

## Status: v0.3.4 Released (March 18, 2026)

## v0.3.3 Completed

### Image Filtering
- Gaussian filter, gaussian_filter1d, gaussian_gradient_magnitude, gaussian_laplace
- Median filter (N-dimensional)
- Rank filters: minimum, maximum, percentile (full n-dimensional support)
- Edge detection: Sobel, Prewitt, Laplacian, Scharr, Roberts
- Bilateral filter (edge-preserving)
- Uniform (box) filter
- Generic filter with custom functions
- N-dimensional convolution (convolve, convolve1d)
- All boundary modes: reflect, nearest, wrap, mirror, constant
- Fourier filters: Gaussian, uniform, ellipsoid, shift

### Morphological Operations
- Binary erosion, dilation, opening, closing, propagation, hole filling
- Binary hit-or-miss transform
- Grayscale erosion, dilation, opening, closing
- White/black top-hat transforms, morphological gradient, Laplace
- Distance transforms: Euclidean (O(n) Felzenszwalb-Huttenlocher), city-block, chessboard
- Connected component labeling, find objects, remove small objects
- Structuring element generators: disk, square, diamond
- Skeletonization (topological thinning)

### Image Measurements
- Region statistics per label: sum, mean, variance, std, min, max
- Raw, central, normalized, and Hu moments
- Region properties: area, perimeter, centroid, bounding box, eccentricity, orientation
- Center of mass (N-dimensional)
- Local and global extrema
- Per-label histograms
- Inertia tensor

### Image Segmentation
- Thresholding: binary, Otsu, adaptive mean/Gaussian
- Standard watershed and marker-controlled watershed
- Active contours (snakes) with gradient vector flow
- Chan-Vese level set segmentation (single and multi-phase)
- Graph cut segmentation with interactive refinement (max-flow/min-cut)
- SLIC superpixels (2D and 3D)
- Atlas-based segmentation (label transfer via registration)

### Feature Detection
- Canny edge detector
- Harris corners, FAST corners
- SIFT descriptor computation
- HOG (Histogram of Oriented Gradients)
- Template matching (NCC, zero-mean NCC)
- Gabor filter bank (multi-scale, multi-orientation)
- Shape analysis (moments-based descriptors, matching)

### Geometric Interpolation
- map_coordinates (0th-5th order splines)
- affine_transform (N-dimensional)
- geometric_transform (general, custom coordinate mapping)
- shift, rotate, zoom
- spline_filter, spline_filter1d

### 3D Volume Analysis
- 3D morphology (all binary and grayscale operations)
- 3D Gaussian, Sobel, Laplacian, bilateral filters
- 3D region properties (surface area, Euler characteristic)
- Slice-by-slice processing for 3D stacks

### Medical Image Processing
- Frangi vesselness filter (multi-scale)
- Bone enhancement for CT
- Lung nodule candidate generation

### Hyperspectral Image Analysis
- Per-band filtering and morphology
- NDVI, NDWI, and custom spectral index computation
- Linear spectral unmixing
- Cloud and shadow masking
- Pan-sharpening (Brovey, IHS, PCA)

### Texture Analysis
- GLCM (gray-level co-occurrence matrix, 2D and 3D)
- Texture features from GLCM: contrast, correlation, energy, homogeneity
- Local binary patterns (LBP)
- Gabor feature maps

### Co-occurrence Matrices
- Multi-direction GLCM computation
- Haralick texture features

### Deep Feature Extraction Interface
- Hooks for forwarding arrays through external feature extractors
- Integration interface with scirs2-neural

### Performance
- SIMD-accelerated morphology and edge detection (via scirs2-core SIMD)
- Rayon parallel processing for large arrays (auto-switch at 10K elements)
- Chunked processing for images larger than RAM
- O(n) EDT via Felzenszwalb-Huttenlocher separable algorithm

## v0.4.0 Roadmap

### GPU-Accelerated Convolutions
- GPU convolution for large 2D and 3D arrays (via scirs2-core GPU backend)
- GPU Gaussian filter, median filter, morphological operations
- Automatic CPU/GPU dispatch based on array size and GPU availability
- Memory-efficient tiled GPU convolution for images larger than VRAM

### 4D (Temporal 3D) Imaging
- 4D array support for time-lapse volumetric data
- 4D optical flow (spatiotemporal motion estimation)
- 4D morphological operations (spatiotemporal erosion/dilation)
- 4D region tracking (object tracking across time)

### Deep Segmentation Models
- UNet-based segmentation integration (via scirs2-neural)
- nnU-Net-style automatic configuration for medical segmentation
- Foundation model interface (SAM-style prompt-based segmentation)
- Transfer learning support for domain-specific segmentation

### Advanced Texture and Material Analysis
- Run-length matrix (RLM) features
- Gray-level size zone matrix (GLSZM)
- Neighborhood gray-tone difference matrix (NGTDM)
- Laws' texture energy measures

### Enhanced Segmentation
- Geodesic active contours (level set with external image energy)
- Topology-preserving segmentation
- 3D watershed with topological constraints
- Conditional Random Fields (CRF) for label smoothing post-processing

### Advanced Measurement
- Graph-based region adjacency
- Reachability and overlap between labeled regions
- Multi-label volumetric statistics
- Radiomics feature extraction (full PyRadiomics-equivalent set)

## Known Issues

- Bilateral filter performance degrades significantly for large kernel sizes (>21x21); use a fast approximation for large kernels
- 3D watershed can be memory-intensive for large volumes; use chunked processing via the `chunked` module
- Chan-Vese segmentation convergence depends strongly on the `mu`, `lambda1`, `lambda2` parameters; automatic initialization is not yet implemented
- SLIC superpixels in 3D may not maintain exactly the requested number of superpixels due to boundary effects
- Atlas-based segmentation requires pre-registered atlas; no built-in registration is performed
