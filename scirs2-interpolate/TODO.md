# scirs2-interpolate TODO

This module provides interpolation functionality similar to SciPy's interpolate module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] 1D interpolation methods
  - [x] Linear interpolation
  - [x] Nearest neighbor interpolation
  - [x] Cubic interpolation
  - [x] Spline interpolation
  - [x] PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
- [x] Multi-dimensional interpolation
  - [x] Regular grid interpolation (tensor product)
  - [x] Scattered data interpolation
- [x] Advanced interpolation methods
  - [x] Akima splines
  - [x] Barycentric interpolation
  - [x] Radial basis functions (RBF)
  - [x] Kriging (Gaussian process regression)
- [x] Utility functions
  - [x] Grid creation and manipulation
  - [x] Derivative and integration of interpolants
- [x] Fixed Clippy warnings for iterator_cloned_collect
- [x] Fixed tests
  - [x] Update barycentric_interpolator_quadratic test
  - [x] Fix make_barycentric_interpolator test
  - [x] Fix kriging_interpolator_prediction test
  - [x] Address rbf_interpolator_2d test
  - [x] Comprehensive fast kriging module tests (15+ test functions added)

## Completing SciPy Parity

- [x] FITPACK replacements with modular design
  - [x] Implement B-spline basis functions with more flexible interface
  - [x] Provide direct control over knot placement
  - [x] Support for various boundary conditions (not-a-knot, natural, clamped, periodic)
  - [x] Internal validation for knot sequences and parameters
- [x] Spline fitting enhancements
  - [x] Variable knot smoothing splines
  - [x] User-selectable smoothing criteria (P-splines penalty, etc.)
  - [x] Advanced boundary condition specification
  - [x] Weight-based fitting for uncertain data
- [x] Multi-dimensional interpolators
  - [x] Complete bivariate spline implementation
  - [x] Improve n-dimensional thin-plate splines
  - [x] Better tensor-product spline interpolation
  - [x] Voronoi-based interpolation methods

## Interpolation Algorithm Extensions

- [ ] Add more interpolation methods
  - [x] PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
  - [x] Bivariate splines for irregularly spaced data
  - [x] Thin-plate splines with full radial basis support
  - [x] Bezier curves and surfaces with control point manipulation
  - [x] NURBS (Non-Uniform Rational B-Splines) implementation
  - [x] Monotonic interpolation methods beyond PCHIP
- [x] Specialized spline techniques
  - [x] Penalized splines (P-splines) with various penalty terms
  - [x] Constrained splines (monotonicity, convexity)
  - [x] Tension splines with adjustable tension parameters
  - [x] Hermite splines with derivative constraints
  - [x] Multiscale B-splines for adaptive refinement

## Advanced Features

- [x] Improve extrapolation methods and boundary handling
  - [x] Configurable extrapolation modes (constant, linear, spline-based)
  - [x] Specialized boundary conditions for physical constraints
  - [x] Domain extension methods that preserve continuity
  - [x] Warning systems for extrapolation reliability
- [x] Enhanced RBF interpolation
  - [x] Expanded kernel function options
  - [x] Automatic kernel parameter selection
  - [x] Multi-scale RBF methods for complex surfaces
  - [x] Compactly supported RBF kernels for sparse linear systems
- [x] Kriging improvements
  - [x] Support for anisotropic variogram models
  - [x] Universal kriging with trend functions
  - [x] Bayesian kriging with uncertainty quantification
  - [x] Fast approximate kriging for large datasets
    - [x] Local kriging (O(k³) per prediction)
    - [x] Fixed rank approximation (low-rank covariance)
    - [x] Tapering methods (sparse matrices)
    - [x] HODLR approximation (hierarchical matrices)
    - [x] Comprehensive test suite (15+ test functions)
    - [x] Performance benchmarking tools
    - [x] Automatic method selection based on dataset size
- [x] Local interpolation techniques
  - [x] Moving least squares interpolation
  - [x] Local polynomial regression models
  - [x] Adaptive bandwidth selection
  - [x] Windowed radial basis functions

## Performance Improvements

- [x] Improve performance for large datasets
  - [x] Optimized data structures for nearest neighbor search (kd-trees, ball trees)
  - [x] Parallelization of computationally intensive operations
  - [x] Add standard `workers` parameter to parallelizable functions
  - [x] Cache-aware algorithm implementations
- [x] Enhance multi-dimensional interpolation
  - [x] Better support for high-dimensional data
  - [x] More efficient scattered data interpolation
  - [x] Dimension reduction techniques for high-dimensional spaces
  - [x] Sparse grid methods for addressing the curse of dimensionality
- [x] Algorithmic optimizations
  - [x] Fast evaluation of B-splines using recursive algorithms
  - [x] Optimized basis function evaluations
  - [x] Structured coefficient matrix operations
  - [x] Memory-efficient representations for large problems

## GPU and SIMD Acceleration

- [x] GPU-accelerated implementations for large datasets
  - [x] RBF interpolation on GPU for many evaluation points
  - [x] Batch evaluation of spline functions
  - [x] Parallelized scattered data interpolation
  - [x] Mixed CPU/GPU workloads for optimal performance
- [x] SIMD optimization for core functions
  - [x] Vectorized distance calculations for spatial search
  - [x] SIMD RBF kernel evaluations (Gaussian, Multiquadric, etc.)
  - [x] Platform-specific optimizations (AVX2, SSE2)
  - [x] SIMD-friendly data layouts for evaluation
  - [x] Vectorized B-spline basis function evaluation (completed)

## Adaptive Methods

- [x] Adaptive resolution techniques
  - [x] Error-based refinement of interpolation domains
  - [x] Hierarchical interpolation methods
  - [x] Multi-level approaches for complex functions
  - [x] Automatic singularity detection and handling
- [x] Learning-based adaptive methods
  - [x] Gaussian process regression with adaptive kernels
  - [x] Neural network enhanced interpolation
  - [x] Active learning approaches for sampling critical regions
  - [x] Hybrid physics-informed interpolation models

## Documentation and Examples

- [x] Add more examples and documentation
  - [x] Tutorial for common interpolation tasks (fast kriging example completed)
  - [ ] Visual examples for different methods
  - [ ] Decision tree for selecting appropriate interpolation methods
  - [ ] Parameter selection guidelines
  - [ ] Performance comparison with SciPy
- [x] Application-specific examples
  - [x] Time series interpolation (completed)
  - [ ] Image and signal processing
  - [x] Geospatial data interpolation (completed)
  - [x] Scientific data reconstruction (fast kriging for large datasets)
  - [ ] Financial data smoothing

## Integration with Other Modules

- [x] Integration with optimization for parameter fitting
  - [x] Cross-validation based model selection
  - [x] Regularization parameter optimization
  - [x] Objective function definitions for common use cases
- [x] Support for specialized domain-specific interpolation
  - [x] Geospatial interpolation methods
  - [x] Time series specific interpolators
  - [ ] Signal processing focused methods
  - [ ] Scientific data reconstruction techniques
- [ ] Integration with differentiation and integration modules
  - [ ] Smooth interfaces for spline differentiation
  - [ ] Accurate integration of interpolated functions
  - [ ] Error bounds for differentiated interpolants
  - [ ] Specialized methods for physical systems

## Long-term Goals

- [ ] Performance comparable to or better than SciPy's interpolate
- [ ] Complete feature parity with SciPy's interpolate
- [ ] Comprehensive benchmarking suite
- [ ] Self-tuning interpolation that adapts to data characteristics
- [ ] Streaming and online interpolation methods
- [ ] Distributed interpolation for extremely large datasets