//! Interpolation module
//!
//! This module provides implementations of various interpolation methods.
//! These methods are used to estimate values at arbitrary points based on a
//! set of known data points.
//!
//! ## Overview
//!
//! * 1D interpolation methods (`interp1d` module)
//!   * Linear, nearest, cubic interpolation
//!   * PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) - shape-preserving interpolation
//! * Spline interpolation (`spline` module)
//! * B-spline basis functions and interpolation (`bspline` module):
//!   * `BSpline` - B-spline basis functions and interpolation
//!   * Support for derivatives, antiderivatives, and integration
//!   * Knot generation with different styles (uniform, average, clamped)
//!   * Least-squares fitting with B-splines
//! * NURBS curves and surfaces (`nurbs` module):
//!   * `NurbsCurve` - Non-Uniform Rational B-Spline curves
//!   * `NurbsSurface` - Non-Uniform Rational B-Spline surfaces
//!   * Utility functions for creating common NURBS shapes (circles, spheres)
//!   * Support for control point weights and derivatives
//! * Bezier curves and surfaces (`bezier` module):
//!   * `BezierCurve` - Parametric curves with control points
//!   * `BezierSurface` - Parametric surfaces with control points
//!   * Bernstein polynomial basis functions
//!   * Curve/surface evaluation, derivatives, and curve splitting
//! * Bivariate splines (`bivariate` module):
//!   * `BivariateSpline` - Base class for bivariate splines
//!   * `SmoothBivariateSpline` - Smooth bivariate spline approximation
//!   * `RectBivariateSpline` - Bivariate spline approximation over a rectangular mesh
//! * Multivariate interpolation (`interpnd` module)
//! * Advanced interpolation methods (`advanced` module):
//!   * Akima spline interpolation - robust to outliers
//!   * Radial Basis Function (RBF) interpolation - for scattered data
//!   * Enhanced RBF interpolation - with automatic parameter selection and multi-scale capabilities
//!   * Kriging (Gaussian process regression) - with uncertainty quantification
//!   * Barycentric interpolation - stable polynomial interpolation
//!   * Thin-plate splines - special case of RBF for smooth interpolation
//! * GPU-accelerated interpolation (`gpu_accelerated` module):
//!   * GPU-accelerated RBF interpolation for large datasets
//!   * Batch spline evaluation using GPU parallelism
//!   * Mixed CPU/GPU workloads for optimal performance
//!   * Multi-GPU support and memory-efficient operations
//! * Grid transformation and resampling (`grid` module):
//!   * Resample scattered data onto regular grids
//!   * Convert between grids of different resolutions
//!   * Map grid data to arbitrary points
//! * Tensor product interpolation (`tensor` module):
//!   * Efficient high-dimensional interpolation on structured grids
//!   * Higher-order interpolation using Lagrange polynomials
//! * Penalized splines (`penalized` module):
//!   * P-splines with various penalty types (ridge, derivatives)
//!   * Cross-validation for optimal smoothing parameter selection
//! * Constrained splines (`constrained` module):
//!   * Splines with explicit monotonicity and convexity constraints
//!   * Support for regional constraints and multiple constraint types
//!   * Constraint-preserving interpolation and least squares fitting
//! * Tension splines (`tension` module):
//!   * Splines with adjustable tension parameters
//!   * Control over the "tightness" of interpolation curves
//!   * Smooth transition between cubic splines and linear interpolation
//! * Hermite splines (`hermite` module):
//!   * Cubic and quintic Hermite interpolation with derivative constraints
//!   * Direct control over function values and derivatives at data points
//!   * Multiple derivative specification options (automatic, fixed, zero, periodic)
//!   * C1 and C2 continuity options (continuous first/second derivatives)
//! * Multiscale B-splines (`multiscale` module):
//!   * Hierarchical B-splines with adaptive refinement
//!   * Different refinement criteria (error-based, curvature-based, combined)
//!   * Multi-level representation with control over precision-complexity tradeoffs
//!   * Efficient representation of functions with varying detail across the domain
//! * Active learning for adaptive sampling (`adaptive_learning` module):
//!   * Intelligent sampling strategies for reducing function evaluations
//!   * Uncertainty-based, error-based, and gradient-based sampling
//!   * Exploration vs exploitation balance for optimal data collection
//!   * Expected improvement and combined sampling strategies
//!   * Statistics tracking and convergence monitoring
//! * Physics-informed interpolation (`physics_informed` module):
//!   * Interpolation methods that respect physical constraints and conservation laws
//!   * Mass, energy, and momentum conservation enforcement
//!   * Positivity-preserving and monotonicity-preserving interpolation
//!   * Boundary condition enforcement and smoothness constraints
//!   * Multi-physics coupling and constraint satisfaction monitoring
//! * Gaussian process regression with adaptive kernels (`adaptive_gp` module):
//!   * Automatic kernel selection and hyperparameter optimization
//!   * Multiple kernel types (RBF, Matérn, Periodic, Linear, Polynomial)
//!   * Uncertainty quantification and probabilistic interpolation
//!   * Model comparison and Bayesian model selection
//!   * Sparse GP methods for scalability to large datasets
//! * Neural network enhanced interpolation (`neural_enhanced` module):
//!   * Hybrid interpolation combining traditional methods with neural networks
//!   * Residual learning to improve base interpolation accuracy
//!   * Multiple neural architectures (MLP, Bayesian, PINN, Transformer)
//!   * Adaptive training with early stopping and regularization
//!   * Uncertainty-aware neural interpolation with Bayesian networks
//! * Advanced extrapolation methods (`extrapolation` module):
//!   * Configurable extrapolation beyond domain boundaries
//!   * Multiple methods: constant, linear, polynomial, periodic, reflected
//!   * Physics-informed extrapolation (exponential, power law)
//!   * Customizable for specific domain knowledge
//! * Enhanced boundary handling (`boundarymode` module):
//!   * Physical boundary conditions for PDEs (Dirichlet, Neumann)
//!   * Domain extension via symmetry, periodicity, and custom mappings
//!   * Separate control of upper and lower boundary behavior
//!   * Support for mixed boundary conditions
//! * Optimization-based parameter fitting (`optimization` module):
//!   * Cross-validation based model selection (K-fold, leave-one-out)
//!   * Regularization parameter optimization with grid search
//!   * Hyperparameter tuning for RBF kernels and spline smoothing
//!   * Model comparison and validation metrics (MSE, MAE, R²)
//!   * Bayesian optimization for efficient parameter search
//! * Time series specific interpolation (`timeseries` module):
//!   * Temporal pattern recognition (trends, seasonality, irregularities)
//!   * Missing data handling with forward/backward fill and interpolation
//!   * Seasonal decomposition and trend-aware interpolation
//!   * Outlier detection and robust temporal smoothing
//!   * Uncertainty estimation for time-dependent predictions
//!   * Holiday and event-aware interpolation strategies
//! * Geospatial interpolation methods (`geospatial` module):
//!   * Geographic coordinate handling (latitude/longitude, projections)
//!   * Spherical interpolation with great circle distances
//!   * Spatial autocorrelation and kriging for Earth science data
//!   * Multi-scale interpolation from local to global coverage
//!   * Elevation-aware 3D interpolation and topographic considerations
//!   * Boundary handling for coastal and land/water transitions
//! * Utility functions (`utils` module):
//!   * Error estimation with cross-validation
//!   * Parameter optimization
//!   * Differentiation and integration of interpolated functions

// Export error types
pub mod error;
pub use error::{InterpolateError, InterpolateResult};

// Interpolation modules
pub mod adaptive_gp;
pub mod adaptive_learning;
pub mod adaptive_singularity;
pub mod advanced;
pub mod bezier;
pub mod bivariate;
pub mod boundarymode;
pub mod bspline;
pub mod cache;
pub mod cache_aware;
pub mod constrained;
pub mod extrapolation;
pub mod fast_bspline;
pub mod geospatial;
pub mod gpu_accelerated;
pub mod grid;
pub mod griddata;
pub mod hermite;
pub mod high_dimensional;
pub mod interp1d;
pub mod interp2d;
pub mod interpnd;
pub mod local;
pub mod multiscale;
pub mod neural_enhanced;
pub mod numerical_stability;
pub mod nurbs;
pub mod optimization;
pub mod parallel;
pub mod penalized;
pub mod physics_informed;
pub mod scattered_optimized;
pub mod simd_optimized;
pub mod structured_matrix;

// SIMD-optimized interpolation methods (optional)
#[cfg(feature = "simd")]
pub mod simd_bspline;
pub mod smoothing;
pub mod sparse_grid;
pub mod spatial;
pub mod spline;
pub mod tension;
pub mod tensor;
pub mod timeseries;
pub mod utils;
pub mod voronoi;

// Re-exports for convenience
pub use adaptive_gp::{
    make_adaptive_gp, AdaptiveGPConfig, AdaptiveGaussianProcess, GPStats, KernelHyperparameters,
    KernelModel, KernelType as GPKernelType,
};
pub use adaptive_learning::{
    make_active_learner, ActiveLearner, ActiveLearningConfig, LearningStats, SamplingCandidate,
    SamplingStrategy,
};
pub use adaptive_singularity::{
    apply_singularity_handling, SingularityDetector, SingularityDetectorConfig, SingularityInfo,
    SingularityType, TreatmentStrategy,
};
pub use advanced::akima::{make_akima_spline, AkimaSpline};
pub use advanced::barycentric::{
    make_barycentric_interpolator, BarycentricInterpolator, BarycentricTriangulation,
};
pub use advanced::enhanced_kriging::{
    make_bayesian_kriging, make_enhanced_kriging, make_universal_kriging, AnisotropicCovariance,
    BayesianKrigingBuilder, BayesianPredictionResult, EnhancedKriging, EnhancedKrigingBuilder,
    TrendFunction,
};
pub use advanced::enhanced_rbf::{
    make_accurate_rbf, make_auto_rbf, make_fast_rbf, EnhancedRBFInterpolator, EnhancedRBFKernel,
    KernelType, KernelWidthStrategy,
};
pub use advanced::fast_kriging::{
    make_fixed_rank_kriging, make_hodlr_kriging, make_local_kriging, make_tapered_kriging,
    FastKriging, FastKrigingBuilder, FastKrigingMethod, FastPredictionResult,
};
pub use advanced::kriging::{make_kriging_interpolator, CovarianceFunction, KrigingInterpolator};
pub use advanced::rbf::{RBFInterpolator, RBFKernel};
pub use advanced::thinplate::{make_thinplate_interpolator, ThinPlateSpline};
pub use bezier::{bernstein, compute_bernstein_all, BezierCurve, BezierSurface};
pub use bivariate::{
    BivariateInterpolator, BivariateSpline, RectBivariateSpline, SmoothBivariateSpline,
    SmoothBivariateSplineBuilder,
};
pub use boundarymode::{
    make_antisymmetric_boundary, make_linear_gradient_boundary, make_periodic_boundary,
    make_symmetric_boundary, make_zero_gradient_boundary, make_zero_value_boundary, BoundaryMode,
    BoundaryParameters, BoundaryResult,
};
pub use bspline::{
    generate_knots, make_interp_bspline, make_lsq_bspline, BSpline, BSplineWorkspace,
    ExtrapolateMode as BSplineExtrapolateMode, WorkspaceMemoryStats,
};
pub use cache::{
    make_cached_bspline, make_cached_bspline_with_config, BSplineCache, CacheConfig, CacheStats,
    CachedBSpline, DistanceMatrixCache,
};
pub use cache_aware::{
    make_cache_aware_rbf, CacheAwareBSpline, CacheAwareRBF, CacheOptimizedConfig,
    CacheOptimizedStats, CacheSizes,
};
pub use constrained::{
    ConstrainedSpline, Constraint, ConstraintRegion, ConstraintType, FittingMethod,
};
pub use extrapolation::{
    make_cubic_extrapolator, make_exponential_extrapolator, make_linear_extrapolator,
    make_periodic_extrapolator, make_reflection_extrapolator, ExtrapolationMethod,
    ExtrapolationParameters, Extrapolator,
};
pub use fast_bspline::{
    make_cached_fast_bspline_evaluator, make_fast_bspline_evaluator, FastBSplineEvaluator,
    TensorProductFastEvaluator,
};
pub use geospatial::{
    make_climate_interpolator, make_elevation_interpolator, CoordinateSystem, GeospatialConfig,
    GeospatialInterpolator, GeospatialResult, InterpolationModel, SpatialStats,
};
pub use gpu_accelerated::{
    get_gpu_device_info, is_gpu_acceleration_available, make_gpu_rbf_interpolator,
    GpuBatchSplineEvaluator, GpuConfig, GpuDeviceInfo, GpuRBFInterpolator, GpuRBFKernel, GpuStats,
};
pub use grid::{
    create_regular_grid, map_grid_to_points, resample_grid_to_grid, resample_to_grid,
    GridTransformMethod,
};
pub use hermite::{
    make_hermite_spline, make_hermite_spline_with_derivatives, make_natural_hermite_spline,
    make_periodic_hermite_spline, make_quintic_hermite_spline, DerivativeSpec, HermiteSpline,
};
pub use high_dimensional::{
    make_knn_interpolator, make_local_rbf_interpolator, make_pca_interpolator,
    DimensionReductionMethod, HighDimensionalInterpolator, HighDimensionalInterpolatorBuilder,
    InterpolatorStats, LocalMethod, LocalRBFType, SparseStrategy, SpatialIndexType,
};
pub use interp1d::{
    cubic_interpolate,
    hyman_interpolate,
    linear_interpolate,
    modified_akima_interpolate,
    monotonic_interpolate,
    nearest_interpolate,
    pchip_interpolate,
    steffen_interpolate,
    Interp1d,
    InterpolationMethod,
    MonotonicInterpolator,
    // Monotonic interpolation methods
    MonotonicMethod,
    PchipInterpolator,
};
pub use interpnd::{
    make_interp_nd, make_interp_scattered, map_coordinates, ExtrapolateMode, GridType,
    RegularGridInterpolator, ScatteredInterpolator,
};
pub use local::mls::{MovingLeastSquares, PolynomialBasis, WeightFunction};
pub use local::polynomial::{
    make_loess, make_robust_loess, LocalPolynomialConfig, LocalPolynomialRegression,
    RegressionResult,
};
pub use multiscale::{make_adaptive_bspline, MultiscaleBSpline, RefinementCriterion};
pub use neural_enhanced::{
    make_neural_enhanced_interpolator, ActivationType, EnhancementStrategy, NeuralArchitecture,
    NeuralEnhancedInterpolator, NeuralTrainingConfig, TrainingStats,
};
pub use numerical_stability::{
    apply_tikhonov_regularization, assess_matrix_condition, check_safe_division, machine_epsilon,
    safe_reciprocal, solve_with_stability_monitoring, ConditionReport, StabilityDiagnostics,
    StabilityLevel,
};
pub use nurbs::{make_nurbs_circle, make_nurbs_sphere, NurbsCurve, NurbsSurface};
pub use optimization::{
    grid_search, make_cross_validator, CrossValidationResult, CrossValidationStrategy,
    CrossValidator, ModelSelector, OptimizationConfig, OptimizationResult, ValidationMetric,
};
pub use parallel::{
    make_parallel_loess, make_parallel_mls, make_parallel_robust_loess, ParallelConfig,
    ParallelEvaluate, ParallelLocalPolynomialRegression, ParallelMovingLeastSquares,
};
pub use penalized::{cross_validate_lambda, pspline_with_custom_penalty, PSpline, PenaltyType};
pub use physics_informed::{
    make_mass_conserving_interpolator, make_monotonic_physics_interpolator,
    make_smooth_physics_interpolator, ConservationLaw, PhysicalConstraint, PhysicsInformedConfig,
    PhysicsInformedInterpolator, PhysicsInformedResult,
};
pub use scattered_optimized::{
    make_optimized_scattered_interpolator, OptimizedScatteredInterpolator, OptimizedScatteredStats,
    ScatteredConfig,
};
#[cfg(feature = "simd")]
pub use simd_bspline::SimdBSplineEvaluator;
pub use simd_optimized::{
    get_simd_config, is_simd_available, simd_bspline_basis_functions, simd_bspline_batch_evaluate,
    simd_distance_matrix, simd_rbf_evaluate, RBFKernel as SimdRBFKernel, SimdConfig,
};
pub use smoothing::{
    make_adaptive_smoothing_spline, make_error_based_smoothing_spline,
    make_optimized_smoothing_spline, KnotStrategy, VariableKnotSpline,
};
pub use sparse_grid::{
    make_adaptive_sparse_grid_interpolator, make_sparse_grid_from_data,
    make_sparse_grid_interpolator, GridPoint, MultiIndex, SparseGridBuilder,
    SparseGridInterpolator, SparseGridStats,
};
pub use spatial::balltree::BallTree;
pub use spatial::kdtree::KdTree;
pub use spline::{make_interp_spline, BoundaryCondition, CubicSpline};
pub use structured_matrix::{
    create_bspline_band_matrix, solve_band_system, solve_sparse_system,
    solve_structured_least_squares, BandMatrix, CSRMatrix,
};
pub use tension::{make_tension_spline, TensionSpline};
pub use tensor::{
    lagrange_tensor_interpolate, tensor_product_interpolate, LagrangeTensorInterpolator,
    TensorProductInterpolator,
};
pub use timeseries::{
    backward_fill, forward_fill, make_daily_interpolator, make_weekly_interpolator,
    MissingDataStrategy, SeasonalityType, TemporalPattern, TemporalStats, TimeSeriesConfig,
    TimeSeriesInterpolator, TimeSeriesResult,
};
pub use voronoi::{
    constant_value_extrapolation, inverse_distance_extrapolation, linear_gradient_extrapolation,
    make_laplace_interpolator, make_natural_neighbor_interpolator,
    make_parallel_laplace_interpolator, make_parallel_natural_neighbor_interpolator,
    make_parallel_sibson_interpolator, make_sibson_interpolator, nearest_neighbor_extrapolation,
    Extrapolation, ExtrapolationMethod as VoronoiExtrapolationMethod, ExtrapolationParams,
    GradientEstimation, InterpolateWithGradient, InterpolateWithGradientResult,
    InterpolationMethod as VoronoiInterpolationMethod, NaturalNeighborInterpolator,
    ParallelNaturalNeighborInterpolator,
};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
