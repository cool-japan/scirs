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
//! * Utility functions (`utils` module):
//!   * Error estimation with cross-validation
//!   * Parameter optimization
//!   * Differentiation and integration of interpolated functions

// Export error types
pub mod error;
pub use error::{InterpolateError, InterpolateResult};

// Interpolation modules
pub mod advanced;
pub mod bezier;
pub mod bivariate;
pub mod boundarymode;
pub mod bspline;
pub mod constrained;
pub mod extrapolation;
pub mod grid;
pub mod hermite;
pub mod interp1d;
pub mod interpnd;
pub mod multiscale;
pub mod nurbs;
pub mod penalized;
pub mod spline;
pub mod tension;
pub mod tensor;
pub mod utils;

// Re-exports for convenience
pub use advanced::akima::{make_akima_spline, AkimaSpline};
pub use advanced::barycentric::{
    make_barycentric_interpolator, BarycentricInterpolator, BarycentricTriangulation,
};
pub use advanced::kriging::{make_kriging_interpolator, CovarianceFunction, KrigingInterpolator};
pub use advanced::enhanced_kriging::{
    AnisotropicCovariance, BayesianKrigingBuilder, BayesianPredictionResult,
    EnhancedKriging, EnhancedKrigingBuilder, TrendFunction,
    make_enhanced_kriging, make_universal_kriging, make_bayesian_kriging,
};
pub use advanced::fast_kriging::{
    FastKriging, FastKrigingBuilder, FastKrigingMethod, FastPredictionResult,
    make_local_kriging, make_fixed_rank_kriging, make_hodlr_kriging, make_tapered_kriging,
};
pub use advanced::rbf::{RBFInterpolator, RBFKernel};
pub use advanced::enhanced_rbf::{
    EnhancedRBFInterpolator, EnhancedRBFKernel, KernelType, KernelWidthStrategy,
    make_auto_rbf, make_accurate_rbf, make_fast_rbf,
};
pub use advanced::thinplate::{ThinPlateSpline, make_thinplate_interpolator};
pub use bezier::{BezierCurve, BezierSurface, bernstein, compute_bernstein_all};
pub use bivariate::{
    BivariateInterpolator, BivariateSpline, RectBivariateSpline, SmoothBivariateSpline,
    SmoothBivariateSplineBuilder,
};
pub use bspline::{
    BSpline, ExtrapolateMode as BSplineExtrapolateMode,
    generate_knots, make_interp_bspline, make_lsq_bspline,
};
pub use grid::{
    create_regular_grid, map_grid_to_points, resample_grid_to_grid, resample_to_grid,
    GridTransformMethod,
};
pub use interp1d::{
    cubic_interpolate, linear_interpolate, nearest_interpolate, pchip_interpolate, Interp1d,
    InterpolationMethod, PchipInterpolator,
    // Monotonic interpolation methods
    MonotonicMethod, MonotonicInterpolator, monotonic_interpolate,
    hyman_interpolate, steffen_interpolate, modified_akima_interpolate,
};
pub use interpnd::{
    make_interp_nd, make_interp_scattered, map_coordinates, ExtrapolateMode, GridType,
    RegularGridInterpolator, ScatteredInterpolator,
};
pub use nurbs::{
    NurbsCurve, NurbsSurface, make_nurbs_circle, make_nurbs_sphere,
};
pub use spline::{make_interp_spline, CubicSpline, BoundaryCondition};
pub use tensor::{
    lagrange_tensor_interpolate, tensor_product_interpolate, LagrangeTensorInterpolator,
    TensorProductInterpolator,
};
pub use penalized::{
    PSpline, PenaltyType, pspline_with_custom_penalty, cross_validate_lambda,
};
pub use constrained::{
    ConstrainedSpline, Constraint, ConstraintType, ConstraintRegion, FittingMethod,
};
pub use tension::{
    TensionSpline, make_tension_spline,
};
pub use hermite::{
    HermiteSpline, DerivativeSpec, make_hermite_spline, make_hermite_spline_with_derivatives,
    make_natural_hermite_spline, make_periodic_hermite_spline, make_quintic_hermite_spline,
};
pub use multiscale::{
    MultiscaleBSpline, RefinementCriterion, make_adaptive_bspline,
};
pub use extrapolation::{
    ExtrapolationMethod, ExtrapolationParameters, Extrapolator,
    make_linear_extrapolator, make_periodic_extrapolator, make_reflection_extrapolator,
    make_cubic_extrapolator, make_exponential_extrapolator,
};
pub use boundarymode::{
    BoundaryMode, BoundaryParameters, BoundaryResult,
    make_zero_gradient_boundary, make_zero_value_boundary, make_periodic_boundary,
    make_symmetric_boundary, make_antisymmetric_boundary, make_linear_gradient_boundary,
};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
