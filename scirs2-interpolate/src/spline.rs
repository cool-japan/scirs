//! Cubic spline interpolation with comprehensive boundary condition support
//!
//! This module provides production-ready cubic spline interpolation that offers C2 continuity
//! (continuous function, first, and second derivatives) making it ideal for smooth curve fitting
//! and scientific applications requiring high-quality interpolation.
//!
//! ## Overview
//!
//! Cubic splines construct piecewise cubic polynomials that pass through all data points while
//! maintaining smoothness properties. Each segment between adjacent data points is represented
//! by a cubic polynomial of the form:
//!
//! ```text
//! y(x) = a + b(x-xᵢ) + c(x-xᵢ)² + d(x-xᵢ)³
//! ```
//!
//! ## Computational Complexity
//!
//! | Operation | Time Complexity | Space Complexity | Notes |
//! |-----------|----------------|------------------|-------|
//! | Construction | O(n) | O(n) | Tridiagonal solve |
//! | Single Evaluation | O(log n) | O(1) | Binary search + polynomial eval |
//! | Batch Evaluation | O(m log n) | O(1) | m = number of evaluation points |
//! | Derivative | O(log n) | O(1) | Analytical differentiation |
//!
//! ## Boundary Conditions
//!
//! Multiple boundary conditions are supported to handle different physical constraints:
//!
//! - **Natural**: Zero second derivative at endpoints (default)
//! - **Not-a-knot**: Maximum smoothness at second and second-to-last points
//! - **Clamped**: Specified first derivatives at endpoints
//! - **Periodic**: Function and derivatives match at endpoints
//! - **Second derivative**: Specified second derivatives at endpoints
//!
//! ## SciPy Compatibility
//!
//! This implementation maintains API compatibility with SciPy's `CubicSpline` class,
//! allowing for easy migration from Python-based workflows.
//!
//! ## Performance Characteristics
//!
//! - **Numerical stability**: Excellent for well-conditioned data
//! - **Memory efficiency**: O(n) storage for unlimited evaluations
//! - **Real-time capable**: Sub-microsecond evaluation after construction
//! - **SIMD optimized**: Vectorized evaluation for batch operations
//!
//! ## Example Usage
//!
//! ```rust
//! use scirs2_core::ndarray::array;
//! use scirs2_interpolate::spline::{CubicSpline, SplineBoundaryCondition};
//!
//! // Basic usage with natural boundary conditions
//! let x = array![0.0, 1.0, 2.0, 3.0];
//! let y = array![0.0, 1.0, 4.0, 9.0];
//! let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
//!
//! // Evaluate at a point
//! let result = spline.evaluate(1.5).unwrap();
//!
//! // Compute derivatives
//! let derivative = spline.derivative(1.5).unwrap();
//! let second_deriv = spline.derivative_n(1.5, 2).unwrap();
//!
//! // Integration
//! let integral = spline.integrate(0.5, 2.5).unwrap();
//!
//! // Advanced usage with builder pattern
//! let spline = CubicSpline::builder()
//!     .x(x.clone())
//!     .y(y.clone())
//!     .boundary_condition(SplineBoundaryCondition::Clamped(0.0, 6.0))
//!     .build()
//!     .unwrap();
//!
//! // SciPy-compatible interface
//! let spline = scirs2_interpolate::spline::cubic_spline_scipy(
//!     &x.view(),
//!     &y.view(),
//!     "not-a-knot",
//!     None,
//!     false
//! ).unwrap();
//! ```
//!
//! ## Architecture
//!
//! This module has been refactored into focused submodules for better maintainability:
//!
//! - **types**: Core type definitions and boundary conditions
//! - **algorithms**: Computational algorithms for different boundary conditions
//! - **core**: Main data structures (CubicSpline, CubicSplineBuilder)
//! - **evaluation**: Evaluation and derivative computation methods
//! - **integration**: Integration methods and extrapolation support
//! - **traits_impl**: Trait implementations for framework compatibility
//! - **api**: Public SciPy-compatible API functions
//! - **utils**: Utility functions and helpers

// Re-export the spline module contents from the subdirectory
mod spline_impl {
    include!("spline_modules/mod.rs");
}

pub use spline_impl::*;
