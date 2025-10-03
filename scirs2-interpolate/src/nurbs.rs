//! NURBS (Non-Uniform Rational B-Splines) implementation
//!
//! This module provides functionality for NURBS curves and surfaces, which
//! are a generalization of B-splines and Bezier curves that can exactly represent
//! conic sections like circles and ellipses, as well as other shapes.
//!
//! NURBS use rational basis functions, which are B-spline basis functions
//! with associated weights. This allows for greater flexibility in representing
//! complex shapes while maintaining the favorable properties of B-splines.
//!
//! ## Architecture
//!
//! This module has been refactored into focused submodules for better maintainability:
//!
//! - **types**: Core type definitions and validation functions
//! - **core**: Basic constructors and accessor methods
//! - **curve**: NURBS curve evaluation and manipulation methods
//! - **surface**: NURBS surface evaluation and manipulation methods
//! - **api**: Public API functions for creating common shapes
//!
//! ## Examples
//!
//! ### Basic NURBS Curve
//!
//! ```rust
//! use scirs2_core::ndarray::array;
//! use scirs2_interpolate::nurbs::{NurbsCurve};
//! use scirs2_interpolate::bspline::ExtrapolateMode;
//!
//! let control_points = array![
//!     [0.0, 0.0],
//!     [1.0, 1.0],
//!     [2.0, 0.0]
//! ];
//! let weights = array![1.0, 1.0, 1.0];
//! let knots = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//! let degree = 2;
//!
//! let curve = NurbsCurve::new(
//!     &control_points.view(),
//!     &weights.view(),
//!     &knots.view(),
//!     degree,
//!     ExtrapolateMode::Extrapolate
//! ).unwrap();
//!
//! // Evaluate the curve at parameter t = 0.5
//! let point = curve.evaluate(0.5).unwrap();
//! ```
//!
//! ### NURBS Surface
//!
//! ```rust
//! use scirs2_core::ndarray::array;
//! use scirs2_interpolate::nurbs::NurbsSurface;
//! use scirs2_interpolate::bspline::ExtrapolateMode;
//!
//! let control_points = array![
//!     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]
//! ];
//! let weights = array![1.0, 1.0, 1.0, 1.0];
//! let knotsu = array![0.0, 0.0, 1.0, 1.0];
//! let knotsv = array![0.0, 0.0, 1.0, 1.0];
//!
//! let surface = NurbsSurface::new(
//!     &control_points.view(),
//!     &weights.view(),
//!     2, 2,  // nu, nv
//!     &knotsu.view(),
//!     &knotsv.view(),
//!     1, 1,  // degreeu, degreev
//!     ExtrapolateMode::Extrapolate
//! ).unwrap();
//!
//! // Evaluate the surface at parameters (u=0.5, v=0.5)
//! let point = surface.evaluate(0.5, 0.5).unwrap();
//! ```
//!
//! ### Creating Common Shapes
//!
//! ```rust
//! use scirs2_interpolate::nurbs::{make_nurbs_circle, make_nurbs_sphere};
//!
//! // Create a circle centered at origin with radius 1.0
//! let circle = make_nurbs_circle([0.0, 0.0], 1.0, Some(0.0), Some(2.0 * std::f64::consts::PI)).unwrap();
//!
//! // Create a sphere centered at origin with radius 1.0
//! let sphere = make_nurbs_sphere([0.0, 0.0, 0.0], 1.0).unwrap();
//! ```

// Re-export the nurbs module contents from the subdirectory
mod nurbs_impl {
    include!("nurbs_modules/mod.rs");
}

pub use nurbs_impl::*;
