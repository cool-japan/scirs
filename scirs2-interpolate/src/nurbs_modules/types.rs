//! Core types and trait definitions for NURBS
//!
//! This module defines the fundamental data structures and traits used throughout
//! the NURBS (Non-Uniform Rational B-Splines) implementation.

use crate::bspline::{BSpline, ExtrapolateMode};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

/// Trait alias for numeric types used in NURBS computations
///
/// This trait combines all the necessary numeric operations and properties
/// required for NURBS curve and surface computations.
pub trait NurbsFloat:
    Float
    + FromPrimitive
    + Debug
    + Display
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::RemAssign
    + scirs2_core::ndarray::ScalarOperand
    + Copy
{
}

// Implement the trait for common floating point types
impl<T> NurbsFloat for T where
    T: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + scirs2_core::ndarray::ScalarOperand
        + Copy
{
}

/// NURBS curve defined by control points, weights, and knot vector
///
/// A NURBS curve is defined by:
/// - Control points in n-dimensional space
/// - Weights associated with each control point
/// - A knot vector
/// - A degree
///
/// The curve is defined as:
/// ```text
/// C(u) = Σ(i=0..n) (w_i * P_i * N_{i,p}(u)) / Σ(i=0..n) (w_i * N_{i,p}(u))
/// ```
///
/// where N_{i,p} are the B-spline basis functions of degree p,
/// P_i are the control points, and w_i are the weights.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{array, Array1, Array2};
/// use scirs2_interpolate::nurbs::{NurbsCurve};
/// use scirs2_interpolate::bspline::ExtrapolateMode;
///
/// // Create a simple NURBS curve
/// let control_points = array![
///     [0.0, 0.0],
///     [1.0, 1.0],
///     [2.0, 0.0]
/// ];
/// let weights = array![1.0, 1.0, 1.0];
/// let knots = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let degree = 2;
///
/// let curve = NurbsCurve::new(
///     &control_points.view(),
///     &weights.view(),
///     &knots.view(),
///     degree,
///     ExtrapolateMode::Extrapolate
/// ).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NurbsCurve<T: NurbsFloat> {
    /// Control points defining the curve (n x dim)
    pub(crate) control_points: Array2<T>,
    /// Weights for each control point (n)
    pub(crate) weights: Array1<T>,
    /// Underlying B-spline representation
    pub(crate) bspline: BSpline<T>,
    /// Dimension of the control points
    pub(crate) dimension: usize,
}

/// NURBS surface defined by control points, weights, and knot vectors
///
/// A NURBS surface is the 2D generalization of NURBS curves, defined by:
/// - A grid of control points in n-dimensional space
/// - Weights associated with each control point
/// - Knot vectors in both u and v directions
/// - Degrees in both u and v directions
///
/// The surface is defined as:
/// ```text
/// S(u,v) = Σ(i=0..n) Σ(j=0..m) (w_{i,j} * P_{i,j} * N_{i,p}(u) * N_{j,q}(v)) /
///           Σ(i=0..n) Σ(j=0..m) (w_{i,j} * N_{i,p}(u) * N_{j,q}(v))
/// ```
///
/// where N_{i,p} and N_{j,q} are the B-spline basis functions,
/// P_{i,j} are the control points, and w_{i,j} are the weights.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{array, Array1, Array2};
/// use scirs2_interpolate::nurbs::NurbsSurface;
/// use scirs2_interpolate::bspline::ExtrapolateMode;
///
/// // Create a simple NURBS surface
/// let control_points = array![
///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]
/// ];
/// let weights = array![1.0, 1.0, 1.0, 1.0];
/// let knotsu = array![0.0, 0.0, 1.0, 1.0];
/// let knotsv = array![0.0, 0.0, 1.0, 1.0];
///
/// let surface = NurbsSurface::new(
///     &control_points.view(),
///     &weights.view(),
///     2, 2,  // nu, nv
///     &knotsu.view(),
///     &knotsv.view(),
///     1, 1,  // degreeu, degreev
///     ExtrapolateMode::Extrapolate
/// ).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NurbsSurface<T: NurbsFloat> {
    /// Control points defining the surface (nu * nv x dim)
    pub(crate) control_points: Array2<T>,
    /// Weights for each control point (nu * nv)
    pub(crate) weights: Array1<T>,
    /// Number of control points in the u direction
    pub(crate) nu: usize,
    /// Number of control points in the v direction
    pub(crate) nv: usize,
    /// Knot vector in the u direction
    pub(crate) knotsu: Array1<T>,
    /// Knot vector in the v direction
    pub(crate) knotsv: Array1<T>,
    /// Degree in the u direction
    pub(crate) degreeu: usize,
    /// Degree in the v direction
    pub(crate) degreev: usize,
    /// Dimension of the control points
    pub(crate) dimension: usize,
    /// Extrapolation mode
    pub(crate) extrapolate: ExtrapolateMode,
}

/// Validation result for NURBS input parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValidationResult {
    /// Input parameters are valid
    Valid,
    /// Control points and weights have mismatched dimensions
    DimensionMismatch,
    /// Knot vector is invalid
    InvalidKnots,
    /// Degree is too high for the number of control points
    InvalidDegree,
    /// Weights contain non-positive values
    InvalidWeights,
    /// Control points are empty or malformed
    InvalidControlPoints,
}

impl ValidationResult {
    /// Check if the validation result indicates success
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationResult::Valid)
    }

    /// Convert validation result to a descriptive error message
    pub fn to_error_message(&self) -> &'static str {
        match self {
            ValidationResult::Valid => "Input parameters are valid",
            ValidationResult::DimensionMismatch => {
                "Control points and weights have mismatched dimensions"
            }
            ValidationResult::InvalidKnots => "Knot vector is invalid",
            ValidationResult::InvalidDegree => {
                "Degree is too high for the number of control points"
            }
            ValidationResult::InvalidWeights => "Weights contain non-positive values",
            ValidationResult::InvalidControlPoints => "Control points are empty or malformed",
        }
    }
}

/// Common validation functions for NURBS parameters
pub struct NurbsValidator;

impl NurbsValidator {
    /// Validate control points and weights for consistency
    pub fn validate_curve_inputs<T: NurbsFloat>(
        control_points: &Array2<T>,
        weights: &Array1<T>,
        knots: &Array1<T>,
        degree: usize,
    ) -> ValidationResult {
        // Check control points
        if control_points.is_empty() || control_points.nrows() == 0 {
            return ValidationResult::InvalidControlPoints;
        }

        // Check dimension consistency
        if control_points.nrows() != weights.len() {
            return ValidationResult::DimensionMismatch;
        }

        // Check weights are positive
        for &weight in weights.iter() {
            if weight <= T::zero() {
                return ValidationResult::InvalidWeights;
            }
        }

        // Check degree validity
        if degree >= control_points.nrows() {
            return ValidationResult::InvalidDegree;
        }

        // Check knot vector validity
        if knots.len() != control_points.nrows() + degree + 1 {
            return ValidationResult::InvalidKnots;
        }

        // Check knot vector is non-decreasing
        for i in 1..knots.len() {
            if knots[i] < knots[i - 1] {
                return ValidationResult::InvalidKnots;
            }
        }

        ValidationResult::Valid
    }

    /// Validate surface parameters
    pub fn validate_surface_inputs<T: NurbsFloat>(
        control_points: &Array2<T>,
        weights: &Array1<T>,
        nu: usize,
        nv: usize,
        knotsu: &Array1<T>,
        knotsv: &Array1<T>,
        degreeu: usize,
        degreev: usize,
    ) -> ValidationResult {
        // Check basic dimensions
        if nu == 0 || nv == 0 {
            return ValidationResult::InvalidControlPoints;
        }

        if control_points.nrows() != nu * nv {
            return ValidationResult::DimensionMismatch;
        }

        if weights.len() != nu * nv {
            return ValidationResult::DimensionMismatch;
        }

        // Check weights are positive
        for &weight in weights.iter() {
            if weight <= T::zero() {
                return ValidationResult::InvalidWeights;
            }
        }

        // Check degrees
        if degreeu >= nu || degreev >= nv {
            return ValidationResult::InvalidDegree;
        }

        // Check knot vectors
        if knotsu.len() != nu + degreeu + 1 {
            return ValidationResult::InvalidKnots;
        }

        if knotsv.len() != nv + degreev + 1 {
            return ValidationResult::InvalidKnots;
        }

        // Check knot vectors are non-decreasing
        for i in 1..knotsu.len() {
            if knotsu[i] < knotsu[i - 1] {
                return ValidationResult::InvalidKnots;
            }
        }

        for i in 1..knotsv.len() {
            if knotsv[i] < knotsv[i - 1] {
                return ValidationResult::InvalidKnots;
            }
        }

        ValidationResult::Valid
    }
}