//! Core types and boundary conditions for cubic spline interpolation
//!
//! This module defines the fundamental types used throughout the spline interpolation
//! system, including boundary conditions and integration region types.

use std::fmt::Debug;

/// Boundary conditions for cubic spline interpolation
///
/// Boundary conditions determine the behavior of the spline at the endpoints and
/// significantly affect the shape and properties of the interpolated curve. Choose
/// the appropriate condition based on your physical constraints and smoothness requirements.
///
/// ## Mathematical Properties
///
/// Each boundary condition imposes different constraints on the spline coefficients,
/// leading to different system of equations to solve during construction.
///
/// ## Performance Impact
///
/// All boundary conditions have the same computational complexity O(n) for construction.
/// The choice primarily affects numerical stability and curve shape, not performance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplineBoundaryCondition<T> {
    /// Natural spline boundary condition (default)
    ///
    /// Sets the second derivative to zero at both endpoints: S''(x₀) = S''(xₙ) = 0
    ///
    /// **Mathematical properties:**
    /// - Minimizes the integral of the second derivative (curvature)
    /// - Results in the "most relaxed" curve shape
    /// - May exhibit unwanted oscillations with poorly distributed data
    ///
    /// **When to use:**
    /// - Default choice when no endpoint derivative information is available
    /// - Physical systems with no constraints at boundaries
    /// - When minimizing overall curvature is desired
    ///
    /// **Numerical stability:** Excellent
    Natural,

    /// Not-a-knot boundary condition
    ///
    /// Forces the third derivative to be continuous at the second and second-to-last
    /// data points, effectively making the first and last polynomial pieces part of
    /// the same cubic.
    ///
    /// **Mathematical properties:**
    /// - Maximizes smoothness at internal points
    /// - Often produces the most visually pleasing curves
    /// - Reduces oscillations compared to natural splines
    ///
    /// **When to use:**
    /// - When maximum smoothness is desired
    /// - For visualization and computer graphics applications
    /// - When data is well-distributed and smooth
    ///
    /// **Numerical stability:** Excellent
    NotAKnot,

    /// Clamped (Complete) spline with specified endpoint derivatives
    ///
    /// Specifies the first derivative at both endpoints: S'(x₀) = dy₀, S'(xₙ) = dyₙ
    ///
    /// **Parameters:**
    /// - First value: left endpoint derivative S'(x₀)
    /// - Second value: right endpoint derivative S'(xₙ)
    ///
    /// **Mathematical properties:**
    /// - Provides exact control over endpoint slopes
    /// - Often the most accurate when derivative information is known
    /// - Eliminates endpoint artifacts
    ///
    /// **When to use:**
    /// - When endpoint derivatives are known from physics or other constraints
    /// - For fitting data with known tangent behavior at boundaries
    /// - When connecting spline pieces with continuous derivatives
    ///
    /// **Numerical stability:** Excellent
    ///
    /// **Example:**
    /// ```rust
    /// use scirs2_interpolate::spline::SplineBoundaryCondition;
    /// // Specify horizontal tangents at both ends
    /// let bc = SplineBoundaryCondition::Clamped(0.0, 0.0);
    /// ```
    Clamped(T, T),

    /// Periodic boundary condition
    ///
    /// Forces the function value, first derivative, and second derivative to match
    /// at the endpoints: S(x₀) = S(xₙ), S'(x₀) = S'(xₙ), S''(x₀) = S''(xₙ)
    ///
    /// **Mathematical properties:**
    /// - Creates a smooth, closed curve when plotted
    /// - Requires y₀ = yₙ (function values must match)
    /// - Reduces the system to n-1 unknowns
    ///
    /// **When to use:**
    /// - For periodic data (circular, seasonal, angular)
    /// - When fitting closed curves or loops
    /// - For data representing periodic phenomena
    ///
    /// **Requirements:**
    /// - First and last y-values must be equal
    /// - Data should represent one complete period
    ///
    /// **Numerical stability:** Good (may be less stable for ill-conditioned data)
    ///
    /// **Example:**
    /// ```rust
    /// use scirs2_core::ndarray::array;
    /// use scirs2_interpolate::spline::SplineBoundaryCondition;
    /// // For angular data from 0 to 2π
    /// let x = array![0.0, 1.57, 3.14, 4.71, 6.28]; // π/2, π, 3π/2, 2π
    /// let y = array![0.0, 1.0, 0.0, -1.0, 0.0]; // sine-like data
    /// let bc: SplineBoundaryCondition<f64> = SplineBoundaryCondition::Periodic;
    /// ```
    Periodic,

    /// Specified second derivative boundary condition
    ///
    /// Sets the second derivative at both endpoints: S''(x₀) = d²y₀, S''(xₙ) = d²yₙ
    ///
    /// **Parameters:**
    /// - First value: left endpoint second derivative S''(x₀)
    /// - Second value: right endpoint second derivative S''(xₙ)
    ///
    /// **Mathematical properties:**
    /// - Provides direct control over endpoint curvature
    /// - Useful when curvature constraints are known
    /// - Natural spline is the special case where both values are 0
    ///
    /// **When to use:**
    /// - When endpoint curvature is known from physical constraints
    /// - For beam bending problems (specify moment/curvature)
    /// - When connecting to other curves with known curvature
    ///
    /// **Numerical stability:** Excellent
    ///
    /// **Example:**
    /// ```rust
    /// use scirs2_interpolate::spline::SplineBoundaryCondition;
    /// // Specify positive curvature (concave up) at left, negative at right
    /// let bc = SplineBoundaryCondition::SecondDerivative(1.0, -1.0);
    /// ```
    SecondDerivative(T, T),

    /// Parabolic runout boundary condition (experimental)
    ///
    /// Sets the second derivative to zero at one endpoint while using not-a-knot
    /// at the other. This is a specialized condition for certain applications.
    ///
    /// **Mathematical properties:**
    /// - Hybrid approach combining natural and not-a-knot
    /// - Asymmetric boundary treatment
    /// - Less commonly used in practice
    ///
    /// **When to use:**
    /// - Specialized applications requiring asymmetric boundary treatment
    /// - Legacy compatibility with certain spline implementations
    ///
    /// **Numerical stability:** Good
    ///
    /// **Note:** This condition is experimental and may change in future versions.
    ParabolicRunout,
}

/// Integration region type for extrapolation-aware integration
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum IntegrationRegion {
    /// Integration region within the spline domain
    Interior,
    /// Integration region to the left of the spline domain (requires extrapolation)
    LeftExtrapolation,
    /// Integration region to the right of the spline domain (requires extrapolation)
    RightExtrapolation,
}