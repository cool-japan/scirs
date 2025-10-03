//! Convenience functions for creating constrained splines
//!
//! This module provides high-level convenience functions for creating
//! commonly used types of constrained splines.

use crate::bspline::ExtrapolateMode;
use crate::error::InterpolateResult;
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

use super::types::{ConstrainedSpline, Constraint};

/// Convenience function to create a monotonically increasing spline
///
/// # Arguments
///
/// * `x` - The x coordinates of the data points
/// * `y` - The y coordinates of the data points
/// * `degree` - Degree of the B-spline (defaults to 3 for cubic splines)
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A new constrained spline that interpolates the data points while guaranteeing monotonicity
#[allow(dead_code)]
pub fn monotone_increasing_spline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    degree: usize,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<ConstrainedSpline<T>>
where
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
        + 'static
        + std::fmt::LowerExp,
{
    let constraint = Constraint::monotone_increasing(None, None);
    ConstrainedSpline::interpolate(x, y, vec![constraint], degree, extrapolate)
}

/// Convenience function to create a monotonically decreasing spline
///
/// # Arguments
///
/// * `x` - The x coordinates of the data points
/// * `y` - The y coordinates of the data points
/// * `degree` - Degree of the B-spline (defaults to 3 for cubic splines)
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A new constrained spline that interpolates the data points while guaranteeing monotonic decrease
#[allow(dead_code)]
pub fn monotone_decreasing_spline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    degree: usize,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<ConstrainedSpline<T>>
where
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
        + 'static
        + std::fmt::LowerExp,
{
    let constraint = Constraint::monotone_decreasing(None, None);
    ConstrainedSpline::interpolate(x, y, vec![constraint], degree, extrapolate)
}

/// Convenience function to create a convex spline
///
/// # Arguments
///
/// * `x` - The x coordinates of the data points
/// * `y` - The y coordinates of the data points
/// * `degree` - Degree of the B-spline (defaults to 3 for cubic splines)
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A new constrained spline that interpolates the data points while guaranteeing convexity
#[allow(dead_code)]
pub fn convex_spline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    degree: usize,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<ConstrainedSpline<T>>
where
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
        + 'static
        + std::fmt::LowerExp,
{
    let constraint = Constraint::convex(None, None);
    ConstrainedSpline::interpolate(x, y, vec![constraint], degree, extrapolate)
}

/// Convenience function to create a concave spline
///
/// # Arguments
///
/// * `x` - The x coordinates of the data points
/// * `y` - The y coordinates of the data points
/// * `degree` - Degree of the B-spline (defaults to 3 for cubic splines)
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A new constrained spline that interpolates the data points while guaranteeing concavity
#[allow(dead_code)]
pub fn concave_spline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    degree: usize,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<ConstrainedSpline<T>>
where
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
        + 'static
        + std::fmt::LowerExp,
{
    let constraint = Constraint::concave(None, None);
    ConstrainedSpline::interpolate(x, y, vec![constraint], degree, extrapolate)
}

/// Convenience function to create a positive spline
///
/// # Arguments
///
/// * `x` - The x coordinates of the data points
/// * `y` - The y coordinates of the data points
/// * `degree` - Degree of the B-spline (defaults to 3 for cubic splines)
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A new constrained spline that interpolates the data points while guaranteeing positivity
#[allow(dead_code)]
pub fn positive_spline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    degree: usize,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<ConstrainedSpline<T>>
where
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
        + 'static
        + std::fmt::LowerExp,
{
    let constraint = Constraint::positive(None, None);
    ConstrainedSpline::interpolate(x, y, vec![constraint], degree, extrapolate)
}

/// Convenience function to create a spline with both monotonicity and convexity constraints
///
/// # Arguments
///
/// * `x` - The x coordinates of the data points
/// * `y` - The y coordinates of the data points
/// * `increasing` - Whether the spline should be monotonically increasing (true) or decreasing (false)
/// * `convex` - Whether the spline should be convex (true) or concave (false)
/// * `degree` - Degree of the B-spline (defaults to 3 for cubic splines)
/// * `extrapolate` - Extrapolation mode
///
/// # Returns
///
/// A new constrained spline that interpolates the data points with both constraints
#[allow(dead_code)]
pub fn monotone_convex_spline<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    increasing: bool,
    convex: bool,
    degree: usize,
    extrapolate: ExtrapolateMode,
) -> InterpolateResult<ConstrainedSpline<T>>
where
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
        + 'static
        + std::fmt::LowerExp,
{
    let mut constraints = Vec::new();

    // Add monotonicity constraint
    if increasing {
        constraints.push(Constraint::monotone_increasing(None, None));
    } else {
        constraints.push(Constraint::monotone_decreasing(None, None));
    }

    // Add convexity constraint
    if convex {
        constraints.push(Constraint::convex(None, None));
    } else {
        constraints.push(Constraint::concave(None, None));
    }

    ConstrainedSpline::interpolate(x, y, constraints, degree, extrapolate)
}
