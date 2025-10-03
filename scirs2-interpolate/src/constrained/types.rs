//! Core types and data structures for constrained splines
//!
//! This module contains the fundamental types used throughout the constrained
//! spline implementation, including constraint types, constraint structures,
//! and the main ConstrainedSpline struct.

use crate::bspline::BSpline;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

/// Types of constraints that can be applied to a spline
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintType {
    /// Monotonically increasing constraint (f'(x) >= 0)
    MonotoneIncreasing,

    /// Monotonically decreasing constraint (f'(x) <= 0)
    MonotoneDecreasing,

    /// Convexity constraint (f''(x) >= 0)
    Convex,

    /// Concavity constraint (f''(x) <= 0)
    Concave,

    /// Positivity constraint (f(x) >= 0)
    Positive,

    /// Upper bound constraint (f(x) <= upper_bound)
    UpperBound,

    /// Lower bound constraint (f(x) >= lower_bound)
    LowerBound,
}

/// A constraint on a spline with a specific region of application
#[derive(Debug, Clone)]
pub struct Constraint<T: Float + FromPrimitive + Debug + Display> {
    /// Type of constraint
    pub(crate) constraint_type: ConstraintType,

    /// Start of the region where the constraint applies (None = start of data)
    pub(crate) x_min: Option<T>,

    /// End of the region where the constraint applies (None = end of data)
    pub(crate) x_max: Option<T>,

    /// Additional parameter for constraints that need it (e.g., bounds)
    pub(crate) parameter: Option<T>,
}

/// Represents the region where a constraint applies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintRegion {
    /// Apply constraint globally across the entire domain
    Global,
    /// Apply constraint in a specific interval
    Interval { start: f64, end: f64 },
    /// Apply constraint at specific points only
    Points(usize),
}

impl<T: Float + FromPrimitive + Debug + Display> Constraint<T> {
    /// Create a new monotonically increasing constraint
    ///
    /// # Arguments
    ///
    /// * `x_min` - Start of the region where the constraint applies (None = start of data)
    /// * `x_max` - End of the region where the constraint applies (None = end of data)
    ///
    /// # Returns
    ///
    /// A new monotonically increasing constraint
    pub fn monotone_increasing(x_min: Option<T>, x_max: Option<T>) -> Self {
        Constraint {
            constraint_type: ConstraintType::MonotoneIncreasing,
            x_min,
            x_max,
            parameter: None,
        }
    }

    /// Create a new monotonically decreasing constraint
    ///
    /// # Arguments
    ///
    /// * `x_min` - Start of the region where the constraint applies (None = start of data)
    /// * `x_max` - End of the region where the constraint applies (None = end of data)
    ///
    /// # Returns
    ///
    /// A new monotonically decreasing constraint
    pub fn monotone_decreasing(x_min: Option<T>, x_max: Option<T>) -> Self {
        Constraint {
            constraint_type: ConstraintType::MonotoneDecreasing,
            x_min,
            x_max,
            parameter: None,
        }
    }

    /// Create a new convexity constraint
    ///
    /// # Arguments
    ///
    /// * `x_min` - Start of the region where the constraint applies (None = start of data)
    /// * `x_max` - End of the region where the constraint applies (None = end of data)
    ///
    /// # Returns
    ///
    /// A new convexity constraint
    pub fn convex(x_min: Option<T>, x_max: Option<T>) -> Self {
        Constraint {
            constraint_type: ConstraintType::Convex,
            x_min,
            x_max,
            parameter: None,
        }
    }

    /// Create a new concavity constraint
    ///
    /// # Arguments
    ///
    /// * `x_min` - Start of the region where the constraint applies (None = start of data)
    /// * `x_max` - End of the region where the constraint applies (None = end of data)
    ///
    /// # Returns
    ///
    /// A new concavity constraint
    pub fn concave(x_min: Option<T>, x_max: Option<T>) -> Self {
        Constraint {
            constraint_type: ConstraintType::Concave,
            x_min,
            x_max,
            parameter: None,
        }
    }

    /// Create a new positivity constraint
    ///
    /// # Arguments
    ///
    /// * `x_min` - Start of the region where the constraint applies (None = start of data)
    /// * `x_max` - End of the region where the constraint applies (None = end of data)
    ///
    /// # Returns
    ///
    /// A new positivity constraint
    pub fn positive(x_min: Option<T>, x_max: Option<T>) -> Self {
        Constraint {
            constraint_type: ConstraintType::Positive,
            x_min,
            x_max,
            parameter: None,
        }
    }

    /// Create a new upper bound constraint
    ///
    /// # Arguments
    ///
    /// * `x_min` - Start of the region where the constraint applies (None = start of data)
    /// * `x_max` - End of the region where the constraint applies (None = end of data)
    /// * `upper_bound` - Maximum allowed value for the spline
    ///
    /// # Returns
    ///
    /// A new upper bound constraint
    pub fn upper_bound(x_min: Option<T>, x_max: Option<T>, upperbound: T) -> Self {
        Constraint {
            constraint_type: ConstraintType::UpperBound,
            x_min,
            x_max,
            parameter: Some(upperbound),
        }
    }

    /// Create a new lower bound constraint
    ///
    /// # Arguments
    ///
    /// * `x_min` - Start of the region where the constraint applies (None = start of data)
    /// * `x_max` - End of the region where the constraint applies (None = end of data)
    /// * `lower_bound` - Minimum allowed value for the spline
    ///
    /// # Returns
    ///
    /// A new lower bound constraint
    pub fn lower_bound(x_min: Option<T>, x_max: Option<T>, lowerbound: T) -> Self {
        Constraint {
            constraint_type: ConstraintType::LowerBound,
            x_min,
            x_max,
            parameter: Some(lowerbound),
        }
    }
}

/// Constrained spline with various possible constraints
///
/// This spline implementation enforces constraints like monotonicity and convexity
/// through optimization, finding the coefficients that best fit the data while
/// satisfying all the specified constraints.
#[derive(Debug, Clone)]
pub struct ConstrainedSpline<T>
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
        + 'static,
{
    /// The underlying B-spline representation
    pub(crate) bspline: BSpline<T>,

    /// List of constraints applied to the spline
    pub(crate) constraints: Vec<Constraint<T>>,

    /// Fitting method used
    pub(crate) method: super::builder::FittingMethod,
}

/// Boundary conditions for constrained splines
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// No specific boundary condition
    None,
    /// First derivative at boundary
    FirstDerivative(f64),
    /// Second derivative at boundary
    SecondDerivative(f64),
    /// Natural boundary (second derivative = 0)
    Natural,
    /// Periodic boundary
    Periodic,
}
