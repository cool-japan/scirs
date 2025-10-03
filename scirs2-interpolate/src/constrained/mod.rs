//! Constrained splines with monotonicity and convexity constraints
//!
//! This module provides spline interpolation methods with explicit constraints
//! on properties such as monotonicity (increasing or decreasing) and convexity
//! (convex or concave). These constraints are enforced through an optimization
//! approach that preserves these properties while still providing a smooth curve.
//!
//! Unlike the monotonic interpolation methods in the interp1d module, which use
//! specific basis functions or filtering, these methods use a more general
//! optimization-based approach to enforce constraints anywhere on the spline.
//!
//! Possible constraints include:
//! - Monotonicity (strictly increasing or decreasing)
//! - Convexity (convex or concave)
//! - Positivity
//! - Range constraints (min/max values)
//! - Fixed values at specific points
//! - Fixed derivatives at specific points
//!
//! These methods are particularly useful for:
//! - Economic modeling (utility functions, demand curves)
//! - Physical models with known constraints
//! - Cumulative distribution functions
//! - Probability density functions
//! - Yield curves and term structures
//!
//! # Module Structure
//!
//! This module is organized into several submodules:
//! - `types`: Core types like `ConstraintType`, `Constraint`, and `BoundaryCondition`
//! - `builder`: The `ConstrainedSplineBuilder` for constructing constrained splines
//! - `solver`: Internal solvers for constrained optimization problems
//! - `utils`: Utility functions for constraint matrix generation and checking
//! - `convenience`: High-level convenience functions for common use cases

pub mod builder;
pub mod convenience;
pub(crate) mod solver;
pub mod types;
pub mod utils;

// Re-export the main public types and functions
pub use builder::FittingMethod;
pub use convenience::{
    concave_spline, convex_spline, monotone_convex_spline, monotone_decreasing_spline,
    monotone_increasing_spline, positive_spline,
};
pub use types::{
    BoundaryCondition, ConstrainedSpline, Constraint, ConstraintRegion, ConstraintType,
};
pub use utils::{ConstraintSatisfactionSummary, ConstraintViolationInfo};

// Re-export the main methods from ConstrainedSpline
impl<T> ConstrainedSpline<T>
where
    T: scirs2_core::numeric::Float
        + scirs2_core::numeric::FromPrimitive
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    // These methods are already defined in builder.rs and utils.rs
    // This impl block is just to document the public API
}
