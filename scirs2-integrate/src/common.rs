//! Common traits and types used throughout the crate
//!
//! This module defines common traits and types that are used by multiple
//! parts of the code to ensure consistency and reduce duplication.

use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display, LowerExp};

/// A trait that bundles all the requirements for floating-point types
/// used in the integration routines.
///
/// This trait simplifies type signatures by combining all the trait bounds
/// that are commonly needed for the numeric operations performed in
/// integration routines.
pub trait IntegrateFloat:
    Float
    + FromPrimitive
    + Debug
    + 'static
    + ScalarOperand
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + Display
    + LowerExp
    + std::iter::Sum
{
}

// Implement IntegrateFloat for any type that satisfies all the trait bounds
impl<T> IntegrateFloat for T where
    T: Float
        + FromPrimitive
        + Debug
        + 'static
        + ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + Display
        + LowerExp
        + std::iter::Sum
{
}
