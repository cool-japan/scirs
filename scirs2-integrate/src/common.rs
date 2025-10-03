//! Common traits and types used throughout the crate
//!
//! This module defines common traits and types that are used by multiple
//! parts of the code to ensure consistency and reduce duplication.

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::{Float, FromPrimitive};
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
    /// Convert to f64 for interfacing with GPU kernels
    fn to_f64(self) -> Option<f64>;
}

// Specific implementations for common float types
impl IntegrateFloat for f32 {
    fn to_f64(self) -> Option<f64> {
        Some(self as f64)
    }
}

impl IntegrateFloat for f64 {
    fn to_f64(self) -> Option<f64> {
        Some(self)
    }
}
