//! Bessel functions with enhanced numerical stability
//!
//! This module provides implementations of Bessel functions
//! with better handling of extreme arguments and improved numerical stability.
//!
//! The implementation includes:
//! - Better handling of extreme arguments (very large, very small, near zeros)
//! - Improved asymptotic expansions for large arguments
//! - Use of pre-computed constants for improved precision
//! - Protection against overflow and underflow
//! - Better convergence properties for series evaluations
//!
//! ## First Kind
//!
//! The Bessel functions of the first kind, denoted as J_v(x), are solutions
//! to the differential equation:
//!
//! x² d²y/dx² + x dy/dx + (x² - v²) y = 0
//!
//! Functions:
//! - j0(x): First kind, order 0
//! - j1(x): First kind, order 1
//! - jn(n, x): First kind, integer order n
//!
//! ## Second Kind
//!
//! The Bessel functions of the second kind, denoted as Y_v(x), are solutions
//! to the same differential equation as the first kind but are linearly independent.
//!
//! Functions:
//! - y0(x): Second kind, order 0
//! - y1(x): Second kind, order 1
//! - yn(n, x): Second kind, integer order n
//!
//! ## Modified Bessel Functions
//!
//! The modified Bessel functions are solutions to the modified Bessel's differential equation:
//!
//! x² d²y/dx² + x dy/dx - (x² + v²) y = 0
//!
//! First kind (I_v):
//! - i0(x): Modified first kind, order 0
//! - i1(x): Modified first kind, order 1
//! - iv(v, x): Modified first kind, arbitrary order v
//!
//! Second kind (K_v):
//! - k0(x): Modified second kind, order 0
//! - k1(x): Modified second kind, order 1
//! - kv(v, x): Modified second kind, arbitrary order v

// No imports needed at the module level

// Re-export all public functions
pub use self::derivatives::{j0_prime, j1_prime, jn_prime, jv_prime, y0_prime, y1_prime, yn_prime};
pub use self::first_kind::{j0, j1, jn, jv};
pub use self::modified::{i0, i1, iv, k0, k1, kv};
pub use self::second_kind::{y0, y1, yn};
pub use self::spherical::{spherical_jn, spherical_jn_scaled, spherical_yn, spherical_yn_scaled};

// The helper functions below are moved to the relevant modules where they are used
// to avoid "unused function" warnings

// Export each set of functions from their own module
pub mod derivatives;
pub mod first_kind;
pub mod modified;
pub mod second_kind;
pub mod spherical;

// Import tests
#[cfg(test)]
mod tests;
