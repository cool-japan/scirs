//! Gamma function and related implementations
//!
//! This module provides enhanced implementations of the gamma function, beta function,
//! and related special functions with better handling of edge cases and numerical stability.
//!
//! ## Mathematical Theory
//!
//! ### The Gamma Function
//!
//! The gamma function Γ(z) is one of the most important special functions in mathematics,
//! extending the factorial function to complex numbers. It is defined by the integral:
//!
//! **Definition (Euler's Integral of the Second Kind)**:
//! ```text
//! Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt,    Re(z) > 0
//! ```
//!
//! **Fundamental Properties**:
//!
//! 1. **Functional Equation**: Γ(z+1) = z·Γ(z)
//!    - **Proof**: Integration by parts on the defining integral
//!    - **Consequence**: For positive integers n, Γ(n) = (n-1)!
//!
//! 2. **Reflection Formula** (Euler): Γ(z)Γ(1-z) = π/sin(πz)
//!    - **Proof**: Contour integration using the residue theorem
//!    - **Application**: Extends Γ(z) to the entire complex plane except negative integers
//!
//! 3. **Multiplication Formula** (Legendre):
//!    ```text
//!    Γ(z)Γ(z+1/n)...Γ(z+(n-1)/n) = (2π)^((n-1)/2) n^(1/2-nz) Γ(nz)
//!    ```
//!
//! ### Alternative Representations
//!
//! **Weierstrass Product Formula**:
//! ```text
//! 1/Γ(z) = z e^(γz) ∏_{n=1}^∞ [(1 + z/n) e^(-z/n)]
//! ```
//! where γ is the Euler-Mascheroni constant.
//!
//! **Euler's Infinite Product**:
//! ```text
//! Γ(z) = lim_{n→∞} n^z n! / [z(z+1)(z+2)...(z+n)]
//! ```
//!
//! ### Asymptotic Behavior
//!
//! **Stirling's Formula** (for large |z|, |arg(z)| < π):
//! ```text
//! Γ(z) ~ √(2π/z) (z/e)^z [1 + 1/(12z) + 1/(288z²) - 139/(51840z³) + ...]
//! ```
//!
//! The error in truncating after the k-th term is bounded by the (k+1)-th term
//! when |arg(z)| ≤ π - δ for any δ > 0.
//!
//! ### Computational Methods
//!
//! This implementation uses several numerical methods depending on the input:
//!
//! 1. **Direct computation** for small positive integers
//! 2. **Series expansion** for values near zero: Γ(z) ≈ 1/z - γ + O(z)
//! 3. **Reflection formula** for negative values
//! 4. **Lanczos approximation** for general complex values
//! 5. **Stirling's approximation** for large values to prevent overflow

pub mod approximations;
pub mod beta;
pub mod complex;
pub mod constants;
pub mod core;
pub mod digamma;
pub mod utils;

// Re-export all public functions for backward compatibility
pub use beta::{beta, beta_safe, betainc, betainc_regularized, betaincinv};
pub use complex::{beta_complex, digamma_complex, gamma_complex, loggamma_complex};
pub use core::{gamma, gamma_safe, gammaln, loggamma};
pub use digamma::{digamma, digamma_safe};
pub use utils::polygamma;

// Also re-export betaln which is defined in core
pub use core::betaln;
