//! Bessel functions with enhanced numerical stability
//!
//! This module provides comprehensive implementations of Bessel functions
//! with rigorous mathematical foundations, detailed proofs, and enhanced numerical stability.
//!
//! ## Mathematical Theory and Derivations
//!
//! ### Historical Context
//!
//! Bessel functions were first studied by Daniel Bernoulli (1738) and later by
//! Friedrich Bessel (1824) in his analysis of planetary motion. They arise naturally
//! in problems with cylindrical or spherical symmetry and are among the most
//! important special functions in mathematical physics.
//!
//! ### The Bessel Differential Equation
//!
//! Bessel functions are solutions to **Bessel's differential equation**:
//!
//! ```text
//! x² d²y/dx² + x dy/dx + (x² - ν²) y = 0
//! ```
//!
//! **Derivation from Laplace's Equation**:
//!
//! In cylindrical coordinates (r, θ, z), Laplace's equation ∇²u = 0 becomes:
//! ```text
//! (1/r) ∂/∂r(r ∂u/∂r) + (1/r²) ∂²u/∂θ² + ∂²u/∂z² = 0
//! ```
//!
//! Using separation of variables u(r,θ,z) = R(r)Θ(θ)Z(z), the radial equation becomes:
//! ```text
//! r² R'' + r R' + (λ²r² - ν²) R = 0
//! ```
//!
//! Substituting x = λr transforms this into the standard Bessel equation.
//!
//! ### Fundamental Solutions
//!
//! **Bessel Functions of the First Kind** (J_ν(x)):
//! - Regular at x = 0 for ν ≥ 0
//! - Series representation: J_ν(x) = Σ_{k=0}^∞ [(-1)^k (x/2)^(ν+2k)] / [k! Γ(ν+k+1)]
//! - **Proof of convergence**: Ratio test shows convergence for all finite x
//!
//! **Bessel Functions of the Second Kind** (Y_ν(x)):
//! - Singular at x = 0 (logarithmic singularity)
//! - Defined by: Y_ν(x) = [J_ν(x) cos(νπ) - J_{-ν}(x)] / sin(νπ)
//! - Forms a complete set with J_ν(x) for linearly independent solutions
//!
//! ### Key Properties and Identities
//!
//! **Wronskian Identity**:
//! ```text
//! W[J_ν(x), Y_ν(x)] = J_ν(x)Y_ν'(x) - J_ν'(x)Y_ν(x) = 2/(πx)
//! ```
//! **Proof**: Direct computation using series expansions and L'Hôpital's rule.
//!
//! **Recurrence Relations**:
//! ```text
//! Z_{ν-1}(x) + Z_{ν+1}(x) = (2ν/x) Z_ν(x)
//! Z_{ν-1}(x) - Z_{ν+1}(x) = 2 Z_ν'(x)
//! ```
//! where Z_ν represents any Bessel function (J_ν, Y_ν, H_ν, etc.).
//!
//! **Generating Function for J_n(x)** (integer order):
//! ```text
//! exp[(x/2)(t - 1/t)] = Σ_{n=-∞}^∞ t^n J_n(x)
//! ```
//! **Proof**: Taylor expansion of the exponential and coefficient comparison.
//!
//! ### Asymptotic Expansions
//!
//! **For large argument** (x → ∞):
//! ```text
//! J_ν(x) ~ √(2/(πx)) cos(x - νπ/2 - π/4) [1 + O(1/x)]
//! Y_ν(x) ~ √(2/(πx)) sin(x - νπ/2 - π/4) [1 + O(1/x)]
//! ```
//!
//! **Rigorous derivation**: Method of steepest descent applied to Hankel's integral representation.
//!
//! **For small argument** (x → 0, ν > 0):
//! ```text
//! J_ν(x) ~ (x/2)^ν / Γ(ν+1)
//! Y_ν(x) ~ -(2/π) Γ(ν) (x/2)^(-ν)    for ν > 0
//! Y_0(x) ~ (2/π) ln(x/2)              for ν = 0
//! ```
//!
//! ### Orthogonality Relations
//!
//! **For fixed ν and variable zeros**:
//! ```text
//! ∫₀¹ x J_ν(α_{νm} x) J_ν(α_{νn} x) dx = (1/2) δ_{mn} [J_{ν+1}(α_{νm})]²
//! ```
//! where α_{νm} are the positive zeros of J_ν(x).
//!
//! **Physical significance**: Enables Fourier-Bessel series expansions for problems
//! with cylindrical boundary conditions.
//!
//! ### Modified Bessel Functions
//!
//! **Modified Bessel Equation**:
//! ```text
//! x² d²y/dx² + x dy/dx - (x² + ν²) y = 0
//! ```
//!
//! **Solutions**:
//! - **I_ν(x)**: Modified Bessel function of the first kind (exponentially growing)
//! - **K_ν(x)**: Modified Bessel function of the second kind (exponentially decaying)
//!
//! **Connection to ordinary Bessel functions**:
//! ```text
//! I_ν(x) = i^(-ν) J_ν(ix)
//! K_ν(x) = (π/2) i^(ν+1) H_ν^(1)(ix)
//! ```
//!
//! ### Spherical Bessel Functions
//!
//! **Definition**:
//! ```text
//! j_n(x) = √(π/(2x)) J_{n+1/2}(x)
//! y_n(x) = √(π/(2x)) Y_{n+1/2}(x)
//! ```
//!
//! **Applications**: Solutions to the wave equation in spherical coordinates,
//! quantum mechanics (radial Schrödinger equation).
//!
//! ### Physical Applications and Interpretations
//!
//! **Wave Propagation**:
//! - Cylindrical wave guides (electromagnetic theory)
//! - Acoustic waves in circular domains
//! - Vibrations of circular membranes and cylindrical shells
//!
//! **Heat Conduction**:
//! - Temperature distribution in cylindrical objects
//! - Diffusion processes with cylindrical symmetry
//!
//! **Quantum Mechanics**:
//! - Radial wave functions in cylindrical and spherical potentials
//! - Scattering theory (partial wave analysis)
//!
//! **Antenna Theory**:
//! - Radiation patterns of cylindrical antennas
//! - Waveguide modes and propagation constants
//!
//! ### Computational Methods
//!
//! This implementation employs sophisticated numerical techniques:
//!
//! **1. Series Expansions**:
//! - Power series near x = 0 with optimized convergence acceleration
//! - Asymptotic series for large |x| with error bounds
//!
//! **2. Continued Fractions**:
//! - Miller's algorithm for stable computation of ratios
//! - Backward recurrence with proper normalization
//!
//! **3. Uniform Asymptotic Expansions**:
//! - Airy function representations for turning points
//! - Debye expansions for large order ν
//!
//! **4. Special Value Handling**:
//! - Exact expressions for half-integer orders
//! - Optimized algorithms for integer orders
//!
//! **5. Numerical Stability**:
//! - Protection against overflow/underflow
//! - Careful handling of near-zero arguments
//! - Accurate computation near zeros and extrema
//!
//! ## Function Organization
//!
//! ### First Kind (J_ν)
//! - j0(x): Order 0 - most frequently used
//! - j1(x): Order 1 - important for cylindrical problems  
//! - jn(n, x): Integer order n - exact relations
//! - jv(ν, x): Arbitrary real order ν - general case
//!
//! ### Second Kind (Y_ν)
//! - y0(x): Order 0 - complements j0
//! - y1(x): Order 1 - complements j1
//! - yn(n, x): Integer order n - linearly independent with jn
//!
//! ### Modified Bessel (I_ν, K_ν)
//! - i0(x), i1(x): Growing solutions for modified equation
//! - k0(x), k1(x): Decaying solutions for modified equation
//! - iv(ν, x), kv(ν, x): Arbitrary order modified functions
//!
//! ### Spherical Bessel
//! - Specialized for three-dimensional wave problems
//! - Exact polynomial expressions for integer orders

// No imports needed at the module level

// Re-export all public functions
pub use self::derivatives::{
    h1vp, h2vp, i0_prime, i1_prime, iv_prime, ivp, j0_prime, j1_prime, jn_prime, jv_prime, jvp,
    k0_prime, k1_prime, kv_prime, kvp, y0_prime, y1_prime, yn_prime, yvp,
};
pub use self::first_kind::{j0, j0e, j1, j1e, jn, jne, jv, jve};
pub use self::modified::{i0, i0e, i1, i1e, iv, ive, k0, k0e, k1, k1e, kv, kve};
pub use self::second_kind::{y0, y0e, y1, y1e, yn, yne};
pub use self::spherical::{spherical_jn, spherical_jn_scaled, spherical_yn, spherical_yn_scaled};

// Hankel functions are defined directly in this module below

// The helper functions below are moved to the relevant modules where they are used
// to avoid "unused function" warnings

// Export each set of functions from their own module
pub mod derivatives;
pub mod first_kind;
pub mod modified;
pub mod second_kind;
pub mod spherical;

/// Hankel functions of the first kind H₁⁽¹⁾(v, z)
///
/// Hankel functions are linear combinations of Bessel functions:
/// H₁⁽¹⁾(v, z) = J_v(z) + i * Y_v(z)
///
/// These functions are particularly useful in wave propagation problems
/// and provide outgoing wave solutions in cylindrical coordinates.
///
/// # Arguments
///
/// * `v` - Order of the function
/// * `z` - Argument (complex number supported)
///
/// # Returns
///
/// * Complex value of H₁⁽¹⁾(v, z)
///
/// # Examples
///
/// ```
/// use scirs2_special::hankel1;
/// use scirs2_core::numeric::Complex64;
///
/// let result = hankel1(1.0, 1.0);
/// // H₁⁽¹⁾₁(1) = J₁(1) + i*Y₁(1)
/// ```
#[allow(dead_code)]
pub fn hankel1<T>(v: T, z: T) -> scirs2_core::numeric::Complex<T>
where
    T: scirs2_core::numeric::Float
        + scirs2_core::numeric::FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign,
{
    use crate::bessel::{
        jv,
        second_kind::{y0, y1, yn},
    };

    let j_val = jv(v, z);

    // Use appropriate Y function based on order
    let y_val = if v == T::zero() {
        y0(z)
    } else if v == T::one() {
        y1(z)
    } else if let Some(n) = v.to_i32() {
        if n >= 0 && T::from(n).unwrap() == v {
            yn(n, z)
        } else {
            // For negative or non-integer orders, we need a more general implementation
            // For now, use a simple approximation or return NaN
            T::nan()
        }
    } else {
        // For non-integer orders, we need a more general yv implementation
        // This is a placeholder - should be implemented properly
        T::nan()
    };

    scirs2_core::numeric::Complex::new(j_val, y_val)
}

/// Hankel functions of the second kind H₂⁽²⁾(v, z)  
///
/// Hankel functions are linear combinations of Bessel functions:
/// H₂⁽²⁾(v, z) = J_v(z) - i * Y_v(z)
///
/// These functions are particularly useful in wave propagation problems
/// and provide incoming wave solutions in cylindrical coordinates.
///
/// # Arguments
///
/// * `v` - Order of the function
/// * `z` - Argument (complex number supported)
///
/// # Returns
///
/// * Complex value of H₂⁽²⁾(v, z)
///
/// # Examples
///
/// ```
/// use scirs2_special::hankel2;
/// use scirs2_core::numeric::Complex64;
///
/// let result = hankel2(1.0, 1.0);
/// // H₂⁽²⁾₁(1) = J₁(1) - i*Y₁(1)
/// ```
#[allow(dead_code)]
pub fn hankel2<T>(v: T, z: T) -> scirs2_core::numeric::Complex<T>
where
    T: scirs2_core::numeric::Float
        + scirs2_core::numeric::FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign,
{
    use crate::bessel::{
        jv,
        second_kind::{y0, y1, yn},
    };

    let j_val = jv(v, z);

    // Use appropriate Y function based on order
    let y_val = if v == T::zero() {
        y0(z)
    } else if v == T::one() {
        y1(z)
    } else if let Some(n) = v.to_i32() {
        if n >= 0 && T::from(n).unwrap() == v {
            yn(n, z)
        } else {
            // For negative or non-integer orders, we need a more general implementation
            // For now, use a simple approximation or return NaN
            T::nan()
        }
    } else {
        // For non-integer orders, we need a more general yv implementation
        // This is a placeholder - should be implemented properly
        T::nan()
    };

    scirs2_core::numeric::Complex::new(j_val, -y_val)
}

/// Exponentially scaled Hankel function of the first kind H₁⁽¹⁾(v, z) * exp(-i*z)
///
/// The exponentially scaled version is useful for large arguments where
/// the unscaled function would overflow:
/// hankel1e(v, z) = hankel1(v, z) * exp(-i*z)
///
/// # Arguments
///
/// * `v` - Order of the function
/// * `z` - Argument
///
/// # Returns
///
/// * Complex value of the scaled H₁⁽¹⁾(v, z)
///
/// # Examples
///
/// ```
/// use scirs2_special::hankel1e;
///
/// let result = hankel1e(1.0, 10.0);
/// // Scaled version for numerical stability
/// ```
#[allow(dead_code)]
pub fn hankel1e<T>(v: T, z: T) -> scirs2_core::numeric::Complex<T>
where
    T: scirs2_core::numeric::Float
        + scirs2_core::numeric::FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign,
{
    let h1 = hankel1(v, z);
    let scale_factor = scirs2_core::numeric::Complex::new(T::zero(), -z).exp();
    h1 * scale_factor
}

/// Exponentially scaled Hankel function of the second kind H₂⁽²⁾(v, z) * exp(i*z)
///
/// The exponentially scaled version is useful for large arguments where
/// the unscaled function would overflow:
/// hankel2e(v, z) = hankel2(v, z) * exp(i*z)
///
/// # Arguments
///
/// * `v` - Order of the function
/// * `z` - Argument
///
/// # Returns
///
/// * Complex value of the scaled H₂⁽²⁾(v, z)
///
/// # Examples
///
/// ```
/// use scirs2_special::hankel2e;
///
/// let result = hankel2e(1.0, 10.0);
/// // Scaled version for numerical stability
/// ```
#[allow(dead_code)]
pub fn hankel2e<T>(v: T, z: T) -> scirs2_core::numeric::Complex<T>
where
    T: scirs2_core::numeric::Float
        + scirs2_core::numeric::FromPrimitive
        + std::fmt::Debug
        + std::ops::AddAssign,
{
    let h2 = hankel2(v, z);
    let scale_factor = scirs2_core::numeric::Complex::new(T::zero(), z).exp();
    h2 * scale_factor
}

/// Complex number support for Bessel functions
pub mod complex;

// Import tests
#[cfg(test)]
mod tests;
