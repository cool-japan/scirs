//! Spheroidal wave functions
//!
//! This module provides implementations of spheroidal wave functions, which arise in the solution
//! of the Helmholtz equation in prolate and oblate spheroidal coordinates. These functions are
//! fundamental in mathematical physics, particularly in electromagnetic scattering theory,
//! quantum mechanics, and acoustic wave propagation.
//!
//! ## Mathematical Background
//!
//! ### Helmholtz Equation in Spheroidal Coordinates
//!
//! The Helmholtz equation ∇²u + k²u = 0 in prolate spheroidal coordinates (ξ, η, φ)
//! with semi-focal distance c = ka (where k is the wave number and a is the semi-focal distance)
//! separates into three ordinary differential equations:
//!
//! 1. **Angular equation**: (1-η²)d²S/dη² - 2η dS/dη + [λ - c²η²]S = 0
//! 2. **Radial equation**: (ξ²-1)d²R/dξ² + 2ξ dR/dξ - [λ - c²ξ²]R = 0
//! 3. **Azimuthal equation**: d²Φ/dφ² + m²Φ = 0
//!
//! where λ is the characteristic value (eigenvalue) and m is the azimuthal quantum number.
//!
//! ### Characteristic Values λₘₙ(c)
//!
//! The characteristic values are determined by the requirement that the angular functions
//! be finite at η = ±1. They satisfy the infinite system of linear equations:
//!
//! For prolate functions:
//! ```text
//! (αᵣ - λ)aᵣ + βᵣ₊₁aᵣ₊₂ + βᵣ₋₁aᵣ₋₂ = 0
//! ```
//! where αᵣ = (r+m)(r+m+1) and βᵣ = c²/[4(2r+1)(2r+3)] for the recurrence coefficients.
//!
//! ### Asymptotic Behavior
//!
//! **Small c expansion (perturbation theory):**
//! λₘₙ(c) ≈ n(n+1) + c²/[2(2n+3)] + O(c⁴)
//!
//! **Large c asymptotic expansion:**
//! λₘₙ(c) ≈ -c²/4 + (2n+1)c + n(n+1) - m²/2 + O(1/c)
//!
//! ## Function Types
//!
//! ### Prolate Spheroidal Functions
//! - **Angular functions Sₘₙ(c,η)**: Solutions regular at η = ±1
//! - **Radial functions of first kind Rₘₙ⁽¹⁾(c,ξ)**: Regular at ξ = 1
//! - **Radial functions of second kind Rₘₙ⁽²⁾(c,ξ)**: Irregular at ξ = 1
//!
//! ### Oblate Spheroidal Functions
//! - **Angular functions Sₘₙ(-ic,η)**: Solutions regular at η = ±1
//! - **Radial functions of first kind Rₘₙ⁽¹⁾(-ic,ξ)**: Regular at ξ = 0
//! - **Radial functions of second kind Rₘₙ⁽²⁾(-ic,ξ)**: Irregular at ξ = 0
//!
//! ## Physical Applications
//!
//! ### Electromagnetic Scattering
//! - **Prolate spheroids**: Cigar-shaped objects (a > b = c)
//! - **Oblate spheroids**: Disk-shaped objects (a = b > c)
//! - **Mie scattering**: Extensions for non-spherical particles
//!
//! ### Quantum Mechanics
//! - **Molecular orbitals**: Diatomic molecules in prolate coordinates
//! - **Nuclear physics**: Deformed nuclei modeling
//! - **Stark effect**: Hydrogen atom in electric fields
//!
//! ### Acoustics and Vibrations
//! - **Sound scattering**: By prolate/oblate objects
//! - **Modal analysis**: Vibrations of spheroidal resonators
//! - **Underwater acoustics**: Sonar applications
//!
//! ## Numerical Considerations
//!
//! ### Stability and Accuracy
//! - Functions computed using continued fractions for moderate parameters
//! - Asymptotic expansions for large |c|
//! - Special handling near parameter boundaries
//! - Automatic precision scaling for extreme values
//!
//! ### Performance Optimizations
//! - Cached characteristic values for repeated computations
//! - Vectorized operations for array inputs
//! - Adaptive algorithms based on parameter ranges
//! - Memory-efficient implementations for large-scale computations

pub mod cf_helpers;
pub mod helpers;
pub mod oblate;
pub mod prolate;
pub mod swf;
pub mod wave_functions;

// Re-export all public functions
pub use helpers::*;
pub use oblate::*;
pub use prolate::*;
pub use swf::{
    spheroidal_eigenvalue_mn, spheroidal_ps, spheroidal_wronskian, SpheroidalEigenvalue,
    SpheroidalKind,
};
pub use wave_functions::{
    angular_spheroidal, associated_legendre, radial_spheroidal_1, spherical_bessel_j,
    spheroidal_eigenvalue as spheroidal_eigenvalue_wf, SpheroidType, SpheroidalConfig,
};
