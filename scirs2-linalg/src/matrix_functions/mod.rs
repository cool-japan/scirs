//! Matrix functions such as matrix exponential, logarithm, and square root
//!
//! This module provides a comprehensive set of matrix functions organized into
//! logical categories. All functions maintain compatibility with the original
//! scirs2_linalg API while providing better code organization.

// Module declarations
pub mod advanced;
pub mod analysis;
pub mod control;
pub mod exponential;
pub mod fractional;
pub mod hyperbolic;
pub mod interpolation;
pub mod markov;
pub mod pade;
pub mod polar_decomp;
pub mod poly_roots;
pub mod roots;
pub mod sign_function;
pub mod special;
pub mod sqrt_matrix;
pub mod sylvester;
pub mod trig_schur;
pub mod trigonometric;
pub mod utils;

#[cfg(test)]
mod tests;

// Re-export all public functions to maintain API compatibility

// Exponential functions
pub use exponential::{expm, logm, logm_parallel, matrix_power, sqrtm, sqrtm_parallel};

// Trigonometric functions
pub use trigonometric::{acosm, asinm, atanm, cosm, sinm, tanm};

// Hyperbolic functions
pub use hyperbolic::{coshm, sinhm, tanhm};

// Special functions
pub use special::{sigmoid, signm, softmax};

// Fractional functions
pub use fractional::{fractionalmatrix_power, spdmatrix_function};

// Analysis functions
pub use analysis::{
    geometric_mean_spd, nuclear_norm, polar_decomposition, spectral_condition_number,
    spectral_radius, tikhonov_regularization,
};

// Markov chain functions
pub use markov::{
    absorption_probabilities, analyze_absorbing_chain, fundamental_matrix, is_stochastic,
    is_stochastic_f64, mean_first_passage_time, mixing_time, stationary_distribution,
};

// Utility functions (not re-exported to maintain clean public API)
// These are available as matrix_functions::utils::function_name if needed
