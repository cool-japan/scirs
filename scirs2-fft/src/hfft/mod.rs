//! Hermitian Fast Fourier Transform (HFFT) module
//!
//! This module provides functions for computing the Hermitian Fast Fourier Transform (HFFT)
//! and its inverse (IHFFT). These functions handle complex-valued signals with real spectra.
//!
//! ## Implementation Notes
//!
//! The HFFT functions are particularly sensitive to numerical precision issues due to their
//! reliance on Hermitian symmetry properties. When using these functions:
//!
//! 1. **Normalization**: Pay close attention to the normalization parameter, as it significantly
//!    affects scaling in round-trip transformations.
//!
//! 2. **Precision**: Hermitian symmetry requires that the imaginary part of certain components
//!    be exactly zero, which may not be possible due to floating-point precision. The functions
//!    apply reasonable tolerances to handle these cases.
//!
//! 3. **Round-Trip Transformations**: When performing hfft followed by ihfft (or vice versa),
//!    you may need to apply scaling factors to recover the original signal amplitudes accurately.
//!
//! 4. **Multi-dimensional Transforms**: 2D and N-dimensional transforms have additional complexity
//!    regarding Hermitian symmetry across multiple dimensions. Care should be taken when working
//!    with these functions.

// Module declarations
mod complex_to_real;
mod real_to_complex;
mod symmetric;
mod utility;

// Re-exports
pub use complex_to_real::{hfft, hfft2, hfftn};
pub use real_to_complex::{ihfft, ihfft2, ihfftn};
pub use symmetric::{
    create_hermitian_symmetric_signal, enforce_hermitian_symmetry, enforce_hermitian_symmetry_nd,
    is_hermitian_symmetric,
};
