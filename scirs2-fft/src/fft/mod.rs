/*!
 * Fast Fourier Transform (FFT) module
 *
 * This module provides functions for computing the Fast Fourier Transform (FFT)
 * and its inverse (IFFT) in 1D, 2D, and N-dimensional cases.
 */

// Private modules
mod algorithms;
mod planning;
mod utility;
// Windowing module now public for doctest access
pub mod windowing;

// Re-export the core FFT algorithms
pub use algorithms::{fft, fft2, fftn, ifft, ifft2, ifftn};

// Re-export the parallel FFT implementations
pub use planning::{fft2_parallel, ifft2_parallel};

// We don't need to re-export NormMode since it's not used elsewhere
// pub(crate) use algorithms::NormMode;

// Re-export useful utility functions
pub use utility::{
    complex_angle, complex_magnitude, complex_to_real, is_power_of_two, next_power_of_two,
    power_spectrum, real_to_complex, validate_fft_axes, validate_fft_shapes, validate_fft_size,
};

// Re-export window functions
pub use windowing::{apply_window, create_window, window_properties, WindowProperties, WindowType};
