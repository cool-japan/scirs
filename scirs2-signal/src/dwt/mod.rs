//! Discrete Wavelet Transform (DWT)
//!
//! This module provides implementations of the Discrete Wavelet Transform (DWT),
//! inverse DWT, and associated wavelet filters. The DWT is useful for
//! multi-resolution analysis, denoising, and compression of signals.
//!
//! The module is organized into submodules:
//! - `filters`: Wavelet filter definitions and generation functions
//! - `transform`: Core DWT decomposition and reconstruction functions
//! - `boundary`: Signal extension methods for handling boundary conditions
//! - `multiscale`: Multi-level transform functions for decomposition and reconstruction

// Declare submodules
mod boundary;
mod filters;
mod multiscale;
mod transform;

// Re-export public items from submodules
pub use filters::{Wavelet, WaveletFilters};
pub use multiscale::{wavedec, waverec};
pub use transform::{dwt_decompose, dwt_reconstruct};

// Re-export boundary extension for advanced users
pub use boundary::extend_signal;
