//! Sparse Fast Fourier Transform Implementation
//!
//! This module provides implementations of Sparse FFT algorithms, which are
//! efficient for signals that have a sparse representation in the frequency domain.
//! These algorithms can achieve sub-linear runtime when the signal has only a few
//! significant frequency components.
//!
//! # Module Organization
//!
//! * [`config`] - Configuration types and enums for sparse FFT
//! * [`algorithms`] - Core sparse FFT algorithm implementations
//! * [`windowing`] - Window function utilities
//! * [`estimation`] - Sparsity estimation methods
//! * [`reconstruction`] - Spectrum reconstruction utilities
//!
//! # Examples
//!
//! ```rust
//! use scirs2_fft::sparse_fft::{sparse_fft, SparseFFTConfig, SparsityEstimationMethod};
//!
//! // Create a sparse signal
//! let signal = vec![1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25, 0.0];
//!
//! // Compute sparse FFT with 4 components
//! let result = sparse_fft(&signal, 4, None, None).unwrap();
//!
//! println!("Found {} sparse components", result.values.len());
//! ```

pub mod algorithms;
pub mod config;
pub mod estimation;
pub mod reconstruction;
pub mod windowing;

#[cfg(test)]
mod tests;

// Re-export main types and functions for backward compatibility
pub use algorithms::{SparseFFT, SparseFFTResult};
pub use config::{SparseFFTAlgorithm, SparseFFTConfig, SparsityEstimationMethod, WindowFunction};

// Re-export main public API functions
pub use algorithms::{
    adaptive_sparse_fft, frequency_pruning_sparse_fft, sparse_fft, sparse_fft2, sparse_fftn,
    spectral_flatness_sparse_fft,
};
pub use reconstruction::{
    reconstruct_filtered, reconstruct_high_resolution, reconstruct_spectrum,
    reconstruct_time_domain,
};

// Re-export utilities
pub use config::try_as_complex;
