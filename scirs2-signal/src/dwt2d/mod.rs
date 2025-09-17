//! Two-dimensional Discrete Wavelet Transform (2D DWT) Module
//!
//! This module provides comprehensive implementations of the 2D Discrete Wavelet Transform,
//! designed for image processing, compression, and multi-resolution analysis of 2D signals.
//!
//! # Overview
//!
//! The 2D Discrete Wavelet Transform (DWT2D) extends the concept of 1D wavelet transforms
//! to two-dimensional data. It is particularly useful for processing images and other 2D signals.
//! By using separable filters, the 2D DWT applies 1D transforms first along rows and then along
//! columns, decomposing the image into four subbands:
//!
//! - **LL (Approximation)**: Low frequencies in both horizontal and vertical directions
//! - **LH (Horizontal Details)**: Low frequencies horizontally, high frequencies vertically
//! - **HL (Vertical Details)**: High frequencies horizontally, low frequencies vertically
//! - **HH (Diagonal Details)**: High frequencies in both directions
//!
//! # Performance Features
//!
//! This implementation includes several optimizations for performance:
//!
//! ## Parallel Processing
//! When compiled with the "parallel" feature, row and column transforms can be computed
//! in parallel using Rayon for improved performance on multi-core systems.
//!
//! ## Memory Efficiency
//! - Minimizes temporary allocations through memory pooling
//! - Uses ndarray views for zero-copy operations
//! - Implements cache-friendly traversal patterns
//! - Configurable memory alignment for SIMD operations
//!
//! ## Algorithm Optimizations
//! - Direct transform paths for common wavelets (Haar, DB2, DB4)
//! - Optimized convolution for filter operations
//! - Efficient boundary handling strategies
//! - SIMD acceleration where available
//!
//! # Module Structure
//!
//! This module is organized into several submodules:
//!
//! - [`types`] - Core data structures and type definitions
//! - [`decomposition`] - 2D decomposition algorithms and functions
//! - Future modules will include:
//!   - `reconstruction` - 2D reconstruction algorithms
//!   - `thresholding` - Coefficient thresholding operations
//!   - `validation` - Comprehensive validation and testing
//!   - `utils` - Utility functions and helpers
//!
//! # Examples
//!
//! ## Basic 2D DWT Decomposition
//!
//! ```rust,no_run
//! use ndarray::Array2;
//! use scirs2_signal::dwt::Wavelet;
//! use scirs2_signal::dwt2d::{dwt2d_decompose, Dwt2dResult};
//!
//! // Create a sample 8x8 image
//! let mut image = Array2::zeros((8, 8));
//! for i in 0..8 {
//!     for j in 0..8 {
//!         image[[i, j]] = ((i + j) % 4) as f64;
//!     }
//! }
//!
//! // Perform 2D DWT decomposition
//! let result: Dwt2dResult = dwt2d_decompose(&image, Wavelet::Haar, None)?;
//!
//! // Access the four subbands
//! println!("Approximation shape: {:?}", result.approx.shape());
//! println!("Horizontal details shape: {:?}", result.detail_h.shape());
//! println!("Vertical details shape: {:?}", result.detail_v.shape());
//! println!("Diagonal details shape: {:?}", result.detail_d.shape());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Advanced Configuration
//!
//! ```rust,no_run
//! use scirs2_signal::dwt2d::{Dwt2dConfig, dwt2d_decompose_with_config};
//! use scirs2_signal::dwt::Wavelet;
//! use ndarray::Array2;
//!
//! // Create custom configuration for memory optimization
//! let config = Dwt2dConfig {
//!     preallocate_memory: true,
//!     use_inplace: false,
//!     memory_alignment: 64,  // AVX-512 alignment
//!     chunk_size: Some(2 * 1024 * 1024),  // 2MB chunks
//! };
//!
//! let image = Array2::zeros((256, 256));
//! let result = dwt2d_decompose_with_config(&image, Wavelet::DB(4), Some(&config))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Thresholding for Denoising
//!
//! ```rust,no_run
//! use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct, threshold_dwt2d, ThresholdMethod};
//! use scirs2_signal::dwt::Wavelet;
//! use ndarray::Array2;
//!
//! let noisy_image = Array2::zeros((64, 64)); // Your noisy image here
//!
//! // Decompose the noisy image
//! let mut decomposition = dwt2d_decompose(&noisy_image, Wavelet::DB(6), None)?;
//!
//! // Apply soft thresholding to remove noise
//! threshold_dwt2d(&mut decomposition, 0.1, ThresholdMethod::Soft);
//!
//! // Reconstruct the denoised image
//! let denoised = dwt2d_reconstruct(&decomposition, Wavelet::DB(6), noisy_image.raw_dim())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Error Handling
//!
//! All functions in this module return `Result` types with descriptive error messages.
//! Common error conditions include:
//!
//! - Invalid input dimensions (must be even for single-level decomposition)
//! - Mismatched dimensions during reconstruction
//! - Unsupported wavelet types for certain operations
//! - Memory allocation failures for large images
//!
//! # Thread Safety
//!
//! This module is designed to be thread-safe. Multiple threads can safely perform
//! 2D DWT operations on different data simultaneously. Internal memory pools use
//! thread-local storage to avoid contention.
//!
//! # See Also
//!
//! - [`crate::dwt`] - 1D Discrete Wavelet Transform
//! - [`crate::swt2d`] - 2D Stationary Wavelet Transform
//! - [`crate::wpt2d`] - 2D Wavelet Packet Transform

// Declare submodules
pub mod types;
pub mod decomposition;
pub mod reconstruction;
pub mod thresholding;
pub mod simd;
pub mod validation;
pub mod utils;

#[cfg(test)]
mod tests;

// Re-export main public types for backward compatibility
pub use types::{
    Dwt2dConfig,
    Dwt2dResult,
    Dwt2dValidationConfig,
    Dwt2dValidationResult,
    MemoryEfficiencyMetrics,
    MemoryPool,
    PerformanceMetrics2d,
    ThresholdMethod,
    WaveletCounts,
    WaveletEnergy,
};

// Re-export main decomposition functions for backward compatibility
pub use decomposition::{
    dwt2d_decompose,
    dwt2d_decompose_optimized,
};

// Re-export reconstruction functions
pub use reconstruction::{
    dwt2d_reconstruct,
    wavedec2,
    waverec2,
};

// Re-export thresholding functions
pub use thresholding::{
    threshold_dwt2d,
    threshold_wavedec2,
    apply_threshold,
    apply_adaptive_thresholding,
    estimate_noise_variance,
    calculate_compression_ratio,
};

// Re-export SIMD functions
pub use simd::{
    simd_threshold_coefficients,
    simd_calculate_energy,
    PlatformCapabilities,
};

// Re-export validation functions
pub use validation::{
    validate_dwt2d_comprehensive,
    calculate_energy,
    count_nonzeros,
    validate_decomposition_level,
};

// Re-export utility functions
pub use utils::{
    calculate_psnr,
    calculate_ssim,
    dwt2d_decompose_adaptive,
    wavedec2_enhanced,
    denoise_dwt2d_adaptive,
    denoise_wavedec2_adaptive,
};

// Function implementations are organized in the following submodules:
//
// ## Core Modules
// - [`types`] - Core data structures and type definitions
// - [`decomposition`] - 2D wavelet decomposition algorithms
// - [`reconstruction`] - 2D wavelet reconstruction algorithms
// - [`thresholding`] - Coefficient thresholding and denoising operations
// - [`simd`] - SIMD-accelerated operations for performance
// - [`validation`] - Comprehensive validation and testing utilities
// - [`utils`] - Utility functions including image quality metrics and adaptive processing
//
// ## Module Organization
//
// ### Decomposition Module
// Contains functions for single-level and multi-level 2D DWT decomposition:
// - `dwt2d_decompose()` - Basic single-level decomposition
// - `dwt2d_decompose_optimized()` - Optimized decomposition with configuration
//
// ### Reconstruction Module
// Contains functions for reconstructing signals from wavelet coefficients:
// - `dwt2d_reconstruct()` - Single-level reconstruction
// - `wavedec2()` - Multi-level decomposition
// - `waverec2()` - Multi-level reconstruction
//
// ### Thresholding Module
// Contains coefficient processing functions for denoising and compression:
// - `threshold_dwt2d()` - Apply thresholding to single-level coefficients
// - `threshold_wavedec2()` - Apply thresholding to multi-level coefficients
// - `estimate_noise_variance()` - Noise estimation for adaptive thresholding
//
// ### SIMD Module
// Contains high-performance SIMD-accelerated operations:
// - `simd_threshold_coefficients()` - Vectorized thresholding
// - `simd_calculate_energy()` - Vectorized energy calculation
// - `PlatformCapabilities` - Runtime SIMD capability detection
//
// ### Validation Module
// Contains comprehensive testing and analysis functions:
// - `validate_dwt2d_comprehensive()` - Full validation suite
// - `calculate_energy()` - Energy conservation analysis
// - `count_nonzeros()` - Sparsity analysis
//
// ### Utils Module
// Contains utility functions for image processing and quality assessment:
// - `calculate_psnr()` - Peak Signal-to-Noise Ratio
// - `calculate_ssim()` - Structural Similarity Index
// - `denoise_dwt2d_adaptive()` - Adaptive denoising
// - `dwt2d_decompose_adaptive()` - Hardware-adaptive decomposition