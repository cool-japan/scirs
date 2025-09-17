//! Enhanced 2D Discrete Wavelet Transform (DWT) Module
//!
//! This module provides a comprehensive, production-ready implementation of 2D DWT
//! with advanced features for scientific computing and signal processing applications.
//!
//! # Key Features
//!
//! ## Performance Optimizations
//! - **SIMD Acceleration**: Leverages SIMD instructions for up to 8x speedup
//! - **Parallel Processing**: Multi-threaded operations for large datasets
//! - **Memory Optimization**: Block-based processing for arbitrarily large images
//! - **Adaptive Processing**: Intelligent selection of optimization strategies
//!
//! ## Reconstruction Capabilities
//! - **Enhanced Reconstruction**: Advanced 2D DWT reconstruction with error correction
//! - **Parallel Reconstruction**: Multi-threaded reconstruction for large images
//! - **SIMD-Optimized**: Vectorized reconstruction operations for maximum performance
//! - **Multilevel Support**: Complete multilevel DWT reconstruction
//! - **Quality Analysis**: Edge preservation and reconstruction quality metrics
//!
//! ## Advanced Boundary Handling
//! - 11 sophisticated boundary modes including adaptive content-aware padding
//! - Symmetric, periodic, anti-symmetric, and smooth extensions
//! - Content-aware and gradient-based extrapolation
//! - Minimal edge artifacts and perfect reconstruction guarantees
//!
//! ## Quality Assessment
//! - Comprehensive quality metrics (energy preservation, compression ratio)
//! - Sparsity and edge preservation analysis
//! - Statistical validation and coefficient analysis
//! - Entropy-based decomposition control
//!
//! ## Robust Denoising
//! - Multiple denoising algorithms (SURE, BayesShrink, BiShrink)
//! - Non-local means in wavelet domain
//! - Adaptive threshold selection
//! - Noise standard deviation estimation
//!
//! ## Production Features
//! - Comprehensive error handling and validation
//! - Configurable precision and tolerance settings
//! - Memory-efficient processing for large datasets
//! - Cross-platform compatibility
//!
//! # Usage Examples
//!
//! ## Basic 2D DWT Decomposition
//! ```rust
//! use scirs2_signal::dwt2d_enhanced::{enhanced_dwt2d_decompose, Dwt2dConfig, BoundaryMode};
//! use scirs2_signal::dwt::Wavelet;
//! use ndarray::Array2;
//!
//! let data = Array2::zeros((128, 128));
//! let config = Dwt2dConfig {
//!     boundary_mode: BoundaryMode::Symmetric,
//!     use_simd: true,
//!     use_parallel: true,
//!     compute_metrics: true,
//!     ..Default::default()
//! };
//!
//! let result = enhanced_dwt2d_decompose(&data, Wavelet::Daubechies4, &config)?;
//! println!("Approximation shape: {:?}", result.approx.dim());
//! ```
//!
//! ## 2D DWT Reconstruction
//! ```rust
//! use scirs2_signal::dwt2d_enhanced::{
//!     enhanced_dwt2d_decompose, enhanced_dwt2d_reconstruct, Dwt2dConfig
//! };
//! use scirs2_signal::dwt::Wavelet;
//! use ndarray::Array2;
//!
//! let data = Array2::zeros((128, 128));
//! let config = Dwt2dConfig::default();
//!
//! // Decompose
//! let result = enhanced_dwt2d_decompose(&data, Wavelet::Daubechies4, &config)?;
//!
//! // Reconstruct
//! let reconstructed = enhanced_dwt2d_reconstruct(&result, Wavelet::Daubechies4, &config)?;
//! println!("Original shape: {:?}, Reconstructed shape: {:?}",
//!          data.dim(), reconstructed.dim());
//! ```
//!
//! ## Multilevel Decomposition and Reconstruction
//! ```rust
//! use scirs2_signal::dwt2d_enhanced::{wavedec2_enhanced, waverec2_enhanced, Dwt2dConfig};
//! use scirs2_signal::dwt::Wavelet;
//! use ndarray::Array2;
//!
//! let data = Array2::zeros((256, 256));
//! let config = Dwt2dConfig::default();
//! let levels = 3;
//!
//! // Multilevel decomposition
//! let multilevel = wavedec2_enhanced(&data, Wavelet::Biorthogonal2_2, levels, &config)?;
//! println!("Decomposed into {} levels", multilevel.details.len());
//!
//! // Multilevel reconstruction
//! let reconstructed = waverec2_enhanced(&multilevel)?;
//! println!("Perfect reconstruction achieved: {}",
//!          (data - &reconstructed).mapv(|x| x.abs()).sum() < 1e-10);
//! ```
//!
//!
//! ## Advanced Denoising
//! ```rust
//! use scirs2_signal::dwt2d_enhanced::{
//!     adaptive_wavelet_denoising, DenoisingMethod
//! };
//! use scirs2_signal::dwt::Wavelet;
//! use ndarray::Array2;
//!
//! let noisy_data = Array2::zeros((128, 128));
//!
//! let denoised = adaptive_wavelet_denoising(
//!     &noisy_data,
//!     Wavelet::DB(8),
//!     None, // Auto noise estimation
//!     DenoisingMethod::BayesShrink
//! )?;
//! ```
//!
//! # Module Organization
//!
//! This module is organized into several submodules for better code organization:
//! - `types`: All type definitions, enums, and configuration structures
//! - `decomposition`: Core 2D DWT decomposition algorithms and multilevel operations
//! - `reconstruction`: Advanced 2D DWT reconstruction with SIMD and parallel optimizations
//! - `denoising`: Advanced wavelet-based denoising algorithms (SURE, BayesShrink, BiShrink, Non-local means)
//! - `boundary`: Enhanced boundary handling implementations with content-aware padding
//! - `validation`: Comprehensive validation and testing utilities for DWT results
//! - `statistics`: Statistical analysis utilities for wavelet decompositions
//!
//! # Performance Considerations
//!
//! - For small images (< 64x64), disable parallel processing for better performance
//! - SIMD operations provide significant speedup on modern CPUs
//! - Memory-optimized mode is recommended for images larger than available RAM
//! - Block size should be tuned based on cache size and memory bandwidth
//!
//! # Thread Safety
//!
//! All operations in this module are thread-safe and can be used concurrently.
//! The parallel processing features use work-stealing scheduling for optimal
//! load balancing across available CPU cores.

// Type definitions
pub mod types;

// Core decomposition algorithms
pub mod decomposition;

// Reconstruction algorithms
pub mod reconstruction;

// Denoising algorithms
pub mod denoising;

// Boundary handling implementations
pub mod boundary;

// Validation and testing utilities
pub mod validation;

// Statistical analysis utilities
pub mod statistics;

// Re-export all public types for backward compatibility and convenience
pub use types::{
    BoundaryMode, DenoisingMethod, Dwt2dConfig, Dwt2dQualityMetrics, Dwt2dStatistics,
    EnhancedDwt2dResult, MultilevelDwt2d,
};

// Additional modules for quality metrics and utilities:
// pub mod quality;        // Quality metrics computation (to be added later)
// pub mod utils;          // Common utilities and helpers (to be added later)
// Re-export main decomposition functions for backward compatibility
pub use decomposition::{enhanced_dwt2d_adaptive, enhanced_dwt2d_decompose, wavedec2_enhanced};

// Re-export main reconstruction functions for backward compatibility
pub use reconstruction::{enhanced_dwt2d_reconstruct, waverec2_enhanced};

// Re-export main denoising functions for backward compatibility
pub use denoising::adaptive_wavelet_denoising;

// Re-export boundary handling functions
pub use boundary::enhanced_boundary_padding;

// Re-export statistical analysis functions
pub use statistics::compute_enhanced_dwt2d_statistics;
