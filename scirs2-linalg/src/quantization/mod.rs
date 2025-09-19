//! Quantization-aware linear algebra operations
//!
//! This module provides functions and types for working with quantized matrices and vectors.
//! Quantization reduces the precision of floating-point numbers to save memory and
//! computational resources, which is particularly useful in machine learning applications.
//!
//! ## Overview
//!
//! * Quantization of matrices and vectors to lower bit-width representations
//! * Linear algebra operations on quantized data
//! * Support for different quantization methods (uniform, symmetric, affine)
//! * Efficient operations with mixed quantized and floating-point data
//!
//! ## Examples
//!
//! Basic quantization:
//!
//! ```
//! use ndarray::{Array2, array};
//! use scirs2_linalg::quantization::{quantize_matrix, dequantize_matrix, QuantizationMethod};
//!
//! let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];
//!
//! // Quantize to 8-bit
//! let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Affine);
//!
//! // Dequantize back to floating point
//! let a_dequantized = dequantize_matrix(&quantized, &params);
//!
//! // Check the error exists but is bounded
//! let max_error = (&a - &a_dequantized).mapv(|x| x.abs()).fold(0.0_f32, |acc, &b| acc.max(b));
//! assert!(max_error > 0.0); // There should be some quantization error
//! assert!(max_error < 10.0); // But it should be bounded
//! ```
//!
//! Quantized matrix multiplication:
//!
//! ```
//! use ndarray::{Array2, array};
//! use scirs2_linalg::quantization::{quantize_matrix, QuantizationMethod, quantized_matmul};
//!
//! let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
//! let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
//!
//! // Quantize both matrices to 8-bit
//! let (a_q, a_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
//! let (b_q, b_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);
//!
//! // Perform quantized matrix multiplication
//! let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();
//!
//! // Regular matrix multiplication for comparison
//! let c = a.dot(&b);
//!
//! // Check the error is acceptable
//! let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();
//! assert!(rel_error < 0.1); // Relative error should be small
//! ```

// Core type definitions
pub mod types;

// Data structure modules
pub mod matrix;
pub mod vector;

// Function modules
pub mod conversions;
pub mod operations;

// Test module
#[cfg(test)]
pub mod tests;

// Existing submodules
pub mod calibration;
pub mod calibration_ema;
pub mod fusion;
pub mod out_of_core;
pub mod quantized_matrixfree;
pub mod simd;
pub mod solvers;
pub mod stability;

// Re-export all public types and functions for backward compatibility

// Types
pub use self::types::{QuantizationMethod, QuantizationParams, QuantizedDataType};

// Matrix types and functions
pub use self::matrix::{get_quantizedmatrix_2d_i8, QuantizedData2D, QuantizedMatrix};

// Vector types and functions
pub use self::vector::{get_quantized_vector_1d_i8, QuantizedData1D, QuantizedVector};

// Conversion functions
pub use self::conversions::{
    dequantize_matrix,
    // TODO: Add these when implemented:
    // quantize_vector,
    // dequantize_vector,
    // fake_quantize,
    // fake_quantize_vector,
    quantize_matrix,
    quantize_matrix_per_channel,
};

// Operation functions
pub use self::operations::{
    quantized_matmul,
    // TODO: Add these when implemented:
    // quantized_matvec,
    // quantized_dot,
};

// Re-export submodule public APIs
pub use calibration::*;
pub use calibration_ema::*;
pub use fusion::*;
pub use out_of_core::*;
pub use quantized_matrixfree::*;
pub use simd::*;
pub use solvers::*;
pub use stability::*;
