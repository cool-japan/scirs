//! Common types and utilities for convolutional and pooling layers
//!
//! This module provides shared types, enums, and utility functions used across
//! all convolutional and pooling layer implementations including padding modes,
//! type aliases for caching, and common validation functions.

use scirs2_core::ndarray::{Array, IxDyn};
use std::sync::{Arc, RwLock};

/// Type alias for caching max indices in 2D pooling operations
pub type MaxIndicesCache = Arc<RwLock<Option<Array<(usize, usize), IxDyn>>>>;

/// Type alias for caching max indices in 3D pooling operations  
pub type MaxIndicesCache3D = Arc<RwLock<Option<Array<(usize, usize, usize), IxDyn>>>>;

/// Padding mode for convolutional layers
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PaddingMode {
    /// No padding (will reduce spatial dimensions)
    #[default]
    Valid,
    /// Padding to preserve spatial dimensions
    Same,
    /// Custom padding values
    Custom(usize),
}

impl PaddingMode {
    /// Calculate padding values for a given kernel size and dilation
    pub fn calculate_padding(
        &self,
        kernel_size: (usize, usize),
        dilation: (usize, usize),
    ) -> (usize, usize) {
        match self {
            PaddingMode::Valid => (0, 0),
            PaddingMode::Same => (
                (kernel_size.0 - 1) * dilation.0 / 2,
                (kernel_size.1 - 1) * dilation.1 / 2,
            ),
            PaddingMode::Custom(pad) => (*pad, *pad),
        }
    }

    /// Get a string representation of the padding mode
    pub fn as_str(&self) -> String {
        match self {
            PaddingMode::Valid => "valid".to_string(),
            PaddingMode::Same => "same".to_string(),
            PaddingMode::Custom(p) => p.to_string(),
        }
    }
}

/// Validate convolution parameters
#[allow(dead_code)]
pub fn validate_conv_params(
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> Result<(), String> {
    if in_channels == 0 {
        return Err("Input _channels must be greater than 0".to_string());
    }
    if out_channels == 0 {
        return Err("Output _channels must be greater than 0".to_string());
    }
    if kernel_size.0 == 0 || kernel_size.1 == 0 {
        return Err("Kernel _size must be greater than 0".to_string());
    }
    if stride.0 == 0 || stride.1 == 0 {
        return Err("Stride must be greater than 0".to_string());
    }
    Ok(())
}

/// Calculate output shape for convolution operations
#[allow(dead_code)]
pub fn calculate_outputshape(
    input_height: usize,
    input_width: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> (usize, usize) {
    let effective_kernel_h = (kernel_size.0 - 1) * dilation.0 + 1;
    let effective_kernel_w = (kernel_size.1 - 1) * dilation.1 + 1;

    let output_height = (input_height + 2 * padding.0 - effective_kernel_h) / stride.0 + 1;
    let output_width = (input_width + 2 * padding.1 - effective_kernel_w) / stride.1 + 1;

    (output_height, output_width)
}

/// Calculate adaptive pooling parameters
#[allow(dead_code)]
pub fn calculate_adaptive_pooling_params(
    input_size: usize,
    output_size: usize,
) -> (usize, usize, usize) {
    // Calculate stride as floor division
    let stride = input_size / output_size;
    // Calculate kernel _size to ensure complete coverage
    let kernel_size = input_size - (output_size - 1) * stride;
    // Calculate padding to center the pooling
    let padding = 0; // No padding for adaptive pooling
    (kernel_size, stride, padding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padding_mode_calculation() {
        let kernel_size = (3, 3);
        let dilation = (1, 1);

        assert_eq!(
            PaddingMode::Valid.calculate_padding(kernel_size, dilation),
            (0, 0)
        );

        assert_eq!(
            PaddingMode::Same.calculate_padding(kernel_size, dilation),
            (1, 1)
        );

        assert_eq!(
            PaddingMode::Custom(2).calculate_padding(kernel_size, dilation),
            (2, 2)
        );
    }

    #[test]
    fn test_outputshape_calculation() {
        // Valid padding, stride 1
        assert_eq!(
            calculate_outputshape(32, 32, (3, 3), (1, 1), (0, 0), (1, 1)),
            (30, 30)
        );

        // Same padding, stride 1
        assert_eq!(
            calculate_outputshape(32, 32, (3, 3), (1, 1), (1, 1), (1, 1)),
            (32, 32)
        );

        // Stride 2
        assert_eq!(
            calculate_outputshape(32, 32, (3, 3), (2, 2), (1, 1), (1, 1)),
            (16, 16)
        );
    }

    #[test]
    fn test_calculate_adaptive_pooling_params() {
        let (kernel_size, stride, padding) = calculate_adaptive_pooling_params(8, 4);
        assert_eq!(stride, 2);
        assert_eq!(kernel_size, 2);
        assert_eq!(padding, 0);
    }
}
