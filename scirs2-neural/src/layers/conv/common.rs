//! Common types and utilities for convolutional and pooling layers
//!
//! This module provides shared types, enums, and utility functions used across
//! all convolutional and pooling layer implementations including padding modes,
//! type aliases for caching, and common validation functions.

use ndarray::{Array, IxDyn};
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
pub fn validate_conv_params(
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<(), crate::error::NeuralError> {
    use crate::error::NeuralError;

    if in_channels == 0 || out_channels == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "Number of channels must be positive".to_string(),
        ));
    }

    if kernel_size.0 == 0 || kernel_size.1 == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "Kernel dimensions must be positive".to_string(),
        ));
    }

    if stride.0 == 0 || stride.1 == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "Stride must be positive".to_string(),
        ));
    }

    if dilation.0 == 0 || dilation.1 == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "Dilation must be positive".to_string(),
        ));
    }

    if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "Invalid groups: {} must divide both in_channels: {} and out_channels: {}",
            groups, in_channels, out_channels
        )));
    }

    Ok(())
}

/// Validate pooling parameters
pub fn validate_pool_params(
    pool_size: (usize, usize),
    stride: (usize, usize),
) -> Result<(), crate::error::NeuralError> {
    use crate::error::NeuralError;

    if pool_size.0 == 0 || pool_size.1 == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "Pool size must be positive".to_string(),
        ));
    }

    if stride.0 == 0 || stride.1 == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "Stride must be positive".to_string(),
        ));
    }

    Ok(())
}

/// Calculate output spatial dimensions for convolution/pooling operations
pub fn calculate_output_shape(
    input_height: usize,
    input_width: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> (usize, usize) {
    let output_height = {
        let padded_height = input_height + 2 * padding.0;
        let dilated_kernel = dilation.0 * (kernel_size.0 - 1);

        if padded_height > dilated_kernel {
            (padded_height - dilated_kernel - 1) / stride.0 + 1
        } else {
            1 // Minimum size
        }
    };

    let output_width = {
        let padded_width = input_width + 2 * padding.1;
        let dilated_kernel = dilation.1 * (kernel_size.1 - 1);

        if padded_width > dilated_kernel {
            (padded_width - dilated_kernel - 1) / stride.1 + 1
        } else {
            1 // Minimum size
        }
    };

    (output_height, output_width)
}

/// Calculate output spatial dimensions for pooling operations
pub fn calculate_pool_output_shape(
    input_height: usize,
    input_width: usize,
    pool_size: (usize, usize),
    stride: (usize, usize),
    padding: Option<(usize, usize)>,
) -> (usize, usize) {
    let (pad_h, pad_w) = padding.unwrap_or((0, 0));

    let output_height = (input_height + 2 * pad_h - pool_size.0) / stride.0 + 1;
    let output_width = (input_width + 2 * pad_w - pool_size.1) / stride.1 + 1;

    (output_height, output_width)
}

/// Calculate adaptive pooling parameters for a given input and output size
pub fn calculate_adaptive_pooling_params(
    input_size: usize,
    output_size: usize,
) -> (usize, usize, usize) {
    // Calculate stride as floor division
    let stride = input_size / output_size;
    // Calculate kernel size to ensure complete coverage
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
    fn test_validate_conv_params() {
        // Valid parameters
        assert!(validate_conv_params(3, 8, (3, 3), (1, 1), (1, 1), 1).is_ok());

        // Invalid channels
        assert!(validate_conv_params(0, 8, (3, 3), (1, 1), (1, 1), 1).is_err());

        // Invalid kernel size
        assert!(validate_conv_params(3, 8, (0, 3), (1, 1), (1, 1), 1).is_err());

        // Invalid stride
        assert!(validate_conv_params(3, 8, (3, 3), (0, 1), (1, 1), 1).is_err());

        // Invalid groups
        assert!(validate_conv_params(3, 8, (3, 3), (1, 1), (1, 1), 2).is_err());
    }

    #[test]
    fn test_validate_pool_params() {
        // Valid parameters
        assert!(validate_pool_params((2, 2), (2, 2)).is_ok());

        // Invalid pool size
        assert!(validate_pool_params((0, 2), (2, 2)).is_err());

        // Invalid stride
        assert!(validate_pool_params((2, 2), (0, 2)).is_err());
    }

    #[test]
    fn test_calculate_output_shape() {
        // No padding, stride 1
        assert_eq!(
            calculate_output_shape(32, 32, (3, 3), (1, 1), (0, 0), (1, 1)),
            (30, 30)
        );

        // Same padding, stride 1
        assert_eq!(
            calculate_output_shape(32, 32, (3, 3), (1, 1), (1, 1), (1, 1)),
            (32, 32)
        );

        // Stride 2
        assert_eq!(
            calculate_output_shape(32, 32, (3, 3), (2, 2), (1, 1), (1, 1)),
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
