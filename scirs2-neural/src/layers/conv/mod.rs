//! Convolutional neural network layers implementation
//!
//! This module provides implementations of convolution layers for neural networks,
//! including Conv2D, Conv3D, and their transpose versions, as well as comprehensive
//! pooling layers for 1D, 2D, and 3D data.
//!
//! # Module Organization
//!
//! - [`common`] - Common types, enums, and utility functions
//! - [`conv2d`] - 2D convolution implementation with im2col operations
//! - [`pooling`] - All pooling layer implementations (standard and adaptive)

pub mod common;
pub mod conv2d;
pub mod pooling;

// Re-export main types and functions for backward compatibility
pub use common::PaddingMode;
pub use conv2d::Conv2D;
pub use pooling::{
    AdaptiveAvgPool1D, AdaptiveAvgPool2D, AdaptiveAvgPool3D, AdaptiveMaxPool1D, AdaptiveMaxPool2D,
    AdaptiveMaxPool3D, GlobalAvgPool2D, MaxPool2D,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Layer;
    use ndarray::Array4;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_conv2d_shape() {
        // Create a 2D convolutional layer
        let mut rng = SmallRng::seed_from_u64(42);
        let conv = Conv2D::<f64>::new(
            3,                 // in_channels
            8,                 // out_channels
            (3, 3),            // kernel_size
            (1, 1),            // stride
            PaddingMode::Same, // padding
            &mut rng,
        )
        .unwrap();

        // Create a batch of input data
        let batch_size = 2;
        let channels = 3;
        let height = 32;
        let width = 32;
        let input = Array4::<f64>::from_elem((batch_size, channels, height, width), 0.1).into_dyn();

        // Forward pass
        let output = conv.forward(&input).unwrap();

        // Check output shape - with Same padding, height and width should be preserved
        assert_eq!(output.shape(), &[batch_size, 8, height, width]);
    }

    #[test]
    fn test_conv2d_valid_padding() {
        // Create a 2D convolutional layer with Valid padding
        let mut rng = SmallRng::seed_from_u64(42);
        let conv = Conv2D::<f64>::new(
            3,                  // in_channels
            8,                  // out_channels
            (3, 3),             // kernel_size
            (1, 1),             // stride
            PaddingMode::Valid, // padding
            &mut rng,
        )
        .unwrap();

        // Create a batch of input data
        let batch_size = 2;
        let channels = 3;
        let height = 32;
        let width = 32;
        let input = Array4::<f64>::from_elem((batch_size, channels, height, width), 0.1).into_dyn();

        // Forward pass
        let output = conv.forward(&input).unwrap();

        // Check output shape - with Valid padding, height and width should be reduced by (kernel_size - 1)
        assert_eq!(output.shape(), &[batch_size, 8, height - 2, width - 2]);
    }

    #[test]
    fn test_maxpool2d_shape() {
        // Create a 2D max pooling layer
        let pool = MaxPool2D::<f64>::new(
            (2, 2), // pool_size
            (2, 2), // stride
            None,   // no padding
        )
        .unwrap();

        // Create a batch of input data
        let batch_size = 2;
        let channels = 3;
        let height = 32;
        let width = 32;
        let input = Array4::<f64>::from_elem((batch_size, channels, height, width), 0.1).into_dyn();

        // Forward pass
        let output = pool.forward(&input).unwrap();

        // Check output shape - height and width should be halved
        assert_eq!(
            output.shape(),
            &[batch_size, channels, height / 2, width / 2]
        );
    }

    #[test]
    fn test_maxpool2d_values() {
        // Create a 2D max pooling layer
        let pool = MaxPool2D::<f64>::new(
            (2, 2), // pool_size
            (2, 2), // stride
            None,   // no padding
        )
        .unwrap();

        // Create a simple 4x4 input with known values
        let mut input = Array4::<f64>::zeros((1, 1, 4, 4));
        // Set values in a pattern where we know what the max should be
        // [0.1, 0.2, 0.3, 0.4]
        // [0.5, 0.6, 0.7, 0.8]
        // [0.9, 1.0, 1.1, 1.2]
        // [1.3, 1.4, 1.5, 1.6]
        let mut val = 0.1;
        for i in 0..4 {
            for j in 0..4 {
                input[[0, 0, i, j]] = val;
                val += 0.1;
            }
        }

        // Forward pass
        let output = pool.forward(&input.into_dyn()).unwrap();

        // Check output shape - should be 2x2
        assert_eq!(output.shape(), &[1, 1, 2, 2]);

        // Check values - max pool should pick the bottom right value of each 2x2 window
        assert!((output[[0, 0, 0, 0]] - 0.6).abs() < 1e-10); // max of top-left 2x2
        assert!((output[[0, 0, 0, 1]] - 0.8).abs() < 1e-10); // max of top-right 2x2
        assert!((output[[0, 0, 1, 0]] - 1.4).abs() < 1e-10); // max of bottom-left 2x2
        assert!((output[[0, 0, 1, 1]] - 1.6).abs() < 1e-10); // max of bottom-right 2x2
    }

    #[test]
    fn test_adaptive_avg_pool2d_shape() {
        // Create an adaptive average pooling layer with output size 2x2
        let pool = AdaptiveAvgPool2D::<f64>::new((2, 2), Some("test_pool")).unwrap();

        // Create a 4x4 input
        let input = Array4::<f64>::from_elem((1, 1, 8, 8), 1.0);

        // Forward pass
        let output = pool.forward(&input.into_dyn()).unwrap();

        // Check output shape - should be 2x2
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_adaptive_avg_pool2d_values() {
        // Create an adaptive average pooling layer with output size 2x2
        let pool = AdaptiveAvgPool2D::<f64>::new((2, 2), Some("test_pool")).unwrap();

        // Create a 4x4 input with known values
        let mut input = Array4::<f64>::zeros((1, 1, 4, 4));
        let mut val = 1.0;
        for i in 0..4 {
            for j in 0..4 {
                input[[0, 0, i, j]] = val;
                val += 1.0;
            }
        }

        // Forward pass
        let output = pool.forward(&input.into_dyn()).unwrap();

        // Check output shape
        assert_eq!(output.shape(), &[1, 1, 2, 2]);

        // Values should be averaged within each 2x2 window
        // Top-left: average of 1,2,5,6 = 3.5
        assert!((output[[0, 0, 0, 0]] - 3.5).abs() < 1e-10);
        // Top-right: average of 3,4,7,8 = 5.5
        assert!((output[[0, 0, 0, 1]] - 5.5).abs() < 1e-10);
        // Bottom-left: average of 9,10,13,14 = 11.5
        assert!((output[[0, 0, 1, 0]] - 11.5).abs() < 1e-10);
        // Bottom-right: average of 11,12,15,16 = 13.5
        assert!((output[[0, 0, 1, 1]] - 13.5).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_max_pool2d_shape() {
        // Create an adaptive max pooling layer with output size 2x2
        let pool = AdaptiveMaxPool2D::<f64>::new((2, 2), Some("test_pool")).unwrap();

        // Create a 4x4 input
        let input = Array4::<f64>::from_elem((1, 1, 8, 8), 1.0);

        // Forward pass
        let output = pool.forward(&input.into_dyn()).unwrap();

        // Check output shape - should be 2x2
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_adaptive_max_pool2d_values() {
        // Create an adaptive max pooling layer with output size 2x2
        let pool = AdaptiveMaxPool2D::<f64>::new((2, 2), Some("test_pool")).unwrap();

        // Create a 4x4 input with known values
        let mut input = Array4::<f64>::zeros((1, 1, 4, 4));
        let mut val = 1.0;
        for i in 0..4 {
            for j in 0..4 {
                input[[0, 0, i, j]] = val;
                val += 1.0;
            }
        }

        // Forward pass
        let output = pool.forward(&input.into_dyn()).unwrap();

        // Check output shape
        assert_eq!(output.shape(), &[1, 1, 2, 2]);

        // Values should be maximum within each 2x2 window
        // Top-left: max of 1,2,5,6 = 6
        assert!((output[[0, 0, 0, 0]] - 6.0).abs() < 1e-10);
        // Top-right: max of 3,4,7,8 = 8
        assert!((output[[0, 0, 0, 1]] - 8.0).abs() < 1e-10);
        // Bottom-left: max of 9,10,13,14 = 14
        assert!((output[[0, 0, 1, 0]] - 14.0).abs() < 1e-10);
        // Bottom-right: max of 11,12,15,16 = 16
        assert!((output[[0, 0, 1, 1]] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_pool_different_output_sizes() {
        // Test with non-square adaptive pooling (3x2 output size)
        let pool = AdaptiveAvgPool2D::<f64>::new((3, 2), Some("test_pool")).unwrap();

        // Create a 6x4 input
        let input = Array4::<f64>::from_elem((1, 1, 6, 4), 1.0);

        // Forward pass
        let output = pool.forward(&input.into_dyn()).unwrap();

        // Check output shape - should be 3x2
        assert_eq!(output.shape(), &[1, 1, 3, 2]);
    }

    #[test]
    fn test_global_avg_pool2d() {
        // Create a global average pooling layer
        let pool = GlobalAvgPool2D::<f64>::new(Some("global_pool")).unwrap();

        // Create input data
        let batch_size = 2;
        let channels = 3;
        let height = 32;
        let width = 32;
        let input = Array4::<f64>::from_elem((batch_size, channels, height, width), 0.1).into_dyn();

        // Forward pass
        let output = pool.forward(&input).unwrap();

        // Check output shape - should be [batch_size, channels, 1, 1]
        assert_eq!(output.shape(), &[batch_size, channels, 1, 1]);

        // Check values - should be 0.1 for all elements
        for b in 0..batch_size {
            for c in 0..channels {
                assert!((output[[b, c, 0, 0]] - 0.1).abs() < 1e-10);
            }
        }
    }

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
}

#[cfg(test)]
mod adaptive_pooling_tests {
    use super::*;
    use crate::layers::Layer;
    use ndarray::Array;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_adaptive_avg_pool_1d() {
        let pool = AdaptiveAvgPool1D::<f64>::new(4, Some("test_pool")).unwrap();

        // Create a 3D input: [batch_size, channels, width]
        let input = Array::from_shape_vec((1, 2, 8), (0..16).map(|i| i as f64).collect()).unwrap();

        // Forward pass
        let output = pool.forward(&input.into_dyn()).unwrap();

        // Check output shape - should be [1, 2, 4]
        assert_eq!(output.shape(), &[1, 2, 4]);
    }

    #[test]
    fn test_adaptive_max_pool_1d() {
        let pool = AdaptiveMaxPool1D::<f64>::new(3, Some("test_pool")).unwrap();

        // Create a 3D input: [batch_size, channels, width]
        let input = Array::from_shape_vec((1, 1, 6), vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0]).unwrap();

        // Forward pass
        let output = pool.forward(&input.into_dyn()).unwrap();

        // Check output shape - should be [1, 1, 3]
        assert_eq!(output.shape(), &[1, 1, 3]);
    }

    #[test]
    fn test_adaptive_avg_pool_3d() {
        let pool = AdaptiveAvgPool3D::<f64>::new((2, 2, 2), Some("test_pool")).unwrap();

        // Create a 5D input: [batch_size, channels, depth, height, width]
        let input = Array::from_elem((1, 1, 4, 4, 4), 1.0);

        // Forward pass
        let output = pool.forward(&input.into_dyn()).unwrap();

        // Check output shape - should be [1, 1, 2, 2, 2]
        assert_eq!(output.shape(), &[1, 1, 2, 2, 2]);
    }

    #[test]
    fn test_adaptive_max_pool_3d() {
        let pool = AdaptiveMaxPool3D::<f64>::new((1, 2, 2), Some("test_pool")).unwrap();

        // Create a 5D input: [batch_size, channels, depth, height, width]
        let input = Array::from_elem((1, 1, 2, 4, 4), 2.0);

        // Forward pass
        let output = pool.forward(&input.into_dyn()).unwrap();

        // Check output shape - should be [1, 1, 1, 2, 2]
        assert_eq!(output.shape(), &[1, 1, 1, 2, 2]);
    }

    #[test]
    fn test_layer_descriptions() {
        // Test conv2d description
        let mut rng = SmallRng::seed_from_u64(42);
        let conv = Conv2D::<f64>::new(3, 8, (3, 3), (1, 1), PaddingMode::Same, &mut rng).unwrap();
        let desc = conv.layer_description();
        assert!(desc.contains("Conv2D"));
        assert!(desc.contains("in_channels:3"));
        assert!(desc.contains("out_channels:8"));

        // Test pooling layer descriptions
        let pool = MaxPool2D::<f64>::new((2, 2), (2, 2), None).unwrap();
        let desc = pool.layer_description();
        assert!(desc.contains("MaxPool2D"));
        assert!(desc.contains("pool_size:(2,2)"));

        let adaptive_pool = AdaptiveAvgPool2D::<f64>::new((7, 7), Some("test")).unwrap();
        let desc = adaptive_pool.layer_description();
        assert!(desc.contains("AdaptiveAvgPool2D"));
        assert!(desc.contains("output_size:(7, 7)"));
    }

    #[test]
    fn test_parameter_counts() {
        let mut rng = SmallRng::seed_from_u64(42);

        // Conv2D should have parameters
        let conv = Conv2D::<f64>::new(3, 8, (3, 3), (1, 1), PaddingMode::Same, &mut rng).unwrap();
        assert!(conv.parameter_count() > 0);

        // Pooling layers should have no parameters
        let pool = MaxPool2D::<f64>::new((2, 2), (2, 2), None).unwrap();
        assert_eq!(pool.parameter_count(), 0);

        let global_pool = GlobalAvgPool2D::<f64>::new(Some("test")).unwrap();
        assert_eq!(global_pool.parameter_count(), 0);

        let adaptive_pool = AdaptiveAvgPool2D::<f64>::new((7, 7), Some("test")).unwrap();
        assert_eq!(adaptive_pool.parameter_count(), 0);
    }

    #[test]
    fn test_layer_types() {
        let mut rng = SmallRng::seed_from_u64(42);

        let conv = Conv2D::<f64>::new(3, 8, (3, 3), (1, 1), PaddingMode::Same, &mut rng).unwrap();
        assert_eq!(conv.layer_type(), "Conv2D");

        let pool = MaxPool2D::<f64>::new((2, 2), (2, 2), None).unwrap();
        assert_eq!(pool.layer_type(), "MaxPool2D");

        let global_pool = GlobalAvgPool2D::<f64>::new(Some("test")).unwrap();
        assert_eq!(global_pool.layer_type(), "GlobalAvgPool2D");

        let adaptive_avg = AdaptiveAvgPool2D::<f64>::new((7, 7), Some("test")).unwrap();
        assert_eq!(adaptive_avg.layer_type(), "AdaptiveAvgPool2D");

        let adaptive_max = AdaptiveMaxPool2D::<f64>::new((7, 7), Some("test")).unwrap();
        assert_eq!(adaptive_max.layer_type(), "AdaptiveMaxPool2D");

        let adaptive_1d_avg = AdaptiveAvgPool1D::<f64>::new(4, Some("test")).unwrap();
        assert_eq!(adaptive_1d_avg.layer_type(), "AdaptiveAvgPool1D");

        let adaptive_1d_max = AdaptiveMaxPool1D::<f64>::new(4, Some("test")).unwrap();
        assert_eq!(adaptive_1d_max.layer_type(), "AdaptiveMaxPool1D");

        let adaptive_3d_avg = AdaptiveAvgPool3D::<f64>::new((2, 2, 2), Some("test")).unwrap();
        assert_eq!(adaptive_3d_avg.layer_type(), "AdaptiveAvgPool3D");

        let adaptive_3d_max = AdaptiveMaxPool3D::<f64>::new((2, 2, 2), Some("test")).unwrap();
        assert_eq!(adaptive_3d_max.layer_type(), "AdaptiveMaxPool3D");
    }
}
