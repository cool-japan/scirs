//! Convolution operations for neural networks
//!
//! This module contains implementations of convolution operations
//! and related functions like pooling, im2col/col2im transformations
//! used in convolutional neural networks.

use ndarray::{Array1, Array2, Array4, ArrayView1, ArrayView2, ArrayView4};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{NeuralError, Result};

/// Performs 2D convolution operation for convolutional neural networks.
///
/// # Arguments
///
/// * `input` - Input tensor with shape [batch_size, in_channels, height, width]
/// * `weight` - Weight tensor with shape [out_channels, in_channels, kernel_height, kernel_width]
/// * `bias` - Optional bias tensor with shape [out_channels]
/// * `stride` - Convolution stride (same for height and width)
/// * `padding` - Convolution padding (same for height and width)
///
/// # Returns
///
/// * Output tensor with shape [batch_size, out_channels, output_height, output_width]
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array1, Array4};
/// use scirs2_neural::linalg::conv2d;
///
/// // Create sample inputs
/// let batch_size = 2;
/// let in_channels = 3;
/// let height = 5;
/// let width = 5;
/// let out_channels = 2;
/// let kernel_size = 3;
///
/// // Initialize with sample values
/// let input = Array::from_shape_fn(
///     (batch_size, in_channels, height, width),
///     |_| 0.1
/// );
///
/// let weight = Array::from_shape_fn(
///     (out_channels, in_channels, kernel_size, kernel_size),
///     |_| 0.1
/// );
///
/// let bias = Some(Array::from_shape_fn(out_channels, |_| 0.1));
///
/// // Apply convolution with stride 1 and padding 1
/// let output = conv2d(&input.view(), &weight.view(), bias.as_ref().map(|b| b.view()), 1, 1).unwrap();
///
/// // Output shape should be [batch_size, out_channels, height, width] with padding 1 and stride 1
/// assert_eq!(output.shape(), &[batch_size, out_channels, height, width]);
/// ```
pub fn conv2d<F>(
    input: &ArrayView4<F>,
    weight: &ArrayView4<F>,
    bias: Option<ArrayView1<F>>,
    stride: usize,
    padding: usize,
) -> Result<Array4<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = input.shape()[0];
    let in_channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];

    let out_channels = weight.shape()[0];
    let weight_in_channels = weight.shape()[1];
    let kernel_height = weight.shape()[2];
    let kernel_width = weight.shape()[3];

    // Validate shapes
    if in_channels != weight_in_channels {
        return Err(NeuralError::ShapeMismatch(
            format!("Input and weight channel mismatch in conv2d: input has {} channels, weight expects {} channels",
                   in_channels, weight_in_channels)
        ));
    }

    if let Some(b) = bias {
        if b.shape()[0] != out_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "Bias shape mismatch in conv2d: bias has {} channels, expected {}",
                b.shape()[0],
                out_channels
            )));
        }
    }

    // Calculate output dimensions
    let out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    let out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

    // Initialize output tensor
    let mut output = Array4::<F>::zeros((batch_size, out_channels, out_height, out_width));

    // Perform convolution using im2col approach

    // Convert input to im2col format
    let mut input_padded = Array4::<F>::zeros((
        batch_size,
        in_channels,
        in_height + 2 * padding,
        in_width + 2 * padding,
    ));

    // Copy input data to padded array (zero padding)
    for b in 0..batch_size {
        for c in 0..in_channels {
            for h in 0..in_height {
                for w in 0..in_width {
                    input_padded[[b, c, h + padding, w + padding]] = input[[b, c, h, w]];
                }
            }
        }
    }

    // Perform convolution
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let h_start = oh * stride;
                    let w_start = ow * stride;

                    let mut sum = F::zero();

                    // Sum over input channels and kernel dimensions
                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                sum = sum
                                    + input_padded[[b, ic, h_start + kh, w_start + kw]]
                                        * weight[[oc, ic, kh, kw]];
                            }
                        }
                    }

                    // Add bias if provided
                    if let Some(b) = bias {
                        sum = sum + b[oc];
                    }

                    output[[b, oc, oh, ow]] = sum;
                }
            }
        }
    }

    Ok(output)
}

/// Performs 2D max pooling operation for convolutional neural networks.
///
/// Downsamples the input by taking the maximum value within pooling windows.
///
/// # Arguments
///
/// * `input` - Input tensor with shape [batch_size, channels, height, width]
/// * `kernel_size` - Size of the pooling window (same for height and width)
/// * `stride` - Pooling stride (same for height and width)
/// * `padding` - Pooling padding (same for height and width)
///
/// # Returns
///
/// * Tuple of (output, indices) where:
///   - output has shape [batch_size, channels, output_height, output_width]
///   - indices stores the positions of maximum values for backpropagation
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array4};
/// use scirs2_neural::linalg::max_pool2d;
///
/// // Create sample input
/// let batch_size = 2;
/// let channels = 3;
/// let height = 6;
/// let width = 6;
///
/// // Initialize with incrementing values for clear max pooling results
/// let mut input = Array::from_shape_fn(
///     (batch_size, channels, height, width),
///     |(b, c, h, w)| (h * width + w) as f32 + 0.1
/// );
///
/// // Apply max pooling with kernel_size 2, stride 2, padding 0
/// let (output, _) = max_pool2d(&input.view(), 2, 2, 0).unwrap();
///
/// // Output shape should be [batch_size, channels, height/stride, width/stride]
/// assert_eq!(output.shape(), &[batch_size, channels, height/2, width/2]);
/// ```
pub fn max_pool2d<F>(
    input: &ArrayView4<F>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<(Array4<F>, Array4<usize>)>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];

    // Calculate output dimensions
    let out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    let out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    // Initialize output tensor and indices
    let mut output = Array4::<F>::zeros((batch_size, channels, out_height, out_width));
    let mut indices = Array4::<usize>::zeros((batch_size, channels, out_height, out_width));

    // Pad input if needed
    let mut input_padded = Array4::<F>::zeros((
        batch_size,
        channels,
        in_height + 2 * padding,
        in_width + 2 * padding,
    ));

    // Use negative infinity for padding in max pooling
    let neg_inf = F::min_value();
    input_padded.fill(neg_inf);

    // Copy input data to padded array
    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..in_height {
                for w in 0..in_width {
                    input_padded[[b, c, h + padding, w + padding]] = input[[b, c, h, w]];
                }
            }
        }
    }

    // Perform max pooling
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let h_start = oh * stride;
                    let w_start = ow * stride;

                    let mut max_val = neg_inf;
                    let mut max_idx = 0;

                    // Find maximum value in the pooling window
                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let h_idx = h_start + kh;
                            let w_idx = w_start + kw;

                            let val = input_padded[[b, c, h_idx, w_idx]];
                            if val > max_val {
                                max_val = val;
                                // Convert to flattened index for backpropagation
                                max_idx = kh * kernel_size + kw;
                            }
                        }
                    }

                    output[[b, c, oh, ow]] = max_val;
                    indices[[b, c, oh, ow]] = max_idx;
                }
            }
        }
    }

    Ok((output, indices))
}

/// Converts image data to column format for efficient convolution.
///
/// Rearranges input data to make convolution operations more efficient by
/// turning them into matrix multiplications.
///
/// # Arguments
///
/// * `input` - Input tensor with shape [batch_size, channels, height, width]
/// * `kernel_height` - Height of the convolution kernel
/// * `kernel_width` - Width of the convolution kernel
/// * `stride` - Convolution stride
/// * `padding` - Convolution padding
///
/// # Returns
///
/// * Rearranged data in column format
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array4};
/// use scirs2_neural::linalg::im2col;
///
/// // Create sample input
/// let batch_size = 2;
/// let channels = 3;
/// let height = 5;
/// let width = 5;
///
/// let input = Array::from_shape_fn(
///     (batch_size, channels, height, width),
///     |_| 0.1
/// );
///
/// // Convert to column format for a 3x3 kernel
/// let col = im2col(&input.view(), 3, 3, 1, 1).unwrap();
///
/// // Check that the output has the expected shape
/// let output_height = height; // With padding 1 and stride 1
/// let output_width = width;  // With padding 1 and stride 1
/// let col_height = channels * 3 * 3;
/// let col_width = batch_size * output_height * output_width;
/// assert_eq!(col.shape(), &[col_height, col_width]);
/// ```
pub fn im2col<F>(
    input: &ArrayView4<F>,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
) -> Result<Array2<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];

    // Calculate output dimensions
    let out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    let out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

    // Pad input if needed
    let mut input_padded = Array4::<F>::zeros((
        batch_size,
        channels,
        in_height + 2 * padding,
        in_width + 2 * padding,
    ));

    // Copy input data to padded array
    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..in_height {
                for w in 0..in_width {
                    input_padded[[b, c, h + padding, w + padding]] = input[[b, c, h, w]];
                }
            }
        }
    }

    // Create output column matrix
    let col_height = channels * kernel_height * kernel_width;
    let col_width = batch_size * out_height * out_width;
    let mut cols = Array2::<F>::zeros((col_height, col_width));

    // Fill columns
    for b in 0..batch_size {
        for oh in 0..out_height {
            for ow in 0..out_width {
                let col_idx = b * (out_height * out_width) + oh * out_width + ow;

                let h_start = oh * stride;
                let w_start = ow * stride;

                // Extract patch and reshape into column
                let mut row_idx = 0;
                for c in 0..channels {
                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            cols[[row_idx, col_idx]] =
                                input_padded[[b, c, h_start + kh, w_start + kw]];
                            row_idx += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(cols)
}

/// Converts column format data back to image format.
///
/// This is the inverse operation of im2col, used during backpropagation.
///
/// # Arguments
///
/// * `cols` - Column data
/// * `batch_size` - Number of samples in the batch
/// * `channels` - Number of input channels
/// * `height` - Height of the input image
/// * `width` - Width of the input image
/// * `kernel_height` - Height of the convolution kernel
/// * `kernel_width` - Width of the convolution kernel
/// * `stride` - Convolution stride
/// * `padding` - Convolution padding
///
/// # Returns
///
/// * Rearranged data in image format
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array2, Array4};
/// use scirs2_neural::linalg::{im2col, col2im};
///
/// // Create sample input
/// let batch_size = 2;
/// let channels = 3;
/// let height = 5;
/// let width = 5;
///
/// let input = Array::from_shape_fn(
///     (batch_size, channels, height, width),
///     |_| 0.1
/// );
///
/// // Convert to column format
/// let col = im2col(&input.view(), 3, 3, 1, 1).unwrap();
///
/// // Convert back to image format
/// let output = col2im(
///     &col.view(), batch_size, channels, height, width, 3, 3, 1, 1
/// ).unwrap();
///
/// assert_eq!(output.shape(), input.shape());
/// ```
#[allow(clippy::too_many_arguments)]
pub fn col2im<F>(
    cols: &ArrayView2<F>,
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_height: usize,
    kernel_width: usize,
    _stride: usize,
    padding: usize,
) -> Result<Array4<F>>
where
    F: Float + Debug,
{
    // Calculate output dimensions
    let out_height = height + 2 * padding - kernel_height + 1;
    let out_width = width + 2 * padding - kernel_width + 1;

    // Validate column shape
    let expected_col_height = channels * kernel_height * kernel_width;
    let expected_col_width = batch_size * out_height * out_width;

    if cols.shape()[0] != expected_col_height || cols.shape()[1] != expected_col_width {
        return Err(NeuralError::ShapeMismatch(format!(
            "Column shape mismatch in col2im: expected [{}, {}], got [{}, {}]",
            expected_col_height,
            expected_col_width,
            cols.shape()[0],
            cols.shape()[1]
        )));
    }

    // Initialize output tensor (with padding)
    let mut output_padded = Array4::<F>::zeros((
        batch_size,
        channels,
        height + 2 * padding,
        width + 2 * padding,
    ));

    // Accumulate values from columns
    for b in 0..batch_size {
        for oh in 0..out_height {
            for ow in 0..out_width {
                let col_idx = b * (out_height * out_width) + oh * out_width + ow;

                let h_start = oh;
                let w_start = ow;

                // Distribute values from column back to image
                let mut row_idx = 0;
                for c in 0..channels {
                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            output_padded[[b, c, h_start + kh, w_start + kw]] = output_padded
                                [[b, c, h_start + kh, w_start + kw]]
                                + cols[[row_idx, col_idx]];
                            row_idx += 1;
                        }
                    }
                }
            }
        }
    }

    // Remove padding
    let mut output = Array4::<F>::zeros((batch_size, channels, height, width));

    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    output[[b, c, h, w]] = output_padded[[b, c, h + padding, w + padding]];
                }
            }
        }
    }

    Ok(output)
}

/// Performs adaptive average pooling for convolutional neural networks.
///
/// Downsamples the input to a specified output size using average pooling
/// with automatically calculated kernel size and stride.
///
/// # Arguments
///
/// * `input` - Input tensor with shape [batch_size, channels, height, width]
/// * `output_height` - Desired output height
/// * `output_width` - Desired output width
///
/// # Returns
///
/// * Output tensor with shape [batch_size, channels, output_height, output_width]
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array4};
/// use scirs2_neural::linalg::adaptive_avg_pool2d;
///
/// // Create sample input
/// let batch_size = 2;
/// let channels = 3;
/// let height = 7;
/// let width = 9;
///
/// let input = Array::from_shape_fn(
///     (batch_size, channels, height, width),
///     |_| 0.1
/// );
///
/// // Adaptively pool to 3x4 output size
/// let output = adaptive_avg_pool2d(&input.view(), 3, 4).unwrap();
///
/// assert_eq!(output.shape(), &[batch_size, channels, 3, 4]);
/// ```
pub fn adaptive_avg_pool2d<F>(
    input: &ArrayView4<F>,
    output_height: usize,
    output_width: usize,
) -> Result<Array4<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];

    // Validate output dimensions
    if output_height > in_height || output_width > in_width {
        return Err(NeuralError::InvalidArgument(
            "Output dimensions must be less than or equal to input dimensions in adaptive_avg_pool2d".to_string()
        ));
    }

    // Initialize output tensor
    let mut output = Array4::<F>::zeros((batch_size, channels, output_height, output_width));

    // Calculate stride and kernel size for each output position
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    // Calculate input region to average over
                    let h_start = (oh * in_height) / output_height;
                    let h_end = ((oh + 1) * in_height) / output_height;

                    let w_start = (ow * in_width) / output_width;
                    let w_end = ((ow + 1) * in_width) / output_width;

                    let kernel_h = h_end - h_start;
                    let kernel_w = w_end - w_start;

                    // Calculate average over the input region
                    let mut sum = F::zero();
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            sum = sum + input[[b, c, h, w]];
                        }
                    }

                    // Divide by kernel size
                    output[[b, c, oh, ow]] = sum / F::from(kernel_h * kernel_w).unwrap();
                }
            }
        }
    }

    Ok(output)
}

/// Computes the backward pass for 2D convolution operation.
///
/// # Arguments
///
/// * `dout` - Gradient of loss with respect to conv2d output, shape [batch_size, out_channels, out_height, out_width]
/// * `input` - Original input to conv2d, shape [batch_size, in_channels, in_height, in_width]
/// * `weight` - Convolution weights, shape [out_channels, in_channels, kernel_height, kernel_width]
/// * `stride` - Convolution stride
/// * `padding` - Convolution padding
///
/// # Returns
///
/// * Tuple of (d_input, d_weight, d_bias) containing gradients for inputs
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array1, Array4};
/// use scirs2_neural::linalg::{conv2d, conv2d_backward};
///
/// // Setup (similar to forward example)
/// let batch_size = 2;
/// let in_channels = 3;
/// let height = 5;
/// let width = 5;
/// let out_channels = 2;
/// let kernel_size = 3;
///
/// let input = Array::from_shape_fn(
///     (batch_size, in_channels, height, width),
///     |_| 0.1
/// );
///
/// let weight = Array::from_shape_fn(
///     (out_channels, in_channels, kernel_size, kernel_size),
///     |_| 0.1
/// );
///
/// // Forward pass
/// let output = conv2d(&input.view(), &weight.view(), None, 1, 1).unwrap();
///
/// // Gradient of loss with respect to output
/// let dout = Array::from_shape_fn(output.raw_dim(), |_| 0.01);
///
/// // Backward pass
/// let (d_input, d_weight, d_bias) = conv2d_backward(
///     &dout.view(), &input.view(), &weight.view(), 1, 1
/// ).unwrap();
///
/// assert_eq!(d_input.shape(), input.shape());
/// assert_eq!(d_weight.shape(), weight.shape());
/// assert_eq!(d_bias.shape()[0], out_channels);
/// ```
pub fn conv2d_backward<F>(
    dout: &ArrayView4<F>,
    input: &ArrayView4<F>,
    weight: &ArrayView4<F>,
    stride: usize,
    padding: usize,
) -> Result<(Array4<F>, Array4<F>, Array1<F>)>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = input.shape()[0];
    let in_channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];

    let out_channels = dout.shape()[1];
    let out_height = dout.shape()[2];
    let out_width = dout.shape()[3];

    let kernel_height = weight.shape()[2];
    let kernel_width = weight.shape()[3];

    // Initialize gradients
    let mut d_input = Array4::<F>::zeros(input.raw_dim());
    let mut d_weight = Array4::<F>::zeros(weight.raw_dim());
    let mut d_bias = Array1::<F>::zeros(out_channels);

    // Pad input for convolution
    let mut input_padded = Array4::<F>::zeros((
        batch_size,
        in_channels,
        in_height + 2 * padding,
        in_width + 2 * padding,
    ));

    // Copy input data to padded array
    for b in 0..batch_size {
        for c in 0..in_channels {
            for h in 0..in_height {
                for w in 0..in_width {
                    input_padded[[b, c, h + padding, w + padding]] = input[[b, c, h, w]];
                }
            }
        }
    }

    // Compute gradient for bias
    for oc in 0..out_channels {
        for b in 0..batch_size {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    d_bias[oc] = d_bias[oc] + dout[[b, oc, oh, ow]];
                }
            }
        }
    }

    // Compute gradient for weights
    for oc in 0..out_channels {
        for ic in 0..in_channels {
            for kh in 0..kernel_height {
                for kw in 0..kernel_width {
                    let mut grad = F::zero();

                    for b in 0..batch_size {
                        for oh in 0..out_height {
                            for ow in 0..out_width {
                                let h_in = oh * stride + kh;
                                let w_in = ow * stride + kw;

                                grad = grad
                                    + dout[[b, oc, oh, ow]] * input_padded[[b, ic, h_in, w_in]];
                            }
                        }
                    }

                    d_weight[[oc, ic, kh, kw]] = grad;
                }
            }
        }
    }

    // Compute gradient for input
    // Initialize padded gradient input
    let mut d_input_padded = Array4::<F>::zeros((
        batch_size,
        in_channels,
        in_height + 2 * padding,
        in_width + 2 * padding,
    ));

    // Compute convolution backward
    for b in 0..batch_size {
        for ic in 0..in_channels {
            for h_in in 0..(in_height + 2 * padding) {
                for w_in in 0..(in_width + 2 * padding) {
                    let mut grad = F::zero();

                    // Find all output positions that use this input position
                    for oc in 0..out_channels {
                        for kh in 0..kernel_height {
                            if h_in < kh {
                                continue;
                            }
                            let oh = (h_in - kh) / stride;
                            if oh >= out_height || (h_in - kh) % stride != 0 {
                                continue;
                            }

                            for kw in 0..kernel_width {
                                if w_in < kw {
                                    continue;
                                }
                                let ow = (w_in - kw) % stride;
                                if ow >= out_width || (w_in - kw) % stride != 0 {
                                    continue;
                                }

                                grad = grad + dout[[b, oc, oh, ow]] * weight[[oc, ic, kh, kw]];
                            }
                        }
                    }

                    d_input_padded[[b, ic, h_in, w_in]] = grad;
                }
            }
        }
    }

    // Copy from padded gradients to output
    for b in 0..batch_size {
        for c in 0..in_channels {
            for h in 0..in_height {
                for w in 0..in_width {
                    d_input[[b, c, h, w]] = d_input_padded[[b, c, h + padding, w + padding]];
                }
            }
        }
    }

    Ok((d_input, d_weight, d_bias))
}

/// Computes the backward pass for 2D max pooling operation.
///
/// # Arguments
///
/// * `dout` - Gradient of loss with respect to pooling output, shape [batch_size, channels, out_height, out_width]
/// * `input` - Original input to pooling, shape [batch_size, channels, in_height, in_width]
/// * `indices` - Indices of maximum values from forward pass
/// * `kernel_size` - Size of the pooling window
/// * `stride` - Pooling stride
/// * `padding` - Pooling padding
///
/// # Returns
///
/// * Gradient with respect to input
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array4};
/// use scirs2_neural::linalg::{max_pool2d, max_pool2d_backward};
///
/// // Setup (similar to forward example)
/// let batch_size = 2;
/// let channels = 3;
/// let height = 6;
/// let width = 6;
///
/// let input = Array::from_shape_fn(
///     (batch_size, channels, height, width),
///     |(b, c, h, w)| (h * width + w) as f32 + 0.1  // Values that create clear max values
/// );
///
/// // Forward pass
/// let (output, indices) = max_pool2d(&input.view(), 2, 2, 0).unwrap();
///
/// // Gradient of loss with respect to output
/// let dout = Array::from_shape_fn(output.raw_dim(), |_| 0.01);
///
/// // Backward pass
/// let d_input = max_pool2d_backward(
///     &dout.view(), &input.view(), &indices.view(), 2, 2, 0
/// ).unwrap();
///
/// assert_eq!(d_input.shape(), input.shape());
/// ```
pub fn max_pool2d_backward<F>(
    dout: &ArrayView4<F>,
    input: &ArrayView4<F>,
    indices: &ArrayView4<usize>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<Array4<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];

    let out_height = dout.shape()[2];
    let out_width = dout.shape()[3];

    // Validate shapes
    if dout.shape()[0] != batch_size || dout.shape()[1] != channels {
        return Err(NeuralError::ShapeMismatch(format!(
            "Gradient shape mismatch in max_pool2d_backward: dout shape {:?}, input shape {:?}",
            dout.shape(),
            input.shape()
        )));
    }

    if indices.shape() != dout.shape() {
        return Err(NeuralError::ShapeMismatch(format!(
            "Indices shape mismatch in max_pool2d_backward: indices shape {:?}, dout shape {:?}",
            indices.shape(),
            dout.shape()
        )));
    }

    // Initialize gradient for input
    let mut d_input = Array4::<F>::zeros(input.raw_dim());

    // Distribute gradients back to input based on max indices
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let h_start = oh * stride;
                    let w_start = ow * stride;

                    // Get the index of the max value
                    let idx = indices[[b, c, oh, ow]];
                    let kh = idx / kernel_size;
                    let kw = idx % kernel_size;

                    // Add gradient to the input position that had the max value
                    let h_idx = h_start + kh - padding;
                    let w_idx = w_start + kw - padding;

                    // Skip if the index is out of bounds (due to padding)
                    if h_idx < in_height && w_idx < in_width {
                        d_input[[b, c, h_idx, w_idx]] =
                            d_input[[b, c, h_idx, w_idx]] + dout[[b, c, oh, ow]];
                    }
                }
            }
        }
    }

    Ok(d_input)
}

/// Computes the backward pass for adaptive average pooling.
///
/// # Arguments
///
/// * `dout` - Gradient of loss with respect to pooling output
/// * `input` - Original input to pooling
/// * `output_height` - Height of the output
/// * `output_width` - Width of the output
///
/// # Returns
///
/// * Gradient with respect to input
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array4};
/// use scirs2_neural::linalg::{adaptive_avg_pool2d, adaptive_avg_pool2d_backward};
///
/// // Setup (similar to forward example)
/// let batch_size = 2;
/// let channels = 3;
/// let height = 7;
/// let width = 9;
///
/// let input = Array::from_shape_fn(
///     (batch_size, channels, height, width),
///     |_| 0.1
/// );
///
/// // Forward pass
/// let output = adaptive_avg_pool2d(&input.view(), 3, 4).unwrap();
///
/// // Gradient of loss with respect to output
/// let dout = Array::from_shape_fn(output.raw_dim(), |_| 0.01);
///
/// // Backward pass
/// let d_input = adaptive_avg_pool2d_backward(
///     &dout.view(), &input.view(), 3, 4
/// ).unwrap();
///
/// assert_eq!(d_input.shape(), input.shape());
/// ```
pub fn adaptive_avg_pool2d_backward<F>(
    dout: &ArrayView4<F>,
    input: &ArrayView4<F>,
    output_height: usize,
    output_width: usize,
) -> Result<Array4<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = input.shape()[0];
    let channels = input.shape()[1];
    let in_height = input.shape()[2];
    let in_width = input.shape()[3];

    // Validate shapes
    if dout.shape() != [batch_size, channels, output_height, output_width] {
        return Err(NeuralError::ShapeMismatch(
            format!("Gradient shape mismatch in adaptive_avg_pool2d_backward: dout shape {:?}, expected [{}, {}, {}, {}]",
                   dout.shape(), batch_size, channels, output_height, output_width)
        ));
    }

    // Initialize gradient for input
    let mut d_input = Array4::<F>::zeros(input.raw_dim());

    // Distribute gradients back to input
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    // Calculate input region that was averaged
                    let h_start = (oh * in_height) / output_height;
                    let h_end = ((oh + 1) * in_height) / output_height;

                    let w_start = (ow * in_width) / output_width;
                    let w_end = ((ow + 1) * in_width) / output_width;

                    let kernel_h = h_end - h_start;
                    let kernel_w = w_end - w_start;

                    // Calculate gradient factor
                    let grad_factor = dout[[b, c, oh, ow]] / F::from(kernel_h * kernel_w).unwrap();

                    // Distribute gradient evenly to all inputs in the region
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            d_input[[b, c, h, w]] = d_input[[b, c, h, w]] + grad_factor;
                        }
                    }
                }
            }
        }
    }

    Ok(d_input)
}
