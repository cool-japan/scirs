//! Specialized operations for convolutional neural networks
//!
//! This module provides efficient implementations of specialized operations
//! that are commonly used in convolutional neural networks, such as im2col/col2im,
//! efficient convolution algorithms, and other related operations.

use ndarray::{Array2, Array4, ArrayView4, ScalarOperand};
use num_traits::{Float, NumAssign, Zero};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Extract patches from an input tensor using the im2col algorithm
///
/// This function implements the im2col (image to column) algorithm, which
/// reformats the input data to allow efficient computation of convolution
/// operations using matrix multiplication.
///
/// # Arguments
///
/// * `input` - Input tensor of shape (batch_size, channels, height, width)
/// * `kernel_size` - Size of the kernel as (kernel_height, kernel_width)
/// * `stride` - Stride as (stride_height, stride_width)
/// * `padding` - Padding as (padding_height, padding_width)
/// * `dilation` - Dilation as (dilation_height, dilation_width)
///
/// # Returns
///
/// * Column matrix of shape (kernel_h * kernel_w * channels, output_h * output_w * batch_size)
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_linalg::convolution::im2col;
///
/// // Create a 1x3x4x4 input tensor (1 batch, 3 channels, 4x4 spatial dimensions)
/// let mut input = Array4::<f32>::zeros((1, 3, 4, 4));
/// // Fill with sample data
/// for c in 0..3 {
///     for h in 0..4 {
///         for w in 0..4 {
///             input[[0, c, h, w]] = (c * 16 + h * 4 + w) as f32;
///         }
///     }
/// }
///
/// // Extract 3x3 patches with stride 1 and no padding
/// let cols = im2col(&input.view(), (3, 3), (1, 1), (0, 0), (1, 1)).unwrap();
///
/// // Resulting matrix has shape (3*3*3, 2*2*1) = (27, 4)
/// // Each column represents a 3x3 patch across all 3 channels
/// assert_eq!(cols.shape(), &[27, 4]);
/// ```
pub fn im2col<F>(
    input: &ArrayView4<F>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Zero + ScalarOperand,
{
    let (batch_size, channels, height, width) = input.dim();
    let (kernel_h, kernel_w) = kernel_size;
    let (stride_h, stride_w) = stride;
    let (padding_h, padding_w) = padding;
    let (dilation_h, dilation_w) = dilation;

    // Calculate output dimensions
    let output_h = ((height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1;
    let output_w = ((width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1;

    // Check for valid dimensions
    if output_h == 0 || output_w == 0 {
        return Err(LinalgError::ShapeError(format!(
            "Invalid output dimensions: ({}, {})",
            output_h, output_w
        )));
    }

    // Allocate output matrix
    let mut cols = Array2::<F>::zeros((
        kernel_h * kernel_w * channels,
        output_h * output_w * batch_size,
    ));

    // Populate the output matrix with patches
    for batch_idx in 0..batch_size {
        for channel_idx in 0..channels {
            for kernel_row in 0..kernel_h {
                for kernel_col in 0..kernel_w {
                    let input_row_offset = kernel_row * dilation_h;
                    let input_col_offset = kernel_col * dilation_w;

                    // Position in the cols matrix
                    let cols_idx =
                        channel_idx * kernel_h * kernel_w + kernel_row * kernel_w + kernel_col;

                    for output_row in 0..output_h {
                        for output_col in 0..output_w {
                            let input_row = output_row * stride_h + input_row_offset;
                            let input_col = output_col * stride_w + input_col_offset;

                            // Position in the cols matrix
                            let cols_pos = batch_idx * output_h * output_w
                                + output_row * output_w
                                + output_col;

                            // Check if we need to pad
                            if input_row < padding_h
                                || input_row >= height + padding_h
                                || input_col < padding_w
                                || input_col >= width + padding_w
                            {
                                // Zero-padding
                                cols[[cols_idx, cols_pos]] = F::zero();
                            } else {
                                // Copy from input
                                let input_val = input[[
                                    batch_idx,
                                    channel_idx,
                                    input_row - padding_h,
                                    input_col - padding_w,
                                ]];
                                cols[[cols_idx, cols_pos]] = input_val;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(cols)
}

/// Convert a column matrix back to an input tensor using the col2im algorithm
///
/// This function implements the col2im (column to image) algorithm, which
/// converts the column matrix back to the original input tensor format.
///
/// # Arguments
///
/// * `cols` - Column matrix of shape (kernel_h * kernel_w * channels, output_h * output_w * batch_size)
/// * `output_shape` - Shape of the output tensor as (batch_size, channels, height, width)
/// * `kernel_size` - Size of the kernel as (kernel_height, kernel_width)
/// * `stride` - Stride as (stride_height, stride_width)
/// * `padding` - Padding as (padding_height, padding_width)
/// * `dilation` - Dilation as (dilation_height, dilation_width)
///
/// # Returns
///
/// * Output tensor of shape (batch_size, channels, height, width)
///
/// # Examples
///
/// ```
/// use ndarray::{Array4, ArrayView4};
/// use scirs2_linalg::convolution::{im2col, col2im};
///
/// // Create a 1x3x4x4 input tensor
/// let mut input = Array4::<f32>::zeros((1, 3, 4, 4));
/// // Fill with sample data
/// for c in 0..3 {
///     for h in 0..4 {
///         for w in 0..4 {
///             input[[0, c, h, w]] = (c * 16 + h * 4 + w) as f32;
///         }
///     }
/// }
///
/// // Convert to columns with im2col
/// let cols = im2col(&input.view(), (3, 3), (1, 1), (0, 0), (1, 1)).unwrap();
///
/// // Convert back to image with col2im
/// let output = col2im(
///     &cols.view(),
///     (1, 3, 4, 4),
///     (3, 3),
///     (1, 1),
///     (0, 0),
///     (1, 1),
/// ).unwrap();
///
/// // Verify output shape
/// assert_eq!(output.shape(), &[1, 3, 4, 4]);
/// ```
pub fn col2im<F>(
    cols: &ndarray::ArrayView2<F>,
    output_shape: (usize, usize, usize, usize),
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> LinalgResult<Array4<F>>
where
    F: Float + NumAssign + Sum + Zero + ScalarOperand,
{
    let (batch_size, channels, height, width) = output_shape;
    let (kernel_h, kernel_w) = kernel_size;
    let (stride_h, stride_w) = stride;
    let (padding_h, padding_w) = padding;
    let (dilation_h, dilation_w) = dilation;

    // Calculate output dimensions
    let output_h = ((height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1;
    let output_w = ((width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1;

    // Check for valid dimensions
    if output_h == 0 || output_w == 0 {
        return Err(LinalgError::ShapeError(format!(
            "Invalid output dimensions: ({}, {})",
            output_h, output_w
        )));
    }

    // Check input columns shape
    if cols.shape()[0] != kernel_h * kernel_w * channels
        || cols.shape()[1] != output_h * output_w * batch_size
    {
        return Err(LinalgError::ShapeError(format!(
            "Invalid cols shape: expected ({}, {}), got ({}, {})",
            kernel_h * kernel_w * channels,
            output_h * output_w * batch_size,
            cols.shape()[0],
            cols.shape()[1]
        )));
    }

    // Allocate output tensor
    let mut output = Array4::<F>::zeros((batch_size, channels, height, width));
    let mut counts = Array4::<usize>::zeros((batch_size, channels, height, width));

    // Accumulate values from cols to output
    for batch_idx in 0..batch_size {
        for channel_idx in 0..channels {
            for kernel_row in 0..kernel_h {
                for kernel_col in 0..kernel_w {
                    let input_row_offset = kernel_row * dilation_h;
                    let input_col_offset = kernel_col * dilation_w;

                    // Position in the cols matrix
                    let cols_idx =
                        channel_idx * kernel_h * kernel_w + kernel_row * kernel_w + kernel_col;

                    for output_row in 0..output_h {
                        for output_col in 0..output_w {
                            let input_row = output_row * stride_h + input_row_offset;
                            let input_col = output_col * stride_w + input_col_offset;

                            // Position in the cols matrix
                            let cols_pos = batch_idx * output_h * output_w
                                + output_row * output_w
                                + output_col;

                            // Check if the position is valid (not padding)
                            if input_row >= padding_h
                                && input_row < height + padding_h
                                && input_col >= padding_w
                                && input_col < width + padding_w
                            {
                                let output_row_idx = input_row - padding_h;
                                let output_col_idx = input_col - padding_w;

                                output[[batch_idx, channel_idx, output_row_idx, output_col_idx]] +=
                                    cols[[cols_idx, cols_pos]];
                                counts[[batch_idx, channel_idx, output_row_idx, output_col_idx]] +=
                                    1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Normalize by count (average overlapping patches)
    for batch_idx in 0..batch_size {
        for channel_idx in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let count = counts[[batch_idx, channel_idx, h, w]];
                    if count > 0 {
                        output[[batch_idx, channel_idx, h, w]] /= F::from(count).unwrap();
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Perform max pooling operation on a 4D input tensor
///
/// Applies max pooling over a 4D tensor, which is commonly used to
/// down-sample feature maps in convolutional neural networks.
///
/// # Arguments
///
/// * `input` - Input tensor of shape (batch_size, channels, height, width)
/// * `pool_size` - Size of the pooling window as (pool_height, pool_width)
/// * `stride` - Stride as (stride_height, stride_width)
/// * `padding` - Padding as (padding_height, padding_width)
///
/// # Returns
///
/// * Output tensor of pooled values and indices of max values (for backward pass)
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_linalg::convolution::max_pool2d;
///
/// // Create a 1x1x4x4 input tensor
/// let mut input = Array4::<f32>::zeros((1, 1, 4, 4));
/// // Fill with sample data
/// for h in 0..4 {
///     for w in 0..4 {
///         input[[0, 0, h, w]] = (h * 4 + w) as f32;
///     }
/// }
///
/// // Apply 2x2 max pooling with stride 2
/// let (output, indices) = max_pool2d(&input.view(), (2, 2), (2, 2), (0, 0)).unwrap();
///
/// // Resulting tensor has shape (1, 1, 2, 2)
/// assert_eq!(output.shape(), &[1, 1, 2, 2]);
/// ```
pub fn max_pool2d<F>(
    input: &ArrayView4<F>,
    pool_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> LinalgResult<(Array4<F>, Array4<usize>)>
where
    F: Float + NumAssign + Sum + Zero + ScalarOperand,
{
    let (batch_size, channels, height, width) = input.dim();
    let (pool_h, pool_w) = pool_size;
    let (stride_h, stride_w) = stride;
    let (padding_h, padding_w) = padding;

    // Calculate output dimensions
    let output_h = ((height + 2 * padding_h - pool_h) / stride_h) + 1;
    let output_w = ((width + 2 * padding_w - pool_w) / stride_w) + 1;

    // Check for valid dimensions
    if output_h == 0 || output_w == 0 {
        return Err(LinalgError::ShapeError(format!(
            "Invalid output dimensions: ({}, {})",
            output_h, output_w
        )));
    }

    // Allocate output tensors
    let mut output = Array4::<F>::zeros((batch_size, channels, output_h, output_w));
    let mut indices = Array4::<usize>::zeros((batch_size, channels, output_h, output_w));

    // Perform max pooling
    for batch_idx in 0..batch_size {
        for channel_idx in 0..channels {
            for output_row in 0..output_h {
                for output_col in 0..output_w {
                    let start_h = output_row * stride_h;
                    let start_w = output_col * stride_w;

                    let mut max_val = F::neg_infinity();
                    let mut max_idx = 0;

                    // Find max value in the pooling window
                    for pool_row in 0..pool_h {
                        for pool_col in 0..pool_w {
                            let input_row = start_h + pool_row;
                            let input_col = start_w + pool_col;

                            // Check if the position is valid (not padding)
                            if input_row >= padding_h
                                && input_row < height + padding_h
                                && input_col >= padding_w
                                && input_col < width + padding_w
                            {
                                let input_row_idx = input_row - padding_h;
                                let input_col_idx = input_col - padding_w;
                                let val =
                                    input[[batch_idx, channel_idx, input_row_idx, input_col_idx]];

                                if val > max_val {
                                    max_val = val;
                                    max_idx = input_row_idx * width + input_col_idx;
                                }
                            }
                        }
                    }

                    output[[batch_idx, channel_idx, output_row, output_col]] = max_val;
                    indices[[batch_idx, channel_idx, output_row, output_col]] = max_idx;
                }
            }
        }
    }

    Ok((output, indices))
}

/// Perform the backward pass of max pooling operation
///
/// Takes the gradients of the pooled outputs and distributes them back to
/// the locations of the maximum values in the original input.
///
/// # Arguments
///
/// * `grad_output` - Gradient of the output tensor of shape (batch_size, channels, output_height, output_width)
/// * `indices` - Indices of the maximum values from the forward pass
/// * `input_shape` - Shape of the original input tensor (batch_size, channels, height, width)
///
/// # Returns
///
/// * Gradient with respect to input
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_linalg::convolution::{max_pool2d, max_pool2d_backward};
///
/// // Create a 1x1x4x4 input tensor
/// let mut input = Array4::<f32>::zeros((1, 1, 4, 4));
/// // Fill with sample data
/// for h in 0..4 {
///     for w in 0..4 {
///         input[[0, 0, h, w]] = (h * 4 + w) as f32;
///     }
/// }
///
/// // Apply max pooling (forward pass)
/// let (output, indices) = max_pool2d(&input.view(), (2, 2), (2, 2), (0, 0)).unwrap();
///
/// // Create gradient of the output
/// let mut grad_output = Array4::<f32>::ones((1, 1, 2, 2));
///
/// // Compute gradient of the input (backward pass)
/// let grad_input = max_pool2d_backward(
///     &grad_output.view(),
///     &indices.view(),
///     (1, 1, 4, 4),
/// ).unwrap();
///
/// // Verify shape
/// assert_eq!(grad_input.shape(), &[1, 1, 4, 4]);
/// ```
pub fn max_pool2d_backward<F>(
    grad_output: &ArrayView4<F>,
    indices: &ndarray::ArrayView4<usize>,
    input_shape: (usize, usize, usize, usize),
) -> LinalgResult<Array4<F>>
where
    F: Float + NumAssign + Sum + Zero + ScalarOperand,
{
    let (batch_size, channels, height, width) = input_shape;
    let (out_batch, out_channels, out_height, out_width) = grad_output.dim();
    let (idx_batch, idx_channels, idx_height, idx_width) = indices.dim();

    // Check that shapes match
    if out_batch != idx_batch
        || out_channels != idx_channels
        || out_height != idx_height
        || out_width != idx_width
    {
        return Err(LinalgError::ShapeError(format!(
            "Shape mismatch between grad_output ({}, {}, {}, {}) and indices ({}, {}, {}, {})",
            out_batch,
            out_channels,
            out_height,
            out_width,
            idx_batch,
            idx_channels,
            idx_height,
            idx_width
        )));
    }

    // Allocate output gradient tensor
    let mut grad_input = Array4::<F>::zeros((batch_size, channels, height, width));

    // Distribute gradients to the locations of the maximum values
    for batch_idx in 0..out_batch {
        for channel_idx in 0..out_channels {
            for h in 0..out_height {
                for w in 0..out_width {
                    let index = indices[[batch_idx, channel_idx, h, w]];
                    let input_h = index / width;
                    let input_w = index % width;

                    if input_h < height && input_w < width {
                        grad_input[[batch_idx, channel_idx, input_h, input_w]] +=
                            grad_output[[batch_idx, channel_idx, h, w]];
                    }
                }
            }
        }
    }

    Ok(grad_input)
}

/// Compute the indices for batch matrix multiplication in a convolutional layer
///
/// This function computes the indices needed for efficient batch matrix multiplication
/// in a convolutional layer, which can be used to implement convolutional layers
/// more efficiently.
///
/// # Arguments
///
/// * `input_shape` - Shape of the input tensor (batch_size, channels, height, width)
/// * `kernel_shape` - Shape of the kernel tensor (out_channels, in_channels, kernel_h, kernel_w)
/// * `stride` - Stride as (stride_height, stride_width)
/// * `padding` - Padding as (padding_height, padding_width)
///
/// # Returns
///
/// * Indices for efficient batch matrix multiplication
///
/// # Examples
///
/// ```
/// use scirs2_linalg::convolution::compute_conv_indices;
///
/// // Compute indices for a simple convolutional layer
/// let indices = compute_conv_indices(
///     (1, 1, 4, 4),    // Input shape: batch_size=1, channels=1, height=4, width=4
///     (1, 1, 2, 2),    // Kernel shape: out_channels=1, in_channels=1, kernel_h=2, kernel_w=2
///     (1, 1),          // Stride: height=1, width=1
///     (0, 0),          // Padding: height=0, width=0
/// ).unwrap();
/// // For a 4x4 input with 2x2 kernel and no padding, we get a 3x3 output
/// // Each output element is computed from 4 input elements (2x2 kernel)
/// // So we should have 3*3*4*5 = 180 values in the indices array
/// assert_eq!(indices.len() % 5, 0); // Should be multiple of 5
/// ```
pub fn compute_conv_indices(
    input_shape: (usize, usize, usize, usize),
    kernel_shape: (usize, usize, usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> LinalgResult<ndarray::Array1<usize>> {
    let (batch_size, in_channels, height, width) = input_shape;
    let (out_channels, _, kernel_h, kernel_w) = kernel_shape;
    let (stride_h, stride_w) = stride;
    let (padding_h, padding_w) = padding;

    // Calculate output dimensions
    let output_h = ((height + 2 * padding_h - kernel_h) / stride_h) + 1;
    let output_w = ((width + 2 * padding_w - kernel_w) / stride_w) + 1;

    // Check for valid dimensions
    if output_h == 0 || output_w == 0 {
        return Err(LinalgError::ShapeError(format!(
            "Invalid output dimensions: ({}, {})",
            output_h, output_w
        )));
    }

    // Calculate total number of elements
    // Each output element can be computed from in_channels * kernel_h * kernel_w input elements
    let total_elements =
        batch_size * out_channels * output_h * output_w * in_channels * kernel_h * kernel_w;

    // Allocate array for indices (5 values per element)
    let mut indices = ndarray::Array1::<usize>::zeros(total_elements * 5);

    // Compute indices for batch matmul
    let mut idx = 0;
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..output_h {
                for ow in 0..output_w {
                    for ic in 0..in_channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                // Check if within padded input
                                if ih >= padding_h
                                    && ih < height + padding_h
                                    && iw >= padding_w
                                    && iw < width + padding_w
                                {
                                    let real_ih = ih - padding_h;
                                    let real_iw = iw - padding_w;

                                    // Output index
                                    let out_idx = b * out_channels * output_h * output_w
                                        + oc * output_h * output_w
                                        + oh * output_w
                                        + ow;

                                    // Input index
                                    let in_idx = b * in_channels * height * width
                                        + ic * height * width
                                        + real_ih * width
                                        + real_iw;

                                    // Kernel index
                                    let kernel_idx = oc * in_channels * kernel_h * kernel_w
                                        + ic * kernel_h * kernel_w
                                        + kh * kernel_w
                                        + kw;

                                    // Store indices
                                    indices[idx] = out_idx;
                                    indices[idx + 1] = in_idx;
                                    indices[idx + 2] = kernel_idx;
                                    indices[idx + 3] = oh * output_w + ow;
                                    indices[idx + 4] = oc;

                                    idx += 5;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Resize the array to remove unused elements
    let indices = indices.slice(ndarray::s![0..idx]).to_owned();
    Ok(indices)
}

/// Apply convolution operation using im2col and matrix multiplication
///
/// This function implements convolution using the im2col algorithm and
/// efficient matrix multiplication, which is often faster than direct
/// convolution for large inputs or kernels.
///
/// # Arguments
///
/// * `input` - Input tensor of shape (batch_size, channels, height, width)
/// * `kernel` - Kernel tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
/// * `bias` - Optional bias tensor of shape (out_channels,)
/// * `stride` - Stride as (stride_height, stride_width)
/// * `padding` - Padding as (padding_height, padding_width)
/// * `dilation` - Dilation as (dilation_height, dilation_width)
///
/// # Returns
///
/// * Output tensor of shape (batch_size, out_channels, output_height, output_width)
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array4};
/// use scirs2_linalg::convolution::conv2d_im2col;
///
/// // Create a 2x3x32x32 input tensor (2 batches, 3 channels, 32x32 spatial dimensions)
/// let input = Array4::<f32>::zeros((2, 3, 32, 32));
///
/// // Create a 16x3x3x3 kernel tensor (16 output channels, 3 input channels, 3x3 kernel)
/// let kernel = Array4::<f32>::zeros((16, 3, 3, 3));
///
/// // Create a bias tensor
/// let bias = Some(Array::zeros(16));
///
/// // Apply convolution
/// let output = conv2d_im2col(
///     &input.view(),
///     &kernel.view(),
///     bias.as_ref().map(|b| b.view()),
///     (1, 1),  // stride
///     (1, 1),  // padding
///     (1, 1),  // dilation
/// ).unwrap();
///
/// // Output shape is (2, 16, 32, 32)
/// assert_eq!(output.shape(), &[2, 16, 32, 32]);
/// ```
pub fn conv2d_im2col<F>(
    input: &ArrayView4<F>,
    kernel: &ArrayView4<F>,
    bias: Option<ndarray::ArrayView1<F>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> LinalgResult<Array4<F>>
where
    F: Float + NumAssign + Sum + Zero + ScalarOperand,
{
    let (batch_size, in_channels, height, width) = input.dim();
    let (out_channels, k_in_channels, kernel_h, kernel_w) = kernel.dim();

    // Check that input and kernel channels match
    if in_channels != k_in_channels {
        return Err(LinalgError::ShapeError(format!(
            "Input channels ({}) must match kernel in_channels ({})",
            in_channels, k_in_channels
        )));
    }

    // Check bias shape if provided
    if let Some(b) = bias {
        if b.len() != out_channels {
            return Err(LinalgError::ShapeError(format!(
                "Bias length ({}) must match out_channels ({})",
                b.len(),
                out_channels
            )));
        }
    }

    // Calculate output dimensions
    let (stride_h, stride_w) = stride;
    let (padding_h, padding_w) = padding;
    let (dilation_h, dilation_w) = dilation;

    let output_h = ((height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1;
    let output_w = ((width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1;

    // Check for valid dimensions
    if output_h == 0 || output_w == 0 {
        return Err(LinalgError::ShapeError(format!(
            "Invalid output dimensions: ({}, {})",
            output_h, output_w
        )));
    }

    // Convert input to columns using im2col
    let cols = im2col(input, (kernel_h, kernel_w), stride, padding, dilation)?;

    // Reshape kernel for matrix multiplication
    let flat_kernel = (*kernel)
        .into_shape_with_order((out_channels, in_channels * kernel_h * kernel_w))
        .map_err(|e| LinalgError::ShapeError(e.to_string()))?;

    // Perform matrix multiplication
    let output_2d = flat_kernel.dot(&cols);

    // Reshape to output tensor
    let mut output = output_2d
        .into_shape_with_order((out_channels, batch_size, output_h, output_w))
        .map_err(|e| LinalgError::ShapeError(e.to_string()))?;

    // Rearrange dimensions to (batch_size, out_channels, output_h, output_w)
    output = output.permuted_axes([1, 0, 2, 3]);

    // Add bias if provided
    if let Some(b) = bias {
        for batch_idx in 0..batch_size {
            for oc in 0..out_channels {
                for h in 0..output_h {
                    for w in 0..output_w {
                        output[[batch_idx, oc, h, w]] += b[oc];
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Apply backward pass of convolution operation for input gradient
///
/// This function computes the gradient of the input in a convolutional layer
/// given the gradient of the output.
///
/// # Arguments
///
/// * `grad_output` - Gradient of the output tensor of shape (batch_size, out_channels, output_h, output_w)
/// * `kernel` - Kernel tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
/// * `input_shape` - Shape of the input tensor (batch_size, in_channels, height, width)
/// * `stride` - Stride as (stride_height, stride_width)
/// * `padding` - Padding as (padding_height, padding_width)
/// * `dilation` - Dilation as (dilation_height, dilation_width)
///
/// # Returns
///
/// * Gradient of the input tensor of shape (batch_size, in_channels, height, width)
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_linalg::convolution::{conv2d_im2col, conv2d_backward_input};
///
/// // Forward pass
/// let input = Array4::<f32>::zeros((2, 3, 32, 32));
/// let kernel = Array4::<f32>::zeros((16, 3, 3, 3));
/// let bias = None;
/// let output = conv2d_im2col(
///     &input.view(),
///     &kernel.view(),
///     bias,
///     (1, 1),
///     (1, 1),
///     (1, 1),
/// ).unwrap();
///
/// // Backward pass
/// let grad_output = Array4::<f32>::ones((2, 16, 32, 32));
/// let grad_input = conv2d_backward_input(
///     &grad_output.view(),
///     &kernel.view(),
///     (2, 3, 32, 32),
///     (1, 1),
///     (1, 1),
///     (1, 1),
/// ).unwrap();
///
/// // Gradient shape matches input shape
/// assert_eq!(grad_input.shape(), &[2, 3, 32, 32]);
/// ```
pub fn conv2d_backward_input<F>(
    grad_output: &ArrayView4<F>,
    kernel: &ArrayView4<F>,
    input_shape: (usize, usize, usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> LinalgResult<Array4<F>>
where
    F: Float + NumAssign + Sum + Zero + ScalarOperand,
{
    let (batch_size, out_channels, _output_h, _output_w) = grad_output.dim();
    let (k_out_channels, in_channels, kernel_h, kernel_w) = kernel.dim();
    let (i_batch_size, i_in_channels, _height, _width) = input_shape;

    // Check that shapes match
    if batch_size != i_batch_size {
        return Err(LinalgError::ShapeError(format!(
            "Batch size mismatch: grad_output ({}) vs input_shape ({})",
            batch_size, i_batch_size
        )));
    }

    if out_channels != k_out_channels {
        return Err(LinalgError::ShapeError(format!(
            "Output channels mismatch: grad_output ({}) vs kernel ({})",
            out_channels, k_out_channels
        )));
    }

    if in_channels != i_in_channels {
        return Err(LinalgError::ShapeError(format!(
            "Input channels mismatch: kernel ({}) vs input_shape ({})",
            in_channels, i_in_channels
        )));
    }

    // Prepare kernel for transposed convolution
    let mut kernel_transposed = Array4::<F>::zeros((in_channels, out_channels, kernel_h, kernel_w));

    // Flip the kernel and transpose input/output channels
    for oc in 0..out_channels {
        for ic in 0..in_channels {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    kernel_transposed[[ic, oc, kernel_h - 1 - kh, kernel_w - 1 - kw]] =
                        kernel[[oc, ic, kh, kw]];
                }
            }
        }
    }

    // Calculate padding for transposed convolution
    let (_stride_h, _stride_w) = stride;
    let (padding_h, padding_w) = padding;
    let (_dilation_h, _dilation_w) = dilation;

    // We need to adjust padding for transposed convolution
    let pad_h = kernel_h - 1 - padding_h;
    let pad_w = kernel_w - 1 - padding_w;

    // Perform transposed convolution
    // For transposed convolution, we swap the roles of stride and dilation
    conv2d_im2col(
        grad_output,
        &kernel_transposed.view(),
        None,
        dilation,       // original dilation becomes stride
        (pad_h, pad_w), // adjusted padding
        stride,         // original stride becomes dilation
    )
}

/// Apply backward pass of convolution operation for kernel gradient
///
/// This function computes the gradient of the kernel in a convolutional layer
/// given the gradient of the output and the input.
///
/// # Arguments
///
/// * `input` - Input tensor of shape (batch_size, in_channels, height, width)
/// * `grad_output` - Gradient of the output tensor of shape (batch_size, out_channels, output_h, output_w)
/// * `kernel_shape` - Shape of the kernel tensor (out_channels, in_channels, kernel_h, kernel_w)
/// * `stride` - Stride as (stride_height, stride_width)
/// * `padding` - Padding as (padding_height, padding_width)
/// * `dilation` - Dilation as (dilation_height, dilation_width)
///
/// # Returns
///
/// * Gradient of the kernel tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_linalg::convolution::{conv2d_im2col, conv2d_backward_kernel};
///
/// // Simple example with smaller dimensions
/// let input = Array4::<f32>::zeros((1, 1, 4, 4));
/// let kernel_shape = (1, 1, 2, 2);
///
/// // Forward pass to get output shape
/// let kernel = Array4::<f32>::zeros(kernel_shape);
/// let output = conv2d_im2col(
///     &input.view(),
///     &kernel.view(),
///     None,
///     (1, 1),  // stride
///     (0, 0),  // padding
///     (1, 1),  // dilation
/// ).unwrap();
///
/// // Backward pass - grad_output must match forward output shape
/// let grad_output = Array4::<f32>::ones(output.dim());
/// let grad_kernel = conv2d_backward_kernel(
///     &input.view(),
///     &grad_output.view(),
///     kernel_shape,
///     (1, 1),
///     (0, 0),
///     (1, 1),
/// ).unwrap();
///
/// // Gradient shape matches kernel shape
/// assert_eq!(grad_kernel.shape(), &[1, 1, 2, 2]);
/// ```
pub fn conv2d_backward_kernel<F>(
    input: &ArrayView4<F>,
    grad_output: &ArrayView4<F>,
    kernel_shape: (usize, usize, usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> LinalgResult<Array4<F>>
where
    F: Float + NumAssign + Sum + Zero + ScalarOperand,
{
    let (batch_size, in_channels, _height, _width) = input.dim();
    let (go_batch_size, out_channels, output_h, output_w) = grad_output.dim();
    let (k_out_channels, k_in_channels, kernel_h, kernel_w) = kernel_shape;

    // Check that shapes match
    if batch_size != go_batch_size {
        return Err(LinalgError::ShapeError(format!(
            "Batch size mismatch: input ({}) vs grad_output ({})",
            batch_size, go_batch_size
        )));
    }

    if out_channels != k_out_channels {
        return Err(LinalgError::ShapeError(format!(
            "Output channels mismatch: grad_output ({}) vs kernel_shape ({})",
            out_channels, k_out_channels
        )));
    }

    if in_channels != k_in_channels {
        return Err(LinalgError::ShapeError(format!(
            "Input channels mismatch: input ({}) vs kernel_shape ({})",
            in_channels, k_in_channels
        )));
    }

    // Convert input to columns using im2col
    let cols = im2col(input, (kernel_h, kernel_w), stride, padding, dilation)?;

    // Reshape grad_output for matrix multiplication
    let grad_output_reshaped = (*grad_output)
        .into_shape_with_order((batch_size * out_channels, output_h * output_w))
        .map_err(|e| LinalgError::ShapeError(e.to_string()))?;

    // Compute kernel gradient using matrix multiplication
    let grad_kernel_flat = grad_output_reshaped.dot(&cols.t());

    // Reshape to kernel shape
    let grad_kernel = grad_kernel_flat
        .into_shape_with_order((out_channels, in_channels, kernel_h, kernel_w))
        .map_err(|e| LinalgError::ShapeError(e.to_string()))?;

    Ok(grad_kernel)
}

/// Apply backward pass of convolution operation for bias gradient
///
/// This function computes the gradient of the bias in a convolutional layer
/// given the gradient of the output.
///
/// # Arguments
///
/// * `grad_output` - Gradient of the output tensor of shape (batch_size, out_channels, output_h, output_w)
///
/// # Returns
///
/// * Gradient of the bias tensor of shape (out_channels,)
///
/// # Examples
///
/// ```
/// use ndarray::Array4;
/// use scirs2_linalg::convolution::conv2d_backward_bias;
///
/// // Backward pass for bias
/// let grad_output = Array4::<f32>::ones((2, 16, 32, 32));
/// let grad_bias = conv2d_backward_bias(&grad_output.view()).unwrap();
///
/// // Gradient shape matches bias shape
/// assert_eq!(grad_bias.shape(), &[16]);
/// ```
pub fn conv2d_backward_bias<F>(grad_output: &ArrayView4<F>) -> LinalgResult<ndarray::Array1<F>>
where
    F: Float + NumAssign + Sum + Zero,
{
    let (batch_size, out_channels, output_h, output_w) = grad_output.dim();

    // Allocate gradient for bias
    let mut grad_bias = ndarray::Array1::<F>::zeros(out_channels);

    // Sum gradients over batch, height, and width dimensions
    for batch_idx in 0..batch_size {
        for oc in 0..out_channels {
            for h in 0..output_h {
                for w in 0..output_w {
                    grad_bias[oc] += grad_output[[batch_idx, oc, h, w]];
                }
            }
        }
    }

    Ok(grad_bias)
}

/// Apply 2D transposed convolution (deconvolution) operation
///
/// This function implements a transposed convolution (also known as deconvolution
/// or fractionally-strided convolution), which is commonly used in convolutional
/// neural networks for upsampling.
///
/// # Arguments
///
/// * `input` - Input tensor of shape (batch_size, channels, height, width)
/// * `kernel` - Kernel tensor of shape (in_channels, out_channels, kernel_h, kernel_w)
/// * `bias` - Optional bias tensor of shape (out_channels,)
/// * `stride` - Stride as (stride_height, stride_width)
/// * `padding` - Padding as (padding_height, padding_width)
/// * `output_padding` - Additional padding for output as (padding_height, padding_width)
/// * `dilation` - Dilation as (dilation_height, dilation_width)
///
/// # Returns
///
/// * Output tensor of shape (batch_size, out_channels, output_height, output_width)
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array4};
/// use scirs2_linalg::convolution::conv_transpose2d;
///
/// // Simple example with smaller dimensions
/// let input = Array4::<f32>::zeros((1, 2, 3, 3));
///
/// // Create a 2x1x2x2 kernel tensor (2 input channels, 1 output channel, 2x2 kernel)
/// let kernel = Array4::<f32>::zeros((2, 1, 2, 2));
///
/// // Apply transposed convolution
/// let output = conv_transpose2d(
///     &input.view(),
///     &kernel.view(),
///     None,        // no bias
///     (1, 1),      // stride
///     (0, 0),      // padding
///     (0, 0),      // output_padding
///     (1, 1),      // dilation
/// ).unwrap();
///
/// // Calculate expected output shape:
/// // output_h = (3 - 1) * 1 - 2 * 0 + 1 * (2 - 1) + 0 + 1 = 2 + 1 + 1 = 4
/// // output_w = (3 - 1) * 1 - 2 * 0 + 1 * (2 - 1) + 0 + 1 = 2 + 1 + 1 = 4
/// assert_eq!(output.shape(), &[1, 1, 4, 4]);
/// ```
pub fn conv_transpose2d<F>(
    input: &ArrayView4<F>,
    kernel: &ArrayView4<F>,
    bias: Option<ndarray::ArrayView1<F>>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    dilation: (usize, usize),
) -> LinalgResult<Array4<F>>
where
    F: Float + NumAssign + Sum + Zero + ScalarOperand,
{
    let (batch_size, in_channels, height, width) = input.dim();
    let (k_in_channels, out_channels, kernel_h, kernel_w) = kernel.dim();

    // Check that channels match
    if in_channels != k_in_channels {
        return Err(LinalgError::ShapeError(format!(
            "Input channels mismatch: input ({}) vs kernel ({})",
            in_channels, k_in_channels
        )));
    }

    // Check bias shape if provided
    if let Some(b) = bias {
        if b.len() != out_channels {
            return Err(LinalgError::ShapeError(format!(
                "Bias length ({}) must match out_channels ({})",
                b.len(),
                out_channels
            )));
        }
    }

    // Calculate output dimensions
    let (stride_h, stride_w) = stride;
    let (padding_h, padding_w) = padding;
    let (output_padding_h, output_padding_w) = output_padding;
    let (dilation_h, dilation_w) = dilation;

    let output_h = (height - 1) * stride_h - 2 * padding_h
        + dilation_h * (kernel_h - 1)
        + output_padding_h
        + 1;
    let output_w =
        (width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    // Allocate output tensor
    let mut output = Array4::<F>::zeros((batch_size, out_channels, output_h, output_w));

    // Perform transposed convolution
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for ic in 0..in_channels {
                for h in 0..height {
                    for w in 0..width {
                        let input_val = input[[b, ic, h, w]];

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                // Calculate output coordinates for transposed convolution
                                // For transposed conv, we're scattering values from input to output
                                // The calculation needs to account for the kernel's position in the opposite way
                                // than normal convolution
                                let out_h = h as isize * stride_h as isize
                                    + kh as isize * dilation_h as isize
                                    - padding_h as isize;
                                let out_w = w as isize * stride_w as isize
                                    + kw as isize * dilation_w as isize
                                    - padding_w as isize;

                                // Only process if coordinates are non-negative
                                if out_h >= 0 && out_w >= 0 {
                                    let out_h = out_h as usize;
                                    let out_w = out_w as usize;

                                    if out_h < output_h && out_w < output_w {
                                        // The transposed convolution can be thought of as:
                                        // 1. For each input position and each kernel position
                                        // 2. Calculate the output position based on stride, padding, and dilation
                                        // 3. Add the product of input and kernel value to that output position
                                        //
                                        // Note: Technically, for a proper mathematical transposed convolution,
                                        // we should use the flipped kernel. However, in practice, ML libraries
                                        // often just reuse the same kernel for simplicity and learn appropriate weights.
                                        output[[b, oc, out_h, out_w]] +=
                                            input_val * kernel[[ic, oc, kh, kw]];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Add bias if provided
            if let Some(b_val) = bias.map(|b| b[oc]) {
                for h in 0..output_h {
                    for w in 0..output_w {
                        output[[b, oc, h, w]] += b_val;
                    }
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array1, Array4};

    #[test]
    fn test_im2col_basic() {
        // Create a simple 1x1x3x3 input
        let mut input = Array4::<f32>::zeros((1, 1, 3, 3));
        for h in 0..3 {
            for w in 0..3 {
                input[[0, 0, h, w]] = (h * 3 + w) as f32;
            }
        }

        // Extract 2x2 patches with stride 1 and no padding
        let cols = im2col(&input.view(), (2, 2), (1, 1), (0, 0), (1, 1)).unwrap();

        // Resulting matrix should be (1*2*2, 2*2*1) = (4, 4)
        assert_eq!(cols.shape(), &[4, 4]);

        // Check the first column (top-left 2x2 patch)
        assert_eq!(cols[[0, 0]], 0.0);
        assert_eq!(cols[[1, 0]], 1.0);
        assert_eq!(cols[[2, 0]], 3.0);
        assert_eq!(cols[[3, 0]], 4.0);

        // Check the second column (top-right 2x2 patch)
        assert_eq!(cols[[0, 1]], 1.0);
        assert_eq!(cols[[1, 1]], 2.0);
        assert_eq!(cols[[2, 1]], 4.0);
        assert_eq!(cols[[3, 1]], 5.0);
    }

    #[test]
    fn test_im2col_with_padding() {
        // Create a simple 1x1x2x2 input
        let mut input = Array4::<f32>::zeros((1, 1, 2, 2));
        input[[0, 0, 0, 0]] = 0.0;
        input[[0, 0, 0, 1]] = 1.0;
        input[[0, 0, 1, 0]] = 2.0;
        input[[0, 0, 1, 1]] = 3.0;

        // Extract 3x3 patches with stride 1 and padding 1
        let cols = im2col(&input.view(), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();

        // Resulting matrix should be (1*3*3, 2*2*1) = (9, 4)
        assert_eq!(cols.shape(), &[9, 4]);

        // Check padding is zero
        assert_eq!(cols[[0, 0]], 0.0); // Top-left padding
        assert_eq!(cols[[2, 0]], 0.0); // Top-right padding
        assert_eq!(cols[[6, 0]], 0.0); // Bottom-left padding
        assert_eq!(cols[[8, 0]], 3.0); // Bottom-right padding - this corresponds to input[1,1] at position (2,2)

        // Check actual values - kernel center (1,1) at patch (0,0) corresponds to input (0,0)
        assert_eq!(cols[[4, 0]], 0.0); // Center of first patch corresponds to input[0,0,0,0]
    }

    #[test]
    fn test_col2im_basic() {
        // Create a simple 1x1x3x3 input
        let mut input = Array4::<f32>::zeros((1, 1, 3, 3));
        for h in 0..3 {
            for w in 0..3 {
                input[[0, 0, h, w]] = (h * 3 + w) as f32;
            }
        }

        // Convert to columns
        let cols = im2col(&input.view(), (2, 2), (1, 1), (0, 0), (1, 1)).unwrap();

        // Convert back to image
        let output = col2im(&cols.view(), (1, 1, 3, 3), (2, 2), (1, 1), (0, 0), (1, 1)).unwrap();

        // Check dimensions
        assert_eq!(output.shape(), input.shape());

        // Check values (note that overlapping patches are averaged)
        assert_relative_eq!(output[[0, 0, 0, 0]], input[[0, 0, 0, 0]], epsilon = 1e-5);
        assert_relative_eq!(output[[0, 0, 0, 1]], input[[0, 0, 0, 1]], epsilon = 1e-5);
        assert_relative_eq!(output[[0, 0, 0, 2]], input[[0, 0, 0, 2]], epsilon = 1e-5);
        assert_relative_eq!(output[[0, 0, 1, 0]], input[[0, 0, 1, 0]], epsilon = 1e-5);
        assert_relative_eq!(output[[0, 0, 1, 1]], input[[0, 0, 1, 1]], epsilon = 1e-5);
        assert_relative_eq!(output[[0, 0, 1, 2]], input[[0, 0, 1, 2]], epsilon = 1e-5);
        assert_relative_eq!(output[[0, 0, 2, 0]], input[[0, 0, 2, 0]], epsilon = 1e-5);
        assert_relative_eq!(output[[0, 0, 2, 1]], input[[0, 0, 2, 1]], epsilon = 1e-5);
        assert_relative_eq!(output[[0, 0, 2, 2]], input[[0, 0, 2, 2]], epsilon = 1e-5);
    }

    #[test]
    fn test_max_pool2d() {
        // Create a simple 1x1x4x4 input
        let mut input = Array4::<f32>::zeros((1, 1, 4, 4));
        for h in 0..4 {
            for w in 0..4 {
                input[[0, 0, h, w]] = (h * 4 + w) as f32;
            }
        }

        // Apply 2x2 max pooling with stride 2
        let (output, indices) = max_pool2d(&input.view(), (2, 2), (2, 2), (0, 0)).unwrap();

        // Check dimensions
        assert_eq!(output.shape(), &[1, 1, 2, 2]);

        // Check values (should take max from each 2x2 region)
        assert_eq!(output[[0, 0, 0, 0]], 5.0); // max of top-left 2x2
        assert_eq!(output[[0, 0, 0, 1]], 7.0); // max of top-right 2x2
        assert_eq!(output[[0, 0, 1, 0]], 13.0); // max of bottom-left 2x2
        assert_eq!(output[[0, 0, 1, 1]], 15.0); // max of bottom-right 2x2

        // Check indices
        assert_eq!(indices[[0, 0, 0, 0]], 5); // index of 5 in flattened input
        assert_eq!(indices[[0, 0, 0, 1]], 7); // index of 7 in flattened input
        assert_eq!(indices[[0, 0, 1, 0]], 13); // index of 13 in flattened input
        assert_eq!(indices[[0, 0, 1, 1]], 15); // index of 15 in flattened input
    }

    #[test]
    fn test_max_pool2d_backward() {
        // Create a simple 1x1x4x4 input
        let mut input = Array4::<f32>::zeros((1, 1, 4, 4));
        for h in 0..4 {
            for w in 0..4 {
                input[[0, 0, h, w]] = (h * 4 + w) as f32;
            }
        }

        // Forward pass
        let (_output, indices) = max_pool2d(&input.view(), (2, 2), (2, 2), (0, 0)).unwrap();

        // Create gradient of output
        let grad_output = Array4::<f32>::ones((1, 1, 2, 2));

        // Backward pass
        let grad_input =
            max_pool2d_backward(&grad_output.view(), &indices.view(), (1, 1, 4, 4)).unwrap();

        // Check dimensions
        assert_eq!(grad_input.shape(), input.shape());

        // Only positions with max values should have gradients
        for h in 0..4 {
            for w in 0..4 {
                let pos = h * 4 + w;
                let expected = if pos == 5 || pos == 7 || pos == 13 || pos == 15 {
                    1.0
                } else {
                    0.0
                };
                assert_eq!(grad_input[[0, 0, h, w]], expected);
            }
        }
    }

    #[test]
    fn test_conv2d_im2col_basic() {
        // Create a simple 1x1x3x3 input
        let mut input = Array4::<f32>::zeros((1, 1, 3, 3));
        for h in 0..3 {
            for w in 0..3 {
                input[[0, 0, h, w]] = (h * 3 + w) as f32;
            }
        }

        // Create a simple 1x1x2x2 kernel (identity)
        let mut kernel = Array4::<f32>::zeros((1, 1, 2, 2));
        kernel[[0, 0, 0, 0]] = 1.0;
        kernel[[0, 0, 0, 1]] = 0.0;
        kernel[[0, 0, 1, 0]] = 0.0;
        kernel[[0, 0, 1, 1]] = 0.0;

        // Apply convolution
        let output =
            conv2d_im2col(&input.view(), &kernel.view(), None, (1, 1), (0, 0), (1, 1)).unwrap();

        // Check dimensions
        assert_eq!(output.shape(), &[1, 1, 2, 2]);

        // Kernel extracts top-left value from each position
        assert_eq!(output[[0, 0, 0, 0]], 0.0);
        assert_eq!(output[[0, 0, 0, 1]], 1.0);
        assert_eq!(output[[0, 0, 1, 0]], 3.0);
        assert_eq!(output[[0, 0, 1, 1]], 4.0);
    }

    #[test]
    fn test_conv2d_im2col_with_bias() {
        // Create a simple 1x1x3x3 input
        let mut input = Array4::<f32>::zeros((1, 1, 3, 3));
        for h in 0..3 {
            for w in 0..3 {
                input[[0, 0, h, w]] = (h * 3 + w) as f32;
            }
        }

        // Create a simple 1x1x2x2 kernel (identity)
        let mut kernel = Array4::<f32>::zeros((1, 1, 2, 2));
        kernel[[0, 0, 0, 0]] = 1.0;
        kernel[[0, 0, 0, 1]] = 0.0;
        kernel[[0, 0, 1, 0]] = 0.0;
        kernel[[0, 0, 1, 1]] = 0.0;

        // Create bias
        let bias = Array1::<f32>::from_elem(1, 10.0);

        // Apply convolution with bias
        let output = conv2d_im2col(
            &input.view(),
            &kernel.view(),
            Some(bias.view()),
            (1, 1),
            (0, 0),
            (1, 1),
        )
        .unwrap();

        // Check dimensions
        assert_eq!(output.shape(), &[1, 1, 2, 2]);

        // Kernel extracts top-left value from each position, plus bias
        assert_eq!(output[[0, 0, 0, 0]], 10.0);
        assert_eq!(output[[0, 0, 0, 1]], 11.0);
        assert_eq!(output[[0, 0, 1, 0]], 13.0);
        assert_eq!(output[[0, 0, 1, 1]], 14.0);
    }

    #[test]
    fn test_conv2d_backward_input() {
        // Create a simple 1x1x3x3 input
        let input = Array4::<f32>::zeros((1, 1, 3, 3));

        // Create a simple 1x1x2x2 kernel
        let mut kernel = Array4::<f32>::zeros((1, 1, 2, 2));
        kernel[[0, 0, 0, 0]] = 1.0;
        kernel[[0, 0, 0, 1]] = 2.0;
        kernel[[0, 0, 1, 0]] = 3.0;
        kernel[[0, 0, 1, 1]] = 4.0;

        // Apply forward pass
        let _output =
            conv2d_im2col(&input.view(), &kernel.view(), None, (1, 1), (0, 0), (1, 1)).unwrap();

        // Create gradient of output
        let grad_output = Array4::<f32>::ones((1, 1, 2, 2));

        // Apply backward pass for input
        let grad_input = conv2d_backward_input(
            &grad_output.view(),
            &kernel.view(),
            (1, 1, 3, 3),
            (1, 1),
            (0, 0),
            (1, 1),
        )
        .unwrap();

        // Check dimensions
        assert_eq!(grad_input.shape(), input.shape());

        // Each position receives weighted gradients from overlapping filters
        // For gradient=1 at each output position:
        // input[0,0] receives 1.0 from output[0,0]
        // input[0,1] receives 2.0 from output[0,0] + 1.0 from output[0,1] = 3.0
        // etc.
        assert_eq!(grad_input[[0, 0, 0, 0]], 1.0);
        assert_eq!(grad_input[[0, 0, 0, 1]], 3.0);
        assert_eq!(grad_input[[0, 0, 1, 0]], 4.0);
        assert_eq!(grad_input[[0, 0, 1, 1]], 10.0);
    }

    #[test]
    fn test_conv2d_backward_kernel() {
        // Create a simple 1x1x3x3 input with all ones
        let input = Array4::<f32>::ones((1, 1, 3, 3));

        // Create gradient of output, all ones
        let grad_output = Array4::<f32>::ones((1, 1, 2, 2));

        // Apply backward pass for kernel
        let grad_kernel = conv2d_backward_kernel(
            &input.view(),
            &grad_output.view(),
            (1, 1, 2, 2),
            (1, 1),
            (0, 0),
            (1, 1),
        )
        .unwrap();

        // Check dimensions
        assert_eq!(grad_kernel.shape(), &[1, 1, 2, 2]);

        // With all ones input and gradient, each kernel position accumulates
        // the number of times it overlaps with the input
        assert_eq!(grad_kernel[[0, 0, 0, 0]], 4.0); // Overlaps 4 times
        assert_eq!(grad_kernel[[0, 0, 0, 1]], 4.0);
        assert_eq!(grad_kernel[[0, 0, 1, 0]], 4.0);
        assert_eq!(grad_kernel[[0, 0, 1, 1]], 4.0);
    }

    #[test]
    fn test_conv2d_backward_bias() {
        // Create gradient of output
        let mut grad_output = Array4::<f32>::zeros((2, 3, 2, 2));
        for b in 0..2 {
            for c in 0..3 {
                for h in 0..2 {
                    for w in 0..2 {
                        grad_output[[b, c, h, w]] = 1.0;
                    }
                }
            }
        }

        // Apply backward pass for bias
        let grad_bias = conv2d_backward_bias(&grad_output.view()).unwrap();

        // Check dimensions
        assert_eq!(grad_bias.shape(), &[3]);

        // Each bias accumulates gradient from all positions and batches
        assert_eq!(grad_bias[0], 8.0); // 2 batches * 2*2 spatial = 8
        assert_eq!(grad_bias[1], 8.0);
        assert_eq!(grad_bias[2], 8.0);
    }

    #[test]
    fn test_conv_transpose2d() {
        // Create a simple 1x1x2x2 input
        let input = Array4::<f32>::ones((1, 1, 2, 2));

        // Create a simple 1x1x3x3 kernel with only top-left value set to 1.0
        let mut kernel = Array4::<f32>::zeros((1, 1, 3, 3));
        kernel[[0, 0, 0, 0]] = 1.0;
        // All other values are 0.0

        // Apply transposed convolution
        let output = conv_transpose2d(
            &input.view(),
            &kernel.view(),
            None,
            (2, 2), // stride
            (1, 1), // padding
            (0, 0), // output_padding
            (1, 1), // dilation
        )
        .unwrap();

        // Check dimensions
        // output_size = (input_size - 1) * stride - 2 * padding + kernel_size
        // = (2-1)*2 - 2*1 + 3 = 2 - 2 + 3 = 3
        assert_eq!(output.shape(), &[1, 1, 3, 3]);

        // Let's carefully trace through the algorithm for each input position:
        // The input is all ones at positions (0,0), (0,1), (1,0), and (1,1)
        // The kernel has a 1.0 at position (0,0,0,0) and zeros elsewhere
        // With stride=(2,2), padding=(1,1), the output coordinates are calculated as:
        //
        // For each input position (h,w) and kernel position (kh,kw):
        //   out_h = h*stride_h + kh*dilation_h - padding_h
        //   out_w = w*stride_w + kw*dilation_w - padding_w
        //
        // For input (0,0) and kernel (0,0):
        //   out_h = 0*2 + 0*1 - 1 = -1 (out of bounds)
        //   out_w = 0*2 + 0*1 - 1 = -1 (out of bounds)
        //
        // For input (0,1) and kernel (0,0):
        //   out_h = 0*2 + 0*1 - 1 = -1 (out of bounds)
        //   out_w = 1*2 + 0*1 - 1 = 1
        //
        // For input (1,0) and kernel (0,0):
        //   out_h = 1*2 + 0*1 - 1 = 1
        //   out_w = 0*2 + 0*1 - 1 = -1 (out of bounds)
        //
        // For input (1,1) and kernel (0,0):
        //   out_h = 1*2 + 0*1 - 1 = 1
        //   out_w = 1*2 + 0*1 - 1 = 1
        //
        // So only the input at (1,1) with kernel at (0,0) contributes to the output at (1,1)

        // Verify the output
        assert_eq!(output[[0, 0, 0, 0]], 0.0); // No contribution
        assert_eq!(output[[0, 0, 0, 1]], 0.0); // No contribution
        assert_eq!(output[[0, 0, 0, 2]], 0.0); // No contribution
        assert_eq!(output[[0, 0, 1, 0]], 0.0); // No contribution
        assert_eq!(output[[0, 0, 1, 1]], 1.0); // From input (1,1) with kernel (0,0)
        assert_eq!(output[[0, 0, 1, 2]], 0.0); // No contribution
        assert_eq!(output[[0, 0, 2, 0]], 0.0); // No contribution
        assert_eq!(output[[0, 0, 2, 1]], 0.0); // No contribution
        assert_eq!(output[[0, 0, 2, 2]], 0.0); // No contribution
    }
}
