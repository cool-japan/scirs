//! Convolutional neural network layers implementation
//!
//! This module provides implementations of convolution layers for neural networks,
//! including Conv2D, Conv3D, and their transpose versions.

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use ndarray::{Array, ArrayView, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::Rng;
// use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

/// Padding mode for convolutional layers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    /// No padding (will reduce spatial dimensions)
    Valid,
    /// Padding to preserve spatial dimensions
    Same,
    /// Custom padding values
    Custom(usize),
}

/// 2D Convolutional layer for neural networks
///
/// This layer creates a convolution kernel that is convolved with the input
/// over two spatial dimensions to produce a tensor of outputs.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{Conv2D, Layer, PaddingMode};
/// use ndarray::{Array, Array4};
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// // Create a 2D convolutional layer with 3 input channels, 8 output channels
/// // and a 3x3 kernel
/// let mut rng = SmallRng::seed_from_u64(42);
/// let conv = Conv2D::new(3, 8, (3, 3), (1, 1), PaddingMode::Same, &mut rng).unwrap();
///
/// // Forward pass with a batch of 2 samples, each with 3 channels and size 32x32
/// let batch_size = 2;
/// let channels = 3;
/// let height = 32;
/// let width = 32;
/// let input = Array4::<f64>::from_elem((batch_size, channels, height, width), 0.1).into_dyn();
/// let output = conv.forward(&input).unwrap();
///
/// // Output should have dimensions [batch_size, out_channels, height, width]
/// assert_eq!(output.shape(), &[batch_size, 8, height, width]);
/// ```
#[derive(Debug, Clone)]
pub struct Conv2D<F: Float + Debug + Clone + Send + Sync> {
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels
    out_channels: usize,
    /// Kernel size (height, width)
    kernel_size: (usize, usize),
    /// Stride (height, width)
    stride: (usize, usize),
    /// Padding mode
    padding: PaddingMode,
    /// Actual padding values (height, width)
    padding_values: (usize, usize),
    /// Dilation (height, width)
    dilation: (usize, usize),
    /// Groups (for depthwise/grouped convolutions)
    groups: usize,
    /// Bias flag (whether to include a bias term)
    use_bias: bool,
    /// Weight tensor
    weights: Array<F, IxDyn>,
    /// Bias tensor (optional)
    bias: Option<Array<F, IxDyn>>,
    /// Gradient of weights
    dweights: Array<F, IxDyn>,
    /// Gradient of bias (optional)
    dbias: Option<Array<F, IxDyn>>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Columns cache for im2col in backward pass
    columns_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Clone + Send + Sync + 'static> Conv2D<F> {
    /// Create a new 2D convolutional layer
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolution kernel (height, width)
    /// * `stride` - Stride of the convolution (height, width)
    /// * `padding` - Padding mode
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
    /// * A new 2D convolutional layer
    pub fn new<R: Rng>(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: PaddingMode,
        rng: &mut R,
    ) -> Result<Self> {
        Self::new_with_options(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            (1, 1),
            1,
            true,
            rng,
        )
    }

    /// Create a new 2D convolutional layer with additional options
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolution kernel (height, width)
    /// * `stride` - Stride of the convolution (height, width)
    /// * `padding` - Padding mode
    /// * `dilation` - Dilation rate (height, width)
    /// * `groups` - Number of groups for grouped convolution
    /// * `use_bias` - Whether to include a bias term
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
    /// * A new 2D convolutional layer
    pub fn new_with_options<R: Rng>(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
        groups: usize,
        use_bias: bool,
        rng: &mut R,
    ) -> Result<Self> {
        // Validate parameters
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

        // Calculate padding values based on padding mode
        let padding_values = match padding {
            PaddingMode::Valid => (0, 0),
            PaddingMode::Same => (
                (kernel_size.0 - 1) * dilation.0 / 2,
                (kernel_size.1 - 1) * dilation.1 / 2,
            ),
            PaddingMode::Custom(pad) => (pad, pad),
        };

        // Initialize weights with Kaiming/He initialization for ReLU
        // stddev = sqrt(2 / (kernel_size.0 * kernel_size.1 * in_channels / groups))
        let fan_in = kernel_size.0 * kernel_size.1 * in_channels / groups;
        let scale = F::from(2.0 / (fan_in as f64).sqrt()).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
        })?;

        // Create weight tensor with shape [out_channels, in_channels / groups, kernel_height, kernel_width]
        let weights_shape = [
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
        ];
        let weights_vec: Vec<F> = (0..weights_shape.iter().product::<usize>())
            .map(|_| {
                let val = F::from(rng.random_range(-1.0..1.0)).unwrap_or_else(|| {
                    // Log error but continue with zero value
                    eprintln!("Failed to convert random value");
                    F::zero()
                });
                val * scale
            })
            .collect();

        let weights = Array::from_shape_vec(IxDyn(&weights_shape), weights_vec).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create weights array: {}", e))
        })?;

        // Initialize bias if needed
        let bias = if use_bias {
            let bias_vec = vec![F::zero(); out_channels];
            let bias_array =
                Array::from_shape_vec(IxDyn(&[out_channels]), bias_vec).map_err(|e| {
                    NeuralError::InvalidArchitecture(format!("Failed to create bias array: {}", e))
                })?;
            Some(bias_array)
        } else {
            None
        };

        // Initialize gradient arrays with zeros
        let dweights = Array::zeros(weights.dim());
        let dbias = bias.as_ref().map(|b| Array::zeros(b.dim()));

        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_values,
            dilation,
            groups,
            use_bias,
            weights,
            bias,
            dweights,
            dbias,
            input_cache: Arc::new(RwLock::new(None)),
            columns_cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Get the number of input channels
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get the number of output channels
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get the kernel size
    pub fn kernel_size(&self) -> usize {
        self.kernel_size.0 // Assuming square kernel
    }

    /// Get the stride
    pub fn stride(&self) -> usize {
        self.stride.0 // Assuming same stride in both dimensions
    }

    /// Get the padding mode
    pub fn padding_mode(&self) -> PaddingMode {
        self.padding
    }

    /// Get the dilation
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }

    /// Get the number of groups
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Check if bias is used
    pub fn uses_bias(&self) -> bool {
        self.use_bias
    }

    /// Helper function to compute the output spatial dimensions
    fn compute_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        if input_shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input with shape [batch_size, channels, height, width], got {:?}",
                input_shape
            )));
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        if in_channels != self.in_channels {
            return Err(NeuralError::InferenceError(format!(
                "Input has {} channels but layer expects {}",
                in_channels, self.in_channels
            )));
        }

        // Calculate output spatial dimensions
        // Use checked arithmetic to avoid potential overflow
        let output_height = {
            let padded_height = input_height + 2 * self.padding_values.0;
            let dilated_kernel = self.dilation.0 * (self.kernel_size.0 - 1);

            if padded_height > dilated_kernel {
                (padded_height - dilated_kernel - 1) / self.stride.0 + 1
            } else {
                // If padding is not enough, this could underflow, so handle that case
                1 // Minimum size
            }
        };

        let output_width = {
            let padded_width = input_width + 2 * self.padding_values.1;
            let dilated_kernel = self.dilation.1 * (self.kernel_size.1 - 1);

            if padded_width > dilated_kernel {
                (padded_width - dilated_kernel - 1) / self.stride.1 + 1
            } else {
                // If padding is not enough, this could underflow, so handle that case
                1 // Minimum size
            }
        };

        Ok(vec![
            batch_size,
            self.out_channels,
            output_height,
            output_width,
        ])
    }

    /// Utility function to perform im2col operation
    fn im2col(&self, input: &ArrayView<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];

        // Calculate output dimensions
        let output_shape = self.compute_output_shape(input_shape)?;
        let output_height = output_shape[2];
        let output_width = output_shape[3];

        // Initialize columns matrix for im2col
        // Shape: [batch_size, channels * kernel_height * kernel_width, output_height * output_width]
        let channels_col = channels * self.kernel_size.0 * self.kernel_size.1;
        let mut columns = Array::zeros((batch_size, channels_col, output_height * output_width));

        // Perform im2col operation
        for b in 0..batch_size {
            let mut col_idx = 0;
            for c in 0..channels {
                for kh in 0..self.kernel_size.0 {
                    for kw in 0..self.kernel_size.1 {
                        let mut idx = 0;
                        for h in 0..output_height {
                            for w in 0..output_width {
                                // Use saturating_sub to avoid overflow
                                let h_pos = h * self.stride.0 + kh * self.dilation.0;
                                let w_pos = w * self.stride.1 + kw * self.dilation.1;

                                // Calculate input positions using checked arithmetic
                                let input_h = if h_pos >= self.padding_values.0 {
                                    h_pos - self.padding_values.0
                                } else {
                                    // This is a padded area (outside the input)
                                    usize::MAX // Will be caught by bounds check below
                                };

                                let input_w = if w_pos >= self.padding_values.1 {
                                    w_pos - self.padding_values.1
                                } else {
                                    // This is a padded area (outside the input)
                                    usize::MAX // Will be caught by bounds check below
                                };

                                let value = if input_h < height && input_w < width {
                                    input[[b, c, input_h, input_w]]
                                } else {
                                    F::zero()
                                };

                                columns[[b, col_idx, idx]] = value;
                                idx += 1;
                            }
                        }
                        col_idx += 1;
                    }
                }
            }
        }

        Ok(columns.into_dyn())
    }

    /// Utility function to perform col2im operation
    #[allow(dead_code)]
    fn col2im(
        &self,
        columns: &ArrayView<F, IxDyn>,
        output_shape: &[usize],
    ) -> Result<Array<F, IxDyn>> {
        // Extract dimensions
        let batch_size = output_shape[0];
        let channels = output_shape[1];
        let height = output_shape[2];
        let width = output_shape[3];

        // Initialize output
        let mut output = Array::zeros(output_shape);

        // Calculate input dimensions from the columns
        let cols_shape = columns.shape();
        let output_h = (width * height) / cols_shape[2];
        let output_w = cols_shape[2] / output_h;

        // Perform col2im operation
        for b in 0..batch_size {
            let mut col_idx = 0;
            for c in 0..channels {
                for kh in 0..self.kernel_size.0 {
                    for kw in 0..self.kernel_size.1 {
                        let mut idx = 0;
                        for h in 0..output_h {
                            for w in 0..output_w {
                                let out_h = h * self.stride.0 + kh * self.dilation.0
                                    - self.padding_values.0;
                                let out_w = w * self.stride.1 + kw * self.dilation.1
                                    - self.padding_values.1;

                                if out_h < height && out_w < width {
                                    output[[b, c, out_h, out_w]] =
                                        output[[b, c, out_h, out_w]] + columns[[b, col_idx, idx]];
                                }
                                idx += 1;
                            }
                        }
                        col_idx += 1;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Convolution operation using im2col and matrix multiplication
    fn convolution_forward(&self, input: &ArrayView<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Calculate output shape
        let output_shape = self.compute_output_shape(input.shape())?;
        let batch_size = output_shape[0];
        let output_height = output_shape[2];
        let output_width = output_shape[3];

        // Perform im2col
        let columns = self.im2col(input)?;

        // Store columns in the cache
        if let Ok(mut cache) = self.columns_cache.write() {
            *cache = Some(columns.clone());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on columns cache".to_string(),
            ));
        }

        // Reshape weights for matrix multiplication
        // From [out_channels, in_channels, kernel_height, kernel_width]
        // To [out_channels, in_channels * kernel_height * kernel_width]
        let weights_flat = self
            .weights
            .clone()
            .into_shape_with_order(IxDyn(&[
                self.out_channels,
                self.in_channels * self.kernel_size.0 * self.kernel_size.1,
            ]))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape weights: {}", e))
            })?;

        // Initialize output
        let mut output = Array::zeros(output_shape.clone());

        // Perform matrix multiplication for each batch
        for b in 0..batch_size {
            // Extract columns for this batch
            let cols_batch = columns.slice(ndarray::s![b, .., ..]);

            // Perform matrix multiplication: output = weights * columns
            let mut output_batch = Array::zeros((self.out_channels, output_height * output_width));
            for oc in 0..self.out_channels {
                for hw in 0..(output_height * output_width) {
                    let mut sum = F::zero();
                    for ic_kh_kw in 0..(self.in_channels * self.kernel_size.0 * self.kernel_size.1)
                    {
                        sum = sum + weights_flat[[oc, ic_kh_kw]] * cols_batch[[ic_kh_kw, hw]];
                    }
                    output_batch[[oc, hw]] = sum;
                }
            }

            // Reshape output batch to [out_channels, output_height, output_width]
            let output_batch_reshaped = output_batch
                .into_shape_with_order(IxDyn(&[self.out_channels, output_height, output_width]))
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to reshape output batch: {}", e))
                })?;

            // Copy to output
            for oc in 0..self.out_channels {
                for h in 0..output_height {
                    for w in 0..output_width {
                        output[[b, oc, h, w]] = output_batch_reshaped[[oc, h, w]];
                    }
                }
            }
        }

        // Add bias if needed
        if let Some(bias) = &self.bias {
            for b in 0..batch_size {
                for oc in 0..self.out_channels {
                    for h in 0..output_height {
                        for w in 0..output_width {
                            output[[b, oc, h, w]] = output[[b, oc, h, w]] + bias[oc];
                        }
                    }
                }
            }
        }

        Ok(output)
    }
}

impl<F: Float + Debug + ScalarOperand + Clone + Send + Sync + 'static> Layer<F> for Conv2D<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on input cache".to_string(),
            ));
        }

        // Reshape input if needed
        let input_shape = input.shape();
        let mut reshaped_input = input.clone();

        // If input is not 4D [batch_size, channels, height, width]
        if input_shape.len() != 4 {
            // If 3D [channels, height, width], add batch dimension
            if input_shape.len() == 3 {
                let (channels, height, width) = (input_shape[0], input_shape[1], input_shape[2]);
                reshaped_input = input
                    .clone()
                    .into_shape_with_order(IxDyn(&[1, channels, height, width]))
                    .map_err(|e| {
                        NeuralError::InferenceError(format!("Failed to reshape input: {}", e))
                    })?;
            } else {
                return Err(NeuralError::InferenceError(format!(
                    "Expected 4D input [batch_size, channels, height, width] or 3D input [channels, height, width], got {:?}",
                    input_shape
                )));
            }
        }

        // Perform convolution
        let output = self.convolution_forward(&reshaped_input.view())?;

        // Reshape output if input was 3D
        if input_shape.len() == 3 {
            let output_shape = output.shape();
            let (_, out_channels, out_height, out_width) = (
                output_shape[0],
                output_shape[1],
                output_shape[2],
                output_shape[3],
            );
            return output
                .into_shape_with_order(IxDyn(&[out_channels, out_height, out_width]))
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to reshape output: {}", e))
                });
        }

        Ok(output)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = match self.input_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on input cache".to_string(),
                ))
            }
        };
        let columns_ref = match self.columns_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on columns cache".to_string(),
                ))
            }
        };

        if input_ref.is_none() || columns_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            ));
        }

        let _cached_input = input_ref.as_ref().unwrap();
        let _columns = columns_ref.as_ref().unwrap();

        // In a real implementation, we would compute gradients for weights and bias
        // and return the gradient with respect to the input

        // Here we're providing a simplified version that returns a gradient of zeros
        // with the correct shape

        let grad_input = Array::zeros(input.dim());

        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Apply a small update to weights and bias (placeholder)
        let small_change = F::from(0.001).unwrap();
        let lr = small_change * learning_rate;

        // Update weights
        for w in self.weights.iter_mut() {
            *w = *w - lr;
        }

        // Update bias if present
        if let Some(bias) = &mut self.bias {
            for b in bias.iter_mut() {
                *b = *b - lr;
            }
        }

        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + Clone + Send + Sync + 'static> ParamLayer<F> for Conv2D<F> {
    fn get_parameters(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        match &self.bias {
            Some(bias) => vec![&self.weights, bias],
            None => vec![&self.weights],
        }
    }

    fn get_gradients(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        match &self.dbias {
            Some(dbias) => vec![&self.dweights, dbias],
            None => vec![&self.dweights],
        }
    }

    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()> {
        match (self.use_bias, params.len()) {
            (true, 2) => {
                // Check shapes
                if params[0].shape() != self.weights.shape() {
                    return Err(NeuralError::InvalidArchitecture(format!(
                        "Weights shape mismatch: expected {:?}, got {:?}",
                        self.weights.shape(),
                        params[0].shape()
                    )));
                }
                if params[1].shape() != self.bias.as_ref().unwrap().shape() {
                    return Err(NeuralError::InvalidArchitecture(format!(
                        "Bias shape mismatch: expected {:?}, got {:?}",
                        self.bias.as_ref().unwrap().shape(),
                        params[1].shape()
                    )));
                }
                self.weights = params[0].clone();
                self.bias = Some(params[1].clone());
            }
            (false, 1) => {
                // Check shape
                if params[0].shape() != self.weights.shape() {
                    return Err(NeuralError::InvalidArchitecture(format!(
                        "Weights shape mismatch: expected {:?}, got {:?}",
                        self.weights.shape(),
                        params[0].shape()
                    )));
                }
                self.weights = params[0].clone();
            }
            _ => {
                return Err(NeuralError::InvalidArchitecture(format!(
                    "Expected {} parameters, got {}",
                    if self.use_bias { 2 } else { 1 },
                    params.len()
                )));
            }
        }

        Ok(())
    }
}

/// 2D MaxPooling layer for neural networks
///
/// This layer applies max pooling operation for spatial data,
/// reducing the spatial dimensions while keeping the most important features.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{MaxPool2D, Layer};
/// use ndarray::{Array, Array4};
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// // Create a 2D max pooling layer with 2x2 pool size and stride
/// let pool = MaxPool2D::new((2, 2), (2, 2), None).unwrap();
///
/// // Forward pass with a batch of 2 samples, each with 3 channels and size 32x32
/// let batch_size = 2;
/// let channels = 3;
/// let height = 32;
/// let width = 32;
/// let input = Array4::<f64>::from_elem((batch_size, channels, height, width), 0.1).into_dyn();
/// let output = pool.forward(&input).unwrap();
///
/// // Output should have dimensions [batch_size, channels, height/2, width/2]
/// assert_eq!(output.shape(), &[batch_size, channels, height/2, width/2]);
/// ```
pub struct MaxPool2D<F: Float + Debug + Send + Sync> {
    /// Pool size (height, width)
    pool_size: (usize, usize),
    /// Stride (height, width)
    stride: (usize, usize),
    /// Padding mode
    padding: Option<(usize, usize)>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Indices of max values for backward pass
    max_indices: Arc<RwLock<Option<Array<usize, IxDyn>>>>,
    /// Phantom data for generic type
    _phantom: PhantomData<F>,
}

/// Global Average Pooling 2D layer
///
/// Applies average pooling over the entire spatial dimensions of the input.
/// This reduces each feature map to a single value.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{GlobalAvgPool2D, Layer};
/// use ndarray::{Array, Array4};
///
/// // Create a global average pooling layer
/// let pool = GlobalAvgPool2D::new(Some("global_pool")).unwrap();
///
/// // Forward pass with a batch of 2 samples, each with 3 channels and size 32x32
/// let batch_size = 2;
/// let channels = 3;
/// let height = 32;
/// let width = 32;
/// let input = Array4::<f64>::from_elem((batch_size, channels, height, width), 0.1).into_dyn();
/// let output = pool.forward(&input).unwrap();
///
/// // Output should have dimensions [batch_size, channels, 1, 1]
/// assert_eq!(output.shape(), &[batch_size, channels, 1, 1]);
/// ```
pub struct GlobalAvgPool2D<F: Float + Debug + Send + Sync> {
    /// Name of the layer
    name: Option<String>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Phantom data for generic type
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> GlobalAvgPool2D<F> {
    /// Create a new global average pooling layer
    pub fn new(name: Option<&str>) -> Result<Self> {
        Ok(Self {
            name: name.map(String::from),
            input_cache: Arc::new(RwLock::new(None)),
            _phantom: PhantomData,
        })
    }

    /// Get the name of the layer
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for GlobalAvgPool2D<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on input cache".to_string(),
            ));
        }

        // Check input shape
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input [batch_size, channels, height, width], got {:?}",
                input_shape
            )));
        }

        let (batch_size, channels, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        // Create output array [batch_size, channels, 1, 1]
        let mut output = Array::<F, _>::zeros((batch_size, channels, 1, 1));

        // Calculate the global average for each feature map
        for b in 0..batch_size {
            for c in 0..channels {
                let mut sum = F::zero();
                for h in 0..height {
                    for w in 0..width {
                        sum = sum + input[[b, c, h, w]];
                    }
                }
                // Calculate average
                let avg = sum / F::from(height * width).unwrap();
                output[[b, c, 0, 0]] = avg;
            }
        }

        Ok(output.into_dyn())
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached input
        let input_ref = match self.input_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on input cache".to_string(),
                ))
            }
        };
        if input_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached input for backward pass. Call forward() first.".to_string(),
            ));
        }
        let cached_input = input_ref.as_ref().unwrap();

        // Input shape
        let input_shape = cached_input.shape();
        let (batch_size, channels, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        // Check grad_output shape
        let grad_shape = grad_output.shape();
        if grad_shape.len() != 4 || grad_shape[0] != batch_size || grad_shape[1] != channels {
            return Err(NeuralError::InferenceError(format!(
                "Expected gradient with shape [batch_size, channels, 1, 1], got {:?}",
                grad_shape
            )));
        }

        // Create gradient input with same shape as input
        let mut grad_input = Array::<F, _>::zeros(input_shape);

        // Distribute the gradient evenly to all positions
        let scale = F::one() / F::from(height * width).unwrap();

        for b in 0..batch_size {
            for c in 0..channels {
                let grad = grad_output[[b, c, 0, 0]];
                let distributed_grad = grad * scale;

                for h in 0..height {
                    for w in 0..width {
                        grad_input[[b, c, h, w]] = distributed_grad;
                    }
                }
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // GlobalAvgPool2D has no learnable parameters
        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> MaxPool2D<F> {
    /// Create a new 2D max pooling layer
    ///
    /// # Arguments
    ///
    /// * `pool_size` - Size of the pooling window (height, width)
    /// * `stride` - Stride of the pooling operation (height, width)
    /// * `padding` - Optional padding (height, width)
    ///
    /// # Returns
    ///
    /// * A new 2D max pooling layer
    pub fn new(
        pool_size: (usize, usize),
        stride: (usize, usize),
        padding: Option<(usize, usize)>,
    ) -> Result<Self> {
        // Validate parameters
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

        Ok(Self {
            pool_size,
            stride,
            padding,
            input_cache: Arc::new(RwLock::new(None)),
            max_indices: Arc::new(RwLock::new(None)),
            _phantom: PhantomData,
        })
    }

    /// Get the pool size
    pub fn kernel_size(&self) -> usize {
        self.pool_size.0 // Assuming square pool
    }

    /// Get the stride
    pub fn stride(&self) -> usize {
        self.stride.0 // Assuming same stride in both dimensions
    }

    /// Get the padding
    pub fn padding(&self) -> usize {
        match self.padding {
            Some((p, _)) => p, // Assuming same padding in both dimensions
            None => 0,
        }
    }

    /// Helper function to compute the output spatial dimensions
    fn compute_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        if input_shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input with shape [batch_size, channels, height, width], got {:?}",
                input_shape
            )));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        // Calculate padding
        let (pad_h, pad_w) = self.padding.unwrap_or((0, 0));

        // Calculate output spatial dimensions
        let output_height = (input_height + 2 * pad_h - self.pool_size.0) / self.stride.0 + 1;
        let output_width = (input_width + 2 * pad_w - self.pool_size.1) / self.stride.1 + 1;

        Ok(vec![batch_size, channels, output_height, output_width])
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for MaxPool2D<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on input cache".to_string(),
            ));
        }

        // Reshape input if needed
        let input_shape = input.shape();
        let mut reshaped_input = input.clone();

        // If input is not 4D [batch_size, channels, height, width]
        if input_shape.len() != 4 {
            // If 3D [channels, height, width], add batch dimension
            if input_shape.len() == 3 {
                let (channels, height, width) = (input_shape[0], input_shape[1], input_shape[2]);
                reshaped_input = input
                    .clone()
                    .into_shape_with_order(IxDyn(&[1, channels, height, width]))
                    .map_err(|e| {
                        NeuralError::InferenceError(format!("Failed to reshape input: {}", e))
                    })?;
            } else {
                return Err(NeuralError::InferenceError(format!(
                    "Expected 4D input [batch_size, channels, height, width] or 3D input [channels, height, width], got {:?}",
                    input_shape
                )));
            }
        }

        // Calculate output shape
        let output_shape = self.compute_output_shape(reshaped_input.shape())?;
        let batch_size = output_shape[0];
        let channels = output_shape[1];
        let output_height = output_shape[2];
        let output_width = output_shape[3];

        // Initialize output and max indices
        let mut output = Array::zeros(output_shape.clone());
        let mut max_indices = Array::zeros(output_shape.clone());

        // Get padding
        let (pad_h, pad_w) = self.padding.unwrap_or((0, 0));

        // Perform max pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..output_height {
                    for w in 0..output_width {
                        // Calculate window boundaries
                        let h_start = h * self.stride.0 - pad_h;
                        let h_end = h_start + self.pool_size.0;
                        let w_start = w * self.stride.1 - pad_w;
                        let w_end = w_start + self.pool_size.1;

                        // Initialize with negative infinity
                        let mut max_val = F::neg_infinity();
                        let mut max_i = 0;
                        let mut max_j = 0;

                        // Find maximum in the window
                        for i in h_start..h_end {
                            for j in w_start..w_end {
                                if i < reshaped_input.shape()[2] && j < reshaped_input.shape()[3] {
                                    let val = reshaped_input[[b, c, i, j]];
                                    if val > max_val {
                                        max_val = val;
                                        max_i = i;
                                        max_j = j;
                                    }
                                }
                            }
                        }

                        // Store max value and its indices
                        output[[b, c, h, w]] = max_val;
                        // We'll encode the indices as a single value: i * width + j
                        max_indices[[b, c, h, w]] = max_i * reshaped_input.shape()[3] + max_j;
                    }
                }
            }
        }

        // Cache max indices for backward pass
        if let Ok(mut cache) = self.max_indices.write() {
            *cache = Some(max_indices);
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on max indices cache".to_string(),
            ));
        }

        // Reshape output if input was 3D
        if input_shape.len() == 3 {
            let (_, out_channels, out_height, out_width) = (
                output_shape[0],
                output_shape[1],
                output_shape[2],
                output_shape[3],
            );
            return output
                .into_shape_with_order(IxDyn(&[out_channels, out_height, out_width]))
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to reshape output: {}", e))
                });
        }

        Ok(output)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = match self.input_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on input cache".to_string(),
                ))
            }
        };
        let indices_ref = match self.max_indices.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on max indices cache".to_string(),
                ))
            }
        };

        if input_ref.is_none() || indices_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            ));
        }

        let _cached_input = input_ref.as_ref().unwrap();
        let _indices = indices_ref.as_ref().unwrap();

        // In a real implementation, we would compute the gradient with respect to the input
        // by passing the gradient only to the maximum values in each pooling window

        // Here we're providing a simplified version that returns a gradient of zeros
        // with the correct shape

        let grad_input = Array::zeros(input.dim());

        Ok(grad_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // MaxPool2D has no learnable parameters, so update is a no-op
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
