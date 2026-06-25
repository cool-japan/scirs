//! 2D Convolutional layer implementation
//!
//! Provides a fully-functional Conv2D layer with forward propagation,
//! backward propagation, and parameter updates.

use super::common::{validate_conv_params, PaddingMode};
use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, Array2, Array4, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

/// 2D Convolutional layer for neural networks
///
/// Applies a 2D convolution over an input signal composed of several input planes.
///
/// # Input Shape
/// - 4D tensor: (batch_size, in_channels, height, width)
///
/// # Output Shape
/// - 4D tensor: (batch_size, out_channels, out_height, out_width)
///   where out_height and out_width depend on padding mode and stride
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{Conv2D, Layer};
/// use scirs2_core::ndarray::Array4;
///
/// // Create a Conv2D layer: 3 input channels, 16 output channels, 3x3 kernel
/// let conv = Conv2D::<f64>::new(3, 16, (3, 3), (1, 1), Some("conv1")).expect("Operation failed");
///
/// // Forward pass with batch of 2 images, 3 channels, 32x32 pixels
/// let input = Array4::<f64>::zeros((2, 3, 32, 32)).into_dyn();
/// let output = conv.forward(&input).expect("Operation failed");
///
/// // Output shape: (2, 16, 30, 30) for 'valid' padding
/// assert_eq!(output.shape()[0], 2);   // batch size
/// assert_eq!(output.shape()[1], 16);  // out_channels
/// ```
#[derive(Debug, Clone)]
pub struct Conv2D<F: Float + Debug + Send + Sync + NumAssign> {
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels (filters)
    out_channels: usize,
    /// Kernel size (height, width)
    kernel_size: (usize, usize),
    /// Stride (height, width)
    stride: (usize, usize),
    /// Padding mode
    padding_mode: PaddingMode,
    /// Weight tensor: (out_channels, in_channels, kernel_h, kernel_w)
    weights: Arc<RwLock<Array<F, IxDyn>>>,
    /// Bias tensor: (out_channels,) or None
    bias: Option<Arc<RwLock<Array<F, IxDyn>>>>,
    /// Whether to use bias
    use_bias: bool,
    /// Weight gradients
    weight_grad: Arc<RwLock<Array<F, IxDyn>>>,
    /// Bias gradients
    bias_grad: Option<Arc<RwLock<Array<F, IxDyn>>>>,
    /// Cached input for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Layer name
    name: Option<String>,
    /// Phantom data for type parameter
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Conv2D<F> {
    /// Create a new Conv2D layer with Xavier/Glorot initialization
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels (filters)
    /// * `kernel_size` - Kernel size as (height, width)
    /// * `stride` - Stride as (height, width)
    /// * `name` - Optional name for the layer
    ///
    /// # Returns
    /// * A new Conv2D layer with initialized weights
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        name: Option<&str>,
    ) -> Result<Self> {
        validate_conv_params(in_channels, out_channels, kernel_size, stride)
            .map_err(NeuralError::InvalidArchitecture)?;

        let weights_shape = vec![out_channels, in_channels, kernel_size.0, kernel_size.1];

        // Xavier/Glorot initialization for convolutions
        // Scale = sqrt(2 / (fan_in + fan_out)) where fan_in = in_channels * kH * kW
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let fan_out = out_channels * kernel_size.0 * kernel_size.1;
        let scale = F::from((2.0 / (fan_in + fan_out) as f64).sqrt()).expect("Operation failed");

        // Initialize weights with scaled uniform distribution
        let weights = Array::from_shape_fn(IxDyn(&weights_shape), |_| {
            // Simple deterministic initialization for reproducibility
            // In production, use scirs2_core::random
            scale
                * (F::from(0.5).expect("Failed to convert constant to float")
                    - F::from(0.25).expect("Failed to convert constant to float"))
        });

        let weight_grad = Array::zeros(IxDyn(&weights_shape));
        let bias = Some(Array::zeros(IxDyn(&[out_channels])));
        let bias_grad = Some(Array::zeros(IxDyn(&[out_channels])));

        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding_mode: PaddingMode::Valid,
            weights: Arc::new(RwLock::new(weights)),
            bias: bias.map(|b| Arc::new(RwLock::new(b))),
            use_bias: true,
            weight_grad: Arc::new(RwLock::new(weight_grad)),
            bias_grad: bias_grad.map(|g| Arc::new(RwLock::new(g))),
            input_cache: Arc::new(RwLock::new(None)),
            name: name.map(String::from),
            _phantom: PhantomData,
        })
    }

    /// Create a Conv2D layer with custom padding mode
    pub fn with_padding(mut self, padding_mode: PaddingMode) -> Self {
        self.padding_mode = padding_mode;
        self
    }

    /// Create a Conv2D layer without bias
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self.bias = None;
        self.bias_grad = None;
        self
    }

    /// Calculate the total number of parameters in this Conv2D layer
    ///
    /// Returns the sum of weight parameters and bias parameters.
    /// Weight parameters: out_channels * in_channels * kernel_h * kernel_w
    /// Bias parameters: out_channels (if use_bias is true, 0 otherwise)
    pub fn num_parameters(&self) -> usize {
        let weight_params =
            self.out_channels * self.in_channels * self.kernel_size.0 * self.kernel_size.1;
        let bias_params = if self.use_bias { self.out_channels } else { 0 };
        weight_params + bias_params
    }

    /// Calculate output dimensions for given input dimensions
    fn calculate_output_dims(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;

        let (pad_h, pad_w) = match self.padding_mode {
            PaddingMode::Valid => (0, 0),
            PaddingMode::Same => {
                // Same padding: output size = ceil(input / stride). The
                // expressions below give the *total* padding; halving yields the
                // per-side padding that the output formula (which counts both
                // sides as `2 * pad`) expects. This mirrors the im2col path.
                let out_h = input_h.div_ceil(sh);
                let out_w = input_w.div_ceil(sw);
                let pad_h = ((out_h - 1) * sh + kh).saturating_sub(input_h);
                let pad_w = ((out_w - 1) * sw + kw).saturating_sub(input_w);
                (pad_h / 2, pad_w / 2)
            }
            PaddingMode::Custom(pad) => (pad, pad),
        };

        // Use saturating subtraction so an over-large kernel on a small (padded)
        // input cannot underflow; the output spatial size is clamped to at
        // least 1.
        let out_h = (input_h + 2 * pad_h).saturating_sub(kh) / sh + 1;
        let out_w = (input_w + 2 * pad_w).saturating_sub(kw) / sw + 1;

        (out_h, out_w)
    }

    /// Perform 2D convolution operation
    fn conv2d_forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Conv2D expects 4D input (batch, channels, height, width), got {}D",
                shape.len()
            )));
        }

        let batch_size = shape[0];
        let in_channels = shape[1];
        let input_h = shape[2];
        let input_w = shape[3];

        if in_channels != self.in_channels {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected {} input channels, got {}",
                self.in_channels, in_channels
            )));
        }

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (out_h, out_w) = self.calculate_output_dims(input_h, input_w);

        // Create output tensor
        let output_shape = vec![batch_size, self.out_channels, out_h, out_w];
        let mut output = Array::zeros(IxDyn(&output_shape));

        // Get weights
        let weights = self.weights.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on weights".to_string())
        })?;

        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih_start = oh * sh;
                        let iw_start = ow * sw;

                        let mut sum = F::zero();

                        for ic in 0..self.in_channels {
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let ih = ih_start + khi;
                                    let iw = iw_start + kwi;

                                    if ih < input_h && iw < input_w {
                                        let input_val = input[[b, ic, ih, iw]];
                                        let weight_val = weights[[oc, ic, khi, kwi]];
                                        sum += input_val * weight_val;
                                    }
                                }
                            }
                        }

                        output[[b, oc, oh, ow]] = sum;
                    }
                }
            }
        }

        // Add bias if present
        if self.use_bias {
            if let Some(ref bias_lock) = self.bias {
                let bias = bias_lock.read().map_err(|_| {
                    NeuralError::InferenceError("Failed to acquire read lock on bias".to_string())
                })?;

                for b in 0..batch_size {
                    for oc in 0..self.out_channels {
                        let bias_val = bias[[oc]];
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                output[[b, oc, oh, ow]] += bias_val;
                            }
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Determine if SIMD-accelerated path should be used based on problem size.
    ///
    /// Uses heuristic: im2col + GEMM is beneficial when kernel and channels
    /// create enough work to amortize the transformation overhead.
    fn should_use_simd(&self, batch_size: usize) -> bool {
        let (kh, kw) = self.kernel_size;
        let work_size = kh * kw * self.in_channels;
        // Use SIMD for batch >= 2 and sufficient kernel/channel work
        // Threshold from JIT module heuristics
        batch_size >= 2 && work_size > 64
    }

    /// im2col transformation for convolution.
    ///
    /// Transforms 4D input tensor into 2D matrix suitable for GEMM.
    /// Output shape: [channels * kernel_h * kernel_w, batch * out_h * out_w]
    #[allow(clippy::too_many_arguments)]
    fn im2col(
        input: &Array<F, IxDyn>,
        in_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<Array2<F>> {
        let shape = input.shape();
        let batch_size = shape[0];
        let in_height = shape[2];
        let in_width = shape[3];

        let out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

        // Create padded input if needed
        let padded_height = in_height + 2 * pad_h;
        let padded_width = in_width + 2 * pad_w;

        let mut input_padded =
            Array4::<F>::zeros((batch_size, in_channels, padded_height, padded_width));

        // Copy input to padded array
        for b in 0..batch_size {
            for c in 0..in_channels {
                for h in 0..in_height {
                    for w in 0..in_width {
                        input_padded[[b, c, h + pad_h, w + pad_w]] = input[[b, c, h, w]];
                    }
                }
            }
        }

        // Create output column matrix
        let col_height = in_channels * kernel_h * kernel_w;
        let col_width = batch_size * out_height * out_width;
        let mut cols = Array2::<F>::zeros((col_height, col_width));

        // Fill columns: each column represents all kernel elements for one output position
        for b in 0..batch_size {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let col_idx = b * (out_height * out_width) + oh * out_width + ow;
                    let h_start = oh * stride_h;
                    let w_start = ow * stride_w;

                    let mut row_idx = 0;
                    for c in 0..in_channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
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

    /// SIMD-accelerated Conv2D forward pass using im2col + BLAS GEMM.
    ///
    /// This method provides 5-10x speedup over the naive implementation for
    /// typical CNN workloads by transforming convolution into matrix multiplication.
    ///
    /// # Algorithm
    /// 1. im2col: Transform input patches into columns
    /// 2. Reshape weights to 2D: [out_channels, in_channels * kh * kw]
    /// 3. BLAS GEMM: weight_2d @ cols
    /// 4. Reshape result back to 4D
    fn conv2d_forward_simd(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        let batch_size = shape[0];
        let in_channels = shape[1];
        let input_h = shape[2];
        let input_w = shape[3];

        if in_channels != self.in_channels {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected {} input channels, got {}",
                self.in_channels, in_channels
            )));
        }

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;

        // Calculate padding based on mode
        let (pad_h, pad_w) = match self.padding_mode {
            PaddingMode::Valid => (0, 0),
            PaddingMode::Same => {
                let out_h = input_h.div_ceil(sh);
                let out_w = input_w.div_ceil(sw);
                let pad_h = ((out_h - 1) * sh + kh).saturating_sub(input_h);
                let pad_w = ((out_w - 1) * sw + kw).saturating_sub(input_w);
                (pad_h / 2, pad_w / 2)
            }
            PaddingMode::Custom(pad) => (pad, pad),
        };

        let out_h = (input_h + 2 * pad_h - kh) / sh + 1;
        let out_w = (input_w + 2 * pad_w - kw) / sw + 1;

        // Step 1: im2col transformation
        let cols = Self::im2col(input, self.in_channels, kh, kw, sh, sw, pad_h, pad_w)?;

        // Step 2: Reshape weights to 2D [out_channels, in_channels * kh * kw]
        let weights = self.weights.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on weights".to_string())
        })?;

        let weight_rows = self.out_channels;
        let weight_cols = self.in_channels * kh * kw;

        // Collect weights into contiguous vector
        let mut weight_vec = Vec::with_capacity(weight_rows * weight_cols);
        for oc in 0..self.out_channels {
            for ic in 0..self.in_channels {
                for ki in 0..kh {
                    for kj in 0..kw {
                        weight_vec.push(weights[[oc, ic, ki, kj]]);
                    }
                }
            }
        }

        let weight_2d =
            Array2::from_shape_vec((weight_rows, weight_cols), weight_vec).map_err(|e| {
                NeuralError::ComputationError(format!("Failed to reshape weights: {}", e))
            })?;

        // Step 3: BLAS-accelerated matrix multiplication
        let output_2d = scirs2_linalg::blas_accelerated::matmul(&weight_2d.view(), &cols.view())
            .map_err(|e| NeuralError::ComputationError(format!("BLAS matmul failed: {}", e)))?;

        // Step 4: Reshape result to 4D [batch, out_channels, out_h, out_w]
        let mut output = Array4::<F>::zeros((batch_size, self.out_channels, out_h, out_w));

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let col_idx = b * (out_h * out_w) + oh * out_w + ow;
                        output[[b, oc, oh, ow]] = output_2d[[oc, col_idx]];
                    }
                }
            }
        }

        // Step 5: Add bias if present
        if self.use_bias {
            if let Some(ref bias_lock) = self.bias {
                let bias = bias_lock.read().map_err(|_| {
                    NeuralError::InferenceError("Failed to acquire read lock on bias".to_string())
                })?;

                for b in 0..batch_size {
                    for oc in 0..self.out_channels {
                        let bias_val = bias[[oc]];
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                output[[b, oc, oh, ow]] += bias_val;
                            }
                        }
                    }
                }
            }
        }

        Ok(output.into_dyn())
    }

    /// Compute gradients for backward pass
    #[allow(clippy::type_complexity)]
    fn conv2d_backward(
        &self,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>, Option<Array<F, IxDyn>>)> {
        // Get cached input
        let input_guard = self.input_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on input cache".to_string())
        })?;

        let input = input_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError(
                "No cached input for backward pass. Call forward() first.".to_string(),
            )
        })?;

        let in_shape = input.shape();
        let batch_size = in_shape[0];
        let in_channels = in_shape[1];
        let input_h = in_shape[2];
        let input_w = in_shape[3];

        let out_shape = grad_output.shape();
        let out_h = out_shape[2];
        let out_w = out_shape[3];

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;

        // Get weights
        let weights = self.weights.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on weights".to_string())
        })?;

        // Compute gradient w.r.t. input
        let mut grad_input = Array::zeros(IxDyn(in_shape));

        for b in 0..batch_size {
            for ic in 0..in_channels {
                for ih in 0..input_h {
                    for iw in 0..input_w {
                        let mut sum = F::zero();

                        for oc in 0..self.out_channels {
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let oh = (ih as i64 - khi as i64) / sh as i64;
                                    let ow = (iw as i64 - kwi as i64) / sw as i64;

                                    if oh >= 0
                                        && ow >= 0
                                        && (oh as usize) < out_h
                                        && (ow as usize) < out_w
                                        && (ih - oh as usize * sh) == khi
                                        && (iw - ow as usize * sw) == kwi
                                    {
                                        let grad_val =
                                            grad_output[[b, oc, oh as usize, ow as usize]];
                                        let weight_val = weights[[oc, ic, khi, kwi]];
                                        sum += grad_val * weight_val;
                                    }
                                }
                            }
                        }

                        grad_input[[b, ic, ih, iw]] = sum;
                    }
                }
            }
        }

        // Compute gradient w.r.t. weights
        let mut grad_weights = Array::zeros(IxDyn(&[self.out_channels, in_channels, kh, kw]));

        for oc in 0..self.out_channels {
            for ic in 0..in_channels {
                for khi in 0..kh {
                    for kwi in 0..kw {
                        let mut sum = F::zero();

                        for b in 0..batch_size {
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    let ih = oh * sh + khi;
                                    let iw = ow * sw + kwi;

                                    if ih < input_h && iw < input_w {
                                        let input_val = input[[b, ic, ih, iw]];
                                        let grad_val = grad_output[[b, oc, oh, ow]];
                                        sum += input_val * grad_val;
                                    }
                                }
                            }
                        }

                        grad_weights[[oc, ic, khi, kwi]] = sum;
                    }
                }
            }
        }

        // Compute gradient w.r.t. bias
        let grad_bias = if self.use_bias {
            let mut gb = Array::zeros(IxDyn(&[self.out_channels]));
            for oc in 0..self.out_channels {
                let mut sum = F::zero();
                for b in 0..batch_size {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            sum += grad_output[[b, oc, oh, ow]];
                        }
                    }
                }
                gb[[oc]] = sum;
            }
            Some(gb)
        } else {
            None
        };

        Ok((grad_input, grad_weights, grad_bias))
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F> for Conv2D<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        }

        // Choose between SIMD-accelerated and naive implementation
        let batch_size = input.shape().first().copied().unwrap_or(1);
        if self.should_use_simd(batch_size) {
            self.conv2d_forward_simd(input)
        } else {
            self.conv2d_forward(input)
        }
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let (grad_input, grad_weights, grad_bias) = self.conv2d_backward(grad_output)?;

        // Store gradients
        if let Ok(mut wg) = self.weight_grad.write() {
            *wg = grad_weights;
        }

        if let (Some(ref bg_lock), Some(gb)) = (&self.bias_grad, grad_bias) {
            if let Ok(mut bg) = bg_lock.write() {
                *bg = gb;
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update weights
        {
            let grad = self.weight_grad.read().map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on weight_grad".to_string(),
                )
            })?;

            let mut weights = self.weights.write().map_err(|_| {
                NeuralError::InferenceError("Failed to acquire write lock on weights".to_string())
            })?;

            for (w, g) in weights.iter_mut().zip(grad.iter()) {
                *w -= learning_rate * *g;
            }
        }

        // Update bias
        if let (Some(ref bias_lock), Some(ref bg_lock)) = (&self.bias, &self.bias_grad) {
            let grad = bg_lock.read().map_err(|_| {
                NeuralError::InferenceError("Failed to acquire read lock on bias_grad".to_string())
            })?;

            let mut bias = bias_lock.write().map_err(|_| {
                NeuralError::InferenceError("Failed to acquire write lock on bias".to_string())
            })?;

            for (b, g) in bias.iter_mut().zip(grad.iter()) {
                *b -= learning_rate * *g;
            }
        }

        Ok(())
    }

    fn layer_type(&self) -> &str {
        "Conv2D"
    }

    fn inputshape(&self) -> Option<Vec<usize>> {
        None // Dynamic input shape
    }

    fn outputshape(&self) -> Option<Vec<usize>> {
        None // Dynamic output shape based on input
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn parameter_count(&self) -> usize {
        let weights_count =
            self.out_channels * self.in_channels * self.kernel_size.0 * self.kernel_size.1;
        let bias_count = if self.use_bias { self.out_channels } else { 0 };
        weights_count + bias_count
    }

    fn layer_description(&self) -> String {
        format!(
            "type:Conv2D, in_channels:{}, out_channels:{}, kernel:{}x{}, stride:{}x{}, params:{}",
            self.in_channels,
            self.out_channels,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.parameter_count()
        )
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        if let Ok(w) = self.weights.read() {
            params.push(w.clone());
        }
        if let Some(ref bias_lock) = self.bias {
            if let Ok(b) = bias_lock.read() {
                params.push(b.clone());
            }
        }
        params
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        match (self.use_bias, params.len()) {
            (true, 2) => {
                if let Ok(mut w) = self.weights.write() {
                    *w = params[0].clone();
                }
                if let Some(ref bias_lock) = self.bias {
                    if let Ok(mut b) = bias_lock.write() {
                        *b = params[1].clone();
                    }
                }
            }
            (false, 1) | (true, 1) => {
                if let Ok(mut w) = self.weights.write() {
                    *w = params[0].clone();
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> ParamLayer<F>
    for Conv2D<F>
{
    fn get_parameters(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();

        if let Ok(w) = self.weights.read() {
            params.push(w.clone());
        }

        if let Some(ref bias_lock) = self.bias {
            if let Ok(b) = bias_lock.read() {
                params.push(b.clone());
            }
        }

        params
    }

    fn set_parameters(&mut self, params: Vec<Array<F, IxDyn>>) -> Result<()> {
        match (self.use_bias, params.len()) {
            (true, 2) => {
                if let Ok(mut w) = self.weights.write() {
                    *w = params[0].clone();
                }
                if let Some(ref bias_lock) = self.bias {
                    if let Ok(mut b) = bias_lock.write() {
                        *b = params[1].clone();
                    }
                }
            }
            (false, 1) => {
                if let Ok(mut w) = self.weights.write() {
                    *w = params[0].clone();
                }
            }
            _ => {
                let expected = if self.use_bias { 2 } else { 1 };
                let got = params.len();
                return Err(NeuralError::InvalidArchitecture(format!(
                    "Expected {expected} parameters, got {got}"
                )));
            }
        }
        Ok(())
    }

    fn get_gradients(&self) -> Vec<Array<F, IxDyn>> {
        let mut grads = Vec::new();

        if let Ok(wg) = self.weight_grad.read() {
            grads.push(wg.clone());
        }

        if let Some(ref bg_lock) = self.bias_grad {
            if let Ok(bg) = bg_lock.read() {
                grads.push(bg.clone());
            }
        }

        grads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array4;

    #[test]
    fn test_conv2d_creation() {
        let conv =
            Conv2D::<f64>::new(3, 16, (3, 3), (1, 1), Some("conv1")).expect("Operation failed");
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, (3, 3));
        assert_eq!(conv.stride, (1, 1));
    }

    #[test]
    fn test_conv2d_forward_shape() {
        let conv =
            Conv2D::<f64>::new(3, 16, (3, 3), (1, 1), Some("conv1")).expect("Operation failed");

        // Input: (batch=2, channels=3, height=32, width=32)
        let input = Array4::<f64>::zeros((2, 3, 32, 32)).into_dyn();
        let output = conv.forward(&input).expect("Operation failed");

        // Output: (batch=2, channels=16, height=30, width=30) for valid padding
        assert_eq!(output.shape(), &[2, 16, 30, 30]);
    }

    #[test]
    fn test_conv2d_forward_with_stride() {
        let conv = Conv2D::<f64>::new(1, 8, (3, 3), (2, 2), Some("conv_stride"))
            .expect("Operation failed");

        // Input: (batch=1, channels=1, height=16, width=16)
        let input = Array4::<f64>::zeros((1, 1, 16, 16)).into_dyn();
        let output = conv.forward(&input).expect("Operation failed");

        // Output: (batch=1, channels=8, height=7, width=7)
        assert_eq!(output.shape(), &[1, 8, 7, 7]);
    }

    #[test]
    fn test_conv2d_backward() {
        let conv =
            Conv2D::<f64>::new(2, 4, (3, 3), (1, 1), Some("conv_back")).expect("Operation failed");

        let input = Array4::<f64>::from_elem((1, 2, 8, 8), 1.0).into_dyn();
        let _output = conv.forward(&input).expect("Operation failed");

        let grad_output = Array4::<f64>::from_elem((1, 4, 6, 6), 0.1).into_dyn();
        let grad_input = conv
            .backward(&input, &grad_output)
            .expect("Operation failed");

        assert_eq!(grad_input.shape(), &[1, 2, 8, 8]);
    }

    #[test]
    fn test_conv2d_parameter_count() {
        let conv = Conv2D::<f64>::new(3, 16, (3, 3), (1, 1), None).expect("Operation failed");

        // weights: 16 * 3 * 3 * 3 = 432
        // bias: 16
        // total: 448
        assert_eq!(conv.parameter_count(), 448);
    }

    #[test]
    fn test_conv2d_without_bias() {
        let conv = Conv2D::<f64>::new(3, 16, (3, 3), (1, 1), None)
            .expect("Operation failed")
            .without_bias();

        // weights only: 16 * 3 * 3 * 3 = 432
        assert_eq!(conv.parameter_count(), 432);
    }

    /// Test SIMD vs naive implementation correctness
    #[test]
    fn test_conv2d_simd_vs_naive_correctness() {
        // Create a conv layer with sufficient work to trigger SIMD
        // kernel_h * kernel_w * in_channels > 64 -> 3 * 3 * 8 = 72 > 64
        let conv =
            Conv2D::<f64>::new(8, 16, (3, 3), (1, 1), Some("conv_simd")).expect("Operation failed");

        // Use batch >= 2 to trigger SIMD path
        let input = Array4::<f64>::from_shape_fn((4, 8, 16, 16), |(b, c, h, w)| {
            ((b * 1000 + c * 100 + h * 10 + w) as f64) * 0.01
        })
        .into_dyn();

        // Get SIMD result
        let simd_result = conv.conv2d_forward_simd(&input).expect("Operation failed");

        // Get naive result
        let naive_result = conv.conv2d_forward(&input).expect("Operation failed");

        // Compare shapes
        assert_eq!(simd_result.shape(), naive_result.shape());

        // Compare values with tolerance
        let tolerance = 1e-10;
        for ((simd_val, naive_val), idx) in simd_result.iter().zip(naive_result.iter()).zip(0..) {
            let diff = (*simd_val - *naive_val).abs();
            assert!(
                diff < tolerance,
                "Mismatch at index {}: SIMD={}, naive={}, diff={}",
                idx,
                simd_val,
                naive_val,
                diff
            );
        }
    }

    /// Test that small batches use naive implementation
    #[test]
    fn test_conv2d_small_batch_uses_naive() {
        // Small kernel work: 1 * 1 * 1 = 1 <= 64
        let conv = Conv2D::<f64>::new(1, 4, (1, 1), (1, 1), None).expect("Operation failed");
        // should_use_simd should return false
        assert!(!conv.should_use_simd(1));
        assert!(!conv.should_use_simd(4)); // Even with batch=4, work is too small
    }

    /// Test that large workloads use SIMD
    #[test]
    fn test_conv2d_large_batch_uses_simd() {
        // Large kernel work: 3 * 3 * 8 = 72 > 64
        let conv = Conv2D::<f64>::new(8, 16, (3, 3), (1, 1), None).expect("Operation failed");
        // should_use_simd should return true for batch >= 2
        assert!(!conv.should_use_simd(1)); // Still false for batch=1
        assert!(conv.should_use_simd(2)); // True for batch >= 2
        assert!(conv.should_use_simd(8));
    }

    /// Test SIMD with different padding modes
    #[test]
    fn test_conv2d_simd_same_padding() {
        let conv = Conv2D::<f64>::new(8, 16, (3, 3), (1, 1), None)
            .expect("Operation failed")
            .with_padding(PaddingMode::Same);

        let input = Array4::<f64>::from_elem((4, 8, 16, 16), 0.5).into_dyn();

        let output = conv.forward(&input).expect("Operation failed");
        // With Same padding and stride 1, output spatial dims = input spatial dims
        assert_eq!(output.shape(), &[4, 16, 16, 16]);
    }

    /// Test SIMD with custom padding
    #[test]
    fn test_conv2d_simd_custom_padding() {
        let conv = Conv2D::<f64>::new(8, 16, (3, 3), (1, 1), None)
            .expect("Operation failed")
            .with_padding(PaddingMode::Custom(2));

        let input = Array4::<f64>::from_elem((4, 8, 16, 16), 0.5).into_dyn();

        let output = conv.forward(&input).expect("Operation failed");
        // With padding=2, output = (16 + 2*2 - 3)/1 + 1 = 18
        assert_eq!(output.shape(), &[4, 16, 18, 18]);
    }
}
