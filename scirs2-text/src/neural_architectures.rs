//! Advanced neural architectures for text processing
//!
//! This module provides various neural network architectures optimized for
//! text processing tasks, including RNNs, CNNs, attention mechanisms, and hybrid models.

use crate::error::{Result, TextError};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use rand::{self, Rng};
use statrs::statistics::Statistics;

/// Activation functions for neural networks
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    /// Hyperbolic tangent
    Tanh,
    /// Sigmoid function
    Sigmoid,
    /// Rectified Linear Unit
    ReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Swish activation
    Swish,
    /// Linear activation (identity)
    Linear,
}

impl ActivationFunction {
    /// Apply activation function to a value
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::GELU => {
                0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
            }
            ActivationFunction::Swish => x / (1.0 + (-x).exp()),
            ActivationFunction::Linear => x,
        }
    }

    /// Apply activation function to an array
    pub fn apply_array(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|val| self.apply(val))
    }

    /// Compute derivative of activation function
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Tanh => {
                let tanh_x = x.tanh();
                1.0 - tanh_x * tanh_x
            }
            ActivationFunction::Sigmoid => {
                let sig_x = self.apply(x);
                sig_x * (1.0 - sig_x)
            }
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::GELU => {
                // Approximate derivative of GELU
                let cdf = 0.5 * (1.0 + (x * 0.7978845608).tanh());
                let pdf = 0.7978845608 * (-0.5 * x * x).exp();
                cdf + x * pdf
            }
            ActivationFunction::Swish => {
                let sig_x = 1.0 / (1.0 + (-x).exp());
                sig_x + x * sig_x * (1.0 - sig_x)
            }
            ActivationFunction::Linear => 1.0,
        }
    }
}

/// Long Short-Term Memory (LSTM) cell
#[derive(Debug, Clone)]
pub struct LSTMCell {
    /// Input gate weights
    w_i: Array2<f64>,
    /// Forget gate weights
    w_f: Array2<f64>,
    /// Output gate weights
    w_o: Array2<f64>,
    /// Candidate gate weights
    w_c: Array2<f64>,
    /// Hidden state weights for input gate
    u_i: Array2<f64>,
    /// Hidden state weights for forget gate
    u_f: Array2<f64>,
    /// Hidden state weights for output gate
    u_o: Array2<f64>,
    /// Hidden state weights for candidate gate
    u_c: Array2<f64>,
    /// Bias vectors
    b_i: Array1<f64>,
    b_f: Array1<f64>,
    b_o: Array1<f64>,
    b_c: Array1<f64>,
    /// Input size
    input_size: usize,
    /// Hidden size
    hidden_size: usize,
}

impl LSTMCell {
    /// Create new LSTM cell
    pub fn new(_input_size: usize, hiddensize: usize) -> Self {
        let scale = (2.0 / (_input_size + hiddensize) as f64).sqrt();

        // Initialize weights with Xavier initialization
        let w_i = Array2::from_shape_fn((hiddensize, _input_size), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_f = Array2::from_shape_fn((hiddensize, _input_size), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_o = Array2::from_shape_fn((hiddensize, _input_size), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_c = Array2::from_shape_fn((hiddensize, _input_size), |_| {
            rand::rng().random_range(-scale..scale)
        });

        let u_i = Array2::from_shape_fn((hiddensize, hiddensize), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let u_f = Array2::from_shape_fn((hiddensize, hiddensize), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let u_o = Array2::from_shape_fn((hiddensize, hiddensize), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let u_c = Array2::from_shape_fn((hiddensize, hiddensize), |_| {
            rand::rng().random_range(-scale..scale)
        });

        // Initialize biases (forget gate bias to 1.0 for better gradient flow)
        let b_i = Array1::zeros(hiddensize);
        let b_f = Array1::ones(hiddensize);
        let b_o = Array1::zeros(hiddensize);
        let b_c = Array1::zeros(hiddensize);

        Self {
            w_i,
            w_f,
            w_o,
            w_c,
            u_i,
            u_f,
            u_o,
            u_c,
            b_i,
            b_f,
            b_o,
            b_c,
            input_size: _input_size,
            hidden_size: hiddensize,
        }
    }

    /// Forward pass through LSTM cell
    pub fn forward(
        &self,
        x: ArrayView1<f64>,
        h_prev: ArrayView1<f64>,
        c_prev: ArrayView1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        if x.len() != self.input_size {
            return Err(TextError::InvalidInput(format!(
                "Expected input size {}, got {}",
                self.input_size,
                x.len()
            )));
        }

        if h_prev.len() != self.hidden_size || c_prev.len() != self.hidden_size {
            return Err(TextError::InvalidInput(format!(
                "Expected hidden size {}, got h: {}, c: {}",
                self.hidden_size,
                h_prev.len(),
                c_prev.len()
            )));
        }

        // Input gate
        let i_t = ActivationFunction::Sigmoid
            .apply_array(&(self.w_i.dot(&x) + self.u_i.dot(&h_prev) + &self.b_i));

        // Forget gate
        let f_t = ActivationFunction::Sigmoid
            .apply_array(&(self.w_f.dot(&x) + self.u_f.dot(&h_prev) + &self.b_f));

        // Output gate
        let o_t = ActivationFunction::Sigmoid
            .apply_array(&(self.w_o.dot(&x) + self.u_o.dot(&h_prev) + &self.b_o));

        // Candidate values
        let c_tilde = ActivationFunction::Tanh
            .apply_array(&(self.w_c.dot(&x) + self.u_c.dot(&h_prev) + &self.b_c));

        // Cell state
        let c_t = &f_t * &c_prev + &i_t * &c_tilde;

        // Hidden state
        let h_t = &o_t * &ActivationFunction::Tanh.apply_array(&c_t);

        Ok((h_t, c_t))
    }
}

/// Gated Recurrent Unit (GRU) cell
#[derive(Debug, Clone)]
pub struct GRUCell {
    /// Update gate weights
    w_z: Array2<f64>,
    /// Reset gate weights  
    w_r: Array2<f64>,
    /// New gate weights
    w_h: Array2<f64>,
    /// Hidden state weights for update gate
    u_z: Array2<f64>,
    /// Hidden state weights for reset gate
    u_r: Array2<f64>,
    /// Hidden state weights for new gate
    u_h: Array2<f64>,
    /// Bias vectors
    b_z: Array1<f64>,
    b_r: Array1<f64>,
    b_h: Array1<f64>,
    /// Input size
    input_size: usize,
    /// Hidden size
    hidden_size: usize,
}

impl GRUCell {
    /// Create new GRU cell
    pub fn new(_input_size: usize, hiddensize: usize) -> Self {
        let scale = (2.0 / (_input_size + hiddensize) as f64).sqrt();

        // Initialize weights with Xavier initialization
        let w_z = Array2::from_shape_fn((hiddensize, _input_size), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_r = Array2::from_shape_fn((hiddensize, _input_size), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_h = Array2::from_shape_fn((hiddensize, _input_size), |_| {
            rand::rng().random_range(-scale..scale)
        });

        let u_z = Array2::from_shape_fn((hiddensize, hiddensize), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let u_r = Array2::from_shape_fn((hiddensize, hiddensize), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let u_h = Array2::from_shape_fn((hiddensize, hiddensize), |_| {
            rand::rng().random_range(-scale..scale)
        });

        // Initialize biases
        let b_z = Array1::zeros(hiddensize);
        let b_r = Array1::zeros(hiddensize);
        let b_h = Array1::zeros(hiddensize);

        Self {
            w_z,
            w_r,
            w_h,
            u_z,
            u_r,
            u_h,
            b_z,
            b_r,
            b_h,
            input_size: _input_size,
            hidden_size: hiddensize,
        }
    }

    /// Forward pass through GRU cell
    pub fn forward(&self, x: ArrayView1<f64>, hprev: ArrayView1<f64>) -> Result<Array1<f64>> {
        if x.len() != self.input_size {
            return Err(TextError::InvalidInput(format!(
                "Expected input size {}, got {}",
                self.input_size,
                x.len()
            )));
        }

        if hprev.len() != self.hidden_size {
            return Err(TextError::InvalidInput(format!(
                "Expected hidden size {}, got {}",
                self.hidden_size,
                hprev.len()
            )));
        }

        // Update gate
        let z_t = ActivationFunction::Sigmoid
            .apply_array(&(self.w_z.dot(&x) + self.u_z.dot(&hprev) + &self.b_z));

        // Reset gate
        let r_t = ActivationFunction::Sigmoid
            .apply_array(&(self.w_r.dot(&x) + self.u_r.dot(&hprev) + &self.b_r));

        // New gate (candidate activation)
        let h_tilde = ActivationFunction::Tanh
            .apply_array(&(self.w_h.dot(&x) + self.u_h.dot(&(&r_t * &hprev)) + &self.b_h));

        // Final hidden state
        let h_t = &(&Array1::ones(self.hidden_size) - &z_t) * &hprev + &z_t * &h_tilde;

        Ok(h_t)
    }
}

/// Bidirectional LSTM layer
pub struct BiLSTM {
    /// Forward LSTM cells
    forward_cells: Vec<LSTMCell>,
    /// Backward LSTM cells
    backward_cells: Vec<LSTMCell>,
    /// Number of layers
    num_layers: usize,
    /// Hidden size
    hidden_size: usize,
}

impl BiLSTM {
    /// Create new bidirectional LSTM
    pub fn new(_input_size: usize, hidden_size: usize, numlayers: usize) -> Self {
        let mut forward_cells = Vec::new();
        let mut backward_cells = Vec::new();

        for i in 0..numlayers {
            let layer_input_size = if i == 0 { _input_size } else { hidden_size * 2 };
            forward_cells.push(LSTMCell::new(layer_input_size, hidden_size));
            backward_cells.push(LSTMCell::new(layer_input_size, hidden_size));
        }

        Self {
            forward_cells,
            backward_cells,
            num_layers: numlayers,
            hidden_size,
        }
    }

    /// Forward pass through bidirectional LSTM
    pub fn forward(&self, sequence: ArrayView2<f64>) -> Result<Array2<f64>> {
        let (seq_len, input_size) = sequence.dim();
        let output_size = self.hidden_size * 2; // Concatenated forward and backward

        let mut current_input = sequence.to_owned();

        for layer in 0..self.num_layers {
            let mut forward_outputs = Vec::new();
            let mut backward_outputs = Vec::new();

            // Forward direction
            let mut h_forward = Array1::zeros(self.hidden_size);
            let mut c_forward = Array1::zeros(self.hidden_size);

            for t in 0..seq_len {
                let (h_new, c_new) = self.forward_cells[layer].forward(
                    current_input.row(t),
                    h_forward.view(),
                    c_forward.view(),
                )?;
                h_forward = h_new;
                c_forward = c_new;
                forward_outputs.push(h_forward.clone());
            }

            // Backward direction
            let mut h_backward = Array1::zeros(self.hidden_size);
            let mut c_backward = Array1::zeros(self.hidden_size);

            for t in (0..seq_len).rev() {
                let (h_new, c_new) = self.backward_cells[layer].forward(
                    current_input.row(t),
                    h_backward.view(),
                    c_backward.view(),
                )?;
                h_backward = h_new;
                c_backward = c_new;
                backward_outputs.push(h_backward.clone());
            }

            // Reverse backward outputs to match forward order
            backward_outputs.reverse();

            // Concatenate forward and backward outputs
            let mut layer_output = Array2::zeros((seq_len, output_size));
            for t in 0..seq_len {
                let mut concat_output = Array1::zeros(output_size);
                concat_output
                    .slice_mut(s![..self.hidden_size])
                    .assign(&forward_outputs[t]);
                concat_output
                    .slice_mut(s![self.hidden_size..])
                    .assign(&backward_outputs[t]);
                layer_output.row_mut(t).assign(&concat_output);
            }

            current_input = layer_output;
        }

        Ok(current_input)
    }
}

/// Convolutional layer for text processing
#[derive(Debug, Clone)]
pub struct Conv1D {
    /// Convolution filters
    filters: Array3<f64>,
    /// Bias terms
    bias: Array1<f64>,
    /// Number of filters
    num_filters: usize,
    /// Kernel size
    kernel_size: usize,
    /// Input channels
    input_channels: usize,
    /// Activation function
    activation: ActivationFunction,
}

impl Conv1D {
    /// Create new 1D convolutional layer
    pub fn new(
        input_channels: usize,
        num_filters: usize,
        kernel_size: usize,
        activation: ActivationFunction,
    ) -> Self {
        let scale = (2.0 / (input_channels * kernel_size) as f64).sqrt();

        // Initialize _filters with Xavier initialization
        let _filters = Array3::from_shape_fn((num_filters, input_channels, kernel_size), |_| {
            rand::rng().random_range(-scale..scale)
        });

        let bias = Array1::zeros(num_filters);

        Self {
            filters: _filters,
            bias,
            num_filters,
            kernel_size,
            input_channels,
            activation,
        }
    }

    /// Forward pass through convolution layer
    pub fn forward(&self, input: ArrayView2<f64>) -> Result<Array2<f64>> {
        let (seq_len, input_dim) = input.dim();

        if input_dim != self.input_channels {
            return Err(TextError::InvalidInput(format!(
                "Expected {} input channels, got {}",
                self.input_channels, input_dim
            )));
        }

        let output_len = seq_len.saturating_sub(self.kernel_size - 1);
        let mut output = Array2::zeros((output_len, self.num_filters));

        for filter_idx in 0..self.num_filters {
            for pos in 0..output_len {
                let mut conv_sum = 0.0;

                for ch in 0..self.input_channels {
                    for k in 0..self.kernel_size {
                        if pos + k < seq_len {
                            conv_sum += input[[pos + k, ch]] * self.filters[[filter_idx, ch, k]];
                        }
                    }
                }

                conv_sum += self.bias[filter_idx];
                output[[pos, filter_idx]] = self.activation.apply(conv_sum);
            }
        }

        Ok(output)
    }
}

/// Max pooling layer for 1D data
#[derive(Debug)]
pub struct MaxPool1D {
    /// Pool size
    pool_size: usize,
    /// Stride
    stride: usize,
}

impl MaxPool1D {
    /// Create new max pooling layer
    pub fn new(poolsize: usize, stride: usize) -> Self {
        Self {
            pool_size: poolsize,
            stride,
        }
    }

    /// Forward pass through max pooling
    pub fn forward(&self, input: ArrayView2<f64>) -> Array2<f64> {
        let (seq_len, channels) = input.dim();
        let output_len = (seq_len - self.pool_size) / self.stride + 1;

        let mut output = Array2::zeros((output_len, channels));

        for ch in 0..channels {
            for i in 0..output_len {
                let start = i * self.stride;
                let end = (start + self.pool_size).min(seq_len);

                let mut max_val = f64::NEG_INFINITY;
                for j in start..end {
                    max_val = max_val.max(input[[j, ch]]);
                }

                output[[i, ch]] = max_val;
            }
        }

        output
    }
}

/// Residual block for CNNs
#[derive(Debug, Clone)]
pub struct ResidualBlock1D {
    /// First convolution layer
    conv1: Conv1D,
    /// Second convolution layer  
    conv2: Conv1D,
    /// Skip connection projection (for dimension matching)
    skip_projection: Option<Array2<f64>>,
    /// Batch normalization parameters
    bn1_scale: Array1<f64>,
    bn1_shift: Array1<f64>,
    bn2_scale: Array1<f64>,
    bn2_shift: Array1<f64>,
}

impl ResidualBlock1D {
    /// Create new residual block
    pub fn new(_input_channels: usize, output_channels: usize, kernelsize: usize) -> Self {
        let conv1 = Conv1D::new(
            _input_channels,
            output_channels,
            kernelsize,
            ActivationFunction::Linear,
        );
        let conv2 = Conv1D::new(
            output_channels,
            output_channels,
            kernelsize,
            ActivationFunction::Linear,
        );

        // Skip projection if input and output _channels differ
        let skip_projection = if _input_channels != output_channels {
            let scale = (2.0 / _input_channels as f64).sqrt();
            Some(Array2::from_shape_fn(
                (output_channels, _input_channels),
                |_| rand::rng().random_range(-scale..scale),
            ))
        } else {
            None
        };

        // Batch normalization parameters
        let bn1_scale = Array1::ones(output_channels);
        let bn1_shift = Array1::zeros(output_channels);
        let bn2_scale = Array1::ones(output_channels);
        let bn2_shift = Array1::zeros(output_channels);

        Self {
            conv1,
            conv2,
            skip_projection,
            bn1_scale,
            bn1_shift,
            bn2_scale,
            bn2_shift,
        }
    }

    /// Forward pass through residual block
    pub fn forward(&self, input: ArrayView2<f64>) -> Result<Array2<f64>> {
        // First convolution + batch norm + ReLU
        let conv1_out = self.conv1.forward(input)?;
        let bn1_out = self.batch_norm(&conv1_out, &self.bn1_scale, &self.bn1_shift);
        let relu1_out = bn1_out.mapv(|x| ActivationFunction::ReLU.apply(x));

        // Second convolution + batch norm
        let conv2_out = self.conv2.forward(relu1_out.view())?;
        let bn2_out = self.batch_norm(&conv2_out, &self.bn2_scale, &self.bn2_shift);

        // Skip connection
        let skip_out = if let Some(ref projection) = self.skip_projection {
            // Project input to match output channels: (seq_len, input_channels) -> (seq_len, output_channels)
            let projected = input.dot(&projection.t());

            // Handle sequence length mismatch due to convolutions
            // Each convolution reduces length by (kernel_size - 1), so total reduction is 2 * (kernel_size - 1)
            let conv_output_len = bn2_out.shape()[0];
            let skip_len = projected.shape()[0];

            if conv_output_len < skip_len {
                // Take center slice of skip connection to match conv output length
                let start = (skip_len - conv_output_len) / 2;
                let end = start + conv_output_len;
                projected.slice(s![start..end, ..]).to_owned()
            } else {
                projected
            }
        } else {
            // Direct skip connection - handle sequence length mismatch
            let conv_output_len = bn2_out.shape()[0];
            let skip_len = input.shape()[0];

            if conv_output_len < skip_len {
                // Take center slice of input to match conv output length
                let start = (skip_len - conv_output_len) / 2;
                let end = start + conv_output_len;
                input.slice(s![start..end, ..]).to_owned()
            } else {
                input.to_owned()
            }
        };

        // Add skip connection and apply ReLU
        let output = &bn2_out + &skip_out;
        Ok(output.mapv(|x| ActivationFunction::ReLU.apply(x)))
    }

    /// Simple batch normalization (simplified implementation)
    fn batch_norm(
        &self,
        input: &Array2<f64>,
        scale: &Array1<f64>,
        shift: &Array1<f64>,
    ) -> Array2<f64> {
        let mut result = input.clone();
        let eps = 1e-5;

        // Normalize over the sequence dimension for each channel
        for ch in 0..input.shape()[1] {
            let channel_data = input.column(ch);
            let mean = channel_data.mean();
            let var = channel_data.mapv(|x| (x - mean).powi(2)).mean();
            let std = (var + eps).sqrt();

            let mut normalized = channel_data.mapv(|x| (x - mean) / std);
            normalized = normalized * scale[ch] + shift[ch];

            result.column_mut(ch).assign(&normalized);
        }

        result
    }
}

/// Multi-scale CNN for text processing
#[derive(Debug)]
pub struct MultiScaleCNN {
    /// Parallel convolution branches with different kernel sizes
    conv_branches: Vec<Conv1D>,
    /// Batch normalization for each branch
    bn_branches: Vec<(Array1<f64>, Array1<f64>)>,
    /// Combination weights
    combinationweights: Array2<f64>,
    /// Global max pooling
    #[allow(dead_code)]
    global_pool: MaxPool1D,
}

impl MultiScaleCNN {
    /// Create new multi-scale CNN
    pub fn new(
        input_channels: usize,
        num_filters_per_scale: usize,
        kernel_sizes: Vec<usize>,
        output_size: usize,
    ) -> Self {
        let mut conv_branches = Vec::new();
        let mut bn_branches = Vec::new();

        // Create convolution branches for different scales
        for &kernel_size in &kernel_sizes {
            conv_branches.push(Conv1D::new(
                input_channels,
                num_filters_per_scale,
                kernel_size,
                ActivationFunction::ReLU,
            ));

            // Batch normalization parameters
            bn_branches.push((
                Array1::ones(num_filters_per_scale),
                Array1::zeros(num_filters_per_scale),
            ));
        }

        // Combination layer
        let total_features = kernel_sizes.len() * num_filters_per_scale;
        let _scale = (2.0 / total_features as f64).sqrt();
        let combination_weights = Array2::from_shape_fn((output_size, total_features), |_| {
            rand::rng().random_range(-_scale.._scale)
        });

        let global_pool = MaxPool1D::new(2, 2);

        Self {
            conv_branches,
            bn_branches,
            combinationweights: combination_weights,
            global_pool,
        }
    }

    /// Forward pass through multi-scale CNN
    pub fn forward(&self, input: ArrayView2<f64>) -> Result<Array1<f64>> {
        let mut branch_outputs = Vec::new();

        // Process each scale branch
        for (i, conv) in self.conv_branches.iter().enumerate() {
            let conv_out = conv.forward(input)?;

            // Apply batch normalization
            let (scale, shift) = &self.bn_branches[i];
            let bn_out = self.batch_norm_branch(&conv_out, scale, shift);

            // Global max pooling over sequence dimension
            let global_max = bn_out.map_axis(Axis(0), |row| {
                row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            });

            branch_outputs.push(global_max);
        }

        // Concatenate all branch outputs
        let mut concatenated = Array1::zeros(branch_outputs.iter().map(|x| x.len()).sum::<usize>());
        let mut offset = 0;
        for branch_output in branch_outputs {
            let end = offset + branch_output.len();
            concatenated
                .slice_mut(s![offset..end])
                .assign(&branch_output);
            offset = end;
        }

        // Final combination layer
        Ok(self.combinationweights.dot(&concatenated))
    }

    /// Batch normalization for a single branch
    fn batch_norm_branch(
        &self,
        input: &Array2<f64>,
        scale: &Array1<f64>,
        shift: &Array1<f64>,
    ) -> Array2<f64> {
        let mut result = input.clone();
        let eps = 1e-5;

        for ch in 0..input.shape()[1] {
            let channel_data = input.column(ch);
            let mean = channel_data.mean();
            let var = channel_data.mapv(|x| (x - mean).powi(2)).mean();
            let std = (var + eps).sqrt();

            let mut normalized = channel_data.mapv(|x| (x - mean) / std);
            normalized = normalized * scale[ch] + shift[ch];

            result.column_mut(ch).assign(&normalized);
        }

        result
    }
}

/// Attention mechanism for sequence-to-sequence models
pub struct AdditiveAttention {
    /// Attention weights
    w_a: Array2<f64>,
    /// Query projection
    #[allow(dead_code)]
    w_q: Array2<f64>,
    /// Key projection
    #[allow(dead_code)]
    w_k: Array2<f64>,
    /// Value projection
    #[allow(dead_code)]
    w_v: Array2<f64>,
    /// Attention vector
    v_a: Array1<f64>,
}

impl AdditiveAttention {
    /// Create new additive attention mechanism
    pub fn new(_encoder_dim: usize, decoder_dim: usize, attentiondim: usize) -> Self {
        let scale = (2.0 / attentiondim as f64).sqrt();

        let w_a = Array2::from_shape_fn((attentiondim, _encoder_dim + decoder_dim), |_| {
            rand::rng().random_range(-scale..scale)
        });

        let w_q = Array2::from_shape_fn((attentiondim, decoder_dim), |_| {
            rand::rng().random_range(-scale..scale)
        });

        let w_k = Array2::from_shape_fn((attentiondim, _encoder_dim), |_| {
            rand::rng().random_range(-scale..scale)
        });

        let w_v = Array2::from_shape_fn((_encoder_dim, _encoder_dim), |_| {
            rand::rng().random_range(-scale..scale)
        });

        let v_a = Array1::from_shape_fn(attentiondim, |_| rand::rng().random_range(-scale..scale));

        Self {
            w_a,
            w_q,
            w_k,
            w_v,
            v_a,
        }
    }

    /// Compute attention scores
    pub fn forward(
        &self,
        query: ArrayView1<f64>,
        encoder_outputs: ArrayView2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let seq_len = encoder_outputs.shape()[0];
        let mut attention_scores = Array1::zeros(seq_len);

        // Compute attention scores for each encoder output
        for i in 0..seq_len {
            let encoder_output = encoder_outputs.row(i);

            // Concatenate query and encoder output
            let mut combined = Array1::zeros(query.len() + encoder_output.len());
            combined.slice_mut(s![..query.len()]).assign(&query);
            combined
                .slice_mut(s![query.len()..])
                .assign(&encoder_output);

            // Compute attention score
            let attention_input = self.w_a.dot(&combined);
            let activated = ActivationFunction::Tanh.apply_array(&attention_input);
            attention_scores[i] = self.v_a.dot(&activated);
        }

        // Apply softmax to get attention weights
        let attention_weights = self.softmax(&attention_scores);

        // Compute context vector
        let context = encoder_outputs.t().dot(&attention_weights);

        Ok((context, attention_weights))
    }

    /// Apply softmax to scores
    fn softmax(&self, scores: &Array1<f64>) -> Array1<f64> {
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores = scores.mapv(|x| (x - max_score).exp());
        let sum_exp = exp_scores.sum();
        exp_scores / sum_exp
    }
}

/// Self-attention mechanism (simplified transformer-style)
#[derive(Debug)]
pub struct SelfAttention {
    /// Query projection
    w_q: Array2<f64>,
    /// Key projection
    w_k: Array2<f64>,
    /// Value projection
    w_v: Array2<f64>,
    /// Output projection
    w_o: Array2<f64>,
    /// Attention dimension
    d_k: usize,
    /// Dropout rate
    #[allow(dead_code)]
    dropout: f64,
}

impl SelfAttention {
    /// Create new self-attention layer
    pub fn new(_dmodel: usize, dropout: f64) -> Self {
        let d_k = _dmodel;
        let scale = (2.0 / _dmodel as f64).sqrt();

        let w_q =
            Array2::from_shape_fn((_dmodel, d_k), |_| rand::rng().random_range(-scale..scale));
        let w_k =
            Array2::from_shape_fn((_dmodel, d_k), |_| rand::rng().random_range(-scale..scale));
        let w_v =
            Array2::from_shape_fn((_dmodel, d_k), |_| rand::rng().random_range(-scale..scale));
        let w_o =
            Array2::from_shape_fn((d_k, _dmodel), |_| rand::rng().random_range(-scale..scale));

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            d_k,
            dropout,
        }
    }

    /// Forward pass through self-attention
    pub fn forward(
        &self,
        input: ArrayView2<f64>,
        mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        let _seq_len = input.shape()[0];

        // Compute Q, K, V
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);

        // Scaled dot-product attention
        let attention_output =
            self.scaled_dot_product_attention(q.view(), k.view(), v.view(), mask)?;

        // Output projection
        Ok(attention_output.dot(&self.w_o))
    }

    /// Scaled dot-product attention computation
    fn scaled_dot_product_attention(
        &self,
        q: ArrayView2<f64>,
        k: ArrayView2<f64>,
        v: ArrayView2<f64>,
        mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        let d_k = self.d_k as f64;

        // Compute attention scores: Q * K^T / sqrt(d_k)
        let scores = q.dot(&k.t()) / d_k.sqrt();

        // Apply mask if provided
        let mut masked_scores = scores;
        if let Some(mask) = mask {
            for ((i, j), &should_mask) in mask.indexed_iter() {
                if should_mask {
                    masked_scores[[i, j]] = f64::NEG_INFINITY;
                }
            }
        }

        // Apply softmax
        let attention_weights = self.softmax_2d(&masked_scores)?;

        // Apply attention to values
        Ok(attention_weights.dot(&v))
    }

    /// Apply softmax to 2D array along last axis
    fn softmax_2d(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let mut result = x.clone();

        for mut row in result.rows_mut() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|x| (x - max_val).exp());
            let sum: f64 = row.sum();
            if sum > 0.0 {
                row /= sum;
            }
        }

        Ok(result)
    }
}

/// Cross-attention mechanism for encoder-decoder architectures
#[derive(Debug)]
pub struct CrossAttention {
    /// Query projection
    w_q: Array2<f64>,
    /// Key projection
    w_k: Array2<f64>,
    /// Value projection
    w_v: Array2<f64>,
    /// Output projection
    w_o: Array2<f64>,
    /// Attention dimension
    d_k: usize,
}

impl CrossAttention {
    /// Create new cross-attention layer
    pub fn new(_dmodel: usize) -> Self {
        let d_k = _dmodel;
        let scale = (2.0 / _dmodel as f64).sqrt();

        let w_q =
            Array2::from_shape_fn((_dmodel, d_k), |_| rand::rng().random_range(-scale..scale));
        let w_k =
            Array2::from_shape_fn((_dmodel, d_k), |_| rand::rng().random_range(-scale..scale));
        let w_v =
            Array2::from_shape_fn((_dmodel, d_k), |_| rand::rng().random_range(-scale..scale));
        let w_o =
            Array2::from_shape_fn((d_k, _dmodel), |_| rand::rng().random_range(-scale..scale));

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            d_k,
        }
    }

    /// Forward pass through cross-attention
    pub fn forward(
        &self,
        query: ArrayView2<f64>,
        key: ArrayView2<f64>,
        value: ArrayView2<f64>,
        mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        // Compute Q, K, V
        let q = query.dot(&self.w_q);
        let k = key.dot(&self.w_k);
        let v = value.dot(&self.w_v);

        // Scaled dot-product attention
        let attention_output =
            self.scaled_dot_product_attention(q.view(), k.view(), v.view(), mask)?;

        // Output projection
        Ok(attention_output.dot(&self.w_o))
    }

    /// Scaled dot-product attention computation
    fn scaled_dot_product_attention(
        &self,
        q: ArrayView2<f64>,
        k: ArrayView2<f64>,
        v: ArrayView2<f64>,
        mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        let d_k = self.d_k as f64;

        // Compute attention scores: Q * K^T / sqrt(d_k)
        let scores = q.dot(&k.t()) / d_k.sqrt();

        // Apply mask if provided
        let mut masked_scores = scores;
        if let Some(mask) = mask {
            for ((i, j), &should_mask) in mask.indexed_iter() {
                if should_mask {
                    masked_scores[[i, j]] = f64::NEG_INFINITY;
                }
            }
        }

        // Apply softmax
        let attention_weights = self.softmax_2d(&masked_scores)?;

        // Apply attention to values
        Ok(attention_weights.dot(&v))
    }

    /// Apply softmax to 2D array along last axis
    fn softmax_2d(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let mut result = x.clone();

        for mut row in result.rows_mut() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|x| (x - max_val).exp());
            let sum: f64 = row.sum();
            if sum > 0.0 {
                row /= sum;
            }
        }

        Ok(result)
    }
}

/// Positional feed-forward network with GELU activation
#[derive(Debug)]
pub struct PositionwiseFeedForward {
    /// First linear transformation
    w1: Array2<f64>,
    /// Second linear transformation
    w2: Array2<f64>,
    /// Bias vectors
    b1: Array1<f64>,
    b2: Array1<f64>,
    /// Dropout rate
    dropout: f64,
}

impl PositionwiseFeedForward {
    /// Create new position-wise feed-forward network
    pub fn new(_dmodel: usize, dff: usize, dropout: f64) -> Self {
        let scale1 = (2.0 / _dmodel as f64).sqrt();
        let scale2 = (2.0 / dff as f64).sqrt();

        let w1 = Array2::from_shape_fn((dff, _dmodel), |_| {
            rand::rng().random_range(-scale1..scale1)
        });
        let w2 = Array2::from_shape_fn((_dmodel, dff), |_| {
            rand::rng().random_range(-scale2..scale2)
        });
        let b1 = Array1::zeros(dff);
        let b2 = Array1::zeros(_dmodel);

        Self {
            w1,
            w2,
            b1,
            b2,
            dropout,
        }
    }

    /// Forward pass through feed-forward network
    pub fn forward(&self, x: ArrayView2<f64>) -> Array2<f64> {
        // First linear transformation + GELU
        let hidden = x.dot(&self.w1.t()) + &self.b1;
        let activated = hidden.mapv(|x| ActivationFunction::GELU.apply(x));

        // Apply dropout (simplified - in practice would use random mask)
        let dropout_mask = if self.dropout > 0.0 {
            1.0 - self.dropout
        } else {
            1.0
        };
        let dropped = activated * dropout_mask;

        // Second linear transformation
        dropped.dot(&self.w2.t()) + &self.b2
    }
}

/// Text CNN architecture for classification
pub struct TextCNN {
    /// Convolutional layers with different kernel sizes
    conv_layers: Vec<Conv1D>,
    /// Max pooling layers
    pool_layers: Vec<MaxPool1D>,
    /// Fully connected layer weights
    fcweights: Array2<f64>,
    /// Fully connected layer bias
    fc_bias: Array1<f64>,
    /// Dropout rate
    dropout_rate: f64,
}

impl TextCNN {
    /// Create new Text CNN
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _vocab_size: usize,
        embedding_dim: usize,
        num_filters: usize,
        filter_sizes: Vec<usize>,
        num_classes: usize,
        dropout_rate: f64,
    ) -> Self {
        let mut conv_layers = Vec::new();
        let mut pool_layers = Vec::new();

        // Create convolutional layers with different filter _sizes
        for &filter_size in &filter_sizes {
            conv_layers.push(Conv1D::new(
                embedding_dim,
                num_filters,
                filter_size,
                ActivationFunction::ReLU,
            ));
            pool_layers.push(MaxPool1D::new(2, 2));
        }

        // Fully connected layer
        let fc_input_size = num_filters * filter_sizes.len();
        let scale = (2.0 / fc_input_size as f64).sqrt();

        let fc_weights = Array2::from_shape_fn((num_classes, fc_input_size), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let fc_bias = Array1::zeros(num_classes);

        Self {
            conv_layers,
            pool_layers,
            fcweights: fc_weights,
            fc_bias,
            dropout_rate,
        }
    }

    /// Forward pass through Text CNN
    pub fn forward(&self, embeddings: ArrayView2<f64>) -> Result<Array1<f64>> {
        let mut feature_maps = Vec::new();

        // Apply each convolutional layer
        for (conv_layer, pool_layer) in self.conv_layers.iter().zip(&self.pool_layers) {
            let conv_output = conv_layer.forward(embeddings)?;
            let pooled_output = pool_layer.forward(conv_output.view());

            // Global max pooling over sequence dimension
            let global_max = pooled_output.map_axis(Axis(0), |row| {
                row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            });

            feature_maps.push(global_max);
        }

        // Concatenate all feature maps
        let mut concatenated_features =
            Array1::zeros(feature_maps.iter().map(|fm| fm.len()).sum::<usize>());
        let mut offset = 0;
        for feature_map in feature_maps {
            let end = offset + feature_map.len();
            concatenated_features
                .slice_mut(s![offset..end])
                .assign(&feature_map);
            offset = end;
        }

        // Apply dropout (simplified - in practice would use random mask)
        let dropout_mask = if self.dropout_rate > 0.0 {
            1.0 - self.dropout_rate
        } else {
            1.0
        };
        concatenated_features *= dropout_mask;

        // Fully connected layer
        let output = self.fcweights.dot(&concatenated_features) + &self.fc_bias;

        Ok(output)
    }
}

/// Hybrid CNN-LSTM architecture
pub struct CNNLSTMHybrid {
    /// CNN feature extractor
    cnn: TextCNN,
    /// LSTM sequence processor
    lstm: BiLSTM,
    /// Final classification layer
    classifier: Array2<f64>,
    /// Classifier bias
    classifier_bias: Array1<f64>,
}

impl CNNLSTMHybrid {
    /// Create new CNN-LSTM hybrid model
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embedding_dim: usize,
        cnn_filters: usize,
        filter_sizes: Vec<usize>,
        lstm_hidden_size: usize,
        lstm_layers: usize,
        num_classes: usize,
    ) -> Self {
        // CNN for local feature extraction
        let cnn = TextCNN::new(
            0, // vocab_size not needed for feature extraction
            embedding_dim,
            cnn_filters,
            filter_sizes.clone(),
            cnn_filters * filter_sizes.len(),
            0.0, // No dropout in feature extraction
        );

        // BiLSTM for sequence modeling
        let lstm_input_size = cnn_filters * filter_sizes.len();
        let lstm = BiLSTM::new(lstm_input_size, lstm_hidden_size, lstm_layers);

        // Final classifier
        let classifier_input_size = lstm_hidden_size * 2; // Bidirectional
        let scale = (2.0 / classifier_input_size as f64).sqrt();

        let classifier = Array2::from_shape_fn((num_classes, classifier_input_size), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let classifier_bias = Array1::zeros(num_classes);

        Self {
            cnn,
            lstm,
            classifier,
            classifier_bias,
        }
    }

    /// Forward pass through hybrid model
    pub fn forward(&self, embeddings: ArrayView2<f64>) -> Result<Array1<f64>> {
        // Extract CNN features (this is simplified - would need proper implementation)
        let cnn_features = self.cnn.forward(embeddings)?;

        // Reshape for LSTM input (simplified)
        let lstm_input = Array2::from_shape_vec((1, cnn_features.len()), cnn_features.to_vec())
            .map_err(|e| TextError::InvalidInput(format!("Reshape error: {e}")))?;

        // Process through LSTM
        let lstm_output = self.lstm.forward(lstm_input.view())?;

        // Take last timestep output
        let final_hidden = lstm_output.row(lstm_output.shape()[0] - 1);

        // Final classification
        let output = self.classifier.dot(&final_hidden) + &self.classifier_bias;

        Ok(output)
    }
}

/// Layer normalization for neural networks
pub struct LayerNorm {
    /// Learnable scale parameters
    weight: Array1<f64>,
    /// Learnable bias parameters
    bias: Array1<f64>,
    /// Small epsilon for numerical stability
    eps: f64,
}

impl LayerNorm {
    /// Create new layer normalization layer
    pub fn new(normalizedshape: usize) -> Self {
        Self {
            weight: Array1::ones(normalizedshape),
            bias: Array1::zeros(normalizedshape),
            eps: 1e-6,
        }
    }

    /// Forward pass with layer normalization
    pub fn forward(&self, x: ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut output = Array2::zeros(x.raw_dim());

        // Normalize along the last dimension for each sample
        for (i, row) in x.outer_iter().enumerate() {
            let mean = row.mean();
            let variance = row.mapv(|v| (v - mean).powi(2)).mean();
            let std = (variance + self.eps).sqrt();

            // Apply normalization and learned parameters
            for (j, &val) in row.iter().enumerate() {
                let normalized = (val - mean) / std;
                output[[i, j]] = normalized * self.weight[j] + self.bias[j];
            }
        }

        Ok(output)
    }
}

/// Dropout layer for regularization
pub struct Dropout {
    /// Dropout probability
    p: f64,
    /// Whether the layer is in training mode
    training: bool,
}

impl Dropout {
    /// Create new dropout layer
    pub fn new(p: f64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            training: true,
        }
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Forward pass with dropout
    pub fn forward(&self, x: ArrayView2<f64>) -> Array2<f64> {
        if !self.training || self.p == 0.0 {
            return x.to_owned();
        }

        let mut output = x.to_owned();
        let scale = 1.0 / (1.0 - self.p);

        for elem in output.iter_mut() {
            if rand::rng().random_range(0.0..1.0) < self.p {
                *elem = 0.0; // Drop the element
            } else {
                *elem *= scale; // Scale to maintain expected value
            }
        }

        output
    }
}

/// Multi-head attention mechanism
pub struct MultiHeadAttention {
    /// Number of attention heads
    num_heads: usize,
    /// Model dimension
    d_model: usize,
    /// Dimension per head
    d_k: usize,
    /// Query projection weights
    w_q: Array2<f64>,
    /// Key projection weights
    w_k: Array2<f64>,
    /// Value projection weights
    w_v: Array2<f64>,
    /// Output projection weights
    w_o: Array2<f64>,
    /// Dropout layer
    dropout: Dropout,
}

impl MultiHeadAttention {
    /// Create new multi-head attention layer
    pub fn new(_dmodel: usize, num_heads: usize, dropoutp: f64) -> Result<Self> {
        if _dmodel % num_heads != 0 {
            return Err(TextError::InvalidInput(
                "Model dimension must be divisible by number of _heads".to_string(),
            ));
        }

        let d_k = _dmodel / num_heads;
        let scale = (2.0 / _dmodel as f64).sqrt();

        let w_q = Array2::from_shape_fn((_dmodel, _dmodel), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_k = Array2::from_shape_fn((_dmodel, _dmodel), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_v = Array2::from_shape_fn((_dmodel, _dmodel), |_| {
            rand::rng().random_range(-scale..scale)
        });
        let w_o = Array2::from_shape_fn((_dmodel, _dmodel), |_| {
            rand::rng().random_range(-scale..scale)
        });

        Ok(Self {
            num_heads,
            d_model: _dmodel,
            d_k,
            w_q,
            w_k,
            w_v,
            w_o,
            dropout: Dropout::new(dropoutp),
        })
    }

    /// Forward pass through multi-head attention
    pub fn forward(
        &self,
        query: ArrayView2<f64>,
        key: ArrayView2<f64>,
        value: ArrayView2<f64>,
        mask: Option<ArrayView2<bool>>,
    ) -> Result<Array2<f64>> {
        let seq_len = query.shape()[0];
        let _batch_size = 1; // Simplified for single sequence

        // Linear projections
        let q = query.dot(&self.w_q);
        let k = key.dot(&self.w_k);
        let v = value.dot(&self.w_v);

        // Reshape for multi-head attention [seq_len, d_model] -> [seq_len, num_heads, d_k]
        let mut q_heads = Array3::zeros((seq_len, self.num_heads, self.d_k));
        let mut k_heads = Array3::zeros((seq_len, self.num_heads, self.d_k));
        let mut v_heads = Array3::zeros((seq_len, self.num_heads, self.d_k));

        for i in 0..seq_len {
            for h in 0..self.num_heads {
                let start = h * self.d_k;
                let _end = start + self.d_k;

                for j in 0..self.d_k {
                    q_heads[[i, h, j]] = q[[i, start + j]];
                    k_heads[[i, h, j]] = k[[i, start + j]];
                    v_heads[[i, h, j]] = v[[i, start + j]];
                }
            }
        }

        // Apply scaled dot-product attention for each head
        let mut attention_outputs = Array3::zeros((seq_len, self.num_heads, self.d_k));

        for h in 0..self.num_heads {
            let q_h = q_heads.slice(s![.., h, ..]);
            let k_h = k_heads.slice(s![.., h, ..]);
            let v_h = v_heads.slice(s![.., h, ..]);

            // Compute attention scores: Q * K^T / sqrt(d_k)
            let scores = q_h.dot(&k_h.t()) / (self.d_k as f64).sqrt();

            // Apply mask if provided
            let mut masked_scores = scores;
            if let Some(mask) = mask {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if mask[[i, j]] {
                            masked_scores[[i, j]] = f64::NEG_INFINITY;
                        }
                    }
                }
            }

            // Apply softmax
            let mut attention_weights = Array2::zeros((seq_len, seq_len));
            for i in 0..seq_len {
                let row = masked_scores.row(i);
                let max_val = row.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                let exp_sum: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();

                for j in 0..seq_len {
                    attention_weights[[i, j]] = (masked_scores[[i, j]] - max_val).exp() / exp_sum;
                }
            }

            // Apply dropout to attention weights
            let attention_weights_dropped = self.dropout.forward(attention_weights.view());

            // Apply attention to values
            let attended = attention_weights_dropped.dot(&v_h);

            // Store result for this head
            for i in 0..seq_len {
                for j in 0..self.d_k {
                    attention_outputs[[i, h, j]] = attended[[i, j]];
                }
            }
        }

        // Concatenate heads and reshape back to [seq_len, d_model]
        let mut concatenated = Array2::zeros((seq_len, self.d_model));
        for i in 0..seq_len {
            for h in 0..self.num_heads {
                let start = h * self.d_k;
                for j in 0..self.d_k {
                    concatenated[[i, start + j]] = attention_outputs[[i, h, j]];
                }
            }
        }

        // Final output projection
        Ok(concatenated.dot(&self.w_o))
    }

    /// Set training mode for dropout
    pub fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        let x = 0.5;

        // Test that activations produce reasonable outputs
        assert!(ActivationFunction::Sigmoid.apply(x) > 0.0);
        assert!(ActivationFunction::Sigmoid.apply(x) < 1.0);
        assert!(ActivationFunction::Tanh.apply(x) > -1.0);
        assert!(ActivationFunction::Tanh.apply(x) < 1.0);
        assert_eq!(ActivationFunction::ReLU.apply(-1.0), 0.0);
        assert_eq!(ActivationFunction::ReLU.apply(1.0), 1.0);
    }

    #[test]
    fn test_lstm_cell() {
        let lstm = LSTMCell::new(10, 20);
        let input = Array1::ones(10);
        let h_prev = Array1::zeros(20);
        let c_prev = Array1::zeros(20);

        let (h_new, c_new) = lstm
            .forward(input.view(), h_prev.view(), c_prev.view())
            .unwrap();

        assert_eq!(h_new.len(), 20);
        assert_eq!(c_new.len(), 20);
    }

    #[test]
    fn test_conv1d() {
        let conv = Conv1D::new(5, 10, 3, ActivationFunction::ReLU);
        let input = Array2::ones((8, 5)); // Sequence length 8, 5 channels

        let output = conv.forward(input.view()).unwrap();
        assert_eq!(output.shape(), &[6, 10]); // (8-3+1, 10)
    }

    #[test]
    fn test_bilstm() {
        let bilstm = BiLSTM::new(10, 20, 2);
        let input = Array2::ones((5, 10)); // 5 timesteps, 10 features

        let output = bilstm.forward(input.view()).unwrap();
        assert_eq!(output.shape(), &[5, 40]); // Bidirectional doubles output size
    }

    #[test]
    fn test_gru_cell() {
        let gru = GRUCell::new(10, 20);
        let input = Array1::ones(10);
        let h_prev = Array1::zeros(20);

        let h_new = gru.forward(input.view(), h_prev.view()).unwrap();

        assert_eq!(h_new.len(), 20);
        // Check that output is not all zeros (some processing happened)
        assert!(h_new.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_self_attention() {
        let attention = SelfAttention::new(8, 0.1);
        let input = Array2::ones((4, 8)); // 4 tokens, 8 dimensions

        let output = attention.forward(input.view(), None).unwrap();
        assert_eq!(output.shape(), &[4, 8]);
    }

    #[test]
    fn test_cross_attention() {
        let attention = CrossAttention::new(8);
        let query = Array2::ones((3, 8));
        let key = Array2::ones((5, 8));
        let value = Array2::ones((5, 8));

        let output = attention
            .forward(query.view(), key.view(), value.view(), None)
            .unwrap();
        assert_eq!(output.shape(), &[3, 8]);
    }

    #[test]
    fn test_residual_block() {
        let block = ResidualBlock1D::new(4, 8, 3);
        let input = Array2::ones((10, 4)); // 10 sequence length, 4 channels

        let output = block.forward(input.view()).unwrap();
        // Two conv layers with kernel_size=3 reduce sequence: 10 -> 8 -> 6
        assert_eq!(output.shape(), &[6, 8]); // Correct convolution output shape
    }

    #[test]
    fn test_multi_scale_cnn() {
        let cnn = MultiScaleCNN::new(
            5,             // input channels
            10,            // filters per scale
            vec![2, 3, 4], // kernel sizes
            30,            // output size
        );
        let input = Array2::ones((8, 5)); // 8 sequence length, 5 channels

        let output = cnn.forward(input.view()).unwrap();
        assert_eq!(output.len(), 30);
    }

    #[test]
    fn test_positionwise_feedforward() {
        let ff = PositionwiseFeedForward::new(8, 16, 0.1);
        let input = Array2::ones((4, 8)); // 4 tokens, 8 dimensions

        let output = ff.forward(input.view());
        assert_eq!(output.shape(), &[4, 8]);
    }
}
