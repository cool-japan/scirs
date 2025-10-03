//! LSTM Network Components for Time Series Forecasting
//!
//! This module provides Long Short-Term Memory (LSTM) network implementations
//! for time series forecasting, including LSTM cells, states, and multi-layer networks.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::config::ActivationFunction;
use crate::error::{Result, TimeSeriesError};

/// LSTM cell state and hidden state
#[derive(Debug, Clone)]
pub struct LSTMState<F: Float> {
    /// Hidden state
    pub hidden: Array1<F>,
    /// Cell state
    pub cell: Array1<F>,
}

/// LSTM cell implementation
#[derive(Debug)]
pub struct LSTMCell<F: Float + Debug> {
    /// Input size
    #[allow(dead_code)]
    input_size: usize,
    /// Hidden size
    #[allow(dead_code)]
    hidden_size: usize,
    /// Forget gate weights
    #[allow(dead_code)]
    w_forget: Array2<F>,
    /// Input gate weights
    #[allow(dead_code)]
    w_input: Array2<F>,
    /// Candidate gate weights
    #[allow(dead_code)]
    w_candidate: Array2<F>,
    /// Output gate weights
    #[allow(dead_code)]
    w_output: Array2<F>,
    /// Bias terms
    #[allow(dead_code)]
    bias: Array1<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive> LSTMCell<F> {
    /// Create new LSTM cell with random initialization
    pub fn new(_input_size: usize, hiddensize: usize) -> Self {
        let total_input_size = _input_size + hiddensize;

        // Initialize weights with Xavier/Glorot initialization
        let scale = F::from(2.0).unwrap() / F::from(total_input_size).unwrap();
        let std_dev = scale.sqrt();

        Self {
            input_size: _input_size,
            hidden_size: hiddensize,
            w_forget: Self::random_matrix(hiddensize, total_input_size, std_dev),
            w_input: Self::random_matrix(hiddensize, total_input_size, std_dev),
            w_candidate: Self::random_matrix(hiddensize, total_input_size, std_dev),
            w_output: Self::random_matrix(hiddensize, total_input_size, std_dev),
            bias: Array1::zeros(4 * hiddensize), // Bias for all gates
        }
    }

    /// Initialize random matrix with given standard deviation
    pub fn random_matrix(_rows: usize, cols: usize, stddev: F) -> Array2<F> {
        let mut matrix = Array2::zeros((_rows, cols));

        // Simple pseudo-random initialization (for production, use proper RNG)
        let mut seed: u32 = 12345;
        for i in 0.._rows {
            for j in 0..cols {
                // Linear congruential generator
                seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
                let rand_val = F::from(seed as f64 / 2147483647.0).unwrap();
                let normalized = (rand_val - F::from(0.5).unwrap()) * F::from(2.0).unwrap();
                matrix[[i, j]] = normalized * stddev;
            }
        }

        matrix
    }

    /// Forward pass through LSTM cell
    pub fn forward(&self, input: &Array1<F>, prevstate: &LSTMState<F>) -> Result<LSTMState<F>> {
        if input.len() != self.input_size {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.input_size,
                actual: input.len(),
            });
        }

        if prevstate.hidden.len() != self.hidden_size || prevstate.cell.len() != self.hidden_size {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.hidden_size,
                actual: prevstate.hidden.len(),
            });
        }

        // Concatenate input and previous hidden _state
        let mut combined_input = Array1::zeros(self.input_size + self.hidden_size);
        for (i, &val) in input.iter().enumerate() {
            combined_input[i] = val;
        }
        for (i, &val) in prevstate.hidden.iter().enumerate() {
            combined_input[self.input_size + i] = val;
        }

        // Compute gate values
        let forget_gate = self.compute_gate(&self.w_forget, &combined_input, 0);
        let input_gate = self.compute_gate(&self.w_input, &combined_input, self.hidden_size);
        let candidate_gate =
            self.compute_gate(&self.w_candidate, &combined_input, 2 * self.hidden_size);
        let output_gate = self.compute_gate(&self.w_output, &combined_input, 3 * self.hidden_size);

        // Apply activations
        let forget_activated = forget_gate.mapv(|x| ActivationFunction::Sigmoid.apply(x));
        let input_activated = input_gate.mapv(|x| ActivationFunction::Sigmoid.apply(x));
        let candidate_activated = candidate_gate.mapv(|x| ActivationFunction::Tanh.apply(x));
        let output_activated = output_gate.mapv(|x| ActivationFunction::Sigmoid.apply(x));

        // Update cell _state
        let mut new_cell = Array1::zeros(self.hidden_size);
        for i in 0..self.hidden_size {
            new_cell[i] = forget_activated[i] * prevstate.cell[i]
                + input_activated[i] * candidate_activated[i];
        }

        // Update hidden _state
        let cell_tanh = new_cell.mapv(|x| x.tanh());
        let mut new_hidden = Array1::zeros(self.hidden_size);
        for i in 0..self.hidden_size {
            new_hidden[i] = output_activated[i] * cell_tanh[i];
        }

        Ok(LSTMState {
            hidden: new_hidden,
            cell: new_cell,
        })
    }

    /// Compute gate output (linear transformation)
    fn compute_gate(
        &self,
        weights: &Array2<F>,
        input: &Array1<F>,
        bias_offset: usize,
    ) -> Array1<F> {
        let mut output = Array1::zeros(self.hidden_size);

        for i in 0..self.hidden_size {
            let mut sum = self.bias[bias_offset + i];
            for j in 0..input.len() {
                sum = sum + weights[[i, j]] * input[j];
            }
            output[i] = sum;
        }

        output
    }

    /// Initialize zero state
    pub fn init_state(&self) -> LSTMState<F> {
        LSTMState {
            hidden: Array1::zeros(self.hidden_size),
            cell: Array1::zeros(self.hidden_size),
        }
    }
}

/// Multi-layer LSTM network
#[derive(Debug)]
pub struct LSTMNetwork<F: Float + Debug> {
    /// LSTM layers
    #[allow(dead_code)]
    layers: Vec<LSTMCell<F>>,
    /// Output projection layer
    #[allow(dead_code)]
    output_layer: Array2<F>,
    /// Output bias
    #[allow(dead_code)]
    output_bias: Array1<F>,
    /// Dropout probability
    #[allow(dead_code)]
    dropout_prob: F,
}

impl<F: Float + Debug + Clone + FromPrimitive> LSTMNetwork<F> {
    /// Create new multi-layer LSTM network
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        dropout_prob: F,
    ) -> Self {
        let mut layers = Vec::new();

        // First layer
        if !hidden_sizes.is_empty() {
            layers.push(LSTMCell::new(input_size, hidden_sizes[0]));

            // Additional layers
            for i in 1..hidden_sizes.len() {
                layers.push(LSTMCell::new(hidden_sizes[i - 1], hidden_sizes[i]));
            }
        }

        let final_hidden_size = hidden_sizes.last().copied().unwrap_or(input_size);

        // Output layer initialization
        let output_scale = F::from(2.0).unwrap() / F::from(final_hidden_size).unwrap();
        let output_std = output_scale.sqrt();
        let output_layer = LSTMCell::random_matrix(output_size, final_hidden_size, output_std);

        Self {
            layers,
            output_layer,
            output_bias: Array1::zeros(output_size),
            dropout_prob,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, inputsequence: &Array2<F>) -> Result<Array2<F>> {
        let (seqlen, _input_size) = inputsequence.dim();

        if self.layers.is_empty() {
            return Err(TimeSeriesError::InvalidModel(
                "No LSTM layers defined".to_string(),
            ));
        }

        let output_size = self.output_layer.nrows();
        let mut outputs = Array2::zeros((seqlen, output_size));

        // Initialize states for all layers
        let mut states: Vec<LSTMState<F>> =
            self.layers.iter().map(|layer| layer.init_state()).collect();

        // Process each time step
        for t in 0..seqlen {
            let mut layer_input = inputsequence.row(t).to_owned();

            // Forward through LSTM layers
            for (i, layer) in self.layers.iter().enumerate() {
                let new_state = layer.forward(&layer_input, &states[i])?;
                layer_input = new_state.hidden.clone();
                states[i] = new_state;
            }

            // Apply dropout (simplified - just scaling)
            if self.dropout_prob > F::zero() {
                let keep_prob = F::one() - self.dropout_prob;
                layer_input = layer_input.mapv(|x| x * keep_prob);
            }

            // Output projection
            let output = self.compute_output(&layer_input);
            for (j, &val) in output.iter().enumerate() {
                outputs[[t, j]] = val;
            }
        }

        Ok(outputs)
    }

    /// Compute final output projection
    fn compute_output(&self, hidden: &Array1<F>) -> Array1<F> {
        let mut output = self.output_bias.clone();

        for i in 0..self.output_layer.nrows() {
            for j in 0..self.output_layer.ncols() {
                output[i] = output[i] + self.output_layer[[i, j]] * hidden[j];
            }
        }

        output
    }

    /// Generate forecast for multiple steps
    pub fn forecast(&self, input_sequence: &Array2<F>, forecaststeps: usize) -> Result<Array1<F>> {
        let (seqlen, _) = input_sequence.dim();

        // Get the last hidden states from input _sequence
        let _ = self.forward(input_sequence)?;

        // Initialize states for forecasting
        let mut states: Vec<LSTMState<F>> =
            self.layers.iter().map(|layer| layer.init_state()).collect();

        // Re-run forward pass to get final states
        for t in 0..seqlen {
            let mut layer_input = input_sequence.row(t).to_owned();
            for (i, layer) in self.layers.iter().enumerate() {
                let new_state = layer.forward(&layer_input, &states[i])?;
                layer_input = new_state.hidden.clone();
                states[i] = new_state;
            }
        }

        let mut forecasts = Array1::zeros(forecaststeps);
        let mut last_output = input_sequence.row(seqlen - 1).to_owned();

        // Generate forecasts step by step
        for step in 0..forecaststeps {
            let mut layer_input = last_output.clone();

            // Forward through LSTM layers
            for (i, layer) in self.layers.iter().enumerate() {
                let new_state = layer.forward(&layer_input, &states[i])?;
                layer_input = new_state.hidden.clone();
                states[i] = new_state;
            }

            // Compute output
            let output = self.compute_output(&layer_input);
            forecasts[step] = output[0]; // Assuming single output for forecasting

            // Use forecast as input for next step (assuming univariate)
            if last_output.len() == 1 {
                last_output[0] = output[0];
            } else {
                // For multivariate, use the forecast as the first feature
                last_output[0] = output[0];
            }
        }

        Ok(forecasts)
    }
}
