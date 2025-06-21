use ndarray::{s, Array, Array1, Array2, Array3};
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};
use std::f32;
use std::f32::consts::PI;

// LSTM Cell implementation (similar to previous examples)
#[derive(Debug, Serialize, Deserialize)]
struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    batch_size: usize,

    // Parameters
    w_ii: Array2<f32>, // Input to input gate
    w_hi: Array2<f32>, // Hidden to input gate
    b_i: Array1<f32>,  // Input gate bias

    w_if: Array2<f32>, // Input to forget gate
    w_hf: Array2<f32>, // Hidden to forget gate
    b_f: Array1<f32>,  // Forget gate bias

    w_ig: Array2<f32>, // Input to cell gate
    w_hg: Array2<f32>, // Hidden to cell gate
    b_g: Array1<f32>,  // Cell gate bias

    w_io: Array2<f32>, // Input to output gate
    w_ho: Array2<f32>, // Hidden to output gate
    b_o: Array1<f32>,  // Output gate bias

    // Hidden and cell states
    h_t: Option<Array2<f32>>, // Current hidden state [batch_size, hidden_size]
    c_t: Option<Array2<f32>>, // Current cell state [batch_size, hidden_size]
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize, batch_size: usize) -> Self {
        // Xavier/Glorot initialization
        let bound = (6.0 / (input_size + hidden_size) as f32).sqrt();

        // Input gate weights
        let uniform = Uniform::new(-bound, bound).unwrap();
        let mut rng = rand::rng();
        let w_ii = Array::from_shape_fn((hidden_size, input_size), |_| uniform.sample(&mut rng));
        let w_hi = Array::from_shape_fn((hidden_size, hidden_size), |_| uniform.sample(&mut rng));
        let b_i = Array1::zeros(hidden_size);

        // Forget gate weights (initialize forget gate bias to 1 to avoid vanishing gradients early in training)
        let w_if = Array::from_shape_fn((hidden_size, input_size), |_| uniform.sample(&mut rng));
        let w_hf = Array::from_shape_fn((hidden_size, hidden_size), |_| uniform.sample(&mut rng));
        let b_f = Array1::ones(hidden_size);

        // Cell gate weights
        let w_ig = Array::from_shape_fn((hidden_size, input_size), |_| uniform.sample(&mut rng));
        let w_hg = Array::from_shape_fn((hidden_size, hidden_size), |_| uniform.sample(&mut rng));
        let b_g = Array1::zeros(hidden_size);

        // Output gate weights
        let w_io = Array::from_shape_fn((hidden_size, input_size), |_| uniform.sample(&mut rng));
        let w_ho = Array::from_shape_fn((hidden_size, hidden_size), |_| uniform.sample(&mut rng));
        let b_o = Array1::zeros(hidden_size);

        LSTMCell {
            input_size,
            hidden_size,
            batch_size,
            w_ii,
            w_hi,
            b_i,
            w_if,
            w_hf,
            b_f,
            w_ig,
            w_hg,
            b_g,
            w_io,
            w_ho,
            b_o,
            h_t: None,
            c_t: None,
        }
    }

    fn reset_state(&mut self) {
        self.h_t = None;
        self.c_t = None;
    }

    fn forward(&mut self, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let batch_size = x.shape()[0];

        // Initialize states if None
        if self.h_t.is_none() {
            self.h_t = Some(Array2::zeros((batch_size, self.hidden_size)));
        }

        if self.c_t.is_none() {
            self.c_t = Some(Array2::zeros((batch_size, self.hidden_size)));
        }

        // Get previous states
        let h_prev = self.h_t.as_ref().unwrap();
        let c_prev = self.c_t.as_ref().unwrap();

        // Input gate: i_t = sigmoid(W_ii * x_t + W_hi * h_prev + b_i)
        let i_t = Self::sigmoid(&(x.dot(&self.w_ii.t()) + h_prev.dot(&self.w_hi.t()) + &self.b_i));

        // Forget gate: f_t = sigmoid(W_if * x_t + W_hf * h_prev + b_f)
        let f_t = Self::sigmoid(&(x.dot(&self.w_if.t()) + h_prev.dot(&self.w_hf.t()) + &self.b_f));

        // Cell gate: g_t = tanh(W_ig * x_t + W_hg * h_prev + b_g)
        let g_t = Self::tanh(&(x.dot(&self.w_ig.t()) + h_prev.dot(&self.w_hg.t()) + &self.b_g));

        // Output gate: o_t = sigmoid(W_io * x_t + W_ho * h_prev + b_o)
        let o_t = Self::sigmoid(&(x.dot(&self.w_io.t()) + h_prev.dot(&self.w_ho.t()) + &self.b_o));

        // Cell state: c_t = f_t * c_prev + i_t * g_t
        let c_t = &f_t * c_prev + &i_t * &g_t;

        // Hidden state: h_t = o_t * tanh(c_t)
        let h_t = &o_t * &Self::tanh(&c_t);

        // Update states
        self.h_t = Some(h_t.clone());
        self.c_t = Some(c_t.clone());

        (h_t, c_t)
    }

    // Activation functions
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn tanh(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.tanh())
    }
}

// Attention mechanism for time series
#[derive(Debug, Serialize, Deserialize)]
struct TimeSeriesAttention {
    hidden_size: usize,

    // Parameters
    w_query: Array2<f32>,
    w_key: Array2<f32>,
    w_value: Array2<f32>,
    v_attention: Array1<f32>,
}

impl TimeSeriesAttention {
    fn new(hidden_size: usize) -> Self {
        // Xavier/Glorot initialization
        let bound = (6.0 / (hidden_size + hidden_size) as f32).sqrt();

        let uniform = Uniform::new(-bound, bound).unwrap();
        let mut rng = rand::rng();
        let w_query =
            Array::from_shape_fn((hidden_size, hidden_size), |_| uniform.sample(&mut rng));
        let w_key = Array::from_shape_fn((hidden_size, hidden_size), |_| uniform.sample(&mut rng));
        let w_value =
            Array::from_shape_fn((hidden_size, hidden_size), |_| uniform.sample(&mut rng));
        let v_attention = Array::from_shape_fn(hidden_size, |_| uniform.sample(&mut rng));

        TimeSeriesAttention {
            hidden_size,
            w_query,
            w_key,
            w_value,
            v_attention,
        }
    }

    fn forward(
        &self,
        query: &Array2<f32>,
        keys: &Array3<f32>,
        values: &Array3<f32>,
    ) -> Array2<f32> {
        // query: [batch_size, hidden_size] - Current hidden state
        // keys: [batch_size, seq_len, hidden_size] - All hidden states from the sequence
        // values: [batch_size, seq_len, hidden_size] - All hidden states from the sequence (usually same as keys)

        let batch_size = query.shape()[0];
        let seq_len = keys.shape()[1];

        // Project query, keys, and values
        let q_proj = query.dot(&self.w_query); // [batch_size, hidden_size]

        // Prepare projected keys and values
        let mut k_proj = Array3::<f32>::zeros((batch_size, seq_len, self.hidden_size));
        let mut v_proj = Array3::<f32>::zeros((batch_size, seq_len, self.hidden_size));

        for b in 0..batch_size {
            for t in 0..seq_len {
                let key = keys.slice(s![b, t, ..]).to_owned();
                let value = values.slice(s![b, t, ..]).to_owned();

                let k_p = key.dot(&self.w_key);
                let v_p = value.dot(&self.w_value);

                for h in 0..self.hidden_size {
                    k_proj[[b, t, h]] = k_p[h];
                    v_proj[[b, t, h]] = v_p[h];
                }
            }
        }

        // Calculate attention scores
        let mut scores = Array2::<f32>::zeros((batch_size, seq_len));

        for b in 0..batch_size {
            let q = q_proj.slice(s![b, ..]).to_owned();

            for t in 0..seq_len {
                let k = k_proj.slice(s![b, t, ..]).to_owned();

                // Calculate similarity
                let mut similarity = 0.0;
                for h in 0..self.hidden_size {
                    similarity += q[h] * k[h];
                }

                scores[[b, t]] = similarity;
            }
        }

        // Apply softmax
        let attention_weights = Self::softmax_by_row(&scores);

        // Calculate weighted sum
        let mut context = Array2::<f32>::zeros((batch_size, self.hidden_size));

        for b in 0..batch_size {
            for t in 0..seq_len {
                let weight = attention_weights[[b, t]];

                for h in 0..self.hidden_size {
                    context[[b, h]] += weight * v_proj[[b, t, h]];
                }
            }
        }

        context
    }

    // Softmax applied to each row separately
    fn softmax_by_row(x: &Array2<f32>) -> Array2<f32> {
        let mut result = Array2::<f32>::zeros(x.raw_dim());

        for (i, row) in x.outer_iter().enumerate() {
            // Find max value for numerical stability
            let max_val = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Calculate exp and sum
            let mut sum = 0.0;
            let mut exp_vals = vec![0.0; row.len()];

            for (j, &val) in row.iter().enumerate() {
                let exp_val = (val - max_val).exp();
                exp_vals[j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for (j, exp_val) in exp_vals.iter().enumerate() {
                result[[i, j]] = exp_val / sum;
            }
        }

        result
    }
}

// LSTM Encoder for time series
#[derive(Debug, Serialize, Deserialize)]
struct LSTMEncoder {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    batch_size: usize,

    // LSTM cells for each layer
    lstm_cells: Vec<LSTMCell>,
}

impl LSTMEncoder {
    fn new(input_size: usize, hidden_size: usize, num_layers: usize, batch_size: usize) -> Self {
        let mut lstm_cells = Vec::with_capacity(num_layers);

        // Create LSTM cells for each layer
        for layer in 0..num_layers {
            let layer_input_size = if layer == 0 { input_size } else { hidden_size };
            lstm_cells.push(LSTMCell::new(layer_input_size, hidden_size, batch_size));
        }

        LSTMEncoder {
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            lstm_cells,
        }
    }

    fn forward(&mut self, x: &Array3<f32>) -> (Array3<f32>, Array2<f32>, Array2<f32>) {
        // x: [batch_size, seq_len, input_size]
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Reset all cell states
        for cell in &mut self.lstm_cells {
            cell.reset_state();
        }

        // Store all hidden states for attention
        let mut all_hidden_states = Array3::<f32>::zeros((batch_size, seq_len, self.hidden_size));

        // Process each time step
        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]).to_owned();

            // Process through each layer
            let mut layer_input = x_t;

            for (layer_idx, cell) in self.lstm_cells.iter_mut().enumerate() {
                let (h_t, _) = cell.forward(&layer_input);

                // Output of this layer becomes input to the next
                layer_input = h_t.clone();

                // Store hidden state if this is the last layer
                if layer_idx == self.num_layers - 1 {
                    for b in 0..batch_size {
                        for h in 0..self.hidden_size {
                            all_hidden_states[[b, t, h]] = h_t[[b, h]];
                        }
                    }
                }
            }
        }

        // Get final hidden and cell states from the last layer
        let h_n = self.lstm_cells.last().unwrap().h_t.clone().unwrap();
        let c_n = self.lstm_cells.last().unwrap().c_t.clone().unwrap();

        (all_hidden_states, h_n, c_n)
    }
}

// Attention-based forecasting model
#[derive(Debug, Serialize, Deserialize)]
struct TimeSeriesForecaster {
    input_size: usize,       // Number of features in input
    hidden_size: usize,      // Size of LSTM hidden state
    output_size: usize,      // Number of outputs (usually 1 for univariate forecasting)
    forecast_horizon: usize, // Number of time steps to forecast

    // Encoder
    encoder: LSTMEncoder,

    // Attention mechanism
    attention: TimeSeriesAttention,

    // Decoder initial state projection
    w_decoder_init: Array2<f32>,
    b_decoder_init: Array1<f32>,

    // Output projection
    w_out: Array2<f32>,
    b_out: Array1<f32>,

    // Decoder LSTM cell
    decoder_cell: LSTMCell,
}

impl TimeSeriesForecaster {
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        num_encoder_layers: usize,
        forecast_horizon: usize,
        batch_size: usize,
    ) -> Self {
        // Create encoder
        let encoder = LSTMEncoder::new(input_size, hidden_size, num_encoder_layers, batch_size);

        // Create attention mechanism
        let attention = TimeSeriesAttention::new(hidden_size);

        // Decoder initial state projection
        let bound_init = (6.0 / (hidden_size + hidden_size) as f32).sqrt();
        let uniform_init = Uniform::new(-bound_init, bound_init).unwrap();
        let mut rng = rand::rng();
        let w_decoder_init = Array::from_shape_fn((hidden_size, hidden_size), |_| {
            uniform_init.sample(&mut rng)
        });
        let b_decoder_init = Array1::zeros(hidden_size);

        // Output projection
        let bound_out = (6.0 / (hidden_size + output_size) as f32).sqrt();
        let uniform_out = Uniform::new(-bound_out, bound_out).unwrap();
        let w_out =
            Array::from_shape_fn((output_size, hidden_size), |_| uniform_out.sample(&mut rng));
        let b_out = Array1::zeros(output_size);

        // Decoder LSTM cell
        let decoder_cell = LSTMCell::new(output_size, hidden_size, batch_size);

        TimeSeriesForecaster {
            input_size,
            hidden_size,
            output_size,
            forecast_horizon,
            encoder,
            attention,
            w_decoder_init,
            b_decoder_init,
            w_out,
            b_out,
            decoder_cell,
        }
    }

    fn forward(&mut self, x: &Array3<f32>, prev_y: Option<&Array2<f32>>) -> Array3<f32> {
        // x: [batch_size, seq_len, input_size] - Input time series
        // prev_y: Optional[batch_size, output_size] - Previous output (for autoregressive prediction)

        let batch_size = x.shape()[0];

        // Encode input sequence
        let (encoder_states, h_n, c_n) = self.encoder.forward(x);

        // Initialize decoder state
        let decoder_h = h_n.dot(&self.w_decoder_init) + &self.b_decoder_init;
        let decoder_c = c_n.clone();

        self.decoder_cell.h_t = Some(decoder_h);
        self.decoder_cell.c_t = Some(decoder_c);

        // Initialize forecast outputs
        let mut forecasts =
            Array3::<f32>::zeros((batch_size, self.forecast_horizon, self.output_size));

        // Initial input to decoder
        let mut decoder_input = match prev_y {
            Some(y) => y.clone(),
            None => Array2::<f32>::zeros((batch_size, self.output_size)),
        };

        // Generate forecast step by step
        for t in 0..self.forecast_horizon {
            // Apply attention
            let context = self.attention.forward(
                self.decoder_cell.h_t.as_ref().unwrap(),
                &encoder_states,
                &encoder_states,
            );

            // Concatenate context with decoder input
            // In a real implementation, we would concatenate them
            // For simplicity, we'll just use the decoder input and context separately

            // Run decoder LSTM cell
            let (h_t, _) = self.decoder_cell.forward(&decoder_input);

            // Combine hidden state with context for output
            let mut combined = Array2::<f32>::zeros((batch_size, self.hidden_size));
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    combined[[b, h]] = h_t[[b, h]] + context[[b, h]]; // Simple addition for combination
                }
            }

            // Project to output
            let output = combined.dot(&self.w_out.t()) + &self.b_out;

            // Store forecast
            for b in 0..batch_size {
                for o in 0..self.output_size {
                    forecasts[[b, t, o]] = output[[b, o]];
                }
            }

            // Next decoder input is current output (autoregressive)
            decoder_input = output;
        }

        forecasts
    }
}

// Normalization helper
struct MinMaxScaler {
    min_vals: Array1<f32>,
    max_vals: Array1<f32>,
}

impl MinMaxScaler {
    #[allow(dead_code)]
    fn new(min_vals: Array1<f32>, max_vals: Array1<f32>) -> Self {
        MinMaxScaler { min_vals, max_vals }
    }

    fn fit(data: &Array2<f32>) -> Self {
        let features = data.shape()[1];
        let mut min_vals = Array1::<f32>::zeros(features);
        let mut max_vals = Array1::<f32>::zeros(features);

        for f in 0..features {
            let column = data.slice(s![.., f]);

            let min_val = column.fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = column.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            min_vals[f] = min_val;
            max_vals[f] = max_val;
        }

        MinMaxScaler { min_vals, max_vals }
    }

    fn transform(&self, data: &Array2<f32>) -> Array2<f32> {
        let (rows, cols) = data.dim();
        let mut result = Array2::<f32>::zeros((rows, cols));

        for r in 0..rows {
            for c in 0..cols {
                let min_val = self.min_vals[c];
                let max_val = self.max_vals[c];

                if max_val > min_val {
                    result[[r, c]] = (data[[r, c]] - min_val) / (max_val - min_val);
                } else {
                    result[[r, c]] = 0.0;
                }
            }
        }

        result
    }

    fn inverse_transform(&self, data: &Array2<f32>) -> Array2<f32> {
        let (rows, cols) = data.dim();
        let mut result = Array2::<f32>::zeros((rows, cols));

        for r in 0..rows {
            for c in 0..cols {
                let min_val = self.min_vals[c];
                let max_val = self.max_vals[c];

                result[[r, c]] = data[[r, c]] * (max_val - min_val) + min_val;
            }
        }

        result
    }
}

// Create sliding window dataset from time series
fn create_sliding_window_dataset(
    data: &Array2<f32>,
    window_size: usize,
    forecast_horizon: usize,
) -> (Array3<f32>, Array3<f32>) {
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    let n_windows = n_samples - window_size - forecast_horizon + 1;

    let mut x = Array3::<f32>::zeros((n_windows, window_size, n_features));
    let mut y = Array3::<f32>::zeros((n_windows, forecast_horizon, n_features));

    for i in 0..n_windows {
        // Extract window
        for t in 0..window_size {
            for f in 0..n_features {
                x[[i, t, f]] = data[[i + t, f]];
            }
        }

        // Extract forecast horizon
        for t in 0..forecast_horizon {
            for f in 0..n_features {
                y[[i, t, f]] = data[[i + window_size + t, f]];
            }
        }
    }

    (x, y)
}

// Generate synthetic time series data
fn generate_synthetic_time_series(n_samples: usize, n_features: usize) -> Array2<f32> {
    let mut data = Array2::<f32>::zeros((n_samples, n_features));

    // Generate time range
    let time: Vec<f32> = (0..n_samples).map(|i| i as f32 / 100.0).collect();

    // Feature 1: Sine wave with noise
    let normal = Normal::new(0.0, 0.1).unwrap();
    let mut rng = rand::rng();
    for (i, &t) in time.iter().enumerate() {
        let sine_val = (t * 2.0 * PI).sin();
        let noise: f32 = normal.sample(&mut rng);
        data[[i, 0]] = sine_val + noise;
    }

    // If we have more features, add them
    if n_features > 1 {
        // Feature 2: Cosine wave with different frequency and noise
        let normal = Normal::new(0.0, 0.1).unwrap();
        let mut rng = rand::rng();
        for (i, &t) in time.iter().enumerate() {
            let cosine_val = (t * 1.5 * PI).cos();
            let noise: f32 = normal.sample(&mut rng);
            data[[i, 1]] = cosine_val + noise;
        }
    }

    // Add more features if needed (e.g., trend, seasonality, etc.)
    if n_features > 2 {
        // Feature 3: Linear trend
        let normal = Normal::new(0.0, 0.05).unwrap();
        let mut rng = rand::rng();
        for i in 0..n_samples {
            let trend = i as f32 / n_samples as f32;
            let noise: f32 = normal.sample(&mut rng);
            data[[i, 2]] = trend + noise;
        }
    }

    data
}

// Mean Absolute Error calculation
fn mean_absolute_error(y_true: &Array3<f32>, y_pred: &Array3<f32>) -> f32 {
    let batch_size = y_true.shape()[0];
    let forecast_horizon = y_true.shape()[1];
    let output_size = y_true.shape()[2];

    let mut total_error = 0.0;
    let mut count = 0;

    for b in 0..batch_size {
        for t in 0..forecast_horizon {
            for o in 0..output_size {
                total_error += (y_true[[b, t, o]] - y_pred[[b, t, o]]).abs();
                count += 1;
            }
        }
    }

    if count > 0 {
        total_error / count as f32
    } else {
        0.0
    }
}

// Train the forecaster (simplified training loop without actual parameter updates)
fn train_forecaster(
    model: &mut TimeSeriesForecaster,
    x_train: &Array3<f32>,
    y_train: &Array3<f32>,
    num_epochs: usize,
    batch_size: usize,
) {
    let n_samples = x_train.shape()[0];
    let n_batches = n_samples.div_ceil(batch_size);

    println!("Training forecaster model...");

    for epoch in 1..=num_epochs {
        let mut total_loss = 0.0;

        // Simple batching (no shuffling for simplicity)
        for batch in 0..n_batches {
            let start_idx = batch * batch_size;
            let end_idx = (start_idx + batch_size).min(n_samples);
            let actual_batch_size = end_idx - start_idx;

            if actual_batch_size == 0 {
                continue;
            }

            // Extract batch
            let mut x_batch =
                Array3::<f32>::zeros((actual_batch_size, x_train.shape()[1], x_train.shape()[2]));
            let mut y_batch =
                Array3::<f32>::zeros((actual_batch_size, y_train.shape()[1], y_train.shape()[2]));

            for i in 0..actual_batch_size {
                for t in 0..x_train.shape()[1] {
                    for f in 0..x_train.shape()[2] {
                        x_batch[[i, t, f]] = x_train[[start_idx + i, t, f]];
                    }
                }

                for t in 0..y_train.shape()[1] {
                    for f in 0..y_train.shape()[2] {
                        y_batch[[i, t, f]] = y_train[[start_idx + i, t, f]];
                    }
                }
            }

            // Forward pass
            let predictions = model.forward(&x_batch, None);

            // Calculate loss (MSE)
            let mut batch_loss = 0.0;
            let mut count = 0;

            for b in 0..actual_batch_size {
                for t in 0..model.forecast_horizon {
                    for o in 0..model.output_size {
                        let error = predictions[[b, t, o]] - y_batch[[b, t, o]];
                        batch_loss += error * error;
                        count += 1;
                    }
                }
            }

            let avg_batch_loss = if count > 0 {
                batch_loss / count as f32
            } else {
                0.0
            };
            total_loss += avg_batch_loss;

            // In a real implementation, we would now:
            // 1. Calculate gradients through backpropagation
            // 2. Update parameters using an optimizer
        }

        let avg_loss = total_loss / n_batches as f32;
        println!("Epoch {}/{} - Loss: {:.6}", epoch, num_epochs, avg_loss);
    }
}

// Evaluate the model
fn evaluate_forecaster(
    model: &mut TimeSeriesForecaster,
    x_test: &Array3<f32>,
    y_test: &Array3<f32>,
    scaler: &MinMaxScaler,
) {
    println!("\nEvaluating forecaster...");

    // Forward pass
    let predictions = model.forward(x_test, None);

    // Calculate metrics
    let mae = mean_absolute_error(y_test, &predictions);
    println!("Mean Absolute Error (scaled): {:.6}", mae);

    // Inverse transform to original scale
    let n_samples = predictions.shape()[0].min(5); // Show only up to 5 samples
    let forecast_horizon = predictions.shape()[1];
    let output_size = predictions.shape()[2];

    for sample_idx in 0..n_samples {
        println!("\nSample {}:", sample_idx + 1);

        // Extract actual forecasts and predictions
        let mut actual = Array2::<f32>::zeros((forecast_horizon, output_size));
        let mut predicted = Array2::<f32>::zeros((forecast_horizon, output_size));

        for t in 0..forecast_horizon {
            for o in 0..output_size {
                actual[[t, o]] = y_test[[sample_idx, t, o]];
                predicted[[t, o]] = predictions[[sample_idx, t, o]];
            }
        }

        // Inverse transform
        let actual_orig = scaler.inverse_transform(&actual);
        let pred_orig = scaler.inverse_transform(&predicted);

        println!("Time | Actual | Predicted");
        println!("--------------------------");
        for t in 0..forecast_horizon {
            println!(
                "{:4} | {:6.3} | {:6.3}",
                t + 1,
                actual_orig[[t, 0]],
                pred_orig[[t, 0]]
            );
        }
    }
}

// Forecast future time steps starting from the last observed values
fn forecast_future(
    model: &mut TimeSeriesForecaster,
    last_window: &Array2<f32>,
    num_steps: usize,
    scaler: &MinMaxScaler,
) {
    println!("\nForecasting future time steps...");

    let window_size = last_window.shape()[0];
    let n_features = last_window.shape()[1];

    // Prepare input with correct shape
    let mut x = Array3::<f32>::zeros((1, window_size, n_features));
    for t in 0..window_size {
        for f in 0..n_features {
            x[[0, t, f]] = last_window[[t, f]];
        }
    }

    // Initial forecast
    let mut all_forecasts = Array2::<f32>::zeros((num_steps, n_features));
    let mut forecast_window = x.clone();

    // Generate forecasts step by step
    for step in 0..num_steps {
        // Generate forecast for next chunk
        let forecast_chunk = model.forward(&forecast_window, None);

        // Store forecasts
        let forecast_horizon = model.forecast_horizon.min(num_steps - step);
        for t in 0..forecast_horizon {
            for f in 0..n_features {
                all_forecasts[[step + t, f]] = forecast_chunk[[0, t, f]];
            }
        }

        // If we need more forecasts, update the window and continue
        if step + forecast_horizon < num_steps {
            // Slide window forward
            for t in 0..window_size - forecast_horizon {
                for f in 0..n_features {
                    forecast_window[[0, t, f]] = forecast_window[[0, t + forecast_horizon, f]];
                }
            }

            // Add new forecasts to the end
            for t in 0..forecast_horizon {
                for f in 0..n_features {
                    forecast_window[[0, window_size - forecast_horizon + t, f]] =
                        forecast_chunk[[0, t, f]];
                }
            }
        } else {
            break;
        }
    }

    // Convert back to original scale
    let forecasts_orig = scaler.inverse_transform(&all_forecasts);

    println!("Future Forecasts:");
    println!("----------------");
    println!("Time | Value");
    for t in 0..num_steps {
        println!("{:4} | {:6.3}", t + 1, forecasts_orig[[t, 0]]);
    }
}

fn main() {
    println!("Time Series Forecasting with LSTM and Attention");
    println!("==============================================");

    // Generate synthetic time series data
    let n_samples = 500;
    let n_features = 1; // For simplicity, we'll use univariate time series

    println!("Generating synthetic time series data...");
    let data = generate_synthetic_time_series(n_samples, n_features);

    // Normalize data
    let scaler = MinMaxScaler::fit(&data);
    let scaled_data = scaler.transform(&data);

    // For time series, sometimes we may want to only normalize the feature values
    // but keep the time information unchanged, especially for interpretability
    // However, for this example we'll use fully normalized data for simplicity

    // Create sliding window dataset
    let window_size = 24; // Look back window
    let forecast_horizon = 8; // How many steps to forecast

    let (x, y) = create_sliding_window_dataset(&scaled_data, window_size, forecast_horizon);

    println!("Created dataset with {} samples", x.shape()[0]);
    println!("Input shape: {:?}", x.shape());
    println!("Output shape: {:?}", y.shape());

    // Split into train and test sets (80% train, 20% test)
    let train_size = (x.shape()[0] as f32 * 0.8) as usize;

    let x_train = x.slice(s![0..train_size, .., ..]).to_owned();
    let y_train = y.slice(s![0..train_size, .., ..]).to_owned();

    let x_test = x.slice(s![train_size.., .., ..]).to_owned();
    let y_test = y.slice(s![train_size.., .., ..]).to_owned();

    // Model parameters
    let hidden_size = 32;
    let num_encoder_layers = 1;
    let batch_size = x_train.shape()[0].min(16); // Use the whole dataset or max 16 samples per batch

    // Create model
    let mut model = TimeSeriesForecaster::new(
        n_features,
        hidden_size,
        n_features, // Output size (same as input for time series forecasting)
        num_encoder_layers,
        forecast_horizon,
        batch_size,
    );

    // Train model
    train_forecaster(&mut model, &x_train, &y_train, 50, batch_size);

    // Evaluate model
    evaluate_forecaster(&mut model, &x_test, &y_test, &scaler);

    // Forecast future values
    let last_window = scaled_data
        .slice(s![scaled_data.shape()[0] - window_size.., ..])
        .to_owned();
    forecast_future(&mut model, &last_window, 24, &scaler);

    println!("\nTime series forecasting model implementation completed!");
}
