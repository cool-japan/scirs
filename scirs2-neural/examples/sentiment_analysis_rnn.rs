use ndarray::{s, Array1, Array2, Array3, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32;

// Define a trait for recurrent classification models
trait RecurrentClassifier {
    fn forward(&mut self, x: &Array3<f32>, is_training: bool) -> Array2<f32>;
    fn backward(&mut self, x: &Array3<f32>, targets: &Array2<f32>) -> f32;
    fn update_params(&mut self, learning_rate: f32);
    fn reset_state(&mut self);
    fn predict(&mut self, x: &Array3<f32>) -> Array1<usize>;
}

// LSTM model for sequence classification
#[derive(Debug, Serialize, Deserialize)]
struct LSTMClassifier {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    batch_size: usize,

    // Parameters: LSTM Cell
    // Input gate
    w_ii: Array2<f32>,
    w_hi: Array2<f32>,
    b_i: Array1<f32>,

    // Forget gate
    w_if: Array2<f32>,
    w_hf: Array2<f32>,
    b_f: Array1<f32>,

    // Cell gate
    w_ig: Array2<f32>,
    w_hg: Array2<f32>,
    b_g: Array1<f32>,

    // Output gate
    w_io: Array2<f32>,
    w_ho: Array2<f32>,
    b_o: Array1<f32>,

    // Parameters: Classification Head
    w_out: Array2<f32>,
    b_out: Array1<f32>,

    // Gradients
    dw_ii: Option<Array2<f32>>,
    dw_hi: Option<Array2<f32>>,
    db_i: Option<Array1<f32>>,
    dw_if: Option<Array2<f32>>,
    dw_hf: Option<Array2<f32>>,
    db_f: Option<Array1<f32>>,
    dw_ig: Option<Array2<f32>>,
    dw_hg: Option<Array2<f32>>,
    db_g: Option<Array1<f32>>,
    dw_io: Option<Array2<f32>>,
    dw_ho: Option<Array2<f32>>,
    db_o: Option<Array1<f32>>,
    dw_out: Option<Array2<f32>>,
    db_out: Option<Array1<f32>>,

    // Hidden and cell states
    h_t: Option<Array2<f32>>,
    c_t: Option<Array2<f32>>,

    // Cache for backward pass
    inputs: Option<Array3<f32>>,
    input_gates: Option<Array3<f32>>,
    forget_gates: Option<Array3<f32>>,
    cell_gates: Option<Array3<f32>>,
    output_gates: Option<Array3<f32>>,
    cell_states: Option<Array3<f32>>,
    hidden_states: Option<Array3<f32>>,
    final_hidden: Option<Array2<f32>>,
    output: Option<Array2<f32>>,
}

impl LSTMClassifier {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, batch_size: usize) -> Self {
        // Xavier/Glorot initialization for weights
        let bound = (6.0 / (input_size + hidden_size) as f32).sqrt();

        // Create a random number generator
        let mut rng = rand::rng();

        // Input gate weights
        let mut w_ii = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hi = Array2::<f32>::zeros((hidden_size, hidden_size));
        for elem in w_ii.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_hi.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        let b_i = Array1::zeros(hidden_size);

        // Forget gate weights
        let mut w_if = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hf = Array2::<f32>::zeros((hidden_size, hidden_size));
        for elem in w_if.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_hf.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        let b_f = Array1::ones(hidden_size); // Initialize to 1s to avoid forgetting early in training

        // Cell gate weights
        let mut w_ig = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hg = Array2::<f32>::zeros((hidden_size, hidden_size));
        for elem in w_ig.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_hg.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        let b_g = Array1::zeros(hidden_size);

        // Output gate weights
        let mut w_io = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_ho = Array2::<f32>::zeros((hidden_size, hidden_size));
        for elem in w_io.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_ho.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        let b_o = Array1::zeros(hidden_size);

        // Output projection weights
        let output_bound = (6.0 / (hidden_size + output_size) as f32).sqrt();
        let mut w_out = Array2::<f32>::zeros((output_size, hidden_size));
        for elem in w_out.iter_mut() {
            *elem = rng.random_range(-output_bound..output_bound);
        }
        let b_out = Array1::zeros(output_size);

        LSTMClassifier {
            input_size,
            hidden_size,
            output_size,
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
            w_out,
            b_out,
            dw_ii: None,
            dw_hi: None,
            db_i: None,
            dw_if: None,
            dw_hf: None,
            db_f: None,
            dw_ig: None,
            dw_hg: None,
            db_g: None,
            dw_io: None,
            dw_ho: None,
            db_o: None,
            dw_out: None,
            db_out: None,
            h_t: None,
            c_t: None,
            inputs: None,
            input_gates: None,
            forget_gates: None,
            cell_gates: None,
            output_gates: None,
            cell_states: None,
            hidden_states: None,
            final_hidden: None,
            output: None,
        }
    }

    // Activation functions and derivatives
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn sigmoid_derivative(sigmoid_output: &Array2<f32>) -> Array2<f32> {
        sigmoid_output * &(1.0 - sigmoid_output)
    }

    fn tanh(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.tanh())
    }

    fn tanh_derivative(tanh_output: &Array2<f32>) -> Array2<f32> {
        1.0 - tanh_output * tanh_output
    }

    fn softmax(x: &Array2<f32>) -> Array2<f32> {
        let max_vals = x.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let exp_shifted = x
            .outer_iter()
            .zip(max_vals.iter())
            .map(|(row, &max_val)| row.mapv(|v| (v - max_val).exp()))
            .collect::<Vec<_>>();

        let exp_sum = exp_shifted.iter().map(|row| row.sum()).collect::<Vec<_>>();

        let result = exp_shifted
            .iter()
            .zip(exp_sum.iter())
            .map(|(row, &sum)| row.mapv(|v| v / sum))
            .collect::<Vec<_>>();

        let mut output = Array2::zeros(x.raw_dim());
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                output[[i, j]] = val;
            }
        }

        output
    }

    // Process sequence and classify
    fn process_sequence(
        &mut self,
        x: &Array3<f32>,
        is_training: bool,
    ) -> (Array2<f32>, Array2<f32>) {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Initialize states if None
        if self.h_t.is_none() {
            self.h_t = Some(Array2::zeros((batch_size, self.hidden_size)));
        }

        if self.c_t.is_none() {
            self.c_t = Some(Array2::zeros((batch_size, self.hidden_size)));
        }

        // Arrays to store values for each time step
        let mut all_hidden_states = Array3::zeros((batch_size, seq_len + 1, self.hidden_size));
        let mut all_cell_states = Array3::zeros((batch_size, seq_len + 1, self.hidden_size));
        let mut all_input_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_forget_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_cell_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_output_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));

        // Set initial hidden and cell states
        if let Some(h_t) = &self.h_t {
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    all_hidden_states[[b, 0, h]] = h_t[[b, h]];
                }
            }
        }

        if let Some(c_t) = &self.c_t {
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    all_cell_states[[b, 0, h]] = c_t[[b, h]];
                }
            }
        }

        // Process sequence
        for t in 0..seq_len {
            // Get input at time t [batch_size, input_size]
            let x_t = x
                .slice(s![.., t, ..])
                .to_owned()
                .into_shape_with_order((batch_size, self.input_size))
                .unwrap();

            // Get previous hidden and cell states
            let h_prev = all_hidden_states.slice(s![.., t, ..]).to_owned();
            let c_prev = all_cell_states.slice(s![.., t, ..]).to_owned();

            // Input gate: i_t = sigmoid(W_ii * x_t + W_hi * h_prev + b_i)
            let i_t =
                Self::sigmoid(&(x_t.dot(&self.w_ii.t()) + h_prev.dot(&self.w_hi.t()) + &self.b_i));

            // Forget gate: f_t = sigmoid(W_if * x_t + W_hf * h_prev + b_f)
            let f_t =
                Self::sigmoid(&(x_t.dot(&self.w_if.t()) + h_prev.dot(&self.w_hf.t()) + &self.b_f));

            // Cell gate: g_t = tanh(W_ig * x_t + W_hg * h_prev + b_g)
            let g_t =
                Self::tanh(&(x_t.dot(&self.w_ig.t()) + h_prev.dot(&self.w_hg.t()) + &self.b_g));

            // Output gate: o_t = sigmoid(W_io * x_t + W_ho * h_prev + b_o)
            let o_t =
                Self::sigmoid(&(x_t.dot(&self.w_io.t()) + h_prev.dot(&self.w_ho.t()) + &self.b_o));

            // Cell state: c_t = f_t * c_prev + i_t * g_t
            let c_t = &f_t * &c_prev + &i_t * &g_t;

            // Hidden state: h_t = o_t * tanh(c_t)
            let h_t = &o_t * &Self::tanh(&c_t);

            // Store values
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    all_input_gates[[b, t, h]] = i_t[[b, h]];
                    all_forget_gates[[b, t, h]] = f_t[[b, h]];
                    all_cell_gates[[b, t, h]] = g_t[[b, h]];
                    all_output_gates[[b, t, h]] = o_t[[b, h]];
                    all_cell_states[[b, t + 1, h]] = c_t[[b, h]];
                    all_hidden_states[[b, t + 1, h]] = h_t[[b, h]];
                }
            }
        }

        // Update current states with the last computed values
        self.h_t = Some(all_hidden_states.slice(s![.., seq_len, ..]).to_owned());
        self.c_t = Some(all_cell_states.slice(s![.., seq_len, ..]).to_owned());

        // Get the final hidden state
        let final_hidden = all_hidden_states.slice(s![.., seq_len, ..]).to_owned();

        // Classification layer
        let logits = final_hidden.dot(&self.w_out.t()) + &self.b_out;
        let probabilities = Self::softmax(&logits);

        // Store for backward pass if in training mode
        if is_training {
            self.inputs = Some(x.clone());
            self.input_gates = Some(all_input_gates.clone());
            self.forget_gates = Some(all_forget_gates.clone());
            self.cell_gates = Some(all_cell_gates.clone());
            self.output_gates = Some(all_output_gates.clone());
            self.cell_states = Some(all_cell_states.clone());
            self.hidden_states = Some(all_hidden_states.clone());
            self.final_hidden = Some(final_hidden.clone());
            self.output = Some(probabilities.clone());
        }

        (final_hidden, probabilities)
    }
}

impl RecurrentClassifier for LSTMClassifier {
    fn forward(&mut self, x: &Array3<f32>, is_training: bool) -> Array2<f32> {
        let (_, probabilities) = self.process_sequence(x, is_training);
        probabilities
    }

    fn backward(&mut self, _x: &Array3<f32>, targets: &Array2<f32>) -> f32 {
        let inputs = self
            .inputs
            .as_ref()
            .expect("No cached inputs for backward pass");
        let input_gates = self
            .input_gates
            .as_ref()
            .expect("No cached input gates for backward pass");
        let forget_gates = self
            .forget_gates
            .as_ref()
            .expect("No cached forget gates for backward pass");
        let cell_gates = self
            .cell_gates
            .as_ref()
            .expect("No cached cell gates for backward pass");
        let output_gates = self
            .output_gates
            .as_ref()
            .expect("No cached output gates for backward pass");
        let cell_states = self
            .cell_states
            .as_ref()
            .expect("No cached cell states for backward pass");
        let hidden_states = self
            .hidden_states
            .as_ref()
            .expect("No cached hidden states for backward pass");
        let final_hidden = self
            .final_hidden
            .as_ref()
            .expect("No cached final hidden state for backward pass");
        let output = self
            .output
            .as_ref()
            .expect("No cached output for backward pass");

        let batch_size = inputs.shape()[0];
        let seq_len = inputs.shape()[1];

        // Initialize gradients
        let mut dw_ii = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_hi = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_i = Array1::zeros(self.hidden_size);

        let mut dw_if = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_hf = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_f = Array1::zeros(self.hidden_size);

        let mut dw_ig = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_hg = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_g = Array1::zeros(self.hidden_size);

        let mut dw_io = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_ho = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_o = Array1::zeros(self.hidden_size);

        let mut dw_out = Array2::zeros((self.output_size, self.hidden_size));
        let mut db_out = Array1::zeros(self.output_size);

        // Calculate cross-entropy loss
        let mut total_loss = 0.0;
        for (output_row, target_row) in output.outer_iter().zip(targets.outer_iter()) {
            for (j, &target) in target_row.iter().enumerate() {
                if j < self.output_size && target > 0.0 {
                    total_loss -= output_row[j].ln() * target;
                }
            }
        }
        let avg_loss = total_loss / (batch_size as f32);

        // Gradient of cross-entropy loss with respect to softmax output
        let mut doutput = output.clone();
        for i in 0..batch_size {
            for j in 0..self.output_size {
                doutput[[i, j]] -= targets[[i, j]];
            }
        }
        doutput /= batch_size as f32;

        // Backpropagate through output layer
        dw_out = dw_out + doutput.t().dot(final_hidden);
        db_out = db_out + doutput.sum_axis(Axis(0));

        // Gradient with respect to final hidden state
        let dh_final = doutput.dot(&self.w_out);

        // Initialize gradients for the last time step
        let mut dh_next = Array2::zeros((batch_size, self.hidden_size));
        let mut dc_next = Array2::zeros((batch_size, self.hidden_size));

        // Set gradient for final hidden state (from classifier)
        for i in 0..batch_size {
            for j in 0..self.hidden_size {
                dh_next[[i, j]] = dh_final[[i, j]];
            }
        }

        // Iterate backwards through time steps
        for t in (0..seq_len).rev() {
            // Get current timestep values
            let i_t = input_gates.slice(s![.., t, ..]).to_owned();
            let f_t = forget_gates.slice(s![.., t, ..]).to_owned();
            let g_t = cell_gates.slice(s![.., t, ..]).to_owned();
            let o_t = output_gates.slice(s![.., t, ..]).to_owned();
            let c_t = cell_states.slice(s![.., t + 1, ..]).to_owned();
            let c_prev = cell_states.slice(s![.., t, ..]).to_owned();
            let h_prev = hidden_states.slice(s![.., t, ..]).to_owned();
            let x_t = inputs
                .slice(s![.., t, ..])
                .to_owned()
                .into_shape_with_order((batch_size, self.input_size))
                .unwrap();

            // Gradient of output gate: do = dh * tanh(c) * sigmoid_derivative(o)
            let tanh_c_t = Self::tanh(&c_t);
            let do_t = &dh_next * &tanh_c_t * &Self::sigmoid_derivative(&o_t);

            // Gradient of cell state: dc = dh * o * tanh_derivative(c) + dc_next
            let dc = &dh_next * &o_t * &Self::tanh_derivative(&tanh_c_t) + &dc_next;

            // Gradient of input gate: di = dc * g * sigmoid_derivative(i)
            let di_t = &dc * &g_t * &Self::sigmoid_derivative(&i_t);

            // Gradient of forget gate: df = dc * c_prev * sigmoid_derivative(f)
            let df_t = &dc * &c_prev * &Self::sigmoid_derivative(&f_t);

            // Gradient of cell gate: dg = dc * i * tanh_derivative(g)
            let dg_t = &dc * &i_t * &Self::tanh_derivative(&g_t);

            // Accumulate gradients for weights and biases
            db_i = db_i + di_t.sum_axis(Axis(0));
            db_f = db_f + df_t.sum_axis(Axis(0));
            db_g = db_g + dg_t.sum_axis(Axis(0));
            db_o = db_o + do_t.sum_axis(Axis(0));

            dw_ii = dw_ii + di_t.t().dot(&x_t);
            dw_if = dw_if + df_t.t().dot(&x_t);
            dw_ig = dw_ig + dg_t.t().dot(&x_t);
            dw_io = dw_io + do_t.t().dot(&x_t);

            dw_hi = dw_hi + di_t.t().dot(&h_prev);
            dw_hf = dw_hf + df_t.t().dot(&h_prev);
            dw_hg = dw_hg + dg_t.t().dot(&h_prev);
            dw_ho = dw_ho + do_t.t().dot(&h_prev);

            // Gradient for previous hidden state
            let dh_i = di_t.dot(&self.w_hi);
            let dh_f = df_t.dot(&self.w_hf);
            let dh_g = dg_t.dot(&self.w_hg);
            let dh_o = do_t.dot(&self.w_ho);

            dh_next = dh_i + dh_f + dh_g + dh_o;

            // Gradient for previous cell state
            dc_next = &dc * &f_t;
        }

        // Store gradients
        self.dw_ii = Some(dw_ii);
        self.dw_hi = Some(dw_hi);
        self.db_i = Some(db_i);
        self.dw_if = Some(dw_if);
        self.dw_hf = Some(dw_hf);
        self.db_f = Some(db_f);
        self.dw_ig = Some(dw_ig);
        self.dw_hg = Some(dw_hg);
        self.db_g = Some(db_g);
        self.dw_io = Some(dw_io);
        self.dw_ho = Some(dw_ho);
        self.db_o = Some(db_o);
        self.dw_out = Some(dw_out);
        self.db_out = Some(db_out);

        avg_loss
    }

    fn update_params(&mut self, learning_rate: f32) {
        // Update input gate parameters
        if let Some(dw_ii) = &self.dw_ii {
            self.w_ii = &self.w_ii - &(dw_ii * learning_rate);
        }
        if let Some(dw_hi) = &self.dw_hi {
            self.w_hi = &self.w_hi - &(dw_hi * learning_rate);
        }
        if let Some(db_i) = &self.db_i {
            self.b_i = &self.b_i - &(db_i * learning_rate);
        }

        // Update forget gate parameters
        if let Some(dw_if) = &self.dw_if {
            self.w_if = &self.w_if - &(dw_if * learning_rate);
        }
        if let Some(dw_hf) = &self.dw_hf {
            self.w_hf = &self.w_hf - &(dw_hf * learning_rate);
        }
        if let Some(db_f) = &self.db_f {
            self.b_f = &self.b_f - &(db_f * learning_rate);
        }

        // Update cell gate parameters
        if let Some(dw_ig) = &self.dw_ig {
            self.w_ig = &self.w_ig - &(dw_ig * learning_rate);
        }
        if let Some(dw_hg) = &self.dw_hg {
            self.w_hg = &self.w_hg - &(dw_hg * learning_rate);
        }
        if let Some(db_g) = &self.db_g {
            self.b_g = &self.b_g - &(db_g * learning_rate);
        }

        // Update output gate parameters
        if let Some(dw_io) = &self.dw_io {
            self.w_io = &self.w_io - &(dw_io * learning_rate);
        }
        if let Some(dw_ho) = &self.dw_ho {
            self.w_ho = &self.w_ho - &(dw_ho * learning_rate);
        }
        if let Some(db_o) = &self.db_o {
            self.b_o = &self.b_o - &(db_o * learning_rate);
        }

        // Update output projection parameters
        if let Some(dw_out) = &self.dw_out {
            self.w_out = &self.w_out - &(dw_out * learning_rate);
        }
        if let Some(db_out) = &self.db_out {
            self.b_out = &self.b_out - &(db_out * learning_rate);
        }
    }

    fn reset_state(&mut self) {
        self.h_t = None;
        self.c_t = None;
    }

    fn predict(&mut self, x: &Array3<f32>) -> Array1<usize> {
        // Forward pass
        let probs = self.forward(x, false);

        // Get predicted class (argmax)
        let mut predictions = Array1::zeros(probs.shape()[0]);
        for (i, row) in probs.outer_iter().enumerate() {
            let mut max_idx = 0;
            let mut max_val = row[0];
            for (j, &val) in row.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            predictions[i] = max_idx;
        }

        predictions
    }
}

// GRU model for sequence classification
#[derive(Debug, Serialize, Deserialize)]
struct GRUClassifier {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    batch_size: usize,

    // Parameters: GRU Cell
    // Reset gate
    w_ir: Array2<f32>,
    w_hr: Array2<f32>,
    b_r: Array1<f32>,

    // Update gate
    w_iz: Array2<f32>,
    w_hz: Array2<f32>,
    b_z: Array1<f32>,

    // New gate
    w_in: Array2<f32>,
    w_hn: Array2<f32>,
    b_n: Array1<f32>,

    // Parameters: Classification Head
    w_out: Array2<f32>,
    b_out: Array1<f32>,

    // Gradients
    dw_ir: Option<Array2<f32>>,
    dw_hr: Option<Array2<f32>>,
    db_r: Option<Array1<f32>>,
    dw_iz: Option<Array2<f32>>,
    dw_hz: Option<Array2<f32>>,
    db_z: Option<Array1<f32>>,
    dw_in: Option<Array2<f32>>,
    dw_hn: Option<Array2<f32>>,
    db_n: Option<Array1<f32>>,
    dw_out: Option<Array2<f32>>,
    db_out: Option<Array1<f32>>,

    // Hidden state
    h_t: Option<Array2<f32>>,

    // Cache for backward pass
    inputs: Option<Array3<f32>>,
    reset_gates: Option<Array3<f32>>,
    update_gates: Option<Array3<f32>>,
    new_gates: Option<Array3<f32>>,
    hidden_states: Option<Array3<f32>>,
    final_hidden: Option<Array2<f32>>,
    output: Option<Array2<f32>>,
}

impl GRUClassifier {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, batch_size: usize) -> Self {
        // Xavier/Glorot initialization for weights
        let bound = (6.0 / (input_size + hidden_size) as f32).sqrt();

        // Create a random number generator
        let mut rng = rand::rng();

        // Reset gate weights
        let mut w_ir = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hr = Array2::<f32>::zeros((hidden_size, hidden_size));
        for elem in w_ir.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_hr.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        let b_r = Array1::zeros(hidden_size);

        // Update gate weights
        let mut w_iz = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hz = Array2::<f32>::zeros((hidden_size, hidden_size));
        for elem in w_iz.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_hz.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        let b_z = Array1::zeros(hidden_size);

        // New gate weights
        let mut w_in = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hn = Array2::<f32>::zeros((hidden_size, hidden_size));
        for elem in w_in.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_hn.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        let b_n = Array1::zeros(hidden_size);

        // Output projection weights
        let output_bound = (6.0 / (hidden_size + output_size) as f32).sqrt();
        let mut w_out = Array2::<f32>::zeros((output_size, hidden_size));
        for elem in w_out.iter_mut() {
            *elem = rng.random_range(-output_bound..output_bound);
        }
        let b_out = Array1::zeros(output_size);

        GRUClassifier {
            input_size,
            hidden_size,
            output_size,
            batch_size,
            w_ir,
            w_hr,
            b_r,
            w_iz,
            w_hz,
            b_z,
            w_in,
            w_hn,
            b_n,
            w_out,
            b_out,
            dw_ir: None,
            dw_hr: None,
            db_r: None,
            dw_iz: None,
            dw_hz: None,
            db_z: None,
            dw_in: None,
            dw_hn: None,
            db_n: None,
            dw_out: None,
            db_out: None,
            h_t: None,
            inputs: None,
            reset_gates: None,
            update_gates: None,
            new_gates: None,
            hidden_states: None,
            final_hidden: None,
            output: None,
        }
    }

    // Activation functions and derivatives
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn sigmoid_derivative(sigmoid_output: &Array2<f32>) -> Array2<f32> {
        sigmoid_output * &(1.0 - sigmoid_output)
    }

    fn tanh(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.tanh())
    }

    fn tanh_derivative(tanh_output: &Array2<f32>) -> Array2<f32> {
        1.0 - tanh_output * tanh_output
    }

    fn softmax(x: &Array2<f32>) -> Array2<f32> {
        let max_vals = x.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let exp_shifted = x
            .outer_iter()
            .zip(max_vals.iter())
            .map(|(row, &max_val)| row.mapv(|v| (v - max_val).exp()))
            .collect::<Vec<_>>();

        let exp_sum = exp_shifted.iter().map(|row| row.sum()).collect::<Vec<_>>();

        let result = exp_shifted
            .iter()
            .zip(exp_sum.iter())
            .map(|(row, &sum)| row.mapv(|v| v / sum))
            .collect::<Vec<_>>();

        let mut output = Array2::zeros(x.raw_dim());
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                output[[i, j]] = val;
            }
        }

        output
    }

    // Process sequence and classify
    fn process_sequence(
        &mut self,
        x: &Array3<f32>,
        is_training: bool,
    ) -> (Array2<f32>, Array2<f32>) {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Initialize hidden state if None
        if self.h_t.is_none() {
            self.h_t = Some(Array2::zeros((batch_size, self.hidden_size)));
        }

        // Arrays to store values for each time step
        let mut all_hidden_states = Array3::zeros((batch_size, seq_len + 1, self.hidden_size));
        let mut all_reset_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_update_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_new_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));

        // Set initial hidden state
        if let Some(h_t) = &self.h_t {
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    all_hidden_states[[b, 0, h]] = h_t[[b, h]];
                }
            }
        }

        // Process sequence
        for t in 0..seq_len {
            // Get input at time t [batch_size, input_size]
            let x_t = x
                .slice(s![.., t, ..])
                .to_owned()
                .into_shape_with_order((batch_size, self.input_size))
                .unwrap();

            // Get previous hidden state
            let h_prev = all_hidden_states.slice(s![.., t, ..]).to_owned();

            // Reset gate: r_t = sigmoid(W_ir * x_t + W_hr * h_prev + b_r)
            let r_t =
                Self::sigmoid(&(x_t.dot(&self.w_ir.t()) + h_prev.dot(&self.w_hr.t()) + &self.b_r));

            // Update gate: z_t = sigmoid(W_iz * x_t + W_hz * h_prev + b_z)
            let z_t =
                Self::sigmoid(&(x_t.dot(&self.w_iz.t()) + h_prev.dot(&self.w_hz.t()) + &self.b_z));

            // New gate: n_t = tanh(W_in * x_t + r_t * (W_hn * h_prev) + b_n)
            let n_t = Self::tanh(
                &(x_t.dot(&self.w_in.t()) + &r_t * &h_prev.dot(&self.w_hn.t()) + &self.b_n),
            );

            // Hidden state: h_t = (1 - z_t) * n_t + z_t * h_prev
            let h_t = (1.0 - &z_t) * &n_t + &z_t * &h_prev;

            // Store values
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    all_reset_gates[[b, t, h]] = r_t[[b, h]];
                    all_update_gates[[b, t, h]] = z_t[[b, h]];
                    all_new_gates[[b, t, h]] = n_t[[b, h]];
                    all_hidden_states[[b, t + 1, h]] = h_t[[b, h]];
                }
            }
        }

        // Update current hidden state with the last computed hidden state
        self.h_t = Some(all_hidden_states.slice(s![.., seq_len, ..]).to_owned());

        // Get the final hidden state
        let final_hidden = all_hidden_states.slice(s![.., seq_len, ..]).to_owned();

        // Classification layer
        let logits = final_hidden.dot(&self.w_out.t()) + &self.b_out;
        let probabilities = Self::softmax(&logits);

        // Store for backward pass if in training mode
        if is_training {
            self.inputs = Some(x.clone());
            self.reset_gates = Some(all_reset_gates.clone());
            self.update_gates = Some(all_update_gates.clone());
            self.new_gates = Some(all_new_gates.clone());
            self.hidden_states = Some(all_hidden_states.clone());
            self.final_hidden = Some(final_hidden.clone());
            self.output = Some(probabilities.clone());
        }

        (final_hidden, probabilities)
    }
}

impl RecurrentClassifier for GRUClassifier {
    fn forward(&mut self, x: &Array3<f32>, is_training: bool) -> Array2<f32> {
        let (_, probabilities) = self.process_sequence(x, is_training);
        probabilities
    }

    fn backward(&mut self, _x: &Array3<f32>, targets: &Array2<f32>) -> f32 {
        let inputs = self
            .inputs
            .as_ref()
            .expect("No cached inputs for backward pass");
        let reset_gates = self
            .reset_gates
            .as_ref()
            .expect("No cached reset gates for backward pass");
        let update_gates = self
            .update_gates
            .as_ref()
            .expect("No cached update gates for backward pass");
        let new_gates = self
            .new_gates
            .as_ref()
            .expect("No cached new gates for backward pass");
        let hidden_states = self
            .hidden_states
            .as_ref()
            .expect("No cached hidden states for backward pass");
        let final_hidden = self
            .final_hidden
            .as_ref()
            .expect("No cached final hidden state for backward pass");
        let output = self
            .output
            .as_ref()
            .expect("No cached output for backward pass");

        let batch_size = inputs.shape()[0];
        let seq_len = inputs.shape()[1];

        // Initialize gradients
        let mut dw_ir = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_hr = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_r = Array1::zeros(self.hidden_size);

        let mut dw_iz = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_hz = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_z = Array1::zeros(self.hidden_size);

        let mut dw_in = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_hn = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_n = Array1::zeros(self.hidden_size);

        let mut dw_out = Array2::zeros((self.output_size, self.hidden_size));
        let mut db_out = Array1::zeros(self.output_size);

        // Calculate cross-entropy loss
        let mut total_loss = 0.0;
        for (output_row, target_row) in output.outer_iter().zip(targets.outer_iter()) {
            for (j, &target) in target_row.iter().enumerate() {
                if j < self.output_size && target > 0.0 {
                    total_loss -= output_row[j].ln() * target;
                }
            }
        }
        let avg_loss = total_loss / (batch_size as f32);

        // Gradient of cross-entropy loss with respect to softmax output
        let mut doutput = output.clone();
        for i in 0..batch_size {
            for j in 0..self.output_size {
                doutput[[i, j]] -= targets[[i, j]];
            }
        }
        doutput /= batch_size as f32;

        // Backpropagate through output layer
        dw_out = dw_out + doutput.t().dot(final_hidden);
        db_out = db_out + doutput.sum_axis(Axis(0));

        // Gradient with respect to final hidden state
        let dh_final = doutput.dot(&self.w_out);

        // Initialize gradient for the last time step
        let mut dh_next = Array2::zeros((batch_size, self.hidden_size));

        // Set gradient for final hidden state (from classifier)
        for i in 0..batch_size {
            for j in 0..self.hidden_size {
                dh_next[[i, j]] = dh_final[[i, j]];
            }
        }

        // Iterate backwards through time steps
        for t in (0..seq_len).rev() {
            // Get current timestep values
            let r_t = reset_gates.slice(s![.., t, ..]).to_owned();
            let z_t = update_gates.slice(s![.., t, ..]).to_owned();
            let n_t = new_gates.slice(s![.., t, ..]).to_owned();
            let h_prev = hidden_states.slice(s![.., t, ..]).to_owned();
            let x_t = inputs
                .slice(s![.., t, ..])
                .to_owned()
                .into_shape_with_order((batch_size, self.input_size))
                .unwrap();

            // Gradient of GRU equations:
            // h_t = (1 - z_t) * n_t + z_t * h_prev
            // n_t = tanh(W_in * x_t + r_t * (W_hn * h_prev) + b_n)
            // r_t = sigmoid(W_ir * x_t + W_hr * h_prev + b_r)
            // z_t = sigmoid(W_iz * x_t + W_hz * h_prev + b_z)

            // Gradient of update gate: dz = dh * (h_prev - n_t) * sigmoid_derivative(z)
            let dz_t = &dh_next * &(&h_prev - &n_t) * &Self::sigmoid_derivative(&z_t);

            // Gradient of new gate: dn = dh * (1 - z_t) * tanh_derivative(n)
            let dn_t = &dh_next * &(1.0 - &z_t) * &Self::tanh_derivative(&n_t);

            // Gradient of reset gate: dr = dn * (W_hn * h_prev) * sigmoid_derivative(r)
            let dr_t = &dn_t * &h_prev.dot(&self.w_hn.t()) * &Self::sigmoid_derivative(&r_t);

            // Accumulate gradients for weights and biases
            db_r = db_r + dr_t.sum_axis(Axis(0));
            db_z = db_z + dz_t.sum_axis(Axis(0));
            db_n = db_n + dn_t.sum_axis(Axis(0));

            dw_ir = dw_ir + dr_t.t().dot(&x_t);
            dw_iz = dw_iz + dz_t.t().dot(&x_t);
            dw_in = dw_in + dn_t.t().dot(&x_t);

            dw_hr = dw_hr + dr_t.t().dot(&h_prev);
            dw_hz = dw_hz + dz_t.t().dot(&h_prev);

            // For w_hn, we need to account for the reset gate
            let dn_h = &dn_t * &r_t;
            dw_hn = dw_hn + dn_h.t().dot(&h_prev);

            // Gradient for previous hidden state
            // dh_prev = dh_next * z_t + dn_t * r_t * W_hn + dr_t * W_hr + dz_t * W_hz
            let dh_z = &dh_next * &z_t;
            let dh_n = &dn_t * &r_t.dot(&self.w_hn);
            let dh_r = dr_t.dot(&self.w_hr);
            let dh_z_gate = dz_t.dot(&self.w_hz);

            dh_next = dh_z + dh_n + dh_r + dh_z_gate;
        }

        // Store gradients
        self.dw_ir = Some(dw_ir);
        self.dw_hr = Some(dw_hr);
        self.db_r = Some(db_r);
        self.dw_iz = Some(dw_iz);
        self.dw_hz = Some(dw_hz);
        self.db_z = Some(db_z);
        self.dw_in = Some(dw_in);
        self.dw_hn = Some(dw_hn);
        self.db_n = Some(db_n);
        self.dw_out = Some(dw_out);
        self.db_out = Some(db_out);

        avg_loss
    }

    fn update_params(&mut self, learning_rate: f32) {
        // Update reset gate parameters
        if let Some(dw_ir) = &self.dw_ir {
            self.w_ir = &self.w_ir - &(dw_ir * learning_rate);
        }
        if let Some(dw_hr) = &self.dw_hr {
            self.w_hr = &self.w_hr - &(dw_hr * learning_rate);
        }
        if let Some(db_r) = &self.db_r {
            self.b_r = &self.b_r - &(db_r * learning_rate);
        }

        // Update update gate parameters
        if let Some(dw_iz) = &self.dw_iz {
            self.w_iz = &self.w_iz - &(dw_iz * learning_rate);
        }
        if let Some(dw_hz) = &self.dw_hz {
            self.w_hz = &self.w_hz - &(dw_hz * learning_rate);
        }
        if let Some(db_z) = &self.db_z {
            self.b_z = &self.b_z - &(db_z * learning_rate);
        }

        // Update new gate parameters
        if let Some(dw_in) = &self.dw_in {
            self.w_in = &self.w_in - &(dw_in * learning_rate);
        }
        if let Some(dw_hn) = &self.dw_hn {
            self.w_hn = &self.w_hn - &(dw_hn * learning_rate);
        }
        if let Some(db_n) = &self.db_n {
            self.b_n = &self.b_n - &(db_n * learning_rate);
        }

        // Update output projection parameters
        if let Some(dw_out) = &self.dw_out {
            self.w_out = &self.w_out - &(dw_out * learning_rate);
        }
        if let Some(db_out) = &self.db_out {
            self.b_out = &self.b_out - &(db_out * learning_rate);
        }
    }

    fn reset_state(&mut self) {
        self.h_t = None;
    }

    fn predict(&mut self, x: &Array3<f32>) -> Array1<usize> {
        // Forward pass
        let probs = self.forward(x, false);

        // Get predicted class (argmax)
        let mut predictions = Array1::zeros(probs.shape()[0]);
        for (i, row) in probs.outer_iter().enumerate() {
            let mut max_idx = 0;
            let mut max_val = row[0];
            for (j, &val) in row.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            predictions[i] = max_idx;
        }

        predictions
    }
}

// Helper function to create training data
fn create_sentiment_data() -> (Vec<Array3<f32>>, Vec<Array2<f32>>) {
    // Simple toy sentiment analysis dataset
    // Format: (text, sentiment) where sentiment is 0 (negative) or 1 (positive)
    let dataset = vec![
        ("good movie great acting", 1),
        ("terrible waste of time", 0),
        ("awesome film loved it", 1),
        ("boring plot bad acting", 0),
        ("excellent performances entertaining", 1),
        ("disappointing storyline", 0),
        ("fantastic direction brilliant script", 1),
        ("awful dialogue worst movie", 0),
        ("masterpiece incredible story", 1),
        ("dull characters poor execution", 0),
    ];

    // Create vocabulary
    let mut vocab = HashMap::new();
    let mut word_id = 0;

    for (text, _) in &dataset {
        for word in text.split_whitespace() {
            if !vocab.contains_key(word) {
                vocab.insert(word.to_string(), word_id);
                word_id += 1;
            }
        }
    }

    let vocab_size = vocab.len();
    println!("Vocabulary size: {}", vocab_size);

    // Encode dataset
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for (text, sentiment) in dataset {
        // Get the sequence length
        let words: Vec<&str> = text.split_whitespace().collect();
        let seq_len = words.len();

        // Create one-hot encoded input
        let mut input = Array3::<f32>::zeros((1, seq_len, vocab_size));

        for (t, _word) in words.iter().enumerate() {
            if let Some(&word_idx) = vocab.get(*_word) {
                input[[0, t, word_idx]] = 1.0;
            }
        }

        // Create target
        let mut target = Array2::<f32>::zeros((1, 2)); // Binary classification
        target[[0, sentiment as usize]] = 1.0;

        inputs.push(input);
        targets.push(target);
    }

    (inputs, targets)
}

fn sentiment_analysis_example() {
    println!("Creating toy sentiment analysis dataset...");
    let (inputs, targets) = create_sentiment_data();

    // Split into train and test sets
    let train_size = 8;
    let test_size = inputs.len() - train_size;

    let train_inputs = inputs[..train_size].to_vec();
    let train_targets = targets[..train_size].to_vec();
    let test_inputs = inputs[train_size..].to_vec();
    let test_targets = targets[train_size..].to_vec();

    // Model parameters
    let input_size = train_inputs[0].shape()[2]; // Vocabulary size
    let hidden_size = 16;
    let output_size = 2; // Binary classification
    let batch_size = 1;

    // Create LSTM and GRU classifiers
    let mut lstm_classifier = LSTMClassifier::new(input_size, hidden_size, output_size, batch_size);
    let mut gru_classifier = GRUClassifier::new(input_size, hidden_size, output_size, batch_size);

    // Training parameters
    let learning_rate = 0.01;
    let num_epochs = 100;

    println!("Training models...");

    // Training loop
    for epoch in 0..num_epochs {
        let mut lstm_total_loss = 0.0;
        let mut gru_total_loss = 0.0;

        for (input, target) in train_inputs.iter().zip(train_targets.iter()) {
            // Train LSTM
            lstm_classifier.reset_state();
            lstm_classifier.forward(input, true);
            lstm_total_loss += lstm_classifier.backward(input, target);
            lstm_classifier.update_params(learning_rate);

            // Train GRU
            gru_classifier.reset_state();
            gru_classifier.forward(input, true);
            gru_total_loss += gru_classifier.backward(input, target);
            gru_classifier.update_params(learning_rate);
        }

        // Calculate average loss
        let lstm_avg_loss = lstm_total_loss / (train_size as f32);
        let gru_avg_loss = gru_total_loss / (train_size as f32);

        // Print progress
        if (epoch + 1) % 10 == 0 {
            println!(
                "Epoch {}/{} - LSTM Loss: {:.6}, GRU Loss: {:.6}",
                epoch + 1,
                num_epochs,
                lstm_avg_loss,
                gru_avg_loss
            );
        }
    }

    // Evaluate models on test set
    println!("\nEvaluating models on test set...");

    let mut lstm_correct = 0;
    let mut gru_correct = 0;

    for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
        // LSTM prediction
        lstm_classifier.reset_state();
        let lstm_pred = lstm_classifier.predict(input)[0];

        // GRU prediction
        gru_classifier.reset_state();
        let gru_pred = gru_classifier.predict(input)[0];

        // Get true label
        let true_label = target
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Count correct predictions
        if lstm_pred == true_label {
            lstm_correct += 1;
        }

        if gru_pred == true_label {
            gru_correct += 1;
        }
    }

    // Calculate accuracy
    let lstm_accuracy = (lstm_correct as f32) / (test_size as f32);
    let gru_accuracy = (gru_correct as f32) / (test_size as f32);

    println!("LSTM Test Accuracy: {:.2}%", lstm_accuracy * 100.0);
    println!("GRU Test Accuracy: {:.2}%", gru_accuracy * 100.0);

    // Test with some new examples
    println!("\nTesting with new examples:");

    let test_examples = [
        "wonderful movie amazing experience",
        "horrible acting terrible script",
    ];

    // Create vocabulary map from the training data
    let mut vocab = HashMap::new();
    for input in &train_inputs {
        for t in 0..input.shape()[1] {
            for v in 0..input.shape()[2] {
                if input[[0, t, v]] > 0.0 {
                    vocab.entry(v).or_insert(true);
                }
            }
        }
    }

    for (i, example) in test_examples.iter().enumerate() {
        // Tokenize example
        let words: Vec<&str> = example.split_whitespace().collect();
        let seq_len = words.len();

        // Create input tensor
        let mut input = Array3::<f32>::zeros((1, seq_len, input_size));

        // Find word indices in the vocabulary
        for (t, _word) in words.iter().enumerate() {
            // This is a simplified approach - in practice, you'd have a proper word->idx mapping
            for word_idx in vocab.keys() {
                if train_inputs
                    .iter()
                    .any(|x| x[[0, t % x.shape()[1], *word_idx]] > 0.0)
                {
                    input[[0, t, *word_idx]] = 1.0;
                    break;
                }
            }
        }

        // Get predictions
        lstm_classifier.reset_state();
        let lstm_pred = lstm_classifier.predict(&input)[0];

        gru_classifier.reset_state();
        let gru_pred = gru_classifier.predict(&input)[0];

        println!("Example {}: \"{}\"", i + 1, example);
        println!(
            "  LSTM prediction: {}",
            if lstm_pred == 1 {
                "Positive"
            } else {
                "Negative"
            }
        );
        println!(
            "  GRU prediction: {}",
            if gru_pred == 1 {
                "Positive"
            } else {
                "Negative"
            }
        );
    }
}

fn main() {
    println!("Sentiment Analysis with Recurrent Neural Networks");
    println!("================================================");

    sentiment_analysis_example();
}
