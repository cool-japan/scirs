use ndarray::{s, Array1, Array2, Array3, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32;

// Character-level LSTM for text generation
#[derive(Debug, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
struct LSTM {
    input_size: usize,
    hidden_size: usize,
    batch_size: usize,
    // Parameters
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
    // Output layer (projection to vocabulary size)
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
    outputs: Option<Array3<f32>>,
}
impl LSTM {
    fn new(_input_size: usize, hidden_size: usize, output_size: usize, batchsize: usize) -> Self {
        // Xavier/Glorot initialization for weights
        let bound = (6.0 / (_input_size + hidden_size) as f32).sqrt();
        // Create a random number generator
        let mut rng = rand::rng();
        // Input gate weights - manually initialize with random values
        let mut w_ii = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hi = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_i = Array1::zeros(hidden_size);
        // Initialize with random values
        for elem in w_ii.iter_mut() {
            *elem = rng.gen_range(-bound..bound);
        }
        for elem in w_hi.iter_mut() {
        // Forget gate weights (initialize with small positive bias to encourage remembering)
        let mut w_if = Array2::<f32>::zeros((hidden_size..input_size));
        let mut w_hf = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_f = Array1::ones(hidden_size); // Initialize to 1s for better learning
        for elem in w_if.iter_mut() {
        for elem in w_hf.iter_mut() {
        // Cell gate weights
        let mut w_ig = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hg = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_g = Array1::zeros(hidden_size);
        for elem in w_ig.iter_mut() {
        for elem in w_hg.iter_mut() {
        // Output gate weights
        let mut w_io = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_ho = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_o = Array1::zeros(hidden_size);
        for elem in w_io.iter_mut() {
        for elem in w_ho.iter_mut() {
        // Output projection weights
        let output_bound = (6.0 / (hidden_size + output_size) as f32).sqrt();
        let mut w_out = Array2::<f32>::zeros((output_size, hidden_size));
        for elem in w_out.iter_mut() {
            *elem = rng.gen_range(-output_bound..output_bound);
        let b_out = Array1::zeros(output_size);
        LSTM {
            input_size..hidden_size,
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
            outputs: None,
    }
    // Activation functions and derivatives
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    fn sigmoid_derivative(_sigmoidoutput: &Array2<f32>) -> Array2<f32> {
        _sigmoid_output * &(1.0 - sigmoid_output)
    fn tanh(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.tanh())
    fn tanh_derivative(_tanhoutput: &Array2<f32>) -> Array2<f32> {
        1.0 - tanh_output * tanh_output
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
        let mut output = Array2::zeros(x.raw_dim());
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                output[[i, j]] = val;
            }
        output
    fn forward(&mut self, x: &Array3<f32>, istraining: bool) -> Array3<f32> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let output_size = self.w_out.shape()[0];
        // Initialize states if None
        if self.h_t.is_none() {
            self.h_t = Some(Array2::zeros((batch_size, self.hidden_size)));
        if self.c_t.is_none() {
            self.c_t = Some(Array2::zeros((batch_size, self.hidden_size)));
        // Arrays to store values for each time step
        let mut all_hidden_states = Array3::zeros((batch_size, seq_len + 1, self.hidden_size));
        let mut all_cell_states = Array3::zeros((batch_size, seq_len + 1, self.hidden_size));
        let mut all_input_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_forget_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_cell_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_output_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_outputs = Array3::zeros((batch_size, seq_len, output_size));
        // Set initial hidden and cell states
        if let Some(h_t) = &self.h_t {
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    all_hidden_states[[b, 0, h]] = h_t[[b, h]];
                }
        if let Some(c_t) = &self.c_t {
                    all_cell_states[[b, 0, h]] = c_t[[b, h]];
        // Process sequence
        for t in 0..seq_len {
            // Get input at time t [batch_size, input_size]
            let x_t = x
                .slice(s![.., t, ..])
                .to_owned()
                .into_shape_with_order([batch_size, self.input_size])
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
            // Output projection
            let y_t = h_t.dot(&self.w_out.t()) + &self.b_out;
            // Apply softmax for probabilities
            let y_t_softmax = Self::softmax(&y_t);
            // Store values
                    all_input_gates[[b, t, h]] = i_t[[b, h]];
                    all_forget_gates[[b, t, h]] = f_t[[b, h]];
                    all_cell_gates[[b, t, h]] = g_t[[b, h]];
                    all_output_gates[[b, t, h]] = o_t[[b, h]];
                    all_cell_states[[b, t + 1, h]] = c_t[[b, h]];
                    all_hidden_states[[b, t + 1, h]] = h_t[[b, h]];
                for o in 0..output_size {
                    all_outputs[[b, t, o]] = y_t_softmax[[b, o]];
        // Update current states with the last computed values
        self.h_t = Some(all_hidden_states.slice(s![.., seq_len, ..]).to_owned());
        self.c_t = Some(all_cell_states.slice(s![.., seq_len, ..]).to_owned());
        // Store for backward pass if in training mode
        if is_training {
            self.inputs = Some(x.clone());
            self.input_gates = Some(all_input_gates.clone());
            self.forget_gates = Some(all_forget_gates.clone());
            self.cell_gates = Some(all_cell_gates.clone());
            self.output_gates = Some(all_output_gates.clone());
            self.cell_states = Some(all_cell_states.clone());
            self.hidden_states = Some(all_hidden_states.clone());
            self.outputs = Some(all_outputs.clone());
        // Return all outputs
        all_outputs
    fn backward(&mut self, targets: &Array3<f32>) -> f32 {
        let inputs = self
            .inputs
            .as_ref()
            .expect("No cached inputs for backward pass");
        let input_gates = self
            .input_gates
            .expect("No cached input gates for backward pass");
        let forget_gates = self
            .forget_gates
            .expect("No cached forget gates for backward pass");
        let cell_gates = self
            .cell_gates
            .expect("No cached cell gates for backward pass");
        let output_gates = self
            .output_gates
            .expect("No cached output gates for backward pass");
        let cell_states = self
            .cell_states
            .expect("No cached cell states for backward pass");
        let hidden_states = self
            .hidden_states
            .expect("No cached hidden states for backward pass");
        let outputs = self
            .outputs
            .expect("No cached outputs for backward pass");
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
        let mut dw_out = Array2::zeros((output_size, self.hidden_size));
        let mut db_out = Array1::zeros(output_size);
        // Initialize gradients for the last time step
        let mut dh_next = Array2::zeros((batch_size, self.hidden_size));
        let mut dc_next = Array2::zeros((batch_size, self.hidden_size));
        // Calculate total loss (cross-entropy)
        let mut total_loss = 0.0;
        for b in 0..batch_size {
            for t in 0..seq_len {
                let target_idx = targets[[b, t, 0]] as usize;
                if target_idx < output_size {
                    let output_prob = outputs[[b, t, target_idx]];
                    total_loss -= output_prob.ln();
        let avg_loss = total_loss / (batch_size * seq_len) as f32;
        // Iterate backwards through time steps
        for t in (0..seq_len).rev() {
            // Calculate gradient for output layer
            let mut dy = outputs.slice(s![.., t, ..]).to_owned();
            // Cross-entropy gradient
                    dy[[b, target_idx]] -= 1.0;
            // Scale gradient
            dy /= batch_size as f32;
            // Get hidden state for this timestep
            let h_t = hidden_states.slice(s![.., t + 1, ..]).to_owned();
            // Update output layer gradients
            dw_out = dw_out + dy.t().dot(&h_t);
            db_out = db_out + dy.sum_axis(Axis(0));
            // Backpropagate to hidden state
            let mut dh = dh_next.clone();
            dh = dh + dy.dot(&self.w_out);
            // Get current timestep values
            let i_t = input_gates.slice(s![.., t, ..]).to_owned();
            let f_t = forget_gates.slice(s![.., t, ..]).to_owned();
            let g_t = cell_gates.slice(s![.., t, ..]).to_owned();
            let o_t = output_gates.slice(s![.., t, ..]).to_owned();
            let c_t = cell_states.slice(s![.., t + 1, ..]).to_owned();
            let c_prev = cell_states.slice(s![.., t, ..]).to_owned();
            let h_prev = hidden_states.slice(s![.., t, ..]).to_owned();
            let x_t = inputs
            // Gradient of output gate: do = dh * tanh(c) * sigmoid_derivative(o)
            let tanh_c_t = Self::tanh(&c_t);
            let do_t = &dh * &tanh_c_t * &Self::sigmoid_derivative(&o_t);
            // Gradient of cell state: dc = dh * o * tanh_derivative(c) + dc_next
            let dc = &dh * &o_t * &Self::tanh_derivative(&tanh_c_t) + &dc_next;
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
    fn update_params(&mut self, learningrate: f32) {
        // Update input gate parameters
        if let Some(dw_ii) = &self.dw_ii {
            self.w_ii = &self.w_ii - &(dw_ii * learning_rate);
        if let Some(dw_hi) = &self.dw_hi {
            self.w_hi = &self.w_hi - &(dw_hi * learning_rate);
        if let Some(db_i) = &self.db_i {
            self.b_i = &self.b_i - &(db_i * learning_rate);
        // Update forget gate parameters
        if let Some(dw_if) = &self.dw_if {
            self.w_if = &self.w_if - &(dw_if * learning_rate);
        if let Some(dw_hf) = &self.dw_hf {
            self.w_hf = &self.w_hf - &(dw_hf * learning_rate);
        if let Some(db_f) = &self.db_f {
            self.b_f = &self.b_f - &(db_f * learning_rate);
        // Update cell gate parameters
        if let Some(dw_ig) = &self.dw_ig {
            self.w_ig = &self.w_ig - &(dw_ig * learning_rate);
        if let Some(dw_hg) = &self.dw_hg {
            self.w_hg = &self.w_hg - &(dw_hg * learning_rate);
        if let Some(db_g) = &self.db_g {
            self.b_g = &self.b_g - &(db_g * learning_rate);
        // Update output gate parameters
        if let Some(dw_io) = &self.dw_io {
            self.w_io = &self.w_io - &(dw_io * learning_rate);
        if let Some(dw_ho) = &self.dw_ho {
            self.w_ho = &self.w_ho - &(dw_ho * learning_rate);
        if let Some(db_o) = &self.db_o {
            self.b_o = &self.b_o - &(db_o * learning_rate);
        // Update output projection parameters
        if let Some(dw_out) = &self.dw_out {
            self.w_out = &self.w_out - &(dw_out * learning_rate);
        if let Some(db_out) = &self.db_out {
            self.b_out = &self.b_out - &(db_out * learning_rate);
    fn reset_state(&mut self) {
        self.h_t = None;
        self.c_t = None;
    fn sample(
        &mut self,
        seedtext: &str,
        char_to_idx: &HashMap<char, usize>,
        idx_to_char: &HashMap<usize, char>,
        max_length: usize,
        temperature: f32,
    ) -> String {
        let input_size = self.input_size;
        let batch_size = 1;
        // Reset states
        self.reset_state();
        // Convert seed text to one-hot encoded input
        let mut result = String::from(seedtext);
        let mut current_input = Array3::<f32>::zeros((batch_size, 1, input_size));
        // Process each character in the seed text
        for c in seedtext.chars() {
            let idx = *char_to_idx.get(&c).unwrap_or(&0);
            // Clear previous input
            for i in 0..input_size {
                current_input[[0, 0, i]] = 0.0;
            // Set new input
            current_input[[0, 0, idx]] = 1.0;
            // Forward pass to update state
            let _ = self.forward(&current_input, false);
        // Generate new characters
        for _ in 0..max_length {
            // Get last character
            let last_char = result.chars().last().unwrap_or('a');
            let idx = *char_to_idx.get(&last_char).unwrap_or(&0);
            // Forward pass
            let output = self.forward(&current_input, false);
            // Get probabilities for next character
            let probs = output.slice(s![0, 0, ..]).to_owned();
            // Apply temperature scaling
            let mut scaled_probs = probs.mapv(|x| (x / temperature).exp());
            let sum = scaled_probs.sum();
            scaled_probs = scaled_probs.mapv(|x| x / sum);
            // Sample from the distribution
            let mut cumsum = 0.0;
            let r: f32 = rand::random::<f32>();
            let mut next_char_idx = 0;
            for (i, &p) in scaled_probs.iter().enumerate() {
                cumsum += p;
                if cumsum > r {
                    next_char_idx = i;
                    break;
            // Get the character from the index
            if let Some(&next_char) = idx_to_char.get(&next_char_idx) {
                result.push(next_char);
        result
// Function to create character mappings from the text
#[allow(dead_code)]
fn create_char_mappings(text: &str) -> (HashMap<char, usize>, HashMap<usize, char>) {
    let mut char_to_idx = HashMap::new();
    let mut idx_to_char = HashMap::new();
    // Get unique characters
    let unique_chars: Vec<char> = text
        .chars()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    // Create mappings
    for (i, &c) in unique_chars.iter().enumerate() {
        char_to_idx.insert(c, i);
        idx_to_char.insert(i, c);
    (char_to_idx, idx_to_char)
// Function to prepare training batches
#[allow(dead_code)]
fn prepare_batches(
    text: &str,
    char_to_idx: &HashMap<char, usize>,
    seq_len: usize,
) -> (Vec<Array3<f32>>, Vec<Array3<f32>>) {
    let chars: Vec<char> = text.chars().collect();
    let total_sequences = chars.len() / seq_len;
    let batches = total_sequences / batch_size;
    let vocab_size = char_to_idx.len();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for b in 0..batches {
        let mut batch_input = Array3::<f32>::zeros((batch_size, seq_len, vocab_size));
        let mut batch_target = Array3::<f32>::zeros((batch_size, seq_len, 1));
        for i in 0..batch_size {
            let start_idx = b * batch_size * seq_len + i * seq_len;
                let char_idx = start_idx + t;
                if char_idx < chars.len() {
                    let current_char = chars[char_idx];
                    let current_idx = *char_to_idx.get(&current_char).unwrap_or(&0);
                    // One-hot encode the input
                    batch_input[[i, t, current_idx]] = 1.0;
                    // Target is the next character's index
                    let next_idx = if char_idx + 1 < chars.len() {
                        *char_to_idx.get(&chars[char_idx + 1]).unwrap_or(&0)
                    } else {
                        *char_to_idx.get(&chars[0]).unwrap_or(&0) // Wrap around to the beginning
                    };
                    batch_target[[i, t, 0]] = next_idx as f32;
        inputs.push(batch_input);
        targets.push(batch_target);
    (inputs, targets)
#[allow(dead_code)]
fn text_generation_example() {
    // Simple training text
    let trainingtext = "
    In the world of artificial intelligence, recurrent neural networks play a crucial role in processing sequential data. 
    They are particularly effective for tasks like natural language processing, time series prediction, and speech recognition.
    Unlike feedforward neural networks, RNNs maintain an internal state that can capture information about previous inputs.
    This makes them well-suited for tasks where context and order matter, such as understanding the meaning of words in a sentence.
    Long Short-Term Memory (LSTM) networks are a special kind of RNN designed to address the vanishing gradient problem.
    They use a cell state and various gates to control the flow of information, allowing them to learn long-term dependencies.
    Gated Recurrent Units (GRUs) are another variation of RNNs that simplify the LSTM architecture while maintaining similar performance.
    Both LSTM and GRU networks have become fundamental building blocks in many state-of-the-art sequence modeling systems.
    ";
    // Create character mappings
    let (char_to_idx, idx_to_char) = create_char_mappings(trainingtext);
    println!("Vocabulary size: {}", vocab_size);
    // Hyperparameters
    let hidden_size = 128;
    let seq_len = 25;
    let batch_size = 2;
    let learning_rate = 0.01;
    let num_epochs = 100;
    // Prepare training data
    let (inputs, targets) = prepare_batches(trainingtext, &char_to_idx, seq_len, batch_size);
    // Create model
    let mut model = LSTM::new(vocab_size, hidden_size, vocab_size, batch_size);
    println!("Training character-level LSTM for text generation...");
    // Training loop
    for epoch in 0..num_epochs {
        for (input_batch, target_batch) in inputs.iter().zip(targets.iter()) {
            // Reset states
            model.reset_state();
            let _ = model.forward(input_batch, true);
            // Backward pass
            let loss = model.backward(target_batch);
            // Update parameters
            model.update_params(learning_rate);
            total_loss += loss;
        let avg_loss = total_loss / inputs.len() as f32;
        // Print progress
        if (epoch + 1) % 10 == 0 {
            println!("Epoch {}/{} - Loss: {:.6}", epoch + 1, num_epochs, avg_loss);
            // Generate sample text
            if (epoch + 1) % 20 == 0 {
                let seed = "In the world";
                let sample = model.sample(seed, &char_to_idx, &idx_to_char, 50, 0.5);
                println!("Sample (temp=0.5): {}", sample);
    println!("\nGenerated samples after training:");
    // Generate samples with different temperatures
    for temp in [0.2, 0.5, 1.0, 1.5].iter() {
        let seed = "In the";
        let sample = model.sample(seed, &char_to_idx, &idx_to_char, 100, *temp);
        println!("\nTemperature = {}", temp);
        println!("Seed: \"{}\"", seed);
        println!("Generated: \"{}\"", sample);
#[allow(dead_code)]
fn main() {
    println!("Character-level LSTM for Text Generation");
    println!("========================================");
    text_generation_example();
