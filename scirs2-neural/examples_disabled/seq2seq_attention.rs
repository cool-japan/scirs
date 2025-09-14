use ndarray::{s, Array1, Array2, Array3};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32;

// Encoder module for Seq2Seq model
#[derive(Debug, Serialize, Deserialize)]
struct Encoder {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    batch_size: usize,
    bidirectional: bool,
    // Parameters for each layer
    lstm_cells: Vec<LSTMCell>,
    // State
    h_n: Option<Array3<f32>>, // [num_layers*directions, batch_size, hidden_size]
    c_n: Option<Array3<f32>>, // [num_layers*directions, batch_size, hidden_size]
}
impl Encoder {
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
        bidirectional: bool,
    ) -> Self {
        let mut lstm_cells = Vec::new();
        let direction_factor = if bidirectional { 2 } else { 1 };
        // Create LSTM cells for each layer and direction
        for layer in 0..num_layers {
            let layer_input_size = if layer == 0 {
                input_size
            } else {
                hidden_size * direction_factor
            };
            // Forward direction
            lstm_cells.push(LSTMCell::new(layer_input_size, hidden_size, batch_size));
            // Backward direction (if bidirectional)
            if bidirectional {
                lstm_cells.push(LSTMCell::new(layer_input_size, hidden_size, batch_size));
            }
        }
        Encoder {
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            bidirectional,
            lstm_cells,
            h_n: None,
            c_n: None,
    }
    fn forward(&mut self, x: &Array3<f32>) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        // x shape: [batch_size, seq_len, input_size]
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let directions = if self.bidirectional { 2 } else { 1 };
        // Initialize states
        let mut h_states =
            Array3::<f32>::zeros((self.num_layers * directions, batch_size, self.hidden_size));
        let mut c_states =
        // Store all hidden states for attention
        // shape: [batch_size, seq_len, hidden_size*directions]
        let mut all_hidden_states =
            Array3::<f32>::zeros((batch_size, seq_len, self.hidden_size * directions));
        // Process each layer
        for layer in 0..self.num_layers {
            let layer_idx = layer * directions;
            let mut layer_outputs = Array3::<f32>::zeros((batch_size, seq_len, self.hidden_size));
            // Initial states for this layer
            let mut h_t = Array2::<f32>::zeros((batch_size, self.hidden_size));
            let mut c_t = Array2::<f32>::zeros((batch_size, self.hidden_size));
            // Get input sequence for this layer
            let layer_input = if layer == 0 {
                x.clone()
                if self.bidirectional {
                    // Concatenate outputs from forward and backward passes of previous layer
                    let mut prev_outputs =
                        Array3::<f32>::zeros((batch_size, seq_len, self.hidden_size * 2));
                    for b in 0..batch_size {
                        for t in 0..seq_len {
                            for h in 0..self.hidden_size {
                                // Forward direction outputs
                                prev_outputs[[b, t, h]] = all_hidden_states[[b, t, h]];
                                // Backward direction outputs
                                prev_outputs[[b, t, h + self.hidden_size]] =
                                    all_hidden_states[[b, t, h + self.hidden_size]];
                            }
                        }
                    }
                    prev_outputs
                } else {
                    // Just use outputs from previous layer
                    all_hidden_states.clone()
                }
            // Reset LSTM cell for forward direction
            self.lstm_cells[layer_idx].reset_state();
            // Forward pass through the sequence
            for t in 0..seq_len {
                let x_t = layer_input.slice(s![.., t, ..]).to_owned();
                let (new_h, new_c) = self.lstm_cells[layer_idx].forward(&x_t);
                // Store hidden state for this timestep
                for b in 0..batch_size {
                    for h in 0..self.hidden_size {
                        layer_outputs[[b, t, h]] = new_h[[b, h]];
                h_t = new_h;
                c_t = new_c;
            // Store final states for this layer and direction
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    h_states[[layer_idx, b, h]] = h_t[[b, h]];
                    c_states[[layer_idx, b, h]] = c_t[[b, h]];
            // If bidirectional, process in backward direction
            if self.bidirectional {
                let back_idx = layer_idx + 1;
                let mut back_outputs =
                    Array3::<f32>::zeros((batch_size, seq_len, self.hidden_size));
                // Reset LSTM cell for backward direction
                self.lstm_cells[back_idx].reset_state();
                // Initial states for backward direction
                let mut h_t_back = Array2::<f32>::zeros((batch_size, self.hidden_size));
                let mut c_t_back = Array2::<f32>::zeros((batch_size, self.hidden_size));
                // Backward pass through the sequence
                for t in (0..seq_len).rev() {
                    let x_t = layer_input.slice(s![.., t, ..]).to_owned();
                    let (new_h, new_c) = self.lstm_cells[back_idx].forward(&x_t);
                    // Store hidden state for this timestep
                        for h in 0..self.hidden_size {
                            back_outputs[[b, t, h]] = new_h[[b, h]];
                    h_t_back = new_h;
                    c_t_back = new_c;
                // Store final states for this layer and backward direction
                        h_states[[back_idx, b, h]] = h_t_back[[b, h]];
                        c_states[[back_idx, b, h]] = c_t_back[[b, h]];
                // Combine outputs from forward and backward directions
                    for t in 0..seq_len {
                            // Forward direction
                            all_hidden_states[[b, t, h]] = layer_outputs[[b, t, h]];
                            // Backward direction
                            all_hidden_states[[b, t, h + self.hidden_size]] =
                                back_outputs[[b, t, h]];
                // If not bidirectional, just copy the forward outputs
        // Store final states
        self.h_n = Some(h_states.clone());
        self.c_n = Some(c_states.clone());
        (all_hidden_states, h_states, c_states)
// LSTM Cell implementation
struct LSTMCell {
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
impl LSTMCell {
    fn new(_input_size: usize, hidden_size: usize, batchsize: usize) -> Self {
        // Xavier/Glorot initialization
        let bound = (6.0 / (_input_size + hidden_size) as f32).sqrt();
        // Create a random number generator
        let mut rng = rand::rng();
        // Input gate weights
        let mut w_ii = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hi = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_i = Array1::zeros(hidden_size);
        // Initialize with random values
        for elem in w_ii.iter_mut() {
            *elem = rng.gen_range(-bound..bound);
        for elem in w_hi.iter_mut() {
        // Forget gate weights (initialize forget gate bias to 1 to avoid vanishing gradients early in training)
        let mut w_if = Array2::<f32>::zeros((hidden_size.._input_size));
        let mut w_hf = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_f = Array1::ones(hidden_size);
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
        LSTMCell {
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
    fn reset_state(&mut self) {
        self.h_t = None;
        self.c_t = None;
    fn forward(&mut self, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        // Initialize states if None
        if self.h_t.is_none() {
            self.h_t = Some(Array2::zeros((batch_size, self.hidden_size)));
        if self.c_t.is_none() {
            self.c_t = Some(Array2::zeros((batch_size, self.hidden_size)));
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
    // Activation functions
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    fn tanh(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.tanh())
// Attention mechanism
struct Attention {
    enc_hidden_size: usize,
    w_attn: Array2<f32>, // Attention weights
    v_attn: Array1<f32>, // Attention vector
impl Attention {
    fn new(_hidden_size: usize, enc_hiddensize: usize) -> Self {
        let bound = (6.0 / (hidden_size + enc_hidden_size) as f32).sqrt();
        let mut w_attn = Array2::<f32>::zeros((hidden_size + enc_hidden_size, hidden_size));
        for elem in w_attn.iter_mut() {
        let mut v_attn = Array1::<f32>::zeros(hidden_size);
        for elem in v_attn.iter_mut() {
        Attention {
            enc_hidden_size,
            w_attn,
            v_attn,
    fn forward(
        &self,
        hidden: &Array2<f32>,
        encoder_outputs: &Array3<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        // hidden: [batch_size, hidden_size]
        // encoder_outputs: [batch_size, src_len, enc_hidden_size]
        let batch_size = encoder_outputs.shape()[0];
        let src_len = encoder_outputs.shape()[1];
        // Create an expanded hidden state for each encoder output
        // [batch_size, src_len, hidden_size]
        let mut repeated_hidden = Array3::<f32>::zeros((batch_size, src_len, self.hidden_size));
        for b in 0..batch_size {
            for t in 0..src_len {
                    repeated_hidden[[b, t, h]] = hidden[[b, h]];
        // Concatenate hidden and encoder outputs
        // [batch_size, src_len, hidden_size + enc_hidden_size]
        let mut concat =
            Array3::<f32>::zeros((batch_size, src_len, self.hidden_size + self.enc_hidden_size));
                    concat[[b, t, h]] = repeated_hidden[[b, t, h]];
                for h in 0..self.enc_hidden_size {
                    concat[[b, t, h + self.hidden_size]] = encoder_outputs[[b, t, h]];
        // Calculate energy
        let mut energy = Array3::<f32>::zeros((batch_size, src_len, self.hidden_size));
                let concat_slice = concat.slice(s![b, t, ..]).to_owned();
                let energy_t = concat_slice.dot(&self.w_attn);
                    energy[[b, t, h]] = energy_t[h];
        // Apply tanh
        energy.mapv_inplace(|v| v.tanh());
        // Calculate attention scores
        // [batch_size, src_len]
        let mut attention = Array2::<f32>::zeros((batch_size, src_len));
                let energy_slice = energy.slice(s![b, t, ..]).to_owned();
                attention[[b, t]] = energy_slice.dot(&self.v_attn);
        // Apply softmax to get attention weights
        let attention_weights = Self::softmax_by_row(&attention);
        // Calculate context vector as weighted sum of encoder outputs
        // [batch_size, enc_hidden_size]
        let mut context = Array2::<f32>::zeros((batch_size, self.enc_hidden_size));
                let weight = attention_weights[[b, t]];
                    context[[b, h]] += weight * encoder_outputs[[b, t, h]];
        (context, attention_weights)
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
            // Normalize
            for (j, exp_val) in exp_vals.iter().enumerate() {
                result[[i, j]] = exp_val / sum;
        result
// Decoder module with attention
struct Decoder {
    input_size: usize,      // Size of embedded tokens
    hidden_size: usize,     // Decoder's hidden size
    output_size: usize,     // Vocabulary size
    enc_hidden_size: usize, // Encoder's hidden size
    // LSTM layers
    // Attention mechanism
    attention: Attention,
    // Output projection
    w_out: Array2<f32>,
    b_out: Array1<f32>,
    // States
    h_t: Option<Array3<f32>>, // [num_layers, batch_size, hidden_size]
    c_t: Option<Array3<f32>>, // [num_layers, batch_size, hidden_size]
impl Decoder {
        output_size: usize,
        enc_hidden_size: usize,
        // Create LSTM cells for each layer
        // First layer input includes both token embedding and context vector
        let first_layer_input_size = _input_size + enc_hidden_size;
        lstm_cells.push(LSTMCell::new(
            first_layer_input_size,
        ));
        // Other layers have regular input size
        for _ in 1..num_layers {
            lstm_cells.push(LSTMCell::new(hidden_size, hidden_size, batch_size));
        // Attention mechanism
        let attention = Attention::new(hidden_size, enc_hidden_size);
        // Output projection
        let output_bound = (6.0 / (hidden_size + output_size) as f32).sqrt();
        let mut w_out = Array2::<f32>::zeros((output_size, hidden_size + enc_hidden_size));
        for elem in w_out.iter_mut() {
            *elem = rng.gen_range(-output_bound..output_bound);
        let b_out = Array1::zeros(output_size);
        Decoder {
            output_size..attention,
            w_out,
            b_out,
    fn init_state(&mut self, encoder_state: &Array3<f32>, encodercell: &Array3<f32>) {
        // encoder_state: [num_layers*directions, batch_size, enc_hidden_size]
        // We'll use the final state of the encoder as our initial state
        // For bidirectional encoders, we'll use the forward direction's state
        let batch_size = encoder_state.shape()[1];
        let mut h_t = Array3::<f32>::zeros((self.num_layers, batch_size, self.hidden_size));
        let mut c_t = Array3::<f32>::zeros((self.num_layers, batch_size, self.hidden_size));
        // Initialize with encoder's final states
        // For each layer, we'll copy from the corresponding encoder layer
            // We only use forward direction states
            let encoder_layer = layer;
            // Copy state with projection if sizes don't match
                    // Simple copy or average if sizes match or encoder is larger
                    if self.hidden_size <= self.enc_hidden_size {
                        h_t[[layer, b, h]] = encoder_state[[encoder_layer, b, h]];
                        c_t[[layer, b, h]] = encoder_cell[[encoder_layer, b, h]];
                    } else {
                        // If decoder is larger, we need to pad or project
                        // Here we just copy with repetition (not ideal but simple)
                        let encoder_idx = h % self.enc_hidden_size;
                        h_t[[layer, b, h]] = encoder_state[[encoder_layer, b, encoder_idx]];
                        c_t[[layer, b, h]] = encoder_cell[[encoder_layer, b, encoder_idx]];
        self.h_t = Some(h_t);
        self.c_t = Some(c_t);
    fn forward_step(
        &mut self,
        input: &Array2<f32>,
        // input: [batch_size, input_size] - current token embedding
        let batch_size = input.shape()[0];
        // Get current states
        let h_t = self.h_t.as_ref().expect("Decoder state not initialized");
        let c_t = self.c_t.as_ref().expect("Decoder state not initialized");
        // Get top layer hidden state for attention
        let top_hidden = h_t.slice(s![self.num_layers - 1, .., ..]).to_owned();
        // Apply attention
        let (context, attn_weights) = self.attention.forward(&top_hidden, encoder_outputs);
        // Concatenate input embedding with context vector for first layer
        let mut combined_input =
            Array2::<f32>::zeros((batch_size, self.input_size + self.enc_hidden_size));
            // Copy input embedding
            for i in 0..self.input_size {
                combined_input[[b, i]] = input[[b, i]];
            // Copy context vector
            for i in 0..self.enc_hidden_size {
                combined_input[[b, i + self.input_size]] = context[[b, i]];
        // Process through LSTM layers
        let mut new_h = Array3::<f32>::zeros((self.num_layers, batch_size, self.hidden_size));
        let mut new_c = Array3::<f32>::zeros((self.num_layers, batch_size, self.hidden_size));
        let mut layer_input = combined_input;
            // Get previous states for this layer
            let h_prev = h_t.slice(s![layer, .., ..]).to_owned();
            let c_prev = c_t.slice(s![layer, .., ..]).to_owned();
            // Update LSTM cell state
            self.lstm_cells[layer].h_t = Some(h_prev);
            self.lstm_cells[layer].c_t = Some(c_prev);
            // Forward pass through this layer
            let (h_new, c_new) = self.lstm_cells[layer].forward(&layer_input);
            // Store new states
                    new_h[[layer, b, h]] = h_new[[b, h]];
                    new_c[[layer, b, h]] = c_new[[b, h]];
            // Output of this layer becomes input to the next
            layer_input = h_new;
        self.h_t = Some(new_h);
        self.c_t = Some(new_c);
        // Concatenate top layer hidden state with context for output projection
        let mut output_input =
            Array2::<f32>::zeros((batch_size, self.hidden_size + self.enc_hidden_size));
            // Copy hidden state
            for h in 0..self.hidden_size {
                output_input[[b, h]] = layer_input[[b, h]]; // layer_input now contains the output of the last layer
            for h in 0..self.enc_hidden_size {
                output_input[[b, h + self.hidden_size]] = context[[b, h]];
        // Final projection to vocabulary size
        let output = output_input.dot(&self.w_out.t()) + &self.b_out;
        (output, attn_weights)
        inputs: &Array3<f32>,
        teacher_forcing_ratio: f32,
    ) -> (Array3<f32>, Array3<f32>) {
        // inputs: [batch_size, tgt_len, input_size]
        let batch_size = inputs.shape()[0];
        let tgt_len = inputs.shape()[1];
        // Outputs and attention weights
        let mut outputs = Array3::<f32>::zeros((batch_size, tgt_len, self.output_size));
        let mut all_attn_weights = Array3::<f32>::zeros((batch_size, tgt_len, src_len));
        // First input is always the first token of the target sequence
        let mut input = inputs.slice(s![.., 0, ..]).to_owned();
        // Process each time step
        for t in 0..tgt_len {
            // Forward step
            let (output, attn_weights) = self.forward_step(&input, encoder_outputs);
            // Store outputs and attention weights
                for v in 0..self.output_size {
                    outputs[[b, t, v]] = output[[b, v]];
                for s in 0..src_len {
                    all_attn_weights[[b, t, s]] = attn_weights[[b, s]];
            // Prepare input for next time step
            if t < tgt_len - 1 {
                // Check for teacher forcing
                let use_teacher_forcing = rand::random::<f32>() < teacher_forcing_ratio;
                if use_teacher_forcing {
                    // Use ground truth as input
                    input = inputs.slice(s![.., t + 1, ..]).to_owned();
                    // Use model's prediction
                    // Apply softmax to get probabilities
                    let probs = Self::softmax_by_row(&output);
                    // Get argmax for each batch
                    let mut pred_tokens = Array2::<f32>::zeros((batch_size, self.input_size));
                        // Find max probability token
                        let mut max_idx = 0;
                        let mut max_val = probs[[b, 0]];
                        for v in 1..self.output_size {
                            if probs[[b, v]] > max_val {
                                max_idx = v;
                                max_val = probs[[b, v]];
                        // In a real implementation, we would convert the token ID to an embedding here
                        // For simplicity, we'll just use one-hot encoding
                        // This assumes input_size == output_size (shared vocabulary)
                        // In practice, you'd use an embedding layer
                        if max_idx < self.input_size {
                            pred_tokens[[b, max_idx]] = 1.0;
                    input = pred_tokens;
        (outputs, all_attn_weights)
// Seq2Seq model that combines encoder and decoder
struct Seq2SeqModel {
    encoder: Encoder,
    decoder: Decoder,
    src_vocab_size: usize,
    tgt_vocab_size: usize,
    embedding_dim: usize,
    // Embedding matrices
    src_embedding: Array2<f32>,
    tgt_embedding: Array2<f32>,
impl Seq2SeqModel {
        src_vocab_size: usize,
        tgt_vocab_size: usize,
        embedding_dim: usize,
        enc_layers: usize,
        dec_layers: usize,
        // Calculate encoder output size
        let enc_output_size = if bidirectional {
            hidden_size * 2
        } else {
            hidden_size
        };
        // Create encoder
        let encoder = Encoder::new(
            embedding_dim,
            enc_layers,
        );
        // Create decoder
        let decoder = Decoder::new(
            tgt_vocab_size,
            enc_output_size,
            dec_layers,
        // Create embedding matrices
        let src_bound = (3.0 / embedding_dim as f32).sqrt();
        let tgt_bound = (3.0 / embedding_dim as f32).sqrt();
        // Initialize embeddings with random values
        let mut src_embedding = Array2::<f32>::zeros((src_vocab_size, embedding_dim));
        for elem in src_embedding.iter_mut() {
            *elem = rng.gen_range(-src_bound..src_bound);
        let mut tgt_embedding = Array2::<f32>::zeros((tgt_vocab_size..embedding_dim));
        for elem in tgt_embedding.iter_mut() {
            *elem = rng.gen_range(-tgt_bound..tgt_bound);
        Seq2SeqModel {
            encoder..decoder,
            src_vocab_size,
            src_embedding,
            tgt_embedding,
        src_tokens: &Array2<usize>,
        tgt_tokens: &Array2<usize>,
        // src_tokens: [batch_size, src_len]
        // tgt_tokens: [batch_size, tgt_len]
        let batch_size = src_tokens.shape()[0];
        let src_len = src_tokens.shape()[1];
        let tgt_len = tgt_tokens.shape()[1];
        // Embed source tokens
        let mut src_embedded = Array3::<f32>::zeros((batch_size, src_len, self.embedding_dim));
                let token_id = src_tokens[[b, t]];
                if token_id < self.src_vocab_size {
                    for e in 0..self.embedding_dim {
                        src_embedded[[b, t, e]] = self.src_embedding[[token_id, e]];
        // Embed target tokens
        let mut tgt_embedded = Array3::<f32>::zeros((batch_size, tgt_len, self.embedding_dim));
            for t in 0..tgt_len {
                let token_id = tgt_tokens[[b, t]];
                if token_id < self.tgt_vocab_size {
                        tgt_embedded[[b, t, e]] = self.tgt_embedding[[token_id, e]];
        // Encode source sequence
        let (encoder_outputs, h_n, c_n) = self.encoder.forward(&src_embedded);
        // Initialize decoder state with encoder final state
        self.decoder.init_state(&h_n, &c_n);
        // Decode target sequence
        let (outputs, attention) =
            self.decoder
                .forward(&tgt_embedded, &encoder_outputs, teacher_forcing_ratio);
        (outputs, attention)
    fn translate(
        max_len: usize,
        sos_token: usize,
        eos_token: usize,
    ) -> (Vec<Vec<usize>>, Array3<f32>) {
        // Start with SOS token
        let mut input = Array2::<f32>::zeros((batch_size, self.embedding_dim));
            for e in 0..self.embedding_dim {
                input[[b, e]] = self.tgt_embedding[[sos_token, e]];
        // Store decoded token IDs and attention weights
        let mut decoded_tokens = vec![vec![sos_token]; batch_size];
        let mut all_attention = Array3::<f32>::zeros((batch_size, max_len, src_len));
        // Decode one token at a time
        for t in 0..max_len {
            // Generate next token
            let (output, attn_weights) = self.decoder.forward_step(&input, &encoder_outputs);
            // Apply softmax to get probabilities
            let probs = Decoder::softmax_by_row(&output);
            // Store attention weights
                    all_attention[[b, t, s]] = attn_weights[[b, s]];
            // Get argmax token for each batch
                // Skip if the sequence has already ended with EOS
                if decoded_tokens[b].len() > 0
                    && decoded_tokens[b][decoded_tokens[b].len() - 1] == eos_token
                {
                    continue;
                // Find max probability token
                let mut max_idx = 0;
                let mut max_val = probs[[b, 0]];
                for v in 1..self.tgt_vocab_size {
                    if probs[[b, v]] > max_val {
                        max_idx = v;
                        max_val = probs[[b, v]];
                // Add token to decoded sequence
                decoded_tokens[b].push(max_idx);
                // Prepare input for next time step
                // Embed the predicted token
                for e in 0..self.embedding_dim {
                    input[[b, e]] = self.tgt_embedding[[max_idx, e]];
            // Check if all sequences have ended with EOS
            let all_ended = decoded_tokens
                .iter()
                .all(|seq| seq.len() > 0 && seq[seq.len() - 1] == eos_token);
            if all_ended {
                break;
        // Remove the initial SOS token
        for seq in &mut decoded_tokens {
            if seq.len() > 1 {
                seq.remove(0);
        (decoded_tokens, all_attention)
// Create a mini parallel corpus for testing
#[allow(dead_code)]
fn create_mini_translation_corpus() -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    // Simple English-French pairs
    let pairs = vec![
        ("Hello world", "Bonjour le monde"),
        ("How are you", "Comment allez-vous"),
        ("I love programming", "J'aime la programmation"),
        ("This is a test", "C'est un test"),
        ("Good morning", "Bonjour"),
        ("What is your name", "Comment t'appelles-tu"),
        ("The weather is nice today", "Le temps est beau aujourd'hui"),
        ("I am learning Rust", "J'apprends Rust"),
        ("Thank you very much", "Merci beaucoup"),
        ("See you tomorrow", "À demain"),
    ];
    let mut english = Vec::new();
    let mut french = Vec::new();
    for (en, fr) in pairs {
        english.push(en.split_whitespace().map(|s| s.to_string()).collect());
        french.push(fr.split_whitespace().map(|s| s.to_string()).collect());
    (english, french)
// Create vocabularies and token mappings
#[allow(dead_code)]
fn create_vocabularies(
    src_sentences: &[Vec<String>],
    tgt_sentences: &[Vec<String>],
) -> (
    HashMap<String, usize>,
    HashMap<usize, String>,
) {
    let mut src_word_to_idx = HashMap::new();
    let mut src_idx_to_word = HashMap::new();
    let mut tgt_word_to_idx = HashMap::new();
    let mut tgt_idx_to_word = HashMap::new();
    // Special tokens
    let special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"];
    // Add special tokens to both vocabularies
    for (i, token) in special_tokens.iter().enumerate() {
        src_word_to_idx.insert(token.to_string(), i);
        src_idx_to_word.insert(i, token.to_string());
        tgt_word_to_idx.insert(token.to_string(), i);
        tgt_idx_to_word.insert(i, token.to_string());
    // Add source words
    let mut idx = special_tokens.len();
    for sentence in src_sentences {
        for word in sentence {
            if !src_word_to_idx.contains_key(word) {
                src_word_to_idx.insert(word.clone(), idx);
                src_idx_to_word.insert(idx, word.clone());
                idx += 1;
    // Add target words
    for sentence in tgt_sentences {
            if !tgt_word_to_idx.contains_key(word) {
                tgt_word_to_idx.insert(word.clone(), idx);
                tgt_idx_to_word.insert(idx, word.clone());
    (
        src_word_to_idx,
        src_idx_to_word,
        tgt_word_to_idx,
        tgt_idx_to_word,
    )
// Convert sentences to token IDs
#[allow(dead_code)]
fn tokenize_sentences(
    sentences: &[Vec<String>],
    word_to_idx: &HashMap<String, usize>,
    add_sos_eos: bool,
    max_len: usize,
) -> Array2<usize> {
    let sos_idx = word_to_idx.get("<sos>").unwrap_or(&0);
    let eos_idx = word_to_idx.get("<eos>").unwrap_or(&0);
    let pad_idx = word_to_idx.get("<pad>").unwrap_or(&0);
    let unk_idx = word_to_idx.get("<unk>").unwrap_or(&0);
    // Calculate sequence length
    let seq_len = if add_sos_eos {
        // Need space for SOS, tokens, EOS, and padding up to max_len
        max_len + 2
    } else {
        max_len
    };
    let mut tokens = Array2::<usize>::from_elem((sentences.len(), seq_len), *pad_idx);
    for (i, sentence) in sentences.iter().enumerate() {
        let mut pos = 0;
        // Add SOS token
        if add_sos_eos {
            tokens[[i, pos]] = *sos_idx;
            pos += 1;
        // Add tokens
        for word in sentence.iter().take(max_len) {
            tokens[[i, pos]] = *word_to_idx.get(word).unwrap_or(unk_idx);
        // Add EOS token
        if add_sos_eos && pos < seq_len {
            tokens[[i, pos]] = *eos_idx;
    tokens
// Training loop for Seq2Seq model
#[allow(dead_code)]
fn train_seq2seq(
    model: &mut Seq2SeqModel,
    src_tokens: &Array2<usize>,
    tgt_tokens: &Array2<usize>,
    num_epochs: usize, _learning_rate: f32,
    // Training parameters
    let teacher_forcing_ratio = 0.5;
    // Print vocabulary sizes
    println!("Source vocabulary size: {}", model.src_vocab_size);
    println!("Target vocabulary size: {}", model.tgt_vocab_size);
    // Training loop
    for epoch in 0..num_epochs {
        // Forward pass
        let (outputs_) = model.forward(src_tokens, tgt_tokens, teacher_forcing_ratio);
        // Calculate loss (cross-entropy)
        let batch_size = tgt_tokens.shape()[0];
        let mut total_loss = 0.0;
                let target = tgt_tokens[[b, t]];
                if target < model.tgt_vocab_size {
                    let output_t = outputs.slice(s![b, t, ..]).to_owned();
                    let mut max_val = output_t[0];
                    for &val in output_t.iter() {
                        if val > max_val {
                            max_val = val;
                    let mut sum_exp = 0.0;
                        sum_exp += (val - max_val).exp();
                    let log_prob = output_t[target] - max_val - sum_exp.ln();
                    total_loss -= log_prob;
        let avg_loss = total_loss / (batch_size * tgt_len) as f32;
        println!("Epoch {}/{} - Loss: {:.4}", epoch + 1, num_epochs, avg_loss);
        // In a real implementation, we would now:
        // 1. Compute gradients with backward pass
        // 2. Update parameters with optimizer
        // 3. Validate on a separate dataset
        // For simplicity, we're skipping the actual parameter updates
        // as implementing backpropagation for this complex model would require significant code
// Evaluate the model on test data
#[allow(dead_code)]
fn evaluate_seq2seq(
    src_idx_to_word: &HashMap<usize, String>,
    tgt_idx_to_word: &HashMap<usize, String>,
    let batch_size = src_tokens.shape()[0];
    let sos_idx = 1; // Assuming 1 is the index for SOS token
    let eos_idx = 2; // Assuming 2 is the index for EOS token
    // Generate translations
    let (translations_attention) = model.translate(src_tokens, 20, sos_idx, eos_idx);
    // Print some examples
    for b in 0..batch_size.min(3) {
        // Source sentence
        let mut src_sentence = String::new();
        for t in 0..src_tokens.shape()[1] {
            let token_id = src_tokens[[b, t]];
            if let Some(word) = src_idx_to_word.get(&token_id) {
                if word != "<pad>" && word != "<sos>" && word != "<eos>" {
                    src_sentence.push_str(word);
                    src_sentence.push(' ');
        // Reference translation
        let mut ref_sentence = String::new();
        for t in 0..tgt_tokens.shape()[1] {
            let token_id = tgt_tokens[[b, t]];
            if let Some(word) = tgt_idx_to_word.get(&token_id) {
                    ref_sentence.push_str(word);
                    ref_sentence.push(' ');
        // Model translation
        let mut model_sentence = String::new();
        for &token_id in &translations[b] {
            if token_id == eos_idx {
                    model_sentence.push_str(word);
                    model_sentence.push(' ');
        // Print results
        println!("Example {}:", b + 1);
        println!("  Source:   {}", src_sentence.trim());
        println!("  Reference: {}", ref_sentence.trim());
        println!("  Generated: {}", model_sentence.trim());
        println!();
#[allow(dead_code)]
fn main() {
    println!("Sequence-to-Sequence Model with Attention");
    println!("=========================================");
    // Create a mini translation dataset
    let (english, french) = create_mini_translation_corpus();
    // Create vocabularies
    let (en_word_to_idx, en_idx_to_word, fr_word_to_idx, fr_idx_to_word) =
        create_vocabularies(&english, &french);
    // Convert sentences to token IDs
    let max_len = 15;
    let src_tokens = tokenize_sentences(&english, &en_word_to_idx, false, max_len);
    let tgt_tokens = tokenize_sentences(&french, &fr_word_to_idx, true, max_len);
    // Model parameters
    let src_vocab_size = en_word_to_idx.len();
    let tgt_vocab_size = fr_word_to_idx.len();
    let embedding_dim = 32;
    let hidden_size = 64;
    let enc_layers = 1;
    let dec_layers = 1;
    let batch_size = english.len();
    let bidirectional = true;
    // Create model
    let mut model = Seq2SeqModel::new(
        src_vocab_size,
        tgt_vocab_size,
        embedding_dim,
        hidden_size,
        enc_layers,
        dec_layers,
        batch_size,
        bidirectional,
    );
    // Train the model
    println!("\nTraining the Seq2Seq model...");
    train_seq2seq(&mut model, &src_tokens, &tgt_tokens, 10, 0.001);
    // Evaluate the model
    println!("\nEvaluating the model...");
    evaluate_seq2seq(
        &mut model,
        &src_tokens,
        &tgt_tokens,
        &en_idx_to_word,
        &fr_idx_to_word,
    println!("\nSequence-to-Sequence with Attention model implementation completed!");
