use ndarray::{s, Array1, Array2, Array3, Axis};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f32;
use statrs::statistics::Statistics;

// Define a trait for recurrent layers
trait RecurrentLayer {
    // Forward pass takes input x and returns output
    // For RNNs, x has shape [batch_size, seq_len, input_size]
    // Output has shape [batch_size, seq_len, hidden_size]
    fn forward(&mut self, x: &Array3<f32>, istraining: bool) -> Array3<f32>;
    // Backward pass takes gradient from next layer and returns gradient to previous layer
    fn backward(&mut self, gradoutput: &Array3<f32>) -> Array3<f32>;
    // Update parameters with calculated gradients
    fn update_params(&mut self, learningrate: f32);
    // Reset hidden state
    fn reset_state(&mut self);
}
// Simple RNN implementation
#[derive(Debug, Serialize, Deserialize)]
struct SimpleRNN {
    input_size: usize,
    hidden_size: usize,
    batch_size: usize,
    // Parameters
    w_ih: Array2<f32>, // Input to hidden weights
    w_hh: Array2<f32>, // Hidden to hidden weights
    b_ih: Array1<f32>, // Input to hidden bias
    b_hh: Array1<f32>, // Hidden to hidden bias
    // Gradients
    dw_ih: Option<Array2<f32>>,
    dw_hh: Option<Array2<f32>>,
    db_ih: Option<Array1<f32>>,
    db_hh: Option<Array1<f32>>,
    // Hidden state
    h_t: Option<Array2<f32>>, // Current hidden state [batch_size, hidden_size]
    // Cache for backward pass
    inputs: Option<Array3<f32>>, // [batch_size, seq_len, input_size]
    hidden_states: Option<Array3<f32>>, // [batch_size, seq_len+1, hidden_size]
impl SimpleRNN {
    fn new(_input_size: usize, hidden_size: usize, batchsize: usize) -> Self {
        // Xavier/Glorot initialization for weights
        let bound = (6.0 / (_input_size + hidden_size) as f32).sqrt();
        // Create a random number generator
        let mut rng = rand::rng();
        // Initialize with random values
        let mut w_ih = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hh = Array2::<f32>::zeros((hidden_size, hidden_size));
        for elem in w_ih.iter_mut() {
            *elem = rng.gen_range(-bound..bound);
        }
        for elem in w_hh.iter_mut() {
        // Initialize biases to zero
        let b_ih = Array1::zeros(hidden_size);
        let b_hh = Array1::zeros(hidden_size);
        SimpleRNN {
            input_size..hidden_size,
            batch_size,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            dw_ih: None,
            dw_hh: None,
            db_ih: None,
            db_hh: None,
            h_t: None,
            inputs: None,
            hidden_states: None,
    }
    // Activation function and its derivative
    fn tanh(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.tanh())
    fn tanh_derivative(_tanhoutput: &Array2<f32>) -> Array2<f32> {
        // derivative of tanh(x) is 1 - tanh(x)^2
        tanh_output.mapv(|v| 1.0 - v * v)
impl RecurrentLayer for SimpleRNN {
    fn forward(&mut self, x: &Array3<f32>, istraining: bool) -> Array3<f32> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        // Initialize hidden state if None
        if self.h_t.is_none() {
            self.h_t = Some(Array2::zeros((batch_size, self.hidden_size)));
        // Store all hidden states, including initial state
        let mut all_hidden_states = Array3::zeros((batch_size, seq_len + 1, self.hidden_size));
        // Set initial hidden state
        if let Some(h_t) = &self.h_t {
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    all_hidden_states[[b, 0, h]] = h_t[[b, h]];
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
            // Get previous hidden state [batch_size, hidden_size]
            let h_prev = all_hidden_states.slice(s![.., t, ..]).to_owned();
            // Calculate new hidden state: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_prev + b_hh)
            // First: W_ih * x_t [batch_size, hidden_size]
            let wx = x_t.dot(&self.w_ih.t());
            // Second: W_hh * h_prev [batch_size, hidden_size]
            let wh = h_prev.dot(&self.w_hh.t());
            // Combine with biases and apply activation
            let mut h_t = wx + wh;
                    h_t[[b, h]] += self.b_ih[h] + self.b_hh[h];
            let h_t = Self::tanh(&h_t);
            // Store hidden state
                    all_hidden_states[[b, t + 1, h]] = h_t[[b, h]];
        // Update current hidden state with the last computed hidden state
        self.h_t = Some(all_hidden_states.slice(s![.., seq_len, ..]).to_owned());
        // Store for backward pass if in training mode
        if is_training {
            self.inputs = Some(x.clone());
            self.hidden_states = Some(all_hidden_states.clone());
        // Return all hidden states except the initial state
        all_hidden_states.slice(s![.., 1.., ..]).to_owned()
    fn backward(&mut self, gradoutput: &Array3<f32>) -> Array3<f32> {
        // Get cached values
        let inputs = self
            .inputs
            .as_ref()
            .expect("No cached inputs for backward pass");
        let hidden_states = self
            .hidden_states
            .expect("No cached hidden states for backward pass");
        let batch_size = inputs.shape()[0];
        let seq_len = inputs.shape()[1];
        // Initialize gradients
        let mut dw_ih = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_hh = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_ih = Array1::zeros(self.hidden_size);
        let mut db_hh = Array1::zeros(self.hidden_size);
        // Gradient of the input
        let mut dx = Array3::zeros(inputs.raw_dim());
        // Initialize hidden state gradient for the last time step
        let mut dh_next = Array2::zeros((batch_size, self.hidden_size));
        // Iterate backwards through time steps
        for t in (0..seq_len).rev() {
            // Add gradient from the output
            let mut dh = dh_next.clone();
                    dh[[b, h]] += grad_output[[b, t, h]];
            // Get current hidden state
            let h_t = hidden_states.slice(s![.., t + 1, ..]).to_owned();
            // Get previous hidden state
            let h_prev = hidden_states.slice(s![.., t, ..]).to_owned();
            // Get current input
            let x_t = inputs
            // Calculate gradients through tanh
            let dtanh = Self::tanh_derivative(&h_t) * &dh;
            // Update bias gradients
                    db_ih[h] += dtanh[[b, h]];
                    db_hh[h] += dtanh[[b, h]];
            // Update weight gradients
            dw_ih = dw_ih + dtanh.t().dot(&x_t);
            dw_hh = dw_hh + dtanh.t().dot(&h_prev);
            // Calculate gradient to previous hidden state
            dh_next = dtanh.dot(&self.w_hh);
            // Calculate gradient to input
            let dx_t = dtanh.dot(&self.w_ih);
                for i in 0..self.input_size {
                    dx[[b, t, i]] = dx_t[[b, i]];
        // Store gradients
        self.dw_ih = Some(dw_ih);
        self.dw_hh = Some(dw_hh);
        self.db_ih = Some(db_ih);
        self.db_hh = Some(db_hh);
        dx
    fn update_params(&mut self, learningrate: f32) {
        if let Some(dw_ih) = &self.dw_ih {
            self.w_ih = &self.w_ih - &(dw_ih * learning_rate);
        if let Some(dw_hh) = &self.dw_hh {
            self.w_hh = &self.w_hh - &(dw_hh * learning_rate);
        if let Some(db_ih) = &self.db_ih {
            self.b_ih = &self.b_ih - &(db_ih * learning_rate);
        if let Some(db_hh) = &self.db_hh {
            self.b_hh = &self.b_hh - &(db_hh * learning_rate);
    fn reset_state(&mut self) {
        self.h_t = None;
// LSTM implementation
struct LSTM {
    // Input gate
    w_ii: Array2<f32>, // Input to input gate
    w_hi: Array2<f32>, // Hidden to input gate
    b_i: Array1<f32>,  // Input gate bias
    // Forget gate
    w_if: Array2<f32>, // Input to forget gate
    w_hf: Array2<f32>, // Hidden to forget gate
    b_f: Array1<f32>,  // Forget gate bias
    // Cell gate
    w_ig: Array2<f32>, // Input to cell gate
    w_hg: Array2<f32>, // Hidden to cell gate
    b_g: Array1<f32>,  // Cell gate bias
    // Output gate
    w_io: Array2<f32>, // Input to output gate
    w_ho: Array2<f32>, // Hidden to output gate
    b_o: Array1<f32>,  // Output gate bias
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
    // Hidden and cell states
    c_t: Option<Array2<f32>>, // Current cell state [batch_size, hidden_size]
    inputs: Option<Array3<f32>>,      // [batch_size, seq_len, input_size]
    input_gates: Option<Array3<f32>>, // [batch_size, seq_len, hidden_size]
    forget_gates: Option<Array3<f32>>, // [batch_size, seq_len, hidden_size]
    cell_gates: Option<Array3<f32>>,  // [batch_size, seq_len, hidden_size]
    output_gates: Option<Array3<f32>>, // [batch_size, seq_len, hidden_size]
    cell_states: Option<Array3<f32>>, // [batch_size, seq_len+1, hidden_size]
impl LSTM {
        // Input gate weights
        let mut w_ii = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hi = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_i = Array1::zeros(hidden_size);
        for elem in w_ii.iter_mut() {
        for elem in w_hi.iter_mut() {
        // Forget gate weights (initialize with small positive bias to encourage remembering)
        let mut w_if = Array2::<f32>::zeros((hidden_size, input_size));
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
        LSTM {
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
            c_t: None,
            input_gates: None,
            forget_gates: None,
            cell_gates: None,
            output_gates: None,
            cell_states: None,
    // Activation functions and derivatives
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    fn sigmoid_derivative(_sigmoidoutput: &Array2<f32>) -> Array2<f32> {
        _sigmoid_output * &(1.0 - sigmoid_output)
        1.0 - tanh_output * tanh_output
impl RecurrentLayer for LSTM {
        // Initialize states if None
        if self.c_t.is_none() {
            self.c_t = Some(Array2::zeros((batch_size, self.hidden_size)));
        // Arrays to store values for each time step
        let mut all_cell_states = Array3::zeros((batch_size, seq_len + 1, self.hidden_size));
        let mut all_input_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_forget_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_cell_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_output_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        // Set initial hidden and cell states
        if let Some(c_t) = &self.c_t {
                    all_cell_states[[b, 0, h]] = c_t[[b, h]];
            // Get previous hidden and cell states
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
                    all_input_gates[[b, t, h]] = i_t[[b, h]];
                    all_forget_gates[[b, t, h]] = f_t[[b, h]];
                    all_cell_gates[[b, t, h]] = g_t[[b, h]];
                    all_output_gates[[b, t, h]] = o_t[[b, h]];
                    all_cell_states[[b, t + 1, h]] = c_t[[b, h]];
        // Update current states with the last computed values
        self.c_t = Some(all_cell_states.slice(s![.., seq_len, ..]).to_owned());
            self.input_gates = Some(all_input_gates.clone());
            self.forget_gates = Some(all_forget_gates.clone());
            self.cell_gates = Some(all_cell_gates.clone());
            self.output_gates = Some(all_output_gates.clone());
            self.cell_states = Some(all_cell_states.clone());
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
        // Initialize gradients for the last time step
        let mut dc_next = Array2::zeros((batch_size, self.hidden_size));
            // Get current timestep values
            let i_t = input_gates.slice(s![.., t, ..]).to_owned();
            let f_t = forget_gates.slice(s![.., t, ..]).to_owned();
            let g_t = cell_gates.slice(s![.., t, ..]).to_owned();
            let o_t = output_gates.slice(s![.., t, ..]).to_owned();
            let c_t = cell_states.slice(s![.., t + 1, ..]).to_owned();
            let c_prev = cell_states.slice(s![.., t, ..]).to_owned();
            // Get gradient from the output plus gradient from next timestep
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
            // Gradient for input x
            let dx_i = di_t.dot(&self.w_ii);
            let dx_f = df_t.dot(&self.w_if);
            let dx_g = dg_t.dot(&self.w_ig);
            let dx_o = do_t.dot(&self.w_io);
            let dx_t = dx_i + dx_f + dx_g + dx_o;
            // Gradient for previous hidden state
            let dh_i = di_t.dot(&self.w_hi);
            let dh_f = df_t.dot(&self.w_hf);
            let dh_g = dg_t.dot(&self.w_hg);
            let dh_o = do_t.dot(&self.w_ho);
            dh_next = dh_i + dh_f + dh_g + dh_o;
            // Gradient for previous cell state
            dc_next = &dc * &f_t;
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
        self.c_t = None;
// GRU implementation
struct GRU {
    // Reset gate
    w_ir: Array2<f32>, // Input to reset gate
    w_hr: Array2<f32>, // Hidden to reset gate
    b_r: Array1<f32>,  // Reset gate bias
    // Update gate
    w_iz: Array2<f32>, // Input to update gate
    w_hz: Array2<f32>, // Hidden to update gate
    b_z: Array1<f32>,  // Update gate bias
    // New gate
    w_in: Array2<f32>, // Input to new gate
    w_hn: Array2<f32>, // Hidden to new gate
    b_n: Array1<f32>,  // New gate bias
    dw_ir: Option<Array2<f32>>,
    dw_hr: Option<Array2<f32>>,
    db_r: Option<Array1<f32>>,
    dw_iz: Option<Array2<f32>>,
    dw_hz: Option<Array2<f32>>,
    db_z: Option<Array1<f32>>,
    dw_in: Option<Array2<f32>>,
    dw_hn: Option<Array2<f32>>,
    db_n: Option<Array1<f32>>,
    reset_gates: Option<Array3<f32>>, // [batch_size, seq_len, hidden_size]
    update_gates: Option<Array3<f32>>, // [batch_size, seq_len, hidden_size]
    new_gates: Option<Array3<f32>>,   // [batch_size, seq_len, hidden_size]
impl GRU {
        // Reset gate weights
        let mut w_ir = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hr = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_r = Array1::zeros(hidden_size);
        for elem in w_ir.iter_mut() {
        for elem in w_hr.iter_mut() {
        // Update gate weights
        let mut w_iz = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hz = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_z = Array1::zeros(hidden_size);
        for elem in w_iz.iter_mut() {
        for elem in w_hz.iter_mut() {
        // New gate weights
        let mut w_in = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hn = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_n = Array1::zeros(hidden_size);
        for elem in w_in.iter_mut() {
        for elem in w_hn.iter_mut() {
        GRU {
            w_ir,
            w_hr,
            b_r,
            w_iz,
            w_hz,
            b_z,
            w_in,
            w_hn,
            b_n,
            dw_ir: None,
            dw_hr: None,
            db_r: None,
            dw_iz: None,
            dw_hz: None,
            db_z: None,
            dw_in: None,
            dw_hn: None,
            db_n: None,
            reset_gates: None,
            update_gates: None,
            new_gates: None,
impl RecurrentLayer for GRU {
        let mut all_reset_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_update_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
        let mut all_new_gates = Array3::zeros((batch_size, seq_len, self.hidden_size));
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
                    all_reset_gates[[b, t, h]] = r_t[[b, h]];
                    all_update_gates[[b, t, h]] = z_t[[b, h]];
                    all_new_gates[[b, t, h]] = n_t[[b, h]];
            self.reset_gates = Some(all_reset_gates.clone());
            self.update_gates = Some(all_update_gates.clone());
            self.new_gates = Some(all_new_gates.clone());
        let reset_gates = self
            .reset_gates
            .expect("No cached reset gates for backward pass");
        let update_gates = self
            .update_gates
            .expect("No cached update gates for backward pass");
        let new_gates = self
            .new_gates
            .expect("No cached new gates for backward pass");
        let mut dw_ir = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_hr = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_r = Array1::zeros(self.hidden_size);
        let mut dw_iz = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_hz = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_z = Array1::zeros(self.hidden_size);
        let mut dw_in = Array2::zeros((self.hidden_size, self.input_size));
        let mut dw_hn = Array2::zeros((self.hidden_size, self.hidden_size));
        let mut db_n = Array1::zeros(self.hidden_size);
        // Initialize gradient for the last time step
            let r_t = reset_gates.slice(s![.., t, ..]).to_owned();
            let z_t = update_gates.slice(s![.., t, ..]).to_owned();
            let n_t = new_gates.slice(s![.., t, ..]).to_owned();
            let _h_t = hidden_states.slice(s![.., t + 1, ..]).to_owned();
            // Gradient of update gate: dz = dh * (h_prev - n_t) * sigmoid_derivative(z)
            let dz_t = &dh * &(&h_prev - &n_t) * &Self::sigmoid_derivative(&z_t);
            // Gradient of new gate: dn = dh * (1 - z_t) * tanh_derivative(n)
            let dn_t = &dh * &(1.0 - &z_t) * &Self::tanh_derivative(&n_t);
            // Gradient of reset gate: dr = dn * (W_hn * h_prev) * sigmoid_derivative(r)
            let dr_t = &dn_t * &h_prev.dot(&self.w_hn.t()) * &Self::sigmoid_derivative(&r_t);
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
            let dx_r = dr_t.dot(&self.w_ir);
            let dx_z = dz_t.dot(&self.w_iz);
            let dx_n = dn_t.dot(&self.w_in);
            let dx_t = dx_r + dx_z + dx_n;
            let dh_r = dr_t.dot(&self.w_hr);
            let dh_z = dz_t.dot(&self.w_hz);
            let dh_n = &dn_t * &r_t.dot(&self.w_hn);
            let dh_direct = &dh * &z_t;
            dh_next = dh_r + dh_z + dh_n + dh_direct;
        self.dw_ir = Some(dw_ir);
        self.dw_hr = Some(dw_hr);
        self.db_r = Some(db_r);
        self.dw_iz = Some(dw_iz);
        self.dw_hz = Some(dw_hz);
        self.db_z = Some(db_z);
        self.dw_in = Some(dw_in);
        self.dw_hn = Some(dw_hn);
        self.db_n = Some(db_n);
        // Update reset gate parameters
        if let Some(dw_ir) = &self.dw_ir {
            self.w_ir = &self.w_ir - &(dw_ir * learning_rate);
        if let Some(dw_hr) = &self.dw_hr {
            self.w_hr = &self.w_hr - &(dw_hr * learning_rate);
        if let Some(db_r) = &self.db_r {
            self.b_r = &self.b_r - &(db_r * learning_rate);
        // Update update gate parameters
        if let Some(dw_iz) = &self.dw_iz {
            self.w_iz = &self.w_iz - &(dw_iz * learning_rate);
        if let Some(dw_hz) = &self.dw_hz {
            self.w_hz = &self.w_hz - &(dw_hz * learning_rate);
        if let Some(db_z) = &self.db_z {
            self.b_z = &self.b_z - &(db_z * learning_rate);
        // Update new gate parameters
        if let Some(dw_in) = &self.dw_in {
            self.w_in = &self.w_in - &(dw_in * learning_rate);
        if let Some(dw_hn) = &self.dw_hn {
            self.w_hn = &self.w_hn - &(dw_hn * learning_rate);
        if let Some(db_n) = &self.db_n {
            self.b_n = &self.b_n - &(db_n * learning_rate);
// Example of using recurrent layers for a simple task - learning to predict the next element in a sequence
#[allow(dead_code)]
fn sequence_prediction_example() {
    // Create a simple sequence: [0, 1, 2, 3, 4, 5, ...]
    let seq_len = 10;
    let batch_size = 5;
    let input_size = 1;
    let hidden_size = 10;
    // Create input sequences and target outputs
    let mut inputs = Array3::<f32>::zeros((batch_size, seq_len, input_size));
    let mut targets = Array3::<f32>::zeros((batch_size, seq_len, input_size));
    for b in 0..batch_size {
        let offset = b as f32; // Different starting point for each sequence in batch
            inputs[[b, t, 0]] = t as f32 + offset;
            targets[[b, t, 0]] = (t + 1) as f32 + offset; // Next value in sequence
    // Normalize inputs and targets to range [0, 1] for better training
    let max_val = inputs.fold(0.0f32, |acc, &x| acc.max(x));
    inputs = inputs.mapv(|x| x / max_val);
    targets = targets.mapv(|x| x / max_val);
    // Create models
    let mut simple_rnn = SimpleRNN::new(input_size, hidden_size, batch_size);
    let mut lstm = LSTM::new(input_size, hidden_size, batch_size);
    let mut gru = GRU::new(input_size, hidden_size, batch_size);
    // Training parameters
    let learning_rate = 0.01;
    let num_epochs = 100;
    println!("Training models on sequence prediction task...");
    // Training loop
    for epoch in 0..num_epochs {
        // Train SimpleRNN
        let rnn_output = simple_rnn.forward(&inputs, true);
        let rnn_loss = (&rnn_output - &targets).mapv(|x| x * x).mean().unwrap();
        let rnn_grad = 2.0 * (&rnn_output - &targets) / (batch_size * seq_len) as f32;
        simple_rnn.backward(&rnn_grad);
        simple_rnn.update_params(learning_rate);
        // Train LSTM
        let lstm_output = lstm.forward(&inputs, true);
        let lstm_loss = (&lstm_output - &targets).mapv(|x| x * x).mean().unwrap();
        let lstm_grad = 2.0 * (&lstm_output - &targets) / (batch_size * seq_len) as f32;
        lstm.backward(&lstm_grad);
        lstm.update_params(learning_rate);
        // Train GRU
        let gru_output = gru.forward(&inputs, true);
        let gru_loss = (&gru_output - &targets).mapv(|x| x * x).mean().unwrap();
        let gru_grad = 2.0 * (&gru_output - &targets) / (batch_size * seq_len) as f32;
        gru.backward(&gru_grad);
        gru.update_params(learning_rate);
        // Print loss every 10 epochs
        if (epoch + 1) % 10 == 0 {
            println!(
                "Epoch {}/{} - RNN Loss: {:.6}, LSTM Loss: {:.6}, GRU Loss: {:.6}",
                epoch + 1,
                num_epochs,
                rnn_loss,
                lstm_loss,
                gru_loss
        // Reset states after each epoch
        simple_rnn.reset_state();
        lstm.reset_state();
        gru.reset_state();
    // Evaluation - prediction
    println!("\nEvaluating models on test sequence:");
    // Create a test sequence starting from a different point
    let mut test_input = Array3::<f32>::zeros((1, seq_len, input_size));
    for t in 0..seq_len {
        test_input[[0, t, 0]] = (t + 20) as f32 / max_val; // Start from 20
    // Make predictions
    simple_rnn.reset_state();
    lstm.reset_state();
    gru.reset_state();
    let rnn_pred = simple_rnn
        .forward(&test_input, false)
        .slice(s![0, seq_len - 1, 0])
        .to_owned();
    let lstm_pred = lstm
    let gru_pred = gru
    // True next value
    let true_next = (20 + seq_len) as f32 / max_val;
    println!(
        "Last input: {:.6}",
        test_input[[0, seq_len - 1, 0]] * max_val
    );
    println!("True next value: {:.6}", true_next * max_val);
    println!("SimpleRNN prediction: {:.6}", rnn_pred * max_val);
    println!("LSTM prediction: {:.6}", lstm_pred * max_val);
    println!("GRU prediction: {:.6}", gru_pred * max_val);
// Function to compare recurrent network performance on time series prediction
#[allow(dead_code)]
fn compare_recurrent_networks() {
    // Generate sine wave data
    let seq_len = 20;
    let hidden_size = 32;
    let batch_size = 10;
    let total_points = 1000;
    // Generate sine wave with some noise
    let mut data = Vec::with_capacity(total_points);
    for i in 0..total_points {
        let t = i as f32 * 0.1;
        let sin_val = (t).sin();
        // Add small noise
        let noise: f32 = rand::random::<f32>() * 0.1 - 0.05;
        data.push(sin_val + noise);
    // Normalize data to [0, 1]
    let max_val = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let data: Vec<f32> = data.iter().map(|&x| (x / max_val + 1.0) / 2.0).collect();
    // Create input/target sequences
    let mut all_inputs = Vec::new();
    let mut all_targets = Vec::new();
    for i in 0..(total_points - seq_len - 1) {
        let input_seq: Vec<f32> = data[i..(i + seq_len)].to_vec();
        let target_seq: Vec<f32> = data[(i + 1)..(i + seq_len + 1)].to_vec();
        all_inputs.push(input_seq);
        all_targets.push(target_seq);
    // Create batches
    let num_batches = all_inputs.len() / batch_size;
    let train_split = (num_batches as f32 * 0.8) as usize;
    let num_epochs = 50;
    println!("Training models on sine wave prediction task...");
        let mut rnn_total_loss = 0.0;
        let mut lstm_total_loss = 0.0;
        let mut gru_total_loss = 0.0;
        // Shuffle training data
        let mut indices: Vec<usize> = (0..train_split * batch_size).collect();
        indices.shuffle(&mut rng);
        for batch in 0..train_split {
            // Prepare batch
            let mut inputs = Array3::<f32>::zeros((batch_size, seq_len, input_size));
            let mut targets = Array3::<f32>::zeros((batch_size, seq_len, input_size));
            for i in 0..batch_size {
                let idx = indices[batch * batch_size + i];
                for t in 0..seq_len {
                    inputs[[i, t, 0]] = all_inputs[idx][t];
                    targets[[i, t, 0]] = all_targets[idx][t];
            // Train SimpleRNN
            let rnn_output = simple_rnn.forward(&inputs, true);
            let rnn_loss = (&rnn_output - &targets).mapv(|x| x * x).mean().unwrap();
            let rnn_grad = 2.0 * (&rnn_output - &targets) / (batch_size * seq_len) as f32;
            simple_rnn.backward(&rnn_grad);
            simple_rnn.update_params(learning_rate);
            rnn_total_loss += rnn_loss;
            // Train LSTM
            let lstm_output = lstm.forward(&inputs, true);
            let lstm_loss = (&lstm_output - &targets).mapv(|x| x * x).mean().unwrap();
            let lstm_grad = 2.0 * (&lstm_output - &targets) / (batch_size * seq_len) as f32;
            lstm.backward(&lstm_grad);
            lstm.update_params(learning_rate);
            lstm_total_loss += lstm_loss;
            // Train GRU
            let gru_output = gru.forward(&inputs, true);
            let gru_loss = (&gru_output - &targets).mapv(|x| x * x).mean().unwrap();
            let gru_grad = 2.0 * (&gru_output - &targets) / (batch_size * seq_len) as f32;
            gru.backward(&gru_grad);
            gru.update_params(learning_rate);
            gru_total_loss += gru_loss;
            // Reset states after each batch
            simple_rnn.reset_state();
            lstm.reset_state();
            gru.reset_state();
        // Calculate average loss
        let avg_rnn_loss = rnn_total_loss / train_split as f32;
        let avg_lstm_loss = lstm_total_loss / train_split as f32;
        let avg_gru_loss = gru_total_loss / train_split as f32;
        // Print loss every 5 epochs
        if (epoch + 1) % 5 == 0 {
                avg_rnn_loss,
                avg_lstm_loss,
                avg_gru_loss
    // Evaluation - calculate validation loss
    let mut rnn_val_loss = 0.0;
    let mut lstm_val_loss = 0.0;
    let mut gru_val_loss = 0.0;
    for batch in train_split..num_batches {
        // Prepare batch
        let mut inputs = Array3::<f32>::zeros((batch_size, seq_len, input_size));
        let mut targets = Array3::<f32>::zeros((batch_size, seq_len, input_size));
        for i in 0..batch_size {
            let idx = batch * batch_size + i;
            for t in 0..seq_len {
                inputs[[i, t, 0]] = all_inputs[idx][t];
                targets[[i, t, 0]] = all_targets[idx][t];
        // Evaluate SimpleRNN
        let rnn_output = simple_rnn.forward(&inputs, false);
        rnn_val_loss += rnn_loss;
        // Evaluate LSTM
        let lstm_output = lstm.forward(&inputs, false);
        lstm_val_loss += lstm_loss;
        // Evaluate GRU
        let gru_output = gru.forward(&inputs, false);
        gru_val_loss += gru_loss;
    // Calculate average validation loss
    let avg_rnn_val_loss = rnn_val_loss / (num_batches - train_split) as f32;
    let avg_lstm_val_loss = lstm_val_loss / (num_batches - train_split) as f32;
    let avg_gru_val_loss = gru_val_loss / (num_batches - train_split) as f32;
    println!("\nValidation Results:");
    println!("SimpleRNN MSE: {:.6}", avg_rnn_val_loss);
    println!("LSTM MSE: {:.6}", avg_lstm_val_loss);
    println!("GRU MSE: {:.6}", avg_gru_val_loss);
    // Generate predictions for next 50 points
    println!("\nGenerating predictions for future timesteps...");
    // Get the last sequence from the data as starting point
    let mut current_rnn_input = Array3::<f32>::zeros((1, seq_len, input_size));
    let mut current_lstm_input = Array3::<f32>::zeros((1, seq_len, input_size));
    let mut current_gru_input = Array3::<f32>::zeros((1, seq_len, input_size));
        let idx = total_points - seq_len - 1 + t;
        current_rnn_input[[0, t, 0]] = data[idx];
        current_lstm_input[[0, t, 0]] = data[idx];
        current_gru_input[[0, t, 0]] = data[idx];
    let future_steps = 50;
    let mut rnn_predictions = Vec::with_capacity(future_steps);
    let mut lstm_predictions = Vec::with_capacity(future_steps);
    let mut gru_predictions = Vec::with_capacity(future_steps);
    for _ in 0..future_steps {
        // Get predictions from each model
        let rnn_output = simple_rnn.forward(&current_rnn_input, false);
        let lstm_output = lstm.forward(&current_lstm_input, false);
        let gru_output = gru.forward(&current_gru_input, false);
        // Extract last prediction
        let rnn_pred = rnn_output[[0, seq_len - 1, 0]];
        let lstm_pred = lstm_output[[0, seq_len - 1, 0]];
        let gru_pred = gru_output[[0, seq_len - 1, 0]];
        // Store predictions
        rnn_predictions.push(rnn_pred);
        lstm_predictions.push(lstm_pred);
        gru_predictions.push(gru_pred);
        // Update inputs for next prediction (shift sequence and add new prediction)
        for t in 0..(seq_len - 1) {
            current_rnn_input[[0, t, 0]] = current_rnn_input[[0, t + 1, 0]];
            current_lstm_input[[0, t, 0]] = current_lstm_input[[0, t + 1, 0]];
            current_gru_input[[0, t, 0]] = current_gru_input[[0, t + 1, 0]];
        current_rnn_input[[0, seq_len - 1, 0]] = rnn_pred;
        current_lstm_input[[0, seq_len - 1, 0]] = lstm_pred;
        current_gru_input[[0, seq_len - 1, 0]] = gru_pred;
    // Convert normalized predictions back to original scale
    println!("\nSample predictions (denormalized):");
    for i in 0..5 {
        let rnn_val = (rnn_predictions[i] * 2.0 - 1.0) * max_val;
        let lstm_val = (lstm_predictions[i] * 2.0 - 1.0) * max_val;
        let gru_val = (gru_predictions[i] * 2.0 - 1.0) * max_val;
        println!(
            "Step {}: RNN: {:.6}, LSTM: {:.6}, GRU: {:.6}",
            i + 1,
            rnn_val,
            lstm_val,
            gru_val
        );
#[allow(dead_code)]
fn main() {
    println!("Recurrent Neural Network Layers Example");
    println!("=======================================");
    println!("\nExample 1: Simple Sequence Prediction");
    sequence_prediction_example();
    println!("\nExample 2: Sine Wave Prediction Comparison");
    compare_recurrent_networks();
