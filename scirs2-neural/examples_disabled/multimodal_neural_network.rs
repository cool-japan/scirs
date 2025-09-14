use ndarray::{s, Array, Array1, Array2, Array3, Array4};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32;

// CNN Components for Image Processing
// ------------------------------------
// Convolutional Layer
#[derive(Debug, Serialize, Deserialize)]
struct Conv2D {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    // Parameters
    weight: Array4<f32>,
    bias: Array1<f32>,
}
impl Conv2D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        // Xavier/Glorot initialization
        let fan_in = in_channels * kernel_size * kernel_size;
        let fan_out = out_channels * kernel_size * kernel_size;
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        // Create a random number generator
        let mut rng = rand::rng();
        // Initialize with random values
        let mut weight =
            Array4::<f32>::zeros((out_channels, in_channels, kernel_size, kernel_size));
        for elem in weight.iter_mut() {
            *elem = rng.gen_range(-bound..bound);
        }
        let bias = Array::zeros(out_channels);
        Conv2D {
            in_channels..out_channels,
            kernel_size,
            stride,
            padding,
            weight,
            bias,
    }
    fn forward(&self, x: &Array4<f32>) -> Array4<f32> {
        // x: [batch_size, in_channels, height, width]
        let batch_size = x.shape()[0];
        let height = x.shape()[2];
        let width = x.shape()[3];
        // Calculate output dimensions with padding
        let out_height = (height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_width = (width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let mut output =
            Array4::<f32>::zeros((batch_size, self.out_channels, out_height, out_width));
        // Create padded input if necessary
        let padded_height = height + 2 * self.padding;
        let padded_width = width + 2 * self.padding;
        let mut padded_input =
            Array4::<f32>::zeros((batch_size, self.in_channels, padded_height, padded_width));
        // Copy input to padded input (center region)
        for b in 0..batch_size {
            for c in 0..self.in_channels {
                for h in 0..height {
                    for w in 0..width {
                        padded_input[[b, c, h + self.padding, w + self.padding]] = x[[b, c, h, w]];
                    }
                }
            }
        // Convolution operation
            for out_c in 0..self.out_channels {
                for out_h in 0..out_height {
                    for out_w in 0..out_width {
                        let h_start = out_h * self.stride;
                        let w_start = out_w * self.stride;
                        let mut sum = 0.0;
                        // Apply kernel
                        for in_c in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    let h_pos = h_start + kh;
                                    let w_pos = w_start + kw;
                                    sum += padded_input[[b, in_c, h_pos, w_pos]]
                                        * self.weight[[out_c, in_c, kh, kw]];
                                }
                            }
                        }
                        // Add bias
                        sum += self.bias[out_c];
                        // Store result
                        output[[b, out_c, out_h, out_w]] = sum;
        output
// MaxPooling Layer
struct MaxPool2D {
impl MaxPool2D {
    fn new(_kernelsize: usize, stride: usize) -> Self {
        MaxPool2D {
        // x: [batch_size, channels, height, width]
        let channels = x.shape()[1];
        // Calculate output dimensions
        let out_height = (height - self._kernel_size) / self.stride + 1;
        let out_width = (width - self._kernel_size) / self.stride + 1;
        let mut output = Array4::<f32>::zeros((batch_size, channels, out_height, out_width));
        // Pooling operation
            for c in 0..channels {
                        let mut max_val = f32::NEG_INFINITY;
                        // Find maximum value in the window
                        for kh in 0..self._kernel_size {
                            for kw in 0..self._kernel_size {
                                let h_pos = h_start + kh;
                                let w_pos = w_start + kw;
                                max_val = max_val.max(x[[b, c, h_pos, w_pos]]);
                        output[[b, c, out_h, out_w]] = max_val;
// ReLU Activation
#[allow(dead_code)]
fn relu(x: &Array4<f32>) -> Array4<f32> {
    x.mapv(|v| v.max(0.0))
// Flatten Layer (for connecting CNN to FC layers)
#[allow(dead_code)]
fn flatten(x: &Array4<f32>) -> Array2<f32> {
    let batch_size = x.shape()[0];
    let flat_size: usize = x.shape()[1..].iter().product();
    let mut output = Array2::<f32>::zeros((batch_size, flat_size));
    for b in 0..batch_size {
        let mut idx = 0;
        for c in 0..x.shape()[1] {
            for h in 0..x.shape()[2] {
                for w in 0..x.shape()[3] {
                    output[[b, idx]] = x[[b, c, h, w]];
                    idx += 1;
    output
// RNN Components for Text Processing
// ----------------------------------
// LSTM Cell for text processing
struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
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
impl LSTMCell {
    fn new(_input_size: usize, hiddensize: usize) -> Self {
        let bound = (6.0 / (input_size + hidden_size) as f32).sqrt();
        // Input gate weights
        let mut w_ii = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hi = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_i = Array1::zeros(hidden_size);
        for elem in w_ii.iter_mut() {
        for elem in w_hi.iter_mut() {
        // Forget gate weights (initialize forget gate bias to 1 to avoid vanishing gradients early in training)
        let mut w_if = Array2::<f32>::zeros((hidden_size, input_size));
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
            input_size,
            hidden_size,
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
    fn forward(
        &self,
        x: &Array2<f32>,
        h_prev: &Array2<f32>,
        c_prev: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        // x: [batch_size, input_size]
        // h_prev: [batch_size, hidden_size]
        // c_prev: [batch_size, hidden_size]
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
        (h_t, c_t)
    // Activation functions
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    fn tanh(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.tanh())
// LSTM layer for processing sequences
struct LSTM {
    cell: LSTMCell,
impl LSTM {
        let cell = LSTMCell::new(input_size, hidden_size);
        LSTM { cell }
    fn forward(&self, x: &Array3<f32>) -> Array2<f32> {
        // x: [batch_size, seq_len, input_size]
        let seq_len = x.shape()[1];
        // Initialize hidden and cell states
        let mut h_t = Array2::<f32>::zeros((batch_size, self.cell.hidden_size));
        let mut c_t = Array2::<f32>::zeros((batch_size, self.cell.hidden_size));
        // Process sequence
        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]).to_owned();
            let (new_h, new_c) = self.cell.forward(&x_t, &h_t, &c_t);
            h_t = new_h;
            c_t = new_c;
        // Return final hidden state
        h_t
// Shared Components
// ----------------
// Embedding layer
struct Embedding {
    vocab_size: usize,
    embedding_dim: usize,
    weight: Array2<f32>,
impl Embedding {
    fn new(_vocab_size: usize, embeddingdim: usize) -> Self {
        let bound = (3.0 / embedding_dim as f32).sqrt();
        let mut weight = Array2::<f32>::zeros((_vocab_size, embedding_dim));
        Embedding {
            vocab_size,
            embedding_dim,
    fn forward(&self, x: &Array2<usize>) -> Array3<f32> {
        // x: [batch_size, seq_len] - Input token IDs
        // Returns: [batch_size, seq_len, embedding_dim] - Embedded vectors
        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.embedding_dim));
            for t in 0..seq_len {
                let token_id = x[[b, t]];
                if token_id < self._vocab_size {
                    for e in 0..self.embedding_dim {
                        output[[b, t, e]] = self.weight[[token_id, e]];
// Fully connected layer
struct Linear {
    in_features: usize,
    out_features: usize,
impl Linear {
    fn new(_in_features: usize, outfeatures: usize) -> Self {
        let bound = (6.0 / (in_features + out_features) as f32).sqrt();
        let mut weight = Array2::<f32>::zeros((out_features, in_features));
        let bias = Array::zeros(out_features);
        Linear {
            in_features,
            out_features,
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // x: [batch_size, in_features]
        // Returns: [batch_size, out_features]
        let result = x.dot(&self.weight.t());
        // Add bias
        let mut output = result.clone();
        for i in 0..output.shape()[0] {
            for j in 0..output.shape()[1] {
                output[[i, j]] += self.bias[j];
// Softmax for classification
#[allow(dead_code)]
fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut result = Array2::<f32>::zeros(x.raw_dim());
    for (i, row) in x.outer_iter().enumerate() {
        // Find max for numerical stability
        let max_val = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        // Calculate exp and sum
        let mut sum = 0.0;
        let mut exp_vals = vec![0.0; row.len()];
        for (j, &val) in row.iter().enumerate() {
            let exp_val = (val - max_val).exp();
            exp_vals[j] = exp_val;
            sum += exp_val;
        // Normalize
        for (j, &exp_val) in exp_vals.iter().enumerate() {
            result[[i, j]] = exp_val / sum;
    result
// ReLU activation for FC layers
#[allow(dead_code)]
fn fc_relu(x: &Array2<f32>) -> Array2<f32> {
// MultiModal Model combining CNN and RNN
// --------------------------------------
struct MultiModalModel {
    // Image processing (CNN)
    conv1: Conv2D,
    conv2: Conv2D,
    pool: MaxPool2D,
    // Text processing (RNN)
    embedding: Embedding,
    lstm: LSTM,
    // Combined processing (FC)
    fc1: Linear,
    fc2: Linear,
    fc_out: Linear,
    // Dimensions
    image_feature_size: usize,
    text_feature_size: usize,
    combined_feature_size: usize,
    num_classes: usize,
impl MultiModalModel {
        image_channels: usize,
        image_height: usize,
        image_width: usize,
        vocab_size: usize,
        embedding_dim: usize,
        hidden_size: usize,
        num_classes: usize,
        // CNN layers
        let conv1 = Conv2D::new(image_channels, 16, 3, 1, 1);
        let conv2 = Conv2D::new(16, 32, 3, 1, 1);
        let pool = MaxPool2D::new(2, 2);
        // Calculate CNN output dimensions
        let conv1_out_h = image_height; // Same padding
        let conv1_out_w = image_width; // Same padding
        let pool1_out_h = conv1_out_h / 2;
        let pool1_out_w = conv1_out_w / 2;
        let conv2_out_h = pool1_out_h; // Same padding
        let conv2_out_w = pool1_out_w; // Same padding
        let pool2_out_h = conv2_out_h / 2;
        let pool2_out_w = conv2_out_w / 2;
        let image_feature_size = 32 * pool2_out_h * pool2_out_w;
        // RNN layers
        let embedding = Embedding::new(_vocab_size, embedding_dim);
        let lstm = LSTM::new(embedding_dim, hidden_size);
        let text_feature_size = hidden_size;
        // Combined feature size
        let combined_feature_size = image_feature_size + text_feature_size;
        // FC layers
        let fc1 = Linear::new(combined_feature_size, 128);
        let fc2 = Linear::new(128, 64);
        let fc_out = Linear::new(64, num_classes);
        MultiModalModel {
            conv1,
            conv2,
            pool,
            embedding,
            lstm,
            fc1,
            fc2,
            fc_out,
            image_feature_size,
            text_feature_size,
            combined_feature_size,
            num_classes,
    fn forward(&self, images: &Array4<f32>, text: &Array2<usize>) -> Array2<f32> {
        // Process images through CNN
        let conv1_out = self.conv1.forward(images);
        let relu1_out = relu(&conv1_out);
        let pool1_out = self.pool.forward(&relu1_out);
        let conv2_out = self.conv2.forward(&pool1_out);
        let relu2_out = relu(&conv2_out);
        let pool2_out = self.pool.forward(&relu2_out);
        // Flatten CNN output
        let image_features = flatten(&pool2_out);
        // Process text through RNN
        let embedded = self.embedding.forward(text);
        let text_features = self.lstm.forward(&embedded);
        // Combine features
        let batch_size = images.shape()[0];
        let mut combined_features = Array2::<f32>::zeros((batch_size, self.combined_feature_size));
            // Copy image features
            for i in 0..self.image_feature_size {
                combined_features[[b, i]] = image_features[[b, i]];
            // Copy text features
            for i in 0..self.text_feature_size {
                combined_features[[b, self.image_feature_size + i]] = text_features[[b, i]];
        // Fully connected layers
        let fc1_out = self.fc1.forward(&combined_features);
        let relu3_out = fc_relu(&fc1_out);
        let fc2_out = self.fc2.forward(&relu3_out);
        let relu4_out = fc_relu(&fc2_out);
        let logits = self.fc_out.forward(&relu4_out);
        // Apply softmax for probabilities
        softmax(&logits)
// Data Generation and Helper Functions
// -----------------------------------
// Generate synthetic image data (simple geometric shapes)
#[allow(dead_code)]
fn generate_synthetic_images(
    num_samples: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Array4<f32> {
    let mut images = Array4::<f32>::zeros((num_samples, channels, height, width));
    for n in 0..num_samples {
        let shape_type = n % 3; // 0: square, 1: circle, 2: triangle
        match shape_type {
            0 => {
                // Draw a square
                let size = height / 3;
                let x_offset = width / 4;
                let y_offset = height / 4;
                for c in 0..channels {
                    let color = match c {
                        0 => 1.0, // Red channel
                        1 => 0.0, // Green channel
                        2 => 0.0, // Blue channel
                        _ => 0.5,
                    };
                    for y in y_offset..(y_offset + size) {
                        for x in x_offset..(x_offset + size) {
                            images[[n, c, y, x]] = color;
            1 => {
                // Draw a "circle" (approximated in the pixel grid)
                let radius = height as f32 / 6.0;
                let center_x = width as f32 / 2.0;
                let center_y = height as f32 / 2.0;
                        0 => 0.0, // Red channel
                        1 => 1.0, // Green channel
                    for y in 0..height {
                        for x in 0..width {
                            let dx = x as f32 - center_x;
                            let dy = y as f32 - center_y;
                            let distance = (dx * dx + dy * dy).sqrt();
                            if distance <= radius {
                                images[[n, c, y, x]] = color;
            2 => {
                // Draw a triangle
                let center_x = width / 2;
                let top_y = height / 4;
                let bottom_y = 3 * height / 4;
                let left_x = width / 3;
                let right_x = 2 * width / 3;
                        2 => 1.0, // Blue channel
                    for y in top_y..bottom_y {
                        // Calculate the width of the triangle at this y-coordinate
                        let progress = (y - top_y) as f32 / (bottom_y - top_y) as f32;
                        let half_width = ((right_x - left_x) as f32 * progress / 2.0) as usize;
                        for x in (center_x - half_width)..(center_x + half_width) {
            _ => {}
    images
// Generate synthetic text descriptions
#[allow(dead_code)]
fn generate_synthetictext(
    _vocab_size: usize,
    max_len: usize,
) -> (Array2<usize>, HashMap<usize, String>) {
    let mut text_data = Array2::<usize>::zeros((num_samples, max_len));
    let mut word_map = HashMap::new();
    // Create a simple vocabulary
    word_map.insert(0, "<pad>".to_string());
    word_map.insert(1, "square".to_string());
    word_map.insert(2, "circle".to_string());
    word_map.insert(3, "triangle".to_string());
    word_map.insert(4, "red".to_string());
    word_map.insert(5, "green".to_string());
    word_map.insert(6, "blue".to_string());
    word_map.insert(7, "this".to_string());
    word_map.insert(8, "is".to_string());
    word_map.insert(9, "a".to_string());
        // Create a simple description sentence
        let (shape_word, color_word) = match shape_type {
            0 => (1, 4), // "square", "red"
            1 => (2, 5), // "circle", "green"
            2 => (3, 6), // "triangle", "blue"
            _ => (0, 0),
        };
        // Fill in the sentence: "this is a [color] [shape]"
        text_data[[n, 0]] = 7; // "this"
        text_data[[n, 1]] = 8; // "is"
        text_data[[n, 2]] = 9; // "a"
        text_data[[n, 3]] = color_word;
        text_data[[n, 4]] = shape_word;
        // Pad the rest with 0s (already initialized to 0)
    (text_data, word_map)
// Create labels for the synthetic data
#[allow(dead_code)]
fn generate_labels(_num_samples: usize, numclasses: usize) -> (Array2<f32>, Array1<usize>) {
    let mut one_hot = Array2::<f32>::zeros((_num_samples, num_classes));
    let mut indices = Array1::<usize>::zeros(_num_samples);
        let class = n % num_classes;
        one_hot[[n, class]] = 1.0;
        indices[n] = class;
    (one_hot, indices)
// Calculate accuracy
#[allow(dead_code)]
fn calculate_accuracy(predictions: &Array2<f32>, targets: &Array1<usize>) -> f32 {
    let _num_samples = predictions.shape()[0];
    let mut correct = 0;
        // Get predicted class (argmax)
        let mut max_idx = 0;
        let mut max_val = predictions[[n, 0]];
        for c in 1..predictions.shape()[1] {
            if predictions[[n, c]] > max_val {
                max_idx = c;
                max_val = predictions[[n, c]];
        // Check if prediction matches target
        if max_idx == targets[n] {
            correct += 1;
    correct as f32 / _num_samples as f32
#[allow(dead_code)]
fn main() {
    println!("Multi-Modal Neural Network (CNN + RNN) Example");
    println!("=============================================");
    // Configuration
    let _num_samples = 300;
    let image_channels = 3; // RGB
    let image_height = 32;
    let image_width = 32;
    let vocab_size = 10; // Simple vocabulary
    let embedding_dim = 16;
    let hidden_size = 32;
    let num_classes = 3; // Square, Circle, Triangle
    let maxtext_len = 10;
    // Generate synthetic data
    println!("Generating synthetic data...");
    let images = generate_synthetic_images(_num_samples, image_channels, image_height, image_width);
    let (text_data, word_map) = generate_synthetictext(_num_samples, vocab_size, maxtext_len);
    let (_, labels) = generate_labels(_num_samples, num_classes);
    // Split into train and test sets (80% train, 20% test)
    let train_size = (_num_samples as f32 * 0.8) as usize;
    let test_size = _num_samples - train_size;
    let train_images = images.slice(s![0..train_size, .., .., ..]).to_owned();
    let traintext = text_data.slice(s![0..train_size, ..]).to_owned();
    let train_labels = labels.slice(s![0..train_size]).to_owned();
    let test_images = images.slice(s![train_size.., .., .., ..]).to_owned();
    let testtext = text_data.slice(s![train_size.., ..]).to_owned();
    let test_labels = labels.slice(s![train_size..]).to_owned();
    println!(
        "Created dataset with {} training and {} test samples",
        train_size, test_size
    );
    // Create model
    println!("Creating multi-modal model...");
    let model = MultiModalModel::new(
        image_channels,
        image_height,
        image_width,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_classes,
    // Forward pass with train data (evaluation only, no training in this example)
    println!("Running forward pass on training data...");
    let train_preds = model.forward(&train_images, &traintext);
    let train_accuracy = calculate_accuracy(&train_preds, &train_labels);
    println!("Training accuracy: {:.2}%", train_accuracy * 100.0);
    // Forward pass with test data
    println!("\nRunning forward pass on test data...");
    let test_preds = model.forward(&test_images, &testtext);
    let test_accuracy = calculate_accuracy(&test_preds, &test_labels);
    println!("Test accuracy: {:.2}%", test_accuracy * 100.0);
    // Show some example predictions
    println!("\nExample predictions:");
    println!("-------------------");
    let num_examples = 5.min(test_size);
    for i in 0..num_examples {
        // Get class names (shapes)
        let true_class = match test_labels[i] {
            0 => "Square (Red)",
            1 => "Circle (Green)",
            2 => "Triangle (Blue)"_ => "Unknown",
        // Get predicted class
        let mut pred_class_idx = 0;
        let mut max_val = test_preds[[i, 0]];
        for c in 1..num_classes {
            if test_preds[[i, c]] > max_val {
                pred_class_idx = c;
                max_val = test_preds[[i, c]];
        let pred_class = match pred_class_idx {
        // Get text description
        let mut description = String::new();
        for t in 0..maxtext_len {
            let word_idx = testtext[[i, t]];
            if word_idx > 0 {
                // Skip padding
                if t > 0 {
                    description.push(' ');
                description.push_str(word_map.get(&word_idx).unwrap_or(&"<unk>".to_string()));
        println!("Example {}:", i + 1);
        println!("  Text: \"{}\"", description);
        println!("  True class: {}", true_class);
        println!(
            "  Predicted: {} (confidence: {:.2}%)",
            pred_class,
            max_val * 100.0
        );
            "  Probabilities: Square: {:.2}%, Circle: {:.2}%, Triangle: {:.2}%",
            test_preds[[i, 0]] * 100.0,
            test_preds[[i, 1]] * 100.0,
            test_preds[[i, 2]] * 100.0
        println!();
    println!("Multi-modal neural network example completed!");
