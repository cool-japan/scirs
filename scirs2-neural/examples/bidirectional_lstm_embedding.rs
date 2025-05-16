use ndarray::{s, Array1, Array2, Array3};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32;

// Embedding layer
#[derive(Debug, Serialize, Deserialize)]
struct Embedding {
    vocab_size: usize,
    embedding_dim: usize,
    weight: Array2<f32>,
}

impl Embedding {
    fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Xavier/Glorot initialization
        let bound = (3.0 / embedding_dim as f32).sqrt();

        // Create a random number generator
        let mut rng = rand::rng();

        // Initialize with random values
        let mut weight = Array2::<f32>::zeros((vocab_size, embedding_dim));
        for elem in weight.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        Embedding {
            vocab_size,
            embedding_dim,
            weight,
        }
    }

    fn forward(&self, x: &Array2<usize>) -> Array3<f32> {
        // x: [batch_size, seq_len] - Input token IDs
        // Returns: [batch_size, seq_len, embedding_dim] - Embedded vectors

        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.embedding_dim));

        for b in 0..batch_size {
            for t in 0..seq_len {
                let token_id = x[[b, t]];
                if token_id < self.vocab_size {
                    for e in 0..self.embedding_dim {
                        output[[b, t, e]] = self.weight[[token_id, e]];
                    }
                }
            }
        }

        output
    }
}

// Bidirectional LSTM layer
#[derive(Debug, Serialize, Deserialize)]
struct BiLSTM {
    input_size: usize,
    hidden_size: usize,
    batch_size: usize,

    // Forward LSTM cell
    forward_cell: LSTMCell,

    // Backward LSTM cell
    backward_cell: LSTMCell,
}

impl BiLSTM {
    fn new(input_size: usize, hidden_size: usize, batch_size: usize) -> Self {
        // Create forward and backward LSTM cells
        let forward_cell = LSTMCell::new(input_size, hidden_size, batch_size);
        let backward_cell = LSTMCell::new(input_size, hidden_size, batch_size);

        BiLSTM {
            input_size,
            hidden_size,
            batch_size,
            forward_cell,
            backward_cell,
        }
    }

    fn forward(&mut self, x: &Array3<f32>) -> Array3<f32> {
        // x: [batch_size, seq_len, input_size]
        // Returns: [batch_size, seq_len, hidden_size*2] - Concatenated forward and backward hidden states

        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Reset cell states
        self.forward_cell.reset_state();
        self.backward_cell.reset_state();

        // Output arrays
        let mut forward_hidden = Array3::<f32>::zeros((batch_size, seq_len, self.hidden_size));
        let mut backward_hidden = Array3::<f32>::zeros((batch_size, seq_len, self.hidden_size));

        // Forward pass (left to right)
        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]).to_owned();
            let (h_t, _) = self.forward_cell.forward(&x_t);

            // Store hidden state
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    forward_hidden[[b, t, h]] = h_t[[b, h]];
                }
            }
        }

        // Backward pass (right to left)
        for t in (0..seq_len).rev() {
            let x_t = x.slice(s![.., t, ..]).to_owned();
            let (h_t, _) = self.backward_cell.forward(&x_t);

            // Store hidden state
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    backward_hidden[[b, t, h]] = h_t[[b, h]];
                }
            }
        }

        // Concatenate forward and backward hidden states
        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.hidden_size * 2));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for h in 0..self.hidden_size {
                    // Forward
                    output[[b, t, h]] = forward_hidden[[b, t, h]];
                    // Backward
                    output[[b, t, h + self.hidden_size]] = backward_hidden[[b, t, h]];
                }
            }
        }

        output
    }
}

// LSTM Cell (same as in previous examples)
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

        // Create a random number generator
        let mut rng = rand::rng();

        // Initialize with random values

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

        // Forget gate weights (initialize forget gate bias to 1 to avoid vanishing gradients early in training)
        let mut w_if = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hf = Array2::<f32>::zeros((hidden_size, hidden_size));
        for elem in w_if.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_hf.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        let b_f = Array1::ones(hidden_size);

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

// Dropout layer
#[allow(dead_code)]
#[derive(Debug)]
struct Dropout {
    p: f32, // Dropout probability
    mask: Option<Array2<f32>>,
    is_training: bool,
}

#[allow(dead_code)]
impl Dropout {
    fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be in [0, 1)");

        Dropout {
            p,
            mask: None,
            is_training: true,
        }
    }

    fn train(&mut self) {
        self.is_training = true;
    }

    fn eval(&mut self) {
        self.is_training = false;
    }

    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        if !self.is_training || self.p == 0.0 {
            return x.clone();
        }

        // Create binary mask (1 = keep, 0 = drop)
        let mut mask = Array2::<f32>::ones(x.raw_dim());
        for val in mask.iter_mut() {
            if ndarray_rand::rand::random::<f32>() < self.p {
                *val = 0.0;
            }
        }

        // Scale by 1/(1-p) to maintain expected value
        let scale = 1.0 / (1.0 - self.p);
        let result = x * &mask * scale;

        self.mask = Some(mask);
        result
    }
}

// Full model for text classification with BiLSTM and embeddings
#[derive(Debug, Serialize, Deserialize)]
struct BiLSTMClassifier {
    embedding: Embedding,
    bilstm: BiLSTM,

    // Dense layer for classification
    output_size: usize,
    w_out: Array2<f32>,
    b_out: Array1<f32>,

    // Attention mechanism (optional)
    use_attention: bool,
    w_attention: Option<Array2<f32>>,
    v_attention: Option<Array1<f32>>,
}

impl BiLSTMClassifier {
    fn new(
        vocab_size: usize,
        embedding_dim: usize,
        hidden_size: usize,
        output_size: usize,
        batch_size: usize,
        use_attention: bool,
    ) -> Self {
        // Create embedding layer
        let embedding = Embedding::new(vocab_size, embedding_dim);

        // Create BiLSTM layer
        let bilstm = BiLSTM::new(embedding_dim, hidden_size, batch_size);

        // Output layer
        let output_input_size = hidden_size * 2; // Concatenated forward and backward hidden states
        let output_bound = (6.0 / (output_input_size + output_size) as f32).sqrt();

        // Create a random number generator
        let mut rng = rand::rng();

        // Initialize output weights with random values
        let mut w_out = Array2::<f32>::zeros((output_size, output_input_size));
        for elem in w_out.iter_mut() {
            *elem = rng.random_range(-output_bound..output_bound);
        }
        let b_out = Array1::zeros(output_size);

        // Attention parameters (if used)
        let (w_attention, v_attention) = if use_attention {
            let attention_bound = (6.0 / (hidden_size * 2) as f32).sqrt();

            // Initialize attention weights with random values
            let mut w_att = Array2::<f32>::zeros((hidden_size * 2, hidden_size * 2));
            for elem in w_att.iter_mut() {
                *elem = rng.random_range(-attention_bound..attention_bound);
            }

            let mut v_att = Array1::<f32>::zeros(hidden_size * 2);
            for elem in v_att.iter_mut() {
                *elem = rng.random_range(-attention_bound..attention_bound);
            }

            (Some(w_att), Some(v_att))
        } else {
            (None, None)
        };

        BiLSTMClassifier {
            embedding,
            bilstm,
            output_size,
            w_out,
            b_out,
            use_attention,
            w_attention,
            v_attention,
        }
    }

    fn forward(&mut self, x: &Array2<usize>) -> Array2<f32> {
        // x: [batch_size, seq_len] - Input token IDs
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Embed tokens
        let embedded = self.embedding.forward(x);

        // Process through BiLSTM
        let bilstm_output = self.bilstm.forward(&embedded);

        // Apply attention or get last hidden state
        let features = if self.use_attention {
            // Attention mechanism
            let w_attention = self.w_attention.as_ref().unwrap();
            let v_attention = self.v_attention.as_ref().unwrap();

            // Calculate attention weights
            let mut attention_scores = Array2::<f32>::zeros((batch_size, seq_len));
            let mut transformed = Array3::<f32>::zeros(bilstm_output.raw_dim());

            // Transform hidden states
            for b in 0..batch_size {
                for t in 0..seq_len {
                    let h_t = bilstm_output.slice(s![b, t, ..]).to_owned();
                    let transformed_h = h_t.dot(w_attention);
                    for h in 0..transformed_h.len() {
                        transformed[[b, t, h]] = transformed_h[h];
                    }
                }
            }

            // Apply tanh
            transformed.mapv_inplace(|v| v.tanh());

            // Calculate attention scores
            for b in 0..batch_size {
                for t in 0..seq_len {
                    let h_t = transformed.slice(s![b, t, ..]).to_owned();
                    attention_scores[[b, t]] = h_t.dot(v_attention);
                }
            }

            // Apply softmax to get attention weights
            let attention_weights = Self::softmax_by_row(&attention_scores);

            // Weighted sum of hidden states
            let mut context = Array2::<f32>::zeros((batch_size, bilstm_output.shape()[2]));
            for b in 0..batch_size {
                for t in 0..seq_len {
                    let weight = attention_weights[[b, t]];
                    for h in 0..bilstm_output.shape()[2] {
                        context[[b, h]] += weight * bilstm_output[[b, t, h]];
                    }
                }
            }

            context
        } else {
            // Just use the last hidden state from BiLSTM
            let mut last_hidden = Array2::<f32>::zeros((batch_size, bilstm_output.shape()[2]));
            for b in 0..batch_size {
                for h in 0..bilstm_output.shape()[2] {
                    last_hidden[[b, h]] = bilstm_output[[b, seq_len - 1, h]];
                }
            }

            last_hidden
        };

        // Output layer
        let logits = features.dot(&self.w_out.t()) + &self.b_out;

        // Apply softmax
        Self::softmax_by_row(&logits)
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

    fn predict(&mut self, x: &Array2<usize>) -> Array1<usize> {
        // Forward pass
        let probs = self.forward(x);

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

// Extended sentiment analysis dataset with more complex examples
fn create_sentiment_dataset() -> (Vec<Vec<String>>, Vec<usize>) {
    // Format: (text, sentiment) where sentiment is 0 (negative), 1 (neutral), or 2 (positive)
    let dataset = vec![
        // Positive examples
        ("this movie was absolutely amazing and I loved it", 2),
        (
            "the performances were outstanding and the direction was brilliant",
            2,
        ),
        (
            "I thoroughly enjoyed every minute of this entertaining film",
            2,
        ),
        ("the best movie I have seen in years, highly recommended", 2),
        (
            "fantastic screenplay with excellent character development",
            2,
        ),
        (
            "a masterpiece of modern cinema that will be remembered for years",
            2,
        ),
        ("stunning visuals and a captivating storyline throughout", 2),
        ("heartwarming and uplifting, exactly what I needed", 2),
        ("the perfect blend of comedy and drama with great acting", 2),
        ("innovative and refreshing take on a classic genre", 2),
        // Neutral examples
        ("the movie was okay but nothing special", 1),
        ("some good moments but also some boring parts", 1),
        ("decent acting but the plot was predictable", 1),
        ("not bad but not great either, just average", 1),
        ("watchable but forgettable, nothing stood out", 1),
        ("had potential but didn't quite deliver on its promises", 1),
        (
            "standard fare for this genre, neither impressive nor terrible",
            1,
        ),
        ("the performances were mixed, some good some mediocre", 1),
        ("visually appealing but lacking in substance overall", 1),
        (
            "an acceptable way to spend two hours but wouldn't watch again",
            1,
        ),
        // Negative examples
        ("this was the worst movie I have ever seen", 0),
        ("terrible acting and a completely nonsensical plot", 0),
        ("a waste of time and money, extremely disappointing", 0),
        ("boring from start to finish with no redeeming qualities", 0),
        ("the dialogue was cringe worthy and the pacing was awful", 0),
        (
            "poorly directed with plot holes you could drive a truck through",
            0,
        ),
        ("an incoherent mess that should never have been made", 0),
        ("painfully bad acting combined with a ridiculous script", 0),
        ("avoid this movie at all costs, absolutely dreadful", 0),
        ("complete disaster from beginning to end, truly terrible", 0),
    ];

    // Tokenize and prepare data
    let mut texts = Vec::new();
    let mut labels = Vec::new();

    for (text, label) in dataset {
        texts.push(
            text.split_whitespace()
                .map(|s| s.to_string().to_lowercase())
                .collect(),
        );
        labels.push(label);
    }

    (texts, labels)
}

// Create vocabulary from tokenized texts
fn create_vocabulary(texts: &[Vec<String>]) -> (HashMap<String, usize>, HashMap<usize, String>) {
    let mut word_to_idx = HashMap::new();
    let mut idx_to_word = HashMap::new();

    // Special tokens
    let special_tokens = ["<pad>", "<unk>"];

    // Add special tokens
    for (i, token) in special_tokens.iter().enumerate() {
        word_to_idx.insert(token.to_string(), i);
        idx_to_word.insert(i, token.to_string());
    }

    // Add words from texts
    let mut idx = special_tokens.len();
    for text in texts {
        for word in text {
            if !word_to_idx.contains_key(word) {
                word_to_idx.insert(word.clone(), idx);
                idx_to_word.insert(idx, word.clone());
                idx += 1;
            }
        }
    }

    (word_to_idx, idx_to_word)
}

// Convert texts to token IDs with padding
fn tokenize_texts(
    texts: &[Vec<String>],
    word_to_idx: &HashMap<String, usize>,
    max_len: usize,
) -> Array2<usize> {
    let pad_idx = *word_to_idx.get("<pad>").unwrap_or(&0);
    let unk_idx = *word_to_idx.get("<unk>").unwrap_or(&1);

    let mut tokens = Array2::<usize>::from_elem((texts.len(), max_len), pad_idx);

    for (i, text) in texts.iter().enumerate() {
        for (j, word) in text.iter().enumerate().take(max_len) {
            tokens[[i, j]] = *word_to_idx.get(word).unwrap_or(&unk_idx);
        }
    }

    tokens
}

// Convert labels to one-hot encoded vectors
fn one_hot_encode(labels: &[usize], num_classes: usize) -> Array2<f32> {
    let mut one_hot = Array2::<f32>::zeros((labels.len(), num_classes));

    for (i, &label) in labels.iter().enumerate() {
        if label < num_classes {
            one_hot[[i, label]] = 1.0;
        }
    }

    one_hot
}

// Shuffle the dataset
fn shuffle_dataset<T: Clone, U: Clone>(xs: &[T], ys: &[U]) -> (Vec<T>, Vec<U>) {
    assert_eq!(
        xs.len(),
        ys.len(),
        "Data and labels must have the same length"
    );

    let mut indices: Vec<usize> = (0..xs.len()).collect();
    let mut rng = ndarray_rand::rand::thread_rng();
    use ndarray_rand::rand::seq::SliceRandom;
    indices.shuffle(&mut rng);

    let mut shuffled_xs = Vec::with_capacity(xs.len());
    let mut shuffled_ys = Vec::with_capacity(ys.len());

    for &idx in &indices {
        shuffled_xs.push(xs[idx].clone());
        shuffled_ys.push(ys[idx].clone());
    }

    (shuffled_xs, shuffled_ys)
}

// Split dataset into training and testing sets
fn train_test_split<T: Clone, U: Clone>(
    xs: &[T],
    ys: &[U],
    test_ratio: f32,
) -> (Vec<T>, Vec<T>, Vec<U>, Vec<U>) {
    let test_size = (xs.len() as f32 * test_ratio).round() as usize;
    let train_size = xs.len() - test_size;

    let (shuffled_xs, shuffled_ys) = shuffle_dataset(xs, ys);

    let train_xs = shuffled_xs[..train_size].to_vec();
    let test_xs = shuffled_xs[train_size..].to_vec();
    let train_ys = shuffled_ys[..train_size].to_vec();
    let test_ys = shuffled_ys[train_size..].to_vec();

    (train_xs, test_xs, train_ys, test_ys)
}

// Train the BiLSTM classifier
fn train_model(
    model: &mut BiLSTMClassifier,
    x_train: &Array2<usize>,
    y_train: &Array2<f32>,
    num_epochs: usize,
    _learning_rate: f32,
) {
    // Training loop
    for epoch in 0..num_epochs {
        // Forward pass
        let output = model.forward(x_train);

        // Calculate loss (cross-entropy)
        let batch_size = x_train.shape()[0];
        let mut total_loss = 0.0;

        for b in 0..batch_size {
            for c in 0..model.output_size {
                if y_train[[b, c]] > 0.0 {
                    // If this is the target class
                    total_loss -= y_train[[b, c]] * output[[b, c]].ln();
                }
            }
        }

        let avg_loss = total_loss / batch_size as f32;

        // Calculate accuracy
        let mut correct = 0;
        for b in 0..batch_size {
            // Get predicted class (argmax of output)
            let mut pred_class = 0;
            let mut max_prob = output[[b, 0]];
            for c in 1..model.output_size {
                if output[[b, c]] > max_prob {
                    max_prob = output[[b, c]];
                    pred_class = c;
                }
            }

            // Get true class (argmax of one-hot y_train)
            let mut true_class = 0;
            let mut max_val = y_train[[b, 0]];
            for c in 1..model.output_size {
                if y_train[[b, c]] > max_val {
                    max_val = y_train[[b, c]];
                    true_class = c;
                }
            }

            if pred_class == true_class {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / batch_size as f32;

        println!(
            "Epoch {}/{} - Loss: {:.4} - Accuracy: {:.2}%",
            epoch + 1,
            num_epochs,
            avg_loss,
            accuracy * 100.0
        );

        // In a real implementation, we would now:
        // 1. Compute gradients with backward pass
        // 2. Update parameters with optimizer
        // 3. Validate on a separate dataset

        // For simplicity, we're skipping the actual parameter updates
        // as implementing backpropagation for this complex model would require significant code
    }
}

// Evaluate the model on test data
fn evaluate_model(model: &mut BiLSTMClassifier, x_test: &Array2<usize>, y_test: &[usize]) {
    let predictions = model.predict(x_test);

    // Calculate accuracy
    let mut correct = 0;
    for (pred, &true_label) in predictions.iter().zip(y_test.iter()) {
        if *pred == true_label {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / y_test.len() as f32;
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);

    // Print confusion matrix
    let num_classes = 3; // Negative, Neutral, Positive
    let mut confusion_matrix = Array2::<usize>::zeros((num_classes, num_classes));

    for (pred, &true_label) in predictions.iter().zip(y_test.iter()) {
        if true_label < num_classes && *pred < num_classes {
            confusion_matrix[[true_label, *pred]] += 1;
        }
    }

    println!("\nConfusion Matrix:");
    println!("             Predicted");
    println!("             Neg  Neu  Pos");
    println!(
        "Actual Neg | {:3}  {:3}  {:3}",
        confusion_matrix[[0, 0]],
        confusion_matrix[[0, 1]],
        confusion_matrix[[0, 2]]
    );
    println!(
        "       Neu | {:3}  {:3}  {:3}",
        confusion_matrix[[1, 0]],
        confusion_matrix[[1, 1]],
        confusion_matrix[[1, 2]]
    );
    println!(
        "       Pos | {:3}  {:3}  {:3}",
        confusion_matrix[[2, 0]],
        confusion_matrix[[2, 1]],
        confusion_matrix[[2, 2]]
    );

    // Calculate precision, recall and F1 score for each class
    println!("\nClassification Report:");
    println!("         Precision  Recall  F1-Score");

    for c in 0..num_classes {
        let true_positives = confusion_matrix[[c, c]];

        // Sum predicted positives (column sum)
        let mut predicted_positives = 0;
        for i in 0..num_classes {
            predicted_positives += confusion_matrix[[i, c]];
        }

        // Sum actual positives (row sum)
        let mut actual_positives = 0;
        for j in 0..num_classes {
            actual_positives += confusion_matrix[[c, j]];
        }

        // Calculate metrics
        let precision = if predicted_positives > 0 {
            true_positives as f32 / predicted_positives as f32
        } else {
            0.0
        };

        let recall = if actual_positives > 0 {
            true_positives as f32 / actual_positives as f32
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let class_name = match c {
            0 => "Negative",
            1 => "Neutral ",
            2 => "Positive",
            _ => "Unknown ",
        };

        println!(
            "{} | {:.4}    {:.4}  {:.4}",
            class_name, precision, recall, f1
        );
    }
}

// Example predictions
fn example_predictions(model: &mut BiLSTMClassifier, word_to_idx: &HashMap<String, usize>) {
    let examples = vec![
        "this is a great movie with amazing performances",
        "the movie was okay but nothing really stood out",
        "absolutely terrible waste of time and money",
        "incredible cinematography and wonderful acting",
        "somewhat bland but has a few good moments",
    ];

    // Tokenize examples
    let tokenized: Vec<Vec<String>> = examples
        .iter()
        .map(|text| {
            text.split_whitespace()
                .map(|s| s.to_string().to_lowercase())
                .collect()
        })
        .collect();

    // Convert to token IDs
    let max_len = 20;
    let x = tokenize_texts(&tokenized, word_to_idx, max_len);

    // Make predictions
    let predictions = model.predict(&x);

    println!("\nExample Predictions:");
    for (i, example) in examples.iter().enumerate() {
        let sentiment = match predictions[i] {
            0 => "Negative",
            1 => "Neutral",
            2 => "Positive",
            _ => "Unknown",
        };
        println!("Text: \"{}\"", example);
        println!("Predicted sentiment: {}\n", sentiment);
    }
}

fn main() {
    println!("Bidirectional LSTM with Embedding for Sentiment Analysis");
    println!("======================================================");

    // Create dataset
    let (texts, labels) = create_sentiment_dataset();

    // Create vocabulary
    let (word_to_idx, _idx_to_word) = create_vocabulary(&texts);
    let vocab_size = word_to_idx.len();

    println!("Vocabulary size: {}", vocab_size);

    // Split dataset
    let (train_texts, test_texts, train_labels, test_labels) =
        train_test_split(&texts, &labels, 0.2);

    // Tokenize texts
    let max_len = 30;
    let x_train = tokenize_texts(&train_texts, &word_to_idx, max_len);
    let x_test = tokenize_texts(&test_texts, &word_to_idx, max_len);

    // One-hot encode labels
    let num_classes = 3; // Negative, Neutral, Positive
    let y_train_onehot = one_hot_encode(&train_labels, num_classes);

    // Model parameters
    let embedding_dim = 32;
    let hidden_size = 64;
    let batch_size = train_texts.len();
    let use_attention = true;

    // Create model
    let mut model = BiLSTMClassifier::new(
        vocab_size,
        embedding_dim,
        hidden_size,
        num_classes,
        batch_size,
        use_attention,
    );

    // Train model
    println!(
        "\nTraining BiLSTM classifier with attention: {}",
        use_attention
    );
    train_model(&mut model, &x_train, &y_train_onehot, 20, 0.01);

    // Evaluate model
    println!("\nEvaluating model on test set:");
    evaluate_model(&mut model, &x_test, &test_labels);

    // Example predictions
    example_predictions(&mut model, &word_to_idx);
}
