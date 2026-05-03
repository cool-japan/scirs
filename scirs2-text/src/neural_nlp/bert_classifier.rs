//! BERT-style classification head fine-tuned on top of a frozen transformer encoder.
//!
//! The encoder is used purely as a feature extractor (frozen).  A linear
//! classification head `W ∈ ℝ^{hidden × n_classes}` is optimised with SGD
//! and softmax cross-entropy loss.

use super::transformer_encoder::{TransformerEncoderConfig, TransformerTextEncoder};
use crate::error::{Result, TextError};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for [`BertClassifier`].
#[derive(Debug, Clone)]
pub struct BertClassifierConfig {
    /// Config forwarded to the inner [`TransformerTextEncoder`].
    pub encoder_config: TransformerEncoderConfig,
    /// Number of output classes.
    pub num_classes: usize,
    /// Dropout applied to mean-pooled embeddings (unused at inference).
    pub dropout: f32,
    /// SGD learning rate.
    pub learning_rate: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// PRNG seed (used to initialise classifier head weights independently).
    pub seed: u64,
}

impl Default for BertClassifierConfig {
    fn default() -> Self {
        Self {
            encoder_config: TransformerEncoderConfig::default(),
            num_classes: 2,
            dropout: 0.1,
            learning_rate: 0.01,
            epochs: 10,
            batch_size: 8,
            seed: 123,
        }
    }
}

// ─── BertClassifier ───────────────────────────────────────────────────────────

/// Linear classification head on top of a frozen [`TransformerTextEncoder`].
pub struct BertClassifier {
    encoder: TransformerTextEncoder,
    /// Weight matrix `[hidden_size, num_classes]`.
    pub classifier_weights: Array2<f32>,
    /// Bias vector `[num_classes]`.
    pub classifier_bias: Array1<f32>,
    config: BertClassifierConfig,
}

// ─── helper ───────────────────────────────────────────────────────────────────

/// Multiply two matrices row-wise and add bias.
fn linear(x: &Array1<f32>, w: &Array2<f32>, b: &Array1<f32>) -> Array1<f32> {
    x.dot(w) + b
}

/// Softmax of a 1-D vector.
fn softmax1(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Array1<f32> = logits.mapv(|v| (v - max).exp());
    let sum = exp.sum();
    if sum > 0.0 {
        exp / sum
    } else {
        exp
    }
}

/// Xavier-style initialisation using a simple LCG.
fn xavier_vec(rows: usize, cols: usize, seed: &mut u64) -> Array2<f32> {
    let scale = (6.0_f32 / (rows + cols) as f32).sqrt();
    Array2::from_shape_fn((rows, cols), |_| {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (*seed >> 33) as f32 / (u32::MAX as f32);
        (v - 0.5) * 2.0 * scale
    })
}

// ─── impl ─────────────────────────────────────────────────────────────────────

impl BertClassifier {
    /// Construct a new classifier.
    pub fn new(config: BertClassifierConfig) -> Result<Self> {
        if config.num_classes < 2 {
            return Err(TextError::InvalidInput(
                "num_classes must be ≥ 2".to_string(),
            ));
        }

        let encoder = TransformerTextEncoder::new(config.encoder_config.clone())?;
        let hidden = config.encoder_config.hidden_size;

        let mut seed = config.seed;
        let classifier_weights = xavier_vec(hidden, config.num_classes, &mut seed);
        let classifier_bias = Array1::zeros(config.num_classes);

        Ok(Self {
            encoder,
            classifier_weights,
            classifier_bias,
            config,
        })
    }

    /// Predict class probabilities for a single token sequence.
    pub fn predict_proba(&self, tokens: &[usize]) -> Result<Array1<f32>> {
        let embedding = self.encoder.encode_sentence(tokens)?;
        let logits = linear(&embedding, &self.classifier_weights, &self.classifier_bias);
        Ok(softmax1(&logits))
    }

    /// Predict the most likely class index.
    pub fn predict(&self, tokens: &[usize]) -> Result<usize> {
        let proba = self.predict_proba(tokens)?;
        proba
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| TextError::InvalidInput("Empty probability vector".to_string()))
    }

    /// Fine-tune the classification head using SGD on (token sequences, labels).
    ///
    /// The encoder is **frozen**; only `classifier_weights` and `classifier_bias`
    /// are updated.  Returns per-epoch mean cross-entropy loss.
    pub fn fine_tune(&mut self, data: &[(Vec<usize>, usize)]) -> Result<Vec<f32>> {
        if data.is_empty() {
            return Err(TextError::InvalidInput(
                "Training data is empty".to_string(),
            ));
        }

        let n_classes = self.config.num_classes;
        let lr = self.config.learning_rate;
        let mut epoch_losses = Vec::with_capacity(self.config.epochs);

        for _epoch in 0..self.config.epochs {
            let mut total_loss = 0.0_f32;
            let mut count = 0usize;

            for (tokens, label) in data.iter() {
                let label = *label;
                if label >= n_classes {
                    return Err(TextError::InvalidInput(format!(
                        "Label {label} out of range for {n_classes} classes"
                    )));
                }

                // Forward
                let emb = self.encoder.encode_sentence(tokens)?;
                let logits = linear(&emb, &self.classifier_weights, &self.classifier_bias);
                let proba = softmax1(&logits);

                // Cross-entropy loss
                let prob_correct = proba[label].max(1e-12);
                total_loss -= prob_correct.ln();
                count += 1;

                // Gradient of softmax cross-entropy: dL/dlogits = proba - one_hot
                let mut grad_logits = proba.clone();
                grad_logits[label] -= 1.0;

                // Gradient w.r.t. weights: outer(emb, grad_logits)
                // classifier_weights[i, j] -= lr * emb[i] * grad_logits[j]
                let hidden = emb.len();
                for i in 0..hidden {
                    for j in 0..n_classes {
                        self.classifier_weights[[i, j]] -= lr * emb[i] * grad_logits[j];
                    }
                }

                // Gradient w.r.t. bias
                for j in 0..n_classes {
                    self.classifier_bias[j] -= lr * grad_logits[j];
                }
            }

            epoch_losses.push(if count > 0 {
                total_loss / count as f32
            } else {
                0.0
            });
        }

        Ok(epoch_losses)
    }

    /// Compute classification accuracy on a test dataset.
    pub fn accuracy(&self, data: &[(Vec<usize>, usize)]) -> Result<f32> {
        if data.is_empty() {
            return Err(TextError::InvalidInput("Test data is empty".to_string()));
        }
        let mut correct = 0usize;
        for (tokens, label) in data.iter() {
            if self.predict(tokens)? == *label {
                correct += 1;
            }
        }
        Ok(correct as f32 / data.len() as f32)
    }

    /// Access the embedded encoder.
    pub fn encoder(&self) -> &TransformerTextEncoder {
        &self.encoder
    }

    /// Access the classifier config.
    pub fn config(&self) -> &BertClassifierConfig {
        &self.config
    }

    /// Encode a batch of token sequences and return the pooled embedding matrix
    /// `[n_samples, hidden_size]` (useful for external evaluation).
    pub fn encode_batch(&self, sequences: &[Vec<usize>]) -> Result<Array2<f32>> {
        if sequences.is_empty() {
            return Err(TextError::InvalidInput("Empty batch".to_string()));
        }
        let hidden = self.config.encoder_config.hidden_size;
        let mut out = Array2::zeros((sequences.len(), hidden));
        for (i, tokens) in sequences.iter().enumerate() {
            let emb = self.encoder.encode_sentence(tokens)?;
            out.row_mut(i).assign(&emb);
        }
        Ok(out)
    }

    /// Predict classes for a batch and return class indices.
    pub fn predict_batch(&self, sequences: &[Vec<usize>]) -> Result<Vec<usize>> {
        sequences
            .iter()
            .map(|tokens| self.predict(tokens))
            .collect()
    }

    /// Mean-pool helper: mean of rows in `[seq, hidden]` over `Axis(0)`.
    #[allow(dead_code)]
    fn mean_pool(ctx: &Array2<f32>) -> Result<Array1<f32>> {
        ctx.mean_axis(Axis(0))
            .ok_or_else(|| TextError::InvalidInput("Cannot mean-pool empty context".to_string()))
    }
}
