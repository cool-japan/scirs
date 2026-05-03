//! Neural NER: transformer encoder + linear tag projection.
//!
//! Uses a frozen transformer encoder to extract per-token contextual embeddings
//! and trains a linear projection to BIO tag logits with SGD.

use super::transformer_encoder::{TransformerEncoderConfig, TransformerTextEncoder};
use crate::error::{Result, TextError};
use scirs2_core::ndarray::{Array1, Array2};

// ─── NerTag ───────────────────────────────────────────────────────────────────

/// BIO NER tag set (IOB2 format, 9 tags).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NerTag {
    /// Outside any entity span.
    O,
    /// Begin-Person.
    BPer,
    /// Inside-Person.
    IPer,
    /// Begin-Organisation.
    BOrg,
    /// Inside-Organisation.
    IOrg,
    /// Begin-Location.
    BLoc,
    /// Inside-Location.
    ILoc,
    /// Begin-Miscellaneous.
    BMisc,
    /// Inside-Miscellaneous.
    IMisc,
}

impl NerTag {
    /// Number of tags in this set.
    pub const N: usize = 9;

    /// Convert an integer index to a [`NerTag`].
    pub fn from_idx(idx: usize) -> Result<Self> {
        match idx {
            0 => Ok(NerTag::O),
            1 => Ok(NerTag::BPer),
            2 => Ok(NerTag::IPer),
            3 => Ok(NerTag::BOrg),
            4 => Ok(NerTag::IOrg),
            5 => Ok(NerTag::BLoc),
            6 => Ok(NerTag::ILoc),
            7 => Ok(NerTag::BMisc),
            8 => Ok(NerTag::IMisc),
            other => Err(TextError::InvalidInput(format!(
                "Invalid NerTag index {other}; max is {}",
                NerTag::N - 1
            ))),
        }
    }

    /// Convert this tag to its integer index.
    pub fn to_idx(self) -> usize {
        match self {
            NerTag::O => 0,
            NerTag::BPer => 1,
            NerTag::IPer => 2,
            NerTag::BOrg => 3,
            NerTag::IOrg => 4,
            NerTag::BLoc => 5,
            NerTag::ILoc => 6,
            NerTag::BMisc => 7,
            NerTag::IMisc => 8,
        }
    }
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for [`NeuralNer`].
#[derive(Debug, Clone)]
pub struct NeuralNerConfig {
    /// Forwarded to the inner [`TransformerTextEncoder`].
    pub encoder_config: TransformerEncoderConfig,
    /// Number of output tags. Defaults to [`NerTag::N`] (9).
    pub n_tags: usize,
    /// SGD learning rate.
    pub learning_rate: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// PRNG seed for weight initialisation.
    pub seed: u64,
}

impl Default for NeuralNerConfig {
    fn default() -> Self {
        Self {
            encoder_config: TransformerEncoderConfig::default(),
            n_tags: NerTag::N,
            learning_rate: 0.01,
            epochs: 5,
            seed: 777,
        }
    }
}

// ─── NeuralNer ────────────────────────────────────────────────────────────────

/// Named Entity Recogniser: frozen transformer + trainable linear tag projection.
pub struct NeuralNer {
    encoder: TransformerTextEncoder,
    /// Tag projection weights `[hidden_size, n_tags]`.
    pub tag_projection: Array2<f32>,
    /// Tag projection bias `[n_tags]`.
    pub tag_bias: Array1<f32>,
    config: NeuralNerConfig,
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Row-wise softmax for a 2-D matrix (in-place).
fn softmax_rows(x: &mut Array2<f32>) {
    let (rows, cols) = x.dim();
    for i in 0..rows {
        let max_val = x.row(i).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for j in 0..cols {
            x[[i, j]] = (x[[i, j]] - max_val).exp();
            sum += x[[i, j]];
        }
        if sum > 0.0 {
            for j in 0..cols {
                x[[i, j]] /= sum;
            }
        }
    }
}

fn argmax_row(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn xavier2(rows: usize, cols: usize, seed: &mut u64) -> Array2<f32> {
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

impl NeuralNer {
    /// Create a new NER model.
    pub fn new(config: NeuralNerConfig) -> Result<Self> {
        if config.n_tags == 0 {
            return Err(TextError::InvalidInput("n_tags must be > 0".to_string()));
        }

        let encoder = TransformerTextEncoder::new(config.encoder_config.clone())?;
        let hidden = config.encoder_config.hidden_size;

        let mut seed = config.seed;
        let tag_projection = xavier2(hidden, config.n_tags, &mut seed);
        let tag_bias = Array1::zeros(config.n_tags);

        Ok(Self {
            encoder,
            tag_projection,
            tag_bias,
            config,
        })
    }

    /// Compute per-token logit matrix `[seq_len, n_tags]` (before softmax).
    fn logits(&self, tokens: &[usize]) -> Result<Array2<f32>> {
        let ctx = self.encoder.encode_tokens(tokens)?; // [seq, hidden]
        let logits = ctx.dot(&self.tag_projection) + &self.tag_bias; // [seq, n_tags]
        Ok(logits)
    }

    /// Predict the NER tag sequence for `tokens`.
    pub fn predict(&self, tokens: &[usize]) -> Result<Vec<NerTag>> {
        let logits = self.logits(tokens)?;
        let seq = logits.shape()[0];
        let mut tags = Vec::with_capacity(seq);
        for i in 0..seq {
            let row: Vec<f32> = logits.row(i).iter().cloned().collect();
            let idx = argmax_row(&row);
            tags.push(NerTag::from_idx(idx)?);
        }
        Ok(tags)
    }

    /// Train on `(token_seqs, tag_seqs)` pairs using token-level SGD.
    ///
    /// The encoder is frozen; only `tag_projection` and `tag_bias` are updated.
    /// Returns per-epoch mean cross-entropy loss.
    pub fn fit(&mut self, data: &[(Vec<usize>, Vec<usize>)]) -> Result<Vec<f32>> {
        if data.is_empty() {
            return Err(TextError::InvalidInput(
                "Training data is empty".to_string(),
            ));
        }

        let n_tags = self.config.n_tags;
        let lr = self.config.learning_rate;
        let mut epoch_losses = Vec::with_capacity(self.config.epochs);

        for _epoch in 0..self.config.epochs {
            let mut total_loss = 0.0_f32;
            let mut total_tokens = 0usize;

            for (tokens, tag_idxs) in data.iter() {
                if tokens.len() != tag_idxs.len() {
                    return Err(TextError::InvalidInput(
                        "Token and tag sequences must have the same length".to_string(),
                    ));
                }
                if tokens.is_empty() {
                    continue;
                }
                let seq = tokens.len();

                // Forward: get contextual embeddings [seq, hidden]
                let ctx = self.encoder.encode_tokens(tokens)?;

                // Compute logits [seq, n_tags]
                let mut logits = ctx.dot(&self.tag_projection) + &self.tag_bias;

                // Softmax in place
                softmax_rows(&mut logits); // now logits holds probabilities

                // Compute loss & gradient for each token
                for (t, &label) in tag_idxs.iter().enumerate() {
                    if label >= n_tags {
                        return Err(TextError::InvalidInput(format!(
                            "Tag index {label} out of range for n_tags {n_tags}"
                        )));
                    }
                    let prob_correct = logits[[t, label]].max(1e-12);
                    total_loss -= prob_correct.ln();
                    total_tokens += 1;

                    // Gradient: proba - one_hot
                    let hidden = self.config.encoder_config.hidden_size;
                    let emb_t: Vec<f32> = ctx.row(t).iter().cloned().collect();
                    let mut grad = vec![0.0_f32; n_tags];
                    for j in 0..n_tags {
                        grad[j] = logits[[t, j]];
                    }
                    grad[label] -= 1.0;

                    // Update tag_projection: [hidden, n_tags]
                    for i in 0..hidden {
                        for j in 0..n_tags {
                            self.tag_projection[[i, j]] -= lr * emb_t[i] * grad[j];
                        }
                    }

                    // Update tag_bias
                    for j in 0..n_tags {
                        self.tag_bias[j] -= lr * grad[j];
                    }
                }

                let _ = seq; // suppress warning
            }

            epoch_losses.push(if total_tokens > 0 {
                total_loss / total_tokens as f32
            } else {
                0.0
            });
        }

        Ok(epoch_losses)
    }

    /// Compute token-level micro-F1 on a test dataset.
    ///
    /// Ignores the `O` tag in numerator/denominator (entity-only F1).
    pub fn f1_score(&self, data: &[(Vec<usize>, Vec<usize>)]) -> Result<f32> {
        if data.is_empty() {
            return Err(TextError::InvalidInput("Test data is empty".to_string()));
        }

        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;

        for (tokens, gold_idxs) in data.iter() {
            let pred_tags = self.predict(tokens)?;
            for (pred, &gold) in pred_tags.iter().zip(gold_idxs.iter()) {
                let pred_idx = pred.to_idx();
                // Only count non-O tags
                let pred_entity = pred_idx != NerTag::O.to_idx();
                let gold_entity = gold != NerTag::O.to_idx();

                if pred_entity && gold_entity && pred_idx == gold {
                    tp += 1;
                } else if pred_entity && (!gold_entity || pred_idx != gold) {
                    fp += 1;
                } else if gold_entity && (!pred_entity || pred_idx != gold) {
                    fn_ += 1;
                }
            }
        }

        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f32 / (tp + fn_) as f32
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Ok(f1)
    }

    /// Access the inner encoder.
    pub fn encoder(&self) -> &TransformerTextEncoder {
        &self.encoder
    }

    /// Access the NER configuration.
    pub fn config(&self) -> &NeuralNerConfig {
        &self.config
    }
}
