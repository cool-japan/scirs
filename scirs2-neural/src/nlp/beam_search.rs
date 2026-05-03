//! Beam search and sampling decoding strategies for sequence generation.
//!
//! Provides:
//! - [`BeamSearchDecoder`]: full beam search with length normalization
//! - [`greedy_decode`]: greedy (argmax) decoding
//! - [`top_k_sampling`]: top-k token sampling with temperature
//! - [`top_p_sampling`]: nucleus (top-p) sampling with temperature

use crate::error::NeuralError;
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use scirs2_core::random::{Rng, RngExt};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for beam search decoding.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams to keep during search.
    pub beam_width: usize,
    /// Maximum sequence length (including BOS token).
    pub max_length: usize,
    /// End-of-sequence token id.
    pub eos_token: usize,
    /// Padding token id.
    pub pad_token: usize,
    /// Length penalty alpha; score is divided by `len^alpha`.
    /// alpha = 0 → no penalty; alpha = 1 → linear length normalization.
    pub length_penalty: f64,
    /// If true, stop as soon as `beam_width` complete hypotheses are found.
    pub early_stopping: bool,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_length: 128,
            eos_token: 2,
            pad_token: 0,
            length_penalty: 0.6,
            early_stopping: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Hypothesis
// ---------------------------------------------------------------------------

/// A single beam search hypothesis.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token ids generated so far (including BOS).
    pub token_ids: Vec<usize>,
    /// Cumulative log-probability score.
    pub score: f64,
    /// Whether the hypothesis has emitted EOS.
    pub is_done: bool,
}

impl BeamHypothesis {
    /// Normalised score used for final ranking.
    ///
    /// `normalised = score / length^alpha`
    pub fn normalised_score(&self, alpha: f64) -> f64 {
        let len = self.token_ids.len().max(1) as f64;
        self.score / len.powf(alpha)
    }
}

// ---------------------------------------------------------------------------
// Beam search decoder
// ---------------------------------------------------------------------------

/// Beam search decoder.
pub struct BeamSearchDecoder {
    config: BeamSearchConfig,
}

impl BeamSearchDecoder {
    /// Create a new decoder with the given configuration.
    pub fn new(config: BeamSearchConfig) -> Self {
        Self { config }
    }

    /// Advance a single step.
    ///
    /// `hypotheses` – active (not-done) hypotheses, one per row of `log_probs`.
    /// `log_probs`  – shape `(active_beams, vocab_size)`.
    ///
    /// Returns `(new_hypotheses, all_done)`.
    pub fn step(
        &self,
        hypotheses: Vec<BeamHypothesis>,
        log_probs: &Array2<f64>,
    ) -> (Vec<BeamHypothesis>, bool) {
        let beam_width = self.config.beam_width;
        let n_beams = hypotheses.len();
        let vocab_size = log_probs.ncols();

        // Collect (extended_score, hyp_idx, token_id) for every candidate
        let mut candidates: Vec<(f64, usize, usize)> = Vec::with_capacity(n_beams * vocab_size);

        for (i, hyp) in hypotheses.iter().enumerate() {
            let row = log_probs.row(i);
            for (tok_id, &lp) in row.iter().enumerate() {
                candidates.push((hyp.score + lp, i, tok_id));
            }
        }

        // Keep top `2 * beam_width` candidates for diversity
        let keep = (2 * beam_width).min(candidates.len());
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(keep);

        let mut new_hyps: Vec<BeamHypothesis> = Vec::with_capacity(beam_width);

        for (score, hyp_idx, tok_id) in candidates {
            if new_hyps.len() >= beam_width {
                break;
            }
            let parent = &hypotheses[hyp_idx];
            let mut tokens = parent.token_ids.clone();
            tokens.push(tok_id);

            let is_done = tok_id == self.config.eos_token || tokens.len() >= self.config.max_length;

            new_hyps.push(BeamHypothesis {
                token_ids: tokens,
                score,
                is_done,
            });
        }

        let all_done = new_hyps.iter().all(|h| h.is_done);
        (new_hyps, all_done)
    }

    /// Run full beam search.
    ///
    /// `bos_token`  – token id for begin-of-sequence.
    /// `vocab_size` – number of tokens in the vocabulary.
    /// `score_fn`   – closure that takes a batch of sequences and returns
    ///                `Array2<f64>` of shape `(batch, vocab_size)` with log-probs
    ///                for the next token.
    pub fn decode<F>(
        &self,
        bos_token: usize,
        vocab_size: usize,
        score_fn: F,
    ) -> Result<Vec<BeamHypothesis>, NeuralError>
    where
        F: Fn(&[Vec<usize>]) -> Result<Array2<f64>, NeuralError>,
    {
        if vocab_size == 0 {
            return Err(NeuralError::InvalidArgument(
                "vocab_size must be > 0".to_string(),
            ));
        }

        // Initialise with a single beam containing BOS
        let mut active: Option<Vec<BeamHypothesis>> = Some(vec![BeamHypothesis {
            token_ids: vec![bos_token],
            score: 0.0,
            is_done: false,
        }]);
        let mut finished: Vec<BeamHypothesis> = Vec::new();

        for _step_idx in 0..self.config.max_length {
            let current_active = match active.take() {
                Some(a) => a,
                None => break,
            };
            let active_seqs: Vec<Vec<usize>> =
                current_active.iter().map(|h| h.token_ids.clone()).collect();
            let log_probs = score_fn(&active_seqs)?;

            if log_probs.nrows() != current_active.len() || log_probs.ncols() != vocab_size {
                return Err(NeuralError::ShapeMismatch(format!(
                    "score_fn returned shape ({}, {}), expected ({}, {})",
                    log_probs.nrows(),
                    log_probs.ncols(),
                    current_active.len(),
                    vocab_size,
                )));
            }

            let (new_hyps, all_done) = self.step(current_active, &log_probs);

            let mut next_active: Vec<BeamHypothesis> = Vec::new();
            for hyp in new_hyps {
                if hyp.is_done {
                    finished.push(hyp);
                } else {
                    next_active.push(hyp);
                }
            }

            if self.config.early_stopping && finished.len() >= self.config.beam_width {
                // Keep remaining active hypotheses as fallback if finished is empty
                if finished.is_empty() {
                    finished.extend(next_active);
                }
                break;
            }

            if next_active.is_empty() || all_done {
                finished.extend(next_active);
                break;
            }

            active = Some(next_active);
        }

        // If no hypotheses were ever completed, use whatever active beams remain.
        if finished.is_empty() {
            if let Some(remaining) = active {
                finished = remaining;
            }
        }

        // Sort by normalised score (descending)
        let alpha = self.config.length_penalty;
        finished.sort_by(|a, b| {
            b.normalised_score(alpha)
                .partial_cmp(&a.normalised_score(alpha))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(finished)
    }

    /// Return the best hypothesis from a list (highest normalised score, i.e. first after sorting).
    pub fn best_hypothesis(hyps: &[BeamHypothesis]) -> Option<&BeamHypothesis> {
        hyps.first()
    }
}

// ---------------------------------------------------------------------------
// Greedy decoding
// ---------------------------------------------------------------------------

/// Greedy (argmax) decoding.
///
/// `score_fn` returns log-probabilities for the next token given the current
/// sequence (length `vocab_size`).
pub fn greedy_decode<F>(
    bos_token: usize,
    eos_token: usize,
    max_length: usize,
    vocab_size: usize,
    score_fn: F,
) -> Result<Vec<usize>, NeuralError>
where
    F: Fn(&[usize]) -> Result<Vec<f64>, NeuralError>,
{
    if vocab_size == 0 {
        return Err(NeuralError::InvalidArgument(
            "vocab_size must be > 0".to_string(),
        ));
    }

    let mut sequence = vec![bos_token];

    for _ in 0..max_length {
        let log_probs = score_fn(&sequence)?;
        if log_probs.len() != vocab_size {
            return Err(NeuralError::ShapeMismatch(format!(
                "score_fn returned {} values, expected {}",
                log_probs.len(),
                vocab_size,
            )));
        }

        // Argmax
        let next_token = log_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        sequence.push(next_token);

        if next_token == eos_token {
            break;
        }
    }

    Ok(sequence)
}

// ---------------------------------------------------------------------------
// Top-k sampling
// ---------------------------------------------------------------------------

/// Top-k sampling with temperature.
///
/// At each step:
/// 1. Apply temperature to logits.
/// 2. Keep only the `k` highest-probability tokens.
/// 3. Re-normalise and sample.
pub fn top_k_sampling<F>(
    bos_token: usize,
    eos_token: usize,
    max_length: usize,
    k: usize,
    temperature: f64,
    score_fn: F,
    rng_seed: u64,
) -> Result<Vec<usize>, NeuralError>
where
    F: Fn(&[usize]) -> Result<Vec<f64>, NeuralError>,
{
    if k == 0 {
        return Err(NeuralError::InvalidArgument(
            "k must be > 0 for top_k_sampling".to_string(),
        ));
    }
    if temperature <= 0.0 {
        return Err(NeuralError::InvalidArgument(
            "temperature must be > 0".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut sequence = vec![bos_token];

    for _ in 0..max_length {
        let log_probs = score_fn(&sequence)?;

        // Apply temperature and convert to probabilities
        let scaled: Vec<f64> = log_probs.iter().map(|&lp| lp / temperature).collect();
        let probs = softmax_vec(&scaled);

        // Select top-k indices by probability
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        // Re-normalise
        let sum: f64 = indexed.iter().map(|(_, p)| p).sum();
        if sum <= 0.0 {
            sequence.push(eos_token);
            break;
        }

        let sample_val: f64 = rng.random::<f64>() * sum;
        let mut cumulative = 0.0;
        let mut chosen = indexed[0].0;
        for (tok, p) in &indexed {
            cumulative += p;
            if sample_val <= cumulative {
                chosen = *tok;
                break;
            }
        }

        sequence.push(chosen);

        if chosen == eos_token {
            break;
        }
    }

    Ok(sequence)
}

// ---------------------------------------------------------------------------
// Top-p (nucleus) sampling
// ---------------------------------------------------------------------------

/// Top-p (nucleus) sampling with temperature.
///
/// At each step:
/// 1. Apply temperature.
/// 2. Sort tokens by descending probability.
/// 3. Keep the smallest set whose cumulative probability ≥ p.
/// 4. Re-normalise and sample.
pub fn top_p_sampling<F>(
    bos_token: usize,
    eos_token: usize,
    max_length: usize,
    p: f64,
    temperature: f64,
    score_fn: F,
    rng_seed: u64,
) -> Result<Vec<usize>, NeuralError>
where
    F: Fn(&[usize]) -> Result<Vec<f64>, NeuralError>,
{
    if !(0.0 < p && p <= 1.0) {
        return Err(NeuralError::InvalidArgument(
            "p must be in (0, 1] for top_p_sampling".to_string(),
        ));
    }
    if temperature <= 0.0 {
        return Err(NeuralError::InvalidArgument(
            "temperature must be > 0".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut sequence = vec![bos_token];

    for _ in 0..max_length {
        let log_probs = score_fn(&sequence)?;

        // Apply temperature and convert to probabilities
        let scaled: Vec<f64> = log_probs.iter().map(|&lp| lp / temperature).collect();
        let probs = softmax_vec(&scaled);

        // Sort by descending probability
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Nucleus: accumulate until cumulative prob >= p
        let mut nucleus: Vec<(usize, f64)> = Vec::new();
        let mut cumulative = 0.0;
        for (tok, prob) in indexed {
            nucleus.push((tok, prob));
            cumulative += prob;
            if cumulative >= p {
                break;
            }
        }

        // Re-normalise
        let sum: f64 = nucleus.iter().map(|(_, prob)| prob).sum();
        if sum <= 0.0 {
            sequence.push(eos_token);
            break;
        }

        let sample_val: f64 = rng.random::<f64>() * sum;
        let mut cumulative2 = 0.0;
        let mut chosen = nucleus[0].0;
        for (tok, prob) in &nucleus {
            cumulative2 += prob;
            if sample_val <= cumulative2 {
                chosen = *tok;
                break;
            }
        }

        sequence.push(chosen);

        if chosen == eos_token {
            break;
        }
    }

    Ok(sequence)
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Numerically stable softmax over a slice.
fn softmax_vec(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        let n = exps.len().max(1);
        return vec![1.0 / n as f64; exps.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // Helper: uniform log-probs
    fn uniform_score(seqs: &[Vec<usize>], vocab_size: usize) -> Result<Array2<f64>, NeuralError> {
        let lp = -(vocab_size as f64).ln();
        Ok(Array2::from_elem((seqs.len(), vocab_size), lp))
    }

    // --- BeamSearchConfig ---

    #[test]
    fn test_beam_config_defaults() {
        let cfg = BeamSearchConfig::default();
        assert_eq!(cfg.beam_width, 4);
        assert_eq!(cfg.max_length, 128);
        assert!(cfg.early_stopping);
    }

    // --- BeamHypothesis ---

    #[test]
    fn test_beam_hypothesis_normalised_score_alpha_zero() {
        let hyp = BeamHypothesis {
            token_ids: vec![0, 1, 2, 3],
            score: -4.0,
            is_done: false,
        };
        // alpha=0 → len^0 = 1 → normalised == score
        assert!((hyp.normalised_score(0.0) - (-4.0)).abs() < 1e-9);
    }

    #[test]
    fn test_beam_hypothesis_normalised_score_alpha_one() {
        let hyp = BeamHypothesis {
            token_ids: vec![0, 1, 2, 3],
            score: -4.0,
            is_done: false,
        };
        // alpha=1 → score / 4 = -1.0
        assert!((hyp.normalised_score(1.0) - (-1.0)).abs() < 1e-9);
    }

    // --- BeamSearchDecoder::step ---

    #[test]
    fn test_beam_step_returns_beam_width_hyps() {
        let cfg = BeamSearchConfig {
            beam_width: 3,
            max_length: 10,
            eos_token: 9,
            pad_token: 0,
            length_penalty: 0.6,
            early_stopping: true,
        };
        let decoder = BeamSearchDecoder::new(cfg);
        let hyps = vec![
            BeamHypothesis {
                token_ids: vec![1],
                score: 0.0,
                is_done: false,
            },
            BeamHypothesis {
                token_ids: vec![2],
                score: -0.5,
                is_done: false,
            },
            BeamHypothesis {
                token_ids: vec![3],
                score: -1.0,
                is_done: false,
            },
        ];
        let log_probs = Array2::from_elem((3, 10), -std::f64::consts::LN_10);
        let (new_hyps, _done) = decoder.step(hyps, &log_probs);
        assert_eq!(new_hyps.len(), 3);
    }

    #[test]
    fn test_beam_step_done_when_eos_emitted() {
        let eos = 5usize;
        let cfg = BeamSearchConfig {
            beam_width: 2,
            max_length: 50,
            eos_token: eos,
            pad_token: 0,
            length_penalty: 0.0,
            early_stopping: true,
        };
        let decoder = BeamSearchDecoder::new(cfg);
        let hyps = vec![BeamHypothesis {
            token_ids: vec![1],
            score: 0.0,
            is_done: false,
        }];
        // Make token `eos` have the highest log-prob
        let mut row = vec![-100.0f64; 10];
        row[eos] = 0.0;
        let log_probs = Array2::from_shape_vec((1, 10), row).expect("shape");
        let (new_hyps, _) = decoder.step(hyps, &log_probs);
        assert!(new_hyps.iter().any(|h| h.is_done));
    }

    // --- BeamSearchDecoder::decode ---

    #[test]
    fn test_beam_decode_returns_nonempty() {
        let vocab_size = 10;
        let cfg = BeamSearchConfig::default();
        let decoder = BeamSearchDecoder::new(cfg);
        let result = decoder
            .decode(1, vocab_size, |seqs| uniform_score(seqs, vocab_size))
            .expect("decode");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_beam_decode_zero_vocab_size_errors() {
        let cfg = BeamSearchConfig::default();
        let decoder = BeamSearchDecoder::new(cfg);
        assert!(decoder
            .decode(0, 0, |seqs: &[Vec<usize>]| Ok(Array2::zeros((
                seqs.len(),
                1
            ))))
            .is_err());
    }

    #[test]
    fn test_beam_decode_shape_mismatch_errors() {
        let vocab_size = 10;
        let cfg = BeamSearchConfig::default();
        let decoder = BeamSearchDecoder::new(cfg);
        let result = decoder.decode(1, vocab_size, |seqs| {
            Ok(Array2::zeros((seqs.len(), vocab_size + 1)))
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_best_hypothesis_is_first() {
        let hyps = vec![
            BeamHypothesis {
                token_ids: vec![1, 2],
                score: -1.0,
                is_done: true,
            },
            BeamHypothesis {
                token_ids: vec![1, 3],
                score: -2.0,
                is_done: true,
            },
        ];
        let best = BeamSearchDecoder::best_hypothesis(&hyps);
        assert_eq!(best.map(|h| h.score), Some(-1.0));
    }

    #[test]
    fn test_best_hypothesis_empty_returns_none() {
        assert!(BeamSearchDecoder::best_hypothesis(&[]).is_none());
    }

    // --- greedy_decode ---

    #[test]
    fn test_greedy_decode_stops_at_eos() {
        let eos = 3usize;
        let vocab_size = 5;
        let result = greedy_decode(0, eos, 100, vocab_size, |_seq| {
            let mut lp = vec![-100.0f64; vocab_size];
            lp[eos] = 0.0;
            Ok(lp)
        })
        .expect("greedy");
        assert_eq!(*result.last().expect("last"), eos);
    }

    #[test]
    fn test_greedy_decode_starts_with_bos() {
        let bos = 1usize;
        let vocab_size = 5;
        let result = greedy_decode(bos, 4, 10, vocab_size, |_| Ok(vec![-1.0f64; vocab_size]))
            .expect("greedy");
        assert_eq!(result[0], bos);
    }

    #[test]
    fn test_greedy_decode_zero_vocab_errors() {
        assert!(greedy_decode(0, 1, 10, 0, |_| Ok(vec![])).is_err());
    }

    #[test]
    fn test_greedy_decode_max_length_respected() {
        let vocab_size = 5;
        let max_length = 10;
        // EOS never emitted (token 99 not in vocab range, so token 0 wins)
        let result = greedy_decode(0, 99, max_length, vocab_size, |_| {
            Ok(vec![-1.0f64; vocab_size])
        })
        .expect("greedy");
        assert!(result.len() <= max_length + 1);
    }

    // --- top_k_sampling ---

    #[test]
    fn test_top_k_sampling_starts_with_bos() {
        let vocab_size = 10;
        let result =
            top_k_sampling(0, 9, 20, 3, 1.0, |_| Ok(vec![-1.0f64; vocab_size]), 42).expect("top_k");
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_top_k_sampling_stops_at_eos() {
        let eos = 5usize;
        let vocab_size = 10;
        // Make eos overwhelmingly likely via extreme log-prob
        let result = top_k_sampling(
            0,
            eos,
            50,
            1,
            1.0,
            |_| {
                let mut lp = vec![-1000.0f64; vocab_size];
                lp[eos] = 0.0;
                Ok(lp)
            },
            123,
        )
        .expect("top_k");
        assert_eq!(*result.last().expect("last"), eos);
    }

    #[test]
    fn test_top_k_sampling_zero_k_errors() {
        assert!(top_k_sampling(0, 1, 10, 0, 1.0, |_| Ok(vec![-1.0; 5]), 0).is_err());
    }

    #[test]
    fn test_top_k_sampling_nonpositive_temperature_errors() {
        assert!(top_k_sampling(0, 1, 10, 3, 0.0, |_| Ok(vec![-1.0; 5]), 0).is_err());
        assert!(top_k_sampling(0, 1, 10, 3, -1.0, |_| Ok(vec![-1.0; 5]), 0).is_err());
    }

    #[test]
    fn test_top_k_sampling_reproducible_with_same_seed() {
        let vocab_size = 20;
        let result1 =
            top_k_sampling(0, 19, 30, 5, 0.8, |_| Ok(vec![-1.0f64; vocab_size]), 777).expect("r1");
        let result2 =
            top_k_sampling(0, 19, 30, 5, 0.8, |_| Ok(vec![-1.0f64; vocab_size]), 777).expect("r2");
        assert_eq!(result1, result2);
    }

    // --- top_p_sampling ---

    #[test]
    fn test_top_p_sampling_starts_with_bos() {
        let vocab_size = 10;
        let result = top_p_sampling(0, 9, 20, 0.9, 1.0, |_| Ok(vec![-1.0f64; vocab_size]), 42)
            .expect("top_p");
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_top_p_sampling_stops_at_eos() {
        let eos = 4usize;
        let vocab_size = 10;
        let result = top_p_sampling(
            0,
            eos,
            50,
            0.95,
            1.0,
            |_| {
                let mut lp = vec![-1000.0f64; vocab_size];
                lp[eos] = 0.0;
                Ok(lp)
            },
            55,
        )
        .expect("top_p");
        assert_eq!(*result.last().expect("last"), eos);
    }

    #[test]
    fn test_top_p_sampling_invalid_p_errors() {
        assert!(top_p_sampling(0, 1, 10, 0.0, 1.0, |_| Ok(vec![-1.0; 5]), 0).is_err());
        assert!(top_p_sampling(0, 1, 10, 1.5, 1.0, |_| Ok(vec![-1.0; 5]), 0).is_err());
    }

    #[test]
    fn test_top_p_sampling_nonpositive_temperature_errors() {
        assert!(top_p_sampling(0, 1, 10, 0.9, 0.0, |_| Ok(vec![-1.0; 5]), 0).is_err());
    }

    #[test]
    fn test_top_p_sampling_reproducible_with_same_seed() {
        let vocab_size = 15;
        let r1 =
            top_p_sampling(0, 14, 20, 0.9, 1.0, |_| Ok(vec![-1.0f64; vocab_size]), 99).expect("r1");
        let r2 =
            top_p_sampling(0, 14, 20, 0.9, 1.0, |_| Ok(vec![-1.0f64; vocab_size]), 99).expect("r2");
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_softmax_vec_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = super::softmax_vec(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
