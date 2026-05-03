//! Language model evaluation utilities.
//!
//! Provides:
//! - [`perplexity`]: compute perplexity from per-token log-probabilities.
//! - [`bleu_score`]: corpus/sentence BLEU with brevity penalty.
//! - [`self_bleu`]: self-BLEU diversity measure.
//! - [`nll_loss`]: mean negative log-likelihood.
//! - [`lm_cross_entropy`]: cross-entropy for next-token LM outputs.

use crate::error::NeuralError;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Perplexity
// ---------------------------------------------------------------------------

/// Compute perplexity from per-token log-probabilities.
///
/// `log_probs` – natural-log probability of each token in the sequence.
///
/// `perplexity = exp(-mean(log_probs))`
///
/// Returns `f64::INFINITY` for an empty sequence.
pub fn perplexity(log_probs: &[f64]) -> f64 {
    if log_probs.is_empty() {
        return f64::INFINITY;
    }
    let mean_neg_log: f64 = -log_probs.iter().sum::<f64>() / log_probs.len() as f64;
    mean_neg_log.exp()
}

// ---------------------------------------------------------------------------
// BLEU score
// ---------------------------------------------------------------------------

/// Compute sentence-level BLEU score.
///
/// Uses modified n-gram precision up to `max_n` with a brevity penalty.
/// `max_n` is typically 4.
///
/// Returns a value in [0, 1].
pub fn bleu_score(hypothesis: &[usize], references: &[Vec<usize>], max_n: usize) -> f64 {
    if hypothesis.is_empty() || references.is_empty() || max_n == 0 {
        return 0.0;
    }

    let hyp_len = hypothesis.len();
    // Choose reference closest in length to hypothesis
    let ref_len = references
        .iter()
        .min_by_key(|r| (r.len() as isize - hyp_len as isize).unsigned_abs())
        .map(|r| r.len())
        .unwrap_or(1);

    let mut log_precision_sum = 0.0;
    let mut count_valid = 0usize;

    for n in 1..=max_n {
        if hyp_len < n {
            break;
        }

        let hyp_counts = ngram_counts(hypothesis, n);
        let hyp_total: usize = hyp_counts.values().sum();

        if hyp_total == 0 {
            break;
        }

        // Clip counts: for each ngram, max reference count across references
        let mut clipped_count = 0usize;
        for (ngram, &h_count) in &hyp_counts {
            let max_ref_count = references
                .iter()
                .map(|r| ngram_counts(r, n).get(ngram).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            clipped_count += h_count.min(max_ref_count);
        }

        if clipped_count == 0 {
            // Any zero-precision order collapses BLEU to 0
            return 0.0;
        }

        log_precision_sum += (clipped_count as f64 / hyp_total as f64).ln();
        count_valid += 1;
    }

    if count_valid == 0 {
        return 0.0;
    }

    // Brevity penalty
    let bp = if hyp_len >= ref_len {
        1.0
    } else {
        ((1.0 - ref_len as f64 / hyp_len as f64).exp()).min(1.0)
    };

    bp * (log_precision_sum / count_valid as f64).exp()
}

/// Compute self-BLEU: mean pairwise BLEU across a set of hypotheses.
///
/// Measures diversity – lower self-BLEU = more diverse outputs.
pub fn self_bleu(hypotheses: &[Vec<usize>], max_n: usize) -> f64 {
    if hypotheses.len() < 2 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0usize;

    for (i, hyp) in hypotheses.iter().enumerate() {
        let references: Vec<Vec<usize>> = hypotheses
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, r)| r.clone())
            .collect();
        total += bleu_score(hyp, &references, max_n);
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    total / count as f64
}

// ---------------------------------------------------------------------------
// NLL loss
// ---------------------------------------------------------------------------

/// Compute mean negative log-likelihood loss.
///
/// `log_probs` – shape `(batch_or_seq_len,)` containing the log-probability of
///               the correct token at each position.
pub fn nll_loss(log_probs: &Array2<f64>, targets: &[usize]) -> f64 {
    let nrows = log_probs.nrows();
    let ncols = log_probs.ncols();

    if nrows == 0 || ncols == 0 || targets.is_empty() {
        return 0.0;
    }

    let len = nrows.min(targets.len());
    let mut total = 0.0;

    for i in 0..len {
        let t = targets[i];
        if t < ncols {
            total -= log_probs[[i, t]];
        }
    }

    total / len as f64
}

// ---------------------------------------------------------------------------
// Cross-entropy for LM
// ---------------------------------------------------------------------------

/// Compute cross-entropy loss for a language model.
///
/// `logits` – raw (unnormalised) scores of shape `(seq_len, vocab_size)`.
/// `targets` – target token ids for each position, length `seq_len`.
/// `ignore_index` – if set, positions where `targets[i] == ignore_index` are
///                  excluded from the loss.
///
/// The function applies log-softmax internally.
pub fn lm_cross_entropy(
    logits: &Array2<f64>,
    targets: &[usize],
    ignore_index: Option<usize>,
) -> f64 {
    let seq_len = logits.nrows();
    let vocab_size = logits.ncols();

    if seq_len == 0 || vocab_size == 0 || targets.is_empty() {
        return 0.0;
    }

    let len = seq_len.min(targets.len());
    let mut total = 0.0;
    let mut valid = 0usize;

    for (i, &t) in targets.iter().enumerate().take(len) {
        if let Some(ign) = ignore_index {
            if t == ign {
                continue;
            }
        }

        if t >= vocab_size {
            continue;
        }

        // Log-softmax for row i
        let row: Vec<f64> = logits.row(i).iter().copied().collect();
        let log_prob_t = log_softmax_at(&row, t);
        total -= log_prob_t;
        valid += 1;
    }

    if valid == 0 {
        return 0.0;
    }

    total / valid as f64
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Count n-gram occurrences in a sequence.
fn ngram_counts(seq: &[usize], n: usize) -> HashMap<Vec<usize>, usize> {
    let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
    if seq.len() < n {
        return counts;
    }
    for window in seq.windows(n) {
        *counts.entry(window.to_vec()).or_insert(0) += 1;
    }
    counts
}

/// Numerically stable log-softmax at position `t`.
fn log_softmax_at(logits: &[f64], t: usize) -> f64 {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = logits.iter().map(|&x| (x - max).exp()).sum();
    (logits[t] - max) - sum_exp.ln()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // --- perplexity ---

    #[test]
    fn test_perplexity_empty_is_infinity() {
        assert_eq!(perplexity(&[]), f64::INFINITY);
    }

    #[test]
    fn test_perplexity_uniform_distribution() {
        // Uniform over V tokens: each log-prob = -ln(V)
        let vocab_size = 10usize;
        let lp = -(vocab_size as f64).ln();
        let log_probs: Vec<f64> = vec![lp; 20];
        let ppl = perplexity(&log_probs);
        // Expected perplexity = V
        assert!((ppl - vocab_size as f64).abs() < 1e-6, "ppl = {ppl}");
    }

    #[test]
    fn test_perplexity_perfect_prediction() {
        // log-prob = 0 → probability = 1 → perplexity = 1
        let log_probs = vec![0.0f64; 10];
        assert!((perplexity(&log_probs) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_perplexity_single_token() {
        let log_probs = vec![-2.0f64];
        let expected = 2.0f64.exp();
        assert!((perplexity(&log_probs) - expected).abs() < 1e-9);
    }

    // --- bleu_score ---

    #[test]
    fn test_bleu_perfect_match() {
        let hyp = vec![1usize, 2, 3, 4];
        let refs = vec![vec![1usize, 2, 3, 4]];
        let score = bleu_score(&hyp, &refs, 4);
        assert!((score - 1.0).abs() < 1e-9, "score = {score}");
    }

    #[test]
    fn test_bleu_zero_for_empty_hyp() {
        let score = bleu_score(&[], &[vec![1, 2, 3]], 4);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_bleu_zero_for_no_overlap() {
        let hyp = vec![10usize, 11, 12];
        let refs = vec![vec![1usize, 2, 3]];
        let score = bleu_score(&hyp, &refs, 4);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_bleu_partial_overlap_less_than_one() {
        let hyp = vec![1usize, 2, 99];
        let refs = vec![vec![1usize, 2, 3]];
        let score = bleu_score(&hyp, &refs, 2);
        assert!(score > 0.0 && score < 1.0, "score = {score}");
    }

    #[test]
    fn test_bleu_max_n_zero_is_zero() {
        let hyp = vec![1usize, 2, 3];
        let refs = vec![vec![1usize, 2, 3]];
        assert_eq!(bleu_score(&hyp, &refs, 0), 0.0);
    }

    #[test]
    fn test_bleu_brevity_penalty_short_hyp() {
        // Short hypothesis relative to reference should be penalised
        let hyp = vec![1usize, 2];
        let refs = vec![vec![1usize, 2, 3, 4, 5, 6]];
        let score = bleu_score(&hyp, &refs, 1);
        // 1-gram precision = 1.0 (both tokens match), BP < 1
        assert!(score < 1.0, "score = {score}");
    }

    #[test]
    fn test_bleu_multiple_references() {
        let hyp = vec![1usize, 2, 3];
        let refs = vec![vec![4usize, 5, 6], vec![1usize, 2, 3]];
        let score = bleu_score(&hyp, &refs, 4);
        assert!((score - 1.0).abs() < 1e-9, "score = {score}");
    }

    // --- self_bleu ---

    #[test]
    fn test_self_bleu_single_hyp_is_zero() {
        let hyps = vec![vec![1usize, 2, 3]];
        assert_eq!(self_bleu(&hyps, 4), 0.0);
    }

    #[test]
    fn test_self_bleu_identical_hyps_is_one() {
        let hyps = vec![vec![1usize, 2, 3, 4], vec![1usize, 2, 3, 4]];
        let score = self_bleu(&hyps, 4);
        assert!((score - 1.0).abs() < 1e-9, "score = {score}");
    }

    #[test]
    fn test_self_bleu_diverse_hyps_is_lower() {
        let identical = vec![vec![1usize, 2, 3, 4], vec![1usize, 2, 3, 4]];
        let diverse = vec![vec![1usize, 2, 3, 4], vec![5usize, 6, 7, 8]];
        let sb_ident = self_bleu(&identical, 4);
        let sb_div = self_bleu(&diverse, 4);
        assert!(
            sb_div <= sb_ident,
            "diverse should be <= identical: {sb_div} vs {sb_ident}"
        );
    }

    // --- nll_loss ---

    #[test]
    fn test_nll_loss_correct_token() {
        // log_prob of the correct token = -1.0 → NLL = 1.0
        let lp = Array2::from_shape_vec((2, 3), vec![-1.0, -2.0, -3.0, -5.0, -1.0, -2.0])
            .expect("shape");
        let targets = vec![0usize, 1];
        let loss = nll_loss(&lp, &targets);
        // (-(-1.0) + -(-1.0)) / 2 = 1.0
        assert!((loss - 1.0).abs() < 1e-9, "loss = {loss}");
    }

    #[test]
    fn test_nll_loss_empty_is_zero() {
        let lp: Array2<f64> = Array2::zeros((0, 5));
        let loss = nll_loss(&lp, &[]);
        assert_eq!(loss, 0.0);
    }

    // --- lm_cross_entropy ---

    #[test]
    fn test_lm_cross_entropy_perfect_logits() {
        // If logit for correct token is very large, cross-entropy ≈ 0
        let mut logits_data = vec![-100.0f64; 10];
        logits_data[3] = 100.0;
        let logits = Array2::from_shape_vec((1, 10), logits_data).expect("shape");
        let ce = lm_cross_entropy(&logits, &[3], None);
        assert!(ce < 0.01, "ce = {ce}");
    }

    #[test]
    fn test_lm_cross_entropy_uniform_is_log_vocab() {
        // Uniform logits → cross-entropy = ln(vocab_size)
        let vocab_size = 4usize;
        let logits = Array2::from_elem((1, vocab_size), 0.0);
        let ce = lm_cross_entropy(&logits, &[0], None);
        let expected = (vocab_size as f64).ln();
        assert!(
            (ce - expected).abs() < 1e-9,
            "ce = {ce}, expected = {expected}"
        );
    }

    #[test]
    fn test_lm_cross_entropy_ignore_index() {
        let vocab_size = 4usize;
        let logits = Array2::from_elem((2, vocab_size), 0.0);
        // Target 99 is the ignore_index; only position 1 (target 0) counts
        let ce_all = lm_cross_entropy(&logits, &[0, 99], None);
        let ce_ign = lm_cross_entropy(&logits, &[0, 99], Some(99));
        // With ignore, only 1 valid position → same CE for that position
        let expected = (vocab_size as f64).ln();
        assert!((ce_ign - expected).abs() < 1e-9, "ce_ign = {ce_ign}");
        // Without ignore, position 1 is still invalid (99 >= vocab_size) → same
        let _ = ce_all; // both ignore out-of-range targets anyway
    }

    #[test]
    fn test_lm_cross_entropy_empty_is_zero() {
        let logits: Array2<f64> = Array2::zeros((0, 5));
        assert_eq!(lm_cross_entropy(&logits, &[], None), 0.0);
    }
}
