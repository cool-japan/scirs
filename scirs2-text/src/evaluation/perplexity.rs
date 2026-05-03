//! Perplexity-based language-model evaluation.
//!
//! Provides the [`LanguageModelLike`] trait, [`perplexity_evaluate`], and
//! [`PerplexityReport`]. An implementation of `LanguageModelLike` for the
//! existing [`crate::language_models::NgramLM`] is included.
//!
//! ## Example
//!
//! ```rust,ignore
//! use scirs2_text::evaluation::perplexity::{
//!     LanguageModelLike, PerplexityReport, perplexity_evaluate,
//! };
//!
//! struct UniformModel { vocab: usize }
//! impl LanguageModelLike for UniformModel {
//!     fn log_prob_sequence(&self, tokens: &[&str]) -> Option<f64> {
//!         if tokens.is_empty() { return None; }
//!         Some(tokens.len() as f64 * -(self.vocab as f64).ln())
//!     }
//!     fn vocabulary_size(&self) -> usize { self.vocab }
//! }
//! let model = UniformModel { vocab: 100 };
//! let corpus = vec![vec!["a", "b", "c"]];
//! let report = perplexity_evaluate(&model, &corpus).unwrap();
//! assert!((report.corpus_perplexity - 100.0).abs() < 1e-6);
//! ```

use crate::error::{Result, TextError};
use std::path::Path;

/// Trait for language models that can compute conditional log-probability
/// over a token sequence.
///
/// Implementors expose string tokens so the trait is usable with both
/// word-level and character-level models without a vocabulary-bridge adapter.
pub trait LanguageModelLike {
    /// Return Σ_t log p(tokens[t] | tokens[0..t]) for the whole sequence.
    ///
    /// Returns `None` when the sequence is empty.
    fn log_prob_sequence(&self, tokens: &[&str]) -> Option<f64>;

    /// Size of the model vocabulary (used by tests / diagnostics only).
    fn vocabulary_size(&self) -> usize;
}

/// Report returned by [`perplexity_evaluate`].
#[derive(Debug, Clone)]
pub struct PerplexityReport {
    /// PP = exp(-1/N Σ log p(wᵢ|w<ᵢ)) across the whole corpus.
    pub corpus_perplexity: f64,
    /// Per-sentence perplexity; `NaN` for empty/failed sentences.
    pub per_sentence_perplexity: Vec<f64>,
    /// Total number of tokens processed.
    pub total_tokens: usize,
    /// Sum of log-probabilities across all tokens.
    pub total_log_prob: f64,
}

/// Compute corpus perplexity for `model` over a pre-tokenized corpus.
///
/// `corpus` is a slice of sentences; each sentence is a `Vec<&str>` of string tokens.
///
/// # Errors
///
/// - [`TextError::InvalidInput`] when `corpus` is empty.
/// - [`TextError::InvalidInput`] when all sentences are empty (zero tokens total).
pub fn perplexity_evaluate(
    model: &dyn LanguageModelLike,
    corpus: &[Vec<&str>],
) -> Result<PerplexityReport> {
    if corpus.is_empty() {
        return Err(TextError::InvalidInput("corpus is empty".into()));
    }

    let mut total_log_prob = 0.0f64;
    let mut total_tokens = 0usize;
    let mut per_sentence = Vec::with_capacity(corpus.len());

    for sentence in corpus {
        if sentence.is_empty() {
            per_sentence.push(f64::NAN);
            continue;
        }
        match model.log_prob_sequence(sentence) {
            Some(lp) => {
                let n = sentence.len();
                let ppl = (-lp / n as f64).exp();
                per_sentence.push(ppl);
                total_log_prob += lp;
                total_tokens += n;
            }
            None => {
                per_sentence.push(f64::NAN);
            }
        }
    }

    if total_tokens == 0 {
        return Err(TextError::InvalidInput(
            "no tokens found in corpus (all sentences are empty)".into(),
        ));
    }

    let corpus_ppl = (-total_log_prob / total_tokens as f64).exp();

    Ok(PerplexityReport {
        corpus_perplexity: corpus_ppl,
        per_sentence_perplexity: per_sentence,
        total_tokens,
        total_log_prob,
    })
}

/// Load a pre-tokenized corpus from a plain text file.
///
/// Each line is one sentence; tokens are whitespace-separated.
/// Lines that tokenize to zero tokens are skipped.
///
/// # Errors
///
/// Returns [`TextError::IoError`] if the file cannot be opened or read.
pub fn load_token_corpus(path: impl AsRef<Path>) -> Result<Vec<Vec<String>>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path.as_ref()).map_err(|e| TextError::IoError(e.to_string()))?;
    let reader = BufReader::new(file);
    let mut result = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| TextError::IoError(e.to_string()))?;
        let tokens: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
        if !tokens.is_empty() {
            result.push(tokens);
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Blanket helper: build owned string references for a sentence (used by impls)
// ---------------------------------------------------------------------------

/// Implement `LanguageModelLike` for `NgramLM` from `language_models` module.
///
/// `NgramLM` uses string-based context windows with Kneser-Ney smoothing.
impl LanguageModelLike for crate::language_models::NgramLM {
    fn log_prob_sequence(&self, tokens: &[&str]) -> Option<f64> {
        if tokens.is_empty() {
            return None;
        }
        let n = self.n;
        let mut log_prob = 0.0f64;

        for i in 0..tokens.len() {
            let ctx_start = if i >= n - 1 { i + 1 - n } else { 0 };
            let context: Vec<&str> = tokens[ctx_start..i].to_vec();
            let word = tokens[i];
            let p = self.probability(word, &context);
            log_prob += if p <= 0.0 { 1e-10_f64.ln() } else { p.ln() };
        }

        Some(log_prob)
    }

    fn vocabulary_size(&self) -> usize {
        // Derive vocab size from the set of unique terminal words in n-gram keys.
        // Each n-gram key Vec<String> has the word at index n-1.
        let mut vocab = std::collections::HashSet::new();
        for key in self.counts.keys() {
            if let Some(word) = key.last() {
                vocab.insert(word.as_str());
            }
        }
        vocab.len().max(1)
    }
}

/// Implement `LanguageModelLike` for the wired [`crate::language_model::NgramModel`].
///
/// Delegates to `NgramModel::probability` with the preceding `n-1` tokens as context.
impl LanguageModelLike for crate::language_model::NgramModel {
    fn log_prob_sequence(&self, tokens: &[&str]) -> Option<f64> {
        if tokens.is_empty() {
            return None;
        }
        let n = self.order();
        let mut log_prob = 0.0f64;

        for i in 0..tokens.len() {
            let ctx_start = if i >= n - 1 { i + 1 - n } else { 0 };
            let context: Vec<&str> = tokens[ctx_start..i].to_vec();
            let word = tokens[i];
            // `probability` can return Err only for context-length mismatch;
            // pad context to exactly n-1 with <START> tokens when needed.
            let padded_ctx: Vec<&str> = if context.len() < n - 1 {
                let needed = n - 1 - context.len();
                let mut v: Vec<&str> = vec!["<START>"; needed];
                v.extend_from_slice(&context);
                v
            } else {
                context
            };
            let p = self.probability(&padded_ctx, word).unwrap_or(1e-10);
            log_prob += if p <= 0.0 { 1e-10_f64.ln() } else { p.ln() };
        }

        Some(log_prob)
    }

    fn vocabulary_size(&self) -> usize {
        self.vocabulary_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal model that always returns uniform log-probs
    struct UniformModel {
        /// Vocabulary size V; assigns probability 1/V to every token.
        vocab: usize,
    }

    impl LanguageModelLike for UniformModel {
        fn log_prob_sequence(&self, tokens: &[&str]) -> Option<f64> {
            if tokens.is_empty() {
                return None;
            }
            Some(tokens.len() as f64 * -(self.vocab as f64).ln())
        }

        fn vocabulary_size(&self) -> usize {
            self.vocab
        }
    }

    // Perfect predictor: assigns probability 1.0 to every token
    struct PerfectModel;

    impl LanguageModelLike for PerfectModel {
        fn log_prob_sequence(&self, tokens: &[&str]) -> Option<f64> {
            if tokens.is_empty() {
                return None;
            }
            Some(0.0_f64) // log(1.0) = 0
        }

        fn vocabulary_size(&self) -> usize {
            1
        }
    }

    #[test]
    fn perplexity_uniform_model_equals_vocab_size() {
        let model = UniformModel { vocab: 100 };
        let corpus = vec![vec!["a", "b", "c", "d", "e"]];
        let report = perplexity_evaluate(&model, &corpus).expect("evaluate");
        // PPL of uniform model = V = 100
        assert!(
            (report.corpus_perplexity - 100.0).abs() < 1e-6,
            "expected 100.0, got {}",
            report.corpus_perplexity
        );
    }

    #[test]
    fn perplexity_of_perfect_predictor_is_one() {
        let model = PerfectModel;
        let corpus = vec![vec!["a", "b", "c"]];
        let report = perplexity_evaluate(&model, &corpus).expect("evaluate");
        assert!(
            (report.corpus_perplexity - 1.0).abs() < 1e-9,
            "expected 1.0, got {}",
            report.corpus_perplexity
        );
    }

    #[test]
    fn perplexity_corpus_aggregates_token_log_probs() {
        let model = UniformModel { vocab: 10 };
        // Two sentences of 3 and 2 tokens: total 5 tokens
        let corpus = vec![vec!["a", "b", "c"], vec!["d", "e"]];
        let report = perplexity_evaluate(&model, &corpus).expect("evaluate");
        assert_eq!(report.total_tokens, 5);
        // log_prob_total = 5 * ln(1/10)
        let expected_lp = 5.0 * -(10.0f64).ln();
        assert!(
            (report.total_log_prob - expected_lp).abs() < 1e-9,
            "expected total_log_prob {expected_lp}, got {}",
            report.total_log_prob
        );
    }

    #[test]
    fn perplexity_empty_corpus_returns_error() {
        let model = UniformModel { vocab: 10 };
        let result = perplexity_evaluate(&model, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn perplexity_per_sentence_are_positive() {
        let model = UniformModel { vocab: 5 };
        let corpus = vec![vec!["a"], vec!["b", "c"]];
        let report = perplexity_evaluate(&model, &corpus).expect("evaluate");
        for &ppl in &report.per_sentence_perplexity {
            assert!(ppl > 0.0 && ppl.is_finite(), "per-sentence ppl = {ppl}");
        }
    }

    #[test]
    fn perplexity_all_empty_sentences_returns_error() {
        let model = UniformModel { vocab: 5 };
        // All sentences empty → total_tokens = 0
        let corpus: Vec<Vec<&str>> = vec![vec![], vec![]];
        let result = perplexity_evaluate(&model, &corpus);
        assert!(result.is_err());
    }
}
