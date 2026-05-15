//! Statistical language models: Unigram, Bigram, N-gram, and perplexity evaluation.
//!
//! This module provides from-scratch implementations of classical statistical
//! language models with smoothing:
//!
//! - `UnigramLM` – unsmoothed maximum-likelihood unigram model
//! - `BigramLM` – bigram model with Laplace smoothing
//! - `NgramLM` – arbitrary-order model with Kneser-Ney smoothing
//! - `PerplexityEval` – perplexity computation for any `NgramLM`

use std::collections::{HashMap, HashSet};

use crate::error::{Result, TextError};

// ---------------------------------------------------------------------------
// Tokenisation helper
// ---------------------------------------------------------------------------

/// Split text into lowercase alpha tokens.
fn simple_tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

// ---------------------------------------------------------------------------
// UnigramLM
// ---------------------------------------------------------------------------

/// A maximum-likelihood unigram language model.
#[derive(Debug, Clone)]
pub struct UnigramLM {
    /// Word → probability (MLE, no smoothing).
    pub probs: HashMap<String, f64>,
    /// Known vocabulary.
    pub vocab: HashSet<String>,
}

impl UnigramLM {
    /// Train a unigram model from a corpus of sentences.
    pub fn train(sentences: &[Vec<String>]) -> Result<UnigramLM> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;
        for sent in sentences {
            for w in sent {
                *counts.entry(w.clone()).or_insert(0) += 1;
                total += 1;
            }
        }
        if total == 0 {
            return Err(TextError::InvalidInput("Empty corpus".to_string()));
        }
        let vocab: HashSet<String> = counts.keys().cloned().collect();
        let probs = counts
            .into_iter()
            .map(|(k, c)| (k, c as f64 / total as f64))
            .collect();
        Ok(UnigramLM { probs, vocab })
    }

    /// Probability of a word.
    ///
    /// Returns `0.0` for out-of-vocabulary words.
    pub fn probability(&self, word: &str) -> f64 {
        self.probs.get(word).copied().unwrap_or(0.0)
    }

    /// Log-probability of a word (returns `f64::NEG_INFINITY` for OOV).
    pub fn log_probability(&self, word: &str) -> f64 {
        let p = self.probability(word);
        if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            p.ln()
        }
    }
}

// ---------------------------------------------------------------------------
// BigramLM
// ---------------------------------------------------------------------------

/// A bigram language model with Laplace smoothing.
#[derive(Debug, Clone)]
pub struct BigramLM {
    /// (prev, curr) → P(curr | prev)
    pub probs: HashMap<(String, String), f64>,
    /// Unigram probabilities (smoothed) for back-off.
    pub unigrams: HashMap<String, f64>,
    /// Vocabulary size used for Laplace denominator.
    vocab_size: usize,
}

impl BigramLM {
    /// Train a bigram model from a corpus of sentences.
    ///
    /// Laplace smoothing is applied with `k = 1` (add-one).
    pub fn train(sentences: &[Vec<String>]) -> Result<BigramLM> {
        let mut uni_counts: HashMap<String, usize> = HashMap::new();
        let mut bi_counts: HashMap<(String, String), usize> = HashMap::new();

        // Collect <s> and </s> boundaries
        const START: &str = "<s>";
        const END: &str = "</s>";

        for sent in sentences {
            if sent.is_empty() {
                continue;
            }
            let padded: Vec<&str> = std::iter::once(START)
                .chain(sent.iter().map(String::as_str))
                .chain(std::iter::once(END))
                .collect();
            for i in 0..padded.len() - 1 {
                *uni_counts.entry(padded[i].to_string()).or_insert(0) += 1;
                *bi_counts
                    .entry((padded[i].to_string(), padded[i + 1].to_string()))
                    .or_insert(0) += 1;
            }
            *uni_counts.entry(END.to_string()).or_insert(0) += 1;
        }

        let vocab_size = uni_counts.len();
        if vocab_size == 0 {
            return Err(TextError::InvalidInput("Empty corpus".to_string()));
        }

        // Laplace-smoothed unigrams
        let total_uni: usize = uni_counts.values().sum();
        let unigrams: HashMap<String, f64> = uni_counts
            .iter()
            .map(|(w, &c)| {
                let p = (c as f64 + 1.0) / (total_uni as f64 + vocab_size as f64);
                (w.clone(), p)
            })
            .collect();

        // Laplace-smoothed bigrams
        let mut probs: HashMap<(String, String), f64> = HashMap::new();
        // Collect all known bigrams
        for ((prev, curr), &c) in &bi_counts {
            let prev_count = uni_counts.get(prev).copied().unwrap_or(0) as f64;
            let p = (c as f64 + 1.0) / (prev_count + vocab_size as f64);
            probs.insert((prev.clone(), curr.clone()), p);
        }

        Ok(BigramLM {
            probs,
            unigrams,
            vocab_size,
        })
    }

    /// P(curr | prev) with Laplace smoothing back-off.
    pub fn probability(&self, prev: &str, curr: &str) -> f64 {
        self.probs
            .get(&(prev.to_string(), curr.to_string()))
            .copied()
            .unwrap_or_else(|| 1.0 / (self.vocab_size as f64 + 1.0))
    }
}

// ---------------------------------------------------------------------------
// NgramLM  (Kneser-Ney smoothing)
// ---------------------------------------------------------------------------

/// An N-gram language model with Kneser-Ney smoothing.
///
/// Uses absolute discounting with discount `d = 0.75` and a recursive
/// lower-order back-off.
#[derive(Debug, Clone)]
pub struct NgramLM {
    /// N-gram order.
    pub n: usize,
    /// Full-context n-gram counts.
    pub counts: HashMap<Vec<String>, usize>,
    /// Context (n-1 gram) counts.
    pub context_counts: HashMap<Vec<String>, usize>,
    /// Kneser-Ney continuation counts for lower-order back-off.
    continuation_counts: HashMap<String, usize>,
    /// Number of distinct bigrams in the corpus (for KN base).
    n_bigrams: usize,
    /// Discount parameter.
    discount: f64,
}

impl NgramLM {
    /// Train an N-gram model from a corpus of sentences.
    pub fn train(n: usize, sentences: &[Vec<String>]) -> Result<NgramLM> {
        if n < 1 {
            return Err(TextError::InvalidInput("n must be >= 1".to_string()));
        }
        const START: &str = "<s>";
        const END: &str = "</s>";

        let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
        let mut context_counts: HashMap<Vec<String>, usize> = HashMap::new();
        let mut continuation_counts: HashMap<String, usize> = HashMap::new();
        let mut bigram_set: HashSet<(String, String)> = HashSet::new();

        for sent in sentences {
            if sent.is_empty() {
                continue;
            }
            // Pad with (n-1) start tokens and 1 end token
            let mut padded: Vec<String> = (0..n - 1).map(|_| START.to_string()).collect();
            padded.extend(sent.iter().cloned());
            padded.push(END.to_string());

            for i in 0..padded.len().saturating_sub(n - 1) {
                let ngram: Vec<String> = padded[i..i + n].to_vec();
                let context: Vec<String> = padded[i..i + n - 1].to_vec();
                *counts.entry(ngram).or_insert(0) += 1;
                if n > 1 {
                    *context_counts.entry(context).or_insert(0) += 1;
                }
            }

            // Continuation counts for KN: unique left-contexts for each word
            for i in 1..padded.len() {
                bigram_set.insert((padded[i - 1].clone(), padded[i].clone()));
                *continuation_counts.entry(padded[i].clone()).or_insert(0) += 0;
                // ensure entry exists
            }
        }

        // Count unique left-contexts per word
        for (_, curr) in &bigram_set {
            *continuation_counts.entry(curr.clone()).or_insert(0) += 1;
        }
        let n_bigrams = bigram_set.len();

        Ok(NgramLM {
            n,
            counts,
            context_counts,
            continuation_counts,
            n_bigrams,
            discount: 0.75,
        })
    }

    /// P(word | context) using Kneser-Ney smoothing.
    ///
    /// For unigrams (n=1) this reduces to Kneser-Ney continuation probability.
    pub fn probability(&self, word: &str, context: &[&str]) -> f64 {
        self.kn_probability(word, context)
    }

    fn kn_probability(&self, word: &str, context: &[&str]) -> f64 {
        if self.n == 1 {
            return self.kn_unigram(word);
        }

        // Use the last (n-1) words of context
        let used_ctx: Vec<String> = if context.len() >= self.n - 1 {
            context[context.len() - (self.n - 1)..]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            context.iter().map(|s| s.to_string()).collect()
        };

        let ngram: Vec<String> = used_ctx
            .iter()
            .cloned()
            .chain(std::iter::once(word.to_string()))
            .collect();

        let c = self.counts.get(&ngram).copied().unwrap_or(0) as f64;
        let c_ctx = self.context_counts.get(&used_ctx).copied().unwrap_or(0) as f64;

        if c_ctx == 0.0 {
            return self.kn_unigram(word);
        }

        // Count of types that follow `used_ctx` (for lambda)
        let types_after_ctx = self
            .counts
            .iter()
            .filter(|(k, &v)| v > 0 && k.len() == self.n && k[..self.n - 1] == used_ctx[..])
            .count() as f64;

        let lambda = self.discount * types_after_ctx / c_ctx;
        let first_term = (c - self.discount).max(0.0) / c_ctx;
        first_term + lambda * self.kn_unigram(word)
    }

    fn kn_unigram(&self, word: &str) -> f64 {
        let c_w = self.continuation_counts.get(word).copied().unwrap_or(0) as f64;
        if self.n_bigrams == 0 {
            return 1e-10;
        }
        (c_w / self.n_bigrams as f64).max(1e-10)
    }

    /// Log-probability of `word` given `context`.
    pub fn log_probability(&self, word: &str, context: &[&str]) -> f64 {
        let p = self.probability(word, context);
        if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            p.ln()
        }
    }
}

// ---------------------------------------------------------------------------
// PerplexityEval
// ---------------------------------------------------------------------------

/// Perplexity evaluation for an `NgramLM`.
pub struct PerplexityEval;

impl PerplexityEval {
    /// Compute per-token perplexity over `test_sentences`.
    ///
    /// PP = exp( -1/N * Σ log P(w_i | w_{i-n+1}..w_{i-1}) )
    pub fn compute(lm: &NgramLM, test_sentences: &[Vec<String>]) -> Result<f64> {
        let mut log_prob_sum = 0.0f64;
        let mut token_count = 0usize;

        const START: &str = "<s>";

        for sent in test_sentences {
            if sent.is_empty() {
                continue;
            }
            // Pad with start tokens
            let mut padded: Vec<String> = (0..lm.n - 1).map(|_| START.to_string()).collect();
            padded.extend(sent.iter().cloned());

            for i in lm.n - 1..padded.len() {
                let word = &padded[i];
                let ctx_start = i.saturating_sub(lm.n - 1);
                let context: Vec<&str> = padded[ctx_start..i].iter().map(String::as_str).collect();
                let lp = lm.log_probability(word, &context);
                if lp.is_finite() {
                    log_prob_sum += lp;
                } else {
                    // Penalty for completely unknown n-gram
                    log_prob_sum += (1e-10_f64).ln();
                }
                token_count += 1;
            }
        }

        if token_count == 0 {
            return Err(TextError::InvalidInput(
                "No tokens in test sentences".to_string(),
            ));
        }

        let avg_log_prob = log_prob_sum / token_count as f64;
        Ok((-avg_log_prob).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<Vec<String>> {
        vec![
            simple_tokenize("the cat sat on the mat"),
            simple_tokenize("the dog ran over the hill"),
            simple_tokenize("a cat and a dog played"),
            simple_tokenize("the cat chased the dog"),
            simple_tokenize("the mat was on the floor"),
        ]
    }

    #[test]
    fn test_unigram_probabilities_sum_to_one() {
        let lm = UnigramLM::train(&corpus()).expect("train failed");
        let total: f64 = lm.probs.values().sum();
        assert!((total - 1.0).abs() < 1e-9, "sum = {}", total);
    }

    #[test]
    fn test_unigram_known_word() {
        let lm = UnigramLM::train(&corpus()).expect("train");
        assert!(lm.probability("cat") > 0.0);
    }

    #[test]
    fn test_unigram_oov() {
        let lm = UnigramLM::train(&corpus()).expect("train");
        assert_eq!(lm.probability("xyzzy"), 0.0);
    }

    #[test]
    fn test_bigram_probability_positive() {
        let lm = BigramLM::train(&corpus()).expect("train");
        let p = lm.probability("the", "cat");
        assert!(p > 0.0 && p <= 1.0, "p = {}", p);
    }

    #[test]
    fn test_bigram_unseen_is_smoothed() {
        let lm = BigramLM::train(&corpus()).expect("train");
        let p = lm.probability("cat", "airplane");
        assert!(p > 0.0, "Laplace smoothed probability should be > 0");
    }

    #[test]
    fn test_ngram_probability_trigram() {
        let lm = NgramLM::train(3, &corpus()).expect("train");
        let p = lm.probability("cat", &["<s>", "the"]);
        assert!(p > 0.0, "p = {}", p);
    }

    #[test]
    fn test_ngram_probability_unseen() {
        let lm = NgramLM::train(2, &corpus()).expect("train");
        let p = lm.probability("airplane", &["the"]);
        // KN back-off should return a small but positive value
        assert!(p > 0.0, "KN probability should be > 0 even for OOV");
    }

    #[test]
    fn test_perplexity_finite() {
        let train = corpus();
        let lm = NgramLM::train(2, &train).expect("train");
        let test_data = vec![simple_tokenize("the cat sat")];
        let pp = PerplexityEval::compute(&lm, &test_data).expect("perplexity");
        assert!(pp.is_finite() && pp > 1.0, "pp = {}", pp);
    }

    #[test]
    fn test_perplexity_lower_on_train_than_random() {
        let train = corpus();
        let lm = NgramLM::train(2, &train).expect("train");

        let train_pp = PerplexityEval::compute(&lm, &train[..2]).expect("train perplexity");
        let random_pp = PerplexityEval::compute(&lm, &[simple_tokenize("xyzzy blorp quux flerb")])
            .expect("random perplexity");

        assert!(
            train_pp <= random_pp,
            "train pp {} should be <= random pp {}",
            train_pp,
            random_pp
        );
    }

    #[test]
    fn test_perplexity_empty_error() {
        let lm = NgramLM::train(2, &corpus()).expect("train");
        let result = PerplexityEval::compute(&lm, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unigram_log_probability() {
        let lm = UnigramLM::train(&corpus()).expect("train");
        let lp = lm.log_probability("cat");
        assert!(lp < 0.0 && lp.is_finite());
    }
}
