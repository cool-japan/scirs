//! Penn Treebank synthetic language modelling dataset generator.
//!
//! Generates tokenised sentences with a Zipfian vocabulary distribution,
//! mimicking the statistical properties of the Penn Treebank corpus.
//! Sentence lengths follow a Poisson distribution.
//!
//! Also provides [`PennTreebankDataset::load_from_text`] to build the dataset
//! from a real space-separated text file.

use crate::error::{DatasetsError, Result};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Penn Treebank synthetic dataset generator.
#[derive(Debug, Clone)]
pub struct PennTreebankConfig {
    /// Vocabulary size (default: 10_000, matching PTB's ~10k unique words).
    pub vocab_size: usize,
    /// Number of sentences to generate (default: 1_000).
    pub n_sentences: usize,
    /// Average sentence length in tokens (default: 20).
    pub avg_sentence_len: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for PennTreebankConfig {
    fn default() -> Self {
        Self {
            vocab_size: 10_000,
            n_sentences: 1_000,
            avg_sentence_len: 20,
            seed: 42,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal LCG (same pattern as text_datasets.rs — no external rand needed for
// the custom Zipf sampler, but we also use it for sentence-length Poisson)
// ─────────────────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            6_364_136_223_846_793_005
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Poisson(lambda) sample via Knuth's algorithm.
    fn next_poisson(&mut self, lambda: f64) -> usize {
        if lambda <= 0.0 {
            return 0;
        }
        let l = (-lambda).exp();
        let mut k = 0usize;
        let mut p = 1.0_f64;
        loop {
            k += 1;
            p *= self.next_f64().max(1e-300);
            if p <= l {
                break;
            }
        }
        k.saturating_sub(1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Zipf cumulative-weight sampler
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed CDF for Zipf(s=1.0) over `vocab_size` entries.
/// Token rank 0 is most common.
struct ZipfSampler {
    cdf: Vec<f64>,
}

impl ZipfSampler {
    fn new(vocab_size: usize) -> Self {
        let mut cdf = Vec::with_capacity(vocab_size);
        let mut cumsum = 0.0_f64;
        for rank in 0..vocab_size {
            cumsum += 1.0 / (rank + 1) as f64;
            cdf.push(cumsum);
        }
        // Normalise
        let total = cumsum;
        for v in &mut cdf {
            *v /= total;
        }
        Self { cdf }
    }

    /// Sample a token index ∈ [0, vocab_size).
    fn sample(&self, u: f64) -> usize {
        // Binary search for the first cdf entry ≥ u
        match self.cdf.partition_point(|&c| c < u) {
            idx if idx < self.cdf.len() => idx,
            _ => self.cdf.len() - 1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PennTreebankDataset
// ─────────────────────────────────────────────────────────────────────────────

/// Synthetic Penn Treebank-style language modelling dataset.
///
/// Token indices are 0-based.  Token 0 is the most frequent (highest Zipf rank).
#[derive(Debug, Clone)]
pub struct PennTreebankDataset {
    tokens: Vec<Vec<usize>>,
    vocab_size: usize,
}

impl PennTreebankDataset {
    /// Generate a synthetic dataset from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn generate(config: PennTreebankConfig) -> Result<Self> {
        if config.vocab_size == 0 {
            return Err(DatasetsError::InvalidFormat(
                "PennTreebankConfig: vocab_size must be > 0".to_string(),
            ));
        }
        if config.n_sentences == 0 {
            return Err(DatasetsError::InvalidFormat(
                "PennTreebankConfig: n_sentences must be > 0".to_string(),
            ));
        }

        let zipf = ZipfSampler::new(config.vocab_size);
        let mut rng = Lcg::new(config.seed);
        let avg = config.avg_sentence_len.max(1) as f64;

        let sentences: Vec<Vec<usize>> = (0..config.n_sentences)
            .map(|_| {
                let len = rng.next_poisson(avg).max(1);
                (0..len)
                    .map(|_| zipf.sample(rng.next_f64()))
                    .collect::<Vec<usize>>()
            })
            .collect();

        Ok(Self {
            tokens: sentences,
            vocab_size: config.vocab_size,
        })
    }

    /// All tokenised sentences.
    pub fn sentences(&self) -> &[Vec<usize>] {
        &self.tokens
    }

    /// All tokens from all sentences concatenated in order.
    pub fn flat_tokens(&self) -> Vec<usize> {
        self.tokens.iter().flatten().copied().collect()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Total number of tokens across all sentences.
    pub fn word_count(&self) -> usize {
        self.tokens.iter().map(|s| s.len()).sum()
    }

    /// Load a Penn Treebank-style dataset from a space-separated text file.
    ///
    /// Words are space / newline separated. The most frequent `vocab_size` words
    /// are assigned indices 1..vocab_size.  All other words map to index 0
    /// (`<unk>`).  Each line becomes one sentence.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or read.
    pub fn load_from_text(path: impl AsRef<Path>, vocab_size: usize) -> Result<Self> {
        let file = fs::File::open(path.as_ref()).map_err(DatasetsError::IoError)?;
        let reader = BufReader::new(file);

        // First pass: count word frequencies
        let mut freq: HashMap<String, usize> = HashMap::new();
        let mut raw_sentences: Vec<Vec<String>> = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(DatasetsError::IoError)?;
            let words: Vec<String> = line.split_whitespace().map(|w| w.to_lowercase()).collect();
            if !words.is_empty() {
                for w in &words {
                    *freq.entry(w.clone()).or_insert(0) += 1;
                }
                raw_sentences.push(words);
            }
        }

        // Build vocab: take top-N by frequency, assign indices 1..=N; 0 = <unk>
        let mut sorted_words: Vec<(String, usize)> = freq.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let vocab: HashMap<String, usize> = sorted_words
            .iter()
            .take(vocab_size.saturating_sub(1)) // reserve index 0 for <unk>
            .enumerate()
            .map(|(i, (word, _))| (word.clone(), i + 1))
            .collect();

        let sentences: Vec<Vec<usize>> = raw_sentences
            .iter()
            .map(|sent| sent.iter().map(|w| *vocab.get(w).unwrap_or(&0)).collect())
            .collect();

        Ok(Self {
            tokens: sentences,
            vocab_size,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_ptb_shape() {
        let cfg = PennTreebankConfig {
            vocab_size: 1_000,
            n_sentences: 100,
            avg_sentence_len: 15,
            seed: 1,
        };
        let ds = PennTreebankDataset::generate(cfg.clone()).expect("generate failed");
        assert_eq!(ds.sentences().len(), cfg.n_sentences);
        assert_eq!(ds.vocab_size(), cfg.vocab_size);
        assert!(ds.word_count() > 0);
    }

    #[test]
    fn test_ptb_deterministic() {
        let cfg = PennTreebankConfig {
            vocab_size: 500,
            n_sentences: 50,
            avg_sentence_len: 10,
            seed: 77,
        };
        let ds1 = PennTreebankDataset::generate(cfg.clone()).expect("generate failed");
        let ds2 = PennTreebankDataset::generate(cfg).expect("generate failed");
        assert_eq!(ds1.flat_tokens(), ds2.flat_tokens());
    }

    #[test]
    fn test_ptb_token_range() {
        let cfg = PennTreebankConfig {
            vocab_size: 200,
            n_sentences: 50,
            avg_sentence_len: 12,
            seed: 5,
        };
        let ds = PennTreebankDataset::generate(cfg.clone()).expect("generate failed");
        for tok in ds.flat_tokens() {
            assert!(tok < cfg.vocab_size, "token {tok} out of vocab range");
        }
    }

    #[test]
    fn test_ptb_flat_tokens_concat() {
        let cfg = PennTreebankConfig {
            vocab_size: 100,
            n_sentences: 10,
            avg_sentence_len: 5,
            seed: 3,
        };
        let ds = PennTreebankDataset::generate(cfg).expect("generate failed");
        let flat = ds.flat_tokens();
        let expected: usize = ds.sentences().iter().map(|s| s.len()).sum();
        assert_eq!(flat.len(), expected);
    }

    #[test]
    fn test_ptb_each_sentence_nonempty() {
        let cfg = PennTreebankConfig {
            vocab_size: 50,
            n_sentences: 20,
            avg_sentence_len: 8,
            seed: 11,
        };
        let ds = PennTreebankDataset::generate(cfg).expect("generate failed");
        for sent in ds.sentences() {
            assert!(
                !sent.is_empty(),
                "every sentence must have at least 1 token"
            );
        }
    }

    #[test]
    fn test_ptb_load_from_text() {
        let mut tmp = std::env::temp_dir();
        tmp.push("ptb_test_corpus.txt");
        {
            let mut f = fs::File::create(&tmp).expect("create tmp file");
            writeln!(f, "the cat sat on the mat").expect("write");
            writeln!(f, "the dog sat on the log").expect("write");
            writeln!(f, "a quick brown fox").expect("write");
        }
        let ds = PennTreebankDataset::load_from_text(&tmp, 20).expect("load failed");
        assert_eq!(ds.vocab_size(), 20);
        assert_eq!(ds.sentences().len(), 3);
        // "the" should map to index 1 (most frequent)
        let flat = ds.flat_tokens();
        assert!(flat.iter().any(|&t| t > 0), "should have known tokens");
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_ptb_error_zero_vocab() {
        let cfg = PennTreebankConfig {
            vocab_size: 0,
            ..PennTreebankConfig::default()
        };
        assert!(PennTreebankDataset::generate(cfg).is_err());
    }
}
