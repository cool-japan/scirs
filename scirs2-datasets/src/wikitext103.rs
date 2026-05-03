//! WikiText-103 synthetic NLP dataset generator.
//!
//! Generates a synthetic dataset mimicking WikiText-103 structure:
//! - Article / paragraph / token hierarchy
//! - Large Zipfian vocabulary (default 100,000 tokens)
//! - Paragraph lengths sampled from a Poisson distribution
//!
//! Also provides [`WikiText103Dataset::load_from_text`] which reads the
//! WikiText on-disk format (articles delimited by `= Title =` lines,
//! paragraphs by blank lines).

use crate::error::{DatasetsError, Result};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the WikiText-103 synthetic dataset generator.
#[derive(Debug, Clone)]
pub struct WikiText103Config {
    /// Vocabulary size (default: 100_000).
    pub vocab_size: usize,
    /// Number of articles to generate (default: 100).
    pub n_articles: usize,
    /// Average number of paragraphs per article (default: 5).
    pub avg_paragraphs: usize,
    /// Average number of tokens per paragraph (default: 100).
    pub avg_para_tokens: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for WikiText103Config {
    fn default() -> Self {
        Self {
            vocab_size: 100_000,
            n_articles: 100,
            avg_paragraphs: 5,
            avg_para_tokens: 100,
            seed: 42,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal LCG
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

    /// Knuth Poisson sampler.
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
// Zipf sampler (shared structure with penn_treebank but independent)
// ─────────────────────────────────────────────────────────────────────────────

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
        let total = cumsum;
        for v in &mut cdf {
            *v /= total;
        }
        Self { cdf }
    }

    fn sample(&self, u: f64) -> usize {
        match self.cdf.partition_point(|&c| c < u) {
            idx if idx < self.cdf.len() => idx,
            _ => self.cdf.len() - 1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WikiText103Dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Synthetic WikiText-103-style NLP dataset.
///
/// Structure: `articles[article_idx][paragraph_idx][token_idx]`.
/// Token 0 is `<unk>`, tokens 1..vocab_size are Zipf-ranked vocabulary entries.
#[derive(Debug, Clone)]
pub struct WikiText103Dataset {
    /// `[article][paragraph][token]`
    articles: Vec<Vec<Vec<usize>>>,
    vocab_size: usize,
}

impl WikiText103Dataset {
    /// Generate a synthetic WikiText-103 dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn generate(config: WikiText103Config) -> Result<Self> {
        if config.vocab_size == 0 {
            return Err(DatasetsError::InvalidFormat(
                "WikiText103Config: vocab_size must be > 0".to_string(),
            ));
        }
        if config.n_articles == 0 {
            return Err(DatasetsError::InvalidFormat(
                "WikiText103Config: n_articles must be > 0".to_string(),
            ));
        }

        let zipf = ZipfSampler::new(config.vocab_size);
        let mut rng = Lcg::new(config.seed);
        let avg_paras = config.avg_paragraphs.max(1) as f64;
        let avg_toks = config.avg_para_tokens.max(1) as f64;

        let articles: Vec<Vec<Vec<usize>>> = (0..config.n_articles)
            .map(|_| {
                let n_paras = rng.next_poisson(avg_paras).max(1);
                (0..n_paras)
                    .map(|_| {
                        let n_toks = rng.next_poisson(avg_toks).max(1);
                        (0..n_toks)
                            .map(|_| zipf.sample(rng.next_f64()))
                            .collect::<Vec<usize>>()
                    })
                    .collect::<Vec<Vec<usize>>>()
            })
            .collect();

        Ok(Self {
            articles,
            vocab_size: config.vocab_size,
        })
    }

    /// All articles in `[article][paragraph][token]` order.
    pub fn articles(&self) -> &[Vec<Vec<usize>>] {
        &self.articles
    }

    /// All tokens from all articles and paragraphs concatenated.
    pub fn flat_tokens(&self) -> Vec<usize> {
        self.articles
            .iter()
            .flat_map(|art| art.iter().flat_map(|para| para.iter().copied()))
            .collect()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Load a WikiText-format dataset from a file.
    ///
    /// Articles are delimited by lines matching `= Title =` (single `=` on
    /// each side, not `==`). Paragraphs are separated by blank lines.
    /// Words are lowercased and space-separated. The most frequent
    /// `vocab_size - 1` words get indices 1..vocab_size; all others map to 0
    /// (`<unk>`).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or read.
    pub fn load_from_text(path: impl AsRef<Path>, vocab_size: usize) -> Result<Self> {
        let file = fs::File::open(path.as_ref()).map_err(DatasetsError::IoError)?;
        let reader = BufReader::new(file);

        let mut freq: HashMap<String, usize> = HashMap::new();
        let mut raw_articles: Vec<Vec<Vec<String>>> = Vec::new();
        let mut current_article: Vec<Vec<String>> = Vec::new();
        let mut current_para: Vec<String> = Vec::new();

        let is_top_level_title = |line: &str| -> bool {
            let s = line.trim();
            // Matches "= Some Title =" but NOT "== Subtitle ==" etc.
            if s.starts_with("= ") && s.ends_with(" =") {
                let inner = &s[2..s.len() - 2];
                !inner.contains('=')
            } else {
                false
            }
        };

        for line in reader.lines() {
            let line = line.map_err(DatasetsError::IoError)?;
            if is_top_level_title(&line) {
                // Save current paragraph and article
                if !current_para.is_empty() {
                    current_article.push(std::mem::take(&mut current_para));
                }
                if !current_article.is_empty() {
                    raw_articles.push(std::mem::take(&mut current_article));
                }
                current_article = Vec::new();
            } else if line.trim().is_empty() {
                // Blank line = paragraph boundary
                if !current_para.is_empty() {
                    current_article.push(std::mem::take(&mut current_para));
                }
            } else {
                let words: Vec<String> =
                    line.split_whitespace().map(|w| w.to_lowercase()).collect();
                for w in &words {
                    *freq.entry(w.clone()).or_insert(0) += 1;
                }
                current_para.extend(words);
            }
        }
        // Flush
        if !current_para.is_empty() {
            current_article.push(current_para);
        }
        if !current_article.is_empty() {
            raw_articles.push(current_article);
        }

        // Build vocabulary (top-N by frequency)
        let mut sorted_words: Vec<(String, usize)> = freq.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let vocab: HashMap<String, usize> = sorted_words
            .iter()
            .take(vocab_size.saturating_sub(1))
            .enumerate()
            .map(|(i, (word, _))| (word.clone(), i + 1))
            .collect();

        let articles: Vec<Vec<Vec<usize>>> = raw_articles
            .iter()
            .map(|art| {
                art.iter()
                    .map(|para| para.iter().map(|w| *vocab.get(w).unwrap_or(&0)).collect())
                    .collect()
            })
            .collect();

        Ok(Self {
            articles,
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
    fn test_wt103_shape() {
        let cfg = WikiText103Config {
            vocab_size: 5_000,
            n_articles: 10,
            avg_paragraphs: 3,
            avg_para_tokens: 20,
            seed: 1,
        };
        let ds = WikiText103Dataset::generate(cfg.clone()).expect("generate failed");
        assert_eq!(ds.articles().len(), cfg.n_articles);
        assert_eq!(ds.vocab_size(), cfg.vocab_size);
        assert!(!ds.flat_tokens().is_empty());
    }

    #[test]
    fn test_wt103_deterministic() {
        let cfg = WikiText103Config {
            vocab_size: 1_000,
            n_articles: 5,
            avg_paragraphs: 2,
            avg_para_tokens: 10,
            seed: 99,
        };
        let ds1 = WikiText103Dataset::generate(cfg.clone()).expect("generate failed");
        let ds2 = WikiText103Dataset::generate(cfg).expect("generate failed");
        assert_eq!(ds1.flat_tokens(), ds2.flat_tokens());
    }

    #[test]
    fn test_wt103_token_range() {
        let cfg = WikiText103Config {
            vocab_size: 500,
            n_articles: 5,
            avg_paragraphs: 2,
            avg_para_tokens: 15,
            seed: 7,
        };
        let ds = WikiText103Dataset::generate(cfg.clone()).expect("generate failed");
        for tok in ds.flat_tokens() {
            assert!(tok < cfg.vocab_size, "token {tok} out of range");
        }
    }

    #[test]
    fn test_wt103_paragraph_structure() {
        let cfg = WikiText103Config {
            vocab_size: 200,
            n_articles: 4,
            avg_paragraphs: 3,
            avg_para_tokens: 10,
            seed: 42,
        };
        let ds = WikiText103Dataset::generate(cfg).expect("generate failed");
        for art in ds.articles() {
            assert!(
                !art.is_empty(),
                "each article must have at least 1 paragraph"
            );
            for para in art {
                assert!(
                    !para.is_empty(),
                    "each paragraph must have at least 1 token"
                );
            }
        }
    }

    #[test]
    fn test_wt103_load_from_text() {
        let mut tmp = std::env::temp_dir();
        tmp.push("wt103_test.txt");
        {
            let mut f = fs::File::create(&tmp).expect("create tmp");
            writeln!(f, "= First Article =").expect("write");
            writeln!(f, "This is the first paragraph of the article.").expect("write");
            writeln!(f).expect("write");
            writeln!(f, "Another paragraph with more words and content here.").expect("write");
            writeln!(f, "= Second Article =").expect("write");
            writeln!(f, "The second article starts here with some text.").expect("write");
        }
        let ds = WikiText103Dataset::load_from_text(&tmp, 50).expect("load failed");
        assert_eq!(ds.vocab_size(), 50);
        assert_eq!(ds.articles().len(), 2);
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_wt103_error_zero_articles() {
        let cfg = WikiText103Config {
            n_articles: 0,
            ..WikiText103Config::default()
        };
        assert!(WikiText103Dataset::generate(cfg).is_err());
    }

    #[test]
    fn test_wt103_flat_tokens_count() {
        let cfg = WikiText103Config {
            vocab_size: 100,
            n_articles: 3,
            avg_paragraphs: 2,
            avg_para_tokens: 5,
            seed: 13,
        };
        let ds = WikiText103Dataset::generate(cfg).expect("generate failed");
        let flat = ds.flat_tokens();
        let expected: usize = ds
            .articles()
            .iter()
            .flat_map(|art| art.iter().map(|p| p.len()))
            .sum();
        assert_eq!(flat.len(), expected);
    }
}
