//! Multilingual BPE tokenizer — shared vocabulary across 50+ languages.
//!
//! Implements temperature-based language-balanced corpus sampling (mBERT/XLM-R
//! style) where the probability of sampling from language `l` is
//! `p_l = (size_l / total_size)^alpha / Z`.
//!
//! - `alpha = 1.0` gives proportional (natural) sampling.
//! - `alpha = 0.0` gives uniform sampling across languages.
//! - Intermediate values up-sample low-resource languages.

use crate::error::{Result, TextError};
use std::collections::HashMap;

/// Type alias for the four-tuple returned by `init_base`.
type InitBaseMaps = (
    HashMap<u8, char>,
    HashMap<char, u8>,
    HashMap<String, u32>,
    Vec<String>,
);

// ─── LanguageCorpus ───────────────────────────────────────────────────────────

/// A single-language corpus with an optional manual weight override.
#[derive(Debug, Clone)]
pub struct LanguageCorpus {
    /// BCP-47 language tag or arbitrary identifier.
    pub language: String,
    /// Raw text documents.
    pub texts: Vec<String>,
    /// Optional explicit weight; if `None` the weight is derived from corpus size.
    pub weight: f64,
}

impl LanguageCorpus {
    /// Construct a corpus with a manual weight.
    pub fn new(language: impl Into<String>, texts: Vec<String>, weight: f64) -> Self {
        LanguageCorpus {
            language: language.into(),
            texts,
            weight,
        }
    }

    /// Construct a corpus whose weight is derived from the number of tokens
    /// (whitespace-split words) in all texts.
    pub fn from_texts(language: impl Into<String>, texts: Vec<String>) -> Self {
        let size: f64 = texts
            .iter()
            .map(|t| t.split_whitespace().count() as f64)
            .sum();
        LanguageCorpus {
            language: language.into(),
            texts,
            weight: size.max(1.0),
        }
    }
}

// ─── MultilingualBpeConfig ────────────────────────────────────────────────────

/// Configuration for [`MultilingualBpeTokenizer`] training.
#[derive(Debug, Clone)]
pub struct MultilingualBpeConfig {
    /// Target vocabulary size.
    pub vocab_size: usize,
    /// Temperature exponent for language-balanced sampling.
    ///
    /// `alpha = 1.0` → proportional; `alpha = 0.0` → uniform.
    pub alpha: f64,
    /// Minimum pair frequency for a merge to be accepted.
    pub min_frequency: usize,
    /// Whether to prepend `Ġ` to non-initial words (GPT-2 style).
    pub add_prefix_space: bool,
}

impl Default for MultilingualBpeConfig {
    fn default() -> Self {
        MultilingualBpeConfig {
            vocab_size: 250_000,
            alpha: 0.5,
            min_frequency: 5,
            add_prefix_space: true,
        }
    }
}

// ─── MultilingualBpeTokenizer ─────────────────────────────────────────────────

/// Multilingual byte-level BPE tokenizer.
///
/// Shares a single vocabulary and merge table across all languages, trained
/// with language-balanced sampling to prevent high-resource languages from
/// dominating the vocabulary.
#[derive(Debug, Clone)]
pub struct MultilingualBpeTokenizer {
    /// token string → integer id
    pub vocab: HashMap<String, u32>,
    /// integer id → token string
    pub id_to_token: Vec<String>,
    /// ordered merge rules
    pub merges: Vec<(String, String)>,
    /// byte → unicode char  (GPT-2 table)
    pub byte_encoder: HashMap<u8, char>,
    /// unicode char → byte
    pub byte_decoder: HashMap<char, u8>,
    /// language sampling probabilities used during training
    pub language_probs: HashMap<String, f64>,
}

impl MultilingualBpeTokenizer {
    /// Build base vocabulary (256 byte-level characters).
    fn init_base() -> InitBaseMaps {
        use super::byte_level_bpe::bytes_to_unicode;
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(&b, &c)| (c, b)).collect();

        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut id_to_token: Vec<String> = Vec::new();

        for b in 0u8..=255u8 {
            let ch = byte_encoder[&b];
            let tok = ch.to_string();
            if !vocab.contains_key(&tok) {
                let id = id_to_token.len() as u32;
                vocab.insert(tok.clone(), id);
                id_to_token.push(tok);
            }
        }
        (byte_encoder, byte_decoder, vocab, id_to_token)
    }

    /// Compute temperature-smoothed language sampling probabilities.
    ///
    /// `p_l = weight_l^alpha / sum_k(weight_k^alpha)`
    ///
    /// Returns `None` only when `corpora` is empty.
    pub fn compute_language_probs(
        corpora: &[LanguageCorpus],
        alpha: f64,
    ) -> Option<HashMap<String, f64>> {
        if corpora.is_empty() {
            return None;
        }
        let powered: Vec<f64> = corpora.iter().map(|c| c.weight.powf(alpha)).collect();
        let z: f64 = powered.iter().sum();
        if z == 0.0 {
            // Uniform fallback
            let p = 1.0 / corpora.len() as f64;
            return Some(corpora.iter().map(|c| (c.language.clone(), p)).collect());
        }
        Some(
            corpora
                .iter()
                .zip(powered.iter())
                .map(|(c, &pw)| (c.language.clone(), pw / z))
                .collect(),
        )
    }

    /// Byte-encode a single string into a sequence of unicode-char tokens.
    fn byte_encode(byte_encoder: &HashMap<u8, char>, s: &str) -> Vec<String> {
        s.bytes()
            .map(|b| {
                byte_encoder
                    .get(&b)
                    .copied()
                    .unwrap_or('\u{FFFD}')
                    .to_string()
            })
            .collect()
    }

    /// Apply all known merges (priority-ordered) to a word token sequence.
    fn apply_merges(merges: &[(String, String)], mut word: Vec<String>) -> Vec<String> {
        let merge_rank: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();
        loop {
            if word.len() < 2 {
                break;
            }
            let mut best_rank = usize::MAX;
            let mut best_idx = usize::MAX;
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = merge_rank.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            let merged = format!("{}{}", word[best_idx], word[best_idx + 1]);
            word.remove(best_idx + 1);
            word[best_idx] = merged;
        }
        word
    }

    /// Train a new [`MultilingualBpeTokenizer`] from a set of language corpora.
    ///
    /// Language probabilities are computed via temperature smoothing
    /// (`alpha` parameter in config).  Pair frequencies are accumulated as a
    /// weighted sum: `count += freq * p_l` for each language `l`.
    pub fn train(corpora: &[LanguageCorpus], config: MultilingualBpeConfig) -> Self {
        let (byte_encoder, byte_decoder, mut vocab, mut id_to_token) = Self::init_base();

        let lang_probs = Self::compute_language_probs(corpora, config.alpha).unwrap_or_default();

        // Build per-language word-frequency maps
        let mut lang_word_freq: Vec<(f64, HashMap<Vec<String>, usize>)> =
            Vec::with_capacity(corpora.len());

        for corpus in corpora {
            let prob = lang_probs.get(&corpus.language).copied().unwrap_or(0.0);
            let mut word_freq: HashMap<Vec<String>, usize> = HashMap::new();
            for text in &corpus.texts {
                let mut first = true;
                for word in text.split_whitespace() {
                    let prefixed = if first || !config.add_prefix_space {
                        word.to_string()
                    } else {
                        format!("\u{0120}{}", word)
                    };
                    first = false;
                    let encoded = Self::byte_encode(&byte_encoder, &prefixed);
                    *word_freq.entry(encoded).or_insert(0) += 1;
                }
            }
            lang_word_freq.push((prob, word_freq));
        }

        let mut merges: Vec<(String, String)> = Vec::new();

        // BPE merge loop with language-weighted pair counting
        while vocab.len() < config.vocab_size {
            let mut pair_freq: HashMap<(String, String), f64> = HashMap::new();

            for (prob, word_freq) in &lang_word_freq {
                for (word, &count) in word_freq {
                    let weighted = count as f64 * prob;
                    for i in 0..word.len().saturating_sub(1) {
                        let pair = (word[i].clone(), word[i + 1].clone());
                        *pair_freq.entry(pair).or_insert(0.0) += weighted;
                    }
                }
            }

            let best = pair_freq
                .iter()
                .filter(|(_, &f)| f >= config.min_frequency as f64)
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let (left, right) = match best {
                Some(((l, r), _)) => (l.clone(), r.clone()),
                None => break,
            };

            merges.push((left.clone(), right.clone()));
            let merged = format!("{}{}", left, right);
            let new_id = id_to_token.len() as u32;
            vocab.insert(merged.clone(), new_id);
            id_to_token.push(merged.clone());

            // Apply merge to all language word maps
            for (_, word_freq) in &mut lang_word_freq {
                let updated: HashMap<Vec<String>, usize> = word_freq
                    .drain()
                    .map(|(word, freq)| (merge_pair(&word, &left, &right), freq))
                    .collect();
                *word_freq = updated;
            }
        }

        MultilingualBpeTokenizer {
            vocab,
            id_to_token,
            merges,
            byte_encoder,
            byte_decoder,
            language_probs: lang_probs,
        }
    }

    /// Encode text using this tokenizer.
    ///
    /// Language is accepted as an argument for API symmetry but the encoding
    /// is purely language-agnostic (same BPE merges for all languages).
    pub fn encode_with_language(&self, text: &str, _lang: &str) -> Vec<u32> {
        self.encode(text)
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut first = true;
        for word in text.split_whitespace() {
            let prefixed = if first {
                word.to_string()
            } else {
                format!("\u{0120}{}", word)
            };
            first = false;
            let chars = Self::byte_encode(&self.byte_encoder, &prefixed);
            let merged = Self::apply_merges(&self.merges, chars);
            for tok in merged {
                if let Some(&id) = self.vocab.get(&tok) {
                    ids.push(id);
                }
            }
        }
        ids
    }

    /// Decode token IDs back to a UTF-8 string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(tok) = self.id_to_token.get(id as usize) {
                for ch in tok.chars() {
                    if let Some(&b) = self.byte_decoder.get(&ch) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Compute vocabulary coverage: fraction of encoded tokens that are not
    /// out-of-vocabulary.
    ///
    /// With byte-level encoding there is no true OOV, but this metric measures
    /// the fraction of whitespace-delimited words that are represented as a
    /// single token (perfectly compressed) versus multiple sub-tokens.
    pub fn vocabulary_coverage(&self, texts: &[&str]) -> f64 {
        let mut total_words = 0usize;
        let mut single_token_words = 0usize;
        for text in texts {
            for word in text.split_whitespace() {
                total_words += 1;
                let chars = Self::byte_encode(&self.byte_encoder, word);
                let merged = Self::apply_merges(&self.merges, chars);
                if merged.len() == 1 {
                    single_token_words += 1;
                }
            }
        }
        if total_words == 0 {
            return 0.0;
        }
        single_token_words as f64 / total_words as f64
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Merge all occurrences of (left, right) adjacent pair in `word`.
fn merge_pair(word: &[String], left: &str, right: &str) -> Vec<String> {
    let mut result = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == left && word[i + 1] == right {
            result.push(format!("{}{}", left, right));
            i += 2;
        } else {
            result.push(word[i].clone());
            i += 1;
        }
    }
    result
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_corpora() -> Vec<LanguageCorpus> {
        vec![
            LanguageCorpus::from_texts(
                "en",
                vec![
                    "hello world the quick brown fox".to_string(),
                    "rust is a great language for systems programming".to_string(),
                    "more english text for training the tokenizer".to_string(),
                    "the tokenizer should learn common english word pieces".to_string(),
                ],
            ),
            LanguageCorpus::from_texts(
                "de",
                vec![
                    "hallo welt schnell braun fuchs".to_string(),
                    "rust ist eine großartige sprache".to_string(),
                ],
            ),
            LanguageCorpus::from_texts(
                "fr",
                vec![
                    "bonjour monde renard brun rapide".to_string(),
                    "rust est un langage de programmation".to_string(),
                ],
            ),
        ]
    }

    #[test]
    fn test_language_probs_sum_to_one() {
        let corpora = sample_corpora();
        let probs = MultilingualBpeTokenizer::compute_language_probs(&corpora, 0.5)
            .expect("should compute probs");
        let sum: f64 = probs.values().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "language probs should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_alpha_zero_uniform() {
        let corpora = sample_corpora();
        let probs = MultilingualBpeTokenizer::compute_language_probs(&corpora, 0.0)
            .expect("should compute probs");
        // alpha=0 means weight^0 = 1.0 for all, so uniform
        let expected = 1.0 / corpora.len() as f64;
        for (lang, &p) in &probs {
            assert!(
                (p - expected).abs() < 1e-9,
                "lang {} prob {} != uniform {}",
                lang,
                p,
                expected
            );
        }
    }

    #[test]
    fn test_alpha_one_proportional() {
        let corpora = sample_corpora();
        let total_weight: f64 = corpora.iter().map(|c| c.weight).sum();
        let probs = MultilingualBpeTokenizer::compute_language_probs(&corpora, 1.0)
            .expect("should compute probs");
        for corpus in &corpora {
            let expected = corpus.weight / total_weight;
            let got = probs[&corpus.language];
            assert!(
                (got - expected).abs() < 1e-9,
                "lang {} prob {} != proportional {}",
                corpus.language,
                got,
                expected
            );
        }
    }

    #[test]
    fn test_train_vocab_size() {
        let corpora = sample_corpora();
        let config = MultilingualBpeConfig {
            vocab_size: 400,
            alpha: 0.5,
            min_frequency: 1,
            add_prefix_space: true,
        };
        let tok = MultilingualBpeTokenizer::train(&corpora, config);
        assert!(tok.vocab_size() <= 400);
        assert!(tok.vocab_size() >= 256);
    }

    #[test]
    fn test_encode_with_language() {
        let corpora = sample_corpora();
        let config = MultilingualBpeConfig {
            vocab_size: 400,
            alpha: 0.5,
            min_frequency: 1,
            add_prefix_space: true,
        };
        let tok = MultilingualBpeTokenizer::train(&corpora, config);
        let ids_en = tok.encode_with_language("hello world", "en");
        let ids_de = tok.encode_with_language("hello world", "de");
        // Language-agnostic: same IDs regardless of lang tag
        assert_eq!(ids_en, ids_de);
    }

    #[test]
    fn test_vocabulary_coverage() {
        let corpora = sample_corpora();
        let config = MultilingualBpeConfig {
            vocab_size: 500,
            alpha: 0.5,
            min_frequency: 1,
            add_prefix_space: false,
        };
        let tok = MultilingualBpeTokenizer::train(&corpora, config);
        let coverage = tok.vocabulary_coverage(&["hello", "rust", "world"]);
        assert!(
            (0.0..=1.0).contains(&coverage),
            "coverage should be in [0,1]"
        );
    }
}
