//! Advanced Keyword Extraction (`keywords.rs`)
//!
//! Provides three complementary single-document keyword extraction algorithms:
//!
//! - [`Rake`] -- Rapid Automatic Keyword Extraction (phrase-level)
//! - [`Yake`] -- Yet Another Keyword Extractor (statistical, n-gram)
//! - [`textrank_keywords`] -- TextRank graph-based keyword extraction
//!
//! All algorithms are purely statistical / heuristic; no external models or
//! corpora are required.

use crate::error::{Result, TextError};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Stop words
// ---------------------------------------------------------------------------

/// Comprehensive English stop-word list used by RAKE and YAKE.
pub fn english_stop_words() -> HashSet<String> {
    const WORDS: &[&str] = &[
        // Articles & conjunctions
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "nor",
        "for",
        "yet",
        "so",
        // Prepositions
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "about",
        "against",
        "along",
        "around",
        "up",
        "down",
        // Pronouns
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        // Auxiliary verbs
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        // Common adverbs / adjectives
        "not",
        "no",
        "nor",
        "very",
        "just",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "than",
        "too",
        "also",
        "any",
        "because",
        "if",
        "while",
        // Numbers spelled out
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ];
    WORDS.iter().map(|w| w.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

fn words_lower(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let tail = current.trim().to_string();
            if !tail.is_empty() {
                sentences.push(tail);
            }
            current.clear();
        }
    }
    let tail = current.trim().to_string();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

// ---------------------------------------------------------------------------
// RAKE
// ---------------------------------------------------------------------------

/// RAKE (Rapid Automatic Keyword Extraction).
///
/// Splits text into candidate phrases using stop-words as delimiters, then
/// scores each phrase using a word co-degree / frequency ratio.
///
/// # Example
///
/// ```rust
/// use scirs2_text::keywords::Rake;
///
/// let rake = Rake::new();
/// let keywords = rake.extract("Rust is a systems programming language.", 5).unwrap();
/// assert!(!keywords.is_empty());
/// ```
pub struct Rake {
    stop_words: HashSet<String>,
    /// Minimum word length (shorter words are treated as stop-words).
    pub min_word_len: usize,
    /// Maximum number of words in an extracted phrase.
    pub max_phrase_words: usize,
}

impl Default for Rake {
    fn default() -> Self {
        Self::new()
    }
}

impl Rake {
    /// Create a new RAKE extractor with the built-in English stop-word list.
    pub fn new() -> Self {
        Self {
            stop_words: english_stop_words(),
            min_word_len: 3,
            max_phrase_words: 5,
        }
    }

    /// Create a RAKE extractor with a custom stop-word list.
    pub fn with_stop_words(words: Vec<String>) -> Self {
        Self {
            stop_words: words.into_iter().collect(),
            min_word_len: 3,
            max_phrase_words: 5,
        }
    }

    /// Extract up to `top_n` keyword phrases from `text`.
    ///
    /// Returns `(phrase, score)` pairs sorted by score descending.
    pub fn extract(&self, text: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        if top_n == 0 {
            return Err(TextError::InvalidInput("top_n must be > 0".to_string()));
        }
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let candidates = self.generate_candidates(text);
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let mut word_freq: HashMap<String, f64> = HashMap::new();
        let mut word_degree: HashMap<String, f64> = HashMap::new();

        for phrase in &candidates {
            let words = self.phrase_words(phrase);
            let degree = words.len() as f64;
            for word in &words {
                *word_freq.entry(word.clone()).or_insert(0.0) += 1.0;
                *word_degree.entry(word.clone()).or_insert(0.0) += degree;
            }
        }

        let word_score: HashMap<String, f64> = word_freq
            .iter()
            .map(|(w, freq)| {
                let deg = word_degree.get(w).copied().unwrap_or(0.0);
                (w.clone(), if *freq > 0.0 { deg / freq } else { 0.0 })
            })
            .collect();

        let mut seen: HashSet<String> = HashSet::new();
        let mut scored: Vec<(String, f64)> = Vec::new();

        for phrase in &candidates {
            let key = phrase.to_lowercase();
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);

            let words = self.phrase_words(phrase);
            if words.is_empty() {
                continue;
            }
            let score: f64 = words
                .iter()
                .map(|w| word_score.get(w).copied().unwrap_or(0.0))
                .sum();
            scored.push((phrase.clone(), score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_n);
        Ok(scored)
    }

    fn is_delimiter(&self, word: &str) -> bool {
        word.is_empty()
            || self.stop_words.contains(&word.to_lowercase())
            || word.len() < self.min_word_len
            || word.chars().all(|c| !c.is_alphanumeric())
    }

    fn phrase_words(&self, phrase: &str) -> Vec<String> {
        phrase
            .split_whitespace()
            .filter(|w| w.len() >= self.min_word_len)
            .map(|w| w.to_lowercase())
            .collect()
    }

    fn generate_candidates(&self, text: &str) -> Vec<String> {
        let mut candidates: Vec<String> = Vec::new();
        let mut current: Vec<String> = Vec::new();

        for raw_token in text.split(|c: char| !c.is_alphanumeric() && c != '\'') {
            let token = raw_token.trim().to_lowercase();
            if self.is_delimiter(&token) {
                if !current.is_empty() {
                    if current.len() <= self.max_phrase_words {
                        candidates.push(current.join(" "));
                    }
                    current.clear();
                }
            } else {
                current.push(token);
            }
        }
        if !current.is_empty() && current.len() <= self.max_phrase_words {
            candidates.push(current.join(" "));
        }
        candidates
    }
}

// ---------------------------------------------------------------------------
// YAKE!
// ---------------------------------------------------------------------------

/// YAKE! (Yet Another Keyword Extractor).
///
/// Statistical, single-document keyword extraction. Lower YAKE scores
/// indicate more important keywords. The `extract` method returns keywords
/// sorted by score ascending (most important first).
///
/// # Example
///
/// ```rust
/// use scirs2_text::keywords::Yake;
///
/// let yake = Yake::new(2);
/// let keywords = yake.extract("Rust is a systems programming language.", 5).unwrap();
/// assert!(!keywords.is_empty());
/// ```
pub struct Yake {
    /// ISO 639-1 language code.
    pub language: String,
    /// Maximum n-gram size.
    pub max_ngram_size: usize,
    /// Deduplication threshold (Jaccard-based similarity).
    pub dedup_threshold: f64,
    /// Context window size (words to left/right for co-occurrence).
    pub window_size: usize,
}

impl Default for Yake {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Yake {
    /// Create a new YAKE extractor.
    pub fn new(max_ngram: usize) -> Self {
        Self {
            language: "en".to_string(),
            max_ngram_size: max_ngram.max(1),
            dedup_threshold: 0.9,
            window_size: 2,
        }
    }

    /// Extract up to `top_n` keywords from `text`.
    ///
    /// Returns `(keyword, score)` sorted by score ascending (lower = more important).
    pub fn extract(&self, text: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        if top_n == 0 {
            return Err(TextError::InvalidInput("top_n must be > 0".to_string()));
        }
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let stop_words = english_stop_words();
        let total_words: Vec<String> = words_lower(text);
        let n = total_words.len();

        if n == 0 {
            return Ok(Vec::new());
        }

        // Per-word statistics
        let mut tf: HashMap<String, usize> = HashMap::new();
        let mut first_pos: HashMap<String, usize> = HashMap::new();
        let mut capitalized: HashMap<String, bool> = HashMap::new();
        let mut left_ctx: HashMap<String, HashSet<String>> = HashMap::new();
        let mut right_ctx: HashMap<String, HashSet<String>> = HashMap::new();

        let orig_words: Vec<&str> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();

        for (i, ow) in orig_words.iter().enumerate() {
            let lower = ow.to_lowercase();
            *tf.entry(lower.clone()).or_insert(0) += 1;
            first_pos.entry(lower.clone()).or_insert(i);
            let is_cap = ow.chars().next().is_some_and(|c| c.is_uppercase());
            capitalized.entry(lower.clone()).or_insert(is_cap);
        }

        for i in 0..n {
            let word = &total_words[i];
            for delta in 1..=self.window_size {
                if i + delta < n {
                    let right = total_words[i + delta].clone();
                    right_ctx
                        .entry(word.clone())
                        .or_default()
                        .insert(right.clone());
                    left_ctx.entry(right).or_default().insert(word.clone());
                }
            }
        }

        let sigma = 1.0_f64;
        let tf_max = tf.values().copied().max().unwrap_or(1) as f64;

        let mut word_scores: HashMap<String, f64> = HashMap::new();

        for (word, &freq) in &tf {
            if stop_words.contains(word) || word.len() < 2 {
                continue;
            }
            let tf_norm = freq as f64 / tf_max;
            let pos = first_pos.get(word).copied().unwrap_or(0) as f64;
            let rel_pos = 1.0 - pos / n.max(1) as f64;
            let left_div = left_ctx.get(word).map_or(0, |s| s.len()) as f64;
            let right_div = right_ctx.get(word).map_or(0, |s| s.len()) as f64;
            let disp = (left_div + right_div + sigma) / (2.0 * freq as f64 + sigma);
            let cap_bonus = if *capitalized.get(word).unwrap_or(&false) {
                0.1
            } else {
                0.0
            };
            let score = (tf_norm * disp) / (rel_pos + cap_bonus + sigma);
            word_scores.insert(word.clone(), score);
        }

        // Build n-gram candidates
        let mut ngram_scores: Vec<(String, f64)> = Vec::new();

        for n_size in 1..=self.max_ngram_size {
            let candidates = self.generate_ngrams(&total_words, n_size, &stop_words);
            for ngram in candidates {
                let words: Vec<&str> = ngram.split_whitespace().collect();
                if words.is_empty() {
                    continue;
                }
                if n_size > 1 {
                    let first = words[0];
                    let last = words[words.len() - 1];
                    if stop_words.contains(first) || stop_words.contains(last) {
                        continue;
                    }
                }

                let prod: f64 = words
                    .iter()
                    .map(|w| word_scores.get(*w).copied().unwrap_or(1.0))
                    .product();

                let coherence: f64 = if n_size > 1 {
                    let pairs = n_size - 1;
                    let pair_count: f64 = (0..pairs)
                        .map(|p| {
                            let left = words[p];
                            let right = words[p + 1];
                            right_ctx
                                .get(left)
                                .map_or(0, |s| if s.contains(right) { 1 } else { 0 })
                                as f64
                        })
                        .sum();
                    (pair_count / pairs as f64).max(0.01)
                } else {
                    1.0
                };

                let score = prod / (n_size as f64 * coherence + sigma);
                ngram_scores.push((ngram, score));
            }
        }

        ngram_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let deduped = self.deduplicate(ngram_scores);
        let mut result: Vec<(String, f64)> = deduped.into_iter().take(top_n).collect();

        if let Some(max_s) = result.iter().map(|(_, s)| *s).reduce(f64::max) {
            if max_s > 0.0 {
                for (_, s) in result.iter_mut() {
                    *s /= max_s;
                }
            }
        }

        Ok(result)
    }

    fn generate_ngrams(
        &self,
        words: &[String],
        n: usize,
        stop_words: &HashSet<String>,
    ) -> Vec<String> {
        if words.len() < n {
            return Vec::new();
        }
        let mut ngrams: HashSet<String> = HashSet::new();

        for window in words.windows(n) {
            if window.iter().all(|w| stop_words.contains(w.as_str())) {
                continue;
            }
            if window.iter().any(|w| w.len() < 2) {
                continue;
            }
            ngrams.insert(window.join(" "));
        }
        ngrams.into_iter().collect()
    }

    fn deduplicate(&self, sorted: Vec<(String, f64)>) -> Vec<(String, f64)> {
        let mut result: Vec<(String, f64)> = Vec::new();

        for candidate in sorted {
            let tokens_c: HashSet<&str> = candidate.0.split_whitespace().collect();
            let is_dup = result.iter().any(|(existing, _)| {
                let tokens_e: HashSet<&str> = existing.split_whitespace().collect();
                let inter = tokens_c.intersection(&tokens_e).count();
                let union = tokens_c.union(&tokens_e).count();
                if union == 0 {
                    return false;
                }
                (inter as f64 / union as f64) >= self.dedup_threshold
            });
            if !is_dup {
                result.push(candidate);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// TextRank keyword extraction (standalone function)
// ---------------------------------------------------------------------------

/// TextRank-based keyword extraction.
///
/// Returns up to `top_n` `(keyword, score)` pairs sorted by score descending.
///
/// # Errors
///
/// Returns [`TextError::InvalidInput`] if `top_n` is zero or `window` < 2.
pub fn textrank_keywords(text: &str, top_n: usize, window: usize) -> Result<Vec<(String, f64)>> {
    if top_n == 0 {
        return Err(TextError::InvalidInput("top_n must be > 0".to_string()));
    }
    if window < 2 {
        return Err(TextError::InvalidInput("window must be >= 2".to_string()));
    }
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let stop_words = english_stop_words();
    let words: Vec<String> = words_lower(text);
    let filtered: Vec<String> = words
        .iter()
        .filter(|w| w.len() >= 3 && !stop_words.contains(*w))
        .cloned()
        .collect();

    if filtered.is_empty() {
        return Ok(Vec::new());
    }

    let mut graph: HashMap<String, HashMap<String, f64>> = HashMap::new();

    for win in filtered.windows(window) {
        for i in 0..win.len() {
            for j in (i + 1)..win.len() {
                let a = &win[i];
                let b = &win[j];
                *graph
                    .entry(a.clone())
                    .or_default()
                    .entry(b.clone())
                    .or_insert(0.0) += 1.0;
                *graph
                    .entry(b.clone())
                    .or_default()
                    .entry(a.clone())
                    .or_insert(0.0) += 1.0;
            }
        }
    }

    let nodes: Vec<String> = graph.keys().cloned().collect();
    let n = nodes.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let node_idx: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i))
        .collect();

    const DAMPING: f64 = 0.85;
    const MAX_ITER: usize = 100;
    const EPS: f64 = 1e-5;

    let mut scores = vec![1.0_f64 / n as f64; n];

    let out_sums: Vec<f64> = nodes
        .iter()
        .map(|node| {
            graph
                .get(node)
                .map(|nbrs| nbrs.values().sum())
                .unwrap_or(0.0)
        })
        .collect();

    for _ in 0..MAX_ITER {
        let mut new_scores = vec![(1.0 - DAMPING) / n as f64; n];
        for (j, node_j) in nodes.iter().enumerate() {
            if out_sums[j] <= 0.0 {
                continue;
            }
            if let Some(nbrs) = graph.get(node_j) {
                for (nbr, &weight) in nbrs {
                    if let Some(&i) = node_idx.get(nbr.as_str()) {
                        new_scores[i] += DAMPING * (weight / out_sums[j]) * scores[j];
                    }
                }
            }
        }
        let diff: f64 = scores
            .iter()
            .zip(&new_scores)
            .map(|(a, b)| (a - b).abs())
            .sum();
        scores = new_scores;
        if diff < EPS {
            break;
        }
    }

    let word_scores: HashMap<String, f64> = nodes.iter().cloned().zip(scores).collect();

    let all_words: Vec<String> = words_lower(text);
    let mut phrases: Vec<(String, f64)> = Vec::new();
    let mut phrase_buf: Vec<String> = Vec::new();
    let mut phrase_score = 0.0_f64;

    for w in &all_words {
        if let Some(&sc) = word_scores.get(w) {
            phrase_buf.push(w.clone());
            phrase_score += sc;
        } else {
            if !phrase_buf.is_empty() {
                phrases.push((phrase_buf.join(" "), phrase_score));
                phrase_buf.clear();
                phrase_score = 0.0;
            }
        }
    }
    if !phrase_buf.is_empty() {
        phrases.push((phrase_buf.join(" "), phrase_score));
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut unique: Vec<(String, f64)> = Vec::new();
    for (phrase, score) in phrases {
        if !seen.contains(&phrase) {
            seen.insert(phrase.clone());
            unique.push((phrase, score));
        }
    }

    unique.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    unique.truncate(top_n);
    Ok(unique)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TEXT: &str = "Rust is a systems programming language that runs blazingly fast, \
         prevents segfaults, and guarantees thread safety. \
         Rust programming combines low-level control with high-level ergonomics. \
         Many developers choose Rust for building reliable and efficient software.";

    // ---- Rake tests ----

    #[test]
    fn test_rake_returns_results() {
        let rake = Rake::new();
        let keywords = rake.extract(SAMPLE_TEXT, 5).expect("RAKE should succeed");
        assert!(!keywords.is_empty(), "RAKE should return keywords");
        assert!(keywords.len() <= 5);
    }

    #[test]
    fn test_rake_descending_scores() {
        let rake = Rake::new();
        let keywords = rake.extract(SAMPLE_TEXT, 10).expect("ok");
        for pair in keywords.windows(2) {
            assert!(
                pair[0].1 >= pair[1].1,
                "Scores should be descending: {:?}",
                keywords
            );
        }
    }

    #[test]
    fn test_rake_empty_text() {
        let rake = Rake::new();
        let result = rake.extract("", 5).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_rake_top_n_zero_errors() {
        let rake = Rake::new();
        assert!(rake.extract(SAMPLE_TEXT, 0).is_err());
    }

    #[test]
    fn test_rake_custom_stop_words() {
        let rake = Rake::with_stop_words(vec!["rust".to_string(), "is".to_string()]);
        let keywords = rake
            .extract("Rust is a systems language. Rust is fast.", 5)
            .expect("ok");
        for (kw, _) in &keywords {
            assert!(
                !kw.contains("rust"),
                "Stop-word 'rust' appeared in results: {}",
                kw
            );
        }
    }

    #[test]
    fn test_rake_no_phrases_longer_than_max() {
        let rake = Rake {
            max_phrase_words: 2,
            ..Rake::new()
        };
        let keywords = rake.extract(SAMPLE_TEXT, 10).expect("ok");
        for (kw, _) in &keywords {
            let wc = kw.split_whitespace().count();
            assert!(wc <= 2, "Phrase '{}' exceeds max length", kw);
        }
    }

    #[test]
    fn test_rake_phrase_scores_positive() {
        let rake = Rake::new();
        let keywords = rake.extract(SAMPLE_TEXT, 5).expect("ok");
        for (_, score) in &keywords {
            assert!(*score >= 0.0, "Score should be non-negative");
        }
    }

    // ---- Yake tests ----

    #[test]
    fn test_yake_returns_results() {
        let yake = Yake::new(2);
        let keywords = yake.extract(SAMPLE_TEXT, 5).expect("YAKE should succeed");
        assert!(!keywords.is_empty(), "YAKE should return keywords");
        assert!(keywords.len() <= 5);
    }

    #[test]
    fn test_yake_scores_ascending() {
        let yake = Yake::new(2);
        let keywords = yake.extract(SAMPLE_TEXT, 10).expect("ok");
        for pair in keywords.windows(2) {
            assert!(
                pair[0].1 <= pair[1].1,
                "YAKE scores should be ascending: {:?}",
                keywords
            );
        }
    }

    #[test]
    fn test_yake_empty_text() {
        let yake = Yake::new(2);
        let result = yake.extract("", 5).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_yake_top_n_zero_errors() {
        let yake = Yake::new(2);
        assert!(yake.extract(SAMPLE_TEXT, 0).is_err());
    }

    #[test]
    fn test_yake_unigram_mode() {
        let yake = Yake::new(1);
        let keywords = yake.extract(SAMPLE_TEXT, 5).expect("ok");
        for (kw, _) in &keywords {
            let wc = kw.split_whitespace().count();
            assert_eq!(
                wc, 1,
                "Unigram mode should return single words, got: {}",
                kw
            );
        }
    }

    #[test]
    fn test_yake_bigram_mode() {
        let yake = Yake::new(2);
        let keywords = yake.extract(SAMPLE_TEXT, 10).expect("ok");
        let has_bigram = keywords
            .iter()
            .any(|(kw, _)| kw.split_whitespace().count() == 2);
        assert!(has_bigram, "Bigram mode should include 2-word phrases");
    }

    #[test]
    fn test_yake_scores_normalized() {
        let yake = Yake::new(2);
        let keywords = yake.extract(SAMPLE_TEXT, 10).expect("ok");
        for (kw, score) in &keywords {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "Score {} for '{}' out of [0,1] range",
                score,
                kw
            );
        }
    }

    // ---- textrank_keywords tests ----

    #[test]
    fn test_textrank_returns_results() {
        let keywords = textrank_keywords(SAMPLE_TEXT, 5, 3).expect("ok");
        assert!(!keywords.is_empty());
        assert!(keywords.len() <= 5);
    }

    #[test]
    fn test_textrank_scores_descending() {
        let keywords = textrank_keywords(SAMPLE_TEXT, 10, 3).expect("ok");
        for pair in keywords.windows(2) {
            assert!(pair[0].1 >= pair[1].1, "Scores should be descending");
        }
    }

    #[test]
    fn test_textrank_empty_text() {
        let result = textrank_keywords("", 5, 3).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_textrank_zero_top_n_errors() {
        assert!(textrank_keywords(SAMPLE_TEXT, 0, 3).is_err());
    }

    #[test]
    fn test_textrank_small_window_errors() {
        assert!(textrank_keywords(SAMPLE_TEXT, 5, 1).is_err());
    }

    #[test]
    fn test_textrank_window_size_2() {
        let keywords = textrank_keywords(SAMPLE_TEXT, 5, 2).expect("ok");
        assert!(!keywords.is_empty());
    }

    #[test]
    fn test_textrank_larger_window() {
        let keywords = textrank_keywords(SAMPLE_TEXT, 5, 5).expect("ok");
        assert!(!keywords.is_empty());
    }

    // ---- Cross-algorithm tests ----

    #[test]
    fn test_all_methods_non_empty_for_real_text() {
        let rake_kw = Rake::new().extract(SAMPLE_TEXT, 5).expect("RAKE ok");
        let yake_kw = Yake::new(2).extract(SAMPLE_TEXT, 5).expect("YAKE ok");
        let tr_kw = textrank_keywords(SAMPLE_TEXT, 5, 3).expect("TextRank ok");

        assert!(!rake_kw.is_empty(), "RAKE returned empty");
        assert!(!yake_kw.is_empty(), "YAKE returned empty");
        assert!(!tr_kw.is_empty(), "TextRank returned empty");
    }

    #[test]
    fn test_all_methods_handle_short_text() {
        let short = "Quick brown fox.";
        let _ = Rake::new().extract(short, 3).expect("RAKE ok");
        let _ = Yake::new(1).extract(short, 3).expect("YAKE ok");
        let _ = textrank_keywords(short, 3, 2).expect("TextRank ok");
    }

    #[test]
    fn test_stop_word_list_not_empty() {
        let sw = english_stop_words();
        assert!(!sw.is_empty());
        assert!(sw.contains("the"));
        assert!(sw.contains("and"));
        assert!(sw.contains("is"));
    }
}
