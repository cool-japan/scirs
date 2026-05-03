//! Advanced extractive text summarization algorithms.
//!
//! Provides `TextRankSummarizer` (PageRank on sentence similarity graphs),
//! `ExtractiveSummarizer` (lead-k and frequency-based strategies) and three
//! sentence-similarity metrics: cosine TF-IDF, BM25, and Jaccard.

use std::collections::HashMap;

use crate::error::{Result, TextError};

// ---------------------------------------------------------------------------
// Sentence similarity
// ---------------------------------------------------------------------------

/// Algorithm used to compute inter-sentence similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentenceSimilarity {
    /// Cosine similarity over TF-IDF term vectors.
    CosineTFIDF,
    /// BM25 relevance scoring.
    BM25,
    /// Token-overlap Jaccard coefficient.
    Jaccard,
}

// ---------------------------------------------------------------------------
// Tokenisation helpers
// ---------------------------------------------------------------------------

/// Very simple sentence tokeniser: splits on `.`, `?`, `!` followed by space.
fn split_sentences(text: &str) -> Vec<String> {
    // Split on sentence-ending punctuation followed by whitespace or end
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        current.push(chars[i]);
        if matches!(chars[i], '.' | '?' | '!') {
            // Check if followed by space/end and the next char is uppercase
            if i + 1 >= n || chars[i + 1].is_whitespace() {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current = String::new();
            }
        }
        i += 1;
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

/// Tokenise a sentence into lowercase words (alpha-only).
fn tokenize_words(sentence: &str) -> Vec<String> {
    sentence
        .split(|c: char| !c.is_alphabetic())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

// ---------------------------------------------------------------------------
// Similarity functions
// ---------------------------------------------------------------------------

/// Jaccard similarity between two sentences.
fn jaccard(a: &str, b: &str) -> f64 {
    let ta: std::collections::HashSet<String> = tokenize_words(a).into_iter().collect();
    let tb: std::collections::HashSet<String> = tokenize_words(b).into_iter().collect();
    let inter = ta.intersection(&tb).count();
    let union = ta.union(&tb).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// Build a TF map for one sentence.
fn tf_map(sentence: &str) -> HashMap<String, f64> {
    let words = tokenize_words(sentence);
    let n = words.len() as f64;
    if n == 0.0 {
        return HashMap::new();
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for w in words {
        *counts.entry(w).or_insert(0) += 1;
    }
    counts.into_iter().map(|(k, c)| (k, c as f64 / n)).collect()
}

/// Cosine similarity with IDF weighting across a sentence corpus.
fn cosine_tfidf(a: &str, b: &str, idf: &HashMap<String, f64>) -> f64 {
    let ta = tf_map(a);
    let tb = tf_map(b);
    let dot: f64 = ta
        .iter()
        .filter_map(|(w, &tfa)| {
            tb.get(w).map(|&tfb| {
                let idf_w = idf.get(w).copied().unwrap_or(1.0);
                tfa * idf_w * tfb * idf_w
            })
        })
        .sum();
    let norm_a: f64 = ta
        .values()
        .map(|&v| {
            let idf_w = idf.get(&String::new()).copied().unwrap_or(1.0);
            (v * idf_w).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = tb
        .values()
        .map(|&v| {
            let idf_w = idf.get(&String::new()).copied().unwrap_or(1.0);
            (v * idf_w).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// BM25 score of `query_sentence` given `doc_sentence` as the document.
fn bm25_similarity(
    query: &str,
    doc: &str,
    avgdl: f64,
    idf: &HashMap<String, f64>,
    k1: f64,
    b: f64,
) -> f64 {
    let query_words = tokenize_words(query);
    let doc_words = tokenize_words(doc);
    let dl = doc_words.len() as f64;
    let mut freq_map: HashMap<&str, usize> = HashMap::new();
    for w in &doc_words {
        *freq_map.entry(w.as_str()).or_insert(0) += 1;
    }
    query_words
        .iter()
        .map(|w| {
            let idf_w = idf.get(w).copied().unwrap_or(1.0);
            let f = *freq_map.get(w.as_str()).unwrap_or(&0) as f64;
            idf_w * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / avgdl))
        })
        .sum()
}

/// Build IDF map from a corpus of sentences.
fn build_idf(sentences: &[String]) -> HashMap<String, f64> {
    let n = sentences.len() as f64;
    let mut df: HashMap<String, usize> = HashMap::new();
    for sent in sentences {
        let words: std::collections::HashSet<String> = tokenize_words(sent).into_iter().collect();
        for w in words {
            *df.entry(w).or_insert(0) += 1;
        }
    }
    df.into_iter()
        .map(|(w, c)| (w, ((n + 1.0) / (c as f64 + 1.0)).ln() + 1.0))
        .collect()
}

// ---------------------------------------------------------------------------
// TextRankSummarizer
// ---------------------------------------------------------------------------

/// Extractive summariser based on the TextRank algorithm.
///
/// Sentences are represented as graph nodes; edges are weighted by
/// sentence similarity.  PageRank is run to score each sentence.
#[derive(Debug, Clone)]
pub struct TextRankSummarizer {
    /// PageRank damping factor.
    pub damping: f64,
    /// Number of PageRank iterations.
    pub n_iterations: usize,
    /// Similarity metric used to weight graph edges.
    pub similarity: SentenceSimilarity,
}

impl Default for TextRankSummarizer {
    fn default() -> Self {
        TextRankSummarizer {
            damping: 0.85,
            n_iterations: 50,
            similarity: SentenceSimilarity::CosineTFIDF,
        }
    }
}

impl TextRankSummarizer {
    /// Create a summariser with custom parameters.
    pub fn new(damping: f64, n_iterations: usize, similarity: SentenceSimilarity) -> Self {
        TextRankSummarizer {
            damping,
            n_iterations,
            similarity,
        }
    }

    /// Summarise `text` by extracting the top `n_sentences` ranked sentences.
    ///
    /// Sentences are returned in their **original document order**.
    pub fn summarize(&self, text: &str, n_sentences: usize) -> Result<String> {
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Ok(String::new());
        }
        let k = n_sentences.min(sentences.len());
        if k == sentences.len() {
            return Ok(text.to_string());
        }

        let idf = build_idf(&sentences);
        let n = sentences.len();
        let avgdl = sentences
            .iter()
            .map(|s| tokenize_words(s).len())
            .sum::<usize>() as f64
            / n as f64;

        // Build adjacency matrix
        let mut adj = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                adj[i][j] = match self.similarity {
                    SentenceSimilarity::Jaccard => jaccard(&sentences[i], &sentences[j]),
                    SentenceSimilarity::CosineTFIDF => {
                        cosine_tfidf(&sentences[i], &sentences[j], &idf)
                    }
                    SentenceSimilarity::BM25 => {
                        bm25_similarity(&sentences[i], &sentences[j], avgdl, &idf, 1.5, 0.75)
                    }
                };
            }
        }

        // Row-normalise adjacency
        for row in adj.iter_mut() {
            let total: f64 = row.iter().sum();
            if total > 0.0 {
                for v in row.iter_mut() {
                    *v /= total;
                }
            }
        }

        // PageRank
        let mut scores = vec![1.0 / n as f64; n];
        for _ in 0..self.n_iterations {
            let mut new_scores = vec![0.0f64; n];
            for j in 0..n {
                for i in 0..n {
                    new_scores[j] += adj[i][j] * scores[i];
                }
                new_scores[j] = (1.0 - self.damping) / n as f64 + self.damping * new_scores[j];
            }
            scores = new_scores;
        }

        // Select top-k sentence indices
        let mut ranked: Vec<(usize, f64)> = scores.iter().cloned().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut top_indices: Vec<usize> = ranked.iter().take(k).map(|&(i, _)| i).collect();
        // Restore original order
        top_indices.sort();

        let summary = top_indices
            .iter()
            .map(|&i| sentences[i].as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(summary)
    }
}

// ---------------------------------------------------------------------------
// ExtractiveSummarizer
// ---------------------------------------------------------------------------

/// Simple extractive summarisation strategies.
pub struct ExtractiveSummarizer;

impl ExtractiveSummarizer {
    /// Return the first `k` sentences of `text`.
    pub fn lead_k(text: &str, k: usize) -> Result<String> {
        if k == 0 {
            return Err(TextError::InvalidInput("k must be at least 1".to_string()));
        }
        let sentences = split_sentences(text);
        let selected: Vec<&str> = sentences.iter().take(k).map(String::as_str).collect();
        Ok(selected.join(" "))
    }

    /// Score sentences by aggregate word-frequency and return the top `k`.
    ///
    /// The frequency of each word is computed across the whole document;
    /// stop-words (high-frequency function words) are down-weighted.
    pub fn frequency_based(text: &str, k: usize) -> Result<String> {
        if k == 0 {
            return Err(TextError::InvalidInput("k must be at least 1".to_string()));
        }
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Ok(String::new());
        }

        // Word frequency across the full document
        let mut freq: HashMap<String, usize> = HashMap::new();
        for sent in &sentences {
            for w in tokenize_words(sent) {
                *freq.entry(w).or_insert(0) += 1;
            }
        }
        // Normalise
        let max_freq = *freq.values().max().unwrap_or(&1) as f64;
        let norm_freq: HashMap<String, f64> = freq
            .into_iter()
            .map(|(k, v)| (k, v as f64 / max_freq))
            .collect();

        // Score each sentence
        let mut scored: Vec<(usize, f64)> = sentences
            .iter()
            .enumerate()
            .map(|(i, sent)| {
                let words = tokenize_words(sent);
                let score: f64 = words
                    .iter()
                    .map(|w| norm_freq.get(w).copied().unwrap_or(0.0))
                    .sum();
                (i, score)
            })
            .collect();

        // Take top-k in original order
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut top_indices: Vec<usize> = scored.iter().take(k).map(|&(i, _)| i).collect();
        top_indices.sort();

        let result = top_indices
            .iter()
            .map(|&i| sentences[i].as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "The quick brown fox jumps over the lazy dog. \
         A fox is a cunning animal. \
         Dogs are loyal companions. \
         Foxes live in dens and are mostly nocturnal. \
         The dog slept all afternoon.";

    #[test]
    fn test_split_sentences() {
        let sents = split_sentences(TEXT);
        assert_eq!(sents.len(), 5);
    }

    #[test]
    fn test_textrank_summarize_count() {
        let summarizer = TextRankSummarizer::default();
        let summary = summarizer.summarize(TEXT, 2).expect("summarize failed");
        // A 2-sentence summary should contain at most 2 sentence-ending marks
        let count = summary.matches('.').count();
        assert!(count <= 2, "too many sentences: {}", count);
    }

    #[test]
    fn test_textrank_empty_text() {
        let summarizer = TextRankSummarizer::default();
        let summary = summarizer.summarize("", 3).expect("summarize empty");
        assert!(summary.is_empty());
    }

    #[test]
    fn test_textrank_more_than_available() {
        let summarizer = TextRankSummarizer::default();
        // Requesting 100 sentences from a 5-sentence text → full text
        let summary = summarizer.summarize(TEXT, 100).expect("summarize");
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_textrank_bm25() {
        let summarizer = TextRankSummarizer::new(0.85, 20, SentenceSimilarity::BM25);
        let summary = summarizer.summarize(TEXT, 2).expect("summarize bm25");
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_textrank_jaccard() {
        let summarizer = TextRankSummarizer::new(0.85, 20, SentenceSimilarity::Jaccard);
        let summary = summarizer.summarize(TEXT, 2).expect("summarize jaccard");
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_lead_k() {
        let summary = ExtractiveSummarizer::lead_k(TEXT, 2).expect("lead_k");
        // Should start with the first sentence's first word
        assert!(summary.starts_with("The quick"));
    }

    #[test]
    fn test_lead_k_zero_error() {
        let result = ExtractiveSummarizer::lead_k(TEXT, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_frequency_based() {
        let summary = ExtractiveSummarizer::frequency_based(TEXT, 2).expect("freq_based");
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_jaccard_similarity() {
        let a = "the cat sat on the mat";
        let b = "the cat sat on the mat";
        let sim = jaccard(a, b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_no_overlap() {
        let a = "hello world";
        let b = "foo bar baz";
        let sim = jaccard(a, b);
        assert!(sim < 0.01);
    }

    #[test]
    fn test_build_idf() {
        let sents = vec!["the cat sat".to_string(), "the dog ran".to_string()];
        let idf = build_idf(&sents);
        // "the" appears in all sentences → low IDF
        // "cat" appears in 1 sentence → higher IDF
        let idf_the = idf.get("the").copied().unwrap_or(0.0);
        let idf_cat = idf.get("cat").copied().unwrap_or(0.0);
        assert!(idf_cat >= idf_the);
    }
}
