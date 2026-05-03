//! Abstractive summarization module
//!
//! Provides multi-document fusion, sentence compression, and enhanced
//! centroid-based summarization with position bias and query-focused mode.
//! Includes ROUGE-like evaluation metrics (ROUGE-N and ROUGE-L).
//!
//! # Structures
//!
//! - [`FusionSummarizer`] – multi-document fusion via semantic sentence clustering
//! - [`CompressionSummarizer`] – sentence compression by dropping low-importance tokens
//! - [`EnhancedCentroidSummarizer`] – centroid summarization with position bias and
//!   optional query focus
//!
//! # Evaluation
//!
//! - [`rouge_n`] – ROUGE-N recall (n-gram overlap)
//! - [`rouge_l`] – ROUGE-L (LCS-based recall)

use crate::error::{Result, TextError};
use crate::tokenize::{SentenceTokenizer, Tokenizer, WordTokenizer};
use crate::vectorize::{TfidfVectorizer, Vectorizer};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Tokenise a sentence into lowercase word tokens, stripping punctuation.
fn word_tokens(sentence: &str) -> Vec<String> {
    sentence
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Cosine similarity between two ndarray row vectors from the same matrix.
fn cosine_sim_rows(matrix: &Array2<f64>, i: usize, j: usize) -> f64 {
    let cols = matrix.ncols();
    let mut dot = 0.0_f64;
    let mut n1 = 0.0_f64;
    let mut n2 = 0.0_f64;
    for c in 0..cols {
        let a = matrix[[i, c]];
        let b = matrix[[j, c]];
        dot += a * b;
        n1 += a * a;
        n2 += b * b;
    }
    let denom = n1.sqrt() * n2.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Cosine similarity between an owned row vector and a centroid vector.
fn cosine_sim_vec(row: &Array1<f64>, centroid: &Array1<f64>) -> f64 {
    let dot = row.dot(centroid);
    let n1 = row.dot(row).sqrt();
    let n2 = centroid.dot(centroid).sqrt();
    if n1 == 0.0 || n2 == 0.0 {
        0.0
    } else {
        dot / (n1 * n2)
    }
}

/// Build a TF-IDF matrix from a slice of sentence strings.
/// Returns the (matrix, vectorizer) pair so callers can reuse the vocabulary.
fn build_tfidf_matrix(sentences: &[String]) -> Result<(Array2<f64>, TfidfVectorizer)> {
    if sentences.is_empty() {
        return Err(TextError::InvalidInput(
            "Cannot build TF-IDF matrix from empty sentence list".to_string(),
        ));
    }
    let refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    let mut vectorizer = TfidfVectorizer::default();
    let matrix = vectorizer.fit_transform(&refs)?;
    Ok((matrix, vectorizer))
}

// ---------------------------------------------------------------------------
// ScoredSentence (local, richer than text_summarization::ScoredSentence)
// ---------------------------------------------------------------------------

/// A sentence annotated with its origin document index and a relevance score.
#[derive(Debug, Clone)]
pub struct ScoredSentence {
    /// The sentence text.
    pub text: String,
    /// Zero-based position within the original document (or global sentence list).
    pub index: usize,
    /// Zero-based index of the source document (for multi-document fusion).
    pub doc_index: usize,
    /// Relevance score (higher is more important).
    pub score: f64,
}

// ---------------------------------------------------------------------------
// FusionSummarizer
// ---------------------------------------------------------------------------

/// Multi-document fusion summarizer.
///
/// The pipeline is:
/// 1. Extract all sentences from all documents together with TF-IDF scores.
/// 2. Cluster sentences semantically (k-means-style cosine clustering).
/// 3. Pick the best representative from each cluster and concatenate up to
///    `max_words` of the resulting fusion summary.
///
/// # Example
///
/// ```rust
/// use scirs2_text::abstractive_summary::FusionSummarizer;
///
/// let docs = vec![
///     "Rust is a systems programming language. It focuses on safety.",
///     "Safety is the primary goal of Rust. Memory safety without a GC.",
/// ];
/// let summarizer = FusionSummarizer::new(3);
/// let summary = summarizer.summarize(&docs, 50).unwrap();
/// assert!(!summary.is_empty());
/// ```
pub struct FusionSummarizer {
    /// Desired number of clusters (= roughly number of output sentences).
    n_clusters: usize,
    /// Maximum PageRank-like iterations for cluster convergence.
    max_iter: usize,
    /// Minimum cosine similarity to assign a sentence to an existing cluster.
    cluster_threshold: f64,
}

impl FusionSummarizer {
    /// Create a new `FusionSummarizer` with `n_clusters` output sentences.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters: n_clusters.max(1),
            max_iter: 30,
            cluster_threshold: 0.1,
        }
    }

    /// Override maximum clustering iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Override the minimum cosine similarity threshold used during clustering.
    pub fn with_cluster_threshold(mut self, threshold: f64) -> Self {
        self.cluster_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Extract sentences from multiple documents, scoring each by TF-IDF.
    ///
    /// Returns a flat list of [`ScoredSentence`] items drawn from all documents.
    pub fn extract_sentences(&self, documents: &[&str]) -> Vec<ScoredSentence> {
        if documents.is_empty() {
            return Vec::new();
        }

        let sentence_tokenizer = SentenceTokenizer::new();
        let mut all_sentences: Vec<ScoredSentence> = Vec::new();
        let mut global_index = 0usize;

        // Collect all raw sentences first.
        let mut raw_per_doc: Vec<Vec<String>> = Vec::new();
        for doc in documents {
            let sents = sentence_tokenizer
                .tokenize(doc)
                .unwrap_or_else(|_| vec![doc.to_string()]);
            raw_per_doc.push(sents);
        }

        // Flatten for TF-IDF fitting across all documents.
        let flat: Vec<String> = raw_per_doc.iter().flatten().cloned().collect();
        if flat.is_empty() {
            return Vec::new();
        }

        // Build TF-IDF on the full corpus.
        let flat_refs: Vec<&str> = flat.iter().map(|s| s.as_str()).collect();
        let mut vectorizer = TfidfVectorizer::default();
        let tfidf = match vectorizer.fit_transform(&flat_refs) {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };

        let cols = tfidf.ncols();
        let n = flat.len();

        for (flat_idx, sentence) in flat.iter().enumerate() {
            // Score = mean TF-IDF of the row.
            let score = if cols > 0 {
                let row_sum: f64 = (0..cols).map(|c| tfidf[[flat_idx, c]]).sum();
                row_sum / cols as f64
            } else {
                0.0
            };

            // Determine which document this sentence came from.
            let mut doc_index = 0usize;
            let mut cumulative = 0usize;
            for (di, sents) in raw_per_doc.iter().enumerate() {
                if flat_idx < cumulative + sents.len() {
                    doc_index = di;
                    break;
                }
                cumulative += sents.len();
            }

            all_sentences.push(ScoredSentence {
                text: sentence.clone(),
                index: global_index,
                doc_index,
                score,
            });
            global_index += 1;
        }

        // Normalise scores to [0,1].
        let max_score = all_sentences
            .iter()
            .map(|s| s.score)
            .fold(0.0_f64, f64::max);
        if max_score > 0.0 {
            for s in &mut all_sentences {
                s.score /= max_score;
            }
        }

        all_sentences
    }

    /// Cluster sentences semantically using cosine similarity.
    ///
    /// Implements a greedy cosine-based k-means variant:
    /// 1. Initialise cluster centroids with the highest-scoring sentences.
    /// 2. Assign each sentence to its nearest centroid.
    /// 3. Recompute centroids and repeat up to `max_iter` times.
    ///
    /// Returns a `Vec` of clusters, each being a `Vec<ScoredSentence>`.
    pub fn cluster_sentences(
        &self,
        sentences: &[ScoredSentence],
        n_clusters: usize,
    ) -> Vec<Vec<ScoredSentence>> {
        let k = n_clusters.min(sentences.len()).max(1);
        if sentences.is_empty() {
            return Vec::new();
        }
        if sentences.len() <= k {
            // Each sentence is its own cluster.
            return sentences.iter().map(|s| vec![s.clone()]).collect();
        }

        // Build TF-IDF matrix from sentence texts.
        let texts: Vec<String> = sentences.iter().map(|s| s.text.clone()).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let mut vectorizer = TfidfVectorizer::default();
        let matrix = match vectorizer.fit_transform(&refs) {
            Ok(m) => m,
            Err(_) => {
                // Fallback: put everything in one cluster.
                return vec![sentences.to_vec()];
            }
        };

        let n = sentences.len();
        let cols = matrix.ncols();

        // Choose initial centroids: the k highest-scoring sentences.
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            sentences[b]
                .score
                .partial_cmp(&sentences[a].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let centroid_indices: Vec<usize> = sorted_indices.into_iter().take(k).collect();

        // Initialise centroid vectors (k x cols).
        let mut centroids: Vec<Array1<f64>> = centroid_indices
            .iter()
            .map(|&ci| matrix.row(ci).to_owned())
            .collect();

        let mut assignments: Vec<usize> = vec![0; n];

        for _iter in 0..self.max_iter {
            let mut changed = false;

            // Assignment step.
            for i in 0..n {
                let row = matrix.row(i).to_owned();
                let best_cluster = centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, centroid)| (ci, cosine_sim_vec(&row, centroid)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(ci, _)| ci)
                    .unwrap_or(0);

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step: recompute centroids as the mean of assigned sentences.
            for ci in 0..k {
                let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == ci).collect();
                if members.is_empty() {
                    // Keep previous centroid.
                    continue;
                }
                let mut new_centroid = Array1::zeros(cols);
                for &mi in &members {
                    new_centroid = new_centroid + matrix.row(mi).to_owned();
                }
                let count = members.len() as f64;
                new_centroid.mapv_inplace(|v| v / count);
                centroids[ci] = new_centroid;
            }
        }

        // Build output clusters.
        let mut clusters: Vec<Vec<ScoredSentence>> = vec![Vec::new(); k];
        for (i, &ci) in assignments.iter().enumerate() {
            clusters[ci].push(sentences[i].clone());
        }

        // Remove empty clusters that may arise from degenerate inputs.
        clusters.retain(|c| !c.is_empty());
        clusters
    }

    /// Generate a fused summary from clusters, limited to `max_words`.
    ///
    /// Picks the highest-scoring sentence from each cluster, then joins them
    /// in order of cluster appearance (preserving reading flow).
    pub fn generate_summary(&self, clusters: &[Vec<ScoredSentence>], max_words: usize) -> String {
        if clusters.is_empty() {
            return String::new();
        }

        // Pick best representative per cluster.
        let mut representatives: Vec<&ScoredSentence> = clusters
            .iter()
            .filter_map(|cluster| {
                cluster.iter().max_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .collect();

        // Sort representatives by their original index to preserve flow.
        representatives.sort_by_key(|s| s.index);

        // Concatenate up to max_words.
        let mut words_used = 0usize;
        let mut collected_words: Vec<&str> = Vec::new();

        'outer: for rep in representatives {
            let sentence_words: Vec<&str> = rep.text.split_whitespace().collect();
            for word in &sentence_words {
                if words_used >= max_words {
                    break 'outer;
                }
                collected_words.push(word);
                words_used += 1;
            }
        }

        collected_words.join(" ")
    }

    /// Convenience method: extract + cluster + generate in one call.
    pub fn summarize(&self, documents: &[&str], max_words: usize) -> Result<String> {
        if documents.is_empty() {
            return Ok(String::new());
        }
        let sentences = self.extract_sentences(documents);
        if sentences.is_empty() {
            return Ok(String::new());
        }
        let clusters = self.cluster_sentences(&sentences, self.n_clusters);
        Ok(self.generate_summary(&clusters, max_words))
    }
}

// ---------------------------------------------------------------------------
// CompressionSummarizer
// ---------------------------------------------------------------------------

/// Sentence compression by dropping low-importance tokens.
///
/// The importance of each token is computed via a TF-IDF-inspired heuristic
/// using term frequency within the sentence and inverse document frequency
/// estimated from a small built-in stop-word list.
///
/// # Example
///
/// ```rust
/// use scirs2_text::abstractive_summary::CompressionSummarizer;
///
/// let cs = CompressionSummarizer::new();
/// let compressed = cs.compress_sentence("The very quick brown fox jumped lazily", 0.6);
/// assert!(!compressed.is_empty());
/// ```
pub struct CompressionSummarizer {
    /// Stop words that always receive very low importance scores.
    stop_words: HashSet<String>,
}

impl CompressionSummarizer {
    /// Create a `CompressionSummarizer` with the built-in English stop-word list.
    pub fn new() -> Self {
        let raw = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "shall", "can", "and", "but", "or", "nor", "for", "yet", "so", "in", "on", "at", "to",
            "from", "by", "with", "of", "about", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "each", "all", "both", "very", "just", "too",
            "also", "then", "than", "that", "this", "these", "those", "i", "me", "my", "we", "our",
            "you", "your", "he", "she", "it", "its", "they", "them", "their", "what", "which",
            "who", "whom", "not", "no",
        ];
        Self {
            stop_words: raw.iter().map(|w| w.to_string()).collect(),
        }
    }

    /// Create a `CompressionSummarizer` with a custom stop-word list.
    pub fn with_stop_words(stop_words: HashSet<String>) -> Self {
        Self { stop_words }
    }

    /// Compute the importance score of a single `token` given its sentence context.
    ///
    /// Score components:
    /// - Term frequency within the sentence.
    /// - Stop-word penalty (×0.1 if the token is a stop word).
    /// - Length bonus: longer tokens receive a slight boost.
    /// - Capitalisation bonus: capitalised mid-sentence tokens receive a boost
    ///   (heuristic for proper nouns).
    pub fn importance_score(&self, token: &str, sentence_tokens: &[String]) -> f64 {
        if sentence_tokens.is_empty() {
            return 0.0;
        }
        let token_lower = token.to_lowercase();

        // Term frequency.
        let tf = sentence_tokens
            .iter()
            .filter(|t| t.to_lowercase() == token_lower)
            .count() as f64
            / sentence_tokens.len() as f64;

        // Stop-word penalty.
        let stop_penalty = if self.stop_words.contains(&token_lower) {
            0.1
        } else {
            1.0
        };

        // Length bonus (normalised to ~[0.5, 1.5]).
        let len_bonus = (1.0 + (token.len() as f64 / 10.0).min(1.0)) * 0.5;

        // Capitalisation bonus for mid-sentence proper-noun heuristic.
        let cap_bonus = if token
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
        {
            0.3
        } else {
            0.0
        };

        (tf * stop_penalty + len_bonus + cap_bonus).max(0.0)
    }

    /// Compress `sentence` by retaining only the fraction `ratio` of tokens
    /// with the highest importance scores.
    ///
    /// `ratio` is clamped to (0.0, 1.0]. A ratio of 1.0 keeps all tokens.
    /// Tokens are retained in their original order.
    ///
    /// Returns an empty string if the sentence has no words.
    pub fn compress_sentence(&self, sentence: &str, ratio: f64) -> String {
        let ratio = ratio.clamp(0.01, 1.0);

        // Preserve original whitespace-split tokens for output.
        let original_tokens: Vec<&str> = sentence.split_whitespace().collect();
        if original_tokens.is_empty() {
            return String::new();
        }

        // Normalised tokens for scoring.
        let norm_tokens: Vec<String> = original_tokens
            .iter()
            .map(|t| {
                t.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .collect();

        let n = original_tokens.len();
        let keep_count = ((n as f64 * ratio).ceil() as usize).clamp(1, n);

        // Score each token.
        let mut scored: Vec<(usize, f64)> = norm_tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (i, self.importance_score(t, &norm_tokens)))
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Collect the top-k indices (then re-sort by position).
        let mut keep_indices: Vec<usize> =
            scored.iter().take(keep_count).map(|&(i, _)| i).collect();
        keep_indices.sort_unstable();

        // Reassemble.
        keep_indices
            .iter()
            .map(|&i| original_tokens[i])
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for CompressionSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EnhancedCentroidSummarizer
// ---------------------------------------------------------------------------

/// Centroid-based summarizer with position bias and optional query focus.
///
/// Extends the basic centroid approach with:
/// - **Position bias**: earlier sentences receive a configurable bonus.
/// - **Query-focused mode**: sentences are re-ranked by their cosine similarity
///   to a query vector in addition to the document centroid.
///
/// # Example
///
/// ```rust
/// use scirs2_text::abstractive_summary::EnhancedCentroidSummarizer;
///
/// let summarizer = EnhancedCentroidSummarizer::new(2)
///     .with_position_bias(0.3)
///     .with_redundancy_threshold(0.85);
///
/// let text = "Natural language processing is a field of AI. \
///             It allows computers to understand human language. \
///             NLP is used in chatbots, translation, and search engines.";
/// let summary = summarizer.summarize(text).unwrap();
/// assert!(!summary.is_empty());
/// ```
pub struct EnhancedCentroidSummarizer {
    num_sentences: usize,
    topic_threshold: f64,
    redundancy_threshold: f64,
    /// Weight of position score relative to centroid score (0 = no position bias).
    position_bias: f64,
}

impl EnhancedCentroidSummarizer {
    /// Create a new `EnhancedCentroidSummarizer` extracting up to `num_sentences` sentences.
    pub fn new(num_sentences: usize) -> Self {
        Self {
            num_sentences: num_sentences.max(1),
            topic_threshold: 0.1,
            redundancy_threshold: 0.95,
            position_bias: 0.2,
        }
    }

    /// Set the position bias weight (0.0 = off, 1.0 = strong bias towards early sentences).
    pub fn with_position_bias(mut self, bias: f64) -> Self {
        self.position_bias = bias.clamp(0.0, 1.0);
        self
    }

    /// Set the TF-IDF topic threshold (terms below this weight are zeroed in the centroid).
    pub fn with_topic_threshold(mut self, threshold: f64) -> Self {
        self.topic_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the redundancy threshold: sentences more similar than this value are excluded.
    pub fn with_redundancy_threshold(mut self, threshold: f64) -> Self {
        self.redundancy_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Standard summarization (not query-focused).
    pub fn summarize(&self, text: &str) -> Result<String> {
        self.summarize_internal(text, None)
    }

    /// Query-focused summarization.
    ///
    /// Sentences are ranked by a linear combination of their similarity to the
    /// document centroid and their similarity to the query vector.
    ///
    /// # Arguments
    ///
    /// * `document` – the source document text.
    /// * `query` – a short query string describing the desired focus.
    /// * `max_sentences` – maximum number of sentences to return.
    pub fn summarize_query_focused(
        &self,
        document: &str,
        query: &str,
        max_sentences: usize,
    ) -> Result<String> {
        let override_self = EnhancedCentroidSummarizer {
            num_sentences: max_sentences.max(1),
            ..*self
        };
        override_self.summarize_internal(document, Some(query))
    }

    fn summarize_internal(&self, text: &str, query: Option<&str>) -> Result<String> {
        let sentence_tokenizer = SentenceTokenizer::new();
        let sentences: Vec<String> = sentence_tokenizer.tokenize(text)?;

        if sentences.is_empty() {
            return Ok(String::new());
        }
        if sentences.len() <= self.num_sentences {
            return Ok(text.to_string());
        }

        // Build TF-IDF vectors for all sentences.
        let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
        let mut vectorizer = TfidfVectorizer::default();
        let tfidf = vectorizer.fit_transform(&sentence_refs)?;

        // Compute document centroid.
        let centroid = self.compute_centroid(&tfidf);

        // Optionally compute query vector by transforming the query sentence.
        let query_vec: Option<Array1<f64>> = if let Some(q) = query {
            vectorizer.transform_batch(&[q]).ok().map(|m| {
                // The query may produce a 1-row matrix; take row 0.
                m.row(0).to_owned()
            })
        } else {
            None
        };

        // Score each sentence.
        let n = sentences.len();
        let mut scored: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let row = tfidf.row(i).to_owned();
                let centroid_sim = cosine_sim_vec(&row, &centroid);
                let query_sim = query_vec
                    .as_ref()
                    .map(|qv| cosine_sim_vec(&row, qv))
                    .unwrap_or(0.0);
                // Combine centroid similarity with query similarity (50/50 if query provided).
                let content_score = if query_vec.is_some() {
                    0.5 * centroid_sim + 0.5 * query_sim
                } else {
                    centroid_sim
                };
                // Position bonus: exponential decay from sentence 0.
                let pos_bonus = (-0.5 * i as f64 / n as f64).exp() * self.position_bias;
                (i, content_score + pos_bonus)
            })
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy selection avoiding redundancy.
        let mut selected: Vec<usize> = Vec::new();
        for (idx, _score) in &scored {
            if selected.len() >= self.num_sentences {
                break;
            }
            let redundant = selected
                .iter()
                .any(|&si| cosine_sim_rows(&tfidf, *idx, si) > self.redundancy_threshold);
            if !redundant {
                selected.push(*idx);
            }
        }

        // Restore original order.
        selected.sort_unstable();

        let summary = selected
            .iter()
            .map(|&i| sentences[i].as_str())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(summary)
    }

    /// Compute the document centroid as the mean TF-IDF vector, zeroing terms
    /// below `topic_threshold`.
    fn compute_centroid(&self, tfidf: &Array2<f64>) -> Array1<f64> {
        let mean = tfidf
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .unwrap_or_else(|| Array1::zeros(tfidf.ncols()));

        mean.mapv(|v| if v > self.topic_threshold { v } else { 0.0 })
    }
}

// ---------------------------------------------------------------------------
// ROUGE evaluation metrics
// ---------------------------------------------------------------------------

/// Compute ROUGE-N recall between a `hypothesis` and a `reference` string.
///
/// ROUGE-N is the fraction of reference n-grams that appear in the hypothesis.
///
/// # Arguments
///
/// * `hypothesis` – generated summary text.
/// * `reference` – gold-standard reference text.
/// * `n` – n-gram order (1 = unigrams, 2 = bigrams, …).
///
/// # Returns
///
/// A recall value in [0.0, 1.0]. Returns `0.0` when the reference contains no
/// n-grams (e.g. `n` larger than the reference length).
///
/// # Example
///
/// ```rust
/// use scirs2_text::abstractive_summary::rouge_n;
///
/// let recall = rouge_n("the cat sat on the mat", "the cat sat", 1);
/// assert!(recall > 0.5);
/// ```
pub fn rouge_n(hypothesis: &str, reference: &str, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let hyp_tokens = word_tokens(hypothesis);
    let ref_tokens = word_tokens(reference);

    if ref_tokens.len() < n {
        return 0.0;
    }

    // Build n-gram counts for reference.
    let ref_ngrams = count_ngrams(&ref_tokens, n);
    if ref_ngrams.is_empty() {
        return 0.0;
    }
    let ref_total: usize = ref_ngrams.values().sum();

    // Build n-gram counts for hypothesis.
    let hyp_ngrams = count_ngrams(&hyp_tokens, n);

    // Clipped overlap count.
    let overlap: usize = ref_ngrams
        .iter()
        .map(|(gram, &ref_count)| {
            let hyp_count = hyp_ngrams.get(gram).copied().unwrap_or(0);
            hyp_count.min(ref_count)
        })
        .sum();

    overlap as f64 / ref_total as f64
}

/// Build a frequency map of n-grams from `tokens`.
fn count_ngrams(tokens: &[String], n: usize) -> HashMap<Vec<String>, usize> {
    let mut map: HashMap<Vec<String>, usize> = HashMap::new();
    if tokens.len() < n {
        return map;
    }
    for i in 0..=(tokens.len() - n) {
        let gram: Vec<String> = tokens[i..i + n].to_vec();
        *map.entry(gram).or_insert(0) += 1;
    }
    map
}

/// Compute ROUGE-L recall based on the Longest Common Subsequence (LCS).
///
/// ROUGE-L is defined as `LCS(hypothesis, reference) / |reference|` in terms of
/// token counts.
///
/// # Arguments
///
/// * `hypothesis` – generated summary text.
/// * `reference` – gold-standard reference text.
///
/// # Returns
///
/// A recall value in [0.0, 1.0]. Returns `0.0` for empty inputs.
///
/// # Example
///
/// ```rust
/// use scirs2_text::abstractive_summary::rouge_l;
///
/// let score = rouge_l("the cat sat", "the cat sat on the mat");
/// assert!(score > 0.4);
/// ```
pub fn rouge_l(hypothesis: &str, reference: &str) -> f64 {
    let hyp_tokens = word_tokens(hypothesis);
    let ref_tokens = word_tokens(reference);

    if ref_tokens.is_empty() {
        return 0.0;
    }

    let lcs_len = lcs_length(&hyp_tokens, &ref_tokens);
    lcs_len as f64 / ref_tokens.len() as f64
}

/// Compute the length of the Longest Common Subsequence between two token sequences.
///
/// Uses the classic O(m×n) dynamic programming algorithm.
fn lcs_length(a: &[String], b: &[String]) -> usize {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return 0;
    }

    // Rolling two-row DP to save memory.
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1] + 1
            } else {
                prev[j].max(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.iter_mut().for_each(|v| *v = 0);
    }

    prev[n]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MULTI_DOC_A: &str =
        "Rust is a systems programming language. It focuses on safety and performance.";
    const MULTI_DOC_B: &str =
        "Memory safety without a garbage collector is a key goal of Rust. The language also \
         emphasises zero-cost abstractions.";
    const LONG_TEXT: &str = "Natural language processing is a field of artificial intelligence. \
        It allows computers to understand and generate human language. \
        Applications include machine translation, chatbots, and sentiment analysis. \
        Deep learning has greatly advanced NLP in recent years. \
        Transformer models such as BERT and GPT are state-of-the-art.";

    // -- FusionSummarizer --

    #[test]
    fn test_fusion_extract_sentences_nonempty() {
        let docs = vec![MULTI_DOC_A, MULTI_DOC_B];
        let fs = FusionSummarizer::new(2);
        let sents = fs.extract_sentences(&docs);
        assert!(!sents.is_empty());
        // Every score should be in [0, 1].
        for s in &sents {
            assert!(
                (0.0..=1.001).contains(&s.score),
                "score out of range: {}",
                s.score
            );
        }
    }

    #[test]
    fn test_fusion_extract_empty_docs() {
        let fs = FusionSummarizer::new(2);
        let sents = fs.extract_sentences(&[]);
        assert!(sents.is_empty());
    }

    #[test]
    fn test_fusion_cluster_basic() {
        let docs = vec![MULTI_DOC_A, MULTI_DOC_B];
        let fs = FusionSummarizer::new(2);
        let sents = fs.extract_sentences(&docs);
        let clusters = fs.cluster_sentences(&sents, 2);
        assert!(!clusters.is_empty());
        // All sentences accounted for.
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, sents.len());
    }

    #[test]
    fn test_fusion_generate_summary_respects_max_words() {
        let docs = vec![MULTI_DOC_A, MULTI_DOC_B];
        let fs = FusionSummarizer::new(2);
        let sents = fs.extract_sentences(&docs);
        let clusters = fs.cluster_sentences(&sents, 2);
        let summary = fs.generate_summary(&clusters, 10);
        let words: usize = summary.split_whitespace().count();
        assert!(words <= 10, "Expected ≤10 words, got {}", words);
    }

    #[test]
    fn test_fusion_summarize_end_to_end() {
        let docs = vec![MULTI_DOC_A, MULTI_DOC_B];
        let fs = FusionSummarizer::new(2);
        let summary = fs.summarize(&docs, 60).expect("summarize should succeed");
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_fusion_single_document() {
        let docs = vec![LONG_TEXT];
        let fs = FusionSummarizer::new(2);
        let summary = fs.summarize(&docs, 80).expect("should succeed");
        assert!(!summary.is_empty());
    }

    // -- CompressionSummarizer --

    #[test]
    fn test_compression_basic() {
        let cs = CompressionSummarizer::new();
        let sentence = "The very quick brown fox jumped lazily over the fence";
        let compressed = cs.compress_sentence(sentence, 0.5);
        let orig_words: usize = sentence.split_whitespace().count();
        let comp_words: usize = compressed.split_whitespace().count();
        assert!(comp_words <= orig_words);
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_compression_ratio_one_keeps_all() {
        let cs = CompressionSummarizer::new();
        let sentence = "Hello world this is a test sentence";
        let compressed = cs.compress_sentence(sentence, 1.0);
        let orig_words = sentence.split_whitespace().count();
        let comp_words = compressed.split_whitespace().count();
        assert_eq!(comp_words, orig_words);
    }

    #[test]
    fn test_compression_empty_sentence() {
        let cs = CompressionSummarizer::new();
        let result = cs.compress_sentence("", 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compression_importance_stop_word_lower() {
        let cs = CompressionSummarizer::new();
        let tokens: Vec<String> = vec!["the".to_string(), "quick".to_string(), "fox".to_string()];
        let stop_score = cs.importance_score("the", &tokens);
        let content_score = cs.importance_score("fox", &tokens);
        assert!(
            content_score > stop_score,
            "Content word should score higher than stop word"
        );
    }

    // -- EnhancedCentroidSummarizer --

    #[test]
    fn test_enhanced_centroid_basic() {
        let s = EnhancedCentroidSummarizer::new(2);
        let summary = s.summarize(LONG_TEXT).expect("should succeed");
        assert!(!summary.is_empty());
        assert!(summary.len() < LONG_TEXT.len());
    }

    #[test]
    fn test_enhanced_centroid_short_text() {
        let s = EnhancedCentroidSummarizer::new(5);
        let text = "One sentence only.";
        let summary = s.summarize(text).expect("should succeed");
        assert_eq!(summary, text);
    }

    #[test]
    fn test_enhanced_centroid_empty() {
        let s = EnhancedCentroidSummarizer::new(2);
        let summary = s.summarize("").expect("should succeed");
        assert!(summary.is_empty());
    }

    #[test]
    fn test_enhanced_centroid_query_focused() {
        let s = EnhancedCentroidSummarizer::new(2);
        let summary = s
            .summarize_query_focused(LONG_TEXT, "transformer models BERT GPT", 2)
            .expect("should succeed");
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_enhanced_centroid_query_focused_max_sentences() {
        let s = EnhancedCentroidSummarizer::new(2);
        let summary = s
            .summarize_query_focused(LONG_TEXT, "deep learning", 1)
            .expect("should succeed");
        // Should return at most 1 sentence.
        let sent_tok = SentenceTokenizer::new();
        let sents = sent_tok.tokenize(&summary).expect("ok");
        assert!(sents.len() <= 1);
    }

    // -- ROUGE-N --

    #[test]
    fn test_rouge1_perfect_match() {
        let recall = rouge_n("the cat sat", "the cat sat", 1);
        assert!((recall - 1.0).abs() < 1e-9, "Expected 1.0, got {recall}");
    }

    #[test]
    fn test_rouge1_partial_overlap() {
        let recall = rouge_n("cat sat", "the cat sat on the mat", 1);
        // 2 out of 6 reference unigrams matched: 2/6 ≈ 0.333
        assert!((recall - 2.0 / 6.0).abs() < 1e-9, "Got {recall}");
    }

    #[test]
    fn test_rouge2_basic() {
        let recall = rouge_n("the cat sat on the mat", "the cat sat on the mat", 2);
        assert!((recall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_n_zero_n() {
        assert_eq!(rouge_n("anything", "reference", 0), 0.0);
    }

    #[test]
    fn test_rouge_n_empty_reference() {
        assert_eq!(rouge_n("hypothesis", "", 1), 0.0);
    }

    #[test]
    fn test_rouge_n_empty_hypothesis() {
        // No n-grams match → recall = 0.
        assert_eq!(rouge_n("", "the cat sat", 1), 0.0);
    }

    // -- ROUGE-L --

    #[test]
    fn test_rouge_l_perfect_match() {
        let score = rouge_l("the cat sat", "the cat sat");
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_l_partial() {
        // LCS("cat sat", "the cat sat on the mat") = "cat sat" length 2 → 2/6
        let score = rouge_l("cat sat", "the cat sat on the mat");
        assert!((score - 2.0 / 6.0).abs() < 1e-9, "Got {score}");
    }

    #[test]
    fn test_rouge_l_empty_reference() {
        assert_eq!(rouge_l("hypothesis", ""), 0.0);
    }

    #[test]
    fn test_rouge_l_empty_hypothesis() {
        assert_eq!(rouge_l("", "reference text"), 0.0);
    }

    #[test]
    fn test_lcs_symmetric() {
        let a = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let b = vec!["b".to_string(), "c".to_string(), "d".to_string()];
        let lcs_ab = lcs_length(&a, &b);
        let lcs_ba = lcs_length(&b, &a);
        assert_eq!(lcs_ab, lcs_ba);
    }
}
