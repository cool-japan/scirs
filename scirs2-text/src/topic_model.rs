//! # Advanced Topic Modeling
//!
//! This module provides advanced topic modeling algorithms distinct from the
//! variational LDA in [`topic_modeling`](crate::topic_modeling):
//!
//! - **LDA via collapsed Gibbs sampling**: The standard Bayesian approach
//! - **Non-negative Matrix Factorization (NMF)**: A linear-algebraic topic model
//! - **Topic coherence scoring**: C_v and UMass metrics
//! - **Topic-document and topic-word distributions**
//! - **Automatic topic number selection via coherence**
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::topic_model::{GibbsLda, GibbsLdaConfig, NmfTopicModel};
//!
//! let docs = vec![
//!     vec!["machine", "learning", "algorithm", "data"],
//!     vec!["deep", "learning", "neural", "network"],
//!     vec!["natural", "language", "processing", "text"],
//!     vec!["cat", "dog", "pet", "animal"],
//!     vec!["pet", "care", "food", "animal"],
//! ];
//!
//! // LDA via Gibbs sampling
//! let config = GibbsLdaConfig {
//!     n_topics: 2,
//!     alpha: 0.1,
//!     beta: 0.01,
//!     n_iterations: 100,
//!     seed: Some(42),
//!     ..Default::default()
//! };
//!
//! let docs_str: Vec<Vec<&str>> = docs.iter().map(|d| d.iter().map(|s| *s).collect()).collect();
//! let mut lda = GibbsLda::new(config);
//! lda.fit(&docs_str).unwrap();
//!
//! let topics = lda.top_words(5);
//! assert_eq!(topics.len(), 2);
//! ```

use crate::error::{Result, TextError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, Rng, RngExt, SeedableRng};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Gibbs LDA
// ---------------------------------------------------------------------------

/// Configuration for collapsed Gibbs sampling LDA.
#[derive(Debug, Clone)]
pub struct GibbsLdaConfig {
    /// Number of topics.
    pub n_topics: usize,
    /// Dirichlet prior on document-topic distribution (alpha).
    pub alpha: f64,
    /// Dirichlet prior on topic-word distribution (beta).
    pub beta: f64,
    /// Number of Gibbs sampling iterations.
    pub n_iterations: usize,
    /// Burn-in iterations to discard.
    pub burn_in: usize,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for GibbsLdaConfig {
    fn default() -> Self {
        Self {
            n_topics: 10,
            alpha: 0.1,
            beta: 0.01,
            n_iterations: 500,
            burn_in: 50,
            seed: None,
        }
    }
}

/// LDA model using collapsed Gibbs sampling (Griffiths & Steyvers, 2004).
///
/// In collapsed Gibbs sampling, we integrate out the topic-word and
/// document-topic distributions and only sample the topic assignments.
#[derive(Debug)]
pub struct GibbsLda {
    config: GibbsLdaConfig,
    /// Vocabulary: word -> index
    vocab: HashMap<String, usize>,
    /// Reverse vocabulary: index -> word
    rev_vocab: Vec<String>,
    /// Topic assignment for each word occurrence: topic_assignments[doc][word_pos]
    topic_assignments: Vec<Vec<usize>>,
    /// Count of topic k in document d: n_dk[d][k]
    n_dk: Vec<Vec<usize>>,
    /// Count of word w assigned to topic k: n_kw[k][w]
    n_kw: Vec<Vec<usize>>,
    /// Total words assigned to topic k: n_k[k]
    n_k: Vec<usize>,
    /// Number of words in each document
    doc_lengths: Vec<usize>,
    /// Document word indices: docs[d][pos] = word_index
    doc_words: Vec<Vec<usize>>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl GibbsLda {
    /// Create a new Gibbs LDA model.
    pub fn new(config: GibbsLdaConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
            rev_vocab: Vec::new(),
            topic_assignments: Vec::new(),
            n_dk: Vec::new(),
            n_kw: Vec::new(),
            n_k: Vec::new(),
            doc_lengths: Vec::new(),
            doc_words: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the LDA model on a tokenized corpus.
    pub fn fit(&mut self, documents: &[Vec<&str>]) -> Result<()> {
        if documents.is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot fit LDA on empty corpus".to_string(),
            ));
        }

        let n_topics = self.config.n_topics;
        if n_topics == 0 {
            return Err(TextError::InvalidInput(
                "Number of topics must be > 0".to_string(),
            ));
        }

        // Build vocabulary
        self.vocab.clear();
        self.rev_vocab.clear();
        let mut word_set: HashSet<String> = HashSet::new();
        for doc in documents {
            for &word in doc {
                word_set.insert(word.to_string());
            }
        }
        let mut sorted_words: Vec<String> = word_set.into_iter().collect();
        sorted_words.sort();
        for (idx, word) in sorted_words.iter().enumerate() {
            self.vocab.insert(word.clone(), idx);
        }
        self.rev_vocab = sorted_words;
        let n_vocab = self.rev_vocab.len();

        if n_vocab == 0 {
            return Err(TextError::InvalidInput(
                "Empty vocabulary after tokenization".to_string(),
            ));
        }

        let n_docs = documents.len();

        // Convert documents to word indices
        self.doc_words = documents
            .iter()
            .map(|doc| {
                doc.iter()
                    .filter_map(|w| self.vocab.get(*w).copied())
                    .collect()
            })
            .collect();

        self.doc_lengths = self.doc_words.iter().map(|d| d.len()).collect();

        // Initialize counts
        self.n_dk = vec![vec![0; n_topics]; n_docs];
        self.n_kw = vec![vec![0; n_vocab]; n_topics];
        self.n_k = vec![0; n_topics];
        self.topic_assignments = Vec::with_capacity(n_docs);

        // Random initialization of topic assignments
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42),
        };

        for d in 0..n_docs {
            let mut doc_topics = Vec::with_capacity(self.doc_words[d].len());
            for &w in &self.doc_words[d] {
                let k = (rng.random::<f64>() * n_topics as f64) as usize % n_topics;
                doc_topics.push(k);
                self.n_dk[d][k] += 1;
                self.n_kw[k][w] += 1;
                self.n_k[k] += 1;
            }
            self.topic_assignments.push(doc_topics);
        }

        // Gibbs sampling iterations
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let beta_sum = beta * n_vocab as f64;

        for _iter in 0..self.config.n_iterations {
            for d in 0..n_docs {
                let n_words_d = self.doc_words[d].len();
                for i in 0..n_words_d {
                    let w = self.doc_words[d][i];
                    let old_k = self.topic_assignments[d][i];

                    // Decrement counts
                    self.n_dk[d][old_k] -= 1;
                    self.n_kw[old_k][w] -= 1;
                    self.n_k[old_k] -= 1;

                    // Compute conditional distribution p(z_i = k | ...)
                    let mut probs = vec![0.0f64; n_topics];
                    for k in 0..n_topics {
                        probs[k] = (self.n_dk[d][k] as f64 + alpha)
                            * (self.n_kw[k][w] as f64 + beta)
                            / (self.n_k[k] as f64 + beta_sum);
                    }

                    // Sample new topic
                    let total: f64 = probs.iter().sum();
                    if total < 1e-15 {
                        // Fallback: uniform
                        let new_k = (rng.random::<f64>() * n_topics as f64) as usize % n_topics;
                        self.topic_assignments[d][i] = new_k;
                        self.n_dk[d][new_k] += 1;
                        self.n_kw[new_k][w] += 1;
                        self.n_k[new_k] += 1;
                        continue;
                    }

                    let threshold = rng.random::<f64>() * total;
                    let mut cumsum = 0.0;
                    let mut new_k = n_topics - 1;
                    for k in 0..n_topics {
                        cumsum += probs[k];
                        if cumsum >= threshold {
                            new_k = k;
                            break;
                        }
                    }

                    // Update counts
                    self.topic_assignments[d][i] = new_k;
                    self.n_dk[d][new_k] += 1;
                    self.n_kw[new_k][w] += 1;
                    self.n_k[new_k] += 1;
                }
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Get the topic-word distribution for topic k: P(w | k).
    pub fn topic_word_distribution(&self, topic: usize) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(TextError::ModelNotFitted("LDA not fitted".to_string()));
        }
        if topic >= self.config.n_topics {
            return Err(TextError::InvalidInput(format!(
                "Topic {} out of range ({})",
                topic, self.config.n_topics
            )));
        }

        let n_vocab = self.rev_vocab.len();
        let beta = self.config.beta;
        let beta_sum = beta * n_vocab as f64;
        let total = self.n_k[topic] as f64 + beta_sum;

        let mut dist = Array1::<f64>::zeros(n_vocab);
        for w in 0..n_vocab {
            dist[w] = (self.n_kw[topic][w] as f64 + beta) / total;
        }
        Ok(dist)
    }

    /// Get the document-topic distribution for document d: P(k | d).
    pub fn doc_topic_distribution(&self, doc: usize) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(TextError::ModelNotFitted("LDA not fitted".to_string()));
        }
        if doc >= self.n_dk.len() {
            return Err(TextError::InvalidInput(format!(
                "Document {} out of range ({})",
                doc,
                self.n_dk.len()
            )));
        }

        let n_topics = self.config.n_topics;
        let alpha = self.config.alpha;
        let total = self.doc_lengths[doc] as f64 + alpha * n_topics as f64;

        let mut dist = Array1::<f64>::zeros(n_topics);
        if total < 1e-15 {
            // Empty document: uniform
            let uniform = 1.0 / n_topics as f64;
            for k in 0..n_topics {
                dist[k] = uniform;
            }
        } else {
            for k in 0..n_topics {
                dist[k] = (self.n_dk[doc][k] as f64 + alpha) / total;
            }
        }
        Ok(dist)
    }

    /// Get the full document-topic matrix.
    pub fn doc_topic_matrix(&self) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(TextError::ModelNotFitted("LDA not fitted".to_string()));
        }

        let n_docs = self.n_dk.len();
        let n_topics = self.config.n_topics;
        let mut matrix = Array2::<f64>::zeros((n_docs, n_topics));
        for d in 0..n_docs {
            let dist = self.doc_topic_distribution(d)?;
            for k in 0..n_topics {
                matrix[[d, k]] = dist[k];
            }
        }
        Ok(matrix)
    }

    /// Get top words for each topic.
    pub fn top_words(&self, n_words: usize) -> Vec<Vec<(String, f64)>> {
        let n_topics = self.config.n_topics;
        let mut result = Vec::with_capacity(n_topics);

        for k in 0..n_topics {
            let dist = match self.topic_word_distribution(k) {
                Ok(d) => d,
                Err(_) => {
                    result.push(Vec::new());
                    continue;
                }
            };

            let mut word_probs: Vec<(usize, f64)> =
                dist.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            word_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top: Vec<(String, f64)> = word_probs
                .iter()
                .take(n_words.min(self.rev_vocab.len()))
                .map(|(idx, prob)| (self.rev_vocab[*idx].clone(), *prob))
                .collect();

            result.push(top);
        }
        result
    }

    /// Get the vocabulary.
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocab
    }

    /// Get the number of topics.
    pub fn n_topics(&self) -> usize {
        self.config.n_topics
    }

    /// Check if the model is fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// ---------------------------------------------------------------------------
// Non-negative Matrix Factorization (NMF) Topic Model
// ---------------------------------------------------------------------------

/// Configuration for NMF topic modeling.
#[derive(Debug, Clone)]
pub struct NmfConfig {
    /// Number of topics (components).
    pub n_topics: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for NmfConfig {
    fn default() -> Self {
        Self {
            n_topics: 10,
            max_iter: 200,
            tolerance: 1e-4,
            seed: 42,
        }
    }
}

/// NMF-based topic model.
///
/// Factorizes a term-document matrix V ~ W * H where:
/// - W (n_docs x n_topics): document-topic weights
/// - H (n_topics x n_terms): topic-term weights
///
/// Uses multiplicative update rules (Lee & Seung, 2001).
#[derive(Debug)]
pub struct NmfTopicModel {
    config: NmfConfig,
    /// Document-topic matrix W.
    w: Option<Array2<f64>>,
    /// Topic-term matrix H.
    h: Option<Array2<f64>>,
    /// Vocabulary.
    vocab: Vec<String>,
    /// Reconstruction error history.
    error_history: Vec<f64>,
    /// Whether fitted.
    fitted: bool,
}

impl NmfTopicModel {
    /// Create a new NMF topic model.
    pub fn new(config: NmfConfig) -> Self {
        Self {
            config,
            w: None,
            h: None,
            vocab: Vec::new(),
            error_history: Vec::new(),
            fitted: false,
        }
    }

    /// Fit NMF on a non-negative document-term matrix.
    ///
    /// `matrix` is (n_docs, n_terms), `vocabulary` maps index -> word.
    pub fn fit(&mut self, matrix: &Array2<f64>, vocabulary: &[String]) -> Result<()> {
        let (n_docs, n_terms) = matrix.dim();
        let n_topics = self.config.n_topics;

        if n_docs == 0 || n_terms == 0 {
            return Err(TextError::InvalidInput(
                "Cannot fit NMF on empty matrix".to_string(),
            ));
        }
        if n_topics > n_docs || n_topics > n_terms {
            return Err(TextError::InvalidInput(format!(
                "n_topics ({}) must not exceed matrix dimensions ({}, {})",
                n_topics, n_docs, n_terms
            )));
        }

        self.vocab = vocabulary.to_vec();

        // Initialize W and H with small random positive values
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut w = Array2::<f64>::zeros((n_docs, n_topics));
        let mut h = Array2::<f64>::zeros((n_topics, n_terms));

        let eps = 1e-10;
        for elem in w.iter_mut() {
            *elem = rng.random::<f64>() * 0.1 + eps;
        }
        for elem in h.iter_mut() {
            *elem = rng.random::<f64>() * 0.1 + eps;
        }

        self.error_history.clear();

        // Multiplicative update rules
        for _iter in 0..self.config.max_iter {
            // Update H: H <- H * (W^T * V) / (W^T * W * H)
            let wt_v = mat_mul_ata_b(&w, matrix);
            let wt_w = mat_mul_ata_b(&w, &w);
            let wt_w_h = mat_mul_ab(&wt_w, &h);

            for i in 0..n_topics {
                for j in 0..n_terms {
                    let denom = wt_w_h[[i, j]] + eps;
                    h[[i, j]] *= wt_v[[i, j]] / denom;
                    if h[[i, j]] < eps {
                        h[[i, j]] = eps;
                    }
                }
            }

            // Update W: W <- W * (V * H^T) / (W * H * H^T)
            let v_ht = mat_mul_abt(matrix, &h);
            let w_h = mat_mul_ab(&w, &h);
            let w_h_ht = mat_mul_abt(&w_h, &h);

            for i in 0..n_docs {
                for j in 0..n_topics {
                    let denom = w_h_ht[[i, j]] + eps;
                    w[[i, j]] *= v_ht[[i, j]] / denom;
                    if w[[i, j]] < eps {
                        w[[i, j]] = eps;
                    }
                }
            }

            // Compute reconstruction error
            let wh = mat_mul_ab(&w, &h);
            let mut error = 0.0;
            for i in 0..n_docs {
                for j in 0..n_terms {
                    let diff = matrix[[i, j]] - wh[[i, j]];
                    error += diff * diff;
                }
            }
            error = error.sqrt();
            self.error_history.push(error);

            // Check convergence
            if self.error_history.len() >= 2 {
                let prev = self.error_history[self.error_history.len() - 2];
                if (prev - error).abs() < self.config.tolerance {
                    break;
                }
            }
        }

        self.w = Some(w);
        self.h = Some(h);
        self.fitted = true;
        Ok(())
    }

    /// Get the document-topic matrix W.
    pub fn doc_topic_matrix(&self) -> Result<&Array2<f64>> {
        self.w
            .as_ref()
            .ok_or_else(|| TextError::ModelNotFitted("NMF not fitted".to_string()))
    }

    /// Get the topic-term matrix H.
    pub fn topic_term_matrix(&self) -> Result<&Array2<f64>> {
        self.h
            .as_ref()
            .ok_or_else(|| TextError::ModelNotFitted("NMF not fitted".to_string()))
    }

    /// Get top words for each topic.
    pub fn top_words(&self, n_words: usize) -> Result<Vec<Vec<(String, f64)>>> {
        let h = self.topic_term_matrix()?;
        let n_topics = h.nrows();
        let mut result = Vec::with_capacity(n_topics);

        for k in 0..n_topics {
            let row = h.row(k);
            let mut word_scores: Vec<(usize, f64)> =
                row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top: Vec<(String, f64)> = word_scores
                .iter()
                .take(n_words.min(self.vocab.len()))
                .filter_map(|(idx, score)| self.vocab.get(*idx).map(|w| (w.clone(), *score)))
                .collect();
            result.push(top);
        }
        Ok(result)
    }

    /// Get the reconstruction error history.
    pub fn error_history(&self) -> &[f64] {
        &self.error_history
    }

    /// Check if fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// ---------------------------------------------------------------------------
// Topic Coherence Scoring
// ---------------------------------------------------------------------------

/// Topic coherence calculator.
///
/// Supports C_v (NPMI-based) and UMass coherence metrics.
#[derive(Debug, Clone)]
pub struct TopicCoherenceScorer {
    /// Window size for co-occurrence.
    window_size: usize,
    /// Smoothing epsilon.
    epsilon: f64,
}

impl Default for TopicCoherenceScorer {
    fn default() -> Self {
        Self {
            window_size: 10,
            epsilon: 1e-12,
        }
    }
}

impl TopicCoherenceScorer {
    /// Create a new coherence scorer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set window size.
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Calculate C_v coherence (NPMI-based, Roder et al. 2015).
    ///
    /// Higher values indicate more coherent topics.
    pub fn cv_coherence(
        &self,
        topic_words: &[Vec<String>],
        documents: &[Vec<String>],
    ) -> Result<f64> {
        if topic_words.is_empty() || documents.is_empty() {
            return Err(TextError::InvalidInput(
                "Topic words and documents must not be empty".to_string(),
            ));
        }

        let n_docs = documents.len() as f64;

        // Compute document frequency and co-document frequency
        let doc_sets: Vec<HashSet<&String>> =
            documents.iter().map(|doc| doc.iter().collect()).collect();

        let mut topic_scores = Vec::with_capacity(topic_words.len());

        for words in topic_words {
            if words.len() < 2 {
                topic_scores.push(0.0);
                continue;
            }

            let mut npmi_sum = 0.0;
            let mut pair_count = 0;

            for i in 0..words.len() {
                for j in (i + 1)..words.len() {
                    let wi = &words[i];
                    let wj = &words[j];

                    let df_i = doc_sets.iter().filter(|s| s.contains(wi)).count() as f64;
                    let df_j = doc_sets.iter().filter(|s| s.contains(wj)).count() as f64;
                    let df_ij = doc_sets
                        .iter()
                        .filter(|s| s.contains(wi) && s.contains(wj))
                        .count() as f64;

                    let p_i = (df_i + self.epsilon) / n_docs;
                    let p_j = (df_j + self.epsilon) / n_docs;
                    let p_ij = (df_ij + self.epsilon) / n_docs;

                    // NPMI = (log(P(i,j) / (P(i) * P(j)))) / (-log(P(i,j)))
                    let pmi = (p_ij / (p_i * p_j)).ln();
                    let neg_log_p_ij = -(p_ij.ln());

                    let npmi = if neg_log_p_ij.abs() > self.epsilon {
                        pmi / neg_log_p_ij
                    } else {
                        0.0
                    };

                    npmi_sum += npmi;
                    pair_count += 1;
                }
            }

            let score = if pair_count > 0 {
                npmi_sum / pair_count as f64
            } else {
                0.0
            };
            topic_scores.push(score);
        }

        let avg = topic_scores.iter().sum::<f64>() / topic_scores.len() as f64;
        Ok(avg)
    }

    /// Calculate UMass coherence (Mimno et al. 2011).
    ///
    /// Uses document co-occurrence. Higher (less negative) values are better.
    pub fn umass_coherence(
        &self,
        topic_words: &[Vec<String>],
        documents: &[Vec<String>],
    ) -> Result<f64> {
        if topic_words.is_empty() || documents.is_empty() {
            return Err(TextError::InvalidInput(
                "Topic words and documents must not be empty".to_string(),
            ));
        }

        let doc_sets: Vec<HashSet<&String>> =
            documents.iter().map(|doc| doc.iter().collect()).collect();

        let mut topic_scores = Vec::with_capacity(topic_words.len());

        for words in topic_words {
            if words.len() < 2 {
                topic_scores.push(0.0);
                continue;
            }

            let mut score = 0.0;
            let mut pair_count = 0;

            for i in 1..words.len() {
                for j in 0..i {
                    let wi = &words[i];
                    let wj = &words[j];

                    let df_j = doc_sets.iter().filter(|s| s.contains(wj)).count() as f64;
                    let df_ij = doc_sets
                        .iter()
                        .filter(|s| s.contains(wi) && s.contains(wj))
                        .count() as f64;

                    // UMass: log((D(wi, wj) + epsilon) / D(wj))
                    score += ((df_ij + self.epsilon) / (df_j + self.epsilon)).ln();
                    pair_count += 1;
                }
            }

            let avg_score = if pair_count > 0 {
                score / pair_count as f64
            } else {
                0.0
            };
            topic_scores.push(avg_score);
        }

        let avg = topic_scores.iter().sum::<f64>() / topic_scores.len() as f64;
        Ok(avg)
    }
}

// ---------------------------------------------------------------------------
// Automatic Topic Number Selection
// ---------------------------------------------------------------------------

/// Select the optimal number of topics by maximizing coherence.
///
/// Fits LDA models with different numbers of topics and returns the one
/// with the highest coherence score.
///
/// # Arguments
///
/// * `documents` - Tokenized documents
/// * `min_topics` - Minimum number of topics to try
/// * `max_topics` - Maximum number of topics to try
/// * `n_iterations` - Gibbs sampling iterations per model
/// * `seed` - Random seed
///
/// # Returns
///
/// (optimal_n_topics, coherence_scores)
pub fn select_n_topics(
    documents: &[Vec<&str>],
    min_topics: usize,
    max_topics: usize,
    n_iterations: usize,
    seed: u64,
) -> Result<(usize, Vec<(usize, f64)>)> {
    if documents.is_empty() {
        return Err(TextError::InvalidInput(
            "Cannot select topics on empty corpus".to_string(),
        ));
    }
    if min_topics == 0 || min_topics > max_topics {
        return Err(TextError::InvalidInput(format!(
            "Invalid topic range: {} to {}",
            min_topics, max_topics
        )));
    }

    let scorer = TopicCoherenceScorer::new();

    // Convert documents to owned strings for coherence calculation
    let doc_strings: Vec<Vec<String>> = documents
        .iter()
        .map(|doc| doc.iter().map(|w| w.to_string()).collect())
        .collect();

    let mut scores: Vec<(usize, f64)> = Vec::new();
    let mut best_k = min_topics;
    let mut best_score = f64::NEG_INFINITY;

    for k in min_topics..=max_topics {
        let config = GibbsLdaConfig {
            n_topics: k,
            alpha: 50.0 / k as f64,
            beta: 0.01,
            n_iterations,
            burn_in: n_iterations / 5,
            seed: Some(seed),
        };

        let mut lda = GibbsLda::new(config);
        lda.fit(documents)?;

        let top_words = lda.top_words(10);
        let topic_word_strs: Vec<Vec<String>> = top_words
            .iter()
            .map(|tw| tw.iter().map(|(w, _)| w.clone()).collect())
            .collect();

        let coherence = scorer.cv_coherence(&topic_word_strs, &doc_strings)?;
        scores.push((k, coherence));

        if coherence > best_score {
            best_score = coherence;
            best_k = k;
        }
    }

    Ok((best_k, scores))
}

// ---------------------------------------------------------------------------
// Matrix helper functions (for NMF)
// ---------------------------------------------------------------------------

/// A * B
fn mat_mul_ab(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (ar, ac) = a.dim();
    let (_br, bc) = b.dim();
    let mut result = Array2::<f64>::zeros((ar, bc));
    for i in 0..ar {
        for k in 0..ac {
            let a_ik = a[[i, k]];
            if a_ik.abs() < 1e-15 {
                continue;
            }
            for j in 0..bc {
                result[[i, j]] += a_ik * b[[k, j]];
            }
        }
    }
    result
}

/// A^T * B
fn mat_mul_ata_b(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (ar, ac) = a.dim();
    let (_br, bc) = b.dim();
    let mut result = Array2::<f64>::zeros((ac, bc));
    for k in 0..ar {
        for i in 0..ac {
            let a_ki = a[[k, i]];
            if a_ki.abs() < 1e-15 {
                continue;
            }
            for j in 0..bc {
                result[[i, j]] += a_ki * b[[k, j]];
            }
        }
    }
    result
}

/// A * B^T
fn mat_mul_abt(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (ar, ac) = a.dim();
    let (br, _bc) = b.dim();
    let mut result = Array2::<f64>::zeros((ar, br));
    for i in 0..ar {
        for j in 0..br {
            let mut sum = 0.0;
            for k in 0..ac {
                sum += a[[i, k]] * b[[j, k]];
            }
            result[[i, j]] = sum;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_docs() -> Vec<Vec<&'static str>> {
        vec![
            vec!["machine", "learning", "algorithm", "data", "model"],
            vec!["deep", "learning", "neural", "network", "training"],
            vec!["natural", "language", "processing", "text", "word"],
            vec!["cat", "dog", "pet", "animal", "food"],
            vec!["pet", "care", "food", "animal", "home"],
            vec!["dog", "cat", "play", "park", "fun"],
        ]
    }

    #[test]
    fn test_gibbs_lda_fit() {
        let docs = sample_docs();
        let config = GibbsLdaConfig {
            n_topics: 2,
            n_iterations: 50,
            seed: Some(42),
            ..Default::default()
        };
        let mut lda = GibbsLda::new(config);
        lda.fit(&docs).expect("fit failed");
        assert!(lda.is_fitted());
    }

    #[test]
    fn test_gibbs_lda_top_words() {
        let docs = sample_docs();
        let config = GibbsLdaConfig {
            n_topics: 2,
            n_iterations: 100,
            seed: Some(42),
            ..Default::default()
        };
        let mut lda = GibbsLda::new(config);
        lda.fit(&docs).expect("fit failed");

        let topics = lda.top_words(5);
        assert_eq!(topics.len(), 2);
        for topic in &topics {
            assert_eq!(topic.len(), 5);
            // Probabilities should sum to something reasonable
            let prob_sum: f64 = topic.iter().map(|(_, p)| p).sum();
            assert!(prob_sum > 0.0);
        }
    }

    #[test]
    fn test_gibbs_lda_doc_topic_distribution() {
        let docs = sample_docs();
        let config = GibbsLdaConfig {
            n_topics: 2,
            n_iterations: 50,
            seed: Some(42),
            ..Default::default()
        };
        let mut lda = GibbsLda::new(config);
        lda.fit(&docs).expect("fit failed");

        let dist = lda.doc_topic_distribution(0).expect("dist failed");
        assert_eq!(dist.len(), 2);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gibbs_lda_topic_word_distribution() {
        let docs = sample_docs();
        let config = GibbsLdaConfig {
            n_topics: 2,
            n_iterations: 50,
            seed: Some(42),
            ..Default::default()
        };
        let mut lda = GibbsLda::new(config);
        lda.fit(&docs).expect("fit failed");

        let dist = lda.topic_word_distribution(0).expect("dist failed");
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gibbs_lda_doc_topic_matrix() {
        let docs = sample_docs();
        let config = GibbsLdaConfig {
            n_topics: 3,
            n_iterations: 50,
            seed: Some(42),
            ..Default::default()
        };
        let mut lda = GibbsLda::new(config);
        lda.fit(&docs).expect("fit failed");

        let matrix = lda.doc_topic_matrix().expect("matrix failed");
        assert_eq!(matrix.dim(), (6, 3));

        // Each row should sum to ~1
        for i in 0..6 {
            let sum: f64 = matrix.row(i).iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gibbs_lda_empty_corpus() {
        let config = GibbsLdaConfig::default();
        let mut lda = GibbsLda::new(config);
        let result = lda.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gibbs_lda_not_fitted() {
        let lda = GibbsLda::new(GibbsLdaConfig::default());
        assert!(lda.doc_topic_distribution(0).is_err());
        assert!(lda.topic_word_distribution(0).is_err());
    }

    #[test]
    fn test_nmf_fit() {
        // Build a simple term-document matrix
        let matrix = Array2::from_shape_vec(
            (4, 5),
            vec![
                1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0,
                0.0, 2.0, 1.0, 2.0,
            ],
        )
        .expect("matrix creation failed");

        let vocab: Vec<String> = vec!["ml", "deep", "cat", "dog", "pet"]
            .into_iter()
            .map(String::from)
            .collect();

        let config = NmfConfig {
            n_topics: 2,
            max_iter: 100,
            ..Default::default()
        };

        let mut nmf = NmfTopicModel::new(config);
        nmf.fit(&matrix, &vocab).expect("nmf fit failed");
        assert!(nmf.is_fitted());
    }

    #[test]
    fn test_nmf_top_words() {
        let matrix = Array2::from_shape_vec(
            (4, 5),
            vec![
                1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0,
                0.0, 2.0, 1.0, 2.0,
            ],
        )
        .expect("matrix creation failed");

        let vocab: Vec<String> = vec!["ml", "deep", "cat", "dog", "pet"]
            .into_iter()
            .map(String::from)
            .collect();

        let config = NmfConfig {
            n_topics: 2,
            max_iter: 100,
            ..Default::default()
        };

        let mut nmf = NmfTopicModel::new(config);
        nmf.fit(&matrix, &vocab).expect("nmf fit failed");

        let topics = nmf.top_words(3).expect("top_words failed");
        assert_eq!(topics.len(), 2);
        for topic in &topics {
            assert!(topic.len() <= 3);
        }
    }

    #[test]
    fn test_nmf_convergence() {
        let matrix = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        )
        .expect("matrix creation failed");

        let vocab: Vec<String> = (0..4).map(|i| format!("w{}", i)).collect();

        let config = NmfConfig {
            n_topics: 2,
            max_iter: 200,
            ..Default::default()
        };

        let mut nmf = NmfTopicModel::new(config);
        nmf.fit(&matrix, &vocab).expect("nmf fit failed");

        let errors = nmf.error_history();
        assert!(!errors.is_empty());
        // Error should generally decrease
        if errors.len() >= 2 {
            assert!(
                errors.last().copied().unwrap_or(f64::MAX)
                    <= errors.first().copied().unwrap_or(0.0) + 1e-6
            );
        }
    }

    #[test]
    fn test_nmf_not_fitted() {
        let nmf = NmfTopicModel::new(NmfConfig::default());
        assert!(nmf.doc_topic_matrix().is_err());
        assert!(nmf.topic_term_matrix().is_err());
    }

    #[test]
    fn test_coherence_cv() {
        let topic_words = vec![
            vec![
                "machine".to_string(),
                "learning".to_string(),
                "algorithm".to_string(),
            ],
            vec!["cat".to_string(), "dog".to_string(), "pet".to_string()],
        ];

        let documents = vec![
            vec![
                "machine".to_string(),
                "learning".to_string(),
                "algorithm".to_string(),
            ],
            vec![
                "deep".to_string(),
                "learning".to_string(),
                "neural".to_string(),
            ],
            vec!["cat".to_string(), "dog".to_string(), "pet".to_string()],
            vec!["cat".to_string(), "play".to_string(), "fun".to_string()],
        ];

        let scorer = TopicCoherenceScorer::new();
        let cv = scorer
            .cv_coherence(&topic_words, &documents)
            .expect("cv failed");
        // Should return a finite value
        assert!(cv.is_finite());
    }

    #[test]
    fn test_coherence_umass() {
        let topic_words = vec![
            vec!["machine".to_string(), "learning".to_string()],
            vec!["cat".to_string(), "dog".to_string()],
        ];

        let documents = vec![
            vec!["machine".to_string(), "learning".to_string()],
            vec!["cat".to_string(), "dog".to_string()],
        ];

        let scorer = TopicCoherenceScorer::new();
        let umass = scorer
            .umass_coherence(&topic_words, &documents)
            .expect("umass failed");
        assert!(umass.is_finite());
    }

    #[test]
    fn test_coherence_empty() {
        let scorer = TopicCoherenceScorer::new();
        assert!(scorer.cv_coherence(&[], &[]).is_err());
        assert!(scorer.umass_coherence(&[], &[]).is_err());
    }

    #[test]
    fn test_select_n_topics() {
        let docs = sample_docs();
        let (best_k, scores) = select_n_topics(&docs, 2, 3, 30, 42).expect("select failed");
        assert!((2..=3).contains(&best_k));
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_select_n_topics_invalid_range() {
        let docs = sample_docs();
        assert!(select_n_topics(&docs, 5, 2, 30, 42).is_err());
    }

    #[test]
    fn test_lda_vocabulary() {
        let docs = sample_docs();
        let config = GibbsLdaConfig {
            n_topics: 2,
            n_iterations: 10,
            seed: Some(42),
            ..Default::default()
        };
        let mut lda = GibbsLda::new(config);
        lda.fit(&docs).expect("fit failed");

        let vocab = lda.vocabulary();
        assert!(vocab.contains_key("machine"));
        assert!(vocab.contains_key("cat"));
    }

    #[test]
    fn test_lda_n_topics() {
        let config = GibbsLdaConfig {
            n_topics: 5,
            ..Default::default()
        };
        let lda = GibbsLda::new(config);
        assert_eq!(lda.n_topics(), 5);
    }

    #[test]
    fn test_coherence_window_size() {
        let scorer = TopicCoherenceScorer::new().with_window_size(5);
        assert_eq!(scorer.window_size, 5);
    }

    #[test]
    fn test_nmf_doc_topic_matrix() {
        let matrix = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        )
        .expect("matrix creation failed");

        let vocab: Vec<String> = (0..4).map(|i| format!("w{}", i)).collect();
        let config = NmfConfig {
            n_topics: 2,
            max_iter: 50,
            ..Default::default()
        };

        let mut nmf = NmfTopicModel::new(config);
        nmf.fit(&matrix, &vocab).expect("fit failed");

        let dtm = nmf.doc_topic_matrix().expect("dtm failed");
        assert_eq!(dtm.dim(), (3, 2));

        // All values should be non-negative
        for &v in dtm.iter() {
            assert!(v >= 0.0);
        }
    }
}
