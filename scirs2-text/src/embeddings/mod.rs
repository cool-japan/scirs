//! # Word Embeddings Module
//!
//! This module provides comprehensive implementations for word embeddings, including
//! Word2Vec, GloVe, and FastText.
//!
//! ## Overview
//!
//! Word embeddings are dense vector representations of words that capture semantic
//! relationships. This module implements:
//!
//! - **Word2Vec Skip-gram**: Predicts context words from a target word
//! - **Word2Vec CBOW**: Predicts a target word from context words
//! - **GloVe**: Global Vectors for Word Representation
//! - **FastText**: Character n-gram based embeddings (handles OOV words)
//! - **Negative sampling**: Efficient training technique for large vocabularies
//! - **Hierarchical softmax**: Alternative to negative sampling for optimization
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_text::embeddings::{Word2Vec, Word2VecConfig, Word2VecAlgorithm};
//!
//! // Basic Word2Vec training
//! let documents = vec![
//!     "the quick brown fox jumps over the lazy dog",
//!     "the dog was lazy but the fox was quick",
//!     "brown fox and lazy dog are common phrases"
//! ];
//!
//! let config = Word2VecConfig {
//!     algorithm: Word2VecAlgorithm::SkipGram,
//!     vector_size: 100,
//!     window_size: 5,
//!     min_count: 1,
//!     learning_rate: 0.025,
//!     epochs: 5,
//!     negative_samples: 5,
//!     ..Default::default()
//! };
//!
//! let mut model = Word2Vec::with_config(config);
//! model.train(&documents).expect("Training failed");
//!
//! // Get word vector
//! if let Ok(vector) = model.get_word_vector("fox") {
//!     println!("Vector for 'fox': {:?}", vector);
//! }
//!
//! // Find similar words
//! let similar = model.most_similar("fox", 3).expect("Failed to find similar words");
//! for (word, similarity) in similar {
//!     println!("{}: {:.4}", word, similarity);
//! }
//! ```
//!
//! ## Advanced Usage
//!
//! ### Custom Configuration
//!
//! ```rust
//! use scirs2_text::embeddings::{Word2Vec, Word2VecConfig, Word2VecAlgorithm};
//!
//! let config = Word2VecConfig {
//!     algorithm: Word2VecAlgorithm::CBOW,
//!     vector_size: 300,        // Larger vectors for better quality
//!     window_size: 10,         // Larger context window
//!     min_count: 5,           // Filter rare words
//!     learning_rate: 0.01,    // Lower learning rate for stability
//!     epochs: 15,             // More training iterations
//!     negative_samples: 10,   // More negative samples
//!     subsample: 1e-4, // Subsample frequent words
//!     batch_size: 128,
//!     hierarchical_softmax: false,
//! };
//! ```
//!
//! ### Incremental Training
//!
//! ```rust
//! # use scirs2_text::embeddings::{Word2Vec, Word2VecConfig};
//! # let mut model = Word2Vec::new().with_min_count(1);
//! // Initial training
//! let batch1 = vec!["the quick brown fox jumps over the lazy dog"];
//! model.train(&batch1).expect("Training failed");
//!
//! // Continue training with new data
//! let batch2 = vec!["the dog was lazy but the fox was quick"];
//! model.train(&batch2).expect("Training failed");
//! ```
//!
//! ### Saving and Loading Models
//!
//! ```rust
//! # use scirs2_text::embeddings::{Word2Vec, Word2VecConfig};
//! # let mut model = Word2Vec::new().with_min_count(1);
//! # let texts = vec!["the quick brown fox jumps over the lazy dog"];
//! # model.train(&texts).expect("Training failed");
//! // Save trained model
//! model.save("my_model.w2v").expect("Failed to save model");
//!
//! // Load model
//! let loaded_model = Word2Vec::load("my_model.w2v")
//!     .expect("Failed to load model");
//! ```
//!
//! ## Performance Tips
//!
//! 1. **Vocabulary Size**: Use `min_count` to filter rare words and reduce memory usage
//! 2. **Vector Dimensions**: Balance between quality (higher dimensions) and speed (lower dimensions)
//! 3. **Training Algorithm**: Skip-gram works better with rare words, CBOW is faster
//! 4. **Negative Sampling**: Usually faster than hierarchical softmax for large vocabularies
//! 5. **Subsampling**: Set `subsample_threshold` to handle frequent words efficiently
//!
//! ## Mathematical Background
//!
//! ### Skip-gram Model
//!
//! The Skip-gram model maximizes the probability of context words given a target word:
//!
//! P(context|target) = ∏ P(w_context|w_target)
//!
//! ### CBOW Model
//!
//! The CBOW model predicts a target word from its context:
//!
//! P(target|context) = P(w_target|w_context1, w_context2, ...)
//!
//! ### Negative Sampling
//!
//! Instead of computing the full softmax, negative sampling approximates it by
//! sampling negative examples, making training much more efficient.

// Sub-modules
pub mod contrastive;
pub mod crosslingual;
pub mod fasttext;
pub mod glove;
pub mod sentence;
pub mod sentence_encoder;
pub mod universal;

// Re-export
pub use fasttext::{FastText, FastTextConfig};
pub use glove::{
    cosine_similarity as glove_cosine_similarity, CooccurrenceMatrix, GloVe, GloVeTrainer,
    GloVeTrainerConfig,
};

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vocabulary::Vocabulary;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

// ─── WordEmbedding trait ─────────────────────────────────────────────────────

/// Shared trait for word embedding models
///
/// Provides a common interface for querying word vectors, computing similarity,
/// and solving analogies across different embedding implementations
/// (Word2Vec, GloVe, FastText).
pub trait WordEmbedding {
    /// Get the embedding vector for a word
    fn embedding(&self, word: &str) -> Result<Array1<f64>>;

    /// Get the dimensionality of the embedding vectors
    fn dimension(&self) -> usize;

    /// Compute cosine similarity between two words
    fn similarity(&self, word1: &str, word2: &str) -> Result<f64> {
        let v1 = self.embedding(word1)?;
        let v2 = self.embedding(word2)?;
        Ok(embedding_cosine_similarity(&v1, &v2))
    }

    /// Find the top-N most similar words to the query word
    fn find_similar(&self, word: &str, top_n: usize) -> Result<Vec<(String, f64)>>;

    /// Solve the analogy: a is to b as c is to ?
    fn solve_analogy(&self, a: &str, b: &str, c: &str, top_n: usize) -> Result<Vec<(String, f64)>>;

    /// Get the number of words in the vocabulary
    fn vocab_size(&self) -> usize;
}

/// Calculate cosine similarity between two embedding vectors
pub fn embedding_cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute pairwise cosine similarity matrix for a list of words
pub fn pairwise_similarity(model: &dyn WordEmbedding, words: &[&str]) -> Result<Vec<Vec<f64>>> {
    let vectors: Vec<Array1<f64>> = words
        .iter()
        .map(|&w| model.embedding(w))
        .collect::<Result<Vec<_>>>()?;

    let n = vectors.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            let sim = embedding_cosine_similarity(&vectors[i], &vectors[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
    }

    Ok(matrix)
}

// ─── WordEmbedding implementations ──────────────────────────────────────────

impl WordEmbedding for GloVe {
    fn embedding(&self, word: &str) -> Result<Array1<f64>> {
        self.get_word_vector(word)
    }

    fn dimension(&self) -> usize {
        self.vector_size()
    }

    fn find_similar(&self, word: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        self.most_similar(word, top_n)
    }

    fn solve_analogy(&self, a: &str, b: &str, c: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        self.analogy(a, b, c, top_n)
    }

    fn vocab_size(&self) -> usize {
        self.vocabulary_size()
    }
}

impl WordEmbedding for FastText {
    fn embedding(&self, word: &str) -> Result<Array1<f64>> {
        self.get_word_vector(word)
    }

    fn dimension(&self) -> usize {
        self.vector_size()
    }

    fn find_similar(&self, word: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        self.most_similar(word, top_n)
    }

    fn solve_analogy(&self, a: &str, b: &str, c: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        self.analogy(a, b, c, top_n)
    }

    fn vocab_size(&self) -> usize {
        self.vocabulary_size()
    }
}

// ─── Huffman Tree for Hierarchical Softmax ──────────────────────────────────

/// A node in the Huffman tree used for hierarchical softmax
#[derive(Debug, Clone)]
struct HuffmanNode {
    /// Index in vocabulary (for leaf nodes) or internal node id
    id: usize,
    /// Frequency (for building the tree)
    frequency: usize,
    /// Left child index in the nodes array
    left: Option<usize>,
    /// Right child index in the nodes array
    right: Option<usize>,
    /// Is this a leaf node?
    is_leaf: bool,
}

/// Huffman tree for hierarchical softmax training
#[derive(Debug, Clone)]
struct HuffmanTree {
    /// Huffman codes for each vocabulary word: sequence of 0/1 decisions
    codes: Vec<Vec<u8>>,
    /// Path of internal node indices for each vocabulary word
    paths: Vec<Vec<usize>>,
    /// Number of internal nodes
    num_internal: usize,
}

impl HuffmanTree {
    /// Build a Huffman tree from word frequencies
    ///
    /// Returns codes and paths for each word in the vocabulary.
    fn build(frequencies: &[usize]) -> Result<Self> {
        let vocab_size = frequencies.len();
        if vocab_size == 0 {
            return Err(TextError::EmbeddingError(
                "Cannot build Huffman tree with empty vocabulary".into(),
            ));
        }
        if vocab_size == 1 {
            // Special case: single word
            return Ok(Self {
                codes: vec![vec![0]],
                paths: vec![vec![0]],
                num_internal: 1,
            });
        }

        // Create leaf nodes
        let mut nodes: Vec<HuffmanNode> = frequencies
            .iter()
            .enumerate()
            .map(|(id, &freq)| HuffmanNode {
                id,
                frequency: freq.max(1), // Avoid zero frequency
                left: None,
                right: None,
                is_leaf: true,
            })
            .collect();

        // Priority queue simulation using a sorted list
        // (index_in_nodes, frequency)
        let mut queue: Vec<(usize, usize)> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (i, n.frequency))
            .collect();
        queue.sort_by_key(|item| std::cmp::Reverse(item.1)); // Sort descending (pop from end = min)

        // Build the tree bottom-up
        while queue.len() > 1 {
            // Pop two smallest
            let (idx1, freq1) = queue
                .pop()
                .ok_or_else(|| TextError::EmbeddingError("Queue empty".into()))?;
            let (idx2, freq2) = queue
                .pop()
                .ok_or_else(|| TextError::EmbeddingError("Queue empty".into()))?;

            let new_id = nodes.len();
            let new_node = HuffmanNode {
                id: new_id,
                frequency: freq1 + freq2,
                left: Some(idx1),
                right: Some(idx2),
                is_leaf: false,
            };
            nodes.push(new_node);

            // Insert new node maintaining sorted order
            let new_freq = freq1 + freq2;
            let insert_pos = queue
                .binary_search_by(|(_, f)| new_freq.cmp(f))
                .unwrap_or_else(|pos| pos);
            queue.insert(insert_pos, (new_id, new_freq));
        }

        // Traverse tree to assign codes and paths
        let num_internal = nodes.len() - vocab_size;
        let mut codes = vec![Vec::new(); vocab_size];
        let mut paths = vec![Vec::new(); vocab_size];

        // DFS traversal
        let root_idx = nodes.len() - 1;
        let mut stack: Vec<(usize, Vec<u8>, Vec<usize>)> = vec![(root_idx, Vec::new(), Vec::new())];

        while let Some((node_idx, code, path)) = stack.pop() {
            let node = &nodes[node_idx];

            if node.is_leaf {
                codes[node.id] = code;
                paths[node.id] = path;
            } else {
                // Internal node index (0-based among internal nodes)
                let internal_idx = node.id - vocab_size;

                if let Some(left_idx) = node.left {
                    let mut left_code = code.clone();
                    left_code.push(0);
                    let mut left_path = path.clone();
                    left_path.push(internal_idx);
                    stack.push((left_idx, left_code, left_path));
                }

                if let Some(right_idx) = node.right {
                    let mut right_code = code.clone();
                    right_code.push(1);
                    let mut right_path = path.clone();
                    right_path.push(internal_idx);
                    stack.push((right_idx, right_code, right_path));
                }
            }
        }

        Ok(Self {
            codes,
            paths,
            num_internal,
        })
    }
}

/// A simplified weighted sampling table
#[derive(Debug, Clone)]
struct SamplingTable {
    /// The cumulative distribution function (CDF)
    cdf: Vec<f64>,
    /// The weights
    weights: Vec<f64>,
}

impl SamplingTable {
    /// Create a new sampling table from weights
    fn new(weights: &[f64]) -> Result<Self> {
        if weights.is_empty() {
            return Err(TextError::EmbeddingError("Weights cannot be empty".into()));
        }

        // Ensure all _weights are positive
        if weights.iter().any(|&w| w < 0.0) {
            return Err(TextError::EmbeddingError("Weights must be positive".into()));
        }

        // Compute the CDF
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            return Err(TextError::EmbeddingError(
                "Sum of _weights must be positive".into(),
            ));
        }

        let mut cdf = Vec::with_capacity(weights.len());
        let mut total = 0.0;

        for &w in weights {
            total += w;
            cdf.push(total / sum);
        }

        Ok(Self {
            cdf,
            weights: weights.to_vec(),
        })
    }

    /// Sample an index based on the weights
    fn sample<R: Rng>(&self, rng: &mut R) -> usize {
        let r = rng.random::<f64>();

        // Binary search for the insertion point
        match self.cdf.binary_search_by(|&cdf_val| {
            cdf_val.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => idx,
            Err(idx) => idx,
        }
    }

    /// Get the weights
    fn weights(&self) -> &[f64] {
        &self.weights
    }
}

/// Word2Vec training algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Word2VecAlgorithm {
    /// Continuous Bag of Words (CBOW) algorithm
    CBOW,
    /// Skip-gram algorithm
    SkipGram,
}

/// Word2Vec training options
#[derive(Debug, Clone)]
pub struct Word2VecConfig {
    /// Size of the word vectors
    pub vector_size: usize,
    /// Maximum distance between the current and predicted word within a sentence
    pub window_size: usize,
    /// Minimum count of words to consider for training
    pub min_count: usize,
    /// Number of iterations (epochs) over the corpus
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Skip-gram or CBOW algorithm
    pub algorithm: Word2VecAlgorithm,
    /// Number of negative samples per positive sample
    pub negative_samples: usize,
    /// Threshold for subsampling frequent words
    pub subsample: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Whether to use hierarchical softmax instead of negative sampling
    pub hierarchical_softmax: bool,
}

impl Default for Word2VecConfig {
    fn default() -> Self {
        Self {
            vector_size: 100,
            window_size: 5,
            min_count: 5,
            epochs: 5,
            learning_rate: 0.025,
            algorithm: Word2VecAlgorithm::SkipGram,
            negative_samples: 5,
            subsample: 1e-3,
            batch_size: 128,
            hierarchical_softmax: false,
        }
    }
}

/// Word2Vec model for training and using word embeddings
///
/// Word2Vec is an algorithm for learning vector representations of words,
/// also known as word embeddings. These vectors capture semantic meanings
/// of words, allowing operations like "king - man + woman" to result in
/// a vector close to "queen".
///
/// This implementation supports both Continuous Bag of Words (CBOW) and
/// Skip-gram models, with negative sampling for efficient training.
pub struct Word2Vec {
    /// Configuration options
    config: Word2VecConfig,
    /// Vocabulary
    vocabulary: Vocabulary,
    /// Input embeddings
    input_embeddings: Option<Array2<f64>>,
    /// Output embeddings
    output_embeddings: Option<Array2<f64>>,
    /// Tokenizer
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    /// Sampling table for negative sampling
    sampling_table: Option<SamplingTable>,
    /// Huffman tree for hierarchical softmax
    huffman_tree: Option<HuffmanTree>,
    /// Hierarchical softmax parameter vectors (one per internal node)
    hs_params: Option<Array2<f64>>,
    /// Current learning rate (gets updated during training)
    current_learning_rate: f64,
}

impl Debug for Word2Vec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Word2Vec")
            .field("config", &self.config)
            .field("vocabulary", &self.vocabulary)
            .field("input_embeddings", &self.input_embeddings)
            .field("output_embeddings", &self.output_embeddings)
            .field("sampling_table", &self.sampling_table)
            .field("huffman_tree", &self.huffman_tree)
            .field("current_learning_rate", &self.current_learning_rate)
            .finish()
    }
}

// Manual Clone implementation to handle the non-Clone tokenizer
impl Default for Word2Vec {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Word2Vec {
    fn clone(&self) -> Self {
        let tokenizer: Box<dyn Tokenizer + Send + Sync> = Box::new(WordTokenizer::default());

        Self {
            config: self.config.clone(),
            vocabulary: self.vocabulary.clone(),
            input_embeddings: self.input_embeddings.clone(),
            output_embeddings: self.output_embeddings.clone(),
            tokenizer,
            sampling_table: self.sampling_table.clone(),
            huffman_tree: self.huffman_tree.clone(),
            hs_params: self.hs_params.clone(),
            current_learning_rate: self.current_learning_rate,
        }
    }
}

impl Word2Vec {
    /// Create a new Word2Vec model with default configuration
    pub fn new() -> Self {
        Self {
            config: Word2VecConfig::default(),
            vocabulary: Vocabulary::new(),
            input_embeddings: None,
            output_embeddings: None,
            tokenizer: Box::new(WordTokenizer::default()),
            sampling_table: None,
            huffman_tree: None,
            hs_params: None,
            current_learning_rate: 0.025,
        }
    }

    /// Create a new Word2Vec model with the specified configuration
    pub fn with_config(config: Word2VecConfig) -> Self {
        let learning_rate = config.learning_rate;
        Self {
            config,
            vocabulary: Vocabulary::new(),
            input_embeddings: None,
            output_embeddings: None,
            tokenizer: Box::new(WordTokenizer::default()),
            sampling_table: None,
            huffman_tree: None,
            hs_params: None,
            current_learning_rate: learning_rate,
        }
    }

    /// Set a custom tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer + Send + Sync>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Set vector size
    pub fn with_vector_size(mut self, vectorsize: usize) -> Self {
        self.config.vector_size = vectorsize;
        self
    }

    /// Set window size
    pub fn with_window_size(mut self, windowsize: usize) -> Self {
        self.config.window_size = windowsize;
        self
    }

    /// Set minimum count
    pub fn with_min_count(mut self, mincount: usize) -> Self {
        self.config.min_count = mincount;
        self
    }

    /// Set number of epochs
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.config.epochs = epochs;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learningrate: f64) -> Self {
        self.config.learning_rate = learningrate;
        self.current_learning_rate = learningrate;
        self
    }

    /// Set algorithm (CBOW or Skip-gram)
    pub fn with_algorithm(mut self, algorithm: Word2VecAlgorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }

    /// Set number of negative samples
    pub fn with_negative_samples(mut self, negativesamples: usize) -> Self {
        self.config.negative_samples = negativesamples;
        self
    }

    /// Set subsampling threshold
    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.config.subsample = subsample;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batchsize: usize) -> Self {
        self.config.batch_size = batchsize;
        self
    }

    /// Build vocabulary from a corpus
    pub fn build_vocabulary(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for building vocabulary".into(),
            ));
        }

        // Count word frequencies
        let mut word_counts = HashMap::new();
        let mut _total_words = 0;

        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;
            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
                _total_words += 1;
            }
        }

        // Create vocabulary with words that meet minimum count
        self.vocabulary = Vocabulary::new();
        for (word, count) in &word_counts {
            if *count >= self.config.min_count {
                self.vocabulary.add_token(word);
            }
        }

        if self.vocabulary.is_empty() {
            return Err(TextError::VocabularyError(
                "No words meet the minimum count threshold".into(),
            ));
        }

        // Initialize embeddings
        let vocab_size = self.vocabulary.len();
        let vector_size = self.config.vector_size;

        // Initialize input and output embeddings with small random values
        let mut rng = scirs2_core::random::rng();
        let input_embeddings = Array2::from_shape_fn((vocab_size, vector_size), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) / vector_size as f64
        });
        let output_embeddings = Array2::from_shape_fn((vocab_size, vector_size), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) / vector_size as f64
        });

        self.input_embeddings = Some(input_embeddings);
        self.output_embeddings = Some(output_embeddings);

        // Create sampling table for negative sampling
        self.create_sampling_table(&word_counts)?;

        // Build Huffman tree for hierarchical softmax if configured
        if self.config.hierarchical_softmax {
            let frequencies: Vec<usize> = (0..vocab_size)
                .map(|i| {
                    self.vocabulary
                        .get_token(i)
                        .and_then(|word| word_counts.get(word).copied())
                        .unwrap_or(1)
                })
                .collect();

            let tree = HuffmanTree::build(&frequencies)?;
            let num_internal = tree.num_internal;

            // Initialize hierarchical softmax parameter vectors
            let hs_params = Array2::zeros((num_internal, vector_size));
            self.hs_params = Some(hs_params);
            self.huffman_tree = Some(tree);
        }

        Ok(())
    }

    /// Create sampling table for negative sampling based on word frequencies
    fn create_sampling_table(&mut self, wordcounts: &HashMap<String, usize>) -> Result<()> {
        // Prepare sampling weights (unigram distribution raised to 3/4 power)
        let mut sampling_weights = vec![0.0; self.vocabulary.len()];

        for (word, &count) in wordcounts.iter() {
            if let Some(idx) = self.vocabulary.get_index(word) {
                // Apply smoothing: frequency^0.75
                sampling_weights[idx] = (count as f64).powf(0.75);
            }
        }

        match SamplingTable::new(&sampling_weights) {
            Ok(table) => {
                self.sampling_table = Some(table);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Train the Word2Vec model on a corpus
    pub fn train(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for training".into(),
            ));
        }

        // Build vocabulary if not already built
        if self.vocabulary.is_empty() {
            self.build_vocabulary(texts)?;
        }

        if self.input_embeddings.is_none() || self.output_embeddings.is_none() {
            return Err(TextError::EmbeddingError(
                "Embeddings not initialized. Call build_vocabulary() first".into(),
            ));
        }

        // Count total number of tokens for progress tracking
        let mut _total_tokens = 0;
        let mut sentences = Vec::new();
        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;
            let filtered_tokens: Vec<usize> = tokens
                .iter()
                .filter_map(|token| self.vocabulary.get_index(token))
                .collect();
            if !filtered_tokens.is_empty() {
                _total_tokens += filtered_tokens.len();
                sentences.push(filtered_tokens);
            }
        }

        // Train for the specified number of epochs
        for epoch in 0..self.config.epochs {
            // Update learning rate for this epoch
            self.current_learning_rate =
                self.config.learning_rate * (1.0 - (epoch as f64 / self.config.epochs as f64));
            self.current_learning_rate = self
                .current_learning_rate
                .max(self.config.learning_rate * 0.0001);

            // Process each sentence
            for sentence in &sentences {
                // Apply subsampling of frequent words
                let subsampled_sentence = if self.config.subsample > 0.0 {
                    self.subsample_sentence(sentence)?
                } else {
                    sentence.clone()
                };

                // Skip empty sentences
                if subsampled_sentence.is_empty() {
                    continue;
                }

                // Train on the sentence
                if self.config.hierarchical_softmax {
                    // Use hierarchical softmax
                    match self.config.algorithm {
                        Word2VecAlgorithm::SkipGram => {
                            self.train_skipgram_hs_sentence(&subsampled_sentence)?;
                        }
                        Word2VecAlgorithm::CBOW => {
                            self.train_cbow_hs_sentence(&subsampled_sentence)?;
                        }
                    }
                } else {
                    // Use negative sampling
                    match self.config.algorithm {
                        Word2VecAlgorithm::CBOW => {
                            self.train_cbow_sentence(&subsampled_sentence)?;
                        }
                        Word2VecAlgorithm::SkipGram => {
                            self.train_skipgram_sentence(&subsampled_sentence)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply subsampling to a sentence
    fn subsample_sentence(&self, sentence: &[usize]) -> Result<Vec<usize>> {
        let mut rng = scirs2_core::random::rng();
        let total_words: f64 = self.vocabulary.len() as f64;
        let threshold = self.config.subsample * total_words;

        // Filter words based on subsampling probability
        let subsampled: Vec<usize> = sentence
            .iter()
            .filter(|&&word_idx| {
                let word_freq = self.get_word_frequency(word_idx);
                if word_freq == 0.0 {
                    return true; // Keep rare words
                }
                // Probability of keeping the word
                let keep_prob = ((word_freq / threshold).sqrt() + 1.0) * (threshold / word_freq);
                rng.random::<f64>() < keep_prob
            })
            .copied()
            .collect();

        Ok(subsampled)
    }

    /// Get the frequency of a word in the vocabulary
    fn get_word_frequency(&self, wordidx: usize) -> f64 {
        // This is a simplified version; ideal implementation would track actual frequencies
        // For now, we'll use the sampling table weights as a proxy
        if let Some(table) = &self.sampling_table {
            table.weights()[wordidx]
        } else {
            1.0 // Default weight if no sampling table exists
        }
    }

    /// Train CBOW model on a single sentence
    fn train_cbow_sentence(&mut self, sentence: &[usize]) -> Result<()> {
        if sentence.len() < 2 {
            return Ok(()); // Need at least 2 words for context
        }

        let input_embeddings = self.input_embeddings.as_mut().expect("Operation failed");
        let output_embeddings = self.output_embeddings.as_mut().expect("Operation failed");
        let vector_size = self.config.vector_size;
        let window_size = self.config.window_size;
        let negative_samples = self.config.negative_samples;

        // For each position in sentence, predict the word from its context
        for pos in 0..sentence.len() {
            // Determine context window (with random size)
            let mut rng = scirs2_core::random::rng();
            let window = 1 + rng.random_range(0..window_size);
            let target_word = sentence[pos];

            // Collect context words and average their vectors
            let mut context_words = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in pos.saturating_sub(window)..=(pos + window).min(sentence.len() - 1) {
                if i != pos {
                    context_words.push(sentence[i]);
                }
            }

            if context_words.is_empty() {
                continue; // No context words
            }

            // Average the context word vectors
            let mut context_sum = Array1::zeros(vector_size);
            for &context_idx in &context_words {
                context_sum += &input_embeddings.row(context_idx);
            }
            let context_avg = &context_sum / context_words.len() as f64;

            // Update target word's output embedding with positive example
            let mut target_output = output_embeddings.row_mut(target_word);
            let dot_product = (&context_avg * &target_output).sum();
            let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
            let error = (1.0 - sigmoid) * self.current_learning_rate;

            // Create a copy for update
            let mut target_update = target_output.to_owned();
            target_update.scaled_add(error, &context_avg);
            target_output.assign(&target_update);

            // Negative sampling
            if let Some(sampler) = &self.sampling_table {
                for _ in 0..negative_samples {
                    let negative_idx = sampler.sample(&mut rng);
                    if negative_idx == target_word {
                        continue; // Skip if we sample the target word
                    }

                    let mut negative_output = output_embeddings.row_mut(negative_idx);
                    let dot_product = (&context_avg * &negative_output).sum();
                    let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                    let error = -sigmoid * self.current_learning_rate;

                    // Create a copy for update
                    let mut negative_update = negative_output.to_owned();
                    negative_update.scaled_add(error, &context_avg);
                    negative_output.assign(&negative_update);
                }
            }

            // Update context word vectors
            for &context_idx in &context_words {
                let mut input_vec = input_embeddings.row_mut(context_idx);

                // Positive example
                let dot_product = (&context_avg * &output_embeddings.row(target_word)).sum();
                let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                let error =
                    (1.0 - sigmoid) * self.current_learning_rate / context_words.len() as f64;

                // Create a copy for update
                let mut input_update = input_vec.to_owned();
                input_update.scaled_add(error, &output_embeddings.row(target_word));

                // Negative examples
                if let Some(sampler) = &self.sampling_table {
                    for _ in 0..negative_samples {
                        let negative_idx = sampler.sample(&mut rng);
                        if negative_idx == target_word {
                            continue;
                        }

                        let dot_product =
                            (&context_avg * &output_embeddings.row(negative_idx)).sum();
                        let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                        let error =
                            -sigmoid * self.current_learning_rate / context_words.len() as f64;

                        input_update.scaled_add(error, &output_embeddings.row(negative_idx));
                    }
                }

                input_vec.assign(&input_update);
            }
        }

        Ok(())
    }

    /// Train Skip-gram model on a single sentence
    fn train_skipgram_sentence(&mut self, sentence: &[usize]) -> Result<()> {
        if sentence.len() < 2 {
            return Ok(()); // Need at least 2 words for context
        }

        let input_embeddings = self.input_embeddings.as_mut().expect("Operation failed");
        let output_embeddings = self.output_embeddings.as_mut().expect("Operation failed");
        let vector_size = self.config.vector_size;
        let window_size = self.config.window_size;
        let negative_samples = self.config.negative_samples;

        // For each position in sentence, predict the context from the word
        for pos in 0..sentence.len() {
            // Determine context window (with random size)
            let mut rng = scirs2_core::random::rng();
            let window = 1 + rng.random_range(0..window_size);
            let target_word = sentence[pos];

            // For each context position
            #[allow(clippy::needless_range_loop)]
            for i in pos.saturating_sub(window)..=(pos + window).min(sentence.len() - 1) {
                if i == pos {
                    continue; // Skip the target word itself
                }

                let context_word = sentence[i];
                let target_input = input_embeddings.row(target_word);
                let mut context_output = output_embeddings.row_mut(context_word);

                // Update context word's output embedding with positive example
                let dot_product = (&target_input * &context_output).sum();
                let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                let error = (1.0 - sigmoid) * self.current_learning_rate;

                // Create a copy for update
                let mut context_update = context_output.to_owned();
                context_update.scaled_add(error, &target_input);
                context_output.assign(&context_update);

                // Gradient for input word vector
                let mut input_update = Array1::zeros(vector_size);
                input_update.scaled_add(error, &context_output);

                // Negative sampling
                if let Some(sampler) = &self.sampling_table {
                    for _ in 0..negative_samples {
                        let negative_idx = sampler.sample(&mut rng);
                        if negative_idx == context_word {
                            continue; // Skip if we sample the context word
                        }

                        let mut negative_output = output_embeddings.row_mut(negative_idx);
                        let dot_product = (&target_input * &negative_output).sum();
                        let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                        let error = -sigmoid * self.current_learning_rate;

                        // Create a copy for update
                        let mut negative_update = negative_output.to_owned();
                        negative_update.scaled_add(error, &target_input);
                        negative_output.assign(&negative_update);

                        // Update input gradient
                        input_update.scaled_add(error, &negative_output);
                    }
                }

                // Apply the accumulated gradient to the input word vector
                let mut target_input_mut = input_embeddings.row_mut(target_word);
                target_input_mut += &input_update;
            }
        }

        Ok(())
    }

    /// Get the vector size
    pub fn vector_size(&self) -> usize {
        self.config.vector_size
    }

    /// Get the embedding vector for a word
    pub fn get_word_vector(&self, word: &str) -> Result<Array1<f64>> {
        if self.input_embeddings.is_none() {
            return Err(TextError::EmbeddingError(
                "Model not trained. Call train() first".into(),
            ));
        }

        match self.vocabulary.get_index(word) {
            Some(idx) => Ok(self
                .input_embeddings
                .as_ref()
                .expect("Operation failed")
                .row(idx)
                .to_owned()),
            None => Err(TextError::VocabularyError(format!(
                "Word '{word}' not in vocabulary"
            ))),
        }
    }

    /// Get the most similar words to a given word
    pub fn most_similar(&self, word: &str, topn: usize) -> Result<Vec<(String, f64)>> {
        let word_vec = self.get_word_vector(word)?;
        self.most_similar_by_vector(&word_vec, topn, &[word])
    }

    /// Get the most similar words to a given vector
    pub fn most_similar_by_vector(
        &self,
        vector: &Array1<f64>,
        top_n: usize,
        exclude_words: &[&str],
    ) -> Result<Vec<(String, f64)>> {
        if self.input_embeddings.is_none() {
            return Err(TextError::EmbeddingError(
                "Model not trained. Call train() first".into(),
            ));
        }

        let input_embeddings = self.input_embeddings.as_ref().expect("Operation failed");
        let vocab_size = self.vocabulary.len();

        // Create a set of indices to exclude
        let exclude_indices: Vec<usize> = exclude_words
            .iter()
            .filter_map(|&word| self.vocabulary.get_index(word))
            .collect();

        // Calculate cosine similarity for all _words
        let mut similarities = Vec::with_capacity(vocab_size);

        for i in 0..vocab_size {
            if exclude_indices.contains(&i) {
                continue;
            }

            let word_vec = input_embeddings.row(i);
            let similarity = cosine_similarity(vector, &word_vec.to_owned());

            if let Some(word) = self.vocabulary.get_token(i) {
                similarities.push((word.to_string(), similarity));
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        let result = similarities.into_iter().take(top_n).collect();
        Ok(result)
    }

    /// Compute the analogy: a is to b as c is to ?
    pub fn analogy(&self, a: &str, b: &str, c: &str, topn: usize) -> Result<Vec<(String, f64)>> {
        if self.input_embeddings.is_none() {
            return Err(TextError::EmbeddingError(
                "Model not trained. Call train() first".into(),
            ));
        }

        // Get vectors for a, b, and c
        let a_vec = self.get_word_vector(a)?;
        let b_vec = self.get_word_vector(b)?;
        let c_vec = self.get_word_vector(c)?;

        // Compute d_vec = b_vec - a_vec + c_vec
        let mut d_vec = b_vec.clone();
        d_vec -= &a_vec;
        d_vec += &c_vec;

        // Normalize the vector
        let norm = (d_vec.iter().fold(0.0, |sum, &val| sum + val * val)).sqrt();
        d_vec.mapv_inplace(|val| val / norm);

        // Find most similar words to d_vec
        self.most_similar_by_vector(&d_vec, topn, &[a, b, c])
    }

    /// Save the Word2Vec model to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        if self.input_embeddings.is_none() {
            return Err(TextError::EmbeddingError(
                "Model not trained. Call train() first".into(),
            ));
        }

        let mut file = File::create(path).map_err(|e| TextError::IoError(e.to_string()))?;

        // Write header: vector_size and vocabulary size
        writeln!(
            &mut file,
            "{} {}",
            self.vocabulary.len(),
            self.config.vector_size
        )
        .map_err(|e| TextError::IoError(e.to_string()))?;

        // Write each word and its vector
        let input_embeddings = self.input_embeddings.as_ref().expect("Operation failed");

        for i in 0..self.vocabulary.len() {
            if let Some(word) = self.vocabulary.get_token(i) {
                // Write the word
                write!(&mut file, "{word} ").map_err(|e| TextError::IoError(e.to_string()))?;

                // Write the vector components
                let vector = input_embeddings.row(i);
                for j in 0..self.config.vector_size {
                    write!(&mut file, "{:.6} ", vector[j])
                        .map_err(|e| TextError::IoError(e.to_string()))?;
                }

                writeln!(&mut file).map_err(|e| TextError::IoError(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Load a Word2Vec model from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| TextError::IoError(e.to_string()))?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header = String::new();
        reader
            .read_line(&mut header)
            .map_err(|e| TextError::IoError(e.to_string()))?;

        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(TextError::EmbeddingError(
                "Invalid model file format".into(),
            ));
        }

        let vocab_size = parts[0].parse::<usize>().map_err(|_| {
            TextError::EmbeddingError("Invalid vocabulary size in model file".into())
        })?;

        let vector_size = parts[1]
            .parse::<usize>()
            .map_err(|_| TextError::EmbeddingError("Invalid vector size in model file".into()))?;

        // Initialize model
        let mut model = Self::new().with_vector_size(vector_size);
        let mut vocabulary = Vocabulary::new();
        let mut input_embeddings = Array2::zeros((vocab_size, vector_size));

        // Read each word and its vector
        let mut i = 0;
        for line in reader.lines() {
            let line = line.map_err(|e| TextError::IoError(e.to_string()))?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() != vector_size + 1 {
                let line_num = i + 2;
                return Err(TextError::EmbeddingError(format!(
                    "Invalid vector format at line {line_num}"
                )));
            }

            let word = parts[0];
            vocabulary.add_token(word);

            for j in 0..vector_size {
                input_embeddings[(i, j)] = parts[j + 1].parse::<f64>().map_err(|_| {
                    TextError::EmbeddingError(format!(
                        "Invalid vector component at line {}, position {}",
                        i + 2,
                        j + 1
                    ))
                })?;
            }

            i += 1;
        }

        if i != vocab_size {
            return Err(TextError::EmbeddingError(format!(
                "Expected {vocab_size} words but found {i}"
            )));
        }

        model.vocabulary = vocabulary;
        model.input_embeddings = Some(input_embeddings);
        model.output_embeddings = None; // Only input embeddings are saved

        Ok(model)
    }

    // Getter methods for model registry serialization

    /// Get the vocabulary as a vector of strings
    pub fn get_vocabulary(&self) -> Vec<String> {
        let mut vocab = Vec::new();
        for i in 0..self.vocabulary.len() {
            if let Some(token) = self.vocabulary.get_token(i) {
                vocab.push(token.to_string());
            }
        }
        vocab
    }

    /// Get the vector size
    pub fn get_vector_size(&self) -> usize {
        self.config.vector_size
    }

    /// Get the algorithm
    pub fn get_algorithm(&self) -> Word2VecAlgorithm {
        self.config.algorithm
    }

    /// Get the window size
    pub fn get_window_size(&self) -> usize {
        self.config.window_size
    }

    /// Get the minimum count
    pub fn get_min_count(&self) -> usize {
        self.config.min_count
    }

    /// Get the embeddings matrix (input embeddings)
    pub fn get_embeddings_matrix(&self) -> Option<Array2<f64>> {
        self.input_embeddings.clone()
    }

    /// Get the number of negative samples
    pub fn get_negative_samples(&self) -> usize {
        self.config.negative_samples
    }

    /// Get the learning rate
    pub fn get_learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    /// Get the number of epochs
    pub fn get_epochs(&self) -> usize {
        self.config.epochs
    }

    /// Get the subsampling threshold
    pub fn get_subsampling_threshold(&self) -> f64 {
        self.config.subsample
    }

    /// Check if hierarchical softmax is enabled
    pub fn uses_hierarchical_softmax(&self) -> bool {
        self.config.hierarchical_softmax
    }

    /// Restore vocabulary and input embeddings from serialized state.
    ///
    /// Validates that:
    /// - `embeddings` row count equals `vocabulary.len()`
    /// - `embeddings` column count equals `self.config.vector_size`
    ///
    /// Returns an error on any dimension mismatch; never panics.
    pub fn restore_weights(
        &mut self,
        vocabulary: Vec<String>,
        embeddings: Array2<f64>,
    ) -> Result<()> {
        let embed_shape = embeddings.shape();
        let n_words = vocabulary.len();

        if embed_shape[0] != n_words {
            return Err(TextError::EmbeddingError(format!(
                "Embedding row count {} does not match vocabulary size {}",
                embed_shape[0], n_words
            )));
        }

        if embed_shape[1] != self.config.vector_size {
            return Err(TextError::EmbeddingError(format!(
                "Embedding dimension {} does not match configured vector_size {}",
                embed_shape[1], self.config.vector_size
            )));
        }

        // Rebuild the vocabulary index
        self.vocabulary = Vocabulary::new();
        for word in &vocabulary {
            self.vocabulary.add_token(word);
        }

        self.input_embeddings = Some(embeddings);
        Ok(())
    }

    // ─── Hierarchical Softmax Training ──────────────────────────────────

    /// Train Skip-gram with hierarchical softmax on a single sentence
    fn train_skipgram_hs_sentence(&mut self, sentence: &[usize]) -> Result<()> {
        if sentence.len() < 2 {
            return Ok(());
        }

        let input_embeddings = self
            .input_embeddings
            .as_mut()
            .ok_or_else(|| TextError::EmbeddingError("Input embeddings not initialized".into()))?;
        let hs_params = self
            .hs_params
            .as_mut()
            .ok_or_else(|| TextError::EmbeddingError("HS params not initialized".into()))?;
        let tree = self
            .huffman_tree
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("Huffman tree not built".into()))?;

        let vector_size = self.config.vector_size;
        let window_size = self.config.window_size;
        let lr = self.current_learning_rate;

        let codes = tree.codes.clone();
        let paths = tree.paths.clone();

        let mut rng = scirs2_core::random::rng();

        for pos in 0..sentence.len() {
            let window = 1 + rng.random_range(0..window_size);
            let target_word = sentence[pos];

            for i in pos.saturating_sub(window)..=(pos + window).min(sentence.len() - 1) {
                if i == pos {
                    continue;
                }

                let context_word = sentence[i];
                let code = &codes[context_word];
                let path = &paths[context_word];

                let mut grad_input = Array1::zeros(vector_size);

                // Walk the Huffman tree path for the context word
                for (step, (&node_idx, &label)) in path.iter().zip(code.iter()).enumerate() {
                    if node_idx >= hs_params.nrows() {
                        continue;
                    }

                    // Compute sigmoid(input_vec . hs_param)
                    let input_vec = input_embeddings.row(target_word);
                    let param_vec = hs_params.row(node_idx);

                    let dot: f64 = input_vec
                        .iter()
                        .zip(param_vec.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    let sigmoid = 1.0 / (1.0 + (-dot).exp());

                    // gradient: (1 - label - sigmoid) * lr
                    let target = if label == 0 { 1.0 } else { 0.0 };
                    let gradient = (target - sigmoid) * lr;

                    // Update gradient for input vector
                    grad_input.scaled_add(gradient, &param_vec.to_owned());

                    // Update HS parameter vector
                    let input_owned = input_vec.to_owned();
                    let mut param_mut = hs_params.row_mut(node_idx);
                    param_mut.scaled_add(gradient, &input_owned);
                }

                // Update input embedding
                let mut input_mut = input_embeddings.row_mut(target_word);
                input_mut += &grad_input;
            }
        }

        Ok(())
    }

    /// Train CBOW with hierarchical softmax on a single sentence
    fn train_cbow_hs_sentence(&mut self, sentence: &[usize]) -> Result<()> {
        if sentence.len() < 2 {
            return Ok(());
        }

        let input_embeddings = self
            .input_embeddings
            .as_mut()
            .ok_or_else(|| TextError::EmbeddingError("Input embeddings not initialized".into()))?;
        let hs_params = self
            .hs_params
            .as_mut()
            .ok_or_else(|| TextError::EmbeddingError("HS params not initialized".into()))?;
        let tree = self
            .huffman_tree
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("Huffman tree not built".into()))?;

        let vector_size = self.config.vector_size;
        let window_size = self.config.window_size;
        let lr = self.current_learning_rate;

        let codes = tree.codes.clone();
        let paths = tree.paths.clone();

        let mut rng = scirs2_core::random::rng();

        for pos in 0..sentence.len() {
            let window = 1 + rng.random_range(0..window_size);
            let target_word = sentence[pos];

            // Collect context words
            let mut context_words = Vec::new();
            for i in pos.saturating_sub(window)..=(pos + window).min(sentence.len() - 1) {
                if i != pos {
                    context_words.push(sentence[i]);
                }
            }

            if context_words.is_empty() {
                continue;
            }

            // Average context word vectors
            let mut context_avg = Array1::zeros(vector_size);
            for &ctx_idx in &context_words {
                context_avg += &input_embeddings.row(ctx_idx);
            }
            context_avg /= context_words.len() as f64;

            // Walk Huffman path for target word
            let code = &codes[target_word];
            let path = &paths[target_word];

            let mut grad_context = Array1::zeros(vector_size);

            for (step, (&node_idx, &label)) in path.iter().zip(code.iter()).enumerate() {
                if node_idx >= hs_params.nrows() {
                    continue;
                }

                let param_vec = hs_params.row(node_idx);

                let dot: f64 = context_avg
                    .iter()
                    .zip(param_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let sigmoid = 1.0 / (1.0 + (-dot).exp());

                let target = if label == 0 { 1.0 } else { 0.0 };
                let gradient = (target - sigmoid) * lr;

                grad_context.scaled_add(gradient, &param_vec.to_owned());

                // Update HS parameter
                let ctx_owned = context_avg.clone();
                let mut param_mut = hs_params.row_mut(node_idx);
                param_mut.scaled_add(gradient, &ctx_owned);
            }

            // Distribute gradient back to context word input embeddings
            let grad_per_word = &grad_context / context_words.len() as f64;
            for &ctx_idx in &context_words {
                let mut input_mut = input_embeddings.row_mut(ctx_idx);
                input_mut += &grad_per_word;
            }
        }

        Ok(())
    }
}

// ─── WordEmbedding trait implementation for Word2Vec ─────────────────────────

impl WordEmbedding for Word2Vec {
    fn embedding(&self, word: &str) -> Result<Array1<f64>> {
        self.get_word_vector(word)
    }

    fn dimension(&self) -> usize {
        self.vector_size()
    }

    fn find_similar(&self, word: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        self.most_similar(word, top_n)
    }

    fn solve_analogy(&self, a: &str, b: &str, c: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        self.analogy(a, b, c, top_n)
    }

    fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Calculate cosine similarity between two vectors
#[allow(dead_code)]
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot_product = (a * b).sum();
    let norm_a = (a.iter().fold(0.0, |sum, &val| sum + val * val)).sqrt();
    let norm_b = (b.iter().fold(0.0, |sum, &val| sum + val * val)).sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let similarity = cosine_similarity(&a, &b);
        let expected = 0.9746318461970762;
        assert_relative_eq!(similarity, expected, max_relative = 1e-10);
    }

    #[test]
    fn test_word2vec_config() {
        let config = Word2VecConfig::default();
        assert_eq!(config.vector_size, 100);
        assert_eq!(config.window_size, 5);
        assert_eq!(config.min_count, 5);
        assert_eq!(config.epochs, 5);
        assert_eq!(config.algorithm, Word2VecAlgorithm::SkipGram);
    }

    #[test]
    fn test_word2vec_builder() {
        let model = Word2Vec::new()
            .with_vector_size(200)
            .with_window_size(10)
            .with_learning_rate(0.05)
            .with_algorithm(Word2VecAlgorithm::CBOW);

        assert_eq!(model.config.vector_size, 200);
        assert_eq!(model.config.window_size, 10);
        assert_eq!(model.config.learning_rate, 0.05);
        assert_eq!(model.config.algorithm, Word2VecAlgorithm::CBOW);
    }

    #[test]
    fn test_build_vocabulary() {
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "a quick brown fox jumps over a lazy dog",
        ];

        let mut model = Word2Vec::new().with_min_count(1);
        let result = model.build_vocabulary(&texts);
        assert!(result.is_ok());

        // Check vocabulary size (unique words: "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "a")
        assert_eq!(model.vocabulary.len(), 9);

        // Check that embeddings were initialized
        assert!(model.input_embeddings.is_some());
        assert!(model.output_embeddings.is_some());
        assert_eq!(
            model
                .input_embeddings
                .as_ref()
                .expect("Operation failed")
                .shape(),
            &[9, 100]
        );
    }

    #[test]
    fn test_skipgram_training_small() {
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "a quick brown fox jumps over a lazy dog",
        ];

        let mut model = Word2Vec::new()
            .with_vector_size(10)
            .with_window_size(2)
            .with_min_count(1)
            .with_epochs(1)
            .with_algorithm(Word2VecAlgorithm::SkipGram);

        let result = model.train(&texts);
        assert!(result.is_ok());

        // Test getting a word vector
        let result = model.get_word_vector("fox");
        assert!(result.is_ok());
        let vec = result.expect("Operation failed");
        assert_eq!(vec.len(), 10);
    }

    // ─── Hierarchical Softmax Tests ──────────────────────────────────

    #[test]
    fn test_huffman_tree_build() {
        let frequencies = vec![5, 3, 8, 1, 2];
        let tree = HuffmanTree::build(&frequencies).expect("Huffman build failed");

        // Each word should have a code
        assert_eq!(tree.codes.len(), 5);
        assert_eq!(tree.paths.len(), 5);

        // All codes should be non-empty
        for code in &tree.codes {
            assert!(!code.is_empty());
        }

        // Internal nodes = vocab_size - 1 (for a binary tree)
        assert_eq!(tree.num_internal, 4);
    }

    #[test]
    fn test_huffman_tree_single_word() {
        let frequencies = vec![10];
        let tree = HuffmanTree::build(&frequencies).expect("Huffman build failed");
        assert_eq!(tree.codes.len(), 1);
        assert_eq!(tree.paths.len(), 1);
    }

    #[test]
    fn test_skipgram_hierarchical_softmax() {
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "a quick brown fox jumps over a lazy dog",
        ];

        let config = Word2VecConfig {
            vector_size: 10,
            window_size: 2,
            min_count: 1,
            epochs: 3,
            learning_rate: 0.025,
            algorithm: Word2VecAlgorithm::SkipGram,
            hierarchical_softmax: true,
            ..Default::default()
        };

        let mut model = Word2Vec::with_config(config);
        let result = model.train(&texts);
        assert!(
            result.is_ok(),
            "HS skipgram training failed: {:?}",
            result.err()
        );

        assert!(model.uses_hierarchical_softmax());

        // Should produce valid word vectors
        let vec = model.get_word_vector("fox");
        assert!(vec.is_ok());
        assert_eq!(vec.expect("get vec").len(), 10);
    }

    #[test]
    fn test_cbow_hierarchical_softmax() {
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "a quick brown fox jumps over a lazy dog",
        ];

        let config = Word2VecConfig {
            vector_size: 10,
            window_size: 2,
            min_count: 1,
            epochs: 3,
            learning_rate: 0.025,
            algorithm: Word2VecAlgorithm::CBOW,
            hierarchical_softmax: true,
            ..Default::default()
        };

        let mut model = Word2Vec::with_config(config);
        let result = model.train(&texts);
        assert!(
            result.is_ok(),
            "HS CBOW training failed: {:?}",
            result.err()
        );

        let vec = model.get_word_vector("dog");
        assert!(vec.is_ok());
    }

    // ─── WordEmbedding Trait Tests ──────────────────────────────────

    #[test]
    fn test_word_embedding_trait_word2vec() {
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "a quick brown fox jumps over a lazy dog",
        ];

        let mut model = Word2Vec::new()
            .with_vector_size(10)
            .with_min_count(1)
            .with_epochs(1);

        model.train(&texts).expect("Training failed");

        // Use via trait
        let emb: &dyn WordEmbedding = &model;
        assert_eq!(emb.dimension(), 10);
        assert!(emb.vocab_size() > 0);

        let vec = emb.embedding("fox");
        assert!(vec.is_ok());

        let sim = emb.similarity("fox", "dog");
        assert!(sim.is_ok());
        assert!(sim.expect("sim").is_finite());

        let similar = emb.find_similar("fox", 2);
        assert!(similar.is_ok());

        let analogy = emb.solve_analogy("the", "fox", "dog", 2);
        assert!(analogy.is_ok());
    }

    #[test]
    fn test_embedding_cosine_similarity_fn() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        assert!((embedding_cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);

        let c = Array1::from_vec(vec![1.0, 1.0]);
        let d = Array1::from_vec(vec![1.0, 1.0]);
        assert!((embedding_cosine_similarity(&c, &d) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pairwise_similarity_fn() {
        let texts = ["the quick brown fox", "the lazy brown dog"];

        let mut model = Word2Vec::new()
            .with_vector_size(10)
            .with_min_count(1)
            .with_epochs(1);
        model.train(&texts).expect("Training failed");

        let words = vec!["the", "fox", "dog"];
        let matrix = pairwise_similarity(&model, &words).expect("pairwise failed");

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);

        // Diagonal should be 1.0
        for i in 0..3 {
            assert!((matrix[i][i] - 1.0).abs() < 1e-6);
        }

        // Symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-10);
            }
        }
    }
}
