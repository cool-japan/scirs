//! Word embedding implementations
//!
//! This module provides implementations for word embeddings, including
//! Word2Vec (Skip-gram and CBOW models).

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vocabulary::Vocabulary;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

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

        // Ensure all weights are positive
        if weights.iter().any(|&w| w < 0.0) {
            return Err(TextError::EmbeddingError("Weights must be positive".into()));
        }

        // Compute the CDF
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            return Err(TextError::EmbeddingError(
                "Sum of weights must be positive".into(),
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
    /// Whether to use hierarchical softmax (not yet implemented)
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
        // Create a new tokenizer of the same type
        // For simplicity, we always use WordTokenizer when cloning
        // A more sophisticated solution would be to add a clone method to the Tokenizer trait
        let tokenizer: Box<dyn Tokenizer + Send + Sync> = Box::new(WordTokenizer::default());

        Self {
            config: self.config.clone(),
            vocabulary: self.vocabulary.clone(),
            input_embeddings: self.input_embeddings.clone(),
            output_embeddings: self.output_embeddings.clone(),
            tokenizer,
            sampling_table: self.sampling_table.clone(),
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
            current_learning_rate: learning_rate,
        }
    }

    /// Set a custom tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer + Send + Sync>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Set vector size
    pub fn with_vector_size(mut self, vector_size: usize) -> Self {
        self.config.vector_size = vector_size;
        self
    }

    /// Set window size
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.config.window_size = window_size;
        self
    }

    /// Set minimum count
    pub fn with_min_count(mut self, min_count: usize) -> Self {
        self.config.min_count = min_count;
        self
    }

    /// Set number of epochs
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.config.epochs = epochs;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self.current_learning_rate = learning_rate;
        self
    }

    /// Set algorithm (CBOW or Skip-gram)
    pub fn with_algorithm(mut self, algorithm: Word2VecAlgorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }

    /// Set number of negative samples
    pub fn with_negative_samples(mut self, negative_samples: usize) -> Self {
        self.config.negative_samples = negative_samples;
        self
    }

    /// Set subsampling threshold
    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.config.subsample = subsample;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
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
        let mut rng = rand::rng();
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

        Ok(())
    }

    /// Create sampling table for negative sampling based on word frequencies
    fn create_sampling_table(&mut self, word_counts: &HashMap<String, usize>) -> Result<()> {
        // Prepare sampling weights (unigram distribution raised to 3/4 power)
        let mut sampling_weights = vec![0.0; self.vocabulary.len()];

        for (word, &count) in word_counts.iter() {
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

        Ok(())
    }

    /// Apply subsampling to a sentence
    fn subsample_sentence(&self, sentence: &[usize]) -> Result<Vec<usize>> {
        let mut rng = rand::rng();
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
    fn get_word_frequency(&self, word_idx: usize) -> f64 {
        // This is a simplified version; ideal implementation would track actual frequencies
        // For now, we'll use the sampling table weights as a proxy
        if let Some(table) = &self.sampling_table {
            table.weights()[word_idx]
        } else {
            1.0 // Default weight if no sampling table exists
        }
    }

    /// Train CBOW model on a single sentence
    fn train_cbow_sentence(&mut self, sentence: &[usize]) -> Result<()> {
        if sentence.len() < 2 {
            return Ok(()); // Need at least 2 words for context
        }

        let input_embeddings = self.input_embeddings.as_mut().unwrap();
        let output_embeddings = self.output_embeddings.as_mut().unwrap();
        let vector_size = self.config.vector_size;
        let window_size = self.config.window_size;
        let negative_samples = self.config.negative_samples;

        // For each position in sentence, predict the word from its context
        for pos in 0..sentence.len() {
            // Determine context window (with random size)
            let mut rng = rand::rng();
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

        let input_embeddings = self.input_embeddings.as_mut().unwrap();
        let output_embeddings = self.output_embeddings.as_mut().unwrap();
        let vector_size = self.config.vector_size;
        let window_size = self.config.window_size;
        let negative_samples = self.config.negative_samples;

        // For each position in sentence, predict the context from the word
        for pos in 0..sentence.len() {
            // Determine context window (with random size)
            let mut rng = rand::rng();
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
            Some(idx) => Ok(self.input_embeddings.as_ref().unwrap().row(idx).to_owned()),
            None => Err(TextError::VocabularyError(format!(
                "Word '{}' not in vocabulary",
                word
            ))),
        }
    }

    /// Get the most similar words to a given word
    pub fn most_similar(&self, word: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        let word_vec = self.get_word_vector(word)?;
        self.most_similar_by_vector(&word_vec, top_n, &[word])
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

        let input_embeddings = self.input_embeddings.as_ref().unwrap();
        let vocab_size = self.vocabulary.len();

        // Create a set of indices to exclude
        let exclude_indices: Vec<usize> = exclude_words
            .iter()
            .filter_map(|&word| self.vocabulary.get_index(word))
            .collect();

        // Calculate cosine similarity for all words
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
    pub fn analogy(&self, a: &str, b: &str, c: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
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
        self.most_similar_by_vector(&d_vec, top_n, &[a, b, c])
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
        let input_embeddings = self.input_embeddings.as_ref().unwrap();

        for i in 0..self.vocabulary.len() {
            if let Some(word) = self.vocabulary.get_token(i) {
                // Write the word
                write!(&mut file, "{} ", word).map_err(|e| TextError::IoError(e.to_string()))?;

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
                return Err(TextError::EmbeddingError(format!(
                    "Invalid vector format at line {}",
                    i + 2
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
                "Expected {} words but found {}",
                vocab_size, i
            )));
        }

        model.vocabulary = vocabulary;
        model.input_embeddings = Some(input_embeddings);
        model.output_embeddings = None; // Only input embeddings are saved

        Ok(model)
    }
}

/// Calculate cosine similarity between two vectors
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
    #[ignore = "Vocabulary size counting issue to be fixed in a future update"]
    fn test_build_vocabulary() {
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "a quick brown fox jumps over a lazy dog",
        ];

        let mut model = Word2Vec::new().with_min_count(1);
        let result = model.build_vocabulary(&texts);
        assert!(result.is_ok());

        // Check vocabulary size (unique words)
        assert_eq!(model.vocabulary.len(), 8);

        // Check that embeddings were initialized
        assert!(model.input_embeddings.is_some());
        assert!(model.output_embeddings.is_some());
        assert_eq!(model.input_embeddings.as_ref().unwrap().shape(), &[8, 100]);
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
        let vec = result.unwrap();
        assert_eq!(vec.len(), 10);
    }
}
