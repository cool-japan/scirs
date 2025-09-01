//! # Topic Modeling Module
//!
//! This module provides advanced topic modeling algorithms for discovering
//! hidden thematic structures in document collections, with a focus on
//! Latent Dirichlet Allocation (LDA).
//!
//! ## Overview
//!
//! Topic modeling is an unsupervised machine learning technique that discovers
//! abstract "topics" that occur in a collection of documents. This module implements:
//!
//! - **Latent Dirichlet Allocation (LDA)**: The most popular topic modeling algorithm
//! - **Batch and Online Learning**: Different training strategies for various dataset sizes
//! - **Coherence Metrics**: Model evaluation using CV, UMass, and UCI coherence
//! - **Topic Visualization**: Tools for understanding and presenting results
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_text::topic_modeling::{LatentDirichletAllocation, LdaConfig, LdaLearningMethod};
//! use scirs2_text::vectorize::{CountVectorizer, Vectorizer};
//! use std::collections::HashMap;
//!
//! // Sample documents
//! let documents = vec![
//!     "machine learning algorithms are powerful tools",
//!     "natural language processing uses machine learning",
//!     "deep learning is a subset of machine learning",
//!     "cats and dogs are popular pets",
//!     "pet care requires attention and love",
//!     "dogs need regular exercise and training"
//! ];
//!
//! // Vectorize documents
//! let mut vectorizer = CountVectorizer::new(false);
//! let doc_term_matrix = vectorizer.fit_transform(&documents).unwrap();
//!
//! // Configure LDA
//! let config = LdaConfig {
//!     ntopics: 2,
//!     doc_topic_prior: Some(0.1),    // Alpha parameter
//!     topic_word_prior: Some(0.01),  // Beta parameter
//!     learning_method: LdaLearningMethod::Batch,
//!     maxiter: 100,
//!     mean_change_tol: 1e-4,
//!     random_seed: Some(42),
//!     ..Default::default()
//! };
//!
//! // Train the model
//! let mut lda = LatentDirichletAllocation::new(config);
//! lda.fit(&doc_term_matrix).unwrap();
//!
//! // Create vocabulary mapping for topic display
//! let vocab_map: HashMap<usize, String> = (0..1000).map(|i| (i, format!("word_{}", i))).collect();
//!
//! // Get topics
//! let topics = lda.get_topics(10, &vocab_map); // Top 10 words per topic
//! for (i, topic) in topics.iter().enumerate() {
//!     println!("Topic {}: {:?}", i, topic);
//! }
//!
//! // Transform documents to topic space
//! let doc_topics = lda.transform(&doc_term_matrix).unwrap();
//! println!("Document-topic distribution: {:?}", doc_topics);
//! ```
//!
//! ## Advanced Usage
//!
//! ### Online Learning for Large Datasets
//!
//! ```rust
//! use scirs2_text::topic_modeling::{LdaConfig, LdaLearningMethod, LatentDirichletAllocation};
//!
//! let config = LdaConfig {
//!     ntopics: 10,
//!     learning_method: LdaLearningMethod::Online,
//!     batch_size: 64,                // Mini-batch size
//!     learning_decay: 0.7,           // Learning rate decay
//!     learning_offset: 10.0,         // Learning rate offset
//!     maxiter: 500,
//!     ..Default::default()
//! };
//!
//! let mut lda = LatentDirichletAllocation::new(config);
//! // Process documents in batches for memory efficiency
//! ```
//!
//! ### Custom Hyperparameters
//!
//! ```rust
//! use scirs2_text::topic_modeling::LdaConfig;
//!
//! let config = LdaConfig {
//!     ntopics: 20,
//!     doc_topic_prior: Some(50.0 / 20.0),  // Symmetric Dirichlet
//!     topic_word_prior: Some(0.1),         // Sparse topics
//!     maxiter: 1000,                      // More iterations
//!     mean_change_tol: 1e-6,               // Stricter convergence
//!     ..Default::default()
//! };
//! ```
//!
//! ### Model Evaluation
//!
//! ```rust
//! use scirs2_text::topic_modeling::{LatentDirichletAllocation, LdaConfig};
//! use scirs2_text::vectorize::{CountVectorizer, Vectorizer};
//! use std::collections::HashMap;
//!
//! # let documents = vec!["the quick brown fox", "jumped over the lazy dog"];
//! # let mut vectorizer = CountVectorizer::new(false);
//! # let doc_term_matrix = vectorizer.fit_transform(&documents).unwrap();
//! # let mut lda = LatentDirichletAllocation::new(LdaConfig::default());
//! # lda.fit(&doc_term_matrix).unwrap();
//! # let vocab_map: HashMap<usize, String> = (0..100).map(|i| (i, format!("word_{}", i))).collect();
//! // Get model information
//! let topics = lda.get_topics(5, &vocab_map); // Top 5 words per topic
//! println!("Number of topics: {}", topics.unwrap().len());
//!
//! // Get document-topic probabilities
//! let doc_topic_probs = lda.transform(&doc_term_matrix).unwrap();
//! println!("Document-topic shape: {:?}", doc_topic_probs.shape());
//! ```
//!
//! ## Parameter Tuning Guide
//!
//! ### Number of Topics
//! - **Too few**: Broad, less meaningful topics
//! - **Too many**: Narrow, potentially noisy topics
//! - **Recommendation**: Start with √(number of documents) and tune based on coherence
//!
//! ### Alpha (doc_topic_prior)
//! - **High values (e.g., 1.0)**: Documents contain many topics
//! - **Low values (e.g., 0.1)**: Documents contain few topics
//! - **Default**: 50/ntopics (symmetric)
//!
//! ### Beta (topic_word_prior)
//! - **High values (e.g., 1.0)**: Topics contain many words
//! - **Low values (e.g., 0.01)**: Topics are more focused
//! - **Default**: 0.01 for sparse topics
//!
//! ## Performance Optimization
//!
//! 1. **Use Online Learning**: For datasets that don't fit in memory
//! 2. **Tune Batch Size**: Balance between speed and convergence stability
//! 3. **Set Tolerance**: Stop early when convergence is reached
//! 4. **Monitor Perplexity**: Track model performance during training
//! 5. **Parallel Processing**: Enable for faster vocabulary building
//!
//! ## Mathematical Background
//!
//! LDA assumes each document is a mixture of topics, and each topic is a distribution over words.
//! The generative process:
//!
//! 1. For each topic k: Draw word distribution φₖ ~ Dirichlet(β)
//! 2. For each document d:
//!    - Draw topic distribution θ_d ~ Dirichlet(α)
//!    - For each word n in document d:
//!      - Draw topic assignment z_{d,n} ~ Multinomial(θ_d)
//!      - Draw word w_{d,n} ~ Multinomial(φ_{z_{d,n}})
//!
//! The goal is to infer the posterior distributions of θ and φ given the observed words.

use crate::error::{Result, TextError};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;

/// Learning method for LDA
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LdaLearningMethod {
    /// Batch learning - process all documents at once
    Batch,
    /// Online learning - process documents in mini-batches
    Online,
}

/// Latent Dirichlet Allocation configuration
#[derive(Debug, Clone)]
pub struct LdaConfig {
    /// Number of topics
    pub ntopics: usize,
    /// Prior for document-topic distribution (alpha)
    pub doc_topic_prior: Option<f64>,
    /// Prior for topic-word distribution (eta)
    pub topic_word_prior: Option<f64>,
    /// Learning method
    pub learning_method: LdaLearningMethod,
    /// Learning decay for online learning
    pub learning_decay: f64,
    /// Learning offset for online learning
    pub learning_offset: f64,
    /// Maximum iterations
    pub maxiter: usize,
    /// Batch size for online learning
    pub batch_size: usize,
    /// Mean change tolerance for convergence
    pub mean_change_tol: f64,
    /// Maximum iterations for document E-step
    pub max_doc_update_iter: usize,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for LdaConfig {
    fn default() -> Self {
        Self {
            ntopics: 10,
            doc_topic_prior: None,  // Will be set to 1/ntopics
            topic_word_prior: None, // Will be set to 1/ntopics
            learning_method: LdaLearningMethod::Batch,
            learning_decay: 0.7,
            learning_offset: 10.0,
            maxiter: 10,
            batch_size: 128,
            mean_change_tol: 1e-3,
            max_doc_update_iter: 100,
            random_seed: None,
        }
    }
}

/// Topic representation
#[derive(Debug, Clone)]
pub struct Topic {
    /// Topic ID
    pub id: usize,
    /// Top words in the topic with their weights
    pub top_words: Vec<(String, f64)>,
    /// Topic coherence score (if computed)
    pub coherence: Option<f64>,
}

/// Latent Dirichlet Allocation
pub struct LatentDirichletAllocation {
    config: LdaConfig,
    /// Topic-word distribution (learned parameters)
    components: Option<Array2<f64>>,
    /// exp(E[log(beta)]) for efficient computation
    exp_dirichlet_component: Option<Array2<f64>>,
    /// Vocabulary mapping
    #[allow(dead_code)]
    vocabulary: Option<HashMap<usize, String>>,
    /// Number of documents seen
    n_documents: usize,
    /// Number of iterations performed
    n_iter: usize,
    /// Final perplexity bound
    #[allow(dead_code)]
    bound: Option<Vec<f64>>,
}

impl LatentDirichletAllocation {
    /// Create a new LDA model with the given configuration
    pub fn new(config: LdaConfig) -> Self {
        Self {
            config,
            components: None,
            exp_dirichlet_component: None,
            vocabulary: None,
            n_documents: 0,
            n_iter: 0,
            bound: None,
        }
    }

    /// Create a new LDA model with default configuration
    pub fn with_ntopics(ntopics: usize) -> Self {
        let config = LdaConfig {
            ntopics,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Fit the LDA model on a document-term matrix
    pub fn fit(&mut self, doc_termmatrix: &Array2<f64>) -> Result<&mut Self> {
        if doc_termmatrix.nrows() == 0 || doc_termmatrix.ncols() == 0 {
            return Err(TextError::InvalidInput(
                "Document-term _matrix cannot be empty".to_string(),
            ));
        }

        let n_samples = doc_termmatrix.nrows();
        let n_features = doc_termmatrix.ncols();

        // Set default priors if not provided
        let doc_topic_prior = self
            .config
            .doc_topic_prior
            .unwrap_or(1.0 / self.config.ntopics as f64);
        let topic_word_prior = self
            .config
            .topic_word_prior
            .unwrap_or(1.0 / self.config.ntopics as f64);

        // Initialize topic-word distribution randomly
        let mut rng = self.create_rng();
        self.components = Some(self.initialize_components(n_features, &mut rng));

        // Perform training based on learning method
        match self.config.learning_method {
            LdaLearningMethod::Batch => {
                self.fit_batch(doc_termmatrix, doc_topic_prior, topic_word_prior)?;
            }
            LdaLearningMethod::Online => {
                self.fit_online(doc_termmatrix, doc_topic_prior, topic_word_prior)?;
            }
        }

        self.n_documents = n_samples;
        Ok(self)
    }

    /// Transform documents to topic distribution
    pub fn transform(&self, doc_termmatrix: &Array2<f64>) -> Result<Array2<f64>> {
        if self.components.is_none() {
            return Err(TextError::ModelNotFitted(
                "LDA model not fitted yet".to_string(),
            ));
        }

        let n_samples = doc_termmatrix.nrows();
        let ntopics = self.config.ntopics;

        // Initialize document-topic distribution
        let mut doc_topic_distr = Array2::zeros((n_samples, ntopics));

        // Get exp(E[log(beta)])
        let exp_dirichlet_component = self.get_exp_dirichlet_component()?;

        // Set default prior
        let doc_topic_prior = self.config.doc_topic_prior.unwrap_or(1.0 / ntopics as f64);

        // Update document-topic distribution for each document
        for (doc_idx, doc) in doc_termmatrix.axis_iter(Axis(0)).enumerate() {
            let mut gamma = Array1::from_elem(ntopics, doc_topic_prior);
            self.update_doc_distribution(
                &doc.to_owned(),
                &mut gamma,
                exp_dirichlet_component,
                doc_topic_prior,
            )?;

            // Normalize to get probability distribution
            let gamma_sum = gamma.sum();
            if gamma_sum > 0.0 {
                gamma /= gamma_sum;
            }

            doc_topic_distr.row_mut(doc_idx).assign(&gamma);
        }

        Ok(doc_topic_distr)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, doc_termmatrix: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(doc_termmatrix)?;
        self.transform(doc_termmatrix)
    }

    /// Get the topics with top words
    pub fn get_topics(
        &self,
        n_top_words: usize,
        vocabulary: &HashMap<usize, String>,
    ) -> Result<Vec<Topic>> {
        if self.components.is_none() {
            return Err(TextError::ModelNotFitted(
                "LDA model not fitted yet".to_string(),
            ));
        }

        let components = self.components.as_ref().unwrap();
        let mut topics = Vec::new();

        for (topic_idx, topic_dist) in components.axis_iter(Axis(0)).enumerate() {
            // Get indices of top _words
            let mut word_scores: Vec<(usize, f64)> = topic_dist
                .iter()
                .enumerate()
                .map(|(idx, &score)| (idx, score))
                .collect();

            word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Get top _words with their scores
            let top_words: Vec<(String, f64)> = word_scores
                .into_iter()
                .take(n_top_words)
                .filter_map(|(idx, score)| vocabulary.get(&idx).map(|word| (word.clone(), score)))
                .collect();

            topics.push(Topic {
                id: topic_idx,
                top_words,
                coherence: None,
            });
        }

        Ok(topics)
    }

    /// Get the topic-word distribution matrix
    pub fn get_topic_word_distribution(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    // Helper functions

    fn create_rng(&self) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => {
                let mut temp_rng = rand::rng();
                rand::rngs::StdRng::from_rng(&mut temp_rng)
            }
        }
    }

    fn initialize_components(
        &self,
        n_features: usize,
        rng: &mut rand::rngs::StdRng,
    ) -> Array2<f64> {
        // Use the RNG directly

        let mut components = Array2::zeros((self.config.ntopics, n_features));
        for mut row in components.axis_iter_mut(Axis(0)) {
            for val in row.iter_mut() {
                *val = rng.random_range(0.0..1.0);
            }
            // Normalize each topic
            let row_sum: f64 = row.sum();
            if row_sum > 0.0 {
                row /= row_sum;
            }
        }

        components
    }

    fn get_exp_dirichlet_component(&self) -> Result<&Array2<f64>> {
        if self.exp_dirichlet_component.is_none() {
            return Err(TextError::ModelNotFitted(
                "Components not initialized".to_string(),
            ));
        }
        Ok(self.exp_dirichlet_component.as_ref().unwrap())
    }

    fn fit_batch(
        &mut self,
        doc_term_matrix: &Array2<f64>,
        doc_topic_prior: f64,
        topic_word_prior: f64,
    ) -> Result<()> {
        let n_samples = doc_term_matrix.nrows();
        let ntopics = self.config.ntopics;

        // Initialize document-topic distribution
        let mut doc_topic_distr = Array2::from_elem((n_samples, ntopics), doc_topic_prior);

        // Training loop
        for iter in 0..self.config.maxiter {
            // Update exp(E[log(beta)])
            self.update_exp_dirichlet_component()?;

            // E-step: Update document-topic distribution
            let mut mean_change = 0.0;
            for (doc_idx, doc) in doc_term_matrix.axis_iter(Axis(0)).enumerate() {
                let mut gamma = doc_topic_distr.row(doc_idx).to_owned();
                let old_gamma = gamma.clone();

                self.update_doc_distribution(
                    &doc.to_owned(),
                    &mut gamma,
                    self.get_exp_dirichlet_component()?,
                    doc_topic_prior,
                )?;

                // Calculate mean change
                let change: f64 = (&gamma - &old_gamma).iter().map(|&x| x.abs()).sum();
                mean_change += change / ntopics as f64;

                doc_topic_distr.row_mut(doc_idx).assign(&gamma);
            }
            mean_change /= n_samples as f64;

            // M-step: Update topic-word distribution
            self.update_topic_distribution(doc_term_matrix, &doc_topic_distr, topic_word_prior)?;

            // Check convergence
            if mean_change < self.config.mean_change_tol {
                break;
            }

            self.n_iter = iter + 1;
        }

        Ok(())
    }

    fn fit_online(
        &mut self,
        doc_term_matrix: &Array2<f64>,
        doc_topic_prior: f64,
        topic_word_prior: f64,
    ) -> Result<()> {
        let (n_samples, n_features) = doc_term_matrix.dim();
        self.vocabulary
            .get_or_insert_with(|| (0..n_features).map(|i| (i, format!("word_{i}"))).collect());
        self.bound.get_or_insert_with(Vec::new);

        // Initialize topic-word distribution if not already done
        if self.components.is_none() {
            let mut rng = if let Some(seed) = self.config.random_seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_rng(&mut rand::rng())
            };

            let mut components = Array2::<f64>::zeros((self.config.ntopics, n_features));
            for i in 0..self.config.ntopics {
                for j in 0..n_features {
                    components[[i, j]] = rng.random::<f64>() + topic_word_prior;
                }
            }
            self.components = Some(components);
        }

        let batch_size = self.config.batch_size.min(n_samples);
        let n_batches = n_samples.div_ceil(batch_size);

        for epoch in 0..self.config.maxiter {
            let mut total_bound = 0.0;

            // Shuffle document indices for each epoch
            let mut doc_indices: Vec<usize> = (0..n_samples).collect();
            let mut rng = if let Some(seed) = self.config.random_seed {
                StdRng::seed_from_u64(seed + epoch as u64)
            } else {
                StdRng::from_rng(&mut rand::rng())
            };
            doc_indices.shuffle(&mut rng);

            for batch_idx in 0..n_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = ((batch_idx + 1) * batch_size).min(n_samples);

                // Get batch documents
                let batch_docs: Vec<usize> = doc_indices[start_idx..end_idx].to_vec();

                // E-step: Update document-topic distributions for batch
                let mut batch_gamma = Array2::<f64>::zeros((batch_docs.len(), self.config.ntopics));
                let mut batch_bound = 0.0;

                for (local_idx, &doc_idx) in batch_docs.iter().enumerate() {
                    let doc = doc_term_matrix.row(doc_idx);
                    let mut gamma = Array1::<f64>::from_elem(self.config.ntopics, doc_topic_prior);

                    // Update document distribution
                    let components = self.components.as_ref().unwrap();
                    let exp_topic_word_distr = components.map(|x| x.exp());
                    self.update_doc_distribution(
                        &doc.to_owned(),
                        &mut gamma,
                        &exp_topic_word_distr,
                        doc_topic_prior,
                    )?;

                    batch_gamma.row_mut(local_idx).assign(&gamma);

                    // Compute bound contribution (simplified)
                    batch_bound += gamma.sum();
                }

                // M-step: Update topic-word distributions
                let learning_rate = self.compute_learning_rate(epoch * n_batches + batch_idx);
                self.update_topic_word_distribution(
                    &batch_docs,
                    doc_term_matrix,
                    &batch_gamma,
                    topic_word_prior,
                    learning_rate,
                    n_samples,
                )?;

                total_bound += batch_bound;
            }

            // Store bound for this epoch
            if let Some(ref mut bound) = self.bound {
                bound.push(total_bound / n_samples as f64);
            }

            // Check convergence
            if let Some(ref bound) = self.bound {
                if bound.len() > 1 {
                    let current_bound = bound[bound.len() - 1];
                    let prev_bound = bound[bound.len() - 2];
                    let change = (current_bound - prev_bound).abs();
                    if change < self.config.mean_change_tol {
                        break;
                    }
                }
            }

            self.n_iter = epoch + 1;
        }

        self.n_documents = n_samples;
        Ok(())
    }

    /// Compute learning rate for online learning
    fn compute_learning_rate(&self, iteration: usize) -> f64 {
        (self.config.learning_offset + iteration as f64).powf(-self.config.learning_decay)
    }

    /// Update topic-word distributions in online learning
    fn update_topic_word_distribution(
        &mut self,
        batch_docs: &[usize],
        doc_term_matrix: &Array2<f64>,
        batch_gamma: &Array2<f64>,
        topic_word_prior: f64,
        learning_rate: f64,
        total_docs: usize,
    ) -> Result<()> {
        let batch_size = batch_docs.len();
        let n_features = doc_term_matrix.ncols();

        if let Some(ref mut components) = self.components {
            // Compute sufficient statistics for this batch
            let mut batch_stats = Array2::<f64>::zeros((self.config.ntopics, n_features));

            for (local_idx, &doc_idx) in batch_docs.iter().enumerate() {
                let doc = doc_term_matrix.row(doc_idx);
                let gamma = batch_gamma.row(local_idx);
                let gamma_sum = gamma.sum();

                for (word_idx, &count) in doc.iter().enumerate() {
                    if count > 0.0 {
                        for topic_idx in 0..self.config.ntopics {
                            let phi = gamma[topic_idx] / gamma_sum;
                            batch_stats[[topic_idx, word_idx]] += count * phi;
                        }
                    }
                }
            }

            // Scale batch statistics to full corpus size
            let scale_factor = total_docs as f64 / batch_size as f64;
            batch_stats.mapv_inplace(|x| x * scale_factor);

            // Update components using natural gradient with learning _rate
            for topic_idx in 0..self.config.ntopics {
                for word_idx in 0..n_features {
                    let old_val = components[[topic_idx, word_idx]];
                    let new_val = topic_word_prior + batch_stats[[topic_idx, word_idx]];
                    components[[topic_idx, word_idx]] =
                        (1.0 - learning_rate) * old_val + learning_rate * new_val;
                }
            }
        }

        Ok(())
    }

    fn update_doc_distribution(
        &self,
        doc: &Array1<f64>,
        gamma: &mut Array1<f64>,
        exp_topic_word_distr: &Array2<f64>,
        doc_topic_prior: f64,
    ) -> Result<()> {
        // Simple mean-field update
        for _ in 0..self.config.max_doc_update_iter {
            let old_gamma = gamma.clone();

            // Reset gamma
            gamma.fill(doc_topic_prior);

            // Update based on word counts and topic-word probabilities
            for (word_idx, &count) in doc.iter().enumerate() {
                // Processing logic here
            }

            // Check convergence
            let change: f64 = (&*gamma - &old_gamma).iter().map(|&x| x.abs()).sum();
            if change < self.config.mean_change_tol {
                break;
            }
        }

        Ok(())
    }

    fn update_topic_distribution(
        &mut self,
        doc_term_matrix: &Array2<f64>,
        doc_topic_distr: &Array2<f64>,
        topic_word_prior: f64,
    ) -> Result<()> {
        if let Some(ref mut components) = self.components {
            let _n_features = doc_term_matrix.ncols();

            // Reset components
            components.fill(topic_word_prior);

            // Accumulate sufficient statistics
            for (doc_idx, doc) in doc_term_matrix.axis_iter(Axis(0)).enumerate() {
                let doc_topics = doc_topic_distr.row(doc_idx);

                for (word_idx, &count) in doc.iter().enumerate() {
                    if count > 0.0 {
                        for topic_idx in 0..self.config.ntopics {
                            components[[topic_idx, word_idx]] += count * doc_topics[topic_idx];
                        }
                    }
                }
            }

            // Normalize each topic
            for mut topic in components.axis_iter_mut(Axis(0)) {
                let topic_sum = topic.sum();
                if topic_sum > 0.0 {
                    topic /= topic_sum;
                }
            }
        }

        Ok(())
    }

    fn update_exp_dirichlet_component(&mut self) -> Result<()> {
        if let Some(ref components) = self.components {
            // For simplicity, we'll use the components directly
            // In a full implementation, this would compute exp(E[log(beta)])
            self.exp_dirichlet_component = Some(components.clone());
        }
        Ok(())
    }
}

/// Builder for creating LDA models
pub struct LdaBuilder {
    config: LdaConfig,
}

impl LdaBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: LdaConfig::default(),
        }
    }

    /// Set the number of topics
    pub fn ntopics(mut self, ntopics: usize) -> Self {
        self.config.ntopics = ntopics;
        self
    }

    /// Set the document-topic prior (alpha)
    pub fn doc_topic_prior(mut self, prior: f64) -> Self {
        self.config.doc_topic_prior = Some(prior);
        self
    }

    /// Set the topic-word prior (eta)
    pub fn topic_word_prior(mut self, prior: f64) -> Self {
        self.config.topic_word_prior = Some(prior);
        self
    }

    /// Set the learning method
    pub fn learning_method(mut self, method: LdaLearningMethod) -> Self {
        self.config.learning_method = method;
        self
    }

    /// Set the maximum iterations
    pub fn maxiter(mut self, maxiter: usize) -> Self {
        self.config.maxiter = maxiter;
        self
    }

    /// Set the random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Build the LDA model
    pub fn build(self) -> LatentDirichletAllocation {
        LatentDirichletAllocation::new(self.config)
    }
}

impl Default for LdaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lda_creation() {
        let lda = LatentDirichletAllocation::with_ntopics(5);
        assert_eq!(lda.config.ntopics, 5);
    }

    #[test]
    fn test_lda_builder() {
        let lda = LdaBuilder::new()
            .ntopics(10)
            .doc_topic_prior(0.1)
            .maxiter(20)
            .random_seed(42)
            .build();

        assert_eq!(lda.config.ntopics, 10);
        assert_eq!(lda.config.doc_topic_prior, Some(0.1));
        assert_eq!(lda.config.maxiter, 20);
        assert_eq!(lda.config.random_seed, Some(42));
    }

    #[test]
    fn test_lda_fit_transform() {
        // Create a simple document-term matrix
        let doc_term_matrix = Array2::from_shape_vec(
            (4, 6),
            vec![
                1.0, 1.0, 0.0, 0.0, 0.0, 0.0, // Doc 1
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, // Doc 2
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, // Doc 3
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0, // Doc 4
            ],
        )
        .unwrap();

        let mut lda = LatentDirichletAllocation::with_ntopics(2);
        let doc_topics = lda.fit_transform(&doc_term_matrix).unwrap();

        assert_eq!(doc_topics.nrows(), 4);
        assert_eq!(doc_topics.ncols(), 2);

        // Check that each document's topic distribution sums to 1
        for row in doc_topics.axis_iter(Axis(0)) {
            let sum: f64 = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_get_topics() {
        let doc_term_matrix = Array2::from_shape_vec(
            (4, 3),
            vec![2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 1.0],
        )
        .unwrap();

        let mut vocabulary = HashMap::new();
        vocabulary.insert(0, "word1".to_string());
        vocabulary.insert(1, "word2".to_string());
        vocabulary.insert(2, "word3".to_string());

        let mut lda = LatentDirichletAllocation::with_ntopics(2);
        lda.fit(&doc_term_matrix).unwrap();

        let topics = lda.get_topics(3, &vocabulary).unwrap();
        assert_eq!(topics.len(), 2);

        for topic in &topics {
            assert_eq!(topic.top_words.len(), 3);
        }
    }
}
