//! Topic modeling functionality
//!
//! This module provides topic modeling algorithms including
//! Latent Dirichlet Allocation (LDA).

use crate::error::{Result, TextError};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
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
    pub n_topics: usize,
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
    pub max_iter: usize,
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
            n_topics: 10,
            doc_topic_prior: None,  // Will be set to 1/n_topics
            topic_word_prior: None, // Will be set to 1/n_topics
            learning_method: LdaLearningMethod::Batch,
            learning_decay: 0.7,
            learning_offset: 10.0,
            max_iter: 10,
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
    bound: Option<f64>,
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
    pub fn with_n_topics(n_topics: usize) -> Self {
        let config = LdaConfig {
            n_topics,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Fit the LDA model on a document-term matrix
    pub fn fit(&mut self, doc_term_matrix: &Array2<f64>) -> Result<&mut Self> {
        if doc_term_matrix.nrows() == 0 || doc_term_matrix.ncols() == 0 {
            return Err(TextError::InvalidInput(
                "Document-term matrix cannot be empty".to_string(),
            ));
        }

        let n_samples = doc_term_matrix.nrows();
        let n_features = doc_term_matrix.ncols();

        // Set default priors if not provided
        let doc_topic_prior = self
            .config
            .doc_topic_prior
            .unwrap_or(1.0 / self.config.n_topics as f64);
        let topic_word_prior = self
            .config
            .topic_word_prior
            .unwrap_or(1.0 / self.config.n_topics as f64);

        // Initialize topic-word distribution randomly
        let mut rng = self.create_rng();
        self.components = Some(self.initialize_components(n_features, &mut rng));

        // Perform training based on learning method
        match self.config.learning_method {
            LdaLearningMethod::Batch => {
                self.fit_batch(doc_term_matrix, doc_topic_prior, topic_word_prior)?;
            }
            LdaLearningMethod::Online => {
                self.fit_online(doc_term_matrix, doc_topic_prior, topic_word_prior)?;
            }
        }

        self.n_documents = n_samples;
        Ok(self)
    }

    /// Transform documents to topic distribution
    pub fn transform(&self, doc_term_matrix: &Array2<f64>) -> Result<Array2<f64>> {
        if self.components.is_none() {
            return Err(TextError::ModelNotFitted(
                "LDA model not fitted yet".to_string(),
            ));
        }

        let n_samples = doc_term_matrix.nrows();
        let n_topics = self.config.n_topics;

        // Initialize document-topic distribution
        let mut doc_topic_distr = Array2::zeros((n_samples, n_topics));

        // Get exp(E[log(beta)])
        let exp_dirichlet_component = self.get_exp_dirichlet_component()?;

        // Set default prior
        let doc_topic_prior = self.config.doc_topic_prior.unwrap_or(1.0 / n_topics as f64);

        // Update document-topic distribution for each document
        for (doc_idx, doc) in doc_term_matrix.axis_iter(Axis(0)).enumerate() {
            let mut gamma = Array1::from_elem(n_topics, doc_topic_prior);
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
    pub fn fit_transform(&mut self, doc_term_matrix: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(doc_term_matrix)?;
        self.transform(doc_term_matrix)
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
            // Get indices of top words
            let mut word_scores: Vec<(usize, f64)> = topic_dist
                .iter()
                .enumerate()
                .map(|(idx, &score)| (idx, score))
                .collect();

            word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Get top words with their scores
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

        let mut components = Array2::zeros((self.config.n_topics, n_features));
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
        let n_topics = self.config.n_topics;

        // Initialize document-topic distribution
        let mut doc_topic_distr = Array2::from_elem((n_samples, n_topics), doc_topic_prior);

        // Training loop
        for iter in 0..self.config.max_iter {
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
                mean_change += change / n_topics as f64;

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
        // Online learning implementation would go here
        // For now, we'll use batch learning as a placeholder
        self.fit_batch(doc_term_matrix, doc_topic_prior, topic_word_prior)
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
                if count > 0.0 {
                    for topic_idx in 0..self.config.n_topics {
                        gamma[topic_idx] += count * exp_topic_word_distr[[topic_idx, word_idx]];
                    }
                }
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
                        for topic_idx in 0..self.config.n_topics {
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
    pub fn n_topics(mut self, n_topics: usize) -> Self {
        self.config.n_topics = n_topics;
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
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
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
        let lda = LatentDirichletAllocation::with_n_topics(5);
        assert_eq!(lda.config.n_topics, 5);
    }

    #[test]
    fn test_lda_builder() {
        let lda = LdaBuilder::new()
            .n_topics(10)
            .doc_topic_prior(0.1)
            .max_iter(20)
            .random_seed(42)
            .build();

        assert_eq!(lda.config.n_topics, 10);
        assert_eq!(lda.config.doc_topic_prior, Some(0.1));
        assert_eq!(lda.config.max_iter, 20);
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

        let mut lda = LatentDirichletAllocation::with_n_topics(2);
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

        let mut lda = LatentDirichletAllocation::with_n_topics(2);
        lda.fit(&doc_term_matrix).unwrap();

        let topics = lda.get_topics(3, &vocabulary).unwrap();
        assert_eq!(topics.len(), 2);

        for topic in &topics {
            assert_eq!(topic.top_words.len(), 3);
        }
    }
}
