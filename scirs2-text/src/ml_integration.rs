//! Integration with machine learning modules
//!
//! This module provides utilities for integrating text processing
//! with machine learning pipelines.

use crate::classification::{TextClassificationPipeline, TextFeatureSelector};
use crate::embeddings::Word2Vec;
use crate::enhanced_vectorize::{EnhancedCountVectorizer, EnhancedTfidfVectorizer};
use crate::error::{Result, TextError};
use crate::multilingual::{Language, LanguageDetector};
use crate::sentiment::LexiconSentimentAnalyzer;
use crate::topic_modeling::LatentDirichletAllocation;
use crate::vectorize::{TfidfVectorizer, Vectorizer};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Feature extraction mode for machine learning
#[derive(Debug, Clone, Copy)]
pub enum FeatureExtractionMode {
    /// Bag of words representation
    BagOfWords,
    /// TF-IDF representation
    TfIdf,
    /// Word embeddings
    WordEmbeddings,
    /// Topic modeling features
    TopicModeling,
    /// Combined features
    Combined,
}

/// Text features for machine learning
#[derive(Debug, Clone)]
pub struct TextFeatures {
    /// The feature matrix
    pub features: Array2<f64>,
    /// Feature names (if available)
    pub feature_names: Option<Vec<String>>,
    /// Feature metadata
    pub metadata: HashMap<String, String>,
}

/// Machine learning text preprocessor
pub struct MLTextPreprocessor {
    /// Feature extraction mode
    mode: FeatureExtractionMode,
    /// TF-IDF vectorizer
    tfidf_vectorizer: Option<TfidfVectorizer>,
    /// Enhanced vectorizer
    enhanced_vectorizer: Option<EnhancedTfidfVectorizer>,
    /// Word embeddings model
    word_embeddings: Option<Word2Vec>,
    /// Topic model
    topic_model: Option<LatentDirichletAllocation>,
    /// Language detector
    language_detector: LanguageDetector,
    /// Sentiment analyzer
    sentiment_analyzer: LexiconSentimentAnalyzer,
    /// Feature selector
    feature_selector: Option<TextFeatureSelector>,
}

impl MLTextPreprocessor {
    /// Create a new ML text preprocessor
    pub fn new(mode: FeatureExtractionMode) -> Self {
        Self {
            mode,
            tfidf_vectorizer: None,
            enhanced_vectorizer: None,
            word_embeddings: None,
            topic_model: None,
            language_detector: LanguageDetector::new(),
            sentiment_analyzer: LexiconSentimentAnalyzer::with_basic_lexicon(),
            feature_selector: None,
        }
    }

    /// Configure TF-IDF parameters
    pub fn with_tfidf_params(
        mut self,
        _min_df: f64,
        _max_df: f64,
        _max_features: Option<usize>,
    ) -> Self {
        // For now, use standard EnhancedTfidfVectorizer
        let vectorizer = EnhancedTfidfVectorizer::new();
        self.enhanced_vectorizer = Some(vectorizer);
        self
    }

    /// Configure topic modeling
    pub fn with_topic_modeling(mut self, n_topics: usize) -> Self {
        self.topic_model = Some(LatentDirichletAllocation::with_n_topics(n_topics));
        self
    }

    /// Configure word embeddings
    pub fn with_word_embeddings(mut self, embeddings: Word2Vec) -> Self {
        self.word_embeddings = Some(embeddings);
        self
    }

    /// Configure feature selection
    pub fn with_feature_selection(mut self, max_features: usize) -> Self {
        self.feature_selector = TextFeatureSelector::new()
            .set_max_features(max_features as f64)
            .ok();
        self
    }

    /// Fit the preprocessor on training data
    pub fn fit(&mut self, texts: &[&str]) -> Result<()> {
        match self.mode {
            FeatureExtractionMode::BagOfWords | FeatureExtractionMode::TfIdf => {
                if let Some(ref mut vectorizer) = self.enhanced_vectorizer {
                    vectorizer.fit(texts)?;
                } else {
                    let mut vectorizer = TfidfVectorizer::default();
                    vectorizer.fit(texts)?;
                    self.tfidf_vectorizer = Some(vectorizer);
                }

                // Fit feature selector if configured
                if let Some(ref mut selector) = self.feature_selector {
                    let features = if let Some(ref vectorizer) = self.enhanced_vectorizer {
                        vectorizer.transform_batch(texts)?
                    } else if let Some(ref vectorizer) = self.tfidf_vectorizer {
                        vectorizer.transform_batch(texts)?
                    } else {
                        return Err(TextError::ModelNotFitted(
                            "Vectorizer not fitted".to_string(),
                        ));
                    };
                    selector.fit(&features)?;
                }
            }
            FeatureExtractionMode::TopicModeling => {
                // First create document-term matrix
                let mut vectorizer = EnhancedCountVectorizer::new();
                let doc_term_matrix = vectorizer.fit_transform(texts)?;

                // Then fit topic model
                if let Some(ref mut topic_model) = self.topic_model {
                    topic_model.fit(&doc_term_matrix)?;
                } else {
                    let mut topic_model = LatentDirichletAllocation::with_n_topics(10);
                    topic_model.fit(&doc_term_matrix)?;
                    self.topic_model = Some(topic_model);
                }
            }
            FeatureExtractionMode::WordEmbeddings => {
                // Word embeddings are typically pre-trained
                if self.word_embeddings.is_none() {
                    return Err(TextError::InvalidInput(
                        "Word embeddings must be provided for this mode".to_string(),
                    ));
                }
            }
            FeatureExtractionMode::Combined => {
                // Fit all components
                self.fit_combined(texts)?;
            }
        }

        Ok(())
    }

    /// Transform texts to feature matrix
    pub fn transform(&self, texts: &[&str]) -> Result<TextFeatures> {
        match self.mode {
            FeatureExtractionMode::BagOfWords | FeatureExtractionMode::TfIdf => {
                self.transform_vectorized(texts)
            }
            FeatureExtractionMode::TopicModeling => self.transform_topics(texts),
            FeatureExtractionMode::WordEmbeddings => self.transform_embeddings(texts),
            FeatureExtractionMode::Combined => self.transform_combined(texts),
        }
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, texts: &[&str]) -> Result<TextFeatures> {
        self.fit(texts)?;
        self.transform(texts)
    }

    // Helper methods

    fn fit_combined(&mut self, texts: &[&str]) -> Result<()> {
        // Fit TF-IDF
        let mut tfidf = TfidfVectorizer::default();
        tfidf.fit(texts)?;
        self.tfidf_vectorizer = Some(tfidf);

        // Fit topic model
        let mut count_vectorizer = EnhancedCountVectorizer::new();
        let doc_term_matrix = count_vectorizer.fit_transform(texts)?;
        let mut topic_model = LatentDirichletAllocation::with_n_topics(10);
        topic_model.fit(&doc_term_matrix)?;
        self.topic_model = Some(topic_model);

        Ok(())
    }

    fn transform_vectorized(&self, texts: &[&str]) -> Result<TextFeatures> {
        let mut features = if let Some(ref vectorizer) = self.enhanced_vectorizer {
            vectorizer.transform_batch(texts)?
        } else if let Some(ref vectorizer) = self.tfidf_vectorizer {
            vectorizer.transform_batch(texts)?
        } else {
            return Err(TextError::ModelNotFitted(
                "Vectorizer not fitted".to_string(),
            ));
        };

        // Apply feature selection if configured
        if let Some(ref selector) = self.feature_selector {
            features = selector.transform(&features)?;
        }

        Ok(TextFeatures {
            features,
            feature_names: None,
            metadata: HashMap::new(),
        })
    }

    fn transform_topics(&self, texts: &[&str]) -> Result<TextFeatures> {
        if let Some(ref topic_model) = self.topic_model {
            // First convert to document-term matrix
            let mut count_vectorizer = EnhancedCountVectorizer::new();
            let doc_term_matrix = count_vectorizer.fit_transform(texts)?;

            // Then get topic distributions
            let features = topic_model.transform(&doc_term_matrix)?;

            let mut metadata = HashMap::new();
            metadata.insert(
                "feature_type".to_string(),
                "topic_distributions".to_string(),
            );

            Ok(TextFeatures {
                features,
                feature_names: None,
                metadata,
            })
        } else {
            Err(TextError::ModelNotFitted(
                "Topic model not fitted".to_string(),
            ))
        }
    }

    fn transform_embeddings(&self, texts: &[&str]) -> Result<TextFeatures> {
        if let Some(ref embeddings) = self.word_embeddings {
            let mut doc_embeddings = Vec::new();

            for text in texts {
                // Simple average of word embeddings
                let words: Vec<&str> = text.split_whitespace().collect();
                let mut doc_embedding = Array1::zeros(embeddings.vector_size());
                let mut count = 0;

                for word in &words {
                    if let Ok(vec) = embeddings.get_word_vector(word) {
                        doc_embedding += &vec;
                        count += 1;
                    }
                }

                if count > 0 {
                    doc_embedding /= count as f64;
                }

                doc_embeddings.push(doc_embedding);
            }

            // Convert to 2D array
            let n_docs = doc_embeddings.len();
            let n_features = embeddings.vector_size();
            let mut features = Array2::zeros((n_docs, n_features));

            for (i, doc_vec) in doc_embeddings.iter().enumerate() {
                features.row_mut(i).assign(doc_vec);
            }

            Ok(TextFeatures {
                features,
                feature_names: None,
                metadata: HashMap::new(),
            })
        } else {
            Err(TextError::ModelNotFitted(
                "Word embeddings not loaded".to_string(),
            ))
        }
    }

    fn transform_combined(&self, texts: &[&str]) -> Result<TextFeatures> {
        let mut all_features = Vec::new();

        // TF-IDF features
        if let Ok(tfidf_features) = self.transform_vectorized(texts) {
            all_features.push(tfidf_features.features);
        }

        // Topic features
        if let Ok(topic_features) = self.transform_topics(texts) {
            all_features.push(topic_features.features);
        }

        // Sentiment features
        let sentiment_features = self.extract_sentiment_features(texts)?;
        all_features.push(sentiment_features);

        // Language features
        let language_features = self.extract_language_features(texts)?;
        all_features.push(language_features);

        // Concatenate all features
        let combined_features = self.concatenate_features(&all_features)?;

        Ok(TextFeatures {
            features: combined_features,
            feature_names: None,
            metadata: HashMap::new(),
        })
    }

    fn extract_sentiment_features(&self, texts: &[&str]) -> Result<Array2<f64>> {
        let mut features = Array2::zeros((texts.len(), 4));

        for (i, text) in texts.iter().enumerate() {
            let result = self.sentiment_analyzer.analyze(text)?;
            features[[i, 0]] = result.score;
            features[[i, 1]] = result.confidence;
            features[[i, 2]] = result.word_counts.positive_words as f64;
            features[[i, 3]] = result.word_counts.negative_words as f64;
        }

        Ok(features)
    }

    fn extract_language_features(&self, texts: &[&str]) -> Result<Array2<f64>> {
        let mut features = Array2::zeros((texts.len(), 2));

        for (i, text) in texts.iter().enumerate() {
            let result = self.language_detector.detect(text)?;
            // Encode language as numeric feature
            features[[i, 0]] = match result.language {
                Language::English => 1.0,
                Language::Spanish => 2.0,
                Language::French => 3.0,
                Language::German => 4.0,
                _ => 0.0,
            };
            features[[i, 1]] = result.confidence;
        }

        Ok(features)
    }

    fn concatenate_features(&self, feature_arrays: &[Array2<f64>]) -> Result<Array2<f64>> {
        if feature_arrays.is_empty() {
            return Err(TextError::InvalidInput(
                "No features to concatenate".to_string(),
            ));
        }

        let n_samples = feature_arrays[0].nrows();
        let total_features: usize = feature_arrays.iter().map(|arr| arr.ncols()).sum();

        let mut combined = Array2::zeros((n_samples, total_features));
        let mut col_offset = 0;

        for array in feature_arrays {
            let n_cols = array.ncols();
            for i in 0..n_samples {
                for j in 0..n_cols {
                    combined[[i, col_offset + j]] = array[[i, j]];
                }
            }
            col_offset += n_cols;
        }

        Ok(combined)
    }
}

/// Text preprocessing pipeline for ML
pub struct TextMLPipeline {
    /// Preprocessing steps
    preprocessor: MLTextPreprocessor,
    /// Classification pipeline (if used)
    classification_pipeline: Option<TextClassificationPipeline>,
}

impl TextMLPipeline {
    /// Create a new pipeline with default TF-IDF features
    pub fn new() -> Self {
        Self {
            preprocessor: MLTextPreprocessor::new(FeatureExtractionMode::TfIdf),
            classification_pipeline: None,
        }
    }

    /// Create a pipeline with specific feature extraction mode
    pub fn with_mode(mode: FeatureExtractionMode) -> Self {
        Self {
            preprocessor: MLTextPreprocessor::new(mode),
            classification_pipeline: None,
        }
    }

    /// Add classification preprocessing
    pub fn with_classification(mut self) -> Self {
        self.classification_pipeline = Some(TextClassificationPipeline::with_tfidf());
        self
    }

    /// Configure the preprocessor
    pub fn configure_preprocessor<F>(mut self, f: F) -> Self
    where
        F: FnOnce(MLTextPreprocessor) -> MLTextPreprocessor,
    {
        self.preprocessor = f(self.preprocessor);
        self
    }

    /// Process texts for machine learning
    pub fn process(&mut self, texts: &[&str]) -> Result<TextFeatures> {
        self.preprocessor.fit_transform(texts)
    }
}

impl Default for TextMLPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch text processor for large datasets
pub struct BatchTextProcessor {
    /// Batch size
    batch_size: usize,
    /// Preprocessor
    preprocessor: MLTextPreprocessor,
}

impl BatchTextProcessor {
    /// Create a new batch processor
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            preprocessor: MLTextPreprocessor::new(FeatureExtractionMode::TfIdf),
        }
    }

    /// Process texts in batches
    pub fn process_batches(&mut self, texts: &[&str]) -> Result<Vec<TextFeatures>> {
        let mut results = Vec::new();

        // First fit on all data
        self.preprocessor.fit(texts)?;

        // Then transform in batches
        for chunk in texts.chunks(self.batch_size) {
            let batch_features = self.preprocessor.transform(chunk)?;
            results.push(batch_features);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_preprocessor_tfidf() {
        let mut preprocessor = MLTextPreprocessor::new(FeatureExtractionMode::TfIdf);
        let texts = vec![
            "This is a test document",
            "Another test document here",
            "Machine learning is great",
        ];

        let features = preprocessor.fit_transform(&texts).unwrap();
        assert_eq!(features.features.nrows(), 3);
        assert!(features.features.ncols() > 0);
    }

    #[test]
    fn test_feature_extraction_modes() {
        let modes = vec![
            FeatureExtractionMode::BagOfWords,
            FeatureExtractionMode::TfIdf,
            FeatureExtractionMode::TopicModeling,
        ];

        for mode in modes {
            let preprocessor = MLTextPreprocessor::new(mode);
            assert!(matches!(preprocessor.mode, _));
        }
    }

    #[test]
    fn test_text_ml_pipeline() {
        let mut pipeline =
            TextMLPipeline::new().configure_preprocessor(|p| p.with_feature_selection(10));

        let texts = vec![
            "Text processing example",
            "Machine learning pipeline",
            "Feature extraction test",
        ];

        let features = pipeline.process(&texts).unwrap();
        assert_eq!(features.features.nrows(), 3);
    }

    #[test]
    fn test_batch_processor() {
        let mut processor = BatchTextProcessor::new(2);
        let texts = vec![
            "First batch text",
            "Second batch text",
            "Third batch text",
            "Fourth batch text",
        ];

        let batches = processor.process_batches(&texts).unwrap();
        assert_eq!(batches.len(), 2); // 4 texts with batch size 2
    }
}
