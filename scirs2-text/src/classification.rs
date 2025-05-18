//! Text classification functionality
//!
//! This module provides tools for text classification including
//! metrics, feature selection, and classification pipelines.

use crate::error::{Result, TextError};
use crate::vectorize::{TfidfVectorizer, Vectorizer};
use ndarray::{Array2, Axis};
use rand::prelude::*;
use rand::SeedableRng;

/// Text feature selector
///
/// This utility selects features based on document frequency.
/// It can filter out features that are too rare or too common.
#[derive(Debug, Clone)]
pub struct TextFeatureSelector {
    /// Minimum document frequency (fraction or count)
    min_df: f64,
    /// Maximum document frequency (fraction or count)
    max_df: f64,
    /// Whether to use raw counts instead of fractions
    use_counts: bool,
    /// Selected feature indices
    selected_features: Option<Vec<usize>>,
}

impl Default for TextFeatureSelector {
    fn default() -> Self {
        Self {
            min_df: 0.0,
            max_df: 1.0,
            use_counts: false,
            selected_features: None,
        }
    }
}

impl TextFeatureSelector {
    /// Create a new feature selector
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum document frequency
    pub fn set_min_df(mut self, min_df: f64) -> Result<Self> {
        if min_df < 0.0 {
            return Err(TextError::InvalidInput(
                "min_df must be non-negative".to_string(),
            ));
        }
        self.min_df = min_df;
        Ok(self)
    }

    /// Set maximum document frequency
    pub fn set_max_df(mut self, max_df: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&max_df) {
            return Err(TextError::InvalidInput(
                "max_df must be between 0 and 1 for fractions".to_string(),
            ));
        }
        self.max_df = max_df;
        Ok(self)
    }

    /// Set maximum document frequency (alias for set_max_df)
    pub fn set_max_features(self, max_features: f64) -> Result<Self> {
        self.set_max_df(max_features)
    }

    /// Set to use absolute counts instead of fractions
    pub fn use_counts(mut self, use_counts: bool) -> Self {
        self.use_counts = use_counts;
        self
    }

    /// Fit the feature selector to data
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut document_frequencies = vec![0; n_features];

        // Count document frequency for each feature
        for sample in x.axis_iter(Axis(0)) {
            for (feature_idx, &value) in sample.iter().enumerate() {
                if value > 0.0 {
                    document_frequencies[feature_idx] += 1;
                }
            }
        }

        // Calculate min and max document counts
        let min_count = if self.use_counts {
            self.min_df
        } else {
            self.min_df * n_samples as f64
        };

        let max_count = if self.use_counts {
            self.max_df
        } else {
            self.max_df * n_samples as f64
        };

        // Select features based on document frequency
        let mut selected_features = Vec::new();
        for (idx, &df) in document_frequencies.iter().enumerate() {
            let df_f64 = df as f64;
            if df_f64 >= min_count && df_f64 <= max_count {
                selected_features.push(idx);
            }
        }

        self.selected_features = Some(selected_features);
        Ok(self)
    }

    /// Transform data using selected features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected_features = self
            .selected_features
            .as_ref()
            .ok_or_else(|| TextError::ModelNotFitted("Feature selector not fitted".to_string()))?;

        if selected_features.is_empty() {
            return Err(TextError::InvalidInput(
                "No features selected. Try adjusting min_df and max_df".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_selected = selected_features.len();

        let mut result = Array2::zeros((n_samples, n_selected));

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            for (j, &feature_idx) in selected_features.iter().enumerate() {
                result[[i, j]] = row[feature_idx];
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Get selected feature indices
    pub fn get_selected_features(&self) -> Option<&Vec<usize>> {
        self.selected_features.as_ref()
    }
}

/// Text classification metrics
#[derive(Debug, Clone)]
pub struct TextClassificationMetrics;

impl Default for TextClassificationMetrics {
    fn default() -> Self {
        Self
    }
}

impl TextClassificationMetrics {
    /// Create a new metrics calculator
    pub fn new() -> Self {
        Self
    }

    /// Calculate precision score
    pub fn precision<T>(
        &self,
        predictions: &[T],
        true_labels: &[T],
        class_idx: Option<T>,
    ) -> Result<f64>
    where
        T: PartialEq + Copy + Default,
    {
        let positive_class = class_idx.unwrap_or_default();

        if predictions.len() != true_labels.len() {
            return Err(TextError::InvalidInput(
                "Predictions and labels must have the same length".to_string(),
            ));
        }

        let mut true_positives = 0;
        let mut predicted_positives = 0;

        for i in 0..predictions.len() {
            if predictions[i] == positive_class {
                predicted_positives += 1;
                if true_labels[i] == positive_class {
                    true_positives += 1;
                }
            }
        }

        if predicted_positives == 0 {
            return Ok(0.0);
        }

        Ok(true_positives as f64 / predicted_positives as f64)
    }

    /// Calculate recall score
    pub fn recall<T>(
        &self,
        predictions: &[T],
        true_labels: &[T],
        class_idx: Option<T>,
    ) -> Result<f64>
    where
        T: PartialEq + Copy + Default,
    {
        let positive_class = class_idx.unwrap_or_default();

        if predictions.len() != true_labels.len() {
            return Err(TextError::InvalidInput(
                "Predictions and labels must have the same length".to_string(),
            ));
        }

        let mut true_positives = 0;
        let mut actual_positives = 0;

        for i in 0..predictions.len() {
            if true_labels[i] == positive_class {
                actual_positives += 1;
                if predictions[i] == positive_class {
                    true_positives += 1;
                }
            }
        }

        if actual_positives == 0 {
            return Ok(0.0);
        }

        Ok(true_positives as f64 / actual_positives as f64)
    }

    /// Calculate F1 score
    pub fn f1_score<T>(
        &self,
        predictions: &[T],
        true_labels: &[T],
        class_idx: Option<T>,
    ) -> Result<f64>
    where
        T: PartialEq + Copy + Default,
    {
        let precision = self.precision(predictions, true_labels, class_idx)?;
        let recall = self.recall(predictions, true_labels, class_idx)?;

        if precision + recall == 0.0 {
            return Ok(0.0);
        }

        Ok(2.0 * precision * recall / (precision + recall))
    }

    /// Calculate accuracy from predictions and true labels
    pub fn accuracy<T>(&self, predictions: &[T], true_labels: &[T]) -> Result<f64>
    where
        T: PartialEq,
    {
        if predictions.len() != true_labels.len() {
            return Err(TextError::InvalidInput(
                "Predictions and labels must have the same length".to_string(),
            ));
        }

        if predictions.is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot calculate accuracy for empty arrays".to_string(),
            ));
        }

        let correct = predictions
            .iter()
            .zip(true_labels.iter())
            .filter(|(pred, true_label)| pred == true_label)
            .count();

        Ok(correct as f64 / predictions.len() as f64)
    }

    /// Calculate precision, recall, and F1 score for binary classification
    pub fn binary_metrics<T>(&self, predictions: &[T], true_labels: &[T]) -> Result<(f64, f64, f64)>
    where
        T: PartialEq + Copy + Default + PartialEq<usize>,
    {
        if predictions.len() != true_labels.len() {
            return Err(TextError::InvalidInput(
                "Predictions and labels must have the same length".to_string(),
            ));
        }

        // Count true positives, false positives, false negatives
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_ = 0;

        for (pred, true_label) in predictions.iter().zip(true_labels.iter()) {
            if *pred == 1 && *true_label == 1 {
                tp += 1;
            } else if *pred == 1 && *true_label == 0 {
                fp += 1;
            } else if *pred == 0 && *true_label == 1 {
                fn_ += 1;
            }
        }

        // Calculate precision, recall, F1
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Ok((precision, recall, f1))
    }
}

/// Text classification dataset
#[derive(Debug, Clone)]
pub struct TextDataset {
    /// The text samples
    pub texts: Vec<String>,
    /// The labels for each text
    pub labels: Vec<String>,
    /// Index mapping for labels
    label_index: Option<std::collections::HashMap<String, usize>>,
}

impl TextDataset {
    /// Create a new text dataset
    pub fn new(texts: Vec<String>, labels: Vec<String>) -> Result<Self> {
        if texts.len() != labels.len() {
            return Err(TextError::InvalidInput(
                "Texts and labels must have the same length".to_string(),
            ));
        }

        Ok(Self {
            texts,
            labels,
            label_index: None,
        })
    }

    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Get the unique labels in the dataset
    pub fn unique_labels(&self) -> Vec<String> {
        let mut unique = std::collections::HashSet::new();
        for label in &self.labels {
            unique.insert(label.clone());
        }
        unique.into_iter().collect()
    }

    /// Build a label index mapping
    pub fn build_label_index(&mut self) -> Result<&mut Self> {
        let mut index = std::collections::HashMap::new();
        let unique_labels = self.unique_labels();

        for (i, label) in unique_labels.iter().enumerate() {
            index.insert(label.clone(), i);
        }

        self.label_index = Some(index);
        Ok(self)
    }

    /// Get label indices
    pub fn get_label_indices(&self) -> Result<Vec<usize>> {
        let index = self
            .label_index
            .as_ref()
            .ok_or_else(|| TextError::ModelNotFitted("Label index not built".to_string()))?;

        self.labels
            .iter()
            .map(|label| {
                index
                    .get(label)
                    .copied()
                    .ok_or_else(|| TextError::InvalidInput(format!("Unknown label: {}", label)))
            })
            .collect()
    }

    /// Split the dataset into train and test sets
    pub fn train_test_split(
        &self,
        test_size: f64,
        random_seed: Option<u64>,
    ) -> Result<(Self, Self)> {
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err(TextError::InvalidInput(
                "test_size must be between 0 and 1".to_string(),
            ));
        }

        if self.is_empty() {
            return Err(TextError::InvalidInput("Dataset is empty".to_string()));
        }

        // Create indices and shuffle them
        let mut indices: Vec<usize> = (0..self.len()).collect();

        // Shuffle indices
        if let Some(seed) = random_seed {
            // Use deterministic RNG with seed
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        } else {
            // Use standard thread_rng
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }

        // Split indices
        let test_size = (self.len() as f64 * test_size).ceil() as usize;
        let test_indices = indices[0..test_size].to_vec();
        let train_indices = indices[test_size..].to_vec();

        // Create datasets
        let train_texts = train_indices
            .iter()
            .map(|&i| self.texts[i].clone())
            .collect();
        let train_labels = train_indices
            .iter()
            .map(|&i| self.labels[i].clone())
            .collect();

        let test_texts = test_indices
            .iter()
            .map(|&i| self.texts[i].clone())
            .collect();
        let test_labels = test_indices
            .iter()
            .map(|&i| self.labels[i].clone())
            .collect();

        let mut train_dataset = Self::new(train_texts, train_labels)?;
        let mut test_dataset = Self::new(test_texts, test_labels)?;

        // If we have a label index, build it for the split datasets
        if self.label_index.is_some() {
            train_dataset.build_label_index()?;
            test_dataset.build_label_index()?;
        }

        Ok((train_dataset, test_dataset))
    }
}

/// Pipeline for text classification
pub struct TextClassificationPipeline {
    /// The vectorizer to use
    vectorizer: TfidfVectorizer,
    /// Optional feature selector
    feature_selector: Option<TextFeatureSelector>,
}

impl TextClassificationPipeline {
    /// Create a new pipeline with a default TF-IDF vectorizer
    pub fn with_tfidf() -> Self {
        Self::new(TfidfVectorizer::default())
    }

    /// Create a new pipeline with the given vectorizer
    pub fn new(vectorizer: TfidfVectorizer) -> Self {
        Self {
            vectorizer,
            feature_selector: None,
        }
    }

    /// Add a feature selector to the pipeline
    pub fn with_feature_selector(mut self, selector: TextFeatureSelector) -> Self {
        self.feature_selector = Some(selector);
        self
    }

    /// Fit the pipeline to training data
    pub fn fit(&mut self, dataset: &TextDataset) -> Result<&mut Self> {
        let texts: Vec<&str> = dataset.texts.iter().map(AsRef::as_ref).collect();
        self.vectorizer.fit(&texts)?;

        Ok(self)
    }

    /// Transform text data using the pipeline
    pub fn transform(&self, dataset: &TextDataset) -> Result<Array2<f64>> {
        let texts: Vec<&str> = dataset.texts.iter().map(AsRef::as_ref).collect();
        let mut features = self.vectorizer.transform_batch(&texts)?;

        if let Some(selector) = &self.feature_selector {
            features = selector.transform(&features)?;
        }

        Ok(features)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, dataset: &TextDataset) -> Result<Array2<f64>> {
        self.fit(dataset)?;
        self.transform(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_dataset() {
        let texts = vec![
            "This is document 1".to_string(),
            "Another document".to_string(),
            "A third document".to_string(),
        ];
        let labels = vec!["A".to_string(), "B".to_string(), "A".to_string()];

        let mut dataset = TextDataset::new(texts, labels).unwrap();

        // Create a manual label index to explicitly control the index values
        let mut label_index = std::collections::HashMap::new();
        label_index.insert("A".to_string(), 0);
        label_index.insert("B".to_string(), 1);
        dataset.label_index = Some(label_index);

        let label_indices = dataset.get_label_indices().unwrap();

        // Now we know exactly what the indices should be
        assert_eq!(label_indices[0], 0); // First label "A" should be index 0
        assert_eq!(label_indices[1], 1); // Second label "B" should be index 1
        assert_eq!(label_indices[2], 0); // Third label "A" should be index 0

        let unique_labels = dataset.unique_labels();
        assert_eq!(unique_labels.len(), 2);
        assert!(unique_labels.contains(&"A".to_string()));
        assert!(unique_labels.contains(&"B".to_string()));
    }

    #[test]
    fn test_train_test_split() {
        let texts = (0..10).map(|i| format!("Text {}", i)).collect();
        let labels = (0..10).map(|_| "A".to_string()).collect();

        let dataset = TextDataset::new(texts, labels).unwrap();
        let (train, test) = dataset.train_test_split(0.3, Some(42)).unwrap();

        assert_eq!(train.len(), 7);
        assert_eq!(test.len(), 3);
    }

    #[test]
    fn test_feature_selector() {
        let mut features = Array2::zeros((5, 3));
        // Feature 0: appears in doc 0, 1, 2 (60% of docs)
        features[[0, 0]] = 1.0;
        features[[1, 0]] = 1.0;
        features[[2, 0]] = 1.0;

        // Feature 1: appears in all docs (100% of docs)
        for i in 0..5 {
            features[[i, 1]] = 1.0;
        }

        // Feature 2: appears in doc 0 only (20% of docs)
        features[[0, 2]] = 1.0;

        let mut selector = TextFeatureSelector::new()
            .set_min_df(0.25)
            .unwrap()
            .set_max_df(0.75)
            .unwrap();

        let filtered = selector.fit_transform(&features).unwrap();
        assert_eq!(filtered.ncols(), 1); // Only feature 0 should pass the filters
    }

    #[test]
    fn test_classification_metrics() {
        let predictions = vec![1_usize, 0, 1, 1, 0];
        let true_labels = vec![1_usize, 0, 1, 0, 0];

        let metrics = TextClassificationMetrics::new();
        let accuracy = metrics.accuracy(&predictions, &true_labels).unwrap();
        assert_eq!(accuracy, 0.8);

        let (precision, recall, f1) = metrics.binary_metrics(&predictions, &true_labels).unwrap();
        assert!((precision - 0.667).abs() < 0.001);
        assert_eq!(recall, 1.0);
        assert!((f1 - 0.8).abs() < 0.001);
    }
}
