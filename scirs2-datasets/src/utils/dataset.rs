//! Core Dataset structure and basic methods
//!
//! This module provides the main Dataset struct used throughout the datasets
//! crate, along with its core methods for creation, metadata management, and
//! basic properties.

use crate::utils::serialization;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a dataset with features, optional targets, and metadata
///
/// The Dataset struct is the core data structure for managing machine learning
/// datasets. It stores the feature matrix, optional target values, and rich
/// metadata including feature names, descriptions, and arbitrary key-value pairs.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_datasets::utils::Dataset;
///
/// let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let dataset = Dataset::new(data, None)
///     .with_feature_names(vec!["feature1".to_string(), "feature2".to_string()])
///     .with_description("Sample dataset".to_string());
///
/// assert_eq!(dataset.n_samples(), 3);
/// assert_eq!(dataset.n_features(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Features/data matrix (n_samples, n_features)
    #[serde(
        serialize_with = "serialization::serialize_array2",
        deserialize_with = "serialization::deserialize_array2"
    )]
    pub data: Array2<f64>,

    /// Optional target values
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<Array1<f64>>,

    /// Optional target names for classification problems
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_names: Option<Vec<String>>,

    /// Optional feature names
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_names: Option<Vec<String>>,

    /// Optional descriptions for each feature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_descriptions: Option<Vec<String>>,

    /// Optional dataset description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Optional dataset metadata
    pub metadata: HashMap<String, String>,
}

impl Dataset {
    /// Create a new dataset with the given data and target
    ///
    /// # Arguments
    ///
    /// * `data` - The feature matrix (n_samples, n_features)
    /// * `target` - Optional target values (n_samples,)
    ///
    /// # Returns
    ///
    /// A new Dataset instance with empty metadata
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::{Array1, Array2};
    /// use scirs2_datasets::utils::Dataset;
    ///
    /// let data = Array2::zeros((100, 5));
    /// let target = Some(Array1::zeros(100));
    /// let dataset = Dataset::new(data, target);
    /// ```
    pub fn new(data: Array2<f64>, target: Option<Array1<f64>>) -> Self {
        Dataset {
            data,
            target,
            target_names: None,
            feature_names: None,
            feature_descriptions: None,
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Add target names to the dataset (builder pattern)
    ///
    /// # Arguments
    ///
    /// * `target_names` - Vector of target class names
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_target_names(mut self, target_names: Vec<String>) -> Self {
        self.target_names = Some(target_names);
        self
    }

    /// Add feature names to the dataset (builder pattern)
    ///
    /// # Arguments
    ///
    /// * `feature_names` - Vector of feature names
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_feature_names(mut self, feature_names: Vec<String>) -> Self {
        self.feature_names = Some(feature_names);
        self
    }

    /// Add feature descriptions to the dataset (builder pattern)
    ///
    /// # Arguments
    ///
    /// * `feature_descriptions` - Vector of feature descriptions
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_feature_descriptions(mut self, feature_descriptions: Vec<String>) -> Self {
        self.feature_descriptions = Some(feature_descriptions);
        self
    }

    /// Add a description to the dataset (builder pattern)
    ///
    /// # Arguments
    ///
    /// * `description` - Dataset description
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add metadata to the dataset (builder pattern)
    ///
    /// # Arguments
    ///
    /// * `key` - Metadata key
    /// * `value` - Metadata value
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get the number of samples in the dataset
    ///
    /// # Returns
    ///
    /// Number of samples (rows) in the dataset
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Get the number of features in the dataset
    ///
    /// # Returns
    ///
    /// Number of features (columns) in the dataset
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Get dataset shape as (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Tuple of (n_samples, n_features)
    pub fn shape(&self) -> (usize, usize) {
        (self.n_samples(), self.n_features())
    }

    /// Check if the dataset has target values
    ///
    /// # Returns
    ///
    /// True if target values are present, false otherwise
    pub fn has_target(&self) -> bool {
        self.target.is_some()
    }

    /// Get a reference to the feature names if available
    ///
    /// # Returns
    ///
    /// Optional reference to feature names vector
    pub fn feature_names(&self) -> Option<&Vec<String>> {
        self.feature_names.as_ref()
    }

    /// Get a reference to the target names if available
    ///
    /// # Returns
    ///
    /// Optional reference to target names vector  
    pub fn target_names(&self) -> Option<&Vec<String>> {
        self.target_names.as_ref()
    }

    /// Get a reference to the dataset description if available
    ///
    /// # Returns
    ///
    /// Optional reference to dataset description
    pub fn description(&self) -> Option<&String> {
        self.description.as_ref()
    }

    /// Get a reference to the metadata
    ///
    /// # Returns
    ///
    /// Reference to metadata HashMap
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Add or update a metadata entry
    ///
    /// # Arguments
    ///
    /// * `key` - Metadata key
    /// * `value` - Metadata value
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get a metadata value by key
    ///
    /// # Arguments
    ///
    /// * `key` - Metadata key to lookup
    ///
    /// # Returns
    ///
    /// Optional reference to the metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dataset_creation() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let target = Some(array![0.0, 1.0, 0.0]);

        let dataset = Dataset::new(data.clone(), target.clone());

        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(dataset.shape(), (3, 2));
        assert!(dataset.has_target());
        assert_eq!(dataset.data, data);
        assert_eq!(dataset.target, target);
    }

    #[test]
    fn test_dataset_builder_pattern() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];

        let dataset = Dataset::new(data, None)
            .with_feature_names(vec!["feat1".to_string(), "feat2".to_string()])
            .with_description("Test dataset".to_string())
            .with_metadata("version", "1.0")
            .with_metadata("author", "test");

        assert_eq!(dataset.feature_names().unwrap().len(), 2);
        assert_eq!(dataset.description().unwrap(), "Test dataset");
        assert_eq!(dataset.get_metadata("version").unwrap(), "1.0");
        assert_eq!(dataset.get_metadata("author").unwrap(), "test");
    }

    #[test]
    fn test_dataset_without_target() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let dataset = Dataset::new(data, None);

        assert!(!dataset.has_target());
        assert!(dataset.target.is_none());
    }

    #[test]
    fn test_metadata_operations() {
        let data = array![[1.0, 2.0]];
        let mut dataset = Dataset::new(data, None);

        dataset.set_metadata("key1", "value1");
        dataset.set_metadata("key2", "value2");

        assert_eq!(dataset.get_metadata("key1").unwrap(), "value1");
        assert_eq!(dataset.get_metadata("key2").unwrap(), "value2");
        assert!(dataset.get_metadata("nonexistent").is_none());

        // Update existing key
        dataset.set_metadata("key1", "updated_value");
        assert_eq!(dataset.get_metadata("key1").unwrap(), "updated_value");
    }
}
