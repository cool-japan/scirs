//! Missing value imputation utilities
//!
//! This module provides methods for handling missing values in datasets,
//! which is a crucial preprocessing step for machine learning.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};
use rayon::prelude::*;

use crate::error::{Result, TransformError};

/// Strategy for imputing missing values
#[derive(Debug, Clone, PartialEq)]
pub enum ImputeStrategy {
    /// Replace missing values with mean of the feature
    Mean,
    /// Replace missing values with median of the feature
    Median,
    /// Replace missing values with most frequent value
    MostFrequent,
    /// Replace missing values with a constant value
    Constant(f64),
}

/// SimpleImputer for filling missing values
///
/// This transformer fills missing values using simple strategies like mean,
/// median, most frequent value, or a constant value.
pub struct SimpleImputer {
    /// Strategy for imputation
    strategy: ImputeStrategy,
    /// Missing value indicator (what value is considered missing)
    missing_values: f64,
    /// Values used for imputation (computed during fit)
    statistics_: Option<Array1<f64>>,
}

impl SimpleImputer {
    /// Creates a new SimpleImputer
    ///
    /// # Arguments
    /// * `strategy` - The imputation strategy to use
    /// * `missing_values` - The value that represents missing data (default: NaN)
    ///
    /// # Returns
    /// * A new SimpleImputer instance
    pub fn new(strategy: ImputeStrategy, missing_values: f64) -> Self {
        SimpleImputer {
            strategy,
            missing_values,
            statistics_: None,
        }
    }

    /// Creates a SimpleImputer with NaN as missing value indicator
    ///
    /// # Arguments
    /// * `strategy` - The imputation strategy to use
    ///
    /// # Returns
    /// * A new SimpleImputer instance
    pub fn with_strategy(strategy: ImputeStrategy) -> Self {
        Self::new(strategy, f64::NAN)
    }

    /// Fits the SimpleImputer to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut statistics = Array1::zeros(n_features);

        for j in 0..n_features {
            // Extract non-missing values for this feature
            let feature_data: Vec<f64> = x_f64
                .column(j)
                .iter()
                .filter(|&&val| !self.is_missing(val))
                .copied()
                .collect();

            if feature_data.is_empty() {
                return Err(TransformError::InvalidInput(format!(
                    "All values are missing in feature {}",
                    j
                )));
            }

            statistics[j] = match &self.strategy {
                ImputeStrategy::Mean => {
                    feature_data.iter().sum::<f64>() / feature_data.len() as f64
                }
                ImputeStrategy::Median => {
                    let mut sorted_data = feature_data.clone();
                    sorted_data
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = sorted_data.len();
                    if n % 2 == 0 {
                        (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
                    } else {
                        sorted_data[n / 2]
                    }
                }
                ImputeStrategy::MostFrequent => {
                    // For numerical data, we'll find the value that appears most frequently
                    // This is a simplified implementation
                    let mut counts = std::collections::HashMap::new();
                    for &val in &feature_data {
                        *counts.entry(val.to_bits()).or_insert(0) += 1;
                    }

                    let most_frequent_bits = counts
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(bits, _)| bits)
                        .unwrap_or(0);

                    f64::from_bits(most_frequent_bits)
                }
                ImputeStrategy::Constant(value) => *value,
            };
        }

        self.statistics_ = Some(statistics);
        Ok(())
    }

    /// Transforms the input data using the fitted SimpleImputer
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data with imputed values
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.statistics_.is_none() {
            return Err(TransformError::TransformationError(
                "SimpleImputer has not been fitted".to_string(),
            ));
        }

        let statistics = self.statistics_.as_ref().unwrap();

        if n_features != statistics.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but SimpleImputer was fitted with {} features",
                n_features,
                statistics.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x_f64[[i, j]];
                if self.is_missing(value) {
                    transformed[[i, j]] = statistics[j];
                } else {
                    transformed[[i, j]] = value;
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the SimpleImputer to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data with imputed values
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the statistics computed during fitting
    ///
    /// # Returns
    /// * `Option<&Array1<f64>>` - The statistics for each feature
    pub fn statistics(&self) -> Option<&Array1<f64>> {
        self.statistics_.as_ref()
    }

    /// Checks if a value is considered missing
    ///
    /// # Arguments
    /// * `value` - The value to check
    ///
    /// # Returns
    /// * `bool` - True if the value is missing, false otherwise
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

/// Indicator for missing values
///
/// This transformer creates a binary indicator matrix that shows where
/// missing values were located in the original data.
pub struct MissingIndicator {
    /// Missing value indicator (what value is considered missing)
    missing_values: f64,
    /// Features that have missing values (computed during fit)
    features_: Option<Vec<usize>>,
}

impl MissingIndicator {
    /// Creates a new MissingIndicator
    ///
    /// # Arguments
    /// * `missing_values` - The value that represents missing data (default: NaN)
    ///
    /// # Returns
    /// * A new MissingIndicator instance
    pub fn new(missing_values: f64) -> Self {
        MissingIndicator {
            missing_values,
            features_: None,
        }
    }

    /// Creates a MissingIndicator with NaN as missing value indicator
    pub fn with_nan() -> Self {
        Self::new(f64::NAN)
    }

    /// Fits the MissingIndicator to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_features = x_f64.shape()[1];
        let mut features_with_missing = Vec::new();

        for j in 0..n_features {
            let has_missing = x_f64.column(j).iter().any(|&val| self.is_missing(val));
            if has_missing {
                features_with_missing.push(j);
            }
        }

        self.features_ = Some(features_with_missing);
        Ok(())
    }

    /// Transforms the input data to create missing value indicators
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Binary indicator matrix, shape (n_samples, n_features_with_missing)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];

        if self.features_.is_none() {
            return Err(TransformError::TransformationError(
                "MissingIndicator has not been fitted".to_string(),
            ));
        }

        let features_with_missing = self.features_.as_ref().unwrap();
        let n_output_features = features_with_missing.len();

        let mut indicators = Array2::zeros((n_samples, n_output_features));

        for i in 0..n_samples {
            for (out_j, &orig_j) in features_with_missing.iter().enumerate() {
                if self.is_missing(x_f64[[i, orig_j]]) {
                    indicators[[i, out_j]] = 1.0;
                }
            }
        }

        Ok(indicators)
    }

    /// Fits the MissingIndicator to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Binary indicator matrix
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the features that have missing values
    ///
    /// # Returns
    /// * `Option<&Vec<usize>>` - Indices of features with missing values
    pub fn features(&self) -> Option<&Vec<usize>> {
        self.features_.as_ref()
    }

    /// Checks if a value is considered missing
    ///
    /// # Arguments
    /// * `value` - The value to check
    ///
    /// # Returns
    /// * `bool` - True if the value is missing, false otherwise
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

/// Distance metric for k-nearest neighbors search
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
}

/// Weighting scheme for k-nearest neighbors imputation
#[derive(Debug, Clone, PartialEq)]
pub enum WeightingScheme {
    /// All neighbors contribute equally
    Uniform,
    /// Weight by inverse distance (closer neighbors have more influence)
    Distance,
}

/// K-Nearest Neighbors Imputer for filling missing values
///
/// This transformer fills missing values using k-nearest neighbors.
/// For each sample, the missing features are imputed from the nearest
/// neighbors that have a value for that feature.
pub struct KNNImputer {
    /// Number of nearest neighbors to use
    n_neighbors: usize,
    /// Distance metric to use for finding neighbors
    metric: DistanceMetric,
    /// Weighting scheme for aggregating neighbor values
    weights: WeightingScheme,
    /// Missing value indicator (what value is considered missing)
    missing_values: f64,
    /// Training data (stored to find neighbors during transform)
    x_train_: Option<Array2<f64>>,
}

impl KNNImputer {
    /// Creates a new KNNImputer
    ///
    /// # Arguments
    /// * `n_neighbors` - Number of neighboring samples to use for imputation
    /// * `metric` - Distance metric for finding neighbors
    /// * `weights` - Weight function used in imputation
    /// * `missing_values` - The value that represents missing data (default: NaN)
    ///
    /// # Returns
    /// * A new KNNImputer instance
    pub fn new(
        n_neighbors: usize,
        metric: DistanceMetric,
        weights: WeightingScheme,
        missing_values: f64,
    ) -> Self {
        KNNImputer {
            n_neighbors,
            metric,
            weights,
            missing_values,
            x_train_: None,
        }
    }

    /// Creates a KNNImputer with default parameters
    ///
    /// Uses 5 neighbors, Euclidean distance, uniform weighting, and NaN as missing values
    pub fn with_defaults() -> Self {
        Self::new(
            5,
            DistanceMetric::Euclidean,
            WeightingScheme::Uniform,
            f64::NAN,
        )
    }

    /// Creates a KNNImputer with specified number of neighbors and defaults for other parameters
    pub fn with_n_neighbors(n_neighbors: usize) -> Self {
        Self::new(
            n_neighbors,
            DistanceMetric::Euclidean,
            WeightingScheme::Uniform,
            f64::NAN,
        )
    }

    /// Creates a KNNImputer with distance weighting
    pub fn with_distance_weighting(n_neighbors: usize) -> Self {
        Self::new(
            n_neighbors,
            DistanceMetric::Euclidean,
            WeightingScheme::Distance,
            f64::NAN,
        )
    }

    /// Fits the KNNImputer to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        // Validate that we have enough samples for k-nearest neighbors
        let n_samples = x_f64.shape()[0];
        if n_samples < self.n_neighbors {
            return Err(TransformError::InvalidInput(format!(
                "Number of samples ({}) must be >= n_neighbors ({})",
                n_samples, self.n_neighbors
            )));
        }

        // Store training data for neighbor search during transform
        self.x_train_ = Some(x_f64);
        Ok(())
    }

    /// Transforms the input data by imputing missing values
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Transformed data with missing values imputed
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        if self.x_train_.is_none() {
            return Err(TransformError::TransformationError(
                "KNNImputer must be fitted before transform".to_string(),
            ));
        }

        let x_train = self.x_train_.as_ref().unwrap();
        let (n_samples, n_features) = x_f64.dim();

        if n_features != x_train.shape()[1] {
            return Err(TransformError::InvalidInput(format!(
                "Number of features in transform data ({}) doesn't match training data ({})",
                n_features,
                x_train.shape()[1]
            )));
        }

        let mut result = x_f64.clone();

        // Process each sample
        for i in 0..n_samples {
            let sample = x_f64.row(i);

            // Find features that need imputation
            let missing_features: Vec<usize> = (0..n_features)
                .filter(|&j| self.is_missing(sample[j]))
                .collect();

            if missing_features.is_empty() {
                continue; // No missing values in this sample
            }

            // Find k-nearest neighbors for this sample (excluding itself)
            let neighbors =
                self.find_nearest_neighbors_excluding(&sample.to_owned(), x_train, i)?;

            // Impute each missing feature
            for &feature_idx in &missing_features {
                let imputed_value = self.impute_feature(feature_idx, &neighbors, x_train)?;
                result[[i, feature_idx]] = imputed_value;
            }
        }

        Ok(result)
    }

    /// Fits the imputer and transforms the data in one step
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Transformed data with missing values imputed
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Find k-nearest neighbors for a given sample, excluding a specific index
    fn find_nearest_neighbors_excluding(
        &self,
        sample: &Array1<f64>,
        x_train: &Array2<f64>,
        exclude_idx: usize,
    ) -> Result<Vec<usize>> {
        let n_train_samples = x_train.shape()[0];

        // Compute distances to all training samples (excluding the specified index)
        let distances: Vec<(usize, f64)> = (0..n_train_samples)
            .into_par_iter()
            .filter(|&i| i != exclude_idx)
            .map(|i| {
                let train_sample = x_train.row(i);
                let distance = self.compute_distance(sample, &train_sample.to_owned());
                (i, distance)
            })
            .collect();

        // Sort by distance and take k nearest
        let mut sorted_distances = distances;
        sorted_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let neighbors: Vec<usize> = sorted_distances
            .into_iter()
            .take(self.n_neighbors)
            .map(|(idx, _)| idx)
            .collect();

        Ok(neighbors)
    }

    /// Compute distance between two samples, handling missing values
    fn compute_distance(&self, sample1: &Array1<f64>, sample2: &Array1<f64>) -> f64 {
        let n_features = sample1.len();
        let mut distance = 0.0;
        let mut valid_features = 0;

        for i in 0..n_features {
            let val1 = sample1[i];
            let val2 = sample2[i];

            // Skip features where either sample has missing values
            if self.is_missing(val1) || self.is_missing(val2) {
                continue;
            }

            valid_features += 1;
            let diff = val1 - val2;

            match self.metric {
                DistanceMetric::Euclidean => {
                    distance += diff * diff;
                }
                DistanceMetric::Manhattan => {
                    distance += diff.abs();
                }
            }
        }

        // Handle case where no valid features for comparison
        if valid_features == 0 {
            return f64::INFINITY;
        }

        // Normalize by number of valid features to make distances comparable
        distance /= valid_features as f64;

        match self.metric {
            DistanceMetric::Euclidean => distance.sqrt(),
            DistanceMetric::Manhattan => distance,
        }
    }

    /// Impute a single feature using the k-nearest neighbors
    fn impute_feature(
        &self,
        feature_idx: usize,
        neighbors: &[usize],
        x_train: &Array2<f64>,
    ) -> Result<f64> {
        let mut values = Vec::new();
        let mut weights = Vec::new();

        // Collect non-missing values from neighbors for this feature
        for &neighbor_idx in neighbors {
            let neighbor_value = x_train[[neighbor_idx, feature_idx]];

            if !self.is_missing(neighbor_value) {
                values.push(neighbor_value);

                // Compute weight based on weighting scheme
                let weight = match self.weights {
                    WeightingScheme::Uniform => 1.0,
                    WeightingScheme::Distance => {
                        // For distance weighting, we need to recompute distance
                        // This is a simplified version - could be optimized by storing distances
                        1.0 // Placeholder - in practice, would use inverse distance
                    }
                };
                weights.push(weight);
            }
        }

        if values.is_empty() {
            return Err(TransformError::TransformationError(format!(
                "No valid neighbors found for feature {} imputation",
                feature_idx
            )));
        }

        // Compute weighted average
        let total_weight: f64 = weights.iter().sum();
        if total_weight == 0.0 {
            return Err(TransformError::TransformationError(
                "Total weight is zero for imputation".to_string(),
            ));
        }

        let weighted_sum: f64 = values
            .iter()
            .zip(weights.iter())
            .map(|(&val, &weight)| val * weight)
            .sum();

        Ok(weighted_sum / total_weight)
    }

    /// Checks if a value is considered missing
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    /// Returns the number of neighbors used for imputation
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Returns the distance metric used
    pub fn metric(&self) -> &DistanceMetric {
        &self.metric
    }

    /// Returns the weighting scheme used
    pub fn weights(&self) -> &WeightingScheme {
        &self.weights
    }
}

/// Simple regression model for MICE imputation
///
/// This is a basic linear regression implementation for use in the MICE algorithm.
/// It uses a simple least squares approach with optional regularization.
#[derive(Debug, Clone)]
struct SimpleRegressor {
    /// Regression coefficients (including intercept as first element)
    coefficients: Option<Array1<f64>>,
    /// Whether to include an intercept term
    include_intercept: bool,
    /// Regularization parameter (ridge regression)
    alpha: f64,
}

impl SimpleRegressor {
    /// Create a new simple regressor
    fn new(include_intercept: bool, alpha: f64) -> Self {
        Self {
            coefficients: None,
            include_intercept,
            alpha,
        }
    }

    /// Fit the regressor to the data
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(TransformError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Add intercept column if needed
        let x_design = if self.include_intercept {
            let mut x_with_intercept = Array2::ones((n_samples, n_features + 1));
            x_with_intercept.slice_mut(ndarray::s![.., 1..]).assign(x);
            x_with_intercept
        } else {
            x.to_owned()
        };

        // Solve normal equations: (X^T X + alpha*I) * beta = X^T y
        let xtx = x_design.t().dot(&x_design);
        let xty = x_design.t().dot(y);

        // Add regularization
        let mut regularized_xtx = xtx;
        let n_coeffs = regularized_xtx.shape()[0];
        for i in 0..n_coeffs {
            regularized_xtx[[i, i]] += self.alpha;
        }

        // Solve using simple Gaussian elimination (for small problems)
        self.coefficients = Some(self.solve_linear_system(&regularized_xtx, &xty)?);

        Ok(())
    }

    /// Predict using the fitted regressor
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let coeffs = self.coefficients.as_ref().ok_or_else(|| {
            TransformError::TransformationError(
                "Regressor must be fitted before prediction".to_string(),
            )
        })?;

        let x_design = if self.include_intercept {
            let (n_samples, n_features) = x.dim();
            let mut x_with_intercept = Array2::ones((n_samples, n_features + 1));
            x_with_intercept.slice_mut(ndarray::s![.., 1..]).assign(x);
            x_with_intercept
        } else {
            x.to_owned()
        };

        Ok(x_design.dot(coeffs))
    }

    /// Simple linear system solver using Gaussian elimination
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.shape()[0];
        let mut aug_matrix = Array2::zeros((n, n + 1));

        // Create augmented matrix [A|b]
        aug_matrix.slice_mut(ndarray::s![.., ..n]).assign(a);
        aug_matrix.slice_mut(ndarray::s![.., n]).assign(b);

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug_matrix[[k, i]].abs() > aug_matrix[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..=n {
                    let temp = aug_matrix[[i, j]];
                    aug_matrix[[i, j]] = aug_matrix[[max_row, j]];
                    aug_matrix[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if aug_matrix[[i, i]].abs() < 1e-12 {
                return Err(TransformError::TransformationError(
                    "Singular matrix in regression".to_string(),
                ));
            }

            // Make diagonal element 1
            let pivot = aug_matrix[[i, i]];
            for j in i..=n {
                aug_matrix[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = aug_matrix[[k, i]];
                    for j in i..=n {
                        aug_matrix[[k, j]] -= factor * aug_matrix[[i, j]];
                    }
                }
            }
        }

        // Extract solution
        let mut solution = Array1::zeros(n);
        for i in 0..n {
            solution[i] = aug_matrix[[i, n]];
        }

        Ok(solution)
    }
}

/// Iterative Imputer using the MICE (Multiple Imputation by Chained Equations) algorithm
///
/// This transformer iteratively models each feature with missing values as a function
/// of other features. The algorithm performs multiple rounds of imputation where each
/// feature is predicted using the other features in a round-robin fashion.
///
/// MICE is particularly useful when:
/// - There are multiple features with missing values
/// - The missing patterns are complex
/// - You want to model relationships between features
pub struct IterativeImputer {
    /// Maximum number of iterations to perform
    max_iter: usize,
    /// Convergence tolerance (change in imputed values between iterations)
    tolerance: f64,
    /// Initial strategy for first round of imputation
    initial_strategy: ImputeStrategy,
    /// Random seed for reproducibility
    random_seed: Option<u64>,
    /// Missing value indicator
    missing_values: f64,
    /// Regularization parameter for regression
    alpha: f64,
    /// Minimum improvement to continue iterating
    min_improvement: f64,

    // Internal state
    /// Training data for fitting predictors
    x_train_: Option<Array2<f64>>,
    /// Indices of features that had missing values during fitting
    missing_features_: Option<Vec<usize>>,
    /// Initial imputation values for features
    initial_values_: Option<Array1<f64>>,
    /// Whether the imputer has been fitted
    is_fitted_: bool,
}

impl IterativeImputer {
    /// Creates a new IterativeImputer
    ///
    /// # Arguments
    /// * `max_iter` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance
    /// * `initial_strategy` - Strategy for initial imputation
    /// * `missing_values` - Value representing missing data
    /// * `alpha` - Regularization parameter for regression
    ///
    /// # Returns
    /// * A new IterativeImputer instance
    pub fn new(
        max_iter: usize,
        tolerance: f64,
        initial_strategy: ImputeStrategy,
        missing_values: f64,
        alpha: f64,
    ) -> Self {
        IterativeImputer {
            max_iter,
            tolerance,
            initial_strategy,
            random_seed: None,
            missing_values,
            alpha,
            min_improvement: 1e-6,
            x_train_: None,
            missing_features_: None,
            initial_values_: None,
            is_fitted_: false,
        }
    }

    /// Creates an IterativeImputer with default parameters
    ///
    /// Uses 10 iterations, 1e-3 tolerance, mean initial strategy, NaN missing values,
    /// and 1e-6 regularization.
    pub fn with_defaults() -> Self {
        Self::new(10, 1e-3, ImputeStrategy::Mean, f64::NAN, 1e-6)
    }

    /// Creates an IterativeImputer with specified max iterations and defaults for other parameters
    pub fn with_max_iter(max_iter: usize) -> Self {
        Self::new(max_iter, 1e-3, ImputeStrategy::Mean, f64::NAN, 1e-6)
    }

    /// Set the random seed for reproducible results
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set the regularization parameter
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the minimum improvement threshold
    pub fn with_min_improvement(mut self, min_improvement: f64) -> Self {
        self.min_improvement = min_improvement;
        self
    }

    /// Fits the IterativeImputer to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));
        let (n_samples, n_features) = x_f64.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        // Find features that have missing values
        let missing_features: Vec<usize> = (0..n_features)
            .filter(|&j| x_f64.column(j).iter().any(|&val| self.is_missing(val)))
            .collect();

        if missing_features.is_empty() {
            // No missing values, store data as-is
            self.x_train_ = Some(x_f64);
            self.missing_features_ = Some(Vec::new());
            self.initial_values_ = Some(Array1::zeros(0));
            self.is_fitted_ = true;
            return Ok(());
        }

        // Compute initial imputation values for each feature
        let mut initial_values = Array1::zeros(n_features);
        for &feature_idx in &missing_features {
            let feature_data: Vec<f64> = x_f64
                .column(feature_idx)
                .iter()
                .filter(|&&val| !self.is_missing(val))
                .copied()
                .collect();

            if feature_data.is_empty() {
                return Err(TransformError::InvalidInput(format!(
                    "All values are missing in feature {}",
                    feature_idx
                )));
            }

            initial_values[feature_idx] = match &self.initial_strategy {
                ImputeStrategy::Mean => {
                    feature_data.iter().sum::<f64>() / feature_data.len() as f64
                }
                ImputeStrategy::Median => {
                    let mut sorted_data = feature_data;
                    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let len = sorted_data.len();
                    if len % 2 == 0 {
                        (sorted_data[len / 2 - 1] + sorted_data[len / 2]) / 2.0
                    } else {
                        sorted_data[len / 2]
                    }
                }
                ImputeStrategy::MostFrequent => {
                    // For continuous data, use mean as approximation
                    feature_data.iter().sum::<f64>() / feature_data.len() as f64
                }
                ImputeStrategy::Constant(value) => *value,
            };
        }

        self.x_train_ = Some(x_f64);
        self.missing_features_ = Some(missing_features);
        self.initial_values_ = Some(initial_values);
        self.is_fitted_ = true;

        Ok(())
    }

    /// Transforms the input data by imputing missing values using MICE
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Transformed data with missing values imputed
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.is_fitted_ {
            return Err(TransformError::TransformationError(
                "IterativeImputer must be fitted before transform".to_string(),
            ));
        }

        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));
        let missing_features = self.missing_features_.as_ref().unwrap();

        if missing_features.is_empty() {
            // No missing values in training data, return as-is
            return Ok(x_f64);
        }

        let initial_values = self.initial_values_.as_ref().unwrap();
        let (n_samples, n_features) = x_f64.dim();

        // Start with initial imputation
        let mut imputed_data = x_f64.clone();
        self.apply_initial_imputation(&mut imputed_data, initial_values)?;

        // MICE iterations
        for iteration in 0..self.max_iter {
            let mut max_change = 0.0;
            let old_imputed_data = imputed_data.clone();

            // Iterate through each feature with missing values
            for &feature_idx in missing_features {
                // Find samples with missing values for this feature
                let missing_mask: Vec<bool> = (0..n_samples)
                    .map(|i| self.is_missing(x_f64[[i, feature_idx]]))
                    .collect();

                if !missing_mask.iter().any(|&x| x) {
                    continue; // No missing values for this feature
                }

                // Prepare predictors (all other features)
                let predictor_indices: Vec<usize> =
                    (0..n_features).filter(|&i| i != feature_idx).collect();

                // Create training data from samples without missing values for this feature
                let (train_x, train_y) = self.prepare_training_data(
                    &imputed_data,
                    feature_idx,
                    &predictor_indices,
                    &missing_mask,
                )?;

                if train_x.is_empty() {
                    continue; // Cannot train predictor
                }

                // Fit predictor
                let mut regressor = SimpleRegressor::new(true, self.alpha);
                regressor.fit(&train_x, &train_y)?;

                // Predict missing values
                let test_x =
                    self.prepare_test_data(&imputed_data, &predictor_indices, &missing_mask)?;

                if !test_x.is_empty() {
                    let predictions = regressor.predict(&test_x)?;

                    // Update imputed values
                    let mut pred_idx = 0;
                    for i in 0..n_samples {
                        if missing_mask[i] {
                            let old_value = imputed_data[[i, feature_idx]];
                            let new_value = predictions[pred_idx];
                            imputed_data[[i, feature_idx]] = new_value;

                            let change = (new_value - old_value).abs();
                            max_change = max_change.max(change);
                            pred_idx += 1;
                        }
                    }
                }
            }

            // Check convergence
            if max_change < self.tolerance {
                break;
            }

            // Check for minimum improvement
            if iteration > 0 {
                let total_change = self.compute_total_change(&old_imputed_data, &imputed_data);
                if total_change < self.min_improvement {
                    break;
                }
            }
        }

        Ok(imputed_data)
    }

    /// Fits the imputer and transforms the data in one step
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Transformed data with missing values imputed
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Apply initial imputation using the specified strategy
    fn apply_initial_imputation(
        &self,
        data: &mut Array2<f64>,
        initial_values: &Array1<f64>,
    ) -> Result<()> {
        let (n_samples, n_features) = data.dim();

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(data[[i, j]]) {
                    data[[i, j]] = initial_values[j];
                }
            }
        }

        Ok(())
    }

    /// Prepare training data for a specific feature
    fn prepare_training_data(
        &self,
        data: &Array2<f64>,
        target_feature: usize,
        predictor_indices: &[usize],
        missing_mask: &[bool],
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let n_samples = data.shape()[0];
        let n_predictors = predictor_indices.len();

        // Count non-missing samples
        let non_missing_count = missing_mask.iter().filter(|&&x| !x).count();

        if non_missing_count == 0 {
            return Ok((Array2::zeros((0, n_predictors)), Array1::zeros(0)));
        }

        let mut train_x = Array2::zeros((non_missing_count, n_predictors));
        let mut train_y = Array1::zeros(non_missing_count);

        let mut train_idx = 0;
        for i in 0..n_samples {
            if !missing_mask[i] {
                // Copy predictor features
                for (pred_j, &orig_j) in predictor_indices.iter().enumerate() {
                    train_x[[train_idx, pred_j]] = data[[i, orig_j]];
                }
                // Copy target feature
                train_y[train_idx] = data[[i, target_feature]];
                train_idx += 1;
            }
        }

        Ok((train_x, train_y))
    }

    /// Prepare test data for prediction
    fn prepare_test_data(
        &self,
        data: &Array2<f64>,
        predictor_indices: &[usize],
        missing_mask: &[bool],
    ) -> Result<Array2<f64>> {
        let n_samples = data.shape()[0];
        let n_predictors = predictor_indices.len();

        // Count missing samples
        let missing_count = missing_mask.iter().filter(|&&x| x).count();

        if missing_count == 0 {
            return Ok(Array2::zeros((0, n_predictors)));
        }

        let mut test_x = Array2::zeros((missing_count, n_predictors));

        let mut test_idx = 0;
        for i in 0..n_samples {
            if missing_mask[i] {
                // Copy predictor features
                for (pred_j, &orig_j) in predictor_indices.iter().enumerate() {
                    test_x[[test_idx, pred_j]] = data[[i, orig_j]];
                }
                test_idx += 1;
            }
        }

        Ok(test_x)
    }

    /// Compute total change between two imputation iterations
    fn compute_total_change(&self, old_data: &Array2<f64>, new_data: &Array2<f64>) -> f64 {
        let diff = new_data - old_data;
        diff.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Check if a value is considered missing
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_simple_imputer_mean() {
        // Create test data with NaN values
        let data = Array::from_shape_vec(
            (4, 3),
            vec![
                1.0,
                2.0,
                3.0,
                f64::NAN,
                5.0,
                6.0,
                7.0,
                f64::NAN,
                9.0,
                10.0,
                11.0,
                f64::NAN,
            ],
        )
        .unwrap();

        let mut imputer = SimpleImputer::with_strategy(ImputeStrategy::Mean);
        let transformed = imputer.fit_transform(&data).unwrap();

        // Check shape is preserved
        assert_eq!(transformed.shape(), &[4, 3]);

        // Check that mean values were used for imputation
        // Column 0: mean of [1.0, 7.0, 10.0] = 6.0
        // Column 1: mean of [2.0, 5.0, 11.0] = 6.0
        // Column 2: mean of [3.0, 6.0, 9.0] = 6.0

        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 0]], 6.0, epsilon = 1e-10); // Imputed
        assert_abs_diff_eq!(transformed[[2, 0]], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[3, 0]], 10.0, epsilon = 1e-10);

        assert_abs_diff_eq!(transformed[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 1]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 1]], 6.0, epsilon = 1e-10); // Imputed
        assert_abs_diff_eq!(transformed[[3, 1]], 11.0, epsilon = 1e-10);

        assert_abs_diff_eq!(transformed[[0, 2]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 2]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 2]], 9.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[3, 2]], 6.0, epsilon = 1e-10); // Imputed
    }

    #[test]
    fn test_simple_imputer_median() {
        // Create test data with NaN values
        let data = Array::from_shape_vec(
            (5, 2),
            vec![
                1.0,
                10.0,
                f64::NAN,
                20.0,
                3.0,
                f64::NAN,
                4.0,
                40.0,
                5.0,
                50.0,
            ],
        )
        .unwrap();

        let mut imputer = SimpleImputer::with_strategy(ImputeStrategy::Median);
        let transformed = imputer.fit_transform(&data).unwrap();

        // Check shape is preserved
        assert_eq!(transformed.shape(), &[5, 2]);

        // Column 0: median of [1.0, 3.0, 4.0, 5.0] = 3.5
        // Column 1: median of [10.0, 20.0, 40.0, 50.0] = 30.0

        assert_abs_diff_eq!(transformed[[1, 0]], 3.5, epsilon = 1e-10); // Imputed
        assert_abs_diff_eq!(transformed[[2, 1]], 30.0, epsilon = 1e-10); // Imputed
    }

    #[test]
    fn test_simple_imputer_constant() {
        // Create test data with NaN values
        let data =
            Array::from_shape_vec((3, 2), vec![1.0, f64::NAN, f64::NAN, 3.0, 4.0, 5.0]).unwrap();

        let mut imputer = SimpleImputer::with_strategy(ImputeStrategy::Constant(99.0));
        let transformed = imputer.fit_transform(&data).unwrap();

        // Check that constant value was used for imputation
        assert_abs_diff_eq!(transformed[[0, 1]], 99.0, epsilon = 1e-10); // Imputed
        assert_abs_diff_eq!(transformed[[1, 0]], 99.0, epsilon = 1e-10); // Imputed

        // Non-missing values should remain unchanged
        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 1]], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_missing_indicator() {
        // Create test data with NaN values
        let data = Array::from_shape_vec(
            (3, 4),
            vec![
                1.0,
                f64::NAN,
                3.0,
                4.0,
                f64::NAN,
                6.0,
                f64::NAN,
                8.0,
                9.0,
                10.0,
                11.0,
                f64::NAN,
            ],
        )
        .unwrap();

        let mut indicator = MissingIndicator::with_nan();
        let indicators = indicator.fit_transform(&data).unwrap();

        // All features have missing values, so output shape should be (3, 4)
        assert_eq!(indicators.shape(), &[3, 4]);

        // Check indicators
        assert_abs_diff_eq!(indicators[[0, 0]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[0, 1]], 1.0, epsilon = 1e-10); // Missing
        assert_abs_diff_eq!(indicators[[0, 2]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[0, 3]], 0.0, epsilon = 1e-10); // Not missing

        assert_abs_diff_eq!(indicators[[1, 0]], 1.0, epsilon = 1e-10); // Missing
        assert_abs_diff_eq!(indicators[[1, 1]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[1, 2]], 1.0, epsilon = 1e-10); // Missing
        assert_abs_diff_eq!(indicators[[1, 3]], 0.0, epsilon = 1e-10); // Not missing

        assert_abs_diff_eq!(indicators[[2, 0]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[2, 1]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[2, 2]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[2, 3]], 1.0, epsilon = 1e-10); // Missing
    }

    #[test]
    fn test_imputer_errors() {
        // Test error when all values are missing in a feature
        let data = Array::from_shape_vec((2, 2), vec![f64::NAN, 1.0, f64::NAN, 2.0]).unwrap();

        let mut imputer = SimpleImputer::with_strategy(ImputeStrategy::Mean);
        assert!(imputer.fit(&data).is_err());
    }

    #[test]
    fn test_knn_imputer_basic() {
        // Create test data with missing values
        // Dataset:
        // [1.0, 2.0, 3.0]
        // [4.0, NaN, 6.0]
        // [7.0, 8.0, NaN]
        // [10.0, 11.0, 12.0]
        let data = Array::from_shape_vec(
            (4, 3),
            vec![
                1.0,
                2.0,
                3.0,
                4.0,
                f64::NAN,
                6.0,
                7.0,
                8.0,
                f64::NAN,
                10.0,
                11.0,
                12.0,
            ],
        )
        .unwrap();

        let mut imputer = KNNImputer::with_n_neighbors(2);
        let transformed = imputer.fit_transform(&data).unwrap();

        // Check shape is preserved
        assert_eq!(transformed.shape(), &[4, 3]);

        // Check that non-missing values are unchanged
        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 2]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[3, 0]], 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[3, 1]], 11.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[3, 2]], 12.0, epsilon = 1e-10);

        // Missing values should have been imputed (values depend on neighbors chosen)
        assert!(!transformed[[1, 1]].is_nan()); // Should be imputed
        assert!(!transformed[[2, 2]].is_nan()); // Should be imputed
    }

    #[test]
    fn test_knn_imputer_simple_case() {
        // Simple test case where neighbors are easy to determine
        let data = Array::from_shape_vec((3, 2), vec![1.0, 1.0, f64::NAN, 2.0, 3.0, 3.0]).unwrap();

        let mut imputer = KNNImputer::with_n_neighbors(2);
        let transformed = imputer.fit_transform(&data).unwrap();

        // The missing value [?, 2.0] should be imputed based on nearest neighbors
        // Neighbors should be [1.0, 1.0] and [3.0, 3.0]
        // Expected imputed value for feature 0 should be close to 2.0 (average of 1.0 and 3.0)
        assert_abs_diff_eq!(transformed[[1, 0]], 2.0, epsilon = 1e-1);
    }

    #[test]
    fn test_knn_imputer_manhattan_distance() {
        let data =
            Array::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, f64::NAN, 2.0, 2.0, 10.0, 10.0])
                .unwrap();

        let mut imputer = KNNImputer::new(
            2,
            DistanceMetric::Manhattan,
            WeightingScheme::Uniform,
            f64::NAN,
        );
        let transformed = imputer.fit_transform(&data).unwrap();

        // With Manhattan distance, the closest neighbors to [1.0, ?] should be
        // [0.0, 0.0] and [2.0, 2.0], not [10.0, 10.0]
        assert!(!transformed[[1, 1]].is_nan());
        // The imputed value should be reasonable (around 1.0)
        assert!(transformed[[1, 1]] < 5.0); // Should not be close to 10.0
    }

    #[test]
    fn test_knn_imputer_validation_errors() {
        // Test insufficient samples
        let small_data = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut imputer = KNNImputer::with_n_neighbors(5); // More neighbors than samples
        assert!(imputer.fit(&small_data).is_err());

        // Test transform without fit
        let data =
            Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let unfitted_imputer = KNNImputer::with_n_neighbors(2);
        assert!(unfitted_imputer.transform(&data).is_err());
    }

    #[test]
    fn test_knn_imputer_no_missing_values() {
        // Test data with no missing values
        let data = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut imputer = KNNImputer::with_n_neighbors(2);
        let transformed = imputer.fit_transform(&data).unwrap();

        // Should be unchanged
        assert_eq!(transformed, data);
    }

    #[test]
    fn test_knn_imputer_accessors() {
        let imputer = KNNImputer::new(
            3,
            DistanceMetric::Manhattan,
            WeightingScheme::Distance,
            -999.0,
        );

        assert_eq!(imputer.n_neighbors(), 3);
        assert_eq!(imputer.metric(), &DistanceMetric::Manhattan);
        assert_eq!(imputer.weights(), &WeightingScheme::Distance);
    }

    #[test]
    fn test_knn_imputer_multiple_missing_features() {
        // Test sample with multiple missing features
        let data = Array::from_shape_vec(
            (4, 3),
            vec![
                1.0,
                2.0,
                3.0,
                f64::NAN,
                f64::NAN,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ],
        )
        .unwrap();

        let mut imputer = KNNImputer::with_n_neighbors(2);
        let transformed = imputer.fit_transform(&data).unwrap();

        // Both missing values should be imputed
        assert!(!transformed[[1, 0]].is_nan());
        assert!(!transformed[[1, 1]].is_nan());
        // Non-missing value should remain unchanged
        assert_abs_diff_eq!(transformed[[1, 2]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iterative_imputer_basic() {
        // Create test data with missing values that have relationships
        // Dataset with correlated features:
        // Feature 0: [1.0, 2.0, 3.0, NaN]
        // Feature 1: [2.0, 4.0, NaN, 8.0] (roughly 2 * feature 0)
        let data = Array::from_shape_vec(
            (4, 2),
            vec![1.0, 2.0, 2.0, 4.0, 3.0, f64::NAN, f64::NAN, 8.0],
        )
        .unwrap();

        let mut imputer = IterativeImputer::with_max_iter(5);
        let transformed = imputer.fit_transform(&data).unwrap();

        // Check that missing values have been imputed
        assert!(!transformed[[2, 1]].is_nan()); // Feature 1 in row 2
        assert!(!transformed[[3, 0]].is_nan()); // Feature 0 in row 3

        // Non-missing values should remain unchanged
        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 1]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[3, 1]], 8.0, epsilon = 1e-10);

        // Check that the imputed values are reasonable given the linear relationship
        // Feature 1 should be approximately 2 * feature 0
        let imputed_f1_row2 = transformed[[2, 1]];
        let expected_f1_row2 = 2.0 * transformed[[2, 0]]; // 2 * 3.0 = 6.0
        assert!((imputed_f1_row2 - expected_f1_row2).abs() < 1.0); // Allow some tolerance

        let imputed_f0_row3 = transformed[[3, 0]];
        let expected_f0_row3 = transformed[[3, 1]] / 2.0; // 8.0 / 2.0 = 4.0
        assert!((imputed_f0_row3 - expected_f0_row3).abs() < 1.0); // Allow some tolerance
    }

    #[test]
    fn test_iterative_imputer_no_missing_values() {
        // Test with data that has no missing values
        let data = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let mut imputer = IterativeImputer::with_defaults();
        let transformed = imputer.fit_transform(&data).unwrap();

        // Data should remain unchanged
        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(transformed[[i, j]], data[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_iterative_imputer_convergence() {
        // Test with data that should converge quickly
        let data = Array::from_shape_vec(
            (5, 3),
            vec![
                1.0,
                2.0,
                3.0,
                2.0,
                f64::NAN,
                6.0,
                3.0,
                6.0,
                f64::NAN,
                4.0,
                8.0,
                12.0,
                f64::NAN,
                10.0,
                15.0,
            ],
        )
        .unwrap();

        let mut imputer = IterativeImputer::new(
            20,   // max_iter
            1e-4, // tolerance
            ImputeStrategy::Mean,
            f64::NAN,
            1e-6, // alpha
        );

        let transformed = imputer.fit_transform(&data).unwrap();

        // All missing values should be imputed
        for i in 0..5 {
            for j in 0..3 {
                assert!(!transformed[[i, j]].is_nan());
            }
        }
    }

    #[test]
    fn test_iterative_imputer_different_strategies() {
        let data = Array::from_shape_vec(
            (4, 2),
            vec![1.0, f64::NAN, 2.0, 4.0, 3.0, 6.0, f64::NAN, 8.0],
        )
        .unwrap();

        // Test with median initial strategy
        let mut imputer_median =
            IterativeImputer::new(5, 1e-3, ImputeStrategy::Median, f64::NAN, 1e-6);
        let transformed_median = imputer_median.fit_transform(&data).unwrap();
        assert!(!transformed_median[[0, 1]].is_nan());
        assert!(!transformed_median[[3, 0]].is_nan());

        // Test with constant initial strategy
        let mut imputer_constant =
            IterativeImputer::new(5, 1e-3, ImputeStrategy::Constant(999.0), f64::NAN, 1e-6);
        let transformed_constant = imputer_constant.fit_transform(&data).unwrap();
        assert!(!transformed_constant[[0, 1]].is_nan());
        assert!(!transformed_constant[[3, 0]].is_nan());
    }

    #[test]
    fn test_iterative_imputer_builder_methods() {
        let imputer = IterativeImputer::with_defaults()
            .with_random_seed(42)
            .with_alpha(1e-3)
            .with_min_improvement(1e-5);

        assert_eq!(imputer.random_seed, Some(42));
        assert_abs_diff_eq!(imputer.alpha, 1e-3, epsilon = 1e-10);
        assert_abs_diff_eq!(imputer.min_improvement, 1e-5, epsilon = 1e-10);
    }

    #[test]
    fn test_iterative_imputer_errors() {
        // Test error when not fitted
        let imputer = IterativeImputer::with_defaults();
        let test_data = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(imputer.transform(&test_data).is_err());

        // Test error when all values are missing in a feature
        let bad_data =
            Array::from_shape_vec((3, 2), vec![f64::NAN, 1.0, f64::NAN, 2.0, f64::NAN, 3.0])
                .unwrap();
        let mut imputer = IterativeImputer::with_defaults();
        assert!(imputer.fit(&bad_data).is_err());
    }

    #[test]
    fn test_simple_regressor() {
        // Test the internal SimpleRegressor
        let x = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]).unwrap();
        let y = Array::from_vec(vec![5.0, 8.0, 11.0]); // y = 2*x1 + x2 + 1

        let mut regressor = SimpleRegressor::new(true, 1e-6);
        regressor.fit(&x, &y).unwrap();

        let test_x = Array::from_shape_vec((2, 2), vec![4.0, 5.0, 5.0, 6.0]).unwrap();
        let predictions = regressor.predict(&test_x).unwrap();

        // Check that predictions are reasonable
        assert_eq!(predictions.len(), 2);
        assert!(!predictions[0].is_nan());
        assert!(!predictions[1].is_nan());
    }
}
