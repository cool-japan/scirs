//! Categorical data encoding utilities
//!
//! This module provides methods for encoding categorical data into numerical
//! formats suitable for machine learning algorithms.

use ndarray::{Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};
use std::collections::HashMap;

use crate::error::{Result, TransformError};

/// OneHotEncoder for converting categorical features to binary features
///
/// This transformer converts categorical features into a one-hot encoded representation,
/// where each category is represented by a binary feature.
pub struct OneHotEncoder {
    /// Categories for each feature (learned during fit)
    categories_: Option<Vec<Vec<u64>>>,
    /// Whether to drop one category per feature to avoid collinearity
    drop: Option<String>,
    /// Whether to handle unknown categories
    handle_unknown: String,
    /// Sparse output (not implemented yet)
    #[allow(dead_code)]
    sparse: bool,
}

impl OneHotEncoder {
    /// Creates a new OneHotEncoder
    ///
    /// # Arguments
    /// * `drop` - Strategy for dropping categories ('first', 'if_binary', or None)
    /// * `handle_unknown` - How to handle unknown categories ('error' or 'ignore')
    /// * `sparse` - Whether to return sparse arrays (not implemented)
    ///
    /// # Returns
    /// * A new OneHotEncoder instance
    pub fn new(drop: Option<String>, handle_unknown: &str, sparse: bool) -> Result<Self> {
        if let Some(ref drop_strategy) = drop {
            if drop_strategy != "first" && drop_strategy != "if_binary" {
                return Err(TransformError::InvalidInput(
                    "drop must be 'first', 'if_binary', or None".to_string(),
                ));
            }
        }

        if handle_unknown != "error" && handle_unknown != "ignore" {
            return Err(TransformError::InvalidInput(
                "handle_unknown must be 'error' or 'ignore'".to_string(),
            ));
        }

        if sparse {
            return Err(TransformError::InvalidInput(
                "Sparse output is not yet implemented".to_string(),
            ));
        }

        Ok(OneHotEncoder {
            categories_: None,
            drop,
            handle_unknown: handle_unknown.to_string(),
            sparse,
        })
    }

    /// Creates a OneHotEncoder with default settings
    pub fn with_defaults() -> Self {
        Self::new(None, "error", false).unwrap()
    }

    /// Fits the OneHotEncoder to the input data
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut categories = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect unique values for this feature
            let mut unique_values: Vec<u64> = x_u64.column(j).to_vec();
            unique_values.sort_unstable();
            unique_values.dedup();

            categories.push(unique_values);
        }

        self.categories_ = Some(categories);
        Ok(())
    }

    /// Transforms the input data using the fitted OneHotEncoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The one-hot encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if self.categories_.is_none() {
            return Err(TransformError::TransformationError(
                "OneHotEncoder has not been fitted".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().unwrap();

        if n_features != categories.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but OneHotEncoder was fitted with {} features",
                n_features,
                categories.len()
            )));
        }

        // Calculate total number of output features
        let mut total_features = 0;
        for (j, feature_categories) in categories.iter().enumerate() {
            let n_cats = feature_categories.len();

            // Apply drop strategy
            let n_output_cats = match &self.drop {
                Some(strategy) if strategy == "first" => n_cats.saturating_sub(1),
                Some(strategy) if strategy == "if_binary" && n_cats == 2 => 1,
                _ => n_cats,
            };

            if n_output_cats == 0 {
                return Err(TransformError::InvalidInput(format!(
                    "Feature {} has only one category after dropping",
                    j
                )));
            }

            total_features += n_output_cats;
        }

        let mut transformed = Array2::zeros((n_samples, total_features));

        // Create mappings from category values to column indices
        let mut category_mappings = Vec::new();
        let mut current_col = 0;

        for feature_categories in categories.iter() {
            let mut mapping = HashMap::new();
            let n_cats = feature_categories.len();

            // Determine how many categories to keep
            let (start_idx, n_output_cats) = match &self.drop {
                Some(strategy) if strategy == "first" => (1, n_cats.saturating_sub(1)),
                Some(strategy) if strategy == "if_binary" && n_cats == 2 => (0, 1),
                _ => (0, n_cats),
            };

            for (cat_idx, &category) in feature_categories.iter().enumerate() {
                if cat_idx >= start_idx && cat_idx < start_idx + n_output_cats {
                    mapping.insert(category, current_col + cat_idx - start_idx);
                }
            }

            category_mappings.push(mapping);
            current_col += n_output_cats;
        }

        // Fill the transformed array
        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x_u64[[i, j]];

                if let Some(&col_idx) = category_mappings[j].get(&value) {
                    transformed[[i, col_idx]] = 1.0;
                } else {
                    // Check if this is a dropped category (which should be represented as all zeros)
                    let feature_categories = &categories[j];
                    let is_dropped_category = match &self.drop {
                        Some(strategy) if strategy == "first" => {
                            // If it's the first category in the sorted list, it was dropped
                            !feature_categories.is_empty() && value == feature_categories[0]
                        }
                        Some(strategy)
                            if strategy == "if_binary" && feature_categories.len() == 2 =>
                        {
                            // If it's the second category (index 1) in a binary feature, it was dropped
                            feature_categories.len() == 2 && value == feature_categories[1]
                        }
                        _ => false,
                    };

                    if !is_dropped_category && self.handle_unknown == "error" {
                        return Err(TransformError::InvalidInput(format!(
                            "Found unknown category {} in feature {}",
                            value, j
                        )));
                    }
                    // If it's a dropped category or handle_unknown == "ignore", we just leave it as 0
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the OneHotEncoder to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The one-hot encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the categories for each feature
    ///
    /// # Returns
    /// * `Option<&Vec<Vec<u64>>>` - The categories for each feature
    pub fn categories(&self) -> Option<&Vec<Vec<u64>>> {
        self.categories_.as_ref()
    }

    /// Gets the feature names for the transformed output
    ///
    /// # Arguments
    /// * `input_features` - Names of input features
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - Names of output features
    pub fn get_feature_names(&self, input_features: Option<&[String]>) -> Result<Vec<String>> {
        if self.categories_.is_none() {
            return Err(TransformError::TransformationError(
                "OneHotEncoder has not been fitted".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().unwrap();
        let mut feature_names = Vec::new();

        for (j, feature_categories) in categories.iter().enumerate() {
            let feature_name = if let Some(names) = input_features {
                if j < names.len() {
                    names[j].clone()
                } else {
                    format!("x{}", j)
                }
            } else {
                format!("x{}", j)
            };

            let n_cats = feature_categories.len();

            // Determine which categories to include based on drop strategy
            let (start_idx, n_output_cats) = match &self.drop {
                Some(strategy) if strategy == "first" => (1, n_cats.saturating_sub(1)),
                Some(strategy) if strategy == "if_binary" && n_cats == 2 => (0, 1),
                _ => (0, n_cats),
            };

            for &category in feature_categories
                .iter()
                .skip(start_idx)
                .take(n_output_cats)
            {
                feature_names.push(format!("{}_cat_{}", feature_name, category));
            }
        }

        Ok(feature_names)
    }
}

/// OrdinalEncoder for converting categorical features to ordinal integers
///
/// This transformer converts categorical features into ordinal integers,
/// where each category is assigned a unique integer.
pub struct OrdinalEncoder {
    /// Categories for each feature (learned during fit)
    categories_: Option<Vec<Vec<u64>>>,
    /// How to handle unknown categories
    handle_unknown: String,
    /// Value to use for unknown categories
    unknown_value: Option<f64>,
}

impl OrdinalEncoder {
    /// Creates a new OrdinalEncoder
    ///
    /// # Arguments
    /// * `handle_unknown` - How to handle unknown categories ('error' or 'use_encoded_value')
    /// * `unknown_value` - Value to use for unknown categories (when handle_unknown='use_encoded_value')
    ///
    /// # Returns
    /// * A new OrdinalEncoder instance
    pub fn new(handle_unknown: &str, unknown_value: Option<f64>) -> Result<Self> {
        if handle_unknown != "error" && handle_unknown != "use_encoded_value" {
            return Err(TransformError::InvalidInput(
                "handle_unknown must be 'error' or 'use_encoded_value'".to_string(),
            ));
        }

        if handle_unknown == "use_encoded_value" && unknown_value.is_none() {
            return Err(TransformError::InvalidInput(
                "unknown_value must be specified when handle_unknown='use_encoded_value'"
                    .to_string(),
            ));
        }

        Ok(OrdinalEncoder {
            categories_: None,
            handle_unknown: handle_unknown.to_string(),
            unknown_value,
        })
    }

    /// Creates an OrdinalEncoder with default settings
    pub fn with_defaults() -> Self {
        Self::new("error", None).unwrap()
    }

    /// Fits the OrdinalEncoder to the input data
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut categories = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect unique values for this feature
            let mut unique_values: Vec<u64> = x_u64.column(j).to_vec();
            unique_values.sort_unstable();
            unique_values.dedup();

            categories.push(unique_values);
        }

        self.categories_ = Some(categories);
        Ok(())
    }

    /// Transforms the input data using the fitted OrdinalEncoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The ordinally encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if self.categories_.is_none() {
            return Err(TransformError::TransformationError(
                "OrdinalEncoder has not been fitted".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().unwrap();

        if n_features != categories.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but OrdinalEncoder was fitted with {} features",
                n_features,
                categories.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        // Create mappings from category values to ordinal values
        let mut category_mappings = Vec::new();
        for feature_categories in categories {
            let mut mapping = HashMap::new();
            for (ordinal, &category) in feature_categories.iter().enumerate() {
                mapping.insert(category, ordinal as f64);
            }
            category_mappings.push(mapping);
        }

        // Fill the transformed array
        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x_u64[[i, j]];

                if let Some(&ordinal_value) = category_mappings[j].get(&value) {
                    transformed[[i, j]] = ordinal_value;
                } else if self.handle_unknown == "error" {
                    return Err(TransformError::InvalidInput(format!(
                        "Found unknown category {} in feature {}",
                        value, j
                    )));
                } else {
                    // handle_unknown == "use_encoded_value"
                    transformed[[i, j]] = self.unknown_value.unwrap();
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the OrdinalEncoder to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The ordinally encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the categories for each feature
    ///
    /// # Returns
    /// * `Option<&Vec<Vec<u64>>>` - The categories for each feature
    pub fn categories(&self) -> Option<&Vec<Vec<u64>>> {
        self.categories_.as_ref()
    }
}

/// TargetEncoder for supervised categorical encoding
///
/// This encoder transforms categorical features using the target variable values,
/// encoding each category with a statistic (mean, median, etc.) of the target values
/// for that category. This is useful for high-cardinality categorical features.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// use scirs2_transform::encoding::TargetEncoder;
///
/// let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]).unwrap();
/// let y = vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5];
///
/// let mut encoder = TargetEncoder::new("mean", 1.0, 0.0).unwrap();
/// let encoded = encoder.fit_transform(&x, &y).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TargetEncoder {
    /// Encoding strategy ('mean', 'median', 'count', 'sum')
    strategy: String,
    /// Smoothing parameter for regularization (higher = more smoothing toward global mean)
    smoothing: f64,
    /// Global statistic to use for smoothing and unknown categories
    global_stat: f64,
    /// Mappings from categories to encoded values for each feature
    encodings_: Option<Vec<HashMap<u64, f64>>>,
    /// Whether the encoder has been fitted
    is_fitted: bool,
    /// Global mean of target values (computed during fit)
    global_mean_: f64,
}

impl TargetEncoder {
    /// Creates a new TargetEncoder
    ///
    /// # Arguments
    /// * `strategy` - Encoding strategy ('mean', 'median', 'count', 'sum')
    /// * `smoothing` - Smoothing parameter (0.0 = no smoothing, higher = more smoothing)
    /// * `global_stat` - Global statistic fallback for unknown categories
    ///
    /// # Returns
    /// * A new TargetEncoder instance
    pub fn new(strategy: &str, smoothing: f64, global_stat: f64) -> Result<Self> {
        if !["mean", "median", "count", "sum"].contains(&strategy) {
            return Err(TransformError::InvalidInput(
                "strategy must be 'mean', 'median', 'count', or 'sum'".to_string(),
            ));
        }

        if smoothing < 0.0 {
            return Err(TransformError::InvalidInput(
                "smoothing parameter must be non-negative".to_string(),
            ));
        }

        Ok(TargetEncoder {
            strategy: strategy.to_string(),
            smoothing,
            global_stat,
            encodings_: None,
            is_fitted: false,
            global_mean_: 0.0,
        })
    }

    /// Creates a TargetEncoder with mean strategy and default smoothing
    pub fn with_mean(smoothing: f64) -> Self {
        TargetEncoder {
            strategy: "mean".to_string(),
            smoothing,
            global_stat: 0.0,
            encodings_: None,
            is_fitted: false,
            global_mean_: 0.0,
        }
    }

    /// Creates a TargetEncoder with median strategy
    pub fn with_median(smoothing: f64) -> Self {
        TargetEncoder {
            strategy: "median".to_string(),
            smoothing,
            global_stat: 0.0,
            encodings_: None,
            is_fitted: false,
            global_mean_: 0.0,
        }
    }

    /// Fits the TargetEncoder to the input data and target values
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    /// * `y` - The target values, length n_samples
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>, y: &[f64]) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if y.len() != n_samples {
            return Err(TransformError::InvalidInput(
                "Number of target values must match number of samples".to_string(),
            ));
        }

        // Compute global mean for smoothing
        self.global_mean_ = y.iter().sum::<f64>() / y.len() as f64;

        let mut encodings = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Group target values by category for this feature
            let mut category_targets: HashMap<u64, Vec<f64>> = HashMap::new();

            for i in 0..n_samples {
                let category = x_u64[[i, j]];
                category_targets.entry(category).or_default().push(y[i]);
            }

            // Compute encoding for each category
            let mut category_encoding = HashMap::new();

            for (category, targets) in category_targets.iter() {
                let encoded_value = match self.strategy.as_str() {
                    "mean" => {
                        let category_mean = targets.iter().sum::<f64>() / targets.len() as f64;
                        let count = targets.len() as f64;

                        // Apply smoothing: (count * category_mean + smoothing * global_mean) / (count + smoothing)
                        if self.smoothing > 0.0 {
                            (count * category_mean + self.smoothing * self.global_mean_)
                                / (count + self.smoothing)
                        } else {
                            category_mean
                        }
                    }
                    "median" => {
                        let mut sorted_targets = targets.clone();
                        sorted_targets.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let median = if sorted_targets.len() % 2 == 0 {
                            let mid = sorted_targets.len() / 2;
                            (sorted_targets[mid - 1] + sorted_targets[mid]) / 2.0
                        } else {
                            sorted_targets[sorted_targets.len() / 2]
                        };

                        // Apply smoothing toward global mean
                        if self.smoothing > 0.0 {
                            let count = targets.len() as f64;
                            (count * median + self.smoothing * self.global_mean_)
                                / (count + self.smoothing)
                        } else {
                            median
                        }
                    }
                    "count" => targets.len() as f64,
                    "sum" => targets.iter().sum::<f64>(),
                    _ => unreachable!(),
                };

                category_encoding.insert(*category, encoded_value);
            }

            encodings.push(category_encoding);
        }

        self.encodings_ = Some(encodings);
        self.is_fitted = true;
        Ok(())
    }

    /// Transforms the input data using the fitted TargetEncoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The target-encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.is_fitted {
            return Err(TransformError::TransformationError(
                "TargetEncoder has not been fitted".to_string(),
            ));
        }

        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        let encodings = self.encodings_.as_ref().unwrap();

        if n_features != encodings.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but TargetEncoder was fitted with {} features",
                n_features,
                encodings.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let category = x_u64[[i, j]];

                if let Some(&encoded_value) = encodings[j].get(&category) {
                    transformed[[i, j]] = encoded_value;
                } else {
                    // Use global statistic for unknown categories
                    transformed[[i, j]] = if self.global_stat != 0.0 {
                        self.global_stat
                    } else {
                        self.global_mean_
                    };
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the TargetEncoder and transforms the data in one step
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    /// * `y` - The target values, length n_samples
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The target-encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>, y: &[f64]) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Returns the learned encodings for each feature
    ///
    /// # Returns
    /// * `Option<&Vec<HashMap<u64, f64>>>` - The category encodings for each feature
    pub fn encodings(&self) -> Option<&Vec<HashMap<u64, f64>>> {
        self.encodings_.as_ref()
    }

    /// Returns whether the encoder has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Returns the global mean computed during fitting
    pub fn global_mean(&self) -> f64 {
        self.global_mean_
    }

    /// Applies cross-validation target encoding to prevent overfitting
    ///
    /// This method performs k-fold cross-validation to compute target encodings,
    /// which helps prevent overfitting when the same data is used for both
    /// fitting and transforming.
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    /// * `y` - The target values, length n_samples
    /// * `cv_folds` - Number of cross-validation folds (default: 5)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The cross-validated target-encoded data
    pub fn fit_transform_cv<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &[f64],
        cv_folds: usize,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if y.len() != n_samples {
            return Err(TransformError::InvalidInput(
                "Number of target values must match number of samples".to_string(),
            ));
        }

        if cv_folds < 2 {
            return Err(TransformError::InvalidInput(
                "cv_folds must be at least 2".to_string(),
            ));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        // Compute global mean
        self.global_mean_ = y.iter().sum::<f64>() / y.len() as f64;

        // Create fold indices
        let fold_size = n_samples / cv_folds;
        let mut fold_indices = Vec::new();
        for fold in 0..cv_folds {
            let start = fold * fold_size;
            let end = if fold == cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };
            fold_indices.push((start, end));
        }

        // For each fold, train on other folds and predict on this fold
        for fold in 0..cv_folds {
            let (val_start, val_end) = fold_indices[fold];

            // Collect training data (all folds except current)
            let mut train_indices = Vec::new();
            for (other_fold, &(start, end)) in fold_indices.iter().enumerate().take(cv_folds) {
                if other_fold != fold {
                    train_indices.extend(start..end);
                }
            }

            // For each feature, compute encodings on training data
            for j in 0..n_features {
                let mut category_targets: HashMap<u64, Vec<f64>> = HashMap::new();

                // Collect target values by category for training data
                for &train_idx in &train_indices {
                    let category = x_u64[[train_idx, j]];
                    category_targets
                        .entry(category)
                        .or_default()
                        .push(y[train_idx]);
                }

                // Compute encodings for this fold
                let mut category_encoding = HashMap::new();
                for (category, targets) in category_targets.iter() {
                    let encoded_value = match self.strategy.as_str() {
                        "mean" => {
                            let category_mean = targets.iter().sum::<f64>() / targets.len() as f64;
                            let count = targets.len() as f64;

                            if self.smoothing > 0.0 {
                                (count * category_mean + self.smoothing * self.global_mean_)
                                    / (count + self.smoothing)
                            } else {
                                category_mean
                            }
                        }
                        "median" => {
                            let mut sorted_targets = targets.clone();
                            sorted_targets.sort_by(|a, b| a.partial_cmp(b).unwrap());

                            let median = if sorted_targets.len() % 2 == 0 {
                                let mid = sorted_targets.len() / 2;
                                (sorted_targets[mid - 1] + sorted_targets[mid]) / 2.0
                            } else {
                                sorted_targets[sorted_targets.len() / 2]
                            };

                            if self.smoothing > 0.0 {
                                let count = targets.len() as f64;
                                (count * median + self.smoothing * self.global_mean_)
                                    / (count + self.smoothing)
                            } else {
                                median
                            }
                        }
                        "count" => targets.len() as f64,
                        "sum" => targets.iter().sum::<f64>(),
                        _ => unreachable!(),
                    };

                    category_encoding.insert(*category, encoded_value);
                }

                // Apply encodings to validation fold
                for val_idx in val_start..val_end {
                    let category = x_u64[[val_idx, j]];

                    if let Some(&encoded_value) = category_encoding.get(&category) {
                        transformed[[val_idx, j]] = encoded_value;
                    } else {
                        // Use global mean for unknown categories
                        transformed[[val_idx, j]] = self.global_mean_;
                    }
                }
            }
        }

        // Now fit on the full data for future transforms
        self.fit(x, y)?;

        Ok(transformed)
    }
}

/// BinaryEncoder for converting categorical features to binary representations
///
/// This transformer converts categorical features into binary representations,
/// where each category is encoded as a unique binary number. This is more
/// memory-efficient than one-hot encoding for high-cardinality categorical features.
///
/// For n unique categories, ceil(log2(n)) binary features are created.
#[derive(Debug, Clone)]
pub struct BinaryEncoder {
    /// Mappings from categories to binary codes for each feature
    categories_: Option<Vec<HashMap<u64, Vec<u8>>>>,
    /// Number of binary features per original feature
    n_binary_features_: Option<Vec<usize>>,
    /// Whether to handle unknown categories
    handle_unknown: String,
    /// Whether the encoder has been fitted
    is_fitted: bool,
}

impl BinaryEncoder {
    /// Creates a new BinaryEncoder
    ///
    /// # Arguments
    /// * `handle_unknown` - How to handle unknown categories ('error' or 'ignore')
    ///   - 'error': Raise an error if unknown categories are encountered
    ///   - 'ignore': Encode unknown categories as all zeros
    ///
    /// # Returns
    /// * `Result<BinaryEncoder>` - The new encoder instance
    pub fn new(handle_unknown: &str) -> Result<Self> {
        if handle_unknown != "error" && handle_unknown != "ignore" {
            return Err(TransformError::InvalidInput(
                "handle_unknown must be 'error' or 'ignore'".to_string(),
            ));
        }

        Ok(BinaryEncoder {
            categories_: None,
            n_binary_features_: None,
            handle_unknown: handle_unknown.to_string(),
            is_fitted: false,
        })
    }

    /// Creates a BinaryEncoder with default settings (handle_unknown='error')
    pub fn with_defaults() -> Self {
        Self::new("error").unwrap()
    }

    /// Fits the BinaryEncoder to the input data
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut categories = Vec::with_capacity(n_features);
        let mut n_binary_features = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect unique categories for this feature
            let mut unique_categories: Vec<u64> = x_u64.column(j).to_vec();
            unique_categories.sort_unstable();
            unique_categories.dedup();

            if unique_categories.is_empty() {
                return Err(TransformError::InvalidInput(
                    "Feature has no valid categories".to_string(),
                ));
            }

            // Calculate number of binary features needed
            let n_cats = unique_categories.len();
            let n_bits = if n_cats <= 1 {
                1
            } else {
                (n_cats as f64).log2().ceil() as usize
            };

            // Create binary mappings
            let mut category_map = HashMap::new();
            for (idx, &category) in unique_categories.iter().enumerate() {
                let binary_code = Self::int_to_binary(idx, n_bits);
                category_map.insert(category, binary_code);
            }

            categories.push(category_map);
            n_binary_features.push(n_bits);
        }

        self.categories_ = Some(categories);
        self.n_binary_features_ = Some(n_binary_features);
        self.is_fitted = true;

        Ok(())
    }

    /// Transforms the input data using the fitted encoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data to transform
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The binary-encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.is_fitted {
            return Err(TransformError::InvalidInput(
                "Encoder has not been fitted yet".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().unwrap();
        let n_binary_features = self.n_binary_features_.as_ref().unwrap();

        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_features != categories.len() {
            return Err(TransformError::InvalidInput(format!(
                "Number of features ({}) does not match fitted features ({})",
                n_features,
                categories.len()
            )));
        }

        // Calculate total number of output features
        let total_binary_features: usize = n_binary_features.iter().sum();
        let mut result = Array2::<f64>::zeros((n_samples, total_binary_features));

        let mut output_col = 0;
        for j in 0..n_features {
            let category_map = &categories[j];
            let n_bits = n_binary_features[j];

            for i in 0..n_samples {
                let category = x_u64[[i, j]];

                if let Some(binary_code) = category_map.get(&category) {
                    // Known category: use binary code
                    for (bit_idx, &bit_val) in binary_code.iter().enumerate() {
                        result[[i, output_col + bit_idx]] = bit_val as f64;
                    }
                } else {
                    // Unknown category
                    match self.handle_unknown.as_str() {
                        "error" => {
                            return Err(TransformError::InvalidInput(format!(
                                "Unknown category {} in feature {}",
                                category, j
                            )));
                        }
                        "ignore" => {
                            // Set all bits to zero (already initialized)
                        }
                        _ => unreachable!(),
                    }
                }
            }

            output_col += n_bits;
        }

        Ok(result)
    }

    /// Fits the encoder and transforms the data in one step
    ///
    /// # Arguments
    /// * `x` - The input categorical data
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The binary-encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns whether the encoder has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Returns the category mappings if fitted
    pub fn categories(&self) -> Option<&Vec<HashMap<u64, Vec<u8>>>> {
        self.categories_.as_ref()
    }

    /// Returns the number of binary features per original feature
    pub fn n_binary_features(&self) -> Option<&Vec<usize>> {
        self.n_binary_features_.as_ref()
    }

    /// Returns the total number of output features
    pub fn n_output_features(&self) -> Option<usize> {
        self.n_binary_features_.as_ref().map(|v| v.iter().sum())
    }

    /// Converts an integer to binary representation
    fn int_to_binary(value: usize, n_bits: usize) -> Vec<u8> {
        let mut binary = Vec::with_capacity(n_bits);
        let mut val = value;

        for _ in 0..n_bits {
            binary.push((val & 1) as u8);
            val >>= 1;
        }

        binary.reverse(); // Most significant bit first
        binary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_one_hot_encoder_basic() {
        // Create test data with categorical values
        let data = Array::from_shape_vec(
            (4, 2),
            vec![
                0.0, 1.0, // categories: [0, 1, 2] and [1, 2, 3]
                1.0, 2.0, 2.0, 3.0, 0.0, 1.0,
            ],
        )
        .unwrap();

        let mut encoder = OneHotEncoder::with_defaults();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Should have 3 + 3 = 6 output features
        assert_eq!(encoded.shape(), &[4, 6]);

        // Check first row: category 0 in feature 0, category 1 in feature 1
        assert_abs_diff_eq!(encoded[[0, 0]], 1.0, epsilon = 1e-10); // cat 0, feature 0
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10); // cat 1, feature 0
        assert_abs_diff_eq!(encoded[[0, 2]], 0.0, epsilon = 1e-10); // cat 2, feature 0
        assert_abs_diff_eq!(encoded[[0, 3]], 1.0, epsilon = 1e-10); // cat 1, feature 1
        assert_abs_diff_eq!(encoded[[0, 4]], 0.0, epsilon = 1e-10); // cat 2, feature 1
        assert_abs_diff_eq!(encoded[[0, 5]], 0.0, epsilon = 1e-10); // cat 3, feature 1

        // Check second row: category 1 in feature 0, category 2 in feature 1
        assert_abs_diff_eq!(encoded[[1, 0]], 0.0, epsilon = 1e-10); // cat 0, feature 0
        assert_abs_diff_eq!(encoded[[1, 1]], 1.0, epsilon = 1e-10); // cat 1, feature 0
        assert_abs_diff_eq!(encoded[[1, 2]], 0.0, epsilon = 1e-10); // cat 2, feature 0
        assert_abs_diff_eq!(encoded[[1, 3]], 0.0, epsilon = 1e-10); // cat 1, feature 1
        assert_abs_diff_eq!(encoded[[1, 4]], 1.0, epsilon = 1e-10); // cat 2, feature 1
        assert_abs_diff_eq!(encoded[[1, 5]], 0.0, epsilon = 1e-10); // cat 3, feature 1
    }

    #[test]
    fn test_one_hot_encoder_drop_first() {
        // Create test data with categorical values
        let data = Array::from_shape_vec((3, 2), vec![0.0, 1.0, 1.0, 2.0, 2.0, 1.0]).unwrap();

        let mut encoder = OneHotEncoder::new(Some("first".to_string()), "error", false).unwrap();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Should have (3-1) + (2-1) = 3 output features (dropped first category of each)
        assert_eq!(encoded.shape(), &[3, 3]);

        // Categories: feature 0: [0, 1, 2] -> keep [1, 2]
        //            feature 1: [1, 2] -> keep [2]

        // First row: category 0 (dropped), category 1 (dropped)
        assert_abs_diff_eq!(encoded[[0, 0]], 0.0, epsilon = 1e-10); // cat 1, feature 0
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10); // cat 2, feature 0
        assert_abs_diff_eq!(encoded[[0, 2]], 0.0, epsilon = 1e-10); // cat 2, feature 1

        // Second row: category 1, category 2
        assert_abs_diff_eq!(encoded[[1, 0]], 1.0, epsilon = 1e-10); // cat 1, feature 0
        assert_abs_diff_eq!(encoded[[1, 1]], 0.0, epsilon = 1e-10); // cat 2, feature 0
        assert_abs_diff_eq!(encoded[[1, 2]], 1.0, epsilon = 1e-10); // cat 2, feature 1
    }

    #[test]
    fn test_ordinal_encoder() {
        // Create test data with categorical values
        let data = Array::from_shape_vec(
            (4, 2),
            vec![
                2.0, 10.0, // categories will be mapped to ordinals
                1.0, 20.0, 3.0, 10.0, 2.0, 30.0,
            ],
        )
        .unwrap();

        let mut encoder = OrdinalEncoder::with_defaults();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Should preserve shape
        assert_eq!(encoded.shape(), &[4, 2]);

        // Categories for feature 0: [1, 2, 3] -> ordinals [0, 1, 2]
        // Categories for feature 1: [10, 20, 30] -> ordinals [0, 1, 2]

        // Check mappings
        assert_abs_diff_eq!(encoded[[0, 0]], 1.0, epsilon = 1e-10); // 2 -> ordinal 1
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10); // 10 -> ordinal 0
        assert_abs_diff_eq!(encoded[[1, 0]], 0.0, epsilon = 1e-10); // 1 -> ordinal 0
        assert_abs_diff_eq!(encoded[[1, 1]], 1.0, epsilon = 1e-10); // 20 -> ordinal 1
        assert_abs_diff_eq!(encoded[[2, 0]], 2.0, epsilon = 1e-10); // 3 -> ordinal 2
        assert_abs_diff_eq!(encoded[[2, 1]], 0.0, epsilon = 1e-10); // 10 -> ordinal 0
        assert_abs_diff_eq!(encoded[[3, 0]], 1.0, epsilon = 1e-10); // 2 -> ordinal 1
        assert_abs_diff_eq!(encoded[[3, 1]], 2.0, epsilon = 1e-10); // 30 -> ordinal 2
    }

    #[test]
    fn test_unknown_category_handling() {
        let train_data = Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();

        let test_data = Array::from_shape_vec(
            (1, 1),
            vec![3.0], // Unknown category
        )
        .unwrap();

        // Test error handling
        let mut encoder = OneHotEncoder::with_defaults(); // with_defaults is handle_unknown="error"
        encoder.fit(&train_data).unwrap();
        assert!(encoder.transform(&test_data).is_err());

        // Test ignore handling
        let mut encoder = OneHotEncoder::new(None, "ignore", false).unwrap();
        encoder.fit(&train_data).unwrap();
        let encoded = encoder.transform(&test_data).unwrap();

        // Should be all zeros (ignored unknown category)
        assert_eq!(encoded.shape(), &[1, 2]);
        assert_abs_diff_eq!(encoded[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ordinal_encoder_unknown_value() {
        let train_data = Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();

        let test_data = Array::from_shape_vec(
            (1, 1),
            vec![3.0], // Unknown category
        )
        .unwrap();

        let mut encoder = OrdinalEncoder::new("use_encoded_value", Some(-1.0)).unwrap();
        encoder.fit(&train_data).unwrap();
        let encoded = encoder.transform(&test_data).unwrap();

        // Should use the specified unknown value
        assert_eq!(encoded.shape(), &[1, 1]);
        assert_abs_diff_eq!(encoded[[0, 0]], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_get_feature_names() {
        let data = Array::from_shape_vec((2, 2), vec![1.0, 10.0, 2.0, 20.0]).unwrap();

        let mut encoder = OneHotEncoder::with_defaults();
        encoder.fit(&data).unwrap();

        let feature_names = encoder.get_feature_names(None).unwrap();
        assert_eq!(feature_names.len(), 4); // 2 cats per feature * 2 features

        let custom_names = vec!["feat_a".to_string(), "feat_b".to_string()];
        let feature_names = encoder.get_feature_names(Some(&custom_names)).unwrap();
        assert!(feature_names[0].starts_with("feat_a_cat_"));
        assert!(feature_names[2].starts_with("feat_b_cat_"));
    }

    #[test]
    fn test_target_encoder_mean_strategy() {
        // Create test data
        let x = Array::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]).unwrap();
        let y = vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5];

        let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).unwrap();
        let encoded = encoder.fit_transform(&x, &y).unwrap();

        // Should preserve shape
        assert_eq!(encoded.shape(), &[6, 1]);

        // Check category encodings:
        // Category 0: targets [1.0, 1.5] -> mean = 1.25
        // Category 1: targets [2.0, 2.5] -> mean = 2.25
        // Category 2: targets [3.0, 3.5] -> mean = 3.25

        assert_abs_diff_eq!(encoded[[0, 0]], 1.25, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[1, 0]], 2.25, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[2, 0]], 3.25, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[3, 0]], 1.25, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[4, 0]], 2.25, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[5, 0]], 3.25, epsilon = 1e-10);

        // Check global mean
        assert_abs_diff_eq!(encoder.global_mean(), 2.25, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_median_strategy() {
        let x = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0];

        let mut encoder = TargetEncoder::new("median", 0.0, 0.0).unwrap();
        let encoded = encoder.fit_transform(&x, &y).unwrap();

        // Category 0: targets [1.0, 3.0] -> median = 2.0
        // Category 1: targets [2.0, 4.0] -> median = 3.0

        assert_abs_diff_eq!(encoded[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[1, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[2, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[3, 0]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_count_strategy() {
        let x = Array::from_shape_vec((5, 1), vec![0.0, 1.0, 0.0, 2.0, 1.0]).unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut encoder = TargetEncoder::new("count", 0.0, 0.0).unwrap();
        let encoded = encoder.fit_transform(&x, &y).unwrap();

        // Category 0: appears 2 times
        // Category 1: appears 2 times
        // Category 2: appears 1 time

        assert_abs_diff_eq!(encoded[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[2, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[3, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[4, 0]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_sum_strategy() {
        let x = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0];

        let mut encoder = TargetEncoder::new("sum", 0.0, 0.0).unwrap();
        let encoded = encoder.fit_transform(&x, &y).unwrap();

        // Category 0: targets [1.0, 3.0] -> sum = 4.0
        // Category 1: targets [2.0, 4.0] -> sum = 6.0

        assert_abs_diff_eq!(encoded[[0, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[1, 0]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[2, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[3, 0]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_smoothing() {
        let x = Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = vec![1.0, 2.0, 3.0];

        let mut encoder = TargetEncoder::new("mean", 1.0, 0.0).unwrap();
        let encoded = encoder.fit_transform(&x, &y).unwrap();

        // Global mean = (1+2+3)/3 = 2.0
        // Category 0: count=1, mean=1.0 -> smoothed = (1*1.0 + 1.0*2.0)/(1+1) = 1.5
        // Category 1: count=1, mean=2.0 -> smoothed = (1*2.0 + 1.0*2.0)/(1+1) = 2.0
        // Category 2: count=1, mean=3.0 -> smoothed = (1*3.0 + 1.0*2.0)/(1+1) = 2.5

        assert_abs_diff_eq!(encoded[[0, 0]], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[2, 0]], 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_unknown_categories() {
        let train_x = Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let train_y = vec![1.0, 2.0, 3.0];

        let test_x = Array::from_shape_vec((2, 1), vec![3.0, 4.0]).unwrap(); // Unknown categories

        let mut encoder = TargetEncoder::new("mean", 0.0, -1.0).unwrap();
        encoder.fit(&train_x, &train_y).unwrap();
        let encoded = encoder.transform(&test_x).unwrap();

        // Should use global_stat for unknown categories
        assert_abs_diff_eq!(encoded[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[1, 0]], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_unknown_categories_global_mean() {
        let train_x = Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let train_y = vec![1.0, 2.0, 3.0];

        let test_x = Array::from_shape_vec((1, 1), vec![3.0]).unwrap(); // Unknown category

        let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).unwrap(); // global_stat = 0.0
        encoder.fit(&train_x, &train_y).unwrap();
        let encoded = encoder.transform(&test_x).unwrap();

        // Should use global_mean for unknown categories when global_stat == 0.0
        assert_abs_diff_eq!(encoded[[0, 0]], 2.0, epsilon = 1e-10); // Global mean = 2.0
    }

    #[test]
    fn test_target_encoder_multi_feature() {
        let x =
            Array::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0];

        let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).unwrap();
        let encoded = encoder.fit_transform(&x, &y).unwrap();

        assert_eq!(encoded.shape(), &[4, 2]);

        // Feature 0: Category 0 -> targets [1.0, 3.0] -> mean = 2.0
        //           Category 1 -> targets [2.0, 4.0] -> mean = 3.0
        // Feature 1: Category 0 -> targets [1.0, 4.0] -> mean = 2.5
        //           Category 1 -> targets [2.0, 3.0] -> mean = 2.5

        assert_abs_diff_eq!(encoded[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[0, 1]], 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[1, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[1, 1]], 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_cross_validation() {
        let x = Array::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )
        .unwrap();
        let y = vec![1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.3, 2.3, 1.1, 2.1];

        let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).unwrap();
        let encoded = encoder.fit_transform_cv(&x, &y, 5).unwrap();

        // Should have same shape
        assert_eq!(encoded.shape(), &[10, 1]);

        // Results should be reasonable (not exact due to CV)
        // All category 0 samples should get similar values
        // All category 1 samples should get similar values
        assert!(encoded[[0, 0]] < encoded[[1, 0]]); // Category 0 < Category 1
        assert!(encoded[[2, 0]] < encoded[[3, 0]]);
    }

    #[test]
    fn test_target_encoder_convenience_methods() {
        let _x = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        let _y = vec![1.0, 2.0, 3.0, 4.0];

        let encoder1 = TargetEncoder::with_mean(1.0);
        assert_eq!(encoder1.strategy, "mean");
        assert_abs_diff_eq!(encoder1.smoothing, 1.0, epsilon = 1e-10);

        let encoder2 = TargetEncoder::with_median(0.5);
        assert_eq!(encoder2.strategy, "median");
        assert_abs_diff_eq!(encoder2.smoothing, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_target_encoder_validation_errors() {
        // Invalid strategy
        assert!(TargetEncoder::new("invalid", 0.0, 0.0).is_err());

        // Negative smoothing
        assert!(TargetEncoder::new("mean", -1.0, 0.0).is_err());

        // Mismatched target length
        let x = Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = vec![1.0, 2.0]; // Wrong length

        let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).unwrap();
        assert!(encoder.fit(&x, &y).is_err());

        // Transform before fit
        let encoder2 = TargetEncoder::new("mean", 0.0, 0.0).unwrap();
        assert!(encoder2.transform(&x).is_err());

        // Wrong number of features
        let train_x = Array::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let test_x = Array::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let train_y = vec![1.0, 2.0];

        let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).unwrap();
        encoder.fit(&train_x, &train_y).unwrap();
        assert!(encoder.transform(&test_x).is_err());

        // Invalid CV folds
        let x = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).unwrap();
        assert!(encoder.fit_transform_cv(&x, &y, 1).is_err()); // cv_folds < 2
    }

    #[test]
    fn test_target_encoder_accessors() {
        let x = Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = vec![1.0, 2.0, 3.0];

        let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).unwrap();

        assert!(!encoder.is_fitted());
        assert!(encoder.encodings().is_none());

        encoder.fit(&x, &y).unwrap();

        assert!(encoder.is_fitted());
        assert!(encoder.encodings().is_some());
        assert_abs_diff_eq!(encoder.global_mean(), 2.0, epsilon = 1e-10);

        let encodings = encoder.encodings().unwrap();
        assert_eq!(encodings.len(), 1); // 1 feature
        assert_eq!(encodings[0].len(), 3); // 3 categories
    }

    #[test]
    fn test_target_encoder_empty_data() {
        let empty_x = Array2::<f64>::zeros((0, 1));
        let empty_y = vec![];

        let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).unwrap();
        assert!(encoder.fit(&empty_x, &empty_y).is_err());
    }

    // ===== BinaryEncoder Tests =====

    #[test]
    fn test_binary_encoder_basic() {
        // Test basic binary encoding with 4 categories (needs 2 bits)
        let data = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();

        let mut encoder = BinaryEncoder::with_defaults();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Should have 2 binary features (ceil(log2(4)) = 2)
        assert_eq!(encoded.shape(), &[4, 2]);

        // Check binary codes: 0=00, 1=01, 2=10, 3=11
        assert_abs_diff_eq!(encoded[[0, 0]], 0.0, epsilon = 1e-10); // 0 -> 00
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[1, 0]], 0.0, epsilon = 1e-10); // 1 -> 01
        assert_abs_diff_eq!(encoded[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[2, 0]], 1.0, epsilon = 1e-10); // 2 -> 10
        assert_abs_diff_eq!(encoded[[2, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[3, 0]], 1.0, epsilon = 1e-10); // 3 -> 11
        assert_abs_diff_eq!(encoded[[3, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_binary_encoder_power_of_two() {
        // Test with exactly 8 categories (power of 2)
        let data =
            Array::from_shape_vec((8, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();

        let mut encoder = BinaryEncoder::with_defaults();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Should have 3 binary features (log2(8) = 3)
        assert_eq!(encoded.shape(), &[8, 3]);

        // Check some specific encodings
        assert_abs_diff_eq!(encoded[[0, 0]], 0.0, epsilon = 1e-10); // 0 -> 000
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[0, 2]], 0.0, epsilon = 1e-10);

        assert_abs_diff_eq!(encoded[[7, 0]], 1.0, epsilon = 1e-10); // 7 -> 111
        assert_abs_diff_eq!(encoded[[7, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[7, 2]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_binary_encoder_non_power_of_two() {
        // Test with 5 categories (not power of 2, needs 3 bits)
        let data = Array::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();

        let mut encoder = BinaryEncoder::with_defaults();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Should have 3 binary features (ceil(log2(5)) = 3)
        assert_eq!(encoded.shape(), &[5, 3]);
        assert_eq!(encoder.n_output_features().unwrap(), 3);
    }

    #[test]
    fn test_binary_encoder_single_category() {
        // Test edge case with only 1 category
        let data = Array::from_shape_vec((3, 1), vec![5.0, 5.0, 5.0]).unwrap();

        let mut encoder = BinaryEncoder::with_defaults();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Should have 1 binary feature for single category
        assert_eq!(encoded.shape(), &[3, 1]);
        assert_eq!(encoder.n_output_features().unwrap(), 1);

        // All values should be encoded as 0
        for i in 0..3 {
            assert_abs_diff_eq!(encoded[[i, 0]], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binary_encoder_multi_feature() {
        // Test with multiple features
        let data = Array::from_shape_vec(
            (4, 2),
            vec![
                0.0, 10.0, // Feature 0: [0,1,2] (2 bits), Feature 1: [10,11] (1 bit)
                1.0, 11.0, 2.0, 10.0, 0.0, 11.0,
            ],
        )
        .unwrap();

        let mut encoder = BinaryEncoder::with_defaults();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Feature 0: 3 categories need 2 bits, Feature 1: 2 categories need 1 bit
        // Total: 2 + 1 = 3 features
        assert_eq!(encoded.shape(), &[4, 3]);
        assert_eq!(encoder.n_output_features().unwrap(), 3);

        let n_binary_features = encoder.n_binary_features().unwrap();
        assert_eq!(n_binary_features, &[2, 1]);
    }

    #[test]
    fn test_binary_encoder_separate_fit_transform() {
        let train_data = Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let test_data = Array::from_shape_vec((2, 1), vec![1.0, 0.0]).unwrap();

        let mut encoder = BinaryEncoder::with_defaults();

        // Fit on training data
        encoder.fit(&train_data).unwrap();
        assert!(encoder.is_fitted());

        // Transform test data
        let encoded = encoder.transform(&test_data).unwrap();
        assert_eq!(encoded.shape(), &[2, 2]); // 3 categories need 2 bits

        // Check that mappings are consistent
        let train_encoded = encoder.transform(&train_data).unwrap();
        assert_abs_diff_eq!(encoded[[0, 0]], train_encoded[[1, 0]], epsilon = 1e-10); // Same category 1
        assert_abs_diff_eq!(encoded[[0, 1]], train_encoded[[1, 1]], epsilon = 1e-10);
    }

    #[test]
    fn test_binary_encoder_unknown_categories_error() {
        let train_data = Array::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let test_data = Array::from_shape_vec((1, 1), vec![2.0]).unwrap(); // Unknown category

        let mut encoder = BinaryEncoder::new("error").unwrap();
        encoder.fit(&train_data).unwrap();

        // Should error on unknown category
        assert!(encoder.transform(&test_data).is_err());
    }

    #[test]
    fn test_binary_encoder_unknown_categories_ignore() {
        let train_data = Array::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let test_data = Array::from_shape_vec((1, 1), vec![2.0]).unwrap(); // Unknown category

        let mut encoder = BinaryEncoder::new("ignore").unwrap();
        encoder.fit(&train_data).unwrap();
        let encoded = encoder.transform(&test_data).unwrap();

        // Unknown category should be encoded as all zeros
        assert_eq!(encoded.shape(), &[1, 1]); // 2 categories need 1 bit
        assert_abs_diff_eq!(encoded[[0, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_binary_encoder_categories_accessor() {
        let data = Array::from_shape_vec((3, 1), vec![10.0, 20.0, 30.0]).unwrap();

        let mut encoder = BinaryEncoder::with_defaults();

        // Before fitting
        assert!(!encoder.is_fitted());
        assert!(encoder.categories().is_none());
        assert!(encoder.n_binary_features().is_none());
        assert!(encoder.n_output_features().is_none());

        encoder.fit(&data).unwrap();

        // After fitting
        assert!(encoder.is_fitted());
        assert!(encoder.categories().is_some());
        assert!(encoder.n_binary_features().is_some());
        assert!(encoder.n_output_features().is_some());

        let categories = encoder.categories().unwrap();
        assert_eq!(categories.len(), 1); // 1 feature
        assert_eq!(categories[0].len(), 3); // 3 categories

        // Check that categories are mapped correctly
        let category_map = &categories[0];
        assert!(category_map.contains_key(&10));
        assert!(category_map.contains_key(&20));
        assert!(category_map.contains_key(&30));
    }

    #[test]
    fn test_binary_encoder_int_to_binary() {
        // Test binary conversion utility function
        assert_eq!(BinaryEncoder::int_to_binary(0, 3), vec![0, 0, 0]);
        assert_eq!(BinaryEncoder::int_to_binary(1, 3), vec![0, 0, 1]);
        assert_eq!(BinaryEncoder::int_to_binary(2, 3), vec![0, 1, 0]);
        assert_eq!(BinaryEncoder::int_to_binary(3, 3), vec![0, 1, 1]);
        assert_eq!(BinaryEncoder::int_to_binary(7, 3), vec![1, 1, 1]);

        // Test with different bit lengths
        assert_eq!(BinaryEncoder::int_to_binary(5, 4), vec![0, 1, 0, 1]);
        assert_eq!(BinaryEncoder::int_to_binary(1, 1), vec![1]);
    }

    #[test]
    fn test_binary_encoder_validation_errors() {
        // Invalid handle_unknown parameter
        assert!(BinaryEncoder::new("invalid").is_err());

        // Empty data
        let empty_data = Array2::<f64>::zeros((0, 1));
        let mut encoder = BinaryEncoder::with_defaults();
        assert!(encoder.fit(&empty_data).is_err());

        // Transform before fit
        let data = Array::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let encoder = BinaryEncoder::with_defaults();
        assert!(encoder.transform(&data).is_err());

        // Wrong number of features in transform
        let train_data = Array::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let test_data = Array::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let mut encoder = BinaryEncoder::with_defaults();
        encoder.fit(&train_data).unwrap();
        assert!(encoder.transform(&test_data).is_err());
    }

    #[test]
    fn test_binary_encoder_consistency() {
        // Test that encoding is consistent across multiple calls
        let data = Array::from_shape_vec((4, 1), vec![3.0, 1.0, 4.0, 1.0]).unwrap();

        let mut encoder = BinaryEncoder::with_defaults();
        let encoded1 = encoder.fit_transform(&data).unwrap();
        let encoded2 = encoder.transform(&data).unwrap();

        // Both should be identical
        for i in 0..encoded1.shape()[0] {
            for j in 0..encoded1.shape()[1] {
                assert_abs_diff_eq!(encoded1[[i, j]], encoded2[[i, j]], epsilon = 1e-10);
            }
        }

        // Same categories should have same encoding
        assert_abs_diff_eq!(encoded1[[1, 0]], encoded1[[3, 0]], epsilon = 1e-10); // Both category 1
        assert_abs_diff_eq!(encoded1[[1, 1]], encoded1[[3, 1]], epsilon = 1e-10);
    }

    #[test]
    fn test_binary_encoder_memory_efficiency() {
        // Test that binary encoding is more memory efficient than one-hot
        // For 10 categories: one-hot needs 10 features, binary needs 4 features
        let data = Array::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();

        let mut binary_encoder = BinaryEncoder::with_defaults();
        let binary_encoded = binary_encoder.fit_transform(&data).unwrap();

        let mut onehot_encoder = OneHotEncoder::with_defaults();
        let onehot_encoded = onehot_encoder.fit_transform(&data).unwrap();

        // Binary should use fewer features
        assert_eq!(binary_encoded.shape()[1], 4); // ceil(log2(10)) = 4
        assert_eq!(onehot_encoded.shape()[1], 10); // 10 categories = 10 features
        assert!(binary_encoded.shape()[1] < onehot_encoded.shape()[1]);
    }
}
