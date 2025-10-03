//! Machine learning pipeline integration
//!
//! This module provides integration utilities for common ML frameworks and pipelines:
//! - Model training data preparation
//! - Cross-validation utilities
//! - Feature engineering pipelines
//! - Model evaluation and metrics
//! - Integration with popular ML libraries

use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::SliceRandomExt;
use scirs2_core::random::Uniform;
use serde::{Deserialize, Serialize};

use crate::error::{DatasetsError, Result};
use crate::utils::{BalancingStrategy, CrossValidationFolds, Dataset};

/// Configuration for ML pipeline integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPipelineConfig {
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Default test size for train/test splits
    pub test_size: f64,
    /// Number of folds for cross-validation
    pub cv_folds: usize,
    /// Whether to stratify splits for classification
    pub stratify: bool,
    /// Data balancing strategy
    pub balancing_strategy: Option<BalancingStrategy>,
    /// Feature scaling method
    pub scaling_method: Option<ScalingMethod>,
}

impl Default for MLPipelineConfig {
    fn default() -> Self {
        Self {
            random_state: Some(42),
            test_size: 0.2,
            cv_folds: 5,
            stratify: true,
            balancing_strategy: None,
            scaling_method: Some(ScalingMethod::StandardScaler),
        }
    }
}

/// Feature scaling methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ScalingMethod {
    /// Z-score normalization
    StandardScaler,
    /// Min-max scaling to [0, 1]
    MinMaxScaler,
    /// Robust scaling using median and MAD
    RobustScaler,
    /// No scaling
    None,
}

/// ML pipeline for data preprocessing and preparation
pub struct MLPipeline {
    config: MLPipelineConfig,
    fitted_scalers: Option<HashMap<String, ScalerParams>>,
}

/// Parameters for fitted scalers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalerParams {
    /// Scaling method used
    pub method: ScalingMethod,
    /// Mean value (for StandardScaler)
    pub mean: Option<f64>,
    /// Standard deviation (for StandardScaler)
    pub std: Option<f64>,
    /// Minimum value (for MinMaxScaler)
    pub min: Option<f64>,
    /// Maximum value (for MinMaxScaler)
    pub max: Option<f64>,
    /// Median value (for RobustScaler)
    pub median: Option<f64>,
    /// Median absolute deviation (for RobustScaler)
    pub mad: Option<f64>,
}

/// ML experiment tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLExperiment {
    /// Experiment name
    pub name: String,
    /// Dataset information
    pub dataset_info: DatasetInfo,
    /// Model configuration
    pub model_config: ModelConfig,
    /// Training results
    pub results: ExperimentResults,
    /// Cross-validation scores
    pub cv_scores: Option<CrossValidationResults>,
}

/// Dataset information for experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Number of samples in the dataset
    pub n_samples: usize,
    /// Number of features in the dataset
    pub n_features: usize,
    /// Number of classes (for classification tasks)
    pub n_classes: Option<usize>,
    /// Distribution of classes in the dataset
    pub class_distribution: Option<HashMap<String, usize>>,
    /// Percentage of missing data
    pub missing_data_percentage: f64,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Type of ML model used
    pub model_type: String,
    /// Hyperparameter settings
    pub hyperparameters: HashMap<String, serde_json::Value>,
    /// List of preprocessing steps applied
    pub preprocessing_steps: Vec<String>,
}

/// Experiment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResults {
    /// Score on training data
    pub training_score: f64,
    /// Score on validation data
    pub validation_score: f64,
    /// Score on test data (if available)
    pub test_score: Option<f64>,
    /// Time taken for training (in seconds)
    pub training_time: f64,
    /// Average inference time per sample (in milliseconds)
    pub inference_time: Option<f64>,
    /// Feature importance scores
    pub feature_importance: Option<Vec<(String, f64)>>,
}

/// Cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResults {
    /// Individual scores for each fold
    pub scores: Vec<f64>,
    /// Mean score across all folds
    pub mean_score: f64,
    /// Standard deviation of scores
    pub std_score: f64,
    /// Detailed results for each fold
    pub fold_details: Vec<FoldResult>,
}

/// Result for a single fold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    /// Index of the fold
    pub fold_index: usize,
    /// Training score for this fold
    pub train_score: f64,
    /// Validation score for this fold
    pub validation_score: f64,
    /// Training time in seconds for this fold
    pub training_time: f64,
}

/// Data split for ML training
#[derive(Debug, Clone)]
pub struct DataSplit {
    /// Training features
    pub x_train: Array2<f64>,
    /// Testing features
    pub x_test: Array2<f64>,
    /// Training targets
    pub y_train: Array1<f64>,
    /// Testing targets
    pub y_test: Array1<f64>,
}

impl Default for MLPipeline {
    fn default() -> Self {
        Self::new(MLPipelineConfig::default())
    }
}

impl MLPipeline {
    /// Create a new ML pipeline
    pub fn new(config: MLPipelineConfig) -> Self {
        Self {
            config,
            fitted_scalers: None,
        }
    }

    /// Prepare dataset for ML training
    pub fn prepare_dataset(&mut self, dataset: &Dataset) -> Result<Dataset> {
        let mut prepared = dataset.clone();

        // Apply balancing if specified
        if let Some(ref strategy) = self.config.balancing_strategy {
            prepared = self.apply_balancing(&prepared, strategy)?;
        }

        // Apply scaling if specified
        if let Some(method) = self.config.scaling_method {
            prepared = self.fit_and_transform_scaling(&prepared, method)?;
        }

        Ok(prepared)
    }

    /// Split dataset into train/test sets
    pub fn train_test_split(&self, dataset: &Dataset) -> Result<DataSplit> {
        let n_samples = dataset.n_samples();
        let test_samples = (n_samples as f64 * self.config.test_size) as usize;
        let train_samples = n_samples - test_samples;

        let indices = self.generate_split_indices(n_samples, dataset.target.as_ref())?;

        let train_indices = &indices[..train_samples];
        let test_indices = &indices[train_samples..];

        let x_train = dataset.data.select(Axis(0), train_indices);
        let x_test = dataset.data.select(Axis(0), test_indices);

        let (y_train, y_test) = if let Some(ref target) = dataset.target {
            let y_train = target.select(Axis(0), train_indices);
            let y_test = target.select(Axis(0), test_indices);
            (y_train, y_test)
        } else {
            return Err(DatasetsError::InvalidFormat(
                "Target variable required for train/test split".to_string(),
            ));
        };

        Ok(DataSplit {
            x_train,
            x_test,
            y_train,
            y_test,
        })
    }

    /// Generate cross-validation folds
    pub fn cross_validation_split(&self, dataset: &Dataset) -> Result<CrossValidationFolds> {
        let target = dataset.target.as_ref().ok_or_else(|| {
            DatasetsError::InvalidFormat(
                "Target variable required for cross-validation".to_string(),
            )
        })?;

        if self.config.stratify {
            crate::utils::stratified_k_fold_split(
                target,
                self.config.cv_folds,
                true,
                self.config.random_state,
            )
        } else {
            crate::utils::k_fold_split(
                dataset.n_samples(),
                self.config.cv_folds,
                true,
                self.config.random_state,
            )
        }
    }

    /// Transform new data using fitted scalers
    pub fn transform(&self, dataset: &Dataset) -> Result<Dataset> {
        let scalers = self.fitted_scalers.as_ref().ok_or_else(|| {
            DatasetsError::InvalidFormat(
                "Pipeline not fitted. Call prepare_dataset first.".to_string(),
            )
        })?;

        let mut transformed_data = dataset.data.clone();

        for (col_idx, mut column) in transformed_data.columns_mut().into_iter().enumerate() {
            let defaultname = format!("feature_{col_idx}");
            let featurename = dataset
                .featurenames
                .as_ref()
                .and_then(|names| names.get(col_idx))
                .map(|s| s.as_str())
                .unwrap_or(&defaultname);

            if let Some(scaler) = scalers.get(featurename) {
                Self::apply_scaler_to_column(&mut column, scaler)?;
            }
        }

        Ok(Dataset {
            data: transformed_data,
            target: dataset.target.clone(),
            featurenames: dataset.featurenames.clone(),
            targetnames: dataset.targetnames.clone(),
            feature_descriptions: dataset.feature_descriptions.clone(),
            description: Some("Transformed dataset".to_string()),
            metadata: dataset.metadata.clone(),
        })
    }

    /// Create an ML experiment tracker
    pub fn create_experiment(&self, name: &str, dataset: &Dataset) -> MLExperiment {
        let dataset_info = self.extract_dataset_info(dataset);

        MLExperiment {
            name: name.to_string(),
            dataset_info,
            model_config: ModelConfig {
                model_type: "undefined".to_string(),
                hyperparameters: HashMap::new(),
                preprocessing_steps: Vec::new(),
            },
            results: ExperimentResults {
                training_score: 0.0,
                validation_score: 0.0,
                test_score: None,
                training_time: 0.0,
                inference_time: None,
                feature_importance: None,
            },
            cv_scores: None,
        }
    }

    /// Evaluate model performance with cross-validation
    pub fn evaluate_with_cv<F>(
        &self,
        dataset: &Dataset,
        train_fn: F,
    ) -> Result<CrossValidationResults>
    where
        F: Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>, &Array1<f64>) -> Result<(f64, f64, f64)>,
    {
        let folds = self.cross_validation_split(dataset)?;
        let mut scores = Vec::new();
        let mut fold_details = Vec::new();

        for (fold_idx, (train_indices, val_indices)) in folds.into_iter().enumerate() {
            let x_train = dataset.data.select(Axis(0), &train_indices);
            let x_val = dataset.data.select(Axis(0), &val_indices);

            let target = dataset.target.as_ref().unwrap();
            let y_train = target.select(Axis(0), &train_indices);
            let y_val = target.select(Axis(0), &val_indices);

            let (train_score, val_score, training_time) =
                train_fn(&x_train, &y_train, &x_val, &y_val)?;

            scores.push(val_score);
            fold_details.push(FoldResult {
                fold_index: fold_idx,
                train_score,
                validation_score: val_score,
                training_time,
            });
        }

        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores
            .iter()
            .map(|score| (score - mean_score).powi(2))
            .sum::<f64>()
            / scores.len() as f64;
        let std_score = variance.sqrt();

        Ok(CrossValidationResults {
            scores,
            mean_score,
            std_score,
            fold_details,
        })
    }

    // Private helper methods

    fn apply_balancing(&self, dataset: &Dataset, strategy: &BalancingStrategy) -> Result<Dataset> {
        // Simplified balancing implementation
        // In a full implementation, you'd use the actual balancing utilities
        match strategy {
            BalancingStrategy::RandomUndersample => self.random_undersample(dataset, None),
            BalancingStrategy::RandomOversample => self.random_oversample(dataset, None),
            _ => Ok(dataset.clone()), // Placeholder for other strategies
        }
    }

    fn random_undersample(&self, dataset: &Dataset, _randomstate: Option<u64>) -> Result<Dataset> {
        let target = dataset.target.as_ref().ok_or_else(|| {
            DatasetsError::InvalidFormat("Target required for balancing".to_string())
        })?;

        // Find minority class size
        let mut class_counts: HashMap<i64, usize> = HashMap::new();
        for &value in target.iter() {
            if !value.is_nan() {
                *class_counts.entry(value as i64).or_insert(0) += 1;
            }
        }

        let min_count = class_counts.values().min().copied().unwrap_or(0);

        // Sample min_count samples from each class
        let mut selected_indices = Vec::new();

        for (class_, _count) in class_counts {
            let class_indices: Vec<usize> = target
                .iter()
                .enumerate()
                .filter(|(_, &val)| !val.is_nan() && val as i64 == class_)
                .map(|(idx, _)| idx)
                .collect();

            let mut sampled_indices = class_indices;
            if sampled_indices.len() > min_count {
                // Simple random sampling (in a real implementation, use proper random sampling)
                sampled_indices.truncate(min_count);
            }

            selected_indices.extend(sampled_indices);
        }

        let balanced_data = dataset.data.select(Axis(0), &selected_indices);
        let balanced_target = target.select(Axis(0), &selected_indices);

        Ok(Dataset {
            data: balanced_data,
            target: Some(balanced_target),
            featurenames: dataset.featurenames.clone(),
            targetnames: dataset.targetnames.clone(),
            feature_descriptions: dataset.feature_descriptions.clone(),
            description: Some("Undersampled dataset".to_string()),
            metadata: dataset.metadata.clone(),
        })
    }

    fn random_oversample(&self, dataset: &Dataset, randomstate: Option<u64>) -> Result<Dataset> {
        use scirs2_core::random::prelude::*;
        use scirs2_core::random::{rngs::StdRng, RngCore, SeedableRng};
        use std::collections::HashMap;

        let target = dataset.target.as_ref().ok_or_else(|| {
            DatasetsError::InvalidFormat("Random oversampling requires target labels".to_string())
        })?;

        if target.len() != dataset.data.nrows() {
            return Err(DatasetsError::InvalidFormat(
                "Target length must match number of samples".to_string(),
            ));
        }

        // Count samples per class
        let mut class_counts: HashMap<i32, usize> = HashMap::new();
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();

        for (idx, &label) in target.iter().enumerate() {
            let class = label as i32;
            *class_counts.entry(class).or_insert(0) += 1;
            class_indices.entry(class).or_default().push(idx);
        }

        // Find the majority class size (the maximum count)
        let max_count = class_counts.values().max().copied().unwrap_or(0);

        if max_count == 0 {
            return Err(DatasetsError::InvalidFormat(
                "No samples found in dataset".to_string(),
            ));
        }

        // Create RNG
        let mut rng: Box<dyn RngCore> = match randomstate {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(thread_rng()),
        };

        // Collect all indices for the oversampled dataset
        let mut all_indices = Vec::new();

        for (_class, indices) in class_indices.iter() {
            let current_count = indices.len();

            // Add all original samples
            all_indices.extend(indices.iter().copied());

            // Add additional samples by random oversampling with replacement
            let samples_needed = max_count - current_count;

            if samples_needed > 0 {
                for _ in 0..samples_needed {
                    let random_idx = rng.sample(Uniform::new(0, indices.len()).unwrap());
                    all_indices.push(indices[random_idx]);
                }
            }
        }

        // Shuffle the final indices to mix classes
        all_indices.shuffle(&mut *rng);

        // Create the oversampled dataset
        let oversampled_data = dataset.data.select(Axis(0), &all_indices);
        let oversampled_target = target.select(Axis(0), &all_indices);

        Ok(Dataset {
            data: oversampled_data,
            target: Some(oversampled_target),
            featurenames: dataset.featurenames.clone(),
            targetnames: dataset.targetnames.clone(),
            feature_descriptions: dataset.feature_descriptions.clone(),
            description: Some(format!(
                "Random oversampled dataset (original: {} samples, oversampled: {} samples)",
                dataset.n_samples(),
                all_indices.len()
            )),
            metadata: dataset.metadata.clone(),
        })
    }

    fn fit_and_transform_scaling(
        &mut self,
        dataset: &Dataset,
        method: ScalingMethod,
    ) -> Result<Dataset> {
        let mut scalers = HashMap::new();
        let mut scaled_data = dataset.data.clone();

        for (col_idx, mut column) in scaled_data.columns_mut().into_iter().enumerate() {
            let featurename = dataset
                .featurenames
                .as_ref()
                .and_then(|names| names.get(col_idx))
                .cloned()
                .unwrap_or_else(|| format!("feature_{col_idx}"));

            let column_view = column.view();
            let scaler_params = Self::fit_scaler(&column_view, method)?;
            Self::apply_scaler_to_column(&mut column, &scaler_params)?;

            scalers.insert(featurename, scaler_params);
        }

        self.fitted_scalers = Some(scalers);

        Ok(Dataset {
            data: scaled_data,
            target: dataset.target.clone(),
            featurenames: dataset.featurenames.clone(),
            targetnames: dataset.targetnames.clone(),
            feature_descriptions: dataset.feature_descriptions.clone(),
            description: Some("Scaled dataset".to_string()),
            metadata: dataset.metadata.clone(),
        })
    }

    fn fit_scaler(
        column: &scirs2_core::ndarray::ArrayView1<f64>,
        method: ScalingMethod,
    ) -> Result<ScalerParams> {
        let values: Vec<f64> = column.iter().copied().filter(|x| !x.is_nan()).collect();

        if values.is_empty() {
            return Ok(ScalerParams {
                method,
                mean: None,
                std: None,
                min: None,
                max: None,
                median: None,
                mad: None,
            });
        }

        match method {
            ScalingMethod::StandardScaler => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                let std = variance.sqrt();

                Ok(ScalerParams {
                    method,
                    mean: Some(mean),
                    std: Some(std),
                    min: None,
                    max: None,
                    median: None,
                    mad: None,
                })
            }
            ScalingMethod::MinMaxScaler => {
                let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                Ok(ScalerParams {
                    method,
                    mean: None,
                    std: None,
                    min: Some(min),
                    max: Some(max),
                    median: None,
                    mad: None,
                })
            }
            ScalingMethod::RobustScaler => {
                let mut sorted_values = values.clone();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let median = Self::percentile(&sorted_values, 0.5).unwrap_or(0.0);
                let mad = Self::compute_mad(&sorted_values, median);

                Ok(ScalerParams {
                    method,
                    mean: None,
                    std: None,
                    min: None,
                    max: None,
                    median: Some(median),
                    mad: Some(mad),
                })
            }
            ScalingMethod::None => Ok(ScalerParams {
                method,
                mean: None,
                std: None,
                min: None,
                max: None,
                median: None,
                mad: None,
            }),
        }
    }

    fn apply_scaler_to_column(
        column: &mut scirs2_core::ndarray::ArrayViewMut1<f64>,
        params: &ScalerParams,
    ) -> Result<()> {
        match params.method {
            ScalingMethod::StandardScaler => {
                if let (Some(mean), Some(std)) = (params.mean, params.std) {
                    if std > 1e-8 {
                        // Avoid division by zero
                        for value in column.iter_mut() {
                            if !value.is_nan() {
                                *value = (*value - mean) / std;
                            }
                        }
                    }
                }
            }
            ScalingMethod::MinMaxScaler => {
                if let (Some(min), Some(max)) = (params.min, params.max) {
                    let range = max - min;
                    if range > 1e-8 {
                        // Avoid division by zero
                        for value in column.iter_mut() {
                            if !value.is_nan() {
                                *value = (*value - min) / range;
                            }
                        }
                    }
                }
            }
            ScalingMethod::RobustScaler => {
                if let (Some(median), Some(mad)) = (params.median, params.mad) {
                    if mad > 1e-8 {
                        // Avoid division by zero
                        for value in column.iter_mut() {
                            if !value.is_nan() {
                                *value = (*value - median) / mad;
                            }
                        }
                    }
                }
            }
            ScalingMethod::None => {
                // No scaling applied
            }
        }

        Ok(())
    }

    fn percentile(sorted_values: &[f64], p: f64) -> Option<f64> {
        if sorted_values.is_empty() {
            return None;
        }

        let index = p * (sorted_values.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            Some(sorted_values[lower])
        } else {
            let weight = index - lower as f64;
            Some(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)
        }
    }

    fn compute_mad(sorted_values: &[f64], median: f64) -> f64 {
        let deviations: Vec<f64> = sorted_values.iter().map(|&x| (x - median).abs()).collect();

        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Self::percentile(&sorted_deviations, 0.5).unwrap_or(1.0)
    }

    fn generate_split_indices(
        &self,
        n_samples: usize,
        target: Option<&Array1<f64>>,
    ) -> Result<Vec<usize>> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Use proper random shuffling based on configuration
        if self.config.stratify && target.is_some() {
            // Implement stratified shuffling
            self.stratified_shuffle(&mut indices, target.unwrap())?;
        } else {
            // Regular shuffling with optional random state
            match self.config.random_state {
                Some(seed) => {
                    let mut rng = StdRng::seed_from_u64(seed);
                    indices.shuffle(&mut rng);
                }
                None => {
                    let mut rng = thread_rng();
                    indices.shuffle(&mut rng);
                }
            }
        }

        Ok(indices)
    }

    /// Perform stratified shuffling to maintain class proportions
    fn stratified_shuffle(&self, indices: &mut Vec<usize>, target: &Array1<f64>) -> Result<()> {
        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();

        for &idx in indices.iter() {
            let class = target[idx] as i32;
            class_indices.entry(class).or_default().push(idx);
        }

        // Shuffle each class group separately
        for class_group in class_indices.values_mut() {
            match self.config.random_state {
                Some(seed) => {
                    let mut rng = StdRng::seed_from_u64(seed);
                    class_group.shuffle(&mut rng);
                }
                None => {
                    let mut rng = thread_rng();
                    class_group.shuffle(&mut rng);
                }
            }
        }

        // Recombine shuffled class groups while maintaining order
        indices.clear();
        let mut class_iterators: HashMap<i32, std::vec::IntoIter<usize>> = class_indices
            .into_iter()
            .map(|(class, group)| (class, group.into_iter()))
            .collect();

        // Interleave samples from different classes to maintain distribution
        while !class_iterators.is_empty() {
            let mut to_remove = Vec::new();
            for (&class, iterator) in class_iterators.iter_mut() {
                if let Some(idx) = iterator.next() {
                    indices.push(idx);
                } else {
                    to_remove.push(class);
                }
            }
            for class in to_remove {
                class_iterators.remove(&class);
            }
        }

        Ok(())
    }

    fn extract_dataset_info(&self, dataset: &Dataset) -> DatasetInfo {
        let n_samples = dataset.n_samples();
        let n_features = dataset.n_features();

        let (n_classes, class_distribution) = if let Some(ref target) = dataset.target {
            let mut class_counts: HashMap<String, usize> = HashMap::new();
            for &value in target.iter() {
                if !value.is_nan() {
                    let classname = format!("{value:.0}");
                    *class_counts.entry(classname).or_insert(0) += 1;
                }
            }

            let n_classes = class_counts.len();
            (Some(n_classes), Some(class_counts))
        } else {
            (None, None)
        };

        // Calculate missing data percentage
        let total_values = n_samples * n_features;
        let missing_values = dataset.data.iter().filter(|&&x| x.is_nan()).count();
        let missing_data_percentage = missing_values as f64 / total_values as f64 * 100.0;

        DatasetInfo {
            n_samples,
            n_features,
            n_classes,
            class_distribution,
            missing_data_percentage,
        }
    }
}

/// Convenience functions for ML pipeline integration
pub mod convenience {
    use super::*;

    /// Quick train/test split with default configuration
    pub fn train_test_split(_dataset: &Dataset, testsize: Option<f64>) -> Result<DataSplit> {
        let mut config = MLPipelineConfig::default();
        if let Some(_size) = testsize {
            config.test_size = _size;
        }

        let pipeline = MLPipeline::new(config);
        pipeline.train_test_split(_dataset)
    }

    /// Prepare dataset for ML with standard preprocessing
    pub fn prepare_for_ml(dataset: &Dataset, scale: bool, balance: bool) -> Result<Dataset> {
        let mut config = MLPipelineConfig::default();

        if !scale {
            config.scaling_method = None;
        }

        if balance {
            config.balancing_strategy = Some(BalancingStrategy::RandomUndersample);
        }

        let mut pipeline = MLPipeline::new(config);
        pipeline.prepare_dataset(dataset)
    }

    /// Generate cross-validation folds
    pub fn cv_split(
        dataset: &Dataset,
        n_folds: Option<usize>,
        stratify: Option<bool>,
    ) -> Result<CrossValidationFolds> {
        let mut config = MLPipelineConfig::default();

        if let Some(_folds) = n_folds {
            config.cv_folds = _folds;
        }

        if let Some(strat) = stratify {
            config.stratify = strat;
        }

        let pipeline = MLPipeline::new(config);
        pipeline.cross_validation_split(dataset)
    }

    /// Create a simple ML experiment
    pub fn create_experiment(name: &str, dataset: &Dataset) -> MLExperiment {
        let pipeline = MLPipeline::default();
        pipeline.create_experiment(name, dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::make_classification;
    use scirs2_core::random::Uniform;

    #[test]
    fn test_ml_pipeline_creation() {
        let pipeline = MLPipeline::default();
        assert_eq!(pipeline.config.test_size, 0.2);
        assert_eq!(pipeline.config.cv_folds, 5);
    }

    #[test]
    fn test_train_test_split() {
        let dataset = make_classification(100, 5, 2, 1, 1, Some(42)).unwrap();
        let split = convenience::train_test_split(&dataset, Some(0.3)).unwrap();

        assert_eq!(split.x_train.nrows() + split.x_test.nrows(), 100);
        assert_eq!(split.y_train.len() + split.y_test.len(), 100);
        assert_eq!(split.x_train.ncols(), 5);
        assert_eq!(split.x_test.ncols(), 5);
    }

    #[test]
    fn test_cross_validation_split() {
        let dataset = make_classification(100, 3, 2, 1, 1, Some(42)).unwrap();
        let folds = convenience::cv_split(&dataset, Some(5), Some(true)).unwrap();

        assert_eq!(folds.len(), 5);

        // Check that all samples are used
        let total_samples: usize = folds
            .iter()
            .map(|(train, test)| train.len() + test.len())
            .sum::<usize>()
            / 5; // Each sample appears in exactly one test set

        assert_eq!(total_samples, 100);
    }

    #[test]
    fn test_dataset_preparation() {
        let dataset = make_classification(50, 4, 2, 1, 1, Some(42)).unwrap();
        let prepared = convenience::prepare_for_ml(&dataset, true, false).unwrap();

        assert_eq!(prepared.n_samples(), dataset.n_samples());
        assert_eq!(prepared.n_features(), dataset.n_features());
    }

    #[test]
    fn test_experiment_creation() {
        let dataset = make_classification(100, 5, 2, 1, 1, Some(42)).unwrap();
        let experiment = convenience::create_experiment("test_experiment", &dataset);

        assert_eq!(experiment.name, "test_experiment");
        assert_eq!(experiment.dataset_info.n_samples, 100);
        assert_eq!(experiment.dataset_info.n_features, 5);
        assert_eq!(experiment.dataset_info.n_classes, Some(2));
    }

    #[test]
    fn test_scaler_fitting() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let array = Array1::from_vec(data);
        let view = array.view();

        let scaler_params = MLPipeline::fit_scaler(&view, ScalingMethod::StandardScaler).unwrap();

        assert!(scaler_params.mean.is_some());
        assert!(scaler_params.std.is_some());
        assert_eq!(scaler_params.mean.unwrap(), 3.0);
    }

    #[test]
    fn test_min_max_scaler() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let array = Array1::from_vec(data);
        let view = array.view();

        let scaler_params = MLPipeline::fit_scaler(&view, ScalingMethod::MinMaxScaler).unwrap();

        assert!(scaler_params.min.is_some());
        assert!(scaler_params.max.is_some());
        assert_eq!(scaler_params.min.unwrap(), 1.0);
        assert_eq!(scaler_params.max.unwrap(), 5.0);
    }
}
