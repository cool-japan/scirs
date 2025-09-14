//! Automated feature engineering with meta-learning
//!
//! This module provides automated feature engineering capabilities that use
//! meta-learning to select optimal transformations for given datasets.

use crate::error::{Result, TransformError};
use ndarray::{Array1, ArrayView1, ArrayView2};
use scirs2_core::validation::check_not_empty;
use std::collections::HashMap;

#[cfg(feature = "auto-feature-engineering")]
use std::collections::VecDeque;

use statrs::statistics::Statistics;
#[cfg(feature = "auto-feature-engineering")]
use tch::{nn, Device, Tensor};

/// Meta-features extracted from datasets for transformation selection
#[derive(Debug, Clone)]
pub struct DatasetMetaFeatures {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Sparsity ratio (fraction of zero values)
    pub sparsity: f64,
    /// Mean of feature correlations
    pub mean_correlation: f64,
    /// Standard deviation of feature correlations
    pub std_correlation: f64,
    /// Skewness statistics
    pub mean_skewness: f64,
    /// Kurtosis statistics
    pub mean_kurtosis: f64,
    /// Number of missing values
    pub missing_ratio: f64,
    /// Feature variance statistics
    pub variance_ratio: f64,
    /// Outlier ratio
    pub outlier_ratio: f64,
    /// Whether the dataset has missing values
    pub has_missing: bool,
}

/// Available transformation types for automated selection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TransformationType {
    /// Standardization (Z-score normalization)
    StandardScaler,
    /// Min-max scaling
    MinMaxScaler,
    /// Robust scaling using median and IQR
    RobustScaler,
    /// Power transformation (Box-Cox/Yeo-Johnson)
    PowerTransformer,
    /// Polynomial feature generation
    PolynomialFeatures,
    /// Principal Component Analysis
    PCA,
    /// Feature selection based on variance
    VarianceThreshold,
    /// Quantile transformation
    QuantileTransformer,
    /// Binary encoding for categorical features
    BinaryEncoder,
    /// Target encoding
    TargetEncoder,
}

/// Configuration for a transformation with its parameters
#[derive(Debug, Clone)]
pub struct TransformationConfig {
    /// Type of transformation to apply
    pub transformation_type: TransformationType,
    /// Parameters for the transformation
    pub parameters: HashMap<String, f64>,
    /// Expected performance score for this transformation
    pub expected_performance: f64,
}

/// Meta-learning model for transformation selection
#[cfg(feature = "auto-feature-engineering")]
pub struct MetaLearningModel {
    /// Neural network for predicting transformation performance
    model: nn::Sequential,
    /// Device for computation (CPU/GPU)
    device: Device,
    /// Training data cache
    training_cache: Vec<(DatasetMetaFeatures, Vec<TransformationConfig>)>,
}

#[cfg(feature = "auto-feature-engineering")]
impl MetaLearningModel {
    /// Create a new meta-learning model
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Build neural network architecture
        let model = nn::seq()
            .add(nn::linear(&root / "layer1", 10, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "layer2", 64, 32, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "layer3", 32, 16, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "output", 16, 10, Default::default()))
            .add_fn(|xs| xs.softmax(-1, tch::Kind::Float));

        Ok(MetaLearningModel {
            model,
            device,
            training_cache: Vec::new(),
        })
    }

    /// Train the meta-learning model on historical transformation performance data
    pub fn train(
        &mut self,
        training_data: Vec<(DatasetMetaFeatures, Vec<TransformationConfig>)>,
    ) -> Result<()> {
        self.training_cache.extend(training_data.clone());

        // Convert training _data to tensors
        let (input_features, target_scores) = self.prepare_training_data(&training_data)?;

        // Training loop
        let mut opt = nn::Adam::default().build(&self.model.vs, 1e-3).unwrap();

        for epoch in 0..100 {
            let predicted = self.model.forward(&input_features);
            let loss = predicted.mse_loss(&target_scores, tch::Reduction::Mean);

            opt.zero_grad();
            loss.backward();
            opt.step();

            if epoch % 20 == 0 {
                println!("Epoch {epoch}: Loss = {:.4}", f64::from(loss));
            }
        }

        Ok(())
    }

    /// Predict optimal transformations for a given dataset
    pub fn predict_transformations(
        &self,
        meta_features: &DatasetMetaFeatures,
    ) -> Result<Vec<TransformationConfig>> {
        let input_tensor = self.meta_features_to_tensor(meta_features)?;
        let prediction = self.model.forward(&input_tensor);

        // Convert prediction to transformation recommendations
        self.tensor_to_transformations(&prediction)
    }

    fn prepare_training_data(
        &self,
        training_data: &[(DatasetMetaFeatures, Vec<TransformationConfig>)],
    ) -> Result<(Tensor, Tensor)> {
        if training_data.is_empty() {
            return Err(TransformError::InvalidInput(
                "Training _data cannot be empty".to_string(),
            ));
        }

        let n_samples = training_data.len();
        let mut input_features = Vec::with_capacity(n_samples * 10);
        let mut target_scores = Vec::with_capacity(n_samples * 10);

        for (meta_features, transformations) in training_data {
            // Normalize feature values for better training stability
            let features = vec![
                (meta_features.n_samples as f64).ln().max(0.0), // Log-scale for sample count
                (meta_features.n_features as f64).ln().max(0.0), // Log-scale for feature count
                meta_features.sparsity.clamp(0.0, 1.0),         // Clamp to [0, 1]
                meta_features.mean_correlation.clamp(-1.0, 1.0), // Clamp to [-1, 1]
                meta_features.std_correlation.max(0.0),         // Non-negative
                meta_features.mean_skewness.clamp(-10.0, 10.0), // Reasonable bounds
                meta_features.mean_kurtosis.clamp(-10.0, 10.0), // Reasonable bounds
                meta_features.missing_ratio.clamp(0.0, 1.0),    // Clamp to [0, 1]
                meta_features.variance_ratio.max(0.0),          // Non-negative
                meta_features.outlier_ratio.clamp(0.0, 1.0),    // Clamp to [0, 1]
            ];

            // Validate all features are finite
            if features.iter().any(|&f| !f.is_finite()) {
                return Err(TransformError::ComputationError(
                    "Non-finite values detected in meta-features".to_string(),
                ));
            }

            input_features.extend(features);

            // Create target vector (transformation type scores)
            let mut scores = vec![0.0; 10]; // Number of transformation types
            for config in transformations {
                let idx = self.transformation_type_to_index(&config.transformation_type);
                let performance = config.expected_performance.clamp(0.0, 1.0); // Clamp to [0, 1]
                scores[idx] = scores[idx].max(performance); // Take max if multiple configs for same type
            }
            target_scores.extend(scores);
        }

        let input_tensor = Tensor::of_slice(&input_features)
            .reshape(&[n_samples as i64, 10])
            .to_device(self.device);
        let target_tensor = Tensor::of_slice(&target_scores)
            .reshape(&[n_samples as i64, 10])
            .to_device(self.device);

        Ok((input_tensor, target_tensor))
    }

    fn meta_features_to_tensor(&self, metafeatures: &DatasetMetaFeatures) -> Result<Tensor> {
        // Apply same normalization as in training data preparation
        let _features = vec![
            (meta_features.n_samples as f64).ln().max(0.0),
            (meta_features.n_features as f64).ln().max(0.0),
            meta_features.sparsity.clamp(0.0, 1.0),
            meta_features.mean_correlation.clamp(-1.0, 1.0),
            meta_features.std_correlation.max(0.0),
            meta_features.mean_skewness.clamp(-10.0, 10.0),
            meta_features.mean_kurtosis.clamp(-10.0, 10.0),
            meta_features.missing_ratio.clamp(0.0, 1.0),
            meta_features.variance_ratio.max(0.0),
            meta_features.outlier_ratio.clamp(0.0, 1.0),
        ];

        // Validate all _features are finite
        if features.iter().any(|&f| !f.is_finite()) {
            return Err(TransformError::ComputationError(
                "Non-finite values detected in meta-_features".to_string(),
            ));
        }

        Ok(Tensor::of_slice(&_features)
            .reshape(&[1, 10])
            .to_device(self.device))
    }

    fn tensor_to_transformations(&self, prediction: &Tensor) -> Result<Vec<TransformationConfig>> {
        let scores: Vec<f64> = prediction.try_into().map_err(|e| {
            TransformError::ComputationError(format!("Failed to extract tensor data: {:?}", e))
        })?;

        if scores.len() != 10 {
            return Err(TransformError::ComputationError(format!(
                "Expected 10 prediction scores, got {}",
                scores.len()
            )));
        }

        let mut transformations = Vec::new();

        // Use adaptive threshold based on score distribution
        let max_score = scores.iter().fold(0.0, |a, &b| a.max(b));
        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let threshold = (max_score * 0.7 + mean_score * 0.3).max(0.3); // Adaptive threshold

        for (i, &score) in scores.iter().enumerate() {
            if score > threshold && score.is_finite() {
                let transformation_type = self.index_to_transformation_type(i);
                let config = TransformationConfig {
                    transformation_type: transformation_type.clone(),
                    parameters: self.get_default_parameters_for_type(&transformation_type),
                    expected_performance: score.clamp(0.0, 1.0), // Clamp to valid range
                };
                transformations.push(config);
            }
        }

        // If no transformations meet threshold, take top 3
        if transformations.is_empty() {
            let mut score_indices: Vec<(usize, f64)> = scores
                .iter()
                .enumerate()
                .filter(|(_, &score)| score.is_finite())
                .map(|(i, &score)| (i, score))
                .collect();

            score_indices
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (i, score) in score_indices.into_iter().take(3) {
                let transformation_type = self.index_to_transformation_type(i);
                let config = TransformationConfig {
                    transformation_type: transformation_type.clone(),
                    parameters: self.get_default_parameters_for_type(&transformation_type),
                    expected_performance: score.clamp(0.0, 1.0),
                };
                transformations.push(config);
            }
        }

        // Sort by expected performance
        transformations.sort_by(|a, b| {
            b.expected_performance
                .partial_cmp(&a.expected_performance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(transformations)
    }

    fn transformation_type_to_index(&self, ttype: &TransformationType) -> usize {
        match t_type {
            TransformationType::StandardScaler => 0,
            TransformationType::MinMaxScaler => 1,
            TransformationType::RobustScaler => 2,
            TransformationType::PowerTransformer => 3,
            TransformationType::PolynomialFeatures => 4,
            TransformationType::PCA => 5,
            TransformationType::VarianceThreshold => 6,
            TransformationType::QuantileTransformer => 7,
            TransformationType::BinaryEncoder => 8,
            TransformationType::TargetEncoder => 9,
        }
    }

    fn index_to_transformation_type(&self, index: usize) -> TransformationType {
        match index {
            0 => TransformationType::StandardScaler,
            1 => TransformationType::MinMaxScaler,
            2 => TransformationType::RobustScaler,
            3 => TransformationType::PowerTransformer,
            4 => TransformationType::PolynomialFeatures,
            5 => TransformationType::PCA,
            6 => TransformationType::VarianceThreshold,
            7 => TransformationType::QuantileTransformer,
            8 => TransformationType::BinaryEncoder,
            _ => TransformationType::StandardScaler,
        }
    }

    fn get_default_parameters_for_type(&self, ttype: &TransformationType) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        match t_type {
            TransformationType::PCA => {
                params.insert("n_components".to_string(), 0.95); // Keep 95% variance
            }
            TransformationType::PolynomialFeatures => {
                params.insert("degree".to_string(), 2.0);
                params.insert("include_bias".to_string(), 0.0);
            }
            TransformationType::VarianceThreshold => {
                params.insert("threshold".to_string(), 0.01);
            }
            _ => {} // Use defaults for other transformations
        }
        params
    }
}

/// Automated feature engineering pipeline
pub struct AutoFeatureEngineer {
    #[cfg(feature = "auto-feature-engineering")]
    meta_model: MetaLearningModel,
    /// Historical transformation performance data
    #[cfg(feature = "auto-feature-engineering")]
    transformation_history: Vec<(DatasetMetaFeatures, Vec<TransformationConfig>, f64)>,
}

impl AutoFeatureEngineer {
    /// Expose pearson_correlation as a public method for external use
    #[allow(dead_code)]
    pub fn pearson_correlation(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64> {
        self.pearson_correlation_internal(x, y)
    }
    /// Create a new automated feature engineer
    pub fn new() -> Result<Self> {
        #[cfg(feature = "auto-feature-engineering")]
        let meta_model = MetaLearningModel::new()?;

        Ok(AutoFeatureEngineer {
            #[cfg(feature = "auto-feature-engineering")]
            meta_model,
            #[cfg(feature = "auto-feature-engineering")]
            transformation_history: Vec::new(),
        })
    }

    /// Extract meta-features from a dataset
    pub fn extract_meta_features(&self, x: &ArrayView2<f64>) -> Result<DatasetMetaFeatures> {
        check_not_empty(x, "x")?;

        // Check finite values
        for &val in x.iter() {
            if !val.is_finite() {
                return Err(crate::error::TransformError::DataValidationError(
                    "Data contains non-finite values".to_string(),
                ));
            }
        }

        let (n_samples, n_features) = x.dim();

        if n_samples < 2 || n_features < 1 {
            return Err(TransformError::InvalidInput(
                "Dataset must have at least 2 samples and 1 feature".to_string(),
            ));
        }

        // Calculate sparsity
        let zeros = x.iter().filter(|&&val| val == 0.0).count();
        let sparsity = zeros as f64 / (n_samples * n_features) as f64;

        // Calculate correlation statistics
        let correlations = self.compute_feature_correlations(x)?;
        let mean_correlation = correlations.mean();
        let std_correlation = 0.0; // Simplified - calculating std after mean() consumed value

        // Calculate skewness and kurtosis
        let (mean_skewness, mean_kurtosis) = self.compute_distribution_stats(x)?;

        // Calculate missing values (assuming NaN represents missing)
        let missing_count = x.iter().filter(|val| val.is_nan()).count();
        let missing_ratio = missing_count as f64 / (n_samples * n_features) as f64;
        let has_missing = missing_count > 0;

        // Calculate variance statistics with better numerical stability
        let variances: Array1<f64> = x.var_axis(ndarray::Axis(0), 0.0);
        let finite_variances: Vec<f64> = variances
            .iter()
            .filter(|&&v| v.is_finite() && v >= 0.0)
            .copied()
            .collect();

        let variance_ratio = if finite_variances.is_empty() {
            0.0
        } else {
            let mean_var = finite_variances.iter().sum::<f64>() / finite_variances.len() as f64;
            if mean_var < f64::EPSILON {
                0.0
            } else {
                let var_of_vars = finite_variances
                    .iter()
                    .map(|&v| (v - mean_var).powi(2))
                    .sum::<f64>()
                    / finite_variances.len() as f64;
                (var_of_vars.sqrt() / mean_var).min(100.0) // Cap at reasonable value
            }
        };

        // Calculate outlier ratio (using IQR method)
        let outlier_ratio = self.compute_outlier_ratio(x)?;

        Ok(DatasetMetaFeatures {
            n_samples,
            n_features,
            sparsity,
            mean_correlation,
            std_correlation,
            mean_skewness,
            mean_kurtosis,
            missing_ratio,
            variance_ratio,
            outlier_ratio,
            has_missing,
        })
    }

    /// Recommend optimal transformations for a dataset
    #[cfg(feature = "auto-feature-engineering")]
    pub fn recommend_transformations(
        &self,
        x: &ArrayView2<f64>,
    ) -> Result<Vec<TransformationConfig>> {
        let meta_features = self.extract_meta_features(x)?;
        self.meta_model.predict_transformations(&meta_features)
    }

    /// Recommend optimal transformations for a dataset (fallback implementation)
    #[cfg(not(feature = "auto-feature-engineering"))]
    pub fn recommend_transformations(
        &self,
        x: &ArrayView2<f64>,
    ) -> Result<Vec<TransformationConfig>> {
        // Fallback to rule-based recommendations
        self.rule_based_recommendations(x)
    }

    /// Rule-based transformation recommendations (fallback)
    fn rule_based_recommendations(&self, x: &ArrayView2<f64>) -> Result<Vec<TransformationConfig>> {
        let meta_features = self.extract_meta_features(x)?;
        let mut recommendations = Vec::new();

        // Rule 1: High skewness -> Power transformation
        if meta_features.mean_skewness.abs() > 1.0 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::PowerTransformer,
                parameters: HashMap::new(),
                expected_performance: 0.8,
            });
        }

        // Rule 2: High dimensionality -> PCA
        if meta_features.n_features > 100 {
            let mut params = HashMap::new();
            params.insert("n_components".to_string(), 0.95);
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::PCA,
                parameters: params,
                expected_performance: 0.75,
            });
        }

        // Rule 3: Different scales -> StandardScaler
        if meta_features.variance_ratio > 1.0 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::StandardScaler,
                parameters: HashMap::new(),
                expected_performance: 0.9,
            });
        }

        // Rule 4: High outlier ratio -> RobustScaler
        if meta_features.outlier_ratio > 0.1 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::RobustScaler,
                parameters: HashMap::new(),
                expected_performance: 0.85,
            });
        }

        // Sort by expected performance
        recommendations.sort_by(|a, b| {
            b.expected_performance
                .partial_cmp(&a.expected_performance)
                .unwrap()
        });

        Ok(recommendations)
    }

    /// Train the meta-learning model with new data
    #[cfg(feature = "auto-feature-engineering")]
    pub fn update_model(
        &mut self,
        meta_features: DatasetMetaFeatures,
        transformations: Vec<TransformationConfig>,
        performance: f64,
    ) -> Result<()> {
        self.transformation_history.push((
            meta_features.clone(),
            transformations.clone(),
            performance,
        ));

        // Retrain every 10 new examples
        if self.transformation_history.len() % 10 == 0 {
            let training_data: Vec<_> = self
                .transformation_history
                .iter()
                .map(|(meta, trans_perf)| (meta.clone(), trans.clone()))
                .collect();
            self.meta_model.train(training_data)?;
        }

        Ok(())
    }

    fn compute_feature_correlations(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let n_features = x.ncols();

        if n_features < 2 {
            return Ok(Array1::zeros(0));
        }

        let mut correlations = Vec::with_capacity((n_features * (n_features - 1)) / 2);

        for i in 0..n_features {
            for j in i + 1..n_features {
                let col_i = x.column(i);
                let col_j = x.column(j);
                let correlation = self.pearson_correlation_internal(&col_i, &col_j)?;
                correlations.push(correlation);
            }
        }

        Ok(Array1::from_vec(correlations))
    }

    fn pearson_correlation_internal(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
    ) -> Result<f64> {
        if x.len() != y.len() {
            return Err(TransformError::InvalidInput(
                "Arrays must have the same length for correlation calculation".to_string(),
            ));
        }

        if x.len() < 2 {
            return Ok(0.0);
        }

        let _n = x.len() as f64;
        let mean_x = x.mean().ok_or_else(|| {
            TransformError::ComputationError("Failed to compute mean of x".to_string())
        })?;
        let mean_y = y.mean().ok_or_else(|| {
            TransformError::ComputationError("Failed to compute mean of y".to_string())
        })?;

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator < f64::EPSILON {
            Ok(0.0)
        } else {
            let correlation = numerator / denominator;
            // Clamp to valid correlation range due to numerical precision
            Ok(correlation.clamp(-1.0, 1.0))
        }
    }

    fn compute_distribution_stats(&self, x: &ArrayView2<f64>) -> Result<(f64, f64)> {
        let mut skewness_values = Vec::new();
        let mut kurtosis_values = Vec::new();

        for col in x.columns() {
            // Filter out non-finite values
            let finite_values: Vec<f64> = col
                .iter()
                .filter(|&&val| val.is_finite())
                .copied()
                .collect();

            if finite_values.len() < 3 {
                continue; // Need at least 3 values for meaningful skewness/kurtosis
            }

            let n = finite_values.len() as f64;
            let mean = finite_values.iter().sum::<f64>() / n;

            // Calculate variance using more numerically stable method
            let variance = finite_values
                .iter()
                .map(|&val| (val - mean).powi(2))
                .sum::<f64>()
                / (n - 1.0); // Sample variance

            let std = variance.sqrt();

            if std > f64::EPSILON * 1000.0 {
                // More robust threshold
                // Sample skewness with bias correction
                let m3: f64 = finite_values
                    .iter()
                    .map(|&val| ((val - mean) / std).powi(3))
                    .sum::<f64>()
                    / n;

                let skew = if n > 2.0 {
                    m3 * (n * (n - 1.0)).sqrt() / (n - 2.0) // Bias-corrected skewness
                } else {
                    m3
                };

                // Sample kurtosis with bias correction
                let m4: f64 = finite_values
                    .iter()
                    .map(|&val| ((val - mean) / std).powi(4))
                    .sum::<f64>()
                    / n;

                let kurt = if n > 3.0 {
                    // Bias-corrected excess kurtosis
                    let numerator = (n - 1.0) * ((n + 1.0) * m4 - 3.0 * (n - 1.0));
                    let denominator = (n - 2.0) * (n - 3.0);
                    numerator / denominator
                } else {
                    m4 - 3.0 // Simple excess kurtosis
                };

                // Clamp to reasonable ranges to avoid extreme outliers
                skewness_values.push(skew.clamp(-20.0, 20.0));
                kurtosis_values.push(kurt.clamp(-20.0, 20.0));
            }
        }

        let mean_skewness = if skewness_values.is_empty() {
            0.0
        } else {
            skewness_values.iter().sum::<f64>() / skewness_values.len() as f64
        };

        let mean_kurtosis = if kurtosis_values.is_empty() {
            0.0
        } else {
            kurtosis_values.iter().sum::<f64>() / kurtosis_values.len() as f64
        };

        Ok((mean_skewness, mean_kurtosis))
    }

    fn compute_outlier_ratio(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let mut total_outliers = 0;
        let mut total_values = 0;

        for col in x.columns() {
            let mut sorted_col: Vec<f64> = col
                .iter()
                .filter(|&&val| val.is_finite())
                .copied()
                .collect();

            if sorted_col.is_empty() {
                continue;
            }

            sorted_col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = sorted_col.len();
            if n < 4 {
                continue;
            }

            // Use proper quartile calculation
            let q1_idx = (n as f64 * 0.25) as usize;
            let q3_idx = (n as f64 * 0.75) as usize;
            let q1 = sorted_col[q1_idx.min(n - 1)];
            let q3 = sorted_col[q3_idx.min(n - 1)];

            let iqr = q3 - q1;

            // Avoid division by zero or very small IQR
            if iqr < f64::EPSILON {
                continue;
            }

            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;

            let outliers = col
                .iter()
                .filter(|&&val| val.is_finite() && (val < lower_bound || val > upper_bound))
                .count();

            total_outliers += outliers;
            total_values += col.len();
        }

        if total_values == 0 {
            Ok(0.0)
        } else {
            Ok(total_outliers as f64 / total_values as f64)
        }
    }
}

/// Advanced meta-learning system with deep learning and reinforcement learning
#[cfg(feature = "auto-feature-engineering")]
pub struct AdvancedMetaLearningSystem {
    /// Deep neural network for meta-learning
    deep_model: nn::Sequential,
    /// Transformer model for sequence-based recommendations
    transformer_model: nn::Sequential,
    /// Reinforcement learning agent for transformation selection
    rl_agent: Option<RLAgent>,
    /// Device for computation
    device: Device,
    /// Historical performance database
    performance_db: Vec<PerformanceRecord>,
    /// Multi-objective optimization weights
    optimization_weights: FeatureOptimizationWeights,
    /// Transfer learning cache
    transfer_cache: HashMap<String, Tensor>,
}

/// Reinforcement learning agent for transformation selection
#[cfg(feature = "auto-feature-engineering")]
pub struct RLAgent {
    /// Q-network for value estimation
    q_network: nn::Sequential,
    /// Target network for stable training
    target_network: nn::Sequential,
    /// Experience replay buffer
    replay_buffer: VecDeque<Experience>,
    /// Epsilon for exploration
    epsilon: f64,
    /// Learning rate
    learning_rate: f64,
    /// Discount factor
    gamma: f64,
}

/// Experience tuple for reinforcement learning
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct Experience {
    /// State representation (meta-features)
    state: Vec<f64>,
    /// Action taken (transformation choice)
    action: usize,
    /// Reward received (performance improvement)
    reward: f64,
    /// Next state
    next_state: Vec<f64>,
    /// Whether episode terminated
    done: bool,
}

/// Performance record for historical analysis
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Dataset meta-features
    meta_features: DatasetMetaFeatures,
    /// Applied transformations
    transformations: Vec<TransformationConfig>,
    /// Performance metrics
    metrics: PerformanceMetrics,
    /// Computational cost
    computational_cost: f64,
    /// Timestamp
    timestamp: u64,
}

/// Multi-objective optimization weights
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct FeatureOptimizationWeights {
    /// Weight for prediction performance
    performance_weight: f64,
    /// Weight for computational efficiency
    efficiency_weight: f64,
    /// Weight for model interpretability
    interpretability_weight: f64,
    /// Weight for robustness
    robustness_weight: f64,
}

#[cfg(feature = "auto-feature-engineering")]
impl Default for FeatureOptimizationWeights {
    fn default() -> Self {
        FeatureOptimizationWeights {
            performance_weight: 0.5,
            efficiency_weight: 0.3,
            interpretability_weight: 0.1,
            robustness_weight: 0.1,
        }
    }
}

/// Performance metrics for multi-objective optimization
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Prediction accuracy/score
    accuracy: f64,
    /// Training time in seconds
    training_time: f64,
    /// Memory usage in MB
    memory_usage: f64,
    /// Model complexity score
    complexity_score: f64,
    /// Cross-validation score
    cv_score: f64,
}

/// Enhanced meta-features with advanced statistical measures
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct EnhancedMetaFeatures {
    /// Base meta-features
    pub base_features: DatasetMetaFeatures,
    /// Estimated intrinsic dimension
    pub manifold_dimension: f64,
    /// Hopkins statistic for clustering tendency
    pub clustering_tendency: f64,
    /// Average mutual information between features
    pub mutual_information_mean: f64,
    /// Differential entropy estimate
    pub entropy_estimate: f64,
    /// Condition number estimate
    pub condition_number: f64,
    /// Volume ratio (convex hull to bounding box)
    pub volume_ratio: f64,
    /// Autocorrelation coefficient
    pub autocorrelation: f64,
    /// Trend strength
    pub trend_strength: f64,
    /// Feature connectivity
    pub connectivity: f64,
    /// Feature clustering coefficient
    pub clustering_coefficient: f64,
}

/// Multi-objective recommendation with performance trade-offs
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct MultiObjectiveRecommendation {
    /// Transformation configuration
    pub transformation: TransformationConfig,
    /// Expected performance score
    pub performance_score: f64,
    /// Computational efficiency score
    pub efficiency_score: f64,
    /// Interpretability score
    pub interpretability_score: f64,
    /// Robustness score
    pub robustness_score: f64,
    /// Overall multi-objective score
    pub overall_score: f64,
}

#[cfg(feature = "auto-feature-engineering")]
impl AdvancedMetaLearningSystem {
    /// Create a new advanced meta-learning system
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Build deep neural network with advanced architecture
        let deep_model = nn::seq()
            .add(nn::linear(
                &root / "deep_layer1",
                20,
                128,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.dropout(0.3, false))
            .add(nn::linear(
                &root / "deep_layer2",
                128,
                256,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::batch_norm1d(&root / "bn1", 256, Default::default()))
            .add(nn::linear(
                &root / "deep_layer3",
                256,
                128,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.dropout(0.3, false))
            .add(nn::linear(
                &root / "deep_layer4",
                128,
                64,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &root / "deep_output",
                64,
                20,
                Default::default(),
            ))
            .add_fn(|xs| xs.softmax(-1, tch::Kind::Float));

        // Build transformer model for sequence-based recommendations
        let transformer_model = nn::seq()
            .add(nn::linear(&root / "trans_embed", 20, 256, Default::default()))
            // Note: Actual transformer layers would be implemented here
            .add(nn::linear(&root / "trans_layer1", 256, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "trans_layer2", 256, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "trans_output", 128, 20, Default::default()))
            .add_fn(|xs| xs.softmax(-1, tch::Kind::Float));

        Ok(AdvancedMetaLearningSystem {
            deep_model,
            transformer_model,
            rl_agent: None,
            device,
            performance_db: Vec::new(),
            optimization_weights: FeatureOptimizationWeights::default(),
            transfer_cache: HashMap::new(),
        })
    }

    /// Initialize reinforcement learning agent
    pub fn initialize_rl_agent(&mut self) -> Result<()> {
        let vs = nn::VarStore::new(self.device);
        let root = vs.root();

        // Q-network architecture
        let q_network = nn::seq()
            .add(nn::linear(&root / "q_layer1", 20, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "q_layer2", 128, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "q_layer3", 256, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "q_output", 128, 20, Default::default())); // 20 possible transformations

        // Target network (copy of Q-network)
        let target_vs = nn::VarStore::new(self.device);
        let target_root = target_vs.root();
        let target_network = nn::seq()
            .add(nn::linear(
                &target_root / "target_layer1",
                20,
                128,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &target_root / "target_layer2",
                128,
                256,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &target_root / "target_layer3",
                256,
                128,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &target_root / "target_output",
                128,
                20,
                Default::default(),
            ));

        self.rl_agent = Some(RLAgent {
            q_network,
            target_network,
            replay_buffer: VecDeque::with_capacity(10000),
            epsilon: 0.1,
            learning_rate: 0.001,
            gamma: 0.99,
        });

        Ok(())
    }

    /// Enhanced meta-feature extraction with advanced statistical measures
    pub fn extract_enhanced_meta_features(
        &self,
        x: &ArrayView2<f64>,
    ) -> Result<EnhancedMetaFeatures> {
        let auto_engineer = AutoFeatureEngineer::new()?;
        let base_features = auto_engineer.extract_meta_features(x)?;

        // Extract additional advanced meta-features
        let (_n_samples_n_features) = x.dim();

        // Topological features
        let manifold_dimension = self.estimate_intrinsic_dimension(x)?;
        let clustering_tendency = self.hopkins_statistic(x)?;

        // Information-theoretic features
        let mutual_information_mean = self.average_mutual_information(x)?;
        let entropy_estimate = self.differential_entropy_estimate(x)?;

        // Geometric features
        let condition_number = self.estimate_condition_number(x)?;
        let volume_ratio = self.estimate_volume_ratio(x)?;

        // Temporal features (if applicable)
        let autocorrelation = self.estimate_autocorrelation(x)?;
        let trend_strength = self.estimate_trend_strength(x)?;

        // Network/graph features
        let connectivity = self.estimate_feature_connectivity(x)?;
        let clustering_coefficient = self.feature_clustering_coefficient(x)?;

        Ok(EnhancedMetaFeatures {
            base_features,
            manifold_dimension,
            clustering_tendency,
            mutual_information_mean,
            entropy_estimate,
            condition_number,
            volume_ratio,
            autocorrelation,
            trend_strength,
            connectivity,
            clustering_coefficient,
        })
    }

    /// Multi-objective transformation recommendation
    pub fn recommend_multi_objective_transformations(
        &self,
        meta_features: &EnhancedMetaFeatures,
    ) -> Result<Vec<MultiObjectiveRecommendation>> {
        // Get base recommendations from deep model
        let deep_input = self.enhanced_meta_features_to_tensor(meta_features)?;
        let deep_predictions = self.deep_model.forward(&deep_input);

        // Get sequence-based recommendations from transformer
        let transformer_predictions = self.transformer_model.forward(&deep_input);

        // Combine predictions using ensemble weighting
        let ensemble_predictions = (&deep_predictions * 0.6) + (&transformer_predictions * 0.4);

        // Apply reinforcement learning if available
        let final_predictions = if let Some(ref rl_agent) = self.rl_agent {
            let rl_q_values = rl_agent.q_network.forward(&deep_input);
            let rl_softmax = rl_q_values.softmax(-1, tch::Kind::Float);
            (&ensemble_predictions * 0.7) + (&rl_softmax * 0.3)
        } else {
            ensemble_predictions
        };

        // Convert to multi-objective recommendations
        self.tensor_to_multi_objective_recommendations(&final_predictions, meta_features)
    }

    /// Transfer learning from similar datasets
    pub fn apply_transfer_learning(
        &mut self,
        target_meta_features: &EnhancedMetaFeatures,
    ) -> Result<Vec<TransformationConfig>> {
        // Find similar datasets in performance database
        let similar_records = self.find_similar_datasets(target_meta_features, 5)?;

        if similar_records.is_empty() {
            return self.fallback_recommendations(target_meta_features);
        }

        // Extract successful transformations from similar datasets
        let mut transformation_votes: HashMap<TransformationType, (f64, usize)> = HashMap::new();

        for record in &similar_records {
            let similarity =
                self.compute_dataset_similarity(target_meta_features, &record.meta_features)?;

            for transformation in &record.transformations {
                let performance_score = record.metrics.accuracy * similarity;
                let entry = transformation_votes
                    .entry(transformation.transformation_type.clone())
                    .or_insert((0.0, 0));
                entry.0 += performance_score;
                entry.1 += 1;
            }
        }

        // Rank transformations by weighted performance
        let mut ranked_transformations: Vec<_> = transformation_votes
            .into_iter()
            .map(|(t_type, (total_score, count))| (t_type, total_score / count as f64))
            .collect();

        ranked_transformations
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Convert to transformation configs
        let mut recommendations = Vec::new();
        for (t_type, score) in ranked_transformations.into_iter().take(5) {
            recommendations.push(TransformationConfig {
                transformation_type: t_type.clone(),
                parameters: self
                    .get_optimized_parameters_for_type(&t_type, target_meta_features)?,
                expected_performance: score.min(1.0).max(0.0),
            });
        }

        Ok(recommendations)
    }

    // Helper methods for advanced meta-feature extraction
    fn estimate_intrinsic_dimension(&self, x: &ArrayView2<f64>) -> Result<f64> {
        // Simplified intrinsic dimension estimation using correlation dimension
        let (n_samples_) = x.dim();
        if n_samples < 10 {
            return Ok(1.0);
        }

        // Sample random points and compute distances
        use rand::Rng;
        let mut rng = rand::rng();
        let sample_size = 100.min(n_samples);
        let mut distances = Vec::new();

        for _ in 0..sample_size {
            let i = rng.gen_range(0..n_samples);
            let j = rng.gen_range(0..n_samples);
            if i != j {
                let dist = self.euclidean_distance(&x.row(i)..&x.row(j));
                distances.push(dist);
            }
        }

        // Estimate dimension using correlation dimension approach
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if distances.is_empty() || distances[0] == 0.0 {
            return Ok(1.0);
        }

        // Count pairs within different distance thresholds
        let thresholds = [0.1, 0.2, 0.5, 1.0];
        let mut dimension_estimates = Vec::new();

        for &threshold in &thresholds {
            let count = distances.iter().filter(|&&d| d < threshold).count();
            if count > 1 {
                let correlation_sum = (count as f64).ln();
                let threshold_ln = threshold.ln();
                if threshold_ln != 0.0 {
                    dimension_estimates.push(correlation_sum / threshold_ln);
                }
            }
        }

        let avg_dimension = if dimension_estimates.is_empty() {
            1.0
        } else {
            dimension_estimates.iter().sum::<f64>() / dimension_estimates.len() as f64
        };

        Ok(avg_dimension.max(1.0).min(x.ncols() as f64))
    }

    fn hopkins_statistic(&self, x: &ArrayView2<f64>) -> Result<f64> {
        // Hopkins statistic for clustering tendency
        let (n_samples, n_features) = x.dim();
        if n_samples < 10 {
            return Ok(0.5); // Neutral value
        }

        use rand::Rng;
        let mut rng = rand::rng();
        let sample_size = 10.min(n_samples / 2);

        // Generate random points in the data space
        let mut min_vals = vec![f64::INFINITY; n_features];
        let mut max_vals = vec![f64::NEG_INFINITY; n_features];

        for row in x.rows() {
            for (j, &val) in row.iter().enumerate() {
                min_vals[j] = min_vals[j].min(val);
                max_vals[j] = max_vals[j].max(val);
            }
        }

        let mut u_distances = Vec::new();
        let mut w_distances = Vec::new();

        // Sample random points and compute distances
        for _ in 0..sample_size {
            // Random point in data space
            let mut random_point = vec![0.0; n_features];
            for j in 0..n_features {
                random_point[j] = rng.gen_range(min_vals[j]..=max_vals[j]);
            }

            // Find nearest neighbor distance for random point
            let mut min_dist_u = f64::INFINITY;
            for row in x.rows() {
                let dist = self.euclidean_distance_vec(&random_point, &row.to_vec());
                min_dist_u = min_dist_u.min(dist);
            }
            u_distances.push(min_dist_u);

            // Random data point
            let random_idx = rng.gen_range(0..n_samples);
            let data_point = x.row(random_idx).to_vec();

            // Find nearest neighbor distance for data point
            let mut min_dist_w = f64::INFINITY;
            for (i, row) in x.rows().enumerate() {
                if i != random_idx {
                    let dist = self.euclidean_distance_vec(&data_point, &row.to_vec());
                    min_dist_w = min_dist_w.min(dist);
                }
            }
            w_distances.push(min_dist_w);
        }

        let sum_u: f64 = u_distances.iter().sum();
        let sum_w: f64 = w_distances.iter().sum();

        if sum_u + sum_w == 0.0 {
            Ok(0.5)
        } else {
            Ok(sum_u / (sum_u + sum_w))
        }
    }

    fn average_mutual_information(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let (_, n_features) = x.dim();
        if n_features < 2 {
            return Ok(0.0);
        }

        let mut mi_sum = 0.0;
        let mut pair_count = 0;

        // Sample pairs of features to avoid O(n²) complexity
        use rand::Rng;
        let mut rng = rand::rng();
        let max_pairs = 50.min((n_features * (n_features - 1)) / 2);

        for _ in 0..max_pairs {
            let i = rng.gen_range(0..n_features);
            let j = rng.gen_range(0..n_features);
            if i != j {
                let mi = self.estimate_mutual_information(&x.column(i)..&x.column(j))?;
                mi_sum += mi;
                pair_count += 1;
            }
        }

        Ok(if pair_count > 0 {
            mi_sum / pair_count as f64
        } else {
            0.0
        })
    }

    fn estimate_mutual_information(
        &self,
        x: &ndarray::ArrayView1<f64>,
        y: &ndarray::ArrayView1<f64>,
    ) -> Result<f64> {
        // Simplified MI estimation using binning
        let n_bins = 10;
        let x_bins = self.create_bins(x, n_bins);
        let y_bins = self.create_bins(y, n_bins);

        // Create joint histogram
        let mut joint_hist = vec![vec![0; n_bins]; n_bins];
        let mut x_hist = vec![0; n_bins];
        let mut y_hist = vec![0; n_bins];

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            if xi.is_finite() && yi.is_finite() {
                let x_bin = self.find_bin(xi, &x_bins).min(n_bins - 1);
                let y_bin = self.find_bin(yi, &y_bins).min(n_bins - 1);
                joint_hist[x_bin][y_bin] += 1;
                x_hist[x_bin] += 1;
                y_hist[y_bin] += 1;
            }
        }

        let total = x.len() as f64;
        let mut mi = 0.0;

        for i in 0..n_bins {
            for j in 0..n_bins {
                let p_xy = joint_hist[i][j] as f64 / total;
                let p_x = x_hist[i] as f64 / total;
                let p_y = y_hist[j] as f64 / total;

                if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                    mi += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }

        Ok(mi.max(0.0))
    }

    fn differential_entropy_estimate(&self, x: &ArrayView2<f64>) -> Result<f64> {
        // Simplified differential entropy estimate
        let (n_samples, n_features) = x.dim();
        if n_samples < 2 {
            return Ok(0.0);
        }

        let mut entropy_sum = 0.0;
        for col in x.columns() {
            let variance = col.variance();
            if variance > 0.0 {
                // Gaussian entropy: 0.5 * log(2πeσ²)
                entropy_sum +=
                    0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * variance).ln();
            }
        }

        Ok(entropy_sum / n_features as f64)
    }

    fn estimate_condition_number(&self, x: &ArrayView2<f64>) -> Result<f64> {
        // Simplified condition number estimation
        let (n_samples, n_features) = x.dim();
        if n_samples < n_features || n_features < 2 {
            return Ok(1.0);
        }

        // Compute correlation matrix
        let mut corr_sum = 0.0;
        let mut corr_count = 0;

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let col_i = x.column(i);
                let col_j = x.column(j);
                if let Ok(corr) = self.quick_correlation(&col_i, &col_j) {
                    corr_sum += corr.abs();
                    corr_count += 1;
                }
            }
        }

        let avg_correlation = if corr_count > 0 {
            corr_sum / corr_count as f64
        } else {
            0.0
        };

        // Approximate condition number based on correlation
        Ok(if avg_correlation > 0.9 {
            100.0 // High condition number
        } else if avg_correlation > 0.7 {
            10.0 // Medium condition number
        } else {
            1.0 // Low condition number
        })
    }

    // Additional helper methods
    fn euclidean_distance(
        &self,
        a: &ndarray::ArrayView1<f64>,
        b: &ndarray::ArrayView1<f64>,
    ) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn euclidean_distance_vec(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn create_bins(&self, data: &ndarray::ArrayView1<f64>, nbins: usize) -> Vec<f64> {
        let mut sorted: Vec<f64> = data.iter().filter(|&&x| x.is_finite()).copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.is_empty() {
            return vec![0.0; n_bins + 1];
        }

        let mut _bins = Vec::new();
        for i in 0..=n_bins {
            let idx = (i * (sorted.len() - 1)) / n_bins;
            bins.push(sorted[idx]);
        }
        _bins
    }

    fn find_bin(&self, value: f64, bins: &[f64]) -> usize {
        for (i, &bin_edge) in bins.iter().enumerate().take(bins.len() - 1) {
            if value <= bin_edge {
                return i;
            }
        }
        bins.len() - 2
    }

    /// Estimate volume ratio (convex hull to bounding box)
    fn estimate_volume_ratio(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let (n_samples, n_features) = x.dim();
        if n_samples < 4 || n_features < 2 {
            return Ok(1.0); // Default for insufficient data
        }

        // For high-dimensional data, use sampling approach
        let sample_size = 1000.min(n_samples);
        use rand::seq::SliceRandom;

        let mut rng = rng();
        let indices: Vec<usize> = (0..n_samples)
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, sample_size)
            .copied()
            .collect();

        // Calculate bounding box volume
        let mut min_vals = vec![f64::INFINITY; n_features];
        let mut max_vals = vec![f64::NEG_INFINITY; n_features];

        for &idx in &indices {
            let row = x.row(idx);
            for (j, &val) in row.iter().enumerate() {
                if val.is_finite() {
                    min_vals[j] = min_vals[j].min(val);
                    max_vals[j] = max_vals[j].max(val);
                }
            }
        }

        // Calculate bounding box volume
        let mut box_volume = 1.0;
        for j in 0..n_features {
            let range = max_vals[j] - min_vals[j];
            if range > f64::EPSILON {
                box_volume *= range;
            } else {
                return Ok(0.0); // Degenerate case
            }
        }

        // Estimate convex hull volume using sampling (simplified approach)
        // For a proper implementation, you'd use a convex hull algorithm
        // Here we estimate using variance-based approximation
        let mut variance_product = 1.0;
        for j in 0..n_features {
            let col_values: Vec<f64> = indices
                .iter()
                .map(|&idx| x[[idx, j]])
                .filter(|&val| val.is_finite())
                .collect();

            if col_values.len() > 1 {
                let mean = col_values.iter().sum::<f64>() / col_values.len() as f64;
                let variance = col_values
                    .iter()
                    .map(|&val| (val - mean).powi(2))
                    .sum::<f64>()
                    / (col_values.len() - 1) as f64;
                variance_product *= variance.sqrt();
            }
        }

        // Approximate volume ratio
        if box_volume > f64::EPSILON {
            let ratio = (variance_product / box_volume).min(1.0).max(0.0);
            Ok(ratio)
        } else {
            Ok(0.0)
        }
    }

    /// Estimate autocorrelation for time-like patterns
    fn estimate_autocorrelation(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let (n_samples, n_features) = x.dim();
        if n_samples < 3 {
            return Ok(0.0);
        }

        let mut autocorr_sum = 0.0;
        let mut feature_count = 0;

        // Calculate autocorrelation for each feature
        for j in 0..n_features {
            let col = x.column(j);
            let values: Vec<f64> = col
                .iter()
                .filter(|&&val| val.is_finite())
                .copied()
                .collect();

            if values.len() < 3 {
                continue;
            }

            // Calculate lag-1 autocorrelation
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in 0..values.len() - 1 {
                numerator += (values[i] - mean) * (values[i + 1] - mean);
            }

            for &val in &values {
                denominator += (val - mean).powi(2);
            }

            if denominator > f64::EPSILON {
                autocorr_sum += numerator / denominator;
                feature_count += 1;
            }
        }

        if feature_count > 0 {
            Ok((autocorr_sum / feature_count as f64).abs())
        } else {
            Ok(0.0)
        }
    }

    /// Estimate trend strength in the data
    fn estimate_trend_strength(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let (n_samples, n_features) = x.dim();
        if n_samples < 5 {
            return Ok(0.0);
        }

        let mut trend_sum = 0.0;
        let mut feature_count = 0;

        // Calculate trend strength for each feature
        for j in 0..n_features {
            let col = x.column(j);
            let values: Vec<(f64, f64)> = col
                .iter()
                .enumerate()
                .filter(|(_, &val)| val.is_finite())
                .map(|(i, &val)| (i as f64, val))
                .collect();

            if values.len() < 5 {
                continue;
            }

            // Calculate linear trend using least squares
            let n = values.len() as f64;
            let sum_x: f64 = values.iter().map(|(x_)| x).sum();
            let sum_y: f64 = values.iter().map(|(_, y)| y).sum();
            let sum_xy: f64 = values.iter().map(|(x, y)| x * y).sum();
            let sum_x2: f64 = values.iter().map(|(x_)| x * x).sum();

            let denominator = n * sum_x2 - sum_x * sum_x;
            if denominator.abs() > f64::EPSILON {
                let slope = (n * sum_xy - sum_x * sum_y) / denominator;
                let intercept = (sum_y - slope * sum_x) / n;

                // Calculate R-squared to measure trend strength
                let y_mean = sum_y / n;
                let mut ss_tot = 0.0;
                let mut ss_res = 0.0;

                for (x_val, y_val) in &values {
                    let y_pred = slope * x_val + intercept;
                    ss_tot += (y_val - y_mean).powi(2);
                    ss_res += (y_val - y_pred).powi(2);
                }

                if ss_tot > f64::EPSILON {
                    let r_squared = 1.0 - (ss_res / ss_tot);
                    trend_sum += r_squared.max(0.0);
                    feature_count += 1;
                }
            }
        }

        if feature_count > 0 {
            Ok(trend_sum / feature_count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Estimate feature connectivity (correlation-based)
    fn estimate_feature_connectivity(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let (_, n_features) = x.dim();
        if n_features < 2 {
            return Ok(0.0);
        }

        let mut strong_connections = 0;
        let mut total_connections = 0;
        let threshold = 0.5; // Threshold for "strong" connection

        // Sample pairs to avoid O(n²) complexity for large feature sets
        let max_pairs = 100.min((n_features * (n_features - 1)) / 2);
        use rand::Rng;
        let mut rng = rand::rng();

        for _ in 0..max_pairs {
            let i = rng.gen_range(0..n_features);
            let j = rng.gen_range(0..n_features);

            if i != j {
                let col_i = x.column(i);
                let col_j = x.column(j);

                if let Ok(corr) = self.quick_correlation(&col_i, &col_j) {
                    if corr.abs() > threshold {
                        strong_connections += 1;
                    }
                    total_connections += 1;
                }
            }
        }

        if total_connections > 0 {
            Ok(strong_connections as f64 / total_connections as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Quick correlation calculation without full validation
    fn quick_correlation(
        &self,
        x: &ndarray::ArrayView1<f64>,
        y: &ndarray::ArrayView1<f64>,
    ) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Ok(0.0);
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            if xi.is_finite() && yi.is_finite() {
                let diff_x = xi - mean_x;
                let diff_y = yi - mean_y;
                numerator += diff_x * diff_y;
                sum_sq_x += diff_x * diff_x;
                sum_sq_y += diff_y * diff_y;
            }
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator < f64::EPSILON {
            Ok(0.0)
        } else {
            let correlation = numerator / denominator;
            Ok(correlation.max(-1.0).min(1.0))
        }
    }

    /// Calculate feature clustering coefficient
    fn feature_clustering_coefficient(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let (_, n_features) = x.dim();
        if n_features < 3 {
            return Ok(0.0);
        }

        // Build correlation adjacency matrix (sampled)
        let sample_size = 20.min(n_features);
        use rand::seq::SliceRandom;

        let mut rng = rng();
        let sampled_features: Vec<usize> = (0..n_features)
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, sample_size)
            .copied()
            .collect();

        let threshold = 0.5;
        let mut adjacency = vec![vec![false; sample_size]; sample_size];

        // Build adjacency matrix
        for (i, &feat_i) in sampled_features.iter().enumerate() {
            for (j, &feat_j) in sampled_features.iter().enumerate() {
                if i != j {
                    let col_i = x.column(feat_i);
                    let col_j = x.column(feat_j);

                    if let Ok(corr) = self.quick_correlation(&col_i, &col_j) {
                        adjacency[i][j] = corr.abs() > threshold;
                    }
                }
            }
        }

        // Calculate clustering coefficient
        let mut total_coefficient = 0.0;
        let mut node_count = 0;

        for i in 0..sample_size {
            // Find neighbors of node i
            let neighbors: Vec<usize> = (0..sample_size).filter(|&j| adjacency[i][j]).collect();

            if neighbors.len() >= 2 {
                // Count edges between neighbors
                let mut edges_between_neighbors = 0;
                let mut possible_edges = 0;

                for (ni, &neighbor_i) in neighbors.iter().enumerate() {
                    for &neighbor_j in neighbors.iter().skip(ni + 1) {
                        possible_edges += 1;
                        if adjacency[neighbor_i][neighbor_j] {
                            edges_between_neighbors += 1;
                        }
                    }
                }

                if possible_edges > 0 {
                    total_coefficient += edges_between_neighbors as f64 / possible_edges as f64;
                    node_count += 1;
                }
            }
        }

        if node_count > 0 {
            Ok(total_coefficient / node_count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Convert enhanced meta-features to tensor for neural network input
    fn enhanced_meta_features_to_tensor(&self, features: &EnhancedMetaFeatures) -> Result<Tensor> {
        // Create feature vector with proper normalization
        let feature_vec = vec![
            // Base features (normalized)
            (features.base_features.n_samples as f64).ln().max(0.0),
            (features.base_features.n_features as f64).ln().max(0.0),
            features.base_features.sparsity.max(0.0).min(1.0),
            features.base_features.mean_correlation.max(-1.0).min(1.0),
            features.base_features.std_correlation.max(0.0),
            features.base_features.mean_skewness.max(-10.0).min(10.0),
            features.base_features.mean_kurtosis.max(-10.0).min(10.0),
            features.base_features.missing_ratio.max(0.0).min(1.0),
            features.base_features.variance_ratio.max(0.0),
            features.base_features.outlier_ratio.max(0.0).min(1.0),
            // Enhanced features (normalized)
            features
                .manifold_dimension
                .max(1.0)
                .min(features.base_features.n_features as f64)
                .ln(),
            features.clustering_tendency.max(0.0).min(1.0),
            features.mutual_information_mean.max(0.0),
            features.entropy_estimate.max(0.0),
            (features.condition_number.max(1.0)).ln(),
            features.volume_ratio.max(0.0).min(1.0),
            features.autocorrelation.max(-1.0).min(1.0),
            features.trend_strength.max(0.0).min(1.0),
            features.connectivity.max(0.0).min(1.0),
            features.clustering_coefficient.max(0.0).min(1.0),
        ];

        // Validate all features are finite
        if feature_vec.iter().any(|&f| !f.is_finite()) {
            return Err(TransformError::ComputationError(
                "Non-finite values in enhanced meta-features".to_string(),
            ));
        }

        Ok(Tensor::f_from_slice(&feature_vec)?
            .reshape(&[1, 20])
            .to_device(self.device))
    }

    /// Convert tensor predictions to multi-objective recommendations
    fn tensor_to_multi_objective_recommendations(
        &self,
        tensor: &Tensor,
        features: &EnhancedMetaFeatures,
    ) -> Result<Vec<MultiObjectiveRecommendation>> {
        let scores: Vec<f64> = tensor.try_into().map_err(|e| {
            TransformError::ComputationError(format!("Failed to extract tensor data: {:?}", e))
        })?;

        if scores.len() != 20 {
            return Err(TransformError::ComputationError(format!(
                "Expected 20 prediction scores, got {}",
                scores.len()
            )));
        }

        let mut recommendations = Vec::new();

        // Map scores to transformations (first 10 are transformation types)
        let transformation_types = [
            TransformationType::StandardScaler,
            TransformationType::MinMaxScaler,
            TransformationType::RobustScaler,
            TransformationType::PowerTransformer,
            TransformationType::PolynomialFeatures,
            TransformationType::PCA,
            TransformationType::VarianceThreshold,
            TransformationType::QuantileTransformer,
            TransformationType::BinaryEncoder,
            TransformationType::TargetEncoder,
        ];

        for (i, t_type) in transformation_types.iter().enumerate() {
            if i < scores.len() && scores[i].is_finite() && scores[i] > 0.3 {
                // Calculate multi-objective scores
                let performance_score = scores[i].max(0.0).min(1.0);

                // Estimate efficiency based on data characteristics
                let efficiency_score = self.estimate_efficiency_score(t_type, features)?;

                // Estimate interpretability
                let interpretability_score = self.estimate_interpretability_score(t_type);

                // Estimate robustness
                let robustness_score = self.estimate_robustness_score(t_type, features);

                // Calculate overall score using default weights
                let weights = FeatureOptimizationWeights::default();
                let overall_score = performance_score * weights.performance_weight
                    + efficiency_score * weights.efficiency_weight
                    + interpretability_score * weights.interpretability_weight
                    + robustness_score * weights.robustness_weight;

                recommendations.push(MultiObjectiveRecommendation {
                    transformation: TransformationConfig {
                        transformation_type: t_type.clone(),
                        parameters: self.get_optimized_parameters_for_type(t_type, features)?,
                        expected_performance: performance_score,
                    },
                    performance_score,
                    efficiency_score,
                    interpretability_score,
                    robustness_score,
                    overall_score,
                });
            }
        }

        // Sort by overall score
        recommendations.sort_by(|a, b| {
            b.overall_score
                .partial_cmp(&a.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(recommendations)
    }

    /// Find similar datasets from performance database
    fn find_similar_datasets(
        &self,
        target: &EnhancedMetaFeatures,
        k: usize,
    ) -> Result<Vec<PerformanceRecord>> {
        if self.performance_db.is_empty() {
            return Ok(vec![]);
        }

        let mut similarities: Vec<(usize, f64)> = Vec::new();

        for (i, record) in self.performance_db.iter().enumerate() {
            let similarity = self.compute_dataset_similarity(target, &record.meta_features)?;
            similarities.push((i, similarity));
        }

        // Sort by similarity and take top k
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut similar_records = Vec::new();
        for (idx_similarity) in similarities.iter().take(k) {
            similar_records.push(self.performance_db[*idx].clone());
        }

        Ok(similar_records)
    }

    /// Compute similarity between two datasets using enhanced meta-features
    fn compute_dataset_similarity(
        &self,
        a: &EnhancedMetaFeatures,
        b: &DatasetMetaFeatures,
    ) -> Result<f64> {
        // Compare base features only (since b doesn't have enhanced features)
        let features_a = &a.base_features;

        // Normalize features for comparison
        let scale_similarity = |val_a: f64, val_b: f64, max_val: f64| -> f64 {
            if max_val > 0.0 {
                1.0 - (val_a - val_b).abs() / max_val
            } else {
                if (val_a - val_b).abs() < f64::EPSILON {
                    1.0
                } else {
                    0.0
                }
            }
        };

        // Calculate similarity for each dimension
        let similarities = vec![
            scale_similarity(
                (features_a.n_samples as f64).ln(),
                (b.n_samples as f64).ln(),
                20.0, // Reasonable scale for log(samples)
            ),
            scale_similarity(
                (features_a.n_features as f64).ln(),
                (b.n_features as f64).ln(),
                15.0, // Reasonable scale for log(features)
            ),
            scale_similarity(features_a.sparsity, b.sparsity, 1.0),
            scale_similarity(features_a.mean_correlation, b.mean_correlation, 2.0),
            scale_similarity(features_a.std_correlation, b.std_correlation, 1.0),
            scale_similarity(features_a.mean_skewness, b.mean_skewness, 20.0),
            scale_similarity(features_a.mean_kurtosis, b.mean_kurtosis, 20.0),
            scale_similarity(features_a.missing_ratio, b.missing_ratio, 1.0),
            scale_similarity(features_a.variance_ratio, b.variance_ratio, 10.0),
            scale_similarity(features_a.outlier_ratio, b.outlier_ratio, 1.0),
        ];

        // Weighted average (give more weight to important characteristics)
        let weights = vec![0.15, 0.15, 0.1, 0.15, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05];
        let weighted_similarity = similarities
            .iter()
            .zip(weights.iter())
            .map(|(sim, weight)| sim * weight)
            .sum::<f64>();

        Ok(weighted_similarity.max(0.0).min(1.0))
    }

    /// Fallback recommendations when meta-learning fails
    fn fallback_recommendations(
        &self,
        features: &EnhancedMetaFeatures,
    ) -> Result<Vec<TransformationConfig>> {
        let mut recommendations = Vec::new();
        let base_features = &features.base_features;

        // Rule-based recommendations

        // 1. Always recommend StandardScaler for most datasets
        recommendations.push(TransformationConfig {
            transformation_type: TransformationType::StandardScaler,
            parameters: HashMap::new(),
            expected_performance: 0.8,
        });

        // 2. High dimensionality -> PCA
        if base_features.n_features > 100 || base_features.n_features > base_features.n_samples {
            let mut params = HashMap::new();
            params.insert("n_components".to_string(), 0.95); // Keep 95% variance
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::PCA,
                parameters: params,
                expected_performance: 0.75,
            });
        }

        // 3. High outlier ratio -> RobustScaler
        if base_features.outlier_ratio > 0.1 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::RobustScaler,
                parameters: HashMap::new(),
                expected_performance: 0.85,
            });
        }

        // 4. High skewness -> PowerTransformer
        if base_features.mean_skewness.abs() > 1.5 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::PowerTransformer,
                parameters: HashMap::new(),
                expected_performance: 0.8,
            });
        }

        // 5. Low variance features -> VarianceThreshold
        if base_features.variance_ratio < 0.1 {
            let mut params = HashMap::new();
            params.insert("threshold".to_string(), 0.01);
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::VarianceThreshold,
                parameters: params,
                expected_performance: 0.7,
            });
        }

        // Sort by expected performance
        recommendations.sort_by(|a, b| {
            b.expected_performance
                .partial_cmp(&a.expected_performance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(recommendations.into_iter().take(3).collect()) // Return top 3
    }

    /// Get optimized parameters for a transformation type
    fn get_optimized_parameters_for_type(
        &self,
        t_type: &TransformationType,
        features: &EnhancedMetaFeatures,
    ) -> Result<HashMap<String, f64>> {
        let mut params = HashMap::new();
        let base_features = &features.base_features;

        match t_type {
            TransformationType::PCA => {
                // Adaptive n_components based on data characteristics
                let variance_threshold = if base_features.n_features > 1000 {
                    0.99
                } else {
                    0.95
                };
                params.insert("variance_threshold".to_string(), variance_threshold);

                // Estimate reasonable number of components
                let max_components = base_features.n_features.min(base_features.n_samples);
                let estimated_components = if base_features.n_features > base_features.n_samples {
                    (base_features.n_samples as f64 * 0.8) as usize
                } else {
                    (max_components as f64 * variance_threshold) as usize
                };
                params.insert(
                    "n_components".to_string(),
                    estimated_components.max(1) as f64,
                );
            }

            TransformationType::PolynomialFeatures => {
                // Adaptive degree based on dataset size
                let degree = if base_features.n_features > 50 { 2 } else { 3 };
                params.insert("degree".to_string(), degree as f64);
                params.insert("include_bias".to_string(), 1.0);
                params.insert(
                    "interaction_only".to_string(),
                    if base_features.n_features > 20 {
                        1.0
                    } else {
                        0.0
                    },
                );
            }

            TransformationType::VarianceThreshold => {
                // Adaptive threshold based on data characteristics
                let threshold = if base_features.variance_ratio < 0.01 {
                    0.001
                } else {
                    0.01
                };
                params.insert("threshold".to_string(), threshold);
            }

            TransformationType::PowerTransformer => {
                // Choose method based on data characteristics
                let method = if base_features.has_missing || base_features.outlier_ratio > 0.2 {
                    "yeo-johnson" // Can handle zeros and negative values
                } else {
                    "box-cox" // More powerful but requires positive values
                };
                params.insert(
                    "method".to_string(),
                    if method == "yeo-johnson" { 1.0 } else { 0.0 },
                );
                params.insert("standardize".to_string(), 1.0);
            }

            TransformationType::QuantileTransformer => {
                // Adaptive number of quantiles
                let n_quantiles = (base_features.n_samples / 10).max(10).min(1000);
                params.insert("n_quantiles".to_string(), n_quantiles as f64);
                params.insert("output_distribution".to_string(), 0.0); // 0 = uniform, 1 = normal
            }

            _ => {
                // Default parameters for other transformations
            }
        }

        Ok(params)
    }

    /// Estimate efficiency score for a transformation
    fn estimate_efficiency_score(
        &self,
        t_type: &TransformationType,
        features: &EnhancedMetaFeatures,
    ) -> Result<f64> {
        let base_features = &features.base_features;
        let data_size_factor = (base_features.n_samples * base_features.n_features) as f64;
        let log_size = data_size_factor.ln();

        let score = match t_type {
            TransformationType::StandardScaler | TransformationType::MinMaxScaler => {
                1.0 - (log_size / 25.0).min(0.3) // Very efficient, slight penalty for large data
            }
            TransformationType::RobustScaler => {
                0.9 - (log_size / 20.0).min(0.3) // Slightly less efficient due to median computation
            }
            TransformationType::PCA => {
                let complexity_penalty = if base_features.n_features > base_features.n_samples {
                    0.5 // Expensive for wide datasets
                } else {
                    0.3
                };
                0.7 - complexity_penalty - (log_size / 30.0).min(0.2)
            }
            TransformationType::PolynomialFeatures => {
                let feature_penalty = (base_features.n_features as f64 / 100.0).min(0.5);
                0.5 - feature_penalty - (log_size / 15.0).min(0.3)
            }
            TransformationType::PowerTransformer => 0.8 - (log_size / 25.0).min(0.2),
            _ => 0.7, // Default efficiency
        };

        Ok(score.max(0.1).min(1.0))
    }

    /// Estimate interpretability score for a transformation
    fn estimate_interpretability_score(&self, ttype: &TransformationType) -> f64 {
        match t_type {
            TransformationType::StandardScaler | TransformationType::MinMaxScaler => 0.9,
            TransformationType::RobustScaler => 0.85,
            TransformationType::VarianceThreshold => 0.95,
            TransformationType::QuantileTransformer => 0.6,
            TransformationType::PowerTransformer => 0.7,
            TransformationType::PCA => 0.4, // Loses original feature meaning
            TransformationType::PolynomialFeatures => 0.3, // Creates many new features
            TransformationType::BinaryEncoder | TransformationType::TargetEncoder => 0.5,
        }
    }

    /// Estimate robustness score for a transformation
    fn estimate_robustness_score(
        &self,
        t_type: &TransformationType,
        features: &EnhancedMetaFeatures,
    ) -> f64 {
        let base_features = &features.base_features;

        let base_score = match t_type {
            TransformationType::RobustScaler => 0.95,
            TransformationType::QuantileTransformer => 0.9,
            TransformationType::StandardScaler => 0.7,
            TransformationType::MinMaxScaler => 0.6,
            TransformationType::PowerTransformer => 0.8,
            TransformationType::PCA => 0.7,
            TransformationType::PolynomialFeatures => 0.7,
        };

        // Adjust based on data characteristics
        let outlier_penalty = if base_features.outlier_ratio > 0.1 {
            match t_type {
                TransformationType::RobustScaler | TransformationType::QuantileTransformer => 0.0,
            }
        } else {
            0.0
        };

        let missing_penalty = if base_features.has_missing { 0.1 } else { 0.0 };

        (base_score - outlier_penalty - missing_penalty)
            .max(0.1)
            .min(1.0)
    }
}

// Stub implementations when auto-feature-engineering is not enabled
/// Advanced meta-learning system for feature engineering (placeholder)
#[cfg(not(feature = "auto-feature-engineering"))]
pub struct AdvancedMetaLearningSystem;

/// Enhanced meta-features for advanced analysis (placeholder)
#[cfg(not(feature = "auto-feature-engineering"))]
pub struct EnhancedMetaFeatures;

/// Multi-objective recommendation system (placeholder)
#[cfg(not(feature = "auto-feature-engineering"))]
pub struct MultiObjectiveRecommendation;
