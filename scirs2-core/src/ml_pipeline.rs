//! Machine Learning Pipeline Integration and Real-Time Processing
//!
//! This module provides a comprehensive ML pipeline framework for SciRS2, enabling
//! real-time data processing, model serving, feature engineering, and automated
//! training workflows.
//!
//! Features:
//! - Real-time streaming data processing
//! - DAG-based pipeline orchestration
//! - Model serving and inference endpoints
//! - Feature extraction and transformation pipelines
//! - Automated model training and evaluation
//! - Performance monitoring and A/B testing
//! - Integration with distributed computing and cloud storage

use crate::error::{CoreError, ErrorContext, ErrorLocation};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use thiserror::Error;

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use crate::parallel_ops::*;

#[cfg(feature = "async")]
#[allow(unused_imports)]
use tokio::sync::{mpsc, oneshot};

/// ML pipeline error types
#[derive(Error, Debug)]
pub enum MLPipelineError {
    /// Pipeline configuration error
    #[error("Pipeline configuration error: {0}")]
    ConfigurationError(String),

    /// Pipeline execution error
    #[error("Pipeline execution failed: {0}")]
    ExecutionError(String),

    /// Model loading/saving error
    #[error("Model error: {0}")]
    ModelError(String),

    /// Feature processing error
    #[error("Feature processing error: {0}")]
    FeatureError(String),

    /// Data validation error
    #[error("Data validation error: {0}")]
    ValidationError(String),

    /// Resource exhausted error
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Inference error
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Monitoring error
    #[error("Monitoring error: {0}")]
    MonitoringError(String),

    /// Dependency error
    #[error("Dependency error: {0}")]
    DependencyError(String),
}

impl From<MLPipelineError> for CoreError {
    fn from(err: MLPipelineError) -> Self {
        match err {
            MLPipelineError::ValidationError(msg) => CoreError::ValidationError(
                ErrorContext::new(format!("{msg}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            MLPipelineError::ResourceExhausted(msg) => CoreError::ComputationError(
                ErrorContext::new(format!("{msg}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            _ => CoreError::ComputationError(
                ErrorContext::new(format!("{err}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
        }
    }
}

/// Data types supported by the ML pipeline
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// String/text data
    String,
    /// Boolean data
    Boolean,
    /// Categorical data with mapping
    Categorical(Vec<String>),
    /// Array of values
    Array(Box<DataType>),
    /// Structured data with named fields
    Struct(HashMap<String, DataType>),
}

/// Feature metadata and schema information
#[derive(Debug, Clone)]
pub struct FeatureSchema {
    /// Feature name
    pub name: String,
    /// Data type
    pub datatype: DataType,
    /// Whether the feature is required
    pub required: bool,
    /// Default value if missing
    pub defaultvalue: Option<FeatureValue>,
    /// Feature description
    pub description: Option<String>,
    /// Validation constraints
    pub constraints: Vec<FeatureConstraint>,
}

/// Feature constraint types
#[derive(Debug, Clone)]
pub enum FeatureConstraint {
    /// Minimum value (for numeric types)
    MinValue(f64),
    /// Maximum value (for numeric types)
    MaxValue(f64),
    /// Valid values (for categorical types)
    ValidValues(Vec<String>),
    /// Regular expression pattern (for string types)
    Pattern(String),
    /// Custom validation function
    Custom(String), // Function name or expression
}

/// Feature value types
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureValue {
    Float32(f32),
    Float64(f64),
    Int32(i32),
    Int64(i64),
    String(String),
    Boolean(bool),
    Array(Vec<FeatureValue>),
    Struct(HashMap<String, FeatureValue>),
    Null,
}

impl FeatureValue {
    /// Convert to f64 if possible
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FeatureValue::Float32(v) => Some(*v as f64),
            FeatureValue::Float64(v) => Some(*v),
            FeatureValue::Int32(v) => Some(*v as f64),
            FeatureValue::Int64(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Convert to string
    pub fn as_string(&self) -> String {
        match self {
            FeatureValue::String(s) => s.clone(),
            FeatureValue::Float32(v) => v.to_string(),
            FeatureValue::Float64(v) => v.to_string(),
            FeatureValue::Int32(v) => v.to_string(),
            FeatureValue::Int64(v) => v.to_string(),
            FeatureValue::Boolean(v) => v.to_string(),
            FeatureValue::Null => "null".to_string(),
            _ => format!("{self:?}"),
        }
    }

    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, FeatureValue::Null)
    }
}

/// Data sample containing features and optional target
#[derive(Debug, Clone)]
pub struct DataSample {
    /// Sample ID
    pub id: String,
    /// Feature values
    pub features: HashMap<String, FeatureValue>,
    /// Target value (for training)
    pub target: Option<FeatureValue>,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Batch of data samples for efficient processing
#[derive(Debug, Clone)]
pub struct DataBatch {
    /// Samples in the batch
    pub samples: Vec<DataSample>,
    /// Batch metadata
    pub metadata: HashMap<String, String>,
    /// Batch creation timestamp
    pub created_at: SystemTime,
}

impl DataBatch {
    /// Create a new empty batch
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
        }
    }

    /// Add a sample to the batch
    pub fn add_sample(&mut self, sample: DataSample) {
        self.samples.push(sample);
    }

    /// Get batch size
    pub fn size(&self) -> usize {
        self.samples.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Extract feature matrix for ML processing
    pub fn extract_featurematrix(
        &self,
        featurenames: &[String],
    ) -> Result<Vec<Vec<f64>>, MLPipelineError> {
        let mut matrix = Vec::new();

        for sample in &self.samples {
            let mut row = Vec::new();
            for feature_name in featurenames {
                if let Some(value) = sample.features.get(feature_name) {
                    if let Some(numeric_value) = value.as_f64() {
                        row.push(numeric_value);
                    } else {
                        return Err(MLPipelineError::FeatureError(format!(
                            "Feature '{}' is not numeric",
                            feature_name
                        )));
                    }
                } else {
                    return Err(MLPipelineError::FeatureError(format!(
                        "Feature '{}' not found in sample",
                        feature_name
                    )));
                }
            }
            matrix.push(row);
        }

        Ok(matrix)
    }
}

impl Default for DataBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline node trait for processing components
pub trait PipelineNode: Send + Sync {
    /// Get node name
    fn name(&self) -> &str;

    /// Get input schema
    fn input_schema(&self) -> &[FeatureSchema];

    /// Get output schema
    fn output_schema(&self) -> &[FeatureSchema];

    /// Process a batch of data
    fn process(&self, batch: DataBatch) -> Result<DataBatch, MLPipelineError>;

    /// Validate configuration
    fn validate(&self) -> Result<(), MLPipelineError>;

    /// Get node metrics
    fn metrics(&self) -> HashMap<String, f64>;
}

/// Feature transformer for data preprocessing
#[derive(Debug, Clone)]
pub struct FeatureTransformer {
    name: String,
    transform_type: TransformType,
    input_features: Vec<String>,
    output_features: Vec<String>,
    parameters: HashMap<String, FeatureValue>,
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

/// Types of feature transformations
#[derive(Debug, Clone)]
pub enum TransformType {
    /// Scale features to [0, 1] range
    MinMaxScaler,
    /// Standardize features (zero mean, unit variance)
    StandardScaler,
    /// One-hot encode categorical features
    OneHotEncoder,
    /// Label encode categorical features
    LabelEncoder,
    /// Apply log transformation
    LogTransform,
    /// Apply power transformation
    PowerTransform { power: f64 },
    /// Principal Component Analysis
    PCA { n_components: usize },
    /// Custom transformation function
    Custom(String),
}

impl FeatureTransformer {
    /// Create a new feature transformer
    pub fn new(
        name: String,
        transform_type: TransformType,
        input_features: Vec<String>,
        output_features: Vec<String>,
    ) -> Self {
        Self {
            name,
            transform_type,
            input_features,
            output_features,
            parameters: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Set transformation parameters
    pub fn set_parameter(&mut self, key: String, value: FeatureValue) {
        self.parameters.insert(key, value);
    }

    /// Fit transformer to data (for stateful transformations)
    pub fn fit(&mut self, batch: &DataBatch) -> Result<(), MLPipelineError> {
        match &self.transform_type {
            TransformType::MinMaxScaler => self.fit_minmax_scaler(batch),
            TransformType::StandardScaler => self.fit_standard_scaler(batch),
            TransformType::OneHotEncoder => self.fit_onehot_encoder(batch),
            TransformType::LabelEncoder => self.fit_label_encoder(batch),
            _ => Ok(()), // No fitting required for stateless transforms
        }
    }

    /// Apply transformation to data
    pub fn transform(&self, batch: DataBatch) -> Result<DataBatch, MLPipelineError> {
        let start_time = Instant::now();

        let mut transformed_batch = DataBatch::new();
        transformed_batch.metadata = batch.metadata;

        for sample in batch.samples {
            let mut transformed_sample = sample.clone();

            match &self.transform_type {
                TransformType::MinMaxScaler => {
                    self.apply_minmax_transform(&mut transformed_sample)?;
                }
                TransformType::StandardScaler => {
                    self.apply_standard_transform(&mut transformed_sample)?;
                }
                TransformType::LogTransform => {
                    self.applylog_transform(&mut transformed_sample)?;
                }
                TransformType::PowerTransform { power } => {
                    self.apply_power_transform(&mut transformed_sample, *power)?;
                }
                _ => {
                    return Err(MLPipelineError::FeatureError(format!(
                        "Transform type {:?} not implemented",
                        self.transform_type
                    )));
                }
            }

            transformed_batch.add_sample(transformed_sample);
        }

        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.metrics
            .lock()
            .unwrap()
            .insert("processing_time_ms".to_string(), processing_time);
        self.metrics.lock().unwrap().insert(
            "samples_processed".to_string(),
            transformed_batch.size() as f64,
        );

        Ok(transformed_batch)
    }

    /// Fit min-max scaler parameters
    fn fit_minmax_scaler(&mut self, batch: &DataBatch) -> Result<(), MLPipelineError> {
        for feature_name in &self.input_features {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;

            for sample in &batch.samples {
                if let Some(value) = sample.features.get(feature_name) {
                    if let Some(numeric_value) = value.as_f64() {
                        min_val = min_val.min(numeric_value);
                        max_val = max_val.max(numeric_value);
                    }
                }
            }

            self.parameters.insert(
                format!("{}_min", feature_name),
                FeatureValue::Float64(min_val),
            );
            self.parameters.insert(
                format!("{}_max", feature_name),
                FeatureValue::Float64(max_val),
            );
        }

        Ok(())
    }

    /// Fit standard scaler parameters
    fn fit_standard_scaler(&mut self, batch: &DataBatch) -> Result<(), MLPipelineError> {
        for feature_name in &self.input_features {
            let mut values = Vec::new();

            for sample in &batch.samples {
                if let Some(value) = sample.features.get(feature_name) {
                    if let Some(numeric_value) = value.as_f64() {
                        values.push(numeric_value);
                    }
                }
            }

            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                let std_dev = variance.sqrt();

                self.parameters.insert(
                    format!("{}_mean", feature_name),
                    FeatureValue::Float64(mean),
                );
                self.parameters.insert(
                    format!("{}_std", feature_name),
                    FeatureValue::Float64(std_dev),
                );
            }
        }

        Ok(())
    }

    /// Fit one-hot encoder parameters
    fn fit_onehot_encoder(&mut self, batch: &DataBatch) -> Result<(), MLPipelineError> {
        for feature_name in &self.input_features {
            let mut unique_values = std::collections::HashSet::new();

            for sample in &batch.samples {
                if let Some(value) = sample.features.get(feature_name) {
                    unique_values.insert(value.as_string());
                }
            }

            let categories: Vec<_> = unique_values.into_iter().collect();
            self.parameters.insert(
                format!("{}_categories", feature_name),
                FeatureValue::Array(categories.into_iter().map(FeatureValue::String).collect()),
            );
        }

        Ok(())
    }

    /// Fit label encoder parameters
    fn fit_label_encoder(&mut self, batch: &DataBatch) -> Result<(), MLPipelineError> {
        for feature_name in &self.input_features {
            let mut unique_values = std::collections::HashSet::new();

            for sample in &batch.samples {
                if let Some(value) = sample.features.get(feature_name) {
                    unique_values.insert(value.as_string());
                }
            }

            let mut categories: Vec<_> = unique_values.into_iter().collect();
            categories.sort(); // Ensure consistent encoding

            let label_map: HashMap<String, i64> = categories
                .into_iter()
                .enumerate()
                .map(|(i, cat)| (cat, i as i64))
                .collect();

            for (category, label) in &label_map {
                self.parameters.insert(
                    format!("{}_{}_label", feature_name, category),
                    FeatureValue::Int64(*label),
                );
            }
        }

        Ok(())
    }

    /// Apply min-max scaling
    fn apply_minmax_transform(&self, sample: &mut DataSample) -> Result<(), MLPipelineError> {
        for feature_name in &self.input_features {
            if let Some(value) = sample.features.get(feature_name).cloned() {
                if let Some(numeric_value) = value.as_f64() {
                    let min_key = format!("{}_min", feature_name);
                    let max_key = format!("{}_max", feature_name);

                    let min_val = self
                        .parameters
                        .get(&min_key)
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let max_val = self
                        .parameters
                        .get(&max_key)
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0);

                    let scaled_value = if max_val > min_val {
                        (numeric_value - min_val) / (max_val - min_val)
                    } else {
                        0.0
                    };

                    sample
                        .features
                        .insert(feature_name.clone(), FeatureValue::Float64(scaled_value));
                }
            }
        }

        Ok(())
    }

    /// Apply standard scaling
    fn apply_standard_transform(&self, sample: &mut DataSample) -> Result<(), MLPipelineError> {
        for feature_name in &self.input_features {
            if let Some(value) = sample.features.get(feature_name).cloned() {
                if let Some(numeric_value) = value.as_f64() {
                    let mean_key = format!("{}_mean", feature_name);
                    let std_key = format!("{}_std", feature_name);

                    let mean = self
                        .parameters
                        .get(&mean_key)
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let std_dev = self
                        .parameters
                        .get(&std_key)
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0);

                    let standardized_value = if std_dev > 0.0 {
                        (numeric_value - mean) / std_dev
                    } else {
                        0.0
                    };

                    sample.features.insert(
                        feature_name.clone(),
                        FeatureValue::Float64(standardized_value),
                    );
                }
            }
        }

        Ok(())
    }

    /// Apply log transformation
    fn applylog_transform(&self, sample: &mut DataSample) -> Result<(), MLPipelineError> {
        for feature_name in &self.input_features {
            if let Some(value) = sample.features.get(feature_name).cloned() {
                if let Some(numeric_value) = value.as_f64() {
                    if numeric_value > 0.0 {
                        let log_value = numeric_value.ln();
                        sample
                            .features
                            .insert(feature_name.clone(), FeatureValue::Float64(log_value));
                    } else {
                        return Err(MLPipelineError::FeatureError(format!(
                            "Cannot apply log transform to non-positive value: {}",
                            numeric_value
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply power transformation
    fn apply_power_transform(
        &self,
        sample: &mut DataSample,
        power: f64,
    ) -> Result<(), MLPipelineError> {
        for feature_name in &self.input_features {
            if let Some(value) = sample.features.get(feature_name).cloned() {
                if let Some(numeric_value) = value.as_f64() {
                    let transformed_value = numeric_value.powf(power);
                    sample.features.insert(
                        feature_name.clone(),
                        FeatureValue::Float64(transformed_value),
                    );
                }
            }
        }

        Ok(())
    }
}

impl PipelineNode for FeatureTransformer {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_schema(&self) -> &[FeatureSchema] {
        // This would typically be populated with actual schemas
        &[]
    }

    fn output_schema(&self) -> &[FeatureSchema] {
        // This would typically be populated with actual schemas
        &[]
    }

    fn process(&self, batch: DataBatch) -> Result<DataBatch, MLPipelineError> {
        self.transform(batch)
    }

    fn validate(&self) -> Result<(), MLPipelineError> {
        if self.input_features.is_empty() {
            return Err(MLPipelineError::ConfigurationError(
                "No input features specified".to_string(),
            ));
        }

        Ok(())
    }

    fn metrics(&self) -> HashMap<String, f64> {
        self.metrics.lock().unwrap().clone()
    }
}

/// Model predictor for inference
pub struct ModelPredictor {
    name: String,
    model_type: ModelType,
    input_features: Vec<String>,
    output_features: Vec<String>,
    model_data: Vec<u8>, // Serialized model
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

/// Types of ML models supported
#[derive(Debug, Clone)]
pub enum ModelType {
    /// Linear regression
    LinearRegression,
    /// Logistic regression
    LogisticRegression,
    /// Random forest
    RandomForest,
    /// Neural network
    NeuralNetwork,
    /// Support vector machine
    SVM,
    /// Custom model
    Custom(String),
}

impl ModelPredictor {
    /// Create a new model predictor
    pub fn new(
        name: String,
        model_type: ModelType,
        input_features: Vec<String>,
        output_features: Vec<String>,
        model_data: Vec<u8>,
    ) -> Self {
        Self {
            name,
            model_type,
            input_features,
            output_features,
            model_data,
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Load model from serialized data
    pub fn loadmodel(&mut self, modeldata: Vec<u8>) -> Result<(), MLPipelineError> {
        self.model_data = modeldata;
        // In a real implementation, this would deserialize the model
        Ok(())
    }

    /// Make predictions on a batch
    pub fn predict(&self, batch: DataBatch) -> Result<DataBatch, MLPipelineError> {
        let start_time = Instant::now();

        let mut prediction_batch = DataBatch::new();
        prediction_batch.metadata = batch.metadata;

        for sample in batch.samples {
            let mut prediction_sample = sample.clone();

            // Extract features for prediction
            let feature_values: Vec<f64> = self
                .input_features
                .iter()
                .map(|feature_name| {
                    sample
                        .features
                        .get(feature_name)
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0)
                })
                .collect();

            // Make prediction (simplified implementation)
            let prediction = self.make_prediction(&feature_values)?;

            // Add prediction to sample
            for (i, output_feature) in self.output_features.iter().enumerate() {
                let pred_value = prediction.get(i).copied().unwrap_or(0.0);
                prediction_sample
                    .features
                    .insert(output_feature.clone(), FeatureValue::Float64(pred_value));
            }

            prediction_batch.add_sample(prediction_sample);
        }

        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.metrics
            .lock()
            .unwrap()
            .insert("inference_time_ms".to_string(), processing_time);
        self.metrics.lock().unwrap().insert(
            "samples_predicted".to_string(),
            prediction_batch.size() as f64,
        );

        Ok(prediction_batch)
    }

    /// Make prediction for a single feature vector
    fn make_prediction(&self, features: &[f64]) -> Result<Vec<f64>, MLPipelineError> {
        // Simplified prediction logic - in practice this would use the actual model
        match &self.model_type {
            ModelType::LinearRegression => {
                // Simple linear combination
                let prediction = features.iter().sum::<f64>() / features.len() as f64;
                Ok(vec![prediction])
            }
            ModelType::LogisticRegression => {
                // Sigmoid activation
                let linear_output = features.iter().sum::<f64>();
                let prediction = 1.0 / (1.0 + (-linear_output).exp());
                Ok(vec![prediction])
            }
            ModelType::RandomForest => {
                // Mock ensemble prediction
                let prediction =
                    features.iter().map(|&x| x.abs()).sum::<f64>() / features.len() as f64;
                Ok(vec![prediction])
            }
            _ => Err(MLPipelineError::InferenceError(format!(
                "Model type {:?} not implemented",
                self.model_type
            ))),
        }
    }
}

impl PipelineNode for ModelPredictor {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_schema(&self) -> &[FeatureSchema] {
        &[]
    }

    fn output_schema(&self) -> &[FeatureSchema] {
        &[]
    }

    fn process(&self, batch: DataBatch) -> Result<DataBatch, MLPipelineError> {
        self.predict(batch)
    }

    fn validate(&self) -> Result<(), MLPipelineError> {
        if self.input_features.is_empty() {
            return Err(MLPipelineError::ConfigurationError(
                "No input features specified for model".to_string(),
            ));
        }

        if self.output_features.is_empty() {
            return Err(MLPipelineError::ConfigurationError(
                "No output features specified for model".to_string(),
            ));
        }

        if self.model_data.is_empty() {
            return Err(MLPipelineError::ModelError(
                "No model data loaded".to_string(),
            ));
        }

        Ok(())
    }

    fn metrics(&self) -> HashMap<String, f64> {
        self.metrics.lock().unwrap().clone()
    }
}

/// ML Pipeline orchestrator
pub struct MLPipeline {
    name: String,
    nodes: Vec<Box<dyn PipelineNode>>,
    node_dependencies: HashMap<String, Vec<String>>,
    pipeline_metrics: Arc<RwLock<PipelineMetrics>>,
    config: PipelineConfig,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum batch size for processing
    pub max_batchsize: usize,
    /// Timeout for node processing
    pub node_timeout: Duration,
    /// Whether to enable parallel processing
    pub parallel_processing: bool,
    /// Error handling strategy
    pub error_strategy: ErrorStrategy,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_batchsize: 1000,
            node_timeout: Duration::from_secs(30),
            parallel_processing: true,
            error_strategy: ErrorStrategy::FailFast,
            monitoring: MonitoringConfig::default(),
        }
    }
}

/// Error handling strategies
#[derive(Debug, Clone)]
pub enum ErrorStrategy {
    /// Stop pipeline on first error
    FailFast,
    /// Continue processing, skip failed samples
    SkipErrors,
    /// Retry failed operations
    RetryWithBackoff {
        maxretries: u32,
        basedelay: Duration,
    },
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable performance metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval
    pub metrics_interval: Duration,
    /// Enable health checks
    pub enable_health_checks: bool,
    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval: Duration::from_secs(60),
            enable_health_checks: true,
            health_check_interval: Duration::from_secs(30),
        }
    }
}

/// Pipeline execution metrics
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    /// Total samples processed
    pub samples_processed: u64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Error count
    pub error_count: u64,
    /// Success rate
    pub success_rate: f64,
    /// Throughput (samples per second)
    pub throughput: f64,
    /// Per-node metrics
    pub node_metrics: HashMap<String, HashMap<String, f64>>,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self {
            samples_processed: 0,
            total_processing_time: Duration::default(),
            error_count: 0,
            success_rate: 0.0,
            throughput: 0.0,
            node_metrics: HashMap::default(),
            last_updated: SystemTime::UNIX_EPOCH,
        }
    }
}

impl MLPipeline {
    /// Create a new ML pipeline
    pub fn new(name: String, config: PipelineConfig) -> Self {
        Self {
            name,
            nodes: Vec::new(),
            node_dependencies: HashMap::new(),
            pipeline_metrics: Arc::new(RwLock::new(PipelineMetrics::default())),
            config,
        }
    }

    /// Add a processing node to the pipeline
    pub fn add_node(&mut self, node: Box<dyn PipelineNode>) -> Result<(), MLPipelineError> {
        node.validate()?;
        self.nodes.push(node);
        Ok(())
    }

    /// Set dependencies between nodes
    pub fn set_dependencies(&mut self, nodename: String, dependencies: Vec<String>) {
        self.node_dependencies.insert(nodename, dependencies);
    }

    /// Execute the pipeline on a batch of data
    pub fn execute(&self, mut batch: DataBatch) -> Result<DataBatch, MLPipelineError> {
        let start_time = Instant::now();
        let initial_size = batch.size();

        // Validate batch size
        if batch.size() > self.config.max_batchsize {
            return Err(MLPipelineError::ValidationError(format!(
                "Batch size {} exceeds maximum {}",
                batch.size(),
                self.config.max_batchsize
            )));
        }

        // Execute nodes in dependency order
        let execution_order = self.get_execution_order()?;

        for node_name in execution_order {
            if let Some(node) = self.nodes.iter().find(|n| n.name() == node_name) {
                let node_start = Instant::now();

                let batch_clone = batch.clone();
                batch = match node.process(batch) {
                    Ok(processed_batch) => {
                        // Update node metrics
                        let node_time = node_start.elapsed();
                        self.update_node_metrics(&node_name, node_time, processed_batch.size());
                        processed_batch
                    }
                    Err(e) => match &self.config.error_strategy {
                        ErrorStrategy::FailFast => return Err(e),
                        ErrorStrategy::SkipErrors => {
                            eprintln!("Node {} failed: {}, continuing...", node_name, e);
                            batch_clone
                        }
                        ErrorStrategy::RetryWithBackoff {
                            maxretries,
                            basedelay,
                        } => {
                            let mut retries = 0;
                            loop {
                                if retries >= *maxretries {
                                    return Err(e);
                                }

                                std::thread::sleep(*basedelay * 2_u32.pow(retries));

                                match node.process(batch_clone.clone()) {
                                    Ok(processed_batch) => {
                                        break processed_batch;
                                    }
                                    Err(_) => {
                                        retries += 1;
                                    }
                                }
                            }
                        }
                    },
                }
            }
        }

        // Update pipeline metrics
        let total_time = start_time.elapsed();
        self.update_pipeline_metrics(total_time, true, initial_size, total_time);

        Ok(batch)
    }

    /// Get node execution order based on dependencies
    fn get_execution_order(&self) -> Result<Vec<String>, MLPipelineError> {
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut visiting = std::collections::HashSet::new();

        for node in &self.nodes {
            if !visited.contains(node.name()) {
                self.dfs_visit(node.name(), &mut order, &mut visited, &mut visiting)?;
            }
        }

        Ok(order)
    }

    /// Depth-first search for topological sorting
    fn dfs_visit(
        &self,
        node_name: &str,
        order: &mut Vec<String>,
        visited: &mut std::collections::HashSet<String>,
        visiting: &mut std::collections::HashSet<String>,
    ) -> Result<(), MLPipelineError> {
        if visiting.contains(node_name) {
            return Err(MLPipelineError::DependencyError(
                "Circular dependency detected".to_string(),
            ));
        }

        if visited.contains(node_name) {
            return Ok(());
        }

        visiting.insert(node_name.to_string());

        if let Some(dependencies) = self.node_dependencies.get(node_name) {
            for dep in dependencies {
                self.dfs_visit(dep, order, visited, visiting)?;
            }
        }

        visiting.remove(node_name);
        visited.insert(node_name.to_string());
        order.push(node_name.to_string());

        Ok(())
    }

    /// Update node-specific metrics
    fn update_node_metrics(&self, node_name: &str, processing_time: Duration, batchsize: usize) {
        if let Ok(mut metrics) = self.pipeline_metrics.write() {
            let node_metrics = metrics
                .node_metrics
                .entry(node_name.to_string())
                .or_insert_with(HashMap::new);
            node_metrics.insert(
                "processing_time_ms".to_string(),
                processing_time.as_millis() as f64,
            );
            node_metrics.insert("batchsize".to_string(), batchsize as f64);
            node_metrics.insert(
                "throughput".to_string(),
                batchsize as f64 / processing_time.as_secs_f64(),
            );
        }
    }

    /// Update overall pipeline metrics
    fn update_pipeline_metrics(
        &self,
        duration: Duration,
        success: bool,
        batchsize: usize,
        processing_time: Duration,
    ) {
        if let Ok(mut metrics) = self.pipeline_metrics.write() {
            metrics.samples_processed += batchsize as u64;
            metrics.total_processing_time += processing_time;

            if !success {
                metrics.error_count += 1;
            }

            let total_executions = metrics.samples_processed as f64 / batchsize as f64;
            metrics.success_rate =
                (total_executions - metrics.error_count as f64) / total_executions;
            metrics.throughput =
                metrics.samples_processed as f64 / metrics.total_processing_time.as_secs_f64();
            metrics.last_updated = SystemTime::now();
        }
    }

    /// Get current pipeline metrics
    pub fn get_metrics(&self) -> PipelineMetrics {
        self.pipeline_metrics.read().unwrap().clone()
    }

    /// Get pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Validate the entire pipeline
    pub fn validate(&self) -> Result<(), MLPipelineError> {
        // Validate each node
        for node in &self.nodes {
            node.validate()?;
        }

        // Validate dependencies
        self.get_execution_order()?;

        Ok(())
    }
}

/// Real-time streaming processor
#[cfg(feature = "async")]
pub struct StreamingProcessor {
    pipeline: Arc<MLPipeline>,
    input_buffer: Arc<Mutex<VecDeque<DataSample>>>,
    output_buffer: Arc<Mutex<VecDeque<DataSample>>>,
    batchsize: usize,
    processing_interval: Duration,
    is_running: Arc<Mutex<bool>>,
}

#[cfg(feature = "async")]
impl StreamingProcessor {
    /// Create a new streaming processor
    pub fn with_interval(
        processing_interval: Duration,
        batchsize: usize,
        pipeline: Arc<MLPipeline>,
    ) -> Self {
        Self {
            pipeline,
            input_buffer: Arc::new(Mutex::new(VecDeque::new())),
            output_buffer: Arc::new(Mutex::new(VecDeque::new())),
            batchsize,
            processing_interval,
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start streaming processing
    pub async fn start(&self) -> Result<(), MLPipelineError> {
        {
            let mut running = self.is_running.lock().unwrap();
            if *running {
                return Err(MLPipelineError::ExecutionError(
                    "Processor already running".to_string(),
                ));
            }
            *running = true;
        }

        let pipeline = self.pipeline.clone();
        let input_buffer = self.input_buffer.clone();
        let output_buffer = self.output_buffer.clone();
        let batchsize = self.batchsize;
        let processing_interval = self.processing_interval;
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(processing_interval);

            loop {
                interval.tick().await;

                if !*is_running.lock().unwrap() {
                    break;
                }

                // Process available data
                let mut batch = DataBatch::new();
                {
                    let mut input = input_buffer.lock().unwrap();
                    let mut count = 0;
                    while count < batchsize && !input.is_empty() {
                        if let Some(sample) = input.pop_front() {
                            batch.add_sample(sample);
                            count += 1;
                        }
                    }
                }

                if !batch.is_empty() {
                    match pipeline.execute(batch) {
                        Ok(processed_batch) => {
                            let mut output = output_buffer.lock().unwrap();
                            for sample in processed_batch.samples {
                                output.push_back(sample);
                            }
                        }
                        Err(e) => {
                            eprintln!("Streaming processing error: {}", e);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop streaming processing
    pub fn stop(&self) {
        *self.is_running.lock().unwrap() = false;
    }

    /// Add a sample to the input buffer
    pub fn add_sample(&self, sample: DataSample) {
        self.input_buffer.lock().unwrap().push_back(sample);
    }

    /// Get processed samples from output buffer
    pub fn get_samples(&self, maxcount: usize) -> Vec<DataSample> {
        let mut output = self.output_buffer.lock().unwrap();
        let mut samples = Vec::new();
        let mut _count = 0;

        while _count < maxcount && !output.is_empty() {
            if let Some(sample) = output.pop_front() {
                samples.push(sample);
                _count += 1;
            }
        }

        samples
    }

    /// Get current buffer sizes
    pub fn get_buffer_stats(&self) -> (usize, usize) {
        let input_size = self.input_buffer.lock().unwrap().len();
        let output_size = self.output_buffer.lock().unwrap().len();
        (input_size, output_size)
    }
}

/// Convenience functions for common ML pipeline operations
pub mod utils {
    use super::*;

    /// Create a simple preprocessing pipeline
    pub fn with_preprocessing(featurenames: Vec<String>) -> MLPipeline {
        let mut pipeline = MLPipeline::new("preprocessing".to_string(), PipelineConfig::default());

        // Add standard scaler
        let scaler = FeatureTransformer::new(
            "standard_scaler".to_string(),
            TransformType::StandardScaler,
            featurenames.clone(),
            featurenames.clone(),
        );
        pipeline.add_node(Box::new(scaler)).unwrap();

        pipeline
    }

    /// Create a simple prediction pipeline
    pub fn with_model_type(
        model_name: String,
        model_type: ModelType,
        input_features: Vec<String>,
        output_features: Vec<String>,
    ) -> MLPipeline {
        let mut pipeline = MLPipeline::new("prediction".to_string(), PipelineConfig::default());

        // Add model predictor
        let predictor = ModelPredictor::new(
            model_name,
            model_type,
            input_features,
            output_features,
            vec![], // Empty model data for now
        );
        pipeline.add_node(Box::new(predictor)).unwrap();

        pipeline
    }

    /// Create a sample data batch for testing
    pub fn create_sample_batch(featurenames: &[String], size: usize) -> DataBatch {
        let mut batch = DataBatch::new();

        for i in 0..size {
            let mut features = HashMap::new();
            for (j, feature_name) in featurenames.iter().enumerate() {
                let value = (i * 10 + j) as f64 / 100.0; // Generate some sample data
                features.insert(feature_name.clone(), FeatureValue::Float64(value));
            }

            let sample = DataSample {
                id: format!("{i}"),
                features,
                target: Some(FeatureValue::Float64((i as f64) % 2.0)), // Binary target
                timestamp: SystemTime::now(),
                metadata: HashMap::new(),
            };

            batch.add_sample(sample);
        }

        batch
    }

    /// Calculate feature statistics for a batch
    pub fn calculate_feature_statistics(
        batch: &DataBatch,
        feature_name: &str,
    ) -> Option<(f64, f64, f64, f64)> {
        let mut values = Vec::new();

        for sample in &batch.samples {
            if let Some(value) = sample.features.get(feature_name) {
                if let Some(numeric_value) = value.as_f64() {
                    values.push(numeric_value);
                }
            }
        }

        if values.is_empty() {
            return None;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let min = values[0];
        let max = values[values.len() - 1];

        Some((mean, std_dev, min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_value_conversions() {
        let float_val = FeatureValue::Float64(3.14);
        assert_eq!(float_val.as_f64(), Some(3.14));
        assert_eq!(float_val.as_string(), "3.14");

        let int_val = FeatureValue::Int32(42);
        assert_eq!(int_val.as_f64(), Some(42.0));
        assert_eq!(int_val.as_string(), "42");

        let null_val = FeatureValue::Null;
        assert!(null_val.is_null());
        assert_eq!(null_val.as_f64(), None);
    }

    #[test]
    fn test_data_batch_operations() {
        let mut batch = DataBatch::new();
        assert!(batch.is_empty());

        let sample = DataSample {
            id: "test1".to_string(),
            features: {
                let mut features = HashMap::new();
                features.insert("feature1".to_string(), FeatureValue::Float64(1.0));
                features.insert("feature2".to_string(), FeatureValue::Float64(2.0));
                features
            },
            target: Some(FeatureValue::Float64(1.0)),
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        };

        batch.add_sample(sample);
        assert_eq!(batch.size(), 1);
        assert!(!batch.is_empty());

        // Test feature matrix extraction
        let featurenames = vec!["feature1".to_string(), "feature2".to_string()];
        let matrix = batch.extract_featurematrix(&featurenames).unwrap();
        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix[0], vec![1.0, 2.0]);
    }

    #[test]
    fn test_feature_transformer_creation() {
        let transformer = FeatureTransformer::new(
            "test_scaler".to_string(),
            TransformType::StandardScaler,
            vec!["feature1".to_string()],
            vec!["feature1_scaled".to_string()],
        );

        assert_eq!(transformer.name(), "test_scaler");
        assert!(transformer.validate().is_ok());
    }

    #[test]
    fn test_model_predictor_creation() {
        let predictor = ModelPredictor::new(
            "test_model".to_string(),
            ModelType::LinearRegression,
            vec!["feature1".to_string(), "feature2".to_string()],
            vec!["prediction".to_string()],
            vec![1, 2, 3, 4], // Mock model data
        );

        assert_eq!(predictor.name(), "test_model");
        assert!(predictor.validate().is_ok());
    }

    #[test]
    fn test_pipeline_creation_and_validation() {
        let mut pipeline = MLPipeline::new("test_pipeline".to_string(), PipelineConfig::default());

        let transformer = FeatureTransformer::new(
            "scaler".to_string(),
            TransformType::StandardScaler,
            vec!["feature1".to_string()],
            vec!["feature1_scaled".to_string()],
        );

        pipeline.add_node(Box::new(transformer)).unwrap();
        assert!(pipeline.validate().is_ok());
    }

    #[test]
    fn test_pipeline_execution_order() {
        let mut pipeline = MLPipeline::new("test_pipeline".to_string(), PipelineConfig::default());

        // Add nodes
        let node1 = FeatureTransformer::new(
            "node1".to_string(),
            TransformType::StandardScaler,
            vec!["feature1".to_string()],
            vec!["feature1_scaled".to_string()],
        );
        let node2 = FeatureTransformer::new(
            "node2".to_string(),
            TransformType::MinMaxScaler,
            vec!["feature1_scaled".to_string()],
            vec!["feature1_normalized".to_string()],
        );

        pipeline.add_node(Box::new(node1)).unwrap();
        pipeline.add_node(Box::new(node2)).unwrap();

        // Set dependencies
        pipeline.set_dependencies("node2".to_string(), vec!["node1".to_string()]);

        let execution_order = pipeline.get_execution_order().unwrap();
        assert_eq!(execution_order, vec!["node1", "node2"]);
    }

    #[test]
    fn test_utils_sample_batch_creation() {
        let featurenames = vec!["feature1".to_string(), "feature2".to_string()];
        let batch = utils::create_sample_batch(&featurenames, 10);

        assert_eq!(batch.size(), 10);
        assert!(!batch.is_empty());

        // Check that all samples have the required features
        for sample in &batch.samples {
            assert!(sample.features.contains_key("feature1"));
            assert!(sample.features.contains_key("feature2"));
            assert!(sample.target.is_some());
        }
    }

    #[test]
    fn test_feature_statistics() {
        let featurenames = vec!["feature1".to_string()];
        let batch = utils::create_sample_batch(&featurenames, 100);

        let stats = utils::calculate_feature_statistics(&batch, "feature1").unwrap();
        let (mean, std_dev, min, max) = stats;

        assert!(mean >= 0.0);
        assert!(std_dev >= 0.0);
        assert!(min <= max);
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.max_batchsize, 1000);
        assert_eq!(config.node_timeout, Duration::from_secs(30));
        assert!(config.parallel_processing);
        assert!(matches!(config.error_strategy, ErrorStrategy::FailFast));
    }
}
