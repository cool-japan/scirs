//! Data pipeline APIs for building complex data processing workflows
//!
//! Provides a flexible framework for constructing data processing pipelines with:
//! - Composable pipeline stages
//! - Error handling and recovery
//! - Progress tracking and monitoring
//! - Parallel and streaming execution
//! - Caching and checkpointing

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::{IoError, Result};
use crate::metadata::{Metadata, ProcessingHistoryEntry};
use scirs2_core::parallel_ops::*;
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod advanced_optimization;
pub mod backpressure;
mod builders;
mod executors;
pub mod sinks;
pub mod sources;
mod stages;
mod transforms;
pub mod typed_transforms;

pub use advanced_optimization::*;
pub use builders::*;
pub use executors::*;
pub use stages::*;
pub use transforms::*;

/// Pipeline data wrapper that carries data and metadata through stages
#[derive(Debug, Clone)]
pub struct PipelineData<T> {
    /// The actual data
    pub data: T,
    /// Associated metadata
    pub metadata: Metadata,
    /// Pipeline execution context
    pub context: PipelineContext,
}

impl<T> PipelineData<T> {
    /// Create new pipeline data
    pub fn new(data: T) -> Self {
        Self {
            data,
            metadata: Metadata::new(),
            context: PipelineContext::new(),
        }
    }

    /// Create with metadata
    pub fn with_metadata(data: T, metadata: Metadata) -> Self {
        Self {
            data,
            metadata,
            context: PipelineContext::new(),
        }
    }

    /// Transform the data while preserving metadata
    pub fn map<U, F>(self, f: F) -> PipelineData<U>
    where
        F: FnOnce(T) -> U,
    {
        PipelineData {
            data: f(self.data),
            metadata: self.metadata,
            context: self.context,
        }
    }

    /// Transform the data with potential failure
    pub fn try_map<U, F>(self, f: F) -> Result<PipelineData<U>>
    where
        F: FnOnce(T) -> Result<U>,
    {
        Ok(PipelineData {
            data: f(self.data)?,
            metadata: self.metadata,
            context: self.context,
        })
    }
}

/// Pipeline execution context
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Shared state between pipeline stages
    pub state: Arc<Mutex<HashMap<String, Box<dyn Any + Send + Sync>>>>,
    /// Execution statistics
    pub stats: Arc<Mutex<PipelineStats>>,
    /// Configuration parameters
    pub config: PipelineConfig,
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineContext {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(PipelineStats::new())),
            config: PipelineConfig::default(),
        }
    }

    /// Store a value in the context
    pub fn set<T: Any + Send + Sync + 'static>(&self, key: &str, value: T) {
        let mut state = self.state.lock().expect("Operation failed");
        state.insert(key.to_string(), Box::new(value));
    }

    /// Retrieve a value from the context
    pub fn get<T>(&self, key: &str) -> Option<T>
    where
        T: Any + Send + Sync + Clone + 'static,
    {
        let state = self.state.lock().expect("Operation failed");
        state.get(key).and_then(|v| v.downcast_ref::<T>()).cloned()
    }
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Enable parallel execution where possible
    pub parallel: bool,
    /// Number of worker threads for parallel execution
    pub num_threads: Option<usize>,
    /// Enable progress tracking
    pub track_progress: bool,
    /// Enable caching of intermediate results
    pub enable_cache: bool,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
    /// Maximum memory usage (bytes)
    pub max_memory: Option<usize>,
    /// Enable checkpointing
    pub checkpoint: bool,
    /// Checkpoint interval
    #[serde(with = "serde_duration")]
    pub checkpoint_interval: Duration,
}

mod serde_duration {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            num_threads: None,
            track_progress: true,
            enable_cache: false,
            cache_dir: None,
            max_memory: None,
            checkpoint: false,
            checkpoint_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Pipeline execution statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total execution time
    pub total_time: Duration,
    /// Execution time per stage
    pub stage_times: HashMap<String, Duration>,
    /// Memory usage per stage
    pub memory_usage: HashMap<String, usize>,
    /// Number of items processed
    pub items_processed: usize,
    /// Number of errors
    pub errors: usize,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStats {
    pub fn new() -> Self {
        Self {
            total_time: Duration::from_secs(0),
            stage_times: HashMap::new(),
            memory_usage: HashMap::new(),
            items_processed: 0,
            errors: 0,
        }
    }
}

/// Main pipeline structure
pub struct Pipeline<I, O> {
    /// Pipeline stages
    stages: Vec<Box<dyn PipelineStage>>,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Input type marker
    _input: PhantomData<I>,
    /// Output type marker
    _output: PhantomData<O>,
}

impl<I, O> Default for Pipeline<I, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I, O> Pipeline<I, O> {
    /// Create a new pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            config: PipelineConfig::default(),
            _input: PhantomData,
            _output: PhantomData,
        }
    }

    /// Set pipeline configuration
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a stage to the pipeline
    pub fn add_stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Execute the pipeline
    pub fn execute(&self, input: I) -> Result<O>
    where
        I: 'static + Send + Sync,
        O: 'static + Send + Sync,
    {
        let start_time = Instant::now();
        let mut data = PipelineData::new(Box::new(input) as Box<dyn Any + Send + Sync>);
        data.context.config = self.config.clone();

        // Execute each stage
        for (i, stage) in self.stages.iter().enumerate() {
            let stage_start = Instant::now();

            // Update metadata with processing history
            let entry = ProcessingHistoryEntry::new(stage.name())
                .with_parameter("stage_index", i as i64)
                .with_parameter("stage_type", stage.stage_type());
            data.metadata.add_processing_history(entry)?;

            // Execute stage
            data = stage.execute(data)?;

            // Update statistics
            let mut stats = data.context.stats.lock().expect("Operation failed");
            stats
                .stage_times
                .insert(stage.name(), stage_start.elapsed());
            stats.items_processed += 1;
        }

        // Update total execution time
        {
            let mut stats = data.context.stats.lock().expect("Operation failed");
            stats.total_time = start_time.elapsed();
        }

        // Extract output
        data.data
            .downcast::<O>()
            .map(|boxed| *boxed)
            .map_err(|_| IoError::Other("Pipeline output type mismatch".to_string()))
    }

    /// Execute the pipeline with progress tracking
    pub fn execute_with_progress<F>(&self, input: I, progress_callback: F) -> Result<O>
    where
        I: 'static + Send + Sync,
        O: 'static + Send + Sync,
        F: Fn(usize, usize, &str),
    {
        let start_time = Instant::now();
        let mut data = PipelineData::new(Box::new(input) as Box<dyn Any + Send + Sync>);
        data.context.config = self.config.clone();

        let total_stages = self.stages.len();

        // Execute each stage with progress
        for (i, stage) in self.stages.iter().enumerate() {
            progress_callback(i + 1, total_stages, &stage.name());

            let stage_start = Instant::now();

            // Update metadata
            let entry = ProcessingHistoryEntry::new(stage.name())
                .with_parameter("stage_index", i as i64)
                .with_parameter("stage_type", stage.stage_type());
            data.metadata.add_processing_history(entry)?;

            // Execute stage
            data = stage.execute(data)?;

            // Update statistics
            let mut stats = data.context.stats.lock().expect("Operation failed");
            stats
                .stage_times
                .insert(stage.name(), stage_start.elapsed());
            stats.items_processed += 1;
        }

        // Update total execution time
        {
            let mut stats = data.context.stats.lock().expect("Operation failed");
            stats.total_time = start_time.elapsed();
        }

        // Extract output
        data.data
            .downcast::<O>()
            .map(|boxed| *boxed)
            .map_err(|_| IoError::Other("Pipeline output type mismatch".to_string()))
    }

    /// Get pipeline statistics after execution
    pub fn get_stats(&self, context: &PipelineContext) -> PipelineStats {
        context.stats.lock().expect("Operation failed").clone()
    }
}

/// Trait for pipeline stages
pub trait PipelineStage: Send + Sync {
    /// Execute the stage
    fn execute(
        &self,
        input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>>;

    /// Get stage name
    fn name(&self) -> String;

    /// Get stage type
    fn stage_type(&self) -> String {
        "generic".to_string()
    }

    /// Check if stage can handle the input type
    fn can_handle(&self, _inputtype: &str) -> bool {
        true
    }
}

/// Result type for pipeline operations
pub type PipelineResult<T> = std::result::Result<T, PipelineError>;

/// Pipeline-specific error type
#[derive(Debug)]
pub enum PipelineError {
    /// Stage execution error
    StageError { stage: String, error: String },
    /// Type mismatch error
    TypeMismatch { expected: String, actual: String },
    /// Configuration error
    ConfigError(String),
    /// IO error
    IoError(IoError),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StageError { stage, error } => write!(f, "Stage '{}' error: {}", stage, error),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "Type mismatch: expected {}, got {}", expected, actual)
            }
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            Self::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<IoError> for PipelineError {
    fn from(error: IoError) -> Self {
        PipelineError::IoError(error)
    }
}

/// Create a simple function-based pipeline stage
#[allow(dead_code)]
pub fn function_stage<F, I, O>(name: &str, f: F) -> Box<dyn PipelineStage>
where
    F: Fn(I) -> Result<O> + Send + Sync + 'static,
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    Box::new(FunctionStage {
        name: name.to_string(),
        function: Box::new(move |input: Box<dyn Any + Send + Sync>| {
            let typed_input = input
                .downcast::<I>()
                .map_err(|_| IoError::Other("Type mismatch in function stage".to_string()))?;
            let output = f(*typed_input)?;
            Ok(Box::new(output) as Box<dyn Any + Send + Sync>)
        }),
    })
}

struct FunctionStage {
    name: String,
    function:
        Box<dyn Fn(Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> + Send + Sync>,
}

impl PipelineStage for FunctionStage {
    fn execute(
        &self,
        mut input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        input.data = (self.function)(input.data)?;
        Ok(input)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn stage_type(&self) -> String {
        "function".to_string()
    }
}

// Advanced Pipeline Features

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Pipeline serialization for saving/loading pipeline configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedPipeline {
    pub name: String,
    pub version: String,
    pub description: String,
    pub stages: Vec<SerializedStage>,
    pub config: PipelineConfig,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedStage {
    pub name: String,
    pub stage_type: String,
    pub config: serde_json::Value,
}

impl<I, O> Pipeline<I, O> {
    /// Save pipeline configuration to a file
    pub fn save_config(&self, path: impl AsRef<Path>) -> Result<()> {
        let serialized = SerializedPipeline {
            name: "pipeline".to_string(),
            version: "1.0.0".to_string(),
            description: String::new(),
            stages: self
                .stages
                .iter()
                .map(|s| SerializedStage {
                    name: s.name(),
                    stage_type: s.stage_type(),
                    config: serde_json::Value::Null, // Stages would need to implement serialization
                })
                .collect(),
            config: self.config.clone(),
            metadata: Metadata::new(),
        };

        let json = serde_json::to_string_pretty(&serialized)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        std::fs::write(path, json).map_err(IoError::Io)
    }

    /// Load pipeline configuration from a file
    pub fn load_config(path: impl AsRef<Path>) -> Result<SerializedPipeline> {
        let content = std::fs::read_to_string(path).map_err(IoError::Io)?;

        serde_json::from_str(&content).map_err(|e| IoError::SerializationError(e.to_string()))
    }
}

/// Pipeline composition for combining multiple pipelines
pub struct PipelineComposer<I, M, O> {
    first: Pipeline<I, M>,
    second: Pipeline<M, O>,
}

impl<I, M, O> PipelineComposer<I, M, O>
where
    I: 'static + Send + Sync,
    M: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    pub fn new(first: Pipeline<I, M>, second: Pipeline<M, O>) -> Self {
        Self { first, second }
    }

    pub fn execute(&self, input: I) -> Result<O> {
        let intermediate = self.first.execute(input)?;
        self.second.execute(intermediate)
    }
}

/// Data lineage tracker for tracking data transformations
#[derive(Debug, Clone)]
pub struct DataLineage {
    pub id: String,
    pub source: String,
    pub transformations: Vec<TransformationRecord>,
    pub created_at: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct TransformationRecord {
    pub stage_name: String,
    pub timestamp: DateTime<Utc>,
    pub input_hash: String,
    pub output_hash: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

impl DataLineage {
    pub fn new(source: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            source: source.into(),
            transformations: Vec::new(),
            created_at: now,
            last_modified: now,
        }
    }

    pub fn add_transformation(&mut self, record: TransformationRecord) {
        self.transformations.push(record);
        self.last_modified = Utc::now();
    }

    /// Generate a lineage graph in DOT format
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph DataLineage {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str(&format!(
            "  source [label=\"{}\" shape=box];\n",
            self.source
        ));

        let mut prev = "source".to_string();
        for (i, transform) in self.transformations.iter().enumerate() {
            let node_id = format!("t{i}");
            dot.push_str(&format!(
                "  {} [label=\"{}\"];\n",
                node_id, transform.stage_name
            ));
            dot.push_str(&format!("  {prev} -> {node_id};\n"));
            prev = node_id;
        }

        dot.push_str("}\n");
        dot
    }
}

/// Pipeline optimizer for reordering stages
pub struct PipelineOptimizer;

impl PipelineOptimizer {
    /// Analyze pipeline and suggest optimizations
    pub fn analyze<I, O>(pipeline: &Pipeline<I, O>) -> OptimizationReport {
        OptimizationReport {
            suggestions: vec![
                OptimizationSuggestion {
                    category: "performance".to_string(),
                    description: "Consider moving filter stages earlier in the _pipeline"
                        .to_string(),
                    impact: "high".to_string(),
                },
                OptimizationSuggestion {
                    category: "memory".to_string(),
                    description: "Enable streaming for large datasets".to_string(),
                    impact: "medium".to_string(),
                },
            ],
            estimated_improvement: 0.25,
        }
    }

    /// Optimize stage ordering for better performance
    pub fn optimize_ordering(stages: Vec<Box<dyn PipelineStage>>) -> Vec<Box<dyn PipelineStage>> {
        // Simple heuristic: move filters and validations earlier
        let mut filters = Vec::new();
        let mut others = Vec::new();

        for stage in stages {
            match stage.stage_type().as_str() {
                "filter" | "validation" => filters.push(stage),
                _ => others.push(stage),
            }
        }

        filters.extend(others);
        filters
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub suggestions: Vec<OptimizationSuggestion>,
    pub estimated_improvement: f64,
}

#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub category: String,
    pub description: String,
    pub impact: String,
}

/// Pipeline configuration DSL for easier pipeline creation
#[macro_export]
macro_rules! pipeline {
    ($($stage:expr),* $(,)?) => {{
        let mut pipeline = Pipeline::new();
        $(
            pipeline = pipeline.add_stage($stage);
        )*
        pipeline
    }};
}

/// Stage creation DSL
#[macro_export]
macro_rules! stage {
    (read $path:expr) => {
        Box::new(FileReadStage::new($path, FileFormat::Auto))
    };
    (transform $func:expr) => {
        function_stage("transform", $func)
    };
    (filter $pred:expr) => {
        function_stage("filter", move |data| {
            if $pred(&data) {
                Ok(data)
            } else {
                Err(IoError::Other("Filtered out".to_string()))
            }
        })
    };
    (write $path:expr) => {
        Box::new(FileWriteStage::new($path, FileFormat::Auto))
    };
}

/// Pipeline monitoring and alerting
pub struct PipelineMonitor {
    thresholds: MonitoringThresholds,
    alerts: Vec<Alert>,
}

#[derive(Debug, Clone)]
pub struct MonitoringThresholds {
    pub max_execution_time: Duration,
    pub max_memory_usage: usize,
    pub max_error_rate: f64,
}

#[derive(Debug, Clone)]
pub struct Alert {
    pub timestamp: DateTime<Utc>,
    pub severity: AlertSeverity,
    pub message: String,
    pub stage: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl PipelineMonitor {
    pub fn new(thresholds: MonitoringThresholds) -> Self {
        Self {
            thresholds,
            alerts: Vec::new(),
        }
    }

    pub fn check_metrics(&mut self, stats: &PipelineStats) {
        // Check execution time
        if stats.total_time > self.thresholds.max_execution_time {
            self.alerts.push(Alert {
                timestamp: Utc::now(),
                severity: AlertSeverity::Warning,
                message: format!(
                    "Pipeline execution time ({:?}) exceeded threshold ({:?})",
                    stats.total_time, self.thresholds.max_execution_time
                ),
                stage: None,
            });
        }

        // Check error rate
        let total = stats.items_processed as f64;
        let error_rate = if total > 0.0 {
            stats.errors as f64 / total
        } else {
            0.0
        };

        if error_rate > self.thresholds.max_error_rate {
            self.alerts.push(Alert {
                timestamp: Utc::now(),
                severity: AlertSeverity::Error,
                message: format!(
                    "Error rate ({:.2}%) exceeded threshold ({:.2}%)",
                    error_rate * 100.0,
                    self.thresholds.max_error_rate * 100.0
                ),
                stage: None,
            });
        }
    }

    pub fn get_alerts(&self) -> &[Alert] {
        &self.alerts
    }
}
