//! Common pipeline stages for data processing

#![allow(dead_code)]
#![allow(missing_docs)]

use super::*;
use crate::csv::{read_csv, write_csv};
use crate::error::Result;
use scirs2_core::ndarray::Array2;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// File reading stage
pub struct FileReadStage {
    path: PathBuf,
    format: FileFormat,
}

#[derive(Debug, Clone)]
pub enum FileFormat {
    Csv,
    Json,
    Binary,
    Text,
    Auto,
}

impl FileReadStage {
    pub fn new(path: impl AsRef<Path>, format: FileFormat) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            format,
        }
    }
}

impl PipelineStage for FileReadStage {
    fn execute(
        &self,
        mut input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        let data = match self.format {
            FileFormat::Csv => {
                let data = read_csv(&self.path, None)?;
                Box::new(data) as Box<dyn Any + Send + Sync>
            }
            FileFormat::Json => {
                let file = File::open(&self.path).map_err(IoError::Io)?;
                let value: serde_json::Value = serde_json::from_reader(file)
                    .map_err(|e| IoError::SerializationError(e.to_string()))?;
                Box::new(value) as Box<dyn Any + Send + Sync>
            }
            FileFormat::Binary => {
                let data = std::fs::read(&self.path).map_err(IoError::Io)?;
                Box::new(data) as Box<dyn Any + Send + Sync>
            }
            FileFormat::Text => {
                let data = std::fs::read_to_string(&self.path).map_err(IoError::Io)?;
                Box::new(data) as Box<dyn Any + Send + Sync>
            }
            FileFormat::Auto => {
                // Auto-detect format based on file extension
                let extension = self
                    .path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("");

                match extension.to_lowercase().as_str() {
                    "csv" => {
                        let data = read_csv(&self.path, None)?;
                        Box::new(data) as Box<dyn Any + Send + Sync>
                    }
                    "json" => {
                        let file = File::open(&self.path).map_err(IoError::Io)?;
                        let value: serde_json::Value = serde_json::from_reader(file)
                            .map_err(|e| IoError::SerializationError(e.to_string()))?;
                        Box::new(value) as Box<dyn Any + Send + Sync>
                    }
                    "txt" | "text" => {
                        let data = std::fs::read_to_string(&self.path).map_err(IoError::Io)?;
                        Box::new(data) as Box<dyn Any + Send + Sync>
                    }
                    _ => {
                        // Default to binary for unknown extensions
                        let data = std::fs::read(&self.path).map_err(IoError::Io)?;
                        Box::new(data) as Box<dyn Any + Send + Sync>
                    }
                }
            }
        };

        input.data = data;
        input
            .metadata
            .set("source_file", self.path.to_string_lossy().to_string());
        Ok(input)
    }

    fn name(&self) -> String {
        format!("read_{:?}", self.format)
    }

    fn stage_type(&self) -> String {
        "input".to_string()
    }
}

/// File writing stage
pub struct FileWriteStage {
    path: PathBuf,
    format: FileFormat,
}

impl FileWriteStage {
    pub fn new(path: impl AsRef<Path>, format: FileFormat) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            format,
        }
    }
}

impl PipelineStage for FileWriteStage {
    fn execute(
        &self,
        input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        match self.format {
            FileFormat::Csv => {
                if let Some(data) = input.data.downcast_ref::<Array2<f64>>() {
                    write_csv(&self.path, data, None, None)?;
                }
            }
            FileFormat::Json => {
                if let Some(value) = input.data.downcast_ref::<serde_json::Value>() {
                    let file = File::create(&self.path).map_err(IoError::Io)?;
                    serde_json::to_writer_pretty(file, value)
                        .map_err(|e| IoError::SerializationError(e.to_string()))?;
                }
            }
            FileFormat::Binary => {
                if let Some(data) = input.data.downcast_ref::<Vec<u8>>() {
                    std::fs::write(&self.path, data).map_err(IoError::Io)?;
                }
            }
            FileFormat::Text => {
                if let Some(data) = input.data.downcast_ref::<String>() {
                    std::fs::write(&self.path, data).map_err(IoError::Io)?;
                }
            }
            FileFormat::Auto => {
                // Auto-detect format based on file extension
                let extension = self
                    .path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("");

                match extension.to_lowercase().as_str() {
                    "csv" => {
                        if let Some(data) = input.data.downcast_ref::<Array2<f64>>() {
                            write_csv(&self.path, data, None, None)?;
                        }
                    }
                    "json" => {
                        if let Some(value) = input.data.downcast_ref::<serde_json::Value>() {
                            let file = File::create(&self.path).map_err(IoError::Io)?;
                            serde_json::to_writer_pretty(file, value)
                                .map_err(|e| IoError::SerializationError(e.to_string()))?;
                        }
                    }
                    "txt" | "text" => {
                        if let Some(data) = input.data.downcast_ref::<String>() {
                            std::fs::write(&self.path, data).map_err(IoError::Io)?;
                        }
                    }
                    _ => {
                        // Default to binary for unknown extensions
                        if let Some(data) = input.data.downcast_ref::<Vec<u8>>() {
                            std::fs::write(&self.path, data).map_err(IoError::Io)?;
                        }
                    }
                }
            }
        }

        Ok(input)
    }

    fn name(&self) -> String {
        format!("write_{:?}", self.format)
    }

    fn stage_type(&self) -> String {
        "output".to_string()
    }
}

/// Data validation stage
pub struct ValidationStage {
    validators: Vec<Box<dyn Validator>>,
}

pub trait Validator: Send + Sync {
    fn validate(&self, data: &dyn Any) -> Result<()>;
    fn name(&self) -> &str;
}

impl Default for ValidationStage {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationStage {
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    pub fn add_validator(mut self, validator: Box<dyn Validator>) -> Self {
        self.validators.push(validator);
        self
    }
}

impl PipelineStage for ValidationStage {
    fn execute(
        &self,
        input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        for validator in &self.validators {
            validator.validate(input.data.as_ref())?;
        }
        Ok(input)
    }

    fn name(&self) -> String {
        "validation".to_string()
    }

    fn stage_type(&self) -> String {
        "validation".to_string()
    }
}

/// Data transformation stage
pub struct TransformStage {
    name: String,
    transformer: Box<dyn DataTransformer>,
}

pub trait DataTransformer: Send + Sync {
    fn transform(&self, data: Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>>;
}

impl TransformStage {
    pub fn new(name: &str, transformer: Box<dyn DataTransformer>) -> Self {
        Self {
            name: name.to_string(),
            transformer,
        }
    }
}

impl PipelineStage for TransformStage {
    fn execute(
        &self,
        mut input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        input.data = self.transformer.transform(input.data)?;
        Ok(input)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn stage_type(&self) -> String {
        "transform".to_string()
    }
}

/// Aggregation stage
pub struct AggregationStage<T> {
    name: String,
    aggregator: Box<dyn Fn(Vec<T>) -> Result<T> + Send + Sync>,
}

impl<T: 'static + Send + Sync> AggregationStage<T> {
    pub fn new<F>(name: &str, aggregator: F) -> Self
    where
        F: Fn(Vec<T>) -> Result<T> + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            aggregator: Box::new(aggregator),
        }
    }
}

impl<T: 'static + Send + Sync> PipelineStage for AggregationStage<T> {
    fn execute(
        &self,
        mut input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        if let Ok(data) = input.data.downcast::<Vec<T>>() {
            let aggregated = (self.aggregator)(*data)?;
            input.data = Box::new(aggregated) as Box<dyn Any + Send + Sync>;
            Ok(input)
        } else {
            Err(IoError::Other(
                "Type mismatch in aggregation stage".to_string(),
            ))
        }
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn stage_type(&self) -> String {
        "aggregation".to_string()
    }
}

/// Filtering stage
pub struct FilterStage<T> {
    name: String,
    predicate: Box<dyn Fn(&T) -> bool + Send + Sync>,
}

impl<T: 'static + Send + Sync + Clone> FilterStage<T> {
    pub fn new<F>(name: &str, predicate: F) -> Self
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            predicate: Box::new(predicate),
        }
    }
}

impl<T: 'static + Send + Sync + Clone> PipelineStage for FilterStage<T> {
    fn execute(
        &self,
        mut input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        if let Ok(data) = input.data.downcast::<Vec<T>>() {
            let filtered: Vec<T> = data
                .iter()
                .filter(|item| (self.predicate)(item))
                .cloned()
                .collect();
            input.data = Box::new(filtered) as Box<dyn Any + Send + Sync>;
            Ok(input)
        } else {
            Err(IoError::Other("Type mismatch in filter stage".to_string()))
        }
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn stage_type(&self) -> String {
        "filter".to_string()
    }
}

/// Enrichment stage - adds metadata or augments data
pub struct EnrichmentStage {
    name: String,
    enricher: Box<dyn DataEnricher>,
}

pub trait DataEnricher: Send + Sync {
    fn enrich(&self, data: &mut PipelineData<Box<dyn Any + Send + Sync>>) -> Result<()>;
}

impl EnrichmentStage {
    pub fn new(name: &str, enricher: Box<dyn DataEnricher>) -> Self {
        Self {
            name: name.to_string(),
            enricher,
        }
    }
}

impl PipelineStage for EnrichmentStage {
    fn execute(
        &self,
        mut input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        self.enricher.enrich(&mut input)?;
        Ok(input)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn stage_type(&self) -> String {
        "enrichment".to_string()
    }
}

/// Cache stage - caches intermediate results
pub struct CacheStage {
    cache_key: String,
    cache_dir: PathBuf,
}

impl CacheStage {
    pub fn new(cache_key: &str, cache_dir: impl AsRef<Path>) -> Self {
        Self {
            cache_key: cache_key.to_string(),
            cache_dir: cache_dir.as_ref().to_path_buf(),
        }
    }
}

impl PipelineStage for CacheStage {
    fn execute(
        &self,
        mut input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        // Create cache directory if needed
        std::fs::create_dir_all(&self.cache_dir).map_err(IoError::Io)?;

        let cache_path = self.cache_dir.join(format!("{}.cache", self.cache_key));

        // Check if cache exists
        if cache_path.exists() {
            // Try to load from cache
            if let Ok(_cache_data) = std::fs::read(&cache_path) {
                // Update metadata to indicate cache hit
                input.metadata.set("cache_hit", true);
                input.metadata.set("cache_key", self.cache_key.clone());

                // For demonstration, we'll store a simple flag in context
                input.context.set("cached_from", self.cache_key.clone());

                return Ok(input);
            }
        }

        // Cache miss - save data for future use
        // Note: In a real implementation, we would serialize the actual data
        // For now, we'll just create a marker file
        let cache_marker = format!(
            "Cache entry for: {}\nCreated: {:?}\n",
            self.cache_key,
            chrono::Utc::now()
        );
        std::fs::write(&cache_path, cache_marker).map_err(IoError::Io)?;

        // Update metadata
        input.metadata.set("cache_hit", false);
        input.metadata.set("cache_key", self.cache_key.clone());

        Ok(input)
    }

    fn name(&self) -> String {
        format!("cache_{}", self.cache_key)
    }

    fn stage_type(&self) -> String {
        "cache".to_string()
    }
}

/// Monitoring stage - logs metrics and progress
pub struct MonitoringStage {
    name: String,
    monitor: Box<dyn Monitor>,
}

pub trait Monitor: Send + Sync {
    fn monitor(&self, data: &PipelineData<Box<dyn Any + Send + Sync>>);
}

impl MonitoringStage {
    pub fn new(name: &str, monitor: Box<dyn Monitor>) -> Self {
        Self {
            name: name.to_string(),
            monitor,
        }
    }
}

impl PipelineStage for MonitoringStage {
    fn execute(
        &self,
        input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        self.monitor.monitor(&input);
        Ok(input)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn stage_type(&self) -> String {
        "monitoring".to_string()
    }
}

/// Error handling stage - catches and handles errors
pub struct ErrorHandlingStage {
    name: String,
    handler: Box<dyn ErrorHandler>,
}

pub trait ErrorHandler: Send + Sync {
    fn handle_error(
        &self,
        error: IoError,
        data: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>>;
}

impl ErrorHandlingStage {
    pub fn new(name: &str, handler: Box<dyn ErrorHandler>) -> Self {
        Self {
            name: name.to_string(),
            handler,
        }
    }
}

impl PipelineStage for ErrorHandlingStage {
    fn execute(
        &self,
        input: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        // In a real pipeline, this would wrap the next stage's execution
        // For now, we'll simulate error handling by checking context for errors

        // Check if there's an error flag in the context
        if let Some(error_msg) = input.context.get::<String>("pipeline_error") {
            // Create an error from the message
            let error = IoError::Other(error_msg);

            // Let the handler decide what to do
            self.handler.handle_error(error, input)
        } else {
            // No error, pass through
            Ok(input)
        }
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn stage_type(&self) -> String {
        "error_handling".to_string()
    }
}

/// Default error handler that logs and retries
pub struct RetryErrorHandler {
    max_retries: usize,
    retry_delay: Duration,
}

impl RetryErrorHandler {
    pub fn new(max_retries: usize) -> Self {
        Self {
            max_retries,
            retry_delay: Duration::from_secs(1),
        }
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.retry_delay = delay;
        self
    }
}

impl ErrorHandler for RetryErrorHandler {
    fn handle_error(
        &self,
        error: IoError,
        mut data: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        // Get current retry count
        let retry_count = data.context.get::<usize>("retry_count").unwrap_or(0);

        if retry_count < self.max_retries {
            // Increment retry count
            data.context.set("retry_count", retry_count + 1);

            // Log retry attempt
            data.metadata.set("last_error", format!("{:?}", error));
            data.metadata.set("retry_attempt", (retry_count + 1) as i64);

            // Clear error flag to retry
            data.context.set::<Option<String>>("pipeline_error", None);

            Ok(data)
        } else {
            // Max retries exceeded
            Err(error)
        }
    }
}

/// Skip error handler that continues on error
pub struct SkipErrorHandler;

impl ErrorHandler for SkipErrorHandler {
    fn handle_error(
        &self,
        _error: IoError,
        mut data: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        // Mark as skipped in metadata
        data.metadata.set("skipped", true);
        data.metadata.set("skip_reason", "error_occurred");

        // Continue processing
        Ok(data)
    }
}

/// Fallback error handler that provides default values
pub struct FallbackErrorHandler<T: Any + Send + Sync + Clone + 'static> {
    fallback_value: T,
}

impl<T: Any + Send + Sync + Clone + 'static> FallbackErrorHandler<T> {
    pub fn new(fallback_value: T) -> Self {
        Self { fallback_value }
    }
}

impl<T: Any + Send + Sync + Clone + 'static> ErrorHandler for FallbackErrorHandler<T> {
    fn handle_error(
        &self,
        _error: IoError,
        mut data: PipelineData<Box<dyn Any + Send + Sync>>,
    ) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        // Replace data with fallback value
        data.data = Box::new(self.fallback_value.clone());
        data.metadata.set("used_fallback", true);

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleValidator;

    impl Validator for SimpleValidator {
        fn validate(&self, data: &dyn Any) -> Result<()> {
            if let Some(nums) = data.downcast_ref::<Vec<i32>>() {
                if nums.is_empty() {
                    return Err(IoError::ValidationError("Empty data".to_string()));
                }
            }
            Ok(())
        }

        fn name(&self) -> &str {
            "simple"
        }
    }

    #[test]
    fn test_validation_stage() {
        let stage = ValidationStage::new().add_validator(Box::new(SimpleValidator));

        let data = PipelineData::new(Box::new(vec![1, 2, 3]) as Box<dyn Any + Send + Sync>);
        let result = stage.execute(data);
        assert!(result.is_ok());

        let empty_data =
            PipelineData::new(Box::new(vec![] as Vec<i32>) as Box<dyn Any + Send + Sync>);
        let result = stage.execute(empty_data);
        assert!(result.is_err());
    }
}
