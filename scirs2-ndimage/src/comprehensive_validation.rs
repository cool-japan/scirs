//! # Enhanced Validation Framework for Advanced Mode
//!
//! This module provides comprehensive validation and error handling
//! capabilities for Advanced mode operations, ensuring robust
//! and reliable performance in production environments.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

use crate::advanced_fusion_algorithms::{AdvancedConfig, AdvancedState};
use crate::error::{NdimageError, NdimageResult};

/// Comprehensive validation framework for Advanced operations
#[derive(Debug, Clone)]
pub struct ComprehensiveValidator {
    /// Validation configuration
    config: ValidationConfig,
    /// Performance benchmarks
    benchmarks: HashMap<String, PerformanceBenchmark>,
    /// Error tracking
    errorhistory: Vec<ValidationError>,
}

/// Validation configuration parameters
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable strict numerical validation
    pub strict_numerical: bool,
    /// Maximum allowed processing time per pixel (nanoseconds)
    pub max_time_per_pixel: u64,
    /// Minimum output quality threshold
    pub min_quality_threshold: f64,
    /// Enable memory usage monitoring
    pub monitor_memory: bool,
    /// Enable quantum coherence validation
    pub validate_quantum_coherence: bool,
    /// Enable consciousness state validation
    pub validate_consciousnessstate: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_numerical: true,
            max_time_per_pixel: 1000, // 1 microsecond per pixel
            min_quality_threshold: 0.95,
            monitor_memory: true,
            validate_quantum_coherence: true,
            validate_consciousnessstate: true,
        }
    }
}

/// Performance benchmark data
#[derive(Debug, Clone)]
pub struct PerformanceBenchmark {
    /// Operation name
    pub operation: String,
    /// Average execution time
    pub avg_time: Duration,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Number of samples
    pub sample_count: usize,
}

/// Validation error information
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Associated operation
    pub operation: String,
}

impl ComprehensiveValidator {
    /// Create new validator with default configuration
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create new validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            config,
            benchmarks: HashMap::new(),
            errorhistory: Vec::new(),
        }
    }

    /// Validate Advanced configuration
    pub fn validate_config(&mut self, config: &AdvancedConfig) -> NdimageResult<()> {
        // Validate consciousness depth
        if config.consciousness_depth == 0 || config.consciousness_depth > 20 {
            return Err(NdimageError::ConfigurationError(
                "Consciousness depth must be between 1 and 20".to_string(),
            ));
        }

        // Validate meta-learning rate
        if config.meta_learning_rate <= 0.0 || config.meta_learning_rate > 1.0 {
            return Err(NdimageError::ConfigurationError(
                "Meta-learning rate must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate advanced-dimensions
        if config.advanced_dimensions == 0 || config.advanced_dimensions > 64 {
            return Err(NdimageError::ConfigurationError(
                "Advanced-dimensions must be between 1 and 64".to_string(),
            ));
        }

        // Validate temporal window
        if config.temporal_window > 1000 {
            return Err(NdimageError::ConfigurationError(
                "Temporal window too large (max 1000)".to_string(),
            ));
        }

        // Validate quantum coherence threshold
        if config.quantum_coherence_threshold < 0.0 || config.quantum_coherence_threshold > 1.0 {
            return Err(NdimageError::ConfigurationError(
                "Quantum coherence threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate input image
    pub fn validate_inputimage<T>(&mut self, image: ArrayView2<T>) -> NdimageResult<()>
    where
        T: Float + FromPrimitive + Copy + Debug,
    {
        let (height, width) = image.dim();

        // Check minimum dimensions
        if height < 2 || width < 2 {
            return Err(NdimageError::DimensionError(
                "Image must be at least 2x2 pixels".to_string(),
            ));
        }

        // Check maximum dimensions for performance
        if height > 10000 || width > 10000 {
            return Err(NdimageError::DimensionError(
                "Image too large for Advanced processing (max 10000x10000)".to_string(),
            ));
        }

        if self.config.strict_numerical {
            // Check for invalid values
            for &pixel in image.iter() {
                if !pixel.is_finite() {
                    return Err(NdimageError::ComputationError(
                        "Input image contains non-finite values".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate processing output
    pub fn validate_output<T>(
        &mut self,
        output: &Array2<T>,
        state: &AdvancedState,
        processing_time: Duration,
    ) -> NdimageResult<ValidationReport>
    where
        T: Float + FromPrimitive + Copy + Debug,
    {
        let mut report = ValidationReport::new();
        let (height, width) = output.dim();
        let total_pixels = height * width;

        // Performance validation
        let time_per_pixel = processing_time.as_nanos() / total_pixels as u128;
        if time_per_pixel > self.config.max_time_per_pixel as u128 {
            report.warnings.push(format!(
                "Processing _time per pixel ({} ns) exceeds threshold ({} ns)",
                time_per_pixel, self.config.max_time_per_pixel
            ));
        }

        // Numerical validation
        if self.config.strict_numerical {
            for &pixel in output.iter() {
                if !pixel.is_finite() {
                    return Err(NdimageError::ComputationError(
                        "Output contains non-finite values".to_string(),
                    ));
                }
            }
        }

        // Quality validation
        let quality_score = self.compute_quality_score(output)?;
        if quality_score < self.config.min_quality_threshold {
            report.warnings.push(format!(
                "Output quality ({:.3}) below threshold ({:.3})",
                quality_score, self.config.min_quality_threshold
            ));
        }

        // State validation
        if self.config.validate_consciousnessstate {
            self.validate_consciousnessstate(state, &mut report)?;
        }

        // Update benchmarks
        self.update_benchmark("enhanced_processing", processing_time, 0, quality_score);

        report.quality_score = quality_score;
        report.processing_time = processing_time;
        report.total_pixels = total_pixels;

        Ok(report)
    }

    /// Compute quality score for output
    fn compute_quality_score<T>(&self, output: &Array2<T>) -> NdimageResult<f64>
    where
        T: Float + FromPrimitive + Copy,
    {
        let mean = output
            .iter()
            .map(|&x| x.to_f64().unwrap_or(0.0))
            .sum::<f64>()
            / output.len() as f64;

        let variance = output
            .iter()
            .map(|&x| {
                let val = x.to_f64().unwrap_or(0.0);
                (val - mean).powi(2)
            })
            .sum::<f64>()
            / output.len() as f64;

        let std_dev = variance.sqrt();

        // Quality score based on distribution properties
        let dynamic_range = if std_dev > 0.0 { std_dev.min(1.0) } else { 0.5 };
        let mean_quality = if mean.is_finite() && mean >= 0.0 && mean <= 1.0 {
            1.0
        } else {
            0.5
        };

        Ok((dynamic_range + mean_quality) / 2.0)
    }

    /// Validate consciousness state
    fn validate_consciousnessstate(
        &self,
        state: &AdvancedState,
        report: &mut ValidationReport,
    ) -> NdimageResult<()> {
        if state.processing_cycles == 0 {
            report
                .warnings
                .push("No processing cycles recorded".to_string());
        }

        if state.processing_cycles > 1000 {
            report.warnings.push(format!(
                "Excessive processing cycles: {}",
                state.processing_cycles
            ));
        }

        Ok(())
    }

    /// Update performance benchmark
    fn update_benchmark(&mut self, operation: &str, time: Duration, memory: usize, quality: f64) {
        let benchmark =
            self.benchmarks
                .entry(operation.to_string())
                .or_insert(PerformanceBenchmark {
                    operation: operation.to_string(),
                    avg_time: time,
                    memory_usage: memory,
                    quality_score: quality,
                    sample_count: 0,
                });

        // Update running averages
        let count = benchmark.sample_count as f64;
        benchmark.avg_time = Duration::from_nanos(
            ((benchmark.avg_time.as_nanos() as f64 * count + time.as_nanos() as f64)
                / (count + 1.0)) as u64,
        );
        benchmark.memory_usage =
            ((benchmark.memory_usage as f64 * count + memory as f64) / (count + 1.0)) as usize;
        benchmark.quality_score = (benchmark.quality_score * count + quality) / (count + 1.0);
        benchmark.sample_count += 1;
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            benchmarks: self.benchmarks.clone(),
            total_operations: self.benchmarks.values().map(|b| b.sample_count).sum(),
            error_count: self.errorhistory.len(),
        }
    }
}

/// Validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Processing time
    pub processing_time: Duration,
    /// Total pixels processed
    pub total_pixels: usize,
    /// Warnings encountered
    pub warnings: Vec<String>,
    /// Validation passed
    pub passed: bool,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            quality_score: 0.0,
            processing_time: Duration::default(),
            total_pixels: 0,
            warnings: Vec::new(),
            passed: true,
        }
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.passed && self.warnings.is_empty()
    }

    /// Get performance metrics
    pub fn get_pixels_per_second(&self) -> f64 {
        if self.processing_time.as_secs_f64() > 0.0 {
            self.total_pixels as f64 / self.processing_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// All benchmarks
    pub benchmarks: HashMap<String, PerformanceBenchmark>,
    /// Total operations performed
    pub total_operations: usize,
    /// Total errors encountered
    pub error_count: usize,
}

impl PerformanceSummary {
    /// Get average quality score across all operations
    pub fn average_quality(&self) -> f64 {
        if self.benchmarks.is_empty() {
            return 0.0;
        }

        self.benchmarks
            .values()
            .map(|b| b.quality_score)
            .sum::<f64>()
            / self.benchmarks.len() as f64
    }

    /// Get total processing time
    pub fn total_processing_time(&self) -> Duration {
        self.benchmarks
            .values()
            .map(|b| b.avg_time)
            .fold(Duration::default(), |acc, t| acc + t)
    }
}

/// Enhanced Advanced processing with validation
#[allow(dead_code)]
pub fn validated_advanced_processing<T>(
    image: ArrayView2<T>,
    config: &AdvancedConfig,
    previousstate: Option<AdvancedState>,
    validator: &mut ComprehensiveValidator,
) -> NdimageResult<(Array2<T>, AdvancedState, ValidationReport)>
where
    T: Float + FromPrimitive + Copy + Send + Sync + Debug,
{
    // Pre-processing validation
    validator.validate_config(config)?;
    validator.validate_inputimage(image)?;

    let start_time = Instant::now();

    // Perform Advanced processing
    let (output, state) =
        crate::advanced_fusion_algorithms::fusion_processing(image, config, previousstate)?;

    let processing_time = start_time.elapsed();

    // Post-processing validation
    let report = validator.validate_output(&output, &state, processing_time)?;

    Ok((output, state, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_validator_creation() {
        let validator = ComprehensiveValidator::new();
        assert!(validator.benchmarks.is_empty());
        assert!(validator.errorhistory.is_empty());
    }

    #[test]
    fn test_input_validation() {
        let mut validator = ComprehensiveValidator::new();
        let validimage = Array2::<f64>::ones((10, 10));
        assert!(validator.validate_inputimage(validimage.view()).is_ok());

        let smallimage = Array2::<f64>::ones((1, 1));
        assert!(validator.validate_inputimage(smallimage.view()).is_err());
    }

    #[test]
    fn test_config_validation() {
        let mut validator = ComprehensiveValidator::new();
        let mut config = crate::advanced_fusion_algorithms::AdvancedConfig::default();

        // Valid configuration should pass
        assert!(validator.validate_config(&config).is_ok());

        // Invalid consciousness depth should fail
        config.consciousness_depth = 0;
        assert!(validator.validate_config(&config).is_err());
    }

    #[test]
    fn test_quality_score_computation() {
        let validator = ComprehensiveValidator::new();
        let output = Array2::<f64>::ones((10, 10));
        let quality = validator.compute_quality_score(&output).unwrap();
        assert!(quality > 0.0 && quality <= 1.0);
    }
}
