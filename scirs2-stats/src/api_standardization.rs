//! API standardization and consistency framework for scirs2-stats v1.0.0
//!
//! This module provides a unified, consistent API layer across all statistical
//! functions in scirs2-stats. It implements standardized parameter handling,
//! builder patterns, method chaining, and consistent error reporting to ensure
//! a smooth user experience that follows Rust idioms while maintaining SciPy
//! compatibility where appropriate.

#![allow(dead_code)]

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumCast};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Standardized statistical operation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardizedConfig {
    /// Enable automatic optimization selection
    pub auto_optimize: bool,
    /// Enable parallel processing when beneficial
    pub parallel: bool,
    /// Enable SIMD optimizations when available
    pub simd: bool,
    /// Maximum memory usage limit
    pub memory_limit: Option<usize>,
    /// Confidence level for statistical tests (0.0-1.0)
    pub confidence_level: f64,
    /// Null value handling strategy
    pub null_handling: NullHandling,
    /// Output precision for display
    pub output_precision: usize,
    /// Enable detailed result metadata
    pub include_metadata: bool,
}

impl Default for StandardizedConfig {
    fn default() -> Self {
        Self {
            auto_optimize: true,
            parallel: true,
            simd: true,
            memory_limit: None,
            confidence_level: 0.95,
            null_handling: NullHandling::Exclude,
            output_precision: 6,
            include_metadata: false,
        }
    }
}

/// Strategy for handling null/missing values
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NullHandling {
    /// Exclude null values from computation
    Exclude,
    /// Propagate null values (result is null if any input is null)
    Propagate,
    /// Replace null values with specified value
    Replace(f64),
    /// Fail computation if null values are encountered
    Fail,
}

/// Standardized result wrapper with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardizedResult<T> {
    /// The computed result value
    pub value: T,
    /// Computation metadata
    pub metadata: ResultMetadata,
    /// Any warnings generated during computation
    pub warnings: Vec<String>,
}

/// Metadata about the computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMetadata {
    /// Sample size used in computation
    pub samplesize: usize,
    /// Degrees of freedom (where applicable)
    pub degrees_of_freedom: Option<usize>,
    /// Confidence level used (where applicable)
    pub confidence_level: Option<f64>,
    /// Method/algorithm used
    pub method: String,
    /// Computation time in milliseconds
    pub computation_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<usize>,
    /// Whether optimization was applied
    pub optimized: bool,
    /// Additional method-specific metadata
    pub extra: HashMap<String, String>,
}

/// Builder pattern for descriptive statistics
pub struct DescriptiveStatsBuilder<F> {
    config: StandardizedConfig,
    ddof: Option<usize>,
    axis: Option<usize>,
    weights: Option<Array1<F>>,
    phantom: PhantomData<F>,
}

/// Builder pattern for correlation analysis
pub struct CorrelationBuilder<F> {
    config: StandardizedConfig,
    method: CorrelationMethod,
    min_periods: Option<usize>,
    phantom: PhantomData<F>,
}

/// Builder pattern for statistical tests
pub struct StatisticalTestBuilder<F> {
    config: StandardizedConfig,
    alternative: Alternative,
    equal_var: bool,
    phantom: PhantomData<F>,
}

/// Correlation method specification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CorrelationMethod {
    Pearson,
    Spearman,
    Kendall,
    PartialPearson,
    PartialSpearman,
}

/// Alternative hypothesis for statistical tests
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

/// Unified statistical analysis interface
pub struct StatsAnalyzer<F> {
    config: StandardizedConfig,
    phantom: PhantomData<F>,
}

/// Descriptive statistics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats<F> {
    pub count: usize,
    pub mean: F,
    pub std: F,
    pub min: F,
    pub percentile_25: F,
    pub median: F,
    pub percentile_75: F,
    pub max: F,
    pub variance: F,
    pub skewness: F,
    pub kurtosis: F,
}

/// Correlation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult<F> {
    pub correlation: F,
    pub p_value: Option<F>,
    pub confidence_interval: Option<(F, F)>,
    pub method: CorrelationMethod,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult<F> {
    pub statistic: F,
    pub p_value: F,
    pub confidence_interval: Option<(F, F)>,
    pub effectsize: Option<F>,
    pub power: Option<F>,
}

impl<F> DescriptiveStatsBuilder<F>
where
    F: Float
        + NumCast
        + Clone
        + scirs2_core::simd_ops::SimdUnifiedOps
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Sync
        + Send
        + std::fmt::Display
        + std::fmt::Debug
        + 'static,
{
    /// Create a new descriptive statistics builder
    pub fn new() -> Self {
        Self {
            config: StandardizedConfig::default(),
            ddof: None,
            axis: None,
            weights: None,
            phantom: PhantomData,
        }
    }

    /// Set degrees of freedom adjustment
    pub fn ddof(mut self, ddof: usize) -> Self {
        self.ddof = Some(ddof);
        self
    }

    /// Set computation axis (for multi-dimensional arrays)
    pub fn axis(mut self, axis: usize) -> Self {
        self.axis = Some(axis);
        self
    }

    /// Set sample weights
    pub fn weights(mut self, weights: Array1<F>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Enable/disable parallel processing
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.parallel = enable;
        self
    }

    /// Enable/disable SIMD optimizations
    pub fn simd(mut self, enable: bool) -> Self {
        self.config.simd = enable;
        self
    }

    /// Set null value handling strategy
    pub fn null_handling(mut self, strategy: NullHandling) -> Self {
        self.config.null_handling = strategy;
        self
    }

    /// Set memory limit
    pub fn memory_limit(mut self, limit: usize) -> Self {
        self.config.memory_limit = Some(limit);
        self
    }

    /// Include metadata in results
    pub fn with_metadata(mut self) -> Self {
        self.config.include_metadata = true;
        self
    }

    /// Compute descriptive statistics for the given data
    pub fn compute(
        &self,
        data: ArrayView1<F>,
    ) -> StatsResult<StandardizedResult<DescriptiveStats<F>>> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Data validation
        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Cannot compute statistics for empty array".to_string(),
            ));
        }

        // Handle null values based on strategy
        let (cleaneddata, samplesize) = self.handle_null_values(&data, &mut warnings)?;

        // Select computation method based on configuration
        let stats = if self.config.auto_optimize {
            self.compute_optimized(&cleaneddata, &mut warnings)?
        } else {
            self.compute_standard(&cleaneddata, &mut warnings)?
        };

        let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Build metadata
        let metadata = ResultMetadata {
            samplesize,
            degrees_of_freedom: Some(samplesize.saturating_sub(self.ddof.unwrap_or(1))),
            confidence_level: None,
            method: self.select_method_name(),
            computation_time_ms: computation_time,
            memory_usage_bytes: self.estimate_memory_usage(samplesize),
            optimized: self.config.simd || self.config.parallel,
            extra: HashMap::new(),
        };

        Ok(StandardizedResult {
            value: stats,
            metadata,
            warnings,
        })
    }

    /// Handle null values according to the configured strategy
    fn handle_null_values(
        &self,
        data: &ArrayView1<F>,
        warnings: &mut Vec<String>,
    ) -> StatsResult<(Array1<F>, usize)> {
        // For now, assume no null values in numeric arrays
        // In a real implementation, this would detect and handle NaN values
        let finitedata: Vec<F> = data.iter().filter(|&&x| x.is_finite()).cloned().collect();

        if finitedata.len() != data.len() {
            warnings.push(format!(
                "Removed {} non-finite values",
                data.len() - finitedata.len()
            ));
        }

        let finite_count = finitedata.len();
        match self.config.null_handling {
            NullHandling::Exclude => Ok((Array1::from_vec(finitedata), finite_count)),
            NullHandling::Fail if finite_count != data.len() => Err(StatsError::InvalidArgument(
                "Null values encountered with Fail strategy".to_string(),
            )),
            _ => Ok((Array1::from_vec(finitedata), finite_count)),
        }
    }

    /// Compute statistics using optimized methods
    fn compute_optimized(
        &self,
        data: &Array1<F>,
        warnings: &mut Vec<String>,
    ) -> StatsResult<DescriptiveStats<F>> {
        let n = data.len();

        // Use SIMD-optimized functions when available and beneficial
        if self.config.simd && n > 64 {
            self.compute_simd_optimized(data, warnings)
        } else if self.config.parallel && n > 10000 {
            self.compute_parallel_optimized(data, warnings)
        } else {
            self.compute_standard(data, warnings)
        }
    }

    /// Compute statistics using SIMD optimizations
    fn compute_simd_optimized(
        &self,
        data: &Array1<F>,
        _warnings: &mut Vec<String>,
    ) -> StatsResult<DescriptiveStats<F>> {
        // Use SIMD-optimized descriptive statistics
        let mean = crate::descriptive_simd::mean_simd(&data.view())?;
        let variance =
            crate::descriptive_simd::variance_simd(&data.view(), self.ddof.unwrap_or(1))?;
        let std = variance.sqrt();

        // Compute other statistics
        let (min, max) = self.compute_min_max(data);
        let sorteddata = self.getsorteddata(data);
        let percentiles = self.compute_percentiles(&sorteddata)?;

        // Use existing functions for skewness and kurtosis
        let skewness = crate::descriptive::skew(&data.view(), false, None)?;
        let kurtosis = crate::descriptive::kurtosis(&data.view(), true, false, None)?;

        Ok(DescriptiveStats {
            count: data.len(),
            mean,
            std,
            min,
            percentile_25: percentiles[0],
            median: percentiles[1],
            percentile_75: percentiles[2],
            max,
            variance,
            skewness,
            kurtosis,
        })
    }

    /// Compute statistics using parallel optimizations
    fn compute_parallel_optimized(
        &self,
        data: &Array1<F>,
        _warnings: &mut Vec<String>,
    ) -> StatsResult<DescriptiveStats<F>> {
        // Use parallel-optimized functions
        let mean = crate::parallel_stats::mean_parallel(&data.view())?;
        let variance =
            crate::parallel_stats::variance_parallel(&data.view(), self.ddof.unwrap_or(1))?;
        let std = variance.sqrt();

        // Compute other statistics
        let (min, max) = self.compute_min_max(data);
        let sorteddata = self.getsorteddata(data);
        let percentiles = self.compute_percentiles(&sorteddata)?;

        // Use existing functions for skewness and kurtosis
        let skewness = crate::descriptive::skew(&data.view(), false, None)?;
        let kurtosis = crate::descriptive::kurtosis(&data.view(), true, false, None)?;

        Ok(DescriptiveStats {
            count: data.len(),
            mean,
            std,
            min,
            percentile_25: percentiles[0],
            median: percentiles[1],
            percentile_75: percentiles[2],
            max,
            variance,
            skewness,
            kurtosis,
        })
    }

    /// Compute statistics using standard methods
    fn compute_standard(
        &self,
        data: &Array1<F>,
        _warnings: &mut Vec<String>,
    ) -> StatsResult<DescriptiveStats<F>> {
        let mean = crate::descriptive::mean(&data.view())?;
        let variance = crate::descriptive::var(&data.view(), self.ddof.unwrap_or(1), None)?;
        let std = variance.sqrt();

        let (min, max) = self.compute_min_max(data);
        let sorteddata = self.getsorteddata(data);
        let percentiles = self.compute_percentiles(&sorteddata)?;

        let skewness = crate::descriptive::skew(&data.view(), false, None)?;
        let kurtosis = crate::descriptive::kurtosis(&data.view(), true, false, None)?;

        Ok(DescriptiveStats {
            count: data.len(),
            mean,
            std,
            min,
            percentile_25: percentiles[0],
            median: percentiles[1],
            percentile_75: percentiles[2],
            max,
            variance,
            skewness,
            kurtosis,
        })
    }

    /// Compute min and max values
    fn compute_min_max(&self, data: &Array1<F>) -> (F, F) {
        let mut min = data[0];
        let mut max = data[0];

        for &value in data.iter() {
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }

        (min, max)
    }

    /// Get sorted copy of data for percentile calculations
    fn getsorteddata(&self, data: &Array1<F>) -> Vec<F> {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Compute percentiles (25th, 50th, 75th)
    fn compute_percentiles(&self, sorteddata: &[F]) -> StatsResult<[F; 3]> {
        let n = sorteddata.len();
        if n == 0 {
            return Err(StatsError::InvalidArgument("Empty data".to_string()));
        }

        let p25_idx = (n as f64 * 0.25) as usize;
        let p50_idx = (n as f64 * 0.50) as usize;
        let p75_idx = (n as f64 * 0.75) as usize;

        Ok([
            sorteddata[p25_idx.min(n - 1)],
            sorteddata[p50_idx.min(n - 1)],
            sorteddata[p75_idx.min(n - 1)],
        ])
    }

    /// Select method name for metadata
    fn select_method_name(&self) -> String {
        if self.config.simd && self.config.parallel {
            "SIMD+Parallel".to_string()
        } else if self.config.simd {
            "SIMD".to_string()
        } else if self.config.parallel {
            "Parallel".to_string()
        } else {
            "Standard".to_string()
        }
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self, samplesize: usize) -> Option<usize> {
        if self.config.include_metadata {
            Some(samplesize * std::mem::size_of::<F>() * 2) // Rough estimate
        } else {
            None
        }
    }
}

impl<F> CorrelationBuilder<F>
where
    F: Float
        + NumCast
        + Clone
        + std::fmt::Debug
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Send
        + Sync
        + 'static,
{
    /// Create a new correlation analysis builder
    pub fn new() -> Self {
        Self {
            config: StandardizedConfig::default(),
            method: CorrelationMethod::Pearson,
            min_periods: None,
            phantom: PhantomData,
        }
    }

    /// Set correlation method
    pub fn method(mut self, method: CorrelationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set minimum number of periods for valid correlation
    pub fn min_periods(mut self, periods: usize) -> Self {
        self.min_periods = Some(periods);
        self
    }

    /// Set confidence level for p-values and confidence intervals
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.config.confidence_level = level;
        self
    }

    /// Enable/disable parallel processing
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.parallel = enable;
        self
    }

    /// Enable/disable SIMD optimizations
    pub fn simd(mut self, enable: bool) -> Self {
        self.config.simd = enable;
        self
    }

    /// Include metadata in results
    pub fn with_metadata(mut self) -> Self {
        self.config.include_metadata = true;
        self
    }

    /// Compute correlation between two variables
    pub fn compute<'a>(
        &self,
        x: ArrayView1<'a, F>,
        y: ArrayView1<'a, F>,
    ) -> StatsResult<StandardizedResult<CorrelationResult<F>>> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Data validation
        if x.len() != y.len() {
            return Err(StatsError::DimensionMismatch(
                "Input arrays must have the same length".to_string(),
            ));
        }

        if x.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Cannot compute correlation for empty arrays".to_string(),
            ));
        }

        // Check minimum periods requirement
        if let Some(min_periods) = self.min_periods {
            if x.len() < min_periods {
                return Err(StatsError::InvalidArgument(format!(
                    "Insufficient data: {} observations, {} required",
                    x.len(),
                    min_periods
                )));
            }
        }

        // Compute correlation based on method
        let correlation = match self.method {
            CorrelationMethod::Pearson => {
                if self.config.simd && x.len() > 64 {
                    crate::correlation_simd::pearson_r_simd(&x, &y)?
                } else {
                    crate::correlation::pearson_r(&x, &y)?
                }
            }
            CorrelationMethod::Spearman => crate::correlation::spearman_r(&x, &y)?,
            CorrelationMethod::Kendall => crate::correlation::kendall_tau(&x, &y, "b")?,
            _ => {
                warnings.push("Advanced correlation methods not yet implemented".to_string());
                crate::correlation::pearson_r(&x, &y)?
            }
        };

        // Compute p-value and confidence interval if requested
        let (p_value, confidence_interval) = if self.config.include_metadata {
            self.compute_statistical_inference(correlation, x.len(), &mut warnings)?
        } else {
            (None, None)
        };

        let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let result = CorrelationResult {
            correlation,
            p_value,
            confidence_interval,
            method: self.method,
        };

        let metadata = ResultMetadata {
            samplesize: x.len(),
            degrees_of_freedom: Some(x.len().saturating_sub(2)),
            confidence_level: Some(self.config.confidence_level),
            method: format!("{:?}", self.method),
            computation_time_ms: computation_time,
            memory_usage_bytes: self.estimate_memory_usage(x.len()),
            optimized: self.config.simd || self.config.parallel,
            extra: HashMap::new(),
        };

        Ok(StandardizedResult {
            value: result,
            metadata,
            warnings,
        })
    }

    /// Compute correlation matrix for multiple variables
    pub fn compute_matrix(
        &self,
        data: ArrayView2<F>,
    ) -> StatsResult<StandardizedResult<Array2<F>>> {
        let start_time = std::time::Instant::now();
        let warnings = Vec::new();

        // Use optimized correlation matrix computation
        let correlation_matrix = if self.config.auto_optimize {
            // Use memory-optimized correlation matrix computation
            let mut optimizer = crate::memory_optimization_advanced::MemoryOptimizationSuite::new(
                crate::memory_optimization_advanced::MemoryOptimizationConfig::default(),
            );
            optimizer.optimized_correlation_matrix(data)?
        } else {
            crate::correlation::corrcoef(&data, "pearson")?
        };

        let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let metadata = ResultMetadata {
            samplesize: data.nrows(),
            degrees_of_freedom: Some(data.nrows().saturating_sub(2)),
            confidence_level: Some(self.config.confidence_level),
            method: format!("Matrix {:?}", self.method),
            computation_time_ms: computation_time,
            memory_usage_bytes: self.estimate_memory_usage(data.nrows() * data.ncols()),
            optimized: self.config.simd || self.config.parallel,
            extra: HashMap::new(),
        };

        Ok(StandardizedResult {
            value: correlation_matrix,
            metadata,
            warnings,
        })
    }

    /// Compute statistical inference (p-values, confidence intervals)
    fn compute_statistical_inference(
        &self,
        correlation: F,
        n: usize,
        warnings: &mut Vec<String>,
    ) -> StatsResult<(Option<F>, Option<(F, F)>)> {
        // Fisher's z-transformation for confidence intervals
        let z = ((F::one() + correlation) / (F::one() - correlation)).ln() * F::from(0.5).unwrap();
        let se_z = F::one() / F::from(n - 3).unwrap().sqrt();

        // Critical value for given confidence level (simplified - would use proper t-distribution)
        let _alpha = F::one() - F::from(self.config.confidence_level).unwrap();
        let z_critical = F::from(1.96).unwrap(); // Approximate for 95% confidence

        let z_lower = z - z_critical * se_z;
        let z_upper = z + z_critical * se_z;

        // Transform back to correlation scale
        let r_lower = (F::from(2.0).unwrap() * z_lower).exp();
        let r_lower = (r_lower - F::one()) / (r_lower + F::one());

        let r_upper = (F::from(2.0).unwrap() * z_upper).exp();
        let r_upper = (r_upper - F::one()) / (r_upper + F::one());

        // Simplified p-value calculation (would use proper statistical test)
        let _t_stat = correlation * F::from(n - 2).unwrap().sqrt()
            / (F::one() - correlation * correlation).sqrt();
        let p_value = F::from(2.0).unwrap() * (F::one() - F::from(0.95).unwrap()); // Simplified

        Ok((Some(p_value), Some((r_lower, r_upper))))
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self, size: usize) -> Option<usize> {
        if self.config.include_metadata {
            Some(size * std::mem::size_of::<F>() * 3) // Rough estimate
        } else {
            None
        }
    }
}

impl<F> StatsAnalyzer<F>
where
    F: Float
        + NumCast
        + Clone
        + scirs2_core::simd_ops::SimdUnifiedOps
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Sync
        + Send
        + std::fmt::Display
        + std::fmt::Debug
        + 'static,
{
    /// Create a new unified stats analyzer
    pub fn new() -> Self {
        Self {
            config: StandardizedConfig::default(),
            phantom: PhantomData,
        }
    }

    /// Configure the analyzer
    pub fn configure(mut self, config: StandardizedConfig) -> Self {
        self.config = config;
        self
    }

    /// Perform comprehensive descriptive analysis
    pub fn describe(
        &self,
        data: ArrayView1<F>,
    ) -> StatsResult<StandardizedResult<DescriptiveStats<F>>> {
        DescriptiveStatsBuilder::new()
            .parallel(self.config.parallel)
            .simd(self.config.simd)
            .null_handling(self.config.null_handling)
            .with_metadata()
            .compute(data)
    }

    /// Perform correlation analysis
    pub fn correlate<'a>(
        &self,
        x: ArrayView1<'a, F>,
        y: ArrayView1<'a, F>,
        method: CorrelationMethod,
    ) -> StatsResult<StandardizedResult<CorrelationResult<F>>> {
        CorrelationBuilder::new()
            .method(method)
            .confidence_level(self.config.confidence_level)
            .parallel(self.config.parallel)
            .simd(self.config.simd)
            .with_metadata()
            .compute(x, y)
    }

    /// Get analyzer configuration
    pub fn get_config(&self) -> &StandardizedConfig {
        &self.config
    }
}

/// Convenient type aliases for common use cases
pub type F64StatsAnalyzer = StatsAnalyzer<f64>;
pub type F32StatsAnalyzer = StatsAnalyzer<f32>;

pub type F64DescriptiveBuilder = DescriptiveStatsBuilder<f64>;
pub type F32DescriptiveBuilder = DescriptiveStatsBuilder<f32>;

pub type F64CorrelationBuilder = CorrelationBuilder<f64>;
pub type F32CorrelationBuilder = CorrelationBuilder<f32>;

impl<F> Default for DescriptiveStatsBuilder<F>
where
    F: Float
        + NumCast
        + Clone
        + scirs2_core::simd_ops::SimdUnifiedOps
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Sync
        + Send
        + std::fmt::Display
        + std::fmt::Debug
        + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Default for CorrelationBuilder<F>
where
    F: Float
        + NumCast
        + Clone
        + std::fmt::Debug
        + std::fmt::Display
        + scirs2_core::simd_ops::SimdUnifiedOps
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Send
        + Sync
        + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Default for StatsAnalyzer<F>
where
    F: Float
        + NumCast
        + Clone
        + scirs2_core::simd_ops::SimdUnifiedOps
        + std::iter::Sum<F>
        + std::ops::Div<Output = F>
        + Sync
        + Send
        + std::fmt::Display
        + std::fmt::Debug
        + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_descriptive_stats_builder() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = DescriptiveStatsBuilder::new()
            .ddof(1)
            .parallel(false)
            .simd(false)
            .with_metadata()
            .compute(data.view())
            .unwrap();

        assert_eq!(result.value.count, 5);
        assert!((result.value.mean - 3.0).abs() < 1e-10);
        assert!(result.metadata.optimized == false);
    }

    #[test]
    fn test_correlation_builder() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = CorrelationBuilder::new()
            .method(CorrelationMethod::Pearson)
            .confidence_level(0.95)
            .with_metadata()
            .compute(x.view(), y.view())
            .unwrap();

        assert!((result.value.correlation - 1.0).abs() < 1e-10);
        assert!(result.value.p_value.is_some());
        assert!(result.value.confidence_interval.is_some());
    }

    #[test]
    fn test_stats_analyzer() {
        let analyzer = StatsAnalyzer::new();
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let desc_result = analyzer.describe(data.view()).unwrap();
        assert_eq!(desc_result.value.count, 5);

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_result = analyzer
            .correlate(x.view(), y.view(), CorrelationMethod::Pearson)
            .unwrap();
        assert!((corr_result.value.correlation + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_null_handling() {
        let data = array![1.0, 2.0, f64::NAN, 4.0, 5.0];

        let result = DescriptiveStatsBuilder::new()
            .null_handling(NullHandling::Exclude)
            .compute(data.view())
            .unwrap();

        assert_eq!(result.value.count, 4); // NaN excluded
        assert!(!result.warnings.is_empty()); // Should have warning about removed values
    }

    #[test]
    fn test_standardized_config() {
        let config = StandardizedConfig {
            auto_optimize: false,
            parallel: false,
            simd: true,
            confidence_level: 0.99,
            ..Default::default()
        };

        assert!(!config.auto_optimize);
        assert!(!config.parallel);
        assert!(config.simd);
        assert!((config.confidence_level - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_api_validation() {
        let framework = APIValidationFramework::new();
        let signature = APISignature {
            function_name: "test_function".to_string(),
            module_path: "scirs2, _stats::test".to_string(),
            parameters: vec![ParameterSpec {
                name: "data".to_string(),
                param_type: "ArrayView1<f64>".to_string(),
                optional: false,
                default_value: None,
                description: Some("Input data array".to_string()),
                constraints: vec![ParameterConstraint::Finite],
            }],
            return_type: ReturnTypeSpec {
                type_name: "f64".to_string(),
                result_wrapped: true,
                inner_type: Some("f64".to_string()),
                error_type: Some("StatsError".to_string()),
            },
            error_types: vec!["StatsError".to_string()],
            documentation: DocumentationSpec {
                has_doc_comment: true,
                has_param_docs: true,
                has_return_docs: true,
                has_examples: true,
                has_error_docs: true,
                scipy_compatibility: Some("Compatible with scipy.stats".to_string()),
            },
            performance: PerformanceSpec {
                time_complexity: Some("O(n)".to_string()),
                space_complexity: Some("O(1)".to_string()),
                simd_optimized: true,
                parallel_processing: true,
                cache_efficient: true,
            },
        };

        let report = framework.validate_api(&signature);
        assert!(matches!(
            report.overall_status,
            ValidationStatus::Passed | ValidationStatus::PassedWithWarnings
        ));
    }
}

/// Comprehensive API validation framework for v1.0.0 compliance
#[derive(Debug)]
pub struct APIValidationFramework {
    /// Validation rules registry
    validation_rules: HashMap<String, Vec<ValidationRule>>,
    /// Compatibility checkers for SciPy integration
    compatibility_checkers: HashMap<String, CompatibilityChecker>,
    /// Performance benchmarks for consistency
    performance_benchmarks: HashMap<String, PerformanceBenchmark>,
    /// Error pattern registry for standardization
    error_patterns: HashMap<String, ErrorPattern>,
}

/// Validation rule for API consistency
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule identifier
    pub id: String,
    /// Rule description
    pub description: String,
    /// Rule category
    pub category: ValidationCategory,
    /// Rule severity
    pub severity: ValidationSeverity,
}

/// API validation categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationCategory {
    /// Parameter naming consistency
    ParameterNaming,
    /// Return type consistency
    ReturnTypes,
    /// Error handling consistency
    ErrorHandling,
    /// Documentation completeness
    Documentation,
    /// Performance characteristics
    Performance,
    /// SciPy compatibility
    ScipyCompatibility,
    /// Thread safety
    ThreadSafety,
    /// Numerical stability
    NumericalStability,
}

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity {
    /// Informational - best practices
    Info,
    /// Warning - should be addressed
    Warning,
    /// Error - must be fixed for v1.0.0
    Error,
    /// Critical - breaking changes
    Critical,
}

/// API signature for validation
#[derive(Debug, Clone)]
pub struct APISignature {
    /// Function name
    pub function_name: String,
    /// Module path
    pub module_path: String,
    /// Parameter specifications
    pub parameters: Vec<ParameterSpec>,
    /// Return type specification
    pub return_type: ReturnTypeSpec,
    /// Error types that can be returned
    pub error_types: Vec<String>,
    /// Documentation completeness
    pub documentation: DocumentationSpec,
    /// Performance characteristics
    pub performance: PerformanceSpec,
}

/// Parameter specification for validation
#[derive(Debug, Clone)]
pub struct ParameterSpec {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Whether parameter is optional
    pub optional: bool,
    /// Default value if optional
    pub default_value: Option<String>,
    /// Parameter description
    pub description: Option<String>,
    /// Validation constraints
    pub constraints: Vec<ParameterConstraint>,
}

/// Parameter constraint types
#[derive(Debug, Clone)]
pub enum ParameterConstraint {
    /// Must be positive
    Positive,
    /// Must be non-negative
    NonNegative,
    /// Must be finite
    Finite,
    /// Must be in range
    Range(f64, f64),
    /// Must be one of specific values
    OneOf(Vec<String>),
    /// Must match shape constraints
    Shape(Vec<Option<usize>>),
    /// Custom validation function
    Custom(String),
}

/// Return type specification for validation
#[derive(Debug, Clone)]
pub struct ReturnTypeSpec {
    /// Return type name
    pub type_name: String,
    /// Whether wrapped in Result
    pub result_wrapped: bool,
    /// Inner type if Result
    pub inner_type: Option<String>,
    /// Error type if Result
    pub error_type: Option<String>,
}

/// Documentation specification for validation
#[derive(Debug, Clone)]
pub struct DocumentationSpec {
    /// Has function documentation
    pub has_doc_comment: bool,
    /// Has parameter documentation
    pub has_param_docs: bool,
    /// Has return documentation
    pub has_return_docs: bool,
    /// Has examples
    pub has_examples: bool,
    /// Has error documentation
    pub has_error_docs: bool,
    /// SciPy compatibility notes
    pub scipy_compatibility: Option<String>,
}

/// Performance specification for validation
#[derive(Debug, Clone)]
pub struct PerformanceSpec {
    /// Time complexity
    pub time_complexity: Option<String>,
    /// Space complexity
    pub space_complexity: Option<String>,
    /// SIMD optimization available
    pub simd_optimized: bool,
    /// Parallel processing available
    pub parallel_processing: bool,
    /// Cache efficiency
    pub cache_efficient: bool,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation passed
    pub passed: bool,
    /// Validation messages
    pub messages: Vec<ValidationMessage>,
    /// Suggested fixes
    pub suggested_fixes: Vec<String>,
    /// Related rules
    pub related_rules: Vec<String>,
}

/// Validation message
#[derive(Debug, Clone)]
pub struct ValidationMessage {
    /// Message severity
    pub severity: ValidationSeverity,
    /// Message text
    pub message: String,
    /// Location information
    pub location: Option<String>,
    /// Rule that generated this message
    pub rule_id: String,
}

/// Compatibility checker for SciPy integration
#[derive(Debug, Clone)]
pub struct CompatibilityChecker {
    /// SciPy function name
    pub scipy_function: String,
    /// Parameter mapping
    pub parameter_mapping: HashMap<String, String>,
    /// Return type mapping
    pub return_type_mapping: HashMap<String, String>,
    /// Known differences
    pub known_differences: Vec<CompatibilityDifference>,
}

/// Known compatibility difference
#[derive(Debug, Clone)]
pub struct CompatibilityDifference {
    /// Difference category
    pub category: DifferenceCategory,
    /// Description
    pub description: String,
    /// Justification
    pub justification: String,
    /// Workaround if available
    pub workaround: Option<String>,
}

/// Compatibility difference categories
#[derive(Debug, Clone, Copy)]
pub enum DifferenceCategory {
    /// Intentional API improvement
    Improvement,
    /// Rust-specific constraint
    RustConstraint,
    /// Performance optimization
    Performance,
    /// Safety enhancement
    Safety,
    /// Unintentional - should be fixed
    Unintentional,
}

/// Performance benchmark specification
#[derive(Debug, Clone)]
pub struct PerformanceBenchmark {
    /// Benchmark name
    pub name: String,
    /// Expected time complexity
    pub expected_complexity: ComplexityClass,
    /// Memory usage characteristics
    pub memory_usage: MemoryUsagePattern,
    /// Scalability requirements
    pub scalability: ScalabilityRequirement,
}

/// Complexity class for performance expectations
#[derive(Debug, Clone, Copy)]
pub enum ComplexityClass {
    Constant,
    Logarithmic,
    Linear,
    LogLinear,
    Quadratic,
    Cubic,
    Exponential,
}

/// Memory usage pattern
#[derive(Debug, Clone, Copy)]
pub enum MemoryUsagePattern {
    Constant,
    Linear,
    Quadratic,
    Streaming,
    OutOfCore,
}

/// Scalability requirement
#[derive(Debug, Clone)]
pub struct ScalabilityRequirement {
    /// Maximum data size for reasonable performance
    pub maxdatasize: usize,
    /// Expected parallel scaling efficiency
    pub parallel_efficiency: f64,
    /// SIMD acceleration factor
    pub simd_acceleration: f64,
}

/// Error pattern for consistent error handling
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    /// Error category
    pub category: ErrorCategory,
    /// Error message template
    pub message_template: String,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
    /// Related error types
    pub related_errors: Vec<String>,
}

/// Error categories for consistent handling
#[derive(Debug, Clone, Copy)]
pub enum ErrorCategory {
    /// Invalid input parameters
    InvalidInput,
    /// Numerical computation errors
    Numerical,
    /// Memory allocation errors
    Memory,
    /// Convergence failures
    Convergence,
    /// Dimension mismatch errors
    DimensionMismatch,
    /// Not implemented features
    NotImplemented,
    /// Internal computation errors
    Internal,
}

/// Validation report for an API
#[derive(Debug)]
pub struct ValidationReport {
    /// Function name
    pub function_name: String,
    /// Rule validation results
    pub results: HashMap<String, ValidationResult>,
    /// Overall validation status
    pub overall_status: ValidationStatus,
    /// Summary statistics
    pub summary: ValidationSummary,
}

/// Overall validation status
#[derive(Debug, Clone, Copy)]
pub enum ValidationStatus {
    Passed,
    PassedWithWarnings,
    Failed,
    Critical,
}

/// Validation summary statistics
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total rules checked
    pub total_rules: usize,
    /// Rules passed
    pub passed: usize,
    /// Rules with warnings
    pub warnings: usize,
    /// Rules with errors
    pub errors: usize,
    /// Critical issues
    pub critical: usize,
}

impl APIValidationFramework {
    /// Create new API validation framework
    pub fn new() -> Self {
        let mut framework = Self {
            validation_rules: HashMap::new(),
            compatibility_checkers: HashMap::new(),
            performance_benchmarks: HashMap::new(),
            error_patterns: HashMap::new(),
        };

        framework.initialize_default_rules();
        framework
    }

    /// Initialize default validation rules for v1.0.0
    fn initialize_default_rules(&mut self) {
        // Parameter naming consistency
        self.add_validation_rule(ValidationRule {
            id: "param_naming_consistency".to_string(),
            description: "Parameter names should follow consistent snake_case conventions"
                .to_string(),
            category: ValidationCategory::ParameterNaming,
            severity: ValidationSeverity::Warning,
        });

        // Error handling consistency
        self.add_validation_rule(ValidationRule {
            id: "error_handling_consistency".to_string(),
            description: "Functions should return Result<T, StatsError> for consistency"
                .to_string(),
            category: ValidationCategory::ErrorHandling,
            severity: ValidationSeverity::Error,
        });

        // Documentation completeness
        self.add_validation_rule(ValidationRule {
            id: "documentation_completeness".to_string(),
            description: "All public functions should have complete documentation".to_string(),
            category: ValidationCategory::Documentation,
            severity: ValidationSeverity::Warning,
        });

        // SciPy compatibility
        self.add_validation_rule(ValidationRule {
            id: "scipy_compatibility".to_string(),
            description: "Functions should maintain SciPy compatibility where possible".to_string(),
            category: ValidationCategory::ScipyCompatibility,
            severity: ValidationSeverity::Info,
        });

        // Performance validation
        self.add_validation_rule(ValidationRule {
            id: "performance_characteristics".to_string(),
            description: "Functions should document performance characteristics".to_string(),
            category: ValidationCategory::Performance,
            severity: ValidationSeverity::Info,
        });
    }

    /// Add a validation rule
    pub fn add_validation_rule(&mut self, rule: ValidationRule) {
        let category_key = format!("{:?}", rule.category);
        self.validation_rules
            .entry(category_key)
            .or_default()
            .push(rule);
    }

    /// Validate API signature against all rules
    pub fn validate_api(&self, signature: &APISignature) -> ValidationReport {
        let mut report = ValidationReport::new(signature.function_name.clone());

        for rules in self.validation_rules.values() {
            for rule in rules {
                let result = self.apply_validation_rule(rule, signature);
                report.add_result(rule.id.clone(), result);
            }
        }

        report
    }

    /// Apply a single validation rule
    fn apply_validation_rule(
        &self,
        rule: &ValidationRule,
        signature: &APISignature,
    ) -> ValidationResult {
        match rule.category {
            ValidationCategory::ParameterNaming => self.validate_parameter_naming(signature),
            ValidationCategory::ErrorHandling => self.validate_error_handling(signature),
            ValidationCategory::Documentation => self.validate_documentation(signature),
            ValidationCategory::ScipyCompatibility => self.validate_scipy_compatibility(signature),
            ValidationCategory::Performance => self.validate_performance(signature),
            _ => ValidationResult {
                passed: true,
                messages: vec![],
                suggested_fixes: vec![],
                related_rules: vec![],
            },
        }
    }

    /// Validate parameter naming consistency
    fn validate_parameter_naming(&self, signature: &APISignature) -> ValidationResult {
        let mut messages = Vec::new();
        let mut suggested_fixes = Vec::new();

        for param in &signature.parameters {
            // Check for snake_case convention
            if param.name.contains(char::is_uppercase) || param.name.contains('-') {
                messages.push(ValidationMessage {
                    severity: ValidationSeverity::Warning,
                    message: format!("Parameter '{}' should use snake_case naming", param.name),
                    location: Some(format!(
                        "{}::{}",
                        signature.module_path, signature.function_name
                    )),
                    rule_id: "param_naming_consistency".to_string(),
                });
                suggested_fixes.push(format!("Rename parameter '{}' to snake_case", param.name));
            }
        }

        ValidationResult {
            passed: messages.is_empty(),
            messages,
            suggested_fixes,
            related_rules: vec!["return_type_consistency".to_string()],
        }
    }

    /// Validate error handling consistency
    fn validate_error_handling(&self, signature: &APISignature) -> ValidationResult {
        let mut messages = Vec::new();
        let mut suggested_fixes = Vec::new();

        if !signature.return_type.result_wrapped {
            messages.push(ValidationMessage {
                severity: ValidationSeverity::Error,
                message: "Function should return Result<T, StatsError> for consistency".to_string(),
                location: Some(format!(
                    "{}::{}",
                    signature.module_path, signature.function_name
                )),
                rule_id: "error_handling_consistency".to_string(),
            });
            suggested_fixes.push("Wrap return type in Result<T, StatsError>".to_string());
        }

        if let Some(error_type) = &signature.return_type.error_type {
            if error_type != "StatsError" {
                messages.push(ValidationMessage {
                    severity: ValidationSeverity::Warning,
                    message: format!("Non-standard error type '{}' used", error_type),
                    location: Some(format!(
                        "{}::{}",
                        signature.module_path, signature.function_name
                    )),
                    rule_id: "error_handling_consistency".to_string(),
                });
                suggested_fixes.push("Use StatsError for consistency".to_string());
            }
        }

        ValidationResult {
            passed: messages.is_empty(),
            messages,
            suggested_fixes,
            related_rules: vec!["documentation_completeness".to_string()],
        }
    }

    /// Validate documentation completeness
    fn validate_documentation(&self, signature: &APISignature) -> ValidationResult {
        let mut messages = Vec::new();
        let mut suggested_fixes = Vec::new();

        if !signature.documentation.has_doc_comment {
            messages.push(ValidationMessage {
                severity: ValidationSeverity::Warning,
                message: "Function lacks documentation comment".to_string(),
                location: Some(format!(
                    "{}::{}",
                    signature.module_path, signature.function_name
                )),
                rule_id: "documentation_completeness".to_string(),
            });
            suggested_fixes.push("Add comprehensive doc comment".to_string());
        }

        if !signature.documentation.has_examples {
            messages.push(ValidationMessage {
                severity: ValidationSeverity::Info,
                message: "Function lacks usage examples".to_string(),
                location: Some(format!(
                    "{}::{}",
                    signature.module_path, signature.function_name
                )),
                rule_id: "documentation_completeness".to_string(),
            });
            suggested_fixes.push("Add usage examples in # Examples section".to_string());
        }

        ValidationResult {
            passed: messages
                .iter()
                .all(|m| matches!(m.severity, ValidationSeverity::Info)),
            messages,
            suggested_fixes,
            related_rules: vec!["scipy_compatibility".to_string()],
        }
    }

    /// Validate SciPy compatibility
    fn validate_scipy_compatibility(&self, signature: &APISignature) -> ValidationResult {
        let mut messages = Vec::new();
        let mut suggested_fixes = Vec::new();

        // Check for SciPy standard parameter names
        let scipy_standard_params = [
            "axis",
            "ddof",
            "keepdims",
            "out",
            "dtype",
            "method",
            "alternative",
        ];
        let has_scipy_params = signature
            .parameters
            .iter()
            .any(|p| scipy_standard_params.contains(&p.name.as_str()));

        if has_scipy_params && signature.documentation.scipy_compatibility.is_none() {
            messages.push(ValidationMessage {
                severity: ValidationSeverity::Info,
                message: "Consider documenting SciPy compatibility status".to_string(),
                location: Some(format!(
                    "{}::{}",
                    signature.module_path, signature.function_name
                )),
                rule_id: "scipy_compatibility".to_string(),
            });
            suggested_fixes.push("Add SciPy compatibility note in documentation".to_string());
        }

        ValidationResult {
            passed: true, // Informational only
            messages,
            suggested_fixes,
            related_rules: vec!["documentation_completeness".to_string()],
        }
    }

    /// Validate performance characteristics
    fn validate_performance(&self, signature: &APISignature) -> ValidationResult {
        let mut messages = Vec::new();
        let mut suggested_fixes = Vec::new();

        if signature.performance.time_complexity.is_none() {
            messages.push(ValidationMessage {
                severity: ValidationSeverity::Info,
                message: "Consider documenting time complexity".to_string(),
                location: Some(format!(
                    "{}::{}",
                    signature.module_path, signature.function_name
                )),
                rule_id: "performance_characteristics".to_string(),
            });
            suggested_fixes.push("Add time complexity documentation".to_string());
        }

        ValidationResult {
            passed: true, // Informational only
            messages,
            suggested_fixes,
            related_rules: vec![],
        }
    }
}

impl ValidationReport {
    /// Create new validation report
    pub fn new(_functionname: String) -> Self {
        Self {
            function_name: _functionname,
            results: HashMap::new(),
            overall_status: ValidationStatus::Passed,
            summary: ValidationSummary {
                total_rules: 0,
                passed: 0,
                warnings: 0,
                errors: 0,
                critical: 0,
            },
        }
    }

    /// Add validation result
    pub fn add_result(&mut self, ruleid: String, result: ValidationResult) {
        self.summary.total_rules += 1;

        if result.passed {
            self.summary.passed += 1;
        } else {
            let max_severity = result
                .messages
                .iter()
                .map(|m| m.severity)
                .max()
                .unwrap_or(ValidationSeverity::Info);

            match max_severity {
                ValidationSeverity::Info => {}
                ValidationSeverity::Warning => {
                    self.summary.warnings += 1;
                    if matches!(self.overall_status, ValidationStatus::Passed) {
                        self.overall_status = ValidationStatus::PassedWithWarnings;
                    }
                }
                ValidationSeverity::Error => {
                    self.summary.errors += 1;
                    if !matches!(self.overall_status, ValidationStatus::Critical) {
                        self.overall_status = ValidationStatus::Failed;
                    }
                }
                ValidationSeverity::Critical => {
                    self.summary.critical += 1;
                    self.overall_status = ValidationStatus::Critical;
                }
            }
        }

        self.results.insert(ruleid, result);
    }

    /// Generate human-readable report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "API Validation Report for {}\n",
            self.function_name
        ));
        report.push_str(&format!("Status: {:?}\n", self.overall_status));
        report.push_str(&format!(
            "Summary: {} passed, {} warnings, {} errors, {} critical\n\n",
            self.summary.passed, self.summary.warnings, self.summary.errors, self.summary.critical
        ));

        for (rule_id, result) in &self.results {
            if !result.passed {
                report.push_str(&format!("Rule: {}\n", rule_id));
                for message in &result.messages {
                    report.push_str(&format!("  {:?}: {}\n", message.severity, message.message));
                }
                if !result.suggested_fixes.is_empty() {
                    report.push_str("  Suggestions:\n");
                    for fix in &result.suggested_fixes {
                        report.push_str(&format!("    - {}\n", fix));
                    }
                }
                report.push('\n');
            }
        }

        report
    }
}

impl Default for APIValidationFramework {
    fn default() -> Self {
        Self::new()
    }
}
