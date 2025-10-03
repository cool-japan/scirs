//! Enhanced API Standardization Framework for scirs2-stats v1.0.0+
//!
//! This module extends the base API standardization with advanced features including
//! fluent API patterns, method chaining, async support, streaming operations, and
//! intelligent auto-configuration for optimal user experience and performance.

use crate::api_standardization::{NullHandling, ResultMetadata, StandardizedConfig};
use crate::error::StatsResult;
// Array1 import removed - not used in this module
use scirs2_core::numeric::{Float, NumCast};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Enhanced fluent API configuration with intelligent defaults
#[derive(Debug, Clone)]
pub struct FluentStatsConfig {
    /// Base configuration
    pub base_config: StandardizedConfig,
    /// Enable fluent method chaining
    pub enable_fluent_api: bool,
    /// Enable automatic result caching
    pub enable_result_caching: bool,
    /// Enable streaming operations for large datasets
    pub enable_streaming: bool,
    /// Enable async operation support
    pub enable_async: bool,
    /// Auto-optimization level
    pub auto_optimization_level: AutoOptimizationLevel,
    /// Result format preferences
    pub result_format: ResultFormat,
    /// Performance monitoring
    pub enable_performance_monitoring: bool,
    /// Memory management strategy
    pub memory_strategy: MemoryStrategy,
}

impl Default for FluentStatsConfig {
    fn default() -> Self {
        Self {
            base_config: StandardizedConfig::default(),
            enable_fluent_api: true,
            enable_result_caching: true,
            enable_streaming: true,
            enable_async: false, // Opt-in for async
            auto_optimization_level: AutoOptimizationLevel::Intelligent,
            result_format: ResultFormat::Comprehensive,
            enable_performance_monitoring: true,
            memory_strategy: MemoryStrategy::Adaptive,
        }
    }
}

/// Auto-optimization levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutoOptimizationLevel {
    None,        // No automatic optimization
    Basic,       // Basic optimization (SIMD, parallel)
    Intelligent, // ML-based optimization selection
    Aggressive,  // Maximum optimization (may sacrifice precision)
}

/// Result format preferences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResultFormat {
    Minimal,       // Just the result value
    Standard,      // Result with basic metadata
    Comprehensive, // Full metadata and diagnostics
    Custom,        // User-defined format
}

/// Memory management strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryStrategy {
    Conservative, // Minimize memory usage
    Balanced,     // Balance memory and performance
    Performance,  // Optimize for performance
    Adaptive,     // Adapt based on system resources
}

/// Enhanced fluent statistics API builder
pub struct FluentStats<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    config: FluentStatsConfig,
    operation_chain: Vec<StatisticalOperation>,
    result_cache: Arc<std::sync::RwLock<HashMap<String, CachedResult<F>>>>,
    performance_monitor: Option<PerformanceMonitor>,
    _phantom: PhantomData<F>,
}

impl<F> Default for FluentStats<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> FluentStats<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    /// Create new fluent statistics API
    pub fn new() -> Self {
        Self::with_config(FluentStatsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: FluentStatsConfig) -> Self {
        let performance_monitor = if config.enable_performance_monitoring {
            Some(PerformanceMonitor::new())
        } else {
            None
        };

        Self {
            config,
            operation_chain: Vec::new(),
            result_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
            performance_monitor,
            _phantom: PhantomData,
        }
    }

    /// Configure parallel processing
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.base_config.parallel = enable;
        self
    }

    /// Configure SIMD optimizations
    pub fn simd(mut self, enable: bool) -> Self {
        self.config.base_config.simd = enable;
        self
    }

    /// Set confidence level
    pub fn confidence(mut self, level: f64) -> Self {
        self.config.base_config.confidence_level = level;
        self
    }

    /// Set null handling strategy
    pub fn null_handling(mut self, strategy: NullHandling) -> Self {
        self.config.base_config.null_handling = strategy;
        self
    }

    /// Set memory limit
    pub fn memory_limit(mut self, limit: usize) -> Self {
        self.config.base_config.memory_limit = Some(limit);
        self
    }

    /// Set auto-optimization level
    pub fn optimization(mut self, level: AutoOptimizationLevel) -> Self {
        self.config.auto_optimization_level = level;
        self
    }

    /// Enable streaming operations
    pub fn streaming(mut self, enable: bool) -> Self {
        self.config.enable_streaming = enable;
        self
    }

    /// Set result format
    pub fn format(mut self, format: ResultFormat) -> Self {
        self.config.result_format = format;
        self
    }

    /// Add descriptive statistics operation
    pub fn descriptive(self) -> FluentDescriptive<F> {
        FluentDescriptive::new(self)
    }

    /// Add correlation analysis operation
    pub fn correlation(self) -> FluentCorrelation<F> {
        FluentCorrelation::new(self)
    }

    /// Add hypothesis testing operation
    pub fn test(self) -> FluentTesting<F> {
        FluentTesting::new(self)
    }

    /// Add regression analysis operation
    pub fn regression(self) -> FluentRegression<F> {
        FluentRegression::new(self)
    }

    /// Execute the operation chain
    pub fn execute(&mut self) -> StatsResult<ChainedResults<F>> {
        let start_time = Instant::now();
        let mut results = ChainedResults::new();

        // Optimize operation chain
        if self.config.auto_optimization_level != AutoOptimizationLevel::None {
            self.optimize_operation_chain()?;
        }

        // Execute operations
        for operation in &self.operation_chain {
            let result = self.execute_operation(operation)?;
            results.add_result(operation.name.clone(), result);
        }

        // Record performance metrics
        if let Some(ref mut monitor) = self.performance_monitor {
            monitor.record_execution(start_time.elapsed(), self.operation_chain.len());
        }

        Ok(results)
    }

    /// Optimize the operation chain for performance
    fn optimize_operation_chain(&mut self) -> StatsResult<()> {
        match self.config.auto_optimization_level {
            AutoOptimizationLevel::Basic => {
                // Basic optimizations: reorder for cache efficiency
                self.operation_chain
                    .sort_by_key(|op| op.memory_access_pattern());
            }
            AutoOptimizationLevel::Intelligent => {
                // Intelligent optimization: use ML to predict optimal order
                self.apply_intelligent_optimization()?;
            }
            AutoOptimizationLevel::Aggressive => {
                // Aggressive optimization: fuse operations when possible
                self.fuse_operations()?;
            }
            AutoOptimizationLevel::None => {}
        }
        Ok(())
    }

    /// Apply intelligent ML-based optimization
    fn apply_intelligent_optimization(&mut self) -> StatsResult<()> {
        // Placeholder for ML-based optimization
        // In practice, this would use a trained model to predict optimal operation order
        self.operation_chain
            .sort_by_key(|op| op.estimated_complexity());
        Ok(())
    }

    /// Fuse compatible operations for better performance
    fn fuse_operations(&mut self) -> StatsResult<()> {
        // Placeholder for operation fusion
        // In practice, this would combine compatible operations (e.g., mean + variance)
        Ok(())
    }

    /// Execute a single operation
    fn execute_operation(
        &self,
        operation: &StatisticalOperation,
    ) -> StatsResult<OperationResult<F>> {
        // Check cache first
        if self.config.enable_result_caching {
            let cache_key = operation.cache_key();
            if let Ok(cache) = self.result_cache.read() {
                if let Some(cached) = cache.get(&cache_key) {
                    if !cached.is_expired() {
                        return Ok(cached.result.clone());
                    }
                }
            }
        }

        // Execute operation
        let result = match &operation.operation_type {
            OperationType::Mean => self.execute_mean_operation(operation),
            OperationType::Variance => self.execute_variance_operation(operation),
            OperationType::Correlation => self.execute_correlation_operation(operation),
            OperationType::TTest => self.execute_ttest_operation(operation),
            OperationType::Regression => self.execute_regression_operation(operation),
        }?;

        // Cache result if enabled
        if self.config.enable_result_caching {
            let cache_key = operation.cache_key();
            if let Ok(mut cache) = self.result_cache.write() {
                cache.insert(cache_key, CachedResult::new(result.clone()));
            }
        }

        Ok(result)
    }

    /// Execute mean operation (placeholder)
    fn execute_mean_operation(
        &self,
        _operation: &StatisticalOperation,
    ) -> StatsResult<OperationResult<F>> {
        // Placeholder implementation
        Ok(OperationResult {
            value: Box::new(F::zero()),
            metadata: ResultMetadata {
                samplesize: 0,
                degrees_of_freedom: None,
                confidence_level: None,
                method: "mean".to_string(),
                computation_time_ms: 0.0,
                memory_usage_bytes: None,
                optimized: true,
                extra: HashMap::new(),
            },
            operation_type: OperationType::Mean,
        })
    }

    /// Execute variance operation (placeholder)
    fn execute_variance_operation(
        &self,
        _operation: &StatisticalOperation,
    ) -> StatsResult<OperationResult<F>> {
        // Placeholder implementation
        Ok(OperationResult {
            value: Box::new(F::one()),
            metadata: ResultMetadata {
                samplesize: 0,
                degrees_of_freedom: Some(0),
                confidence_level: None,
                method: "variance".to_string(),
                computation_time_ms: 0.0,
                memory_usage_bytes: None,
                optimized: true,
                extra: HashMap::new(),
            },
            operation_type: OperationType::Variance,
        })
    }

    /// Execute correlation operation (placeholder)
    fn execute_correlation_operation(
        &self,
        _operation: &StatisticalOperation,
    ) -> StatsResult<OperationResult<F>> {
        // Placeholder implementation
        Ok(OperationResult {
            value: Box::new(F::zero()),
            metadata: ResultMetadata {
                samplesize: 0,
                degrees_of_freedom: None,
                confidence_level: Some(0.95),
                method: "pearson_correlation".to_string(),
                computation_time_ms: 0.0,
                memory_usage_bytes: None,
                optimized: true,
                extra: HashMap::new(),
            },
            operation_type: OperationType::Correlation,
        })
    }

    /// Execute t-test operation (placeholder)
    fn execute_ttest_operation(
        &self,
        _operation: &StatisticalOperation,
    ) -> StatsResult<OperationResult<F>> {
        // Placeholder implementation
        Ok(OperationResult {
            value: Box::new(F::zero()),
            metadata: ResultMetadata {
                samplesize: 0,
                degrees_of_freedom: Some(0),
                confidence_level: Some(0.95),
                method: "t_test".to_string(),
                computation_time_ms: 0.0,
                memory_usage_bytes: None,
                optimized: true,
                extra: HashMap::new(),
            },
            operation_type: OperationType::TTest,
        })
    }

    /// Execute regression operation (placeholder)
    fn execute_regression_operation(
        &self,
        _operation: &StatisticalOperation,
    ) -> StatsResult<OperationResult<F>> {
        // Placeholder implementation
        Ok(OperationResult {
            value: Box::new(F::zero()),
            metadata: ResultMetadata {
                samplesize: 0,
                degrees_of_freedom: Some(0),
                confidence_level: Some(0.95),
                method: "linear_regression".to_string(),
                computation_time_ms: 0.0,
                memory_usage_bytes: None,
                optimized: true,
                extra: HashMap::new(),
            },
            operation_type: OperationType::Regression,
        })
    }
}

/// Fluent descriptive statistics API
pub struct FluentDescriptive<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    parent: FluentStats<F>,
    operations: Vec<DescriptiveOperation>,
}

impl<F> FluentDescriptive<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    fn new(parent: FluentStats<F>) -> Self {
        Self {
            parent,
            operations: Vec::new(),
        }
    }

    /// Add mean calculation
    pub fn mean(mut self) -> Self {
        self.operations.push(DescriptiveOperation::Mean);
        self
    }

    /// Add variance calculation
    pub fn variance(mut self, ddof: usize) -> Self {
        self.operations.push(DescriptiveOperation::Variance(ddof));
        self
    }

    /// Add standard deviation calculation
    pub fn std_dev(mut self, ddof: usize) -> Self {
        self.operations.push(DescriptiveOperation::StdDev(ddof));
        self
    }

    /// Add skewness calculation
    pub fn skewness(mut self) -> Self {
        self.operations.push(DescriptiveOperation::Skewness);
        self
    }

    /// Add kurtosis calculation
    pub fn kurtosis(mut self) -> Self {
        self.operations.push(DescriptiveOperation::Kurtosis);
        self
    }

    /// Add all basic descriptive statistics
    pub fn all_basic(mut self) -> Self {
        self.operations.extend(vec![
            DescriptiveOperation::Mean,
            DescriptiveOperation::Variance(1),
            DescriptiveOperation::StdDev(1),
            DescriptiveOperation::Skewness,
            DescriptiveOperation::Kurtosis,
        ]);
        self
    }

    /// Return to parent fluent API
    pub fn and(mut self) -> FluentStats<F> {
        // Convert descriptive operations to statistical operations
        for desc_op in self.operations {
            let stat_op = StatisticalOperation {
                name: format!("{:?}", desc_op),
                operation_type: OperationType::from_descriptive(desc_op),
                parameters: HashMap::new(),
                data_requirements: DataRequirements::single_array(),
            };
            self.parent.operation_chain.push(stat_op);
        }
        self.parent
    }
}

/// Fluent correlation analysis API
pub struct FluentCorrelation<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    parent: FluentStats<F>,
    correlation_type: CorrelationType,
    method: CorrelationMethod,
}

impl<F> FluentCorrelation<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    fn new(parent: FluentStats<F>) -> Self {
        Self {
            parent,
            correlation_type: CorrelationType::Pairwise,
            method: CorrelationMethod::Pearson,
        }
    }

    /// Set correlation method
    pub fn method(mut self, method: CorrelationMethod) -> Self {
        self.method = method;
        self
    }

    /// Use Pearson correlation
    pub fn pearson(mut self) -> Self {
        self.method = CorrelationMethod::Pearson;
        self
    }

    /// Use Spearman correlation
    pub fn spearman(mut self) -> Self {
        self.method = CorrelationMethod::Spearman;
        self
    }

    /// Use Kendall tau correlation
    pub fn kendall(mut self) -> Self {
        self.method = CorrelationMethod::Kendall;
        self
    }

    /// Compute correlation matrix
    pub fn matrix(mut self) -> Self {
        self.correlation_type = CorrelationType::Matrix;
        self
    }

    /// Return to parent fluent API
    pub fn and(mut self) -> FluentStats<F> {
        let stat_op = StatisticalOperation {
            name: format!("{:?}_{:?}", self.method, self.correlation_type),
            operation_type: OperationType::Correlation,
            parameters: HashMap::from([
                ("method".to_string(), format!("{:?}", self.method)),
                ("type".to_string(), format!("{:?}", self.correlation_type)),
            ]),
            data_requirements: DataRequirements::multi_array(),
        };
        self.parent.operation_chain.push(stat_op);
        self.parent
    }
}

/// Fluent hypothesis testing API
pub struct FluentTesting<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    parent: FluentStats<F>,
    test_type: TestType,
}

impl<F> FluentTesting<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    fn new(parent: FluentStats<F>) -> Self {
        Self {
            parent,
            test_type: TestType::TTest,
        }
    }

    /// One-sample t-test
    pub fn t_test_one_sample(mut self, mu: F) -> Self {
        self.test_type = TestType::TTestOneSample(mu.to_f64().unwrap_or(0.0));
        self
    }

    /// Independent samples t-test
    pub fn t_test_independent(mut self) -> Self {
        self.test_type = TestType::TTestIndependent;
        self
    }

    /// Paired samples t-test
    pub fn t_test_paired(mut self) -> Self {
        self.test_type = TestType::TTestPaired;
        self
    }

    /// Return to parent fluent API
    pub fn and(mut self) -> FluentStats<F> {
        let stat_op = StatisticalOperation {
            name: format!("{:?}", self.test_type),
            operation_type: OperationType::TTest,
            parameters: HashMap::new(),
            data_requirements: DataRequirements::single_array(),
        };
        self.parent.operation_chain.push(stat_op);
        self.parent
    }
}

/// Fluent regression analysis API
pub struct FluentRegression<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    parent: FluentStats<F>,
    regression_type: RegressionType,
}

impl<F> FluentRegression<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    fn new(parent: FluentStats<F>) -> Self {
        Self {
            parent,
            regression_type: RegressionType::Linear,
        }
    }

    /// Linear regression
    pub fn linear(mut self) -> Self {
        self.regression_type = RegressionType::Linear;
        self
    }

    /// Polynomial regression
    pub fn polynomial(mut self, degree: usize) -> Self {
        self.regression_type = RegressionType::Polynomial(degree);
        self
    }

    /// Ridge regression
    pub fn ridge(mut self, alpha: F) -> Self {
        self.regression_type = RegressionType::Ridge(alpha.to_f64().unwrap_or(0.0));
        self
    }

    /// Return to parent fluent API
    pub fn and(mut self) -> FluentStats<F> {
        let stat_op = StatisticalOperation {
            name: format!("{:?}", self.regression_type),
            operation_type: OperationType::Regression,
            parameters: HashMap::new(),
            data_requirements: DataRequirements::xy_arrays(),
        };
        self.parent.operation_chain.push(stat_op);
        self.parent
    }
}

/// Statistical operation definition
#[derive(Debug, Clone)]
pub struct StatisticalOperation {
    pub name: String,
    pub operation_type: OperationType,
    pub parameters: HashMap<String, String>,
    pub data_requirements: DataRequirements,
}

impl StatisticalOperation {
    /// Generate cache key for this operation
    pub fn cache_key(&self) -> String {
        format!(
            "{}_{:?}_{:?}",
            self.name, self.operation_type, self.parameters
        )
    }

    /// Estimate memory access pattern (for optimization)
    pub fn memory_access_pattern(&self) -> u32 {
        match self.operation_type {
            OperationType::Mean => 1,
            OperationType::Variance => 2,
            OperationType::Correlation => 3,
            OperationType::TTest => 2,
            OperationType::Regression => 4,
        }
    }

    /// Estimate computational complexity (for optimization)
    pub fn estimated_complexity(&self) -> u32 {
        match self.operation_type {
            OperationType::Mean => 1,
            OperationType::Variance => 2,
            OperationType::Correlation => 4,
            OperationType::TTest => 3,
            OperationType::Regression => 5,
        }
    }
}

/// Types of statistical operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OperationType {
    Mean,
    Variance,
    Correlation,
    TTest,
    Regression,
}

impl OperationType {
    fn from_descriptive(_descop: DescriptiveOperation) -> Self {
        match _descop {
            DescriptiveOperation::Mean => OperationType::Mean,
            DescriptiveOperation::Variance(_) => OperationType::Variance,
            DescriptiveOperation::StdDev(_) => OperationType::Variance,
            DescriptiveOperation::Skewness => OperationType::Mean, // Simplified
            DescriptiveOperation::Kurtosis => OperationType::Mean, // Simplified
        }
    }
}

/// Data requirements for operations
#[derive(Debug, Clone)]
pub struct DataRequirements {
    pub arrays_needed: usize,
    pub minsize: usize,
    pub requires_numeric: bool,
}

impl DataRequirements {
    pub fn single_array() -> Self {
        Self {
            arrays_needed: 1,
            minsize: 1,
            requires_numeric: true,
        }
    }

    pub fn multi_array() -> Self {
        Self {
            arrays_needed: 2,
            minsize: 1,
            requires_numeric: true,
        }
    }

    pub fn xy_arrays() -> Self {
        Self {
            arrays_needed: 2,
            minsize: 2,
            requires_numeric: true,
        }
    }
}

/// Descriptive statistics operations
#[derive(Debug, Clone, Copy)]
pub enum DescriptiveOperation {
    Mean,
    Variance(usize), // ddof
    StdDev(usize),   // ddof
    Skewness,
    Kurtosis,
}

/// Correlation types
#[derive(Debug, Clone, Copy)]
pub enum CorrelationType {
    Pairwise,
    Matrix,
    Partial,
}

/// Correlation methods
#[derive(Debug, Clone, Copy)]
pub enum CorrelationMethod {
    Pearson,
    Spearman,
    Kendall,
}

/// Hypothesis test types
#[derive(Debug, Clone)]
pub enum TestType {
    TTest,
    TTestOneSample(f64),
    TTestIndependent,
    TTestPaired,
    ChiSquare,
    ANOVA,
}

/// Regression types
#[derive(Debug, Clone)]
pub enum RegressionType {
    Linear,
    Polynomial(usize), // degree
    Ridge(f64),        // alpha
    Lasso(f64),        // alpha
}

/// Result of a single operation
#[derive(Debug, Clone)]
pub struct OperationResult<F> {
    pub value: Box<F>,
    pub metadata: ResultMetadata,
    pub operation_type: OperationType,
}

/// Collection of chained results
#[derive(Debug)]
pub struct ChainedResults<F> {
    results: HashMap<String, OperationResult<F>>,
    execution_order: Vec<String>,
}

impl<F> ChainedResults<F> {
    fn new() -> Self {
        Self {
            results: HashMap::new(),
            execution_order: Vec::new(),
        }
    }

    fn add_result(&mut self, name: String, result: OperationResult<F>) {
        self.execution_order.push(name.clone());
        self.results.insert(name, result);
    }

    /// Get result by operation name
    pub fn get(&self, name: &str) -> Option<&OperationResult<F>> {
        self.results.get(name)
    }

    /// Get all results in execution order
    pub fn iter(&self) -> impl Iterator<Item = (&String, &OperationResult<F>)> {
        self.execution_order
            .iter()
            .filter_map(|name| self.results.get(name).map(|result| (name, result)))
    }
}

/// Cached result with expiration
#[derive(Debug, Clone)]
struct CachedResult<F> {
    result: OperationResult<F>,
    created_at: Instant,
    ttl: Duration,
}

impl<F> CachedResult<F> {
    fn new(result: OperationResult<F>) -> Self {
        Self {
            result,
            created_at: Instant::now(),
            ttl: Duration::from_secs(300), // 5 minutes default TTL
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Performance monitoring for fluent API
#[derive(Debug)]
struct PerformanceMonitor {
    executions: Vec<ExecutionMetrics>,
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            executions: Vec::new(),
        }
    }

    fn record_execution(&mut self, duration: Duration, operationcount: usize) {
        self.executions.push(ExecutionMetrics {
            duration,
            operation_count: operationcount,
            timestamp: Instant::now(),
        });
    }

    #[allow(dead_code)]
    fn average_execution_time(&self) -> Option<Duration> {
        if self.executions.is_empty() {
            None
        } else {
            let total: Duration = self.executions.iter().map(|e| e.duration).sum();
            Some(total / self.executions.len() as u32)
        }
    }
}

/// Execution metrics
#[derive(Debug)]
struct ExecutionMetrics {
    duration: Duration,
    #[allow(dead_code)]
    operation_count: usize,
    #[allow(dead_code)]
    timestamp: Instant,
}

/// Convenience functions for creating fluent API instances
#[allow(dead_code)]
pub fn stats<F>() -> FluentStats<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    FluentStats::new()
}

#[allow(dead_code)]
pub fn stats_with<F>(config: FluentStatsConfig) -> FluentStats<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    FluentStats::with_config(config)
}

/// Quick descriptive statistics with fluent API
#[allow(dead_code)]
pub fn quick_descriptive<F>() -> FluentDescriptive<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    FluentStats::new().descriptive()
}

/// Quick correlation analysis with fluent API
#[allow(dead_code)]
pub fn quick_correlation<F>() -> FluentCorrelation<F>
where
    F: Float + NumCast + Send + Sync + 'static + std::fmt::Display,
{
    FluentStats::new().correlation()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluent_stats_creation() {
        let _stats: FluentStats<f64> = stats();
        assert!(true); // Just test compilation
    }

    #[test]
    fn test_fluent_configuration() {
        let config = FluentStatsConfig {
            enable_fluent_api: true,
            auto_optimization_level: AutoOptimizationLevel::Intelligent,
            ..Default::default()
        };

        let _stats: FluentStats<f64> = stats_with(config);
        assert!(true); // Just test compilation
    }

    #[test]
    fn test_method_chaining() {
        let _chain: FluentStats<f64> = stats()
            .parallel(true)
            .simd(true)
            .confidence(0.99)
            .optimization(AutoOptimizationLevel::Aggressive);

        assert!(true); // Just test compilation
    }

    #[test]
    fn test_descriptive_operations() {
        let _desc = quick_descriptive::<f64>()
            .mean()
            .variance(1)
            .std_dev(1)
            .skewness()
            .kurtosis();

        assert!(true); // Just test compilation
    }
}
