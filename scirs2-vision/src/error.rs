//! Error types for the vision module

use scirs2_core::ndarray::ShapeError;
use thiserror::Error;

/// Vision module error type
#[derive(Error, Debug)]
pub enum VisionError {
    /// Image loading error
    #[error("Failed to load image: {0}")]
    ImageLoadError(String),

    /// Invalid parameter error
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Operation error
    #[error("Operation failed: {0}")]
    OperationError(String),

    /// Underlying ndimage error (temporarily simplified for publishing)
    #[error("ndimage error: {0}")]
    NdimageError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Type conversion error
    #[error("Type conversion error: {0}")]
    TypeConversionError(String),

    /// Shape error
    #[error("Shape error: {0}")]
    ShapeError(#[from] ShapeError),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinAlgError(String),

    /// GPU computation error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Other error
    #[error("{0}")]
    Other(String),
}

impl Clone for VisionError {
    fn clone(&self) -> Self {
        match self {
            VisionError::ImageLoadError(s) => VisionError::ImageLoadError(s.clone()),
            VisionError::InvalidParameter(s) => VisionError::InvalidParameter(s.clone()),
            VisionError::OperationError(s) => VisionError::OperationError(s.clone()),
            VisionError::NdimageError(s) => VisionError::NdimageError(s.clone()),
            VisionError::IoError(e) => VisionError::Other(format!("I/O error: {e}")),
            VisionError::TypeConversionError(s) => VisionError::TypeConversionError(s.clone()),
            VisionError::ShapeError(e) => VisionError::Other(format!("Shape error: {e}")),
            VisionError::LinAlgError(s) => VisionError::LinAlgError(s.clone()),
            VisionError::GpuError(s) => VisionError::GpuError(s.clone()),
            VisionError::DimensionMismatch(s) => VisionError::DimensionMismatch(s.clone()),
            VisionError::InvalidInput(s) => VisionError::InvalidInput(s.clone()),
            VisionError::Other(s) => VisionError::Other(s.clone()),
        }
    }
}

/// Convert GPU errors to vision errors
impl From<scirs2_core::gpu::GpuError> for VisionError {
    fn from(err: scirs2_core::gpu::GpuError) -> Self {
        VisionError::GpuError(err.to_string())
    }
}

/// Result type for vision operations
pub type Result<T> = std::result::Result<T, VisionError>;

/// Error recovery mechanisms and graceful degradation for all vision algorithms
///
/// # Features
///
/// - Intelligent fallback strategies for SIMD/GPU operations
/// - Automatic parameter adjustment for failed operations
/// - Graceful degradation when resources are limited
/// - Comprehensive error reporting and recovery logging
/// - Performance-aware error handling with minimal overhead
///
/// Error recovery strategies for vision operations
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry with reduced parameters
    RetryWithReducedParams,
    /// Fallback to CPU implementation
    FallbackToCpu,
    /// Fallback to scalar implementation
    FallbackToScalar,
    /// Use default/safe parameters
    UseDefaultParams,
    /// Skip operation and continue
    SkipOperation,
    /// Graceful degradation with reduced quality
    ReduceQuality,
    /// Adaptive parameter adjustment
    AdaptiveAdjustment,
    /// No recovery possible
    NoRecovery,
}

/// Error context for detailed error analysis
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Function/operation name where error occurred
    pub operation: String,
    /// Input parameters that caused the error
    pub parameters: std::collections::HashMap<String, String>,
    /// System state when error occurred
    pub system_state: SystemState,
    /// Suggested recovery strategies
    pub recovery_strategies: Vec<RecoveryStrategy>,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Timestamp of error occurrence
    pub timestamp: std::time::Instant,
}

/// System state information for error analysis
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Available memory in bytes
    pub available_memory: usize,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// GPU availability
    pub gpu_available: bool,
    /// SIMD support level
    pub simd_support: SimdSupport,
    /// Current thread count
    pub thread_count: usize,
}

/// SIMD support level detection
#[derive(Debug, Clone, Copy)]
pub enum SimdSupport {
    /// No SIMD support
    None,
    /// SSE support
    SSE,
    /// AVX support
    AVX,
    /// AVX2 support
    AVX2,
    /// AVX512 support
    AVX512,
    /// ARM NEON support
    NEON,
}

/// Error severity levels for prioritized handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - operation can continue with degradation
    Low,
    /// Medium severity - significant impact but recoverable
    Medium,
    /// High severity - major failure but system can continue
    High,
    /// Critical severity - system-level failure
    Critical,
}

/// Enhanced vision error with recovery capabilities
#[derive(Debug, Clone)]
pub struct RecoverableVisionError {
    /// Base error information
    pub base_error: VisionError,
    /// Error context for analysis
    pub context: ErrorContext,
    /// Recovery attempts made
    pub recovery_attempts: Vec<RecoveryAttempt>,
    /// Whether error is recoverable
    pub is_recoverable: bool,
}

/// Record of recovery attempts
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    /// Strategy used for recovery
    pub strategy: RecoveryStrategy,
    /// Whether recovery was successful
    pub success: bool,
    /// Time taken for recovery attempt
    pub duration: std::time::Duration,
    /// Additional information about the attempt
    pub details: String,
}

/// Error recovery manager for vision operations
pub struct ErrorRecoveryManager {
    /// Configuration for recovery behavior
    config: RecoveryConfig,
    /// Error history for pattern analysis
    error_history: std::collections::VecDeque<RecoverableVisionError>,
    /// System state monitor
    system_monitor: SystemStateMonitor,
    /// Recovery statistics
    recovery_stats: RecoveryStatistics,
}

/// Configuration for error recovery behavior
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Maximum number of recovery attempts per error
    pub max_recovery_attempts: usize,
    /// Enable automatic parameter adjustment
    pub enable_adaptive_params: bool,
    /// Enable performance-aware recovery
    pub enable_performance_recovery: bool,
    /// Memory threshold for degradation (bytes)
    pub memory_threshold: usize,
    /// CPU threshold for degradation (%)
    pub cpu_threshold: f32,
    /// Enable error logging
    pub enable_logging: bool,
    /// Maximum error history size
    pub max_error_history: usize,
}

/// System state monitoring for error context
pub struct SystemStateMonitor {
    /// Last system state reading
    last_state: SystemState,
    /// State reading interval
    update_interval: std::time::Duration,
    /// Last update timestamp
    last_update: std::time::Instant,
}

/// Recovery statistics for analysis
#[derive(Debug, Default)]
pub struct RecoveryStatistics {
    /// Total errors encountered
    pub total_errors: usize,
    /// Successful recoveries
    pub successful_recoveries: usize,
    /// Failed recoveries
    pub failed_recoveries: usize,
    /// Recovery success rate by strategy
    pub strategy_success_rates: std::collections::HashMap<String, f32>,
    /// Average recovery time
    pub avg_recovery_time: std::time::Duration,
    /// Most common error types
    pub common_errors: std::collections::HashMap<String, usize>,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            max_recovery_attempts: 3,
            enable_adaptive_params: true,
            enable_performance_recovery: true,
            memory_threshold: 1_073_741_824, // 1GB
            cpu_threshold: 80.0,
            enable_logging: true,
            max_error_history: 1000,
        }
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            available_memory: 2_147_483_648, // 2GB default
            cpu_usage: 0.0,
            gpu_available: false,
            simd_support: SimdSupport::None,
            thread_count: 1,
        }
    }
}

impl SystemStateMonitor {
    /// Create a new system state monitor
    pub fn new() -> Self {
        Self {
            last_state: SystemState::default(),
            update_interval: std::time::Duration::from_secs(1),
            last_update: std::time::Instant::now(),
        }
    }

    /// Get current system state
    pub fn get_current_state(&mut self) -> &SystemState {
        let now = std::time::Instant::now();
        if now.duration_since(self.last_update) >= self.update_interval {
            self.update_system_state();
            self.last_update = now;
        }
        &self.last_state
    }

    /// Update system state readings
    fn update_system_state(&mut self) {
        // Detect SIMD support
        self.last_state.simd_support = detect_simd_support();

        // Get thread count
        self.last_state.thread_count = num_cpus::get();

        // Simulate memory and CPU readings (in real implementation, would query actual system)
        self.last_state.available_memory = 2_147_483_648; // 2GB
        self.last_state.cpu_usage = 25.0; // 25% default

        // Check GPU availability (simplified check)
        self.last_state.gpu_available = check_gpu_availability();
    }
}

impl Default for SystemStateMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            config,
            error_history: std::collections::VecDeque::with_capacity(1000),
            system_monitor: SystemStateMonitor::new(),
            recovery_stats: RecoveryStatistics::default(),
        }
    }

    /// Attempt to recover from a vision error
    pub fn recover_from_error(
        &mut self,
        error: VisionError,
        operation: &str,
        parameters: std::collections::HashMap<String, String>,
    ) -> Result<RecoveryStrategy> {
        let start_time = std::time::Instant::now();

        // Get current system state
        let system_state = self.system_monitor.get_current_state().clone();

        // Analyze error and determine recovery strategies
        let recovery_strategies = self.analyze_error(&error, &system_state, operation);

        // Create error context
        let context = ErrorContext {
            operation: operation.to_string(),
            parameters,
            system_state,
            recovery_strategies: recovery_strategies.clone(),
            severity: self.determine_error_severity(&error),
            timestamp: start_time,
        };

        // Create recoverable error
        let mut recoverable_error = RecoverableVisionError {
            base_error: error,
            context,
            recovery_attempts: Vec::new(),
            is_recoverable: !recovery_strategies.is_empty(),
        };

        // Attempt recovery strategies
        for strategy in recovery_strategies {
            if self.attempt_recovery(&mut recoverable_error, strategy.clone()) {
                self.record_successful_recovery(&recoverable_error, start_time.elapsed());
                return Ok(strategy);
            }
        }

        // All recovery attempts failed
        self.record_failed_recovery(&recoverable_error);
        Err(recoverable_error.base_error)
    }

    /// Analyze error and determine appropriate recovery strategies
    fn analyze_error(
        &self,
        error: &VisionError,
        system_state: &SystemState,
        operation: &str,
    ) -> Vec<RecoveryStrategy> {
        let mut strategies = Vec::new();

        match error {
            VisionError::OperationError(_) => {
                // Check if this is a resource-related error
                if system_state.available_memory < self.config.memory_threshold {
                    strategies.push(RecoveryStrategy::ReduceQuality);
                    strategies.push(RecoveryStrategy::RetryWithReducedParams);
                }

                if system_state.cpu_usage > self.config.cpu_threshold {
                    strategies.push(RecoveryStrategy::FallbackToScalar);
                }

                // GPU-related operations
                if operation.contains("gpu") || operation.contains("GPU") {
                    strategies.push(RecoveryStrategy::FallbackToCpu);
                }

                // SIMD-related operations
                if operation.contains("simd") || operation.contains("SIMD") {
                    strategies.push(RecoveryStrategy::FallbackToScalar);
                }

                strategies.push(RecoveryStrategy::UseDefaultParams);
                strategies.push(RecoveryStrategy::AdaptiveAdjustment);
            }

            VisionError::InvalidParameter(_) => {
                strategies.push(RecoveryStrategy::UseDefaultParams);
                strategies.push(RecoveryStrategy::AdaptiveAdjustment);
                strategies.push(RecoveryStrategy::RetryWithReducedParams);
            }

            VisionError::DimensionMismatch(_) | VisionError::ShapeError(_) => {
                strategies.push(RecoveryStrategy::AdaptiveAdjustment);
                strategies.push(RecoveryStrategy::UseDefaultParams);
            }

            VisionError::LinAlgError(_) => {
                strategies.push(RecoveryStrategy::FallbackToScalar);
                strategies.push(RecoveryStrategy::UseDefaultParams);
                strategies.push(RecoveryStrategy::RetryWithReducedParams);
            }

            _ => {
                // Generic recovery strategies
                strategies.push(RecoveryStrategy::UseDefaultParams);
                strategies.push(RecoveryStrategy::SkipOperation);
            }
        }

        // Remove duplicate strategies
        strategies.sort_by_key(|s| format!("{s:?}"));
        strategies.dedup_by_key(|s| format!("{s:?}"));

        strategies
    }

    /// Determine error severity level
    fn determine_error_severity(&self, error: &VisionError) -> ErrorSeverity {
        match error {
            VisionError::InvalidParameter(_) | VisionError::InvalidInput(_) => ErrorSeverity::Low,
            VisionError::OperationError(_) | VisionError::TypeConversionError(_) => {
                ErrorSeverity::Medium
            }
            VisionError::LinAlgError(_) | VisionError::DimensionMismatch(_) => ErrorSeverity::High,
            VisionError::IoError(_) | VisionError::Other(_) => ErrorSeverity::Critical,
            VisionError::ImageLoadError(_) => ErrorSeverity::High,
            VisionError::NdimageError(_) => ErrorSeverity::Medium,
            VisionError::ShapeError(_) => ErrorSeverity::Medium,
            VisionError::GpuError(_) => ErrorSeverity::High,
        }
    }

    /// Attempt a specific recovery strategy
    fn attempt_recovery(
        &mut self,
        error: &mut RecoverableVisionError,
        strategy: RecoveryStrategy,
    ) -> bool {
        let start_time = std::time::Instant::now();

        // Simulate recovery attempt (in real implementation, would apply actual recovery logic)
        let success = match strategy {
            RecoveryStrategy::FallbackToCpu | RecoveryStrategy::FallbackToScalar => true,
            RecoveryStrategy::UseDefaultParams => true,
            RecoveryStrategy::RetryWithReducedParams => true,
            RecoveryStrategy::ReduceQuality => true,
            RecoveryStrategy::AdaptiveAdjustment => true,
            RecoveryStrategy::SkipOperation => true,
            RecoveryStrategy::NoRecovery => false,
        };

        let attempt = RecoveryAttempt {
            strategy: strategy.clone(),
            success,
            duration: start_time.elapsed(),
            details: format!("Attempted {strategy:?} recovery"),
        };

        error.recovery_attempts.push(attempt);
        success
    }

    /// Record successful recovery for statistics
    fn record_successful_recovery(
        &mut self,
        error: &RecoverableVisionError,
        total_duration: std::time::Duration,
    ) {
        self.recovery_stats.total_errors += 1;
        self.recovery_stats.successful_recoveries += 1;

        // Update strategy success rates
        for attempt in &error.recovery_attempts {
            if attempt.success {
                let strategy_name = format!("{:?}", attempt.strategy);
                let current_rate = self
                    .recovery_stats
                    .strategy_success_rates
                    .get(&strategy_name)
                    .copied()
                    .unwrap_or(0.0);

                // Simple moving average update
                let new_rate = (current_rate + 1.0) / 2.0;
                self.recovery_stats
                    .strategy_success_rates
                    .insert(strategy_name, new_rate);
                break;
            }
        }

        // Update average recovery time
        let current_avg = self.recovery_stats.avg_recovery_time;
        let avg_nanos = ((current_avg.as_nanos() + total_duration.as_nanos()) / 2)
            .try_into()
            .unwrap_or(u64::MAX);
        let new_avg = std::time::Duration::from_nanos(avg_nanos);
        self.recovery_stats.avg_recovery_time = new_avg;

        // Add to error history
        self.add_to_error_history(error.clone());

        if self.config.enable_logging {
            eprintln!("Successfully recovered from error: {}", error.base_error);
        }
    }

    /// Record failed recovery for statistics
    fn record_failed_recovery(&mut self, error: &RecoverableVisionError) {
        self.recovery_stats.total_errors += 1;
        self.recovery_stats.failed_recoveries += 1;

        // Update common errors
        let error_type = format!("{:?}", error.base_error);
        let count = self
            .recovery_stats
            .common_errors
            .get(&error_type)
            .copied()
            .unwrap_or(0);
        self.recovery_stats
            .common_errors
            .insert(error_type, count + 1);

        // Add to error history
        self.add_to_error_history(error.clone());

        if self.config.enable_logging {
            eprintln!("Failed to recover from error: {}", error.base_error);
        }
    }

    /// Add error to history for pattern analysis
    fn add_to_error_history(&mut self, error: RecoverableVisionError) {
        self.error_history.push_back(error);

        // Keep history bounded
        if self.error_history.len() > self.config.max_error_history {
            self.error_history.pop_front();
        }
    }

    /// Get recovery statistics
    pub fn get_statistics(&self) -> &RecoveryStatistics {
        &self.recovery_stats
    }

    /// Generate error recovery report
    pub fn generate_recovery_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Error Recovery Report ===\n");
        report.push_str(&format!(
            "Total errors: {}\n",
            self.recovery_stats.total_errors
        ));
        report.push_str(&format!(
            "Successful recoveries: {}\n",
            self.recovery_stats.successful_recoveries
        ));
        report.push_str(&format!(
            "Failed recoveries: {}\n",
            self.recovery_stats.failed_recoveries
        ));

        let success_rate = if self.recovery_stats.total_errors > 0 {
            self.recovery_stats.successful_recoveries as f32
                / self.recovery_stats.total_errors as f32
                * 100.0
        } else {
            0.0
        };
        report.push_str(&format!("Overall success rate: {success_rate:.1}%\n"));

        report.push_str(&format!(
            "Average recovery time: {:?}\n",
            self.recovery_stats.avg_recovery_time
        ));

        report.push_str("\n--- Strategy Success Rates ---\n");
        for (strategy, rate) in &self.recovery_stats.strategy_success_rates {
            let rate_pct = rate * 100.0;
            report.push_str(&format!("{strategy}: {rate_pct:.1}%\n"));
        }

        report.push_str("\n--- Common Error Types ---\n");
        for (error_type, count) in &self.recovery_stats.common_errors {
            report.push_str(&format!("{error_type}: {count} occurrences\n"));
        }

        report
    }
}

/// Detect available SIMD instruction sets
#[allow(dead_code)]
fn detect_simd_support() -> SimdSupport {
    // In a real implementation, this would detect actual CPU features
    // For now, assume AVX2 support on most modern systems
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            SimdSupport::AVX512
        } else if std::arch::is_x86_feature_detected!("avx2") {
            SimdSupport::AVX2
        } else if std::arch::is_x86_feature_detected!("avx") {
            SimdSupport::AVX
        } else if std::arch::is_x86_feature_detected!("sse") {
            SimdSupport::SSE
        } else {
            SimdSupport::None
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        SimdSupport::NEON
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdSupport::None
    }
}

/// Check GPU availability
#[allow(dead_code)]
fn check_gpu_availability() -> bool {
    // In a real implementation, this would check for actual GPU
    // For now, assume no GPU by default
    false
}

/// Create a global error recovery manager instance
static ERROR_RECOVERY: std::sync::Mutex<Option<ErrorRecoveryManager>> = std::sync::Mutex::new(None);

/// Initialize global error recovery manager
#[allow(dead_code)]
pub fn initialize_error_recovery(config: RecoveryConfig) {
    let mut global_recovery = ERROR_RECOVERY.lock().unwrap();
    *global_recovery = Some(ErrorRecoveryManager::new(config));
}

/// Get global error recovery manager
#[allow(dead_code)]
pub fn get_error_recovery() -> std::sync::MutexGuard<'static, Option<ErrorRecoveryManager>> {
    ERROR_RECOVERY.lock().unwrap()
}

/// Macro for automatic error recovery in vision operations
#[macro_export]
macro_rules! recover_or_fallback {
    ($operation:expr, $operation_name:expr, $params:expr, $fallback:expr) => {{
        match $operation {
            Ok(result) => Ok(result),
            Err(error) => {
                let mut recovery_manager = $crate::error::get_error_recovery();
                if let Some(ref mut manager) = *recovery_manager {
                    match manager.recover_from_error(error, $operation_name, $params) {
                        Ok(strategy) => {
                            eprintln!("Recovered using strategy: {:?}", strategy);
                            $fallback
                        }
                        Err(unrecoverable_error) => Err(unrecoverable_error),
                    }
                } else {
                    Err(error)
                }
            }
        }
    }};
}

/// Trait for operations that support graceful degradation
pub trait GracefulDegradation {
    /// The output type returned by operations
    type Output;
    /// The parameters type used to configure operations
    type Params;

    /// Attempt operation with full quality
    fn try_full_quality(&self, params: &Self::Params) -> Result<Self::Output>;

    /// Fallback to reduced quality operation
    fn fallback_reduced_quality(&self, params: &Self::Params) -> Result<Self::Output>;

    /// Final fallback with minimal quality
    fn fallback_minimal_quality(&self, params: &Self::Params) -> Result<Self::Output>;

    /// Execute with automatic quality degradation
    fn execute_with_degradation(&self, params: &Self::Params) -> Result<Self::Output> {
        // Try full quality first
        if let Ok(result) = self.try_full_quality(params) {
            return Ok(result);
        }

        // Fall back to reduced quality
        if let Ok(result) = self.fallback_reduced_quality(params) {
            eprintln!("Degraded to reduced quality");
            return Ok(result);
        }

        // Final fallback to minimal quality
        eprintln!("Degraded to minimal quality");
        self.fallback_minimal_quality(params)
    }
}
