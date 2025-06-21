//! Advanced error diagnostics and reporting for ``SciRS2``
//!
//! This module provides enhanced error diagnostics including:
//! - Contextual error analysis
//! - Performance impact assessment
//! - Environment diagnostics
//! - Error pattern recognition
//! - Automated troubleshooting suggestions

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use crate::error::CoreError;

/// Environment information for error diagnostics
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    /// Operating system information
    pub os: String,
    /// Architecture (x86_64, aarch64, etc.)
    pub arch: String,
    /// Available memory in bytes
    pub available_memory: Option<u64>,
    /// Number of CPU cores
    pub cpu_cores: Option<usize>,
    /// Rust compiler version
    pub rustc_version: Option<String>,
    /// ``SciRS2`` version
    pub scirs2_version: String,
    /// Enabled features
    pub features: Vec<String>,
    /// Environment variables of interest
    pub env_vars: HashMap<String, String>,
}

impl Default for EnvironmentInfo {
    fn default() -> Self {
        let mut env_vars = HashMap::new();

        // Collect relevant environment variables
        let relevant_vars = [
            "RUST_LOG",
            "RUST_BACKTRACE",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "RAYON_NUM_THREADS",
            "CARGO_MANIFEST_DIR",
        ];

        for var in &relevant_vars {
            if let Ok(value) = std::env::var(var) {
                env_vars.insert(var.to_string(), value);
            }
        }

        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            available_memory: Self::get_available_memory(),
            cpu_cores: std::thread::available_parallelism().ok().map(|n| n.get()),
            rustc_version: option_env!("RUSTC_VERSION").map(|s| s.to_string()),
            scirs2_version: env!("CARGO_PKG_VERSION").to_string(),
            features: Self::get_enabled_features(),
            env_vars,
        }
    }
}

impl EnvironmentInfo {
    /// Get available memory in bytes (platform-specific)
    fn get_available_memory() -> Option<u64> {
        // This is a simplified implementation
        // In a real implementation, you'd use platform-specific APIs
        #[cfg(unix)]
        {
            if let Ok(pages) = std::process::Command::new("getconf")
                .args(["_PHYS_PAGES"])
                .output()
            {
                if let Ok(pages_str) = String::from_utf8(pages.stdout) {
                    if let Ok(pages_num) = pages_str.trim().parse::<u64>() {
                        if let Ok(page_size) = std::process::Command::new("getconf")
                            .args(["PAGE_SIZE"])
                            .output()
                        {
                            if let Ok(size_str) = String::from_utf8(page_size.stdout) {
                                if let Ok(size_num) = size_str.trim().parse::<u64>() {
                                    return Some(pages_num * size_num);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Get list of enabled features
    #[allow(clippy::vec_init_then_push)]
    fn get_enabled_features() -> Vec<String> {
        #[allow(unused_mut)]
        let mut features = Vec::new();

        #[cfg(feature = "parallel")]
        features.push("parallel".to_string());

        #[cfg(feature = "simd")]
        features.push("simd".to_string());

        #[cfg(feature = "gpu")]
        features.push("gpu".to_string());

        #[cfg(feature = "openblas")]
        features.push("openblas".to_string());

        #[cfg(feature = "intel-mkl")]
        features.push("intel-mkl".to_string());

        #[cfg(feature = "profiling")]
        features.push("profiling".to_string());

        features
    }
}

/// Error occurrence tracking for pattern recognition
#[derive(Debug, Clone)]
pub struct ErrorOccurrence {
    /// Error type
    pub error_type: String,
    /// Timestamp when error occurred
    pub timestamp: SystemTime,
    /// Context where error occurred
    pub context: String,
    /// Function or module where error occurred
    pub location: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ErrorOccurrence {
    /// Create a new error occurrence
    pub fn new(error: &CoreError, context: String) -> Self {
        let error_type = format!("{:?}", error)
            .split('(')
            .next()
            .unwrap_or("Unknown")
            .to_string();

        Self {
            error_type,
            timestamp: SystemTime::now(),
            context,
            location: None,
            metadata: HashMap::new(),
        }
    }

    /// Add location information
    pub fn with_location<S: Into<String>>(mut self, location: S) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Add metadata
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Error pattern analysis
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    /// Pattern description
    pub description: String,
    /// Error types involved in this pattern
    pub error_types: Vec<String>,
    /// Frequency of this pattern
    pub frequency: usize,
    /// Common contexts where this pattern occurs
    pub common_contexts: Vec<String>,
    /// Suggested actions for this pattern
    pub suggestions: Vec<String>,
}

/// Error diagnostics engine
#[derive(Debug)]
pub struct ErrorDiagnostics {
    /// Environment information
    environment: EnvironmentInfo,
    /// Recent error occurrences
    error_history: Arc<Mutex<Vec<ErrorOccurrence>>>,
    /// Maximum number of errors to keep in history
    max_history: usize,
    /// Known error patterns
    patterns: Vec<ErrorPattern>,
}

static GLOBAL_DIAGNOSTICS: OnceLock<ErrorDiagnostics> = OnceLock::new();

impl ErrorDiagnostics {
    /// Create a new error diagnostics engine
    pub fn new() -> Self {
        Self {
            environment: EnvironmentInfo::default(),
            error_history: Arc::new(Mutex::new(Vec::new())),
            max_history: 1000,
            patterns: Self::initialize_patterns(),
        }
    }

    /// Get the global diagnostics instance
    pub fn global() -> &'static ErrorDiagnostics {
        GLOBAL_DIAGNOSTICS.get_or_init(Self::new)
    }

    /// Record an error occurrence
    pub fn record_error(&self, error: &CoreError, context: String) {
        let occurrence = ErrorOccurrence::new(error, context);

        let mut history = self.error_history.lock().unwrap();
        history.push(occurrence);

        // Keep only the most recent errors
        if history.len() > self.max_history {
            history.remove(0);
        }
    }

    /// Analyze an error and provide comprehensive diagnostics
    pub fn analyze_error(&self, error: &CoreError) -> ErrorDiagnosticReport {
        let mut report = ErrorDiagnosticReport::new(error.clone());

        // Add environment information
        report.environment = Some(self.environment.clone());

        // Analyze error patterns
        report.patterns = self.find_matching_patterns(error);

        // Check for recent similar errors
        report.recent_occurrences =
            self.find_recent_similar_errors(error, Duration::from_secs(300)); // 5 minutes

        // Assess performance impact
        report.performance_impact = self.assess_performance_impact(error);

        // Generate contextual suggestions
        report.contextual_suggestions = self.generate_contextual_suggestions(error, &report);

        // Add environment-specific diagnostics
        report.environment_diagnostics = self.diagnose_environment_issues(error);

        report
    }

    /// Find patterns matching the given error
    fn find_matching_patterns(&self, error: &CoreError) -> Vec<ErrorPattern> {
        let error_type = format!("{:?}", error)
            .split('(')
            .next()
            .unwrap_or("Unknown")
            .to_string();

        self.patterns
            .iter()
            .filter(|pattern| pattern.error_types.contains(&error_type))
            .cloned()
            .collect()
    }

    /// Find recent similar errors
    fn find_recent_similar_errors(
        &self,
        error: &CoreError,
        window: Duration,
    ) -> Vec<ErrorOccurrence> {
        let error_type = format!("{:?}", error)
            .split('(')
            .next()
            .unwrap_or("Unknown")
            .to_string();
        let cutoff = SystemTime::now() - window;

        let history = self.error_history.lock().unwrap();
        history
            .iter()
            .filter(|occurrence| {
                occurrence.error_type == error_type && occurrence.timestamp >= cutoff
            })
            .cloned()
            .collect()
    }

    /// Assess the performance impact of an error
    fn assess_performance_impact(&self, error: &CoreError) -> PerformanceImpact {
        match error {
            CoreError::MemoryError(_) => PerformanceImpact::High,
            CoreError::TimeoutError(_) => PerformanceImpact::High,
            CoreError::ConvergenceError(_) => PerformanceImpact::Medium,
            CoreError::ComputationError(_) => PerformanceImpact::Medium,
            CoreError::DomainError(_) | CoreError::ValueError(_) => PerformanceImpact::Low,
            _ => PerformanceImpact::Unknown,
        }
    }

    /// Generate contextual suggestions based on error analysis
    fn generate_contextual_suggestions(
        &self,
        _error: &CoreError,
        report: &ErrorDiagnosticReport,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Environment-based suggestions
        if let Some(env) = &report.environment {
            if env.available_memory.is_some_and(|mem| mem < 2_000_000_000) {
                // Less than 2GB
                suggestions.push(
                    "Consider using memory-efficient algorithms for large datasets".to_string(),
                );
            }

            if env.cpu_cores == Some(1) {
                suggestions.push(
                    "Single-core system detected - parallel algorithms may not provide benefits"
                        .to_string(),
                );
            }

            if !env.features.contains(&"simd".to_string()) {
                suggestions.push(
                    "SIMD optimizations not enabled - consider enabling for better performance"
                        .to_string(),
                );
            }
        }

        // Pattern-based suggestions
        for pattern in &report.patterns {
            suggestions.extend(pattern.suggestions.clone());
        }

        // Frequency-based suggestions
        if report.recent_occurrences.len() > 3 {
            suggestions.push("This error has occurred frequently recently - consider reviewing input data or algorithm parameters".to_string());
        }

        suggestions
    }

    /// Diagnose environment-specific issues with enhanced Alpha 6 analysis
    fn diagnose_environment_issues(&self, error: &CoreError) -> Vec<String> {
        let mut diagnostics = Vec::new();

        match error {
            CoreError::MemoryError(_) => {
                if let Some(mem) = self.environment.available_memory {
                    diagnostics.push(format!(
                        "Available memory: {:.2} GB",
                        mem as f64 / 1_000_000_000.0
                    ));

                    // Enhanced memory analysis
                    if mem < 4_000_000_000 {
                        diagnostics.push(
                            "Low memory detected - consider using memory-efficient algorithms"
                                .to_string(),
                        );
                    }
                    if mem > 64_000_000_000 {
                        diagnostics.push(
                            "High memory system - can use in-memory algorithms for large datasets"
                                .to_string(),
                        );
                    }
                }

                // Check for memory-related environment variables
                if let Some(threads) = self.environment.env_vars.get("OMP_NUM_THREADS") {
                    diagnostics.push(format!("OpenMP threads: {}", threads));
                }

                // Check for memory management features
                if !self
                    .environment
                    .features
                    .contains(&"memory_efficient".to_string())
                {
                    diagnostics.push(
                        "Memory-efficient algorithms not enabled - consider enabling this feature"
                            .to_string(),
                    );
                }
            }

            CoreError::ComputationError(_) => {
                if let Some(cores) = self.environment.cpu_cores {
                    diagnostics.push(format!("CPU cores available: {}", cores));

                    // Enhanced CPU analysis
                    if cores == 1 {
                        diagnostics.push(
                            "Single-core system - parallel algorithms won't help".to_string(),
                        );
                    } else if cores > 32 {
                        diagnostics.push(
                            "High-core count system - consider NUMA-aware algorithms".to_string(),
                        );
                    }
                }

                // Check compiler optimizations
                #[cfg(debug_assertions)]
                diagnostics.push("Running in debug mode - performance may be reduced".to_string());

                // Check for performance features
                if !self.environment.features.contains(&"parallel".to_string()) {
                    diagnostics.push(
                        "Parallel processing not enabled - could improve performance".to_string(),
                    );
                }
                if !self.environment.features.contains(&"simd".to_string()) {
                    diagnostics.push(
                        "SIMD optimizations not enabled - could improve numerical performance"
                            .to_string(),
                    );
                }
            }

            CoreError::TimeoutError(_) => {
                diagnostics
                    .push("Timeout detected - operation took longer than expected".to_string());
                if let Some(cores) = self.environment.cpu_cores {
                    if cores > 1 && !self.environment.features.contains(&"parallel".to_string()) {
                        diagnostics.push(
                            "Multi-core system but parallel processing not enabled".to_string(),
                        );
                    }
                }
            }

            _ => {}
        }

        // General environment diagnostics
        self.add_general_environment_diagnostics(&mut diagnostics);

        diagnostics
    }

    /// Add general environment diagnostics for Alpha 6
    fn add_general_environment_diagnostics(&self, diagnostics: &mut Vec<String>) {
        // Check for optimal thread configuration
        if let (Some(cores), Some(omp_threads)) = (
            self.environment.cpu_cores,
            self.environment.env_vars.get("OMP_NUM_THREADS"),
        ) {
            if let Ok(omp_count) = omp_threads.parse::<usize>() {
                if omp_count > cores {
                    diagnostics.push(
                        "OMP_NUM_THREADS exceeds available cores - may cause oversubscription"
                            .to_string(),
                    );
                }
            }
        }

        // Check for BLAS backend optimization
        if self.environment.features.contains(&"openblas".to_string()) {
            diagnostics.push("Using OpenBLAS backend - good for general computations".to_string());
        } else if self.environment.features.contains(&"intel-mkl".to_string()) {
            diagnostics.push("Using Intel MKL backend - optimized for Intel CPUs".to_string());
        } else if self.environment.features.contains(&"linalg".to_string()) {
            diagnostics.push(
                "Linear algebra backend available but no optimized BLAS detected".to_string(),
            );
        }

        // Check for GPU capabilities
        if self.environment.features.contains(&"gpu".to_string()) {
            diagnostics.push("GPU acceleration available".to_string());
            if self.environment.features.contains(&"cuda".to_string()) {
                diagnostics.push("CUDA backend enabled for NVIDIA GPUs".to_string());
            }
        }
    }

    /// Perform predictive error analysis based on historical patterns (Alpha 6 feature)
    pub fn predict_potential_errors(&self, context: &str) -> Vec<String> {
        let mut predictions = Vec::new();
        let history = self.error_history.lock().unwrap();

        // Analyze error frequency patterns
        let mut error_counts: HashMap<String, usize> = HashMap::new();
        let recent_cutoff = SystemTime::now() - Duration::from_secs(3600); // Last hour

        for occurrence in history.iter() {
            if occurrence.timestamp >= recent_cutoff {
                *error_counts
                    .entry(occurrence.error_type.clone())
                    .or_insert(0) += 1;
            }
        }

        // Predict based on high-frequency patterns
        for (error_type, count) in error_counts {
            if count >= 3 {
                predictions.push(format!(
                    "High risk of {} based on recent frequency ({}x in last hour)",
                    error_type, count
                ));
            }
        }

        // Context-based predictions
        if (context.contains("matrix") || context.contains("linear_algebra"))
            && self
                .environment
                .available_memory
                .is_some_and(|mem| mem < 8_000_000_000)
        {
            predictions.push("Potential memory issues with large matrix operations".to_string());
        }

        if context.contains("optimization") || context.contains("solver") {
            predictions.push(
                "Potential convergence issues - consider robust initial conditions".to_string(),
            );
        }

        if context.contains("parallel") && self.environment.cpu_cores == Some(1) {
            predictions
                .push("Parallel algorithms may not be effective on single-core system".to_string());
        }

        predictions
    }

    /// Generate domain-specific recovery strategies (Alpha 6 feature)
    pub fn suggest_domain_recovery(&self, error: &CoreError, domain: &str) -> Vec<String> {
        let mut strategies = Vec::new();

        match domain {
            "linear_algebra" => match error {
                CoreError::MemoryError(_) => {
                    strategies.extend(vec![
                        "Use iterative solvers instead of direct factorization".to_string(),
                        "Implement block algorithms for large matrices".to_string(),
                        "Consider sparse matrix representations".to_string(),
                        "Use out-of-core matrix algorithms".to_string(),
                    ]);
                }
                CoreError::ConvergenceError(_) => {
                    strategies.extend(vec![
                        "Apply preconditioning (ILU, SSOR, or Jacobi)".to_string(),
                        "Use more robust factorization (SVD instead of LU)".to_string(),
                        "Check matrix conditioning and apply regularization".to_string(),
                        "Try different solver algorithms (GMRES, BiCGSTAB)".to_string(),
                    ]);
                }
                _ => {}
            },

            "optimization" => match error {
                CoreError::ConvergenceError(_) => {
                    strategies.extend(vec![
                        "Use trust region methods for better global convergence".to_string(),
                        "Apply line search algorithms with backtracking".to_string(),
                        "Try multiple random starting points".to_string(),
                        "Use gradient-free methods for noisy objectives".to_string(),
                        "Implement adaptive step size control".to_string(),
                    ]);
                }
                CoreError::DomainError(_) => {
                    strategies.extend(vec![
                        "Add constraint handling and projection operators".to_string(),
                        "Use barrier methods for constrained optimization".to_string(),
                        "Implement bounds checking and clipping".to_string(),
                    ]);
                }
                _ => {}
            },

            "statistics" => match error {
                CoreError::DomainError(_) => {
                    strategies.extend(vec![
                        "Use robust statistical methods for outliers".to_string(),
                        "Apply data transformation (log, Box-Cox)".to_string(),
                        "Implement missing data handling strategies".to_string(),
                        "Use non-parametric methods for non-normal data".to_string(),
                    ]);
                }
                CoreError::ComputationError(_) => {
                    strategies.extend(vec![
                        "Use numerically stable algorithms (Welford for variance)".to_string(),
                        "Apply importance sampling for rare events".to_string(),
                        "Use bootstrap methods for robust estimates".to_string(),
                    ]);
                }
                _ => {}
            },

            "signal_processing" => match error {
                CoreError::MemoryError(_) => {
                    strategies.extend(vec![
                        "Use streaming FFT algorithms".to_string(),
                        "Implement overlap-add/overlap-save methods".to_string(),
                        "Use decimation for reduced sample rates".to_string(),
                    ]);
                }
                CoreError::ComputationError(_) => {
                    strategies.extend(vec![
                        "Use windowing to reduce spectral leakage".to_string(),
                        "Apply zero-padding for better frequency resolution".to_string(),
                        "Use advanced spectral estimation methods".to_string(),
                    ]);
                }
                _ => {}
            },

            _ => {
                // Generic domain-agnostic strategies
                strategies.push("Consider using more robust numerical algorithms".to_string());
                strategies.push("Implement error checking and data validation".to_string());
                strategies.push("Use iterative refinement for better accuracy".to_string());
            }
        }

        strategies
    }

    /// Initialize known error patterns with enhanced Alpha 6 patterns
    fn initialize_patterns() -> Vec<ErrorPattern> {
        vec![
            ErrorPattern {
                description: "Memory allocation failures in large matrix operations".to_string(),
                error_types: vec!["MemoryError".to_string()],
                frequency: 0,
                common_contexts: vec![
                    "matrix_multiplication".to_string(),
                    "decomposition".to_string(),
                ],
                suggestions: vec![
                    "Use chunked processing for large matrices".to_string(),
                    "Consider using f32 instead of f64 to reduce memory usage".to_string(),
                    "Enable out-of-core algorithms if available".to_string(),
                    "Use memory-mapped arrays for very large datasets".to_string(),
                    "Consider distributed computing for extremely large problems".to_string(),
                ],
            },
            ErrorPattern {
                description: "Convergence failures in iterative algorithms".to_string(),
                error_types: vec!["ConvergenceError".to_string()],
                frequency: 0,
                common_contexts: vec!["optimization".to_string(), "linear_solver".to_string()],
                suggestions: vec![
                    "Increase maximum iteration count".to_string(),
                    "Adjust convergence tolerance".to_string(),
                    "Try different initial conditions".to_string(),
                    "Use preconditioning to improve convergence".to_string(),
                    "Consider adaptive step sizes or trust region methods".to_string(),
                    "Check problem conditioning and scaling".to_string(),
                ],
            },
            ErrorPattern {
                description: "Shape mismatches in array operations".to_string(),
                error_types: vec!["ShapeError".to_string(), "DimensionError".to_string()],
                frequency: 0,
                common_contexts: vec!["matrix_operations".to_string(), "broadcasting".to_string()],
                suggestions: vec![
                    "Check input array shapes before operations".to_string(),
                    "Use reshaping or broadcasting to make arrays compatible".to_string(),
                    "Verify matrix multiplication dimension compatibility (A: m×k, B: k×n)"
                        .to_string(),
                    "Consider using automatic broadcasting utilities".to_string(),
                    "Use array protocol for mixed array type compatibility".to_string(),
                ],
            },
            ErrorPattern {
                description: "Domain errors with mathematical functions".to_string(),
                error_types: vec!["DomainError".to_string()],
                frequency: 0,
                common_contexts: vec!["special_functions".to_string(), "statistics".to_string()],
                suggestions: vec![
                    "Check input ranges for mathematical functions".to_string(),
                    "Handle edge cases (zero, negative values, infinities)".to_string(),
                    "Use input validation before calling functions".to_string(),
                    "Consider using robust numerical methods for ill-conditioned problems"
                        .to_string(),
                    "Use IEEE 754 special value handling where appropriate".to_string(),
                ],
            },
            // New Alpha 6 patterns
            ErrorPattern {
                description: "GPU memory exhaustion in accelerated computations".to_string(),
                error_types: vec!["MemoryError".to_string(), "ComputationError".to_string()],
                frequency: 0,
                common_contexts: vec![
                    "gpu_acceleration".to_string(),
                    "neural_networks".to_string(),
                ],
                suggestions: vec![
                    "Reduce batch size to fit GPU memory".to_string(),
                    "Use gradient accumulation for large batches".to_string(),
                    "Enable mixed precision training (fp16/fp32)".to_string(),
                    "Consider model parallelism for very large models".to_string(),
                    "Use CPU fallback for computations that don't fit on GPU".to_string(),
                ],
            },
            ErrorPattern {
                description: "Numerical instability in scientific computations".to_string(),
                error_types: vec![
                    "ComputationError".to_string(),
                    "ConvergenceError".to_string(),
                ],
                frequency: 0,
                common_contexts: vec![
                    "linear_algebra".to_string(),
                    "ode_solving".to_string(),
                    "pde_solving".to_string(),
                ],
                suggestions: vec![
                    "Use higher precision (f64 instead of f32)".to_string(),
                    "Apply numerical stabilization techniques".to_string(),
                    "Check condition numbers and matrix rank".to_string(),
                    "Use pivoting strategies in decompositions".to_string(),
                    "Consider regularization for ill-posed problems".to_string(),
                    "Use iterative refinement for better accuracy".to_string(),
                ],
            },
            ErrorPattern {
                description: "Parallel processing overhead and contention".to_string(),
                error_types: vec!["TimeoutError".to_string(), "ComputationError".to_string()],
                frequency: 0,
                common_contexts: vec!["parallel_computing".to_string(), "rayon".to_string()],
                suggestions: vec![
                    "Adjust thread pool size based on workload".to_string(),
                    "Use work-stealing for better load balancing".to_string(),
                    "Minimize false sharing in parallel algorithms".to_string(),
                    "Consider chunking strategies for better cache locality".to_string(),
                    "Profile thread utilization and adjust accordingly".to_string(),
                    "Use NUMA-aware allocation for large systems".to_string(),
                ],
            },
            ErrorPattern {
                description: "Data type overflow and underflow in scientific calculations"
                    .to_string(),
                error_types: vec!["ValueError".to_string(), "ComputationError".to_string()],
                frequency: 0,
                common_contexts: vec![
                    "numerical_integration".to_string(),
                    "statistics".to_string(),
                ],
                suggestions: vec![
                    "Use logarithmic computations for very large/small values".to_string(),
                    "Implement numerical scaling and normalization".to_string(),
                    "Use arbitrary precision arithmetic for extreme ranges".to_string(),
                    "Apply Kahan summation for better numerical stability".to_string(),
                    "Check for intermediate overflow in complex calculations".to_string(),
                ],
            },
            ErrorPattern {
                description: "I/O and serialization failures in scientific data".to_string(),
                error_types: vec!["IoError".to_string(), "SerializationError".to_string()],
                frequency: 0,
                common_contexts: vec!["data_loading".to_string(), "checkpointing".to_string()],
                suggestions: vec![
                    "Use streaming I/O for large datasets".to_string(),
                    "Implement progressive loading with error recovery".to_string(),
                    "Use compression to reduce I/O overhead".to_string(),
                    "Implement chunked serialization for large arrays".to_string(),
                    "Use memory-mapped files for random access patterns".to_string(),
                    "Consider distributed storage for very large datasets".to_string(),
                ],
            },
        ]
    }
}

impl Default for ErrorDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance impact assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceImpact {
    Unknown,
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for PerformanceImpact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => write!(f, "Unknown"),
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Comprehensive error diagnostic report with Alpha 6 enhancements
#[derive(Debug)]
pub struct ErrorDiagnosticReport {
    /// The original error
    pub error: CoreError,
    /// Environment information
    pub environment: Option<EnvironmentInfo>,
    /// Matching error patterns
    pub patterns: Vec<ErrorPattern>,
    /// Recent similar occurrences
    pub recent_occurrences: Vec<ErrorOccurrence>,
    /// Performance impact assessment
    pub performance_impact: PerformanceImpact,
    /// Contextual suggestions
    pub contextual_suggestions: Vec<String>,
    /// Environment-specific diagnostics
    pub environment_diagnostics: Vec<String>,
    /// Alpha 6: Predictive error analysis
    pub predictions: Vec<String>,
    /// Alpha 6: Domain-specific recovery strategies
    pub domain_strategies: Vec<String>,
    /// Timestamp when report was generated
    pub generated_at: SystemTime,
}

impl ErrorDiagnosticReport {
    /// Create a new diagnostic report
    pub fn new(error: CoreError) -> Self {
        Self {
            error,
            environment: None,
            patterns: Vec::new(),
            recent_occurrences: Vec::new(),
            performance_impact: PerformanceImpact::Unknown,
            contextual_suggestions: Vec::new(),
            environment_diagnostics: Vec::new(),
            predictions: Vec::new(),
            domain_strategies: Vec::new(),
            generated_at: SystemTime::now(),
        }
    }

    /// Generate a comprehensive report string
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        // Header
        report.push_str("🔍 `SciRS2` Error Diagnostic Report\n");
        report.push_str(&format!("Generated: {:?}\n", self.generated_at));
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        // Error information
        report.push_str("🚨 Error Details:\n");
        report.push_str(&format!("   {}\n\n", self.error));

        // Performance impact
        report.push_str(&format!(
            "⚡ Performance Impact: {}\n\n",
            self.performance_impact
        ));

        // Environment information
        if let Some(env) = &self.environment {
            report.push_str("🖥️  Environment Information:\n");
            report.push_str(&format!("   OS: {} ({})\n", env.os, env.arch));
            report.push_str(&format!("   `SciRS2` Version: {}\n", env.scirs2_version));

            if let Some(cores) = env.cpu_cores {
                report.push_str(&format!("   CPU Cores: {}\n", cores));
            }

            if let Some(memory) = env.available_memory {
                report.push_str(&format!(
                    "   Available Memory: {:.2} GB\n",
                    memory as f64 / 1_000_000_000.0
                ));
            }

            if !env.features.is_empty() {
                report.push_str(&format!(
                    "   Enabled Features: {}\n",
                    env.features.join(", ")
                ));
            }

            report.push('\n');
        }

        // Environment diagnostics
        if !self.environment_diagnostics.is_empty() {
            report.push_str("🔧 Environment Diagnostics:\n");
            for diagnostic in &self.environment_diagnostics {
                report.push_str(&format!("   • {}\n", diagnostic));
            }
            report.push('\n');
        }

        // Error patterns
        if !self.patterns.is_empty() {
            report.push_str("📊 Matching Error Patterns:\n");
            for pattern in &self.patterns {
                report.push_str(&format!("   • {}\n", pattern.description));
                if !pattern.suggestions.is_empty() {
                    report.push_str("     Suggestions:\n");
                    for suggestion in &pattern.suggestions {
                        report.push_str(&format!("     - {}\n", suggestion));
                    }
                }
            }
            report.push('\n');
        }

        // Recent occurrences
        if !self.recent_occurrences.is_empty() {
            report.push_str(&format!(
                "📈 Recent Similar Errors: {} in the last 5 minutes\n",
                self.recent_occurrences.len()
            ));
            if self.recent_occurrences.len() > 3 {
                report.push_str(
                    "   ⚠️  High frequency detected - this may indicate a systematic issue\n",
                );
            }
            report.push('\n');
        }

        // Alpha 6: Predictive analysis
        if !self.predictions.is_empty() {
            report.push_str("🔮 Predictive Analysis:\n");
            for prediction in &self.predictions {
                report.push_str(&format!("   • {}\n", prediction));
            }
            report.push('\n');
        }

        // Alpha 6: Domain-specific recovery strategies
        if !self.domain_strategies.is_empty() {
            report.push_str("🎯 Domain-Specific Recovery Strategies:\n");
            for (i, strategy) in self.domain_strategies.iter().enumerate() {
                report.push_str(&format!("   {}. {}\n", i + 1, strategy));
            }
            report.push('\n');
        }

        // Contextual suggestions
        if !self.contextual_suggestions.is_empty() {
            report.push_str("💡 Contextual Suggestions:\n");
            for (i, suggestion) in self.contextual_suggestions.iter().enumerate() {
                report.push_str(&format!("   {}. {}\n", i + 1, suggestion));
            }
            report.push('\n');
        }

        // Footer
        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("For more help, visit: https://github.com/cool-japan/scirs/issues\n");

        report
    }
}

impl fmt::Display for ErrorDiagnosticReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.generate_report())
    }
}

/// Convenience function to create a diagnostic report for an error
pub fn diagnose_error(error: &CoreError) -> ErrorDiagnosticReport {
    ErrorDiagnostics::global().analyze_error(error)
}

/// Convenience function to create a diagnostic report with context
pub fn diagnose_error_with_context(error: &CoreError, context: String) -> ErrorDiagnosticReport {
    let diagnostics = ErrorDiagnostics::global();
    diagnostics.record_error(error, context);
    diagnostics.analyze_error(error)
}

/// Macro to create a diagnostic error with automatic context
#[macro_export]
macro_rules! diagnostic_error {
    ($error_type:ident, $message:expr) => {{
        let error = $crate::error::CoreError::$error_type($crate::error_context!($message));
        let context = format!("{}:{}", file!(), line!());
        $crate::error::diagnostics::diagnose_error_with_context(&error, context);
        error
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorContext;

    #[test]
    fn test_environment_info() {
        let env = EnvironmentInfo::default();
        assert!(!env.os.is_empty());
        assert!(!env.arch.is_empty());
        assert!(!env.scirs2_version.is_empty());
    }

    #[test]
    fn test_error_occurrence() {
        let error = CoreError::DomainError(ErrorContext::new("Test error"));
        let occurrence = ErrorOccurrence::new(&error, "test_context".to_string())
            .with_location("test_function")
            .with_metadata("key", "value");

        assert_eq!(occurrence.error_type, "DomainError");
        assert_eq!(occurrence.context, "test_context");
        assert_eq!(occurrence.location, Some("test_function".to_string()));
        assert_eq!(occurrence.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_error_diagnostics() {
        let diagnostics = ErrorDiagnostics::new();
        let error = CoreError::MemoryError(ErrorContext::new("Out of memory"));

        let report = diagnostics.analyze_error(&error);
        assert!(matches!(report.error, CoreError::MemoryError(_)));
        assert!(matches!(report.performance_impact, PerformanceImpact::High));
    }

    #[test]
    fn test_diagnostic_report_generation() {
        let error = CoreError::ShapeError(ErrorContext::new("Shape mismatch"));
        let report = diagnose_error(&error);

        let report_string = report.generate_report();
        assert!(report_string.contains("Error Diagnostic Report"));
        assert!(report_string.contains("Shape mismatch"));
    }
}
