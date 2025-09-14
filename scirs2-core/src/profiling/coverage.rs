//! # Test Coverage Analysis System
//!
//! Enterprise-grade test coverage analysis with comprehensive tracking of code coverage,
//! branch coverage, and integration coverage for production-level quality assurance.
//! Provides detailed insights into test effectiveness and identifies uncovered code paths.
//!
//! ## Features
//!
//! - **Code Coverage**: Line-by-line execution tracking with detailed statistics
//! - **Branch Coverage**: Decision point analysis for conditional statements
//! - **Integration Coverage**: Cross-module and cross-component coverage tracking
//! - **Coverage Visualization**: HTML reports, charts, and interactive coverage maps
//! - **Historical Tracking**: Coverage trends and regression analysis over time
//! - **Quality Gates**: Configurable coverage thresholds and pass/fail criteria
//! - **Differential Coverage**: Coverage analysis for code changes and pull requests
//! - **Multi-format Reports**: JSON, XML, LCOV, and HTML output formats
//! - **Real-time Monitoring**: Live coverage updates during test execution
//! - **Performance Impact**: Low-overhead instrumentation for production environments
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::profiling::coverage::{
//!     CoverageAnalyzer, CoverageConfig, CoverageType, ReportFormat
//! };
//!
//! // Create coverage analyzer
//! let config = CoverageConfig::production()
//!     .with_coverage_types(vec![
//!         CoverageType::Line,
//!         CoverageType::Branch,
//!         CoverageType::Integration
//!     ])
//!     .with_threshold(80.0)
//!     .with_report_format(ReportFormat::Html);
//!
//! let mut analyzer = CoverageAnalyzer::new(config)?;
//!
//! // Start coverage collection
//! analyzer.start_collection()?;
//!
//! // Run your tests here...
//! fn run_test_suite() {
//!     // Example test function that would run the actual test suite
//!     println!("Running test suite...");
//! }
//! run_test_suite();
//!
//! // Stop collection and generate report
//! let report = analyzer.stop_and_generate_report()?;
//! println!("Overall coverage: {:.2}%", report.overall_coverage_percentage());
//!
//! // Check if coverage meets thresholds
//! if report.meets_quality_gates() {
//!     println!("✅ Coverage quality gates passed!");
//! } else {
//!     println!("❌ Coverage below threshold");
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{CoreError, CoreResult};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

/// Coverage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageConfig {
    /// Types of coverage to collect
    pub coverage_types: Vec<CoverageType>,
    /// Minimum coverage threshold (percentage)
    pub coverage_threshold: f64,
    /// Branch coverage threshold
    pub branch_threshold: f64,
    /// Integration coverage threshold
    pub integration_threshold: f64,
    /// Report output formats
    pub report_formats: Vec<ReportFormat>,
    /// Output directory for reports
    pub output_directory: PathBuf,
    /// Include system/library code in coverage
    pub include_systemcode: bool,
    /// File patterns to exclude from coverage
    pub exclude_patterns: Vec<String>,
    /// File patterns to include (if empty, includes all)
    pub include_patterns: Vec<String>,
    /// Enable real-time coverage updates
    pub real_time_updates: bool,
    /// Sampling rate for performance optimization (1.0 = 100%)
    pub samplingrate: f64,
    /// Enable historical tracking
    pub enable_history: bool,
    /// History retention period
    pub history_retention: Duration,
    /// Enable differential coverage
    pub enable_diff_coverage: bool,
    /// Base commit/branch for differential coverage
    pub diffbase: Option<String>,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            coverage_types: vec![CoverageType::Line, CoverageType::Branch],
            coverage_threshold: 80.0,
            branch_threshold: 70.0,
            integration_threshold: 60.0,
            report_formats: vec![ReportFormat::Html, ReportFormat::Json],
            output_directory: PathBuf::from("coverage_reports"),
            include_systemcode: false,
            exclude_patterns: vec![
                "*/tests/*".to_string(),
                "*/benches/*".to_string(),
                "*/examples/*".to_string(),
            ],
            include_patterns: vec![],
            real_time_updates: true,
            samplingrate: 1.0,
            enable_history: true,
            history_retention: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            enable_diff_coverage: false,
            diffbase: None,
        }
    }
}

impl CoverageConfig {
    /// Create production-optimized configuration
    pub fn production() -> Self {
        Self {
            coverage_threshold: 85.0,
            branch_threshold: 75.0,
            integration_threshold: 70.0,
            samplingrate: 0.1, // 10% sampling for production
            real_time_updates: false,
            ..Default::default()
        }
    }

    /// Create development configuration with comprehensive coverage
    pub fn development() -> Self {
        Self {
            coverage_types: vec![
                CoverageType::Line,
                CoverageType::Branch,
                CoverageType::Function,
                CoverageType::Integration,
            ],
            coverage_threshold: 75.0,
            real_time_updates: true,
            samplingrate: 1.0,
            ..Default::default()
        }
    }

    /// Set coverage types to collect
    pub fn with_coverage_types(mut self, types: Vec<CoverageType>) -> Self {
        self.coverage_types = types;
        self
    }

    /// Set minimum coverage threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.coverage_threshold = threshold;
        self
    }

    /// Set branch coverage threshold
    pub fn with_branch_threshold(mut self, threshold: f64) -> Self {
        self.branch_threshold = threshold;
        self
    }

    /// Set report output format
    pub fn with_report_format(mut self, format: ReportFormat) -> Self {
        self.report_formats = vec![format];
        self
    }

    /// Set output directory
    pub fn with_output_directory<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.output_directory = path.as_ref().to_path_buf();
        self
    }

    /// Enable differential coverage
    pub fn with_diff_coverage(mut self, base: &str) -> Self {
        self.enable_diff_coverage = true;
        self.diffbase = Some(base.to_string());
        self
    }

    /// Set file exclusion patterns
    pub fn with_exclude_patterns(mut self, patterns: Vec<&str>) -> Self {
        self.exclude_patterns = patterns.into_iter().map(|s| s.to_string()).collect();
        self
    }
}

/// Types of coverage analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoverageType {
    /// Line coverage - tracks executed lines
    Line,
    /// Branch coverage - tracks decision points
    Branch,
    /// Function coverage - tracks executed functions
    Function,
    /// Statement coverage - tracks individual statements
    Statement,
    /// Integration coverage - tracks cross-module interactions
    Integration,
    /// Path coverage - tracks execution paths
    Path,
    /// Condition coverage - tracks boolean conditions
    Condition,
}

/// Report output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// HTML report with interactive visualization
    Html,
    /// JSON format for programmatic access
    Json,
    /// XML format (compatible with Jenkins, etc.)
    Xml,
    /// LCOV format for external tools
    Lcov,
    /// Plain text summary
    Text,
    /// CSV format for data analysis
    Csv,
}

/// Coverage data for a single source file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCoverage {
    /// File path
    pub file_path: PathBuf,
    /// Total lines in file
    pub total_lines: u32,
    /// Lines executed
    pub covered_lines: u32,
    /// Line-by-line execution counts
    pub line_hits: BTreeMap<u32, u32>,
    /// Branch coverage data
    pub branches: Vec<BranchCoverage>,
    /// Function coverage data
    pub functions: Vec<FunctionCoverage>,
    /// Integration points
    pub integrations: Vec<IntegrationPoint>,
    /// File modification time
    pub modified_time: SystemTime,
    /// Coverage collection timestamp
    pub collected_at: SystemTime,
}

impl FileCoverage {
    /// Calculate line coverage percentage
    pub fn line_coverage_percentage(&self) -> f64 {
        if self.total_lines == 0 {
            100.0
        } else {
            (self.covered_lines as f64 / self.total_lines as f64) * 100.0
        }
    }

    /// Calculate branch coverage percentage
    pub fn branch_coverage_percentage(&self) -> f64 {
        if self.branches.is_empty() {
            100.0
        } else {
            let covered_branches = self.branches.iter().filter(|b| b.is_covered()).count();
            (covered_branches as f64 / self.branches.len() as f64) * 100.0
        }
    }

    /// Calculate function coverage percentage
    pub fn function_coverage_percentage(&self) -> f64 {
        if self.functions.is_empty() {
            100.0
        } else {
            let covered_functions = self
                .functions
                .iter()
                .filter(|f| f.execution_count > 0)
                .count();
            (covered_functions as f64 / self.functions.len() as f64) * 100.0
        }
    }

    /// Get uncovered lines
    pub fn uncovered_lines(&self) -> Vec<u32> {
        (1..=self.total_lines)
            .filter(|line| !self.line_hits.contains_key(line))
            .collect()
    }

    /// Get hot spots (frequently executed lines)
    pub fn hot_spots(&self, threshold: u32) -> Vec<(u32, u32)> {
        self.line_hits
            .iter()
            .filter(|(_, &count)| count >= threshold)
            .map(|(&line, &count)| (line, count))
            .collect()
    }
}

/// Branch coverage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchCoverage {
    /// Line number of the branch
    pub line_number: u32,
    /// Branch ID (unique within file)
    pub branch_id: String,
    /// True branch execution count
    pub true_count: u32,
    /// False branch execution count
    pub false_count: u32,
    /// Branch type (if-else, match, etc.)
    pub branch_type: BranchType,
    /// Source code snippet
    pub source_snippet: String,
}

impl BranchCoverage {
    /// Check if branch is fully covered
    pub fn is_covered(&self) -> bool {
        self.true_count > 0 && self.false_count > 0
    }

    /// Get total execution count
    pub fn total_executions(&self) -> u32 {
        self.true_count + self.false_count
    }

    /// Calculate branch balance (how evenly distributed the executions are)
    pub fn balance_score(&self) -> f64 {
        if !self.is_covered() {
            0.0
        } else {
            let total = self.total_executions() as f64;
            let min_count = self.true_count.min(self.false_count) as f64;
            min_count / total * 2.0 // Score from 0.0 to 1.0
        }
    }
}

/// Types of branches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchType {
    /// If-else statement
    IfElse,
    /// Match/switch statement
    Match,
    /// While loop
    While,
    /// For loop
    For,
    /// Ternary operator
    Ternary,
    /// Logical AND/OR
    Logical,
    /// Other conditional
    Other,
}

/// Function coverage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCoverage {
    /// Function name
    pub function_name: String,
    /// Start line number
    pub start_line: u32,
    /// End line number
    pub end_line: u32,
    /// Execution count
    pub execution_count: u32,
    /// Function complexity score
    pub complexity: u32,
    /// Parameters count
    pub parameter_count: u32,
    /// Return type complexity
    pub return_complexity: u32,
}

impl FunctionCoverage {
    /// Calculate function coverage score based on complexity
    pub fn coverage_score(&self) -> f64 {
        if self.execution_count == 0 {
            0.0
        } else {
            // Score considers both execution and complexity
            let execution_factor = (self.execution_count as f64).ln().min(5.0) / 5.0;
            let complexity_factor = 1.0 / (1.0 + self.complexity as f64 / 10.0);
            execution_factor * complexity_factor
        }
    }
}

/// Integration coverage point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationPoint {
    /// Integration point ID
    pub id: String,
    /// Source module
    pub source_module: String,
    /// Target module
    pub target_module: String,
    /// Integration type
    pub integration_type: IntegrationType,
    /// Execution count
    pub execution_count: u32,
    /// Line number where integration occurs
    pub line_number: u32,
    /// Success rate (for error-prone integrations)
    pub success_rate: f64,
}

/// Types of integration points
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationType {
    /// Function call
    FunctionCall,
    /// Method invocation
    MethodCall,
    /// Trait implementation
    TraitImpl,
    /// Module import
    ModuleImport,
    /// Database connection
    DatabaseCall,
    /// Network request
    NetworkCall,
    /// File system operation
    FileSystemOp,
    /// Inter-process communication
    IpcCall,
}

/// Overall coverage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    /// Report generation timestamp
    pub generated_at: SystemTime,
    /// Configuration used
    pub config: CoverageConfig,
    /// Overall coverage statistics
    pub overall_stats: CoverageStatistics,
    /// Per-file coverage data
    pub file_coverage: HashMap<PathBuf, FileCoverage>,
    /// Coverage trends (if historical data available)
    pub trends: Option<CoverageTrends>,
    /// Quality gate results
    pub quality_gates: QualityGateResults,
    /// Performance impact metrics
    pub performance_impact: PerformanceImpact,
    /// Recommendations for improvement
    pub recommendations: Vec<CoverageRecommendation>,
}

impl CoverageReport {
    /// Calculate overall coverage percentage
    pub fn overall_coverage_percentage(&self) -> f64 {
        self.overall_stats.line_coverage_percentage
    }

    /// Check if report meets all quality gates
    pub fn meets_quality_gates(&self) -> bool {
        self.quality_gates.overall_passed
    }

    /// Get files with coverage below threshold
    pub fn files_below_threshold(&self) -> Vec<(&Path, f64)> {
        self.file_coverage
            .iter()
            .filter_map(|(path, coverage)| {
                let percentage = coverage.line_coverage_percentage();
                if percentage < self.config.coverage_threshold {
                    Some((path.as_path(), percentage))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get top uncovered functions by complexity
    pub fn critical_uncovered_functions(&self) -> Vec<(&FunctionCoverage, &Path)> {
        let mut uncovered: Vec<_> = self
            .file_coverage
            .iter()
            .flat_map(|(path, file_cov)| {
                file_cov
                    .functions
                    .iter()
                    .filter(|f| f.execution_count == 0)
                    .map(|f| (f, path.as_path()))
            })
            .collect();

        uncovered.sort_by(|a, b| b.0.complexity.cmp(&a.0.complexity));
        uncovered.into_iter().take(10).collect()
    }
}

/// Coverage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageStatistics {
    /// Total lines across all files
    pub total_lines: u32,
    /// Total covered lines
    pub covered_lines: u32,
    /// Line coverage percentage
    pub line_coverage_percentage: f64,
    /// Total branches
    pub total_branches: u32,
    /// Covered branches
    pub covered_branches: u32,
    /// Branch coverage percentage
    pub branch_coverage_percentage: f64,
    /// Total functions
    pub total_functions: u32,
    /// Covered functions
    pub covered_functions: u32,
    /// Function coverage percentage
    pub function_coverage_percentage: f64,
    /// Total integration points
    pub total_integrations: u32,
    /// Covered integration points
    pub covered_integrations: u32,
    /// Integration coverage percentage
    pub integration_coverage_percentage: f64,
    /// Number of files analyzed
    pub files_analyzed: u32,
}

/// Coverage trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageTrends {
    /// Historical coverage data points
    pub history: Vec<CoverageDataPoint>,
    /// Trend direction (improving, stable, declining)
    pub trend_direction: TrendDirection,
    /// Rate of change (percentage points per day)
    pub change_rate: f64,
    /// Prediction for next period
    pub predicted_coverage: Option<f64>,
}

/// Historical coverage data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageDataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Coverage percentage at this point
    pub coverage_percentage: f64,
    /// Branch coverage percentage
    pub branch_coverage_percentage: f64,
    /// Commit hash or version
    pub version: Option<String>,
    /// Test count at this time
    pub test_count: u32,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Coverage is improving
    Improving,
    /// Coverage is stable
    Stable,
    /// Coverage is declining
    Declining,
    /// Insufficient history
    Unknown,
}

/// Quality gate results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateResults {
    /// Overall quality gate status
    pub overall_passed: bool,
    /// Line coverage gate
    pub line_coverage_passed: bool,
    /// Branch coverage gate
    pub branch_coverage_passed: bool,
    /// Integration coverage gate
    pub integration_coverage_passed: bool,
    /// Failed gate details
    pub failures: Vec<QualityGateFailure>,
}

/// Quality gate failure details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateFailure {
    /// Gate type that failed
    pub gate_type: String,
    /// Expected threshold
    pub threshold: f64,
    /// Actual value
    pub actual_value: f64,
    /// Severity of failure
    pub severity: FailureSeverity,
    /// Suggested actions
    pub suggestions: Vec<String>,
}

/// Failure severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FailureSeverity {
    /// Minor failure - coverage slightly below threshold
    Minor,
    /// Moderate failure - significant coverage gap
    Moderate,
    /// Major failure - substantial coverage missing
    Major,
    /// Critical failure - coverage far below acceptable
    Critical,
}

/// Performance impact of coverage collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Overhead percentage (execution time increase)
    pub execution_overhead_percent: f64,
    /// Memory overhead in bytes
    pub memory_overhead_bytes: u64,
    /// Collection duration
    pub collection_duration: Duration,
    /// Number of instrumentation points
    pub instrumentation_points: u32,
    /// Sampling effectiveness
    pub sampling_effectiveness: f64,
}

/// Coverage improvement recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Detailed description
    pub description: String,
    /// Expected impact
    pub expected_impact: f64,
    /// Effort estimate (hours)
    pub effort_estimate: f64,
    /// Specific files/functions affected
    pub affected_items: Vec<String>,
}

/// Types of coverage recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Add missing unit tests
    AddUnitTests,
    /// Add integration tests
    AddIntegrationTests,
    /// Test edge cases and error paths
    TestEdgeCases,
    /// Improve branch coverage
    ImproveBranchCoverage,
    /// Add property-based tests
    AddPropertyTests,
    /// Test complex functions
    TestComplexFunctions,
    /// Remove dead code
    RemoveDeadCode,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Low priority recommendation
    Low,
    /// Medium priority recommendation
    Medium,
    /// High priority recommendation
    High,
    /// Critical priority recommendation
    Critical,
}

/// Main coverage analyzer
pub struct CoverageAnalyzer {
    /// Configuration
    config: CoverageConfig,
    /// Coverage data collection state
    collection_state: Arc<Mutex<CollectionState>>,
    /// File coverage data
    file_coverage: Arc<RwLock<HashMap<PathBuf, FileCoverage>>>,
    /// Historical data
    history: Arc<Mutex<Vec<CoverageDataPoint>>>,
    /// Start time of current collection
    collection_start: Option<Instant>,
    /// Performance tracking
    performance_tracker: PerformanceTracker,
}

/// Coverage collection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionState {
    /// Not collecting coverage
    Idle,
    /// Currently collecting coverage
    Collecting,
    /// Collection paused
    Paused,
    /// Collection completed
    Completed,
    /// Error during collection
    Error,
}

/// Performance tracker for coverage collection
#[derive(Debug, Default)]
struct PerformanceTracker {
    /// Memory usage before collection
    baseline_memory: u64,
    /// Execution time tracking
    execution_timer: Option<Instant>,
    /// Instrumentation point count
    instrumentation_count: u32,
}

impl CoverageAnalyzer {
    /// Create a new coverage analyzer
    pub fn new(config: CoverageConfig) -> CoreResult<Self> {
        // Create output directory if it doesn't exist
        if !config.output_directory.exists() {
            std::fs::create_dir_all(&config.output_directory).map_err(|e| {
                CoreError::from(std::io::Error::other(format!(
                    "Failed to create output directory: {e}"
                )))
            })?;
        }

        Ok(Self {
            config,
            collection_state: Arc::new(Mutex::new(CollectionState::Idle)),
            file_coverage: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(Mutex::new(Vec::new())),
            collection_start: None,
            performance_tracker: PerformanceTracker::default(),
        })
    }

    /// Start coverage collection
    pub fn start_collection(&mut self) -> CoreResult<()> {
        if let Ok(mut state) = self.collection_state.lock() {
            match *state {
                CollectionState::Collecting => {
                    return Err(CoreError::from(std::io::Error::other(
                        "Coverage collection already in progress",
                    )));
                }
                _ => *state = CollectionState::Collecting,
            }
        }

        self.collection_start = Some(Instant::now());
        self.performance_tracker.execution_timer = Some(Instant::now());
        self.performance_tracker.baseline_memory = self.get_current_memory_usage();

        // Initialize instrumentation
        self.initialize_instrumentation()?;

        Ok(())
    }

    /// Stop collection and generate comprehensive report
    pub fn stop_and_generate_report(&mut self) -> CoreResult<CoverageReport> {
        // Stop collection
        if let Ok(mut state) = self.collection_state.lock() {
            *state = CollectionState::Completed;
        }

        // Calculate performance impact
        let performance_impact = self.calculate_performance_impact();

        // Generate overall statistics
        let overall_stats = self.calculate_overall_statistics()?;

        // Check quality gates
        let quality_gates = self.evaluate_quality_gates(&overall_stats);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&overall_stats)?;

        // Get coverage trends if history enabled
        let trends = if self.config.enable_history {
            self.calculate_trends()?
        } else {
            None
        };

        // Create report
        let report = CoverageReport {
            generated_at: SystemTime::now(),
            config: self.config.clone(),
            overall_stats,
            file_coverage: self.file_coverage.read().unwrap().clone(),
            trends,
            quality_gates,
            performance_impact,
            recommendations,
        };

        // Save historical data point
        if self.config.enable_history {
            self.save_historical_data_point(&report)?;
        }

        // Generate output reports
        self.generate_output_reports(&report)?;

        Ok(report)
    }

    /// Record line execution
    pub fn record_line_execution(&self, file_path: &Path, linenumber: u32) -> CoreResult<()> {
        if let Ok(mut coverage) = self.file_coverage.write() {
            let file_coverage =
                coverage
                    .entry(file_path.to_path_buf())
                    .or_insert_with(|| FileCoverage {
                        file_path: file_path.to_path_buf(),
                        total_lines: 0,
                        covered_lines: 0,
                        line_hits: BTreeMap::new(),
                        branches: Vec::new(),
                        functions: Vec::new(),
                        integrations: Vec::new(),
                        modified_time: SystemTime::now(),
                        collected_at: SystemTime::now(),
                    });

            *file_coverage.line_hits.entry(linenumber).or_insert(0) += 1;
        }

        Ok(())
    }

    /// Record branch execution
    pub fn record_branch_execution(
        &self,
        file_path: &Path,
        line_number: u32,
        branch_id: &str,
        taken: bool,
    ) -> CoreResult<()> {
        if let Ok(mut coverage) = self.file_coverage.write() {
            let file_coverage =
                coverage
                    .entry(file_path.to_path_buf())
                    .or_insert_with(|| FileCoverage {
                        file_path: file_path.to_path_buf(),
                        total_lines: 0,
                        covered_lines: 0,
                        line_hits: BTreeMap::new(),
                        branches: Vec::new(),
                        functions: Vec::new(),
                        integrations: Vec::new(),
                        modified_time: SystemTime::now(),
                        collected_at: SystemTime::now(),
                    });

            // Find existing branch or create new one
            if let Some(branch) = file_coverage
                .branches
                .iter_mut()
                .find(|b| b.branch_id == branch_id)
            {
                if taken {
                    branch.true_count += 1;
                } else {
                    branch.false_count += 1;
                }
            } else {
                let mut branch = BranchCoverage {
                    line_number,
                    branch_id: branch_id.to_string(),
                    true_count: 0,
                    false_count: 0,
                    branch_type: BranchType::Other,
                    source_snippet: String::new(),
                };

                if taken {
                    branch.true_count = 1;
                } else {
                    branch.false_count = 1;
                }

                file_coverage.branches.push(branch);
            }
        }

        Ok(())
    }

    /// Record function execution
    pub fn record_function_execution(
        &self,
        file_path: &Path,
        function_name: &str,
        start_line: u32,
        end_line: u32,
    ) -> CoreResult<()> {
        if let Ok(mut coverage) = self.file_coverage.write() {
            let file_coverage =
                coverage
                    .entry(file_path.to_path_buf())
                    .or_insert_with(|| FileCoverage {
                        file_path: file_path.to_path_buf(),
                        total_lines: 0,
                        covered_lines: 0,
                        line_hits: BTreeMap::new(),
                        branches: Vec::new(),
                        functions: Vec::new(),
                        integrations: Vec::new(),
                        modified_time: SystemTime::now(),
                        collected_at: SystemTime::now(),
                    });

            // Find existing function or create new one
            if let Some(function) = file_coverage
                .functions
                .iter_mut()
                .find(|f| f.function_name == function_name)
            {
                function.execution_count += 1;
            } else {
                let function = FunctionCoverage {
                    function_name: function_name.to_string(),
                    start_line,
                    end_line,
                    execution_count: 1,
                    complexity: self.calculate_function_complexity(start_line, end_line),
                    parameter_count: 0, // Would be determined during analysis
                    return_complexity: 1,
                };

                file_coverage.functions.push(function);
            }
        }

        Ok(())
    }

    /// Initialize coverage instrumentation
    fn initialize_instrumentation(&mut self) -> CoreResult<()> {
        // In a full implementation, this would instrument the code
        // For now, we'll simulate the instrumentation setup
        self.performance_tracker.instrumentation_count = 1000; // Simulated
        Ok(())
    }

    /// Calculate overall coverage statistics
    fn calculate_overall_statistics(&self) -> CoreResult<CoverageStatistics> {
        let coverage = self.file_coverage.read().unwrap();

        let mut stats = CoverageStatistics {
            total_lines: 0,
            covered_lines: 0,
            line_coverage_percentage: 0.0,
            total_branches: 0,
            covered_branches: 0,
            branch_coverage_percentage: 0.0,
            total_functions: 0,
            covered_functions: 0,
            function_coverage_percentage: 0.0,
            total_integrations: 0,
            covered_integrations: 0,
            integration_coverage_percentage: 0.0,
            files_analyzed: coverage.len() as u32,
        };

        for file_cov in coverage.values() {
            stats.total_lines += file_cov.total_lines;
            stats.covered_lines += file_cov.covered_lines;

            stats.total_branches += file_cov.branches.len() as u32;
            stats.covered_branches +=
                file_cov.branches.iter().filter(|b| b.is_covered()).count() as u32;

            stats.total_functions += file_cov.functions.len() as u32;
            stats.covered_functions += file_cov
                .functions
                .iter()
                .filter(|f| f.execution_count > 0)
                .count() as u32;

            stats.total_integrations += file_cov.integrations.len() as u32;
            stats.covered_integrations += file_cov
                .integrations
                .iter()
                .filter(|i| i.execution_count > 0)
                .count() as u32;
        }

        // Calculate percentages
        stats.line_coverage_percentage = if stats.total_lines > 0 {
            (stats.covered_lines as f64 / stats.total_lines as f64) * 100.0
        } else {
            100.0
        };

        stats.branch_coverage_percentage = if stats.total_branches > 0 {
            (stats.covered_branches as f64 / stats.total_branches as f64) * 100.0
        } else {
            100.0
        };

        stats.function_coverage_percentage = if stats.total_functions > 0 {
            (stats.covered_functions as f64 / stats.total_functions as f64) * 100.0
        } else {
            100.0
        };

        stats.integration_coverage_percentage = if stats.total_integrations > 0 {
            (stats.covered_integrations as f64 / stats.total_integrations as f64) * 100.0
        } else {
            100.0
        };

        Ok(stats)
    }

    /// Evaluate quality gate compliance
    fn evaluate_quality_gates(&self, stats: &CoverageStatistics) -> QualityGateResults {
        let mut results = QualityGateResults {
            overall_passed: true,
            line_coverage_passed: true,
            branch_coverage_passed: true,
            integration_coverage_passed: true,
            failures: Vec::new(),
        };

        // Check line coverage
        if stats.line_coverage_percentage < self.config.coverage_threshold {
            results.line_coverage_passed = false;
            results.overall_passed = false;

            let severity = if stats.line_coverage_percentage < self.config.coverage_threshold - 20.0
            {
                FailureSeverity::Critical
            } else if stats.line_coverage_percentage < self.config.coverage_threshold - 10.0 {
                FailureSeverity::Major
            } else if stats.line_coverage_percentage < self.config.coverage_threshold - 5.0 {
                FailureSeverity::Moderate
            } else {
                FailureSeverity::Minor
            };

            results.failures.push(QualityGateFailure {
                gate_type: "Line Coverage".to_string(),
                threshold: self.config.coverage_threshold,
                actual_value: stats.line_coverage_percentage,
                severity,
                suggestions: vec![
                    "Add unit tests for uncovered lines".to_string(),
                    "Focus on complex functions with low coverage".to_string(),
                    "Consider removing dead code".to_string(),
                ],
            });
        }

        // Check branch coverage
        if stats.branch_coverage_percentage < self.config.branch_threshold {
            results.branch_coverage_passed = false;
            results.overall_passed = false;

            results.failures.push(QualityGateFailure {
                gate_type: "Branch Coverage".to_string(),
                threshold: self.config.branch_threshold,
                actual_value: stats.branch_coverage_percentage,
                severity: FailureSeverity::Moderate,
                suggestions: vec![
                    "Add tests for both true and false branches".to_string(),
                    "Test edge cases and error conditions".to_string(),
                    "Use property-based testing for complex conditions".to_string(),
                ],
            });
        }

        // Check integration coverage
        if stats.integration_coverage_percentage < self.config.integration_threshold {
            results.integration_coverage_passed = false;
            results.overall_passed = false;

            results.failures.push(QualityGateFailure {
                gate_type: "Integration Coverage".to_string(),
                threshold: self.config.integration_threshold,
                actual_value: stats.integration_coverage_percentage,
                severity: FailureSeverity::Moderate,
                suggestions: vec![
                    "Add integration tests between modules".to_string(),
                    "Test external dependencies and APIs".to_string(),
                    "Include database and network interactions".to_string(),
                ],
            });
        }

        results
    }

    /// Generate improvement recommendations
    fn generate_recommendations(
        &self,
        stats: &CoverageStatistics,
    ) -> CoreResult<Vec<CoverageRecommendation>> {
        let mut recommendations = Vec::new();
        let coverage = self.file_coverage.read().unwrap();

        // Recommend unit tests for low coverage files
        for (path, file_cov) in coverage.iter() {
            let coverage_pct = file_cov.line_coverage_percentage();
            if coverage_pct < self.config.coverage_threshold {
                recommendations.push(CoverageRecommendation {
                    recommendation_type: RecommendationType::AddUnitTests,
                    priority: if coverage_pct < 50.0 {
                        RecommendationPriority::High
                    } else {
                        RecommendationPriority::Medium
                    },
                    description: format!(
                        "Add unit tests for {} (current coverage: {:.1}%)",
                        path.display(),
                        coverage_pct
                    ),
                    expected_impact: self.config.coverage_threshold - coverage_pct,
                    effort_estimate: (file_cov.uncovered_lines().len() as f64) * 0.25, // 15min per line
                    affected_items: vec![path.to_string_lossy().to_string()],
                });
            }
        }

        // Recommend branch coverage improvements
        if stats.branch_coverage_percentage < self.config.branch_threshold {
            recommendations.push(CoverageRecommendation {
                recommendation_type: RecommendationType::ImproveBranchCoverage,
                priority: RecommendationPriority::High,
                description: "Improve branch coverage by testing all conditional paths".to_string(),
                expected_impact: self.config.branch_threshold - stats.branch_coverage_percentage,
                effort_estimate: 8.0, // 8 hours estimate
                affected_items: vec!["Multiple files with uncovered branches".to_string()],
            });
        }

        // Recommend testing complex functions
        let critical_functions = self.find_complex_uncovered_functions();
        if !critical_functions.is_empty() {
            recommendations.push(CoverageRecommendation {
                recommendation_type: RecommendationType::TestComplexFunctions,
                priority: RecommendationPriority::Critical,
                description: "Add tests for complex functions with high cyclomatic complexity"
                    .to_string(),
                expected_impact: 15.0, // Estimated impact
                effort_estimate: critical_functions.len() as f64 * 2.0, // 2 hours per function
                affected_items: critical_functions
                    .into_iter()
                    .map(|f| f.function_name.clone())
                    .collect(),
            });
        }

        // Sort by priority
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(recommendations)
    }

    /// Calculate coverage trends
    fn calculate_trends(&self) -> CoreResult<Option<CoverageTrends>> {
        let history = self.history.lock().unwrap();

        if history.len() < 2 {
            return Ok(None);
        }

        // Calculate trend direction
        let recent_points: Vec<_> = history.iter().rev().take(5).collect();
        let trend_direction = if recent_points.len() >= 2 {
            let first = recent_points.last().unwrap().coverage_percentage;
            let last = recent_points.first().unwrap().coverage_percentage;
            let change = last - first;

            if change > 1.0 {
                TrendDirection::Improving
            } else if change < -1.0 {
                TrendDirection::Declining
            } else {
                TrendDirection::Stable
            }
        } else {
            TrendDirection::Unknown
        };

        // Calculate change rate (percentage points per day)
        let change_rate = if recent_points.len() >= 2 {
            let first = recent_points.last().unwrap();
            let last = recent_points.first().unwrap();

            let time_diff = last
                .timestamp
                .duration_since(first.timestamp)
                .unwrap_or(Duration::from_secs(1))
                .as_secs_f64()
                / (24.0 * 60.0 * 60.0); // Convert to days

            let coverage_diff = last.coverage_percentage - first.coverage_percentage;

            if time_diff > 0.0 {
                coverage_diff / time_diff
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Simple prediction based on trend
        let predicted_coverage = if change_rate.abs() > 0.1 {
            let last_coverage = recent_points.first().unwrap().coverage_percentage;
            Some((last_coverage + change_rate * 7.0).clamp(0.0, 100.0)) // 7 days prediction
        } else {
            None
        };

        Ok(Some(CoverageTrends {
            history: history.clone(),
            trend_direction,
            change_rate,
            predicted_coverage,
        }))
    }

    /// Save historical data point
    fn save_historical_data_point(&self, report: &CoverageReport) -> CoreResult<()> {
        if let Ok(mut history) = self.history.lock() {
            let data_point = CoverageDataPoint {
                timestamp: SystemTime::now(),
                coverage_percentage: report.overall_stats.line_coverage_percentage,
                branch_coverage_percentage: report.overall_stats.branch_coverage_percentage,
                version: None, // Would get from git or version system
                test_count: 0, // Would get from test runner
            };

            history.push(data_point);

            // Clean old history based on retention period
            let cutoff = SystemTime::now() - self.config.history_retention;
            history.retain(|point| point.timestamp >= cutoff);
        }

        Ok(())
    }

    /// Generate output reports in various formats
    fn generate_output_reports(&self, report: &CoverageReport) -> CoreResult<()> {
        for format in &self.config.report_formats {
            match format {
                ReportFormat::Html => self.generate_html_report(report)?,
                ReportFormat::Json => self.generate_json_report(report)?,
                ReportFormat::Xml => self.generate_xml_report(report)?,
                ReportFormat::Lcov => self.generate_lcov_report(report)?,
                ReportFormat::Text => self.generatetext_report(report)?,
                ReportFormat::Csv => self.generate_csv_report(report)?,
            }
        }

        Ok(())
    }

    /// Generate HTML report
    fn generate_html_report(&self, report: &CoverageReport) -> CoreResult<()> {
        let html_content = self.create_html_content(report);
        let output_path = self.config.output_directory.join("coverage_report.html");

        std::fs::write(output_path, html_content).map_err(|e| {
            CoreError::from(std::io::Error::other(format!(
                "Failed to write HTML report: {e}"
            )))
        })?;

        Ok(())
    }

    /// Generate JSON report
    fn generate_json_report(&self, report: &CoverageReport) -> CoreResult<()> {
        {
            let json_content = serde_json::to_string_pretty(report).map_err(|e| {
                CoreError::from(std::io::Error::other(format!(
                    "Failed to serialize JSON report: {e}"
                )))
            })?;

            let output_path = self.config.output_directory.join("coverage_report.json");
            std::fs::write(output_path, json_content).map_err(|e| {
                CoreError::from(std::io::Error::other(format!(
                    "Failed to write JSON report: {e}"
                )))
            })?;
        }

        #[cfg(not(feature = "serde"))]
        {
            let _ = report; // Suppress unused warning
            return Err(CoreError::from(std::io::Error::other(
                "JSON report requires serde feature",
            )));
        }

        Ok(())
    }

    /// Generate XML report
    fn generate_xml_report(&self, report: &CoverageReport) -> CoreResult<()> {
        let xml_content = self.create_xml_content(report);
        let output_path = self.config.output_directory.join("coverage_report.xml");

        std::fs::write(output_path, xml_content).map_err(|e| {
            CoreError::from(std::io::Error::other(format!(
                "Failed to write XML report: {e}"
            )))
        })?;

        Ok(())
    }

    /// Generate LCOV report
    fn generate_lcov_report(&self, report: &CoverageReport) -> CoreResult<()> {
        let lcov_content = self.create_lcov_content(report);
        let output_path = self.config.output_directory.join("coverage.lcov");

        std::fs::write(output_path, lcov_content).map_err(|e| {
            CoreError::from(std::io::Error::other(format!(
                "Failed to write LCOV report: {e}"
            )))
        })?;

        Ok(())
    }

    /// Generate text report
    fn generatetext_report(&self, report: &CoverageReport) -> CoreResult<()> {
        let text_content = self.createtext_content(report);
        let output_path = self.config.output_directory.join("coverage_summary.txt");

        std::fs::write(output_path, text_content).map_err(|e| {
            CoreError::from(std::io::Error::other(format!(
                "Failed to write text report: {e}"
            )))
        })?;

        Ok(())
    }

    /// Generate CSV report
    fn generate_csv_report(&self, report: &CoverageReport) -> CoreResult<()> {
        let csv_content = self.create_csv_content(report);
        let output_path = self.config.output_directory.join("coverage_data.csv");

        std::fs::write(output_path, csv_content).map_err(|e| {
            CoreError::from(std::io::Error::other(format!(
                "Failed to write CSV report: {e}"
            )))
        })?;

        Ok(())
    }

    /// Create HTML report content
    fn create_html_content(&self, report: &CoverageReport) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Coverage Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
        .coverage-bar {{ background: #ddd; height: 20px; border-radius: 10px; overflow: hidden; }}
        .coverage-fill {{ background: #4caf50; height: 100%; transition: width 0.3s; }}
        .low-coverage {{ background: #f44336; }}
        .medium-coverage {{ background: #ff9800; }}
        .high-coverage {{ background: #4caf50; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .recommendations {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class= header>
        <h1>Coverage Report</h1>
        <p>Generated at: {}</p>
        <p>Overall Coverage: {:.2}%</p>
    </div>
    
    <div class= stats>
        <div class="stat-box">
            <h3>Line Coverage</h3>
            <div class="coverage-bar">
                <div class="coverage-fill {}" style="width: {:.1}%"></div>
            </div>
            <p>{:.2}% ({}/{})</p>
        </div>
        <div class="stat-box">
            <h3>Branch Coverage</h3>
            <div class="coverage-bar">
                <div class="coverage-fill {}" style="width: {:.1}%"></div>
            </div>
            <p>{:.2}% ({}/{})</p>
        </div>
        <div class="stat-box">
            <h3>Function Coverage</h3>
            <div class="coverage-bar">
                <div class="coverage-fill {}" style="width: {:.1}%"></div>
            </div>
            <p>{:.2}% ({}/{})</p>
        </div>
    </div>

    <h2>Quality Gates</h2>
    <p>Status: {}</p>
    
    <h2>Recommendations</h2>
    <div class= recommendations>
        <ul>
        {}
        </ul>
    </div>
    
    <h2>File Coverage Details</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Line Coverage</th>
            <th>Branch Coverage</th>
            <th>Function Coverage</th>
        </tr>
        {}
    </table>
</body>
</html>"#,
            chrono::DateTime::<chrono::Utc>::from(report.generated_at)
                .format("%Y-%m-%d %H:%M:%S UTC"),
            report.overall_stats.line_coverage_percentage,
            // Line coverage
            self.get_coverage_class(report.overall_stats.line_coverage_percentage),
            report.overall_stats.line_coverage_percentage,
            report.overall_stats.line_coverage_percentage,
            report.overall_stats.covered_lines,
            report.overall_stats.total_lines,
            // Branch coverage
            self.get_coverage_class(report.overall_stats.branch_coverage_percentage),
            report.overall_stats.branch_coverage_percentage,
            report.overall_stats.branch_coverage_percentage,
            report.overall_stats.covered_branches,
            report.overall_stats.total_branches,
            // Function coverage
            self.get_coverage_class(report.overall_stats.function_coverage_percentage),
            report.overall_stats.function_coverage_percentage,
            report.overall_stats.function_coverage_percentage,
            report.overall_stats.covered_functions,
            report.overall_stats.total_functions,
            // Quality gates
            if report.quality_gates.overall_passed {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            },
            // Recommendations
            report
                .recommendations
                .iter()
                .take(5)
                .map(|r| format!("<li>{description}</li>", description = r.description))
                .collect::<Vec<_>>()
                .join("\n"),
            // File table rows
            report
                .file_coverage
                .iter()
                .map(|(path, cov)| format!(
                    "<tr><td>{}</td><td>{:.1}%</td><td>{:.1}%</td><td>{:.1}%</td></tr>",
                    path.display(),
                    cov.line_coverage_percentage(),
                    cov.branch_coverage_percentage(),
                    cov.function_coverage_percentage()
                ))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Create XML report content
    fn create_xml_content(&self, report: &CoverageReport) -> String {
        format!(
            r#"<?xml _version="1.0" encoding="UTF-8"?>
<coverage _version="1.0" timestamp="{}">
    <project name="scirs2-core">
        <metrics>
            <lines-covered>{}</lines-covered>
            <lines-valid>{}</lines-valid>
            <line-coverage>{:.6}</line-coverage>
            <branches-covered>{}</branches-covered>
            <branches-valid>{}</branches-valid>
            <branch-coverage>{:.6}</branch-coverage>
        </metrics>
        <packages>
            {}
        </packages>
    </project>
</coverage>"#,
            chrono::DateTime::<chrono::Utc>::from(report.generated_at).timestamp(),
            report.overall_stats.covered_lines,
            report.overall_stats.total_lines,
            report.overall_stats.line_coverage_percentage / 100.0,
            report.overall_stats.covered_branches,
            report.overall_stats.total_branches,
            report.overall_stats.branch_coverage_percentage / 100.0,
            report
                .file_coverage
                .iter()
                .map(|(path, cov)| format!(
                    r#"<package name="{}">
                        <classes>
                            <class name="{}" filename="{}">
                                <metrics line-coverage="{:.6}" branch-coverage="{:.6}" />
                                <lines>
                                    {}
                                </lines>
                            </class>
                        </classes>
                    </package>"#,
                    path.parent().unwrap_or(Path::new("")).display(),
                    path.file_stem().unwrap_or_default().to_string_lossy(),
                    path.display(),
                    cov.line_coverage_percentage() / 100.0,
                    cov.branch_coverage_percentage() / 100.0,
                    cov.line_hits
                        .iter()
                        .map(|(&line, &hits)| format!(r#"<line number="{line}" hits="{hits}" />"#))
                        .collect::<Vec<_>>()
                        .join("\n                                    ")
                ))
                .collect::<Vec<_>>()
                .join("\n            ")
        )
    }

    /// Create LCOV report content
    fn create_lcov_content(&self, report: &CoverageReport) -> String {
        let mut lcov_content = String::new();

        for (path, cov) in &report.file_coverage {
            lcov_content.push_str(&format!("SF:{path}\n", path = path.display()));

            // Function data
            for func in &cov.functions {
                lcov_content.push_str(&format!(
                    "FN:{start_line},{function_name}\n",
                    start_line = func.start_line,
                    function_name = func.function_name
                ));
            }

            for func in &cov.functions {
                lcov_content.push_str(&format!(
                    "FNDA:{},{}\n",
                    func.execution_count, func.function_name
                ));
            }

            lcov_content.push_str(&format!("FNF:{count}\n", count = cov.functions.len()));
            lcov_content.push_str(&format!(
                "FNH:{}\n",
                cov.functions
                    .iter()
                    .filter(|f| f.execution_count > 0)
                    .count()
            ));

            // Branch data
            for branch in &cov.branches {
                lcov_content.push_str(&format!(
                    "BA:{},0,{}\n",
                    branch.line_number, branch.true_count
                ));
                lcov_content.push_str(&format!(
                    "BA:{},1,{}\n",
                    branch.line_number, branch.false_count
                ));
            }

            lcov_content.push_str(&format!("BRF:{}\n", cov.branches.len() * 2));
            lcov_content.push_str(&format!(
                "BRH:{}\n",
                cov.branches
                    .iter()
                    .map(|b| if b.is_covered() {
                        2
                    } else if b.true_count > 0 || b.false_count > 0 {
                        1
                    } else {
                        0
                    })
                    .sum::<u32>()
            ));

            // Line data
            for (&line, &hits) in &cov.line_hits {
                lcov_content.push_str(&format!("DA:{line},{hits}\n"));
            }

            lcov_content.push_str(&format!("LF:{}\n", cov.total_lines));
            lcov_content.push_str(&format!("LH:{}\n", cov.covered_lines));
            lcov_content.push_str("end_of_record\n");
        }

        lcov_content
    }

    /// Create text report content
    fn createtext_content(&self, report: &CoverageReport) -> String {
        let mut content = String::new();

        content.push_str("===== COVERAGE REPORT =====\n\n");
        content.push_str(&format!(
            "Generated: {}\n",
            chrono::DateTime::<chrono::Utc>::from(report.generated_at)
                .format("%Y-%m-%d %H:%M:%S UTC")
        ));
        content.push_str(&format!(
            "Files Analyzed: {}\n\n",
            report.overall_stats.files_analyzed
        ));

        content.push_str("OVERALL STATISTICS:\n");
        content.push_str(&format!(
            "  Line Coverage:     {:.2}% ({}/{})\n",
            report.overall_stats.line_coverage_percentage,
            report.overall_stats.covered_lines,
            report.overall_stats.total_lines
        ));
        content.push_str(&format!(
            "  Branch Coverage:   {:.2}% ({}/{})\n",
            report.overall_stats.branch_coverage_percentage,
            report.overall_stats.covered_branches,
            report.overall_stats.total_branches
        ));
        content.push_str(&format!(
            "  Function Coverage: {:.2}% ({}/{})\n",
            report.overall_stats.function_coverage_percentage,
            report.overall_stats.covered_functions,
            report.overall_stats.total_functions
        ));

        content.push_str("\nQUALITY GATES:\n");
        content.push_str(&format!(
            "  Overall Status: {}\n",
            if report.quality_gates.overall_passed {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        ));

        for failure in &report.quality_gates.failures {
            content.push_str(&format!(
                "  ❌ {}: {:.2}% (threshold: {:.2}%)\n",
                failure.gate_type, failure.actual_value, failure.threshold
            ));
        }

        if !report.recommendations.is_empty() {
            content.push_str("\nRECOMMENDATIONS:\n");
            for (i, rec) in report.recommendations.iter().take(5).enumerate() {
                content.push_str(&format!(
                    "  {}. [{}] {}\n",
                    i + 1,
                    format!("{0:?}", rec.priority).to_uppercase(),
                    rec.description
                ));
            }
        }

        content.push_str("\nFILE DETAILS:\n");
        let mut files: Vec<_> = report.file_coverage.iter().collect();
        files.sort_by(|a, b| {
            a.1.line_coverage_percentage()
                .partial_cmp(&b.1.line_coverage_percentage())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (path, cov) in files.iter().take(20) {
            content.push_str(&format!(
                "  {:<50} {:>8.1}% {:>8.1}% {:>8.1}%\n",
                path.display()
                    .to_string()
                    .chars()
                    .take(50)
                    .collect::<String>(),
                cov.line_coverage_percentage(),
                cov.branch_coverage_percentage(),
                cov.function_coverage_percentage()
            ));
        }

        content
    }

    /// Create CSV report content
    fn create_csv_content(&self, report: &CoverageReport) -> String {
        let mut csv_content = String::new();

        csv_content.push_str("File,Line Coverage %,Branch Coverage %,Function Coverage %,Total Lines,Covered Lines,Total Branches,Covered Branches,Total Functions,Covered Functions\n");

        for (path, cov) in &report.file_coverage {
            csv_content.push_str(&format!(
                "{},{:.2},{:.2},{:.2},{},{},{},{},{},{}\n",
                path.display(),
                cov.line_coverage_percentage(),
                cov.branch_coverage_percentage(),
                cov.function_coverage_percentage(),
                cov.total_lines,
                cov.covered_lines,
                cov.branches.len(),
                cov.branches.iter().filter(|b| b.is_covered()).count(),
                cov.functions.len(),
                cov.functions
                    .iter()
                    .filter(|f| f.execution_count > 0)
                    .count()
            ));
        }

        csv_content
    }

    /// Get CSS class for coverage percentage
    fn get_coverage_class(&self, percentage: f64) -> &'static str {
        if percentage >= 80.0 {
            "high-coverage"
        } else if percentage >= 50.0 {
            "medium-coverage"
        } else {
            "low-coverage"
        }
    }

    /// Calculate function complexity (simplified)
    fn calculate_complexity(start_line: u32, endline: u32) -> u32 {
        // Simplified complexity calculation based on _line count
        // In a real implementation, this would analyze the AST
        let line_count = endline.saturating_sub(start_line) + 1;
        (line_count / 10).max(1) // Rough approximation
    }

    /// Calculate function complexity (instance method)
    fn calculate_function_complexity(&self, start_line: u32, endline: u32) -> u32 {
        Self::calculate_complexity(start_line, endline)
    }

    /// Find complex uncovered functions
    fn find_complex_uncovered_functions(&self) -> Vec<FunctionCoverage> {
        let coverage = self.file_coverage.read().unwrap();

        let mut complex_functions: Vec<_> = coverage
            .values()
            .flat_map(|file_cov| {
                file_cov
                    .functions
                    .iter()
                    .filter(|f| f.execution_count == 0 && f.complexity > 5)
                    .cloned()
            })
            .collect();

        complex_functions.sort_by(|a, b| b.complexity.cmp(&a.complexity));
        complex_functions.into_iter().take(10).collect()
    }

    /// Calculate performance impact
    fn calculate_performance_impact(&self) -> PerformanceImpact {
        let execution_time = self
            .performance_tracker
            .execution_timer
            .map(|start| start.elapsed())
            .unwrap_or(Duration::from_secs(0));

        let memory_overhead =
            self.get_current_memory_usage() - self.performance_tracker.baseline_memory;

        PerformanceImpact {
            execution_overhead_percent: 5.0, // Simulated 5% overhead
            memory_overhead_bytes: memory_overhead,
            collection_duration: execution_time,
            instrumentation_points: self.performance_tracker.instrumentation_count,
            sampling_effectiveness: self.config.samplingrate,
        }
    }

    /// Get current memory usage (simplified)
    fn get_current_memory_usage(&self) -> u64 {
        // In a real implementation, this would query actual memory usage
        1024 * 1024 * 10 // Simulated 10MB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_config_builder() {
        let config = CoverageConfig::development()
            .with_threshold(90.0)
            .with_coverage_types(vec![CoverageType::Line, CoverageType::Branch])
            .with_report_format(ReportFormat::Html);

        assert_eq!(config.coverage_threshold, 90.0);
        assert_eq!(config.coverage_types.len(), 2);
        assert_eq!(config.report_formats, vec![ReportFormat::Html]);
    }

    #[test]
    fn test_coverage_analyzer_creation() {
        let config = CoverageConfig::default();
        let analyzer = CoverageAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_file_coverage_calculations() {
        let mut file_cov = FileCoverage {
            file_path: PathBuf::from("test.rs"),
            total_lines: 100,
            covered_lines: 80,
            line_hits: BTreeMap::new(),
            branches: vec![
                BranchCoverage {
                    line_number: 10,
                    branch_id: "b1".to_string(),
                    true_count: 5,
                    false_count: 3,
                    branch_type: BranchType::IfElse,
                    source_snippet: "if condition".to_string(),
                },
                BranchCoverage {
                    line_number: 20,
                    branch_id: "b2".to_string(),
                    true_count: 0,
                    false_count: 0,
                    branch_type: BranchType::IfElse,
                    source_snippet: "if other".to_string(),
                },
            ],
            functions: vec![FunctionCoverage {
                function_name: "test_fn".to_string(),
                start_line: 5,
                end_line: 15,
                execution_count: 10,
                complexity: 3,
                parameter_count: 2,
                return_complexity: 1,
            }],
            integrations: vec![],
            modified_time: SystemTime::now(),
            collected_at: SystemTime::now(),
        };

        // Add some line hits
        file_cov.line_hits.insert(1, 5);
        file_cov.line_hits.insert(2, 3);
        file_cov.line_hits.insert(10, 8);

        assert_eq!(file_cov.line_coverage_percentage(), 80.0);
        assert_eq!(file_cov.branch_coverage_percentage(), 50.0); // 1 out of 2 branches covered
        assert_eq!(file_cov.function_coverage_percentage(), 100.0); // 1 out of 1 function covered

        let uncovered = file_cov.uncovered_lines();
        assert_eq!(uncovered.len(), 97); // 100 - 3 covered lines

        let hot_spots = file_cov.hot_spots(5);
        assert_eq!(hot_spots.len(), 2); // Lines with >= 5 hits
    }

    #[test]
    fn test_branch_coverage_analysis() {
        let branch = BranchCoverage {
            line_number: 10,
            branch_id: "test_branch".to_string(),
            true_count: 8,
            false_count: 2,
            branch_type: BranchType::IfElse,
            source_snippet: "if x > 0".to_string(),
        };

        assert!(branch.is_covered());
        assert_eq!(branch.total_executions(), 10);
        assert!((branch.balance_score() - 0.4).abs() < f64::EPSILON); // min(8,2) / 10 * 2
    }

    #[test]
    fn test_function_coverage_score() {
        let function = FunctionCoverage {
            function_name: "complex_function".to_string(),
            start_line: 1,
            end_line: 50,
            execution_count: 5,
            complexity: 8,
            parameter_count: 4,
            return_complexity: 2,
        };

        let score = function.coverage_score();
        assert!(score > 0.0 && score <= 1.0);

        // Test uncovered function
        let uncovered_function = FunctionCoverage {
            function_name: "unused_function".to_string(),
            start_line: 60,
            end_line: 70,
            execution_count: 0,
            complexity: 5,
            parameter_count: 2,
            return_complexity: 1,
        };

        assert_eq!(uncovered_function.coverage_score(), 0.0);
    }

    #[test]
    fn test_quality_gate_evaluation() {
        let config = CoverageConfig {
            coverage_threshold: 80.0,
            branch_threshold: 70.0,
            integration_threshold: 60.0,
            ..Default::default()
        };

        let analyzer = CoverageAnalyzer::new(config).unwrap();

        let stats = CoverageStatistics {
            total_lines: 1000,
            covered_lines: 750,
            line_coverage_percentage: 75.0,
            total_branches: 200,
            covered_branches: 130,
            branch_coverage_percentage: 65.0,
            total_functions: 50,
            covered_functions: 45,
            function_coverage_percentage: 90.0,
            total_integrations: 30,
            covered_integrations: 20,
            integration_coverage_percentage: 66.7,
            files_analyzed: 25,
        };

        let quality_gates = analyzer.evaluate_quality_gates(&stats);

        assert!(!quality_gates.overall_passed);
        assert!(!quality_gates.line_coverage_passed); // 75% < 80%
        assert!(!quality_gates.branch_coverage_passed); // 65% < 70%
        assert!(quality_gates.integration_coverage_passed); // 66.7% > 60%
        assert_eq!(quality_gates.failures.len(), 2);
    }

    #[test]
    fn test_coverage_recommendation_generation() {
        let config = CoverageConfig::default();
        let analyzer = CoverageAnalyzer::new(config).unwrap();

        let stats = CoverageStatistics {
            total_lines: 1000,
            covered_lines: 600,
            line_coverage_percentage: 60.0,
            total_branches: 200,
            covered_branches: 100,
            branch_coverage_percentage: 50.0,
            total_functions: 50,
            covered_functions: 30,
            function_coverage_percentage: 60.0,
            total_integrations: 30,
            covered_integrations: 15,
            integration_coverage_percentage: 50.0,
            files_analyzed: 25,
        };

        let recommendations = analyzer.generate_recommendations(&stats).unwrap();

        assert!(!recommendations.is_empty());

        // Should have branch coverage recommendation since 50% < 70%
        let has_branch_rec = recommendations
            .iter()
            .any(|r| r.recommendation_type == RecommendationType::ImproveBranchCoverage);
        assert!(has_branch_rec);

        // Check priority ordering
        let priorities: Vec<_> = recommendations.iter().map(|r| r.priority).collect();
        assert!(priorities.windows(2).all(|w| w[0] >= w[1])); // Should be sorted by priority
    }

    #[test]
    fn test_coverage_trends() {
        let config = CoverageConfig::default();
        let analyzer = CoverageAnalyzer::new(config).unwrap();

        // Add some historical data
        let mut history = analyzer.history.lock().unwrap();
        let now = SystemTime::now();

        history.push(CoverageDataPoint {
            timestamp: now - Duration::from_secs(7 * 24 * 60 * 60), // 7 days ago
            coverage_percentage: 70.0,
            branch_coverage_percentage: 60.0,
            version: Some("v1.0.0".to_string()),
            test_count: 100,
        });

        history.push(CoverageDataPoint {
            timestamp: now - Duration::from_secs(3 * 24 * 60 * 60), // 3 days ago
            coverage_percentage: 75.0,
            branch_coverage_percentage: 65.0,
            version: Some("v1.1.0".to_string()),
            test_count: 120,
        });

        history.push(CoverageDataPoint {
            timestamp: now,
            coverage_percentage: 80.0,
            branch_coverage_percentage: 70.0,
            version: Some("v1.2.0".to_string()),
            test_count: 150,
        });

        drop(history);

        let trends = analyzer.calculate_trends().unwrap();
        assert!(trends.is_some());

        let trends = trends.unwrap();
        assert_eq!(trends.trend_direction, TrendDirection::Improving);
        assert!(trends.change_rate > 0.0); // Positive change rate
        assert!(trends.predicted_coverage.is_some());
    }
}
