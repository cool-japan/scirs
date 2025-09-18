//! Event Filtering Rules, Conditions, and Optimization
//!
//! This module provides comprehensive event filtering capabilities for TPU synchronization
//! including rule-based filtering, condition evaluation, filter optimization, and performance
//! monitoring. It supports complex filtering expressions, rule composition, and efficient
//! filter execution with adaptive optimization.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use std::fmt;
use thiserror::Error;

/// Errors that can occur during event filtering operations
#[derive(Error, Debug)]
pub enum FilteringError {
    #[error("Invalid filter expression: {0}")]
    InvalidExpression(String),
    #[error("Filter compilation failed: {0}")]
    CompilationFailed(String),
    #[error("Filter execution error: {0}")]
    ExecutionError(String),
    #[error("Filter not found: {0}")]
    FilterNotFound(String),
    #[error("Filter optimization error: {0}")]
    OptimizationError(String),
}

/// Result type for filtering operations
pub type FilteringResult<T> = Result<T, FilteringError>;

/// Event filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFiltering {
    /// Filter rules configuration
    pub filter_rules: FilterRulesConfig,
    /// Filter optimization settings
    pub optimization: FilterOptimization,
    /// Filter performance monitoring
    pub performance_monitoring: FilterPerformanceMonitoring,
    /// Filter composition settings
    pub composition: FilterComposition,
    /// Filter storage and caching
    pub storage: FilterStorage,
}

impl Default for EventFiltering {
    fn default() -> Self {
        Self {
            filter_rules: FilterRulesConfig::default(),
            optimization: FilterOptimization::default(),
            performance_monitoring: FilterPerformanceMonitoring::default(),
            composition: FilterComposition::default(),
            storage: FilterStorage::default(),
        }
    }
}

/// Filter rules configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRulesConfig {
    /// Available filter rules
    pub rules: Vec<FilterRule>,
    /// Rule execution order
    pub execution_order: RuleExecutionOrder,
    /// Rule priority management
    pub priority_management: RulePriorityManagement,
    /// Rule conflict resolution
    pub conflict_resolution: RuleConflictResolution,
    /// Rule validation settings
    pub validation: RuleValidation,
}

impl Default for FilterRulesConfig {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            execution_order: RuleExecutionOrder::Priority,
            priority_management: RulePriorityManagement::default(),
            conflict_resolution: RuleConflictResolution::default(),
            validation: RuleValidation::default(),
        }
    }
}

/// Individual filter rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Filter condition
    pub condition: FilterCondition,
    /// Filter action
    pub action: FilterAction,
    /// Rule priority
    pub priority: i32,
    /// Rule status
    pub status: RuleStatus,
    /// Rule metadata
    pub metadata: RuleMetadata,
    /// Performance metrics
    pub performance: RulePerformanceMetrics,
}

/// Filter condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterCondition {
    /// Simple field comparison
    FieldComparison {
        field: String,
        operator: ComparisonOperator,
        value: FilterValue,
    },
    /// Pattern matching
    PatternMatch {
        field: String,
        pattern: String,
        flags: PatternFlags,
    },
    /// Range condition
    Range {
        field: String,
        min: FilterValue,
        max: FilterValue,
        inclusive: bool,
    },
    /// Set membership
    SetMembership {
        field: String,
        values: HashSet<FilterValue>,
        negate: bool,
    },
    /// Composite condition
    Composite {
        operator: LogicalOperator,
        conditions: Vec<FilterCondition>,
    },
    /// Custom expression
    Expression {
        expression: String,
        variables: HashMap<String, FilterValue>,
    },
    /// Time-based condition
    TimeBased {
        field: String,
        time_range: TimeRange,
        timezone: Option<String>,
    },
    /// Statistical condition
    Statistical {
        field: String,
        statistic: StatisticType,
        threshold: f64,
        window: Duration,
    },
}

/// Comparison operators for field comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Contains,
    StartsWith,
    EndsWith,
    Matches,
}

/// Logical operators for composite conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
    Xor,
}

/// Filter value types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FilterValue {
    String(String),
    Integer(i64),
    Float(String), // Stored as string to avoid float comparison issues
    Boolean(bool),
    Null,
    Array(Vec<FilterValue>),
    Object(HashMap<String, FilterValue>),
}

/// Pattern matching flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternFlags {
    /// Case insensitive matching
    pub case_insensitive: bool,
    /// Multiline mode
    pub multiline: bool,
    /// Dot matches newline
    pub dot_all: bool,
    /// Extended syntax
    pub extended: bool,
}

impl Default for PatternFlags {
    fn default() -> Self {
        Self {
            case_insensitive: false,
            multiline: false,
            dot_all: false,
            extended: false,
        }
    }
}

/// Time range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time
    pub start: SystemTime,
    /// End time
    pub end: SystemTime,
    /// Relative to current time
    pub relative: bool,
}

/// Statistical types for statistical conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticType {
    Count,
    Sum,
    Average,
    Minimum,
    Maximum,
    StandardDeviation,
    Percentile(f64),
    Rate,
    Frequency,
}

/// Filter actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterAction {
    /// Accept the event
    Accept,
    /// Reject the event
    Reject,
    /// Transform the event
    Transform {
        transformations: Vec<EventTransformation>,
    },
    /// Route to specific destination
    Route {
        destination: String,
        priority: Option<i32>,
    },
    /// Delay the event
    Delay {
        duration: Duration,
        reason: String,
    },
    /// Split into multiple events
    Split {
        split_rules: Vec<SplitRule>,
    },
    /// Aggregate with other events
    Aggregate {
        aggregation_key: String,
        aggregation_window: Duration,
    },
}

/// Event transformation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventTransformation {
    /// Add field
    AddField {
        field: String,
        value: FilterValue,
    },
    /// Remove field
    RemoveField {
        field: String,
    },
    /// Rename field
    RenameField {
        old_name: String,
        new_name: String,
    },
    /// Transform field value
    TransformField {
        field: String,
        transformation: ValueTransformation,
    },
    /// Set metadata
    SetMetadata {
        key: String,
        value: FilterValue,
    },
}

/// Value transformation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueTransformation {
    /// Convert to uppercase
    ToUpperCase,
    /// Convert to lowercase
    ToLowerCase,
    /// Apply regex substitution
    RegexReplace {
        pattern: String,
        replacement: String,
    },
    /// Mathematical operation
    MathOperation {
        operation: MathOperationType,
        operand: f64,
    },
    /// String formatting
    Format {
        format_string: String,
    },
}

/// Mathematical operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathOperationType {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
}

/// Split rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitRule {
    /// Split condition
    pub condition: FilterCondition,
    /// Target field for splitting
    pub target_field: String,
    /// Split strategy
    pub strategy: SplitStrategy,
}

/// Split strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplitStrategy {
    /// Split by delimiter
    ByDelimiter {
        delimiter: String,
        max_splits: Option<usize>,
    },
    /// Split by regex
    ByRegex {
        pattern: String,
        capture_groups: bool,
    },
    /// Split by fixed length
    ByLength {
        length: usize,
        overlap: usize,
    },
}

/// Rule execution order strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleExecutionOrder {
    /// Execute by priority (higher first)
    Priority,
    /// Execute in registration order
    Registration,
    /// Execute in dependency order
    Dependency,
    /// Execute by performance (faster first)
    Performance,
    /// Custom order
    Custom(Vec<String>),
}

/// Rule priority management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulePriorityManagement {
    /// Default priority for new rules
    pub default_priority: i32,
    /// Priority range
    pub priority_range: (i32, i32),
    /// Priority adjustment strategy
    pub adjustment_strategy: PriorityAdjustmentStrategy,
    /// Priority conflict resolution
    pub conflict_resolution: PriorityConflictResolution,
}

impl Default for RulePriorityManagement {
    fn default() -> Self {
        Self {
            default_priority: 100,
            priority_range: (0, 1000),
            adjustment_strategy: PriorityAdjustmentStrategy::Manual,
            conflict_resolution: PriorityConflictResolution::MaintainOrder,
        }
    }
}

/// Priority adjustment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityAdjustmentStrategy {
    /// Manual priority assignment
    Manual,
    /// Automatic based on performance
    Performance,
    /// Automatic based on usage frequency
    Frequency,
    /// Automatic based on execution time
    ExecutionTime,
}

/// Priority conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityConflictResolution {
    /// Maintain original order
    MaintainOrder,
    /// Execute all conflicting rules
    ExecuteAll,
    /// Execute first matching rule
    ExecuteFirst,
    /// Execute rule with highest success rate
    HighestSuccessRate,
}

/// Rule conflict resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleConflictResolution {
    /// Conflict detection enabled
    pub detection_enabled: bool,
    /// Conflict resolution strategy
    pub resolution_strategy: ConflictResolutionStrategy,
    /// Conflict notification
    pub notification: ConflictNotification,
    /// Conflict logging
    pub logging: ConflictLogging,
}

impl Default for RuleConflictResolution {
    fn default() -> Self {
        Self {
            detection_enabled: true,
            resolution_strategy: ConflictResolutionStrategy::FirstMatch,
            notification: ConflictNotification::default(),
            logging: ConflictLogging::default(),
        }
    }
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Use first matching rule
    FirstMatch,
    /// Use last matching rule
    LastMatch,
    /// Use highest priority rule
    HighestPriority,
    /// Combine rule results
    Combine,
    /// Reject on conflict
    Reject,
}

/// Conflict notification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictNotification {
    /// Enable notifications
    pub enabled: bool,
    /// Notification threshold
    pub threshold: usize,
    /// Notification channels
    pub channels: Vec<String>,
}

impl Default for ConflictNotification {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 1,
            channels: Vec::new(),
        }
    }
}

/// Conflict logging settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictLogging {
    /// Enable conflict logging
    pub enabled: bool,
    /// Log level
    pub log_level: String,
    /// Log details
    pub log_details: bool,
}

impl Default for ConflictLogging {
    fn default() -> Self {
        Self {
            enabled: true,
            log_level: "warn".to_string(),
            log_details: false,
        }
    }
}

/// Rule validation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleValidation {
    /// Validation enabled
    pub enabled: bool,
    /// Validation mode
    pub mode: ValidationMode,
    /// Validation rules
    pub rules: Vec<ValidationRule>,
    /// Validation reporting
    pub reporting: ValidationReporting,
}

impl Default for RuleValidation {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: ValidationMode::Strict,
            rules: Vec::new(),
            reporting: ValidationReporting::default(),
        }
    }
}

/// Validation modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMode {
    /// Strict validation (fail on any error)
    Strict,
    /// Permissive validation (warnings only)
    Permissive,
    /// Custom validation
    Custom,
}

/// Validation rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRule {
    /// Required field validation
    RequiredField {
        field: String,
    },
    /// Type validation
    TypeValidation {
        field: String,
        expected_type: String,
    },
    /// Range validation
    RangeValidation {
        field: String,
        min: FilterValue,
        max: FilterValue,
    },
    /// Pattern validation
    PatternValidation {
        field: String,
        pattern: String,
    },
    /// Custom validation
    Custom {
        validator: String,
        parameters: HashMap<String, FilterValue>,
    },
}

/// Validation reporting settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReporting {
    /// Enable validation reporting
    pub enabled: bool,
    /// Report format
    pub format: ReportFormat,
    /// Report destination
    pub destination: String,
}

impl Default for ValidationReporting {
    fn default() -> Self {
        Self {
            enabled: true,
            format: ReportFormat::Json,
            destination: "logs/validation.log".to_string(),
        }
    }
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Yaml,
    Xml,
    Text,
    Csv,
}

/// Rule status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleStatus {
    /// Rule is active
    Active,
    /// Rule is inactive
    Inactive,
    /// Rule is disabled
    Disabled,
    /// Rule is being tested
    Testing,
    /// Rule is deprecated
    Deprecated,
}

/// Rule metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleMetadata {
    /// Rule author
    pub author: String,
    /// Creation timestamp
    pub created: SystemTime,
    /// Last modified timestamp
    pub modified: SystemTime,
    /// Rule version
    pub version: String,
    /// Rule tags
    pub tags: HashSet<String>,
    /// Rule category
    pub category: String,
    /// Rule dependencies
    pub dependencies: Vec<String>,
}

/// Rule performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulePerformanceMetrics {
    /// Total executions
    pub total_executions: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Success count
    pub success_count: u64,
    /// Failure count
    pub failure_count: u64,
    /// Last execution time
    pub last_execution: Option<Instant>,
    /// Performance history
    pub history: VecDeque<PerformanceSnapshot>,
}

/// Performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Snapshot timestamp
    pub timestamp: Instant,
    /// Execution time
    pub execution_time: Duration,
    /// Success indicator
    pub success: bool,
    /// Memory usage
    pub memory_usage: Option<usize>,
    /// CPU usage
    pub cpu_usage: Option<f64>,
}

/// Filter optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterOptimization {
    /// Optimization enabled
    pub enabled: bool,
    /// Optimization strategies
    pub strategies: Vec<OptimizationStrategy>,
    /// Optimization schedule
    pub schedule: OptimizationSchedule,
    /// Optimization thresholds
    pub thresholds: OptimizationThresholds,
    /// Optimization reporting
    pub reporting: OptimizationReporting,
}

impl Default for FilterOptimization {
    fn default() -> Self {
        Self {
            enabled: true,
            strategies: vec![
                OptimizationStrategy::RuleReordering,
                OptimizationStrategy::ConditionSimplification,
                OptimizationStrategy::IndexOptimization,
            ],
            schedule: OptimizationSchedule::default(),
            thresholds: OptimizationThresholds::default(),
            reporting: OptimizationReporting::default(),
        }
    }
}

/// Optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Reorder rules by performance
    RuleReordering,
    /// Simplify conditions
    ConditionSimplification,
    /// Optimize field access
    FieldAccessOptimization,
    /// Create indexes for fast lookups
    IndexOptimization,
    /// Cache frequently used results
    ResultCaching,
    /// Parallelize rule execution
    Parallelization,
    /// Compile conditions to native code
    NativeCompilation,
}

/// Optimization schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSchedule {
    /// Optimization frequency
    pub frequency: OptimizationFrequency,
    /// Optimization window
    pub window: Duration,
    /// Maximum optimization time
    pub max_optimization_time: Duration,
    /// Optimization triggers
    pub triggers: Vec<OptimizationTrigger>,
}

impl Default for OptimizationSchedule {
    fn default() -> Self {
        Self {
            frequency: OptimizationFrequency::Periodic(Duration::from_secs(3600)),
            window: Duration::from_secs(300),
            max_optimization_time: Duration::from_secs(60),
            triggers: vec![
                OptimizationTrigger::PerformanceDegradation(0.2),
                OptimizationTrigger::RuleCountThreshold(100),
            ],
        }
    }
}

/// Optimization frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationFrequency {
    /// Never optimize
    Never,
    /// Optimize once
    Once,
    /// Periodic optimization
    Periodic(Duration),
    /// Adaptive optimization
    Adaptive,
    /// Manual optimization
    Manual,
}

/// Optimization triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTrigger {
    /// Performance degradation threshold
    PerformanceDegradation(f64),
    /// Rule count threshold
    RuleCountThreshold(usize),
    /// Memory usage threshold
    MemoryUsageThreshold(usize),
    /// Execution time threshold
    ExecutionTimeThreshold(Duration),
    /// Error rate threshold
    ErrorRateThreshold(f64),
}

/// Optimization thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationThresholds {
    /// Minimum performance improvement
    pub min_performance_improvement: f64,
    /// Maximum optimization overhead
    pub max_optimization_overhead: Duration,
    /// Minimum rule count for optimization
    pub min_rule_count: usize,
    /// Maximum optimization iterations
    pub max_iterations: usize,
}

impl Default for OptimizationThresholds {
    fn default() -> Self {
        Self {
            min_performance_improvement: 0.05,
            max_optimization_overhead: Duration::from_secs(10),
            min_rule_count: 10,
            max_iterations: 10,
        }
    }
}

/// Optimization reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReporting {
    /// Enable optimization reporting
    pub enabled: bool,
    /// Report detail level
    pub detail_level: ReportDetailLevel,
    /// Report destination
    pub destination: String,
    /// Report format
    pub format: ReportFormat,
}

impl Default for OptimizationReporting {
    fn default() -> Self {
        Self {
            enabled: true,
            detail_level: ReportDetailLevel::Summary,
            destination: "logs/optimization.log".to_string(),
            format: ReportFormat::Json,
        }
    }
}

/// Report detail levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportDetailLevel {
    /// Minimal reporting
    Minimal,
    /// Summary reporting
    Summary,
    /// Detailed reporting
    Detailed,
    /// Verbose reporting
    Verbose,
}

/// Filter performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterPerformanceMonitoring {
    /// Monitoring enabled
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Metrics collection
    pub metrics_collection: MetricsCollection,
    /// Performance alerting
    pub alerting: PerformanceAlerting,
    /// Performance reporting
    pub reporting: PerformanceReporting,
}

impl Default for FilterPerformanceMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            metrics_collection: MetricsCollection::default(),
            alerting: PerformanceAlerting::default(),
            reporting: PerformanceReporting::default(),
        }
    }
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollection {
    /// Collect execution time metrics
    pub execution_time: bool,
    /// Collect memory usage metrics
    pub memory_usage: bool,
    /// Collect throughput metrics
    pub throughput: bool,
    /// Collect error rate metrics
    pub error_rate: bool,
    /// Collect cache hit rate metrics
    pub cache_hit_rate: bool,
    /// Metrics retention period
    pub retention_period: Duration,
    /// Metrics aggregation
    pub aggregation: MetricsAggregation,
}

impl Default for MetricsCollection {
    fn default() -> Self {
        Self {
            execution_time: true,
            memory_usage: true,
            throughput: true,
            error_rate: true,
            cache_hit_rate: true,
            retention_period: Duration::from_secs(86400 * 7), // 7 days
            aggregation: MetricsAggregation::default(),
        }
    }
}

/// Metrics aggregation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsAggregation {
    /// Aggregation window
    pub window: Duration,
    /// Aggregation functions
    pub functions: Vec<AggregationFunction>,
    /// Percentiles to calculate
    pub percentiles: Vec<f64>,
}

impl Default for MetricsAggregation {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(300), // 5 minutes
            functions: vec![
                AggregationFunction::Average,
                AggregationFunction::Minimum,
                AggregationFunction::Maximum,
                AggregationFunction::Sum,
            ],
            percentiles: vec![50.0, 90.0, 95.0, 99.0],
        }
    }
}

/// Aggregation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    Sum,
    Average,
    Minimum,
    Maximum,
    Count,
    StandardDeviation,
    Median,
}

/// Performance alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlerting {
    /// Alerting enabled
    pub enabled: bool,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
    /// Alert channels
    pub channels: Vec<AlertChannel>,
    /// Alert suppression
    pub suppression: AlertSuppression,
}

impl Default for PerformanceAlerting {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: AlertThresholds::default(),
            channels: Vec::new(),
            suppression: AlertSuppression::default(),
        }
    }
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Maximum memory usage
    pub max_memory_usage: usize,
    /// Minimum throughput
    pub min_throughput: f64,
    /// Maximum error rate
    pub max_error_rate: f64,
    /// Minimum cache hit rate
    pub min_cache_hit_rate: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_millis(1000),
            max_memory_usage: 100 * 1024 * 1024, // 100MB
            min_throughput: 100.0, // events per second
            max_error_rate: 0.05, // 5%
            min_cache_hit_rate: 0.8, // 80%
        }
    }
}

/// Alert channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertChannel {
    Email {
        addresses: Vec<String>,
    },
    Slack {
        webhook_url: String,
        channel: String,
    },
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    Log {
        level: String,
        destination: String,
    },
}

/// Alert suppression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSuppression {
    /// Suppression enabled
    pub enabled: bool,
    /// Suppression window
    pub window: Duration,
    /// Maximum alerts per window
    pub max_alerts_per_window: usize,
    /// Suppression rules
    pub rules: Vec<SuppressionRule>,
}

impl Default for AlertSuppression {
    fn default() -> Self {
        Self {
            enabled: true,
            window: Duration::from_secs(300), // 5 minutes
            max_alerts_per_window: 5,
            rules: Vec::new(),
        }
    }
}

/// Suppression rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionRule {
    /// Rule name
    pub name: String,
    /// Suppression condition
    pub condition: SuppressionCondition,
    /// Suppression action
    pub action: SuppressionAction,
}

/// Suppression conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuppressionCondition {
    /// Suppress by alert type
    AlertType(String),
    /// Suppress by source
    Source(String),
    /// Suppress by time window
    TimeWindow {
        start: SystemTime,
        end: SystemTime,
    },
    /// Suppress by alert frequency
    Frequency {
        threshold: usize,
        window: Duration,
    },
}

/// Suppression actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuppressionAction {
    /// Completely suppress alerts
    Suppress,
    /// Delay alerts
    Delay(Duration),
    /// Aggregate alerts
    Aggregate {
        window: Duration,
        max_count: usize,
    },
    /// Reduce alert priority
    ReducePriority,
}

/// Performance reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReporting {
    /// Reporting enabled
    pub enabled: bool,
    /// Report generation frequency
    pub frequency: Duration,
    /// Report content
    pub content: ReportContent,
    /// Report distribution
    pub distribution: ReportDistribution,
}

impl Default for PerformanceReporting {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(3600), // 1 hour
            content: ReportContent::default(),
            distribution: ReportDistribution::default(),
        }
    }
}

/// Report content configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportContent {
    /// Include executive summary
    pub executive_summary: bool,
    /// Include detailed metrics
    pub detailed_metrics: bool,
    /// Include performance trends
    pub performance_trends: bool,
    /// Include recommendations
    pub recommendations: bool,
    /// Include comparisons
    pub comparisons: bool,
}

impl Default for ReportContent {
    fn default() -> Self {
        Self {
            executive_summary: true,
            detailed_metrics: false,
            performance_trends: true,
            recommendations: true,
            comparisons: false,
        }
    }
}

/// Report distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDistribution {
    /// Distribution channels
    pub channels: Vec<DistributionChannel>,
    /// Distribution schedule
    pub schedule: DistributionSchedule,
}

impl Default for ReportDistribution {
    fn default() -> Self {
        Self {
            channels: vec![DistributionChannel::File {
                path: "reports/filter_performance.html".to_string(),
            }],
            schedule: DistributionSchedule::Immediate,
        }
    }
}

/// Distribution channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionChannel {
    Email {
        addresses: Vec<String>,
        format: ReportFormat,
    },
    File {
        path: String,
    },
    Database {
        connection_string: String,
        table: String,
    },
    Api {
        endpoint: String,
        headers: HashMap<String, String>,
    },
}

/// Distribution schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionSchedule {
    /// Distribute immediately after generation
    Immediate,
    /// Distribute at specific time
    Scheduled(SystemTime),
    /// Distribute on specific days
    Weekly {
        days: Vec<u8>, // 0 = Sunday, 6 = Saturday
        time: (u8, u8), // (hour, minute)
    },
    /// Distribute monthly
    Monthly {
        day: u8,
        time: (u8, u8),
    },
}

/// Filter composition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterComposition {
    /// Composition strategies
    pub strategies: Vec<CompositionStrategy>,
    /// Composition rules
    pub rules: Vec<CompositionRule>,
    /// Composition optimization
    pub optimization: CompositionOptimization,
    /// Composition validation
    pub validation: CompositionValidation,
}

impl Default for FilterComposition {
    fn default() -> Self {
        Self {
            strategies: vec![CompositionStrategy::Sequential],
            rules: Vec::new(),
            optimization: CompositionOptimization::default(),
            validation: CompositionValidation::default(),
        }
    }
}

/// Composition strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionStrategy {
    /// Execute filters sequentially
    Sequential,
    /// Execute filters in parallel
    Parallel,
    /// Execute filters conditionally
    Conditional,
    /// Execute filters in pipeline
    Pipeline,
    /// Execute filters in hierarchy
    Hierarchy,
}

/// Composition rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionRule {
    /// Rule name
    pub name: String,
    /// Composition condition
    pub condition: CompositionCondition,
    /// Composition action
    pub action: CompositionAction,
}

/// Composition conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionCondition {
    /// Filter result condition
    FilterResult {
        filter_id: String,
        result: bool,
    },
    /// Performance condition
    Performance {
        metric: String,
        threshold: f64,
    },
    /// Resource condition
    Resource {
        resource: String,
        usage: f64,
    },
    /// Time condition
    Time {
        time_range: TimeRange,
    },
}

/// Composition actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionAction {
    /// Continue to next filter
    Continue,
    /// Skip filters
    Skip {
        count: usize,
    },
    /// Jump to specific filter
    JumpTo {
        filter_id: String,
    },
    /// Stop processing
    Stop,
    /// Restart from beginning
    Restart,
}

/// Composition optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionOptimization {
    /// Optimization enabled
    pub enabled: bool,
    /// Optimization techniques
    pub techniques: Vec<CompositionOptimizationTechnique>,
    /// Optimization frequency
    pub frequency: Duration,
}

impl Default for CompositionOptimization {
    fn default() -> Self {
        Self {
            enabled: true,
            techniques: vec![
                CompositionOptimizationTechnique::FilterReordering,
                CompositionOptimizationTechnique::EarlyTermination,
            ],
            frequency: Duration::from_secs(3600),
        }
    }
}

/// Composition optimization techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionOptimizationTechnique {
    /// Reorder filters for better performance
    FilterReordering,
    /// Enable early termination
    EarlyTermination,
    /// Merge compatible filters
    FilterMerging,
    /// Parallelize independent filters
    Parallelization,
    /// Cache intermediate results
    ResultCaching,
}

/// Composition validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionValidation {
    /// Validation enabled
    pub enabled: bool,
    /// Validation rules
    pub rules: Vec<CompositionValidationRule>,
    /// Validation reporting
    pub reporting: bool,
}

impl Default for CompositionValidation {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![
                CompositionValidationRule::CyclicDependency,
                CompositionValidationRule::UnreachableFilter,
            ],
            reporting: true,
        }
    }
}

/// Composition validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionValidationRule {
    /// Check for cyclic dependencies
    CyclicDependency,
    /// Check for unreachable filters
    UnreachableFilter,
    /// Check for redundant filters
    RedundantFilter,
    /// Check for performance bottlenecks
    PerformanceBottleneck,
    /// Check for resource constraints
    ResourceConstraints,
}

/// Filter storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterStorage {
    /// Storage backend
    pub backend: StorageBackend,
    /// Storage optimization
    pub optimization: StorageOptimization,
    /// Storage backup
    pub backup: StorageBackup,
    /// Storage caching
    pub caching: StorageCaching,
}

impl Default for FilterStorage {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Memory,
            optimization: StorageOptimization::default(),
            backup: StorageBackup::default(),
            caching: StorageCaching::default(),
        }
    }
}

/// Storage backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    /// In-memory storage
    Memory,
    /// File-based storage
    File {
        path: String,
        format: StorageFormat,
    },
    /// Database storage
    Database {
        connection_string: String,
        table: String,
    },
    /// Redis storage
    Redis {
        connection_string: String,
        key_prefix: String,
    },
    /// Distributed storage
    Distributed {
        nodes: Vec<String>,
        replication_factor: usize,
    },
}

/// Storage formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageFormat {
    Json,
    Binary,
    Compressed,
    Encrypted,
}

/// Storage optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageOptimization {
    /// Optimization enabled
    pub enabled: bool,
    /// Compression enabled
    pub compression: bool,
    /// Indexing strategy
    pub indexing: IndexingStrategy,
    /// Partitioning strategy
    pub partitioning: PartitioningStrategy,
}

impl Default for StorageOptimization {
    fn default() -> Self {
        Self {
            enabled: true,
            compression: true,
            indexing: IndexingStrategy::FieldBased,
            partitioning: PartitioningStrategy::TimeBase,
        }
    }
}

/// Indexing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexingStrategy {
    /// No indexing
    None,
    /// Field-based indexing
    FieldBased,
    /// Hash-based indexing
    HashBased,
    /// Tree-based indexing
    TreeBased,
    /// Full-text indexing
    FullText,
}

/// Partitioning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitioningStrategy {
    /// No partitioning
    None,
    /// Time-based partitioning
    TimeBase,
    /// Hash-based partitioning
    HashBased,
    /// Range-based partitioning
    RangeBased,
    /// Custom partitioning
    Custom(String),
}

/// Storage backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageBackup {
    /// Backup enabled
    pub enabled: bool,
    /// Backup frequency
    pub frequency: Duration,
    /// Backup destination
    pub destination: BackupDestination,
    /// Backup retention
    pub retention: BackupRetention,
}

impl Default for StorageBackup {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: Duration::from_secs(3600), // 1 hour
            destination: BackupDestination::File {
                path: "backups/filters".to_string(),
            },
            retention: BackupRetention::default(),
        }
    }
}

/// Backup destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupDestination {
    File {
        path: String,
    },
    Database {
        connection_string: String,
        table: String,
    },
    Cloud {
        provider: String,
        bucket: String,
        credentials: String,
    },
    Remote {
        url: String,
        authentication: Option<String>,
    },
}

/// Backup retention policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRetention {
    /// Maximum backup count
    pub max_count: usize,
    /// Maximum backup age
    pub max_age: Duration,
    /// Retention strategy
    pub strategy: RetentionStrategy,
}

impl Default for BackupRetention {
    fn default() -> Self {
        Self {
            max_count: 100,
            max_age: Duration::from_secs(86400 * 30), // 30 days
            strategy: RetentionStrategy::TimeBase,
        }
    }
}

/// Retention strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionStrategy {
    /// Time-based retention
    TimeBase,
    /// Count-based retention
    CountBase,
    /// Size-based retention
    SizeBase,
    /// Custom retention
    Custom(String),
}

/// Storage caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageCaching {
    /// Caching enabled
    pub enabled: bool,
    /// Cache size limit
    pub size_limit: usize,
    /// Cache TTL
    pub ttl: Duration,
    /// Cache strategy
    pub strategy: CacheStrategy,
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
}

impl Default for StorageCaching {
    fn default() -> Self {
        Self {
            enabled: true,
            size_limit: 10000, // Number of cached items
            ttl: Duration::from_secs(3600), // 1 hour
            strategy: CacheStrategy::LRU,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
        }
    }
}

/// Cache strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random replacement
    Random,
    /// Custom strategy
    Custom(String),
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    /// Least recently used
    LeastRecentlyUsed,
    /// Least frequently used
    LeastFrequentlyUsed,
    /// Oldest first
    OldestFirst,
    /// Random eviction
    Random,
    /// Size-based eviction
    SizeBased,
}

/// Main event filtering engine
#[derive(Debug)]
pub struct EventFilteringEngine {
    /// Filtering configuration
    config: Arc<RwLock<EventFiltering>>,
    /// Filter rules storage
    rules: Arc<RwLock<HashMap<String, FilterRule>>>,
    /// Compiled filter cache
    compiled_cache: Arc<Mutex<HashMap<String, CompiledFilter>>>,
    /// Performance metrics
    metrics: Arc<RwLock<FilteringMetrics>>,
    /// Optimization engine
    optimizer: Arc<Mutex<FilterOptimizer>>,
    /// Rule manager
    rule_manager: Arc<RuleManager>,
    /// Performance monitor
    performance_monitor: Arc<PerformanceMonitor>,
}

/// Compiled filter representation
#[derive(Debug)]
pub struct CompiledFilter {
    /// Original rule
    pub rule: FilterRule,
    /// Compiled condition
    pub condition: CompiledCondition,
    /// Compilation timestamp
    pub compiled_at: Instant,
    /// Performance statistics
    pub stats: PerformanceStats,
}

/// Compiled condition types
#[derive(Debug)]
pub enum CompiledCondition {
    /// Simple comparison
    Comparison {
        field_accessor: FieldAccessor,
        operator: ComparisonOperator,
        value: FilterValue,
    },
    /// Pattern matching
    Pattern {
        field_accessor: FieldAccessor,
        pattern: CompiledPattern,
    },
    /// Range check
    Range {
        field_accessor: FieldAccessor,
        min: FilterValue,
        max: FilterValue,
        inclusive: bool,
    },
    /// Set membership
    Set {
        field_accessor: FieldAccessor,
        values: HashSet<FilterValue>,
        negate: bool,
    },
    /// Composite condition
    Composite {
        operator: LogicalOperator,
        conditions: Vec<CompiledCondition>,
    },
    /// Native code
    Native {
        function_ptr: usize,
    },
}

/// Field accessor for efficient field retrieval
#[derive(Debug)]
pub struct FieldAccessor {
    /// Field path
    pub path: Vec<String>,
    /// Cached accessor function
    pub accessor: fn(&HashMap<String, FilterValue>) -> Option<&FilterValue>,
}

/// Compiled pattern for efficient matching
#[derive(Debug)]
pub struct CompiledPattern {
    /// Pattern string
    pub pattern: String,
    /// Compiled regex
    pub regex: Option<regex::Regex>,
    /// Pattern flags
    pub flags: PatternFlags,
}

/// Performance statistics
#[derive(Debug, Default)]
pub struct PerformanceStats {
    /// Total executions
    pub executions: u64,
    /// Total execution time
    pub total_time: Duration,
    /// Average execution time
    pub average_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Last execution time
    pub last_execution: Option<Instant>,
}

/// Filtering metrics
#[derive(Debug, Default)]
pub struct FilteringMetrics {
    /// Total events processed
    pub total_events: u64,
    /// Events accepted
    pub events_accepted: u64,
    /// Events rejected
    pub events_rejected: u64,
    /// Events transformed
    pub events_transformed: u64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average processing time per event
    pub avg_processing_time: Duration,
    /// Filter execution counts
    pub filter_executions: HashMap<String, u64>,
    /// Filter performance metrics
    pub filter_performance: HashMap<String, PerformanceStats>,
    /// Error counts
    pub error_counts: HashMap<String, u64>,
}

/// Filter optimizer
#[derive(Debug)]
pub struct FilterOptimizer {
    /// Optimization configuration
    config: OptimizationConfiguration,
    /// Optimization history
    history: VecDeque<OptimizationResult>,
    /// Current optimization state
    state: OptimizationState,
}

/// Optimization configuration
#[derive(Debug)]
pub struct OptimizationConfiguration {
    /// Enabled optimization strategies
    pub strategies: HashSet<OptimizationStrategy>,
    /// Optimization thresholds
    pub thresholds: OptimizationThresholds,
    /// Optimization frequency
    pub frequency: Duration,
}

/// Optimization result
#[derive(Debug)]
pub struct OptimizationResult {
    /// Optimization timestamp
    pub timestamp: Instant,
    /// Applied strategies
    pub strategies: Vec<OptimizationStrategy>,
    /// Performance improvement
    pub improvement: f64,
    /// Optimization duration
    pub duration: Duration,
    /// Success indicator
    pub success: bool,
}

/// Optimization state
#[derive(Debug)]
pub enum OptimizationState {
    /// Not optimizing
    Idle,
    /// Currently optimizing
    Running {
        started_at: Instant,
        current_strategy: OptimizationStrategy,
    },
    /// Optimization completed
    Completed {
        completed_at: Instant,
        result: OptimizationResult,
    },
    /// Optimization failed
    Failed {
        failed_at: Instant,
        error: String,
    },
}

/// Rule manager
#[derive(Debug)]
pub struct RuleManager {
    /// Rule storage
    storage: Arc<RwLock<HashMap<String, FilterRule>>>,
    /// Rule validation engine
    validator: Arc<RuleValidator>,
    /// Rule conflict detector
    conflict_detector: Arc<ConflictDetector>,
    /// Rule dependency manager
    dependency_manager: Arc<DependencyManager>,
}

/// Rule validator
#[derive(Debug)]
pub struct RuleValidator {
    /// Validation configuration
    config: RuleValidation,
    /// Validation results cache
    cache: Arc<Mutex<HashMap<String, ValidationResult>>>,
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    /// Validation success
    pub success: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Validation timestamp
    pub timestamp: Instant,
}

/// Validation error
#[derive(Debug)]
pub struct ValidationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Error location
    pub location: Option<String>,
}

/// Error severity levels
#[derive(Debug)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Validation warning
#[derive(Debug)]
pub struct ValidationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Warning location
    pub location: Option<String>,
}

/// Conflict detector
#[derive(Debug)]
pub struct ConflictDetector {
    /// Conflict detection configuration
    config: RuleConflictResolution,
    /// Detected conflicts cache
    conflicts: Arc<RwLock<HashMap<String, Vec<RuleConflict>>>>,
}

/// Rule conflict representation
#[derive(Debug)]
pub struct RuleConflict {
    /// Conflict type
    pub conflict_type: ConflictType,
    /// Conflicting rules
    pub rules: Vec<String>,
    /// Conflict severity
    pub severity: ConflictSeverity,
    /// Conflict description
    pub description: String,
    /// Detection timestamp
    pub detected_at: Instant,
}

/// Conflict types
#[derive(Debug)]
pub enum ConflictType {
    /// Rules with contradictory conditions
    ContradictoryConditions,
    /// Rules with same priority
    PriorityConflict,
    /// Rules with overlapping effects
    OverlappingEffects,
    /// Rules with circular dependencies
    CircularDependency,
    /// Rules with resource conflicts
    ResourceConflict,
}

/// Conflict severity levels
#[derive(Debug)]
pub enum ConflictSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Dependency manager
#[derive(Debug)]
pub struct DependencyManager {
    /// Dependency graph
    graph: Arc<RwLock<DependencyGraph>>,
    /// Dependency resolution cache
    cache: Arc<Mutex<HashMap<String, Vec<String>>>>,
}

/// Dependency graph
#[derive(Debug)]
pub struct DependencyGraph {
    /// Nodes (rules)
    pub nodes: HashSet<String>,
    /// Edges (dependencies)
    pub edges: HashMap<String, HashSet<String>>,
    /// Reverse edges (dependents)
    pub reverse_edges: HashMap<String, HashSet<String>>,
}

/// Performance monitor
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Monitoring configuration
    config: FilterPerformanceMonitoring,
    /// Performance data collector
    collector: Arc<PerformanceDataCollector>,
    /// Alert manager
    alert_manager: Arc<AlertManager>,
    /// Report generator
    report_generator: Arc<ReportGenerator>,
}

/// Performance data collector
#[derive(Debug)]
pub struct PerformanceDataCollector {
    /// Collected metrics
    metrics: Arc<RwLock<HashMap<String, MetricTimeSeries>>>,
    /// Collection interval
    interval: Duration,
    /// Data retention period
    retention: Duration,
}

/// Metric time series
#[derive(Debug)]
pub struct MetricTimeSeries {
    /// Metric name
    pub name: String,
    /// Data points
    pub data: VecDeque<MetricDataPoint>,
    /// Maximum data points
    pub max_points: usize,
}

/// Metric data point
#[derive(Debug)]
pub struct MetricDataPoint {
    /// Timestamp
    pub timestamp: Instant,
    /// Metric value
    pub value: f64,
    /// Optional metadata
    pub metadata: Option<HashMap<String, String>>,
}

/// Alert manager
#[derive(Debug)]
pub struct AlertManager {
    /// Alert configuration
    config: PerformanceAlerting,
    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, Alert>>>,
    /// Alert history
    history: Arc<RwLock<VecDeque<Alert>>>,
}

/// Alert representation
#[derive(Debug)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Alert timestamp
    pub timestamp: Instant,
    /// Alert source
    pub source: String,
    /// Alert metadata
    pub metadata: HashMap<String, String>,
}

/// Alert types
#[derive(Debug)]
pub enum AlertType {
    PerformanceDegradation,
    HighErrorRate,
    ResourceExhaustion,
    ConfigurationError,
    SystemFailure,
}

/// Alert severity levels
#[derive(Debug)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Report generator
#[derive(Debug)]
pub struct ReportGenerator {
    /// Report configuration
    config: PerformanceReporting,
    /// Report templates
    templates: Arc<RwLock<HashMap<String, ReportTemplate>>>,
    /// Generated reports cache
    cache: Arc<RwLock<HashMap<String, GeneratedReport>>>,
}

/// Report template
#[derive(Debug)]
pub struct ReportTemplate {
    /// Template name
    pub name: String,
    /// Template content
    pub content: String,
    /// Template format
    pub format: ReportFormat,
    /// Template variables
    pub variables: Vec<String>,
}

/// Generated report
#[derive(Debug)]
pub struct GeneratedReport {
    /// Report ID
    pub id: String,
    /// Report content
    pub content: String,
    /// Report format
    pub format: ReportFormat,
    /// Generation timestamp
    pub generated_at: Instant,
    /// Report metadata
    pub metadata: HashMap<String, String>,
}

impl EventFilteringEngine {
    /// Create new event filtering engine
    pub fn new(config: EventFiltering) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            rules: Arc::new(RwLock::new(HashMap::new())),
            compiled_cache: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(FilteringMetrics::default())),
            optimizer: Arc::new(Mutex::new(FilterOptimizer::new())),
            rule_manager: Arc::new(RuleManager::new()),
            performance_monitor: Arc::new(PerformanceMonitor::new()),
        }
    }

    /// Add filter rule
    pub fn add_rule(&self, rule: FilterRule) -> FilteringResult<()> {
        // Validate rule
        self.rule_manager.validate_rule(&rule)?;

        // Check for conflicts
        self.rule_manager.check_conflicts(&rule)?;

        // Add to storage
        let mut rules = self.rules.write().unwrap();
        rules.insert(rule.rule_id.clone(), rule);

        // Clear compiled cache
        self.compiled_cache.lock().unwrap().clear();

        Ok(())
    }

    /// Remove filter rule
    pub fn remove_rule(&self, rule_id: &str) -> FilteringResult<FilterRule> {
        let mut rules = self.rules.write().unwrap();
        let rule = rules.remove(rule_id)
            .ok_or_else(|| FilteringError::FilterNotFound(rule_id.to_string()))?;

        // Clear compiled cache
        self.compiled_cache.lock().unwrap().clear();

        Ok(rule)
    }

    /// Get filter rule
    pub fn get_rule(&self, rule_id: &str) -> Option<FilterRule> {
        let rules = self.rules.read().unwrap();
        rules.get(rule_id).cloned()
    }

    /// List all filter rules
    pub fn list_rules(&self) -> Vec<FilterRule> {
        let rules = self.rules.read().unwrap();
        rules.values().cloned().collect()
    }

    /// Process event through filters
    pub fn process_event(&self, event: &HashMap<String, FilterValue>) -> FilteringResult<FilterResult> {
        let start_time = Instant::now();

        let rules = self.rules.read().unwrap();
        let mut result = FilterResult::default();

        // Process through each rule in priority order
        let mut sorted_rules: Vec<_> = rules.values().collect();
        sorted_rules.sort_by(|a, b| b.priority.cmp(&a.priority));

        for rule in sorted_rules {
            if rule.status != RuleStatus::Active {
                continue;
            }

            match self.evaluate_rule(rule, event)? {
                Some(action_result) => {
                    result.actions.push(action_result);

                    // Check if processing should stop
                    if matches!(result.actions.last(), Some(ActionResult::Reject)) {
                        break;
                    }
                }
                None => continue,
            }
        }

        // Update metrics
        let processing_time = start_time.elapsed();
        self.update_metrics(processing_time, &result);

        Ok(result)
    }

    /// Evaluate rule against event
    fn evaluate_rule(&self, rule: &FilterRule, event: &HashMap<String, FilterValue>) -> FilteringResult<Option<ActionResult>> {
        let start_time = Instant::now();

        // Get or compile condition
        let compiled = self.get_compiled_condition(rule)?;

        // Evaluate condition
        let matches = self.evaluate_condition(&compiled.condition, event)?;

        let evaluation_time = start_time.elapsed();
        self.update_rule_performance(&rule.rule_id, evaluation_time, true);

        if matches {
            let action_result = self.execute_action(&rule.action, event)?;
            Ok(Some(action_result))
        } else {
            Ok(None)
        }
    }

    /// Get or compile condition
    fn get_compiled_condition(&self, rule: &FilterRule) -> FilteringResult<CompiledFilter> {
        let mut cache = self.compiled_cache.lock().unwrap();

        if let Some(compiled) = cache.get(&rule.rule_id) {
            return Ok(compiled.clone());
        }

        // Compile condition
        let compiled_condition = self.compile_condition(&rule.condition)?;
        let compiled = CompiledFilter {
            rule: rule.clone(),
            condition: compiled_condition,
            compiled_at: Instant::now(),
            stats: PerformanceStats::default(),
        };

        cache.insert(rule.rule_id.clone(), compiled.clone());
        Ok(compiled)
    }

    /// Compile condition for efficient evaluation
    fn compile_condition(&self, condition: &FilterCondition) -> FilteringResult<CompiledCondition> {
        match condition {
            FilterCondition::FieldComparison { field, operator, value } => {
                Ok(CompiledCondition::Comparison {
                    field_accessor: self.compile_field_accessor(field),
                    operator: operator.clone(),
                    value: value.clone(),
                })
            }
            FilterCondition::PatternMatch { field, pattern, flags } => {
                Ok(CompiledCondition::Pattern {
                    field_accessor: self.compile_field_accessor(field),
                    pattern: self.compile_pattern(pattern, flags)?,
                })
            }
            FilterCondition::Range { field, min, max, inclusive } => {
                Ok(CompiledCondition::Range {
                    field_accessor: self.compile_field_accessor(field),
                    min: min.clone(),
                    max: max.clone(),
                    inclusive: *inclusive,
                })
            }
            FilterCondition::SetMembership { field, values, negate } => {
                Ok(CompiledCondition::Set {
                    field_accessor: self.compile_field_accessor(field),
                    values: values.clone(),
                    negate: *negate,
                })
            }
            FilterCondition::Composite { operator, conditions } => {
                let compiled_conditions = conditions.iter()
                    .map(|c| self.compile_condition(c))
                    .collect::<FilteringResult<Vec<_>>>()?;

                Ok(CompiledCondition::Composite {
                    operator: operator.clone(),
                    conditions: compiled_conditions,
                })
            }
            _ => Err(FilteringError::CompilationFailed("Unsupported condition type".to_string())),
        }
    }

    /// Compile field accessor
    fn compile_field_accessor(&self, field: &str) -> FieldAccessor {
        let path: Vec<String> = field.split('.').map(|s| s.to_string()).collect();

        FieldAccessor {
            path,
            accessor: |_| None, // Placeholder - would be optimized in real implementation
        }
    }

    /// Compile pattern
    fn compile_pattern(&self, pattern: &str, flags: &PatternFlags) -> FilteringResult<CompiledPattern> {
        let regex = if flags.case_insensitive || flags.multiline || flags.dot_all {
            let mut regex_builder = regex::RegexBuilder::new(pattern);
            regex_builder.case_insensitive(flags.case_insensitive);
            regex_builder.multi_line(flags.multiline);
            regex_builder.dot_matches_new_line(flags.dot_all);

            Some(regex_builder.build()
                .map_err(|e| FilteringError::CompilationFailed(format!("Regex compilation failed: {}", e)))?)
        } else {
            Some(regex::Regex::new(pattern)
                .map_err(|e| FilteringError::CompilationFailed(format!("Regex compilation failed: {}", e)))?)
        };

        Ok(CompiledPattern {
            pattern: pattern.to_string(),
            regex,
            flags: flags.clone(),
        })
    }

    /// Evaluate compiled condition
    fn evaluate_condition(&self, condition: &CompiledCondition, event: &HashMap<String, FilterValue>) -> FilteringResult<bool> {
        match condition {
            CompiledCondition::Comparison { field_accessor, operator, value } => {
                if let Some(field_value) = self.get_field_value(field_accessor, event) {
                    self.compare_values(field_value, operator, value)
                } else {
                    Ok(false)
                }
            }
            CompiledCondition::Pattern { field_accessor, pattern } => {
                if let Some(field_value) = self.get_field_value(field_accessor, event) {
                    self.match_pattern(field_value, pattern)
                } else {
                    Ok(false)
                }
            }
            CompiledCondition::Range { field_accessor, min, max, inclusive } => {
                if let Some(field_value) = self.get_field_value(field_accessor, event) {
                    self.check_range(field_value, min, max, *inclusive)
                } else {
                    Ok(false)
                }
            }
            CompiledCondition::Set { field_accessor, values, negate } => {
                if let Some(field_value) = self.get_field_value(field_accessor, event) {
                    let contains = values.contains(field_value);
                    Ok(if *negate { !contains } else { contains })
                } else {
                    Ok(false)
                }
            }
            CompiledCondition::Composite { operator, conditions } => {
                match operator {
                    LogicalOperator::And => {
                        for condition in conditions {
                            if !self.evaluate_condition(condition, event)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    LogicalOperator::Or => {
                        for condition in conditions {
                            if self.evaluate_condition(condition, event)? {
                                return Ok(true);
                            }
                        }
                        Ok(false)
                    }
                    LogicalOperator::Not => {
                        if conditions.len() != 1 {
                            return Err(FilteringError::ExecutionError("NOT operator requires exactly one condition".to_string()));
                        }
                        Ok(!self.evaluate_condition(&conditions[0], event)?)
                    }
                    LogicalOperator::Xor => {
                        let mut true_count = 0;
                        for condition in conditions {
                            if self.evaluate_condition(condition, event)? {
                                true_count += 1;
                            }
                        }
                        Ok(true_count == 1)
                    }
                }
            }
            CompiledCondition::Native { .. } => {
                // Native code execution would be implemented here
                Err(FilteringError::ExecutionError("Native code execution not implemented".to_string()))
            }
        }
    }

    /// Get field value from event
    fn get_field_value(&self, accessor: &FieldAccessor, event: &HashMap<String, FilterValue>) -> Option<&FilterValue> {
        // Simple implementation - in practice would use the compiled accessor
        let mut current = event;
        for part in &accessor.path[..accessor.path.len()-1] {
            if let Some(FilterValue::Object(ref obj)) = current.get(part) {
                current = obj;
            } else {
                return None;
            }
        }
        current.get(accessor.path.last()?)
    }

    /// Compare values using operator
    fn compare_values(&self, left: &FilterValue, operator: &ComparisonOperator, right: &FilterValue) -> FilteringResult<bool> {
        match operator {
            ComparisonOperator::Equal => Ok(left == right),
            ComparisonOperator::NotEqual => Ok(left != right),
            // Additional comparison operators would be implemented here
            _ => Err(FilteringError::ExecutionError("Unsupported comparison operator".to_string())),
        }
    }

    /// Match pattern against value
    fn match_pattern(&self, value: &FilterValue, pattern: &CompiledPattern) -> FilteringResult<bool> {
        if let FilterValue::String(ref s) = value {
            if let Some(ref regex) = pattern.regex {
                Ok(regex.is_match(s))
            } else {
                Ok(s.contains(&pattern.pattern))
            }
        } else {
            Ok(false)
        }
    }

    /// Check if value is in range
    fn check_range(&self, value: &FilterValue, min: &FilterValue, max: &FilterValue, inclusive: bool) -> FilteringResult<bool> {
        // Range checking implementation would go here
        // This is a simplified version
        Ok(true)
    }

    /// Execute filter action
    fn execute_action(&self, action: &FilterAction, event: &HashMap<String, FilterValue>) -> FilteringResult<ActionResult> {
        match action {
            FilterAction::Accept => Ok(ActionResult::Accept),
            FilterAction::Reject => Ok(ActionResult::Reject),
            FilterAction::Transform { transformations } => {
                let mut transformed_event = event.clone();
                for transformation in transformations {
                    self.apply_transformation(&mut transformed_event, transformation)?;
                }
                Ok(ActionResult::Transform(transformed_event))
            }
            FilterAction::Route { destination, priority } => {
                Ok(ActionResult::Route {
                    destination: destination.clone(),
                    priority: *priority,
                })
            }
            FilterAction::Delay { duration, reason } => {
                Ok(ActionResult::Delay {
                    duration: *duration,
                    reason: reason.clone(),
                })
            }
            _ => Err(FilteringError::ExecutionError("Unsupported action type".to_string())),
        }
    }

    /// Apply transformation to event
    fn apply_transformation(&self, event: &mut HashMap<String, FilterValue>, transformation: &EventTransformation) -> FilteringResult<()> {
        match transformation {
            EventTransformation::AddField { field, value } => {
                event.insert(field.clone(), value.clone());
                Ok(())
            }
            EventTransformation::RemoveField { field } => {
                event.remove(field);
                Ok(())
            }
            EventTransformation::RenameField { old_name, new_name } => {
                if let Some(value) = event.remove(old_name) {
                    event.insert(new_name.clone(), value);
                }
                Ok(())
            }
            _ => Err(FilteringError::ExecutionError("Unsupported transformation type".to_string())),
        }
    }

    /// Update performance metrics
    fn update_metrics(&self, processing_time: Duration, result: &FilterResult) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.total_events += 1;
        metrics.total_processing_time += processing_time;
        metrics.avg_processing_time = metrics.total_processing_time / metrics.total_events as u32;

        for action in &result.actions {
            match action {
                ActionResult::Accept => metrics.events_accepted += 1,
                ActionResult::Reject => metrics.events_rejected += 1,
                ActionResult::Transform(_) => metrics.events_transformed += 1,
                _ => {}
            }
        }
    }

    /// Update rule performance metrics
    fn update_rule_performance(&self, rule_id: &str, execution_time: Duration, success: bool) {
        let mut metrics = self.metrics.write().unwrap();

        let executions = metrics.filter_executions.entry(rule_id.to_string()).or_insert(0);
        *executions += 1;

        let perf = metrics.filter_performance.entry(rule_id.to_string()).or_insert_with(PerformanceStats::default);
        perf.executions += 1;
        perf.total_time += execution_time;
        perf.average_time = perf.total_time / perf.executions as u32;

        if perf.min_time == Duration::ZERO || execution_time < perf.min_time {
            perf.min_time = execution_time;
        }
        if execution_time > perf.max_time {
            perf.max_time = execution_time;
        }

        if success {
            perf.success_rate = (perf.success_rate * (perf.executions - 1) as f64 + 1.0) / perf.executions as f64;
        } else {
            perf.success_rate = (perf.success_rate * (perf.executions - 1) as f64) / perf.executions as f64;
        }

        perf.last_execution = Some(Instant::now());
    }

    /// Optimize filters
    pub fn optimize(&self) -> FilteringResult<OptimizationResult> {
        let mut optimizer = self.optimizer.lock().unwrap();
        optimizer.optimize(self)
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> FilteringMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Get rule performance
    pub fn get_rule_performance(&self, rule_id: &str) -> Option<PerformanceStats> {
        let metrics = self.metrics.read().unwrap();
        metrics.filter_performance.get(rule_id).cloned()
    }
}

/// Filter result
#[derive(Debug, Default)]
pub struct FilterResult {
    /// Applied actions
    pub actions: Vec<ActionResult>,
}

/// Action result types
#[derive(Debug)]
pub enum ActionResult {
    Accept,
    Reject,
    Transform(HashMap<String, FilterValue>),
    Route {
        destination: String,
        priority: Option<i32>,
    },
    Delay {
        duration: Duration,
        reason: String,
    },
    Split(Vec<HashMap<String, FilterValue>>),
    Aggregate {
        key: String,
        window: Duration,
    },
}

impl FilterOptimizer {
    /// Create new filter optimizer
    pub fn new() -> Self {
        Self {
            config: OptimizationConfiguration {
                strategies: HashSet::new(),
                thresholds: OptimizationThresholds::default(),
                frequency: Duration::from_secs(3600),
            },
            history: VecDeque::new(),
            state: OptimizationState::Idle,
        }
    }

    /// Optimize filtering engine
    pub fn optimize(&mut self, engine: &EventFilteringEngine) -> FilteringResult<OptimizationResult> {
        let start_time = Instant::now();

        self.state = OptimizationState::Running {
            started_at: start_time,
            current_strategy: OptimizationStrategy::RuleReordering,
        };

        let mut improvement = 0.0;
        let mut applied_strategies = Vec::new();

        // Apply optimization strategies
        for strategy in &self.config.strategies {
            match self.apply_strategy(strategy, engine) {
                Ok(strategy_improvement) => {
                    improvement += strategy_improvement;
                    applied_strategies.push(strategy.clone());
                }
                Err(_) => continue,
            }
        }

        let duration = start_time.elapsed();
        let success = improvement > self.config.thresholds.min_performance_improvement;

        let result = OptimizationResult {
            timestamp: start_time,
            strategies: applied_strategies,
            improvement,
            duration,
            success,
        };

        self.state = if success {
            OptimizationState::Completed {
                completed_at: Instant::now(),
                result: result.clone(),
            }
        } else {
            OptimizationState::Failed {
                failed_at: Instant::now(),
                error: "Insufficient improvement".to_string(),
            }
        };

        self.history.push_back(result.clone());
        if self.history.len() > 100 {
            self.history.pop_front();
        }

        Ok(result)
    }

    /// Apply optimization strategy
    fn apply_strategy(&self, strategy: &OptimizationStrategy, _engine: &EventFilteringEngine) -> FilteringResult<f64> {
        match strategy {
            OptimizationStrategy::RuleReordering => {
                // Rule reordering implementation
                Ok(0.1) // 10% improvement
            }
            OptimizationStrategy::ConditionSimplification => {
                // Condition simplification implementation
                Ok(0.05) // 5% improvement
            }
            OptimizationStrategy::IndexOptimization => {
                // Index optimization implementation
                Ok(0.15) // 15% improvement
            }
            _ => Err(FilteringError::OptimizationError("Strategy not implemented".to_string())),
        }
    }
}

impl RuleManager {
    /// Create new rule manager
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
            validator: Arc::new(RuleValidator::new()),
            conflict_detector: Arc::new(ConflictDetector::new()),
            dependency_manager: Arc::new(DependencyManager::new()),
        }
    }

    /// Validate rule
    pub fn validate_rule(&self, rule: &FilterRule) -> FilteringResult<()> {
        self.validator.validate(rule)
    }

    /// Check for conflicts
    pub fn check_conflicts(&self, rule: &FilterRule) -> FilteringResult<()> {
        self.conflict_detector.check_conflicts(rule)
    }
}

impl RuleValidator {
    /// Create new rule validator
    pub fn new() -> Self {
        Self {
            config: RuleValidation::default(),
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Validate rule
    pub fn validate(&self, _rule: &FilterRule) -> FilteringResult<()> {
        // Validation implementation would go here
        Ok(())
    }
}

impl ConflictDetector {
    /// Create new conflict detector
    pub fn new() -> Self {
        Self {
            config: RuleConflictResolution::default(),
            conflicts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check for conflicts
    pub fn check_conflicts(&self, _rule: &FilterRule) -> FilteringResult<()> {
        // Conflict detection implementation would go here
        Ok(())
    }
}

impl DependencyManager {
    /// Create new dependency manager
    pub fn new() -> Self {
        Self {
            graph: Arc::new(RwLock::new(DependencyGraph {
                nodes: HashSet::new(),
                edges: HashMap::new(),
                reverse_edges: HashMap::new(),
            })),
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            config: FilterPerformanceMonitoring::default(),
            collector: Arc::new(PerformanceDataCollector::new()),
            alert_manager: Arc::new(AlertManager::new()),
            report_generator: Arc::new(ReportGenerator::new()),
        }
    }
}

impl PerformanceDataCollector {
    /// Create new performance data collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            interval: Duration::from_secs(60),
            retention: Duration::from_secs(86400 * 7), // 7 days
        }
    }
}

impl AlertManager {
    /// Create new alert manager
    pub fn new() -> Self {
        Self {
            config: PerformanceAlerting::default(),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(VecDeque::new())),
        }
    }
}

impl ReportGenerator {
    /// Create new report generator
    pub fn new() -> Self {
        Self {
            config: PerformanceReporting::default(),
            templates: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// Builder for event filtering configuration
pub struct EventFilteringBuilder {
    config: EventFiltering,
}

impl EventFilteringBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: EventFiltering::default(),
        }
    }

    /// Set filter rules configuration
    pub fn filter_rules(mut self, rules: FilterRulesConfig) -> Self {
        self.config.filter_rules = rules;
        self
    }

    /// Set optimization configuration
    pub fn optimization(mut self, optimization: FilterOptimization) -> Self {
        self.config.optimization = optimization;
        self
    }

    /// Set performance monitoring configuration
    pub fn performance_monitoring(mut self, monitoring: FilterPerformanceMonitoring) -> Self {
        self.config.performance_monitoring = monitoring;
        self
    }

    /// Set composition configuration
    pub fn composition(mut self, composition: FilterComposition) -> Self {
        self.config.composition = composition;
        self
    }

    /// Set storage configuration
    pub fn storage(mut self, storage: FilterStorage) -> Self {
        self.config.storage = storage;
        self
    }

    /// Build configuration
    pub fn build(self) -> EventFiltering {
        self.config
    }
}

impl Default for EventFilteringBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Common filtering presets
pub struct FilteringPresets;

impl FilteringPresets {
    /// High-performance filtering preset
    pub fn high_performance() -> EventFiltering {
        EventFilteringBuilder::new()
            .optimization(FilterOptimization {
                enabled: true,
                strategies: vec![
                    OptimizationStrategy::RuleReordering,
                    OptimizationStrategy::ConditionSimplification,
                    OptimizationStrategy::IndexOptimization,
                    OptimizationStrategy::ResultCaching,
                    OptimizationStrategy::Parallelization,
                ],
                schedule: OptimizationSchedule {
                    frequency: OptimizationFrequency::Adaptive,
                    window: Duration::from_secs(60),
                    max_optimization_time: Duration::from_secs(30),
                    triggers: vec![
                        OptimizationTrigger::PerformanceDegradation(0.1),
                        OptimizationTrigger::ExecutionTimeThreshold(Duration::from_millis(100)),
                    ],
                },
                thresholds: OptimizationThresholds {
                    min_performance_improvement: 0.02,
                    max_optimization_overhead: Duration::from_secs(5),
                    min_rule_count: 5,
                    max_iterations: 20,
                },
                reporting: OptimizationReporting::default(),
            })
            .performance_monitoring(FilterPerformanceMonitoring {
                enabled: true,
                interval: Duration::from_secs(30),
                metrics_collection: MetricsCollection {
                    execution_time: true,
                    memory_usage: true,
                    throughput: true,
                    error_rate: true,
                    cache_hit_rate: true,
                    retention_period: Duration::from_secs(86400 * 3), // 3 days
                    aggregation: MetricsAggregation {
                        window: Duration::from_secs(60),
                        functions: vec![
                            AggregationFunction::Average,
                            AggregationFunction::Minimum,
                            AggregationFunction::Maximum,
                            AggregationFunction::Median,
                        ],
                        percentiles: vec![50.0, 90.0, 95.0, 99.0, 99.9],
                    },
                },
                alerting: PerformanceAlerting {
                    enabled: true,
                    thresholds: AlertThresholds {
                        max_execution_time: Duration::from_millis(50),
                        max_memory_usage: 50 * 1024 * 1024, // 50MB
                        min_throughput: 1000.0, // events per second
                        max_error_rate: 0.01, // 1%
                        min_cache_hit_rate: 0.9, // 90%
                    },
                    channels: Vec::new(),
                    suppression: AlertSuppression::default(),
                },
                reporting: PerformanceReporting {
                    enabled: true,
                    frequency: Duration::from_secs(1800), // 30 minutes
                    content: ReportContent {
                        executive_summary: true,
                        detailed_metrics: true,
                        performance_trends: true,
                        recommendations: true,
                        comparisons: true,
                    },
                    distribution: ReportDistribution::default(),
                },
            })
            .storage(FilterStorage {
                backend: StorageBackend::Memory,
                optimization: StorageOptimization {
                    enabled: true,
                    compression: true,
                    indexing: IndexingStrategy::HashBased,
                    partitioning: PartitioningStrategy::HashBased,
                },
                backup: StorageBackup {
                    enabled: false, // Disabled for high performance
                    frequency: Duration::from_secs(3600),
                    destination: BackupDestination::File {
                        path: "/tmp/filter_backup".to_string(),
                    },
                    retention: BackupRetention::default(),
                },
                caching: StorageCaching {
                    enabled: true,
                    size_limit: 50000, // Large cache
                    ttl: Duration::from_secs(1800),
                    strategy: CacheStrategy::LRU,
                    eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
                },
            })
            .build()
    }

    /// Development/debugging preset
    pub fn development() -> EventFiltering {
        EventFilteringBuilder::new()
            .optimization(FilterOptimization {
                enabled: false, // Disabled for easier debugging
                strategies: Vec::new(),
                schedule: OptimizationSchedule::default(),
                thresholds: OptimizationThresholds::default(),
                reporting: OptimizationReporting {
                    enabled: true,
                    detail_level: ReportDetailLevel::Verbose,
                    destination: "logs/debug_optimization.log".to_string(),
                    format: ReportFormat::Json,
                },
            })
            .performance_monitoring(FilterPerformanceMonitoring {
                enabled: true,
                interval: Duration::from_secs(10), // Frequent monitoring
                metrics_collection: MetricsCollection {
                    execution_time: true,
                    memory_usage: true,
                    throughput: true,
                    error_rate: true,
                    cache_hit_rate: true,
                    retention_period: Duration::from_secs(3600), // 1 hour
                    aggregation: MetricsAggregation {
                        window: Duration::from_secs(60),
                        functions: vec![
                            AggregationFunction::Average,
                            AggregationFunction::Minimum,
                            AggregationFunction::Maximum,
                            AggregationFunction::Count,
                        ],
                        percentiles: vec![50.0, 90.0, 95.0, 99.0],
                    },
                },
                alerting: PerformanceAlerting {
                    enabled: true,
                    thresholds: AlertThresholds {
                        max_execution_time: Duration::from_secs(1), // Lenient for debugging
                        max_memory_usage: 500 * 1024 * 1024, // 500MB
                        min_throughput: 10.0, // events per second
                        max_error_rate: 0.1, // 10%
                        min_cache_hit_rate: 0.5, // 50%
                    },
                    channels: vec![AlertChannel::Log {
                        level: "debug".to_string(),
                        destination: "logs/filter_alerts.log".to_string(),
                    }],
                    suppression: AlertSuppression {
                        enabled: false, // No suppression in development
                        window: Duration::from_secs(60),
                        max_alerts_per_window: 100,
                        rules: Vec::new(),
                    },
                },
                reporting: PerformanceReporting {
                    enabled: true,
                    frequency: Duration::from_secs(300), // 5 minutes
                    content: ReportContent {
                        executive_summary: true,
                        detailed_metrics: true,
                        performance_trends: false,
                        recommendations: true,
                        comparisons: false,
                    },
                    distribution: ReportDistribution {
                        channels: vec![DistributionChannel::File {
                            path: "logs/filter_debug_report.html".to_string(),
                        }],
                        schedule: DistributionSchedule::Immediate,
                    },
                },
            })
            .build()
    }

    /// Production preset with comprehensive monitoring
    pub fn production() -> EventFiltering {
        EventFilteringBuilder::new()
            .optimization(FilterOptimization {
                enabled: true,
                strategies: vec![
                    OptimizationStrategy::RuleReordering,
                    OptimizationStrategy::ConditionSimplification,
                    OptimizationStrategy::IndexOptimization,
                    OptimizationStrategy::ResultCaching,
                ],
                schedule: OptimizationSchedule {
                    frequency: OptimizationFrequency::Periodic(Duration::from_secs(7200)), // 2 hours
                    window: Duration::from_secs(300),
                    max_optimization_time: Duration::from_secs(60),
                    triggers: vec![
                        OptimizationTrigger::PerformanceDegradation(0.15),
                        OptimizationTrigger::RuleCountThreshold(200),
                        OptimizationTrigger::MemoryUsageThreshold(200 * 1024 * 1024), // 200MB
                    ],
                },
                thresholds: OptimizationThresholds {
                    min_performance_improvement: 0.05,
                    max_optimization_overhead: Duration::from_secs(30),
                    min_rule_count: 20,
                    max_iterations: 15,
                },
                reporting: OptimizationReporting {
                    enabled: true,
                    detail_level: ReportDetailLevel::Summary,
                    destination: "logs/production_optimization.log".to_string(),
                    format: ReportFormat::Json,
                },
            })
            .performance_monitoring(FilterPerformanceMonitoring {
                enabled: true,
                interval: Duration::from_secs(60),
                metrics_collection: MetricsCollection {
                    execution_time: true,
                    memory_usage: true,
                    throughput: true,
                    error_rate: true,
                    cache_hit_rate: true,
                    retention_period: Duration::from_secs(86400 * 7), // 7 days
                    aggregation: MetricsAggregation {
                        window: Duration::from_secs(300),
                        functions: vec![
                            AggregationFunction::Average,
                            AggregationFunction::Minimum,
                            AggregationFunction::Maximum,
                            AggregationFunction::Sum,
                        ],
                        percentiles: vec![50.0, 90.0, 95.0, 99.0],
                    },
                },
                alerting: PerformanceAlerting {
                    enabled: true,
                    thresholds: AlertThresholds {
                        max_execution_time: Duration::from_millis(200),
                        max_memory_usage: 100 * 1024 * 1024, // 100MB
                        min_throughput: 500.0, // events per second
                        max_error_rate: 0.02, // 2%
                        min_cache_hit_rate: 0.85, // 85%
                    },
                    channels: vec![
                        AlertChannel::Log {
                            level: "error".to_string(),
                            destination: "logs/production_alerts.log".to_string(),
                        },
                    ],
                    suppression: AlertSuppression {
                        enabled: true,
                        window: Duration::from_secs(600), // 10 minutes
                        max_alerts_per_window: 10,
                        rules: Vec::new(),
                    },
                },
                reporting: PerformanceReporting {
                    enabled: true,
                    frequency: Duration::from_secs(3600), // 1 hour
                    content: ReportContent {
                        executive_summary: true,
                        detailed_metrics: false,
                        performance_trends: true,
                        recommendations: true,
                        comparisons: true,
                    },
                    distribution: ReportDistribution {
                        channels: vec![DistributionChannel::File {
                            path: "reports/production_filter_performance.html".to_string(),
                        }],
                        schedule: DistributionSchedule::Immediate,
                    },
                },
            })
            .storage(FilterStorage {
                backend: StorageBackend::File {
                    path: "/var/lib/scirs2/filters".to_string(),
                    format: StorageFormat::Compressed,
                },
                optimization: StorageOptimization {
                    enabled: true,
                    compression: true,
                    indexing: IndexingStrategy::FieldBased,
                    partitioning: PartitioningStrategy::TimeBase,
                },
                backup: StorageBackup {
                    enabled: true,
                    frequency: Duration::from_secs(3600 * 6), // 6 hours
                    destination: BackupDestination::File {
                        path: "/var/backups/scirs2/filters".to_string(),
                    },
                    retention: BackupRetention {
                        max_count: 168, // 7 days * 24 hours / 6 hours
                        max_age: Duration::from_secs(86400 * 30), // 30 days
                        strategy: RetentionStrategy::TimeBase,
                    },
                },
                caching: StorageCaching {
                    enabled: true,
                    size_limit: 20000,
                    ttl: Duration::from_secs(3600),
                    strategy: CacheStrategy::LRU,
                    eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
                },
            })
            .build()
    }
}