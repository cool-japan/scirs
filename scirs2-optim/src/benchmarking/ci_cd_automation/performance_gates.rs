//! Performance Gates and Monitoring
//!
//! This module provides comprehensive performance gate evaluation, monitoring,
//! and validation capabilities for CI/CD automation, ensuring performance
//! regressions are caught before deployment.

use crate::benchmarking::performance_regression_detector::{
    MetricType, MetricValue, PerformanceMeasurement,
};
use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::config::{
    PerformanceGatesConfig, MetricGate, GateType, ComparisonOperator,
    GateSeverity, GateEvaluationStrategy, GateFailureHandling,
    GateFailureAction, GateFailureNotificationConfig
};
use super::test_execution::{CiCdTestResult, TestSuiteStatistics};

/// Performance gate evaluator
#[derive(Debug, Clone)]
pub struct PerformanceGateEvaluator {
    /// Gate configuration
    pub config: PerformanceGatesConfig,
    /// Baseline metrics for comparison
    pub baseline_metrics: HashMap<MetricType, BaselineMetric>,
    /// Gate evaluation history
    pub evaluation_history: Vec<GateEvaluationResult>,
    /// Current gate states
    pub gate_states: HashMap<MetricType, GateState>,
    /// Performance trend analyzer
    pub trend_analyzer: PerformanceTrendAnalyzer,
    /// Alert manager
    pub alert_manager: AlertManager,
}

/// Baseline metric for gate comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetric {
    /// Metric type
    pub metric_type: MetricType,
    /// Baseline value
    pub baseline_value: f64,
    /// Acceptable variance (percentage)
    pub variance_threshold: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Sample size used for baseline
    pub sample_size: usize,
    /// Last updated timestamp
    pub last_updated: SystemTime,
    /// Statistical significance
    pub statistical_significance: f64,
}

/// Current state of a performance gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateState {
    /// Gate metric type
    pub metric_type: MetricType,
    /// Current gate status
    pub status: GateStatus,
    /// Current metric value
    pub current_value: f64,
    /// Baseline value for comparison
    pub baseline_value: f64,
    /// Percentage change from baseline
    pub percentage_change: f64,
    /// Gate evaluation timestamp
    pub evaluated_at: SystemTime,
    /// Gate configuration used
    pub gate_config: MetricGate,
    /// Evaluation details
    pub evaluation_details: GateEvaluationDetails,
}

/// Gate status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GateStatus {
    /// Gate passed successfully
    Passed,
    /// Gate failed
    Failed,
    /// Gate evaluation warning
    Warning,
    /// Gate was skipped
    Skipped,
    /// Gate evaluation error
    Error,
    /// Gate not evaluated yet
    NotEvaluated,
}

/// Detailed gate evaluation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateEvaluationDetails {
    /// Evaluation method used
    pub evaluation_method: String,
    /// Statistical test results
    pub statistical_tests: Vec<StatisticalTestResult>,
    /// Confidence level
    pub confidence_level: f64,
    /// P-value (for statistical tests)
    pub p_value: Option<f64>,
    /// Effect size
    pub effect_size: Option<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    /// Test name
    pub test_name: String,
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: Option<u32>,
    /// Test conclusion
    pub conclusion: TestConclusion,
}

/// Statistical test conclusion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestConclusion {
    /// Significant difference detected
    Significant,
    /// No significant difference
    NotSignificant,
    /// Inconclusive result
    Inconclusive,
}

/// Gate evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateEvaluationResult {
    /// Evaluation ID
    pub evaluation_id: String,
    /// Evaluation timestamp
    pub timestamp: SystemTime,
    /// Overall evaluation status
    pub overall_status: OverallGateStatus,
    /// Individual gate results
    pub gate_results: Vec<IndividualGateResult>,
    /// Evaluation summary
    pub summary: GateEvaluationSummary,
    /// Actions taken
    pub actions_taken: Vec<GateAction>,
    /// Evaluation duration
    pub evaluation_duration: Duration,
}

/// Overall gate evaluation status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OverallGateStatus {
    /// All gates passed
    AllPassed,
    /// Some gates failed
    SomeFailed,
    /// All gates failed
    AllFailed,
    /// Gates were skipped
    Skipped,
    /// Evaluation error occurred
    Error,
}

/// Individual gate evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndividualGateResult {
    /// Metric type
    pub metric_type: MetricType,
    /// Gate status
    pub status: GateStatus,
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Percentage deviation
    pub percentage_deviation: f64,
    /// Gate severity
    pub severity: GateSeverity,
    /// Failure reason (if failed)
    pub failure_reason: Option<String>,
    /// Evaluation details
    pub details: GateEvaluationDetails,
}

/// Gate evaluation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateEvaluationSummary {
    /// Total gates evaluated
    pub total_gates: usize,
    /// Gates passed
    pub gates_passed: usize,
    /// Gates failed
    pub gates_failed: usize,
    /// Gates with warnings
    pub gates_warnings: usize,
    /// Gates skipped
    pub gates_skipped: usize,
    /// Critical failures
    pub critical_failures: usize,
    /// Performance improvement detected
    pub improvements_detected: usize,
    /// Regression severity
    pub regression_severity: RegressionSeverity,
}

/// Regression severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegressionSeverity {
    /// No regression detected
    None,
    /// Minor regression
    Minor,
    /// Moderate regression
    Moderate,
    /// Major regression
    Major,
    /// Critical regression
    Critical,
}

/// Actions taken during gate evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateAction {
    /// Action type
    pub action_type: GateActionType,
    /// Action description
    pub description: String,
    /// Action timestamp
    pub timestamp: SystemTime,
    /// Action result
    pub result: ActionResult,
}

/// Types of gate actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GateActionType {
    /// Send notification
    SendNotification,
    /// Create issue/ticket
    CreateIssue,
    /// Fail build
    FailBuild,
    /// Mark as unstable
    MarkUnstable,
    /// Log warning
    LogWarning,
    /// Update baseline
    UpdateBaseline,
    /// Override gate
    OverrideGate,
}

/// Result of a gate action
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionResult {
    /// Action succeeded
    Success,
    /// Action failed
    Failed,
    /// Action was skipped
    Skipped,
}

/// Performance trend analyzer for gate evaluation
#[derive(Debug, Clone)]
pub struct PerformanceTrendAnalyzer {
    /// Historical performance data
    pub performance_history: Vec<PerformanceDataPoint>,
    /// Trend analysis configuration
    pub config: TrendAnalysisConfig,
    /// Detected trends
    pub detected_trends: HashMap<MetricType, PerformanceTrend>,
}

/// Performance data point for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Metric type
    pub metric_type: MetricType,
    /// Metric value
    pub value: f64,
    /// Context information
    pub context: HashMap<String, String>,
    /// Data quality score
    pub quality_score: f64,
}

/// Trend analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisConfig {
    /// Minimum data points for trend analysis
    pub min_data_points: usize,
    /// Analysis window size
    pub window_size: usize,
    /// Trend detection sensitivity
    pub sensitivity: f64,
    /// Confidence level for trends
    pub confidence_level: f64,
    /// Enable seasonal adjustment
    pub enable_seasonal_adjustment: bool,
}

/// Detected performance trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Metric type
    pub metric_type: MetricType,
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Trend significance
    pub significance: f64,
    /// Trend slope
    pub slope: f64,
    /// Trend duration
    pub duration: Duration,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Last updated
    pub last_updated: SystemTime,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Degrading trend
    Degrading,
    /// Stable trend
    Stable,
    /// Volatile/unclear trend
    Volatile,
}

/// Alert manager for performance gates
#[derive(Debug, Clone)]
pub struct AlertManager {
    /// Alert configuration
    pub config: AlertConfiguration,
    /// Active alerts
    pub active_alerts: HashMap<String, PerformanceAlert>,
    /// Alert history
    pub alert_history: Vec<PerformanceAlert>,
    /// Alert escalation rules
    pub escalation_rules: Vec<EscalationRule>,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfiguration {
    /// Enable alerts
    pub enabled: bool,
    /// Alert severity threshold
    pub severity_threshold: GateSeverity,
    /// Alert cooldown period
    pub cooldown_period: Duration,
    /// Maximum alerts per hour
    pub max_alerts_per_hour: u32,
    /// Alert channels
    pub alert_channels: Vec<AlertChannel>,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert ID
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert title
    pub title: String,
    /// Alert description
    pub description: String,
    /// Affected metrics
    pub affected_metrics: Vec<MetricType>,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert status
    pub status: AlertStatus,
    /// Related gate evaluation
    pub gate_evaluation_id: Option<String>,
    /// Alert metadata
    pub metadata: HashMap<String, String>,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlertType {
    /// Performance regression alert
    PerformanceRegression,
    /// Threshold breach alert
    ThresholdBreach,
    /// Trend change alert
    TrendChange,
    /// Gate failure alert
    GateFailure,
    /// System anomaly alert
    SystemAnomaly,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Error alert
    Error,
    /// Critical alert
    Critical,
}

/// Alert status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlertStatus {
    /// Alert is active
    Active,
    /// Alert has been acknowledged
    Acknowledged,
    /// Alert has been resolved
    Resolved,
    /// Alert has been suppressed
    Suppressed,
}

/// Alert channels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlertChannel {
    /// Email alerts
    Email,
    /// Slack alerts
    Slack,
    /// Webhook alerts
    Webhook,
    /// Dashboard alerts
    Dashboard,
    /// Log alerts
    Log,
}

/// Alert escalation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRule {
    /// Rule name
    pub name: String,
    /// Trigger condition
    pub trigger_condition: EscalationTrigger,
    /// Escalation delay
    pub delay: Duration,
    /// Target alert channels
    pub target_channels: Vec<AlertChannel>,
    /// Target recipients
    pub target_recipients: Vec<String>,
    /// Rule enabled
    pub enabled: bool,
}

/// Escalation triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationTrigger {
    /// Time-based escalation
    TimeElapsed(Duration),
    /// Severity-based escalation
    SeverityLevel(AlertSeverity),
    /// Multiple failures
    MultipleFailures(u32),
    /// Custom condition
    Custom(String),
}

/// Gate evaluation result for public API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// Overall gate status
    pub status: OverallGateStatus,
    /// Individual gate results
    pub gate_results: Vec<IndividualGateResult>,
    /// Gate evaluation summary
    pub summary: GateEvaluationSummary,
    /// Evaluation timestamp
    pub timestamp: SystemTime,
}

impl PerformanceGateEvaluator {
    /// Create a new performance gate evaluator
    pub fn new(config: PerformanceGatesConfig) -> Result<Self> {
        Ok(Self {
            config,
            baseline_metrics: HashMap::new(),
            evaluation_history: Vec::new(),
            gate_states: HashMap::new(),
            trend_analyzer: PerformanceTrendAnalyzer::new(TrendAnalysisConfig::default())?,
            alert_manager: AlertManager::new(AlertConfiguration::default())?,
        })
    }

    /// Evaluate performance gates against test results
    pub fn evaluate_gates(
        &mut self,
        test_results: &[CiCdTestResult],
        statistics: &TestSuiteStatistics,
    ) -> Result<GateResult> {
        let evaluation_id = uuid::Uuid::new_v4().to_string();
        let start_time = SystemTime::now();

        // Extract metrics from test results
        let current_metrics = self.extract_metrics_from_results(test_results)?;

        // Update trend analyzer with new data
        self.update_trend_analysis(&current_metrics)?;

        // Evaluate individual gates
        let mut gate_results = Vec::new();
        let mut overall_status = OverallGateStatus::AllPassed;

        for (metric_type, gate_config) in &self.config.metric_gates {
            if !gate_config.enabled {
                continue;
            }

            if let Some(current_value) = current_metrics.get(metric_type) {
                let gate_result = self.evaluate_individual_gate(
                    metric_type,
                    *current_value,
                    gate_config,
                )?;

                if gate_result.status == GateStatus::Failed {
                    overall_status = match overall_status {
                        OverallGateStatus::AllPassed => OverallGateStatus::SomeFailed,
                        OverallGateStatus::SomeFailed => OverallGateStatus::SomeFailed,
                        _ => OverallGateStatus::AllFailed,
                    };
                }

                gate_results.push(gate_result);
            }
        }

        // Create evaluation summary
        let summary = self.create_evaluation_summary(&gate_results);

        // Handle gate failures
        let actions_taken = if overall_status != OverallGateStatus::AllPassed {
            self.handle_gate_failures(&gate_results)?
        } else {
            Vec::new()
        };

        // Create evaluation result
        let evaluation_result = GateEvaluationResult {
            evaluation_id: evaluation_id.clone(),
            timestamp: start_time,
            overall_status: overall_status.clone(),
            gate_results: gate_results.clone(),
            summary: summary.clone(),
            actions_taken,
            evaluation_duration: SystemTime::now().duration_since(start_time).unwrap_or_default(),
        };

        // Store evaluation result
        self.evaluation_history.push(evaluation_result);

        // Update gate states
        self.update_gate_states(&gate_results)?;

        // Generate alerts if necessary
        self.generate_alerts(&gate_results)?;

        Ok(GateResult {
            status: overall_status,
            gate_results,
            summary,
            timestamp: start_time,
        })
    }

    /// Extract performance metrics from test results
    fn extract_metrics_from_results(&self, test_results: &[CiCdTestResult]) -> Result<HashMap<MetricType, f64>> {
        let mut metrics = HashMap::new();

        // Calculate aggregate metrics
        if !test_results.is_empty() {
            // Execution time metrics
            let execution_times: Vec<f64> = test_results
                .iter()
                .filter_map(|r| r.duration)
                .map(|d| d.as_secs_f64())
                .collect();

            if !execution_times.is_empty() {
                let avg_execution_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
                metrics.insert(MetricType::ExecutionTime, avg_execution_time);
            }

            // Memory usage metrics
            let memory_usage: Vec<f64> = test_results
                .iter()
                .map(|r| r.resource_usage.peak_memory_mb)
                .collect();

            if !memory_usage.is_empty() {
                let avg_memory = memory_usage.iter().sum::<f64>() / memory_usage.len() as f64;
                metrics.insert(MetricType::MemoryUsage, avg_memory);
            }

            // CPU usage metrics
            let cpu_usage: Vec<f64> = test_results
                .iter()
                .map(|r| r.resource_usage.peak_cpu_percent)
                .collect();

            if !cpu_usage.is_empty() {
                let avg_cpu = cpu_usage.iter().sum::<f64>() / cpu_usage.len() as f64;
                metrics.insert(MetricType::CpuUsage, avg_cpu);
            }

            // Throughput metrics (simplified calculation)
            let total_duration: f64 = execution_times.iter().sum();
            if total_duration > 0.0 {
                let throughput = test_results.len() as f64 / total_duration;
                metrics.insert(MetricType::Throughput, throughput);
            }
        }

        Ok(metrics)
    }

    /// Evaluate an individual performance gate
    fn evaluate_individual_gate(
        &self,
        metric_type: &MetricType,
        current_value: f64,
        gate_config: &MetricGate,
    ) -> Result<IndividualGateResult> {
        let baseline_value = self.get_baseline_value(metric_type);
        let threshold_value = self.calculate_threshold(baseline_value, gate_config);

        let status = match gate_config.gate_type {
            GateType::Absolute => self.evaluate_absolute_gate(current_value, threshold_value, &gate_config.operator),
            GateType::Relative => self.evaluate_relative_gate(current_value, baseline_value, gate_config.threshold, &gate_config.operator),
            GateType::Statistical => self.evaluate_statistical_gate(metric_type, current_value, baseline_value)?,
            GateType::Trend => self.evaluate_trend_gate(metric_type, current_value)?,
        };

        let percentage_deviation = if baseline_value != 0.0 {
            ((current_value - baseline_value) / baseline_value) * 100.0
        } else {
            0.0
        };

        let failure_reason = if status == GateStatus::Failed {
            Some(self.create_failure_reason(metric_type, current_value, threshold_value, &gate_config.operator))
        } else {
            None
        };

        Ok(IndividualGateResult {
            metric_type: metric_type.clone(),
            status,
            current_value,
            threshold_value,
            percentage_deviation,
            severity: gate_config.severity.clone(),
            failure_reason,
            details: self.create_evaluation_details(metric_type, current_value, baseline_value)?,
        })
    }

    /// Evaluate absolute threshold gate
    fn evaluate_absolute_gate(&self, current_value: f64, threshold: f64, operator: &ComparisonOperator) -> GateStatus {
        let passed = match operator {
            ComparisonOperator::LessThan => current_value < threshold,
            ComparisonOperator::LessThanOrEqual => current_value <= threshold,
            ComparisonOperator::GreaterThan => current_value > threshold,
            ComparisonOperator::GreaterThanOrEqual => current_value >= threshold,
            ComparisonOperator::Equal => (current_value - threshold).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (current_value - threshold).abs() >= f64::EPSILON,
        };

        if passed {
            GateStatus::Passed
        } else {
            GateStatus::Failed
        }
    }

    /// Evaluate relative threshold gate
    fn evaluate_relative_gate(&self, current_value: f64, baseline_value: f64, threshold: f64, operator: &ComparisonOperator) -> GateStatus {
        if baseline_value == 0.0 {
            return GateStatus::Error;
        }

        let percentage_change = ((current_value - baseline_value) / baseline_value).abs();

        let passed = match operator {
            ComparisonOperator::LessThanOrEqual => percentage_change <= threshold,
            ComparisonOperator::LessThan => percentage_change < threshold,
            _ => false, // Other operators not typically used for relative gates
        };

        if passed {
            GateStatus::Passed
        } else {
            GateStatus::Failed
        }
    }

    /// Evaluate statistical significance gate
    fn evaluate_statistical_gate(&self, metric_type: &MetricType, current_value: f64, baseline_value: f64) -> Result<GateStatus> {
        // Simplified statistical test - in reality, this would use proper statistical methods
        let difference = (current_value - baseline_value).abs();
        let relative_difference = if baseline_value != 0.0 {
            difference / baseline_value
        } else {
            difference
        };

        // Simple threshold-based evaluation (would use t-test, Mann-Whitney U, etc. in practice)
        if relative_difference > 0.1 { // 10% difference threshold
            Ok(GateStatus::Failed)
        } else {
            Ok(GateStatus::Passed)
        }
    }

    /// Evaluate trend-based gate
    fn evaluate_trend_gate(&self, metric_type: &MetricType, current_value: f64) -> Result<GateStatus> {
        if let Some(trend) = self.trend_analyzer.detected_trends.get(metric_type) {
            match trend.direction {
                TrendDirection::Degrading if trend.strength > 0.7 => Ok(GateStatus::Failed),
                TrendDirection::Degrading if trend.strength > 0.4 => Ok(GateStatus::Warning),
                _ => Ok(GateStatus::Passed),
            }
        } else {
            Ok(GateStatus::Passed)
        }
    }

    /// Get baseline value for metric
    fn get_baseline_value(&self, metric_type: &MetricType) -> f64 {
        self.baseline_metrics
            .get(metric_type)
            .map(|baseline| baseline.baseline_value)
            .unwrap_or(0.0)
    }

    /// Calculate threshold value based on gate configuration
    fn calculate_threshold(&self, baseline_value: f64, gate_config: &MetricGate) -> f64 {
        match gate_config.gate_type {
            GateType::Absolute => gate_config.threshold,
            GateType::Relative => baseline_value * (1.0 + gate_config.threshold),
            _ => gate_config.threshold,
        }
    }

    /// Create failure reason message
    fn create_failure_reason(&self, metric_type: &MetricType, current_value: f64, threshold: f64, operator: &ComparisonOperator) -> String {
        format!(
            "{:?} value {:.2} does not satisfy {:?} {:.2}",
            metric_type, current_value, operator, threshold
        )
    }

    /// Create detailed evaluation information
    fn create_evaluation_details(&self, metric_type: &MetricType, current_value: f64, baseline_value: f64) -> Result<GateEvaluationDetails> {
        Ok(GateEvaluationDetails {
            evaluation_method: "threshold_comparison".to_string(),
            statistical_tests: Vec::new(), // Would include actual test results
            confidence_level: 0.95,
            p_value: None,
            effect_size: Some((current_value - baseline_value).abs()),
            metadata: HashMap::new(),
        })
    }

    /// Create evaluation summary
    fn create_evaluation_summary(&self, gate_results: &[IndividualGateResult]) -> GateEvaluationSummary {
        let total_gates = gate_results.len();
        let gates_passed = gate_results.iter().filter(|r| r.status == GateStatus::Passed).count();
        let gates_failed = gate_results.iter().filter(|r| r.status == GateStatus::Failed).count();
        let gates_warnings = gate_results.iter().filter(|r| r.status == GateStatus::Warning).count();
        let gates_skipped = gate_results.iter().filter(|r| r.status == GateStatus::Skipped).count();
        let critical_failures = gate_results.iter()
            .filter(|r| r.status == GateStatus::Failed && r.severity == GateSeverity::Critical)
            .count();
        let improvements_detected = gate_results.iter()
            .filter(|r| r.percentage_deviation < -5.0) // 5% improvement threshold
            .count();

        let regression_severity = if critical_failures > 0 {
            RegressionSeverity::Critical
        } else if gates_failed > total_gates / 2 {
            RegressionSeverity::Major
        } else if gates_failed > 0 {
            RegressionSeverity::Moderate
        } else {
            RegressionSeverity::None
        };

        GateEvaluationSummary {
            total_gates,
            gates_passed,
            gates_failed,
            gates_warnings,
            gates_skipped,
            critical_failures,
            improvements_detected,
            regression_severity,
        }
    }

    /// Handle gate failures
    fn handle_gate_failures(&mut self, gate_results: &[IndividualGateResult]) -> Result<Vec<GateAction>> {
        let mut actions = Vec::new();

        // Determine action based on configuration
        let action_type = match self.config.failure_handling.failure_action {
            GateFailureAction::FailBuild => GateActionType::FailBuild,
            GateFailureAction::MarkUnstable => GateActionType::MarkUnstable,
            GateFailureAction::LogWarning => GateActionType::LogWarning,
            GateFailureAction::NotifyOnly => GateActionType::SendNotification,
        };

        let action = GateAction {
            action_type,
            description: format!("Gate failure action triggered for {} failed gates",
                gate_results.iter().filter(|r| r.status == GateStatus::Failed).count()),
            timestamp: SystemTime::now(),
            result: ActionResult::Success, // Simplified
        };

        actions.push(action);

        // Send notifications if configured
        if self.config.failure_handling.notifications.send_email ||
           self.config.failure_handling.notifications.send_slack ||
           self.config.failure_handling.notifications.send_webhooks {
            let notification_action = GateAction {
                action_type: GateActionType::SendNotification,
                description: "Notification sent for gate failures".to_string(),
                timestamp: SystemTime::now(),
                result: ActionResult::Success,
            };
            actions.push(notification_action);
        }

        Ok(actions)
    }

    /// Update gate states
    fn update_gate_states(&mut self, gate_results: &[IndividualGateResult]) -> Result<()> {
        for result in gate_results {
            let baseline_value = self.get_baseline_value(&result.metric_type);

            let gate_state = GateState {
                metric_type: result.metric_type.clone(),
                status: result.status.clone(),
                current_value: result.current_value,
                baseline_value,
                percentage_change: result.percentage_deviation,
                evaluated_at: SystemTime::now(),
                gate_config: self.config.metric_gates.get(&result.metric_type)
                    .cloned()
                    .unwrap_or_else(|| MetricGate {
                        gate_type: GateType::Absolute,
                        threshold: 0.0,
                        operator: ComparisonOperator::LessThanOrEqual,
                        severity: GateSeverity::Warning,
                        enabled: true,
                    }),
                evaluation_details: result.details.clone(),
            };

            self.gate_states.insert(result.metric_type.clone(), gate_state);
        }

        Ok(())
    }

    /// Update trend analysis with new metrics
    fn update_trend_analysis(&mut self, metrics: &HashMap<MetricType, f64>) -> Result<()> {
        let timestamp = SystemTime::now();

        for (metric_type, value) in metrics {
            let data_point = PerformanceDataPoint {
                timestamp,
                metric_type: metric_type.clone(),
                value: *value,
                context: HashMap::new(),
                quality_score: 1.0,
            };

            self.trend_analyzer.add_data_point(data_point)?;
        }

        // Update detected trends
        self.trend_analyzer.analyze_trends()?;

        Ok(())
    }

    /// Generate alerts for gate failures
    fn generate_alerts(&mut self, gate_results: &[IndividualGateResult]) -> Result<()> {
        for result in gate_results {
            if result.status == GateStatus::Failed {
                let alert_severity = match result.severity {
                    GateSeverity::Critical => AlertSeverity::Critical,
                    GateSeverity::Error => AlertSeverity::Error,
                    GateSeverity::Warning => AlertSeverity::Warning,
                    GateSeverity::Info => AlertSeverity::Info,
                };

                let alert = PerformanceAlert {
                    id: uuid::Uuid::new_v4().to_string(),
                    alert_type: AlertType::GateFailure,
                    severity: alert_severity,
                    title: format!("Performance Gate Failed: {:?}", result.metric_type),
                    description: result.failure_reason.clone().unwrap_or_else(|| "Gate failed".to_string()),
                    affected_metrics: vec![result.metric_type.clone()],
                    timestamp: SystemTime::now(),
                    status: AlertStatus::Active,
                    gate_evaluation_id: None,
                    metadata: HashMap::new(),
                };

                self.alert_manager.add_alert(alert)?;
            }
        }

        Ok(())
    }

    /// Set baseline metric
    pub fn set_baseline(&mut self, metric_type: MetricType, baseline: BaselineMetric) {
        self.baseline_metrics.insert(metric_type, baseline);
    }

    /// Get gate evaluation history
    pub fn get_evaluation_history(&self) -> &[GateEvaluationResult] {
        &self.evaluation_history
    }

    /// Get current gate states
    pub fn get_gate_states(&self) -> &HashMap<MetricType, GateState> {
        &self.gate_states
    }
}

impl PerformanceTrendAnalyzer {
    /// Create a new trend analyzer
    pub fn new(config: TrendAnalysisConfig) -> Result<Self> {
        Ok(Self {
            performance_history: Vec::new(),
            config,
            detected_trends: HashMap::new(),
        })
    }

    /// Add a performance data point
    pub fn add_data_point(&mut self, data_point: PerformanceDataPoint) -> Result<()> {
        self.performance_history.push(data_point);

        // Keep only recent data points
        let max_points = self.config.window_size * 2;
        if self.performance_history.len() > max_points {
            self.performance_history.drain(0..self.performance_history.len() - max_points);
        }

        Ok(())
    }

    /// Analyze trends in performance data
    pub fn analyze_trends(&mut self) -> Result<()> {
        let metrics: std::collections::HashSet<MetricType> = self.performance_history
            .iter()
            .map(|dp| dp.metric_type.clone())
            .collect();

        for metric_type in metrics {
            let metric_data: Vec<&PerformanceDataPoint> = self.performance_history
                .iter()
                .filter(|dp| dp.metric_type == metric_type)
                .collect();

            if metric_data.len() >= self.config.min_data_points {
                let trend = self.detect_trend(&metric_data)?;
                self.detected_trends.insert(metric_type, trend);
            }
        }

        Ok(())
    }

    /// Detect trend in metric data
    fn detect_trend(&self, data: &[&PerformanceDataPoint]) -> Result<PerformanceTrend> {
        let values: Vec<f64> = data.iter().map(|dp| dp.value).collect();

        // Simple linear regression for trend detection
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = x_values.iter().zip(values.iter()).map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        let direction = if slope > self.config.sensitivity {
            TrendDirection::Improving
        } else if slope < -self.config.sensitivity {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        let strength = slope.abs().min(1.0);

        Ok(PerformanceTrend {
            metric_type: data[0].metric_type.clone(),
            direction,
            strength,
            significance: 0.8, // Simplified
            slope,
            duration: data.last().unwrap().timestamp.duration_since(data[0].timestamp).unwrap_or_default(),
            confidence_interval: (slope - 0.1, slope + 0.1), // Simplified
            last_updated: SystemTime::now(),
        })
    }
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new(config: AlertConfiguration) -> Result<Self> {
        Ok(Self {
            config,
            active_alerts: HashMap::new(),
            alert_history: Vec::new(),
            escalation_rules: Vec::new(),
        })
    }

    /// Add a new alert
    pub fn add_alert(&mut self, alert: PerformanceAlert) -> Result<()> {
        self.active_alerts.insert(alert.id.clone(), alert.clone());
        self.alert_history.push(alert);
        Ok(())
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&mut self, alert_id: &str) -> Result<()> {
        if let Some(alert) = self.active_alerts.get_mut(alert_id) {
            alert.status = AlertStatus::Acknowledged;
        }
        Ok(())
    }

    /// Resolve an alert
    pub fn resolve_alert(&mut self, alert_id: &str) -> Result<()> {
        if let Some(alert) = self.active_alerts.remove(alert_id) {
            // Move to history with resolved status
            let mut resolved_alert = alert;
            resolved_alert.status = AlertStatus::Resolved;
            self.alert_history.push(resolved_alert);
        }
        Ok(())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<&PerformanceAlert> {
        self.active_alerts.values().collect()
    }
}

// Default implementations

impl Default for TrendAnalysisConfig {
    fn default() -> Self {
        Self {
            min_data_points: 10,
            window_size: 50,
            sensitivity: 0.1,
            confidence_level: 0.95,
            enable_seasonal_adjustment: false,
        }
    }
}

impl Default for AlertConfiguration {
    fn default() -> Self {
        Self {
            enabled: true,
            severity_threshold: GateSeverity::Warning,
            cooldown_period: Duration::from_secs(300), // 5 minutes
            max_alerts_per_hour: 10,
            alert_channels: vec![AlertChannel::Log],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_evaluator_creation() {
        let config = PerformanceGatesConfig::default();
        let evaluator = PerformanceGateEvaluator::new(config);
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_gate_status_equality() {
        assert_eq!(GateStatus::Passed, GateStatus::Passed);
        assert_ne!(GateStatus::Passed, GateStatus::Failed);
    }

    #[test]
    fn test_trend_direction() {
        assert_eq!(TrendDirection::Improving, TrendDirection::Improving);
        assert_ne!(TrendDirection::Improving, TrendDirection::Degrading);
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Error);
        assert!(AlertSeverity::Error < AlertSeverity::Critical);
    }

    #[test]
    fn test_baseline_metric() {
        let baseline = BaselineMetric {
            metric_type: MetricType::ExecutionTime,
            baseline_value: 1.0,
            variance_threshold: 0.1,
            confidence_interval: (0.9, 1.1),
            sample_size: 100,
            last_updated: SystemTime::now(),
            statistical_significance: 0.95,
        };

        assert_eq!(baseline.baseline_value, 1.0);
        assert_eq!(baseline.sample_size, 100);
    }
}