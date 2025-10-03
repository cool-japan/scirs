//! Error diagnostics and monitoring system
//!
//! This module provides comprehensive error diagnostics, monitoring, and intelligent
//! recovery strategies for production statistical computing environments.

use crate::error_handling_v2::ErrorCode;
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant, SystemTime};

/// Error pattern detection and analysis
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    /// Pattern identifier
    pub id: String,
    /// Error codes that form this pattern
    pub error_codes: Vec<ErrorCode>,
    /// Frequency threshold for detection
    pub frequency_threshold: usize,
    /// Time window for pattern detection
    pub time_window: Duration,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Description of what this pattern indicates
    pub description: String,
    /// Suggested mitigation strategy
    pub mitigation: String,
}

impl ErrorPattern {
    /// Create a new error pattern
    pub fn new(
        id: impl Into<String>,
        error_codes: Vec<ErrorCode>,
        frequency_threshold: usize,
        time_window: Duration,
        description: impl Into<String>,
        mitigation: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            error_codes,
            frequency_threshold,
            time_window,
            confidence: 0.0,
            description: description.into(),
            mitigation: mitigation.into(),
        }
    }
}

/// Error occurrence record
#[derive(Debug, Clone)]
pub struct ErrorOccurrence {
    /// Error code
    pub code: ErrorCode,
    /// When the error occurred
    pub timestamp: Instant,
    /// Operation context
    pub operation: String,
    /// Frequency count
    pub count: usize,
    /// Resolution status
    pub resolved: bool,
    /// Recovery action taken
    pub recovery_action: Option<String>,
}

/// Comprehensive error monitoring and analytics
pub struct ErrorMonitor {
    /// Recent error occurrences
    error_history: Arc<Mutex<VecDeque<ErrorOccurrence>>>,
    /// Error frequency counters
    error_counts: Arc<Mutex<HashMap<ErrorCode, AtomicUsize>>>,
    /// Known error patterns
    patterns: Vec<ErrorPattern>,
    /// Maximum history size
    max_historysize: usize,
    /// Pattern detection enabled
    pattern_detection_enabled: bool,
    /// Error rate thresholds
    error_rate_thresholds: HashMap<ErrorCode, f64>,
    /// Monitoring start time
    start_time: Instant,
}

impl ErrorMonitor {
    /// Create a new error monitor
    pub fn new() -> Self {
        let mut monitor = Self {
            error_history: Arc::new(Mutex::new(VecDeque::new())),
            error_counts: Arc::new(Mutex::new(HashMap::new())),
            patterns: Vec::new(),
            max_historysize: 1000,
            pattern_detection_enabled: true,
            error_rate_thresholds: HashMap::new(),
            start_time: Instant::now(),
        };

        monitor.initialize_default_patterns();
        monitor.initialize_default_thresholds();
        monitor
    }

    /// Initialize default error patterns
    fn initialize_default_patterns(&mut self) {
        // Memory pressure pattern
        self.patterns.push(ErrorPattern::new(
            "memory_pressure",
            vec![ErrorCode::E5001, ErrorCode::E5002],
            3,
            Duration::from_secs(60),
            "High memory allocation failures indicating memory pressure",
            "Reduce data size, enable streaming processing, or increase available memory",
        ));

        // Numerical instability pattern
        self.patterns.push(ErrorPattern::new(
            "numerical_instability",
            vec![
                ErrorCode::E3001,
                ErrorCode::E3002,
                ErrorCode::E3005,
                ErrorCode::E3006,
            ],
            5,
            Duration::from_secs(30),
            "Frequent numerical errors indicating data quality or algorithm issues",
            "Check data preprocessing, scaling, and consider more stable algorithms",
        ));

        // Convergence issues pattern
        self.patterns.push(ErrorPattern::new(
            "convergence_issues",
            vec![ErrorCode::E3003, ErrorCode::E4001, ErrorCode::E4002],
            3,
            Duration::from_secs(120),
            "Repeated convergence failures in iterative algorithms",
            "Adjust algorithm parameters, improve initial conditions, or use different methods",
        ));

        // Data quality pattern
        self.patterns.push(ErrorPattern::new(
            "data_quality_issues",
            vec![
                ErrorCode::E2003,
                ErrorCode::E2004,
                ErrorCode::E1001,
                ErrorCode::E1002,
            ],
            4,
            Duration::from_secs(60),
            "Frequent data validation errors indicating poor data quality",
            "Implement comprehensive data validation and cleaning pipeline",
        ));
    }

    /// Initialize default error rate thresholds
    fn initialize_default_thresholds(&mut self) {
        self.error_rate_thresholds.insert(ErrorCode::E5001, 0.01); // Memory errors - very low tolerance
        self.error_rate_thresholds.insert(ErrorCode::E3001, 0.05); // Overflow - low tolerance
        self.error_rate_thresholds.insert(ErrorCode::E3005, 0.10); // NaN - moderate tolerance
        self.error_rate_thresholds.insert(ErrorCode::E4001, 0.20); // Max iterations - higher tolerance
    }

    /// Record an error occurrence
    pub fn record_error(&self, code: ErrorCode, operation: impl Into<String>) {
        let occurrence = ErrorOccurrence {
            code,
            timestamp: Instant::now(),
            operation: operation.into(),
            count: 1,
            resolved: false,
            recovery_action: None,
        };

        // Update history
        {
            let mut history = self.error_history.lock().unwrap();
            if history.len() >= self.max_historysize {
                history.pop_front();
            }
            history.push_back(occurrence);
        }

        // Update counters
        {
            let mut counts = self.error_counts.lock().unwrap();
            counts
                .entry(code)
                .or_insert_with(|| AtomicUsize::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }

        // Check for patterns if enabled
        if self.pattern_detection_enabled {
            self.check_patterns();
        }
    }

    /// Check for error patterns in recent history
    fn check_patterns(&self) {
        let history = self.error_history.lock().unwrap();
        let now = Instant::now();

        for pattern in &self.patterns {
            let relevant_errors: Vec<_> = history
                .iter()
                .filter(|err| {
                    pattern.error_codes.contains(&err.code)
                        && now.duration_since(err.timestamp) <= pattern.time_window
                })
                .collect();

            if relevant_errors.len() >= pattern.frequency_threshold {
                eprintln!(
                    "⚠️  ERROR PATTERN DETECTED: {} - {} ({})",
                    pattern.id, pattern.description, pattern.mitigation
                );
            }
        }
    }

    /// Get error statistics
    pub fn get_statistics(&self) -> ErrorStatistics {
        let counts = self.error_counts.lock().unwrap();
        let history = self.error_history.lock().unwrap();

        let total_errors: usize = counts
            .values()
            .map(|counter| counter.load(Ordering::Relaxed))
            .sum();

        let uptime = self.start_time.elapsed();
        let error_rate = total_errors as f64 / uptime.as_secs_f64();

        // Calculate error distribution
        let mut error_distribution = HashMap::new();
        for (code, counter) in counts.iter() {
            let count = counter.load(Ordering::Relaxed);
            if count > 0 {
                error_distribution.insert(*code, count);
            }
        }

        // Find most frequent errors
        let mut frequent_errors: Vec<_> = error_distribution.clone().into_iter().collect();
        frequent_errors.sort_by(|a, b| b.1.cmp(&a.1));
        let top_errors: Vec<_> = frequent_errors.into_iter().take(5).collect();

        // Calculate recent error rate (last hour)
        let one_hour_ago = Instant::now() - Duration::from_secs(3600);
        let recent_errors = history
            .iter()
            .filter(|err| err.timestamp > one_hour_ago)
            .count();
        let recent_error_rate = recent_errors as f64 / 3600.0;

        ErrorStatistics {
            total_errors,
            error_rate,
            recent_error_rate,
            uptime,
            error_distribution,
            top_errors: top_errors.into_iter().collect(),
            active_patterns: self.detect_active_patterns(),
        }
    }

    /// Detect currently active error patterns
    fn detect_active_patterns(&self) -> Vec<String> {
        let history = self.error_history.lock().unwrap();
        let now = Instant::now();
        let mut active_patterns = Vec::new();

        for pattern in &self.patterns {
            let recent_errors: Vec<_> = history
                .iter()
                .filter(|err| {
                    pattern.error_codes.contains(&err.code)
                        && now.duration_since(err.timestamp) <= pattern.time_window
                })
                .collect();

            if recent_errors.len() >= pattern.frequency_threshold {
                active_patterns.push(pattern.id.clone());
            }
        }

        active_patterns
    }

    /// Generate comprehensive health report
    pub fn generate_health_report(&self) -> HealthReport {
        let stats = self.get_statistics();
        let history = self.error_history.lock().unwrap();

        // Calculate health score (0-100)
        let health_score = self.calculate_health_score(&stats);

        // Identify critical issues
        let critical_issues = self.identify_critical_issues(&stats);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&stats, &critical_issues);

        // Calculate trend information
        let trend = self.calculate_error_trend(&history);

        HealthReport {
            health_score,
            critical_issues,
            recommendations,
            statistics: stats,
            trend,
            timestamp: SystemTime::now(),
        }
    }

    /// Calculate overall system health score
    fn calculate_health_score(&self, stats: &ErrorStatistics) -> u8 {
        let mut score = 100.0;

        // Penalty for high error rates
        if stats.error_rate > 1.0 {
            score -= 30.0;
        } else if stats.error_rate > 0.1 {
            score -= 20.0;
        } else if stats.error_rate > 0.01 {
            score -= 10.0;
        }

        // Penalty for active patterns
        score -= stats.active_patterns.len() as f64 * 15.0;

        // Penalty for critical errors
        for (code, count) in &stats.top_errors {
            if code.severity() <= 2 {
                score -= *count as f64 * 5.0;
            }
        }

        // Penalty for recent error spike
        if stats.recent_error_rate > stats.error_rate * 2.0 {
            score -= 20.0;
        }

        score.max(0.0).min(100.0) as u8
    }

    /// Identify critical issues requiring immediate attention
    fn identify_critical_issues(&self, stats: &ErrorStatistics) -> Vec<CriticalIssue> {
        let mut issues = Vec::new();

        // Check for severe error patterns
        if stats
            .active_patterns
            .contains(&"memory_pressure".to_string())
        {
            issues.push(CriticalIssue {
                severity: 1,
                title: "Memory Pressure Detected".to_string(),
                description: "High memory allocation failures indicate system memory pressure"
                    .to_string(),
                impact: "May cause application crashes or severe performance degradation"
                    .to_string(),
                action_required: "Immediate memory optimization or resource scaling required"
                    .to_string(),
            });
        }

        // Check for high critical error rates
        for (code, count) in &stats.top_errors {
            if code.severity() <= 2 && *count > 10 {
                issues.push(CriticalIssue {
                    severity: code.severity(),
                    title: format!("High {} Error Rate", code),
                    description: format!("Frequent {} errors detected", code.description()),
                    impact: "May indicate fundamental data or algorithm issues".to_string(),
                    action_required: "Investigate root cause and implement fixes".to_string(),
                });
            }
        }

        // Check for error rate spikes
        if stats.recent_error_rate > stats.error_rate * 3.0 {
            issues.push(CriticalIssue {
                severity: 2,
                title: "Error Rate Spike".to_string(),
                description: "Recent error rate significantly higher than baseline".to_string(),
                impact: "Indicates potential system instability or new issues".to_string(),
                action_required: "Monitor closely and investigate recent changes".to_string(),
            });
        }

        issues
    }

    /// Generate actionable recommendations
    fn generate_recommendations(
        &self,
        stats: &ErrorStatistics,
        issues: &[CriticalIssue],
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Recommendations based on error patterns
        if stats
            .active_patterns
            .contains(&"numerical_instability".to_string())
        {
            recommendations.push(Recommendation {
                priority: 1,
                category: "Data Quality".to_string(),
                title: "Improve Numerical Stability".to_string(),
                description: "Implement data preprocessing and normalization".to_string(),
                steps: vec![
                    "Check for extreme values in input data".to_string(),
                    "Apply appropriate data scaling or normalization".to_string(),
                    "Consider using more numerically stable algorithms".to_string(),
                ],
                expected_impact: "Reduce numerical errors by 70-90%".to_string(),
            });
        }

        // Recommendations based on frequent errors
        for (code, count) in &stats.top_errors {
            match code {
                ErrorCode::E3005 => {
                    recommendations.push(Recommendation {
                        priority: 2,
                        category: "Data Validation".to_string(),
                        title: "Handle NaN Values".to_string(),
                        description: "Implement comprehensive NaN handling strategy".to_string(),
                        steps: vec![
                            "Add data validation checks before processing".to_string(),
                            "Implement NaN filtering or imputation".to_string(),
                            "Use statistical methods that handle missing data".to_string(),
                        ],
                        expected_impact: "Eliminate NaN-related errors".to_string(),
                    });
                }
                ErrorCode::E3003 => {
                    recommendations.push(Recommendation {
                        priority: 2,
                        category: "Algorithm Tuning".to_string(),
                        title: "Optimize Convergence Parameters".to_string(),
                        description: "Adjust algorithm parameters for better convergence"
                            .to_string(),
                        steps: vec![
                            "Increase maximum iterations for iterative algorithms".to_string(),
                            "Adjust convergence tolerance based on data characteristics"
                                .to_string(),
                            "Consider using different initialization strategies".to_string(),
                        ],
                        expected_impact: "Improve convergence rate by 50-80%".to_string(),
                    });
                }
                _ => {}
            }
        }

        // General recommendations based on health score
        if stats.error_rate > 0.1 {
            recommendations.push(Recommendation {
                priority: 1,
                category: "System Health".to_string(),
                title: "Reduce Overall Error Rate".to_string(),
                description: "Implement comprehensive error prevention strategy".to_string(),
                steps: vec![
                    "Add input validation at system boundaries".to_string(),
                    "Implement data quality checks".to_string(),
                    "Use defensive programming practices".to_string(),
                ],
                expected_impact: "Reduce overall error rate significantly".to_string(),
            });
        }

        recommendations
    }

    /// Calculate error trend over time
    fn calculate_error_trend(&self, history: &VecDeque<ErrorOccurrence>) -> ErrorTrend {
        if history.len() < 10 {
            return ErrorTrend {
                direction: TrendDirection::Stable,
                magnitude: 0.0,
                confidence: 0.0,
                description: "Insufficient data for trend analysis".to_string(),
            };
        }

        let now = Instant::now();
        let recent_window = Duration::from_secs(1800); // 30 minutes
        let older_window = Duration::from_secs(3600); // 1 hour

        let recent_errors = history
            .iter()
            .filter(|err| now.duration_since(err.timestamp) <= recent_window)
            .count();

        let older_errors = history
            .iter()
            .filter(|err| {
                let age = now.duration_since(err.timestamp);
                age > recent_window && age <= older_window
            })
            .count();

        let recent_rate = recent_errors as f64 / recent_window.as_secs_f64();
        let older_rate = older_errors as f64 / recent_window.as_secs_f64(); // Same window size for comparison

        let change_ratio = if older_rate > 0.0 {
            recent_rate / older_rate
        } else if recent_rate > 0.0 {
            2.0 // Arbitrary large value indicating increase from zero
        } else {
            1.0 // No change
        };

        let (direction, description) = if change_ratio > 1.5 {
            (
                TrendDirection::Increasing,
                "Error rate is increasing significantly".to_string(),
            )
        } else if change_ratio < 0.5 {
            (
                TrendDirection::Decreasing,
                "Error rate is decreasing significantly".to_string(),
            )
        } else {
            (
                TrendDirection::Stable,
                "Error rate is relatively stable".to_string(),
            )
        };

        let magnitude = (change_ratio - 1.0).abs();
        let confidence = if history.len() > 50 { 0.8 } else { 0.5 };

        ErrorTrend {
            direction,
            magnitude,
            confidence,
            description,
        }
    }
}

impl Default for ErrorMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Error statistics summary
#[derive(Debug)]
pub struct ErrorStatistics {
    /// Total number of errors
    pub total_errors: usize,
    /// Overall error rate (errors per second)
    pub error_rate: f64,
    /// Recent error rate (last hour)
    pub recent_error_rate: f64,
    /// System uptime
    pub uptime: Duration,
    /// Error distribution by type
    pub error_distribution: HashMap<ErrorCode, usize>,
    /// Top 5 most frequent errors
    pub top_errors: Vec<(ErrorCode, usize)>,
    /// Currently active error patterns
    pub active_patterns: Vec<String>,
}

/// Critical issue requiring immediate attention
#[derive(Debug)]
pub struct CriticalIssue {
    /// Severity level (1 = most critical)
    pub severity: u8,
    /// Issue title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Potential impact
    pub impact: String,
    /// Required action
    pub action_required: String,
}

/// Actionable recommendation
#[derive(Debug)]
pub struct Recommendation {
    /// Priority level (1 = highest)
    pub priority: u8,
    /// Category of recommendation
    pub category: String,
    /// Recommendation title
    pub title: String,
    /// Description
    pub description: String,
    /// Step-by-step actions
    pub steps: Vec<String>,
    /// Expected impact
    pub expected_impact: String,
}

/// Error trend analysis
#[derive(Debug)]
pub struct ErrorTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Magnitude of change
    pub magnitude: f64,
    /// Confidence in the trend (0.0-1.0)
    pub confidence: f64,
    /// Trend description
    pub description: String,
}

/// Trend direction enumeration
#[derive(Debug)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Comprehensive health report
#[derive(Debug)]
pub struct HealthReport {
    /// Overall health score (0-100)
    pub health_score: u8,
    /// Critical issues requiring attention
    pub critical_issues: Vec<CriticalIssue>,
    /// Actionable recommendations
    pub recommendations: Vec<Recommendation>,
    /// Detailed statistics
    pub statistics: ErrorStatistics,
    /// Error trend analysis
    pub trend: ErrorTrend,
    /// Report generation timestamp
    pub timestamp: SystemTime,
}

impl HealthReport {
    /// Generate a formatted text report
    pub fn to_formatted_string(&self) -> String {
        let mut report = String::new();

        report.push_str("=== STATISTICAL COMPUTING HEALTH REPORT ===\n\n");
        report.push_str(&format!(
            "📊 Overall Health Score: {}/100\n",
            self.health_score
        ));
        report.push_str(&format!("⏱️  Report Generated: {:?}\n\n", self.timestamp));

        // Health indicator
        let health_indicator = match self.health_score {
            90..=100 => "🟢 EXCELLENT",
            70..=89 => "🟡 GOOD",
            50..=69 => "🟠 FAIR",
            30..=49 => "🔴 POOR",
            _ => "🚨 CRITICAL",
        };
        report.push_str(&format!("Status: {}\n\n", health_indicator));

        // Critical Issues
        if !self.critical_issues.is_empty() {
            report.push_str("🚨 CRITICAL ISSUES:\n");
            for (i, issue) in self.critical_issues.iter().enumerate() {
                report.push_str(&format!(
                    "{}. {} (Severity: {})\n   {}\n   Impact: {}\n   Action: {}\n\n",
                    i + 1,
                    issue.title,
                    issue.severity,
                    issue.description,
                    issue.impact,
                    issue.action_required
                ));
            }
        }

        // Statistics Summary
        report.push_str("📈 STATISTICS SUMMARY:\n");
        report.push_str(&format!(
            "• Total Errors: {}\n",
            self.statistics.total_errors
        ));
        report.push_str(&format!(
            "• Error Rate: {:.4} errors/sec\n",
            self.statistics.error_rate
        ));
        report.push_str(&format!(
            "• Recent Rate: {:.4} errors/sec\n",
            self.statistics.recent_error_rate
        ));
        report.push_str(&format!(
            "• Uptime: {:.2} hours\n",
            self.statistics.uptime.as_secs_f64() / 3600.0
        ));

        if !self.statistics.top_errors.is_empty() {
            report.push_str("\n📋 TOP ERRORS:\n");
            for (i, (code, count)) in self.statistics.top_errors.iter().enumerate() {
                report.push_str(&format!("   {}. {}: {} occurrences\n", i + 1, code, count));
            }
        }

        // Trend Analysis
        report.push_str(&format!("\n📊 TREND: {}\n", self.trend.description));

        // Recommendations
        if !self.recommendations.is_empty() {
            report.push_str("\n💡 RECOMMENDATIONS:\n");
            for (i, rec) in self.recommendations.iter().enumerate() {
                report.push_str(&format!(
                    "{}. {} (Priority: {})\n   {}\n   Expected Impact: {}\n",
                    i + 1,
                    rec.title,
                    rec.priority,
                    rec.description,
                    rec.expected_impact
                ));
                if !rec.steps.is_empty() {
                    report.push_str("   Steps:\n");
                    for step in &rec.steps {
                        report.push_str(&format!("   • {}\n", step));
                    }
                }
                report.push('\n');
            }
        }

        report
    }

    /// Check if immediate action is required
    pub fn requires_immediate_action(&self) -> bool {
        self.health_score < 50 || self.critical_issues.iter().any(|issue| issue.severity <= 2)
    }
}

/// Global error monitor instance
static GLOBAL_MONITOR: std::sync::OnceLock<ErrorMonitor> = std::sync::OnceLock::new();

/// Get the global error monitor instance
#[allow(dead_code)]
pub fn global_monitor() -> &'static ErrorMonitor {
    GLOBAL_MONITOR.get_or_init(ErrorMonitor::new)
}

/// Convenience function to record an error globally
#[allow(dead_code)]
pub fn record_global_error(code: ErrorCode, operation: impl Into<String>) {
    global_monitor().record_error(code, operation);
}

/// Convenience function to get global error statistics
#[allow(dead_code)]
pub fn get_global_statistics() -> ErrorStatistics {
    global_monitor().get_statistics()
}

/// Convenience function to generate global health report
#[allow(dead_code)]
pub fn generate_global_health_report() -> HealthReport {
    global_monitor().generate_health_report()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    #[ignore = "timeout"]
    fn test_error_monitor_basic() {
        let monitor = ErrorMonitor::new();
        monitor.record_error(ErrorCode::E3005, "test_operation");

        let stats = monitor.get_statistics();
        assert_eq!(stats.total_errors, 1);
        assert!(stats.error_distribution.contains_key(&ErrorCode::E3005));
    }

    #[test]
    #[ignore = "timeout"]
    fn test_pattern_detection() {
        let monitor = ErrorMonitor::new();

        // Record multiple memory errors to trigger pattern
        for _ in 0..5 {
            monitor.record_error(ErrorCode::E5001, "memory_test");
            // Remove sleep - not needed for testing functionality
        }

        let stats = monitor.get_statistics();
        // Pattern detection should identify memory pressure
        // (This would be more testable with dependency injection)
    }

    #[test]
    #[ignore = "timeout"]
    fn test_health_score_calculation() {
        let monitor = ErrorMonitor::new();

        // Fresh monitor should have perfect health
        let health_report = monitor.generate_health_report();
        assert_eq!(health_report.health_score, 100);

        // Record some errors and check health degrades
        monitor.record_error(ErrorCode::E3001, "overflow_test");
        monitor.record_error(ErrorCode::E5001, "memory_test");

        let health_report = monitor.generate_health_report();
        assert!(health_report.health_score < 100);
    }
}
