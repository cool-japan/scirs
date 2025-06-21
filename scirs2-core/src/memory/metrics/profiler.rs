//! Memory profiler for real-time memory monitoring and analysis
//!
//! This module provides a comprehensive memory profiler that combines
//! real-time monitoring with advanced analytics and automated reporting.

use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::memory::metrics::{
    analytics::{LeakDetectionConfig, MemoryAnalytics},
    collector::MemoryMetricsCollector,
    MemoryEvent, MemoryReport,
};

#[cfg(test)]
use crate::memory::metrics::MemoryEventType;

#[cfg(feature = "memory_metrics")]
use serde::{Deserialize, Serialize};

/// Memory profiler configuration
#[derive(Debug, Clone)]
pub struct MemoryProfilerConfig {
    /// Whether the profiler is enabled
    pub enabled: bool,
    /// Profiling interval for periodic reporting
    pub profiling_interval: Duration,
    /// Whether to automatically detect memory leaks
    pub auto_leak_detection: bool,
    /// Whether to generate optimization recommendations
    pub auto_recommendations: bool,
    /// Whether to save profiling results to file
    pub save_to_file: bool,
    /// File path for saving results (if save_to_file is true)
    pub output_file_path: Option<String>,
    /// Maximum number of profiling reports to keep in memory
    pub max_reports_in_memory: usize,
    /// Whether to enable call stack capture
    pub capture_call_stacks: bool,
}

impl Default for MemoryProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            profiling_interval: Duration::from_secs(30),
            auto_leak_detection: true,
            auto_recommendations: true,
            save_to_file: false,
            output_file_path: None,
            max_reports_in_memory: 100,
            capture_call_stacks: cfg!(feature = "memory_call_stack"),
        }
    }
}

/// Profiling session information
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "memory_metrics",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ProfilingSession {
    /// Session identifier
    pub id: String,
    /// Session start time (timestamp in microseconds since epoch)
    pub start_time_micros: u64,
    /// Session duration in microseconds
    pub duration_micros: u64,
    /// Number of memory events recorded
    pub event_count: usize,
    /// Number of components tracked
    pub component_count: usize,
    /// Peak memory usage during session
    pub peak_memory_usage: usize,
    /// Whether any leaks were detected
    pub leaks_detected: bool,
}

/// Memory profiling result containing comprehensive analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub struct ProfilingResult {
    /// Session information
    pub session: ProfilingSession,
    /// Basic memory report
    pub memory_report: MemoryReport,
    /// Leak detection results
    pub leak_results: Vec<crate::memory::metrics::analytics::LeakDetectionResult>,
    /// Pattern analysis results
    pub pattern_analysis: Vec<crate::memory::metrics::analytics::MemoryPatternAnalysis>,
    /// Performance impact analysis
    pub performance_impact: PerformanceImpactAnalysis,
    /// Summary and recommendations
    pub summary: ProfilingSummary,
}

/// Performance impact analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub struct PerformanceImpactAnalysis {
    /// Total time spent in memory allocation operations
    pub total_allocation_time: Duration,
    /// Average allocation time
    pub avg_allocation_time: Duration,
    /// Number of potential performance bottlenecks
    pub performance_bottlenecks: usize,
    /// Memory bandwidth utilization estimate
    pub memory_bandwidth_utilization: f64,
    /// Cache miss estimate
    pub cache_miss_estimate: f64,
}

/// Profiling summary with key insights
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub struct ProfilingSummary {
    /// Overall memory health score (0.0 to 1.0)
    pub health_score: f64,
    /// Key insights discovered
    pub key_insights: Vec<String>,
    /// Priority recommendations
    pub priority_recommendations: Vec<String>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}

/// Risk assessment for memory usage
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub enum RiskAssessment {
    /// Low risk - memory usage is healthy
    Low,
    /// Medium risk - some issues detected but manageable
    Medium { issues: Vec<String> },
    /// High risk - critical issues that need immediate attention
    High { critical_issues: Vec<String> },
}

/// Memory profiler for comprehensive memory analysis
pub struct MemoryProfiler {
    /// Configuration
    config: MemoryProfilerConfig,
    /// Memory metrics collector
    collector: Arc<MemoryMetricsCollector>,
    /// Memory analytics engine
    analytics: Arc<Mutex<MemoryAnalytics>>,
    /// Profiling results history
    results_history: Arc<RwLock<Vec<ProfilingResult>>>,
    /// Current session information
    current_session: Arc<Mutex<Option<ProfilingSession>>>,
    /// Background thread handle
    _background_thread: Option<thread::JoinHandle<()>>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new(config: MemoryProfilerConfig) -> Self {
        let collector = Arc::new(MemoryMetricsCollector::new(
            crate::memory::metrics::MemoryMetricsConfig::default(),
        ));

        let analytics = Arc::new(Mutex::new(MemoryAnalytics::new(
            LeakDetectionConfig::default(),
        )));

        let results_history = Arc::new(RwLock::new(Vec::new()));
        let current_session = Arc::new(Mutex::new(None));

        let mut profiler = Self {
            config,
            collector,
            analytics,
            results_history,
            current_session,
            _background_thread: None,
        };

        // Start background profiling if enabled
        if profiler.config.enabled {
            profiler.start_background_profiling();
        }

        profiler
    }

    /// Start a new profiling session
    pub fn start_session(&self, session_id: Option<String>) -> String {
        let session_id = session_id.unwrap_or_else(|| {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            format!("session_{}", timestamp)
        });

        let now = SystemTime::now();
        let start_time_micros = now
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let session = ProfilingSession {
            id: session_id.clone(),
            start_time_micros,
            duration_micros: 0,
            event_count: 0,
            component_count: 0,
            peak_memory_usage: 0,
            leaks_detected: false,
        };

        {
            let mut current = self.current_session.lock().unwrap();
            *current = Some(session);
        }

        // Reset collector and analytics
        self.collector.reset();
        self.analytics.lock().unwrap().clear();

        session_id
    }

    /// End the current profiling session and generate results
    pub fn end_session(&self) -> Option<ProfilingResult> {
        let session = {
            let mut current = self.current_session.lock().unwrap();
            current.take()?
        };

        // Update session with final metrics
        let memory_report = self.collector.generate_report();
        let analytics = self.analytics.lock().unwrap();
        let leak_results = analytics.get_leak_detection_results();
        let pattern_analysis = analytics.get_pattern_analysis_results();

        let now_micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let duration_micros = now_micros.saturating_sub(session.start_time_micros);

        let updated_session = ProfilingSession {
            duration_micros,
            event_count: memory_report.total_allocation_count,
            component_count: memory_report.component_stats.len(),
            peak_memory_usage: memory_report.total_peak_usage,
            leaks_detected: leak_results.iter().any(|r| r.leak_detected),
            ..session
        };

        let performance_impact = self.analyze_performance_impact(&memory_report, &pattern_analysis);
        let summary = self.generate_summary(&memory_report, &leak_results, &pattern_analysis);

        let result = ProfilingResult {
            session: updated_session,
            memory_report,
            leak_results,
            pattern_analysis,
            performance_impact,
            summary,
        };

        // Store result in history
        {
            let mut history = self.results_history.write().unwrap();
            history.push(result.clone());

            // Limit history size
            while history.len() > self.config.max_reports_in_memory {
                history.remove(0);
            }
        }

        // Save to file if configured
        if self.config.save_to_file {
            if let Some(path) = &self.config.output_file_path {
                let _ = self.save_result_to_file(&result, path);
            }
        }

        Some(result)
    }

    /// Record a memory event for profiling
    pub fn record_event(&self, event: MemoryEvent) {
        // Record in collector
        self.collector.record_event(event.clone());

        // Record in analytics
        self.analytics.lock().unwrap().record_event(event);
    }

    /// Start background profiling thread
    fn start_background_profiling(&mut self) {
        if self._background_thread.is_some() {
            return; // Already running
        }

        let interval = self.config.profiling_interval;
        let collector = Arc::clone(&self.collector);
        let analytics = Arc::clone(&self.analytics);
        let _results_history = Arc::clone(&self.results_history);
        let _current_session = Arc::clone(&self.current_session);
        let config = self.config.clone();

        let handle = thread::spawn(move || {
            let mut last_report_time = Instant::now();

            loop {
                thread::sleep(Duration::from_secs(1));

                if last_report_time.elapsed() >= interval {
                    // Generate periodic report
                    let memory_report = collector.generate_report();
                    let analytics_guard = analytics.lock().unwrap();
                    let leak_results = analytics_guard.get_leak_detection_results();
                    let _pattern_analysis = analytics_guard.get_pattern_analysis_results();
                    drop(analytics_guard);

                    // Check for critical issues
                    let critical_leaks = leak_results
                        .iter()
                        .any(|r| r.leak_detected && r.confidence > 0.8);
                    let high_memory_usage = memory_report.total_current_usage > 1024 * 1024 * 1024; // 1 GB

                    if critical_leaks || high_memory_usage {
                        // Log critical issues
                        eprintln!("MEMORY PROFILER WARNING: Critical memory issues detected!");
                        if critical_leaks {
                            eprintln!("  - Memory leaks detected in components");
                        }
                        if high_memory_usage {
                            eprintln!(
                                "  - High memory usage: {} MB",
                                memory_report.total_current_usage / (1024 * 1024)
                            );
                        }
                    }

                    last_report_time = Instant::now();
                }

                // Check if we should continue running
                if !config.enabled {
                    break;
                }
            }
        });

        self._background_thread = Some(handle);
    }

    /// Analyze performance impact based on memory patterns
    fn analyze_performance_impact(
        &self,
        memory_report: &MemoryReport,
        pattern_analysis: &[crate::memory::metrics::analytics::MemoryPatternAnalysis],
    ) -> PerformanceImpactAnalysis {
        // Calculate performance metrics based on allocation patterns
        let total_allocations = memory_report.total_allocation_count;
        let duration = memory_report.duration;

        // Estimate allocation time (this would be more accurate with actual timing data)
        let avg_allocation_time = if total_allocations > 0 {
            Duration::from_nanos(100) // Estimate: 100ns per allocation
        } else {
            Duration::from_nanos(0)
        };

        let total_allocation_time = avg_allocation_time * total_allocations as u32;

        // Count performance bottlenecks
        let performance_bottlenecks = pattern_analysis
            .iter()
            .map(|analysis| {
                analysis.potential_issues
                    .iter()
                    .filter(|issue| matches!(
                        issue,
                        crate::memory::metrics::analytics::MemoryIssue::HighAllocationFrequency { .. }
                    ))
                    .count()
            })
            .sum();

        // Estimate memory bandwidth utilization (simplified)
        let bytes_per_second = if duration.as_secs() > 0 {
            memory_report.total_allocated_bytes as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        // Assume peak memory bandwidth of 100 GB/s (modern systems)
        let memory_bandwidth_utilization =
            (bytes_per_second / (100.0 * 1024.0 * 1024.0 * 1024.0)).min(1.0);

        // Estimate cache miss rate based on allocation patterns
        let cache_miss_estimate = pattern_analysis
            .iter()
            .map(|analysis| analysis.efficiency.fragmentation_estimate)
            .sum::<f64>()
            / pattern_analysis.len().max(1) as f64;

        PerformanceImpactAnalysis {
            total_allocation_time,
            avg_allocation_time,
            performance_bottlenecks,
            memory_bandwidth_utilization,
            cache_miss_estimate,
        }
    }

    /// Generate profiling summary with insights and recommendations
    fn generate_summary(
        &self,
        memory_report: &MemoryReport,
        leak_results: &[crate::memory::metrics::analytics::LeakDetectionResult],
        pattern_analysis: &[crate::memory::metrics::analytics::MemoryPatternAnalysis],
    ) -> ProfilingSummary {
        let mut health_score = 1.0;
        let mut key_insights = Vec::new();
        let mut priority_recommendations = Vec::new();
        let mut risk_issues = Vec::new();

        // Analyze memory health
        let total_memory_mb = memory_report.total_current_usage / (1024 * 1024);
        if total_memory_mb > 1000 {
            health_score -= 0.2;
            key_insights.push(format!(
                "High memory usage detected: {} MB",
                total_memory_mb
            ));
            priority_recommendations
                .push("Consider implementing memory optimization strategies".to_string());
        }

        // Check for memory leaks
        let critical_leaks = leak_results
            .iter()
            .filter(|r| r.leak_detected && r.confidence > 0.8)
            .count();
        if critical_leaks > 0 {
            health_score -= 0.3 * critical_leaks as f64;
            key_insights.push(format!(
                "{} potential memory leaks detected",
                critical_leaks
            ));
            priority_recommendations
                .push("Investigate and fix memory leaks immediately".to_string());
            risk_issues.push(format!("{} critical memory leaks", critical_leaks));
        }

        // Check allocation efficiency
        let avg_reuse_ratio = pattern_analysis
            .iter()
            .map(|p| p.efficiency.reuse_ratio)
            .sum::<f64>()
            / pattern_analysis.len().max(1) as f64;

        if avg_reuse_ratio > 5.0 {
            health_score -= 0.1;
            key_insights.push("Low memory reuse efficiency detected".to_string());
            priority_recommendations
                .push("Implement buffer pooling to improve memory reuse".to_string());
        }

        // Check allocation frequency
        let high_frequency_components = pattern_analysis
            .iter()
            .filter(|p| p.efficiency.allocation_frequency > 100.0)
            .count();

        if high_frequency_components > 0 {
            health_score -= 0.1;
            key_insights.push(format!(
                "{} components with high allocation frequency",
                high_frequency_components
            ));
            priority_recommendations
                .push("Consider batching allocations for better performance".to_string());
        }

        // Determine risk assessment
        let risk_assessment = if health_score > 0.8 {
            RiskAssessment::Low
        } else if health_score > 0.5 {
            RiskAssessment::Medium {
                issues: key_insights.clone(),
            }
        } else {
            RiskAssessment::High {
                critical_issues: risk_issues,
            }
        };

        // Add general insights
        if memory_report.component_stats.len() > 10 {
            key_insights
                .push("Large number of components tracked - consider consolidation".to_string());
        }

        if memory_report.duration.as_secs() < 60 {
            key_insights.push(
                "Short profiling duration - longer sessions provide better insights".to_string(),
            );
        }

        ProfilingSummary {
            health_score: health_score.max(0.0),
            key_insights,
            priority_recommendations,
            risk_assessment,
        }
    }

    /// Save profiling result to file
    fn save_result_to_file(
        &self,
        result: &ProfilingResult,
        file_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "memory_metrics")]
        {
            let json = serde_json::to_string_pretty(result)?;
            std::fs::write(file_path, json)?;
        }

        #[cfg(not(feature = "memory_metrics"))]
        {
            // Just write a simple summary
            let summary = format!(
                "Memory Profiling Session: {}\nDuration: {} micros\nPeak Usage: {} bytes\nLeaks Detected: {}\n",
                result.session.id,
                result.session.duration_micros,
                result.session.peak_memory_usage,
                result.session.leaks_detected
            );
            std::fs::write(file_path, summary)?;
        }

        Ok(())
    }

    /// Get profiling results history
    pub fn get_results_history(&self) -> Vec<ProfilingResult> {
        self.results_history.read().unwrap().clone()
    }

    /// Get current session information
    pub fn get_current_session(&self) -> Option<ProfilingSession> {
        self.current_session.lock().unwrap().clone()
    }

    /// Generate a quick health check report
    pub fn health_check(&self) -> ProfilingSummary {
        let memory_report = self.collector.generate_report();
        let analytics = self.analytics.lock().unwrap();
        let leak_results = analytics.get_leak_detection_results();
        let pattern_analysis = analytics.get_pattern_analysis_results();
        drop(analytics);

        self.generate_summary(&memory_report, &leak_results, &pattern_analysis)
    }

    /// Clear all profiling data
    pub fn clear_all_data(&self) {
        self.collector.reset();
        self.analytics.lock().unwrap().clear();
        self.results_history.write().unwrap().clear();
        *self.current_session.lock().unwrap() = None;
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new(MemoryProfilerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler_creation() {
        let profiler = MemoryProfiler::new(MemoryProfilerConfig::default());
        assert!(profiler.get_current_session().is_none());
        assert!(profiler.get_results_history().is_empty());
    }

    #[test]
    fn test_profiling_session_lifecycle() {
        let profiler = MemoryProfiler::new(MemoryProfilerConfig {
            enabled: false, // Disable background thread for testing
            ..Default::default()
        });

        // Start session
        let session_id = profiler.start_session(Some("test_session".to_string()));
        assert_eq!(session_id, "test_session");
        assert!(profiler.get_current_session().is_some());

        // Record some events
        let event = MemoryEvent::new(MemoryEventType::Allocation, "TestComponent", 1024, 0x1000);
        profiler.record_event(event);

        // End session
        std::thread::sleep(Duration::from_millis(10)); // Ensure some duration
        let result = profiler.end_session();
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.session.id, "test_session");
        assert!(result.session.duration_micros > 0);
        assert!(profiler.get_current_session().is_none());
    }

    #[test]
    fn test_health_check() {
        let profiler = MemoryProfiler::new(MemoryProfilerConfig {
            enabled: false,
            ..Default::default()
        });

        let health = profiler.health_check();
        assert!(health.health_score >= 0.0 && health.health_score <= 1.0);
    }

    #[test]
    fn test_performance_impact_analysis() {
        let profiler = MemoryProfiler::new(MemoryProfilerConfig {
            enabled: false,
            ..Default::default()
        });

        let memory_report = crate::memory::metrics::MemoryReport {
            total_current_usage: 1024,
            total_peak_usage: 2048,
            total_allocation_count: 10,
            total_allocated_bytes: 4096,
            component_stats: std::collections::HashMap::new(),
            duration: Duration::from_secs(60),
        };

        let pattern_analysis = Vec::new();
        let impact = profiler.analyze_performance_impact(&memory_report, &pattern_analysis);

        assert_eq!(impact.avg_allocation_time, Duration::from_nanos(100));
        assert_eq!(impact.performance_bottlenecks, 0);
    }
}
