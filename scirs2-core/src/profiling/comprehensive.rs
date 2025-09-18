//! Comprehensive profiling integration that combines application profiling with system monitoring

use crate::profiling::advanced::{
    BottleneckConfig, BottleneckDetector, BottleneckReport, FlameGraphGenerator,
};
use crate::profiling::flame_graph_svg::{
    EnhancedFlameGraph, SvgFlameGraphConfig, SvgFlameGraphGenerator,
};
use crate::profiling::profiler::Profiler;
use crate::profiling::system_monitor::{
    AlertConfig, SystemAlert, SystemAlerter, SystemMonitor, SystemMonitorConfig,
};
use crate::profiling::timer::Timer;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Comprehensive profiling session that combines multiple profiling techniques
pub struct ComprehensiveProfiler {
    /// Application profiler
    app_profiler: Arc<Mutex<Profiler>>,
    /// System resource monitor
    systemmonitor: SystemMonitor,
    /// System alerter
    system_alerter: SystemAlerter,
    /// Flame graph generator
    flame_graph_generator: FlameGraphGenerator,
    /// Session start time
    session_start: Instant,
    /// Session configuration
    config: ComprehensiveConfig,
}

/// Configuration for comprehensive profiling
#[derive(Debug, Clone)]
pub struct ComprehensiveConfig {
    /// System monitoring configuration
    pub systemconfig: SystemMonitorConfig,
    /// Alert configuration
    pub alertconfig: AlertConfig,
    /// SVG flame graph configuration
    pub svgconfig: SvgFlameGraphConfig,
    /// Enable automatic bottleneck detection
    pub enable_bottleneck_detection: bool,
    /// Enable automatic alert notifications
    pub enable_alerts: bool,
    /// Enable flame graph generation
    pub enable_flame_graphs: bool,
    /// Session name for reports
    pub session_name: String,
}

impl Default for ComprehensiveConfig {
    fn default() -> Self {
        Self {
            systemconfig: SystemMonitorConfig::default(),
            alertconfig: AlertConfig::default(),
            svgconfig: SvgFlameGraphConfig::default(),
            enable_bottleneck_detection: true,
            enable_alerts: true,
            enable_flame_graphs: true,
            session_name: "Profiling Session".to_string(),
        }
    }
}

impl ComprehensiveProfiler {
    /// Create a new comprehensive profiler
    pub fn new(config: ComprehensiveConfig) -> Self {
        Self {
            app_profiler: Arc::new(Mutex::new(Profiler::new())),
            systemmonitor: SystemMonitor::new(config.systemconfig.clone()),
            system_alerter: SystemAlerter::new(config.alertconfig.clone()),
            flame_graph_generator: FlameGraphGenerator::new(),
            session_start: Instant::now(),
            config,
        }
    }

    /// Start comprehensive profiling
    pub fn start(&mut self) -> Result<(), crate::profiling::system_monitor::SystemMonitorError> {
        // Start application profiler
        self.app_profiler.lock().unwrap().start();

        // Start system monitor
        self.systemmonitor.start()?;

        self.session_start = Instant::now();
        Ok(())
    }

    /// Stop comprehensive profiling
    pub fn stop(&mut self) {
        self.app_profiler.lock().unwrap().stop();
        self.systemmonitor.stop();
    }

    /// Time a function with comprehensive profiling
    pub fn time_function<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Start flame graph tracking
        self.flame_graph_generator.start_call(name);

        // Execute with application profiling
        let result = Timer::time_function(name, f);

        // End flame graph tracking
        self.flame_graph_generator.end_call();

        // Check for alerts if enabled
        if self.config.enable_alerts {
            if let Ok(current_metrics) = self.systemmonitor.get_current_metrics() {
                let alerts = self.system_alerter.check_alerts(&current_metrics);
                for alert in alerts {
                    self.handle_alert(&alert);
                }
            }
        }

        result
    }

    /// Generate comprehensive profiling report
    pub fn generate_report(&mut self) -> ComprehensiveReport {
        let app_report = self.app_profiler.lock().unwrap().get_report();
        let system_metrics = self.systemmonitor.get_metrics_history();
        let alerts = self.system_alerter.get_alert_history();

        let mut bottleneck_reports = Vec::new();
        if self.config.enable_bottleneck_detection {
            let mut detector = BottleneckDetector::new(BottleneckConfig::default());
            bottleneck_reports = detector.analyze(&self.app_profiler.lock().unwrap());
        }

        let flame_graph = if self.config.enable_flame_graphs {
            Some(self.flame_graph_generator.generate())
        } else {
            None
        };

        ComprehensiveReport {
            session_name: self.config.session_name.clone(),
            session_duration: self.session_start.elapsed(),
            application_report: app_report,
            system_metrics,
            alerts,
            bottleneck_reports,
            flame_graph,
            generated_at: Instant::now(),
        }
    }

    /// Export comprehensive report to multiple formats
    pub fn export_report(&mut self, basepath: &str) -> Result<(), std::io::Error> {
        let report = self.generate_report();

        // Export text report
        std::fs::write(format!("{basepath}_report.txt"), report.totext_format())?;

        // Export JSON report
        std::fs::write(format!("{basepath}_report.json"), report.to_json_format())?;

        // Export flame graph if available
        if let Some(ref flame_graph) = report.flame_graph {
            let svg_generator = SvgFlameGraphGenerator::new(self.config.svgconfig.clone());
            svg_generator.export_to_file(flame_graph, &format!("{basepath}_flamegraph.svg"))?;

            // Export enhanced flame graph with system metrics
            let enhanced = EnhancedFlameGraph {
                performance: flame_graph.clone(),
                memory: None,
                cpu_usage: report
                    .system_metrics
                    .iter()
                    .map(|m| (m.timestamp.duration_since(self.session_start), m.cpu_usage))
                    .collect(),
                memory_usage: report
                    .system_metrics
                    .iter()
                    .map(|m| {
                        (
                            m.timestamp.duration_since(self.session_start),
                            m.memory_usage,
                        )
                    })
                    .collect(),
                total_duration: self.session_start.elapsed(),
            };
            enhanced.export_enhanced_svg(&format!("{basepath}_enhanced_flamegraph.svg"))?;
        }

        Ok(())
    }

    /// Handle system alerts
    fn handle_alert(&self, alert: &SystemAlert) {
        // In a real implementation, this could send notifications, log to files, etc.
        println!("ALERT: {}", alert.message);
    }

    /// Get application profiler reference
    pub fn app_profiler(&self) -> Arc<Mutex<Profiler>> {
        Arc::clone(&self.app_profiler)
    }

    /// Get current system metrics
    pub fn get_current_system_metrics(
        &self,
    ) -> Result<
        crate::profiling::system_monitor::SystemMetrics,
        crate::profiling::system_monitor::SystemMonitorError,
    > {
        self.systemmonitor.get_current_metrics()
    }

    /// Get recent alerts
    pub fn get_recent_alerts(&self, duration: Duration) -> Vec<SystemAlert> {
        self.system_alerter.get_recent_alerts(duration)
    }
}

impl Drop for ComprehensiveProfiler {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Comprehensive profiling report
#[derive(Debug)]
pub struct ComprehensiveReport {
    /// Session name
    pub session_name: String,
    /// Total session duration
    pub session_duration: Duration,
    /// Application profiling report
    pub application_report: String,
    /// System metrics collected during session
    pub system_metrics: Vec<crate::profiling::system_monitor::SystemMetrics>,
    /// System alerts triggered during session
    pub alerts: Vec<SystemAlert>,
    /// Bottleneck analysis results
    pub bottleneck_reports: Vec<BottleneckReport>,
    /// Flame graph data
    pub flame_graph: Option<crate::profiling::advanced::FlameGraphNode>,
    /// Report generation timestamp
    pub generated_at: Instant,
}

impl ComprehensiveReport {
    /// Convert report to text format
    pub fn totext_format(&self) -> String {
        use std::fmt::Write;
        let mut report = String::new();

        writeln!(report, "=== {} ===", self.session_name).unwrap();
        writeln!(
            report,
            "Session Duration: {:.2} seconds",
            self.session_duration.as_secs_f64()
        )
        .unwrap();
        writeln!(report, "Generated At: {:?}", self.generated_at).unwrap();
        writeln!(report).unwrap();

        // Application profiling
        writeln!(report, "=== Application Performance ===").unwrap();
        writeln!(report, "{}", self.application_report).unwrap();

        // System metrics summary
        if !self.system_metrics.is_empty() {
            writeln!(report, "=== System Resource Summary ===").unwrap();
            let avg_cpu = self.system_metrics.iter().map(|m| m.cpu_usage).sum::<f64>()
                / self.system_metrics.len() as f64;
            let avg_memory = self
                .system_metrics
                .iter()
                .map(|m| m.memory_usage)
                .sum::<usize>()
                / self.system_metrics.len();
            let max_cpu = self
                .system_metrics
                .iter()
                .map(|m| m.cpu_usage)
                .fold(0.0, f64::max);
            let max_memory = self
                .system_metrics
                .iter()
                .map(|m| m.memory_usage)
                .max()
                .unwrap_or(0);

            writeln!(report, "Average CPU Usage: {avg_cpu:.1}%").unwrap();
            writeln!(report, "Maximum CPU Usage: {max_cpu:.1}%").unwrap();
            writeln!(
                report,
                "Average Memory Usage: {:.1} MB",
                avg_memory as f64 / (1024.0 * 1024.0)
            )
            .unwrap();
            writeln!(
                report,
                "Maximum Memory Usage: {:.1} MB",
                max_memory as f64 / (1024.0 * 1024.0)
            )
            .unwrap();
            writeln!(report).unwrap();
        }

        // Alerts
        if !self.alerts.is_empty() {
            writeln!(report, "=== System Alerts ({}) ===", self.alerts.len()).unwrap();
            for alert in &self.alerts {
                writeln!(report, "[{:?}] {}", alert.severity, alert.message).unwrap();
            }
            writeln!(report).unwrap();
        }

        // Bottlenecks
        if !self.bottleneck_reports.is_empty() {
            writeln!(
                report,
                "=== Performance Bottlenecks ({}) ===",
                self.bottleneck_reports.len()
            )
            .unwrap();
            for bottleneck in &self.bottleneck_reports {
                writeln!(report, "Operation: {}", bottleneck.operation).unwrap();
                writeln!(report, "Type: {:?}", bottleneck.bottleneck_type).unwrap();
                writeln!(report, "Severity: {:.2}", bottleneck.severity).unwrap();
                writeln!(report, "Description: {}", bottleneck.description).unwrap();
                if !bottleneck.suggestions.is_empty() {
                    writeln!(report, "Suggestions:").unwrap();
                    for suggestion in &bottleneck.suggestions {
                        writeln!(report, "  - {suggestion}").unwrap();
                    }
                }
                writeln!(report).unwrap();
            }
        }

        report
    }

    /// Convert report to JSON format
    pub fn to_json_format(&self) -> String {
        // Simplified JSON generation - in a real implementation would use serde
        use std::fmt::Write;
        let mut json = String::new();

        writeln!(json, "{{").unwrap();
        writeln!(json, "  \"session_name\": \"{}\",", self.session_name).unwrap();
        writeln!(
            json,
            "  \"session_duration_seconds\": {},",
            self.session_duration.as_secs_f64()
        )
        .unwrap();
        writeln!(json, "  \"alert_count\": {},", self.alerts.len()).unwrap();
        writeln!(
            json,
            "  \"bottleneck_count\": {},",
            self.bottleneck_reports.len()
        )
        .unwrap();
        writeln!(
            json,
            "  \"system_sample_count\": {}",
            self.system_metrics.len()
        )
        .unwrap();

        if !self.system_metrics.is_empty() {
            let avg_cpu = self.system_metrics.iter().map(|m| m.cpu_usage).sum::<f64>()
                / self.system_metrics.len() as f64;
            let max_cpu = self
                .system_metrics
                .iter()
                .map(|m| m.cpu_usage)
                .fold(0.0, f64::max);
            writeln!(json, "  \"average_cpu_usage\": {avg_cpu},").unwrap();
            writeln!(json, "  \"maximum_cpu_usage\": {max_cpu}").unwrap();
        }

        writeln!(json, "}}").unwrap();
        json
    }

    /// Print comprehensive report to console
    pub fn print(&self) {
        println!("{}", self.totext_format());
    }
}

/// Convenience macro for comprehensive profiling
#[macro_export]
macro_rules! comprehensive_profile {
    ($profiler:expr, $name:expr, $body:block) => {{
        $profiler.time_function($name, || $body)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_comprehensive_profiler() {
        let config = ComprehensiveConfig {
            session_name: "Test Session".to_string(),
            ..Default::default()
        };

        let mut profiler = ComprehensiveProfiler::new(config);
        profiler.start().unwrap();

        // Profile some work
        let result = profiler.time_function("test_work", || {
            thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);

        // Generate report
        let report = profiler.generate_report();
        assert_eq!(report.session_name, "Test Session");
        assert!(report.session_duration > Duration::from_millis(5));

        profiler.stop();
    }

    #[test]
    fn test_comprehensive_report() {
        let report = ComprehensiveReport {
            session_name: "Test".to_string(),
            session_duration: Duration::from_secs(1),
            application_report: "Test report".to_string(),
            system_metrics: Vec::new(),
            alerts: Vec::new(),
            bottleneck_reports: Vec::new(),
            flame_graph: None,
            generated_at: Instant::now(),
        };

        let text = report.totext_format();
        assert!(text.contains("Test"));

        let json = report.to_json_format();
        assert!(json.contains("session_name"));
    }
}
