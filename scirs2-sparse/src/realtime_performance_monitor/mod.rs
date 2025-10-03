//! Real-Time Performance Monitoring and Adaptation for Advanced Processors
//!
//! This module provides comprehensive real-time monitoring and adaptive optimization
//! for all Advanced mode processors, including quantum-inspired, neural-adaptive,
//! and hybrid processors.
//!
//! ## Architecture
//!
//! The real-time performance monitoring system consists of several interconnected components:
//!
//! - **Configuration**: Flexible configuration system for different monitoring scenarios
//! - **Metrics Collection**: Comprehensive performance metrics from various processor types
//! - **History Management**: Efficient storage and analysis of performance history data
//! - **Alert System**: Rule-based alerting with multiple notification channels
//! - **Real-Time Monitoring**: Continuous monitoring with adaptive sampling rates
//! - **Processor Registry**: Registration and management of different processor types
//!
//! ## Usage
//!
//! ```rust,ignore
//! use scirs2_sparse::realtime_performance_monitor::{
//!     RealTimePerformanceMonitor, PerformanceMonitorConfig
//! };
//!
//! // Create configuration
//! let config = PerformanceMonitorConfig::default()
//!     .with_monitoring_interval_ms(100)
//!     .with_adaptive_tuning(true)
//!     .with_alerts(true);
//!
//! // Create monitor
//! let monitor = RealTimePerformanceMonitor::new(config);
//!
//! // Start monitoring
//! monitor.start_monitoring()?;
//!
//! // Register processors for monitoring
//! // monitor.register_quantum_processor(my_quantum_processor)?;
//! // monitor.register_neural_processor(my_neural_processor)?;
//!
//! // Get monitoring data
//! let summary = monitor.get_monitoring_summary();
//! let alerts = monitor.get_active_alerts();
//! let recent_samples = monitor.get_recent_samples(100);
//! ```
//!
//! ## Performance Optimization
//!
//! The system automatically adapts monitoring frequency based on system load and provides
//! detailed performance analytics:
//!
//! ```rust,ignore
//! // Get processor performance summary
//! let processor_summary = monitor.get_processor_summary();
//! for summary in processor_summary {
//!     println!("Processor {}: {:.2} ops/sec, {:.1}% efficiency",
//!         summary.processor_id,
//!         summary.avg_throughput,
//!         summary.efficiency_score * 100.0
//!     );
//! }
//!
//! // Get system health metrics
//! if let Some(system_metrics) = monitor.get_system_metrics() {
//!     println!("System health score: {:.1}%", system_metrics.health_score() * 100.0);
//! }
//! ```

pub mod alerts;
pub mod config;
pub mod history;
pub mod metrics;
pub mod monitor;

// Re-export main types for convenience
pub use alerts::{
    Alert, AlertCondition, AlertManager, AlertRule, AlertSeverity, AlertStats, NotificationChannel,
};
pub use config::{PerformanceMonitorConfig, UseCase};
pub use history::{PerformanceHistory, PerformanceTrend, ProcessorSummary};
pub use metrics::{
    AggregatedMetrics, ExecutionTimer, PerformanceSample, ProcessorType, SystemMetrics,
};
pub use monitor::{
    HybridProcessorMonitor, MemoryCompressorMonitor, MonitoringSummary, NeuralProcessorMonitor,
    QuantumProcessorMonitor, RealTimePerformanceMonitor,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = PerformanceMonitorConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.monitoring_interval_ms, 100);
        assert!(config.adaptive_tuning);
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = PerformanceMonitorConfig::new()
            .with_monitoring_interval_ms(200)
            .with_adaptive_tuning(false)
            .with_alerts(true)
            .with_prediction(false);

        assert!(config.validate().is_ok());
        assert_eq!(config.monitoring_interval_ms, 200);
        assert!(!config.adaptive_tuning);
        assert!(config.enable_alerts);
        assert!(!config.enable_prediction);
    }

    #[test]
    fn test_predefined_configurations() {
        let lightweight = PerformanceMonitorConfig::lightweight();
        assert!(lightweight.validate().is_ok());
        assert_eq!(lightweight.monitoring_interval_ms, 500);
        assert!(!lightweight.adaptive_tuning);

        let high_perf = PerformanceMonitorConfig::high_performance();
        assert!(high_perf.validate().is_ok());
        assert_eq!(high_perf.monitoring_interval_ms, 50);
        assert!(high_perf.adaptive_tuning);

        let debug = PerformanceMonitorConfig::debug();
        assert!(debug.validate().is_ok());
        assert_eq!(debug.monitoring_interval_ms, 10);
        assert!(debug.adaptive_tuning);
    }

    #[test]
    fn test_use_case_configurations() {
        let production = PerformanceMonitorConfig::recommended_for_use_case(UseCase::Production);
        assert!(production.validate().is_ok());

        let development = PerformanceMonitorConfig::recommended_for_use_case(UseCase::Development);
        assert!(development.validate().is_ok());

        let testing = PerformanceMonitorConfig::recommended_for_use_case(UseCase::Testing);
        assert!(testing.validate().is_ok());

        let benchmarking =
            PerformanceMonitorConfig::recommended_for_use_case(UseCase::Benchmarking);
        assert!(benchmarking.validate().is_ok());

        let low_resource = PerformanceMonitorConfig::recommended_for_use_case(UseCase::LowResource);
        assert!(low_resource.validate().is_ok());
    }

    #[test]
    fn test_processor_type_enumeration() {
        let all_types = ProcessorType::all();
        assert_eq!(all_types.len(), 4);
        assert!(all_types.contains(&ProcessorType::QuantumInspired));
        assert!(all_types.contains(&ProcessorType::NeuralAdaptive));
        assert!(all_types.contains(&ProcessorType::QuantumNeuralHybrid));
        assert!(all_types.contains(&ProcessorType::MemoryCompression));
    }

    #[test]
    fn test_processor_type_capabilities() {
        assert!(ProcessorType::QuantumInspired.supports_quantum_metrics());
        assert!(!ProcessorType::QuantumInspired.supports_neural_metrics());
        assert!(!ProcessorType::QuantumInspired.supports_compression_metrics());

        assert!(!ProcessorType::NeuralAdaptive.supports_quantum_metrics());
        assert!(ProcessorType::NeuralAdaptive.supports_neural_metrics());
        assert!(!ProcessorType::NeuralAdaptive.supports_compression_metrics());

        assert!(ProcessorType::QuantumNeuralHybrid.supports_quantum_metrics());
        assert!(ProcessorType::QuantumNeuralHybrid.supports_neural_metrics());
        assert!(!ProcessorType::QuantumNeuralHybrid.supports_compression_metrics());

        assert!(!ProcessorType::MemoryCompression.supports_quantum_metrics());
        assert!(!ProcessorType::MemoryCompression.supports_neural_metrics());
        assert!(ProcessorType::MemoryCompression.supports_compression_metrics());
    }

    #[test]
    fn test_performance_sample_creation() {
        let sample =
            PerformanceSample::new(ProcessorType::QuantumInspired, "test-processor".to_string())
                .with_execution_time(100.0)
                .with_throughput(500.0)
                .with_cache_hit_ratio(0.8);

        assert_eq!(sample.processor_type, ProcessorType::QuantumInspired);
        assert_eq!(sample.processor_id, "test-processor");
        assert_eq!(sample.execution_time_ms, 100.0);
        assert_eq!(sample.throughput_ops_per_sec, 500.0);
        assert_eq!(sample.cache_hit_ratio, 0.8);
    }

    #[test]
    fn test_execution_timer() {
        let timer = ExecutionTimer::new();
        std::thread::sleep(std::time::Duration::from_millis(10));

        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10.0);

        let sample = timer.to_sample(ProcessorType::QuantumInspired, "test".to_string());
        assert!(sample.execution_time_ms >= 10.0);
    }

    #[test]
    fn test_performance_history() {
        let mut history = PerformanceHistory::new(100);

        let sample = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string())
            .with_execution_time(100.0);

        history.add_sample(sample);
        assert_eq!(history.samples.len(), 1);
        assert_eq!(history.aggregated_metrics.sample_count, 1);

        let recent = history.get_recent_samples(10);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Critical > AlertSeverity::Error);
        assert!(AlertSeverity::Error > AlertSeverity::Warning);
        assert!(AlertSeverity::Warning > AlertSeverity::Info);
    }

    #[test]
    fn test_alert_manager() {
        let manager = AlertManager::new(1000);
        assert!(!manager.alert_rules.is_empty()); // Should have default rules

        let stats = manager.get_alert_stats();
        assert_eq!(stats.total_alerts, 0);
        assert_eq!(stats.active_alerts, 0);
    }

    #[test]
    fn test_monitor_creation_and_summary() {
        let config = PerformanceMonitorConfig::default();
        let monitor = RealTimePerformanceMonitor::new(config);

        assert!(!monitor.is_monitoring_active());

        let summary = monitor.get_monitoring_summary();
        assert!(!summary.monitoring_active);
        assert_eq!(summary.total_samples, 0);
        assert_eq!(summary.active_alerts, 0);
        assert_eq!(summary.registered_processors, 0);
    }

    #[test]
    fn test_system_metrics() {
        let metrics = SystemMetrics::new();
        assert_eq!(metrics.cpu_usage, 0.0);
        assert_eq!(metrics.memory_usage, 0.0);
        assert!(metrics.timestamp > 0);

        let health_score = metrics.health_score();
        assert!((0.0..=1.0).contains(&health_score));
    }

    #[test]
    fn test_aggregated_metrics() {
        let mut metrics = AggregatedMetrics::new();

        let sample = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string())
            .with_execution_time(100.0)
            .with_throughput(500.0);

        metrics.update_with_sample(&sample);
        assert_eq!(metrics.sample_count, 1);
        assert_eq!(metrics.avg_execution_time, 100.0);
        assert_eq!(metrics.avg_throughput, 500.0);
    }

    #[test]
    fn test_performance_trend_display() {
        assert_eq!(PerformanceTrend::Improving.to_string(), "Improving");
        assert_eq!(PerformanceTrend::Stable.to_string(), "Stable");
        assert_eq!(PerformanceTrend::Degrading.to_string(), "Degrading");
        assert_eq!(
            PerformanceTrend::Insufficient.to_string(),
            "Insufficient Data"
        );
    }

    #[test]
    fn test_memory_usage_estimation() {
        let config = PerformanceMonitorConfig::default();
        let memory_usage = config.estimated_memory_usage();
        assert!(memory_usage > 0);

        let lightweight = PerformanceMonitorConfig::lightweight();
        let lightweight_memory = lightweight.estimated_memory_usage();
        assert!(lightweight_memory < memory_usage);
    }

    #[test]
    fn test_realtime_suitability() {
        let config = PerformanceMonitorConfig::default();
        assert!(config.is_realtime_suitable());

        let debug_config = PerformanceMonitorConfig::debug();
        // Debug config might have very high frequency monitoring
        let _is_suitable = debug_config.is_realtime_suitable();
    }
}
