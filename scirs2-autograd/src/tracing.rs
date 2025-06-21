//! Enhanced tracing and recording capabilities for computation graphs
//!
//! This module provides comprehensive tracing, debugging, and profiling
//! capabilities for automatic differentiation computations.

use crate::tensor::Tensor;
use crate::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Global tracing system
static GLOBAL_TRACER: std::sync::OnceLock<Arc<Mutex<ExecutionTracer>>> = std::sync::OnceLock::new();

/// Execution tracer for computation graphs
pub struct ExecutionTracer {
    /// Configuration for tracing
    config: TracingConfig,
    /// Current trace session
    current_session: Option<TraceSession>,
    /// Historical trace records
    trace_history: Vec<TraceRecord>,
    /// Performance statistics
    performance_stats: PerformanceStats,
    /// Active recordings
    recordings: HashMap<String, Recording>,
}

impl ExecutionTracer {
    /// Create a new execution tracer
    pub fn new() -> Self {
        Self {
            config: TracingConfig::default(),
            current_session: None,
            trace_history: Vec::new(),
            performance_stats: PerformanceStats::default(),
            recordings: HashMap::new(),
        }
    }

    /// Configure tracing settings
    pub fn configure(&mut self, config: TracingConfig) {
        self.config = config;
    }

    /// Start a new trace session
    pub fn start_session(&mut self, name: &str) -> TraceSessionId {
        let session_id = TraceSessionId::new();
        let session = TraceSession {
            id: session_id.clone(),
            name: name.to_string(),
            start_time: Instant::now(),
            events: Vec::new(),
            metadata: HashMap::new(),
            status: SessionStatus::Active,
        };

        self.current_session = Some(session);
        session_id
    }

    /// End the current trace session
    pub fn end_session(&mut self) -> Option<TraceRecord> {
        if let Some(mut session) = self.current_session.take() {
            session.status = SessionStatus::Completed;
            let duration = session.start_time.elapsed();

            let record = TraceRecord {
                session_id: session.id.clone(),
                session_name: session.name.clone(),
                start_time: session.start_time,
                duration,
                events: session.events.clone(),
                summary: self.create_session_summary(&session),
            };

            self.trace_history.push(record.clone());
            Some(record)
        } else {
            None
        }
    }

    /// Record an execution event
    pub fn record_event(&mut self, event: ExecutionEvent) {
        let should_record = self.should_record_event(&event);

        if let Some(ref mut session) = self.current_session {
            if should_record {
                session.events.push(event.clone());
            }
        }

        // Update performance statistics
        self.update_performance_stats(&event);
    }

    /// Record operation execution
    pub fn record_operation<F: Float>(
        &mut self,
        op_name: &str,
        inputs: &[&Tensor<F>],
        output: &Tensor<F>,
        duration: Duration,
    ) {
        let event = ExecutionEvent {
            timestamp: Instant::now(),
            event_type: EventType::OperationExecution {
                operation: op_name.to_string(),
                input_shapes: inputs.iter().map(|_t| vec![/* placeholder */]).collect(),
                output_shape: vec![/* placeholder */],
                duration,
                memory_usage: self.estimate_memory_usage(inputs, output),
            },
            metadata: HashMap::new(),
        };

        self.record_event(event);
    }

    /// Record gradient computation
    pub fn record_gradient<F: Float>(
        &mut self,
        target_name: &str,
        gradient: &Tensor<F>,
        duration: Duration,
    ) {
        let event = ExecutionEvent {
            timestamp: Instant::now(),
            event_type: EventType::GradientComputation {
                target: target_name.to_string(),
                gradient_shape: vec![/* placeholder */],
                gradient_norm: self.compute_gradient_norm(gradient),
                duration,
            },
            metadata: HashMap::new(),
        };

        self.record_event(event);
    }

    /// Record memory allocation
    pub fn record_memory_allocation(&mut self, size: usize, location: &str) {
        let event = ExecutionEvent {
            timestamp: Instant::now(),
            event_type: EventType::MemoryAllocation {
                size,
                location: location.to_string(),
                allocation_type: AllocationType::TensorData,
            },
            metadata: HashMap::new(),
        };

        self.record_event(event);
    }

    /// Record performance bottleneck
    pub fn record_bottleneck(&mut self, operation: &str, duration: Duration, reason: &str) {
        let event = ExecutionEvent {
            timestamp: Instant::now(),
            event_type: EventType::PerformanceBottleneck {
                operation: operation.to_string(),
                duration,
                reason: reason.to_string(),
                severity: if duration.as_millis() > 100 {
                    BottleneckSeverity::High
                } else if duration.as_millis() > 10 {
                    BottleneckSeverity::Medium
                } else {
                    BottleneckSeverity::Low
                },
            },
            metadata: HashMap::new(),
        };

        self.record_event(event);
    }

    /// Start a recording for detailed analysis
    pub fn start_recording(&mut self, name: &str, config: RecordingConfig) -> RecordingId {
        let recording_id = RecordingId::new();
        let recording = Recording {
            id: recording_id.clone(),
            name: name.to_string(),
            config,
            start_time: Instant::now(),
            traces: Vec::new(),
            status: RecordingStatus::Active,
        };

        self.recordings.insert(recording_id.0.clone(), recording);
        recording_id
    }

    /// Stop a recording
    pub fn stop_recording(&mut self, recording_id: &RecordingId) -> Option<Recording> {
        if let Some(mut recording) = self.recordings.remove(&recording_id.0) {
            recording.status = RecordingStatus::Completed;
            Some(recording)
        } else {
            None
        }
    }

    /// Get trace history
    pub fn get_trace_history(&self) -> &[TraceRecord] {
        &self.trace_history
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> &PerformanceStats {
        &self.performance_stats
    }

    /// Analyze performance patterns
    pub fn analyze_performance(&self) -> PerformanceAnalysis {
        let mut analysis = PerformanceAnalysis::default();

        // Analyze operation performance
        let mut operation_times: HashMap<String, Vec<Duration>> = HashMap::new();
        let mut memory_patterns: Vec<usize> = Vec::new();

        for record in &self.trace_history {
            for event in &record.events {
                match &event.event_type {
                    EventType::OperationExecution {
                        operation,
                        duration,
                        memory_usage,
                        ..
                    } => {
                        operation_times
                            .entry(operation.clone())
                            .or_default()
                            .push(*duration);
                        memory_patterns.push(*memory_usage);
                    }
                    EventType::PerformanceBottleneck {
                        operation,
                        duration,
                        severity,
                        ..
                    } => {
                        analysis.bottlenecks.push(BottleneckInfo {
                            operation: operation.clone(),
                            average_duration: *duration,
                            frequency: 1,
                            severity: *severity,
                        });
                    }
                    _ => {}
                }
            }
        }

        // Calculate operation statistics
        for (operation, times) in operation_times {
            if !times.is_empty() {
                let total_time: Duration = times.iter().sum();
                let avg_time = total_time / times.len() as u32;
                let min_time = *times.iter().min().unwrap();
                let max_time = *times.iter().max().unwrap();

                analysis.operation_stats.insert(
                    operation,
                    OperationStats {
                        call_count: times.len(),
                        total_time,
                        average_time: avg_time,
                        min_time,
                        max_time,
                    },
                );
            }
        }

        // Calculate memory statistics
        if !memory_patterns.is_empty() {
            analysis.memory_stats = MemoryStats {
                peak_usage: *memory_patterns.iter().max().unwrap(),
                average_usage: memory_patterns.iter().sum::<usize>() / memory_patterns.len(),
                allocation_count: memory_patterns.len(),
            };
        }

        analysis
    }

    /// Export traces to JSON format
    pub fn export_traces(&self, format: ExportFormat) -> Result<String, TracingError> {
        match format {
            ExportFormat::Json => {
                // Create a simplified exportable version
                let exportable_traces: Vec<_> = self
                    .trace_history
                    .iter()
                    .map(|record| {
                        serde_json::json!({
                            "session_id": record.session_id.0,
                            "session_name": record.session_name,
                            "duration_ms": record.duration.as_millis(),
                            "events_count": record.events.len(),
                            "summary": {
                                "total_operations": record.summary.total_operations,
                                "total_gradients": record.summary.total_gradients,
                                "memory_allocated": record.summary.memory_allocated,
                                "bottlenecks_detected": record.summary.bottlenecks_detected
                            }
                        })
                    })
                    .collect();

                serde_json::to_string_pretty(&exportable_traces)
                    .map_err(|e| TracingError::ExportError(e.to_string()))
            }
            ExportFormat::Chrome => self.export_chrome_trace(),
        }
    }

    /// Export in Chrome tracing format for detailed analysis
    #[allow(dead_code)]
    fn export_chrome_trace(&self) -> Result<String, TracingError> {
        let mut events = Vec::new();

        for record in &self.trace_history {
            let session_start = record.start_time;

            for event in &record.events {
                let timestamp_us = event.timestamp.duration_since(session_start).as_micros() as u64;

                match &event.event_type {
                    EventType::OperationExecution {
                        operation,
                        duration,
                        ..
                    } => {
                        events.push(serde_json::json!({
                            "name": operation,
                            "cat": "operation",
                            "ph": "X",
                            "ts": timestamp_us,
                            "dur": duration.as_micros(),
                            "pid": 1,
                            "tid": 1
                        }));
                    }
                    EventType::GradientComputation {
                        target, duration, ..
                    } => {
                        events.push(serde_json::json!({
                            "name": format!("grad({})", target),
                            "cat": "gradient",
                            "ph": "X",
                            "ts": timestamp_us,
                            "dur": duration.as_micros(),
                            "pid": 1,
                            "tid": 2
                        }));
                    }
                    _ => {}
                }
            }
        }

        let trace = serde_json::json!({
            "traceEvents": events,
            "displayTimeUnit": "ms"
        });

        Ok(trace.to_string())
    }

    /// Helper methods
    #[allow(dead_code)]
    fn should_record_event(&self, event: &ExecutionEvent) -> bool {
        match &event.event_type {
            EventType::OperationExecution { duration, .. } => {
                *duration >= self.config.min_operation_duration
            }
            EventType::MemoryAllocation { size, .. } => *size >= self.config.min_memory_threshold,
            _ => true,
        }
    }

    #[allow(dead_code)]
    fn estimate_memory_usage<F: Float>(&self, inputs: &[&Tensor<F>], _output: &Tensor<F>) -> usize {
        // Simplified memory estimation - in practice would calculate actual tensor sizes
        let estimated_input_memory = inputs.len() * 1000 * std::mem::size_of::<f64>();
        let estimated_output_memory = 1000 * std::mem::size_of::<f64>();
        estimated_input_memory + estimated_output_memory
    }

    #[allow(dead_code)]
    fn compute_gradient_norm<F: Float>(&self, _gradient: &Tensor<F>) -> f64 {
        // Simplified implementation - would compute actual L2 norm
        1.0
    }

    #[allow(dead_code)]
    fn update_performance_stats(&mut self, event: &ExecutionEvent) {
        match &event.event_type {
            EventType::OperationExecution {
                operation,
                duration,
                ..
            } => {
                self.performance_stats.total_operations += 1;
                self.performance_stats.total_execution_time += *duration;

                let entry = self
                    .performance_stats
                    .operation_counts
                    .entry(operation.clone())
                    .or_insert(0);
                *entry += 1;
            }
            EventType::MemoryAllocation { size, .. } => {
                self.performance_stats.total_memory_allocated += *size;
                self.performance_stats.peak_memory_usage =
                    self.performance_stats.peak_memory_usage.max(*size);
            }
            _ => {}
        }
    }

    #[allow(dead_code)]
    fn create_session_summary(&self, session: &TraceSession) -> SessionSummary {
        let mut summary = SessionSummary::default();

        for event in &session.events {
            match &event.event_type {
                EventType::OperationExecution { duration, .. } => {
                    summary.total_operations += 1;
                    summary.total_execution_time += *duration;
                }
                EventType::GradientComputation { duration, .. } => {
                    summary.total_gradients += 1;
                    summary.total_gradient_time += *duration;
                }
                EventType::MemoryAllocation { size, .. } => {
                    summary.memory_allocated += *size;
                }
                EventType::PerformanceBottleneck { .. } => {
                    summary.bottlenecks_detected += 1;
                }
            }
        }

        summary
    }
}

impl Default for ExecutionTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for tracing behavior
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Enable operation tracing
    pub trace_operations: bool,
    /// Enable gradient tracing
    pub trace_gradients: bool,
    /// Enable memory tracing
    pub trace_memory: bool,
    /// Enable performance bottleneck detection
    pub detect_bottlenecks: bool,
    /// Minimum operation duration to trace
    pub min_operation_duration: Duration,
    /// Minimum memory allocation size to trace
    pub min_memory_threshold: usize,
    /// Maximum number of events per session
    pub max_events_per_session: usize,
    /// Maximum number of sessions to keep in history
    pub max_session_history: usize,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            trace_operations: true,
            trace_gradients: true,
            trace_memory: false,
            detect_bottlenecks: true,
            min_operation_duration: Duration::from_micros(10),
            min_memory_threshold: 1024, // 1KB
            max_events_per_session: 10000,
            max_session_history: 100,
        }
    }
}

/// Unique identifier for trace sessions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceSessionId(String);

impl TraceSessionId {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(format!("session_{}", id))
    }
}

/// Unique identifier for recordings
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordingId(String);

impl RecordingId {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(format!("recording_{}", id))
    }
}

/// A trace session containing execution events
#[derive(Debug, Clone)]
pub struct TraceSession {
    pub id: TraceSessionId,
    pub name: String,
    pub start_time: Instant,
    pub events: Vec<ExecutionEvent>,
    pub metadata: HashMap<String, String>,
    pub status: SessionStatus,
}

/// Status of a trace session
#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    Active,
    Completed,
    Cancelled,
}

/// An execution event in the trace
#[derive(Debug, Clone)]
pub struct ExecutionEvent {
    pub timestamp: Instant,
    pub event_type: EventType,
    pub metadata: HashMap<String, String>,
}

/// Types of execution events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    OperationExecution {
        operation: String,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
        duration: Duration,
        memory_usage: usize,
    },
    GradientComputation {
        target: String,
        gradient_shape: Vec<usize>,
        gradient_norm: f64,
        duration: Duration,
    },
    MemoryAllocation {
        size: usize,
        location: String,
        allocation_type: AllocationType,
    },
    PerformanceBottleneck {
        operation: String,
        duration: Duration,
        reason: String,
        severity: BottleneckSeverity,
    },
}

/// Types of memory allocations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AllocationType {
    TensorData,
    GraphNode,
    Temporary,
}

/// Severity levels for performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// A complete trace record
#[derive(Debug, Clone)]
pub struct TraceRecord {
    pub session_id: TraceSessionId,
    pub session_name: String,
    pub start_time: Instant,
    pub duration: Duration,
    pub events: Vec<ExecutionEvent>,
    pub summary: SessionSummary,
}

/// Summary statistics for a trace session
#[derive(Debug, Clone, Default)]
pub struct SessionSummary {
    pub total_operations: usize,
    pub total_gradients: usize,
    pub total_execution_time: Duration,
    pub total_gradient_time: Duration,
    pub memory_allocated: usize,
    pub bottlenecks_detected: usize,
}

/// Recording configuration for detailed analysis
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    pub record_all_operations: bool,
    pub record_memory_patterns: bool,
    pub record_gradient_flow: bool,
    pub sample_rate: f64, // 0.0 to 1.0
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            record_all_operations: true,
            record_memory_patterns: true,
            record_gradient_flow: true,
            sample_rate: 1.0,
        }
    }
}

/// A recording session for detailed analysis
#[derive(Debug, Clone)]
pub struct Recording {
    pub id: RecordingId,
    pub name: String,
    pub config: RecordingConfig,
    pub start_time: Instant,
    pub traces: Vec<DetailedTrace>,
    pub status: RecordingStatus,
}

/// Status of a recording
#[derive(Debug, Clone, PartialEq)]
pub enum RecordingStatus {
    Active,
    Completed,
    Cancelled,
}

/// Detailed trace information for advanced analysis
#[derive(Debug, Clone)]
pub struct DetailedTrace {
    pub timestamp: Instant,
    pub operation_stack: Vec<String>,
    pub memory_snapshot: MemorySnapshot,
    pub computational_cost: ComputationalCost,
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub active_tensors: usize,
    pub garbage_collectible: usize,
}

/// Computational cost metrics
#[derive(Debug, Clone)]
pub struct ComputationalCost {
    pub flops: u64,
    pub memory_bandwidth: u64,
    pub cache_efficiency: f64,
}

/// Performance statistics across all sessions
#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub total_operations: usize,
    pub total_execution_time: Duration,
    pub total_memory_allocated: usize,
    pub peak_memory_usage: usize,
    pub operation_counts: HashMap<String, usize>,
}

/// Comprehensive performance analysis
#[derive(Debug, Clone, Default)]
pub struct PerformanceAnalysis {
    pub operation_stats: HashMap<String, OperationStats>,
    pub memory_stats: MemoryStats,
    pub bottlenecks: Vec<BottleneckInfo>,
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Statistics for individual operations
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub call_count: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub peak_usage: usize,
    pub average_usage: usize,
    pub allocation_count: usize,
}

/// Information about performance bottlenecks
#[derive(Debug, Clone)]
pub struct BottleneckInfo {
    pub operation: String,
    pub average_duration: Duration,
    pub frequency: usize,
    pub severity: BottleneckSeverity,
}

/// Optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub category: RecommendationCategory,
    pub description: String,
    pub expected_improvement: f64, // Estimated speedup factor
}

/// Categories of optimization recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationCategory {
    MemoryOptimization,
    ComputeOptimization,
    AlgorithmChoice,
    ParallelizationOpportunity,
}

/// Export formats for trace data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExportFormat {
    Json,
    Chrome, // Chrome DevTools tracing format
}

/// Errors that can occur during tracing
#[derive(Debug, thiserror::Error)]
pub enum TracingError {
    #[error("Session not found: {0}")]
    SessionNotFound(String),
    #[error("Recording not found: {0}")]
    RecordingNotFound(String),
    #[error("Export error: {0}")]
    ExportError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Global API functions
/// Initialize the global tracer
pub fn init_tracer() -> Arc<Mutex<ExecutionTracer>> {
    GLOBAL_TRACER
        .get_or_init(|| Arc::new(Mutex::new(ExecutionTracer::new())))
        .clone()
}

/// Configure global tracing
pub fn configure_tracing(config: TracingConfig) -> Result<(), TracingError> {
    let tracer = init_tracer();
    let mut tracer_guard = tracer
        .lock()
        .map_err(|_| TracingError::ConfigError("Failed to acquire tracer lock".to_string()))?;
    tracer_guard.configure(config);
    Ok(())
}

/// Start a new trace session
pub fn start_trace_session(name: &str) -> Result<TraceSessionId, TracingError> {
    let tracer = init_tracer();
    let mut tracer_guard = tracer
        .lock()
        .map_err(|_| TracingError::ConfigError("Failed to acquire tracer lock".to_string()))?;
    Ok(tracer_guard.start_session(name))
}

/// End the current trace session
pub fn end_trace_session() -> Result<Option<TraceRecord>, TracingError> {
    let tracer = init_tracer();
    let mut tracer_guard = tracer
        .lock()
        .map_err(|_| TracingError::ConfigError("Failed to acquire tracer lock".to_string()))?;
    Ok(tracer_guard.end_session())
}

/// Record an operation execution
pub fn trace_operation<F: Float>(
    op_name: &str,
    inputs: &[&Tensor<F>],
    output: &Tensor<F>,
    duration: Duration,
) -> Result<(), TracingError> {
    let tracer = init_tracer();
    let mut tracer_guard = tracer
        .lock()
        .map_err(|_| TracingError::ConfigError("Failed to acquire tracer lock".to_string()))?;
    tracer_guard.record_operation(op_name, inputs, output, duration);
    Ok(())
}

/// Get performance analysis
pub fn get_performance_analysis() -> Result<PerformanceAnalysis, TracingError> {
    let tracer = init_tracer();
    let tracer_guard = tracer
        .lock()
        .map_err(|_| TracingError::ConfigError("Failed to acquire tracer lock".to_string()))?;
    Ok(tracer_guard.analyze_performance())
}

/// Export traces to a file
pub fn export_traces(format: ExportFormat) -> Result<String, TracingError> {
    let tracer = init_tracer();
    let tracer_guard = tracer
        .lock()
        .map_err(|_| TracingError::ConfigError("Failed to acquire tracer lock".to_string()))?;
    tracer_guard.export_traces(format)
}

/// Enable or disable tracing globally
pub fn set_tracing_enabled(enabled: bool) -> Result<(), TracingError> {
    let config = TracingConfig {
        trace_operations: enabled,
        trace_gradients: enabled,
        ..Default::default()
    };
    configure_tracing(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_creation() {
        let tracer = ExecutionTracer::new();
        assert!(tracer.current_session.is_none());
        assert!(tracer.trace_history.is_empty());
    }

    #[test]
    fn test_session_lifecycle() {
        let mut tracer = ExecutionTracer::new();

        let _session_id = tracer.start_session("test_session");
        assert!(tracer.current_session.is_some());

        let record = tracer.end_session();
        assert!(record.is_some());
        assert!(tracer.current_session.is_none());
        assert_eq!(tracer.trace_history.len(), 1);
    }

    #[test]
    fn test_event_recording() {
        let mut tracer = ExecutionTracer::new();
        tracer.start_session("test");

        let event = ExecutionEvent {
            timestamp: Instant::now(),
            event_type: EventType::OperationExecution {
                operation: "add".to_string(),
                input_shapes: vec![vec![2, 2], vec![2, 2]],
                output_shape: vec![2, 2],
                duration: Duration::from_millis(1),
                memory_usage: 1024,
            },
            metadata: HashMap::new(),
        };

        tracer.record_event(event);

        let session = tracer.current_session.as_ref().unwrap();
        assert_eq!(session.events.len(), 1);
    }

    #[test]
    fn test_performance_analysis() {
        let mut tracer = ExecutionTracer::new();
        tracer.start_session("perf_test");

        // Record some operations
        for i in 0..5 {
            let event = ExecutionEvent {
                timestamp: Instant::now(),
                event_type: EventType::OperationExecution {
                    operation: "matmul".to_string(),
                    input_shapes: vec![vec![100, 100], vec![100, 100]],
                    output_shape: vec![100, 100],
                    duration: Duration::from_millis(i + 1),
                    memory_usage: 40000,
                },
                metadata: HashMap::new(),
            };
            tracer.record_event(event);
        }

        tracer.end_session();
        let analysis = tracer.analyze_performance();

        assert!(analysis.operation_stats.contains_key("matmul"));
        let matmul_stats = &analysis.operation_stats["matmul"];
        assert_eq!(matmul_stats.call_count, 5);
    }

    #[test]
    fn test_tracing_config() {
        let config = TracingConfig {
            trace_operations: false,
            min_operation_duration: Duration::from_millis(5),
            ..Default::default()
        };

        assert!(!config.trace_operations);
        assert_eq!(config.min_operation_duration, Duration::from_millis(5));
    }

    #[test]
    fn test_global_tracer() {
        let session_id = start_trace_session("global_test").unwrap();
        assert!(!session_id.0.is_empty());

        let record = end_trace_session().unwrap();
        assert!(record.is_some());
    }

    #[test]
    fn test_export_json() {
        let mut tracer = ExecutionTracer::new();
        tracer.start_session("export_test");
        tracer.end_session();

        let json_export = tracer.export_traces(ExportFormat::Json).unwrap();
        assert!(json_export.contains("session_"));
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut tracer = ExecutionTracer::new();
        tracer.record_bottleneck(
            "slow_op",
            Duration::from_millis(150),
            "inefficient algorithm",
        );

        assert_eq!(tracer.performance_stats.total_operations, 0); // Bottlenecks aren't operations
    }
}
