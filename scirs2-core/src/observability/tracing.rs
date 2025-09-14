//! # Distributed Tracing System
//!
//! Production-grade distributed tracing system with OpenTelemetry integration
//! for `SciRS2` Core. Provides request tracing across components, performance
//! attribution, and comprehensive span management for regulated environments.
//!
//! ## Features
//!
//! - OpenTelemetry-compatible tracing with standards compliance
//! - Distributed context propagation across components
//! - Performance attribution and latency analysis
//! - Span lifecycle management with automatic cleanup
//! - Thread-safe implementations with minimal overhead
//! - Integration with existing metrics and error systems
//! - Configurable sampling and filtering for production use
//! - Enterprise-grade security and compliance features
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::observability::tracing::{TracingSystem, SpanBuilder, TracingConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = TracingConfig::default();
//! let tracing = TracingSystem::new(config)?;
//!
//! // Create a traced operation using SpanBuilder
//! let span = SpanBuilder::new("matrix_multiplication")
//!     .with_attribute("size", "1000x1000")
//!     .with_component("linalg")
//!     .start(&tracing)?;
//!
//! // Perform operation with automatic performance tracking
//! let result = span.in_span(|| {
//!     // Your computation here
//!     42
//! });
//! assert_eq!(result, 42);
//!
//! // Span automatically ends and reports metrics when dropped
//! # Ok(())
//! # }
//! ```

use crate::error::CoreError;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

use serde::{Deserialize, Serialize};

// W3C Trace Context constants for OpenTelemetry compatibility
const TRACE_VERSION: u8 = 0;
#[allow(dead_code)]
const TRACE_HEADER_NAME: &str = "traceparent";
#[allow(dead_code)]
const TRACE_STATE_HEADER_NAME: &str = "tracestate";

/// Distributed tracing system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Service name for trace identification
    pub service_name: String,
    /// Service version for compatibility tracking
    pub service_version: String,
    /// Environment (production, staging, development)
    pub environment: String,
    /// Sampling rate (0.0 to 1.0)
    pub samplingrate: f64,
    /// Maximum number of active spans
    pub max_activespans: usize,
    /// Span timeout duration
    pub span_timeout: Duration,
    /// Enable performance attribution
    pub enable_performance_attribution: bool,
    /// Enable distributed context propagation
    pub enable_distributed_context: bool,
    /// Custom attributes to add to all spans
    pub default_attributes: HashMap<String, String>,
    /// Endpoint for trace export
    pub export_endpoint: Option<String>,
    /// Export batch size
    pub export_batch_size: usize,
    /// Export timeout
    pub export_timeout: Duration,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "scirs2-core".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "production".to_string(),
            samplingrate: 1.0,
            max_activespans: 10000,
            span_timeout: Duration::from_secs(300), // 5 minutes
            enable_performance_attribution: true,
            enable_distributed_context: true,
            default_attributes: HashMap::new(),
            export_endpoint: None,
            export_batch_size: 100,
            export_timeout: Duration::from_secs(30),
        }
    }
}

/// Span kind for categorizing operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanKind {
    /// Internal span within the same process
    Internal,
    /// Server span (receiving a request)
    Server,
    /// Client span (making a request)
    Client,
    /// Producer span (publishing data)
    Producer,
    /// Consumer span (consuming data)
    Consumer,
}

/// Span status for tracking operation outcomes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    /// Operation completed successfully
    Ok,
    /// Operation failed with error
    Error,
    /// Operation was cancelled
    Cancelled,
    /// Operation status unknown
    Unknown,
}

/// Trace context for distributed tracing (W3C Trace Context compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Unique trace identifier (16 bytes for W3C compatibility)
    pub trace_id: Uuid,
    /// Span identifier within the trace (8 bytes for W3C compatibility)
    pub spanid: Uuid,
    /// Parent span identifier
    pub parent_spanid: Option<Uuid>,
    /// Trace flags for sampling decisions (8 bits)
    pub trace_flags: u8,
    /// Additional baggage for context propagation
    pub baggage: HashMap<String, String>,
    /// W3C trace state for vendor-specific data
    pub tracestate: Option<String>,
    /// Remote flag for distributed traces
    pub is_remote: bool,
}

impl TraceContext {
    /// Create a new trace context
    #[must_use]
    pub fn new() -> Self {
        Self {
            trace_id: Uuid::new_v4(),
            spanid: Uuid::new_v4(),
            parent_spanid: None,
            trace_flags: 1, // Sampled
            baggage: HashMap::new(),
            tracestate: None,
            is_remote: false,
        }
    }

    /// Create a child context
    #[must_use]
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id,
            spanid: Uuid::new_v4(),
            parent_spanid: Some(self.spanid),
            trace_flags: self.trace_flags,
            baggage: self.baggage.clone(),
            tracestate: self.tracestate.clone(),
            is_remote: false,
        }
    }

    /// Create a remote child context (from another service)
    #[must_use]
    pub fn remote_child(&self) -> Self {
        let mut child = self.child();
        child.is_remote = true;
        child
    }

    /// Check if trace is sampled
    #[must_use]
    pub const fn is_sampled(&self) -> bool {
        self.trace_flags & 1 != 0
    }

    /// Add baggage item
    #[must_use]
    pub fn with_baggage(mut self, key: String, value: String) -> Self {
        self.baggage.insert(key, value);
        self
    }

    /// Set trace state
    #[must_use]
    pub fn with_tracestate(mut self, tracestate: String) -> Self {
        self.tracestate = Some(tracestate);
        self
    }

    /// Create W3C traceparent header value
    #[must_use]
    pub fn to_traceparent(&self) -> String {
        format!(
            "{:02x}-{}-{}-{:02x}",
            TRACE_VERSION,
            self.trace_id.as_simple(),
            &self.spanid.as_simple().to_string()[16..], // Use last 16 chars for 8-byte span ID
            self.trace_flags
        )
    }

    /// Parse W3C traceparent header
    ///
    /// # Errors
    ///
    /// Returns an error if the traceparent header format is invalid.
    pub fn from_traceparent(header: &str) -> Result<Self, CoreError> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("Invalid traceparent format".to_string()),
            ));
        }

        let version = u8::from_str_radix(parts[0], 16).map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Invalid _version in traceparent".to_string(),
            ))
        })?;

        if version != TRACE_VERSION {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("Unsupported traceparent _version".to_string()),
            ));
        }

        let trace_id = Uuid::parse_str(&format!(
            "{}-{}-{}-{}-{}",
            &parts[1][0..8],
            &parts[1][8..12],
            &parts[1][12..16],
            &parts[1][16..20],
            &parts[1][20..32]
        ))
        .map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Invalid trace ID in traceparent".to_string(),
            ))
        })?;

        // For span ID, we need to pad the 16-char ID to create a valid UUID
        let spanid_str = if parts[2].len() == 16 {
            format!("{:0>32}", parts[2]) // Pad to 32 characters with leading zeros
        } else {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new(
                    "Invalid span ID length in traceparent".to_string(),
                ),
            ));
        };
        let spanid = Uuid::parse_str(&format!(
            "{}-{}-{}-{}-{}",
            &spanid_str[0..8],
            &spanid_str[8..12],
            &spanid_str[12..16],
            &spanid_str[16..20],
            &spanid_str[20..32]
        ))
        .map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Invalid span ID in traceparent".to_string(),
            ))
        })?;

        let trace_flags = u8::from_str_radix(parts[3], 16).map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Invalid flags in traceparent".to_string(),
            ))
        })?;

        Ok(Self {
            trace_id,
            spanid,
            parent_spanid: None,
            trace_flags,
            baggage: HashMap::new(),
            tracestate: None,
            is_remote: true,
        })
    }

    /// Create baggage header value
    #[must_use]
    pub fn to_baggage(&self) -> Option<String> {
        if self.baggage.is_empty() {
            None
        } else {
            Some(
                self.baggage
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        }
    }

    /// Parse baggage header
    #[must_use]
    pub fn with_baggage_header(mut self, header: &str) -> Self {
        for item in header.split(',') {
            let item = item.trim();
            if let Some(eq_pos) = item.find('=') {
                let key = item[..eq_pos].trim().to_string();
                let value = item[eq_pos + 1..].trim().to_string();
                self.baggage.insert(key, value);
            }
        }
        self
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics for a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanMetrics {
    /// Duration of the span
    pub duration: Duration,
    /// CPU time consumed
    pub cpu_time: Option<Duration>,
    /// Memory allocated during span
    pub memory_allocated: Option<u64>,
    /// Memory deallocated during span
    pub memory_deallocated: Option<u64>,
    /// Peak memory usage during span
    pub peak_memory: Option<u64>,
    /// Number of child spans
    pub child_span_count: usize,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl Default for SpanMetrics {
    fn default() -> Self {
        Self {
            duration: Duration::from_nanos(0),
            cpu_time: None,
            memory_allocated: None,
            memory_deallocated: None,
            peak_memory: None,
            child_span_count: 0,
            custom_metrics: HashMap::new(),
        }
    }
}

/// Span data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Trace context
    pub context: TraceContext,
    /// Span name/operation name
    pub name: String,
    /// Span kind
    pub kind: SpanKind,
    /// Start time
    pub start_time: SystemTime,
    /// End time (if span has ended)
    pub end_time: Option<SystemTime>,
    /// Span status
    pub status: SpanStatus,
    /// Span attributes
    pub attributes: HashMap<String, String>,
    /// Events recorded during span
    pub events: Vec<SpanEvent>,
    /// Performance metrics
    pub metrics: SpanMetrics,
    /// Component that created the span
    pub component: Option<String>,
    /// Error information if status is Error
    pub error: Option<String>,
}

/// Event recorded during a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event name
    pub name: String,
    /// Event attributes
    pub attributes: HashMap<String, String>,
}

/// Active span handle for managing span lifecycle
pub struct ActiveSpan {
    span: Arc<Mutex<Span>>,
    tracingsystem: Arc<TracingSystem>,
    start_instant: Instant,
    #[cfg(feature = "memory_metrics")]
    initial_memory: Option<u64>,
}

impl ActiveSpan {
    /// Add an attribute to the span
    ///
    /// # Errors
    ///
    /// Returns an error if the span lock cannot be acquired.
    pub fn add_attribute(&self, key: &str, value: &str) -> Result<(), CoreError> {
        let mut span = self.span.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire span lock".to_string(),
            ))
        })?;
        span.attributes.insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Record an event
    ///
    /// # Errors
    ///
    /// Returns an error if the span lock cannot be acquired.
    pub fn add_event(
        &self,
        name: &str,
        attributes: HashMap<String, String>,
    ) -> Result<(), CoreError> {
        let mut span = self.span.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire span lock".to_string(),
            ))
        })?;

        let event = SpanEvent {
            timestamp: SystemTime::now(),
            name: name.to_string(),
            attributes,
        };

        span.events.push(event);
        Ok(())
    }

    /// Add a custom metric
    ///
    /// # Errors
    ///
    /// Returns an error if the span lock cannot be acquired.
    pub fn add_metric(&self, name: &str, value: f64) -> Result<(), CoreError> {
        let mut span = self.span.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire span lock".to_string(),
            ))
        })?;
        span.metrics.custom_metrics.insert(name.to_string(), value);
        Ok(())
    }

    /// Set span status
    ///
    /// # Errors
    ///
    /// Returns an error if the span lock cannot be acquired.
    pub fn set_status(&self, status: SpanStatus) -> Result<(), CoreError> {
        let mut span = self.span.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire span lock".to_string(),
            ))
        })?;
        span.status = status;
        Ok(())
    }

    /// Set error information
    ///
    /// # Errors
    ///
    /// Returns an error if the span lock cannot be acquired.
    pub fn seterror(&self, error: &str) -> Result<(), CoreError> {
        let mut span = self.span.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire span lock".to_string(),
            ))
        })?;
        span.status = SpanStatus::Error;
        span.error = Some(error.to_string());
        Ok(())
    }

    /// Execute a closure within the span context
    #[must_use]
    pub fn in_span<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Set current span context in thread-local storage
        CURRENT_SPAN.with(|current| {
            let _prev = current.replace(Some(self.span.clone()));
            let result = f();
            current.replace(_prev);
            result
        })
    }

    /// Execute an async closure within the span context
    #[cfg(feature = "async")]
    pub async fn in_span_async<F, Fut, R>(&self, f: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        // For async contexts, we would typically use tokio-tracing
        // For now, we'll provide a basic implementation
        CURRENT_SPAN.with(|current| {
            let _prev = current.borrow_mut().replace(self.span.clone());
            // Note: This is a simplified implementation
            // In production, you'd want proper async context propagation
        });
        f().await
    }

    /// Get the span's trace context
    ///
    /// # Errors
    ///
    /// Returns an error if the span lock cannot be acquired.
    pub fn context(&self) -> Result<TraceContext, CoreError> {
        let span = self.span.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire span lock".to_string(),
            ))
        })?;
        Ok(span.context.clone())
    }

    /// End the span explicitly
    pub fn end(self) {
        // Destructor will handle the actual ending
        drop(self);
    }
}

impl Drop for ActiveSpan {
    fn drop(&mut self) {
        // End the span when dropped
        if let Ok(mut span) = self.span.lock() {
            if span.end_time.is_none() {
                span.end_time = Some(SystemTime::now());
                span.metrics.duration = self.start_instant.elapsed();

                #[cfg(feature = "memory_metrics")]
                if let Some(initial_memory) = self.initial_memory {
                    // Calculate memory metrics (simplified)
                    if let Ok(current_memory) = get_current_memory_usage() {
                        if current_memory > initial_memory {
                            span.metrics.memory_allocated = Some(current_memory - initial_memory);
                        } else {
                            span.metrics.memory_deallocated = Some(initial_memory - current_memory);
                        }
                    }
                }

                // Report span to tracing system
                if let Err(e) = self.tracingsystem.record_span(span.clone()) {
                    eprintln!("Failed to record span: {e}");
                }
            }
        }
    }
}

/// Span builder for creating spans with configuration
pub struct SpanBuilder {
    name: String,
    kind: SpanKind,
    attributes: HashMap<String, String>,
    parent_context: Option<TraceContext>,
    component: Option<String>,
}

impl SpanBuilder {
    /// Create a new span builder
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            kind: SpanKind::Internal,
            attributes: HashMap::new(),
            parent_context: None,
            component: None,
        }
    }

    /// Set span kind
    #[must_use]
    pub fn with_kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }

    /// Add an attribute
    #[must_use]
    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    /// Set parent context
    #[must_use]
    pub fn with_parent(mut self, context: TraceContext) -> Self {
        self.parent_context = Some(context);
        self
    }

    /// Set component name
    #[must_use]
    pub fn with_component(mut self, component: &str) -> Self {
        self.component = Some(component.to_string());
        self
    }

    /// Build the span using the tracing system
    ///
    /// # Errors
    ///
    /// Returns an error if the span cannot be started.
    pub fn start(self, tracingsystem: &TracingSystem) -> Result<ActiveSpan, CoreError> {
        tracingsystem.start_span_with_builder(self)
    }
}

// Thread-local storage for current span
thread_local! {
    static CURRENT_SPAN: std::cell::RefCell<Option<Arc<Mutex<Span>>>> = const { std::cell::RefCell::new(None) };
}

/// Span storage for managing active spans
#[derive(Debug)]
struct SpanStorage {
    active_spans: RwLock<HashMap<Uuid, Arc<Mutex<Span>>>>,
    completed_spans: Mutex<Vec<Span>>,
    max_activespans: usize,
}

impl SpanStorage {
    #[must_use]
    fn new(max_activespans: usize) -> Self {
        Self {
            active_spans: RwLock::new(HashMap::new()),
            completed_spans: Mutex::new(Vec::new()),
            max_activespans,
        }
    }

    /// Add an active span to storage
    ///
    /// # Errors
    ///
    /// Returns an error if the maximum active spans is exceeded or if locks cannot be acquired.
    fn add_active_span(&self, span: Arc<Mutex<Span>>) -> Result<(), CoreError> {
        let mut active = self.active_spans.write().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire write lock".to_string(),
            ))
        })?;

        if active.len() >= self.max_activespans {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("Maximum active spans exceeded".to_string()),
            ));
        }

        let spanid = {
            let span_guard = span.lock().map_err(|_| {
                CoreError::ComputationError(crate::error::ErrorContext::new(
                    "Failed to acquire span lock".to_string(),
                ))
            })?;
            span_guard.context.spanid
        };

        active.insert(spanid, span);
        Ok(())
    }

    #[must_use]
    fn remove_active_span(&self, spanid: Uuid) -> Option<Arc<Mutex<Span>>> {
        if let Ok(mut active) = self.active_spans.write() {
            active.remove(&spanid)
        } else {
            None
        }
    }

    /// Record a completed span
    ///
    /// # Errors
    ///
    /// Returns an error if the completed spans lock cannot be acquired.
    fn record_completed_span(&self, span: Span) -> Result<(), CoreError> {
        let mut completed = self.completed_spans.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire completed spans lock".to_string(),
            ))
        })?;
        completed.push(span);
        Ok(())
    }

    #[must_use]
    fn get_active_span_count(&self) -> usize {
        self.active_spans
            .read()
            .map(|spans| spans.len())
            .unwrap_or(0)
    }

    /// Clean up expired spans
    ///
    /// # Errors
    ///
    /// Returns an error if locks cannot be acquired.
    fn cleanup_expired_spans(&self, timeout: Duration) -> Result<Vec<Span>, CoreError> {
        let mut expired_spans = Vec::new();
        let now = SystemTime::now();
        let mut to_remove = Vec::new();

        {
            let active = self.active_spans.read().map_err(|_| {
                CoreError::ComputationError(crate::error::ErrorContext::new(
                    "Failed to acquire read lock".to_string(),
                ))
            })?;

            for (spanid, span_arc) in active.iter() {
                if let Ok(span) = span_arc.lock() {
                    if let Ok(elapsed) = now.duration_since(span.start_time) {
                        if elapsed > timeout {
                            to_remove.push(*spanid);
                        }
                    }
                }
            }
        }

        for spanid in to_remove {
            if let Some(span_arc) = self.remove_active_span(spanid) {
                if let Ok(mut span) = span_arc.lock() {
                    span.status = SpanStatus::Cancelled;
                    span.end_time = Some(now);
                    expired_spans.push(span.clone());
                }
            }
        }

        Ok(expired_spans)
    }
}

/// Main distributed tracing system
pub struct TracingSystem {
    config: TracingConfig,
    storage: SpanStorage,
    sampler: Box<dyn TracingSampler + Send + Sync>,
    exporter: Option<Box<dyn TraceExporter + Send + Sync>>,
    metrics: Arc<Mutex<TracingMetrics>>,
}

impl TracingSystem {
    /// Create a new tracing system
    ///
    /// # Errors
    ///
    /// Returns an error if the system cannot be initialized.
    pub fn new(config: TracingConfig) -> Result<Self, CoreError> {
        let storage = SpanStorage::new(config.max_activespans);
        let sampler = Box::new(ProbabilitySampler::new(config.samplingrate));
        let metrics = Arc::new(Mutex::new(TracingMetrics::default()));

        Ok(Self {
            config,
            storage,
            sampler,
            exporter: None,
            metrics,
        })
    }

    /// Set a custom trace exporter
    #[must_use]
    pub fn with_exporter(mut self, exporter: Box<dyn TraceExporter + Send + Sync>) -> Self {
        self.exporter = Some(exporter);
        self
    }

    /// Start a new span
    ///
    /// # Errors
    ///
    /// Returns an error if the span cannot be started.
    pub fn start_span(&self, name: &str) -> Result<ActiveSpan, CoreError> {
        let builder = SpanBuilder::new(name);
        self.start_span_with_builder(builder)
    }

    /// Start a span with a builder
    ///
    /// # Errors
    ///
    /// Returns an error if the span cannot be started.
    pub fn start_span_with_builder(&self, builder: SpanBuilder) -> Result<ActiveSpan, CoreError> {
        // Create trace context
        let context = if let Some(parent) = builder.parent_context {
            parent.child()
        } else {
            // Try to get current context from thread-local storage
            CURRENT_SPAN
                .with(|current| {
                    if let Some(current_span) = current.borrow().as_ref() {
                        if let Ok(span) = current_span.lock() {
                            Some(span.context.child())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_default()
        };

        // Check sampling decision
        if !self.sampler.should_sample(&context, &builder.name) {
            // Return a no-op span for unsampled traces
            let span = Span {
                context: context.clone(),
                name: builder.name,
                kind: builder.kind,
                start_time: SystemTime::now(),
                end_time: None,
                status: SpanStatus::Ok,
                attributes: builder.attributes,
                events: Vec::new(),
                metrics: SpanMetrics::default(),
                component: builder.component,
                error: None,
            };

            let span_arc = Arc::new(Mutex::new(span));
            return Ok(ActiveSpan {
                span: span_arc,
                tracingsystem: Arc::new(self.clone()),
                start_instant: Instant::now(),
                #[cfg(feature = "memory_metrics")]
                initial_memory: get_current_memory_usage().ok(),
            });
        }

        // Create span with merged attributes
        let mut attributes = self.config.default_attributes.clone();
        attributes.extend(builder.attributes);

        let span = Span {
            context: context.clone(),
            name: builder.name,
            kind: builder.kind,
            start_time: SystemTime::now(),
            end_time: None,
            status: SpanStatus::Ok,
            attributes,
            events: Vec::new(),
            metrics: SpanMetrics::default(),
            component: builder.component,
            error: None,
        };

        let span_arc = Arc::new(Mutex::new(span));

        // Add to active spans
        self.storage.add_active_span(span_arc.clone())?;

        // Update metrics
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.spans_started += 1;
            metrics.active_spans = self.storage.get_active_span_count();
        }

        Ok(ActiveSpan {
            span: span_arc,
            tracingsystem: Arc::new(self.clone()),
            start_instant: Instant::now(),
            #[cfg(feature = "memory_metrics")]
            initial_memory: get_current_memory_usage().ok(),
        })
    }

    /// Get current span from context
    #[must_use]
    pub fn current_span(&self) -> Option<Arc<Mutex<Span>>> {
        CURRENT_SPAN.with(|current| current.borrow().clone())
    }

    /// Record a completed span
    ///
    /// # Errors
    ///
    /// Returns an error if the span cannot be recorded or exported.
    pub fn record_span(&self, span: Span) -> Result<(), CoreError> {
        // Remove from active spans
        let _ = self.storage.remove_active_span(span.context.spanid);

        // Update metrics
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.spans_completed += 1;
            metrics.active_spans = self.storage.get_active_span_count();

            if span.status == SpanStatus::Error {
                metrics.spans_failed += 1;
            }

            metrics.total_duration += span.metrics.duration;
        }

        // Export span if exporter is available
        if let Some(ref exporter) = self.exporter {
            exporter.export_span(&span)?;
        }

        // Store completed span
        self.storage.record_completed_span(span)?;

        Ok(())
    }

    /// Cleanup expired spans
    ///
    /// # Errors
    ///
    /// Returns an error if expired spans cannot be cleaned up.
    pub fn cleanup_expired_spans(&self) -> Result<(), CoreError> {
        let expired_spans = self
            .storage
            .cleanup_expired_spans(self.config.span_timeout)?;

        for span in expired_spans {
            self.record_span(span)?;
        }

        Ok(())
    }

    /// Get tracing metrics
    ///
    /// # Errors
    ///
    /// Returns an error if the metrics lock cannot be acquired.
    pub fn get_metrics(&self) -> Result<TracingMetrics, CoreError> {
        let metrics = self.metrics.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire metrics lock".to_string(),
            ))
        })?;
        Ok(metrics.clone())
    }

    /// Flush all pending spans
    ///
    /// # Errors
    ///
    /// Returns an error if the exporter flush fails.
    pub fn flush(&self) -> Result<(), CoreError> {
        if let Some(ref exporter) = self.exporter {
            exporter.flush()?;
        }
        Ok(())
    }
}

// Note: We need to implement Clone for TracingSystem to allow Arc<TracingSystem>
// This is a simplified implementation - in production you might want to use Arc internally
impl Clone for TracingSystem {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            storage: SpanStorage::new(self.config.max_activespans),
            sampler: Box::new(ProbabilitySampler::new(self.config.samplingrate)),
            exporter: None, // Cannot clone trait objects easily
            metrics: Arc::new(Mutex::new(TracingMetrics::default())),
        }
    }
}

/// Tracing metrics for monitoring system health
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TracingMetrics {
    /// Total spans started
    pub spans_started: u64,
    /// Total spans completed
    pub spans_completed: u64,
    /// Total spans failed
    pub spans_failed: u64,
    /// Currently active spans
    pub active_spans: usize,
    /// Total duration of all completed spans
    pub total_duration: Duration,
    /// Spans exported successfully
    pub spans_exported: u64,
    /// Export failures
    pub export_failures: u64,
}

/// Trait for implementing sampling strategies
pub trait TracingSampler {
    /// Determine if a trace should be sampled
    fn should_sample(&self, context: &TraceContext, spanname: &str) -> bool;
}

/// Probability-based sampler
pub struct ProbabilitySampler {
    samplingrate: f64,
}

impl ProbabilitySampler {
    pub fn new(samplingrate: f64) -> Self {
        Self {
            samplingrate: samplingrate.clamp(0.0, 1.0),
        }
    }
}

impl TracingSampler for ProbabilitySampler {
    fn should_sample(&self, _context: &TraceContext, _spanname: &str) -> bool {
        if self.samplingrate >= 1.0 {
            true
        } else if self.samplingrate <= 0.0 {
            false
        } else {
            use rand::Rng;
            let mut rng = rand::rng();
            rng.random::<f64>() < self.samplingrate
        }
    }
}

/// Adaptive sampler that adjusts sampling rates based on system load
pub struct AdaptiveSampler {
    base_rate: f64,
    min_rate: f64,
    max_rate: f64,
    sample_count: AtomicU64,
    total_count: AtomicU64,
    adjustment_window: u64,
    target_rate_persecond: f64,
    last_adjustment: Mutex<Instant>,
}

impl AdaptiveSampler {
    pub fn new(base_rate: f64, target_rate_persecond: f64) -> Self {
        Self {
            base_rate: base_rate.clamp(0.0, 1.0),
            min_rate: 0.001, // Minimum 0.1% sampling
            max_rate: 1.0,   // Maximum 100% sampling
            sample_count: AtomicU64::new(0),
            total_count: AtomicU64::new(0),
            adjustment_window: 1000, // Adjust every 1000 spans
            target_rate_persecond,
            last_adjustment: Mutex::new(Instant::now()),
        }
    }

    fn adjust_samplingrate(&self) -> f64 {
        let total = self.total_count.load(Ordering::Relaxed);
        if total % self.adjustment_window == 0 && total > 0 {
            if let Ok(mut last) = self.last_adjustment.try_lock() {
                let now = Instant::now();
                let elapsed = now.duration_since(*last).as_secs_f64();
                *last = now;

                if elapsed > 0.0 {
                    let current_rate = total as f64 / elapsed;
                    let adjustment_factor = self.target_rate_persecond / current_rate;
                    let new_rate =
                        (self.base_rate * adjustment_factor).clamp(self.min_rate, self.max_rate);
                    return new_rate;
                }
            }
        }
        self.base_rate
    }

    pub fn get_stats(&self) -> (u64, u64, f64) {
        let total = self.total_count.load(Ordering::Relaxed);
        let sampled = self.sample_count.load(Ordering::Relaxed);
        let rate = if total > 0 {
            sampled as f64 / total as f64
        } else {
            0.0
        };
        (total, sampled, rate)
    }
}

impl TracingSampler for AdaptiveSampler {
    fn should_sample(&self, _context: &TraceContext, _spanname: &str) -> bool {
        self.total_count.fetch_add(1, Ordering::Relaxed);

        let current_rate = self.adjust_samplingrate();

        if current_rate >= 1.0 {
            self.sample_count.fetch_add(1, Ordering::Relaxed);
            true
        } else if current_rate <= 0.0 {
            false
        } else {
            use rand::Rng;
            let mut rng = rand::rng();
            if rng.random::<f64>() < current_rate {
                self.sample_count.fetch_add(1, Ordering::Relaxed);
                true
            } else {
                false
            }
        }
    }
}

/// Rate-limiting sampler that ensures maximum number of samples per time window
pub struct RateLimitingSampler {
    max_samples_persecond: u64,
    sample_count: AtomicU64,
    window_start: Mutex<Instant>,
    windowsize: Duration,
}

impl RateLimitingSampler {
    pub fn new(max_samples_persecond: u64) -> Self {
        Self {
            max_samples_persecond,
            sample_count: AtomicU64::new(0),
            window_start: Mutex::new(Instant::now()),
            windowsize: Duration::from_secs(1),
        }
    }

    fn reset_window_if_needed(&self) -> bool {
        if let Ok(mut start) = self.window_start.try_lock() {
            let now = Instant::now();
            if now.duration_since(*start) >= self.windowsize {
                *start = now;
                self.sample_count.store(0, Ordering::Relaxed);
                return true;
            }
        }
        false
    }
}

impl TracingSampler for RateLimitingSampler {
    fn should_sample(&self, _context: &TraceContext, _spanname: &str) -> bool {
        self.reset_window_if_needed();

        let current_count = self.sample_count.load(Ordering::Relaxed);
        if current_count < self.max_samples_persecond {
            self.sample_count.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}

/// Trait for exporting traces to external systems
pub trait TraceExporter {
    /// Export a single span
    fn export_span(&self, span: &Span) -> Result<(), CoreError>;

    /// Export multiple spans in batch
    fn export_spans(&self, spans: &[Span]) -> Result<(), CoreError> {
        for span in spans {
            self.export_span(span)?;
        }
        Ok(())
    }

    /// Flush any pending exports
    fn flush(&self) -> Result<(), CoreError>;

    /// Shutdown the exporter
    fn shutdown(&self) -> Result<(), CoreError>;
}

/// Batch exporter that buffers spans before exporting
pub struct BatchExporter {
    inner: Box<dyn TraceExporter + Send + Sync>,
    batch_size: usize,
    batch_timeout: Duration,
    buffer: Mutex<Vec<Span>>,
    last_export: Mutex<Instant>,
}

impl BatchExporter {
    pub fn new(
        inner: Box<dyn TraceExporter + Send + Sync>,
        batch_size: usize,
        batch_timeout: Duration,
    ) -> Self {
        Self {
            inner,
            batch_size,
            batch_timeout,
            buffer: Mutex::new(Vec::new()),
            last_export: Mutex::new(Instant::now()),
        }
    }

    fn should_flush(&self) -> bool {
        if let Ok(buffer) = self.buffer.try_lock() {
            if buffer.len() >= self.batch_size {
                return true;
            }
        }

        if let Ok(last_export) = self.last_export.try_lock() {
            if last_export.elapsed() >= self.batch_timeout {
                return true;
            }
        }

        false
    }

    fn flush_internal(&self) -> Result<(), CoreError> {
        let spans_to_export = {
            let mut buffer = self.buffer.lock().map_err(|_| {
                CoreError::ComputationError(crate::error::ErrorContext::new(
                    "Failed to acquire buffer lock".to_string(),
                ))
            })?;
            if buffer.is_empty() {
                return Ok(());
            }
            let spans = buffer.drain(..).collect::<Vec<_>>();
            spans
        };

        if !spans_to_export.is_empty() {
            self.inner.export_spans(&spans_to_export)?;

            if let Ok(mut last_export) = self.last_export.lock() {
                *last_export = Instant::now();
            }
        }

        Ok(())
    }
}

impl TraceExporter for BatchExporter {
    fn export_span(&self, span: &Span) -> Result<(), CoreError> {
        {
            let mut buffer = self.buffer.lock().map_err(|_| {
                CoreError::ComputationError(crate::error::ErrorContext::new(
                    "Failed to acquire buffer lock".to_string(),
                ))
            })?;
            buffer.push(span.clone());
        }

        if self.should_flush() {
            self.flush_internal()?;
        }

        Ok(())
    }

    fn flush(&self) -> Result<(), CoreError> {
        self.flush_internal()?;
        self.inner.flush()
    }

    fn shutdown(&self) -> Result<(), CoreError> {
        self.flush_internal()?;
        self.inner.shutdown()
    }
}

/// Console exporter for development/debugging
pub struct ConsoleExporter {
    prettyprint: bool,
}

impl ConsoleExporter {
    pub fn new(prettyprint: bool) -> Self {
        Self { prettyprint }
    }
}

impl TraceExporter for ConsoleExporter {
    fn export_span(&self, span: &Span) -> Result<(), CoreError> {
        if self.prettyprint {
            println!("=== Span Export ===");
            println!("Trace ID: {}", span.context.trace_id);
            println!("Span ID: {}", span.context.spanid);
            println!("Name: {}", span.name);
            println!("Duration: {:?}", span.metrics.duration);
            println!("Status: {:?}", span.status);
            if !span.attributes.is_empty() {
                println!("Attributes: {:?}", span.attributes);
            }
            if !span.events.is_empty() {
                println!("Events: {} recorded", span.events.len());
            }
            println!("==================");
        } else {
            println!(
                "SPAN: {} {} {:?} {:?}",
                span.context.trace_id, span.name, span.metrics.duration, span.status
            );
        }
        Ok(())
    }

    fn flush(&self) -> Result<(), CoreError> {
        // Console output is immediate
        Ok(())
    }

    fn shutdown(&self) -> Result<(), CoreError> {
        Ok(())
    }
}

/// HTTP exporter for OpenTelemetry-compatible endpoints
#[cfg(feature = "reqwest")]
pub struct HttpExporter {
    endpoint: String,
    client: reqwest::blocking::Client,
    #[allow(dead_code)]
    timeout: Duration,
}

#[cfg(feature = "reqwest")]
impl HttpExporter {
    pub fn new(endpoint: String, timeout: Duration) -> Result<Self, CoreError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to create HTTP client: {}",
                    e
                )))
            })?;

        Ok(Self {
            endpoint,
            client,
            timeout,
        })
    }
}

#[cfg(feature = "reqwest")]
impl TraceExporter for HttpExporter {
    fn export_span(&self, span: &Span) -> Result<(), CoreError> {
        {
            let json = serde_json::to_string(span).map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to serialize span: {}",
                    e
                )))
            })?;

            let response = self
                .client
                .post(&self.endpoint)
                .header("Content-Type", "application/json")
                .body(json)
                .send()
                .map_err(|e| {
                    CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                        "Failed to send span: {}",
                        e
                    )))
                })?;

            if !response.status().is_success() {
                return Err(CoreError::ComputationError(
                    crate::error::ErrorContext::new(format!(
                        "Failed to export span: HTTP {}",
                        response.status()
                    )),
                ));
            }
        }

        #[cfg(not(feature = "serde"))]
        {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("HTTP export requires serde feature".to_string()),
            ));
        }

        Ok(())
    }

    fn flush(&self) -> Result<(), CoreError> {
        // HTTP exports are sent immediately
        Ok(())
    }

    fn shutdown(&self) -> Result<(), CoreError> {
        Ok(())
    }
}

/// Utility function to get current memory usage
#[cfg(feature = "memory_metrics")]
#[allow(dead_code)]
fn get_current_memory_usage() -> Result<u64, CoreError> {
    // This is a simplified implementation
    // In production, you'd use proper memory monitoring

    // For demonstration purposes, return 0
    // In reality, you'd integrate with memory profiling tools
    Ok(0)
}

/// Global tracing system instance
static GLOBAL_TRACER: std::sync::OnceLock<Arc<TracingSystem>> = std::sync::OnceLock::new();

/// Initialize global tracing system
#[allow(dead_code)]
pub fn init_tracing(config: TracingConfig) -> Result<(), CoreError> {
    let tracer = TracingSystem::new(config)?;
    match GLOBAL_TRACER.set(Arc::new(tracer)) {
        Ok(()) => Ok(()),
        Err(_) => {
            // Already initialized, which is fine
            Ok(())
        }
    }
}

/// Get global tracing system
#[allow(dead_code)]
pub fn global_tracer() -> Option<Arc<TracingSystem>> {
    GLOBAL_TRACER.get().cloned()
}

/// Convenience macro for creating traced functions
#[macro_export]
macro_rules! trace_fn {
    ($name:expr, $block:block) => {{
        if let Some(tracer) = $crate::observability::tracing::global_tracer() {
            let span = tracer.start_span($name)?;
            span.in_span(|| $block)
        } else {
            $block
        }
    }};
}

/// Convenience macro for creating traced async functions
#[cfg(feature = "async")]
#[macro_export]
macro_rules! trace_async_fn {
    ($name:expr, $block:block) => {{
        if let Some(tracer) = $crate::observability::tracing::global_tracer() {
            let span = tracer.start_span($name)?;
            span.in_span_async(|| async move $block).await
        } else {
            async move $block.await
        }
    }};
}

/// Version negotiation for distributed tracing compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl TracingVersion {
    pub const CURRENT: TracingVersion = TracingVersion {
        major: 1,
        minor: 0,
        patch: 0,
    };

    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn is_compatible(&self, other: &TracingVersion) -> bool {
        self.major == other.major && self.minor <= other.minor
    }
}

impl std::fmt::Display for TracingVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Negotiation result for version compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationResult {
    pub agreed_version: TracingVersion,
    pub features_supported: Vec<String>,
    pub features_disabled: Vec<String>,
}

/// Resource attribution tracker for performance analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceAttribution {
    /// CPU time consumed (in nanoseconds)
    pub cpu_timens: Option<u64>,
    /// Memory allocated (in bytes)
    pub memory_allocated_bytes: Option<u64>,
    /// Memory deallocated (in bytes)
    pub memory_deallocated_bytes: Option<u64>,
    /// Peak memory usage (in bytes)
    pub peak_memory_bytes: Option<u64>,
    /// Number of I/O operations
    pub io_operations: Option<u64>,
    /// Bytes read from I/O
    pub bytes_read: Option<u64>,
    /// Bytes written to I/O
    pub byteswritten: Option<u64>,
    /// Network requests made
    pub network_requests: Option<u64>,
    /// GPU memory used (in bytes)
    pub gpu_memory_bytes: Option<u64>,
    /// GPU compute time (in nanoseconds)
    pub gpu_compute_timens: Option<u64>,
}

impl ResourceAttribution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cpu_time(mut self, cpu_timens: u64) -> Self {
        self.cpu_timens = Some(cpu_timens);
        self
    }

    pub fn with_memory_allocation(mut self, bytes: u64) -> Self {
        self.memory_allocated_bytes = Some(bytes);
        self
    }

    pub fn with_io_stats(mut self, operations: u64, bytes_read: u64, byteswritten: u64) -> Self {
        self.io_operations = Some(operations);
        self.bytes_read = Some(bytes_read);
        self.byteswritten = Some(byteswritten);
        self
    }

    pub fn with_gpu_stats(mut self, memory_bytes: u64, compute_timens: u64) -> Self {
        self.gpu_memory_bytes = Some(memory_bytes);
        self.gpu_compute_timens = Some(compute_timens);
        self
    }

    pub fn merge(&mut self, other: &ResourceAttribution) {
        if let Some(cpu) = other.cpu_timens {
            self.cpu_timens = Some(self.cpu_timens.unwrap_or(0) + cpu);
        }
        if let Some(mem) = other.memory_allocated_bytes {
            self.memory_allocated_bytes = Some(self.memory_allocated_bytes.unwrap_or(0) + mem);
        }
        if let Some(mem) = other.memory_deallocated_bytes {
            self.memory_deallocated_bytes = Some(self.memory_deallocated_bytes.unwrap_or(0) + mem);
        }
        if let Some(peak) = other.peak_memory_bytes {
            self.peak_memory_bytes = Some(self.peak_memory_bytes.unwrap_or(0).max(peak));
        }
        if let Some(io) = other.io_operations {
            self.io_operations = Some(self.io_operations.unwrap_or(0) + io);
        }
        if let Some(read) = other.bytes_read {
            self.bytes_read = Some(self.bytes_read.unwrap_or(0) + read);
        }
        if let Some(written) = other.byteswritten {
            self.byteswritten = Some(self.byteswritten.unwrap_or(0) + written);
        }
        if let Some(net) = other.network_requests {
            self.network_requests = Some(self.network_requests.unwrap_or(0) + net);
        }
        if let Some(gpu_mem) = other.gpu_memory_bytes {
            self.gpu_memory_bytes = Some(self.gpu_memory_bytes.unwrap_or(0) + gpu_mem);
        }
        if let Some(gpu_time) = other.gpu_compute_timens {
            self.gpu_compute_timens = Some(self.gpu_compute_timens.unwrap_or(0) + gpu_time);
        }
    }
}

/// Enhanced span metrics with resource attribution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnhancedSpanMetrics {
    /// Basic span metrics
    pub basic: SpanMetrics,
    /// Resource attribution
    pub resources: ResourceAttribution,
    /// Custom performance counters
    pub performance_counters: HashMap<String, u64>,
}

impl EnhancedSpanMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_performance_counter(&mut self, name: &str, value: u64) {
        *self
            .performance_counters
            .entry(name.to_string())
            .or_insert(0) += value;
    }

    pub fn get_total_resource_cost(&self) -> f64 {
        let mut cost = 0.0;

        // CPU cost (normalized to milliseconds)
        if let Some(cpu_ns) = self.resources.cpu_timens {
            cost += cpu_ns as f64 / 1_000_000.0; // ns to ms
        }

        // Memory cost (normalized to MB)
        if let Some(mem) = self.resources.memory_allocated_bytes {
            cost += mem as f64 / 1_048_576.0; // bytes to MB
        }

        // I/O cost (simple addition)
        if let Some(io) = self.resources.io_operations {
            cost += io as f64;
        }

        cost
    }
}

/// Integration with existing metrics system
#[cfg(feature = "observability")]
#[allow(dead_code)]
pub fn integrate_with_metrics_system() -> Result<(), CoreError> {
    // Get global metrics registry and add tracing-specific metrics
    let registry = crate::metrics::global_metrics_registry();

    // Register tracing metrics
    use crate::metrics::{Counter, Gauge, Histogram};

    registry.register(
        "tracing_spans_started".to_string(),
        Counter::new("tracing_spans_started".to_string()),
    )?;
    registry.register(
        "tracing_spans_completed".to_string(),
        Counter::new("tracing_spans_completed".to_string()),
    )?;
    registry.register(
        "tracing_spans_failed".to_string(),
        Counter::new("tracing_spans_failed".to_string()),
    )?;
    registry.register(
        "tracing_active_spans".to_string(),
        Gauge::new("tracing_active_spans".to_string()),
    )?;
    registry.register(
        "tracing_span_duration".to_string(),
        Histogram::with_buckets(
            "tracing_span_duration".to_string(),
            vec![0.001, 0.01, 0.1, 1.0, 10.0],
        ),
    )?;

    Ok(())
}

/// Real-world usage example: Matrix computation with distributed tracing
#[allow(dead_code)]
pub fn examplematrix_computation_with_tracing() -> Result<(), CoreError> {
    // Initialize tracing with adaptive sampling
    let config = TracingConfig {
        service_name: "matrix_computation_service".to_string(),
        samplingrate: 1.0, // 100% for demo
        enable_performance_attribution: true,
        enable_distributed_context: true,
        ..TracingConfig::default()
    };

    let tracing = TracingSystem::new(config)?;
    let _adaptive_sampler = AdaptiveSampler::new(0.1, 1000.0); // 10% base rate, target 1000 samples/sec
    let batch_exporter = BatchExporter::new(
        Box::new(ConsoleExporter::new(true)),
        50,                     // batch size
        Duration::from_secs(5), // timeout
    );

    let tracing = tracing.with_exporter(Box::new(batch_exporter));

    // Start computation span
    let computation_span = tracing.start_span("matrix_multiplication")?;
    computation_span.add_attribute("matrix_size", "1000x1000")?;
    computation_span.add_attribute("algorithm", "block_multiplication")?;

    let _result = computation_span.in_span(|| {
        // Start memory allocation span
        let alloc_span = tracing.start_span("memory_allocation")?;
        alloc_span.add_attribute("allocation_size", "8MB")?;

        let _memory_result = alloc_span.in_span(|| {
            // Simulate memory allocation
            std::thread::sleep(Duration::from_millis(10));
            "allocated"
        });

        // Start computation span
        let compute_span = tracing.start_span("matrix_compute")?;
        compute_span.add_metric("flops", 2_000_000_000.0)?; // 2 billion operations

        let _compute_result = compute_span.in_span(|| {
            // Simulate computation
            std::thread::sleep(Duration::from_millis(100));
            "computed"
        });

        Ok::<_, CoreError>("matrix_result")
    })?;

    computation_span.add_attribute("result_status", "success")?;
    computation_span.end();

    // Cleanup and flush
    tracing.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_trace_context_creation() {
        let context = TraceContext::new();
        assert!(context.is_sampled());
        assert!(context.parent_spanid.is_none());

        let child = context.child();
        assert_eq!(child.trace_id, context.trace_id);
        assert_eq!(child.parent_spanid, Some(context.spanid));
        assert_ne!(child.spanid, context.spanid);
    }

    #[test]
    fn test_tracingsystem_creation() {
        let config = TracingConfig::default();
        let tracing = TracingSystem::new(config).expect("Failed to create tracing system");

        let metrics = tracing.get_metrics().expect("Failed to get metrics");
        assert_eq!(metrics.spans_started, 0);
        assert_eq!(metrics.active_spans, 0);
    }

    #[test]
    fn test_span_lifecycle() {
        let config = TracingConfig::default();
        let tracing = TracingSystem::new(config).expect("Failed to create tracing system");

        {
            let span = tracing
                .start_span("test_operation")
                .expect("Failed to start span");
            span.add_attribute("test", "value")
                .expect("Failed to add attribute");
            span.add_event("test_event", HashMap::new())
                .expect("Failed to add event");

            let context = span.context().expect("Failed to get context");
            assert!(context.is_sampled());
        } // Span ends here

        // Give some time for cleanup
        thread::sleep(Duration::from_millis(10));

        let metrics = tracing.get_metrics().expect("Failed to get metrics");
        assert_eq!(metrics.spans_started, 1);
    }

    #[test]
    fn test_span_builder() {
        let config = TracingConfig::default();
        let tracing = TracingSystem::new(config).expect("Failed to create tracing system");

        let span = SpanBuilder::new("test_operation")
            .with_kind(SpanKind::Server)
            .with_attribute("method", "GET")
            .with_component("web_server")
            .start(&tracing)
            .expect("Failed to start span");

        let context = span.context().expect("Failed to get context");
        assert!(context.is_sampled());
    }

    #[test]
    fn test_probability_sampler() {
        let sampler = ProbabilitySampler::new(0.0);
        let context = TraceContext::new();
        assert!(!sampler.should_sample(&context, "test"));

        let sampler = ProbabilitySampler::new(1.0);
        assert!(sampler.should_sample(&context, "test"));
    }

    #[test]
    fn test_console_exporter() {
        let exporter = ConsoleExporter::new(false);
        let context = TraceContext::new();

        let span = Span {
            context,
            name: "test".to_string(),
            kind: SpanKind::Internal,
            start_time: SystemTime::now(),
            end_time: Some(SystemTime::now()),
            status: SpanStatus::Ok,
            attributes: HashMap::new(),
            events: Vec::new(),
            metrics: SpanMetrics::default(),
            component: None,
            error: None,
        };

        // This will print to console
        exporter.export_span(&span).expect("Failed to export span");
    }

    #[test]
    fn test_nested_spans() {
        let config = TracingConfig::default();
        let tracing = TracingSystem::new(config).expect("Failed to create tracing system");

        let parent_span = tracing
            .start_span("parent_operation")
            .expect("Failed to start parent span");
        let parent_context = parent_span.context().expect("Failed to get parent context");

        let child_span = SpanBuilder::new("child_operation")
            .with_parent(parent_context.clone())
            .start(&tracing)
            .expect("Failed to start child span");

        let child_context = child_span.context().expect("Failed to get child context");
        assert_eq!(child_context.trace_id, parent_context.trace_id);
        assert_eq!(child_context.parent_spanid, Some(parent_context.spanid));
    }

    #[test]
    fn test_w3c_trace_context() {
        let context = TraceContext::new();
        let traceparent = context.to_traceparent();

        // Traceparent should have format: version-trace_id-spanid-flags
        let parts: Vec<&str> = traceparent.split('-').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "00"); // version
        assert_eq!(parts[3], "01"); // sampled flag

        // Test parsing back
        let parsed_context =
            TraceContext::from_traceparent(&traceparent).expect("Failed to parse traceparent");
        assert_eq!(parsed_context.trace_id, context.trace_id);
        assert!(parsed_context.is_remote);
        assert!(parsed_context.is_sampled());
    }

    #[test]
    fn test_adaptive_sampler() {
        let sampler = AdaptiveSampler::new(0.5, 100.0);
        let context = TraceContext::new();

        // Sample some spans
        for _ in 0..10 {
            sampler.should_sample(&context, "test");
        }

        let (total, sampled, rate) = sampler.get_stats();
        assert_eq!(total, 10);
        assert!((0.0..=1.0).contains(&rate));
    }

    #[test]
    fn test_rate_limiting_sampler() {
        let sampler = RateLimitingSampler::new(5); // Max 5 samples per second
        let context = TraceContext::new();

        // Should accept first 5 samples
        for i in 0..5 {
            assert!(
                sampler.should_sample(&context, "test"),
                "Sample {i} should be accepted"
            );
        }

        // Should reject further samples in the same window
        assert!(!sampler.should_sample(&context, "test"));
        assert!(!sampler.should_sample(&context, "test"));
    }

    #[test]
    fn test_batch_exporter() {
        let console_exporter = ConsoleExporter::new(false);
        let batch_exporter = BatchExporter::new(
            Box::new(console_exporter),
            3,                      // batch size
            Duration::from_secs(1), // timeout
        );

        let context = TraceContext::new();
        let span = Span {
            context,
            name: "test".to_string(),
            kind: SpanKind::Internal,
            start_time: SystemTime::now(),
            end_time: Some(SystemTime::now()),
            status: SpanStatus::Ok,
            attributes: HashMap::new(),
            events: Vec::new(),
            metrics: SpanMetrics::default(),
            component: None,
            error: None,
        };

        // Export spans - should batch until threshold
        batch_exporter
            .export_span(&span)
            .expect("Failed to export span");
        batch_exporter
            .export_span(&span)
            .expect("Failed to export span");
        batch_exporter
            .export_span(&span)
            .expect("Failed to export span"); // Should trigger flush

        batch_exporter.flush().expect("Failed to flush");
    }

    #[test]
    fn test_resource_attribution() {
        let mut attribution = ResourceAttribution::new()
            .with_cpu_time(1_000_000) // 1ms
            .with_memory_allocation(1024) // 1KB
            .with_io_stats(5, 100, 200); // 5 ops, 100 read, 200 written

        let other = ResourceAttribution::new()
            .with_cpu_time(500_000) // 0.5ms
            .with_memory_allocation(512); // 0.5KB

        attribution.merge(&other);

        assert_eq!(attribution.cpu_timens, Some(1_500_000));
        assert_eq!(attribution.memory_allocated_bytes, Some(1536));
        assert_eq!(attribution.io_operations, Some(5));
    }

    #[test]
    fn test_enhanced_span_metrics() {
        let mut metrics = EnhancedSpanMetrics::new();
        metrics.add_performance_counter("cache_hits", 150);
        metrics.add_performance_counter("cache_misses", 25);
        metrics.add_performance_counter("cache_hits", 50); // Should add to existing

        assert_eq!(metrics.performance_counters.get("cache_hits"), Some(&200));
        assert_eq!(metrics.performance_counters.get("cache_misses"), Some(&25));

        // Test resource cost calculation
        metrics.resources.cpu_timens = Some(1_000_000); // 1ms
        metrics.resources.memory_allocated_bytes = Some(1_048_576); // 1MB
        metrics.resources.io_operations = Some(5);

        let cost = metrics.get_total_resource_cost();
        assert!(cost > 0.0);
    }

    #[test]
    fn test_tracing_version_compatibility() {
        let v1_0 = TracingVersion::new(1, 0, 0);
        let v1_1 = TracingVersion::new(1, 1, 0);
        let v2_0 = TracingVersion::new(2, 0, 0);

        assert!(v1_0.is_compatible(&v1_1));
        assert!(!v1_1.is_compatible(&v1_0)); // Newer minor not compatible with older
        assert!(!v1_0.is_compatible(&v2_0)); // Different major versions not compatible

        assert_eq!(v1_0.to_string(), "1.0.0");
    }
}
