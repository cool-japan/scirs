//! # Tracing Framework for SciRS2 v0.2.0
//!
//! This module provides structured logging and tracing capabilities using the `tracing` crate.
//! It enables zero-overhead instrumentation when disabled and comprehensive observability when enabled.
//!
//! # Features
//!
//! - **Structured Logging**: Rich, structured event logging with context
//! - **Span Management**: Hierarchical span tracking for nested operations
//! - **Multiple Outputs**: Console, file, JSON, and custom exporters
//! - **Performance Zones**: Named zones for performance tracking
//! - **Zero Overhead**: Compile-time elimination when feature is disabled
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_core::profiling::tracing_framework::{TracingConfig, init_tracing};
//! use tracing::{info, span, Level};
//!
//! // Initialize tracing with default configuration
//! let _guard = init_tracing(TracingConfig::default()).expect("Failed to initialize tracing");
//!
//! // Create a span for a computation
//! let span = span!(Level::INFO, "matrix_multiply", size = 1000);
//! let _enter = span.enter();
//!
//! info!("Starting matrix multiplication");
//! // ... perform computation ...
//! info!(result = "success", "Matrix multiplication completed");
//! ```

#[cfg(feature = "profiling_advanced")]
use crate::CoreResult;
#[cfg(feature = "profiling_advanced")]
use std::path::PathBuf;
#[cfg(feature = "profiling_advanced")]
use tracing::Level;
#[cfg(feature = "profiling_advanced")]
use tracing_appender::non_blocking::WorkerGuard;
#[cfg(feature = "profiling_advanced")]
use tracing_subscriber::{
    fmt, layer::SubscriberExt, registry::LookupSpan, util::SubscriberInitExt, EnvFilter, Layer,
};

/// Configuration for tracing framework
#[cfg(feature = "profiling_advanced")]
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Output format: "compact", "pretty", "json"
    pub format: TracingFormat,
    /// Log level filter
    pub level: Level,
    /// Enable ANSI color codes
    pub ansi_colors: bool,
    /// Log to file
    pub log_to_file: bool,
    /// Log file path
    pub log_file_path: Option<PathBuf>,
    /// Log file rotation: "hourly", "daily", "never"
    pub log_rotation: LogRotation,
    /// Enable flame graph generation
    pub enable_flame: bool,
    /// Enable Chrome DevTools format
    pub enable_chrome: bool,
    /// Custom environment filter
    pub env_filter: Option<String>,
}

/// Output format for tracing
#[cfg(feature = "profiling_advanced")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracingFormat {
    /// Compact single-line format
    Compact,
    /// Pretty multi-line format with colors
    Pretty,
    /// JSON format for structured logging
    Json,
}

/// Log rotation strategy
#[cfg(feature = "profiling_advanced")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogRotation {
    /// Rotate logs hourly
    Hourly,
    /// Rotate logs daily
    Daily,
    /// No rotation
    Never,
}

#[cfg(feature = "profiling_advanced")]
impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            format: TracingFormat::Pretty,
            level: Level::INFO,
            ansi_colors: true,
            log_to_file: false,
            log_file_path: None,
            log_rotation: LogRotation::Daily,
            enable_flame: false,
            enable_chrome: false,
            env_filter: None,
        }
    }
}

#[cfg(feature = "profiling_advanced")]
impl TracingConfig {
    /// Create a production configuration (minimal overhead, JSON format)
    pub fn production() -> Self {
        Self {
            format: TracingFormat::Json,
            level: Level::WARN,
            ansi_colors: false,
            log_to_file: true,
            log_file_path: Some(PathBuf::from("/var/log/scirs2")),
            log_rotation: LogRotation::Daily,
            enable_flame: false,
            enable_chrome: false,
            env_filter: Some("scirs2=warn,scirs2_core=warn".to_string()),
        }
    }

    /// Create a development configuration (verbose, pretty format)
    pub fn development() -> Self {
        Self {
            format: TracingFormat::Pretty,
            level: Level::DEBUG,
            ansi_colors: true,
            log_to_file: false,
            log_file_path: None,
            log_rotation: LogRotation::Never,
            enable_flame: true,
            enable_chrome: false,
            env_filter: Some("scirs2=debug".to_string()),
        }
    }

    /// Create a configuration for benchmarking (flamegraph enabled)
    pub fn benchmark() -> Self {
        Self {
            format: TracingFormat::Compact,
            level: Level::INFO,
            ansi_colors: false,
            log_to_file: false,
            log_file_path: None,
            log_rotation: LogRotation::Never,
            enable_flame: true,
            enable_chrome: true,
            env_filter: Some("scirs2=info".to_string()),
        }
    }

    /// Set output format
    pub fn with_format(mut self, format: TracingFormat) -> Self {
        self.format = format;
        self
    }

    /// Set log level
    pub fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }

    /// Enable file logging
    pub fn with_file_logging(mut self, path: PathBuf, rotation: LogRotation) -> Self {
        self.log_to_file = true;
        self.log_file_path = Some(path);
        self.log_rotation = rotation;
        self
    }

    /// Enable flame graph generation
    pub fn with_flame_graph(mut self, enable: bool) -> Self {
        self.enable_flame = enable;
        self
    }

    /// Set environment filter
    pub fn with_env_filter(mut self, filter: String) -> Self {
        self.env_filter = Some(filter);
        self
    }
}

/// Initialize the tracing framework
///
/// Returns a `WorkerGuard` that must be kept alive for the duration of the program.
/// When dropped, it will flush any remaining logs.
#[cfg(feature = "profiling_advanced")]
pub fn init_tracing(config: TracingConfig) -> CoreResult<Option<WorkerGuard>> {
    let env_filter = if let Some(ref filter) = config.env_filter {
        EnvFilter::try_new(filter).map_err(|e| {
            crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                "Invalid env filter: {}",
                e
            )))
        })?
    } else {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(config.level.to_string()))
    };

    let registry = tracing_subscriber::registry().with(env_filter);

    // Configure file appender if enabled
    let guard = if config.log_to_file {
        let log_path = config.log_file_path.clone().ok_or_else(|| {
            crate::CoreError::ConfigError(crate::error::ErrorContext::new(
                "Log file path not specified".to_string(),
            ))
        })?;

        let file_appender = match config.log_rotation {
            LogRotation::Hourly => tracing_appender::rolling::hourly(&log_path, "scirs2.log"),
            LogRotation::Daily => tracing_appender::rolling::daily(&log_path, "scirs2.log"),
            LogRotation::Never => tracing_appender::rolling::never(&log_path, "scirs2.log"),
        };

        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        let file_layer = fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)
            .json();

        registry.with(file_layer).init();

        Some(guard)
    } else {
        // Console output
        match config.format {
            TracingFormat::Compact => {
                registry
                    .with(fmt::layer().compact().with_ansi(config.ansi_colors))
                    .init();
            }
            TracingFormat::Pretty => {
                registry
                    .with(fmt::layer().pretty().with_ansi(config.ansi_colors))
                    .init();
            }
            TracingFormat::Json => {
                registry.with(fmt::layer().json().with_ansi(false)).init();
            }
        }

        None
    };

    Ok(guard)
}

/// Macro for creating a traced function
///
/// This macro automatically creates a span for the function and enters it.
#[macro_export]
#[cfg(feature = "profiling_advanced")]
macro_rules! traced_function {
    ($name:expr) => {
        let _span = tracing::info_span!($name).entered();
    };
    ($name:expr, $($field:tt)*) => {
        let _span = tracing::info_span!($name, $($field)*).entered();
    };
}

/// Macro for creating a performance zone
///
/// This creates a span that can be used for performance analysis.
#[macro_export]
#[cfg(feature = "profiling_advanced")]
macro_rules! perf_zone {
    ($name:expr) => {
        let _zone = tracing::trace_span!("perf", zone = $name).entered();
    };
    ($name:expr, $($field:tt)*) => {
        let _zone = tracing::trace_span!("perf", zone = $name, $($field)*).entered();
    };
}

/// Span guard for automatic span management
#[cfg(feature = "profiling_advanced")]
pub struct SpanGuard {
    _span: tracing::span::EnteredSpan,
}

#[cfg(feature = "profiling_advanced")]
impl SpanGuard {
    /// Create a new span guard
    pub fn new(name: &str) -> Self {
        Self {
            _span: tracing::info_span!(target: "scirs2::profiling", "{}", name).entered(),
        }
    }

    /// Create a span guard with fields
    pub fn with_fields(name: &str, fields: &[(&str, &dyn std::fmt::Display)]) -> Self {
        let span = tracing::info_span!(target: "scirs2::profiling", "{}", name);
        for (key, value) in fields {
            span.record(*key, &tracing::field::display(value));
        }
        Self {
            _span: span.entered(),
        }
    }
}

/// Performance zone marker for critical sections
#[cfg(feature = "profiling_advanced")]
pub struct PerfZone {
    _span: tracing::span::EnteredSpan,
    name: String,
    start: std::time::Instant,
}

#[cfg(feature = "profiling_advanced")]
impl PerfZone {
    /// Start a new performance zone
    pub fn start(name: &str) -> Self {
        let span = tracing::trace_span!(
            target: "scirs2::perf",
            "perf_zone",
            zone = name
        )
        .entered();

        Self {
            _span: span,
            name: name.to_string(),
            start: std::time::Instant::now(),
        }
    }

    /// End the performance zone and log duration
    pub fn end(self) {
        let duration = self.start.elapsed();
        tracing::info!(
            target: "scirs2::perf",
            zone = %self.name,
            duration_us = duration.as_micros(),
            "Performance zone completed"
        );
    }
}

#[cfg(feature = "profiling_advanced")]
impl Drop for PerfZone {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        tracing::debug!(
            target: "scirs2::perf",
            zone = %self.name,
            duration_us = duration.as_micros(),
            "Performance zone ended"
        );
    }
}

/// Stub implementations when profiling_advanced feature is disabled
#[cfg(not(feature = "profiling_advanced"))]
use crate::CoreResult;

#[cfg(not(feature = "profiling_advanced"))]
pub struct TracingConfig;

#[cfg(not(feature = "profiling_advanced"))]
impl Default for TracingConfig {
    fn default() -> Self {
        Self
    }
}

#[cfg(not(feature = "profiling_advanced"))]
impl TracingConfig {
    pub fn production() -> Self {
        Self
    }
    pub fn development() -> Self {
        Self
    }
    pub fn benchmark() -> Self {
        Self
    }
}

#[cfg(not(feature = "profiling_advanced"))]
pub fn init_tracing(_config: TracingConfig) -> CoreResult<Option<()>> {
    Ok(None)
}

#[cfg(test)]
#[cfg(feature = "profiling_advanced")]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_config_defaults() {
        let config = TracingConfig::default();
        assert_eq!(config.format, TracingFormat::Pretty);
        assert_eq!(config.level, Level::INFO);
        assert!(config.ansi_colors);
        assert!(!config.log_to_file);
    }

    #[test]
    fn test_production_config() {
        let config = TracingConfig::production();
        assert_eq!(config.format, TracingFormat::Json);
        assert_eq!(config.level, Level::WARN);
        assert!(!config.ansi_colors);
        assert!(config.log_to_file);
    }

    #[test]
    fn test_development_config() {
        let config = TracingConfig::development();
        assert_eq!(config.format, TracingFormat::Pretty);
        assert_eq!(config.level, Level::DEBUG);
        assert!(config.ansi_colors);
        assert!(!config.log_to_file);
        assert!(config.enable_flame);
    }

    #[test]
    fn test_span_guard_creation() {
        let _guard = SpanGuard::new("test_span");
        // Should not panic
    }

    #[test]
    fn test_perf_zone() {
        let zone = PerfZone::start("test_zone");
        std::thread::sleep(std::time::Duration::from_millis(10));
        zone.end();
        // Should not panic
    }
}
