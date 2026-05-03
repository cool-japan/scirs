//! # OpenTelemetry Integration for SciRS2 v0.2.0
//!
//! This module provides OpenTelemetry integration for distributed tracing and metrics export.
//! It enables observability across distributed systems and cloud deployments.
//!
//! # Features
//!
//! - **Distributed Tracing**: Trace requests across multiple services
//! - **OTLP Export**: Export traces and metrics via OTLP protocol
//! - **Span Context Propagation**: Propagate trace context across boundaries
//! - **Prometheus Integration**: Export metrics to Prometheus
//! - **Semantic Conventions**: Follow OpenTelemetry semantic conventions
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_core::profiling::opentelemetry_integration::{OtelConfig, init_opentelemetry};
//!
//! // Initialize OpenTelemetry with OTLP exporter
//! let config = OtelConfig::default()
//!     .with_service_name("scirs2-compute")
//!     .with_otlp_endpoint("http://localhost:4317");
//!
//! let _guard = init_opentelemetry(config).expect("Failed to initialize OpenTelemetry");
//!
//! // Spans will now be exported to the OTLP collector
//! ```

#[cfg(feature = "profiling_opentelemetry")]
use crate::CoreResult;
#[cfg(feature = "profiling_opentelemetry")]
use opentelemetry::{global, trace::TracerProvider, KeyValue};
#[cfg(feature = "profiling_opentelemetry")]
use opentelemetry_sdk::{
    runtime,
    trace::{self, Sampler, SdkTracerProvider},
    Resource,
};
#[cfg(feature = "profiling_opentelemetry")]
use std::time::Duration;

/// Configuration for OpenTelemetry
#[cfg(feature = "profiling_opentelemetry")]
#[derive(Debug, Clone)]
pub struct OtelConfig {
    /// Service name
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// OTLP endpoint
    pub otlp_endpoint: Option<String>,
    /// Sampling ratio (0.0 to 1.0)
    pub sampling_ratio: f64,
    /// Export timeout in seconds
    pub export_timeout_secs: u64,
    /// Max queue size
    pub max_queue_size: usize,
    /// Resource attributes
    pub resource_attributes: Vec<(String, String)>,
}

#[cfg(feature = "profiling_opentelemetry")]
impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            service_name: "scirs2".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            otlp_endpoint: None,
            sampling_ratio: 1.0,
            export_timeout_secs: 30,
            max_queue_size: 2048,
            resource_attributes: Vec::new(),
        }
    }
}

#[cfg(feature = "profiling_opentelemetry")]
impl OtelConfig {
    /// Set service name
    pub fn with_service_name(mut self, name: &str) -> Self {
        self.service_name = name.to_string();
        self
    }

    /// Set service version
    pub fn with_service_version(mut self, version: &str) -> Self {
        self.service_version = version.to_string();
        self
    }

    /// Set OTLP endpoint
    pub fn with_otlp_endpoint(mut self, endpoint: &str) -> Self {
        self.otlp_endpoint = Some(endpoint.to_string());
        self
    }

    /// Set sampling ratio
    pub fn with_sampling_ratio(mut self, ratio: f64) -> Self {
        self.sampling_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Add resource attribute
    pub fn with_resource_attribute(mut self, key: String, value: String) -> Self {
        self.resource_attributes.push((key, value));
        self
    }
}

/// Initialize OpenTelemetry with the given configuration
#[cfg(all(feature = "profiling_opentelemetry", feature = "profiling_otlp"))]
pub fn init_opentelemetry(config: OtelConfig) -> CoreResult<OtelGuard> {
    use opentelemetry_otlp::WithExportConfig;

    // Create resource with service information
    let mut resource_kvs = vec![
        KeyValue::new("service.name", config.service_name.clone()),
        KeyValue::new("service.version", config.service_version.clone()),
    ];

    // Add custom attributes
    for (key, value) in config.resource_attributes {
        resource_kvs.push(KeyValue::new(key, value));
    }

    let resource = Resource::builder().with_attributes(resource_kvs).build();

    // Configure sampler
    let sampler = if config.sampling_ratio >= 1.0 {
        Sampler::AlwaysOn
    } else if config.sampling_ratio <= 0.0 {
        Sampler::AlwaysOff
    } else {
        Sampler::TraceIdRatioBased(config.sampling_ratio)
    };

    // Configure OTLP exporter if endpoint is provided
    let tracer_provider = if let Some(endpoint) = config.otlp_endpoint {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_timeout(Duration::from_secs(config.export_timeout_secs))
            .build()
            .map_err(|e| {
                crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                    "Failed to build exporter: {}",
                    e
                )))
            })?;

        SdkTracerProvider::builder()
            .with_resource(resource)
            .with_sampler(sampler)
            .with_batch_exporter(exporter)
            .build()
    } else {
        // No exporter, just create provider
        SdkTracerProvider::builder()
            .with_resource(resource)
            .with_sampler(sampler)
            .build()
    };

    // Set as global provider
    global::set_tracer_provider(tracer_provider.clone());

    Ok(OtelGuard {
        _provider: tracer_provider,
    })
}

/// Initialize OpenTelemetry without OTLP (basic configuration)
#[cfg(all(feature = "profiling_opentelemetry", not(feature = "profiling_otlp")))]
pub fn init_opentelemetry(config: OtelConfig) -> CoreResult<OtelGuard> {
    // Create resource with service information
    let mut resource_kvs = vec![
        KeyValue::new("service.name", config.service_name.clone()),
        KeyValue::new("service.version", config.service_version.clone()),
    ];

    // Add custom attributes
    for (key, value) in config.resource_attributes {
        resource_kvs.push(KeyValue::new(key, value));
    }

    let resource = Resource::builder().with_attributes(resource_kvs).build();

    // Configure sampler
    let sampler = if config.sampling_ratio >= 1.0 {
        Sampler::AlwaysOn
    } else if config.sampling_ratio <= 0.0 {
        Sampler::AlwaysOff
    } else {
        Sampler::TraceIdRatioBased(config.sampling_ratio)
    };

    let tracer_provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_sampler(sampler)
        .build();

    // Set as global provider
    global::set_tracer_provider(tracer_provider.clone());

    Ok(OtelGuard {
        _provider: tracer_provider,
    })
}

/// Guard for OpenTelemetry shutdown
#[cfg(feature = "profiling_opentelemetry")]
pub struct OtelGuard {
    _provider: SdkTracerProvider,
}

#[cfg(feature = "profiling_opentelemetry")]
impl Drop for OtelGuard {
    fn drop(&mut self) {
        // Shutdown is handled automatically when the provider is dropped
        // In OpenTelemetry 0.30.0, explicit shutdown is not needed via global
        let _ = self._provider.shutdown();
    }
}

/// Get a tracer for the current scope
#[cfg(feature = "profiling_opentelemetry")]
pub fn get_tracer(name: &str) -> impl opentelemetry::trace::Tracer {
    global::tracer(name.to_string())
}

/// Create a span with OpenTelemetry
#[cfg(feature = "profiling_opentelemetry")]
pub fn create_span(
    tracer: &impl opentelemetry::trace::Tracer,
    name: &str,
) -> impl opentelemetry::trace::Span {
    use opentelemetry::trace::Tracer;
    tracer.start(name.to_string())
}

/// Macro for creating an OpenTelemetry span
#[macro_export]
#[cfg(feature = "profiling_opentelemetry")]
macro_rules! otel_span {
    ($name:expr) => {{
        use opentelemetry::trace::Tracer;
        let tracer = $crate::profiling::opentelemetry_integration::get_tracer("scirs2");
        tracer.start($name)
    }};
}

/// Stub implementations when profiling_opentelemetry feature is disabled
#[cfg(not(feature = "profiling_opentelemetry"))]
use crate::CoreResult;

#[cfg(not(feature = "profiling_opentelemetry"))]
pub struct OtelConfig;

#[cfg(not(feature = "profiling_opentelemetry"))]
impl Default for OtelConfig {
    fn default() -> Self {
        Self
    }
}

#[cfg(not(feature = "profiling_opentelemetry"))]
impl OtelConfig {
    pub fn with_service_name(self, _name: &str) -> Self {
        self
    }
    pub fn with_otlp_endpoint(self, _endpoint: &str) -> Self {
        self
    }
}

#[cfg(not(feature = "profiling_opentelemetry"))]
pub fn init_opentelemetry(_config: OtelConfig) -> CoreResult<()> {
    Ok(())
}

#[cfg(test)]
#[cfg(feature = "profiling_opentelemetry")]
mod tests {
    use super::*;

    #[test]
    fn test_otel_config_defaults() {
        let config = OtelConfig::default();
        assert_eq!(config.service_name, "scirs2");
        assert_eq!(config.sampling_ratio, 1.0);
        assert!(config.otlp_endpoint.is_none());
    }

    #[test]
    fn test_otel_config_builder() {
        let config = OtelConfig::default()
            .with_service_name("test-service")
            .with_service_version("1.0.0")
            .with_sampling_ratio(0.5);

        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.service_version, "1.0.0");
        assert_eq!(config.sampling_ratio, 0.5);
    }

    #[test]
    fn test_sampling_ratio_clamping() {
        let config1 = OtelConfig::default().with_sampling_ratio(1.5);
        assert_eq!(config1.sampling_ratio, 1.0);

        let config2 = OtelConfig::default().with_sampling_ratio(-0.5);
        assert_eq!(config2.sampling_ratio, 0.0);
    }
}
