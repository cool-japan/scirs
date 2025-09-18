//! Clock Synchronization Module
//!
//! This module provides comprehensive clock synchronization capabilities for TPU pod coordination.
//! The module is organized into focused sub-modules that handle different aspects of time synchronization:
//!
//! - [`core`] - Main synchronization manager and coordination logic
//! - [`protocols`] - Synchronization protocols (NTP, PTP, GPS, etc.)
//! - [`sources`] - Time source management and selection algorithms
//! - [`gps`] - GPS signal processing and error correction
//! - [`network`] - Network synchronization, messaging, and load balancing
//! - [`quality`] - Quality monitoring and assessment
//! - [`drift`] - Drift compensation and prediction
//! - [`health`] - Health monitoring and recovery
//! - [`statistics`] - Performance tracking and reporting
//!
//! # Architecture
//!
//! The clock synchronization system follows a modular architecture where each component
//! has a specific responsibility but can work together to provide robust time synchronization:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    ClockSynchronizationManager                  │
//! │                         (core module)                          │
//! └─────────────────────────┬───────────────────────────────────────┘
//!                           │
//! ┌─────────────────────────┼───────────────────────────────────────┐
//! │         TimeSourceManager        │        ProtocolManager        │
//! │         (sources module)         │       (protocols module)      │
//! └─────────────────────────┬───────┴───────┬───────────────────────┘
//!                           │               │
//! ┌─────────────────────────┼───────────────┼───────────────────────┐
//! │      GPS Processing     │   Network     │    Quality & Health   │
//! │      (gps module)       │  (network)    │  (quality & health)   │
//! └─────────────────────────┼───────────────┼───────────────────────┘
//!                           │               │
//! ┌─────────────────────────┼───────────────┼───────────────────────┐
//! │   Drift Compensation    │  Statistics   │      Reporting        │
//! │     (drift module)      │ (statistics)  │    (statistics)       │
//! └─────────────────────────┴───────────────┴───────────────────────┘
//! ```
//!
//! # Usage
//!
//! Basic usage of the clock synchronization system:
//!
//! ```rust
//! use crate::tpu::pod_coordination::synchronization::clocks::{
//!     ClockSynchronizationManager, ClockSynchronizationConfig
//! };
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create synchronization manager with default configuration
//! let mut sync_manager = ClockSynchronizationManager::new(
//!     ClockSynchronizationConfig::default()
//! )?;
//!
//! // Start synchronization
//! sync_manager.start_synchronization()?;
//!
//! // Perform synchronization
//! sync_manager.synchronize()?;
//!
//! // Get synchronization status
//! let status = sync_manager.get_synchronization_status();
//! println!("Sync status: {:?}", status);
//!
//! // Stop synchronization
//! sync_manager.stop_synchronization()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! The clock synchronization system is designed for high-performance operation with:
//! - Minimal latency overhead
//! - Efficient memory usage
//! - Scalable to large TPU clusters
//! - Real-time operation capabilities
//! - Adaptive algorithms for varying network conditions

// Core synchronization components
pub mod core;
pub mod protocols;
pub mod sources;

// Specialized synchronization modules
pub mod gps;
pub mod network;

// Monitoring and analysis modules
pub mod quality;
pub mod drift;
pub mod health;
pub mod statistics;

// Re-export main types from core module
pub use core::{
    ClockSynchronizationManager, ClockSynchronizationConfig, ClockSynchronizer,
    ClockSynchronizationState, ClockSynchronizationStatus, ClockOffset,
    SynchronizationEvent, SynchronizationResult,
};

// Re-export protocol types
pub use protocols::{
    ClockSyncProtocol, NtpConfig, PtpConfig, SntpConfig, BerkeleyConfig,
    CristianConfig, CustomProtocolConfig, ProtocolManager, ProtocolError,
    NtpSynchronizer, PtpSynchronizer, SntpSynchronizer,
};

// Re-export source management types
pub use sources::{
    ClockSource, TimeSource, TimeSourceManager, TimeSourceConfig,
    SourceSelectionAlgorithm, SourceSelectionCriteria, SourceValidation,
    AtomicClockType, RadioTimeStation, SystemClockConfig,
};

// Re-export GPS synchronization types
pub use gps::{
    GpsConfig, GpsReceiverType, GpsSignalProcessing, GpsErrorCorrection,
    GpsSynchronizationManager, GpsTime, GpsError, AntennaConfig,
    IonosphericCorrection, TroposphericCorrection, SatelliteClockCorrection,
};

// Re-export network synchronization types
pub use network::{
    NetworkSyncConfig, NetworkTopology, MessagePassingConfig, NetworkLoadBalancing,
    NetworkFaultTolerance, NetworkSynchronizationManager, NetworkSyncError,
    SyncMessageType, MessagePriority, LoadBalancingAlgorithm,
};

// Re-export quality monitoring types
pub use quality::{
    ClockAccuracyRequirements, QualityRequirements, SourceQualityMonitoring,
    QualityMetric, QualityThresholds, ClockQualityMonitor, QualitySnapshot,
    QualityGrade, QualityMonitoringConfig, QualityAssessment,
};

// Re-export drift compensation types
pub use drift::{
    DriftCompensationConfig, DriftCompensationAlgorithm, DriftMeasurementConfig,
    DriftPredictionConfig, DriftCompensator, DriftModel, DriftPredictionEngine,
    DriftCompensationError, DriftMeasurement, DriftCompensationStatus,
};

// Re-export health monitoring types
pub use health::{
    SourceHealthMonitor, HealthCheck, HealthCheckType, HealthMonitorConfig,
    HealthThresholds, AlertConfiguration, RecoveryConfiguration, HealthAlert,
    HealthStatus, AlertSeverity, HealthMonitorError, SourceFailoverConfig,
};

// Re-export statistics and reporting types
pub use statistics::{
    ClockStatistics, PerformanceTracking, PerformanceHistory, PerformanceMeasurement,
    QualityReporting, ReportGeneration, StatisticsCollector, PerformanceReport,
    StatisticsError, TrendDirection, ReliabilityStatistics,
};

// Convenience type aliases
pub type Result<T> = std::result::Result<T, ClockSynchronizationError>;
pub type Duration = std::time::Duration;
pub type Instant = std::time::Instant;

/// Main error type for clock synchronization operations
#[derive(Debug)]
pub enum ClockSynchronizationError {
    /// Core synchronization error
    CoreError(core::ClockSynchronizationError),
    /// Protocol error
    ProtocolError(protocols::ProtocolError),
    /// Source management error
    SourceError(sources::SourceManagementError),
    /// GPS synchronization error
    GpsError(gps::GpsError),
    /// Network synchronization error
    NetworkError(network::NetworkSyncError),
    /// Quality monitoring error
    QualityError(quality::QualityMonitorError),
    /// Drift compensation error
    DriftError(drift::DriftCompensationError),
    /// Health monitoring error
    HealthError(health::HealthMonitorError),
    /// Statistics error
    StatisticsError(statistics::StatisticsError),
    /// Configuration error
    ConfigurationError(String),
    /// System error
    SystemError(String),
}

impl std::fmt::Display for ClockSynchronizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClockSynchronizationError::CoreError(e) => write!(f, "Core synchronization error: {}", e),
            ClockSynchronizationError::ProtocolError(e) => write!(f, "Protocol error: {}", e),
            ClockSynchronizationError::SourceError(e) => write!(f, "Source management error: {}", e),
            ClockSynchronizationError::GpsError(e) => write!(f, "GPS synchronization error: {}", e),
            ClockSynchronizationError::NetworkError(e) => write!(f, "Network synchronization error: {}", e),
            ClockSynchronizationError::QualityError(e) => write!(f, "Quality monitoring error: {}", e),
            ClockSynchronizationError::DriftError(e) => write!(f, "Drift compensation error: {}", e),
            ClockSynchronizationError::HealthError(e) => write!(f, "Health monitoring error: {}", e),
            ClockSynchronizationError::StatisticsError(e) => write!(f, "Statistics error: {}", e),
            ClockSynchronizationError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            ClockSynchronizationError::SystemError(msg) => write!(f, "System error: {}", msg),
        }
    }
}

impl std::error::Error for ClockSynchronizationError {}

// Error conversions for seamless error handling
impl From<core::ClockSynchronizationError> for ClockSynchronizationError {
    fn from(err: core::ClockSynchronizationError) -> Self {
        ClockSynchronizationError::CoreError(err)
    }
}

impl From<protocols::ProtocolError> for ClockSynchronizationError {
    fn from(err: protocols::ProtocolError) -> Self {
        ClockSynchronizationError::ProtocolError(err)
    }
}

impl From<sources::SourceManagementError> for ClockSynchronizationError {
    fn from(err: sources::SourceManagementError) -> Self {
        ClockSynchronizationError::SourceError(err)
    }
}

impl From<gps::GpsError> for ClockSynchronizationError {
    fn from(err: gps::GpsError) -> Self {
        ClockSynchronizationError::GpsError(err)
    }
}

impl From<network::NetworkSyncError> for ClockSynchronizationError {
    fn from(err: network::NetworkSyncError) -> Self {
        ClockSynchronizationError::NetworkError(err)
    }
}

impl From<quality::QualityMonitorError> for ClockSynchronizationError {
    fn from(err: quality::QualityMonitorError) -> Self {
        ClockSynchronizationError::QualityError(err)
    }
}

impl From<drift::DriftCompensationError> for ClockSynchronizationError {
    fn from(err: drift::DriftCompensationError) -> Self {
        ClockSynchronizationError::DriftError(err)
    }
}

impl From<health::HealthMonitorError> for ClockSynchronizationError {
    fn from(err: health::HealthMonitorError) -> Self {
        ClockSynchronizationError::HealthError(err)
    }
}

impl From<statistics::StatisticsError> for ClockSynchronizationError {
    fn from(err: statistics::StatisticsError) -> Self {
        ClockSynchronizationError::StatisticsError(err)
    }
}

/// Builder for configuring clock synchronization
///
/// Provides a fluent interface for configuring the various aspects
/// of clock synchronization with sensible defaults.
#[derive(Debug)]
pub struct ClockSynchronizationBuilder {
    core_config: Option<core::ClockSynchronizationConfig>,
    protocol_configs: Vec<protocols::ClockSyncProtocol>,
    source_configs: Vec<sources::TimeSource>,
    gps_config: Option<gps::GpsConfig>,
    network_config: Option<network::NetworkSyncConfig>,
    quality_config: Option<quality::QualityMonitoringConfig>,
    drift_config: Option<drift::DriftCompensationConfig>,
    health_config: Option<health::HealthMonitorConfig>,
    statistics_config: Option<statistics::StatisticsCollectionConfig>,
}

impl ClockSynchronizationBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Self {
        Self {
            core_config: None,
            protocol_configs: Vec::new(),
            source_configs: Vec::new(),
            gps_config: None,
            network_config: None,
            quality_config: None,
            drift_config: None,
            health_config: None,
            statistics_config: None,
        }
    }

    /// Set core synchronization configuration
    pub fn with_core_config(mut self, config: core::ClockSynchronizationConfig) -> Self {
        self.core_config = Some(config);
        self
    }

    /// Add synchronization protocol
    pub fn with_protocol(mut self, protocol: protocols::ClockSyncProtocol) -> Self {
        self.protocol_configs.push(protocol);
        self
    }

    /// Add time source
    pub fn with_source(mut self, source: sources::TimeSource) -> Self {
        self.source_configs.push(source);
        self
    }

    /// Set GPS configuration
    pub fn with_gps_config(mut self, config: gps::GpsConfig) -> Self {
        self.gps_config = Some(config);
        self
    }

    /// Set network synchronization configuration
    pub fn with_network_config(mut self, config: network::NetworkSyncConfig) -> Self {
        self.network_config = Some(config);
        self
    }

    /// Set quality monitoring configuration
    pub fn with_quality_config(mut self, config: quality::QualityMonitoringConfig) -> Self {
        self.quality_config = Some(config);
        self
    }

    /// Set drift compensation configuration
    pub fn with_drift_config(mut self, config: drift::DriftCompensationConfig) -> Self {
        self.drift_config = Some(config);
        self
    }

    /// Set health monitoring configuration
    pub fn with_health_config(mut self, config: health::HealthMonitorConfig) -> Self {
        self.health_config = Some(config);
        self
    }

    /// Set statistics collection configuration
    pub fn with_statistics_config(mut self, config: statistics::StatisticsCollectionConfig) -> Self {
        self.statistics_config = Some(config);
        self
    }

    /// Build the clock synchronization manager
    pub fn build(self) -> Result<ClockSynchronizationManager> {
        let core_config = self.core_config.unwrap_or_default();

        // Create and configure the synchronization manager
        let mut manager = ClockSynchronizationManager::new(core_config)?;

        // Configure protocols
        for protocol in self.protocol_configs {
            manager.add_protocol(protocol)?;
        }

        // Configure sources
        for source in self.source_configs {
            manager.add_time_source(sources::ClockSource::from_time_source(source))?;
        }

        // Apply additional configurations
        if let Some(gps_config) = self.gps_config {
            manager.configure_gps(gps_config)?;
        }

        if let Some(network_config) = self.network_config {
            manager.configure_network(network_config)?;
        }

        if let Some(quality_config) = self.quality_config {
            manager.configure_quality_monitoring(quality_config)?;
        }

        if let Some(drift_config) = self.drift_config {
            manager.configure_drift_compensation(drift_config)?;
        }

        if let Some(health_config) = self.health_config {
            manager.configure_health_monitoring(health_config)?;
        }

        if let Some(statistics_config) = self.statistics_config {
            manager.configure_statistics(statistics_config)?;
        }

        Ok(manager)
    }
}

impl Default for ClockSynchronizationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for clock synchronization
pub mod utils {
    use super::*;

    /// Create a basic NTP-based synchronization setup
    pub fn create_ntp_sync_manager(ntp_servers: Vec<String>) -> Result<ClockSynchronizationManager> {
        let mut builder = ClockSynchronizationBuilder::new();

        // Add NTP protocol
        builder = builder.with_protocol(protocols::ClockSyncProtocol::NTP {
            version: 4,
            servers: ntp_servers,
            authentication: false,
        });

        // Add network time sources
        for (i, server) in builder.source_configs.iter().enumerate() {
            if let sources::TimeSource::Network { server, .. } = server {
                builder = builder.with_source(sources::TimeSource::Network {
                    server: server.clone(),
                    port: 123,
                    protocol: network::NetworkTimeProtocol::NTP,
                    authentication: None,
                });
            }
        }

        // Enable basic monitoring
        builder = builder.with_quality_config(quality::QualityMonitoringConfig::default());
        builder = builder.with_health_config(health::HealthMonitorConfig::default());

        builder.build()
    }

    /// Create a GPS-based synchronization setup
    pub fn create_gps_sync_manager(gps_config: gps::GpsConfig) -> Result<ClockSynchronizationManager> {
        let mut builder = ClockSynchronizationBuilder::new();

        // Add GPS configuration
        builder = builder.with_gps_config(gps_config.clone());

        // Add GPS time source
        builder = builder.with_source(sources::TimeSource::GPS {
            receiver_config: gps_config,
        });

        // Enable comprehensive monitoring for GPS
        builder = builder.with_quality_config(quality::QualityMonitoringConfig::default());
        builder = builder.with_drift_config(drift::DriftCompensationConfig::default());
        builder = builder.with_health_config(health::HealthMonitorConfig::default());

        builder.build()
    }

    /// Create a high-precision synchronization setup
    pub fn create_precision_sync_manager() -> Result<ClockSynchronizationManager> {
        let mut builder = ClockSynchronizationBuilder::new();

        // Use PTP for high precision
        builder = builder.with_protocol(protocols::ClockSyncProtocol::PTP {
            version: protocols::PtpVersion::V2,
            domain: 0,
            transport: protocols::PtpTransport::Ethernet,
            profile: protocols::PtpProfile::Default,
        });

        // Add atomic clock source
        builder = builder.with_source(sources::TimeSource::AtomicClock {
            clock_type: sources::AtomicClockType::Cesium {
                frequency: 9192631770.0,
                stability: 1e-15,
            },
            calibration: sources::ClockCalibration {
                frequency: Duration::from_secs(3600),
                reference: sources::CalibrationReference::PrimaryStandard {
                    standard: "NIST-F1".to_string(),
                },
                method: sources::CalibrationMethod::DirectComparison,
                accuracy: 1e-15,
            },
        });

        // Enable all monitoring and compensation
        builder = builder.with_quality_config(quality::QualityMonitoringConfig::default());
        builder = builder.with_drift_config(drift::DriftCompensationConfig::default());
        builder = builder.with_health_config(health::HealthMonitorConfig::default());
        builder = builder.with_statistics_config(statistics::StatisticsCollectionConfig::default());

        builder.build()
    }

    /// Convert duration to human-readable string
    pub fn duration_to_string(duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let days = total_seconds / 86400;
        let hours = (total_seconds % 86400) / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let millis = duration.subsec_millis();
        let micros = duration.subsec_micros() % 1000;
        let nanos = duration.subsec_nanos() % 1000;

        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, seconds)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}.{:03}s", seconds, millis)
        } else if millis > 0 {
            format!("{}.{:03}ms", millis, micros)
        } else if micros > 0 {
            format!("{}.{:03}μs", micros, nanos)
        } else {
            format!("{}ns", nanos)
        }
    }

    /// Validate clock offset against requirements
    pub fn validate_clock_offset(
        offset: ClockOffset,
        requirements: &quality::ClockAccuracyRequirements,
    ) -> bool {
        offset <= requirements.max_skew
    }

    /// Calculate quality score from multiple metrics
    pub fn calculate_quality_score(metrics: &std::collections::HashMap<String, f64>) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let sum: f64 = metrics.values().sum();
        sum / metrics.len() as f64
    }

    /// Get system uptime
    pub fn get_system_uptime() -> Duration {
        // This would be implemented to get actual system uptime
        // For now, return a placeholder
        Duration::from_secs(86400) // 1 day
    }
}

/// Prelude module for common imports
pub mod prelude {
    pub use super::{
        ClockSynchronizationManager, ClockSynchronizationConfig, ClockSynchronizationBuilder,
        ClockOffset, Result, ClockSynchronizationError,
    };

    pub use super::protocols::{ClockSyncProtocol, NtpConfig, PtpConfig};
    pub use super::sources::{ClockSource, TimeSource, AtomicClockType};
    pub use super::quality::{QualityGrade, QualityMetric, TrendDirection};
    pub use super::health::{AlertSeverity, HealthCheckType};
    pub use super::statistics::{PerformanceMetric, ReportFormat};
    pub use super::utils;
}

// Module-level documentation tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_pattern() {
        let builder = ClockSynchronizationBuilder::new()
            .with_protocol(protocols::ClockSyncProtocol::NTP {
                version: 4,
                servers: vec!["pool.ntp.org".to_string()],
                authentication: false,
            });

        // Builder should be constructible
        assert!(builder.protocol_configs.len() == 1);
    }

    #[test]
    fn test_error_conversions() {
        let core_error = core::ClockSynchronizationError::ConfigurationError("test".to_string());
        let sync_error: ClockSynchronizationError = core_error.into();

        match sync_error {
            ClockSynchronizationError::CoreError(_) => {},
            _ => panic!("Error conversion failed"),
        }
    }

    #[test]
    fn test_utility_functions() {
        // Test duration formatting
        let duration = Duration::from_millis(1500);
        let formatted = utils::duration_to_string(duration);
        assert!(formatted.contains("s"));

        // Test quality score calculation
        let mut metrics = std::collections::HashMap::new();
        metrics.insert("accuracy".to_string(), 0.9);
        metrics.insert("stability".to_string(), 0.8);
        let score = utils::calculate_quality_score(&metrics);
        assert_eq!(score, 0.85);
    }
}