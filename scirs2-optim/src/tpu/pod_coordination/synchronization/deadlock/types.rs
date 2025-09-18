//! Core Types for Deadlock Detection
//!
//! This module contains the fundamental types, structures, and enums
//! used throughout the deadlock detection system.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use crate::tpu::tpu_backend::DeviceId;

/// Deadlock detector for resource management
#[derive(Debug)]
pub struct DeadlockDetector {
    /// Detection configuration
    pub config: DeadlockDetectionConfig,
    /// Resource dependency graph
    pub dependency_graph: super::graph::DependencyGraph,
    /// Detection state
    pub detection_state: DetectionState,
    /// Detection statistics
    pub statistics: super::performance::DeadlockStatistics,
    /// Prevention system
    pub prevention_system: super::prevention::DeadlockPreventionSystem,
    /// Recovery system
    pub recovery_system: super::recovery::DeadlockRecoverySystem,
}

/// Deadlock detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlockDetectionConfig {
    /// Enable deadlock detection
    pub enable: bool,
    /// Detection algorithm
    pub algorithm: super::algorithms::DeadlockDetectionAlgorithm,
    /// Detection frequency
    pub frequency: Duration,
    /// Detection sensitivity
    pub sensitivity: DeadlockSensitivity,
    /// Prevention strategies
    pub prevention: super::prevention::DeadlockPrevention,
    /// Recovery strategies
    pub recovery: super::recovery::DeadlockRecovery,
    /// Performance tuning
    pub performance: DeadlockPerformanceConfig,
    /// Advanced features
    pub advanced: AdvancedDeadlockConfig,
}

/// Detection state management
#[derive(Debug, Clone)]
pub struct DetectionState {
    /// Current detection status
    pub status: DetectionStatus,
    /// Last detection time
    pub last_detection: Instant,
    /// Active deadlock count
    pub active_deadlocks: usize,
    /// State history
    pub history: Vec<DetectionEvent>,
}

/// Detection status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectionStatus {
    /// System is idle
    Idle,
    /// Detection in progress
    Running,
    /// Deadlock detected
    DeadlockDetected,
    /// Prevention active
    PreventionActive,
    /// Recovery in progress
    RecoveryInProgress,
    /// System error
    Error,
}

/// Detection event for history tracking
#[derive(Debug, Clone)]
pub struct DetectionEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Event type
    pub event_type: DetectionEventType,
    /// Event details
    pub details: String,
    /// Associated resources
    pub resources: Vec<String>,
}

/// Types of detection events
#[derive(Debug, Clone, Copy)]
pub enum DetectionEventType {
    /// Detection started
    DetectionStarted,
    /// Detection completed
    DetectionCompleted,
    /// Deadlock found
    DeadlockFound,
    /// Prevention triggered
    PreventionTriggered,
    /// Recovery initiated
    RecoveryInitiated,
    /// Error occurred
    Error,
}

/// Deadlock sensitivity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlockSensitivity {
    /// Base sensitivity level
    pub base_level: f64,
    /// Adaptive sensitivity
    pub adaptive: AdaptiveSensitivity,
    /// Sensitivity metrics
    pub metrics: Vec<SensitivityMetric>,
}

/// Adaptive sensitivity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSensitivity {
    /// Enable adaptive adjustment
    pub enabled: bool,
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Adaptation window size
    pub window_size: usize,
    /// Minimum sensitivity threshold
    pub min_threshold: f64,
    /// Maximum sensitivity threshold
    pub max_threshold: f64,
}

/// Sensitivity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensitivityMetric {
    /// False positive rate
    FalsePositiveRate,
    /// False negative rate
    FalseNegativeRate,
    /// Detection latency
    DetectionLatency,
    /// Resource overhead
    ResourceOverhead,
}

/// Performance configuration for deadlock detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlockPerformanceConfig {
    /// Maximum detection time
    pub max_detection_time: Duration,
    /// Resource usage limits
    pub resource_limits: ResourceLimits,
    /// Optimization settings
    pub optimization: PerformanceOptimization,
}

/// Resource usage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage (bytes)
    pub max_memory: usize,
    /// Maximum CPU usage (percentage)
    pub max_cpu: f64,
    /// Maximum thread count
    pub max_threads: usize,
}

/// Performance optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimization {
    /// Enable aggressive optimization
    pub aggressive: bool,
    /// Optimization level (0-3)
    pub level: u8,
    /// Trade accuracy for speed
    pub speed_over_accuracy: bool,
}

/// Advanced deadlock configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedDeadlockConfig {
    /// Enable experimental features
    pub experimental: bool,
    /// Custom algorithms
    pub custom_algorithms: Vec<String>,
    /// Advanced diagnostics
    pub diagnostics: AdvancedDiagnostics,
    /// Integration settings
    pub integration: IntegrationSettings,
}

/// Advanced diagnostics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedDiagnostics {
    /// Enable detailed logging
    pub detailed_logging: bool,
    /// Performance profiling
    pub profiling: bool,
    /// Trace collection
    pub trace_collection: bool,
    /// Statistics collection interval
    pub stats_interval: Duration,
}

/// Integration settings for external systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationSettings {
    /// External monitoring systems
    pub monitoring_endpoints: Vec<String>,
    /// Notification systems
    pub notification_config: NotificationConfig,
    /// Export formats
    pub export_formats: Vec<ExportFormat>,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Enable notifications
    pub enabled: bool,
    /// Notification types
    pub types: Vec<NotificationType>,
    /// Delivery methods
    pub delivery: Vec<DeliveryMethod>,
}

/// Notification types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    /// Deadlock detected
    DeadlockDetected,
    /// Prevention activated
    PreventionActivated,
    /// Recovery completed
    RecoveryCompleted,
    /// System error
    SystemError,
}

/// Delivery methods for notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryMethod {
    /// Email notification
    Email { address: String },
    /// Webhook notification
    Webhook { url: String },
    /// System log
    SystemLog,
    /// Custom handler
    Custom { handler: String },
}

/// Export formats for diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// Binary format
    Binary,
    /// Custom format
    Custom { format: String },
}

impl Default for DeadlockDetectionConfig {
    fn default() -> Self {
        Self {
            enable: true,
            algorithm: super::algorithms::DeadlockDetectionAlgorithm::WaitForGraph {
                cycle_detection: super::algorithms::CycleDetectionMethod::DepthFirstSearch,
                optimization: super::algorithms::GraphOptimization::default(),
            },
            frequency: Duration::from_millis(100),
            sensitivity: DeadlockSensitivity::default(),
            prevention: super::prevention::DeadlockPrevention::default(),
            recovery: super::recovery::DeadlockRecovery::default(),
            performance: DeadlockPerformanceConfig::default(),
            advanced: AdvancedDeadlockConfig::default(),
        }
    }
}

impl Default for DeadlockSensitivity {
    fn default() -> Self {
        Self {
            base_level: 0.8,
            adaptive: AdaptiveSensitivity::default(),
            metrics: vec![
                SensitivityMetric::FalsePositiveRate,
                SensitivityMetric::DetectionLatency,
            ],
        }
    }
}

impl Default for AdaptiveSensitivity {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_rate: 0.01,
            window_size: 100,
            min_threshold: 0.1,
            max_threshold: 1.0,
        }
    }
}

impl Default for DeadlockPerformanceConfig {
    fn default() -> Self {
        Self {
            max_detection_time: Duration::from_secs(1),
            resource_limits: ResourceLimits::default(),
            optimization: PerformanceOptimization::default(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 100 * 1024 * 1024, // 100 MB
            max_cpu: 50.0,                 // 50%
            max_threads: 8,
        }
    }
}

impl Default for PerformanceOptimization {
    fn default() -> Self {
        Self {
            aggressive: false,
            level: 2,
            speed_over_accuracy: false,
        }
    }
}

impl Default for AdvancedDeadlockConfig {
    fn default() -> Self {
        Self {
            experimental: false,
            custom_algorithms: Vec::new(),
            diagnostics: AdvancedDiagnostics::default(),
            integration: IntegrationSettings::default(),
        }
    }
}

impl Default for AdvancedDiagnostics {
    fn default() -> Self {
        Self {
            detailed_logging: false,
            profiling: false,
            trace_collection: false,
            stats_interval: Duration::from_secs(60),
        }
    }
}

impl Default for IntegrationSettings {
    fn default() -> Self {
        Self {
            monitoring_endpoints: Vec::new(),
            notification_config: NotificationConfig::default(),
            export_formats: vec![ExportFormat::JSON],
        }
    }
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            types: vec![NotificationType::DeadlockDetected],
            delivery: vec![DeliveryMethod::SystemLog],
        }
    }
}