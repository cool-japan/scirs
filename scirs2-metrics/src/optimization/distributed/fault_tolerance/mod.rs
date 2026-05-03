//! Fault tolerance and recovery management
//!
//! This module provides comprehensive fault tolerance capabilities:
//! - Automatic failure detection and recovery
//! - Health monitoring and alerting
//! - Node replacement strategies
//! - Data backup and restoration
//! - Circuit breaker patterns

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

pub use super::config::{FaultToleranceConfig, NodeReplacementStrategy};

/// Recovery action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryActionType {
    /// Node failover to backup
    NodeFailover,
    /// Data replication to maintain redundancy
    DataReplication,
    /// Network healing and reconnection
    NetworkHeal,
    /// Service restart
    ServiceRestart,
    /// Resource scaling
    ResourceScaling,
    /// Configuration rollback
    ConfigRollback,
}

/// Node health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeHealthStatus {
    /// Node is healthy and responsive
    Healthy,
    /// Node is degraded but functional
    Degraded,
    /// Node has failed
    Failed,
    /// Node status is unknown
    Unknown,
    /// Node is recovering from failure
    Recovering,
    /// Node is being maintained
    Maintenance,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Error alert
    Error,
    /// Critical alert
    Critical,
    /// Emergency alert
    Emergency,
}

/// Recovery strategy options
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Immediate recovery action
    Immediate,
    /// Graceful recovery with delay
    Graceful { delay: Duration },
    /// Manual recovery only
    Manual,
    /// Automatic with fallback to manual
    AutomaticWithFallback,
    /// Progressive recovery (try multiple strategies)
    Progressive,
}

/// Fault recovery manager
#[derive(Debug)]
pub struct FaultRecoveryManager {
    /// Configuration
    config: FaultToleranceConfig,
    /// Health monitor
    health_monitor: HealthMonitor,
    /// Recovery actions history
    recovery_history: Arc<RwLock<VecDeque<RecoveryAction>>>,
    /// Alert thresholds
    alert_thresholds: AlertThresholds,
    /// Active recovery operations
    active_recoveries: Arc<Mutex<HashMap<String, RecoveryOperation>>>,
    /// Node replacement queue
    replacement_queue: Arc<Mutex<VecDeque<NodeReplacementRequest>>>,
}

impl FaultRecoveryManager {
    /// Create a new fault recovery manager
    pub fn new(config: FaultToleranceConfig) -> Self {
        Self {
            health_monitor: HealthMonitor::new(config.health_check_interval),
            alert_thresholds: AlertThresholds::default(),
            recovery_history: Arc::new(RwLock::new(VecDeque::new())),
            active_recoveries: Arc::new(Mutex::new(HashMap::new())),
            replacement_queue: Arc::new(Mutex::new(VecDeque::new())),
            config,
        }
    }

    /// Start fault monitoring and recovery
    pub fn start(&mut self) -> Result<()> {
        self.health_monitor.start()?;
        Ok(())
    }

    /// Stop fault monitoring
    pub fn stop(&mut self) -> Result<()> {
        self.health_monitor.stop()?;
        Ok(())
    }

    /// Register a node for monitoring
    pub fn register_node(&mut self, node_id: String, metrics: NodeMetrics) -> Result<()> {
        self.health_monitor.register_node(node_id, metrics)
    }

    /// Unregister a node from monitoring
    pub fn unregister_node(&mut self, node_id: &str) -> Result<()> {
        self.health_monitor.unregister_node(node_id)
    }

    /// Update node metrics
    pub fn update_node_metrics(&mut self, node_id: &str, metrics: NodeMetrics) -> Result<()> {
        self.health_monitor
            .update_metrics(node_id, metrics.clone())?;

        // Check if recovery action is needed
        if let Some(action) = self.evaluate_recovery_need(node_id, &metrics)? {
            self.trigger_recovery(action)?;
        }

        Ok(())
    }

    /// Evaluate if recovery action is needed for a node
    fn evaluate_recovery_need(
        &self,
        node_id: &str,
        metrics: &NodeMetrics,
    ) -> Result<Option<RecoveryAction>> {
        // Check CPU threshold
        if metrics.cpu_usage > self.alert_thresholds.cpu_critical {
            return Ok(Some(RecoveryAction {
                id: format!(
                    "recovery_{}_cpu_{}",
                    node_id,
                    Instant::now().elapsed().as_millis()
                ),
                action_type: RecoveryActionType::ResourceScaling,
                target_node: node_id.to_string(),
                severity: AlertSeverity::Critical,
                description: format!("High CPU usage: {}%", metrics.cpu_usage),
                strategy: RecoveryStrategy::Immediate,
                created_at: SystemTime::now(),
                started_at: None,
                completed_at: None,
                status: RecoveryStatus::Pending,
                error: None,
            }));
        }

        // Check memory threshold
        if metrics.memory_usage > self.alert_thresholds.memory_critical {
            return Ok(Some(RecoveryAction {
                id: format!(
                    "recovery_{}_memory_{}",
                    node_id,
                    Instant::now().elapsed().as_millis()
                ),
                action_type: RecoveryActionType::ResourceScaling,
                target_node: node_id.to_string(),
                severity: AlertSeverity::Critical,
                description: format!("High memory usage: {}%", metrics.memory_usage),
                strategy: RecoveryStrategy::Immediate,
                created_at: SystemTime::now(),
                started_at: None,
                completed_at: None,
                status: RecoveryStatus::Pending,
                error: None,
            }));
        }

        // Check if node is unresponsive
        let last_heartbeat_age = metrics
            .last_heartbeat
            .elapsed()
            .unwrap_or_else(|_| Duration::from_secs(0));
        if last_heartbeat_age > Duration::from_secs(self.config.health_check_interval * 3) {
            return Ok(Some(RecoveryAction {
                id: format!(
                    "recovery_{}_heartbeat_{}",
                    node_id,
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis()
                ),
                action_type: RecoveryActionType::NodeFailover,
                target_node: node_id.to_string(),
                severity: AlertSeverity::Emergency,
                description: format!("Node unresponsive for {:?}", last_heartbeat_age),
                strategy: RecoveryStrategy::Immediate,
                created_at: SystemTime::now(),
                started_at: None,
                completed_at: None,
                status: RecoveryStatus::Pending,
                error: None,
            }));
        }

        Ok(None)
    }

    /// Trigger a recovery action
    pub fn trigger_recovery(&mut self, action: RecoveryAction) -> Result<String> {
        if !self.config.auto_recovery && action.strategy != RecoveryStrategy::Manual {
            // Log the action but don't execute if auto recovery is disabled
            self.log_action(action);
            return Ok(
                "Recovery action logged but not executed (auto recovery disabled)".to_string(),
            );
        }

        let action_id = action.id.clone();
        let mut active_recoveries = self.active_recoveries.lock().expect("Operation failed");

        let recovery_op = RecoveryOperation {
            action: action.clone(),
            progress: 0.0,
            estimated_completion: None,
        };

        active_recoveries.insert(action_id.clone(), recovery_op);
        drop(active_recoveries);

        // Execute the recovery action
        match action.action_type {
            RecoveryActionType::NodeFailover => {
                self.execute_node_failover(&action)?;
            }
            RecoveryActionType::DataReplication => {
                self.execute_data_replication(&action)?;
            }
            RecoveryActionType::NetworkHeal => {
                self.execute_network_heal(&action)?;
            }
            RecoveryActionType::ServiceRestart => {
                self.execute_service_restart(&action)?;
            }
            RecoveryActionType::ResourceScaling => {
                self.execute_resource_scaling(&action)?;
            }
            RecoveryActionType::ConfigRollback => {
                self.execute_config_rollback(&action)?;
            }
        }

        self.log_action(action);
        Ok(action_id)
    }

    /// Execute node failover.
    ///
    /// Marks the failed node in the monitoring table as `Failed` and queues
    /// a replacement request.  The actual workload migration (shard
    /// reassignment) is coordinated by the higher-level coordinator; here we
    /// record the action and enqueue the request.
    fn execute_node_failover(&self, action: &RecoveryAction) -> Result<()> {
        let node_id = &action.target_node;

        // Mark node as Failed in health monitor.
        if let Ok(mut nodes) = self.health_monitor.nodes.write() {
            if let Some(info) = nodes.get_mut(node_id.as_str()) {
                info.health_status = NodeHealthStatus::Failed;
                info.failure_count += 1;
            }
        }

        // Queue a replacement request.
        match self.config.replacement_strategy {
            NodeReplacementStrategy::Immediate | NodeReplacementStrategy::HotStandby => {
                let mut queue = self.replacement_queue.lock().expect("Operation failed");
                queue.push_back(NodeReplacementRequest {
                    failed_node: node_id.clone(),
                    replacement_type: self.config.replacement_strategy.clone(),
                    requested_at: SystemTime::now(),
                    priority: match action.severity {
                        AlertSeverity::Emergency | AlertSeverity::Critical => {
                            ReplacementPriority::High
                        }
                        _ => ReplacementPriority::Normal,
                    },
                });
            }
            _ => {}
        }

        Ok(())
    }

    /// Execute data replication.
    ///
    /// Records that replication was triggered for the target node.
    /// In a real system this would dispatch data copy jobs to replicas.
    fn execute_data_replication(&self, action: &RecoveryAction) -> Result<()> {
        // Mark the node as recovering so the health sweep counts it correctly.
        if let Ok(mut nodes) = self.health_monitor.nodes.write() {
            if let Some(info) = nodes.get_mut(action.target_node.as_str()) {
                info.health_status = NodeHealthStatus::Recovering;
                info.recovery_attempts += 1;
            }
        }
        Ok(())
    }

    /// Execute network healing.
    ///
    /// Resets the failure count for the target node and marks it Healthy
    /// (assuming the network link has been re-established).
    fn execute_network_heal(&self, action: &RecoveryAction) -> Result<()> {
        if let Ok(mut nodes) = self.health_monitor.nodes.write() {
            if let Some(info) = nodes.get_mut(action.target_node.as_str()) {
                // Treat this as a fresh start: clear failure counter.
                info.failure_count = 0;
                info.health_status = NodeHealthStatus::Recovering;
                // Update heartbeat to now so the next sweep sees it as alive.
                info.current_metrics.last_heartbeat = SystemTime::now();
            }
        }
        Ok(())
    }

    /// Execute service restart.
    ///
    /// Marks the node as `Recovering`, resets consecutive failure count,
    /// and updates the last-heartbeat so the node is not immediately
    /// re-triggered for failover.
    fn execute_service_restart(&self, action: &RecoveryAction) -> Result<()> {
        if let Ok(mut nodes) = self.health_monitor.nodes.write() {
            if let Some(info) = nodes.get_mut(action.target_node.as_str()) {
                info.health_status = NodeHealthStatus::Recovering;
                info.failure_count = 0;
                info.recovery_attempts += 1;
                // Simulate grace period by pretending a heartbeat just arrived.
                info.current_metrics.last_heartbeat = SystemTime::now();
            }
        }
        Ok(())
    }

    /// Execute resource scaling.
    ///
    /// Updates the node's resource-usage metric so that subsequent health
    /// checks see reduced load.  A real implementation would provision
    /// additional capacity.
    fn execute_resource_scaling(&self, action: &RecoveryAction) -> Result<()> {
        if let Ok(mut nodes) = self.health_monitor.nodes.write() {
            if let Some(info) = nodes.get_mut(action.target_node.as_str()) {
                // Simulate capacity increase: halve current resource utilisation.
                info.current_metrics.cpu_usage =
                    (info.current_metrics.cpu_usage / 2.0).clamp(0.0, 100.0);
                info.current_metrics.memory_usage =
                    (info.current_metrics.memory_usage / 2.0).clamp(0.0, 100.0);
                // Re-evaluate health after scaling.
                info.health_status = self
                    .health_monitor
                    .determine_health_status(&info.current_metrics);
            }
        }
        Ok(())
    }

    /// Execute configuration rollback.
    ///
    /// Records the rollback event in the recovery history.  A real
    /// implementation would restore a saved configuration snapshot from a
    /// version-history stack.
    fn execute_config_rollback(&self, action: &RecoveryAction) -> Result<()> {
        // Mark node as Recovering after config rollback.
        if let Ok(mut nodes) = self.health_monitor.nodes.write() {
            if let Some(info) = nodes.get_mut(action.target_node.as_str()) {
                info.health_status = NodeHealthStatus::Recovering;
                info.recovery_attempts += 1;
            }
        }
        Ok(())
    }

    /// Log a recovery action
    fn log_action(&self, action: RecoveryAction) {
        let mut history = self.recovery_history.write().expect("Operation failed");
        history.push_back(action);

        // Keep only recent history
        while history.len() > 10000 {
            history.pop_front();
        }
    }

    /// Get recovery history
    pub fn get_recovery_history(&self) -> Vec<RecoveryAction> {
        let history = self.recovery_history.read().expect("Operation failed");
        history.iter().cloned().collect()
    }

    /// Get active recovery operations
    pub fn get_active_recoveries(&self) -> Vec<RecoveryOperation> {
        let active = self.active_recoveries.lock().expect("Operation failed");
        active.values().cloned().collect()
    }

    /// Complete a recovery operation
    pub fn complete_recovery(
        &mut self,
        action_id: &str,
        success: bool,
        error: Option<String>,
    ) -> Result<()> {
        let mut active_recoveries = self.active_recoveries.lock().expect("Operation failed");

        if let Some(mut recovery_op) = active_recoveries.remove(action_id) {
            recovery_op.action.completed_at = Some(SystemTime::now());
            recovery_op.action.status = if success {
                RecoveryStatus::Completed
            } else {
                RecoveryStatus::Failed
            };
            recovery_op.action.error = error;
            recovery_op.progress = 1.0;

            // Update history
            self.log_action(recovery_op.action);
        }

        Ok(())
    }

    /// Get cluster health summary
    pub fn get_health_summary(&self) -> HealthSummary {
        self.health_monitor.get_health_summary()
    }

    /// Update alert thresholds
    pub fn update_alert_thresholds(&mut self, thresholds: AlertThresholds) {
        self.alert_thresholds = thresholds;
    }

    /// Process node replacement requests
    pub fn process_replacement_requests(&mut self) -> Result<Vec<NodeReplacementRequest>> {
        let mut queue = self.replacement_queue.lock().expect("Operation failed");
        let requests: Vec<_> = queue.drain(..).collect();
        Ok(requests)
    }
}

/// Recovery action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAction {
    /// Unique action ID
    pub id: String,
    /// Type of recovery action
    pub action_type: RecoveryActionType,
    /// Target node for the action
    pub target_node: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Human-readable description
    pub description: String,
    /// Recovery strategy
    pub strategy: RecoveryStrategy,
    /// When action was created
    pub created_at: SystemTime,
    /// When action started execution
    pub started_at: Option<SystemTime>,
    /// When action completed
    pub completed_at: Option<SystemTime>,
    /// Current status
    pub status: RecoveryStatus,
    /// Error message if failed
    pub error: Option<String>,
}

/// Recovery action status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecoveryStatus {
    /// Action is pending execution
    Pending,
    /// Action is in progress
    InProgress,
    /// Action completed successfully
    Completed,
    /// Action failed
    Failed,
    /// Action was cancelled
    Cancelled,
}

/// Recovery operation (action with progress tracking)
#[derive(Debug, Clone)]
pub struct RecoveryOperation {
    /// The recovery action
    pub action: RecoveryAction,
    /// Progress percentage (0.0 - 1.0)
    pub progress: f64,
    /// Estimated completion time
    pub estimated_completion: Option<SystemTime>,
}

/// Health monitoring system
#[derive(Debug)]
pub struct HealthMonitor {
    /// Monitored nodes
    nodes: Arc<RwLock<HashMap<String, NodeMonitoringInfo>>>,
    /// Check interval in seconds
    check_interval: u64,
    /// Flag to signal the background thread to stop.
    running: Arc<AtomicBool>,
    /// Background thread handle (held so we can join on stop).
    monitor_thread: Option<std::thread::JoinHandle<()>>,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(check_interval: u64) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            check_interval,
            running: Arc::new(AtomicBool::new(false)),
            monitor_thread: None,
        }
    }

    /// Start the background monitoring thread.
    pub fn start(&mut self) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            // Already running.
            return Ok(());
        }

        self.running.store(true, Ordering::SeqCst);

        let running = Arc::clone(&self.running);
        let nodes = Arc::clone(&self.nodes);
        let interval_secs = self.check_interval;

        let handle = std::thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                // Perform a health check sweep.
                if let Ok(mut node_map) = nodes.write() {
                    for info in node_map.values_mut() {
                        let heartbeat_age = info
                            .current_metrics
                            .last_heartbeat
                            .elapsed()
                            .unwrap_or(Duration::from_secs(0));

                        info.health_status =
                            if heartbeat_age > Duration::from_secs(interval_secs * 3) {
                                NodeHealthStatus::Failed
                            } else if info.current_metrics.cpu_usage > 85.0
                                || info.current_metrics.memory_usage > 85.0
                            {
                                NodeHealthStatus::Degraded
                            } else {
                                NodeHealthStatus::Healthy
                            };

                        info.last_check = Instant::now();
                    }
                }

                // Sleep between sweeps.
                std::thread::sleep(Duration::from_secs(interval_secs.max(1)));
            }
        });

        self.monitor_thread = Some(handle);
        Ok(())
    }

    /// Signal the background thread to stop and wait for it to exit.
    pub fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);

        if let Some(handle) = self.monitor_thread.take() {
            // Best-effort join; we don't propagate thread panics as errors.
            let _ = handle.join();
        }

        Ok(())
    }

    /// Register a node for monitoring
    pub fn register_node(&mut self, node_id: String, metrics: NodeMetrics) -> Result<()> {
        let mut nodes = self.nodes.write().expect("Operation failed");

        let monitoring_info = NodeMonitoringInfo {
            node_id: node_id.clone(),
            current_metrics: metrics,
            health_status: NodeHealthStatus::Healthy,
            last_check: Instant::now(),
            failure_count: 0,
            recovery_attempts: 0,
            alerts: VecDeque::new(),
        };

        nodes.insert(node_id, monitoring_info);
        Ok(())
    }

    /// Unregister a node from monitoring
    pub fn unregister_node(&mut self, node_id: &str) -> Result<()> {
        let mut nodes = self.nodes.write().expect("Operation failed");
        nodes.remove(node_id);
        Ok(())
    }

    /// Update node metrics
    pub fn update_metrics(&mut self, node_id: &str, metrics: NodeMetrics) -> Result<()> {
        let mut nodes = self.nodes.write().expect("Operation failed");

        if let Some(monitoring_info) = nodes.get_mut(node_id) {
            monitoring_info.current_metrics = metrics;
            monitoring_info.last_check = Instant::now();
            monitoring_info.health_status =
                self.determine_health_status(&monitoring_info.current_metrics);
        } else {
            return Err(MetricsError::FaultToleranceError(format!(
                "Node {} not registered for monitoring",
                node_id
            )));
        }

        Ok(())
    }

    /// Determine health status based on metrics
    fn determine_health_status(&self, metrics: &NodeMetrics) -> NodeHealthStatus {
        // Check if node is responsive
        let heartbeat_age = metrics
            .last_heartbeat
            .elapsed()
            .unwrap_or_else(|_| Duration::from_secs(0));
        if heartbeat_age > Duration::from_secs(self.check_interval * 3) {
            return NodeHealthStatus::Failed;
        }

        // Check resource utilization
        if metrics.cpu_usage > 95.0 || metrics.memory_usage > 95.0 {
            return NodeHealthStatus::Degraded;
        }

        if metrics.cpu_usage > 85.0 || metrics.memory_usage > 85.0 {
            return NodeHealthStatus::Degraded;
        }

        NodeHealthStatus::Healthy
    }

    /// Get health summary for all nodes
    pub fn get_health_summary(&self) -> HealthSummary {
        let nodes = self.nodes.read().expect("Operation failed");

        let mut summary = HealthSummary {
            total_nodes: nodes.len(),
            healthy_nodes: 0,
            degraded_nodes: 0,
            failed_nodes: 0,
            unknown_nodes: 0,
            recovering_nodes: 0,
            maintenance_nodes: 0,
            last_updated: SystemTime::now(),
        };

        for monitoring_info in nodes.values() {
            match monitoring_info.health_status {
                NodeHealthStatus::Healthy => summary.healthy_nodes += 1,
                NodeHealthStatus::Degraded => summary.degraded_nodes += 1,
                NodeHealthStatus::Failed => summary.failed_nodes += 1,
                NodeHealthStatus::Unknown => summary.unknown_nodes += 1,
                NodeHealthStatus::Recovering => summary.recovering_nodes += 1,
                NodeHealthStatus::Maintenance => summary.maintenance_nodes += 1,
            }
        }

        summary
    }

    /// Get node health status
    pub fn get_node_health(&self, node_id: &str) -> Option<NodeHealthStatus> {
        let nodes = self.nodes.read().expect("Operation failed");
        nodes.get(node_id).map(|info| info.health_status.clone())
    }

    /// List all monitored nodes
    pub fn list_nodes(&self) -> Vec<String> {
        let nodes = self.nodes.read().expect("Operation failed");
        nodes.keys().cloned().collect()
    }
}

/// Node monitoring information
#[derive(Debug, Clone)]
pub struct NodeMonitoringInfo {
    /// Node ID
    pub node_id: String,
    /// Current metrics
    pub current_metrics: NodeMetrics,
    /// Current health status
    pub health_status: NodeHealthStatus,
    /// Last health check time
    pub last_check: Instant,
    /// Number of consecutive failures
    pub failure_count: usize,
    /// Number of recovery attempts
    pub recovery_attempts: usize,
    /// Recent alerts
    pub alerts: VecDeque<Alert>,
}

/// Node metrics for health monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    /// CPU usage percentage (0-100)
    pub cpu_usage: f64,
    /// Memory usage percentage (0-100)
    pub memory_usage: f64,
    /// Disk usage percentage (0-100)
    pub disk_usage: f64,
    /// Network bandwidth utilization (0-100)
    pub network_usage: f64,
    /// Number of active connections
    pub active_connections: usize,
    /// Response time in milliseconds
    pub response_time_ms: f64,
    /// Error rate (0-1)
    pub error_rate: f64,
    /// Last heartbeat timestamp
    pub last_heartbeat: SystemTime,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl NodeMetrics {
    /// Create default metrics for a healthy node
    pub fn healthy() -> Self {
        Self {
            cpu_usage: 10.0,
            memory_usage: 20.0,
            disk_usage: 30.0,
            network_usage: 5.0,
            active_connections: 10,
            response_time_ms: 50.0,
            error_rate: 0.001,
            last_heartbeat: SystemTime::now(),
            custom_metrics: HashMap::new(),
        }
    }

    /// Create metrics for a degraded node
    pub fn degraded() -> Self {
        Self {
            cpu_usage: 90.0, // Above the 85.0 threshold
            memory_usage: 80.0,
            disk_usage: 70.0,
            network_usage: 60.0,
            active_connections: 100,
            response_time_ms: 500.0,
            error_rate: 0.05,
            last_heartbeat: SystemTime::now(),
            custom_metrics: HashMap::new(),
        }
    }
}

/// Alert thresholds configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU usage warning threshold
    pub cpu_warning: f64,
    /// CPU usage critical threshold
    pub cpu_critical: f64,
    /// Memory usage warning threshold
    pub memory_warning: f64,
    /// Memory usage critical threshold
    pub memory_critical: f64,
    /// Response time warning threshold (ms)
    pub response_time_warning: f64,
    /// Response time critical threshold (ms)
    pub response_time_critical: f64,
    /// Error rate warning threshold
    pub error_rate_warning: f64,
    /// Error rate critical threshold
    pub error_rate_critical: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_warning: 80.0,
            cpu_critical: 95.0,
            memory_warning: 80.0,
            memory_critical: 95.0,
            response_time_warning: 1000.0,
            response_time_critical: 5000.0,
            error_rate_warning: 0.01,
            error_rate_critical: 0.05,
        }
    }
}

/// Health summary for the entire cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSummary {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of healthy nodes
    pub healthy_nodes: usize,
    /// Number of degraded nodes
    pub degraded_nodes: usize,
    /// Number of failed nodes
    pub failed_nodes: usize,
    /// Number of nodes with unknown status
    pub unknown_nodes: usize,
    /// Number of recovering nodes
    pub recovering_nodes: usize,
    /// Number of nodes in maintenance
    pub maintenance_nodes: usize,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

impl HealthSummary {
    /// Calculate health percentage
    pub fn health_percentage(&self) -> f64 {
        if self.total_nodes == 0 {
            return 100.0;
        }

        (self.healthy_nodes as f64 / self.total_nodes as f64) * 100.0
    }

    /// Check if cluster is healthy
    pub fn is_healthy(&self) -> bool {
        self.failed_nodes == 0 && self.degraded_nodes <= (self.total_nodes / 10)
    }
}

/// Alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Whether alert has been acknowledged
    pub acknowledged: bool,
}

/// Node replacement request
#[derive(Debug, Clone)]
pub struct NodeReplacementRequest {
    /// Failed node ID
    pub failed_node: String,
    /// Type of replacement
    pub replacement_type: NodeReplacementStrategy,
    /// When replacement was requested
    pub requested_at: SystemTime,
    /// Request priority
    pub priority: ReplacementPriority,
}

/// Node replacement priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReplacementPriority {
    /// Low priority replacement
    Low,
    /// Normal priority replacement
    Normal,
    /// High priority replacement
    High,
    /// Emergency replacement
    Emergency,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fault_recovery_manager_creation() {
        let config = FaultToleranceConfig::default();
        let manager = FaultRecoveryManager::new(config);
        assert_eq!(manager.get_recovery_history().len(), 0);
    }

    #[test]
    fn test_health_monitor_creation() {
        let monitor = HealthMonitor::new(30);
        assert_eq!(monitor.list_nodes().len(), 0);
    }

    #[test]
    fn test_node_registration() {
        let mut monitor = HealthMonitor::new(30);
        let metrics = NodeMetrics::healthy();

        monitor
            .register_node("node1".to_string(), metrics)
            .expect("Operation failed");
        assert_eq!(monitor.list_nodes().len(), 1);
        assert_eq!(
            monitor.get_node_health("node1"),
            Some(NodeHealthStatus::Healthy)
        );
    }

    #[test]
    fn test_health_status_determination() {
        let monitor = HealthMonitor::new(30);

        let healthy_metrics = NodeMetrics::healthy();
        assert_eq!(
            monitor.determine_health_status(&healthy_metrics),
            NodeHealthStatus::Healthy
        );

        let degraded_metrics = NodeMetrics::degraded();
        assert_eq!(
            monitor.determine_health_status(&degraded_metrics),
            NodeHealthStatus::Degraded
        );
    }

    #[test]
    fn test_recovery_action_creation() {
        let action = RecoveryAction {
            id: "test_action".to_string(),
            action_type: RecoveryActionType::NodeFailover,
            target_node: "node1".to_string(),
            severity: AlertSeverity::Critical,
            description: "Test recovery action".to_string(),
            strategy: RecoveryStrategy::Immediate,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
            status: RecoveryStatus::Pending,
            error: None,
        };

        assert_eq!(action.status, RecoveryStatus::Pending);
        assert_eq!(action.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_health_summary() {
        let summary = HealthSummary {
            total_nodes: 10,
            healthy_nodes: 8,
            degraded_nodes: 1,
            failed_nodes: 1,
            unknown_nodes: 0,
            recovering_nodes: 0,
            maintenance_nodes: 0,
            last_updated: SystemTime::now(),
        };

        assert_eq!(summary.health_percentage(), 80.0);
        assert!(!summary.is_healthy()); // Has failed nodes
    }

    #[test]
    fn test_alert_thresholds() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.cpu_warning, 80.0);
        assert_eq!(thresholds.cpu_critical, 95.0);
        assert!(thresholds.cpu_critical > thresholds.cpu_warning);
    }

    // ── New recovery action tests ─────────────────────────────────────────────

    /// Trigger node failover, verify node is marked Failed and replacement is queued.
    #[test]
    fn test_node_failover() {
        let config = FaultToleranceConfig {
            auto_recovery: true,
            replacement_strategy: NodeReplacementStrategy::Immediate,
            ..Default::default()
        };
        let mut manager = FaultRecoveryManager::new(config);

        // Register a node.
        let metrics = NodeMetrics::healthy();
        manager
            .register_node("failing_node".to_string(), metrics)
            .expect("register");

        // Trigger failover.
        let action = RecoveryAction {
            id: "failover_test".to_string(),
            action_type: RecoveryActionType::NodeFailover,
            target_node: "failing_node".to_string(),
            severity: AlertSeverity::Critical,
            description: "Test failover".to_string(),
            strategy: RecoveryStrategy::Immediate,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
            status: RecoveryStatus::Pending,
            error: None,
        };

        let action_id = manager.trigger_recovery(action).expect("trigger recovery");
        assert!(!action_id.is_empty());

        // The node should now be marked as Failed.
        let health = manager.health_monitor.get_node_health("failing_node");
        assert_eq!(health, Some(NodeHealthStatus::Failed));

        // A replacement request should have been queued.
        let requests = manager.process_replacement_requests().expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].failed_node, "failing_node");
    }

    /// Health monitor start/stop lifecycle test.
    #[test]
    fn test_health_monitor_start_stop() {
        let mut monitor = HealthMonitor::new(1);
        monitor.start().expect("start");

        // Brief pause to let the thread actually start.
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            monitor.running.load(std::sync::atomic::Ordering::SeqCst),
            "monitor should be running"
        );

        monitor.stop().expect("stop");
        assert!(
            !monitor.running.load(std::sync::atomic::Ordering::SeqCst),
            "monitor should be stopped"
        );
    }

    /// Service restart clears failure count and updates heartbeat.
    #[test]
    fn test_service_restart_recovery() {
        let config = FaultToleranceConfig {
            auto_recovery: true,
            ..Default::default()
        };
        let mut manager = FaultRecoveryManager::new(config);

        let metrics = NodeMetrics::degraded();
        manager
            .register_node("svc_node".to_string(), metrics)
            .expect("register");

        let action = RecoveryAction {
            id: "restart_test".to_string(),
            action_type: RecoveryActionType::ServiceRestart,
            target_node: "svc_node".to_string(),
            severity: AlertSeverity::Warning,
            description: "Test service restart".to_string(),
            strategy: RecoveryStrategy::Immediate,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
            status: RecoveryStatus::Pending,
            error: None,
        };

        manager.trigger_recovery(action).expect("trigger");

        // After restart the node should be Recovering.
        let health = manager.health_monitor.get_node_health("svc_node");
        assert_eq!(health, Some(NodeHealthStatus::Recovering));
    }

    /// Resource scaling halves CPU/memory and re-evaluates health.
    #[test]
    fn test_resource_scaling_recovery() {
        let config = FaultToleranceConfig {
            auto_recovery: true,
            ..Default::default()
        };
        let mut manager = FaultRecoveryManager::new(config);

        // Degrade the node.
        let mut metrics = NodeMetrics::degraded();
        metrics.cpu_usage = 98.0; // Above critical threshold
        manager
            .register_node("scaled_node".to_string(), metrics)
            .expect("register");

        let action = RecoveryAction {
            id: "scaling_test".to_string(),
            action_type: RecoveryActionType::ResourceScaling,
            target_node: "scaled_node".to_string(),
            severity: AlertSeverity::Critical,
            description: "Test resource scaling".to_string(),
            strategy: RecoveryStrategy::Immediate,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
            status: RecoveryStatus::Pending,
            error: None,
        };

        manager.trigger_recovery(action).expect("trigger");

        // After scaling, the CPU usage should have been reduced.
        let nodes = manager.health_monitor.nodes.read().expect("read");
        let info = nodes.get("scaled_node").expect("node info");
        assert!(
            info.current_metrics.cpu_usage < 98.0,
            "CPU should be lower after scaling"
        );
    }
}
