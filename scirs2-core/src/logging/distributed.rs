//! # Distributed Logging
//!
//! This module provides distributed logging capabilities for multi-node computations,
//! log aggregation, and centralized logging infrastructure.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Error types for distributed logging
#[derive(Error, Debug)]
pub enum DistributedLogError {
    /// Network communication error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Node not found
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

/// Distributed log entry
#[derive(Debug, Clone)]
pub struct DistributedLogEntry {
    /// Unique entry ID
    pub id: String,
    /// Node ID that generated the log
    pub node_id: String,
    /// Log level
    pub level: crate::logging::LogLevel,
    /// Log message
    pub message: String,
    /// Additional context fields
    pub context: HashMap<String, String>,
    /// Timestamp when log was generated
    pub timestamp: SystemTime,
    /// Optional correlation ID for tracing
    pub correlation_id: Option<String>,
    /// Service or component name
    pub service: String,
}

impl DistributedLogEntry {
    /// Create a new distributed log entry
    pub fn new(
        node_id: String,
        level: crate::logging::LogLevel,
        message: String,
        service: String,
    ) -> Self {
        Self {
            id: Self::generate_id(),
            node_id,
            level,
            message,
            context: HashMap::new(),
            timestamp: SystemTime::now(),
            correlation_id: None,
            service,
        }
    }

    /// Add context field
    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }

    /// Set correlation ID
    pub fn with_correlation_id(mut self, correlation_id: String) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    /// Generate unique ID
    fn generate_id() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis();
        let counter = COUNTER.fetch_add(1, Ordering::SeqCst);

        format!("{:x}_{:x}", timestamp, counter)
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> String {
        // Simplified JSON serialization - real implementation would use serde
        format!(
            r#"{{"id":"{}","node_id":"{}","level":"{}","message":"{}","service":"{}","timestamp":"{}"}}"#,
            self.id,
            self.node_id,
            format!("{:?}", self.level),
            self.message.replace('"', "\\\""),
            self.service,
            self.timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs()
        )
    }
}

/// Node information in the distributed logging cluster
#[derive(Debug, Clone)]
pub struct LogNode {
    /// Node ID
    pub id: String,
    /// Node address (IP:port)
    pub address: String,
    /// Node role
    pub role: NodeRole,
    /// Last heartbeat timestamp
    pub last_heartbeat: SystemTime,
    /// Node status
    pub status: NodeStatus,
}

/// Node roles in distributed logging
#[derive(Debug, Clone, PartialEq)]
pub enum NodeRole {
    /// Log aggregator/collector
    Aggregator,
    /// Log producer
    Producer,
    /// Log storage
    Storage,
    /// Log forwarder
    Forwarder,
}

/// Node status
#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    /// Node is healthy and active
    Healthy,
    /// Node is experiencing issues
    Degraded,
    /// Node is not responding
    Unreachable,
}

/// Distributed logging configuration
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Current node ID
    pub node_id: String,
    /// Current node role
    pub node_role: NodeRole,
    /// Aggregator nodes
    pub aggregators: Vec<String>,
    /// Buffer size for log entries
    pub buffer_size: usize,
    /// Flush interval for buffered logs
    pub flush_interval: Duration,
    /// Retry attempts for failed sends
    pub retry_attempts: usize,
    /// Enable compression for network transport
    pub enable_compression: bool,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            node_id: format!("node-{}", uuid::Uuid::new_v4()),
            node_role: NodeRole::Producer,
            aggregators: Vec::new(),
            buffer_size: 1000,
            flush_interval: Duration::from_secs(5),
            retry_attempts: 3,
            enable_compression: false,
            heartbeat_interval: Duration::from_secs(30),
        }
    }
}

/// Log aggregation strategy
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    /// Simple forwarding to aggregators
    Forward,
    /// Load balance across aggregators
    LoadBalance,
    /// Replicate to all aggregators
    Replicate,
    /// Custom aggregation logic
    Custom(String),
}

/// Distributed logger
pub struct DistributedLogger {
    config: DistributedConfig,
    buffer: Arc<Mutex<Vec<DistributedLogEntry>>>,
    nodes: Arc<Mutex<HashMap<String, LogNode>>>,
    aggregation_strategy: AggregationStrategy,
    running: Arc<Mutex<bool>>,
}

impl DistributedLogger {
    /// Create a new distributed logger
    pub fn new(config: DistributedConfig) -> Self {
        Self {
            config,
            buffer: Arc::new(Mutex::new(Vec::new())),
            nodes: Arc::new(Mutex::new(HashMap::new())),
            aggregation_strategy: AggregationStrategy::Forward,
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the distributed logger
    pub fn start(&self) -> Result<(), DistributedLogError> {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Ok(());
        }
        *running = true;

        // Initialize aggregator nodes
        for aggregator_addr in &self.config.aggregators {
            let node = LogNode {
                id: format!("aggregator-{}", aggregator_addr),
                address: aggregator_addr.clone(),
                role: NodeRole::Aggregator,
                last_heartbeat: SystemTime::now(),
                status: NodeStatus::Healthy,
            };
            self.nodes.lock().unwrap().insert(node.id.clone(), node);
        }

        // Start background tasks
        self.start_flush_task();
        self.start_heartbeat_task();

        Ok(())
    }

    /// Stop the distributed logger
    pub fn stop(&self) {
        *self.running.lock().unwrap() = false;
    }

    /// Log a distributed entry
    pub fn log(&self, level: crate::logging::LogLevel, message: &str, service: &str) {
        let entry = DistributedLogEntry::new(
            self.config.node_id.clone(),
            level,
            message.to_string(),
            service.to_string(),
        );

        self.buffer.lock().unwrap().push(entry);

        // Check if buffer needs immediate flush
        if self.buffer.lock().unwrap().len() >= self.config.buffer_size {
            self.flush_buffer();
        }
    }

    /// Log with correlation ID for distributed tracing
    pub fn log_with_correlation(
        &self,
        level: crate::logging::LogLevel,
        message: &str,
        service: &str,
        correlation_id: &str,
    ) {
        let entry = DistributedLogEntry::new(
            self.config.node_id.clone(),
            level,
            message.to_string(),
            service.to_string(),
        )
        .with_correlation_id(correlation_id.to_string());

        self.buffer.lock().unwrap().push(entry);
    }

    /// Flush buffered log entries
    pub fn flush_buffer(&self) {
        let mut buffer = self.buffer.lock().unwrap();
        if buffer.is_empty() {
            return;
        }

        let entries: Vec<_> = buffer.drain(..).collect();
        drop(buffer);

        match self.aggregation_strategy {
            AggregationStrategy::Forward => {
                self.forward_entries(&entries);
            }
            AggregationStrategy::LoadBalance => {
                self.load_balance_entries(&entries);
            }
            AggregationStrategy::Replicate => {
                self.replicate_entries(&entries);
            }
            AggregationStrategy::Custom(_) => {
                // Custom logic would be implemented here
                self.forward_entries(&entries);
            }
        }
    }

    /// Forward entries to first available aggregator
    fn forward_entries(&self, entries: &[DistributedLogEntry]) {
        let nodes = self.nodes.lock().unwrap();
        let aggregator = nodes
            .values()
            .find(|node| node.role == NodeRole::Aggregator && node.status == NodeStatus::Healthy);

        if let Some(node) = aggregator {
            self.send_entries_to_node(entries, node);
        }
    }

    /// Load balance entries across available aggregators
    fn load_balance_entries(&self, entries: &[DistributedLogEntry]) {
        let nodes = self.nodes.lock().unwrap();
        let aggregators: Vec<_> = nodes
            .values()
            .filter(|node| node.role == NodeRole::Aggregator && node.status == NodeStatus::Healthy)
            .collect();

        if aggregators.is_empty() {
            return;
        }

        for (i, entry) in entries.iter().enumerate() {
            let node = &aggregators[i % aggregators.len()];
            self.send_entries_to_node(&[entry.clone()], node);
        }
    }

    /// Replicate entries to all available aggregators
    fn replicate_entries(&self, entries: &[DistributedLogEntry]) {
        let nodes = self.nodes.lock().unwrap();
        for node in nodes.values() {
            if node.role == NodeRole::Aggregator && node.status == NodeStatus::Healthy {
                self.send_entries_to_node(entries, node);
            }
        }
    }

    /// Send entries to a specific node
    fn send_entries_to_node(&self, entries: &[DistributedLogEntry], node: &LogNode) {
        // In a real implementation, this would use HTTP, gRPC, or message queues
        // For now, we'll simulate the send operation

        for entry in entries {
            println!("Sending log to {}: {}", node.address, entry.to_json());
        }
    }

    /// Start background flush task
    fn start_flush_task(&self) {
        let buffer = Arc::clone(&self.buffer);
        let running = Arc::clone(&self.running);
        let flush_interval = self.config.flush_interval;

        std::thread::spawn(move || {
            while *running.lock().unwrap() {
                std::thread::sleep(flush_interval);

                // This is a simplified version - real implementation would call flush_buffer
                let buffer_size = buffer.lock().unwrap().len();
                if buffer_size > 0 {
                    println!("Background flush: {} entries buffered", buffer_size);
                }
            }
        });
    }

    /// Start heartbeat task
    fn start_heartbeat_task(&self) {
        let nodes = Arc::clone(&self.nodes);
        let running = Arc::clone(&self.running);
        let heartbeat_interval = self.config.heartbeat_interval;

        std::thread::spawn(move || {
            while *running.lock().unwrap() {
                std::thread::sleep(heartbeat_interval);

                // Send heartbeats and check node health
                let mut nodes_guard = nodes.lock().unwrap();
                for node in nodes_guard.values_mut() {
                    // In real implementation, would send actual heartbeat
                    if node.last_heartbeat.elapsed().unwrap_or(Duration::ZERO)
                        > heartbeat_interval * 2
                    {
                        node.status = NodeStatus::Unreachable;
                    }
                }
            }
        });
    }

    /// Get node statistics
    pub fn get_node_stats(&self) -> NodeStats {
        let buffer_size = self.buffer.lock().unwrap().len();
        let nodes = self.nodes.lock().unwrap();

        let healthy_nodes = nodes
            .values()
            .filter(|n| n.status == NodeStatus::Healthy)
            .count();

        let total_nodes = nodes.len();

        NodeStats {
            buffer_size,
            healthy_nodes,
            total_nodes,
            node_id: self.config.node_id.clone(),
            uptime: SystemTime::now(),
        }
    }

    /// Set aggregation strategy
    pub fn set_aggregation_strategy(&mut self, strategy: AggregationStrategy) {
        self.aggregation_strategy = strategy;
    }
}

impl Drop for DistributedLogger {
    fn drop(&mut self) {
        self.stop();
        self.flush_buffer();
    }
}

/// Node statistics
#[derive(Debug)]
pub struct NodeStats {
    /// Current buffer size
    pub buffer_size: usize,
    /// Number of healthy nodes
    pub healthy_nodes: usize,
    /// Total number of nodes
    pub total_nodes: usize,
    /// Current node ID
    pub node_id: String,
    /// Node uptime
    pub uptime: SystemTime,
}

/// Log aggregator for collecting logs from multiple nodes
pub struct LogAggregator {
    config: DistributedConfig,
    collected_logs: Arc<Mutex<Vec<DistributedLogEntry>>>,
    running: Arc<Mutex<bool>>,
}

impl LogAggregator {
    /// Create a new log aggregator
    pub fn new(config: DistributedConfig) -> Self {
        Self {
            config,
            collected_logs: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the aggregator
    pub fn start(&self) -> Result<(), DistributedLogError> {
        *self.running.lock().unwrap() = true;

        // In a real implementation, would start HTTP/gRPC server
        println!("Log aggregator started on node: {}", self.config.node_id);

        Ok(())
    }

    /// Stop the aggregator
    pub fn stop(&self) {
        *self.running.lock().unwrap() = false;
    }

    /// Receive log entries from remote nodes
    pub fn receive_logs(&self, entries: Vec<DistributedLogEntry>) {
        let mut logs = self.collected_logs.lock().unwrap();
        logs.extend(entries);

        // Keep only recent logs (simple cleanup)
        if logs.len() > 10000 {
            logs.drain(0..1000);
        }
    }

    /// Query logs by criteria
    pub fn query_logs(
        &self,
        service: Option<&str>,
        level: Option<crate::logging::LogLevel>,
    ) -> Vec<DistributedLogEntry> {
        let logs = self.collected_logs.lock().unwrap();

        logs.iter()
            .filter(|entry| {
                if let Some(svc) = service {
                    if entry.service != svc {
                        return false;
                    }
                }
                if let Some(lvl) = level {
                    if std::mem::discriminant(&entry.level) != std::mem::discriminant(&lvl) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    /// Get aggregated statistics
    pub fn get_stats(&self) -> AggregatorStats {
        let logs = self.collected_logs.lock().unwrap();

        let mut service_counts = HashMap::new();
        let mut level_counts = HashMap::new();

        for entry in logs.iter() {
            *service_counts.entry(entry.service.clone()).or_insert(0) += 1;
            *level_counts
                .entry(format!("{:?}", entry.level))
                .or_insert(0) += 1;
        }

        AggregatorStats {
            total_logs: logs.len(),
            service_counts,
            level_counts,
            aggregator_id: self.config.node_id.clone(),
        }
    }
}

/// Aggregator statistics
#[derive(Debug)]
pub struct AggregatorStats {
    /// Total number of logs collected
    pub total_logs: usize,
    /// Log counts per service
    pub service_counts: HashMap<String, usize>,
    /// Log counts per level
    pub level_counts: HashMap<String, usize>,
    /// Aggregator node ID
    pub aggregator_id: String,
}

/// Convenience functions for distributed logging
pub mod utils {
    use super::*;

    /// Create a simple distributed logging setup
    pub fn create_simple_setup(node_role: NodeRole) -> DistributedLogger {
        let config = DistributedConfig {
            node_role,
            ..Default::default()
        };

        DistributedLogger::new(config)
    }

    /// Create a distributed logging cluster
    pub fn create_cluster(aggregator_addresses: Vec<String>) -> Vec<DistributedLogger> {
        let mut loggers = Vec::new();

        // Create aggregators
        for addr in &aggregator_addresses {
            let config = DistributedConfig {
                node_id: format!("aggregator-{}", addr),
                node_role: NodeRole::Aggregator,
                ..Default::default()
            };
            loggers.push(DistributedLogger::new(config));
        }

        // Create producers pointing to aggregators
        let producer_config = DistributedConfig {
            aggregators: aggregator_addresses,
            node_role: NodeRole::Producer,
            ..Default::default()
        };
        loggers.push(DistributedLogger::new(producer_config));

        loggers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_log_entry() {
        let entry = DistributedLogEntry::new(
            "test-node".to_string(),
            crate::logging::LogLevel::Info,
            "Test message".to_string(),
            "test-service".to_string(),
        );

        assert_eq!(entry.node_id, "test-node");
        assert_eq!(entry.message, "Test message");
        assert_eq!(entry.service, "test-service");
        assert!(!entry.id.is_empty());
    }

    #[test]
    fn test_distributed_logger_creation() {
        let config = DistributedConfig::default();
        let logger = DistributedLogger::new(config);

        assert_eq!(*logger.running.lock().unwrap(), false);
    }

    #[test]
    fn test_log_aggregator() {
        let config = DistributedConfig {
            node_role: NodeRole::Aggregator,
            ..Default::default()
        };
        let aggregator = LogAggregator::new(config);

        let entry = DistributedLogEntry::new(
            "producer-1".to_string(),
            crate::logging::LogLevel::Error,
            "Error occurred".to_string(),
            "api-service".to_string(),
        );

        aggregator.receive_logs(vec![entry]);

        let logs = aggregator.query_logs(Some("api-service"), None);
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].message, "Error occurred");
    }

    #[test]
    fn test_node_stats() {
        let config = DistributedConfig::default();
        let logger = DistributedLogger::new(config);

        let stats = logger.get_node_stats();
        assert_eq!(stats.buffer_size, 0);
        assert_eq!(stats.total_nodes, 0);
    }
}
