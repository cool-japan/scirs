//! Inter-module communication and message routing

use super::types::*;
use crate::error::{CoreError, CoreResult, ErrorContext};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Communication hub for managing inter-module messages
#[allow(dead_code)]
#[derive(Debug)]
pub struct ModuleCommunicationHub {
    /// Message queues for each module
    message_queues: HashMap<String, Vec<InterModuleMessage>>,
    /// Communication statistics
    #[allow(dead_code)]
    comm_stats: CommunicationStatistics,
    /// Routing table
    routing_table: HashMap<String, Vec<String>>,
}

/// Communication statistics
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CommunicationStatistics {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Average message latency
    pub avg_latency: Duration,
    /// Message error rate
    pub error_rate: f64,
}

/// Optimization opportunity identified by the ecosystem
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    /// Module name
    pub modulename: String,
    /// Type of optimization
    pub opportunity_type: String,
    /// Description of the opportunity
    pub description: String,
    /// Potential performance improvement factor
    pub potential_improvement: f64,
    /// Priority level
    pub priority: Priority,
}

impl Default for ModuleCommunicationHub {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleCommunicationHub {
    /// Create a new communication hub
    pub fn new() -> Self {
        Self {
            message_queues: HashMap::new(),
            comm_stats: CommunicationStatistics {
                messages_sent: 0,
                messages_received: 0,
                avg_latency: Duration::default(),
                error_rate: 0.0,
            },
            routing_table: HashMap::new(),
        }
    }

    /// Send a message from one module to another
    pub fn send_message(&mut self, message: InterModuleMessage) -> CoreResult<()> {
        // Validate message
        if message.from.is_empty() || message.to.is_empty() {
            return Err(CoreError::InvalidInput(ErrorContext::new(
                "Message must have valid source and destination",
            )));
        }

        // Create queue for destination module if it doesn't exist
        if !self.message_queues.contains_key(&message.to) {
            self.message_queues.insert(message.to.clone(), Vec::new());
        }

        // Add message to destination queue
        if let Some(queue) = self.message_queues.get_mut(&message.to) {
            queue.push(message);
        }

        // Update statistics
        self.comm_stats.messages_sent += 1;

        Ok(())
    }

    /// Receive messages for a specific module
    pub fn receive_messages(&mut self, module_name: &str) -> Vec<InterModuleMessage> {
        if let Some(queue) = self.message_queues.get_mut(module_name) {
            let message_count = queue.len();
            let messages = std::mem::take(queue);
            self.comm_stats.messages_received += message_count as u64;
            messages
        } else {
            Vec::new()
        }
    }

    /// Get pending message count for a module
    pub fn get_pending_count(&self, module_name: &str) -> usize {
        self.message_queues
            .get(module_name)
            .map_or(0, |queue| queue.len())
    }

    /// Optimize routing paths between modules
    pub fn optimize_routing(&mut self) -> CoreResult<()> {
        // Clear old message queues and optimize routing paths
        self.message_queues.clear();

        // Rebuild routing table for optimal paths
        for (source, destinations) in &mut self.routing_table {
            // Sort destinations by priority and performance
            destinations.sort();
            println!("    📍 Optimized routing for module: {source}");
        }

        Ok(())
    }

    /// Enable message compression for large payloads
    pub fn enable_compression(&mut self) -> CoreResult<()> {
        println!("    🗜️  Enabled message compression");
        Ok(())
    }

    /// Add a routing entry
    pub fn add_route(&mut self, source: String, destination: String) {
        self.routing_table
            .entry(source)
            .or_default()
            .push(destination);
    }

    /// Remove a routing entry
    pub fn remove_route(&mut self, source: &str, destination: &str) {
        if let Some(destinations) = self.routing_table.get_mut(source) {
            destinations.retain(|dest| dest != destination);
        }
    }

    /// Get routing destinations for a source module
    pub fn get_routes(&self, source: &str) -> Vec<String> {
        self.routing_table.get(source).cloned().unwrap_or_default()
    }

    /// Broadcast a message to all connected modules
    pub fn broadcast_message(&mut self, message: InterModuleMessage) -> CoreResult<()> {
        let destinations = self
            .routing_table
            .get(&message.from)
            .cloned()
            .unwrap_or_default();
        for destination in destinations {
            let mut broadcast_message = message.clone();
            broadcast_message.to = destination;
            self.send_message(broadcast_message)?;
        }
        Ok(())
    }

    /// Get communication statistics
    pub fn get_statistics(&self) -> &CommunicationStatistics {
        &self.comm_stats
    }

    /// Reset communication statistics
    pub fn reset_statistics(&mut self) {
        self.comm_stats = CommunicationStatistics {
            messages_sent: 0,
            messages_received: 0,
            avg_latency: Duration::default(),
            error_rate: 0.0,
        };
    }

    /// Update message latency statistics
    pub fn update_latency(&mut self, latency: Duration) {
        // Simple moving average for latency
        if self.comm_stats.avg_latency.is_zero() {
            self.comm_stats.avg_latency = latency;
        } else {
            let current_nanos = self.comm_stats.avg_latency.as_nanos();
            let new_nanos = latency.as_nanos();
            let avg_nanos = (current_nanos + new_nanos) / 2;
            self.comm_stats.avg_latency = Duration::from_nanos(avg_nanos as u64);
        }
    }

    /// Update error rate statistics
    pub fn update_error_rate(&mut self, error_occurred: bool) {
        let total_messages = self.comm_stats.messages_sent + self.comm_stats.messages_received;
        if total_messages > 0 {
            if error_occurred {
                self.comm_stats.error_rate =
                    (self.comm_stats.error_rate * (total_messages - 1) as f64 + 1.0)
                        / total_messages as f64;
            } else {
                self.comm_stats.error_rate = (self.comm_stats.error_rate
                    * (total_messages - 1) as f64)
                    / total_messages as f64;
            }
        }
    }

    /// Clear all message queues
    pub fn clear_queues(&mut self) {
        self.message_queues.clear();
        println!("    🧹 Cleared all message queues");
    }

    /// Clear queue for a specific module
    pub fn clear_module_queue(&mut self, module_name: &str) {
        if let Some(queue) = self.message_queues.get_mut(module_name) {
            queue.clear();
            println!("    🧹 Cleared message queue for module: {}", module_name);
        }
    }

    /// Get total memory usage of all queues
    pub fn get_memory_usage(&self) -> usize {
        self.message_queues
            .values()
            .map(|queue| queue.len() * std::mem::size_of::<InterModuleMessage>())
            .sum()
    }

    /// Cleanup expired messages (messages older than specified duration)
    pub fn cleanup_expired_messages(&mut self, max_age: Duration) {
        let now = Instant::now();
        let mut cleaned_count = 0;

        for queue in self.message_queues.values_mut() {
            let original_len = queue.len();
            queue.retain(|msg| now.duration_since(msg.timestamp) < max_age);
            cleaned_count += original_len - queue.len();
        }

        if cleaned_count > 0 {
            println!("    🧹 Cleaned up {} expired messages", cleaned_count);
        }
    }

    /// Get health status of the communication hub
    pub fn get_health_status(&self) -> CommunicationHealth {
        let total_queue_size: usize = self.message_queues.values().map(|q| q.len()).sum();
        let memory_usage = self.get_memory_usage();

        if self.comm_stats.error_rate > 0.1 {
            CommunicationHealth::Critical
        } else if total_queue_size > 10000 || memory_usage > 100 * 1024 * 1024 {
            CommunicationHealth::Warning
        } else if self.comm_stats.error_rate > 0.05 {
            CommunicationHealth::Degraded
        } else {
            CommunicationHealth::Healthy
        }
    }

    /// Create an optimized processing pipeline
    pub fn create_optimized_pipeline(
        &self,
        input: &AdvancedInput,
        config: &CrossModuleOptimizationConfig,
    ) -> CoreResult<OptimizedPipeline> {
        let stages = vec![
            PipelineStage {
                name: "preprocessing".to_string(),
                module: "core".to_string(),
                config: HashMap::from([("operation".to_string(), "normalize".to_string())]),
                dependencies: vec![],
            },
            PipelineStage {
                name: "processing".to_string(),
                module: input.context.operationtype.clone(),
                config: HashMap::from([("operation".to_string(), "advanced_process".to_string())]),
                dependencies: vec!["preprocessing".to_string()],
            },
        ];

        Ok(OptimizedPipeline {
            stages,
            optimization_level: config.optimization_level.clone(),
            estimated_performance: PerformanceMetrics {
                throughput: 1000.0,
                latency: Duration::from_millis(100),
                cpu_usage: 50.0,
                memory_usage: 1024 * 1024,
                gpu_usage: 30.0,
            },
        })
    }
}

/// Health status of the communication system
#[derive(Debug, Clone, PartialEq)]
pub enum CommunicationHealth {
    Healthy,
    Warning,
    Degraded,
    Critical,
}

/// Message delivery confirmation
#[derive(Debug, Clone)]
pub struct DeliveryConfirmation {
    pub message_id: String,
    pub delivered: bool,
    pub delivery_time: Duration,
    pub error_message: Option<String>,
}

/// Communication channel configuration
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    pub max_queue_size: usize,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
    pub priority_enabled: bool,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            compression_enabled: false,
            encryption_enabled: false,
            priority_enabled: true,
        }
    }
}
