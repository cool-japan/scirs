//! Resource management for cross-platform testing
//!
//! This module manages computational resources, cost tracking, and
//! resource allocation for distributed testing environments.

use crate::error::Result;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::config::*;
use super::types::*;

/// Platform resource manager
#[derive(Debug)]
pub struct PlatformResourceManager {
    limits: ResourceLimits,
    usage_tracker: ResourceUsageTracker,
    cost_tracker: CostTracker,
    start_time: SystemTime,
}

/// Resource usage tracker
#[derive(Debug)]
pub struct ResourceUsageTracker {
    current_usage: ResourceUsage,
    peak_usage: ResourceUsage,
    history: Vec<ResourceUsage>,
}

/// Cost tracker for cloud resources
#[derive(Debug)]
pub struct CostTracker {
    total_cost: f64,
    cost_by_provider: HashMap<String, f64>,
    cost_history: Vec<CostEntry>,
}

/// Cost entry for tracking
#[derive(Debug, Clone)]
pub struct CostEntry {
    pub timestamp: SystemTime,
    pub provider: String,
    pub resource_type: String,
    pub cost: f64,
    pub description: String,
}

impl PlatformResourceManager {
    /// Create new resource manager
    pub fn new(limits: ResourceLimits) -> Result<Self> {
        Ok(Self {
            limits,
            usage_tracker: ResourceUsageTracker::new(),
            cost_tracker: CostTracker::new(),
            start_time: SystemTime::now(),
        })
    }

    /// Check if resources are available
    pub fn check_resource_availability(&self, required: &HashMap<ResourceType, f64>) -> bool {
        for (resource_type, amount) in required {
            if !self.is_resource_available(resource_type, *amount) {
                return false;
            }
        }
        true
    }

    /// Check if specific resource is available
    fn is_resource_available(&self, resource_type: &ResourceType, amount: f64) -> bool {
        match resource_type {
            ResourceType::CPU => self.usage_tracker.current_usage.cpu_usage + amount < 100.0,
            ResourceType::Memory => {
                let current_mb = self.usage_tracker.current_usage.memory_usage;
                current_mb + amount as usize < self.limits.max_memory_usage
            }
            ResourceType::Storage => amount < self.limits.max_disk_usage as f64,
            ResourceType::Network => true, // Simplified
            ResourceType::GPU => true, // Simplified
        }
    }

    /// Allocate resources
    pub fn allocate_resources(&mut self, required: &HashMap<ResourceType, f64>) -> Result<String> {
        if !self.check_resource_availability(required) {
            return Err(crate::error::OptimError::InsufficientResources(
                "Insufficient resources available".to_string()
            ));
        }

        // Update usage
        for (resource_type, amount) in required {
            match resource_type {
                ResourceType::CPU => {
                    self.usage_tracker.current_usage.cpu_usage += amount;
                }
                ResourceType::Memory => {
                    self.usage_tracker.current_usage.memory_usage += *amount as usize;
                }
                _ => {} // Simplified for other types
            }
        }

        // Generate allocation ID
        let allocation_id = format!("alloc_{}", SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs());

        Ok(allocation_id)
    }

    /// Track cost
    pub fn track_cost(&mut self, provider: &str, resource_type: &str, cost: f64, description: &str) {
        self.cost_tracker.add_cost(provider, resource_type, cost, description);
    }

    /// Get total resource usage
    pub fn get_total_usage(&self) -> ResourceUsage {
        self.usage_tracker.current_usage.clone()
    }

    /// Get total cost
    pub fn get_total_cost(&self) -> f64 {
        self.cost_tracker.total_cost
    }

    /// Get start time
    pub fn get_start_time(&self) -> SystemTime {
        self.start_time
    }

    /// Get total execution time
    pub fn get_total_execution_time(&self) -> Duration {
        SystemTime::now().duration_since(self.start_time).unwrap_or_default()
    }

    /// Update resource usage
    pub fn update_usage(&mut self, usage: ResourceUsage) {
        self.usage_tracker.update_usage(usage);
    }
}

impl ResourceUsageTracker {
    fn new() -> Self {
        Self {
            current_usage: ResourceUsage::default(),
            peak_usage: ResourceUsage::default(),
            history: Vec::new(),
        }
    }

    fn update_usage(&mut self, usage: ResourceUsage) {
        // Update peak usage
        if usage.cpu_usage > self.peak_usage.cpu_usage {
            self.peak_usage.cpu_usage = usage.cpu_usage;
        }
        if usage.memory_usage > self.peak_usage.memory_usage {
            self.peak_usage.memory_usage = usage.memory_usage;
        }

        // Add to history
        self.history.push(usage.clone());

        // Update current usage
        self.current_usage = usage;
    }
}

impl CostTracker {
    fn new() -> Self {
        Self {
            total_cost: 0.0,
            cost_by_provider: HashMap::new(),
            cost_history: Vec::new(),
        }
    }

    fn add_cost(&mut self, provider: &str, resource_type: &str, cost: f64, description: &str) {
        self.total_cost += cost;
        *self.cost_by_provider.entry(provider.to_string()).or_insert(0.0) += cost;

        let entry = CostEntry {
            timestamp: SystemTime::now(),
            provider: provider.to_string(),
            resource_type: resource_type.to_string(),
            cost,
            description: description.to_string(),
        };

        self.cost_history.push(entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_manager_creation() {
        let limits = ResourceLimits::default();
        let manager = PlatformResourceManager::new(limits);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_resource_allocation() {
        let limits = ResourceLimits::default();
        let mut manager = PlatformResourceManager::new(limits).unwrap();

        let mut required = HashMap::new();
        required.insert(ResourceType::CPU, 2.0);
        required.insert(ResourceType::Memory, 1024.0);

        let result = manager.allocate_resources(&required);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cost_tracking() {
        let limits = ResourceLimits::default();
        let mut manager = PlatformResourceManager::new(limits).unwrap();

        manager.track_cost("aws", "ec2", 0.50, "Test instance");
        assert_eq!(manager.get_total_cost(), 0.50);

        manager.track_cost("azure", "vm", 0.30, "Test VM");
        assert_eq!(manager.get_total_cost(), 0.80);
    }
}