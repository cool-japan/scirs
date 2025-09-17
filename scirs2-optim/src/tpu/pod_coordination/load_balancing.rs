//! Load Balancing for TPU Pod Coordination
//!
//! This module provides comprehensive load balancing functionality for TPU pod coordination,
//! including device load monitoring, rebalancing policies, and migration management.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use super::DeviceId;
use crate::error::{OptimError, Result};

/// Device load information
#[derive(Debug, Clone)]
pub struct DeviceLoad {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,

    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,

    /// Communication utilization (0.0 to 1.0)
    pub communication_utilization: f64,

    /// Queue length
    pub queue_length: usize,

    /// Active tasks
    pub active_tasks: usize,

    /// Temperature
    pub temperature: f64,

    /// Power consumption
    pub power_consumption: f64,
}

/// Load snapshot for history
#[derive(Debug, Clone)]
pub struct LoadSnapshot {
    /// Timestamp
    pub timestamp: Instant,

    /// Device loads
    pub device_loads: HashMap<DeviceId, DeviceLoad>,

    /// Overall load balance
    pub load_balance_metric: f64,

    /// Hotspots
    pub hotspots: Vec<DeviceId>,
}

/// Rebalancing policies
#[derive(Debug, Clone)]
pub struct RebalancingPolicy {
    /// Policy trigger
    pub trigger: RebalancingTrigger,

    /// Policy action
    pub action: RebalancingAction,

    /// Policy priority
    pub priority: usize,

    /// Cooldown period
    pub cooldown: Duration,
}

/// Rebalancing triggers
#[derive(Debug, Clone)]
pub enum RebalancingTrigger {
    LoadImbalance(f64),
    HighUtilization(f64),
    LowUtilization(f64),
    QueueBacklog(usize),
    TemperatureThreshold(f64),
    Custom(String),
}

/// Rebalancing actions
#[derive(Debug, Clone)]
pub enum RebalancingAction {
    MigrateTasks,
    RedistributeLoad,
    ScaleUp,
    ScaleDown,
    Throttle,
    Custom(String),
}

/// Load balancing strategies for pods
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    Static,
    Dynamic,
    PredictiveDynamic,
    WorkStealing,
    LoadAware,
    LatencyAware,
    BandwidthAware,
    Adaptive,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    LeastLoaded,
    WeightedRoundRobin,
    CapacityBased,
}

/// Migration manager type alias
type MigrationManager = HashMap<DeviceId, f64>;

/// Load balance statistics type alias
pub type LoadBalanceStatistics = HashMap<String, f64>;

/// Device availability information
#[derive(Debug, Clone)]
pub struct DeviceAvailability {
    pub available_memory: usize,
    pub compute_capacity: f64,
    pub communication_bandwidth: f64,
    pub current_load: f64,
    pub reserved_until: Option<Instant>,
}

/// Pod load balancer
#[derive(Debug)]
pub struct PodLoadBalancer {
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,

    /// Device loads
    device_loads: HashMap<DeviceId, DeviceLoad>,

    /// Load history
    load_history: VecDeque<LoadSnapshot>,

    /// Rebalancing policies
    rebalancing_policies: Vec<RebalancingPolicy>,

    /// Migration manager
    migration_manager: MigrationManager,
}

impl PodLoadBalancer {
    /// Create a new pod load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Result<Self> {
        let rebalancing_policies = vec![
            RebalancingPolicy {
                trigger: RebalancingTrigger::LoadImbalance(0.3),
                action: RebalancingAction::RedistributeLoad,
                priority: 1,
                cooldown: Duration::from_secs(30),
            },
            RebalancingPolicy {
                trigger: RebalancingTrigger::HighUtilization(0.9),
                action: RebalancingAction::MigrateTasks,
                priority: 2,
                cooldown: Duration::from_secs(60),
            },
            RebalancingPolicy {
                trigger: RebalancingTrigger::TemperatureThreshold(80.0),
                action: RebalancingAction::Throttle,
                priority: 3,
                cooldown: Duration::from_secs(15),
            },
        ];

        Ok(Self {
            strategy,
            device_loads: HashMap::new(),
            load_history: VecDeque::with_capacity(1000),
            rebalancing_policies,
            migration_manager: HashMap::new(),
        })
    }

    /// Update device load information
    pub fn update_device_load(&mut self, device_id: DeviceId, load: DeviceLoad) {
        self.device_loads.insert(device_id, load);
        self.record_load_snapshot();
    }

    /// Get current device load
    pub fn get_device_load(&self, device_id: DeviceId) -> Option<&DeviceLoad> {
        self.device_loads.get(&device_id)
    }

    /// Calculate load balance metric
    pub fn calculate_load_balance_metric(&self) -> f64 {
        if self.device_loads.is_empty() {
            return 1.0;
        }

        let utilizations: Vec<f64> = self
            .device_loads
            .values()
            .map(|load| (load.cpu_utilization + load.memory_utilization) / 2.0)
            .collect();

        let mean: f64 = utilizations.iter().sum::<f64>() / utilizations.len() as f64;
        let variance: f64 = utilizations
            .iter()
            .map(|u| (u - mean).powi(2))
            .sum::<f64>()
            / utilizations.len() as f64;

        // Return inverse of variance (higher is better balance)
        if variance == 0.0 {
            1.0
        } else {
            1.0 / (1.0 + variance)
        }
    }

    /// Identify hotspot devices
    pub fn identify_hotspots(&self, threshold: f64) -> Vec<DeviceId> {
        self.device_loads
            .iter()
            .filter(|(_, load)| {
                (load.cpu_utilization + load.memory_utilization) / 2.0 > threshold
            })
            .map(|(device_id, _)| *device_id)
            .collect()
    }

    /// Check if rebalancing is needed
    pub fn check_rebalancing_needed(&self) -> Vec<&RebalancingPolicy> {
        let mut triggered_policies = Vec::new();

        for policy in &self.rebalancing_policies {
            if self.evaluate_trigger(&policy.trigger) {
                triggered_policies.push(policy);
            }
        }

        // Sort by priority
        triggered_policies.sort_by_key(|policy| policy.priority);
        triggered_policies
    }

    /// Execute rebalancing action
    pub async fn execute_rebalancing(&mut self, policy: &RebalancingPolicy) -> Result<()> {
        match &policy.action {
            RebalancingAction::MigrateTasks => {
                self.migrate_tasks().await?;
            }
            RebalancingAction::RedistributeLoad => {
                self.redistribute_load().await?;
            }
            RebalancingAction::ScaleUp => {
                self.scale_up().await?;
            }
            RebalancingAction::ScaleDown => {
                self.scale_down().await?;
            }
            RebalancingAction::Throttle => {
                self.throttle_high_load_devices().await?;
            }
            RebalancingAction::Custom(action) => {
                self.execute_custom_action(action).await?;
            }
        }

        Ok(())
    }

    /// Record load snapshot for history
    fn record_load_snapshot(&mut self) {
        let snapshot = LoadSnapshot {
            timestamp: Instant::now(),
            device_loads: self.device_loads.clone(),
            load_balance_metric: self.calculate_load_balance_metric(),
            hotspots: self.identify_hotspots(0.8),
        };

        self.load_history.push_back(snapshot);
        if self.load_history.len() > 1000 {
            self.load_history.pop_front();
        }
    }

    /// Evaluate rebalancing trigger
    fn evaluate_trigger(&self, trigger: &RebalancingTrigger) -> bool {
        match trigger {
            RebalancingTrigger::LoadImbalance(threshold) => {
                self.calculate_load_balance_metric() < *threshold
            }
            RebalancingTrigger::HighUtilization(threshold) => {
                self.device_loads.values().any(|load| {
                    (load.cpu_utilization + load.memory_utilization) / 2.0 > *threshold
                })
            }
            RebalancingTrigger::LowUtilization(threshold) => {
                self.device_loads.values().all(|load| {
                    (load.cpu_utilization + load.memory_utilization) / 2.0 < *threshold
                })
            }
            RebalancingTrigger::QueueBacklog(threshold) => {
                self.device_loads
                    .values()
                    .any(|load| load.queue_length > *threshold)
            }
            RebalancingTrigger::TemperatureThreshold(threshold) => {
                self.device_loads
                    .values()
                    .any(|load| load.temperature > *threshold)
            }
            RebalancingTrigger::Custom(_) => false, // Custom triggers need custom evaluation
        }
    }

    /// Migrate tasks from overloaded devices
    async fn migrate_tasks(&mut self) -> Result<()> {
        let hotspots = self.identify_hotspots(0.8);
        let underutilized: Vec<DeviceId> = self
            .device_loads
            .iter()
            .filter(|(_, load)| {
                (load.cpu_utilization + load.memory_utilization) / 2.0 < 0.5
            })
            .map(|(device_id, _)| *device_id)
            .collect();

        for hotspot in hotspots {
            if let Some(target) = underutilized.first() {
                // Simulate task migration
                self.migration_manager.insert(hotspot, 0.2); // Migrate 20% of load
                println!(
                    "Migrating tasks from device {:?} to device {:?}",
                    hotspot, target
                );
            }
        }

        Ok(())
    }

    /// Redistribute load across devices
    async fn redistribute_load(&mut self) -> Result<()> {
        let total_load: f64 = self
            .device_loads
            .values()
            .map(|load| (load.cpu_utilization + load.memory_utilization) / 2.0)
            .sum();
        let target_load = total_load / self.device_loads.len() as f64;

        println!("Redistributing load with target utilization: {:.2}", target_load);

        // Update migration manager with redistribution plan
        for (device_id, load) in &self.device_loads {
            let current_load = (load.cpu_utilization + load.memory_utilization) / 2.0;
            let adjustment = target_load - current_load;
            self.migration_manager.insert(*device_id, adjustment);
        }

        Ok(())
    }

    /// Scale up resources
    async fn scale_up(&mut self) -> Result<()> {
        println!("Scaling up resources - requesting additional devices");
        // In a real implementation, this would request additional resources
        Ok(())
    }

    /// Scale down resources
    async fn scale_down(&mut self) -> Result<()> {
        println!("Scaling down resources - releasing underutilized devices");
        // In a real implementation, this would release underutilized resources
        Ok(())
    }

    /// Throttle high load devices
    async fn throttle_high_load_devices(&mut self) -> Result<()> {
        let hotspots = self.identify_hotspots(0.9);
        for hotspot in hotspots {
            println!("Throttling device {:?} due to high utilization", hotspot);
            // In a real implementation, this would reduce task assignment to the device
        }
        Ok(())
    }

    /// Execute custom rebalancing action
    async fn execute_custom_action(&mut self, action: &str) -> Result<()> {
        println!("Executing custom rebalancing action: {}", action);
        // Custom action implementation would go here
        Ok(())
    }

    /// Get load balancing statistics
    pub fn get_statistics(&self) -> LoadBalanceStatistics {
        let mut stats = HashMap::new();

        stats.insert("load_variance".to_string(), {
            let utilizations: Vec<f64> = self
                .device_loads
                .values()
                .map(|load| (load.cpu_utilization + load.memory_utilization) / 2.0)
                .collect();

            if utilizations.is_empty() {
                0.0
            } else {
                let mean: f64 = utilizations.iter().sum::<f64>() / utilizations.len() as f64;
                utilizations
                    .iter()
                    .map(|u| (u - mean).powi(2))
                    .sum::<f64>()
                    / utilizations.len() as f64
            }
        });

        stats.insert(
            "rebalancing_events".to_string(),
            self.rebalancing_policies.len() as f64,
        );

        stats.insert(
            "active_migrations".to_string(),
            self.migration_manager.len() as f64,
        );

        stats.insert(
            "load_balance_metric".to_string(),
            self.calculate_load_balance_metric(),
        );

        stats.insert(
            "hotspot_count".to_string(),
            self.identify_hotspots(0.8).len() as f64,
        );

        stats.insert(
            "average_utilization".to_string(),
            if self.device_loads.is_empty() {
                0.0
            } else {
                self.device_loads
                    .values()
                    .map(|load| (load.cpu_utilization + load.memory_utilization) / 2.0)
                    .sum::<f64>()
                    / self.device_loads.len() as f64
            },
        );

        stats
    }

    /// Get load history
    pub fn get_load_history(&self) -> &VecDeque<LoadSnapshot> {
        &self.load_history
    }

    /// Set rebalancing policies
    pub fn set_rebalancing_policies(&mut self, policies: Vec<RebalancingPolicy>) {
        self.rebalancing_policies = policies;
    }

    /// Add rebalancing policy
    pub fn add_rebalancing_policy(&mut self, policy: RebalancingPolicy) {
        self.rebalancing_policies.push(policy);
    }

    /// Get current strategy
    pub fn get_strategy(&self) -> LoadBalancingStrategy {
        self.strategy
    }

    /// Update strategy
    pub fn set_strategy(&mut self, strategy: LoadBalancingStrategy) {
        self.strategy = strategy;
    }
}

/// Load balancer for resource allocation
#[derive(Debug)]
pub struct LoadBalancer {
    balancing_algorithm: LoadBalancingAlgorithm,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new() -> Self {
        Self {
            balancing_algorithm: LoadBalancingAlgorithm::RoundRobin,
        }
    }

    /// Create with specific algorithm
    pub fn with_algorithm(algorithm: LoadBalancingAlgorithm) -> Self {
        Self {
            balancing_algorithm: algorithm,
        }
    }

    /// Select optimal devices for resource allocation
    pub fn select_optimal_devices(
        &self,
        available_devices: &[DeviceId],
        device_availability: &HashMap<DeviceId, DeviceAvailability>,
    ) -> Vec<DeviceId> {
        match self.balancing_algorithm {
            LoadBalancingAlgorithm::RoundRobin => {
                available_devices.iter().take(4).cloned().collect()
            }
            LoadBalancingAlgorithm::LeastLoaded => {
                let mut devices_with_load: Vec<_> = available_devices
                    .iter()
                    .filter_map(|&device_id| {
                        device_availability
                            .get(&device_id)
                            .map(|availability| (device_id, availability.current_load))
                    })
                    .collect();

                devices_with_load.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                devices_with_load
                    .into_iter()
                    .take(4)
                    .map(|(device_id, _)| device_id)
                    .collect()
            }
            LoadBalancingAlgorithm::WeightedRoundRobin => {
                // Implement weighted round robin based on device capacity
                let mut weighted_devices: Vec<_> = available_devices
                    .iter()
                    .filter_map(|&device_id| {
                        device_availability.get(&device_id).map(|availability| {
                            // Calculate weight based on available capacity
                            let capacity_weight = availability.compute_capacity;
                            let memory_weight = availability.available_memory as f64
                                / (16.0 * 1024.0 * 1024.0 * 1024.0); // Normalize to 16GB
                            let load_weight = 1.0 - availability.current_load;
                            let bandwidth_weight = availability.communication_bandwidth / 100.0; // Normalize to 100 GB/s

                            let combined_weight =
                                (capacity_weight + memory_weight + load_weight + bandwidth_weight)
                                    / 4.0;
                            (device_id, combined_weight)
                        })
                    })
                    .collect();

                // Sort by weight (highest first)
                weighted_devices
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Select devices based on weighted round robin
                let mut selected = Vec::new();
                let total_weight: f64 = weighted_devices.iter().map(|(_, weight)| weight).sum();

                if total_weight > 0.0 {
                    let mut accumulated_weight = 0.0;
                    let weight_per_device = total_weight / 4.0; // Target 4 devices

                    for (device_id, weight) in &weighted_devices {
                        accumulated_weight += weight;
                        if accumulated_weight >= weight_per_device * (selected.len() + 1) as f64 {
                            selected.push(*device_id);
                            if selected.len() >= 4 {
                                break;
                            }
                        }
                    }

                    // Fill remaining slots if needed
                    while selected.len() < 4 && selected.len() < weighted_devices.len() {
                        for (device_id, _) in &weighted_devices {
                            if !selected.contains(device_id) {
                                selected.push(*device_id);
                                break;
                            }
                        }
                    }
                }

                selected
            }
            LoadBalancingAlgorithm::CapacityBased => {
                // Implement capacity-based selection prioritizing highest capacity devices
                let mut capacity_ranked_devices: Vec<_> = available_devices
                    .iter()
                    .filter_map(|&device_id| {
                        device_availability.get(&device_id).map(|availability| {
                            // Calculate comprehensive capacity score
                            let compute_score = availability.compute_capacity;
                            let memory_score = availability.available_memory as f64
                                / (32_u64 * 1024 * 1024 * 1024) as f64; // Normalize to 32GB max
                            let bandwidth_score = availability.communication_bandwidth / 200.0; // Normalize to 200 GB/s max
                            let load_efficiency = (1.0 - availability.current_load).max(0.1); // Avoid division by zero

                            // Weighted capacity score prioritizing compute > memory > bandwidth
                            let capacity_score = (
                                compute_score * 0.5 +      // 50% weight on compute capacity
                                memory_score * 0.3 +       // 30% weight on memory capacity
                                bandwidth_score * 0.2
                                // 20% weight on communication bandwidth
                            ) * load_efficiency; // Adjusted by current load efficiency

                            (device_id, capacity_score)
                        })
                    })
                    .collect();

                // Sort by capacity score (highest first)
                capacity_ranked_devices
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Select top capacity devices up to limit
                let selected_devices: Vec<DeviceId> = capacity_ranked_devices
                    .into_iter()
                    .take(4) // Take top 4 highest capacity devices
                    .map(|(device_id, _)| device_id)
                    .collect();

                // If we have fewer than 4 devices, ensure we have at least one
                if selected_devices.is_empty() && !available_devices.is_empty() {
                    vec![available_devices[0]]
                } else {
                    selected_devices
                }
            }
        }
    }

    /// Update balancing algorithm
    pub fn set_algorithm(&mut self, algorithm: LoadBalancingAlgorithm) {
        self.balancing_algorithm = algorithm;
    }

    /// Get current algorithm
    pub fn get_algorithm(&self) -> LoadBalancingAlgorithm {
        self.balancing_algorithm
    }
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

// Default implementations
impl Default for DeviceLoad {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            communication_utilization: 0.0,
            queue_length: 0,
            active_tasks: 0,
            temperature: 25.0, // Room temperature
            power_consumption: 0.0,
        }
    }
}

impl Default for DeviceAvailability {
    fn default() -> Self {
        Self {
            available_memory: 16 * 1024 * 1024 * 1024, // 16GB
            compute_capacity: 1.0,
            communication_bandwidth: 100.0, // GB/s
            current_load: 0.0,
            reserved_until: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pod_load_balancer_creation() {
        let balancer = PodLoadBalancer::new(LoadBalancingStrategy::Dynamic);
        assert!(balancer.is_ok());
    }

    #[test]
    fn test_load_balance_metric_calculation() {
        let mut balancer = PodLoadBalancer::new(LoadBalancingStrategy::Dynamic).unwrap();

        // Add some test devices with varying loads
        balancer.update_device_load(DeviceId(0), DeviceLoad {
            cpu_utilization: 0.5,
            memory_utilization: 0.6,
            ..Default::default()
        });

        balancer.update_device_load(DeviceId(1), DeviceLoad {
            cpu_utilization: 0.8,
            memory_utilization: 0.7,
            ..Default::default()
        });

        let metric = balancer.calculate_load_balance_metric();
        assert!(metric > 0.0 && metric <= 1.0);
    }

    #[test]
    fn test_hotspot_identification() {
        let mut balancer = PodLoadBalancer::new(LoadBalancingStrategy::Dynamic).unwrap();

        balancer.update_device_load(DeviceId(0), DeviceLoad {
            cpu_utilization: 0.9,
            memory_utilization: 0.95,
            ..Default::default()
        });

        balancer.update_device_load(DeviceId(1), DeviceLoad {
            cpu_utilization: 0.3,
            memory_utilization: 0.4,
            ..Default::default()
        });

        let hotspots = balancer.identify_hotspots(0.8);
        assert_eq!(hotspots.len(), 1);
        assert_eq!(hotspots[0], DeviceId(0));
    }

    #[test]
    fn test_load_balancer_device_selection() {
        let balancer = LoadBalancer::new();
        let devices = vec![DeviceId(0), DeviceId(1), DeviceId(2), DeviceId(3), DeviceId(4)];
        let mut availability = HashMap::new();

        for &device in &devices {
            availability.insert(device, DeviceAvailability::default());
        }

        let selected = balancer.select_optimal_devices(&devices, &availability);
        assert!(!selected.is_empty());
        assert!(selected.len() <= 4);
    }
}