//! Resource allocation and management for cluster nodes
//!
//! This module provides comprehensive resource allocation capabilities
//! including various allocation strategies and optimization algorithms.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::HashMap;

#[cfg(feature = "logging")]
use log;

use super::types::{
    AllocationId, AllocationStrategy, ComputeCapacity, NodeInfo, NodeStatus, ResourceAllocation,
    ResourceRequirements, TaskId,
};

/// Resource allocation and management
#[derive(Debug)]
pub struct ResourceAllocator {
    allocations: HashMap<TaskId, ResourceAllocation>,
    available_resources: ComputeCapacity,
    allocation_strategy: AllocationStrategy,
}

impl Default for ResourceAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl ResourceAllocator {
    /// Create a new resource allocator
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            available_resources: ComputeCapacity::default(),
            allocation_strategy: AllocationStrategy::FirstFit,
        }
    }

    /// Update available resources based on current node status
    pub fn update_available_resources(&mut self, nodes: &[NodeInfo]) -> CoreResult<()> {
        self.available_resources = ComputeCapacity::default();

        for node in nodes {
            if node.status == NodeStatus::Healthy {
                self.available_resources.cpu_cores += node.capabilities.cpu_cores;
                self.available_resources.memory_gb += node.capabilities.memory_gb;
                self.available_resources.gpu_count += node.capabilities.gpu_count;
                self.available_resources.disk_space_gb += node.capabilities.disk_space_gb;
            }
        }

        // Subtract already allocated resources
        for allocation in self.allocations.values() {
            self.available_resources.cpu_cores = self
                .available_resources
                .cpu_cores
                .saturating_sub(allocation.allocated_resources.cpu_cores);
            self.available_resources.memory_gb = self
                .available_resources
                .memory_gb
                .saturating_sub(allocation.allocated_resources.memory_gb);
            self.available_resources.gpu_count = self
                .available_resources
                .gpu_count
                .saturating_sub(allocation.allocated_resources.gpu_count);
            self.available_resources.disk_space_gb = self
                .available_resources
                .disk_space_gb
                .saturating_sub(allocation.allocated_resources.disk_space_gb);
        }

        Ok(())
    }

    /// Allocate resources based on requirements
    pub fn allocate_resources(
        &self,
        requirements: &ResourceRequirements,
    ) -> CoreResult<ResourceAllocation> {
        // Check if resources are available
        if !self.can_satisfy_requirements(requirements) {
            return Err(CoreError::ResourceError(ErrorContext::new(
                "Insufficient resources available",
            )));
        }

        // Create allocation
        Ok(ResourceAllocation {
            allocation_id: AllocationId::generate(),
            allocated_resources: ComputeCapacity {
                cpu_cores: requirements.cpu_cores,
                memory_gb: requirements.memory_gb,
                gpu_count: requirements.gpu_count,
                disk_space_gb: requirements.disk_space_gb,
            },
            assigned_nodes: Vec::new(), // Would be populated with actual nodes
            created_at: std::time::Instant::now(),
            expires_at: None,
        })
    }

    /// Check if requirements can be satisfied with available resources
    fn can_satisfy_requirements(&self, requirements: &ResourceRequirements) -> bool {
        self.available_resources.cpu_cores >= requirements.cpu_cores
            && self.available_resources.memory_gb >= requirements.memory_gb
            && self.available_resources.gpu_count >= requirements.gpu_count
            && self.available_resources.disk_space_gb >= requirements.disk_space_gb
    }

    /// Optimize resource allocation using the configured strategy
    pub fn optimize_resource_allocation(&mut self) -> CoreResult<()> {
        // Implement resource optimization strategies
        match self.allocation_strategy {
            AllocationStrategy::FirstFit => {
                // First-fit allocation (already implemented)
            }
            AllocationStrategy::BestFit => {
                // Best-fit allocation
                self.optimize_best_fit()?;
            }
            AllocationStrategy::LoadBalanced => {
                // Load-balanced allocation
                self.optimize_load_balanced()?;
            }
        }
        Ok(())
    }

    /// Optimize using best-fit strategy
    fn optimize_best_fit(&mut self) -> CoreResult<()> {
        // Best-fit optimization: minimize resource fragmentation by allocating
        // to nodes that most closely match the resource requirements

        // Get all current allocations sorted by resource usage
        let mut allocations: Vec<(TaskId, ResourceAllocation)> = self
            .allocations
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Sort allocations by total resource "weight" (descending)
        // This helps identify heavy allocations that could be better placed
        allocations.sort_by(|a, b| {
            let weight_a = a.1.allocated_resources.cpu_cores
                + a.1.allocated_resources.memory_gb
                + a.1.allocated_resources.gpu_count * 4  // Weight GPUs more heavily
                + a.1.allocated_resources.disk_space_gb / 10; // Weight disk less
            let weight_b = b.1.allocated_resources.cpu_cores
                + b.1.allocated_resources.memory_gb
                + b.1.allocated_resources.gpu_count * 4
                + b.1.allocated_resources.disk_space_gb / 10;
            weight_b.cmp(&weight_a)
        });

        // Optimization strategy: consolidate small allocations onto fewer nodes
        // and ensure large allocations get dedicated resources

        // Track optimization improvements
        let mut optimizations_made = 0;
        let fragmentation_score_before = self.calculate_fragmentation_score();

        // Group allocations by size category
        let (large_allocations, medium_allocations, small_allocations): (Vec<_>, Vec<_>, Vec<_>) = {
            let mut large = Vec::new();
            let mut medium = Vec::new();
            let mut small = Vec::new();

            for (taskid, allocation) in allocations {
                let total_resources = allocation.allocated_resources.cpu_cores
                    + allocation.allocated_resources.memory_gb
                    + allocation.allocated_resources.gpu_count * 4;

                if total_resources >= 32 {
                    large.push((taskid.clone(), allocation.clone()));
                } else if total_resources >= 8 {
                    medium.push((taskid.clone(), allocation.clone()));
                } else {
                    small.push((taskid.clone(), allocation.clone()));
                }
            }

            (large, medium, small)
        };

        // Best-fit strategy for large allocations:
        // Ensure they get dedicated, high-capacity nodes
        for (taskid, allocation) in large_allocations {
            if allocation.assigned_nodes.len() > 1 {
                // Try to consolidate onto a single high-capacity node
                if self.attempt_consolidation(&taskid, &allocation)? {
                    optimizations_made += 1;
                }
            }
        }

        // Best-fit strategy for medium allocations:
        // Pair them efficiently to minimize waste
        for (taskid, allocation) in medium_allocations {
            if self.attempt_best_fit_pairing(&taskid, &allocation)? {
                optimizations_made += 1;
            }
        }

        // Best-fit strategy for small allocations:
        // Pack them tightly onto shared nodes
        for (taskid, allocation) in small_allocations {
            if self.attempt_small_allocation_packing(&taskid, &allocation)? {
                optimizations_made += 1;
            }
        }

        // Calculate improvement
        let fragmentation_score_after = self.calculate_fragmentation_score();
        let _improvement = fragmentation_score_before - fragmentation_score_after;

        if optimizations_made > 0 {
            #[cfg(feature = "logging")]
            log::info!(
                "Best-fit optimization completed: {optimizations_made} optimizations, fragmentation improved by {_improvement:.2}"
            );
        }

        Ok(())
    }

    /// Optimize using load-balanced strategy
    fn optimize_load_balanced(&mut self) -> CoreResult<()> {
        // Load-balanced optimization: distribute workload evenly across nodes
        // to prevent hot spots and maximize overall cluster throughput

        // Calculate current load distribution across nodes
        let mut nodeloads = HashMap::new();
        let mut total_load = 0.0f64;

        // Calculate load for each node based on current allocations
        for allocation in self.allocations.values() {
            for nodeid in &allocation.assigned_nodes {
                let load_weight =
                    self.calculate_allocation_load_weight(&allocation.allocated_resources);
                *nodeloads.entry(nodeid.clone()).or_insert(0.0) += load_weight;
                total_load += load_weight;
            }
        }

        // Identify the target load per node (assuming uniform node capabilities)
        let num_active_nodes = nodeloads.len().max(1);
        let target_load_per_node = total_load / num_active_nodes as f64;
        let load_variance_threshold = target_load_per_node * 0.15f64; // 15% variance allowed

        // Find overloaded and underloaded nodes
        let mut overloaded_nodes = Vec::new();
        let mut underloaded_nodes = Vec::new();

        for (nodeid, &current_load) in &nodeloads {
            let load_diff = current_load - target_load_per_node;
            if load_diff > load_variance_threshold {
                overloaded_nodes.push((nodeid.clone(), current_load, load_diff));
            } else if load_diff < -load_variance_threshold {
                underloaded_nodes.push((nodeid.clone(), current_load, -load_diff));
            }
        }

        // Sort by load difference (most extreme first)
        overloaded_nodes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        underloaded_nodes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        let mut rebalancing_actions = 0;
        let initial_variance = self.calculate_load_variance(&nodeloads);

        // Rebalancing algorithm: move allocations from overloaded to underloaded nodes
        for (overloaded_node, current_load, overloaded_amount) in overloaded_nodes {
            // Find allocations on this overloaded node that can be moved
            let moveable_allocations = self.find_moveable_allocations(&overloaded_node);

            for (taskid, allocation) in moveable_allocations {
                // Find the best underloaded node for this allocation
                if let Some((target_node, _)) = self.find_best_target_node(
                    &allocation.allocated_resources,
                    &underloaded_nodes
                        .iter()
                        .map(|(nodeid, load, _)| (nodeid.clone(), *load))
                        .collect::<Vec<_>>(),
                )? {
                    // Attempt to move the allocation
                    if self.attempt_allocation_migration(&taskid, &target_node)? {
                        rebalancing_actions += 1;

                        // Update node loads tracking
                        let allocation_weight =
                            self.calculate_allocation_load_weight(&allocation.allocated_resources);
                        if let Some(old_load) = nodeloads.get_mut(&overloaded_node) {
                            *old_load -= allocation_weight;
                        }
                        if let Some(new_load) = nodeloads.get_mut(&target_node) {
                            *new_load += allocation_weight;
                        }

                        // Check if we've balanced enough
                        if nodeloads.get(&overloaded_node).copied().unwrap_or(0.0)
                            <= target_load_per_node + load_variance_threshold
                        {
                            break; // This node is now balanced
                        }
                    }
                }
            }
        }

        // Secondary optimization: spread single large allocations across multiple nodes
        let single_node_allocations: Vec<(TaskId, ResourceAllocation)> = self
            .allocations
            .iter()
            .filter(|(_, allocation)| allocation.assigned_nodes.len() == 1)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        for (taskid, allocation) in single_node_allocations {
            let load_weight =
                self.calculate_allocation_load_weight(&allocation.allocated_resources);
            if load_weight > target_load_per_node * 0.6 {
                // Large allocation
                if self.attempt_allocation_spreading(&taskid, &allocation)? {
                    rebalancing_actions += 1;
                }
            }
        }

        // Calculate improvement in load balance
        let final_variance = self.calculate_load_variance(&nodeloads);
        let _variance_improvement = initial_variance - final_variance;

        if rebalancing_actions > 0 {
            #[cfg(feature = "logging")]
            log::info!(
                "Load-balanced optimization completed: {rebalancing_actions} rebalancing actions, \
                 load variance improved by {_variance_improvement:.2}"
            );
        }

        Ok(())
    }

    /// Get available capacity
    pub fn get_available_capacity(&self) -> ComputeCapacity {
        self.available_resources.clone()
    }

    // Helper methods for optimization algorithms

    /// Calculate fragmentation score (lower is better)
    fn calculate_fragmentation_score(&self) -> f64 {
        // Calculate how fragmented the resource allocation is
        // Lower score = better (less fragmented)
        let total_allocated_resources = self.allocations.len() as f64;
        if total_allocated_resources == 0.0 {
            return 0.0f64;
        }

        // Count allocations that are split across multiple nodes
        let split_allocations = self
            .allocations
            .values()
            .filter(|alloc| alloc.assigned_nodes.len() > 1)
            .count() as f64;

        // Calculate average resource utilization efficiency
        let mut total_efficiency = 0.0f64;
        for allocation in self.allocations.values() {
            let resource_efficiency =
                self.calculate_resource_efficiency(&allocation.allocated_resources);
            total_efficiency += resource_efficiency;
        }
        let avg_efficiency = total_efficiency / total_allocated_resources;

        // Fragmentation score: high split ratio + low efficiency = high fragmentation
        let split_ratio = split_allocations / total_allocated_resources;
        (split_ratio * 0.6 + (1.0 - avg_efficiency) * 0.4f64) * 100.0
    }

    /// Calculate resource efficiency (1.0 = perfect, 0.0 = inefficient)
    fn calculate_resource_efficiency(&self, resources: &ComputeCapacity) -> f64 {
        // Calculate how efficiently resources are being used
        // 1.0 = perfect efficiency, 0.0 = completely inefficient

        // Check resource balance (CPU:Memory:GPU ratio)
        let cpu_ratio = resources.cpu_cores as f64;
        let _memory_ratio = resources.memory_gb as f64 / 4.0f64; // Assume 4GB per CPU core is balanced
        let gpu_ratio = resources.gpu_count as f64 * 8.0f64; // Each GPU equivalent to 8 CPU cores

        let total_compute = cpu_ratio + gpu_ratio;
        let balanced_memory = total_compute * 4.0f64;

        // Efficiency is higher when memory allocation matches compute needs
        let memory_efficiency = if resources.memory_gb as f64 > 0.0 {
            balanced_memory.min(resources.memory_gb as f64)
                / balanced_memory.max(resources.memory_gb as f64)
        } else {
            1.0
        };

        // Also consider if resources are "too small" (overhead penalty)
        let scale_efficiency = if total_compute < 2.0 {
            total_compute / 2.0 // Penalty for very small allocations
        } else {
            1.0
        };

        let combined_efficiency = memory_efficiency * 0.7 + scale_efficiency * 0.3f64;
        combined_efficiency.min(1.0)
    }

    /// Calculate the load weight of an allocation
    fn calculate_allocation_load_weight(&self, resources: &ComputeCapacity) -> f64 {
        // Calculate the "load weight" of an allocation for load balancing
        // Higher weight = more demanding allocation
        let cpu_weight = resources.cpu_cores as f64;
        let memory_weight = resources.memory_gb as f64 * 0.25f64; // Memory is less constraining than CPU
        let gpu_weight = resources.gpu_count as f64 * 8.0f64; // GPUs are very constraining
        let disk_weight = resources.disk_space_gb as f64 * 0.01f64; // Disk is least constraining

        cpu_weight + memory_weight + gpu_weight + disk_weight
    }

    /// Calculate load variance across nodes
    fn calculate_load_variance(&self, nodeloads: &HashMap<String, f64>) -> f64 {
        // Calculate variance in load distribution across nodes
        if nodeloads.len() <= 1 {
            return 0.0f64;
        }

        let total_load: f64 = nodeloads.values().sum();
        let mean_load = total_load / nodeloads.len() as f64;

        let variance = nodeloads
            .values()
            .map(|&load| (load - mean_load).powi(2))
            .sum::<f64>()
            / nodeloads.len() as f64;

        variance.sqrt() // Return standard deviation
    }

    /// Find allocations that can be moved from a specific node
    fn find_moveable_allocations(&self, nodeid: &str) -> Vec<(TaskId, ResourceAllocation)> {
        // Find allocations on a specific node that can potentially be moved
        self.allocations
            .iter()
            .filter(|(_, allocation)| allocation.assigned_nodes.contains(&nodeid.to_string()))
            .map(|(taskid, allocation)| (taskid.clone(), allocation.clone()))
            .collect()
    }

    /// Get reference to available capacity
    pub fn available_capacity(&self) -> &ComputeCapacity {
        &self.available_resources
    }

    /// Attempt to consolidate an allocation onto fewer nodes
    pub fn attempt_consolidation(
        &mut self,
        _taskid: &TaskId,
        _allocation: &ResourceAllocation,
    ) -> CoreResult<bool> {
        // Placeholder implementation
        Ok(false)
    }

    /// Attempt to pair allocations for better fit
    pub fn attempt_best_fit_pairing(
        &mut self,
        _taskid: &TaskId,
        _allocation: &ResourceAllocation,
    ) -> CoreResult<bool> {
        // Placeholder implementation
        Ok(false)
    }

    /// Attempt to pack small allocations tightly
    pub fn attempt_small_allocation_packing(
        &mut self,
        _taskid: &TaskId,
        _allocation: &ResourceAllocation,
    ) -> CoreResult<bool> {
        // Placeholder implementation
        Ok(false)
    }

    /// Find the best target node for an allocation
    pub fn find_best_target_node(
        &mut self,
        _resources: &ComputeCapacity,
        _underloaded_nodes: &[(String, f64)],
    ) -> CoreResult<Option<(String, f64)>> {
        // Placeholder implementation
        Ok(None)
    }

    /// Attempt to migrate an allocation to a different node
    pub fn attempt_allocation_migration(
        &mut self,
        _taskid: &TaskId,
        _to_node: &str,
    ) -> CoreResult<bool> {
        // Placeholder implementation
        Ok(false)
    }

    /// Attempt to spread an allocation across multiple nodes
    pub fn attempt_allocation_spreading(
        &mut self,
        _taskid: &TaskId,
        _allocation: &ResourceAllocation,
    ) -> CoreResult<bool> {
        // Placeholder implementation
        Ok(false)
    }

    /// Set allocation strategy
    pub fn set_allocation_strategy(&mut self, strategy: AllocationStrategy) {
        self.allocation_strategy = strategy;
    }

    /// Get current allocation strategy
    pub fn get_allocation_strategy(&self) -> AllocationStrategy {
        self.allocation_strategy
    }

    /// Add an allocation
    pub fn add_allocation(&mut self, task_id: TaskId, allocation: ResourceAllocation) {
        self.allocations.insert(task_id, allocation);
    }

    /// Remove an allocation
    pub fn remove_allocation(&mut self, task_id: &TaskId) -> Option<ResourceAllocation> {
        self.allocations.remove(task_id)
    }

    /// Get all current allocations
    pub fn get_allocations(&self) -> &HashMap<TaskId, ResourceAllocation> {
        &self.allocations
    }
}
