//! Activation checkpoint management for memory-efficient training
//!
//! This module provides activation checkpointing capabilities to trade
//! computation for memory by selectively storing intermediate activations
//! and recomputing others during backward pass.

use std::collections::HashMap;
use std::time::Instant;
use ndarray::Array2;
use num_traits::Float;

use crate::error::{OptimError, Result};
use super::config::{CheckpointStrategy, EvictionPolicy, RecomputationCost};

/// Checkpoint manager for activation recomputation
#[derive(Debug)]
pub struct CheckpointManager<T: Float> {
    /// Active checkpoints indexed by layer ID
    checkpoints: HashMap<String, ActivationCheckpoint<T>>,

    /// Checkpoint strategy being used
    strategy: CheckpointStrategy,

    /// Maximum number of checkpoints to maintain
    max_checkpoints: usize,

    /// Total memory used by checkpoints (bytes)
    checkpoint_memory: usize,

    /// Recomputation cost analysis for each layer
    recomputation_costs: HashMap<String, RecomputationCost>,

    /// Eviction policy when checkpoint limit is reached
    eviction_policy: EvictionPolicy,

    /// Layer ordering for uniform checkpointing
    layer_order: Vec<String>,

    /// Current checkpoint interval for uniform strategy
    current_interval: usize,

    /// Memory pressure threshold for adaptive checkpointing
    memory_pressure_threshold: f32,

    /// Total memory saved through checkpointing
    memory_savings: usize,

    /// Performance statistics
    stats: CheckpointStats,
}

impl<T: Float + Clone + Default> CheckpointManager<T> {
    /// Create a new checkpoint manager
    pub fn new(strategy: CheckpointStrategy, max_checkpoints: usize) -> Self {
        Self {
            checkpoints: HashMap::new(),
            strategy,
            max_checkpoints,
            checkpoint_memory: 0,
            recomputation_costs: HashMap::new(),
            eviction_policy: EvictionPolicy::CostBased,
            layer_order: Vec::new(),
            current_interval: match strategy {
                CheckpointStrategy::Uniform(interval) => interval,
                _ => 4,
            },
            memory_pressure_threshold: 0.8,
            memory_savings: 0,
            stats: CheckpointStats::default(),
        }
    }

    /// Configure checkpoint manager parameters
    pub fn configure(
        &mut self,
        strategy: CheckpointStrategy,
        max_checkpoints: usize,
        eviction_policy: EvictionPolicy,
        memory_pressure_threshold: f32,
    ) {
        self.strategy = strategy;
        self.max_checkpoints = max_checkpoints;
        self.eviction_policy = eviction_policy;
        self.memory_pressure_threshold = memory_pressure_threshold;

        if let CheckpointStrategy::Uniform(interval) = strategy {
            self.current_interval = interval;
        }
    }

    /// Set layer execution order for uniform checkpointing
    pub fn set_layer_order(&mut self, layer_order: Vec<String>) {
        self.layer_order = layer_order;
    }

    /// Decide whether to checkpoint a layer based on strategy
    pub fn should_checkpoint(&self, layer_id: &str, memory_pressure: f32) -> bool {
        match self.strategy {
            CheckpointStrategy::Uniform(interval) => {
                if let Some(pos) = self.layer_order.iter().position(|l| l == layer_id) {
                    pos % interval == 0
                } else {
                    false
                }
            }
            CheckpointStrategy::Adaptive => {
                memory_pressure > self.memory_pressure_threshold
            }
            CheckpointStrategy::Optimal => {
                self.should_checkpoint_optimal(layer_id)
            }
            CheckpointStrategy::Manual => {
                // Manual checkpoints would be controlled externally
                false
            }
        }
    }

    /// Create a checkpoint for layer activations
    pub fn create_checkpoint(
        &mut self,
        layer_id: String,
        activations: Array2<T>,
        recomputation_cost: RecomputationCost,
    ) -> Result<()> {
        let memory_usage = activations.len() * std::mem::size_of::<T>();

        // Check if we need to evict existing checkpoints
        if self.checkpoints.len() >= self.max_checkpoints {
            self.evict_checkpoint()?;
        }

        let checkpoint = ActivationCheckpoint {
            layer_id: layer_id.clone(),
            activations,
            timestamp: Instant::now(),
            memory_usage,
            access_count: 0,
            recomputation_cost: recomputation_cost.clone(),
        };

        // Update tracking
        self.checkpoint_memory += memory_usage;
        self.recomputation_costs.insert(layer_id.clone(), recomputation_cost);
        self.checkpoints.insert(layer_id, checkpoint);

        // Update statistics
        self.stats.total_checkpoints_created += 1;
        self.stats.current_checkpoints = self.checkpoints.len();

        Ok(())
    }

    /// Retrieve a checkpoint and increment access count
    pub fn get_checkpoint(&mut self, layer_id: &str) -> Option<&Array2<T>> {
        if let Some(checkpoint) = self.checkpoints.get_mut(layer_id) {
            checkpoint.access_count += 1;
            Some(&checkpoint.activations)
        } else {
            None
        }
    }

    /// Remove a specific checkpoint
    pub fn remove_checkpoint(&mut self, layer_id: &str) -> Result<()> {
        if let Some(checkpoint) = self.checkpoints.remove(layer_id) {
            self.checkpoint_memory -= checkpoint.memory_usage;
            self.stats.total_checkpoints_removed += 1;
            self.stats.current_checkpoints = self.checkpoints.len();
        }
        Ok(())
    }

    /// Clear all checkpoints
    pub fn clear_all_checkpoints(&mut self) {
        let count = self.checkpoints.len();
        self.checkpoints.clear();
        self.checkpoint_memory = 0;
        self.stats.total_checkpoints_removed += count;
        self.stats.current_checkpoints = 0;
    }

    /// Get checkpoint memory usage
    pub fn get_memory_usage(&self) -> usize {
        self.checkpoint_memory
    }

    /// Get checkpoint statistics
    pub fn get_statistics(&self) -> &CheckpointStats {
        &self.stats
    }

    /// Estimate memory savings from checkpointing
    pub fn estimate_memory_savings(&self, total_activation_memory: usize) -> usize {
        // Estimate based on proportion of activations that are checkpointed vs recomputed
        let checkpoint_ratio = self.checkpoints.len() as f32 / (self.layer_order.len() as f32).max(1.0);
        (total_activation_memory as f32 * (1.0 - checkpoint_ratio)) as usize
    }

    /// Get recomputation cost for a layer
    pub fn get_recomputation_cost(&self, layer_id: &str) -> Option<&RecomputationCost> {
        self.recomputation_costs.get(layer_id)
    }

    /// Update recomputation cost for a layer
    pub fn update_recomputation_cost(
        &mut self,
        layer_id: String,
        cost: RecomputationCost,
    ) {
        self.recomputation_costs.insert(layer_id, cost);
    }

    // Private helper methods

    /// Evict a checkpoint based on eviction policy
    fn evict_checkpoint(&mut self) -> Result<()> {
        let layer_id_to_evict = match self.eviction_policy {
            EvictionPolicy::LRU => self.find_lru_checkpoint(),
            EvictionPolicy::LFU => self.find_lfu_checkpoint(),
            EvictionPolicy::CostBased => self.find_cost_based_checkpoint(),
            EvictionPolicy::FIFO => self.find_fifo_checkpoint(),
        };

        if let Some(layer_id) = layer_id_to_evict {
            self.remove_checkpoint(&layer_id)?;
            self.stats.total_evictions += 1;
        }

        Ok(())
    }

    fn find_lru_checkpoint(&self) -> Option<String> {
        self.checkpoints
            .iter()
            .min_by_key(|(_, checkpoint)| checkpoint.timestamp)
            .map(|(layer_id, _)| layer_id.clone())
    }

    fn find_lfu_checkpoint(&self) -> Option<String> {
        self.checkpoints
            .iter()
            .min_by_key(|(_, checkpoint)| checkpoint.access_count)
            .map(|(layer_id, _)| layer_id.clone())
    }

    fn find_cost_based_checkpoint(&self) -> Option<String> {
        // Evict checkpoint with lowest cost-benefit ratio for keeping
        self.checkpoints
            .iter()
            .min_by(|(_, a), (_, b)| {
                let a_benefit = a.recomputation_cost.cost_benefit_ratio / (a.access_count as f32 + 1.0);
                let b_benefit = b.recomputation_cost.cost_benefit_ratio / (b.access_count as f32 + 1.0);
                a_benefit.partial_cmp(&b_benefit).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(layer_id, _)| layer_id.clone())
    }

    fn find_fifo_checkpoint(&self) -> Option<String> {
        // Same as LRU for simplicity
        self.find_lru_checkpoint()
    }

    fn should_checkpoint_optimal(&self, layer_id: &str) -> bool {
        // Simplified optimal strategy: checkpoint layers with high recomputation cost
        if let Some(cost) = self.recomputation_costs.get(layer_id) {
            cost.cost_benefit_ratio > 1.0
        } else {
            false
        }
    }
}

/// Activation checkpoint data structure
#[derive(Debug, Clone)]
pub struct ActivationCheckpoint<T: Float> {
    /// Layer identifier
    pub layer_id: String,

    /// Stored activation tensors
    pub activations: Array2<T>,

    /// Timestamp when checkpoint was created
    pub timestamp: Instant,

    /// Memory usage in bytes
    pub memory_usage: usize,

    /// Number of times this checkpoint has been accessed
    pub access_count: usize,

    /// Cost analysis for recomputing this activation
    pub recomputation_cost: RecomputationCost,
}

impl<T: Float> ActivationCheckpoint<T> {
    /// Get the age of this checkpoint
    pub fn age(&self) -> std::time::Duration {
        Instant::now().duration_since(self.timestamp)
    }

    /// Calculate access frequency (accesses per minute)
    pub fn access_frequency(&self) -> f32 {
        let age_minutes = self.age().as_secs_f32() / 60.0;
        if age_minutes > 0.0 {
            self.access_count as f32 / age_minutes
        } else {
            self.access_count as f32
        }
    }

    /// Calculate priority score for eviction (lower = more likely to evict)
    pub fn eviction_priority(&self) -> f32 {
        let age_factor = self.age().as_secs_f32() / 3600.0; // Age in hours
        let access_factor = (self.access_count as f32 + 1.0).ln();
        let cost_factor = self.recomputation_cost.cost_benefit_ratio;

        // Higher cost, higher access, and newer checkpoints get higher priority (less likely to evict)
        cost_factor * access_factor / (age_factor + 1.0)
    }
}

/// Checkpoint performance statistics
#[derive(Debug, Clone, Default)]
pub struct CheckpointStats {
    /// Total number of checkpoints created
    pub total_checkpoints_created: usize,

    /// Total number of checkpoints removed
    pub total_checkpoints_removed: usize,

    /// Current number of active checkpoints
    pub current_checkpoints: usize,

    /// Total number of evictions due to memory constraints
    pub total_evictions: usize,

    /// Total recomputation operations performed
    pub total_recomputations: usize,

    /// Average checkpoint access frequency
    pub avg_access_frequency: f32,

    /// Memory savings achieved (bytes)
    pub memory_savings: usize,

    /// Recomputation overhead (nanoseconds)
    pub recomputation_overhead: u64,
}

impl CheckpointStats {
    /// Calculate checkpoint hit rate
    pub fn hit_rate(&self) -> f32 {
        let total_accesses = self.total_checkpoints_created + self.total_recomputations;
        if total_accesses > 0 {
            self.total_checkpoints_created as f32 / total_accesses as f32
        } else {
            0.0
        }
    }

    /// Calculate memory efficiency (savings per checkpoint)
    pub fn memory_efficiency(&self) -> f32 {
        if self.current_checkpoints > 0 {
            self.memory_savings as f32 / self.current_checkpoints as f32
        } else {
            0.0
        }
    }
}

/// Checkpoint optimizer for finding optimal checkpoint placement
pub struct CheckpointOptimizer {
    /// Layer computational costs
    layer_costs: HashMap<String, u64>,

    /// Layer memory requirements
    layer_memory: HashMap<String, usize>,

    /// Dependencies between layers
    layer_dependencies: HashMap<String, Vec<String>>,

    /// Maximum memory budget
    memory_budget: usize,
}

impl CheckpointOptimizer {
    /// Create a new checkpoint optimizer
    pub fn new(memory_budget: usize) -> Self {
        Self {
            layer_costs: HashMap::new(),
            layer_memory: HashMap::new(),
            layer_dependencies: HashMap::new(),
            memory_budget,
        }
    }

    /// Add layer information
    pub fn add_layer(
        &mut self,
        layer_id: String,
        compute_cost: u64,
        memory_requirement: usize,
        dependencies: Vec<String>,
    ) {
        self.layer_costs.insert(layer_id.clone(), compute_cost);
        self.layer_memory.insert(layer_id.clone(), memory_requirement);
        self.layer_dependencies.insert(layer_id, dependencies);
    }

    /// Find optimal checkpoint placement using dynamic programming
    pub fn optimize_checkpoints(&self) -> Result<Vec<String>> {
        // Simplified optimal checkpoint placement
        // In practice, this would use sophisticated algorithms like
        // Chen et al.'s "Training Deep Nets with Sublinear Memory Cost"

        let mut checkpoints = Vec::new();
        let mut current_memory = 0;

        // Sort layers by cost-benefit ratio
        let mut layers: Vec<_> = self.layer_costs.keys().collect();
        layers.sort_by(|&a, &b| {
            let a_ratio = self.calculate_benefit_ratio(a);
            let b_ratio = self.calculate_benefit_ratio(b);
            b_ratio.partial_cmp(&a_ratio).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Greedily select checkpoints
        for &layer_id in layers {
            let memory_req = self.layer_memory.get(layer_id).unwrap_or(&0);
            if current_memory + memory_req <= self.memory_budget {
                checkpoints.push(layer_id.clone());
                current_memory += memory_req;
            }
        }

        Ok(checkpoints)
    }

    fn calculate_benefit_ratio(&self, layer_id: &str) -> f32 {
        let compute_cost = *self.layer_costs.get(layer_id).unwrap_or(&1) as f32;
        let memory_cost = *self.layer_memory.get(layer_id).unwrap_or(&1) as f32;

        // Benefit ratio: recomputation cost saved per unit memory
        compute_cost / memory_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_manager_creation() {
        let manager = CheckpointManager::<f32>::new(CheckpointStrategy::Uniform(4), 10);
        assert_eq!(manager.max_checkpoints, 10);
        assert_eq!(manager.checkpoint_memory, 0);
    }

    #[test]
    fn test_checkpoint_creation_and_retrieval() {
        let mut manager = CheckpointManager::<f32>::new(CheckpointStrategy::Uniform(4), 10);

        let activations = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let cost = RecomputationCost {
            compute_cost: 1000,
            memory_cost: 24, // 6 f32s * 4 bytes
            time_cost: 5000,
            cost_benefit_ratio: 1.5,
        };

        manager.create_checkpoint("layer1".to_string(), activations, cost).unwrap();

        assert_eq!(manager.checkpoints.len(), 1);
        assert!(manager.get_checkpoint("layer1").is_some());
        assert!(manager.get_checkpoint("layer2").is_none());
    }

    #[test]
    fn test_uniform_checkpointing_strategy() {
        let manager = CheckpointManager::<f32>::new(CheckpointStrategy::Uniform(3), 10);
        let mut manager = manager;
        manager.set_layer_order(vec![
            "layer0".to_string(),
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
            "layer4".to_string(),
            "layer5".to_string(),
        ]);

        assert!(manager.should_checkpoint("layer0", 0.5)); // 0 % 3 == 0
        assert!(!manager.should_checkpoint("layer1", 0.5)); // 1 % 3 != 0
        assert!(!manager.should_checkpoint("layer2", 0.5)); // 2 % 3 != 0
        assert!(manager.should_checkpoint("layer3", 0.5)); // 3 % 3 == 0
    }

    #[test]
    fn test_adaptive_checkpointing_strategy() {
        let manager = CheckpointManager::<f32>::new(CheckpointStrategy::Adaptive, 10);

        assert!(!manager.should_checkpoint("layer1", 0.7)); // Below threshold
        assert!(manager.should_checkpoint("layer1", 0.9)); // Above threshold
    }

    #[test]
    fn test_checkpoint_eviction() {
        let mut manager = CheckpointManager::<f32>::new(CheckpointStrategy::Uniform(1), 2);

        let activations1 = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let activations2 = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let activations3 = Array2::from_shape_vec((1, 2), vec![5.0, 6.0]).unwrap();

        let cost = RecomputationCost::default();

        manager.create_checkpoint("layer1".to_string(), activations1, cost.clone()).unwrap();
        manager.create_checkpoint("layer2".to_string(), activations2, cost.clone()).unwrap();

        assert_eq!(manager.checkpoints.len(), 2);

        // Adding third checkpoint should trigger eviction
        manager.create_checkpoint("layer3".to_string(), activations3, cost).unwrap();

        assert_eq!(manager.checkpoints.len(), 2);
        assert!(manager.stats.total_evictions > 0);
    }

    #[test]
    fn test_checkpoint_optimizer() {
        let mut optimizer = CheckpointOptimizer::new(1000);

        optimizer.add_layer("layer1".to_string(), 100, 200, vec![]);
        optimizer.add_layer("layer2".to_string(), 200, 300, vec!["layer1".to_string()]);
        optimizer.add_layer("layer3".to_string(), 150, 250, vec!["layer2".to_string()]);

        let checkpoints = optimizer.optimize_checkpoints().unwrap();
        assert!(!checkpoints.is_empty());
    }

    #[test]
    fn test_activation_checkpoint_metrics() {
        let activations = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let cost = RecomputationCost {
            compute_cost: 1000,
            memory_cost: 16,
            time_cost: 5000,
            cost_benefit_ratio: 2.0,
        };

        let mut checkpoint = ActivationCheckpoint {
            layer_id: "test_layer".to_string(),
            activations,
            timestamp: Instant::now(),
            memory_usage: 16,
            access_count: 5,
            recomputation_cost: cost,
        };

        assert_eq!(checkpoint.access_count, 5);
        assert!(checkpoint.eviction_priority() > 0.0);

        // Simulate some time passing and more accesses
        std::thread::sleep(std::time::Duration::from_millis(10));
        checkpoint.access_count += 3;

        assert!(checkpoint.age().as_millis() >= 10);
        assert!(checkpoint.access_frequency() > 0.0);
    }
}