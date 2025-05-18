//! Parameter groups for different learning rates and configurations
//!
//! This module provides support for parameter groups, allowing different
//! sets of parameters to have different hyperparameters (learning rate,
//! weight decay, etc.) within the same optimizer.

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Configuration for a parameter group
#[derive(Debug, Clone)]
pub struct ParameterGroupConfig<A: Float> {
    /// Learning rate for this group
    pub learning_rate: Option<A>,
    /// Weight decay for this group
    pub weight_decay: Option<A>,
    /// Momentum for this group (if applicable)
    pub momentum: Option<A>,
    /// Custom parameters as key-value pairs
    pub custom_params: HashMap<String, A>,
}

impl<A: Float> Default for ParameterGroupConfig<A> {
    fn default() -> Self {
        Self {
            learning_rate: None,
            weight_decay: None,
            momentum: None,
            custom_params: HashMap::new(),
        }
    }
}

impl<A: Float> ParameterGroupConfig<A> {
    /// Create a new parameter group configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: A) -> Self {
        self.learning_rate = Some(lr);
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, wd: A) -> Self {
        self.weight_decay = Some(wd);
        self
    }

    /// Set momentum
    pub fn with_momentum(mut self, momentum: A) -> Self {
        self.momentum = Some(momentum);
        self
    }

    /// Add custom parameter
    pub fn with_custom_param(mut self, key: String, value: A) -> Self {
        self.custom_params.insert(key, value);
        self
    }
}

/// A parameter group with its own configuration
#[derive(Debug)]
pub struct ParameterGroup<A: Float, D: Dimension> {
    /// Unique identifier for this group
    pub id: usize,
    /// Parameters in this group
    pub params: Vec<Array<A, D>>,
    /// Configuration for this group
    pub config: ParameterGroupConfig<A>,
    /// Internal state for optimization (optimizer-specific)
    pub state: HashMap<String, Vec<Array<A, D>>>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> ParameterGroup<A, D> {
    /// Create a new parameter group
    pub fn new(id: usize, params: Vec<Array<A, D>>, config: ParameterGroupConfig<A>) -> Self {
        Self {
            id,
            params,
            config,
            state: HashMap::new(),
        }
    }

    /// Get the number of parameters in this group
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Get learning rate for this group
    pub fn learning_rate(&self, default: A) -> A {
        self.config.learning_rate.unwrap_or(default)
    }

    /// Get weight decay for this group
    pub fn weight_decay(&self, default: A) -> A {
        self.config.weight_decay.unwrap_or(default)
    }

    /// Get momentum for this group
    pub fn momentum(&self, default: A) -> A {
        self.config.momentum.unwrap_or(default)
    }

    /// Get custom parameter
    pub fn get_custom_param(&self, key: &str, default: A) -> A {
        self.config
            .custom_params
            .get(key)
            .copied()
            .unwrap_or(default)
    }
}

/// Optimizer with parameter group support
pub trait GroupedOptimizer<A: Float + ScalarOperand + Debug, D: Dimension>:
    Optimizer<A, D>
{
    /// Add a parameter group
    fn add_group(
        &mut self,
        params: Vec<Array<A, D>>,
        config: ParameterGroupConfig<A>,
    ) -> Result<usize>;

    /// Get parameter group by ID
    fn get_group(&self, group_id: usize) -> Result<&ParameterGroup<A, D>>;

    /// Get mutable parameter group by ID
    fn get_group_mut(&mut self, group_id: usize) -> Result<&mut ParameterGroup<A, D>>;

    /// Get all parameter groups
    fn groups(&self) -> &[ParameterGroup<A, D>];

    /// Get all parameter groups mutably
    fn groups_mut(&mut self) -> &mut [ParameterGroup<A, D>];

    /// Step for a specific group
    fn step_group(
        &mut self,
        group_id: usize,
        gradients: &[Array<A, D>],
    ) -> Result<Vec<Array<A, D>>>;

    /// Set learning rate for a specific group
    fn set_group_learning_rate(&mut self, group_id: usize, lr: A) -> Result<()>;

    /// Set weight decay for a specific group
    fn set_group_weight_decay(&mut self, group_id: usize, wd: A) -> Result<()>;
}

/// Helper struct for managing parameter groups
#[derive(Debug)]
pub struct GroupManager<A: Float, D: Dimension> {
    groups: Vec<ParameterGroup<A, D>>,
    next_id: usize,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> Default for GroupManager<A, D> {
    fn default() -> Self {
        Self {
            groups: Vec::new(),
            next_id: 0,
        }
    }
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> GroupManager<A, D> {
    /// Create a new group manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new parameter group
    pub fn add_group(
        &mut self,
        params: Vec<Array<A, D>>,
        config: ParameterGroupConfig<A>,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.groups.push(ParameterGroup::new(id, params, config));
        id
    }

    /// Get group by ID
    pub fn get_group(&self, id: usize) -> Result<&ParameterGroup<A, D>> {
        self.groups
            .iter()
            .find(|g| g.id == id)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Group {} not found", id)))
    }

    /// Get mutable group by ID
    pub fn get_group_mut(&mut self, id: usize) -> Result<&mut ParameterGroup<A, D>> {
        self.groups
            .iter_mut()
            .find(|g| g.id == id)
            .ok_or_else(|| OptimError::InvalidConfig(format!("Group {} not found", id)))
    }

    /// Get all groups
    pub fn groups(&self) -> &[ParameterGroup<A, D>] {
        &self.groups
    }

    /// Get all groups mutably
    pub fn groups_mut(&mut self) -> &mut [ParameterGroup<A, D>] {
        &mut self.groups
    }

    /// Get total number of parameters across all groups
    pub fn total_params(&self) -> usize {
        self.groups.iter().map(|g| g.num_params()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_parameter_group_config() {
        let config = ParameterGroupConfig::new()
            .with_learning_rate(0.01)
            .with_weight_decay(0.0001)
            .with_momentum(0.9)
            .with_custom_param("beta1".to_string(), 0.9)
            .with_custom_param("beta2".to_string(), 0.999);

        assert_eq!(config.learning_rate, Some(0.01));
        assert_eq!(config.weight_decay, Some(0.0001));
        assert_eq!(config.momentum, Some(0.9));
        assert_eq!(config.custom_params.get("beta1"), Some(&0.9));
        assert_eq!(config.custom_params.get("beta2"), Some(&0.999));
    }

    #[test]
    fn test_parameter_group() {
        let params = vec![Array1::zeros(5), Array1::ones(3)];
        let config = ParameterGroupConfig::new().with_learning_rate(0.01);

        let group = ParameterGroup::new(0, params, config);

        assert_eq!(group.id, 0);
        assert_eq!(group.num_params(), 2);
        assert_eq!(group.learning_rate(0.001), 0.01);
        assert_eq!(group.weight_decay(0.0), 0.0);
    }

    #[test]
    fn test_group_manager() {
        let mut manager: GroupManager<f64, ndarray::Ix1> = GroupManager::new();

        // Add first group
        let params1 = vec![Array1::zeros(5)];
        let config1 = ParameterGroupConfig::new().with_learning_rate(0.01);
        let id1 = manager.add_group(params1, config1);

        // Add second group
        let params2 = vec![Array1::ones(3), Array1::zeros(4)];
        let config2 = ParameterGroupConfig::new().with_learning_rate(0.001);
        let id2 = manager.add_group(params2, config2);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(manager.groups().len(), 2);
        assert_eq!(manager.total_params(), 3);

        // Test group access
        let group1 = manager.get_group(id1).unwrap();
        assert_eq!(group1.learning_rate(0.0), 0.01);

        let group2 = manager.get_group(id2).unwrap();
        assert_eq!(group2.learning_rate(0.0), 0.001);
    }
}
