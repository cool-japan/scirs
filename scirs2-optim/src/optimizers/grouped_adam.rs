//! Adam optimizer with parameter group support

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use crate::parameter_groups::{
    GroupManager, GroupedOptimizer, ParameterGroup, ParameterGroupConfig,
};
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// Adam optimizer with parameter group support
///
/// This optimizer allows different parameter groups to have different
/// hyperparameters (learning rate, weight decay, betas).
///
/// # Example
///
/// ```no_run
/// use ndarray::Array1;
/// use scirs2_optim::optimizers::{GroupedAdam, Optimizer};
/// use scirs2_optim::parameter_groups::{GroupedOptimizer, ParameterGroupConfig};
///
/// // Create grouped optimizer
/// let mut optimizer = GroupedAdam::new(0.001);
///
/// // Add parameter groups with different learning rates
/// let params_fast = vec![Array1::zeros(5)];
/// let config_fast = ParameterGroupConfig::new().with_learning_rate(0.01);
/// let group_fast = optimizer.add_group(params_fast, config_fast).unwrap();
///
/// let params_slow = vec![Array1::zeros(3)];
/// let config_slow = ParameterGroupConfig::new().with_learning_rate(0.0001);
/// let group_slow = optimizer.add_group(params_slow, config_slow).unwrap();
///
/// // Optimize each group separately
/// let grads_fast = vec![Array1::ones(5)];
/// let updated_fast = optimizer.step_group(group_fast, &grads_fast).unwrap();
///
/// let grads_slow = vec![Array1::ones(3)];
/// let updated_slow = optimizer.step_group(group_slow, &grads_slow).unwrap();
/// ```
#[derive(Debug)]
pub struct GroupedAdam<A: Float, D: Dimension> {
    /// Default learning rate
    default_lr: A,
    /// Default beta1
    default_beta1: A,
    /// Default beta2
    default_beta2: A,
    /// Default weight decay
    default_weight_decay: A,
    /// Epsilon to prevent division by zero
    epsilon: A,
    /// AMSGrad flag
    amsgrad: bool,
    /// Parameter groups
    group_manager: GroupManager<A, D>,
    /// Global step counter
    step: usize,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> GroupedAdam<A, D> {
    /// Create a new grouped Adam optimizer
    pub fn new(default_lr: A) -> Self {
        Self {
            default_lr,
            default_beta1: A::from(0.9).unwrap(),
            default_beta2: A::from(0.999).unwrap(),
            default_weight_decay: A::zero(),
            epsilon: A::from(1e-8).unwrap(),
            amsgrad: false,
            group_manager: GroupManager::new(),
            step: 0,
        }
    }

    /// Set default beta1
    pub fn with_beta1(mut self, beta1: A) -> Self {
        self.default_beta1 = beta1;
        self
    }

    /// Set default beta2
    pub fn with_beta2(mut self, beta2: A) -> Self {
        self.default_beta2 = beta2;
        self
    }

    /// Set default weight decay
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.default_weight_decay = weight_decay;
        self
    }

    /// Enable AMSGrad
    pub fn with_amsgrad(mut self) -> Self {
        self.amsgrad = true;
        self
    }

    /// Initialize state for a group
    fn init_group_state(&mut self, group_id: usize) -> Result<()> {
        let group = self.group_manager.get_group_mut(group_id)?;

        if group.state.is_empty() {
            let mut m_t = Vec::new();
            let mut v_t = Vec::new();
            let mut v_hat_max = Vec::new();

            for param in &group.params {
                m_t.push(Array::zeros(param.raw_dim()));
                v_t.push(Array::zeros(param.raw_dim()));
                if self.amsgrad {
                    v_hat_max.push(Array::zeros(param.raw_dim()));
                }
            }

            group.state.insert("m_t".to_string(), m_t);
            group.state.insert("v_t".to_string(), v_t);
            if self.amsgrad {
                group.state.insert("v_hat_max".to_string(), v_hat_max);
            }
        }

        Ok(())
    }

    /// Step for a specific group
    fn step_group_internal(
        &mut self,
        group_id: usize,
        gradients: &[Array<A, D>],
    ) -> Result<Vec<Array<A, D>>> {
        let t = A::from(self.step + 1).unwrap();

        // Initialize state if needed
        self.init_group_state(group_id)?;

        let group = self.group_manager.get_group_mut(group_id)?;

        if gradients.len() != group.params.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Number of gradients ({}) doesn't match number of parameters ({})",
                gradients.len(),
                group.params.len()
            )));
        }

        // Get hyperparameters for this group
        let lr = group.learning_rate(self.default_lr);
        let beta1 = group.get_custom_param("beta1", self.default_beta1);
        let beta2 = group.get_custom_param("beta2", self.default_beta2);
        let weight_decay = group.weight_decay(self.default_weight_decay);

        let mut updated_params = Vec::new();

        // Process each parameter
        for i in 0..group.params.len() {
            let param = &group.params[i];
            let grad = &gradients[i];

            // Apply weight decay
            let grad_with_decay = if weight_decay > A::zero() {
                grad + &(param * weight_decay)
            } else {
                grad.clone()
            };

            // Update states and compute new parameters
            let updated = {
                // Update first moment
                let m_t = group.state.get_mut("m_t").unwrap();
                m_t[i] = &m_t[i] * beta1 + &grad_with_decay * (A::one() - beta1);
                let m_hat = &m_t[i] / (A::one() - beta1.powi(t.to_i32().unwrap()));

                // Update second moment
                let v_t = group.state.get_mut("v_t").unwrap();
                v_t[i] = &v_t[i] * beta2 + &grad_with_decay * &grad_with_decay * (A::one() - beta2);
                let v_hat = &v_t[i] / (A::one() - beta2.powi(t.to_i32().unwrap()));

                // Update parameters
                if self.amsgrad {
                    let v_hat_max = group.state.get_mut("v_hat_max").unwrap();
                    v_hat_max[i].zip_mut_with(&v_hat, |a, &b| *a = a.max(b));
                    param - &(&m_hat * lr / (&v_hat_max[i].mapv(|x| x.sqrt()) + self.epsilon))
                } else {
                    param - &(&m_hat * lr / (&v_hat.mapv(|x| x.sqrt()) + self.epsilon))
                }
            };

            updated_params.push(updated);
        }

        // Update group parameters
        group.params = updated_params.clone();

        Ok(updated_params)
    }
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> GroupedOptimizer<A, D> for GroupedAdam<A, D> {
    fn add_group(
        &mut self,
        params: Vec<Array<A, D>>,
        config: ParameterGroupConfig<A>,
    ) -> Result<usize> {
        Ok(self.group_manager.add_group(params, config))
    }

    fn get_group(&self, group_id: usize) -> Result<&ParameterGroup<A, D>> {
        self.group_manager.get_group(group_id)
    }

    fn get_group_mut(&mut self, group_id: usize) -> Result<&mut ParameterGroup<A, D>> {
        self.group_manager.get_group_mut(group_id)
    }

    fn groups(&self) -> &[ParameterGroup<A, D>] {
        self.group_manager.groups()
    }

    fn groups_mut(&mut self) -> &mut [ParameterGroup<A, D>] {
        self.group_manager.groups_mut()
    }

    fn step_group(
        &mut self,
        group_id: usize,
        gradients: &[Array<A, D>],
    ) -> Result<Vec<Array<A, D>>> {
        self.step += 1;
        self.step_group_internal(group_id, gradients)
    }

    fn set_group_learning_rate(&mut self, group_id: usize, lr: A) -> Result<()> {
        let group = self.group_manager.get_group_mut(group_id)?;
        group.config.learning_rate = Some(lr);
        Ok(())
    }

    fn set_group_weight_decay(&mut self, group_id: usize, wd: A) -> Result<()> {
        let group = self.group_manager.get_group_mut(group_id)?;
        group.config.weight_decay = Some(wd);
        Ok(())
    }
}

// Standard optimizer implementation for default behavior
impl<A: Float + ScalarOperand + Debug, D: Dimension> Optimizer<A, D> for GroupedAdam<A, D> {
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // For single parameter optimization, create a temporary group
        let params_vec = vec![params.clone()];
        let gradients_vec = vec![gradients.clone()];
        let config = ParameterGroupConfig::new();

        let group_id = self.add_group(params_vec, config)?;
        let result = self.step_group(group_id, &gradients_vec)?;

        Ok(result.into_iter().next().unwrap())
    }

    fn get_learning_rate(&self) -> A {
        self.default_lr
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.default_lr = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_grouped_adam_creation() {
        let optimizer: GroupedAdam<f64, ndarray::Ix1> = GroupedAdam::new(0.001);
        assert_eq!(optimizer.default_lr, 0.001);
        assert_eq!(optimizer.default_beta1, 0.9);
        assert_eq!(optimizer.default_beta2, 0.999);
    }

    #[test]
    fn test_grouped_adam_multiple_groups() {
        let mut optimizer = GroupedAdam::new(0.001);

        // Add first group with high learning rate
        let params1 = vec![Array1::from_vec(vec![1.0, 2.0])];
        let config1 = ParameterGroupConfig::new().with_learning_rate(0.01);
        let group1 = optimizer.add_group(params1, config1).unwrap();

        // Add second group with low learning rate
        let params2 = vec![Array1::from_vec(vec![3.0, 4.0, 5.0])];
        let config2 = ParameterGroupConfig::new().with_learning_rate(0.0001);
        let group2 = optimizer.add_group(params2, config2).unwrap();

        // Update first group
        let grads1 = vec![Array1::from_vec(vec![0.1, 0.2])];
        let updated1 = optimizer.step_group(group1, &grads1).unwrap();

        // Update second group
        let grads2 = vec![Array1::from_vec(vec![0.3, 0.4, 0.5])];
        let updated2 = optimizer.step_group(group2, &grads2).unwrap();

        // Verify different updates due to different learning rates
        assert!(updated1[0][0] < 1.0); // Should decrease more
        assert!(updated2[0][0] > 2.9); // Should decrease less
    }

    #[test]
    fn test_grouped_adam_custom_betas() {
        let mut optimizer = GroupedAdam::new(0.001);

        // Add group with custom betas
        let params = vec![Array1::from_vec(vec![1.0, 2.0])];
        let config = ParameterGroupConfig::new()
            .with_custom_param("beta1".to_string(), 0.8)
            .with_custom_param("beta2".to_string(), 0.99);
        let group = optimizer.add_group(params, config).unwrap();

        // Verify custom parameters are used
        let group_ref = optimizer.get_group(group).unwrap();
        assert_eq!(group_ref.get_custom_param("beta1", 0.0), 0.8);
        assert_eq!(group_ref.get_custom_param("beta2", 0.0), 0.99);
    }

    #[test]
    fn test_grouped_adam_clear() {
        let mut optimizer = GroupedAdam::new(0.001);

        // Add groups
        let params1 = vec![Array1::zeros(2)];
        let config1 = ParameterGroupConfig::new();
        optimizer.add_group(params1, config1).unwrap();

        assert_eq!(optimizer.groups().len(), 1);

        // Clear groups
        optimizer.group_manager = GroupManager::new();
        optimizer.step = 0;

        assert_eq!(optimizer.groups().len(), 0);
        assert_eq!(optimizer.step, 0);
    }
}
