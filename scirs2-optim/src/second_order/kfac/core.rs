//! Core K-FAC optimizer implementation
//!
//! This module contains the main K-FAC (Kronecker-Factored Approximate Curvature)
//! optimizer implementation, providing second-order optimization through efficient
//! Fisher information matrix approximation.

use crate::error::{OptimError, Result};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use super::config::{KFACConfig, KFACStats, LayerInfo};
use super::layer_state::KFACLayerState;

/// Main K-FAC optimizer
#[derive(Debug)]
pub struct KFAC<T: Float> {
    /// Configuration
    config: KFACConfig<T>,

    /// Per-layer state
    layer_states: HashMap<String, KFACLayerState<T>>,

    /// Global step counter
    step_count: usize,

    /// Acceptance ratio for damping adjustment
    acceptance_ratio: T,

    /// Previous loss for loss-based damping
    previous_loss: Option<T>,

    /// Eigenvalue regularization history
    eigenvalue_history: Vec<T>,

    /// Performance statistics
    stats: KFACStats<T>,
}

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::iter::Sum
            + ndarray::ScalarOperand
            + 'static
            + num_traits::FromPrimitive,
    > KFAC<T>
{
    /// Create a new K-FAC optimizer
    pub fn new(config: KFACConfig<T>) -> Self {
        Self {
            config,
            layer_states: HashMap::new(),
            step_count: 0,
            acceptance_ratio: num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero()),
            previous_loss: None,
            eigenvalue_history: Vec::new(),
            stats: KFACStats::default(),
        }
    }

    /// Register a layer with the optimizer
    pub fn register_layer(&mut self, layer_info: LayerInfo) -> Result<()> {
        let layer_name = layer_info.name.clone();
        let state = KFACLayerState::new(layer_info, self.config.damping);
        self.layer_states.insert(layer_name, state);
        Ok(())
    }

    /// Update covariance matrices with new activations and gradients
    pub fn update_covariance_matrices(
        &mut self,
        layer_name: &str,
        activations: &Array2<T>,
        gradients: &Array2<T>,
    ) -> Result<()> {
        let should_update = {
            let state = self
                .layer_states
                .get(layer_name)
                .ok_or_else(|| OptimError::InvalidParameter(format!("Layer {} not found", layer_name)))?;

            self.step_count.saturating_sub(state.last_cov_update) >= self.config.cov_update_freq
        };

        if should_update {
            let state = self
                .layer_states
                .get_mut(layer_name)
                .ok_or_else(|| OptimError::InvalidParameter(format!("Layer {} not found", layer_name)))?;

            // Update input covariance matrix
            state.update_input_covariance(activations, self.config.stat_decay);

            // Update output gradient covariance matrix
            state.update_output_covariance(gradients, self.config.stat_decay);

            state.last_cov_update = self.step_count;
            self.stats.cov_updates += 1;
        }

        Ok(())
    }

    /// Update inverse covariance matrices
    pub fn update_inverse_matrices(&mut self, layer_name: &str) -> Result<()> {
        let should_update = {
            let state = self
                .layer_states
                .get(layer_name)
                .ok_or_else(|| OptimError::InvalidParameter(format!("Layer {} not found", layer_name)))?;

            self.step_count.saturating_sub(state.last_inv_update) >= self.config.inv_update_freq
        };

        if should_update {
            let current_damping = self.get_adaptive_damping(layer_name)?;

            let state = self
                .layer_states
                .get_mut(layer_name)
                .ok_or_else(|| OptimError::InvalidParameter(format!("Layer {} not found", layer_name)))?;

            state.compute_inverses(current_damping, current_damping)?;
            self.stats.inv_updates += 1;

            // Update condition number statistics
            let (a_cond, g_cond) = state.condition_number_estimate();
            let avg_cond = (a_cond + g_cond) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero());

            // Update running average of condition numbers
            let decay = num_traits::cast::cast(0.95).unwrap_or_else(|| T::zero());
            self.stats.avg_condition_number = decay * self.stats.avg_condition_number
                + (T::one() - decay) * avg_cond;
        }

        Ok(())
    }

    /// Apply K-FAC update to gradients
    pub fn apply_update(
        &mut self,
        layer_name: &str,
        gradients: &Array2<T>,
    ) -> Result<Array2<T>> {
        let state = self
            .layer_states
            .get(layer_name)
            .ok_or_else(|| OptimError::InvalidParameter(format!("Layer {} not found", layer_name)))?;

        if !state.is_ready() {
            // If inverses aren't computed yet, return scaled gradients
            return Ok(gradients * self.config.learning_rate);
        }

        let a_inv = state.a_cov_inv.as_ref().unwrap();
        let g_inv = state.g_cov_inv.as_ref().unwrap();

        // Apply K-FAC natural gradient update: G^{-1} * grad * A^{-1}
        let natural_gradients = g_inv.dot(gradients).dot(a_inv);

        // Scale by learning rate and apply weight decay
        let mut update = natural_gradients * self.config.learning_rate;

        if self.config.weight_decay > T::zero() {
            // Add weight decay: update = update + weight_decay * parameters
            // Note: In practice, parameters would be passed as an argument
            // For now, we just apply learning rate scaling
        }

        Ok(update)
    }

    /// Perform a complete optimization step
    pub fn step<F>(
        &mut self,
        layer_gradients: HashMap<String, (&Array2<T>, &Array2<T>)>,
        loss_fn: Option<F>,
    ) -> Result<HashMap<String, Array2<T>>>
    where
        F: FnOnce() -> T,
    {
        self.step_count += 1;
        self.stats.total_steps += 1;

        let mut updates = HashMap::new();

        // Update covariance matrices for all layers
        for (layer_name, (activations, gradients)) in &layer_gradients {
            self.update_covariance_matrices(layer_name, activations, gradients)?;
        }

        // Update inverse matrices if needed
        for layer_name in layer_gradients.keys() {
            self.update_inverse_matrices(layer_name)?;
        }

        // Compute natural gradient updates
        for (layer_name, (_, gradients)) in &layer_gradients {
            let update = self.apply_update(layer_name, gradients)?;
            updates.insert(layer_name.clone(), update);
        }

        // Update damping based on acceptance ratio if enabled
        if self.config.auto_damping {
            if let Some(loss_fn) = loss_fn {
                let current_loss = loss_fn();
                self.update_damping(current_loss);
            }
        }

        Ok(updates)
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &KFACStats<T> {
        &self.stats
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        for state in self.layer_states.values_mut() {
            state.reset();
        }
        self.step_count = 0;
        self.acceptance_ratio = num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero());
        self.previous_loss = None;
        self.eigenvalue_history.clear();
        self.stats = KFACStats::default();
    }

    /// Estimate memory usage in bytes
    pub fn estimate_memory_usage(&self) -> usize {
        let mut total = 0;

        for state in self.layer_states.values() {
            total += state.memory_usage();
        }

        // Add overhead for the optimizer itself
        total += std::mem::size_of::<Self>();
        total += self.eigenvalue_history.capacity() * std::mem::size_of::<T>();

        // Note: memory usage tracked in stats
        total
    }

    /// Get layer state for inspection
    pub fn get_layer_state(&self, layer_name: &str) -> Option<&KFACLayerState<T>> {
        self.layer_states.get(layer_name)
    }

    /// Set layer-specific damping parameters
    pub fn set_layer_damping(
        &mut self,
        layer_name: &str,
        damping_a: T,
        damping_g: T,
    ) -> Result<()> {
        let state = self
            .layer_states
            .get_mut(layer_name)
            .ok_or_else(|| OptimError::InvalidParameter(format!("Layer {} not found", layer_name)))?;

        state.damping_a = damping_a;
        state.damping_g = damping_g;
        Ok(())
    }

    /// Get the number of registered layers
    pub fn num_layers(&self) -> usize {
        self.layer_states.len()
    }

    /// Get list of registered layer names
    pub fn layer_names(&self) -> Vec<String> {
        self.layer_states.keys().cloned().collect()
    }

    /// Check if a layer is registered
    pub fn has_layer(&self, layer_name: &str) -> bool {
        self.layer_states.contains_key(layer_name)
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get current acceptance ratio
    pub fn acceptance_ratio(&self) -> T {
        self.acceptance_ratio
    }

    // Private helper methods

    fn get_adaptive_damping(&self, layer_name: &str) -> Result<T> {
        if !self.config.auto_damping {
            return Ok(self.config.damping);
        }

        // Simple adaptive damping based on acceptance ratio
        let base_damping = self.config.damping;
        let ratio_diff = self.acceptance_ratio - self.config.target_acceptance_ratio;

        if ratio_diff > T::zero() {
            // Acceptance ratio is too high, reduce damping
            Ok(base_damping * num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()))
        } else {
            // Acceptance ratio is too low, increase damping
            Ok(base_damping * num_traits::cast::cast(1.1).unwrap_or_else(|| T::zero()))
        }
    }

    fn update_damping(&mut self, current_loss: T) {
        if let Some(prev_loss) = self.previous_loss {
            // Update acceptance ratio based on loss improvement
            let loss_ratio = current_loss / prev_loss;
            let decay = num_traits::cast::cast(0.95).unwrap_or_else(|| T::zero());

            if loss_ratio <= T::one() {
                // Loss improved, increase acceptance ratio
                self.acceptance_ratio = decay * self.acceptance_ratio
                    + (T::one() - decay) * num_traits::cast::cast(1.2).unwrap_or_else(|| T::zero());
            } else {
                // Loss got worse, decrease acceptance ratio
                self.acceptance_ratio = decay * self.acceptance_ratio
                    + (T::one() - decay) * num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero());
            }

            // Clamp acceptance ratio to reasonable bounds
            let min_ratio = num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero());
            let max_ratio = num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero());
            self.acceptance_ratio = self.acceptance_ratio.max(min_ratio).min(max_ratio);
        }

        self.previous_loss = Some(current_loss);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::second_order::kfac::config::{LayerType, LayerInfo};

    #[test]
    fn test_kfac_creation() {
        let config = KFACConfig::<f32>::default();
        let kfac = KFAC::new(config);

        assert_eq!(kfac.num_layers(), 0);
        assert_eq!(kfac.step_count(), 0);
    }

    #[test]
    fn test_layer_registration() {
        let config = KFACConfig::<f64>::default();
        let mut kfac = KFAC::new(config);

        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 128,
            output_dim: 64,
            layer_type: LayerType::Dense,
            has_bias: true,
        };

        assert!(kfac.register_layer(layer_info).is_ok());
        assert_eq!(kfac.num_layers(), 1);
        assert!(kfac.has_layer("test_layer"));
    }

    #[test]
    fn test_covariance_update() {
        let config = KFACConfig::<f32>::default();
        let mut kfac = KFAC::new(config);

        let layer_info = LayerInfo {
            name: "test_layer".to_string(),
            input_dim: 4,
            output_dim: 2,
            layer_type: LayerType::Dense,
            has_bias: false,
        };

        kfac.register_layer(layer_info).unwrap();

        let activations = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let gradients = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        assert!(kfac.update_covariance_matrices("test_layer", &activations, &gradients).is_ok());

        let stats = kfac.get_stats();
        assert!(stats.cov_updates > 0);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let config = KFACConfig::<f64>::default();
        let mut kfac = KFAC::new(config);

        let layer_info = LayerInfo {
            name: "large_layer".to_string(),
            input_dim: 1000,
            output_dim: 500,
            layer_type: LayerType::Dense,
            has_bias: true,
        };

        kfac.register_layer(layer_info).unwrap();
        let memory_usage = kfac.estimate_memory_usage();

        assert!(memory_usage > 0);
        // Should be substantial for large matrices
        assert!(memory_usage > 1000000); // At least 1MB
    }

    #[test]
    fn test_damping_adjustment() {
        let mut config = KFACConfig::<f32>::default();
        config.auto_damping = true;
        config.target_acceptance_ratio = 0.75;

        let mut kfac = KFAC::new(config);

        // Simulate improving loss
        kfac.update_damping(1.0);
        kfac.update_damping(0.9); // Loss improved

        // Acceptance ratio should increase
        assert!(kfac.acceptance_ratio() > 1.0);

        // Simulate worsening loss
        kfac.update_damping(1.1); // Loss got worse

        // Acceptance ratio should decrease
        assert!(kfac.acceptance_ratio() < 1.2);
    }
}