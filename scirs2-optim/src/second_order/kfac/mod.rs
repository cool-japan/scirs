//! K-FAC (Kronecker-Factored Approximate Curvature) Second-Order Optimizer
//!
//! This module provides a complete implementation of the K-FAC second-order optimization
//! algorithm, which approximates the Fisher information matrix using Kronecker products
//! for efficient computation of natural gradients.
//!
//! # Features
//!
//! - **Efficient Fisher Information Approximation**: Uses Kronecker factorization to
//!   approximate the Fisher information matrix with manageable computational complexity
//! - **Layer-Specific Optimization**: Maintains separate covariance matrices for each
//!   layer type (dense, convolutional, batch normalization)
//! - **Adaptive Damping**: Automatic adjustment of regularization parameters based on
//!   optimization progress and acceptance ratios
//! - **Natural Gradients**: Computes preconditioned gradients using Fisher information
//!   matrix inverses for improved convergence
//! - **Memory Efficient**: Stores only necessary covariance matrices and their inverses
//!
//! # Module Structure
//!
//! - [`config`] - Configuration structures and validation
//! - [`layer_state`] - Per-layer state management and covariance computation
//! - [`core`] - Main K-FAC optimizer implementation
//! - [`natural_gradients`] - Natural gradient computation and matrix operations
//! - [`utils`] - Mathematical utilities and layer-specific operations
//!
//! # Usage
//!
//! ```rust
//! use scirs2_optim::second_order::kfac::{KFAC, KFACConfig, LayerInfo, LayerType};
//! use ndarray::Array2;
//!
//! // Create K-FAC configuration
//! let config = KFACConfig::<f32>::default();
//! let mut kfac = KFAC::new(config);
//!
//! // Register layers
//! let layer_info = LayerInfo::dense("layer1".to_string(), 128, 64, true);
//! kfac.register_layer(layer_info).unwrap();
//!
//! // Optimization step with activations and gradients
//! let activations = Array2::zeros((32, 128)); // batch_size x input_dim
//! let gradients = Array2::zeros((32, 64));    // batch_size x output_dim
//!
//! let mut layer_gradients = std::collections::HashMap::new();
//! layer_gradients.insert("layer1".to_string(), (&activations, &gradients));
//!
//! let updates = kfac.step(layer_gradients, None).unwrap();
//! ```
//!
//! # Algorithm Details
//!
//! K-FAC approximates the Fisher information matrix F as a Kronecker product:
//! F ≈ A ⊗ G where:
//! - A is the covariance of layer inputs (activations)
//! - G is the covariance of layer output gradients
//!
//! The natural gradient update becomes:
//! θ_{t+1} = θ_t - η * (A^{-1} ⊗ G^{-1}) * ∇L(θ_t)
//!
//! This can be computed efficiently as:
//! ΔW = η * G^{-1} * ∇W * A^{-1}
//!
//! # References
//!
//! - Martens, J., & Grosse, R. (2015). Optimizing neural networks with kronecker-factored
//!   approximate curvature. In International conference on machine learning (pp. 2408-2417).

pub mod config;
pub mod core;
pub mod layer_state;
pub mod natural_gradients;
pub mod utils;

// Re-export main types for convenient access
pub use config::{KFACConfig, KFACStats, LayerInfo, LayerType};
pub use core::KFAC;
pub use layer_state::KFACLayerState;
pub use natural_gradients::{NaturalGradientCompute, NaturalGradientConfig};
pub use utils::{KFACUtils, OrderedFloat};

// Re-export specific functions that might be used independently
pub use natural_gradients::NaturalGradientCompute as NGCompute;
pub use utils::KFACUtils as Utils;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use ndarray::Array2;
    use std::collections::HashMap;

    #[test]
    fn test_kfac_integration_dense_layer() {
        let config = KFACConfig::<f32>::default();
        let mut kfac = KFAC::new(config);

        // Register a dense layer
        let layer_info = LayerInfo::dense("dense1".to_string(), 4, 2, true);
        assert!(kfac.register_layer(layer_info).is_ok());

        // Create sample data
        let activations = Array2::from_shape_vec((3, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ]).unwrap();

        let gradients = Array2::from_shape_vec((3, 2), vec![
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6,
        ]).unwrap();

        // Perform optimization step
        let mut layer_gradients = HashMap::new();
        layer_gradients.insert("dense1".to_string(), (&activations, &gradients));

        let updates = kfac.step(layer_gradients, None).unwrap();
        assert!(updates.contains_key("dense1"));
        assert_eq!(updates["dense1"].dim(), gradients.dim());
    }

    #[test]
    fn test_kfac_integration_multiple_layers() {
        let config = KFACConfig::<f64>::for_small_model();
        let mut kfac = KFAC::new(config);

        // Register multiple layers
        let layer1 = LayerInfo::dense("layer1".to_string(), 8, 4, true);
        let layer2 = LayerInfo::dense("layer2".to_string(), 4, 2, false);

        assert!(kfac.register_layer(layer1).is_ok());
        assert!(kfac.register_layer(layer2).is_ok());

        assert_eq!(kfac.num_layers(), 2);
        assert!(kfac.has_layer("layer1"));
        assert!(kfac.has_layer("layer2"));

        // Create sample data for both layers
        let batch_size = 5;
        let activations1 = Array2::ones((batch_size, 8));
        let gradients1 = Array2::ones((batch_size, 4)) * 0.1;

        let activations2 = Array2::ones((batch_size, 4));
        let gradients2 = Array2::ones((batch_size, 2)) * 0.2;

        let mut layer_gradients = HashMap::new();
        layer_gradients.insert("layer1".to_string(), (&activations1, &gradients1));
        layer_gradients.insert("layer2".to_string(), (&activations2, &gradients2));

        // Multiple optimization steps
        for step in 0..3 {
            let updates = kfac.step(layer_gradients.clone(), None).unwrap();

            assert_eq!(updates.len(), 2);
            assert!(updates.contains_key("layer1"));
            assert!(updates.contains_key("layer2"));

            // Check that updates have correct dimensions
            assert_eq!(updates["layer1"].dim(), gradients1.dim());
            assert_eq!(updates["layer2"].dim(), gradients2.dim());

            // Verify step count increases
            assert_eq!(kfac.step_count(), step + 1);
        }

        // Check statistics
        let stats = kfac.get_stats();
        assert_eq!(stats.total_steps, 3);
        assert!(stats.cov_updates > 0);
    }

    #[test]
    fn test_kfac_memory_usage() {
        let config = KFACConfig::<f32>::default();
        let mut kfac = KFAC::new(config);

        // Register a large layer to test memory estimation
        let layer_info = LayerInfo::dense("large_layer".to_string(), 512, 256, true);
        kfac.register_layer(layer_info).unwrap();

        let memory_usage = kfac.estimate_memory_usage();
        assert!(memory_usage > 0);

        // Should be substantial for large matrices (at least several MB)
        let expected_minimum = (513 * 513 + 256 * 256) * 2 * std::mem::size_of::<f32>();
        assert!(memory_usage >= expected_minimum);
    }

    #[test]
    fn test_kfac_adaptive_damping() {
        let mut config = KFACConfig::<f32>::default();
        config.auto_damping = true;
        config.target_acceptance_ratio = 0.8;

        let mut kfac = KFAC::new(config);

        let layer_info = LayerInfo::dense("test_layer".to_string(), 4, 2, false);
        kfac.register_layer(layer_info).unwrap();

        let activations = Array2::ones((2, 4));
        let gradients = Array2::ones((2, 2)) * 0.1;

        let mut layer_gradients = HashMap::new();
        layer_gradients.insert("test_layer".to_string(), (&activations, &gradients));

        // Test with improving loss
        let loss_fn = || 1.0; // Constant loss for first step
        kfac.step(layer_gradients.clone(), Some(loss_fn)).unwrap();

        let improving_loss_fn = || 0.8; // Improving loss
        kfac.step(layer_gradients.clone(), Some(improving_loss_fn)).unwrap();

        // Acceptance ratio should reflect the improvement
        assert!(kfac.acceptance_ratio() >= 1.0);

        // Test with worsening loss
        let worsening_loss_fn = || 1.2; // Worsening loss
        kfac.step(layer_gradients, Some(worsening_loss_fn)).unwrap();

        // Acceptance ratio should decrease
        assert!(kfac.acceptance_ratio() < 1.2);
    }

    #[test]
    fn test_kfac_layer_specific_damping() {
        let config = KFACConfig::<f64>::default();
        let mut kfac = KFAC::new(config);

        let layer_info = LayerInfo::dense("test_layer".to_string(), 3, 2, false);
        kfac.register_layer(layer_info).unwrap();

        // Set custom damping for the layer
        assert!(kfac.set_layer_damping("test_layer", 0.01, 0.02).is_ok());

        let state = kfac.get_layer_state("test_layer").unwrap();
        assert!((state.damping_a - 0.01).abs() < 1e-10);
        assert!((state.damping_g - 0.02).abs() < 1e-10);

        // Test error for non-existent layer
        assert!(kfac.set_layer_damping("nonexistent", 0.01, 0.02).is_err());
    }

    #[test]
    fn test_kfac_reset() {
        let config = KFACConfig::<f32>::default();
        let mut kfac = KFAC::new(config);

        let layer_info = LayerInfo::dense("test_layer".to_string(), 2, 2, false);
        kfac.register_layer(layer_info).unwrap();

        let activations = Array2::ones((2, 2));
        let gradients = Array2::ones((2, 2)) * 0.1;

        let mut layer_gradients = HashMap::new();
        layer_gradients.insert("test_layer".to_string(), (&activations, &gradients));

        // Perform some steps
        kfac.step(layer_gradients.clone(), None).unwrap();
        kfac.step(layer_gradients, None).unwrap();

        assert_eq!(kfac.step_count(), 2);
        assert!(kfac.get_stats().total_steps > 0);

        // Reset and verify state
        kfac.reset();

        assert_eq!(kfac.step_count(), 0);
        assert_eq!(kfac.get_stats().total_steps, 0);
        assert!((kfac.acceptance_ratio() - 1.0).abs() < 1e-6);

        // Layer should still be registered but reset
        assert!(kfac.has_layer("test_layer"));
        let state = kfac.get_layer_state("test_layer").unwrap();
        assert_eq!(state.num_updates, 0);
        assert!(!state.is_ready()); // Inverses should be cleared
    }
}