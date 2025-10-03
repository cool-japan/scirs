//! Basic Advanced functionality test
//!
//! This test verifies that the Advanced mode can be instantiated and performs basic operations.

use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_optimize::prelude::*;

/// Simple quadratic test function
#[allow(dead_code)]
fn quadratic_function(x: &ArrayView1<f64>) -> f64 {
    x.iter().map(|&xi| xi.powi(2)).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_coordinator_creation() {
        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let config = AdvancedConfig::default();
        let coordinator = AdvancedCoordinator::new(config, &initial_params.view());

        // Test that coordinator is created successfully
        assert_eq!(coordinator.state.global_best_solution.len(), 3);
    }

    #[test]
    fn test_advanced_basic_optimization() {
        let initial_params = Array1::from_vec(vec![1.0, 2.0]);
        let config = AdvancedConfig {
            strategy: AdvancedStrategy::AdaptiveSelection,
            max_nit: 10,
            max_evaluations: 100,
            tolerance: 1e-4,
            ..Default::default()
        };

        let result = advanced_optimize(quadratic_function, &initial_params.view(), Some(config));

        // Test that optimization runs without panicking
        assert!(result.is_ok());
        let opt_result = result.unwrap();

        // Test that some improvement was made
        let initial_obj = quadratic_function(&initial_params.view());
        assert!(opt_result.fun <= initial_obj);
    }

    #[test]
    fn test_neuromorphic_optimizer() {
        let config = NeuromorphicConfig {
            num_neurons: 10,
            total_time: 0.1, // Short time for quick test
            ..Default::default()
        };
        let initial_params = Array1::from_vec(vec![0.5, -0.5]);

        let result =
            neuromorphic_optimize(quadratic_function, &initial_params.view(), Some(config));

        assert!(result.is_ok());
        let opt_result = result.unwrap();
        assert!(opt_result.nit > 0);
    }
}
