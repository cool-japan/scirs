//! Tests for the Lion optimizer

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};
use scirs2_optim::optimizers::{Lion, Optimizer};

#[test]
fn test_lion_new_creates_optimizer_with_default_values() {
    let optimizer: Lion<f64> = Lion::new(0.001);
    assert_abs_diff_eq!(optimizer.learning_rate(), 0.001);
    assert_abs_diff_eq!(optimizer.get_beta1(), 0.9);
    assert_abs_diff_eq!(optimizer.get_beta2(), 0.99);
    assert_abs_diff_eq!(optimizer.get_weight_decay(), 0.0);
}

#[test]
fn test_lion_new_with_config() {
    let optimizer: Lion<f64> = Lion::new_with_config(0.01, 0.85, 0.98, 0.001);
    assert_abs_diff_eq!(optimizer.learning_rate(), 0.01);
    assert_abs_diff_eq!(optimizer.get_beta1(), 0.85);
    assert_abs_diff_eq!(optimizer.get_beta2(), 0.98);
    assert_abs_diff_eq!(optimizer.get_weight_decay(), 0.001);
}

#[test]
fn test_lion_setters() {
    let mut optimizer: Lion<f64> = Lion::new(0.001);

    optimizer.set_lr(0.002);
    assert_abs_diff_eq!(optimizer.learning_rate(), 0.002);

    optimizer.set_beta1(0.8);
    assert_abs_diff_eq!(optimizer.get_beta1(), 0.8);

    optimizer.set_beta2(0.95);
    assert_abs_diff_eq!(optimizer.get_beta2(), 0.95);

    optimizer.set_weight_decay(0.01);
    assert_abs_diff_eq!(optimizer.get_weight_decay(), 0.01);
}

#[test]
fn test_lion_1d_optimization() {
    let mut optimizer: Lion<f64> = Lion::new(0.1);

    // Start from x = 2.0, trying to minimize x^2
    let mut params = Array1::from_vec(vec![2.0]);

    // Run a few optimization steps
    for _ in 0..10 {
        // Gradient of x^2 is 2x
        let gradients = Array1::from_vec(vec![2.0 * params[0]]);
        params = optimizer.step(&params, &gradients).unwrap();
    }

    // Should be moving towards 0
    assert!(params[0].abs() < 1.0);
}

#[test]
fn test_lion_2d_optimization() {
    let mut optimizer: Lion<f64> = Lion::new(0.1);

    // Start from (2.0, 3.0), trying to minimize x^2 + y^2
    let mut params = Array1::from_vec(vec![2.0, 3.0]);

    // Run optimization steps
    for _ in 0..20 {
        // Gradient of x^2 + y^2 is (2x, 2y)
        let gradients = Array1::from_vec(vec![2.0 * params[0], 2.0 * params[1]]);
        params = optimizer.step(&params, &gradients).unwrap();
    }

    // Should be moving towards (0, 0)
    assert!(params[0].abs() < 1.0);
    assert!(params[1].abs() < 1.0);
}

#[test]
fn test_lion_with_weight_decay() {
    let mut optimizer: Lion<f64> = Lion::new_with_config(0.1, 0.9, 0.99, 0.1);

    // Start from (1.0, 1.0)
    let mut params = Array1::from_vec(vec![1.0, 1.0]);

    // Run optimization with zero gradients (only weight decay effect)
    for _ in 0..10 {
        let gradients = Array1::zeros(2);
        params = optimizer.step(&params, &gradients).unwrap();
    }

    // Weight decay should shrink parameters
    assert!(params[0] < 1.0);
    assert!(params[1] < 1.0);
}

#[test]
fn test_lion_reset() {
    let mut optimizer: Lion<f64> = Lion::new(0.1);

    // Perform a step to initialize internal state
    let params = Array1::from_vec(vec![1.0, 2.0]);
    let gradients = Array1::from_vec(vec![0.1, 0.2]);
    let _ = optimizer.step(&params, &gradients).unwrap();

    // Reset the optimizer
    optimizer.reset();

    // After reset, the next step should behave like the first step
    let new_params = optimizer.step(&params, &gradients).unwrap();

    // This should match the behavior of a fresh optimizer
    let mut fresh_optimizer: Lion<f64> = Lion::new(0.1);
    let fresh_params = fresh_optimizer.step(&params, &gradients).unwrap();

    assert_abs_diff_eq!(new_params[0], fresh_params[0], epsilon = 1e-6);
    assert_abs_diff_eq!(new_params[1], fresh_params[1], epsilon = 1e-6);
}

#[test]
fn test_lion_multiple_dimensions() {
    let mut optimizer: Lion<f64> = Lion::new(0.1);

    // Test with 2D array
    let params_2d = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let gradients_2d = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

    let updated_2d = optimizer.step(&params_2d, &gradients_2d).unwrap();

    // Check dimensions are preserved
    assert_eq!(updated_2d.shape(), params_2d.shape());

    // Check that parameters were updated
    assert_ne!(updated_2d[[0, 0]], params_2d[[0, 0]]);
    assert_ne!(updated_2d[[1, 2]], params_2d[[1, 2]]);
}

#[test]
fn test_lion_trait_methods() {
    let mut optimizer: Lion<f64> = Lion::new(0.001);

    // Test direct methods (not trait methods to avoid dimension inference issues)
    assert_abs_diff_eq!(optimizer.learning_rate(), 0.001);

    optimizer.set_lr(0.005);
    assert_abs_diff_eq!(optimizer.learning_rate(), 0.005);
}
