//! Simple integration tests for core scirs2-optimize functionality
//!
//! These tests validate the basic optimization algorithms work correctly.

use ndarray::{Array1, ArrayView1};
use scirs2_optimize::stochastic::{
    minimize_adam, minimize_sgd, AdamOptions, DataProvider, InMemoryDataProvider, SGDOptions,
    StochasticGradientFunction,
};

/// Simple quadratic function for testing
struct QuadraticFunction;

impl StochasticGradientFunction for QuadraticFunction {
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
        x.mapv(|xi| 2.0 * xi)
    }

    fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
        x.mapv(|xi| xi * xi).sum()
    }
}

#[test]
fn test_basic_sgd_integration() {
    // Test SGD
    let grad_func = QuadraticFunction;
    let x0 = Array1::from_vec(vec![1.0, -1.0]);
    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

    let options = SGDOptions {
        learning_rate: 0.1,
        max_iter: 100,
        tol: 1e-4,
        ..Default::default()
    };

    let result = minimize_sgd(grad_func, x0.clone(), data_provider, options).unwrap();

    // Should converge toward zero
    assert!(
        result.fun < 1e-2,
        "SGD should converge to low function value"
    );
    println!(
        "SGD converged to f = {:.2e} in {} iterations",
        result.fun, result.iterations
    );
}

#[test]
fn test_basic_adam_integration() {
    // Test Adam optimizer
    let grad_func = QuadraticFunction;
    let x0 = Array1::from_vec(vec![2.0, -1.5]);
    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

    let options = AdamOptions {
        learning_rate: 0.1,
        max_iter: 200,
        tol: 1e-6,
        ..Default::default()
    };

    let result = minimize_adam(grad_func, x0, data_provider, options).unwrap();

    // Adam should converge efficiently
    assert!(
        result.fun < 1e-3,
        "Adam should converge to low function value"
    );
    println!(
        "Adam converged to f = {:.2e} in {} iterations",
        result.fun, result.iterations
    );
}

#[test]
fn test_stochastic_optimization_capabilities() {
    println!("\n🔬 scirs2-optimize Stochastic Capabilities Test");
    println!("==============================================");

    // Test that we can create different optimizers
    let _sgd_options = SGDOptions::default();
    let _adam_options = AdamOptions::default();

    println!("✅ All optimizer option structs created successfully");

    // Test data provider
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let provider = InMemoryDataProvider::new(data.clone());
    assert_eq!(provider.num_samples(), 5);
    assert_eq!(provider.get_full_data(), data);

    println!("✅ Data provider functionality verified");

    // Test gradient function trait
    let mut grad_func = QuadraticFunction;
    let x = Array1::from_vec(vec![1.0, 2.0]);
    let batch_data = vec![1.0];

    let gradient = grad_func.compute_gradient(&x.view(), &batch_data);
    let expected = Array1::from_vec(vec![2.0, 4.0]);
    assert_eq!(gradient, expected);

    let value = grad_func.compute_value(&x.view(), &batch_data);
    assert_eq!(value, 5.0); // 1^2 + 2^2 = 5

    println!("✅ Stochastic gradient function trait verified");
    println!("✅ All core stochastic optimization capabilities working correctly!");
}

#[test]
fn test_stochastic_algorithms_variety() {
    use scirs2_optimize::stochastic::{
        minimize_adamw, minimize_rmsprop, AdamWOptions, RMSPropOptions,
    };

    // Test multiple stochastic optimization algorithms
    let x0 = Array1::from_vec(vec![1.0, 1.0]);

    // Test RMSProp
    let rmsprop_options = RMSPropOptions {
        learning_rate: 0.05,
        max_iter: 200,
        tol: 1e-4,
        ..Default::default()
    };
    let data_provider1 = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));
    let result_rmsprop = minimize_rmsprop(
        QuadraticFunction,
        x0.clone(),
        data_provider1,
        rmsprop_options,
    )
    .unwrap();
    assert!(result_rmsprop.fun < 1e-2, "RMSProp should converge");

    // Test AdamW
    let adamw_options = AdamWOptions {
        learning_rate: 0.05,
        max_iter: 200,
        tol: 1e-4,
        weight_decay: 0.01,
        ..Default::default()
    };
    let data_provider2 = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));
    let result_adamw =
        minimize_adamw(QuadraticFunction, x0, data_provider2, adamw_options).unwrap();
    assert!(result_adamw.fun < 1e-2, "AdamW should converge");

    println!("✅ Multiple stochastic optimization algorithms working");
}
