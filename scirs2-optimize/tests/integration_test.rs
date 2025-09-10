//! Integration tests for scirs2-optimize
//!
//! These tests validate the key functionality of the optimization library
//! across different algorithm categories.

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use scirs2_optimize::{
    stochastic::{
        minimize_adam, minimize_sgd, AdamOptions, DataProvider, InMemoryDataProvider, SGDOptions,
        StochasticGradientFunction,
    },
    unconstrained::{minimize_bfgs, Options as UnconstrainedOptions},
};

/// Simple quadratic function for testing
struct QuadraticFunction;

impl StochasticGradientFunction for QuadraticFunction {
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batchdata: &[f64]) -> Array1<f64> {
        x.mapv(|xi| 2.0 * xi)
    }

    fn compute_value(&mut self, x: &ArrayView1<f64>, _batchdata: &[f64]) -> f64 {
        x.mapv(|xi| xi * xi).sum()
    }
}

#[test]
#[allow(dead_code)]
fn test_stochastic_optimization_integration() {
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

    let result = minimize_sgd(grad_func, x0.clone(), data_provider, options);
    assert!(result.is_ok());
    let result = result.unwrap();

    // Should converge toward zero
    assert!(result.fun < 1e-2);
    println!(
        "SGD converged to f = {:.2e} in {} iterations",
        result.fun, result.nit
    );
}

#[test]
#[allow(dead_code)]
fn test_adam_optimization_integration() {
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

    let result = minimize_adam(grad_func, x0, data_provider, options);
    assert!(result.is_ok());
    let result = result.unwrap();

    // Adam should converge efficiently
    assert!(result.fun < 1e-3);
    println!(
        "Adam converged to f = {:.2e} in {} iterations",
        result.fun, result.nit
    );
}

#[test]
#[allow(dead_code)]
fn test_bfgs_optimization_integration() {
    // Test BFGS on a simple function
    let func = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };

    let x0 = Array1::from_vec(vec![3.0, -2.0]);
    let options = UnconstrainedOptions::default();

    let result = minimize_bfgs(func, x0, &options);
    assert!(result.is_ok());
    let result = result.unwrap();

    // BFGS should converge quickly for quadratic functions
    assert!(result.success);
    assert!(result.fun < 1e-6);
    println!(
        "BFGS converged to f = {:.2e} in {} iterations",
        result.fun, result.nit
    );
}

#[test]
#[allow(dead_code)]
fn test_optimization_library_capabilities() {
    println!("\n🔬 scirs2-optimize Library Capabilities Test");
    println!("============================================");

    // Test that we can create different optimizers
    let _sgd_options = SGDOptions::default();
    let _adam_options = AdamOptions::default();
    let _bfgs_options = UnconstrainedOptions::default();

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
    println!("✅ All core library capabilities working correctly!");
}

#[test]
#[allow(dead_code)]
fn test_comprehensive_optimization_workflows() {
    println!("\n🔧 Comprehensive Optimization Workflows Test");
    println!("===========================================");

    // Test 1: Basic unconstrained optimization workflow
    println!("Testing unconstrained optimization workflow...");
    test_unconstrained_workflow();

    // Test 2: Stochastic optimization workflow
    println!("Testing stochastic optimization workflow...");
    test_stochastic_workflow();

    // Test 3: Different problem types
    println!("Testing various problem types...");
    test_problem_types();

    println!("✅ All optimization workflows working correctly!");
}

#[allow(dead_code)]
fn test_unconstrained_workflow() {
    use scirs2_optimize::unconstrained::{minimize_lbfgs, minimize_powell};

    // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
        let x1 = x[0];
        let x2 = x[1];
        (1.0 - x1).powi(2) + 100.0 * (x2 - x1 * x1).powi(2)
    };

    let x0 = Array1::from_vec(vec![-1.2, 1.0]);
    let options = UnconstrainedOptions {
        max_iter: 1000,
        gtol: 1e-6,
        ftol: 1e-12,
        ..Default::default()
    };

    // Test L-BFGS
    let result_lbfgs = minimize_lbfgs(rosenbrock, x0.clone(), &options).unwrap();
    assert!(result_lbfgs.success, "L-BFGS failed to converge");
    assert!(
        result_lbfgs.fun < 1e-1,
        "L-BFGS didn't reach target accuracy (got {:.2e})",
        result_lbfgs.fun
    );

    // Test Powell's method (derivative-free)
    let result_powell = minimize_powell(rosenbrock, x0.clone(), &options).unwrap();
    assert!(
        result_powell.fun < 1e-3,
        "Powell's method didn't reach reasonable accuracy"
    );

    println!("  ✅ Unconstrained optimization algorithms working");
}

#[allow(dead_code)]
fn test_stochastic_workflow() {
    use scirs2_optimize::stochastic::{
        minimize_adamw, minimize_rmsprop, AdamWOptions, RMSPropOptions,
    };

    // Test with a noisy quadratic function (simulating ML scenario)
    struct NoisyQuadratic;
    impl StochasticGradientFunction for NoisyQuadratic {
        fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batchdata: &[f64]) -> Array1<f64> {
            // Add small noise to simulate stochastic gradients
            let mut rng = rand::rng();
            x.mapv(|xi| 2.0 * xi + rng.random_range(-0.01..0.01))
        }

        fn compute_value(&mut self, x: &ArrayView1<f64>, _batchdata: &[f64]) -> f64 {
            x.mapv(|xi| xi * xi).sum()
        }
    }

    let x0 = Array1::from_vec(vec![2.0, -1.5, 0.5]);
    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 200]));

    // Test RMSProp
    let rmsprop_options = RMSPropOptions {
        learning_rate: 0.01,
        max_iter: 500,
        tol: 1e-4,
        ..Default::default()
    };
    let result_rmsprop =
        minimize_rmsprop(NoisyQuadratic, x0.clone(), data_provider, rmsprop_options).unwrap();
    assert!(
        result_rmsprop.fun < 1e-2,
        "RMSProp didn't converge adequately"
    );

    // Test AdamW
    let data_provider2 = Box::new(InMemoryDataProvider::new(vec![1.0; 200]));
    let adamw_options = AdamWOptions {
        learning_rate: 0.01,
        max_iter: 500,
        weight_decay: 0.001,
        tol: 1e-4,
        ..Default::default()
    };
    let result_adamw = minimize_adamw(NoisyQuadratic, x0, data_provider2, adamw_options).unwrap();
    assert!(result_adamw.fun < 1e-2, "AdamW didn't converge adequately");

    println!("  ✅ Stochastic optimization algorithms working");
}

#[allow(dead_code)]
fn test_problem_types() {
    // Test different problem characteristics

    // 1. Ill-conditioned quadratic
    let ill_conditioned = |x: &ArrayView1<f64>| -> f64 {
        1000.0 * x[0] * x[0] + x[1] * x[1] // Condition number = 1000
    };

    let x0 = Array1::from_vec(vec![1.0, 1.0]);
    let options = UnconstrainedOptions {
        max_iter: 500,
        gtol: 1e-6,
        ..Default::default()
    };

    let result = minimize_bfgs(ill_conditioned, x0.clone(), &options).unwrap();
    assert!(
        result.fun < 1e-3,
        "Failed on ill-conditioned problem (got {:.2e})",
        result.fun
    );

    // 2. High-dimensional problem
    let high_dim_size = 50;
    let high_dim_quad = |x: &ArrayView1<f64>| -> f64 { x.mapv(|xi| xi * xi).sum() };

    let x0_high_dim = Array1::ones(high_dim_size);
    let result_high_dim = minimize_bfgs(high_dim_quad, x0_high_dim, &options).unwrap();
    assert!(
        result_high_dim.success,
        "Failed on high-dimensional problem"
    );
    assert!(
        result_high_dim.fun < 1e-6,
        "High-dimensional problem didn't reach target accuracy"
    );

    println!("  ✅ Various problem types handled correctly");
}

#[test]
#[allow(dead_code)]
fn test_algorithm_robustness() {
    println!("\n🛡️  Testing Algorithm Robustness");
    println!("===============================");

    // Test algorithms on challenging problems
    test_challenging_problems();
    test_edge_cases();

    println!("✅ All robustness tests passed!");
}

#[allow(dead_code)]
fn test_challenging_problems() {
    use scirs2_optimize::unconstrained::minimize_nelder_mead;

    // Himmelblau's function (multiple global minima)
    let himmelblau = |x: &ArrayView1<f64>| -> f64 {
        let x1 = x[0];
        let x2 = x[1];
        (x1 * x1 + x2 - 11.0).powi(2) + (x1 + x2 * x2 - 7.0).powi(2)
    };

    let x0 = Array1::from_vec(vec![0.0, 0.0]);
    let options = UnconstrainedOptions {
        max_iter: 1000,
        ftol: 1e-6,
        ..Default::default()
    };

    // Test with derivative-free method
    let result = minimize_nelder_mead(himmelblau, x0, &options).unwrap();
    assert!(
        result.fun < 1e-3,
        "Failed to find good minimum for Himmelblau's function"
    );

    println!("  ✅ Challenging multi-modal problems handled");
}

#[allow(dead_code)]
fn test_edge_cases() {
    // Test with very small problems
    let simple_1d = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] };
    let x0_1d = Array1::from_vec(vec![5.0]);
    let options = UnconstrainedOptions {
        max_iter: 100,
        gtol: 1e-8,
        ..Default::default()
    };

    let result_1d = minimize_bfgs(simple_1d, x0_1d, &options).unwrap();
    assert!(result_1d.success, "Failed on 1D problem");
    assert!(
        (result_1d.x[0]).abs() < 1e-6,
        "1D minimum not found accurately"
    );

    // Test with starting point at optimum
    let x0_optimal = Array1::from_vec(vec![0.0, 0.0]);
    let quad_2d = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };
    let result_optimal =
        minimize_bfgs(quad_2d, x0_optimal, &UnconstrainedOptions::default()).unwrap();
    assert!(result_optimal.success, "Failed when starting at optimum");

    println!("  ✅ Edge cases handled correctly");
}

#[test]
#[allow(dead_code)]
fn test_stochastic_integration_comprehensive() {
    println!("\n📊 Comprehensive Stochastic Integration Test");
    println!("==========================================");

    // Test complete stochastic optimization pipeline
    test_machine_learning_workflow();
    test_large_scale_optimization();

    println!("✅ Comprehensive stochastic integration completed!");
}

#[allow(dead_code)]
fn test_machine_learning_workflow() {
    use scirs2_optimize::stochastic::{minimize_sgd_momentum, MomentumOptions};

    // Simulate a simple logistic regression scenario
    struct LogisticRegression {
        features: Vec<Vec<f64>>,
        labels: Vec<f64>,
    }

    impl LogisticRegression {
        fn new() -> Self {
            // Simple 2D dataset with linear separability
            let features = vec![
                vec![1.0, 2.0],
                vec![2.0, 3.0],
                vec![3.0, 4.0], // Class 1
                vec![-1.0, -2.0],
                vec![-2.0, -3.0],
                vec![-3.0, -4.0], // Class 0
            ];
            let labels = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
            Self { features, labels }
        }
    }

    impl StochasticGradientFunction for LogisticRegression {
        fn compute_gradient(
            &mut self,
            params: &ArrayView1<f64>,
            batch_indices: &[f64],
        ) -> Array1<f64> {
            let mut grad = Array1::zeros(params.len());

            for &idx in batch_indices {
                let i = idx as usize % self.features.len();
                let x = &self.features[i];
                let y = self.labels[i];

                let z = params[0] * x[0] + params[1] * x[1] + params[2]; // w1*x1 + w2*x2 + bias
                let pred = 1.0 / (1.0 + (-z).exp());
                let error = pred - y;

                grad[0] += error * x[0];
                grad[1] += error * x[1];
                grad[2] += error; // bias gradient
            }

            grad / batch_indices.len() as f64
        }

        fn compute_value(&mut self, params: &ArrayView1<f64>, batchindices: &[f64]) -> f64 {
            let mut loss = 0.0;

            for &idx in batchindices {
                let i = idx as usize % self.features.len();
                let x = &self.features[i];
                let y = self.labels[i];

                let z = params[0] * x[0] + params[1] * x[1] + params[2];
                let pred = 1.0 / (1.0 + (-z).exp());

                // Cross-entropy loss
                loss += -y * pred.ln() - (1.0 - y) * (1.0 - pred).ln();
            }

            loss / batchindices.len() as f64
        }
    }

    let logreg = LogisticRegression::new();
    let x0 = Array1::zeros(3); // [w1, w2, bias]
    let data_provider = Box::new(InMemoryDataProvider::new(
        (0..6).map(|i| i as f64).collect(),
    ));

    let options = MomentumOptions {
        learning_rate: 0.1,
        momentum: 0.9,
        max_iter: 200,
        tol: 1e-4,
        batch_size: Some(3),
        ..Default::default()
    };

    let result = minimize_sgd_momentum(logreg, x0, data_provider, options).unwrap();
    assert!(
        result.fun < 1.0,
        "Logistic regression didn't converge to reasonable loss"
    );

    println!("  ✅ Machine learning workflow tested successfully");
}

#[allow(dead_code)]
fn test_large_scale_optimization() {
    // Test with larger problem size to verify scalability
    struct LargeQuadratic {
        #[allow(dead_code)]
        dimension: usize,
    }

    impl LargeQuadratic {
        fn new(dim: usize) -> Self {
            Self { dimension: dim }
        }
    }

    impl StochasticGradientFunction for LargeQuadratic {
        fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batchdata: &[f64]) -> Array1<f64> {
            // Scaled quadratic: gradient = 2x with different scales per dimension
            x.iter()
                .enumerate()
                .map(|(i, &xi)| {
                    let scale = 1.0 + i as f64 * 0.1; // Increasing scale
                    2.0 * scale * xi
                })
                .collect()
        }

        fn compute_value(&mut self, x: &ArrayView1<f64>, _batchdata: &[f64]) -> f64 {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| {
                    let scale = 1.0 + i as f64 * 0.1;
                    scale * xi * xi
                })
                .sum()
        }
    }

    let large_dim = 100;
    let large_quad = LargeQuadratic::new(large_dim);
    let x0_large = Array1::ones(large_dim);
    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 500]));

    let adam_options = AdamOptions {
        learning_rate: 0.01,
        max_iter: 300,
        tol: 1e-4,
        batch_size: Some(50),
        ..Default::default()
    };

    let result_large = minimize_adam(large_quad, x0_large, data_provider, adam_options).unwrap();
    assert!(result_large.fun < 1e-2, "Large-scale optimization failed");

    println!(
        "  ✅ Large-scale optimization (dim={}) successful",
        large_dim
    );
}
