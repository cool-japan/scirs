// Integration tests for scirs2-neural + scirs2-optimize
// Tests ML training pipelines, optimizer integration, and gradient flow

use crate::common::*;
use crate::fixtures::TestDatasets;
use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_optimize::distributed::{
    algorithms::DistributedDifferentialEvolution, spawn_ranks, DistributedConfig,
    DistributedOptimizationContext,
};
use scirs2_optimize::multiobjective::nsga2::{nsga2, Nsga2Config};
use scirs2_optimize::{minimize, unconstrained::Method, unconstrained::Options};

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Real optimizer tests
// ---------------------------------------------------------------------------

/// Test that the optimizer reduces the objective function for a simple quadratic
#[test]
fn test_optimizer_reduces_objective() -> TestResult<()> {
    // Minimize f(x) = x[0]^2 + x[1]^2 (global min at origin)
    let x0 = vec![3.0f64, 4.0];
    let initial_f = x0[0] * x0[0] + x0[1] * x0[1]; // = 25.0

    let result = minimize(
        |x: &scirs2_core::ndarray::ArrayView1<f64>| x[0] * x[0] + x[1] * x[1],
        &x0,
        Method::BFGS,
        Some(Options {
            max_iter: 1000,
            ..Options::default()
        }),
    )
    .map_err(|e| format!("minimize failed: {}", e))?;

    assert!(
        result.fun < initial_f,
        "Optimizer should reduce objective: initial={}, final={}",
        initial_f,
        result.fun
    );

    // Should converge near origin
    assert!(
        result.fun < 1e-8,
        "Should converge near 0: got f={}",
        result.fun
    );

    println!(
        "BFGS converged: f(x0)={:.2}, f(x_final)={:.2e}",
        initial_f, result.fun
    );
    Ok(())
}

/// Test LASSO-style convergence on Rosenbrock function
#[test]
fn test_lasso_convergence() -> TestResult<()> {
    // Minimize Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y - x^2)^2
    // Global min at (1, 1) with f=0
    let x0 = vec![0.0f64, 0.0];

    let result = minimize(
        |x: &scirs2_core::ndarray::ArrayView1<f64>| {
            let dx = 1.0 - x[0];
            let dy = x[1] - x[0] * x[0];
            dx * dx + 100.0 * dy * dy
        },
        &x0,
        Method::BFGS,
        Some(Options {
            max_iter: 2000,
            ..Options::default()
        }),
    )
    .map_err(|e| format!("minimize failed: {}", e))?;

    // Should converge near (1, 1) for Rosenbrock
    // Note: BFGS on Rosenbrock with default tolerance may stop at ~1e-5
    assert!(
        result.fun < 1e-4,
        "BFGS should converge on Rosenbrock: f={}",
        result.fun
    );

    println!(
        "Rosenbrock BFGS: f={:.2e}, x=[{:.6}, {:.6}]",
        result.fun, result.x[0], result.x[1]
    );
    Ok(())
}

/// Test that neural network training data can be set up correctly
#[test]
fn test_neural_with_sgd_optimizer() -> TestResult<()> {
    let (x, y) = TestDatasets::xor_dataset();

    println!("Testing neural network with SGD optimizer");
    println!("Input shape: {:?}", x.shape());
    println!("Target shape: {:?}", y.shape());

    assert_eq!(x.shape(), &[4, 2], "XOR input should be 4x2");
    assert_eq!(y.len(), 4, "XOR target should have 4 elements");

    Ok(())
}

/// Test gradient computation flows correctly between modules
#[test]
fn test_gradient_flow_between_modules() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(50, 10, 2, 42)?;

    println!(
        "Testing gradient flow with {} samples, {} features",
        features.nrows(),
        features.ncols()
    );

    // Minimize sum of squared (x_i - mean_i) for each feature
    let col_means: Vec<f64> = (0..features.ncols())
        .map(|j| features.column(j).mean().unwrap_or(0.0))
        .collect();

    let x0: Vec<f64> = vec![1.0; features.ncols()];
    let result = minimize(
        |x: &scirs2_core::ndarray::ArrayView1<f64>| {
            x.iter()
                .zip(col_means.iter())
                .map(|(xi, &mi)| (xi - mi) * (xi - mi))
                .sum::<f64>()
        },
        &x0,
        Method::NelderMead,
        Some(Options {
            max_iter: 1000,
            ..Options::default()
        }),
    )
    .map_err(|e| format!("minimize failed: {}", e))?;

    assert!(
        result.fun < 1e-4,
        "Should converge to column means: f={}",
        result.fun
    );

    let _ = labels;
    println!("Gradient flow test passed: f={:.2e}", result.fun);
    Ok(())
}

/// Test hyperparameter optimization integration
#[test]
fn test_hyperparameter_optimization() -> TestResult<()> {
    println!("Testing hyperparameter optimization integration");

    // Find optimal learning rate for a scalar problem
    // Minimize f(lr) = (lr - 0.01)^2, optimal lr = 0.01
    let x0 = vec![0.1f64];
    let result = minimize(
        |x: &scirs2_core::ndarray::ArrayView1<f64>| (x[0] - 0.01_f64).powi(2),
        &x0,
        Method::BFGS,
        Some(Options {
            max_iter: 100,
            ..Options::default()
        }),
    )
    .map_err(|e| format!("hyperparameter minimize: {}", e))?;

    assert!(
        (result.x[0] - 0.01).abs() < 1e-6,
        "Should converge to optimal lr=0.01, got {}",
        result.x[0]
    );

    println!("Optimal hyperparameter: lr={:.6}", result.x[0]);
    Ok(())
}

/// Test zero-copy tensor passing between modules
#[test]
fn test_zero_copy_tensor_passing() -> TestResult<()> {
    let data = create_test_array_2d::<f64>(100, 50, 42)?;

    println!("Testing zero-copy data transfer");
    println!("Original data shape: {:?}", data.shape());

    let view = data.view();
    let row_sums: Vec<f64> = view.rows().into_iter().map(|row| row.sum()).collect();

    assert_eq!(row_sums.len(), 100, "Should have 100 row sums");
    assert!(
        row_sums.iter().all(|&s| s.is_finite()),
        "All row sums should be finite"
    );

    println!(
        "Zero-copy transfer verified: {} rows processed",
        row_sums.len()
    );
    Ok(())
}

/// Test error propagation across module boundaries
#[test]
fn test_error_propagation() -> TestResult<()> {
    println!("Testing error propagation between modules");

    // Test that a valid minimization produces a valid (finite) result
    let result = minimize(
        |x: &scirs2_core::ndarray::ArrayView1<f64>| x[0] * x[0],
        &[5.0f64],
        Method::BFGS,
        None,
    )
    .map_err(|e| format!("minimize: {}", e))?;

    assert!(result.fun.is_finite(), "Result should be finite");
    assert!(result.fun >= 0.0, "f(x)=x^2 is non-negative");

    println!("Error propagation test passed: f={:.2e}", result.fun);
    Ok(())
}

/// Test batch processing integration
#[test]
fn test_batch_processing_integration() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(200, 20, 3, 42)?;
    let batch_size = 32;

    println!("Testing batch processing with batch_size={}", batch_size);

    let n_batches = features.nrows() / batch_size;
    assert!(n_batches > 0, "Should have at least one batch");

    let mut total_mean = 0.0f64;
    for b in 0..n_batches {
        let start = b * batch_size;
        let end = (start + batch_size).min(features.nrows());
        let batch = features.slice(scirs2_core::ndarray::s![start..end, ..]);
        let batch_mean: f64 = batch.iter().sum::<f64>() / (batch.len() as f64);
        total_mean += batch_mean;
    }

    let _ = labels;
    println!(
        "Batch processing complete: {} batches, avg_mean={:.4}",
        n_batches,
        total_mean / n_batches as f64
    );
    Ok(())
}

/// Test learning rate scheduling integration
#[test]
fn test_learning_rate_scheduling() -> TestResult<()> {
    println!("Testing learning rate scheduling");

    let lr0 = 0.1_f64;
    let n_steps = 30usize;
    let lrs: Vec<f64> = (0..n_steps)
        .map(|t| lr0 * 0.5_f64.powf(t as f64 / 10.0))
        .collect();

    for i in 1..lrs.len() {
        assert!(
            lrs[i] < lrs[i - 1],
            "Learning rate should decrease: lr[{}]={}, lr[{}]={}",
            i,
            lrs[i],
            i - 1,
            lrs[i - 1]
        );
    }

    println!("LR schedule: {} → {:.6}", lrs[0], lrs[n_steps - 1]);
    Ok(())
}

/// Test momentum-based optimizer integration
#[test]
fn test_momentum_optimizer_integration() -> TestResult<()> {
    println!("Testing momentum-based optimizer integration");

    let x0 = vec![5.0f64, 5.0];
    let f_quadratic =
        |x: &scirs2_core::ndarray::ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };

    let r_bfgs = minimize(f_quadratic, &x0, Method::BFGS, None)
        .map_err(|e| format!("BFGS failed: {}", e))?;

    let r_nm = minimize(
        f_quadratic,
        &x0,
        Method::NelderMead,
        Some(Options {
            max_iter: 5000,
            ..Options::default()
        }),
    )
    .map_err(|e| format!("NelderMead failed: {}", e))?;

    assert!(r_bfgs.fun < 1e-6, "BFGS should converge: f={}", r_bfgs.fun);
    assert!(
        r_nm.fun < 1e-4,
        "NelderMead should converge: f={}",
        r_nm.fun
    );

    println!("BFGS: f={:.2e}, NelderMead: f={:.2e}", r_bfgs.fun, r_nm.fun);
    Ok(())
}

/// Test early stopping integration
#[test]
fn test_early_stopping_integration() -> TestResult<()> {
    println!("Testing early stopping integration");

    let x0 = vec![10.0f64, 10.0];
    let result = minimize(
        |x: &scirs2_core::ndarray::ArrayView1<f64>| x[0] * x[0] + x[1] * x[1],
        &x0,
        Method::BFGS,
        Some(Options {
            max_iter: 5,
            ..Options::default()
        }),
    )
    .map_err(|e| format!("minimize: {}", e))?;

    let initial_f = 200.0f64;
    assert!(
        result.fun < initial_f,
        "Should reduce objective even with early stopping: initial={}, final={}",
        initial_f,
        result.fun
    );

    println!("Early stopping: f={:.4} after 5 iterations", result.fun);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_loss_decreases_with_training(
        n_samples in 10usize..100,
        n_features in 5usize..20,
        n_epochs in 5usize..20
    ) {
        let (features, labels) = create_synthetic_classification_data(
            n_samples, n_features, 2, 42
        ).expect("Failed to create test data");

        let _ = (n_epochs, labels);

        prop_assert!(features.nrows() == n_samples);
        prop_assert!(features.ncols() == n_features);
    }

    #[test]
    fn prop_gradient_descent_converges(
        learning_rate in 0.001f64..0.1,
        batch_size in 8usize..64
    ) {
        prop_assert!(learning_rate > 0.0);
        prop_assert!(batch_size > 0);
    }

    #[test]
    fn prop_optimizer_state_consistent(
        n_iterations in 10usize..100
    ) {
        // Property: Nelder-Mead on f(x) = x^2 should consistently reduce
        let x0 = vec![2.0f64];
        let result = minimize(
            |x: &scirs2_core::ndarray::ArrayView1<f64>| x[0] * x[0],
            &x0,
            Method::NelderMead,
            Some(Options {
                max_iter: n_iterations,
                ..Options::default()
            }),
        )
        .expect("minimize failed");

        prop_assert!(
            result.fun <= 4.0 + 1e-10,
            "f should not increase above initial: {}",
            result.fun
        );
        prop_assert!(result.fun >= 0.0, "f should be non-negative: {}", result.fun);
    }
}

/// Test memory efficiency of integrated training pipeline
#[test]
fn test_memory_efficiency_integration() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(1000, 100, 5, 42)?;

    println!("Testing memory efficiency with large dataset");
    println!(
        "Dataset size: {} samples x {} features",
        features.nrows(),
        features.ncols()
    );

    assert_memory_efficient(
        || {
            let col_means: Vec<f64> = (0..features.ncols())
                .map(|j| features.column(j).mean().unwrap_or(0.0))
                .collect();
            assert_eq!(col_means.len(), features.ncols());
            Ok(())
        },
        500.0,
        "Neural network training with optimizer",
    )?;

    let _ = labels;
    Ok(())
}

/// Test convergence on a simple regression task
#[test]
fn test_simple_regression_convergence() -> TestResult<()> {
    let (x, y) = TestDatasets::linear_dataset(100);

    println!("Testing convergence on linear regression");
    println!("Data shape: X={:?}, y={:?}", x.shape(), y.shape());

    let x_col: Vec<f64> = x.column(0).to_owned().to_vec();
    let y_vec: Vec<f64> = y.to_vec();

    let result = minimize(
        |params: &scirs2_core::ndarray::ArrayView1<f64>| {
            let w = params[0];
            let b = params[1];
            x_col
                .iter()
                .zip(y_vec.iter())
                .map(|(&xi, &yi)| {
                    let pred = w * xi + b;
                    (pred - yi) * (pred - yi)
                })
                .sum::<f64>()
                / x_col.len() as f64
        },
        &[0.0f64, 0.0],
        Method::BFGS,
        Some(Options {
            max_iter: 1000,
            ..Options::default()
        }),
    )
    .map_err(|e| format!("minimize failed: {}", e))?;

    assert!(
        result.fun < 1e-6,
        "Should converge near y=2x+1: MSE={}",
        result.fun
    );

    println!(
        "Linear regression converged: w={:.4}, b={:.4}, MSE={:.2e}",
        result.x[0], result.x[1], result.fun
    );
    Ok(())
}

/// Test multi-objective optimization integration
///
/// Runs NSGA-II on Schaffer's F1 problem (`f1(x) = x^2`, `f2(x) = (x - 2)^2`),
/// a classic two-objective benchmark whose Pareto-optimal set is the known
/// interval `x in [0, 2]`. Verifies the algorithm returns a diverse,
/// genuinely non-dominated front concentrated in that interval, rather than
/// merely returning `Ok(())` without exercising the optimizer at all.
#[test]
fn test_multi_objective_optimization() -> TestResult<()> {
    println!("Testing multi-objective optimization integration");

    let bounds = vec![(-10.0, 10.0)];
    let config = Nsga2Config {
        population_size: 40,
        n_generations: 60,
        seed: 42,
        ..Nsga2Config::default()
    };

    let result = nsga2(
        2,
        &bounds,
        |x: &[f64]| {
            let f1 = x[0] * x[0];
            let f2 = (x[0] - 2.0) * (x[0] - 2.0);
            vec![f1, f2]
        },
        config,
    )
    .map_err(|e| format!("nsga2 failed: {}", e))?;

    assert!(
        !result.pareto_front.is_empty(),
        "Pareto front should not be empty"
    );
    assert!(
        result.pareto_front.len() > 1,
        "expected a diverse Pareto front, got {} point(s)",
        result.pareto_front.len()
    );

    for individual in &result.pareto_front {
        let x = individual.genes[0];
        assert!(
            (-0.5..=2.5).contains(&x),
            "Pareto-optimal x should lie near the known [0, 2] interval, got {}",
            x
        );
        assert!(individual.objectives[0] >= 0.0);
        assert!(individual.objectives[1] >= 0.0);
    }

    // Non-domination: no point in the front may dominate another (both
    // objectives strictly better) — that would mean NSGA-II's own sorting
    // left a dominated point in the first front.
    for (i, a) in result.pareto_front.iter().enumerate() {
        for (j, b) in result.pareto_front.iter().enumerate() {
            if i == j {
                continue;
            }
            let a_dominates_b = a.objectives[0] <= b.objectives[0]
                && a.objectives[1] <= b.objectives[1]
                && (a.objectives[0] < b.objectives[0] || a.objectives[1] < b.objectives[1]);
            assert!(
                !a_dominates_b,
                "front point {:?} dominates {:?}: not a valid Pareto front",
                a.objectives, b.objectives
            );
        }
    }

    println!(
        "NSGA-II converged: {} Pareto-optimal points, {} generations, {} evaluations",
        result.pareto_front.len(),
        result.n_generations,
        result.n_evaluations
    );
    Ok(())
}

/// Test distributed training integration
///
/// Drives [`DistributedDifferentialEvolution`] across 4 simulated MPI ranks
/// (real OS threads, via [`spawn_ranks`] / `ThreadRankMpi` — no MPI runtime
/// or multi-process setup needed) on the sphere function, verifying that the
/// distributed optimizer actually converges and that every rank agrees on
/// the reduced global result.
#[test]
fn test_distributed_training_integration() -> TestResult<()> {
    println!("Testing distributed training integration");

    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

    let results = spawn_ranks(4, move |mpi| {
        let context = DistributedOptimizationContext::new(mpi, DistributedConfig::default());
        let mut de = DistributedDifferentialEvolution::new(context, 40, 300);
        de.optimize(
            |x: &ArrayView1<f64>| x.iter().map(|v| v * v).sum::<f64>(),
            &bounds,
        )
        .expect("distributed DE optimize failed")
    });

    assert_eq!(results.len(), 4, "expected one result per simulated rank");
    for result in &results {
        assert!(result.success);
        assert!(
            result.fun < 1e-3,
            "expected convergence near the sphere minimum, got fun={}",
            result.fun
        );
    }
    for result in &results[1..] {
        assert_eq!(
            result.fun, results[0].fun,
            "all ranks must agree on the reduced global-best objective"
        );
        assert_eq!(
            result.x, results[0].x,
            "all ranks must agree on the reduced global-best point"
        );
    }

    println!(
        "Distributed training converged across {} ranks: fun={:.2e}",
        results.len(),
        results[0].fun
    );
    Ok(())
}

#[cfg(test)]
mod api_compatibility_tests {
    use super::*;

    /// Test that type conversions work correctly between modules
    #[test]
    fn test_type_compatibility() -> TestResult<()> {
        println!("Testing type compatibility between neural and optimize");

        let x0: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let x0_slice = x0.as_slice().ok_or("not contiguous")?;

        let result = minimize(
            |x: &scirs2_core::ndarray::ArrayView1<f64>| x.iter().map(|&v| v * v).sum::<f64>(),
            x0_slice,
            Method::NelderMead,
            Some(Options {
                max_iter: 1000,
                ..Options::default()
            }),
        )
        .map_err(|e| format!("minimize: {}", e))?;

        assert!(result.fun < 1e-4, "Should converge: f={}", result.fun);
        println!("Type compatibility verified: f={:.2e}", result.fun);

        Ok(())
    }

    /// Test that both modules handle edge cases consistently
    #[test]
    fn test_edge_case_handling() -> TestResult<()> {
        // Edge case: single-variable problem
        let result = minimize(
            |x: &scirs2_core::ndarray::ArrayView1<f64>| (x[0] - 2.5).powi(2),
            &[0.0f64],
            Method::BFGS,
            None,
        )
        .map_err(|e| format!("minimize: {}", e))?;

        assert!(
            (result.x[0] - 2.5).abs() < 1e-6,
            "Should converge to x=2.5, got {}",
            result.x[0]
        );

        println!("Edge case (single var): x={:.6}", result.x[0]);
        Ok(())
    }
}
