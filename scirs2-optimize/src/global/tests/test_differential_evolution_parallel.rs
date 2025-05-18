//! Tests for differential evolution with parallel computation

use crate::differential_evolution::{differential_evolution, DifferentialEvolutionOptions};
use crate::parallel::ParallelOptions;
use ndarray::{array, ArrayView1};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// Test function that tracks the number of evaluations
fn test_function_with_counter(counter: Arc<AtomicUsize>) -> impl Fn(&ArrayView1<f64>) -> f64 {
    move |x: &ArrayView1<f64>| {
        counter.fetch_add(1, Ordering::Relaxed);
        // Simple quadratic function with minimum at (1.0, 2.0)
        let x0 = x[0];
        let x1 = x[1];
        (x0 - 1.0).powi(2) + (x1 - 2.0).powi(2)
    }
}

#[test]
fn test_parallel_vs_sequential() {
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
    
    // Test sequential execution
    let sequential_counter = Arc::new(AtomicUsize::new(0));
    let sequential_func = test_function_with_counter(sequential_counter.clone());
    
    let mut sequential_opts = DifferentialEvolutionOptions::default();
    sequential_opts.popsize = 10;
    sequential_opts.maxiter = 5;
    sequential_opts.seed = Some(42);
    sequential_opts.parallel = None;
    
    let sequential_result = differential_evolution(
        sequential_func,
        bounds.clone(),
        Some(sequential_opts),
        None,
    ).unwrap();
    
    let sequential_evals = sequential_counter.load(Ordering::Relaxed);
    
    // Test parallel execution
    let parallel_counter = Arc::new(AtomicUsize::new(0));
    let parallel_func = test_function_with_counter(parallel_counter.clone());
    
    let mut parallel_opts = DifferentialEvolutionOptions::default();
    parallel_opts.popsize = 10;
    parallel_opts.maxiter = 5;
    parallel_opts.seed = Some(42);
    parallel_opts.parallel = Some(ParallelOptions {
        num_workers: Some(2), // Use 2 workers for consistency
        min_parallel_size: 4,
        chunk_size: 1,
        parallel_evaluations: true,
        parallel_gradient: true,
    });
    
    let parallel_result = differential_evolution(
        parallel_func,
        bounds,
        Some(parallel_opts),
        None,
    ).unwrap();
    
    let parallel_evals = parallel_counter.load(Ordering::Relaxed);
    
    // Both should produce similar results
    assert!((parallel_result.fun - sequential_result.fun).abs() < 1e-6);
    assert!((parallel_result.x[0] - sequential_result.x[0]).abs() < 1e-2);
    assert!((parallel_result.x[1] - sequential_result.x[1]).abs() < 1e-2);
    
    // Both should use the same number of function evaluations
    assert_eq!(parallel_evals, sequential_evals);
}

#[test]
fn test_parallel_correctness() {
    // Test that parallel execution produces correct results
    let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
    
    // Rosenbrock function with minimum at (1, 1)
    let rosenbrock = |x: &ArrayView1<f64>| {
        let x0 = x[0];
        let x1 = x[1];
        (1.0 - x0).powi(2) + 100.0 * (x1 - x0.powi(2)).powi(2)
    };
    
    let mut options = DifferentialEvolutionOptions::default();
    options.popsize = 20;
    options.maxiter = 50;
    options.seed = Some(123);
    options.parallel = Some(ParallelOptions {
        num_workers: None, // Use all available cores
        min_parallel_size: 5,
        chunk_size: 1,
        parallel_evaluations: true,
        parallel_gradient: true,
    });
    
    let result = differential_evolution(
        rosenbrock,
        bounds,
        Some(options),
        None,
    ).unwrap();
    
    // Check that we found the minimum
    assert!(result.fun < 0.01); // Should be very close to 0
    assert!((result.x[0] - 1.0).abs() < 0.1);
    assert!((result.x[1] - 1.0).abs() < 0.1);
}

#[test]
fn test_parallel_options_disabled() {
    // Test that parallel options can be disabled
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
    
    let simple_func = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
    
    let mut options = DifferentialEvolutionOptions::default();
    options.popsize = 8;
    options.maxiter = 10;
    options.seed = Some(456);
    options.parallel = Some(ParallelOptions {
        num_workers: Some(4),
        min_parallel_size: 100, // Set very high to disable parallel
        chunk_size: 1,
        parallel_evaluations: false, // Explicitly disable
        parallel_gradient: false,
    });
    
    let result = differential_evolution(
        simple_func,
        bounds,
        Some(options),
        None,
    ).unwrap();
    
    // Should still find minimum at (0, 0)
    assert!(result.fun < 0.01);
    assert!(result.x[0].abs() < 0.1);
    assert!(result.x[1].abs() < 0.1);
}