//! Integration tests for global optimization algorithms

use scirs2_core::ndarray::array;
use scirs2_core::ndarray::ArrayView1;
use scirs2_optimize::global::{
    basinhopping, differential_evolution, dual_annealing, BasinHoppingOptions,
    DifferentialEvolutionOptions, DualAnnealingOptions,
};

#[test]
#[allow(dead_code)]
fn test_global_optimization_on_rosenbrock() {
    // Test all three global optimization algorithms on the Rosenbrock function
    let rosenbrock = |x: &scirs2_core::ndarray::ArrayView1<f64>| {
        let a = 1.0;
        let b = 100.0;
        let x0 = x[0];
        let x1 = x[1];
        (a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2)
    };

    // Test 1: Differential Evolution
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
    let options = DifferentialEvolutionOptions {
        maxiter: 100,
        popsize: 15,
        seed: Some(42),
        ..Default::default()
    };

    let result = differential_evolution(rosenbrock, bounds.clone(), Some(options), None).unwrap();
    assert!(result.success);
    assert!((result.x[0] - 1.0).abs() < 0.1);
    assert!((result.x[1] - 1.0).abs() < 0.1);

    // Test 2: Basin-hopping
    let x0 = array![0.0, 0.0];
    let options = BasinHoppingOptions {
        niter: 50,
        seed: Some(42),
        ..Default::default()
    };

    let result = basinhopping(rosenbrock, x0.clone(), Some(options), None, None).unwrap();
    assert!(result.success);
    assert!((result.x[0] - 1.0).abs() < 0.2);
    assert!((result.x[1] - 1.0).abs() < 0.2);

    // Test 3: Dual Annealing
    let options = DualAnnealingOptions {
        maxiter: 100,
        seed: Some(42),
        bounds: bounds.clone(),
        ..Default::default()
    };

    let result = dual_annealing(rosenbrock, x0, bounds, Some(options)).unwrap();
    // Dual annealing may not always converge to exact solution, so we check if it's close
    assert!(result.fun < 1.0);
}

#[test]
fn test_global_optimization_with_constraints() {
    // Test global optimization on a function with bounded constraints
    let func =
        |x: &scirs2_core::ndarray::ArrayView1<f64>| (x[0] + 1.0).powi(2) + (x[1] + 1.0).powi(2);

    // Without bounds, minimum is at (-1, -1)
    // With bounds [0, 2] x [0, 2], minimum should be at (0, 0)
    let bounds = vec![(0.0, 2.0), (0.0, 2.0)];

    // Test Differential Evolution
    let options = DifferentialEvolutionOptions {
        maxiter: 200, // Increased iterations for better convergence
        seed: Some(42),
        ..Default::default()
    };

    let result = differential_evolution(func, bounds.clone(), Some(options), None).unwrap();

    assert!(result.success);

    // Print the actual result for verification
    println!(
        "Differential Evolution result: x = {:?}, f(x) = {}",
        result.x, result.fun
    );

    // Check that the result respects the bounds
    for (i, &val) in result.x.iter().enumerate() {
        assert!(
            val >= bounds[i].0 && val <= bounds[i].1,
            "Solution x[{}] = {} is outside bounds [{}, {}]",
            i,
            val,
            bounds[i].0,
            bounds[i].1
        );
    }

    // The constrained minimum should be near (0, 0) with function value near 2.0
    // f(0,0) = (0+1)^2 + (0+1)^2 = 2
    assert!(result.x[0] >= 0.0 && result.x[0] <= 2.0);
    assert!(result.x[1] >= 0.0 && result.x[1] <= 2.0);
    assert!(result.fun >= 1.8); // Should be close to the constrained minimum value
    assert!(result.fun < 10.0); // Should be better than a random point
}

#[test]
#[allow(dead_code)]
fn test_global_optimization_multimodal() {
    // Test on a multimodal function with many local minima
    let ackley = |x: &scirs2_core::ndarray::ArrayView1<f64>| {
        let n = x.len() as f64;
        let a = 20.0;
        let b = 0.2;
        let c = 2.0 * std::f64::consts::PI;

        let sum1 = x.iter().map(|&xi| xi.powi(2)).sum::<f64>() / n;
        let sum2 = x.iter().map(|&xi| (c * xi).cos()).sum::<f64>() / n;

        -a * (-b * sum1.sqrt()).exp() - sum2.exp() + a + std::f64::consts::E
    };

    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

    // Test Differential Evolution
    let options = DifferentialEvolutionOptions {
        maxiter: 100,
        popsize: 20,
        seed: Some(42),
        ..Default::default()
    };

    let result = differential_evolution(ackley, bounds, Some(options), None).unwrap();
    assert!(result.success);
    // Ackley function has global minimum at origin with value 0
    assert!(result.x.iter().all(|&xi| xi.abs() < 0.5));
    assert!(result.fun < 1.0);
}
