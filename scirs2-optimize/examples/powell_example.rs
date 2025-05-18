//! Example of using Powell's optimization method
//!
//! This example demonstrates Powell's method for unconstrained and
//! constrained optimization problems.

use ndarray::{array, ArrayView1};
use scirs2_optimize::unconstrained::{minimize_powell, Bounds, Options};

fn main() {
    println!("Powell's Method Examples");
    println!("========================\n");

    // Example 1: Simple quadratic function
    println!("Example 1: Simple quadratic function");
    println!("Minimize: f(x) = x₁² + x₂²");

    let quadratic = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
    let x0 = array![1.0, 1.0];

    let result = minimize_powell(quadratic, x0.clone(), &Options::default()).unwrap();

    println!("Starting point: {:?}", x0);
    println!("Solution: {:?}", result.x);
    println!("Minimum value: {:.6}", result.fun);
    println!("Iterations: {}", result.nit);
    println!("Success: {}\n", result.success);

    // Example 2: Rosenbrock function
    println!("Example 2: Rosenbrock function");
    println!("Minimize: f(x) = (1-x₁)² + 100(x₂-x₁²)²");

    let rosenbrock = |x: &ArrayView1<f64>| {
        let a = 1.0;
        let b = 100.0;
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    };

    let x0 = array![0.0, 0.0];
    let result = minimize_powell(rosenbrock, x0.clone(), &Options::default()).unwrap();

    println!("Starting point: {:?}", x0);
    println!("Solution: {:?}", result.x);
    println!("Minimum value: {:.6}", result.fun);
    println!("Iterations: {}", result.nit);
    println!("Success: {}\n", result.success);

    // Example 3: Constrained optimization with bounds
    println!("Example 3: Quadratic with bounds");
    println!("Minimize: f(x) = (x₁-2)² + (x₂-3)²");
    println!("Subject to: 0 ≤ x₁ ≤ 1, 0 ≤ x₂ ≤ 1");

    let quadratic_shifted = |x: &ArrayView1<f64>| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);

    let x0 = array![0.0, 0.0];
    let mut options = Options::default();

    // Set bounds: [0, 1] x [0, 1]
    let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
    options.bounds = Some(bounds);

    let result = minimize_powell(quadratic_shifted, x0.clone(), &options).unwrap();

    println!("Starting point: {:?}", x0);
    println!("Solution: {:?}", result.x);
    println!("Minimum value: {:.6}", result.fun);
    println!("Iterations: {}", result.nit);
    println!("Success: {}\n", result.success);

    println!("Note: Since the optimal point (2,3) is outside the bounds,");
    println!("the algorithm finds the closest point within bounds: (1,1)");

    // Example 4: More complex function
    println!("\nExample 4: Himmelblau function");
    println!("Minimize: f(x) = (x₁²+x₂-11)² + (x₁+x₂²-7)²");

    let himmelblau = |x: &ArrayView1<f64>| {
        (x[0].powi(2) + x[1] - 11.0).powi(2) + (x[0] + x[1].powi(2) - 7.0).powi(2)
    };

    // Try different starting points
    let starting_points = vec![array![0.0, 0.0], array![1.0, 1.0], array![-1.0, -1.0]];

    for (i, x0) in starting_points.iter().enumerate() {
        let result = minimize_powell(himmelblau, x0.clone(), &Options::default()).unwrap();

        println!("\nStarting point {}: {:?}", i + 1, x0);
        println!("Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
        println!("Minimum value: {:.8}", result.fun);
        println!("Iterations: {}", result.nit);
    }

    println!("\nNote: Himmelblau function has 4 local minima:");
    println!("(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)");
    println!("The algorithm finds different minima depending on starting point.");
}
