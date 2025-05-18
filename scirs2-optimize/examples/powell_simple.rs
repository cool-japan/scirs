//! Simple Powell method example based on actual tests

use ndarray::{array, ArrayView1};
use scirs2_optimize::unconstrained::{minimize_powell, Bounds, Options};

fn main() {
    // Example from test_powell_simple
    println!("Powell's Method - Simple Quadratic");
    let quadratic = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };

    let x0 = array![1.0, 1.0];
    let options = Options::default();

    let result = minimize_powell(quadratic, x0, &options).unwrap();

    println!("Success: {}", result.success);
    println!("Solution: {:?}", result.x);
    println!("Function value: {}", result.fun);
    println!("Iterations: {}", result.nit);

    // Example from test_powell_rosenbrock
    println!("\nPowell's Method - Rosenbrock Function");
    let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
        let a = 1.0;
        let b = 100.0;
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    };

    let x0 = array![0.0, 0.0];
    let options = Options::default();

    let result = minimize_powell(rosenbrock, x0, &options).unwrap();

    println!("Success: {}", result.success);
    println!("Solution: {:?}", result.x);
    println!("Function value: {}", result.fun);
    println!("Iterations: {}", result.nit);

    // Example from test_powell_with_bounds
    println!("\nPowell's Method - With Bounds");
    let quadratic = |x: &ArrayView1<f64>| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) };

    let x0 = array![0.0, 0.0];
    let mut options = Options::default();

    // Constrain solution to [0, 1] x [0, 1]
    let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
    options.bounds = Some(bounds);

    let result = minimize_powell(quadratic, x0, &options).unwrap();

    println!("Success: {}", result.success);
    println!("Solution: {:?}", result.x);
    println!("Function value: {}", result.fun);
    println!("Iterations: {}", result.nit);
    println!("Note: Optimal point (2,3) is outside bounds, so result is (1,1)");
}
