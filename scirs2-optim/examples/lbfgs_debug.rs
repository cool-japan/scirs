//! Debug LBFGS optimizer behavior

use ndarray::Array1;
use scirs2_optim::optimizers::{Optimizer, LBFGS};

fn main() {
    let mut optimizer: LBFGS<f64> = LBFGS::new(0.01);

    // Minimize a simple quadratic function: f(x) = x^2
    let mut params = Array1::from_vec(vec![5.0]);

    println!("LBFGS optimizer debug - minimizing x^2");
    println!("Initial params: {:?}", params);

    for i in 0..10 {
        // Gradient of x^2 is 2x
        let gradients = Array1::from_vec(vec![2.0 * params[0]]);

        println!("Iteration {}: gradient = {:?}", i, gradients);

        params = optimizer.step(&params, &gradients).unwrap();

        println!("  new params = {:?}", params);

        if params[0].is_nan() {
            println!("  ERROR: NaN detected!");
            break;
        }
    }

    println!("Final params: {:?}", params);
}
