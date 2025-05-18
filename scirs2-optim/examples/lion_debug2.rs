//! Debug Lion optimizer behavior with smaller start

use ndarray::Array1;
use scirs2_optim::optimizers::{Lion, Optimizer};

fn main() {
    let mut optimizer: Lion<f64> = Lion::new(0.01);

    // Minimize a simple quadratic function: f(x) = x^2
    let mut params = Array1::from_vec(vec![5.0]);

    println!("Lion optimizer debug - minimizing x^2");
    println!("Initial params: {:?}", params);

    for i in 0..100 {
        // Gradient of x^2 is 2x
        let gradients = Array1::from_vec(vec![2.0 * params[0]]);
        params = optimizer.step(&params, &gradients).unwrap();

        if i % 10 == 0 || i < 10 {
            println!(
                "Iteration {}: params = {:.6}, gradient = {:.6}",
                i,
                params[0],
                2.0 * params[0]
            );
        }
    }

    println!("Final params: {:.6}", params[0]);
    println!("Final absolute value: {:.6}", params[0].abs());
}
