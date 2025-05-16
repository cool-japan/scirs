use scirs2_sparse::linalg::{cgs, CGSOptions, DiagonalOperator};

fn main() {
    // Simple diagonal matrix test
    let diag = DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
    let b = vec![2.0, 6.0, 12.0];

    let mut options = CGSOptions::default();
    options.atol = 1e-10;
    options.rtol = 1e-8;
    options.max_iter = 20;

    println!("Matrix A:");
    println!("[2.0  0.0  0.0]");
    println!("[0.0  3.0  0.0]");
    println!("[0.0  0.0  4.0]");
    println!("\nRHS b: {:?}", b);
    println!("Expected solution: [1.0, 2.0, 3.0]");

    let result = cgs(&diag, &b, options).unwrap();

    println!("\n--- Result ---");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("Residual norm: {}", result.residual_norm);
    println!("Solution: {:?}", result.x);

    // Check residual
    let mut residual = vec![0.0; 3];
    for i in 0..3 {
        residual[i] = b[i] - diag.diagonal()[i] * result.x[i];
    }
    println!("Actual residual: {:?}", residual);

    // Calculate actual residual norm
    let rnorm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
    println!("Actual residual norm: {}", rnorm);
}
