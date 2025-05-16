use scirs2_sparse::linalg::{cgs, CGSOptions, DiagonalOperator, LinearOperator};

fn main() {
    // Simple diagonal matrix
    let diag = DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
    let b = vec![2.0, 6.0, 12.0];

    let mut options = CGSOptions::default();
    options.max_iter = 50; // More iterations to converge

    let result = cgs(&diag, &b, options).unwrap();

    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("Residual norm: {}", result.residual_norm);
    println!("Solution: {:?}", result.x);

    // Expected solution: [1.0, 2.0, 3.0]
    println!("\nExpected: [1.0, 2.0, 3.0]");

    // Let's check Ax = b
    let ax = diag.matvec(&result.x).unwrap();
    println!("\nA*x = {:?}", ax);
    println!("b   = {:?}", b);
}
