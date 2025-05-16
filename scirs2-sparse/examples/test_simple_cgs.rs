use scirs2_sparse::csr::CsrMatrix;
use scirs2_sparse::linalg::AsLinearOperator;
use scirs2_sparse::linalg::{cgs, CGSOptions};

fn main() {
    // Test with a simple well-conditioned non-symmetric matrix
    let rows = vec![0, 0, 1, 1, 2, 2];
    let cols = vec![0, 1, 0, 1, 0, 1];
    let data = vec![2.0, 0.5, 0.5, 2.0, 0.0, 2.0]; // Well-conditioned
    let shape = (3, 3);

    let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
    let op = matrix.as_linear_operator();

    let b = vec![2.5, 2.5, 2.0];

    let mut options = CGSOptions::default();
    options.atol = 1e-6;
    options.rtol = 1e-6;
    options.max_iter = 100;

    println!("Matrix:");
    for i in 0..3 {
        for j in 0..3 {
            print!("{:8.3} ", matrix.get(i, j));
        }
        println!();
    }
    println!("\nRHS b: {:?}", b);

    let result = cgs(op.as_ref(), &b, options).unwrap();

    println!("\n--- Result ---");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("Residual norm: {}", result.residual_norm);
    println!("Solution: {:?}", result.x);

    // Check residual
    let ax = op.matvec(&result.x).unwrap();
    let residual: Vec<f64> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
    println!("Actual residual: {:?}", residual);
    let rnorm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
    println!("Actual residual norm: {}", rnorm);
}
