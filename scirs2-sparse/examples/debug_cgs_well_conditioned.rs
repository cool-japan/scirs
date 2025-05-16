use scirs2_sparse::csr::CsrMatrix;
use scirs2_sparse::linalg::{cgs, AsLinearOperator, CGSOptions};

fn main() {
    // Test CGS on a well-conditioned non-symmetric matrix
    // This matrix is diagonally dominant
    let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let data = vec![4.0, 0.5, 0.5, 0.5, 4.0, 0.5, 0.0, 0.5, 4.0];
    let shape = (3, 3);

    let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
    let op = matrix.as_linear_operator();

    println!("Matrix:");
    for i in 0..3 {
        for j in 0..3 {
            print!("{:8.3} ", matrix.get(i, j));
        }
        println!();
    }

    let b = vec![5.0, 5.0, 4.5];
    println!("\nRHS b: {:?}", b);

    let mut options = CGSOptions::default();
    options.rtol = 1e-8;
    options.atol = 1e-10;
    options.max_iter = 100;

    println!("\nTolerance: rtol={}, atol={}", options.rtol, options.atol);

    let result = cgs(op.as_ref(), &b, options).unwrap();

    println!("\n--- Result ---");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("Residual norm: {}", result.residual_norm);
    println!("Solution: {:?}", result.x);

    // Verify solution
    let ax = op.matvec(&result.x).unwrap();
    let residual: Vec<f64> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
    let residual_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();

    println!("\nActual residual: {:?}", residual);
    println!("Actual residual norm: {}", residual_norm);

    // For diagonal dominant matrices, expected solution can be approximated
    println!("\nApproximate expected solution (by inspection):");
    println!("[1.0, 1.0, 1.125]");
}
