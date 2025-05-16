use scirs2_sparse::csr::CsrMatrix;
use scirs2_sparse::linalg::{bicg, AsLinearOperator, BiCGOptions};

fn main() {
    // Create a non-symmetric matrix: [[2, -1], [1, 3]]
    let rows = vec![0, 0, 1, 1];
    let cols = vec![0, 1, 0, 1];
    let data = vec![2.0, -1.0, 1.0, 3.0];
    let matrix = CsrMatrix::new(data, rows, cols, (2, 2)).unwrap();

    // Use AsLinearOperator trait to convert to LinearOperator
    let op = matrix.as_linear_operator();

    // Right-hand side
    let b = vec![3.0, 2.0];

    // Solve using BiCG with different options
    let mut options = BiCGOptions::default();
    options.atol = 1e-8;
    options.rtol = 1e-8;
    options.max_iter = 10;

    println!("Solving Ax = b where:");
    println!("A = [[2, -1], [1, 3]]");
    println!("b = [3, 2]");
    println!("Expected solution: x = [11/7, 1/7] = [1.571428..., 0.142857...]");
    println!();

    let result = bicg(op.as_ref(), &b, options).unwrap();

    println!("BiCG result:");
    println!("  converged: {}", result.converged);
    println!("  iterations: {}", result.iterations);
    println!("  residual_norm: {}", result.residual_norm);
    println!("  solution: {:?}", result.x);

    // Check the solution by computing Ax
    let mut ax: Vec<f64> = vec![0.0; 2];
    for i in 0..2 {
        for j in 0..2 {
            ax[i] += matrix.get(i, j) * result.x[j];
        }
    }
    println!("  Ax = {:?} (should be [3, 2])", ax);

    // Compute residual
    let mut residual: Vec<f64> = vec![0.0; 2];
    for i in 0..2 {
        residual[i] = b[i] - ax[i];
    }
    let residual_norm: f64 = (residual[0].powi(2) + residual[1].powi(2)).sqrt();
    println!("  Computed residual norm: {}", residual_norm);
}
