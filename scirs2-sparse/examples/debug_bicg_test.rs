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

    println!("Matrix:");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:8.3} ", matrix.get(i, j));
        }
        println!();
    }

    // Right-hand side
    let b = vec![3.0, 2.0];
    println!("\nRHS b: {:?}", b);

    // Solve using BiCG
    let mut options = BiCGOptions::default();
    options.atol = 1e-8;
    options.rtol = 1e-8;
    options.max_iter = 100;

    println!("\nStarting BiCG with options:");
    println!(
        "atol = {}, rtol = {}, max_iter = {}",
        options.atol, options.rtol, options.max_iter
    );

    let result = bicg(op.as_ref(), &b, options).unwrap();

    // Check convergence
    println!(
        "\nBiCG result: converged={}, iterations={}, residual_norm={}",
        result.converged, result.iterations, result.residual_norm
    );
    println!("BiCG solution: {:?}", result.x);

    // Check actual residual
    let residual = {
        let ax = op.matvec(&result.x).unwrap();
        let r: Vec<f64> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
        println!("\nAx = {:?}", ax);
        println!("residual = b - Ax = {:?}", r);
        r.iter().map(|&x| x * x).sum::<f64>().sqrt()
    };

    println!("Actual residual norm: {}", residual);

    // Expected solution
    // Solving: 2x - y = 3, x + 3y = 2
    // Solution: x = 11/7, y = 1/7
    println!("\nExpected solution: x = {}, y = {}", 11.0 / 7.0, 1.0 / 7.0);
    println!(
        "Difference from expected: dx = {}, dy = {}",
        result.x[0] - 11.0 / 7.0,
        result.x[1] - 1.0 / 7.0
    );

    // Let's also test with a different initial guess
    println!("\n--- Testing with initial guess ---");
    let x0 = vec![1.0, 1.0];
    let mut options2 = BiCGOptions::default();
    options2.atol = 1e-8;
    options2.rtol = 1e-8;
    options2.max_iter = 100;
    options2.x0 = Some(x0);

    let result2 = bicg(op.as_ref(), &b, options2).unwrap();
    println!(
        "BiCG result2: converged={}, iterations={}, residual_norm={}",
        result2.converged, result2.iterations, result2.residual_norm
    );
    println!("BiCG solution2: {:?}", result2.x);
}
