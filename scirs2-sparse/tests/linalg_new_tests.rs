use approx::assert_relative_eq;
use scirs2_sparse::csr::CsrMatrix;
use scirs2_sparse::linalg::{bicg, cg, expm_multiply, BiCGOptions, CGOptions};

#[test]
fn test_cg_solver() {
    // Create a simple positive definite matrix: [[2, -1], [-1, 2]]
    let rows = vec![0, 0, 1, 1];
    let cols = vec![0, 1, 0, 1];
    let data = vec![2.0, -1.0, -1.0, 2.0];
    let matrix = CsrMatrix::new(data, rows, cols, (2, 2)).unwrap();

    // Use AsLinearOperator trait to convert to LinearOperator
    use scirs2_sparse::linalg::AsLinearOperator;
    let op = matrix.as_linear_operator();

    // Right-hand side
    let b = vec![1.0, 0.0];

    // Solve using CG
    let options = CGOptions::default();
    let result = cg(op.as_ref(), &b, options).unwrap();

    // Check convergence
    assert!(result.converged);

    // Check solution
    // Solving: 2x - y = 1, -x + 2y = 0
    // Solution should be [2/3, 1/3]
    assert_relative_eq!(result.x[0], 2.0 / 3.0, epsilon = 1e-6);
    assert_relative_eq!(result.x[1], 1.0 / 3.0, epsilon = 1e-6);
}

#[test]
fn test_bicg_solver() {
    // Create a non-symmetric matrix: [[2, -1], [1, 3]]
    let rows = vec![0, 0, 1, 1];
    let cols = vec![0, 1, 0, 1];
    let data = vec![2.0, -1.0, 1.0, 3.0];
    let matrix = CsrMatrix::new(data, rows, cols, (2, 2)).unwrap();

    // Use AsLinearOperator trait to convert to LinearOperator
    use scirs2_sparse::linalg::AsLinearOperator;
    let op = matrix.as_linear_operator();

    // Right-hand side
    let b = vec![3.0, 2.0];

    // Solve using BiCG
    let mut options = BiCGOptions::default();
    options.atol = 1e-8;
    options.rtol = 1e-8;
    let result = bicg(op.as_ref(), &b, options).unwrap();

    // Check convergence (for some small problems, BiCG may struggle)
    println!(
        "BiCG result: converged={}, iterations={}, residual_norm={}",
        result.converged, result.iterations, result.residual_norm
    );
    println!("BiCG solution: {:?}", result.x);

    // Check actual residual even if BiCG didn't converge
    let residual = {
        let ax = op.matvec(&result.x).unwrap();
        let r: Vec<f64> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
        r.iter().map(|&x| x * x).sum::<f64>().sqrt()
    };

    assert!(residual < 1e-5, "Residual too large: {}", residual);

    // Check solution
    // Solving: 2x - y = 3, x + 3y = 2
    // Solution: x = 11/7, y = 1/7
    assert_relative_eq!(result.x[0], 11.0 / 7.0, epsilon = 1e-5);
    assert_relative_eq!(result.x[1], 1.0 / 7.0, epsilon = 1e-5);
}

#[test]
fn test_expm_multiply() {
    // Create a dense matrix: [[1, 0], [0, -1]]
    let rows = vec![0, 1];
    let cols = vec![0, 1];
    let data = vec![1.0, -1.0];
    let matrix = CsrMatrix::new(data, rows, cols, (2, 2)).unwrap();

    // Use AsLinearOperator trait to convert to LinearOperator
    use scirs2_sparse::linalg::AsLinearOperator;
    let op = matrix.as_linear_operator();

    // Vector to multiply - use a different vector to avoid degeneracy
    let v = vec![1.0, 0.0];

    // Compute exp(A) * v
    let result = expm_multiply(op.as_ref(), &v, 1.0, None, None).unwrap();

    // Debug output
    println!("expm_multiply result: {:?}", result);

    // For diagonal matrix, exp(diag([a, b])) = diag([exp(a), exp(b)])
    // So exp(diag([1, -1])) = diag([e, 1/e])
    // And exp(A) * [1, 0] = [e, 0]
    let e = std::f64::consts::E;
    assert_relative_eq!(result[0], e, epsilon = 1e-5);
    assert_relative_eq!(result[1], 0.0, epsilon = 1e-5);
}
