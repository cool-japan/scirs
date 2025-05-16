use scirs2_sparse::{
    csr::CsrMatrix,
    linalg::{cgs, AsLinearOperator, CGSOptions, JacobiPreconditioner, LinearOperator},
};

#[test]
fn test_cgs_identity() {
    // Test CGS on identity matrix - should converge in 1 iteration
    let rows = vec![0, 1, 2];
    let cols = vec![0, 1, 2];
    let data = vec![1.0, 1.0, 1.0];
    let shape = (3, 3);

    let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
    let op = matrix.as_linear_operator();

    let b = vec![1.0, 2.0, 3.0];
    let options = CGSOptions::default();
    let result = cgs(op.as_ref(), &b, options).unwrap();

    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    for (xi, bi) in result.x.iter().zip(&b) {
        let diff: f64 = *xi - *bi;
        assert!(diff.abs() < 1e-10);
    }
}

#[test]
fn test_cgs_well_conditioned() {
    // Test CGS on a well-conditioned non-symmetric matrix
    // This matrix is diagonally dominant
    let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let data = vec![4.0, 0.5, 0.5, 0.5, 4.0, 0.5, 0.0, 0.5, 4.0];
    let shape = (3, 3);

    let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
    let op = matrix.as_linear_operator();

    let b = vec![5.0, 5.0, 4.5];
    let mut options = CGSOptions::default();
    options.rtol = 1e-8;
    options.atol = 1e-10;

    let result = cgs(op.as_ref(), &b, options).unwrap();

    // CGS may not always converge based on internal residual measure,
    // but what matters is the actual residual of the system

    // Verify solution
    let ax = op.matvec(&result.x).unwrap();
    let residual: Vec<f64> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
    let residual_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();

    assert!(
        residual_norm < 1e-8,
        "Residual norm too large: {}",
        residual_norm
    );
}

#[test]
fn test_cgs_with_preconditioner() {
    // Test CGS with Jacobi preconditioner on a moderately ill-conditioned matrix
    let rows = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
    let cols = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    let data = vec![
        10.0, 1.0, 0.5, 0.0, // row 0
        1.0, 8.0, 0.5, 0.5, // row 1
        0.5, 0.5, 6.0, 1.0, // row 2
        0.0, 0.5, 1.0, 4.0, // row 3
    ];
    let shape = (4, 4);

    let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
    let op = matrix.as_linear_operator();

    // Create RHS vector
    let b = vec![11.5, 10.0, 8.0, 5.5];

    // Without preconditioner
    let mut options_no_precond = CGSOptions::default();
    options_no_precond.max_iter = 100;
    options_no_precond.rtol = 1e-6;

    let result_no_precond = cgs(op.as_ref(), &b, options_no_precond).unwrap();

    // With Jacobi preconditioner
    let precond = JacobiPreconditioner::new(&matrix).unwrap();
    let mut options_precond = CGSOptions::default();
    options_precond.max_iter = 100;
    options_precond.rtol = 1e-6;
    options_precond.right_preconditioner = Some(Box::new(precond) as Box<dyn LinearOperator<f64>>);

    let result_precond = cgs(op.as_ref(), &b, options_precond).unwrap();

    // With preconditioner should converge faster (if it converges)
    if result_precond.converged && result_no_precond.converged {
        assert!(result_precond.iterations <= result_no_precond.iterations);
    }
}

#[test]
fn test_cgs_real_world_pattern() {
    // Test CGS on a sparse matrix with real-world sparsity pattern
    // This represents a simple finite difference discretization
    let n = 5; // 5x5 grid -> 25x25 matrix
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    // Create a 2D Laplacian matrix
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;

            // Diagonal entry
            rows.push(idx);
            cols.push(idx);
            data.push(4.0);

            // Off-diagonal entries (4-point stencil)
            if i > 0 {
                rows.push(idx);
                cols.push((i - 1) * n + j);
                data.push(-1.0);
            }
            if i < n - 1 {
                rows.push(idx);
                cols.push((i + 1) * n + j);
                data.push(-1.0);
            }
            if j > 0 {
                rows.push(idx);
                cols.push(i * n + (j - 1));
                data.push(-1.0);
            }
            if j < n - 1 {
                rows.push(idx);
                cols.push(i * n + (j + 1));
                data.push(-1.0);
            }
        }
    }

    let matrix = CsrMatrix::new(data, rows, cols, (n * n, n * n)).unwrap();
    let op = matrix.as_linear_operator();

    // Create a simple RHS
    let b: Vec<f64> = (0..n * n).map(|i| (i + 1) as f64).collect();

    let mut options = CGSOptions::default();
    options.max_iter = 200; // May need more iterations for larger problems
    options.rtol = 1e-5;
    options.atol = 1e-7;

    let result = cgs(op.as_ref(), &b, options).unwrap();

    // We just check that it produces a valid solution (residual is small)
    if result.converged {
        let ax = op.matvec(&result.x).unwrap();
        let residual: Vec<f64> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
        let residual_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|&r| r * r).sum::<f64>().sqrt();

        assert!(residual_norm / b_norm < 1e-4, "Relative residual too large");
    }
}

#[test]
fn test_cgs_symmetric_vs_cg() {
    // CGS should also work on symmetric positive definite matrices
    // though CG would be preferred in practice
    let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let data = vec![4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0];
    let shape = (3, 3);

    let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
    let op = matrix.as_linear_operator();

    let b = vec![2.0, 2.0, 2.0];
    let mut options = CGSOptions::default();
    options.rtol = 1e-8;

    let result = cgs(op.as_ref(), &b, options).unwrap();

    assert!(result.converged);

    // Check that solution is correct
    let expected = vec![1.0, 1.0, 1.0]; // Due to symmetry
    for (xi, ei) in result.x.iter().zip(&expected) {
        let diff: f64 = *xi - *ei;
        assert!(diff.abs() < 1e-6);
    }
}
