use scirs2_sparse::{
    csr::CsrMatrix,
    linalg::{bicgstab, AsLinearOperator, BiCGSTABOptions, JacobiPreconditioner, LinearOperator},
};

#[test]
fn test_bicgstab_with_preconditioner() {
    // Create a non-symmetric sparse matrix
    let rows = vec![0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3];
    let cols = vec![0, 1, 0, 1, 2, 1, 2, 3, 0, 2, 3, 3];
    let data = vec![
        10.0, 1.0, 2.0, 12.0, 1.0, 1.0, 8.0, 2.0, 3.0, 1.0, 15.0, 1.0,
    ];
    let shape = (4, 4);

    let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
    let op = matrix.as_linear_operator();

    // Create RHS vector
    let b = vec![11.0, 15.0, 11.0, 20.0];

    // Create preconditioner
    let m = JacobiPreconditioner::new(&matrix).unwrap();

    // Setup BiCGSTAB with preconditioner
    let options = BiCGSTABOptions {
        max_iter: 100,
        rtol: 1e-6,
        atol: 1e-8,
        x0: None,
        left_preconditioner: Some(Box::new(m) as Box<dyn LinearOperator<f64>>),
        right_preconditioner: None,
    };

    // Solve
    let result = bicgstab(op.as_ref(), &b, options).unwrap();

    assert!(result.converged);
    assert!(result.iterations < 50); // Should converge quickly with preconditioner

    // Verify solution
    let residual_norm = {
        let ax = op.matvec(&result.x).unwrap();
        let residual: Vec<f64> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
        residual.iter().map(|&r| r * r).sum::<f64>().sqrt()
    };

    assert!(
        residual_norm < 1e-6,
        "Residual norm too large: {}",
        residual_norm
    );
}

#[test]
fn test_bicgstab_complex_system() {
    // Test BiCGSTAB on a more complex non-symmetric system
    // This matrix is based on a simple convection-diffusion problem
    let n = 10;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for i in 0..n {
        // Main diagonal
        rows.push(i);
        cols.push(i);
        data.push(4.0);

        // Off-diagonals
        if i > 0 {
            rows.push(i);
            cols.push(i - 1);
            data.push(-1.5); // Lower diagonal with convection

            rows.push(i - 1);
            cols.push(i);
            data.push(-0.5); // Upper diagonal with convection
        }
    }

    let matrix = CsrMatrix::new(data, rows, cols, (n, n)).unwrap();
    let op = matrix.as_linear_operator();

    // Create RHS vector
    let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();

    // Solve without preconditioner
    let options = BiCGSTABOptions {
        max_iter: 200,
        rtol: 1e-8,
        atol: 1e-10,
        x0: None,
        left_preconditioner: None,
        right_preconditioner: None,
    };

    let result = bicgstab(op.as_ref(), &b, options).unwrap();

    assert!(result.converged);

    // Verify the solution
    let residual_norm = {
        let ax = op.matvec(&result.x).unwrap();
        let residual: Vec<f64> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
        residual.iter().map(|&r| r * r).sum::<f64>().sqrt()
    };

    assert!(
        residual_norm < 5e-7,
        "Residual norm too large: {}",
        residual_norm
    );
}

#[test]
fn test_bicgstab_breakdown_detection() {
    // Test BiCGSTAB behavior when it encounters breakdown conditions
    // Create a singular matrix (not invertible)
    let rows = vec![0, 0, 1, 1];
    let cols = vec![0, 1, 0, 1];
    let data = vec![1.0, 1.0, 1.0, 1.0]; // All rows are the same
    let shape = (2, 2);

    let matrix = CsrMatrix::new(data, rows, cols, shape).unwrap();
    let op = matrix.as_linear_operator();

    // Create an inconsistent RHS
    let b = vec![1.0, 2.0];

    let options = BiCGSTABOptions::default();
    let result = bicgstab(op.as_ref(), &b, options).unwrap();

    // Should not converge on singular system
    assert!(!result.converged);
    assert!(result.message.contains("breakdown"));
}
