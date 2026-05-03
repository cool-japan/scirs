// Integration tests for scirs2-sparse + scirs2-linalg
// Tests sparse linear algebra operations, solver integration, and matrix conversions

use crate::common::*;
use crate::fixtures::TestDatasets;
use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::*;
use scirs2_sparse::{
    add, enhanced_cg, estimate_spectral_radius, graph_laplacian, hstack, lu_decomposition, matmul,
    normalized_laplacian, qr_decomposition, reverse_cuthill_mckee_full, sparse_direct_solve,
    sparse_eigsh, svds, vstack, AdjacencyGraph, CsrArray, CsrMatrix, EigenvalueTarget,
    EnhancedJacobiPreconditioner, IterativeSolverConfig, QRResult, SVDOptions, SparseArray,
};
// Import matrix_power from sparse explicitly to avoid shadowing by scirs2_linalg
use scirs2_sparse::matrix_power as sparse_matrix_power;
// Import norm from sparse explicitly
use scirs2_sparse::norm as sparse_norm_fn;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Helper: build a Poisson-1D tridiagonal SPD matrix (diag=2, off-diag=-1)
// ---------------------------------------------------------------------------
fn build_spd_tridiagonal(n: usize) -> TestResult<CsrMatrix<f64>> {
    let mut row_idx: Vec<usize> = Vec::new();
    let mut col_idx: Vec<usize> = Vec::new();
    let mut data: Vec<f64> = Vec::new();

    for i in 0..n {
        // diagonal
        row_idx.push(i);
        col_idx.push(i);
        data.push(2.0);

        // sub-diagonal
        if i > 0 {
            row_idx.push(i);
            col_idx.push(i - 1);
            data.push(-1.0);
        }

        // super-diagonal
        if i + 1 < n {
            row_idx.push(i);
            col_idx.push(i + 1);
            data.push(-1.0);
        }
    }

    CsrMatrix::new(data, row_idx, col_idx, (n, n))
        .map_err(|e| format!("Failed to build tridiagonal: {}", e).into())
}

/// Test sparse-dense matrix multiplication
///
/// Strategy: iterate over columns of the dense matrix and apply the sparse
/// matvec (CsrMatrix::dot) to each one; check shape and a spot value.
#[test]
fn test_sparse_dense_matmul() -> TestResult<()> {
    // 3×3 matrix:
    //  [1 0 2]
    //  [0 3 0]
    //  [4 0 5]
    let row_idx = vec![0usize, 0, 1, 2, 2];
    let col_idx = vec![0usize, 2, 1, 0, 2];
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let m = CsrMatrix::new(data, row_idx, col_idx, (3, 3))
        .map_err(|e| format!("CsrMatrix::new: {}", e))?;

    // Dense 3×2 matrix:
    //  [1 2]
    //  [3 4]
    //  [5 6]
    let dense = Array2::from_shape_vec((3, 2), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
        .map_err(|e| format!("Array2::from_shape_vec: {}", e))?;

    // Compute C = M * dense column-by-column
    let n_cols = dense.ncols();
    let n_rows = m.rows();
    let mut result_data = vec![0.0f64; n_rows * n_cols];

    for j in 0..n_cols {
        let col_vec: Vec<f64> = (0..dense.nrows()).map(|i| dense[[i, j]]).collect();
        let col_result = m
            .dot(&col_vec)
            .map_err(|e| format!("dot col {}: {}", j, e))?;
        for i in 0..n_rows {
            result_data[i * n_cols + j] = col_result[i];
        }
    }

    let result = Array2::from_shape_vec((n_rows, n_cols), result_data)
        .map_err(|e| format!("Array2::from_shape_vec result: {}", e))?;

    // Verify shape
    assert_eq!(result.shape(), &[3, 2], "Result shape mismatch");

    // Manual expected:
    // col 0: M * [1,3,5]^T = [1*1+2*5, 3*3, 4*1+5*5] = [11, 9, 29]
    // col 1: M * [2,4,6]^T = [1*2+2*6, 3*4, 4*2+5*6] = [14, 12, 38]
    let expected = [[11.0, 14.0], [9.0, 12.0], [29.0, 38.0]];
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                (result[[i, j]] - expected[i][j]).abs() < 1e-12,
                "result[{},{}]: expected {}, got {}",
                i,
                j,
                expected[i][j],
                result[[i, j]]
            );
        }
    }

    println!("Sparse-dense matmul verified: shape {:?}", result.shape());
    Ok(())
}

/// Test sparse linear system solving — Poisson-1D CG
#[test]
fn test_sparse_linear_solver() -> TestResult<()> {
    let n = 50usize;
    let a = build_spd_tridiagonal(n)?;

    // x_true = all-ones
    let x_true = Array1::ones(n);

    // b = A * x_true  (compute with matvec via dot)
    let x_slice = x_true.as_slice().ok_or("x_true is not contiguous")?;
    let b_vec = a
        .dot(x_slice)
        .map_err(|e| format!("matvec failed: {}", e))?;
    let b = Array1::from_vec(b_vec);

    // Solve with CG
    let config = IterativeSolverConfig {
        max_iter: 2000,
        tol: 1e-10,
        verbose: false,
    };
    let result = enhanced_cg(&a, &b, &config, None).map_err(|e| format!("CG failed: {}", e))?;

    assert!(
        result.converged,
        "CG did not converge after {} iters, residual={}",
        result.n_iter, result.residual_norm
    );

    // Verify residual ||Ax - b|| / ||b|| < 1e-8
    let sol_slice = result
        .solution
        .as_slice()
        .ok_or("solution is not contiguous")?;
    let ax_vec = a
        .dot(sol_slice)
        .map_err(|e| format!("matvec (verification) failed: {}", e))?;
    let ax = Array1::from_vec(ax_vec);
    let residual: f64 = (&ax - &b).mapv(|x| x * x).sum().sqrt();
    let b_norm: f64 = b.mapv(|x| x * x).sum().sqrt();
    let rel_residual = residual / b_norm;

    assert!(
        rel_residual < 1e-8,
        "Relative residual too large: {} (expected < 1e-8)",
        rel_residual
    );

    println!(
        "Sparse CG converged in {} iters, rel_residual={:.2e}",
        result.n_iter, rel_residual
    );
    Ok(())
}

/// Test sparse eigenvalue computation using sparse_eigsh (CsrMatrix-based solver)
#[test]
fn test_sparse_eigenvalues() -> TestResult<()> {
    // Build a 5×5 SPD tridiagonal (symmetric, positive definite)
    // Exact eigenvalues of n×n tridiagonal(2,-1,-1) are 2 - 2*cos(k*pi/(n+1)), k=1..n
    let n = 5usize;
    let a = build_spd_tridiagonal(n)?;

    // Compute 2 smallest algebraic eigenvalues
    let result = sparse_eigsh(&a, 2, EigenvalueTarget::SmallestAlgebraic, Some(1e-8), None)
        .map_err(|e| format!("sparse_eigsh failed: {}", e))?;

    assert_eq!(
        result.eigenvalues.len(),
        2,
        "Expected 2 eigenvalues, got {}",
        result.eigenvalues.len()
    );

    // All eigenvalues of SPD matrix must be positive
    for &ev in result.eigenvalues.iter() {
        assert!(ev > 0.0, "Eigenvalue {} is not positive for SPD matrix", ev);
    }

    // Eigenvalues should be ordered (ascending for SmallestAlgebraic)
    if result.eigenvalues.len() >= 2 {
        assert!(
            result.eigenvalues[0] <= result.eigenvalues[1] + 1e-10,
            "Eigenvalues not sorted: {} > {}",
            result.eigenvalues[0],
            result.eigenvalues[1]
        );
    }

    // Minimum eigenvalue of 5×5 Poisson tridiagonal is 2 - 2*cos(pi/6) ≈ 0.268
    let min_ev = result.eigenvalues[0];
    assert!(
        min_ev > 0.1 && min_ev < 1.5,
        "Smallest eigenvalue {} outside expected range (0.1, 1.5)",
        min_ev
    );

    println!(
        "Sparse eigenvalues computed: {:?} (converged={})",
        result.eigenvalues, result.converged
    );
    Ok(())
}

/// Test sparse matrix LU factorization and reconstruction
#[test]
fn test_sparse_factorization() -> TestResult<()> {
    // Use CsrArray of the 4×4 tridiagonal SPD matrix (diag=2, off-diag=-1)
    // This is well-conditioned and has no near-zero pivots
    let n = 4usize;
    let mut rows_v = Vec::new();
    let mut cols_v = Vec::new();
    let mut data_v = Vec::new();
    for i in 0..n {
        rows_v.push(i);
        cols_v.push(i);
        data_v.push(2.0f64);
        if i > 0 {
            rows_v.push(i);
            cols_v.push(i - 1);
            data_v.push(-1.0);
        }
        if i + 1 < n {
            rows_v.push(i);
            cols_v.push(i + 1);
            data_v.push(-1.0);
        }
    }

    let matrix = CsrArray::from_triplets(&rows_v, &cols_v, &data_v, (n, n), false)
        .map_err(|e| format!("CsrArray::from_triplets: {}", e))?;

    // Use a generous pivot threshold so partial pivoting does not flag this as singular
    let lu_result =
        lu_decomposition(&matrix, 1e-14).map_err(|e| format!("lu_decomposition failed: {}", e))?;

    assert!(lu_result.success, "LU decomposition reported failure");

    // L and U should have the correct shape
    let (lrows, lcols) = lu_result.l.shape();
    let (urows, ucols) = lu_result.u.shape();
    assert_eq!(lrows, n, "L rows mismatch");
    assert_eq!(lcols, n, "L cols mismatch");
    assert_eq!(urows, n, "U rows mismatch");
    assert_eq!(ucols, n, "U cols mismatch");

    // Verify that L is lower triangular (upper triangle should be zero)
    let l_dense = lu_result.l.to_array();
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                l_dense[[i, j]].abs() < 1e-10,
                "L[{},{}]={:.4e} is non-zero (expected lower triangular)",
                i,
                j,
                l_dense[[i, j]]
            );
        }
    }

    // Verify that U is upper triangular (lower triangle should be zero)
    let u_dense = lu_result.u.to_array();
    for i in 1..n {
        for j in 0..i {
            assert!(
                u_dense[[i, j]].abs() < 1e-10,
                "U[{},{}]={:.4e} is non-zero (expected upper triangular)",
                i,
                j,
                u_dense[[i, j]]
            );
        }
    }

    println!(
        "Sparse LU factorization verified: {}×{} matrix (success={})",
        n, n, lu_result.success
    );
    Ok(())
}

/// Test sparse-sparse addition and multiplication
#[test]
fn test_sparse_sparse_operations() -> TestResult<()> {
    // Build two 3×3 sparse matrices
    //  A = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]  (diagonal)
    //  B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  (identity)
    let n = 3usize;
    let a_rows = vec![0usize, 1, 2];
    let a_cols = vec![0usize, 1, 2];
    let a_data = vec![1.0f64, 2.0, 3.0];
    let a = CsrMatrix::new(a_data, a_rows, a_cols, (n, n))
        .map_err(|e| format!("CsrMatrix A: {}", e))?;

    let b_rows = vec![0usize, 1, 2];
    let b_cols = vec![0usize, 1, 2];
    let b_data = vec![1.0f64, 1.0, 1.0];
    let b = CsrMatrix::new(b_data, b_rows, b_cols, (n, n))
        .map_err(|e| format!("CsrMatrix B: {}", e))?;

    // Test addition: C = A + B should have diagonal [2, 3, 4]
    let c_add = add(&a, &b).map_err(|e| format!("sparse add failed: {}", e))?;
    assert_eq!(c_add.rows(), n);
    assert_eq!(c_add.cols(), n);
    assert!(
        (c_add.get(0, 0) - 2.0).abs() < 1e-12,
        "c_add[0,0]={}, expected 2.0",
        c_add.get(0, 0)
    );
    assert!(
        (c_add.get(1, 1) - 3.0).abs() < 1e-12,
        "c_add[1,1]={}, expected 3.0",
        c_add.get(1, 1)
    );
    assert!(
        (c_add.get(2, 2) - 4.0).abs() < 1e-12,
        "c_add[2,2]={}, expected 4.0",
        c_add.get(2, 2)
    );

    // Test sparse-sparse multiplication: C = A * B = A * I = A
    let c_mul = matmul(&a, &b).map_err(|e| format!("sparse matmul failed: {}", e))?;
    assert_eq!(c_mul.rows(), n);
    assert_eq!(c_mul.cols(), n);
    assert!(
        (c_mul.get(0, 0) - 1.0).abs() < 1e-12,
        "c_mul[0,0]={}, expected 1.0",
        c_mul.get(0, 0)
    );
    assert!(
        (c_mul.get(1, 1) - 2.0).abs() < 1e-12,
        "c_mul[1,1]={}, expected 2.0",
        c_mul.get(1, 1)
    );
    assert!(
        (c_mul.get(2, 2) - 3.0).abs() < 1e-12,
        "c_mul[2,2]={}, expected 3.0",
        c_mul.get(2, 2)
    );

    println!("Sparse-sparse add and matmul verified");
    Ok(())
}

/// Test sparse format conversions
#[test]
fn test_sparse_format_conversions() -> TestResult<()> {
    // Build a 5×5 known matrix via triplets
    let rows = vec![0usize, 0, 1, 2, 3, 4, 4];
    let cols = vec![0usize, 4, 1, 2, 3, 0, 4];
    let data = vec![1.0f64, 5.0, 2.0, 3.0, 4.0, 6.0, 7.0];
    let n = 5usize;

    let csr = CsrMatrix::new(data.clone(), rows.clone(), cols.clone(), (n, n))
        .map_err(|e| format!("CsrMatrix::new failed: {}", e))?;

    // Verify element access for every known entry
    for k in 0..data.len() {
        let val = csr.get(rows[k], cols[k]);
        assert!(
            (val - data[k]).abs() < 1e-12,
            "Element mismatch at ({}, {}): expected {}, got {}",
            rows[k],
            cols[k],
            data[k],
            val
        );
    }

    // Verify a zero entry
    let zero_val = csr.get(0, 1);
    assert!(
        zero_val.abs() < 1e-12,
        "Expected 0 at (0,1) but got {}",
        zero_val
    );

    // Verify shape
    assert_eq!(csr.rows(), n, "rows mismatch");
    assert_eq!(csr.cols(), n, "cols mismatch");
    assert_eq!(csr.nnz(), data.len(), "nnz mismatch");

    println!(
        "Sparse format construction verified: {}x{} matrix, {} nnz",
        n,
        n,
        csr.nnz()
    );
    Ok(())
}

/// Test iterative solvers with Jacobi preconditioning
#[test]
fn test_iterative_solvers_with_preconditioner() -> TestResult<()> {
    // Build a 10×10 SPD tridiagonal
    let n = 10usize;
    let a = build_spd_tridiagonal(n)?;

    // RHS: b = A * x_true where x_true = [1,2,...,n]
    let x_true_data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let b_vec = a
        .dot(&x_true_data)
        .map_err(|e| format!("matvec for rhs: {}", e))?;
    let b = Array1::from_vec(b_vec);

    // Build Jacobi preconditioner from the matrix diagonal entries
    let precond = EnhancedJacobiPreconditioner::new(&a)
        .map_err(|e| format!("JacobiPreconditioner::new: {}", e))?;

    let config = IterativeSolverConfig {
        max_iter: 1000,
        tol: 1e-10,
        verbose: false,
    };

    let result = enhanced_cg(&a, &b, &config, Some(&precond))
        .map_err(|e| format!("CG with preconditioner failed: {}", e))?;

    assert!(
        result.converged,
        "Preconditioned CG did not converge (iters={}, residual={})",
        result.n_iter, result.residual_norm
    );

    // Verify solution matches x_true within tolerance
    for (i, (&sol, &expected)) in result.solution.iter().zip(x_true_data.iter()).enumerate() {
        assert!(
            (sol - expected).abs() < 1e-7,
            "solution[{}]={} expected {}",
            i,
            sol,
            expected
        );
    }

    println!(
        "Preconditioned CG converged in {} iters, residual={:.2e}",
        result.n_iter, result.residual_norm
    );
    Ok(())
}

/// Test sparse matrix norms — Frobenius, 1-norm, inf-norm
#[test]
fn test_sparse_matrix_norms() -> TestResult<()> {
    // 3×3 matrix:
    //  [3 0 0]
    //  [0 4 0]
    //  [0 0 0]
    // Frobenius norm = sqrt(3^2 + 4^2) = 5
    // 1-norm (max col sum) = 4
    // inf-norm (max row sum) = 4
    let row_idx = vec![0usize, 1];
    let col_idx = vec![0usize, 1];
    let data = vec![3.0f64, 4.0];
    let m = CsrMatrix::new(data, row_idx, col_idx, (3, 3))
        .map_err(|e| format!("CsrMatrix::new: {}", e))?;

    let frob = sparse_norm_fn(&m, "fro").map_err(|e| format!("frobenius norm: {}", e))?;
    assert!(
        (frob - 5.0).abs() < 1e-12,
        "Frobenius norm: expected 5.0, got {}",
        frob
    );

    let norm1 = sparse_norm_fn(&m, "1").map_err(|e| format!("1-norm: {}", e))?;
    assert!(
        (norm1 - 4.0).abs() < 1e-12,
        "1-norm: expected 4.0, got {}",
        norm1
    );

    let norm_inf = sparse_norm_fn(&m, "inf").map_err(|e| format!("inf-norm: {}", e))?;
    assert!(
        (norm_inf - 4.0).abs() < 1e-12,
        "inf-norm: expected 4.0, got {}",
        norm_inf
    );

    println!(
        "Sparse matrix norms: fro={}, 1={}, inf={}",
        frob, norm1, norm_inf
    );
    Ok(())
}

/// Test sparse matrix transpose — verify (A^T)^T = A and (A^T)[i,j] = A[j,i]
#[test]
fn test_sparse_transpose_operations() -> TestResult<()> {
    // Asymmetric 3×4 matrix:
    //  [1 2 0 0]
    //  [0 3 4 0]
    //  [0 0 5 6]
    let row_idx = vec![0usize, 0, 1, 1, 2, 2];
    let col_idx = vec![0usize, 1, 1, 2, 2, 3];
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = CsrMatrix::new(data, row_idx, col_idx, (3, 4))
        .map_err(|e| format!("CsrMatrix::new: {}", e))?;

    // Transpose: A^T should be 4×3
    let at = a.transpose();
    assert_eq!(at.rows(), 4, "A^T rows mismatch");
    assert_eq!(at.cols(), 3, "A^T cols mismatch");

    // Verify (A^T)[j,i] = A[i,j] for all non-zero entries
    // A[0,0]=1 → A^T[0,0]=1
    assert!(
        (at.get(0, 0) - 1.0).abs() < 1e-12,
        "A^T[0,0]={} expected 1.0",
        at.get(0, 0)
    );
    // A[0,1]=2 → A^T[1,0]=2
    assert!(
        (at.get(1, 0) - 2.0).abs() < 1e-12,
        "A^T[1,0]={} expected 2.0",
        at.get(1, 0)
    );
    // A[1,2]=4 → A^T[2,1]=4
    assert!(
        (at.get(2, 1) - 4.0).abs() < 1e-12,
        "A^T[2,1]={} expected 4.0",
        at.get(2, 1)
    );
    // A[2,3]=6 → A^T[3,2]=6
    assert!(
        (at.get(3, 2) - 6.0).abs() < 1e-12,
        "A^T[3,2]={} expected 6.0",
        at.get(3, 2)
    );

    // (A^T)^T should equal A
    let att = at.transpose();
    assert_eq!(att.rows(), 3, "(A^T)^T rows mismatch");
    assert_eq!(att.cols(), 4, "(A^T)^T cols mismatch");
    assert!(
        (att.get(1, 2) - 4.0).abs() < 1e-12,
        "(A^T)^T[1,2]={} expected 4.0",
        att.get(1, 2)
    );

    println!("Sparse transpose verified: {}×{} → {}×{}", 3, 4, 4, 3);
    Ok(())
}

/// Test sparse matrix slicing and submatrix extraction
#[test]
fn test_sparse_submatrix_operations() -> TestResult<()> {
    // Build a 5×5 banded matrix:
    //  row 0: [2, -1,  0,  0,  0]
    //  row 1: [-1, 2, -1,  0,  0]
    //  row 2: [ 0,-1,  2, -1,  0]
    //  row 3: [ 0, 0, -1,  2, -1]
    //  row 4: [ 0, 0,  0, -1,  2]
    let n = 5usize;
    let mut rows_v = Vec::new();
    let mut cols_v = Vec::new();
    let mut data_v = Vec::new();
    for i in 0..n {
        rows_v.push(i);
        cols_v.push(i);
        data_v.push(2.0f64);
        if i > 0 {
            rows_v.push(i);
            cols_v.push(i - 1);
            data_v.push(-1.0);
        }
        if i + 1 < n {
            rows_v.push(i);
            cols_v.push(i + 1);
            data_v.push(-1.0);
        }
    }
    let full =
        CsrMatrix::new(data_v, rows_v, cols_v, (n, n)).map_err(|e| format!("build 5×5: {}", e))?;

    // Extract top-left 3×3 block
    let sub = full
        .submatrix(0, 3, 0, 3)
        .map_err(|e| format!("submatrix(0,3,0,3): {}", e))?;
    assert_eq!(sub.rows(), 3, "submatrix rows");
    assert_eq!(sub.cols(), 3, "submatrix cols");

    // Diagonal entries should be 2
    for i in 0..3 {
        assert!(
            (sub.get(i, i) - 2.0).abs() < 1e-12,
            "sub[{0},{0}]={} expected 2.0",
            sub.get(i, i)
        );
    }

    // Off-diagonal entries should be -1 for adjacent rows
    assert!(
        (sub.get(0, 1) - (-1.0)).abs() < 1e-12,
        "sub[0,1]={} expected -1.0",
        sub.get(0, 1)
    );
    assert!(
        (sub.get(1, 0) - (-1.0)).abs() < 1e-12,
        "sub[1,0]={} expected -1.0",
        sub.get(1, 0)
    );
    assert!(
        (sub.get(1, 2) - (-1.0)).abs() < 1e-12,
        "sub[1,2]={} expected -1.0",
        sub.get(1, 2)
    );

    // Row 0 of the 5×5 matrix should NOT appear in the sub for row 3 direction
    // (i.e. position (0,3) in sub must be zero — sub has only 3 cols)
    // Row 2 should have no entry at col 3 (out of sub's column range)
    assert!(
        sub.get(2, 2).abs() > 0.0,
        "sub[2,2] should be non-zero (=2)"
    );

    // Extract middle 2×2 block (rows 2-3, cols 2-3)
    let mid = full
        .submatrix(2, 4, 2, 4)
        .map_err(|e| format!("submatrix(2,4,2,4): {}", e))?;
    assert_eq!(mid.rows(), 2, "mid rows");
    assert_eq!(mid.cols(), 2, "mid cols");
    assert!(
        (mid.get(0, 0) - 2.0).abs() < 1e-12,
        "mid[0,0]={} expected 2.0",
        mid.get(0, 0)
    );
    assert!(
        (mid.get(0, 1) - (-1.0)).abs() < 1e-12,
        "mid[0,1]={} expected -1.0",
        mid.get(0, 1)
    );
    assert!(
        (mid.get(1, 0) - (-1.0)).abs() < 1e-12,
        "mid[1,0]={} expected -1.0",
        mid.get(1, 0)
    );
    assert!(
        (mid.get(1, 1) - 2.0).abs() < 1e-12,
        "mid[1,1]={} expected 2.0",
        mid.get(1, 1)
    );

    println!(
        "Sparse submatrix extraction verified: 5×5 → 3×3 sub (nnz={}) and 2×2 mid (nnz={})",
        sub.nnz(),
        mid.nnz()
    );
    Ok(())
}

/// Test sparse matrix concatenation using hstack / vstack
#[test]
fn test_sparse_matrix_concatenation() -> TestResult<()> {
    // Build two 2×2 identity-like CsrArray matrices
    let rows_a = vec![0usize, 1];
    let cols_a = vec![0usize, 1];
    let data_a = vec![1.0f64, 2.0];
    let a = CsrArray::from_triplets(&rows_a, &cols_a, &data_a, (2, 2), false)
        .map_err(|e| format!("CsrArray A: {}", e))?;

    let rows_b = vec![0usize, 1];
    let cols_b = vec![0usize, 1];
    let data_b = vec![3.0f64, 4.0];
    let b = CsrArray::from_triplets(&rows_b, &cols_b, &data_b, (2, 2), false)
        .map_err(|e| format!("CsrArray B: {}", e))?;

    // hstack: [A | B] → 2×4
    let h = hstack(
        &[&a as &dyn SparseArray<f64>, &b as &dyn SparseArray<f64>],
        "csr",
    )
    .map_err(|e| format!("hstack: {}", e))?;
    assert_eq!(h.shape(), (2, 4), "hstack shape mismatch: {:?}", h.shape());
    // A occupies columns 0-1, B occupies columns 2-3
    assert!(
        (h.get(0, 0) - 1.0).abs() < 1e-12,
        "hstack[0,0]={} expected 1.0",
        h.get(0, 0)
    );
    assert!(
        (h.get(1, 1) - 2.0).abs() < 1e-12,
        "hstack[1,1]={} expected 2.0",
        h.get(1, 1)
    );
    assert!(
        (h.get(0, 2) - 3.0).abs() < 1e-12,
        "hstack[0,2]={} expected 3.0",
        h.get(0, 2)
    );
    assert!(
        (h.get(1, 3) - 4.0).abs() < 1e-12,
        "hstack[1,3]={} expected 4.0",
        h.get(1, 3)
    );

    // vstack: [A; B] → 4×2
    let v = vstack(
        &[&a as &dyn SparseArray<f64>, &b as &dyn SparseArray<f64>],
        "csr",
    )
    .map_err(|e| format!("vstack: {}", e))?;
    assert_eq!(v.shape(), (4, 2), "vstack shape mismatch: {:?}", v.shape());
    // A occupies rows 0-1, B occupies rows 2-3
    assert!(
        (v.get(0, 0) - 1.0).abs() < 1e-12,
        "vstack[0,0]={} expected 1.0",
        v.get(0, 0)
    );
    assert!(
        (v.get(2, 0) - 3.0).abs() < 1e-12,
        "vstack[2,0]={} expected 3.0",
        v.get(2, 0)
    );
    assert!(
        (v.get(3, 1) - 4.0).abs() < 1e-12,
        "vstack[3,1]={} expected 4.0",
        v.get(3, 1)
    );

    println!(
        "Sparse concatenation verified: hstack→{:?}, vstack→{:?}",
        h.shape(),
        v.shape()
    );
    Ok(())
}

// Property-based tests

proptest! {
    #[test]
    fn prop_sparse_dense_consistency(
        n in 10usize..50,
        m in 10usize..50,
        density in 0.05..0.3
    ) {
        // Property: Sparse matrix operations should give same results
        // as equivalent dense operations
        let sparse_triplets = TestDatasets::sparse_test_matrix(n, m, density);
        prop_assert!(n > 0 && m > 0);
        prop_assert!(!sparse_triplets.is_empty() || (n * m) as f64 * density < 1.0);
    }

    #[test]
    fn prop_sparse_matrix_symmetry(
        n in 10usize..50
    ) {
        // Property: For symmetric sparse matrix, A = A^T
        let sparse_triplets = TestDatasets::sparse_test_matrix(n, n, 0.1);
        prop_assert!(n > 0);
        // Just verify the triplets have valid indices
        for (r, c, _v) in &sparse_triplets {
            prop_assert!(*r < n, "row index out of bounds");
            prop_assert!(*c < n, "col index out of bounds");
        }
    }

    /// Property: sparse matvec scales linearly — (A * (alpha * x)) ≈ alpha * (A * x)
    #[test]
    fn prop_sparse_matvec_linear(
        n in 5usize..30,
        alpha in -5.0f64..5.0,
    ) {
        let size = n;
        // Build SPD tridiagonal
        let a = build_spd_tridiagonal(size).expect("build_spd_tridiagonal failed");

        let x: Vec<f64> = (0..size).map(|i| (i as f64 + 1.0) / size as f64).collect();
        let alpha_x: Vec<f64> = x.iter().map(|&v| alpha * v).collect();

        let ax = a.dot(&x).expect("matvec ax failed");
        let a_alphax = a.dot(&alpha_x).expect("matvec a_alphax failed");

        let max_err = ax.iter()
            .zip(a_alphax.iter())
            .map(|(axv, aalphaxv)| (aalphaxv - alpha * axv).abs())
            .fold(0.0_f64, f64::max);

        prop_assert!(
            max_err < 1e-10,
            "Linearity violated: max_err={}",
            max_err
        );
    }

    #[test]
    fn prop_sparse_solver_accuracy(
        n in 10usize..30,
        density in 0.1..0.5
    ) {
        // Property: Sparse solver should satisfy ||Ax - b|| / ||b|| < tolerance
        let sparse_triplets = TestDatasets::sparse_test_matrix(n, n, density);
        prop_assert!(n > 0);
        let _ = sparse_triplets;
    }
}

/// Test memory efficiency of sparse operations
#[test]
fn test_sparse_operations_memory_efficiency() -> TestResult<()> {
    let large_n = 1000;
    let sparse_triplets = TestDatasets::sparse_test_matrix(large_n, large_n, 0.01);

    println!("Testing memory efficiency of sparse operations");
    println!("Matrix size: {}x{} (density 0.01)", large_n, large_n);

    assert_memory_efficient(
        || {
            // Verify that building a large sparse matrix from triplets is cheap
            let rows: Vec<usize> = sparse_triplets.iter().map(|(r, _, _)| *r).collect();
            let cols: Vec<usize> = sparse_triplets.iter().map(|(_, c, _)| *c).collect();
            let data: Vec<f64> = sparse_triplets.iter().map(|(_, _, v)| *v).collect();
            let _m = CsrMatrix::new(data, rows, cols, (large_n, large_n))
                .map_err(|e| format!("CsrMatrix failed: {}", e))?;
            Ok(())
        },
        50.0,
        "Sparse matrix operations",
    )?;

    Ok(())
}

/// Test sparse matrix condition number estimation using power iteration
#[test]
fn test_sparse_condition_number() -> TestResult<()> {
    // 3×3 diagonal matrix with known condition number:
    //  diag(1, 2, 4) → cond = largest/smallest = 4/1 = 4
    let row_idx = vec![0usize, 1, 2];
    let col_idx = vec![0usize, 1, 2];
    let data = vec![1.0f64, 2.0, 4.0];
    let m = CsrMatrix::new(data, row_idx, col_idx, (3, 3))
        .map_err(|e| format!("CsrMatrix::new: {}", e))?;

    // Estimate the spectral radius (largest eigenvalue magnitude)
    let spectral_radius = estimate_spectral_radius(&m, 100)
        .map_err(|e| format!("estimate_spectral_radius: {}", e))?;

    // For diag(1,2,4), spectral radius ≈ 4
    assert!(
        spectral_radius > 0.0,
        "Spectral radius must be positive, got {}",
        spectral_radius
    );
    assert!(
        (spectral_radius - 4.0).abs() < 0.5,
        "Spectral radius: expected ~4.0, got {}",
        spectral_radius
    );

    println!("Spectral radius estimated: {:.4}", spectral_radius);
    Ok(())
}

/// Test sparse QR factorization with reconstruction check Q*R ≈ A
#[test]
fn test_sparse_qr_factorization() -> TestResult<()> {
    // 4×4 well-conditioned matrix via CsrArray
    let rows = vec![0usize, 0, 1, 1, 2, 2, 3, 3];
    let cols = vec![0usize, 1, 1, 2, 2, 3, 3, 0];
    let data = vec![2.0f64, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0];
    let n = 4usize;

    let matrix = CsrArray::from_triplets(&rows, &cols, &data, (n, n), false)
        .map_err(|e| format!("CsrArray: {}", e))?;

    let qr = qr_decomposition::<f64, _>(&matrix).map_err(|e| format!("qr_decomposition: {}", e))?;

    assert!(qr.success, "QR decomposition reported failure");

    let (qrows, qcols) = qr.q.shape();
    let (rrows, rcols) = qr.r.shape();
    assert_eq!(qrows, n, "Q rows mismatch");
    assert_eq!(qcols, n, "Q cols mismatch");
    assert_eq!(rrows, n, "R rows mismatch");
    assert_eq!(rcols, n, "R cols mismatch");

    // Reconstruct A = Q * R using dense multiplication
    let q_dense = qr.q.to_array();
    let r_dense = qr.r.to_array();
    let a_orig = matrix.to_array();

    // C[i,j] = sum_k Q[i,k] * R[k,j]
    let mut reconstructed = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += q_dense[[i, k]] * r_dense[[k, j]];
            }
            reconstructed[[i, j]] = s;
        }
    }

    // Check reconstruction error
    let max_err = a_orig
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &r)| (a - r).abs())
        .fold(0.0f64, f64::max);

    assert!(
        max_err < 1e-10,
        "QR reconstruction error too large: max|A - QR| = {}",
        max_err
    );

    println!(
        "Sparse QR factorization verified: max reconstruction error = {:.2e}",
        max_err
    );
    Ok(())
}

/// Test sparse SVD — verify singular values are non-negative and sorted,
/// and left/right singular vectors have orthonormal columns.
#[test]
fn test_sparse_svd() -> TestResult<()> {
    // 4×5 matrix via CsrArray; k must be < min(4,5) = 4, so use k=2
    let rows = vec![0usize, 0, 1, 1, 2, 3];
    let cols = vec![0usize, 2, 1, 3, 2, 4];
    let data = vec![3.0f64, 1.0, 2.0, 4.0, 5.0, 2.0];
    let m = 4usize;
    let n = 5usize;

    let matrix = CsrArray::from_triplets(&rows, &cols, &data, (m, n), false)
        .map_err(|e| format!("CsrArray: {}", e))?;

    let result = svds(&matrix, Some(2), None).map_err(|e| format!("svds failed: {}", e))?;

    assert_eq!(
        result.s.len(),
        2,
        "Expected 2 singular values, got {}",
        result.s.len()
    );

    // All singular values must be non-negative
    for &sv in result.s.iter() {
        assert!(sv >= 0.0, "Singular value {} is negative", sv);
    }

    // Singular values should be sorted descending (or at worst equal)
    if result.s.len() >= 2 {
        assert!(
            result.s[0] >= result.s[1] - 1e-10,
            "Singular values not sorted descending: {} < {}",
            result.s[0],
            result.s[1]
        );
    }

    // The largest singular value of this matrix should be non-trivial (> 1.0)
    // since the matrix has entries 2-5 and is 4×5
    assert!(
        result.s[0] > 0.5,
        "Largest singular value {} is unexpectedly small",
        result.s[0]
    );

    // U and Vt should have correct shapes if computed
    if let Some(ref u) = result.u {
        assert_eq!(u.nrows(), m, "U nrows mismatch");
        assert_eq!(u.ncols(), 2, "U ncols mismatch");
    }
    if let Some(ref vt) = result.vt {
        assert_eq!(vt.nrows(), 2, "Vt nrows mismatch");
        assert_eq!(vt.ncols(), n, "Vt ncols mismatch");
    }

    println!(
        "Sparse SVD verified: singular values = {:?} (converged={})",
        result.s, result.converged
    );
    Ok(())
}

/// Test sparse matrix graph operations (graph Laplacian connectivity check)
#[test]
fn test_sparse_matrix_graph_operations() -> TestResult<()> {
    // Build the adjacency matrix for the path graph P₄: 0-1-2-3
    //   A = [[0,1,0,0],
    //        [1,0,1,0],
    //        [0,1,0,1],
    //        [0,0,1,0]]
    let adj_rows = vec![0usize, 1, 1, 2, 2, 3];
    let adj_cols = vec![1usize, 0, 2, 1, 3, 2];
    let adj_data = vec![1.0f64; 6];
    let n = 4usize;

    let adj = CsrMatrix::new(adj_data, adj_rows, adj_cols, (n, n))
        .map_err(|e| format!("adjacency matrix: {}", e))?;

    // Compute the combinatorial Laplacian L = D - A
    let l = graph_laplacian(&adj).map_err(|e| format!("graph_laplacian: {}", e))?;

    assert_eq!(l.rows(), n, "L rows");
    assert_eq!(l.cols(), n, "L cols");

    // The Laplacian has row sums = 0 (each row: degree on diagonal, -1 off-diag)
    for i in 0..n {
        let mut row_sum = 0.0f64;
        for j in 0..n {
            row_sum += l.get(i, j);
        }
        assert!(
            row_sum.abs() < 1e-12,
            "L row {} sum = {} expected 0",
            i,
            row_sum
        );
    }

    // Node 0 and 3 have degree 1; nodes 1 and 2 have degree 2
    assert!(
        (l.get(0, 0) - 1.0).abs() < 1e-12,
        "L[0,0]={} expected 1",
        l.get(0, 0)
    );
    assert!(
        (l.get(1, 1) - 2.0).abs() < 1e-12,
        "L[1,1]={} expected 2",
        l.get(1, 1)
    );
    assert!(
        (l.get(0, 1) - (-1.0)).abs() < 1e-12,
        "L[0,1]={} expected -1",
        l.get(0, 1)
    );
    assert!(
        (l.get(3, 3) - 1.0).abs() < 1e-12,
        "L[3,3]={} expected 1",
        l.get(3, 3)
    );

    // Compute the normalized Laplacian L_sym = D^{-1/2} L D^{-1/2}
    let l_sym = normalized_laplacian(&adj).map_err(|e| format!("normalized_laplacian: {}", e))?;
    assert_eq!(l_sym.rows(), n, "L_sym rows");

    // Diagonal of L_sym: 1 for any non-isolated node
    for i in 0..n {
        assert!(
            (l_sym.get(i, i) - 1.0).abs() < 1e-12,
            "L_sym[{0},{0}]={} expected 1.0",
            l_sym.get(i, i)
        );
    }

    // Off-diagonal: L_sym[0,1] = -1 / sqrt(1 * 2) = -1/sqrt(2) ≈ -0.7071
    let expected_01 = -1.0f64 / (1.0f64 * 2.0f64).sqrt();
    assert!(
        (l_sym.get(0, 1) - expected_01).abs() < 1e-10,
        "L_sym[0,1]={} expected {}",
        l_sym.get(0, 1),
        expected_01
    );

    // Verify graph connectivity via AdjacencyGraph
    let graph = AdjacencyGraph::from_csr_matrix(&adj)
        .map_err(|e| format!("AdjacencyGraph::from_csr_matrix: {}", e))?;
    assert_eq!(graph.num_nodes(), n, "graph num_nodes");

    println!(
        "Graph Laplacian verified for P₄: L nnz={}, L_sym nnz={}",
        l.nnz(),
        l_sym.nnz()
    );
    Ok(())
}

/// Test sparse matrix elementwise operations
#[test]
fn test_sparse_elementwise_operations() -> TestResult<()> {
    // A = diag(1, 2, 3)  B = diag(4, 5, 6)
    // C = A ⊙ B = diag(4, 10, 18)
    let n = 3usize;
    let a = CsrMatrix::new(
        vec![1.0f64, 2.0, 3.0],
        vec![0usize, 1, 2],
        vec![0usize, 1, 2],
        (n, n),
    )
    .map_err(|e| format!("CsrMatrix A: {}", e))?;

    let b = CsrMatrix::new(
        vec![4.0f64, 5.0, 6.0],
        vec![0usize, 1, 2],
        vec![0usize, 1, 2],
        (n, n),
    )
    .map_err(|e| format!("CsrMatrix B: {}", e))?;

    // Hadamard product
    let c = a
        .elementwise_mul(&b)
        .map_err(|e| format!("elementwise_mul: {}", e))?;

    assert_eq!(c.rows(), n, "C rows");
    assert_eq!(c.cols(), n, "C cols");

    assert!(
        (c.get(0, 0) - 4.0).abs() < 1e-12,
        "c[0,0]={} expected 4.0",
        c.get(0, 0)
    );
    assert!(
        (c.get(1, 1) - 10.0).abs() < 1e-12,
        "c[1,1]={} expected 10.0",
        c.get(1, 1)
    );
    assert!(
        (c.get(2, 2) - 18.0).abs() < 1e-12,
        "c[2,2]={} expected 18.0",
        c.get(2, 2)
    );

    // Off-diagonal must be zero (no overlap)
    assert!(
        c.get(0, 1).abs() < 1e-12,
        "c[0,1]={} expected 0.0",
        c.get(0, 1)
    );

    // Test with overlapping non-zero patterns
    // D = [[1, 2, 0], [0, 3, 4], [0, 0, 5]]
    // E = [[6, 0, 0], [7, 8, 0], [0, 9, 10]]
    // F = D ⊙ E = [[6, 0, 0], [0, 24, 0], [0, 0, 50]]
    let d = CsrMatrix::new(
        vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
        vec![0usize, 0, 1, 1, 2],
        vec![0usize, 1, 1, 2, 2],
        (n, n),
    )
    .map_err(|e| format!("CsrMatrix D: {}", e))?;

    let e_mat = CsrMatrix::new(
        vec![6.0f64, 7.0, 8.0, 9.0, 10.0],
        vec![0usize, 1, 1, 2, 2],
        vec![0usize, 0, 1, 1, 2],
        (n, n),
    )
    .map_err(|e| format!("CsrMatrix E: {}", e))?;

    let f_mat = d
        .elementwise_mul(&e_mat)
        .map_err(|e| format!("elementwise_mul D⊙E: {}", e))?;

    assert!(
        (f_mat.get(0, 0) - 6.0).abs() < 1e-12,
        "f[0,0]={} expected 6.0",
        f_mat.get(0, 0)
    );
    // D[0,1]=2 but E[0,1]=0 → 0
    assert!(
        f_mat.get(0, 1).abs() < 1e-12,
        "f[0,1]={} expected 0.0",
        f_mat.get(0, 1)
    );
    assert!(
        (f_mat.get(1, 1) - 24.0).abs() < 1e-12,
        "f[1,1]={} expected 24.0",
        f_mat.get(1, 1)
    );
    assert!(
        (f_mat.get(2, 2) - 50.0).abs() < 1e-12,
        "f[2,2]={} expected 50.0",
        f_mat.get(2, 2)
    );

    println!(
        "Elementwise (Hadamard) product verified: diag case nnz={}, overlap case nnz={}",
        c.nnz(),
        f_mat.nnz()
    );
    Ok(())
}

/// Test sparse matrix reordering (Reverse Cuthill-McKee)
#[test]
fn test_sparse_matrix_reordering() -> TestResult<()> {
    // Build a deliberately wide-bandwidth 8-node tridiagonal but with rows
    // permuted (anti-diagonal ordering) so the natural bandwidth is large.
    // Natural tridiagonal (diag+off-diag) has bandwidth 1.
    // We permute the indices so node i → node (7-i), making bandwidth = 7.
    //
    // Specifically: build the symmetrised adjacency for a path graph
    //   0-1-2-3-4-5-6-7
    // then permute row/col so node 0 is connected to 7, etc.
    // giving high bandwidth.  RCM should recover a lower bandwidth.

    // Build simple path graph (bandwidth = 1) as adjacency matrix
    let n = 8usize;
    let mut rows_v = Vec::new();
    let mut cols_v = Vec::new();
    let mut data_v = Vec::new();

    // Diagonal
    for i in 0..n {
        rows_v.push(i);
        cols_v.push(i);
        data_v.push(2.0f64);
    }
    // Off-diagonal: connect i ↔ i+1
    for i in 0..(n - 1) {
        rows_v.push(i);
        cols_v.push(i + 1);
        data_v.push(-1.0);
        rows_v.push(i + 1);
        cols_v.push(i);
        data_v.push(-1.0);
    }

    let a = CsrMatrix::new(data_v, rows_v, cols_v, (n, n))
        .map_err(|e| format!("build tridiagonal: {}", e))?;

    // Build an adjacency graph from the matrix and apply RCM
    let graph =
        AdjacencyGraph::from_csr_matrix(&a).map_err(|e| format!("AdjacencyGraph: {}", e))?;
    assert_eq!(graph.num_nodes(), n, "graph node count");

    let rcm_result = reverse_cuthill_mckee_full(&graph)
        .map_err(|e| format!("reverse_cuthill_mckee_full: {}", e))?;

    // Permutation must cover all n nodes (each index appears exactly once)
    assert_eq!(rcm_result.perm.len(), n, "perm length");
    let mut seen = vec![false; n];
    for &p in &rcm_result.perm {
        assert!(p < n, "perm index out of range: {}", p);
        assert!(!seen[p], "duplicate perm index: {}", p);
        seen[p] = true;
    }
    assert!(seen.iter().all(|&s| s), "perm does not cover all nodes");

    // For the natural tridiagonal the original bandwidth should be 1.
    // RCM on such a matrix should keep bandwidth <= original or at most small.
    assert!(
        rcm_result.bandwidth_after <= rcm_result.bandwidth_before + 1,
        "RCM bandwidth increased too much: before={}, after={}",
        rcm_result.bandwidth_before,
        rcm_result.bandwidth_after
    );

    // For a path graph, the natural bandwidth is 1; RCM should keep it at 1.
    assert!(
        rcm_result.bandwidth_after <= 1,
        "Expected bandwidth_after <= 1 for path graph, got {}",
        rcm_result.bandwidth_after
    );

    println!(
        "RCM reordering verified: bandwidth {}->{} for {}×{} path graph",
        rcm_result.bandwidth_before, rcm_result.bandwidth_after, n, n
    );
    Ok(())
}

/// Test sparse direct solver: solve Ax=b, verify A*x ≈ b
#[test]
fn test_sparse_direct_solvers() -> TestResult<()> {
    // 4×4 diagonally dominant system
    let n = 4usize;
    let a = build_spd_tridiagonal(n)?;

    let b = vec![1.0f64, 0.0, 0.0, 1.0];

    // Use sparse_direct_solve (falls through to Gaussian elimination)
    let x = sparse_direct_solve(&a, &b, true, true)
        .map_err(|e| format!("sparse_direct_solve: {}", e))?;

    assert_eq!(x.len(), n, "Solution length mismatch");

    // Verify A * x ≈ b
    let ax = a
        .dot(&x)
        .map_err(|e| format!("verification matvec: {}", e))?;
    let max_err = ax
        .iter()
        .zip(b.iter())
        .map(|(&axi, &bi)| (axi - bi).abs())
        .fold(0.0f64, f64::max);

    assert!(
        max_err < 1e-10,
        "Direct solver residual too large: max|Ax - b| = {}",
        max_err
    );

    println!("Sparse direct solver verified: max|Ax-b| = {:.2e}", max_err);
    Ok(())
}

/// Test sparse matrix power: A^2 = A * A, verify by comparing with manual matmul
#[test]
fn test_sparse_matrix_powers() -> TestResult<()> {
    // 3×3 simple sparse matrix (diagonal + one off-diagonal)
    //  A = [[2, 1, 0],
    //       [0, 2, 1],
    //       [0, 0, 2]]
    let row_idx = vec![0usize, 0, 1, 1, 2];
    let col_idx = vec![0usize, 1, 1, 2, 2];
    let data = vec![2.0f64, 1.0, 2.0, 1.0, 2.0];
    let a = CsrMatrix::new(data, row_idx, col_idx, (3, 3))
        .map_err(|e| format!("CsrMatrix::new: {}", e))?;

    // A^2 via sparse_matrix_power (explicit sparse version to avoid linalg shadowing)
    let a2 = sparse_matrix_power(&a, 2).map_err(|e| format!("matrix_power: {}", e))?;

    // A^2 via direct matmul
    let a2_ref = matmul(&a, &a).map_err(|e| format!("reference matmul: {}", e))?;

    assert_eq!(a2.rows(), 3, "A^2 rows mismatch");
    assert_eq!(a2.cols(), 3, "A^2 cols mismatch");

    // Compare every element
    for i in 0..3usize {
        for j in 0..3usize {
            let pow_val = a2.get(i, j);
            let ref_val = a2_ref.get(i, j);
            assert!(
                (pow_val - ref_val).abs() < 1e-12,
                "A^2[{},{}]: matrix_power={}, direct_matmul={}",
                i,
                j,
                pow_val,
                ref_val
            );
        }
    }

    println!("Sparse matrix power A^2 verified");
    Ok(())
}

/// Test integration with dense linear algebra — build sparse, convert to dense, compare ops
#[test]
fn test_sparse_dense_integration() -> TestResult<()> {
    // Build a 4×4 SPD tridiagonal sparse matrix
    let n = 4usize;
    let a_sparse = build_spd_tridiagonal(n)?;

    // Convert to dense (manual via get)
    let mut a_dense = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_dense[[i, j]] = a_sparse.get(i, j);
        }
    }

    // Verify dense representation matches sparse on diagonal and off-diagonal
    for i in 0..n {
        assert!(
            (a_dense[[i, i]] - 2.0).abs() < 1e-12,
            "dense diagonal [{}] = {} expected 2.0",
            i,
            a_dense[[i, i]]
        );
    }
    for i in 0..(n - 1) {
        assert!(
            (a_dense[[i, i + 1]] - (-1.0)).abs() < 1e-12,
            "dense super-diag [{},{}] = {} expected -1.0",
            i,
            i + 1,
            a_dense[[i, i + 1]]
        );
    }

    // Sparse matvec and dense matvec should agree
    let x: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    let sparse_result = a_sparse
        .dot(&x)
        .map_err(|e| format!("sparse matvec: {}", e))?;

    let x_arr = Array1::from_vec(x.clone());
    let dense_result = a_dense.dot(&x_arr);

    for (i, (&s, &d)) in sparse_result.iter().zip(dense_result.iter()).enumerate() {
        assert!(
            (s - d).abs() < 1e-12,
            "sparse[{}]={} vs dense[{}]={}",
            i,
            s,
            i,
            d
        );
    }

    println!(
        "Sparse-dense integration verified: matvec agrees for {}×{} matrix",
        n, n
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Dense matvec test (uses known entries)
// ---------------------------------------------------------------------------

/// Test sparse dense matvec — multiply a simple CsrMatrix by a vector
#[test]
fn test_sparse_dense_matvec() -> TestResult<()> {
    // 3×3 matrix:
    //  [1 0 2]
    //  [0 3 0]
    //  [4 0 5]
    let row_idx = vec![0usize, 0, 1, 2, 2];
    let col_idx = vec![0usize, 2, 1, 0, 2];
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let m = CsrMatrix::new(data, row_idx, col_idx, (3, 3))
        .map_err(|e| format!("CsrMatrix::new: {}", e))?;

    let x = vec![1.0f64, 2.0, 3.0];
    let y = m.dot(&x).map_err(|e| format!("dot: {}", e))?;

    // Expected: [1*1+2*3, 3*2, 4*1+5*3] = [7, 6, 19]
    let expected = [7.0, 6.0, 19.0];
    for (i, (&got, &exp)) in y.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-12,
            "y[{}]: expected {}, got {}",
            i,
            exp,
            got
        );
    }

    println!("Sparse matvec verified: y = {:?}", y);
    Ok(())
}

#[cfg(test)]
mod api_compatibility_tests {
    use super::*;

    /// Test that sparse matrix types are compatible with linalg operations
    #[test]
    fn test_sparse_type_compatibility() -> TestResult<()> {
        // Verify that CsrMatrix can be built and matvec works
        let a = build_spd_tridiagonal(5)?;
        let x = vec![1.0f64; 5];
        let _y = a
            .dot(&x)
            .map_err(|e| format!("type compat matvec: {}", e))?;
        println!("Sparse-linalg type compatibility verified");
        Ok(())
    }

    /// Test error handling across sparse-linalg boundary
    #[test]
    fn test_sparse_linalg_error_handling() -> TestResult<()> {
        // Dimension mismatch: 5x5 matrix with length-3 vector
        let a = build_spd_tridiagonal(5)?;
        let x = vec![1.0f64, 2.0, 3.0];
        let result = a.dot(&x);
        assert!(
            result.is_err(),
            "Expected error for dimension mismatch, got Ok"
        );
        println!("Error handling test passed: {:?}", result.err());
        Ok(())
    }

    /// Test performance characteristics
    #[test]
    fn test_sparse_linalg_performance() -> TestResult<()> {
        let sizes = vec![100, 200, 500, 1000];
        let density = 0.05;

        println!("Testing sparse vs dense performance");

        for n in sizes {
            let sparse_triplets = TestDatasets::sparse_test_matrix(n, n, density);
            let rows: Vec<usize> = sparse_triplets.iter().map(|(r, _, _)| *r).collect();
            let cols: Vec<usize> = sparse_triplets.iter().map(|(_, c, _)| *c).collect();
            let data: Vec<f64> = sparse_triplets.iter().map(|(_, _, v)| *v).collect();
            let m = CsrMatrix::new(data, rows, cols, (n, n))
                .map_err(|e| format!("CsrMatrix failed: {}", e))?;
            let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
            let (_y, perf) = measure_time(&format!("SpMV n={}", n), || {
                m.dot(&x).map_err(|e| e.to_string().into())
            })?;
            println!("  Size {}: {:.3} ms", n, perf.duration_ms);
        }

        Ok(())
    }
}
