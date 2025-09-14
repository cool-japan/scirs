//! Showcase of SciPy-compatible API functions
//!
//! This example demonstrates the expanded SciPy linalg API compatibility
//! coverage provided by scirs2-linalg. It shows how to use common SciPy
//! functions with the same interface and parameter names.

use ndarray::array;
use scirs2_linalg::compat;
use scirs2_linalg::error::LinalgResult;

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("=== SciPy-compatible Linear Algebra API Showcase ===\n");

    // Test data
    let a = array![[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]];
    let b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let vector = array![3.0, 4.0, 5.0];

    // 1. Basic Matrix Operations
    println!("1. Basic Matrix Operations");
    println!("--------------------------");

    // Determinant (SciPy: scipy.linalg.det)
    let det_result = compat::det(&a.view(), false, true)?;
    println!("Determinant: {:.4}", det_result);

    // Matrix inverse (SciPy: scipy.linalg.inv)
    let inv_result = compat::inv(&a.view(), false, true)?;
    println!("Inverse:\n{:.4}", inv_result);

    // Pseudoinverse (SciPy: scipy.linalg.pinv)
    let pinv_result = compat::pinv(&a.view(), None, false, true)?;
    println!("Pseudoinverse:\n{:.4}", pinv_result);

    // 2. Matrix Norms and Properties
    println!("\n2. Matrix Norms and Properties");
    println!("------------------------------");

    // Frobenius norm (SciPy: scipy.linalg.norm)
    let frobenius_norm = compat::norm(&a.view(), Some("fro"), None, false, true)?;
    println!("Frobenius norm: {:.4}", frobenius_norm);

    // 1-norm (SciPy: scipy.linalg.norm with ord=1)
    let norm_1 = compat::norm(&a.view(), Some("1"), None, false, true)?;
    println!("1-norm: {:.4}", norm_1);

    // Infinity norm (SciPy: scipy.linalg.norm with ord=np.inf)
    let norm_inf = compat::norm(&a.view(), Some("inf"), None, false, true)?;
    println!("Infinity norm: {:.4}", norm_inf);

    // Vector norms (SciPy: scipy.linalg.norm for vectors)
    let vector_norm_2 = compat::vector_norm(&vector.view(), Some(2.0), true)?;
    println!("Vector 2-norm: {:.4}", vector_norm_2);

    let vector_norm_1 = compat::vector_norm(&vector.view(), Some(1.0), true)?;
    println!("Vector 1-norm: {:.4}", vector_norm_1);

    // Condition number (SciPy: scipy.linalg.cond)
    let cond_2 = compat::cond(&a.view(), Some("2"))?;
    println!("Condition number (2-norm): {:.4}", cond_2);

    // Matrix rank (SciPy: scipy.linalg.matrix_rank)
    let rank = compat::matrix_rank(&a.view(), None, false, true)?;
    println!("Matrix rank: {}", rank);

    // 3. Matrix Decompositions
    println!("\n3. Matrix Decompositions");
    println!("------------------------");

    // LU decomposition (SciPy: scipy.linalg.lu)
    let (p, l, u) = compat::lu(&a.view(), false, false, true, false)?;
    println!("LU decomposition successful");
    println!("P:\n{:.4}", p);
    println!("L:\n{:.4}", l);
    println!("U:\n{:.4}", u);

    // QR decomposition (SciPy: scipy.linalg.qr)
    let (q_opt, r) = compat::qr(&a.view(), false, None, "full", false, true)?;
    if let Some(q) = q_opt {
        println!("QR decomposition successful");
        println!("Q:\n{:.4}", q);
        println!("R:\n{:.4}", r);
    }

    // SVD (SciPy: scipy.linalg.svd)
    let (u_opt, s, vt_opt) = compat::svd(&a.view(), true, true, false, true, "gesdd")?;
    if let (Some(u), Some(vt)) = (u_opt, vt_opt) {
        println!("SVD decomposition successful");
        println!("U:\n{:.4}", u);
        println!("Singular values: {:?}", s);
        println!("Vt:\n{:.4}", vt);
    }

    // Cholesky decomposition (SciPy: scipy.linalg.cholesky)
    let chol_result = compat::cholesky(&a.view(), true, false, true)?;
    println!("Cholesky decomposition (lower):\n{:.4}", chol_result);

    // 4. Eigenvalue Problems
    println!("\n4. Eigenvalue Problems");
    println!("----------------------");

    // Symmetric eigenvalues and eigenvectors (SciPy: scipy.linalg.eigh)
    let (eigenvals, eigenvecs_opt) = compat::eigh(
        &a.view(),
        None,
        false,
        false,
        false,
        false,
        true,
        None,
        None,
        None,
        1,
    )?;

    println!("Eigenvalues: {:?}", eigenvals);
    if let Some(eigenvecs) = eigenvecs_opt {
        println!("Eigenvectors:\n{:.4}", eigenvecs);
    }

    // Eigenvalues only (SciPy: scipy.linalg.eigvalsh)
    let eigenvals_only = compat::eigh(
        &a.view(),
        None,
        false,
        true,
        false,
        false,
        true,
        None,
        None,
        None,
        1,
    )?;
    println!("Eigenvalues only: {:?}", eigenvals_only);

    // 5. Linear System Solvers
    println!("\n5. Linear System Solvers");
    println!("------------------------");

    // General linear system solve (SciPy: scipy.linalg.solve)
    let solve_result =
        compat::compat_solve(&a.view(), &b.view(), false, false, false, true, None, false)?;
    println!("Linear system solution:\n{:.4}", solve_result);

    // Least squares (SciPy: scipy.linalg.lstsq)
    let (lstsq_solution, residuals_opt, lstsq_rank, sing_vals) =
        compat::lstsq(&a.view(), &b.view(), None, false, false, true, None)?;
    println!("Least squares solution:\n{:.4}", lstsq_solution);
    println!("Rank: {}", lstsq_rank);
    println!("Singular values: {:?}", sing_vals);
    if let Some(residuals) = residuals_opt {
        println!("Residuals: {:?}", residuals);
    }

    // 6. Matrix Functions
    println!("\n6. Matrix Functions");
    println!("-------------------");

    // Matrix exponential (SciPy: scipy.linalg.expm)
    let exp_result = compat::expm(&a.view(), None)?;
    println!("Matrix exponential:\n{:.4}", exp_result);

    // Matrix square root (SciPy: scipy.linalg.sqrtm)
    let sqrt_result = compat::sqrtm(&a.view(), None)?;
    println!("Matrix square root:\n{:.4}", sqrt_result);

    // Matrix logarithm (SciPy: scipy.linalg.logm)
    let log_result = compat::logm(&a.view())?;
    println!("Matrix logarithm:\n{:.4}", log_result);

    // General matrix function (SciPy: scipy.linalg.funm)
    let exp_via_funm = compat::funm(&a.view(), "exp", false)?;
    println!("Matrix exp via funm:\n{:.4}", exp_via_funm);

    // 7. Advanced Decompositions
    println!("\n7. Advanced Decompositions");
    println!("--------------------------");

    // RQ decomposition (SciPy: scipy.linalg.rq)
    let (r_rq, q_rq) = compat::rq(&a.view(), false, None, "full", true)?;
    println!("RQ decomposition successful");
    println!("R:\n{:.4}", r_rq);
    println!("Q:\n{:.4}", q_rq);

    // Polar decomposition (SciPy: scipy.linalg.polar)
    let (u_polar, p_polar) = compat::polar(&a.view(), "right")?;
    println!("Polar decomposition (right):");
    println!("U:\n{:.4}", u_polar);
    println!("P:\n{:.4}", p_polar);

    // 8. Utility Functions
    println!("\n8. Utility Functions");
    println!("--------------------");

    // Block diagonal matrix (SciPy: scipy.linalg.block_diag)
    let block1 = array![[1.0, 2.0], [3.0, 4.0]];
    let block2 = array![[5.0]];
    let block3 = array![[6.0, 7.0], [8.0, 9.0]];

    let blocks = [block1.view(), block2.view(), block3.view()];
    let block_diagonal = compat::block_diag(&blocks)?;
    println!("Block diagonal matrix:\n{:.1}", block_diagonal);

    // 9. Error Handling Demo
    println!("\n9. Error Handling Demo");
    println!("----------------------");

    // Try operations that are not yet implemented
    println!("Attempting Schur decomposition (not yet implemented):");
    match compat::schur(&a.view(), "real", None, false, None, true) {
        Ok(_) => println!("Schur decomposition succeeded"),
        Err(e) => println!("Expected error: {}", e),
    }

    println!("Attempting matrix sine (not yet implemented):");
    match compat::sinm(&a.view()) {
        Ok(_) => println!("Matrix sine succeeded"),
        Err(e) => println!("Expected error: {}", e),
    }

    println!("Attempting banded solver (not yet implemented):");
    let dummy_banded = array![[1.0, 2.0], [3.0, 4.0]];
    match compat::solve_banded(&dummy_banded.view(), &b.view(), false, false, true) {
        Ok(_) => println!("Banded solver succeeded"),
        Err(e) => println!("Expected error: {}", e),
    }

    println!("\n=== SciPy API Compatibility Showcase Complete ===");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_scipy_compat_basic_ops() {
        let a = array![[2.0, 1.0], [1.0, 2.0]];

        // Test determinant
        let det_result = compat::det(&a.view(), false, true).unwrap();
        assert!((det_result - 3.0_f64).abs() < 1e-10);

        // Test matrix norm
        let norm_result = compat::norm(&a.view(), Some("fro"), None, false, true).unwrap();
        assert!(norm_result > 0.0);

        // Test matrix rank
        let rank = compat::matrix_rank(&a.view(), None, false, true).unwrap();
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_scipy_compat_decompositions() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];

        // Test LU decomposition
        let (p, l, u) = compat::lu(&a.view(), false, false, true, false).unwrap();
        assert_eq!(p.shape(), [2, 2]);
        assert_eq!(l.shape(), [2, 2]);
        assert_eq!(u.shape(), [2, 2]);

        // Test QR decomposition
        let (q_opt, r) = compat::qr(&a.view(), false, None, "full", false, true).unwrap();
        assert!(q_opt.is_some());
        assert_eq!(r.shape(), [2, 2]);

        // Test SVD
        let (u_opt, s, vt_opt) = compat::svd(&a.view(), true, true, false, true, "gesdd").unwrap();
        assert!(u_opt.is_some());
        assert_eq!(s.len(), 2);
        assert!(vt_opt.is_some());
    }

    #[test]
    fn test_scipy_compatmatrix_functions() {
        let a = array![[1.0, 0.1], [0.1, 1.0]];

        // Test matrix exponential
        let exp_result = compat::expm(&a.view(), None).unwrap();
        assert_eq!(exp_result.shape(), [2, 2]);

        // Test matrix square root
        let sqrt_result = compat::sqrtm(&a.view(), None).unwrap();
        assert_eq!(sqrt_result.shape(), [2, 2]);

        // Test pseudoinverse
        let pinv_result = compat::pinv(&a.view(), None, false, true).unwrap();
        assert_eq!(pinv_result.shape(), [2, 2]);
    }

    #[test]
    fn test_scipy_compat_vector_norms() {
        let v = array![3.0, 4.0];

        // Test vector 2-norm
        let norm_2 = compat::vector_norm(&v.view(), Some(2.0), true).unwrap();
        assert!((norm_2 - 5.0_f64).abs() < 1e-10);

        // Test vector 1-norm
        let norm_1 = compat::vector_norm(&v.view(), Some(1.0), true).unwrap();
        assert!((norm_1 - 7.0_f64).abs() < 1e-10);

        // Test vector infinity norm
        let norm_inf = compat::vector_norm(&v.view(), Some(f64::INFINITY), true).unwrap();
        assert!((norm_inf - 4.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_scipy_compat_utility_functions() {
        let block1 = array![[1.0, 2.0], [3.0, 4.0]];
        let block2 = array![[5.0]];

        let blocks = [block1.view(), block2.view()];
        let block_diag = compat::block_diag(&blocks).unwrap();

        assert_eq!(block_diag.shape(), [3, 3]);
        assert_eq!(block_diag[[0, 0]], 1.0);
        assert_eq!(block_diag[[1, 1]], 4.0);
        assert_eq!(block_diag[[2, 2]], 5.0);
        assert_eq!(block_diag[[0, 2]], 0.0); // Off-diagonal should be zero
    }

    #[test]
    fn test_scipy_compat_error_handling() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];

        // Test that Schur decomposition works with valid input
        assert!(compat::schur(&a.view(), "real", None, false, None, true).is_ok());
        // Test that trigonometric matrix functions work
        assert!(compat::sinm(&a.view()).is_ok());
        assert!(compat::cosm(&a.view()).is_ok());
        assert!(compat::tanm(&a.view()).is_ok());

        let b = array![[1.0], [2.0]];
        assert!(compat::solve_banded(&a.view(), &b.view(), false, false, true).is_err());
    }
}
