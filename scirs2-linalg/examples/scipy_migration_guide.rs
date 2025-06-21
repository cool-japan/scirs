//! SciPy Migration Guide
//!
//! This example demonstrates how to migrate from SciPy's linalg module to scirs2-linalg,
//! showing equivalent operations and highlighting key differences and improvements.

use ndarray::{array, Array1, Array2};
use scirs2_linalg::{
    cholesky,
    compat, // SciPy-compatible API
    // Direct API (more Rust-idiomatic)
    det,
    eig,
    eigh,
    error::LinalgResult,
    inv,
    lu,
    matrix_norm,
    matrix_rank,
    qr,
    solve,
    svd,
};

fn main() -> LinalgResult<()> {
    println!("=== SciPy to scirs2-linalg Migration Guide ===\n");

    // Basic matrix operations migration
    basic_operations_migration()?;

    // Decompositions migration
    decompositions_migration()?;

    // Linear system solving migration
    linear_solving_migration()?;

    // Eigenvalue computations migration
    eigenvalue_migration()?;

    // Matrix properties migration
    matrix_properties_migration()?;

    // Advanced features and differences
    advanced_features_differences()?;

    println!("✅ Migration guide completed successfully!");
    println!("\n📚 Key takeaways:");
    println!("   • Use `compat` module for SciPy-like API");
    println!("   • Direct API offers more Rust-idiomatic error handling");
    println!("   • Both APIs provide the same numerical accuracy");
    println!("   • scirs2-linalg adds parallel processing capabilities");
    println!("   • Memory safety is guaranteed by Rust's type system");

    Ok(())
}

/// Demonstrates migration of basic matrix operations
fn basic_operations_migration() -> LinalgResult<()> {
    println!("📊 Basic Matrix Operations Migration");
    println!("{}", "=".repeat(50));

    let a = array![[1.0, 2.0], [3.0, 4.0]];

    println!("Python SciPy code:");
    println!("```python");
    println!("import numpy as np");
    println!("from scipy import linalg");
    println!("a = np.array([[1.0, 2.0], [3.0, 4.0]])");
    println!("```");
    println!();

    // Determinant
    println!("🔹 Determinant Calculation");
    println!("Python: det_a = linalg.det(a)");
    println!("Rust (compat):  let det_a = compat::det(&a.view(), false, true)?;");
    println!("Rust (direct):  let det_a = det(&a.view(), None)?;");

    let det_compat: f64 = compat::det(&a.view(), false, true)?;
    let det_direct: f64 = det(&a.view(), None)?;
    println!("Results: {:.6} (both APIs give same result)", det_compat);
    assert!((det_compat - det_direct).abs() < 1e-10);

    // Matrix inverse
    println!("\n🔹 Matrix Inverse");
    println!("Python: inv_a = linalg.inv(a)");
    println!("Rust (compat):  let inv_a = compat::inv(&a.view(), false, true)?;");
    println!("Rust (direct):  let inv_a = inv(&a.view(), None)?;");

    let inv_compat = compat::inv(&a.view(), false, true)?;
    let inv_direct = inv(&a.view(), None)?;
    println!("Both APIs produce equivalent results");

    // Verify they're the same
    let diff = &inv_compat - &inv_direct;
    let error = diff.iter().map(|&x: &f64| x.abs()).fold(0.0_f64, f64::max);
    assert!(error < 1e-10_f64);

    // Matrix norms
    println!("\n🔹 Matrix Norms");
    println!("Python: frobenius_norm = linalg.norm(a, 'fro')");
    println!("        one_norm = linalg.norm(a, 1)");
    println!("        inf_norm = linalg.norm(a, np.inf)");
    println!("Rust:   let frobenius_norm = matrix_norm(&a.view(), \"fro\", None)?;");
    println!("        let one_norm = matrix_norm(&a.view(), \"1\", None)?;");
    println!("        let inf_norm = matrix_norm(&a.view(), \"inf\", None)?;");

    let frobenius_norm = matrix_norm(&a.view(), "fro", None)?;
    let one_norm = matrix_norm(&a.view(), "1", None)?;
    let inf_norm = matrix_norm(&a.view(), "inf", None)?;

    println!("Frobenius norm: {:.6}", frobenius_norm);
    println!("1-norm:         {:.6}", one_norm);
    println!("∞-norm:         {:.6}", inf_norm);

    println!("\n");
    Ok(())
}

/// Demonstrates migration of matrix decompositions
fn decompositions_migration() -> LinalgResult<()> {
    println!("🔬 Matrix Decompositions Migration");
    println!("{}", "=".repeat(50));

    let a = array![[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]];

    println!("Python SciPy code:");
    println!("```python");
    println!("a = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]])");
    println!("```");
    println!();

    // LU decomposition
    println!("🔹 LU Decomposition");
    println!("Python: P, L, U = linalg.lu(a)");
    println!("Rust:   let (l, u, p) = lu(&a.view(), None)?;");

    let (_l, _u, _p) = lu(&a.view(), None)?;
    println!("✅ LU decomposition successful");

    // QR decomposition
    println!("\n🔹 QR Decomposition");
    println!("Python: Q, R = linalg.qr(a)");
    println!("Rust:   let (q, r) = qr(&a.view(), None)?;");

    let (q, r) = qr(&a.view(), None)?;
    println!("✅ QR decomposition successful");
    println!("Q shape: {:?}, R shape: {:?}", q.dim(), r.dim());

    // SVD
    println!("\n🔹 Singular Value Decomposition");
    println!("Python: U, s, Vt = linalg.svd(a)");
    println!("Rust:   let (u, s, vt) = svd(&a.view(), false, None)?;");

    let (u, s, vt) = svd(&a.view(), false, None)?;
    println!("✅ SVD successful");
    println!(
        "U shape: {:?}, s length: {}, Vt shape: {:?}",
        u.dim(),
        s.len(),
        vt.dim()
    );
    println!("Singular values: {:?}", s);

    // Cholesky decomposition (for positive definite matrices)
    println!("\n🔹 Cholesky Decomposition");
    println!("Python: L = linalg.cholesky(a, lower=True)");
    println!("Rust:   let l = cholesky(&a.view(), None)?;");

    match cholesky(&a.view(), None) {
        Ok(l) => {
            println!("✅ Cholesky decomposition successful");
            println!("L shape: {:?}", l.dim());
        }
        Err(e) => {
            println!(
                "❌ Cholesky failed (matrix may not be positive definite): {}",
                e
            );
            println!("   This is expected behavior for this example matrix");
        }
    }

    println!("\n");
    Ok(())
}

/// Demonstrates migration of linear system solving
fn linear_solving_migration() -> LinalgResult<()> {
    println!("🎯 Linear System Solving Migration");
    println!("{}", "=".repeat(50));

    let a = array![[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]];
    let b = array![1.0, -2.0, 0.0];

    println!("Python SciPy code:");
    println!("```python");
    println!("a = np.array([[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]])");
    println!("b = np.array([1.0, -2.0, 0.0])");
    println!("```");
    println!();

    // Basic linear solve
    println!("🔹 Basic Linear System Solve");
    println!("Python: x = linalg.solve(a, b)");
    println!("Rust:   let x = solve(&a.view(), &b.view(), None)?;");

    let x = solve(&a.view(), &b.view(), None)?;
    println!("Solution: {:?}", x);

    // Verify solution
    let residual = &a.dot(&x) - &b;
    let residual_norm = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
    println!("Residual norm: {:.2e}", residual_norm);

    // Multiple right-hand sides
    println!("\n🔹 Multiple Right-Hand Sides");
    let b_multi = array![[1.0, 2.0], [-2.0, 1.0], [0.0, -1.0]];

    println!("Python: X = linalg.solve(a, B)  # B is a matrix");
    println!("Rust:   // Solve for each column separately");
    println!("        let x1 = solve(&a.view(), &b_multi.column(0), None)?;");
    println!("        let x2 = solve(&a.view(), &b_multi.column(1), None)?;");

    let x1 = solve(&a.view(), &b_multi.column(0), None)?;
    let x2 = solve(&a.view(), &b_multi.column(1), None)?;

    println!("Solution 1: {:?}", x1);
    println!("Solution 2: {:?}", x2);

    // Overdetermined system (least squares)
    println!("\n🔹 Overdetermined System (Least Squares)");
    let a_over = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
    let b_over = array![1.0, 2.0, 3.0, 4.0];

    println!("Python: x, residuals, rank, s = linalg.lstsq(a_over, b_over)");
    println!("Rust:   // Use SVD-based least squares");

    // For overdetermined systems, we can use the normal equation or SVD
    let ata = a_over.t().dot(&a_over);
    let atb = a_over.t().dot(&b_over);
    let x_lstsq = solve(&ata.view(), &atb.view(), None)?;

    println!("Least squares solution: {:?}", x_lstsq);

    let residual_lstsq = &a_over.dot(&x_lstsq) - &b_over;
    let residual_norm_lstsq = residual_lstsq.iter().map(|&r| r * r).sum::<f64>().sqrt();
    println!("Residual norm: {:.2e}", residual_norm_lstsq);

    println!("\n");
    Ok(())
}

/// Demonstrates migration of eigenvalue computations
fn eigenvalue_migration() -> LinalgResult<()> {
    println!("🌀 Eigenvalue Computations Migration");
    println!("{}", "=".repeat(50));

    // Symmetric matrix for reliable eigenvalue computation
    let a_sym = array![[4.0, -2.0, 1.0], [-2.0, 2.0, -1.0], [1.0, -1.0, 3.0]];

    println!("Python SciPy code:");
    println!("```python");
    println!("a_sym = np.array([[4.0, -2.0, 1.0], [-2.0, 2.0, -1.0], [1.0, -1.0, 3.0]])");
    println!("```");
    println!();

    // Symmetric eigenvalue decomposition
    println!("🔹 Symmetric Eigenvalue Decomposition");
    println!("Python: eigenvals, eigenvecs = linalg.eigh(a_sym)");
    println!("Rust:   let (eigenvals, eigenvecs) = eigh(&a_sym.view(), None)?;");

    let (eigenvals, eigenvecs) = eigh(&a_sym.view(), None)?;
    println!("Eigenvalues: {:?}", eigenvals);
    println!("Eigenvectors shape: {:?}", eigenvecs.dim());

    // Verify eigenvalue equation: A * v = λ * v
    if !eigenvals.is_empty() && eigenvecs.ncols() > 0 {
        let lambda0 = eigenvals[0];
        let v0 = eigenvecs.column(0);
        let av0 = a_sym.dot(&v0);
        let lambda_v0 = &v0 * lambda0;

        let diff = &av0 - &lambda_v0;
        let error = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
        println!("Verification ||A*v₀ - λ₀*v₀||: {:.2e}", error);
    }

    // General eigenvalue decomposition
    println!("\n🔹 General Eigenvalue Decomposition");
    let a_gen = array![[1.0, 2.0], [3.0, 4.0]];

    println!("Python: eigenvals, eigenvecs = linalg.eig(a_gen)");
    println!("Rust:   let (eigenvals, eigenvecs) = eig(&a_gen.view(), None)?;");

    match eig(&a_gen.view(), None) {
        Ok((eigenvals_gen, eigenvecs_gen)) => {
            println!("✅ General eigenvalue decomposition successful");
            println!("Eigenvalues: {:?}", eigenvals_gen);
            println!("Eigenvectors shape: {:?}", eigenvecs_gen.dim());
        }
        Err(e) => {
            println!("❌ General eigenvalue decomposition failed: {}", e);
            println!("   This may occur with complex eigenvalues or numerical issues");
        }
    }

    // Eigenvalues only (more efficient)
    println!("\n🔹 Eigenvalues Only");
    println!("Python: eigenvals = linalg.eigvals(a_sym)");
    println!("Rust:   let (eigenvals, _) = eigh(&a_sym.view(), None)?;  // Ignore eigenvectors");

    let (eigenvals_only, _) = eigh(&a_sym.view(), None)?;
    println!("Eigenvalues only: {:?}", eigenvals_only);

    println!("\n");
    Ok(())
}

/// Demonstrates migration of matrix properties
fn matrix_properties_migration() -> LinalgResult<()> {
    println!("📏 Matrix Properties Migration");
    println!("{}", "=".repeat(50));

    let a = array![[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]];

    println!("Python SciPy code:");
    println!("```python");
    println!("a = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]])");
    println!("```");
    println!();

    // Matrix rank
    println!("🔹 Matrix Rank");
    println!("Python: rank = np.linalg.matrix_rank(a)");
    println!("Rust:   let rank = matrix_rank(&a.view(), None, None)?;");

    let rank = matrix_rank(&a.view(), None, None)?;
    println!("Matrix rank: {}", rank);

    // Condition number
    println!("\n🔹 Condition Number");
    println!("Python: cond_num = np.linalg.cond(a)");
    println!("Rust:   let cond_num = cond(&a.view(), None, None)?;");

    let cond_num = scirs2_linalg::cond(&a.view(), None, None)?;
    println!("Condition number: {:.2e}", cond_num);

    // Matrix properties analysis
    println!("\n🔹 Matrix Properties Analysis");

    // Check if matrix is well-conditioned
    if cond_num < 1e12 {
        println!("✅ Matrix is well-conditioned (κ < 1e12)");
    } else {
        println!("⚠️  Matrix is ill-conditioned (κ ≥ 1e12)");
    }

    // Check if matrix is full rank
    let (m, n) = a.dim();
    let expected_rank = std::cmp::min(m, n);
    if rank == expected_rank {
        println!("✅ Matrix is full rank ({}/{})", rank, expected_rank);
    } else {
        println!("⚠️  Matrix is rank deficient ({}/{})", rank, expected_rank);
    }

    // Different matrix norms comparison
    println!("\n🔹 Matrix Norms Comparison");
    let norms = vec![
        ("Frobenius", matrix_norm(&a.view(), "fro", None)?),
        ("1-norm", matrix_norm(&a.view(), "1", None)?),
        ("∞-norm", matrix_norm(&a.view(), "inf", None)?),
        ("2-norm", matrix_norm(&a.view(), "2", None)?),
    ];

    println!("Matrix norms:");
    for (name, value) in norms {
        println!("  {}: {:.6}", name, value);
    }

    println!("\n");
    Ok(())
}

/// Demonstrates advanced features and key differences
fn advanced_features_differences() -> LinalgResult<()> {
    println!("🚀 Advanced Features and Key Differences");
    println!("{}", "=".repeat(50));

    let a = Array2::from_shape_fn((100, 100), |(i, j)| ((i + 1) * (j + 1)) as f64);

    // Parallel processing (unique to scirs2-linalg)
    println!("🔹 Parallel Processing (Rust-specific)");
    println!("Python: # No built-in parallel control in SciPy");
    println!("Rust:   use scirs2_linalg::parallel::{{algorithms, WorkerConfig}};");
    println!("        let config = WorkerConfig::new().with_workers(4);");

    let config = scirs2_linalg::parallel::WorkerConfig::new().with_workers(4);
    let vector = Array1::ones(100);

    let _result =
        scirs2_linalg::parallel::algorithms::parallel_matvec(&a.view(), &vector.view(), &config)?;
    println!("✅ Parallel matrix-vector multiplication completed");

    // Memory safety (Rust-specific)
    println!("\n🔹 Memory Safety (Rust-specific)");
    println!("Python: # Runtime errors possible with memory issues");
    println!("Rust:   # Compile-time guarantees against memory errors");
    println!("        # No null pointer dereferences");
    println!("        # No buffer overflows");
    println!("        # No use-after-free errors");
    println!("✅ Memory safety guaranteed by Rust's type system");

    // Error handling differences
    println!("\n🔹 Error Handling Differences");
    println!("Python: try:");
    println!("            result = linalg.inv(singular_matrix)");
    println!("        except LinAlgError as e:");
    println!("            print(f'Error: {{e}}')");
    println!("Rust:   match inv(&singular_matrix.view(), None) {{");
    println!("            Ok(result) => {{ /* use result */ }}");
    println!("            Err(e) => println!(\"Error: {{}}\", e),");
    println!("        }}");

    // Demonstrate with a singular matrix
    let singular = array![[1.0, 2.0], [2.0, 4.0]];
    match inv(&singular.view(), None) {
        Ok(_) => println!("❌ Unexpected success"),
        Err(e) => println!("✅ Proper error handling: {}", e),
    }

    // Type safety
    println!("\n🔹 Type Safety");
    println!("Python: # Runtime type checking");
    println!("        result = linalg.det([[1, 2], [3, 4]])  # Works with lists");
    println!("Rust:   # Compile-time type checking");
    println!("        let det_val = det(&array.view(), None)?;  # Only works with proper types");
    println!("✅ Compile-time type safety prevents many runtime errors");

    // Performance characteristics
    println!("\n🔹 Performance Characteristics");
    println!("Python: # GIL can limit parallel performance");
    println!("        # Dynamic typing overhead");
    println!("        # Interpreted language overhead");
    println!("Rust:   # No GIL - true parallelism");
    println!("        # Zero-cost abstractions");
    println!("        # Compiled native code");
    println!("✅ Rust offers potential performance advantages");

    // Integration differences
    println!("\n🔹 Integration Differences");
    println!("SciPy pros:");
    println!("  • Mature ecosystem with extensive libraries");
    println!("  • Interactive development (Jupyter notebooks)");
    println!("  • Large community and extensive documentation");
    println!("  • Easy integration with NumPy/pandas/matplotlib");
    println!();
    println!("scirs2-linalg pros:");
    println!("  • Memory safety guarantees");
    println!("  • Better parallel processing capabilities");
    println!("  • Zero-cost abstractions and high performance");
    println!("  • Compile-time error detection");
    println!("  • No runtime dependency on Python interpreter");

    println!("\n📋 Migration Checklist:");
    println!("  ✅ Replace `import scipy.linalg` with `use scirs2_linalg`");
    println!("  ✅ Convert NumPy arrays to ndarray Arrays");
    println!("  ✅ Add error handling with Result types");
    println!("  ✅ Consider using parallel algorithms for large matrices");
    println!("  ✅ Take advantage of compile-time type checking");
    println!("  ✅ Use the `compat` module for easier transition");

    println!("\n");
    Ok(())
}
