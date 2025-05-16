use ndarray::{array, Array2};
use scirs2_linalg::compat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // This example demonstrates the SciPy-compatible API
    println!("SciPy-compatible API example for scirs2-linalg");

    // Create a test matrix
    let a: Array2<f64> = array![[4.0, 2.0], [2.0, 5.0]];
    println!("Matrix A:");
    println!("{}", a);

    // Compute determinant with SciPy-style parameters
    let det = compat::det(&a.view(), false, true)?;
    println!("\nDeterminant (scipy.linalg.det style): {}", det);

    // Compute inverse
    let inv = compat::inv(&a.view(), false, true)?;
    println!("\nInverse (scipy.linalg.inv style):");
    println!("{}", inv);

    // LU decomposition
    let (p, l, u) = compat::lu(&a.view(), false, false, true, false)?;
    println!("\nLU decomposition (scipy.linalg.lu style):");
    println!("P =\n{}", p);
    println!("L =\n{}", l);
    println!("U =\n{}", u);

    // QR decomposition
    let (q, r) = compat::qr(&a.view(), false, None, "full", false, true)?;
    println!("\nQR decomposition (scipy.linalg.qr style):");
    println!("Q =\n{}", q.unwrap());
    println!("R =\n{}", r);

    // SVD decomposition
    let (u, s, vt) = compat::svd(&a.view(), true, true, false, true, "gesdd")?;
    println!("\nSVD decomposition (scipy.linalg.svd style):");
    println!("U =\n{}", u.unwrap());
    println!("S =\n{}", s);
    println!("Vt =\n{}", vt.unwrap());

    // Cholesky decomposition (for positive definite matrix)
    let l = compat::cholesky(&a.view(), true, false, true)?;
    println!("\nCholesky decomposition (scipy.linalg.cholesky style):");
    println!("L =\n{}", l);

    // Solve linear system
    let b: Array2<f64> = array![[1.0], [2.0]];
    println!("\nVector b:");
    println!("{}", b);

    let x = compat::compat_solve(&a.view(), &b.view(), false, false, false, true, None, false)?;
    println!("\nSolution to Ax = b (scipy.linalg.solve style):");
    println!("{}", x);

    // Eigenvalue decomposition (for symmetric matrix)
    let (eigenvalues, eigenvectors) = compat::eigh(
        &a.view(),
        None,
        true,
        false,
        false,
        false,
        true,
        None,
        None,
        None,
        1,
    )?;
    println!("\nEigenvalue decomposition for symmetric matrix (scipy.linalg.eigh style):");
    println!("Eigenvalues =\n{}", eigenvalues);
    if let Some(v) = eigenvectors {
        println!("Eigenvectors =\n{}", v);
    }

    println!("\nNote: The compat module provides SciPy-compatible function signatures");
    println!("while delegating to the efficient scirs2-linalg implementations.");
    println!("Some parameters are ignored (like overwrite_a) but the API matches SciPy.");

    Ok(())
}
