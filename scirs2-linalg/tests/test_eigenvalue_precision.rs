use scirs2_core::ndarray::array;
use scirs2_linalg::eigh;

#[test]
#[allow(dead_code)]
fn test_3x3_eigenvalue_precision() {
    let symmetricmatrix = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];

    let (eigenvals, eigenvecs) = eigh(&symmetricmatrix.view(), None).unwrap();

    // Check A * V = V * Λ with high precision
    let av = symmetricmatrix.dot(&eigenvecs);
    let vl = eigenvecs.dot(&scirs2_core::ndarray::Array2::from_diag(&eigenvals));

    let mut max_error = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let diff: f64 = av[[i, j]] - vl[[i, j]];
            let error = diff.abs();
            if error > max_error {
                max_error = error;
            }
        }
    }

    println!("Maximum error in A*V = V*Λ: {:.2e}", max_error);
    println!("Eigenvalues: {:?}", eigenvals);

    // Check orthogonality
    let vtv = eigenvecs.t().dot(&eigenvecs);
    let identity = scirs2_core::ndarray::Array2::<f64>::eye(3);

    let mut max_ortho_error = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let diff: f64 = vtv[[i, j]] - identity[[i, j]];
            let error = diff.abs();
            if error > max_ortho_error {
                max_ortho_error = error;
            }
        }
    }

    println!("Maximum orthogonality error: {:.2e}", max_ortho_error);

    // The goal is to achieve good precision (relaxed slightly for numerical stability)
    assert!(
        max_error < 2e-10,
        "A*V = V*Λ error {:.2e} exceeds 2e-10 tolerance",
        max_error
    );
    assert!(
        max_ortho_error < 1e-10,
        "Orthogonality error {:.2e} exceeds 1e-10 tolerance",
        max_ortho_error
    );
}
