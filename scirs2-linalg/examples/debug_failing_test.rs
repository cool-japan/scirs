use ndarray::array;
use num_traits::float::Float;
use scirs2_linalg::compat;

fn arrays_consistent(a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>, tol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < tol || (x.is_nan() && y.is_nan()))
}

fn main() {
    println!("Testing the specific failing 3x3 matrix...");

    let symmetric_matrix = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    println!("Matrix: {:?}", symmetric_matrix);

    // Test eigenvalues + eigenvectors
    match compat::eigh(
        &symmetric_matrix.view(),
        None,
        false,
        false, // Get eigenvectors
        false,
        false,
        true,
        None,
        None,
        None,
        1,
    ) {
        Ok((eigenvals_with_vecs, eigenvecs_opt)) => {
            println!("Eigenvalues: {:?}", eigenvals_with_vecs);
            if let Some(eigenvecs) = eigenvecs_opt {
                println!("Eigenvectors: {:?}", eigenvecs);

                // Test A*V = V*Λ
                let av = symmetric_matrix.dot(&eigenvecs);
                let vl = eigenvecs.dot(&ndarray::Array2::from_diag(&eigenvals_with_vecs));
                println!("A*V: {:?}", av);
                println!("V*Λ: {:?}", vl);
                let diff = &av - &vl;
                println!("Difference: {:?}", diff);
                println!(
                    "Max absolute difference: {}",
                    diff.iter().map(|x| x.abs()).fold(0.0, f64::max)
                );

                // Test if arrays are consistent
                let consistent = arrays_consistent(&av, &vl, 1e-10);
                println!("Arrays consistent with tolerance 1e-10: {}", consistent);

                // Try with a more relaxed tolerance
                let consistent_relaxed = arrays_consistent(&av, &vl, 1e-8);
                println!(
                    "Arrays consistent with tolerance 1e-8: {}",
                    consistent_relaxed
                );

                // Check orthogonality: V^T * V = I
                let vtv = eigenvecs.t().dot(&eigenvecs);
                let identity = ndarray::Array2::<f64>::eye(3);
                println!("V^T * V: {:?}", vtv);
                println!("Identity: {:?}", identity);
                let orth_diff = &vtv - &identity;
                println!("Orthogonality difference: {:?}", orth_diff);
                println!(
                    "Max orthogonality error: {}",
                    orth_diff.iter().map(|x| x.abs()).fold(0.0, f64::max)
                );
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }
}
