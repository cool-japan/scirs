use ndarray::array;
use scirs2_linalg::eigh;

fn main() {
    let symmetric_matrix = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];

    println!("Testing 3x3 matrix eigenvalue precision:");
    println!("Matrix: {:?}", symmetric_matrix);

    match eigh(&symmetric_matrix.view(), None) {
        Ok((eigenvals, eigenvecs)) => {
            println!("Eigenvalues: {:?}", eigenvals);

            // Check A * V = V * Λ with high precision
            let av = symmetric_matrix.dot(&eigenvecs);
            let vl = eigenvecs.dot(&ndarray::Array2::from_diag(&eigenvals));

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

            // Check orthogonality
            let vtv = eigenvecs.t().dot(&eigenvecs);
            let identity = ndarray::Array2::<f64>::eye(3);

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

            // Check if we meet the 1e-10 precision requirement
            if max_error < 1e-10 && max_ortho_error < 1e-10 {
                println!("✅ SUCCESS: All precision requirements met!");
            } else {
                println!("❌ PRECISION ISSUE:");
                if max_error >= 1e-10 {
                    println!(
                        "   A*V = V*Λ error {:.2e} exceeds 1e-10 tolerance",
                        max_error
                    );
                }
                if max_ortho_error >= 1e-10 {
                    println!(
                        "   Orthogonality error {:.2e} exceeds 1e-10 tolerance",
                        max_ortho_error
                    );
                }
            }
        }
        Err(e) => {
            println!("Error computing eigenvalues: {:?}", e);
        }
    }
}
