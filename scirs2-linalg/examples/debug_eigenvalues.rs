use scirs2_core::ndarray::array;
use scirs2_linalg::compat;

#[allow(dead_code)]
fn main() {
    println!("Testing eigenvalue computation...");

    // Test 1: Diagonal matrix (should have eigenvalues [1, 3, 5])
    let diagmatrix = array![[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
    println!("Diagonal matrix: {:?}", diagmatrix);

    match compat::eigh(
        &diagmatrix.view(),
        None,
        false,
        false, // Get eigenvectors too
        false,
        false,
        true,
        None,
        None,
        None,
        1,
    ) {
        Ok((eigenvals, eigenvecs_opt)) => {
            println!("Eigenvalues: {:?}", eigenvals);
            if let Some(eigenvecs) = eigenvecs_opt {
                println!("Eigenvectors shape: {:?}", eigenvecs.shape());
                println!("Eigenvectors: {:?}", eigenvecs);

                // Test A*V = V*Λ
                let av = diagmatrix.dot(&eigenvecs);
                let vl = eigenvecs.dot(&scirs2_core::ndarray::Array2::from_diag(&eigenvals));
                println!("A*V: {:?}", av);
                println!("V*Λ: {:?}", vl);
                println!("Difference: {:?}", &av - &vl);
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }

    // Test 2: Simple 2x2 symmetric matrix
    let simplematrix = array![[2.0, 1.0], [1.0, 3.0]];
    println!("\nSimple 2x2 matrix: {:?}", simplematrix);

    match compat::eigh(
        &simplematrix.view(),
        None,
        false,
        false, // Get eigenvectors too
        false,
        false,
        true,
        None,
        None,
        None,
        1,
    ) {
        Ok((eigenvals, eigenvecs_opt)) => {
            println!("Eigenvalues: {:?}", eigenvals);
            if let Some(eigenvecs) = eigenvecs_opt {
                println!("Eigenvectors: {:?}", eigenvecs);

                // Test A*V = V*Λ
                let av = simplematrix.dot(&eigenvecs);
                let vl = eigenvecs.dot(&scirs2_core::ndarray::Array2::from_diag(&eigenvals));
                println!("A*V: {:?}", av);
                println!("V*Λ: {:?}", vl);
                println!("Difference: {:?}", &av - &vl);
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }
}
