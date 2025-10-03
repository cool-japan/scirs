use scirs2_core::ndarray::{array, Array2};
use scirs2_linalg::lapack::svd;

#[test]
#[allow(dead_code)]
fn debug_svd_implementation() {
    // Test with a simple 2x2 matrix first
    let a = array![[1.0_f64, 0.0], [0.0, 1.0]];

    println!("Testing SVD with 2x2 identity matrix");
    match svd(&a.view(), false) {
        Ok(result) => {
            println!("U shape: {:?}", result.u.shape());
            println!("S: {:?}", result.s);
            println!("Vt shape: {:?}", result.vt.shape());
        }
        Err(e) => {
            println!("SVD failed with error: {:?}", e);
        }
    }

    // Now test with 3x3 identity
    println!("\nTesting SVD with 3x3 identity matrix");
    let a3: Array2<f64> = Array2::eye(3);

    // Test by directly computing A^T * A
    let a_t = a3.t();
    let ata = a_t.dot(&a3);
    println!("A^T * A shape: {:?}", ata.shape());
    println!("A^T * A:\n{:?}", ata);

    // Now try calling eigh directly on this
    use scirs2_linalg::eigh;
    match eigh(&ata.view(), None) {
        Ok((eigenvalues, eigenvectors)) => {
            println!("Eigenvalues: {:?}", eigenvalues);
            println!("Eigenvectors shape: {:?}", eigenvectors.shape());
        }
        Err(e) => {
            println!("eigh failed with error: {:?}", e);
        }
    }
}
