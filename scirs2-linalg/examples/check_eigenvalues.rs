use scirs2_core::ndarray::array;

#[allow(dead_code)]
fn main() {
    // Matrix: [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
    // Characteristic polynomial: det(A - λI) = 0
    // (4-λ)(3-λ)(2-λ) - (1)(1)(2-λ) - (1)(1)(4-λ)
    // = (4-λ)(3-λ)(2-λ) - (2-λ) - (4-λ)
    // = (4-λ)(3-λ)(2-λ) - (2-λ) - (4-λ)

    let matrix = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];

    // Let's verify using trace and determinant
    let trace = matrix.diag().sum(); // sum of eigenvalues
    let det = matrix[[0, 0]] * (matrix[[1, 1]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 1]])
        - matrix[[0, 1]] * (matrix[[1, 0]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 0]])
        + matrix[[0, 2]] * (matrix[[1, 0]] * matrix[[2, 1]] - matrix[[1, 1]] * matrix[[2, 0]]);

    println!("Matrix: {:?}", matrix);
    println!("Trace (sum of eigenvalues): {}", trace);
    println!("Determinant (product of eigenvalues): {}", det);

    // For a 3x3 symmetric matrix, we can solve the characteristic equation
    // Let's try some test values
    for lambda in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] {
        let char_val =
            (4.0 - lambda) * (3.0 - lambda) * (2.0 - lambda) - (2.0 - lambda) - (4.0 - lambda);
        println!("λ = {}: characteristic value = {}", lambda, char_val);
    }
}
