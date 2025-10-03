#[cfg(test)]
mod tests {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::extended_precision::{extended_matmul, extended_matvec, extended_solve};

    #[test]
    fn test_extended_matmul() {
        // Create a matrix in f32 precision
        let a = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let b = array![[9.0_f32, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];

        // Compute with extended precision
        let c = extended_matmul::<_, f64>(&a.view(), &b.view()).unwrap();

        // Compute with standard precision for comparison
        let c_standard = a.dot(&b);

        // Verify results match within epsilon
        for i in 0..3 {
            for j in 0..3 {
                assert!((c[[i, j]] - c_standard[[i, j]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_extended_matvec() {
        // Create a matrix and vector in f32 precision
        let a = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let x = array![1.0_f32, 2.0, 3.0];

        // Compute with extended precision
        let y = extended_matvec::<_, f64>(&a.view(), &x.view()).unwrap();

        // Compute with standard precision for comparison
        let y_standard = a.dot(&x);

        // Verify results match within epsilon
        for i in 0..3 {
            assert!((y[i] - y_standard[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_extended_solve() {
        let a = array![[4.0_f32, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 2.0, 5.0]];

        let b = array![6.0_f32, 6.0, 8.0];

        // Solve the system with extended precision
        let x = extended_solve::<_, f64>(&a.view(), &b.view()).unwrap();

        // Verify A*x ≈ b
        let ax = a.dot(&x);

        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-5);
        }
    }
}
