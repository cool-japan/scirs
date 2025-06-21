//! Tests for matrix functions (sqrtm, logm, powm)

use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

#[test]
fn test_matrix_sqrt() {
    ag::run(|g| {
        // Test with a simple 2x2 positive definite matrix
        let a = convert_to_tensor(array![[4.0_f32, 0.0], [0.0, 9.0]], g);
        let sqrt_a = sqrtm(&a);
        let sqrt_result = sqrt_a.eval(g).unwrap();

        // For diagonal matrix, sqrt should be [[2, 0], [0, 3]]
        assert!((sqrt_result[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((sqrt_result[[1, 1]] - 3.0).abs() < 1e-5);
        assert!(sqrt_result[[0, 1]].abs() < 1e-5);
        assert!(sqrt_result[[1, 0]].abs() < 1e-5);

        // Verify: sqrt(A) * sqrt(A) = A
        let squared = matmul(&sqrt_a, &sqrt_a);
        let squared_result = squared.eval(g).unwrap();
        let a_result = a.eval(g).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (squared_result[[i, j]] - a_result[[i, j]]).abs() < 1e-4,
                    "sqrt(A)^2 != A at [{}, {}]: {} != {}",
                    i,
                    j,
                    squared_result[[i, j]],
                    a_result[[i, j]]
                );
            }
        }

        // Test with a symmetric positive definite matrix
        let b = convert_to_tensor(array![[4.0_f32, 1.0], [1.0, 3.0]], g);
        let sqrt_b = sqrtm(&b);
        let sqrt_b_result = sqrt_b.eval(g).unwrap();

        // Verify it's computed successfully
        assert_eq!(sqrt_b_result.shape(), &[2, 2]);

        // Verify sqrt(B) * sqrt(B) ≈ B
        let b_squared = matmul(&sqrt_b, &sqrt_b);
        let b_squared_result = b_squared.eval(g).unwrap();
        let b_result = b.eval(g).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (b_squared_result[[i, j]] - b_result[[i, j]]).abs() < 1e-4,
                    "sqrt(B)^2 != B at [{}, {}]",
                    i,
                    j
                );
            }
        }
    });
}

#[test]
fn test_matrix_log() {
    ag::run(|g| {
        // Test with a simple diagonal matrix
        let a = convert_to_tensor(
            array![
                [std::f32::consts::E, 0.0],
                [0.0, std::f32::consts::E.powi(2)]
            ],
            g,
        ); // e and e^2
        let log_a = logm(&a);
        let log_result = log_a.eval(g).unwrap();

        // For diagonal matrix with e and e^2, log should be [[1, 0], [0, 2]]
        assert!((log_result[[0, 0]] - 1.0).abs() < 1e-4);
        assert!((log_result[[1, 1]] - 2.0).abs() < 1e-4);
        assert!(log_result[[0, 1]].abs() < 1e-5);
        assert!(log_result[[1, 0]].abs() < 1e-5);

        // Test with identity matrix (log(I) = 0)
        let i = convert_to_tensor(array![[1.0_f32, 0.0], [0.0, 1.0]], g);
        let log_i = logm(&i);
        let log_i_result = log_i.eval(g).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    log_i_result[[i, j]].abs() < 1e-5,
                    "log(I) != 0 at [{}, {}]: {}",
                    i,
                    j,
                    log_i_result[[i, j]]
                );
            }
        }
    });
}

#[test]
fn test_matrix_power() {
    ag::run(|g| {
        // Test integer powers
        let a = convert_to_tensor(array![[2.0_f32, 1.0], [0.0, 3.0]], g);

        // Test A^0 = I
        let a0 = powm(&a, 0.0);
        let a0_result = a0.eval(g).unwrap();
        assert!((a0_result[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((a0_result[[1, 1]] - 1.0).abs() < 1e-5);
        assert!(a0_result[[0, 1]].abs() < 1e-5);
        assert!(a0_result[[1, 0]].abs() < 1e-5);

        // Test A^1 = A
        let a1 = powm(&a, 1.0);
        let a1_result = a1.eval(g).unwrap();
        let a_result = a.eval(g).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((a1_result[[i, j]] - a_result[[i, j]]).abs() < 1e-5);
            }
        }

        // Test A^2
        let a2 = powm(&a, 2.0);
        let a2_result = a2.eval(g).unwrap();
        let a2_expected = matmul(&a, &a);
        let a2_expected_result = a2_expected.eval(g).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a2_result[[i, j]] - a2_expected_result[[i, j]]).abs() < 1e-4,
                    "A^2 mismatch at [{}, {}]",
                    i,
                    j
                );
            }
        }

        // Test A^(-1) = inverse(A)
        let a_inv = powm(&a, -1.0);
        let _a_inv_result = a_inv.eval(g).unwrap();
        let identity = matmul(&a, &a_inv);
        let identity_result = identity.eval(g).unwrap();

        // Check if A * A^(-1) = I
        assert!((identity_result[[0, 0]] - 1.0).abs() < 1e-4);
        assert!((identity_result[[1, 1]] - 1.0).abs() < 1e-4);
        assert!(identity_result[[0, 1]].abs() < 1e-4);
        assert!(identity_result[[1, 0]].abs() < 1e-4);

        // Test fractional power with symmetric positive definite matrix
        let b = convert_to_tensor(array![[4.0_f32, 0.0], [0.0, 9.0]], g);
        let b_half = powm(&b, 0.5);
        let b_half_result = b_half.eval(g).unwrap();

        // For diagonal matrix, A^0.5 should be [[2, 0], [0, 3]]
        assert!((b_half_result[[0, 0]] - 2.0).abs() < 1e-4);
        assert!((b_half_result[[1, 1]] - 3.0).abs() < 1e-4);
    });
}

#[test]
fn test_matrix_functions_consistency() {
    ag::run(|g| {
        // Test that exp(log(A)) = A for positive definite matrix
        let a = convert_to_tensor(array![[2.0_f32, 0.5], [0.5, 3.0]], g);
        let log_a = logm(&a);
        let exp_log_a = matrix_exp(&log_a);
        let result = exp_log_a.eval(g).unwrap();
        let a_result = a.eval(g).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (result[[i, j]] - a_result[[i, j]]).abs() < 1e-3,
                    "exp(log(A)) != A at [{}, {}]: {} != {}",
                    i,
                    j,
                    result[[i, j]],
                    a_result[[i, j]]
                );
            }
        }

        // Test that sqrt(A) = A^0.5
        let sqrt_a = sqrtm(&a);
        let a_half = powm(&a, 0.5);
        let sqrt_result = sqrt_a.eval(g).unwrap();
        let half_result = a_half.eval(g).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (sqrt_result[[i, j]] - half_result[[i, j]]).abs() < 1e-3,
                    "sqrt(A) != A^0.5 at [{}, {}]",
                    i,
                    j
                );
            }
        }
    });
}

fn main() {
    println!("Running tests for matrix functions (sqrtm, logm, powm)...");
}
