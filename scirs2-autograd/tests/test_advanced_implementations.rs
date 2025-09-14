//! Tests for advanced linear algebra implementations

use ag::tensor_ops::PreconditionerType;
use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

#[test]
#[allow(dead_code)]
fn test_advanced_svd() {
    ag::run(|g| {
        // Test Jacobi SVD
        let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]], g);
        let _u_s_vt = svd_jacobi(&a, false);

        // For now, just test that we can create the operations without panic
        // The extraction operators need special handling in the framework

        // Test randomized SVD
        let _u_r_s_r_vt_r = randomized_svd(&a, 2, 2, 3);

        // Operations created successfully
        println!("SVD operations created successfully");
    });
}

#[test]
#[allow(dead_code)]
fn test_iterative_solvers() {
    ag::run(|g| {
        // Test Conjugate Gradient
        let a = convert_to_tensor(array![[4.0_f32, 1.0], [1.0, 3.0]], g); // Positive definite
        let b = convert_to_tensor(array![1.0_f32, 2.0], g);

        let x_cg = conjugate_gradient_solve(&a, &b, 100, Some(1e-6));
        let result_cg = x_cg.eval(g).unwrap();
        assert_eq!(result_cg.shape(), &[2]);

        // Verify Ax = b
        let ax = matmul(a, reshape(x_cg, &[2, 1]));
        let ax_flat = reshape(ax, &[2]);
        let diff = sub(ax_flat, b);
        let error = normfro(&reshape(diff, &[2, 1]));
        let error_val = error.eval(g).unwrap()[ndarray::IxDyn(&[])];
        assert!(error_val < 1e-4, "CG solver error too large: {}", error_val);

        // Test GMRES
        let x_gmres = gmres_solve(&a, &b, 100, 10, Some(1e-6));
        let result_gmres = x_gmres.eval(g).unwrap();
        assert_eq!(result_gmres.shape(), &[2]);

        // Test BiCGSTAB
        let x_bicgstab = bicgstab_solve(&a, &b, 100, Some(1e-6));
        let result_bicgstab = x_bicgstab.eval(g).unwrap();
        assert_eq!(result_bicgstab.shape(), &[2]);
    });
}

#[test]
#[allow(dead_code)]
fn test_matrix_trig_functions() {
    ag::run(|g| {
        // Test matrix sine
        let a = convert_to_tensor(array![[0.0_f32, 1.0], [-1.0, 0.0]], g);
        let sin_a = sinm(&a);
        let sin_result = sin_a.eval(g).unwrap();
        assert_eq!(sin_result.shape(), &[2, 2]);

        // Test matrix cosine
        let cos_a = cosm(&a);
        let cos_result = cos_a.eval(g).unwrap();
        assert_eq!(cos_result.shape(), &[2, 2]);

        // Test identity: sin²(A) + cos²(A) = I (approximately for small matrices)
        let sin2 = matmul(sin_a, sin_a);
        let cos2 = matmul(cos_a, cos_a);
        let sum = add(sin2, cos2);
        let sum_result = sum.eval(g).unwrap();

        // Should be close to identity for this specific matrix
        assert!((sum_result[[0, 0]] - 1.0).abs() < 0.1);
        assert!((sum_result[[1, 1]] - 1.0).abs() < 0.1);

        // Test hyperbolic functions
        let sinh_a = sinhm(&a);
        let cosh_a = coshm(&a);

        assert_eq!(sinh_a.eval(g).unwrap().shape(), &[2, 2]);
        assert_eq!(cosh_a.eval(g).unwrap().shape(), &[2, 2]);
    });
}

#[test]
#[allow(dead_code)]
fn test_generalized_eigen() {
    ag::run(|g| {
        // Test generalized eigenvalue problem
        let a = convert_to_tensor(array![[2.0_f32, 1.0], [1.0, 2.0]], g);
        let b = convert_to_tensor(array![[1.0_f32, 0.0], [0.0, 1.0]], g);

        let _eigenvalues_eigenvectors = generalized_eigen(&a, &b);

        // For now, just test that we can create the operations without panic
        println!("Generalized eigenvalue operation created successfully");
    });
}

#[test]
#[allow(dead_code)]
fn test_qr_with_pivoting() {
    ag::run(|g| {
        // Test QR decomposition with column pivoting
        let a = convert_to_tensor(array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], g);
        let _q_r_p = qr_pivot(&a);

        // For now, just test that we can create the operations without panic
        println!("QR pivot operation created successfully");
    });
}

#[test]
#[allow(dead_code)]
fn test_preconditioned_cg() {
    ag::run(|g| {
        // Test Preconditioned Conjugate Gradient
        let a = convert_to_tensor(array![[4.0_f32, 1.0], [1.0, 3.0]], g);
        let b = convert_to_tensor(array![1.0_f32, 2.0], g);

        // Test with Jacobi preconditioner
        let x_pcg = pcg_solve(&a, &b, 100, Some(1e-6), PreconditionerType::Jacobi);

        let result_pcg = x_pcg.eval(g).unwrap();
        assert_eq!(result_pcg.shape(), &[2]);

        // Verify solution
        let ax = matmul(a, reshape(x_pcg, &[2, 1]));
        let ax_flat = reshape(ax, &[2]);
        let diff = sub(ax_flat, b);
        let error = normfro(&reshape(diff, &[2, 1]));
        let error_val = error.eval(g).unwrap()[ndarray::IxDyn(&[])];
        assert!(
            error_val < 1e-4,
            "PCG solver error too large: {}",
            error_val
        );
    });
}

#[test]
#[allow(dead_code)]
fn test_matrix_sign_function() {
    ag::run(|g| {
        // Test matrix sign function
        let a = convert_to_tensor(array![[1.0_f32, 2.0], [0.0, -1.0]], g);
        let sign_a = signm(&a);
        let sign_result = sign_a.eval(g).unwrap();

        assert_eq!(sign_result.shape(), &[2, 2]);

        // Test property: sign(A)² = I for non-singular matrices
        let sign2 = matmul(sign_a, sign_a);
        let sign2_result = sign2.eval(g).unwrap();

        assert!((sign2_result[[0, 0]] - 1.0).abs() < 0.1);
        assert!((sign2_result[[1, 1]] - 1.0).abs() < 0.1);
    });
}

#[test]
#[allow(dead_code)]
fn test_general_matrix_function() {
    ag::run(|g| {
        // Test general matrix function with exponential on a symmetric matrix
        // Using a symmetric matrix since the current implementation only works for symmetric matrices
        let a = convert_to_tensor(array![[0.0_f32, 1.0], [1.0, 0.0]], g);
        let exp_a = funm(&a, |x: f32| x.exp(), "exp");
        let exp_result = exp_a.eval(g).unwrap();

        assert_eq!(exp_result.shape(), &[2, 2]);

        // For this symmetric matrix [[0, 1], [1, 0]], eigenvalues are ±1
        // exp(A) will have specific values based on eigendecomposition
        // Just verify the result is not zero and has reasonable values
        let sum = exp_result.iter().map(|x| x.abs()).sum::<f32>();
        assert!(sum > 1.0, "Matrix function should produce non-zero result");

        // Verify symmetry is preserved
        assert!(
            (exp_result[[0, 1]] - exp_result[[1, 0]]).abs() < 1e-5,
            "Result should be symmetric"
        );
    });
}

#[allow(dead_code)]
fn main() {
    println!("Running tests for advanced linear algebra implementations...");
}
