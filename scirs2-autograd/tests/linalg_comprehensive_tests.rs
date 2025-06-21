use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

const EPSILON: f64 = 1e-5;
const MATRIX_FUNC_EPSILON: f64 = 1e-2; // Larger tolerance for matrix function approximations

#[test]
fn test_matrix_operations() {
    ag::run::<f64, _, _>(|g| {
        // Test matrix inverse
        let a = convert_to_tensor(array![[2.0, 1.0], [1.0, 3.0]], g);
        let inv = matrix_inverse(&a);
        let identity = matmul(&a, &inv);
        let result = identity.eval(g).unwrap();

        assert!((result[[0, 0]] - 1.0_f64).abs() < EPSILON);
        assert!((result[[1, 1]] - 1.0_f64).abs() < EPSILON);
        assert!((result[[0, 1]] as f64).abs() < EPSILON);
        assert!((result[[1, 0]] as f64).abs() < EPSILON);

        // Test determinant
        let det = determinant(&a);
        let det_val = det.eval(g).unwrap();
        assert!((det_val[[]] - 5.0_f64).abs() < EPSILON); // det([[2,1],[1,3]]) = 6-1 = 5
    });
}

#[test]
fn test_matrix_functions() {
    ag::run::<f64, _, _>(|g| {
        // Test matrix exponential and logarithm
        let a = convert_to_tensor(array![[0.5, 0.1], [0.1, 0.3]], g);
        let exp_a = matrix_exp(&a);
        let log_exp_a = matrix_log(&exp_a);
        let result = log_exp_a.eval(g).unwrap();
        let original = a.eval(g).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                let diff = (result[[i, j]] - original[[i, j]]) as f64;
                println!(
                    "Log(Exp(A)) vs A - Position [{}, {}]: result={}, original={}, diff={}",
                    i,
                    j,
                    result[[i, j]],
                    original[[i, j]],
                    diff
                );
                assert!(
                    diff.abs() < MATRIX_FUNC_EPSILON,
                    "Matrix log(exp(A)) differs from A at position [{}, {}]: diff={}",
                    i,
                    j,
                    diff
                );
            }
        }

        // Test matrix square root
        let b = convert_to_tensor(array![[4.0, 0.0], [0.0, 9.0]], g);
        let sqrt_b = matrix_sqrt(&b);
        let result = sqrt_b.eval(g).unwrap();
        assert!(((result[[0, 0]] - 2.0) as f64).abs() < EPSILON);
        assert!(((result[[1, 1]] - 3.0) as f64).abs() < EPSILON);

        // Test matrix power
        let c = convert_to_tensor(array![[2.0, 0.0], [0.0, 3.0]], g);
        let c_squared = powm(&c, 2.0);
        let result = c_squared.eval(g).unwrap();
        assert!(((result[[0, 0]] - 4.0) as f64).abs() < EPSILON);
        assert!(((result[[1, 1]] - 9.0) as f64).abs() < EPSILON);
    });
}

#[test]
fn test_special_matrices() {
    ag::run::<f64, _, _>(|g| {
        // Test Cholesky decomposition
        let a = convert_to_tensor(array![[4.0, 2.0], [2.0, 5.0]], g); // Positive definite matrix
        let l = cholesky(&a);
        let reconstructed = matmul(&l, &transpose(&l, &[1, 0]));
        let result = reconstructed.eval(g).unwrap();
        let original = a.eval(g).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(((result[[i, j]] - original[[i, j]]) as f64).abs() < EPSILON);
            }
        }

        // Test symmetrize
        let b = convert_to_tensor(array![[1.0, 2.0], [3.0, 4.0]], g);
        let sym = symmetrize(&b);
        let result = sym.eval(g).unwrap();
        assert!(((result[[0, 1]] - result[[1, 0]]) as f64).abs() < EPSILON);
        assert!(((result[[0, 1]] - 2.5) as f64).abs() < EPSILON); // (2+3)/2 = 2.5

        // Test triangular extraction
        let c = convert_to_tensor(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], g);
        let lower = tril(&c, 0);
        let upper = triu(&c, 0);

        let lower_result = lower.eval(g).unwrap();
        let upper_result = upper.eval(g).unwrap();

        // Check lower triangular
        assert!((lower_result[[0, 1]] as f64).abs() < EPSILON);
        assert!((lower_result[[0, 2]] as f64).abs() < EPSILON);
        assert!((lower_result[[1, 2]] as f64).abs() < EPSILON);

        // Check upper triangular
        assert!((upper_result[[1, 0]] as f64).abs() < EPSILON);
        assert!((upper_result[[2, 0]] as f64).abs() < EPSILON);
        assert!((upper_result[[2, 1]] as f64).abs() < EPSILON);
    });
}

#[test]
fn test_eigenvalue_decomposition() {
    ag::run::<f64, _, _>(|g| {
        // Test with symmetric matrix
        let a = convert_to_tensor(array![[3.0, 1.0], [1.0, 3.0]], g);
        let (eigenvalues, eigenvectors) = eigen(&a);

        // Debug shapes by evaluating
        let eigenvals_eval = eigenvalues.eval(g).unwrap();
        let eigenvecs_eval = eigenvectors.eval(g).unwrap();
        println!("Debug: eigenvalues shape: {:?}", eigenvals_eval.shape());
        println!("Debug: eigenvectors shape: {:?}", eigenvecs_eval.shape());

        // Reconstruct matrix: A = V * Λ * V^T
        let lambda = diag(&eigenvalues);
        let lambda_eval = lambda.eval(g).unwrap();
        println!("Debug: lambda shape: {:?}", lambda_eval.shape());

        let v_lambda = matmul(&eigenvectors, &lambda);
        let reconstructed = matmul(&v_lambda, &transpose(&eigenvectors, &[1, 0]));

        let result = reconstructed.eval(g).unwrap();
        let original = a.eval(g).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(((result[[i, j]] - original[[i, j]]) as f64).abs() < 1e-4);
            }
        }

        // Test eigenvalues separately
        // Extract eigenvalues from the previous decomposition
        let eigenvals_result = eigenvalues.eval(g).unwrap();

        // Since we're just testing that we can evaluate the eigenvalues
        assert!(eigenvals_result.shape() == &[2]);
    });
}

#[test]
fn test_linear_solvers() {
    ag::run::<f64, _, _>(|g| {
        // Test linear system solver
        let a = convert_to_tensor(array![[3.0, 1.0], [1.0, 2.0]], g);
        let b = convert_to_tensor(array![[9.0], [8.0]], g);
        let x = solve(&a, &b);

        // Verify Ax = b
        let ax = matmul(&a, &x);
        let result = ax.eval(g).unwrap();
        let b_val = b.eval(g).unwrap();

        for i in 0..2 {
            assert!(((result[[i, 0]] - b_val[[i, 0]]) as f64).abs() < EPSILON);
        }

        // Test least squares solver
        let a_overdetermined = convert_to_tensor(array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], g);
        let b_overdetermined = convert_to_tensor(array![[6.0], [5.0], [7.0]], g);
        let x_ls = lstsq(&a_overdetermined, &b_overdetermined);

        // The solution should minimize ||Ax - b||^2
        let x_result = x_ls.eval(g).unwrap();
        assert!(x_result.shape() == &[2, 1]);
    });
}

#[test]
fn test_gradient_computation() {
    ag::run::<f64, _, _>(|g| {
        // Test gradient through matrix operations
        let a = variable(array![[2.0, 1.0], [1.0, 3.0]], g);
        let b = variable(array![[1.0], [2.0]], g);

        // Test gradient through inverse
        let inv_a = matrix_inverse(&a);
        let y = matmul(&inv_a, &b);
        let loss = sum_all(square(&y));

        let grads = grad(&[&loss], &[&a]);
        let grad_a = &grads[0];
        assert!(grad_a.eval(g).is_ok());

        // Test gradient through solve
        let x = solve(&a, &b);
        let loss2 = sum_all(square(&x));

        let grads2 = grad(&[&loss2], &[&a, &b]);
        assert!(grads2[0].eval(g).is_ok());
        assert!(grads2[1].eval(g).is_ok());
    });
}

#[test]
fn test_complex_linear_algebra_pipeline() {
    ag::run::<f64, _, _>(|g| {
        // Create a complex pipeline using multiple operations
        let a = variable(array![[4.0, 2.0], [2.0, 5.0]], g); // Positive definite

        // Cholesky decomposition
        let l = cholesky(&a);

        // Solve system using Cholesky factorization
        let b = convert_to_tensor(array![[1.0], [2.0]], g);
        let y = solve(&l, &b);
        let x = solve(&transpose(&l, &[1, 0]), &y);

        // Compute determinant using Cholesky
        let det_l = determinant(&l);
        let det_a = square(&det_l); // det(A) = det(L)^2 for A = L*L^T

        // Test results
        let x_direct = solve(&a, &b);
        let x_result = x.eval(g).unwrap();
        let x_direct_result = x_direct.eval(g).unwrap();

        for i in 0..2 {
            assert!(((x_result[[i, 0]] - x_direct_result[[i, 0]]) as f64).abs() < EPSILON);
        }

        let det_a_result = det_a.eval(g).unwrap();
        let det_direct = determinant(&a).eval(g).unwrap();
        assert!(((det_a_result[[]] - det_direct[[]]) as f64).abs() < EPSILON);
    });
}
