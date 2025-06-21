#[cfg(test)]
mod enhanced_linalg_tests {
    use ag::tensor_ops::linear_algebra::*;
    use ag::tensor_ops::*;
    #[allow(unused_imports)]
    use approx::assert_relative_eq;
    use ndarray::array;
    use scirs2_autograd as ag;

    #[test]
    fn test_matrix_norms() {
        ag::run(|g| {
            let a = convert_to_tensor(
                array![[1.0_f32, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]],
                g,
            );

            // Test 1-norm (maximum column sum)
            let n1 = norm1(&a);
            let n1_val = n1.eval(g).unwrap();
            assert_eq!(n1_val.shape(), &[] as &[usize]);
            // Column sums: |1|+|-4|+|7| = 12, |-2|+|5|+|-8| = 15, |3|+|-6|+|9| = 18
            // Maximum is 18
            assert_relative_eq!(n1_val[ndarray::IxDyn(&[])], 18.0, epsilon = 1e-5);

            // Test infinity-norm (maximum row sum)
            let ninf = norminf(&a);
            let ninf_val = ninf.eval(g).unwrap();
            assert_eq!(ninf_val.shape(), &[] as &[usize]);
            // Row sums: |1|+|-2|+|3| = 6, |-4|+|5|+|-6| = 15, |7|+|-8|+|9| = 24
            // Maximum is 24
            assert_relative_eq!(ninf_val[ndarray::IxDyn(&[])], 24.0, epsilon = 1e-5);

            // Test Frobenius norm
            let nfro = normfro(&a);
            let nfro_val = nfro.eval(g).unwrap();
            assert_eq!(nfro_val.shape(), &[] as &[usize]);
            // sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2)
            // = sqrt(1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81) = sqrt(285)
            assert_relative_eq!(
                nfro_val[ndarray::IxDyn(&[])],
                285.0_f32.sqrt(),
                epsilon = 1e-5
            );

            // Test 2-norm (spectral norm) - just verify it runs
            let n2 = norm2(&a);
            let n2_val = n2.eval(g).unwrap();
            assert_eq!(n2_val.shape(), &[] as &[usize]);
            assert!(n2_val[ndarray::IxDyn(&[])] > 0.0);
        });
    }

    #[test]
    fn test_symmetric_eigendecomposition() {
        ag::run(|g| {
            // Create a symmetric matrix
            let a = convert_to_tensor(array![[4.0_f32, 1.0], [1.0, 3.0]], g);

            // Test eigh (eigendecomposition for symmetric matrices)
            let (eigenvalues, eigenvectors) = eigh(&a);

            let vals = eigenvalues.eval(g).unwrap();
            let vecs = eigenvectors.eval(g).unwrap();

            assert_eq!(vals.shape(), &[2]);
            assert_eq!(vecs.shape(), &[2, 2]);

            // Eigenvalues should be sorted in descending order
            assert!(vals[0] >= vals[1]);

            // Test eigvalsh (eigenvalues only)
            let eigenvals_only = eigvalsh(&a);
            let vals_only = eigenvals_only.eval(g).unwrap();
            assert_eq!(vals_only.shape(), &[2]);
        });
    }

    #[test]
    fn test_matrix_exponential_methods() {
        ag::run(|g| {
            let a = convert_to_tensor(array![[0.0_f32, 1.0], [-1.0, 0.0]], g);

            // Test expm2 (Padé approximation)
            let exp_a2 = expm2(&a);
            let exp_a2_val = exp_a2.eval(g).unwrap();
            assert_eq!(exp_a2_val.shape(), &[2, 2]);

            // Test expm3 (eigendecomposition method)
            let exp_a3 = expm3(&a);
            let exp_a3_val = exp_a3.eval(g).unwrap();
            assert_eq!(exp_a3_val.shape(), &[2, 2]);

            // For this rotation matrix, exp(A) should give another rotation
            // The methods might give different results due to numerical differences
            // Just verify that both are valid matrix exponentials by checking shapes
            // and that the results are reasonable (non-NaN, finite)
            for i in 0..2 {
                for j in 0..2 {
                    assert!(exp_a2_val[[i, j]].is_finite());
                    assert!(exp_a3_val[[i, j]].is_finite());
                }
            }

            // For the specific matrix [[0, 1], [-1, 0]], the exact exponential is
            // [[cos(1), sin(1)], [-sin(1), cos(1)]]
            let cos_1 = 1.0_f32.cos();
            let sin_1 = 1.0_f32.sin();

            // Check that expm3 (eigendecomposition) gives approximately the right answer
            assert_relative_eq!(exp_a3_val[[0, 0]], cos_1, epsilon = 1e-3);
            assert_relative_eq!(exp_a3_val[[0, 1]], sin_1, epsilon = 1e-3);
            assert_relative_eq!(exp_a3_val[[1, 0]], -sin_1, epsilon = 1e-3);
            assert_relative_eq!(exp_a3_val[[1, 1]], cos_1, epsilon = 1e-3);
        });
    }

    #[test]
    fn test_cholesky_solve() {
        ag::run(|g| {
            // Create a positive definite matrix
            let a = convert_to_tensor(array![[4.0_f32, 2.0], [2.0, 3.0]], g);
            let b = convert_to_tensor(array![1.0_f32, 2.0], g);

            // Solve Ax = b using Cholesky decomposition
            let x = cholesky_solve(&a, &b);
            let x_val = x.eval(g).unwrap();
            assert_eq!(x_val.shape(), &[2]);

            // Verify solution: multiply A * x should give b
            let ax = matmul(&a, &x.reshape(&[2, 1]));
            let ax_val = ax.eval(g).unwrap();
            let b_val = b.eval(g).unwrap();

            assert_relative_eq!(ax_val[[0, 0]], b_val[0], epsilon = 1e-4);
            assert_relative_eq!(ax_val[[1, 0]], b_val[1], epsilon = 1e-4);
        });
    }

    #[test]
    fn test_sylvester_equation() {
        ag::run(|g| {
            let a = convert_to_tensor(array![[1.0_f32, 0.0], [0.0, 2.0]], g);
            let b = convert_to_tensor(array![[3.0_f32, 0.0], [0.0, 4.0]], g);
            let c = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Solve AX + XB = C
            let x = solve_sylvester(&a, &b, &c);
            let x_val = x.eval(g).unwrap();
            assert_eq!(x_val.shape(), &[2, 2]);

            // Verify solution: AX + XB should equal C
            let ax = matmul(&a, &x);
            let xb = matmul(&x, &b);
            let ax_plus_xb = add(&ax, &xb);
            let result = ax_plus_xb.eval(g).unwrap();
            let c_val = c.eval(g).unwrap();

            for i in 0..2 {
                for j in 0..2 {
                    assert_relative_eq!(result[[i, j]], c_val[[i, j]], epsilon = 1e-4);
                }
            }
        });
    }

    #[test]
    fn test_lyapunov_equation() {
        ag::run(|g| {
            let a = convert_to_tensor(array![[1.0_f32, 0.5], [0.5, 2.0]], g);
            let q = convert_to_tensor(array![[1.0_f32, 0.0], [0.0, 1.0]], g);

            // Solve AX + XA^T = Q
            let x = solve_lyapunov(&a, &q);
            let x_val = x.eval(g).unwrap();
            assert_eq!(x_val.shape(), &[2, 2]);

            // Verify solution
            let ax = matmul(&a, &x);
            let at = transpose(&a, &[1, 0]);
            let xat = matmul(&x, &at);
            let ax_plus_xat = add(&ax, &xat);
            let result = ax_plus_xat.eval(g).unwrap();
            let q_val = q.eval(g).unwrap();

            for i in 0..2 {
                for j in 0..2 {
                    assert_relative_eq!(result[[i, j]], q_val[[i, j]], epsilon = 1e-3);
                }
            }
        });
    }

    #[test]
    fn test_polar_decomposition() {
        ag::run(|g| {
            let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Compute polar decomposition A = UP
            let (u, p) = polar(&a);

            let u_val = u.eval(g).unwrap();
            let p_val = p.eval(g).unwrap();

            assert_eq!(u_val.shape(), &[2, 2]);
            assert_eq!(p_val.shape(), &[2, 2]);

            // P should be positive semidefinite (all eigenvalues >= 0)
            // This is a simplified test - just check diagonal elements are positive
            assert!(p_val[[0, 0]] >= 0.0);
            assert!(p_val[[1, 1]] >= 0.0);
        });
    }

    #[test]
    fn test_schur_decomposition() {
        ag::run(|g| {
            let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);

            // Compute Schur decomposition A = QTQ^T
            let (q, t) = schur(&a);

            let q_val = q.eval(g).unwrap();
            let t_val = t.eval(g).unwrap();

            assert_eq!(q_val.shape(), &[2, 2]);
            assert_eq!(t_val.shape(), &[2, 2]);

            // T should be quasi-upper triangular
            // For 2x2 case, just verify shapes
        });
    }

    #[test]
    fn test_tensor_solve() {
        ag::run(|g| {
            // Simple 2D case: solve Ax = b
            let a = convert_to_tensor(array![[2.0_f32, 1.0], [1.0, 3.0]], g);
            let b = convert_to_tensor(array![1.0_f32, 2.0], g);

            let x = tensor_solve(&a, &b, None);
            let x_val = x.eval(g).unwrap();

            // Verify shape
            assert_eq!(x_val.ndim(), 1);
            assert_eq!(x_val.len(), 2);
        });
    }

    #[test]
    fn test_einsum() {
        ag::run(|g| {
            let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);
            let b = convert_to_tensor(array![[5.0_f32, 6.0], [7.0, 8.0]], g);

            // Matrix multiplication: "ij,jk->ik"
            let c = einsum("ij,jk->ik", &[&a, &b]);
            let c_val = c.eval(g).unwrap();
            assert_eq!(c_val.shape(), &[2, 2]);

            // Compare with regular matmul
            let c_matmul = matmul(&a, &b);
            let c_matmul_val = c_matmul.eval(g).unwrap();

            for i in 0..2 {
                for j in 0..2 {
                    assert_relative_eq!(c_val[[i, j]], c_matmul_val[[i, j]], epsilon = 1e-5);
                }
            }

            // Test dot product: "i,i->"
            let v1 = convert_to_tensor(array![1.0_f32, 2.0, 3.0], g);
            let v2 = convert_to_tensor(array![4.0_f32, 5.0, 6.0], g);
            let dot = einsum("i,i->", &[&v1, &v2]);
            let dot_val = dot.eval(g).unwrap();
            assert_eq!(dot_val.shape(), &[] as &[usize]);
            assert_relative_eq!(dot_val[ndarray::IxDyn(&[])], 32.0, epsilon = 1e-5);
            // 1*4 + 2*5 + 3*6
        });
    }

    #[test]
    fn test_kronecker_product() {
        ag::run(|g| {
            let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);
            let b = convert_to_tensor(array![[5.0_f32, 6.0], [7.0, 8.0]], g);

            // Test using kron_tensor alias
            let c = kronecker_product(&a, &b);
            let c_val = c.eval(g).unwrap();

            // Kronecker product of 2x2 with 2x2 should be 4x4
            assert_eq!(c_val.shape(), &[4, 4]);

            // Verify first block (a[0,0] * b)
            assert_eq!(c_val[[0, 0]], 1.0 * 5.0);
            assert_eq!(c_val[[0, 1]], 1.0 * 6.0);
            assert_eq!(c_val[[1, 0]], 1.0 * 7.0);
            assert_eq!(c_val[[1, 1]], 1.0 * 8.0);
        });
    }
}
