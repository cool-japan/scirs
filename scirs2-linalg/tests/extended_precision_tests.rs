#[cfg(feature = "extended-test")]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{array, Array2};
    #[allow(unused_imports)]
    use scirs2_linalg::extended_precision::{
        extended_cholesky, extended_eig, extended_eigvals, extended_eigvalsh, extended_lu,
        extended_matmul, extended_matvec, extended_qr, extended_solve, extended_svd,
    };

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
                assert_relative_eq!(c[[i, j]], c_standard[[i, j]], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_extended_lu() {
        let a = array![[2.0_f32, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]];

        // Compute LU decomposition with extended precision
        let (p, l, u) = extended_lu::<_, f64>(&a.view()).unwrap();

        // Verify P*A = L*U
        let pa = p.dot(&a);
        let lu = l.dot(&u);

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(pa[[i, j]], lu[[i, j]], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_extended_qr() {
        let a = array![
            [12.0_f32, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0]
        ];

        // Compute QR decomposition with extended precision
        let (q, r) = extended_qr::<_, f64>(&a.view()).unwrap();

        // Verify Q is orthogonal: Q^T * Q = I
        let qt = q.t();
        let qtq = qt.dot(&q);

        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(qtq[[i, j]], 1.0, epsilon = 1e-5);
                } else {
                    assert_relative_eq!(qtq[[i, j]], 0.0, epsilon = 1e-5);
                }
            }
        }

        // Verify A = Q*R
        let qr = q.dot(&r);

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(a[[i, j]], qr[[i, j]], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_extended_cholesky() {
        let a = array![[4.0_f32, 1.0, 1.0], [1.0, 5.0, 2.0], [1.0, 2.0, 6.0]];

        // Compute Cholesky decomposition with extended precision
        let l = extended_cholesky::<_, f64>(&a.view()).unwrap();

        // Verify A = L*L^T
        let lt = l.t();
        let llt = l.dot(&lt);

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(a[[i, j]], llt[[i, j]], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_extended_eigen() {
        let a = array![[1.0_f32, 2.0], [2.0, 1.0]];

        // Compute eigenvalues with extended precision
        let eigvals = extended_eigvalsh::<_, f64>(&a.view(), None, None).unwrap();

        // The eigenvalues of this matrix are 3 and -1
        assert!((eigvals[0] - 3.0).abs() < 1e-5 || (eigvals[0] + 1.0).abs() < 1e-5);
        assert!((eigvals[1] - 3.0).abs() < 1e-5 || (eigvals[1] + 1.0).abs() < 1e-5);
        assert!((eigvals[0] - eigvals[1]).abs() > 1.0); // Eigenvalues are different
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
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_extended_svd() {
        let a = array![[1.0_f32, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.2]];

        // Compute SVD with extended precision
        let (u, s, vh) = extended_svd::<_, f64>(&a.view(), true, None, None).unwrap();

        // Check that the singular values are correct
        assert_relative_eq!(s[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(s[1], 0.5, epsilon = 1e-5);
        assert_relative_eq!(s[2], 0.2, epsilon = 1e-5);

        // Check that U and V are orthogonal
        let ut = u.t();
        let utu = ut.dot(&u);

        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(utu[[i, j]], 1.0, epsilon = 1e-5);
                } else {
                    assert_relative_eq!(utu[[i, j]], 0.0, epsilon = 1e-5);
                }
            }
        }

        let vht = vh.t();
        let vhtvh = vh.dot(&vht);

        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(vhtvh[[i, j]], 1.0, epsilon = 1e-5);
                } else {
                    assert_relative_eq!(vhtvh[[i, j]], 0.0, epsilon = 1e-5);
                }
            }
        }

        // Check that A = U*S*V^H
        let mut usv = Array2::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    usv[[i, j]] += u[[i, k]] * s[k] * vh[[k, j]];
                }
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(a[[i, j]], usv[[i, j]], epsilon = 1e-5);
            }
        }
    }
}
