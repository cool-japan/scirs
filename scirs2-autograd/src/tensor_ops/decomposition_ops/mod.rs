//! Matrix decomposition operations — QR, SVD, Cholesky, eigendecomposition, LU,
//! and matrix functions (exp, log, power).

mod backward_ops;
mod cholesky_eigen_ops;
mod lu_ops;
mod qr_ops;
mod svd_ops;

// Public re-exports — maintain exactly the same public API as the original file
pub(crate) use backward_ops::{
    CholeskyBackwardOp, LUExtractBackwardOp, QRExtractBackwardOp, SVDBackwardOp,
};
pub use cholesky_eigen_ops::{cholesky, matrix_exp, matrix_log, matrix_power, symmetric_eigen};
pub use cholesky_eigen_ops::{
    CholeskyOp, MatrixExpOp, MatrixLogOp, MatrixPowerOp, SymmetricEigenOp,
};
pub use lu_ops::{lu, LUExtractOp, LUOp};
pub use qr_ops::{qr, QRExtractOp, QROp};
pub use svd_ops::{svd, SVDExtractOp, SVDOp};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::convert_to_tensor;
    use scirs2_core::ndarray::{array, Array2, Ix2};
    use scirs2_core::numeric::FromPrimitive;

    /// Helper: compute max absolute error of A - U * diag(s) * Vt
    fn svd_reconstruction_error(
        a: &Array2<f64>,
        u: &scirs2_core::ndarray::ArrayD<f64>,
        s: &scirs2_core::ndarray::ArrayD<f64>,
        vt: &scirs2_core::ndarray::ArrayD<f64>,
    ) -> f64 {
        let m = a.shape()[0];
        let n = a.shape()[1];
        let k = m.min(n);

        let u2 = u.view().into_dimensionality::<Ix2>().expect("u 2d");
        let vt2 = vt.view().into_dimensionality::<Ix2>().expect("vt 2d");

        let mut max_err = 0.0_f64;
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0_f64;
                for r in 0..k {
                    let sigma_r = if r < s.len() { s[r] } else { 0.0 };
                    val += u2[[i, r]] * sigma_r * vt2[[r, j]];
                }
                let err = (a[[i, j]] - val).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        max_err
    }

    #[test]
    fn test_svd_identity_3x3() {
        // SVD of 3×3 identity → singular values [1,1,1]
        crate::run(|g| {
            let eye3 = Array2::<f64>::eye(3);
            let mat = convert_to_tensor(eye3.into_dyn(), g);
            let (u, s, vt) = svd(&mat);

            let s_val = s.eval(g).expect("s eval");
            assert_eq!(s_val.shape(), &[3], "sigma shape wrong");

            for i in 0..3 {
                let diff = (s_val[i] - 1.0_f64).abs();
                assert!(diff < 1e-10, "sigma[{i}] = {}, expected 1.0", s_val[i]);
            }

            // Check descending order
            assert!(s_val[0] >= s_val[1], "sigma not descending at 0,1");
            assert!(s_val[1] >= s_val[2], "sigma not descending at 1,2");

            // Verify reconstruction
            let u_val = u.eval(g).expect("u eval");
            let vt_val = vt.eval(g).expect("vt eval");
            let eye3 = Array2::<f64>::eye(3);
            let err = svd_reconstruction_error(&eye3, &u_val, &s_val, &vt_val);
            assert!(err < 1e-10, "reconstruction error too large: {err}");
        });
    }

    #[test]
    fn test_svd_4x4_random_reconstruction() {
        // 4×4 matrix A: verify A ≈ U diag(sigma) V^T within 1e-10
        crate::run(|g| {
            let a_data = array![
                [4.0_f64, 3.0, 1.0, -2.0],
                [-1.0, 2.0, 5.0, 1.0],
                [2.0, 0.0, -3.0, 4.0],
                [1.0, -1.0, 2.0, 3.0],
            ];
            let mat = convert_to_tensor(a_data.clone().into_dyn(), g);
            let (u, s, vt) = svd(&mat);

            let u_val = u.eval(g).expect("u eval");
            let s_val = s.eval(g).expect("s eval");
            let vt_val = vt.eval(g).expect("vt eval");

            // Shapes
            assert_eq!(u_val.shape(), &[4, 4], "U shape wrong");
            assert_eq!(s_val.shape(), &[4], "sigma shape wrong");
            assert_eq!(vt_val.shape(), &[4, 4], "Vt shape wrong");

            // Reconstruction error
            let err = svd_reconstruction_error(&a_data, &u_val, &s_val, &vt_val);
            assert!(err < 1e-8, "4x4 reconstruction error: {err} > 1e-8");

            // Singular values non-negative
            for i in 0..4 {
                assert!(s_val[i] >= 0.0, "sigma[{i}] = {} is negative", s_val[i]);
            }

            // Descending order
            for i in 0..3 {
                assert!(
                    s_val[i] >= s_val[i + 1] - 1e-12,
                    "sigma not descending: sigma[{i}]={} > sigma[{}]={}",
                    s_val[i],
                    i + 1,
                    s_val[i + 1]
                );
            }
        });
    }

    #[test]
    fn test_svd_singular_values_non_negative_descending() {
        crate::run(|g| {
            let a_data = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];
            let mat = convert_to_tensor(a_data.clone().into_dyn(), g);
            let (_u, s, _vt) = svd(&mat);
            let s_val = s.eval(g).expect("s eval");

            assert_eq!(s_val.shape(), &[3]);
            for i in 0..3 {
                assert!(s_val[i] >= -1e-12, "sigma[{i}] = {} is negative", s_val[i]);
            }
            for i in 0..2 {
                assert!(
                    s_val[i] >= s_val[i + 1] - 1e-10,
                    "sigma not descending: sigma[{}]={} < sigma[{}]={}",
                    i,
                    s_val[i],
                    i + 1,
                    s_val[i + 1]
                );
            }
        });
    }

    #[test]
    fn test_svd_rank_deficient() {
        // Rank-deficient matrix (rank 2 of 3×3) → at least one near-zero singular value
        crate::run(|g| {
            // Row 3 = Row 1 + Row 2 → rank 2
            let a_data = array![
                [1.0_f64, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [5.0, 7.0, 9.0], // sum of rows 1 and 2
            ];
            let mat = convert_to_tensor(a_data.clone().into_dyn(), g);
            let (u, s, vt) = svd(&mat);

            let u_val = u.eval(g).expect("u eval");
            let s_val = s.eval(g).expect("s eval");
            let vt_val = vt.eval(g).expect("vt eval");

            // Reconstruction must still work
            let err = svd_reconstruction_error(&a_data, &u_val, &s_val, &vt_val);
            assert!(err < 1e-8, "rank-deficient reconstruction error: {err}");

            // Smallest singular value should be near zero
            let min_s = s_val[s_val.len() - 1];
            assert!(
                min_s < 1e-6,
                "Expected near-zero singular value, got {min_s}"
            );
        });
    }

    #[test]
    fn test_svd_rectangular_wide() {
        // 3×5 matrix (wide)
        crate::run(|g| {
            let a_data = array![
                [1.0_f64, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
            ];
            let mat = convert_to_tensor(a_data.clone().into_dyn(), g);
            let (u, s, vt) = svd(&mat);

            let u_val = u.eval(g).expect("u eval");
            let s_val = s.eval(g).expect("s eval");
            let vt_val = vt.eval(g).expect("vt eval");

            // k = min(3,5) = 3
            assert_eq!(u_val.shape(), &[3, 3], "U shape for 3x5");
            assert_eq!(s_val.shape(), &[3], "sigma shape for 3x5");
            assert_eq!(vt_val.shape(), &[3, 5], "Vt shape for 3x5");

            let err = svd_reconstruction_error(&a_data, &u_val, &s_val, &vt_val);
            assert!(err < 1e-8, "3x5 reconstruction error: {err}");
        });
    }

    #[test]
    fn test_svd_rectangular_tall() {
        // 5×3 matrix (tall)
        crate::run(|g| {
            let a_data = array![
                [1.0_f64, -1.0, 2.0],
                [0.0, 3.0, 1.0],
                [-2.0, 1.0, 0.0],
                [1.0, 1.0, -1.0],
                [3.0, 0.0, 2.0],
            ];
            let mat = convert_to_tensor(a_data.clone().into_dyn(), g);
            let (u, s, vt) = svd(&mat);

            let u_val = u.eval(g).expect("u eval");
            let s_val = s.eval(g).expect("s eval");
            let vt_val = vt.eval(g).expect("vt eval");

            // k = min(5,3) = 3
            assert_eq!(u_val.shape(), &[5, 3], "U shape for 5x3");
            assert_eq!(s_val.shape(), &[3], "sigma shape for 5x3");
            assert_eq!(vt_val.shape(), &[3, 3], "Vt shape for 5x3");

            let err = svd_reconstruction_error(&a_data, &u_val, &s_val, &vt_val);
            assert!(err < 1e-8, "5x3 reconstruction error: {err}");
        });
    }

    #[test]
    fn test_svd_diagonal_matrix() {
        // Diagonal matrix → singular values are the diagonal (sorted descending)
        crate::run(|g| {
            let a_data = array![[3.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0],];
            let mat = convert_to_tensor(a_data.clone().into_dyn(), g);
            let (_u, s, _vt) = svd(&mat);
            let s_val = s.eval(g).expect("s eval");

            // Singular values should be {1, 3, 5} sorted descending → {5, 3, 1}
            assert!(s_val[0] > s_val[1], "sigma not descending at 0,1");
            assert!(s_val[1] > s_val[2], "sigma not descending at 1,2");

            let mut vals: Vec<f64> = s_val.iter().copied().collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let expected = [1.0_f64, 3.0, 5.0];
            for i in 0..3 {
                assert!(
                    (vals[i] - expected[i]).abs() < 1e-8,
                    "sorted sigma[{i}] = {}, expected {}",
                    vals[i],
                    expected[i]
                );
            }
        });
    }
}
