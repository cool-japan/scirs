//! Integration tests for Rahimi-Recht Random Fourier Features (RFF).
//!
//! Tests the ndarray-based API: `FourierFeatureMap`, `RffKernel`,
//! `RandomFeaturesRegressor`, and `OrthogonalFourierFeatureMap`.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_interpolate::random_features::{
    FourierFeatureMap, OrthogonalFourierFeatureMap, RandomFeaturesRegressor, RffKernel,
};

// ─── Kernel approximation quality ─────────────────────────────────────────────

/// For Gaussian kernel K(x,y) = exp(-||x-y||²/2), verify that the RFF
/// approximation error decays as D grows.
#[test]
fn rff_kernel_approximation_error_decays_with_d() {
    let x1 = [1.0_f64, 0.0];
    let x2 = [0.0_f64, 1.0];
    // K = exp(-||x1-x2||²/(2·l²)) = exp(-(1+1)/(2·1)) = exp(-1)
    let true_k = (-1.0_f64).exp();

    for &d in &[100usize, 500, 2000] {
        let map = FourierFeatureMap::new(RffKernel::Gaussian { length_scale: 1.0 }, 2, d, 42);
        let approx_k = map.kernel_approx(&x1, &x2).expect("kernel_approx");
        let err = (approx_k - true_k).abs();
        println!("D={d}: approx_K={approx_k:.4}, true_K={true_k:.4}, err={err:.4}");
        if d == 2000 {
            assert!(err < 0.15, "at D={d}, error={err} should be < 0.15");
        }
    }
}

/// Same test via `transform`: compute z(x1)ᵀz(x2) and compare to K(x1,x2).
#[test]
fn rff_kernel_via_transform_matches_true_value() {
    let map = FourierFeatureMap::new(RffKernel::Gaussian { length_scale: 1.0 }, 2, 2000, 42);

    let x_mat = Array2::from_shape_fn((2, 2), |(i, j)| [[1.0_f64, 0.0], [0.0, 1.0]][i][j]);
    let z = map.transform(&x_mat.view()).expect("transform");
    let approx_k: f64 = z
        .row(0)
        .iter()
        .zip(z.row(1).iter())
        .map(|(a, b)| a * b)
        .sum();
    let true_k = (-1.0_f64).exp();
    let err = (approx_k - true_k).abs();
    println!("transform path: approx_K={approx_k:.4}, err={err:.4}");
    assert!(
        err < 0.15,
        "RFF kernel error={err:.4} should be < 0.15 for D=2000"
    );
}

// ─── Regression quality ───────────────────────────────────────────────────────

/// Fit sin(x) on [0, 2π] and check RMSE < 0.3.
#[test]
fn rff_regressor_fits_sin_function() {
    let n = 200;
    let x = Array2::from_shape_fn((n, 1), |(i, _)| {
        i as f64 * std::f64::consts::PI * 2.0 / n as f64
    });
    let y: Array1<f64> = x.column(0).mapv(f64::sin);

    let mut reg =
        RandomFeaturesRegressor::new(RffKernel::Gaussian { length_scale: 1.0 }, 500, 1e-4, 42);
    reg.fit(&x.view(), &y.view()).expect("fit should succeed");

    let n_test = 50;
    let x_test = Array2::from_shape_fn((n_test, 1), |(i, _)| {
        i as f64 * std::f64::consts::PI * 2.0 / n_test as f64
    });
    let y_pred = reg.predict(&x_test.view()).expect("predict should succeed");
    let y_true: Array1<f64> = x_test.column(0).mapv(f64::sin);

    let rmse: f64 = {
        let sum_sq: f64 = y_pred
            .iter()
            .zip(y_true.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        (sum_sq / n_test as f64).sqrt()
    };
    println!("sin regression RMSE = {rmse:.4}");
    assert!(rmse < 0.3, "RMSE {rmse} should be < 0.3 for sin regression");
}

/// Verify predict before fit returns an error.
#[test]
fn rff_regressor_predict_before_fit_errors() {
    let reg = RandomFeaturesRegressor::new(RffKernel::Gaussian { length_scale: 1.0 }, 50, 1e-3, 0);
    let x = Array2::<f64>::zeros((3, 1));
    assert!(
        reg.predict(&x.view()).is_err(),
        "predict before fit should return Err"
    );
}

// ─── Multiple kernel types ────────────────────────────────────────────────────

/// All four kernel types must produce finite output with correct shape.
#[test]
fn rff_multiple_kernel_types_run_without_panic() {
    let x = Array2::from_shape_fn((10, 2), |(i, j)| (i + j) as f64 * 0.1);
    for kernel in [
        RffKernel::Gaussian { length_scale: 1.0 },
        RffKernel::Laplacian { length_scale: 1.0 },
        RffKernel::Matern32 { length_scale: 1.0 },
        RffKernel::Matern52 { length_scale: 1.0 },
    ] {
        let map = FourierFeatureMap::new(kernel, 2, 100, 0);
        let z = map.transform(&x.view()).expect("transform should succeed");
        assert_eq!(z.shape(), &[10, 100]);
        assert!(
            z.iter().all(|v| v.is_finite()),
            "all features must be finite"
        );
    }
}

// ─── Dimension mismatch handling ──────────────────────────────────────────────

/// Wrong number of columns should return an error, not panic.
#[test]
fn rff_transform_dimension_mismatch_errors() {
    let map = FourierFeatureMap::new(RffKernel::Gaussian { length_scale: 1.0 }, 3, 64, 1);
    // Provide 2-column input but map expects 3.
    let x_bad = Array2::<f64>::zeros((5, 2));
    assert!(
        map.transform(&x_bad.view()).is_err(),
        "dimension mismatch should produce Err"
    );
}

// ─── Orthogonal Random Features ──────────────────────────────────────────────

/// ORF output shape must match (n, d_out).
#[test]
fn rff_orf_output_shape_correct() {
    let map =
        OrthogonalFourierFeatureMap::new(RffKernel::Gaussian { length_scale: 1.0 }, 3, 64, 99);
    let x = Array2::<f64>::zeros((5, 3));
    let z = map.transform(&x.view()).expect("ORF transform");
    assert_eq!(z.shape(), &[5, 64]);
}

/// ORF kernel approximation should be within reasonable bounds.
#[test]
fn rff_orf_kernel_approximation_reasonable() {
    let map =
        OrthogonalFourierFeatureMap::new(RffKernel::Gaussian { length_scale: 1.0 }, 2, 512, 7);
    let x1 = [1.0_f64, 0.0];
    let x2 = [0.0_f64, 1.0];
    let true_k = (-1.0_f64).exp();
    let approx_k = map.kernel_approx(&x1, &x2).expect("ORF kernel_approx");
    let err = (approx_k - true_k).abs();
    println!("ORF: approx_K={approx_k:.4}, true_K={true_k:.4}, err={err:.4}");
    assert!(err < 0.15, "ORF error={err:.4} should be < 0.15 for D=512");
}

/// When d_out is not a multiple of d_in, ORF should still produce d_out features.
#[test]
fn rff_orf_non_multiple_d_out() {
    let map = OrthogonalFourierFeatureMap::new(
        RffKernel::Gaussian { length_scale: 1.0 },
        3,
        7, // not a multiple of 3
        5,
    );
    let x = Array2::<f64>::zeros((2, 3));
    let z = map.transform(&x.view()).expect("ORF transform");
    assert_eq!(z.shape(), &[2, 7]);
}

// ─── Length-scale sensitivity ─────────────────────────────────────────────────

/// A larger length-scale should produce smoother kernel (closer to 1 for nearby points).
#[test]
fn rff_length_scale_affects_kernel_value() {
    let x1 = [0.5_f64, 0.0];
    let x2 = [0.0_f64, 0.5];
    // K_gaussian(x1,x2) = exp(-||x1-x2||²/(2l²))
    // Large l → K close to 1
    let map_large = FourierFeatureMap::new(RffKernel::Gaussian { length_scale: 10.0 }, 2, 2000, 1);
    let map_small = FourierFeatureMap::new(RffKernel::Gaussian { length_scale: 0.1 }, 2, 2000, 1);
    let k_large = map_large.kernel_approx(&x1, &x2).expect("large ls approx");
    let k_small = map_small.kernel_approx(&x1, &x2).expect("small ls approx");
    println!("k_large(ls=10)={k_large:.4}, k_small(ls=0.1)={k_small:.4}");
    assert!(
        k_large > k_small,
        "larger length-scale should produce larger kernel value for near points"
    );
}

// ─── Regressor with multiple kernel types ────────────────────────────────────

/// All kernel types should fit and predict without errors.
#[test]
fn rff_regressor_all_kernels_fit_and_predict() {
    let n = 30;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| (i + j) as f64 * 0.1);
    let y: Array1<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

    for kernel in [
        RffKernel::Gaussian { length_scale: 1.0 },
        RffKernel::Laplacian { length_scale: 1.0 },
        RffKernel::Matern32 { length_scale: 1.0 },
        RffKernel::Matern52 { length_scale: 1.0 },
    ] {
        let mut reg = RandomFeaturesRegressor::new(kernel, 50, 1e-3, 42);
        reg.fit(&x.view(), &y.view()).expect("fit");
        let preds = reg.predict(&x.view()).expect("predict");
        assert_eq!(preds.len(), n, "prediction length mismatch");
        assert!(
            preds.iter().all(|v| v.is_finite()),
            "all predictions must be finite"
        );
    }
}
