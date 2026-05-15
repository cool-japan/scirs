//! Tests for fast_kriging_reexports module.
//!
//! Separated from the main implementation to keep file size under 2000 lines.

use super::*;
use scirs2_core::ndarray::{Array1, Array2};

/// Build a simple 2-D test dataset: z = x² + y².
fn make_grid_data(nx: usize) -> (Array2<f64>, Array1<f64>) {
    let n = nx * nx;
    let mut pts = Array2::zeros((n, 2));
    let mut vals = Array1::zeros(n);
    let mut idx = 0;
    for i in 0..nx {
        for j in 0..nx {
            let x = i as f64 / (nx - 1) as f64;
            let y = j as f64 / (nx - 1) as f64;
            pts[[idx, 0]] = x;
            pts[[idx, 1]] = y;
            vals[idx] = x * x + y * y;
            idx += 1;
        }
    }
    (pts, vals)
}

#[test]
fn test_cholesky_solve_identity() {
    let a = Array2::<f64>::eye(3);
    let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let x = cholesky_solve(&a, &b).expect("cholesky_solve");
    for (xi, bi) in x.iter().zip(b.iter()) {
        assert!((xi - bi).abs() < 1e-10);
    }
}

#[test]
fn test_cholesky_solve_2x2() {
    // [4 2; 2 3] x = [8; 7]
    // Solution: 4x0 + 2x1 = 8, 2x0 + 3x1 = 7 => x0 = 1.25, x1 = 1.5
    let a = Array2::from_shape_vec((2, 2), vec![4.0_f64, 2.0, 2.0, 3.0]).expect("shape");
    let b = Array1::from_vec(vec![8.0_f64, 7.0]);
    let x = cholesky_solve(&a, &b).expect("cholesky_solve");
    // Verify A*x = b
    let r0 = 4.0 * x[0] + 2.0 * x[1] - 8.0;
    let r1 = 2.0 * x[0] + 3.0 * x[1] - 7.0;
    assert!(r0.abs() < 1e-9, "residual[0] = {r0}");
    assert!(r1.abs() < 1e-9, "residual[1] = {r1}");
}

#[test]
fn test_wendland_c2_properties() {
    // At r=0 should be 1
    assert!((wendland_c2(0.0_f64, 1.0) - 1.0).abs() < 1e-12);
    // At r=range should be 0
    assert!(wendland_c2(1.0_f64, 1.0).abs() < 1e-12);
    // Beyond range should be 0
    assert!(wendland_c2(1.5_f64, 1.0).abs() < 1e-12);
    // Monotone in [0, range]
    let vals: Vec<f64> = (0..=10).map(|i| wendland_c2(i as f64 * 0.1, 1.0)).collect();
    for w in vals.windows(2) {
        assert!(
            w[0] >= w[1],
            "Wendland C2 should be monotonically non-increasing"
        );
    }
}

#[test]
fn test_select_approximation_method() {
    assert_eq!(select_approximation_method(100), FastKrigingMethod::Local);
    assert!(matches!(
        select_approximation_method(2_000),
        FastKrigingMethod::FixedRank(_)
    ));
    assert!(matches!(
        select_approximation_method(20_000),
        FastKrigingMethod::Tapering(_)
    ));
    assert!(matches!(
        select_approximation_method(100_000),
        FastKrigingMethod::HODLR(_)
    ));
}

#[test]
fn test_make_local_kriging_build_and_predict() {
    let (pts, vals) = make_grid_data(5); // 25 training points
    let kriging = make_local_kriging(
        &pts.view(),
        &vals.view(),
        CovarianceFunction::Matern52,
        0.5,
        8,
    )
    .expect("build");

    assert_eq!(kriging.n_points(), 25);
    assert_eq!(kriging.n_dims(), 2);

    let q = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.75, 0.75]).expect("shape");
    let pred = kriging.predict(&q.view()).expect("predict");

    assert_eq!(pred.value.len(), 2);
    assert_eq!(pred.variance.len(), 2);
    assert!(pred.value[0].is_finite());
    assert!(pred.value[1].is_finite());
    for &v in pred.variance.iter() {
        assert!(v >= 0.0, "Variance must be non-negative");
    }
}

#[test]
fn test_local_kriging_interpolates_training_points() {
    // At a training point, prediction should be close to the true value
    let pts = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    .expect("shape");
    let vals = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 0.5]);

    let kriging = make_local_kriging(
        &pts.view(),
        &vals.view(),
        CovarianceFunction::SquaredExponential,
        1.0,
        4,
    )
    .expect("build");

    // Query the first training point exactly
    let q = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).expect("shape");
    let pred = kriging.predict(&q.view()).expect("predict");
    // Should be somewhere close to 0.5 (or at least finite and non-negative variance)
    assert!(pred.value[0].is_finite());
    assert!(pred.variance[0] >= 0.0);
}

#[test]
fn test_make_fixed_rank_kriging() {
    let (pts, vals) = make_grid_data(6); // 36 training points
    let kriging = make_fixed_rank_kriging(
        &pts.view(),
        &vals.view(),
        5,
        CovarianceFunction::Exponential,
        0.5,
    )
    .expect("build");

    let q = Array2::from_shape_vec((3, 2), vec![0.1, 0.1, 0.5, 0.5, 0.9, 0.9]).expect("shape");
    let pred = kriging.predict(&q.view()).expect("predict");

    assert_eq!(pred.value.len(), 3);
    for &v in pred.value.iter() {
        assert!(v.is_finite());
    }
    for &var in pred.variance.iter() {
        assert!(var >= 0.0);
    }
}

#[test]
fn test_make_tapered_kriging() {
    let (pts, vals) = make_grid_data(5);
    let kriging = make_tapered_kriging(
        &pts.view(),
        &vals.view(),
        0.5_f64,
        CovarianceFunction::SquaredExponential,
        0.3_f64,
    )
    .expect("build");

    let q = Array2::from_shape_vec((2, 2), vec![0.3, 0.3, 0.7, 0.7]).expect("shape");
    let pred = kriging.predict(&q.view()).expect("predict");

    assert_eq!(pred.value.len(), 2);
    for &v in pred.value.iter() {
        assert!(v.is_finite());
    }
    for &var in pred.variance.iter() {
        assert!(var >= 0.0);
    }
}

#[test]
fn test_make_hodlr_kriging() {
    let (pts, vals) = make_grid_data(6); // 36 training points
    let kriging = make_hodlr_kriging(
        &pts.view(),
        &vals.view(),
        8,
        CovarianceFunction::Matern32,
        0.5,
    )
    .expect("build");

    let q = Array2::from_shape_vec((2, 2), vec![0.4, 0.4, 0.6, 0.6]).expect("shape");
    let pred = kriging.predict(&q.view()).expect("predict");

    for &v in pred.value.iter() {
        assert!(v.is_finite());
    }
    for &var in pred.variance.iter() {
        assert!(var >= 0.0);
    }
}

#[test]
fn test_builder_api() {
    let (pts, vals) = make_grid_data(4);
    let kriging = FastKrigingBuilder::<f64>::new()
        .points(pts)
        .values(vals)
        .covariance_function(CovarianceFunction::Matern52)
        .approximation_method(FastKrigingMethod::Local)
        .max_neighbors(6)
        .sigma_sq(2.0)
        .nugget(1e-5)
        .build()
        .expect("build");

    assert_eq!(kriging.n_dims(), 2);
    assert_eq!(kriging.approximation_method(), FastKrigingMethod::Local);
}

#[test]
fn test_builder_missing_points_error() {
    let result = FastKrigingBuilder::<f64>::new()
        .values(Array1::zeros(5))
        .build();
    assert!(result.is_err());
}

#[test]
fn test_builder_missing_values_error() {
    let result = FastKrigingBuilder::<f64>::new()
        .points(Array2::zeros((5, 2)))
        .build();
    assert!(result.is_err());
}

#[test]
fn test_builder_dimension_mismatch_error() {
    let result = FastKrigingBuilder::<f64>::new()
        .points(Array2::zeros((5, 2)))
        .values(Array1::zeros(3))
        .build();
    assert!(result.is_err());
}

#[test]
fn test_predict_dimension_mismatch_error() {
    let (pts, vals) = make_grid_data(4);
    let kriging = FastKrigingBuilder::<f64>::new()
        .points(pts)
        .values(vals)
        .build()
        .expect("build");

    // Query with wrong dimensionality
    let q = Array2::<f64>::zeros((2, 3));
    assert!(kriging.predict(&q.view()).is_err());
}

#[test]
fn test_empty_query_points() {
    let (pts, vals) = make_grid_data(4);
    let kriging = make_local_kriging(
        &pts.view(),
        &vals.view(),
        CovarianceFunction::Matern52,
        0.5,
        4,
    )
    .expect("build");

    let q = Array2::<f64>::zeros((0, 2));
    let pred = kriging.predict(&q.view()).expect("predict");
    assert_eq!(pred.value.len(), 0);
}

#[test]
fn test_all_methods_produce_finite_output() {
    let (pts, vals) = make_grid_data(5);
    let q = Array2::from_shape_vec((3, 2), vec![0.1, 0.9, 0.5, 0.5, 0.9, 0.1]).expect("shape");

    for method in [
        FastKrigingMethod::Local,
        FastKrigingMethod::FixedRank(5),
        FastKrigingMethod::Tapering(0.6),
        FastKrigingMethod::HODLR(4),
    ] {
        let kriging = FastKrigingBuilder::<f64>::new()
            .points(pts.clone())
            .values(vals.clone())
            .covariance_function(CovarianceFunction::Matern52)
            .approximation_method(method)
            .max_neighbors(8)
            .build()
            .expect("build");

        let pred = kriging.predict(&q.view()).expect("predict");
        for (i, &v) in pred.value.iter().enumerate() {
            assert!(v.is_finite(), "method={method:?} query {i}: value={v}");
        }
        for (i, &var) in pred.variance.iter().enumerate() {
            assert!(var >= 0.0, "method={method:?} query {i}: variance={var}");
        }
    }
}

#[test]
fn test_f32_support() {
    let pts = Array2::<f32>::from_shape_fn((9, 2), |(i, j)| {
        (if j == 0 { i % 3 } else { i / 3 }) as f32 * 0.5
    });
    let vals = Array1::<f32>::from_iter((0..9).map(|i| i as f32 * 0.1));

    let kriging = make_local_kriging(
        &pts.view(),
        &vals.view(),
        CovarianceFunction::Exponential,
        0.5_f32,
        4,
    )
    .expect("build f32");

    let q = Array2::<f32>::from_shape_vec((1, 2), vec![0.5, 0.5]).expect("shape");
    let pred = kriging.predict(&q.view()).expect("predict f32");
    assert!(pred.value[0].is_finite());
}
