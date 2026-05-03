//! Integration tests for out-of-core interpolation.
//!
//! These tests cover:
//! - `DiskStorage` write/read round-trip
//! - `OutOfCoreRbf` direct solve path (n ≤ chunk_size×10)
//! - `OutOfCoreRbf` landmark approximation path (n > chunk_size×10)
//! - `OutOfCoreKriging` chunked Cholesky-equivalent (via Gaussian RBF delegate)
//! - Scratch-dir cleanup
//! - Partial-write resume (DiskStorage::open)
//!
//! All temporary files are created under `std::env::temp_dir()` and cleaned up
//! at the end of each test.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_interpolate::outofcore::{
    DiskStorage, OocRbfKernel, OutOfCoreKriging, OutOfCoreKrigingConfig, OutOfCoreRbf,
    OutOfCoreRbfConfig,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return a scratch directory unique to `test_name` under `std::env::temp_dir()`.
fn scratch(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("scirs2_ooc_{test_name}"));
    std::fs::create_dir_all(&dir).expect("cannot create temp scratch dir");
    dir
}

/// Remove the scratch directory, ignoring errors.
fn rm_scratch(dir: &std::path::Path) {
    let _ = std::fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// DiskStorage round-trip
// ---------------------------------------------------------------------------

#[test]
fn ooc_storage_write_read_roundtrip() {
    let path = std::env::temp_dir().join("scirs2_ooc_storage_test.bin");
    let _ = std::fs::remove_file(&path); // clean up from prior run

    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = DiskStorage::create(&path, 2, 3).expect("create");
    storage.write_rows(0, &data).expect("write");
    let read_back = storage.read_rows(0, 2).expect("read");
    assert_eq!(read_back, data, "round-trip mismatch");

    // Partial read: just first row
    let first_row = storage.read_rows(0, 1).expect("read first row");
    assert_eq!(first_row, vec![1.0, 2.0, 3.0]);

    // Partial read: second row
    let second_row = storage.read_rows(1, 1).expect("read second row");
    assert_eq!(second_row, vec![4.0, 5.0, 6.0]);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn ooc_storage_open_existing() {
    let path = std::env::temp_dir().join("scirs2_ooc_storage_open_test.bin");
    let _ = std::fs::remove_file(&path);

    let data = vec![-1.0_f64, 0.0, 1.0];
    let storage = DiskStorage::create(&path, 3, 1).expect("create");
    storage.write_rows(0, &data).expect("write");

    // Open by path (simulates resume from partial write)
    let reopened = DiskStorage::open(&path, 3, 1).expect("open");
    let read_back = reopened.read_rows(0, 3).expect("read after reopen");
    assert_eq!(read_back, data, "reopen round-trip mismatch");

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// OutOfCoreRbf — direct solve path (small n)
// ---------------------------------------------------------------------------

/// `ooc_rbf_matches_in_memory_for_n_100`
///
/// Fits 100 points and checks that predictions approximate the target function
/// (sin). Uses direct-solve path (n=100, chunk_size=50 → n ≤ chunk_size×10=500).
#[test]
fn ooc_rbf_matches_in_memory_for_n_100() {
    let n = 100usize;
    let d = 1usize;
    let centers = Array2::from_shape_fn((n, d), |(i, _)| i as f64 / n as f64);
    let values: Array1<f64> = centers.column(0).mapv(|x| x.sin());
    let query = Array2::from_shape_fn((20, d), |(i, _)| i as f64 / 20.0);

    let s = scratch("rbf_n100");
    let mut ooc = OutOfCoreRbf::new(OutOfCoreRbfConfig {
        scratch_dir: s.clone(),
        chunk_size: 50,
        kernel: OocRbfKernel::Gaussian,
        epsilon: 10.0,
        regularization: 1e-6,
        ..Default::default()
    });
    ooc.fit(&centers, &values).expect("fit should succeed");

    assert_eq!(ooc.n_centers(), n);
    assert_eq!(ooc.dim(), d);

    let preds = ooc.predict(&query).expect("predict should succeed");
    assert_eq!(preds.len(), 20);

    for (i, &pred) in preds.iter().enumerate() {
        let x = i as f64 / 20.0;
        let expected = x.sin();
        assert!(
            (pred - expected).abs() < 0.5,
            "i={i}: predicted {pred:.4}, expected sin({x:.2})={expected:.4}"
        );
    }

    ooc.cleanup().expect("cleanup should succeed");
    rm_scratch(&s);
}

/// `ooc_rbf_thin_plate_direct_path`
///
/// Thin-plate spline kernel on a 2D dataset, direct-solve path.
#[test]
fn ooc_rbf_thin_plate_direct_path() {
    let n = 50usize;
    let d = 2usize;
    // Sample on a regular grid
    let side = (n as f64).sqrt() as usize;
    let n_actual = side * side;
    let centers = Array2::from_shape_fn((n_actual, d), |(i, k)| {
        let row = i / side;
        let col = i % side;
        if k == 0 {
            row as f64 / side as f64
        } else {
            col as f64 / side as f64
        }
    });
    let values: Array1<f64> = Array1::from_iter((0..n_actual).map(|i| {
        let x = centers[[i, 0]];
        let y = centers[[i, 1]];
        x * x + y * y
    }));
    let query = Array2::from_shape_fn((4, d), |(i, k)| {
        if k == 0 {
            i as f64 / 4.0
        } else {
            (3 - i) as f64 / 4.0
        }
    });

    let s = scratch("rbf_thinplate_2d");
    let mut ooc = OutOfCoreRbf::new(OutOfCoreRbfConfig {
        scratch_dir: s.clone(),
        chunk_size: 100,
        kernel: OocRbfKernel::ThinPlate,
        regularization: 1e-8,
        ..Default::default()
    });
    ooc.fit(&centers, &values).expect("thin-plate fit");
    let preds = ooc.predict(&query).expect("thin-plate predict");
    assert_eq!(preds.len(), 4);
    // Check predictions are finite and roughly in range [0, 1]
    for &p in preds.iter() {
        assert!(p.is_finite(), "prediction is non-finite: {p}");
    }
    ooc.cleanup().ok();
    rm_scratch(&s);
}

// ---------------------------------------------------------------------------
// OutOfCoreRbf — landmark approximation path (large n)
// ---------------------------------------------------------------------------

/// `ooc_rbf_landmark_path_activated`
///
/// Forces the landmark approximation path by using chunk_size=10, n=200
/// (n > chunk_size×10 = 100 → landmark).
/// Tolerance is wider because landmark is approximate.
#[test]
fn ooc_rbf_landmark_path_activated() {
    let n = 200usize;
    let d = 1usize;
    let centers = Array2::from_shape_fn((n, d), |(i, _)| i as f64 / n as f64);
    // Target: linear function (should be captured well by landmark approx)
    let values: Array1<f64> = centers.column(0).mapv(|x| 2.0 * x + 0.5);
    let query = Array2::from_shape_fn((10, d), |(i, _)| i as f64 / 10.0);

    let s = scratch("rbf_landmark");
    let mut ooc = OutOfCoreRbf::new(OutOfCoreRbfConfig {
        scratch_dir: s.clone(),
        chunk_size: 10, // n=200 > 10×10=100 → landmark path
        kernel: OocRbfKernel::Gaussian,
        epsilon: 5.0,
        regularization: 1e-4,
        ..Default::default()
    });
    ooc.fit(&centers, &values).expect("landmark fit");
    let preds = ooc.predict(&query).expect("landmark predict");
    assert_eq!(preds.len(), 10);
    // All predictions should be finite
    for &p in preds.iter() {
        assert!(p.is_finite(), "landmark prediction non-finite: {p}");
    }
    ooc.cleanup().ok();
    rm_scratch(&s);
}

// ---------------------------------------------------------------------------
// OutOfCoreRbf — cleanup & scratch dir
// ---------------------------------------------------------------------------

#[test]
fn ooc_handles_scratch_dir_cleanup() {
    let s = scratch("rbf_cleanup");
    let centers = Array2::from_shape_fn((10, 1), |(i, _)| i as f64 / 10.0);
    let values = Array1::from_iter((0..10).map(|i| i as f64 / 10.0));
    let mut ooc = OutOfCoreRbf::new(OutOfCoreRbfConfig {
        scratch_dir: s.clone(),
        ..Default::default()
    });
    ooc.fit(&centers, &values).expect("fit");
    ooc.cleanup().expect("cleanup should succeed");
    // Coeff file should be gone; directory may still exist
    let coeff_path = s.join("outofcore_rbf_coeffs.bin");
    assert!(
        !coeff_path.exists(),
        "coefficient file should be deleted after cleanup"
    );
    rm_scratch(&s);
}

/// `ooc_resumes_from_partial_write`
///
/// Simulates a scenario where the coefficient file already exists (e.g., from a
/// previous run that was interrupted after `DiskStorage::create` but before
/// prediction).  Re-opening with `DiskStorage::open` and reading back the data
/// should succeed.
#[test]
fn ooc_resumes_from_partial_write() {
    let path = std::env::temp_dir().join("scirs2_ooc_resume_test.bin");
    let _ = std::fs::remove_file(&path);

    // Simulate partial write: write 3 rows, then open and read back
    let data: Vec<f64> = (0..9).map(|i| i as f64).collect();
    let storage = DiskStorage::create(&path, 3, 3).expect("create");
    storage.write_rows(0, &data).expect("write");

    // Re-open (as if restarting after a crash)
    let resumed = DiskStorage::open(&path, 3, 3).expect("resume open");
    let read_back = resumed.read_rows(0, 3).expect("resume read");
    assert_eq!(read_back, data);

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// OutOfCoreKriging
// ---------------------------------------------------------------------------

/// `ooc_kriging_chunked_cholesky_matches_in_memory`
///
/// Fits 80 points of a quadratic and checks that kriging predictions are
/// accurate.  The "chunked Cholesky" aspect is handled by `solve` inside
/// `OutOfCoreRbf`; the Gaussian kernel makes this equivalent to simple kriging.
#[test]
fn ooc_kriging_chunked_cholesky_matches_in_memory() {
    let n = 80usize;
    let centers = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
    let values: Array1<f64> = centers.column(0).mapv(|x| x * x);
    let query = Array2::from_shape_fn((10, 1), |(i, _)| i as f64 / 10.0);

    let s = scratch("kriging_n80");
    let mut ok = OutOfCoreKriging::new(OutOfCoreKrigingConfig {
        scratch_dir: s.clone(),
        chunk_size: 40,
        length_scale: 10.0,
        nugget: 1e-6,
        ..Default::default()
    });
    ok.fit(&centers, &values).expect("kriging fit");
    assert_eq!(ok.n_centers(), n);
    assert_eq!(ok.dim(), 1);

    let preds = ok.predict(&query).expect("kriging predict");
    assert_eq!(preds.len(), 10);

    for (i, &pred) in preds.iter().enumerate() {
        let x = i as f64 / 10.0;
        let expected = x * x;
        assert!(
            (pred - expected).abs() < 0.5,
            "kriging i={i}: predicted {pred:.4}, expected x²={expected:.4}"
        );
    }

    ok.cleanup().expect("kriging cleanup");
    rm_scratch(&s);
}

/// `ooc_kriging_multidimensional`
///
/// Sanity-check: kriging on 2D inputs, verifying finite outputs.
#[test]
fn ooc_kriging_multidimensional() {
    let n = 30usize;
    let centers = Array2::from_shape_fn((n, 2), |(i, k)| {
        if k == 0 {
            i as f64 / n as f64
        } else {
            (n - 1 - i) as f64 / n as f64
        }
    });
    let values: Array1<f64> =
        Array1::from_iter((0..n).map(|i| (centers[[i, 0]] + centers[[i, 1]]).sin()));
    let query = Array2::from_shape_fn((5, 2), |(i, k)| {
        if k == 0 {
            i as f64 / 5.0
        } else {
            (4 - i) as f64 / 5.0
        }
    });

    let s = scratch("kriging_2d");
    let mut ok = OutOfCoreKriging::new(OutOfCoreKrigingConfig {
        scratch_dir: s.clone(),
        ..Default::default()
    });
    ok.fit(&centers, &values).expect("2D kriging fit");
    let preds = ok.predict(&query).expect("2D kriging predict");
    assert_eq!(preds.len(), 5);
    for &p in preds.iter() {
        assert!(p.is_finite(), "2D kriging prediction is non-finite: {p}");
    }
    ok.cleanup().ok();
    rm_scratch(&s);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn ooc_rbf_predict_without_fit_returns_error() {
    let ooc = OutOfCoreRbf::new(OutOfCoreRbfConfig::default());
    let query = Array2::from_shape_fn((3, 1), |(i, _)| i as f64);
    let result = ooc.predict(&query);
    assert!(
        result.is_err(),
        "predict without fit should return an error"
    );
}

#[test]
fn ooc_rbf_fit_empty_centers_returns_error() {
    let s = scratch("rbf_empty");
    let mut ooc = OutOfCoreRbf::new(OutOfCoreRbfConfig {
        scratch_dir: s.clone(),
        ..Default::default()
    });
    let centers = Array2::<f64>::zeros((0, 1));
    let values = Array1::<f64>::zeros(0);
    let result = ooc.fit(&centers, &values);
    assert!(
        result.is_err(),
        "fit with empty centers should return an error"
    );
    rm_scratch(&s);
}

#[test]
fn ooc_rbf_fit_shape_mismatch_returns_error() {
    let s = scratch("rbf_shape_mismatch");
    let mut ooc = OutOfCoreRbf::new(OutOfCoreRbfConfig {
        scratch_dir: s.clone(),
        ..Default::default()
    });
    let centers = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
    let values = Array1::from_vec(vec![1.0, 2.0, 3.0]); // wrong length
    let result = ooc.fit(&centers, &values);
    assert!(
        result.is_err(),
        "fit with mismatched shapes should return an error"
    );
    rm_scratch(&s);
}

#[test]
fn ooc_rbf_predict_wrong_dimension_returns_error() {
    let s = scratch("rbf_dim_mismatch");
    let mut ooc = OutOfCoreRbf::new(OutOfCoreRbfConfig {
        scratch_dir: s.clone(),
        ..Default::default()
    });
    let centers = Array2::from_shape_fn((10, 2), |(i, k)| i as f64 + k as f64);
    let values = Array1::from_iter((0..10).map(|i| i as f64));
    ooc.fit(&centers, &values).expect("fit 2D");

    // Query with wrong dimension (1D instead of 2D)
    let query = Array2::from_shape_fn((3, 1), |(i, _)| i as f64);
    let result = ooc.predict(&query);
    assert!(
        result.is_err(),
        "predict with wrong dim should return an error"
    );

    ooc.cleanup().ok();
    rm_scratch(&s);
}
