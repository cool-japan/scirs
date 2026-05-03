//! Tests for Grid Resampling Utilities (Item #8).

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_interpolate::resampling::{
    resample_scattered_to_grid, Aggregator, GridSpec, ResampleStrategy,
};

// ---------------------------------------------------------------------------
// resample_rasterize_mean_on_known_grid
// ---------------------------------------------------------------------------

/// 4 points on a 2×2 grid, verify means.
///
/// Points:
///   (0, 0) → 0.0
///   (1, 0) → 1.0
///   (0, 1) → 2.0
///   (1, 1) → 3.0
/// Grid: 2×2 with axes [0, 1] × [0, 1]
/// Expected cell means: cell (0,0)=0.0, (1,0)=1.0, (0,1)=2.0, (1,1)=3.0
#[test]
fn resample_rasterize_mean_on_known_grid() {
    let pts = Array2::from_shape_vec((4, 2), vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        .expect("shape");
    let vals = Array1::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0]);

    let grid = GridSpec::uniform(2, &[(0.0, 1.0, 2), (0.0, 1.0, 2)]);
    let out = resample_scattered_to_grid(
        &pts,
        &vals,
        &grid,
        ResampleStrategy::Rasterize(Aggregator::Mean),
    )
    .expect("resample");

    assert_eq!(out.shape(), &[2, 2]);

    // C-order: index [i, j] → flat i*2 + j
    // Cell (0,0): point (0,0) → 0.0
    assert!(
        (out[[0, 0].as_ref()] - 0.0).abs() < 1e-10,
        "cell (0,0): {}",
        out[[0, 0].as_ref()]
    );
    // Cell (1,0): point (1,0) → 1.0
    assert!(
        (out[[1, 0].as_ref()] - 1.0).abs() < 1e-10,
        "cell (1,0): {}",
        out[[1, 0].as_ref()]
    );
    // Cell (0,1): point (0,1) → 2.0
    assert!(
        (out[[0, 1].as_ref()] - 2.0).abs() < 1e-10,
        "cell (0,1): {}",
        out[[0, 1].as_ref()]
    );
    // Cell (1,1): point (1,1) → 3.0
    assert!(
        (out[[1, 1].as_ref()] - 3.0).abs() < 1e-10,
        "cell (1,1): {}",
        out[[1, 1].as_ref()]
    );
}

// ---------------------------------------------------------------------------
// resample_handles_empty_cells
// ---------------------------------------------------------------------------

/// Cells that receive no scattered points get NaN (or 0 for Count).
#[test]
fn resample_handles_empty_cells() {
    // Only one point: goes to cell (0,0) of a 3×3 grid.
    let pts = Array2::from_shape_vec((1, 2), vec![0.0_f64, 0.0]).expect("shape");
    let vals = Array1::from_vec(vec![42.0_f64]);

    let grid = GridSpec::uniform(2, &[(0.0, 2.0, 3), (0.0, 2.0, 3)]);
    let out = resample_scattered_to_grid(
        &pts,
        &vals,
        &grid,
        ResampleStrategy::Rasterize(Aggregator::Mean),
    )
    .expect("resample");

    assert_eq!(out.shape(), &[3, 3]);

    // The cell containing (0,0) should be 42.
    let first = out[[0, 0].as_ref()];
    assert!(
        (first - 42.0).abs() < 1e-10,
        "cell (0,0) should be 42, got {first}"
    );

    // Other cells should be NaN (no points).
    let other = out[[2, 2].as_ref()];
    assert!(other.is_nan(), "empty cell should be NaN, got {other}");
}

// ---------------------------------------------------------------------------
// resample_1d_mean_matches_analytic
// ---------------------------------------------------------------------------

/// 100 uniformly spaced points on [0, 1], values = x.
/// A 5-cell grid [0, 0.25, 0.5, 0.75, 1.0] should accumulate ~20 points per
/// cell with mean ≈ cell centre.
#[test]
fn resample_1d_mean_matches_analytic() {
    let n = 100_usize;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let ys: Vec<f64> = xs.clone();

    let pts = Array2::from_shape_vec((n, 1), xs).expect("shape");
    let vals = Array1::from_vec(ys);

    let n_cells = 5_usize;
    let grid = GridSpec::uniform(1, &[(0.0, 1.0, n_cells)]);
    let out = resample_scattered_to_grid(
        &pts,
        &vals,
        &grid,
        ResampleStrategy::Rasterize(Aggregator::Mean),
    )
    .expect("resample");

    assert_eq!(out.shape(), &[n_cells]);

    // Cell centres are at 0.0, 0.25, 0.5, 0.75, 1.0.
    // Mean of y=x in each cell should be approximately the cell centre.
    for i in 0..n_cells {
        let cell_centre = i as f64 / (n_cells - 1) as f64;
        let val = out[[i].as_ref()];
        assert!(
            val.is_finite(),
            "cell {i} (centre={cell_centre}) is not finite: {val}"
        );
        assert!(
            (val - cell_centre).abs() < 0.15,
            "cell {i} (centre={cell_centre}) mean={val}, expected ~{cell_centre}"
        );
    }
}

// ---------------------------------------------------------------------------
// resample_conservative_matches_mean_for_uniform
// ---------------------------------------------------------------------------

/// Conservative strategy should equal Mean for uniform point distributions.
#[test]
fn resample_conservative_matches_mean_for_uniform() {
    let pts = Array2::from_shape_vec((4, 2), vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        .expect("shape");
    let vals = Array1::from_vec(vec![1.0_f64; 4]);

    let grid = GridSpec::uniform(2, &[(0.0, 1.0, 2), (0.0, 1.0, 2)]);

    let mean_out = resample_scattered_to_grid(
        &pts,
        &vals,
        &grid,
        ResampleStrategy::Rasterize(Aggregator::Mean),
    )
    .expect("mean resample");

    let cons_out = resample_scattered_to_grid(&pts, &vals, &grid, ResampleStrategy::Conservative)
        .expect("conservative resample");

    assert_eq!(mean_out.shape(), cons_out.shape());
    for (m, c) in mean_out.iter().zip(cons_out.iter()) {
        assert!(
            (m - c).abs() < 1e-10 || (m.is_nan() && c.is_nan()),
            "Mean={m}, Conservative={c} should match for uniform single-point-per-cell"
        );
    }
}

// ---------------------------------------------------------------------------
// resample_count_aggregator
// ---------------------------------------------------------------------------

/// Count aggregator should return the number of points per cell.
#[test]
fn resample_count_aggregator() {
    // 6 points, 2 landing in cell 0, 4 in cell 1 of a 1-D 2-cell grid.
    let pts =
        Array2::from_shape_vec((6, 1), vec![0.0_f64, 0.1, 0.2, 1.0, 1.0, 1.0]).expect("shape");
    let vals = Array1::from_vec(vec![1.0_f64; 6]);

    let grid = GridSpec::uniform(1, &[(0.0, 1.0, 2)]);
    let out = resample_scattered_to_grid(
        &pts,
        &vals,
        &grid,
        ResampleStrategy::Rasterize(Aggregator::Count),
    )
    .expect("count resample");

    assert_eq!(out.shape(), &[2]);

    let c0 = out[[0].as_ref()];
    let c1 = out[[1].as_ref()];

    // 0.0, 0.1, 0.2 are nearest to 0.0; 1.0, 1.0, 1.0 nearest to 1.0.
    assert!((c0 - 3.0).abs() < 1e-10, "cell 0 count = {c0}, expected 3");
    assert!((c1 - 3.0).abs() < 1e-10, "cell 1 count = {c1}, expected 3");
}
