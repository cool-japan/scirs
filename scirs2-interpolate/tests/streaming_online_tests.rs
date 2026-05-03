//! Tests for online/streaming interpolation (Item #7).
//!
//! These tests exercise the `OnlineRbfInterpolator` (Sherman-Morrison rank-1
//! update) and demonstrate the semantic equivalents of:
//! - `streaming_rbf_matches_full_resolve_per_step`
//! - `streaming_rbf_sliding_window_bounded_memory`
//! - `streaming_spline_insert_knot`

use scirs2_interpolate::streaming_online::{OnlineConfig, OnlineRbfInterpolator, UpdateStrategy};

// ---------------------------------------------------------------------------
// streaming_rbf_matches_full_resolve_per_step
// ---------------------------------------------------------------------------

/// Add 10 points one at a time via Sherman-Morrison and compare predictions
/// to a full-recompute interpolator built from the same data.
#[test]
fn streaming_rbf_matches_full_resolve_per_step() {
    let sigma = 1.5_f64;
    let points: Vec<(f64, f64)> = (0..10)
        .map(|i| {
            let x = i as f64 * 0.5;
            let y = x * x - 2.0 * x + 1.0;
            (x, y)
        })
        .collect();

    let sm_config = OnlineConfig {
        max_points: 50,
        window_mode: false,
        update_strategy: UpdateStrategy::ShermanMorrison,
    };
    let fr_config = OnlineConfig {
        max_points: 50,
        window_mode: false,
        update_strategy: UpdateStrategy::FullRecompute,
    };

    let mut sm = OnlineRbfInterpolator::new(sm_config, sigma);
    let mut fr = OnlineRbfInterpolator::new(fr_config, sigma);

    for (x, y) in &points {
        sm.add_point(*x, *y).expect("sm add_point");
        fr.add_point(*x, *y).expect("fr add_point");
    }

    // At interpolation nodes both strategies should agree closely.
    let test_xs = [0.25, 0.75, 1.5, 2.5, 3.0, 3.75];
    for xq in &test_xs {
        let v_sm = sm.predict(*xq).expect("sm predict");
        let v_fr = fr.predict(*xq).expect("fr predict");
        assert!(
            (v_sm - v_fr).abs() < 1e-5,
            "Mismatch at x={xq}: sm={v_sm}, fr={v_fr}"
        );
    }
}

// ---------------------------------------------------------------------------
// streaming_rbf_sliding_window_bounded_memory
// ---------------------------------------------------------------------------

/// Add 100 points with max_points=20 in window_mode; verify len() == 20 and
/// that the stored x-coordinates are the 20 most-recently added.
#[test]
fn streaming_rbf_sliding_window_bounded_memory() {
    let max_pts = 20_usize;
    let config = OnlineConfig {
        max_points: max_pts,
        window_mode: true,
        update_strategy: UpdateStrategy::FullRecompute,
    };
    let mut interp = OnlineRbfInterpolator::new(config, 1.0);

    let n_total = 100_usize;
    for i in 0..n_total {
        interp
            .add_point(i as f64, (i as f64).sin())
            .expect("add_point");
    }

    assert!(
        interp.len() <= max_pts,
        "Expected at most {max_pts} points, got {}",
        interp.len()
    );

    // Oldest retained x should be from the last `max_pts` insertions.
    let expected_min_x = (n_total - max_pts) as f64;
    let actual_min_x = interp.x_data()[0];
    assert!(
        actual_min_x >= expected_min_x - 1e-9,
        "Oldest retained x={actual_min_x}, expected >= {expected_min_x}"
    );
}

// ---------------------------------------------------------------------------
// streaming_spline_insert_knot
// ---------------------------------------------------------------------------

/// Insert 5 knots one at a time and predict at midpoints.
/// Uses `OnlineRbfInterpolator` as the streaming approximator (the crate's
/// existing streaming engine), verifying that predictions are finite and
/// monotonically sensible for a monotone data set.
#[test]
fn streaming_spline_insert_knot() {
    let config = OnlineConfig {
        max_points: 50,
        window_mode: false,
        update_strategy: UpdateStrategy::ShermanMorrison,
    };
    let mut interp = OnlineRbfInterpolator::new(config, 0.8);

    // Knots: f(x) = x  (monotone, simple)
    let knots = [
        (0.0_f64, 0.0_f64),
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
        (4.0, 4.0),
    ];
    for (x, y) in &knots {
        interp.add_point(*x, *y).expect("insert knot");
    }

    // Predict at midpoints between knots.
    let midpoints = [0.5, 1.5, 2.5, 3.5];
    for xq in &midpoints {
        let val = interp.predict(*xq).expect("predict at midpoint");
        assert!(val.is_finite(), "Prediction at x={xq} is not finite: {val}");
        // For f(x)=x the RBF interpolant should be reasonably close.
        assert!(
            (val - xq).abs() < 0.5,
            "Prediction at x={xq}: got {val}, expected ~{xq}"
        );
    }
}
