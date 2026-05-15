//! Integration tests for the NUMA-aware parallel path in `regression::discover`.
//!
//! All tests are gated on `#[cfg(feature = "numa")]` — they are meaningless
//! without the feature because the dispatch constant and the parallel code
//! path do not exist in non-numa builds.

#[cfg(feature = "numa")]
mod numa_tests {
    use ndarray::{Array1, Array2};
    use scirs2_symbolic::regression::{discover, SrConfig, NUMA_DISPATCH_THRESHOLD};

    // ── Helper: build a simple single-feature dataset (y = x) ──────────────

    fn make_identity_dataset(n: usize) -> (Array2<f64>, Array1<f64>) {
        let xs: Vec<f64> = (0..n)
            .map(|i| (i as f64 - (n as f64 / 2.0)) * 0.01)
            .collect();
        let features = Array2::from_shape_vec((n, 1), xs.clone()).expect("shape");
        let targets = Array1::from_vec(xs);
        (features, targets)
    }

    // ── Test 1: above-threshold correctness ───────────────────────────────────

    /// With 4096 samples (> NUMA_DISPATCH_THRESHOLD = 1024), `predict` routes
    /// through `predict_parallel` / `par_map_chunks`.  The identity formula
    /// `Var(0)` is in the initial population so MSE on the best candidate must
    /// be near zero regardless of whether we use the parallel or serial path.
    ///
    /// To confirm the parallel and serial paths agree, we also run the same
    /// search at 512 samples (below threshold) and verify both return the
    /// same top formula kind (identity) with MSE < 1e-10.
    #[test]
    fn predict_parallel_above_threshold_correctness() {
        let n = 4096; // well above NUMA_DISPATCH_THRESHOLD
        assert!(
            n >= NUMA_DISPATCH_THRESHOLD,
            "test invariant: n must be >= NUMA_DISPATCH_THRESHOLD"
        );

        let (features_large, targets_large) = make_identity_dataset(n);
        let (features_small, targets_small) = make_identity_dataset(512);

        let config = SrConfig::default().with_max_iter(5).with_top_n(1);

        let results_large = discover(features_large.view(), targets_large.view(), &config);
        let results_small = discover(features_small.view(), targets_small.view(), &config);

        // Both runs must find the identity formula with near-zero MSE.
        assert!(
            !results_large.is_empty(),
            "parallel path returned no formulas"
        );
        assert!(
            !results_small.is_empty(),
            "serial path returned no formulas"
        );

        let mse_large = results_large[0].fitness.mse;
        let mse_small = results_small[0].fitness.mse;

        assert!(
            mse_large < 1e-10,
            "parallel path MSE = {mse_large} (expected < 1e-10)"
        );
        assert!(
            mse_small < 1e-10,
            "serial path MSE = {mse_small} (expected < 1e-10)"
        );

        // The discovered formula structure must match across paths.
        assert_eq!(
            results_large[0].op, results_small[0].op,
            "parallel and serial paths must discover the same top formula"
        );
    }

    // ── Test 2: below-threshold correctness ───────────────────────────────────

    /// With 256 samples (< NUMA_DISPATCH_THRESHOLD), `predict` stays on the
    /// serial path.  The identity formula must still be discoverable and the
    /// MSE must be near zero.  This guards against regressions where the
    /// dispatch rewrite accidentally breaks the serial fallback.
    #[test]
    fn predict_parallel_below_threshold_uses_rayon() {
        let n = 256; // well below NUMA_DISPATCH_THRESHOLD
        assert!(
            n < NUMA_DISPATCH_THRESHOLD,
            "test invariant: n must be < NUMA_DISPATCH_THRESHOLD"
        );

        let (features, targets) = make_identity_dataset(n);
        let config = SrConfig::default().with_max_iter(10).with_top_n(1);

        let results = discover(features.view(), targets.view(), &config);

        assert!(!results.is_empty(), "serial path returned no formulas");
        let mse = results[0].fitness.mse;
        assert!(mse < 1e-10, "serial path MSE = {mse} (expected < 1e-10)");
    }

    // ── Test 3: constant value assertion ──────────────────────────────────────

    /// Asserts that the dispatch constant used to route between the serial and
    /// NUMA-parallel paths is exactly 1024.  This number is referenced in
    /// documentation, benchmarks, and the TODO item; a silent change would
    /// break profiling expectations.
    #[test]
    fn numa_dispatch_threshold_constant() {
        assert_eq!(
            NUMA_DISPATCH_THRESHOLD, 1024,
            "NUMA_DISPATCH_THRESHOLD must be 1024 (see regression/discover.rs)"
        );
    }
}
