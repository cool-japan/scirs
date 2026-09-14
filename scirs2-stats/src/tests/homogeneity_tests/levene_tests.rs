#[cfg(test)]
mod tests {
    use crate::tests::homogeneity::levene;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array1};

    // Test data from SciPy documentation
    const A: [f64; 10] = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
    const B: [f64; 10] = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
    const C: [f64; 10] = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];

    #[test]
    fn test_levene_median() {
        let a = array![A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9]];
        let b = array![B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9]];
        let c = array![C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9]];

        let samples = vec![a.view(), b.view(), c.view()];

        let (statistic, p_value) =
            levene(&samples, "median", 0.05).expect("Test: operation failed");

        // The expected values are from SciPy's implementation
        // Using an epsilon value to account for different implementations
        assert!(
            statistic > 5.0 && statistic < 8.0,
            "Expected statistic around 6.95, got {}",
            statistic
        );
        assert!(p_value < 0.01, "Expected p_value < 0.01, got {}", p_value);
    }

    #[test]
    fn test_levene_mean() {
        let a = array![A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9]];
        let b = array![B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9]];
        let c = array![C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9]];

        let samples = vec![a.view(), b.view(), c.view()];

        let (statistic, p_value) = levene(&samples, "mean", 0.05).expect("Test: operation failed");

        // Mean and median give different results
        assert!(
            statistic > 3.0,
            "Expected statistic > 3.0, got {}",
            statistic
        );
        assert!(p_value < 0.05, "Expected p_value < 0.05, got {}", p_value);
    }

    #[test]
    fn test_levene_trimmed() {
        let a = array![A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9]];
        let b = array![B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9]];
        let c = array![C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9]];

        let samples = vec![a.view(), b.view(), c.view()];

        let (statistic, p_value) =
            levene(&samples, "trimmed", 0.1).expect("Test: operation failed");

        // Results should be different from mean and median
        assert!(
            statistic > 0.0,
            "Expected positive statistic, got {}",
            statistic
        );
        assert!(
            p_value > 0.0 && p_value < 1.0,
            "Expected p_value in (0,1), got {}",
            p_value
        );
    }

    #[test]
    fn test_levene_equal_variances() {
        // Create samples with equal variances
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 6.0];
        let c = array![3.0, 4.0, 5.0, 6.0, 7.0];

        let samples = vec![a.view(), b.view(), c.view()];

        let (_, p_value) = levene(&samples, "median", 0.05).expect("Test: operation failed");

        // With equal variances, p-value should be high (non-significant)
        assert!(p_value > 0.05, "Expected p_value > 0.05, got {}", p_value);
    }

    #[test]
    fn test_levene_different_variances() {
        // Create samples with clearly different variances
        let a = array![1.0, 1.1, 1.2, 0.9, 1.0]; // low variance
        let b = array![1.0, 3.0, 5.0, 7.0, 9.0]; // high variance

        let samples = vec![a.view(), b.view()];

        let (_, p_value) = levene(&samples, "median", 0.05).expect("Test: operation failed");

        // With very different variances, p-value should be low (significant)
        assert!(p_value < 0.05, "Expected p_value < 0.05, got {}", p_value);
    }

    #[test]
    fn test_levene_zero_trim() {
        // Test that "trimmed" with 0.0 proportion gives same result as "mean"
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 6.0];

        let samples = vec![a.view(), b.view()];

        let (stat1, p1) = levene(&samples, "mean", 0.05).expect("Test: operation failed");
        let (stat2, p2) = levene(&samples, "trimmed", 0.0).expect("Test: operation failed");

        assert_abs_diff_eq!(stat1, stat2, epsilon = 1e-10);
        assert_abs_diff_eq!(p1, p2, epsilon = 1e-10);
    }

    #[test]
    fn test_levene_invalid_center() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let samples = vec![a.view(), b.view()];

        let result = levene(&samples, "invalid", 0.05);
        assert!(result.is_err());
    }

    #[test]
    fn test_levene_too_few_samples() {
        let a = array![1.0, 2.0, 3.0];

        let samples = vec![a.view()];

        let result = levene(&samples, "median", 0.05);
        assert!(result.is_err());
    }

    #[test]
    fn test_levene_empty_sample() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![];

        let samples = vec![a.view(), b.view()];

        let result = levene(&samples, "median", 0.05);
        assert!(result.is_err());
    }

    /// Regression test for the numerical-stability defect class of
    /// cool-japan/scirs#131. Levene's p-value used to be built from a Lanczos
    /// *gamma* approximation of `B(df2/2, df1/2)`, which overflows to `inf`
    /// above ~142: every test with `df2 = n_total - k >= 284` (about 286
    /// observations) returned `inf` or `NaN`. The statistic and p-value below
    /// were computed independently with mpmath at 50 digits, NOT derived from
    /// this crate.
    #[test]
    fn test_levene_large_sample_pvalue_is_finite() {
        // Three groups of 100 with clearly different spreads:
        // n_total = 300, k = 3, so df1 = 2 and df2 = 297.
        let a: Array1<f64> = Array1::from_iter((0..100).map(|i| (i % 17) as f64 * 0.1));
        let b: Array1<f64> = Array1::from_iter((0..100).map(|i| (i % 23) as f64 * 0.25));
        let c: Array1<f64> = Array1::from_iter((0..100).map(|i| (i % 11) as f64 * 0.5));
        let samples = vec![a.view(), b.view(), c.view()];

        let (statistic, p_value) =
            levene(&samples, "median", 0.05).expect("Test: operation failed");

        assert_abs_diff_eq!(statistic, 71.17399631112818, epsilon = 1e-8);
        assert!(!p_value.is_nan(), "p-value evaluates to NaN");
        assert!(
            p_value.is_finite() && (0.0..=1.0).contains(&p_value),
            "expected a probability, got {}",
            p_value
        );
        let expected = 5.5879012840629727e-26;
        let relative = ((p_value - expected) / expected).abs();
        assert!(
            relative < 1e-9,
            "got {:e}, want {:e} (relative error {:e})",
            p_value,
            expected,
            relative
        );
    }

    /// Two groups of 150 observations, i.e. df2 = 298 -- one past the cliff
    /// where the old Beta function returned exactly 0 and the p-value `inf`.
    #[test]
    fn test_levene_two_large_groups_pvalue_matches_reference() {
        let a: Array1<f64> = Array1::from_iter((0..150).map(|i| (i % 19) as f64 * 0.2));
        let b: Array1<f64> = Array1::from_iter((0..150).map(|i| (i % 29) as f64 * 0.45));
        let samples = vec![a.view(), b.view()];

        let (statistic, p_value) =
            levene(&samples, "median", 0.05).expect("Test: operation failed");

        assert_abs_diff_eq!(statistic, 220.55876676712205, epsilon = 1e-8);
        let expected = 1.0034461323451784e-37;
        let relative = ((p_value - expected) / expected).abs();
        assert!(
            relative < 1e-9,
            "got {:e}, want {:e} (relative error {:e})",
            p_value,
            expected,
            relative
        );
    }
}
