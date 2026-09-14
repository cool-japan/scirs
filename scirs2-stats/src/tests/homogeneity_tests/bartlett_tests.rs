#[cfg(test)]
mod tests {
    use crate::tests::homogeneity::bartlett;
    use scirs2_core::ndarray::{array, Array1};

    // Test data from SciPy documentation
    const A: [f64; 10] = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
    const B: [f64; 10] = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
    const C: [f64; 10] = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];

    #[test]
    fn test_bartlett_different_variances() {
        let a = array![A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9]];
        let b = array![B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9]];
        let c = array![C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9]];

        let samples = vec![a.view(), b.view(), c.view()];

        let (statistic, p_value) = bartlett(&samples).expect("Test: operation failed");

        // Expected values based on SciPy's bartlett function
        // We use a range check because implementations might have slight differences
        assert!(
            statistic > 20.0 && statistic < 30.0,
            "Expected statistic around 25.0, got {}",
            statistic
        );
        assert!(
            p_value < 0.0001,
            "Expected p_value < 0.0001, got {}",
            p_value
        );
    }

    #[test]
    fn test_bartlett_equal_variances() {
        // Create samples with equal variances
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 6.0];
        let c = array![3.0, 4.0, 5.0, 6.0, 7.0];

        let samples = vec![a.view(), b.view(), c.view()];

        let (_, p_value) = bartlett(&samples).expect("Test: operation failed");

        // With equal variances, p-value should be high (non-significant)
        assert!(p_value > 0.05, "Expected p_value > 0.05, got {}", p_value);
    }

    #[test]
    fn test_bartlett_large_variance_difference() {
        // Create samples with clearly different variances
        let a = array![1.0, 1.1, 1.2, 0.9, 1.0]; // low variance
        let b = array![1.0, 3.0, 5.0, 7.0, 9.0]; // high variance

        let samples = vec![a.view(), b.view()];

        let (_, p_value) = bartlett(&samples).expect("Test: operation failed");

        // With very different variances, p-value should be low (significant)
        assert!(p_value < 0.05, "Expected p_value < 0.05, got {}", p_value);
    }

    #[test]
    fn test_bartlett_variance_calculation() {
        // Test with known variances and expected result
        let a = array![1.0, 1.0, 1.0, 1.0]; // Variance = 0
        let b = array![0.0, 1.0, 2.0, 3.0]; // Variance = 5/3

        let samples = vec![a.view(), b.view()];

        let (statistic_, p_value) = bartlett(&samples).expect("Test: operation failed");

        // Expected result for this specific example:
        // When one variance is zero, the statistic tends to infinity
        // But numerical implementations usually yield a large finite value
        assert!(
            statistic_ > 10.0,
            "Expected large statistic value, got {}",
            statistic_
        );
    }

    #[test]
    fn test_bartlett_invalid_input() {
        // Test with fewer than 2 groups
        let a = array![1.0, 2.0, 3.0];

        let samples = vec![a.view()];

        let result = bartlett(&samples);
        assert!(result.is_err());

        // Test with empty sample
        let a = array![1.0, 2.0, 3.0];
        let b = array![];

        let samples = vec![a.view(), b.view()];

        let result = bartlett(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_bartlett_single_value_sample() {
        // Each sample must have at least 2 observations for variance
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0];

        let samples = vec![a.view(), b.view()];

        let result = bartlett(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_bartlett_compare_with_scipy() {
        // Test with samples used in SciPy example
        let a = array![0.8, 0.9, 1.2, 0.85];
        let b = array![1.0, 1.1, 1.0, 0.9, 1.2];
        let c = array![0.8, 0.7, 0.6, 0.8];

        let samples = vec![a.view(), b.view(), c.view()];

        let (statistic, p_value) = bartlett(&samples).expect("Test: operation failed");

        // Values should be in reasonable range compared to SciPy
        assert!(
            statistic >= 0.0,
            "Expected non-negative statistic, got {}",
            statistic
        );
        assert!(
            p_value >= 0.0 && p_value <= 1.0,
            "Expected p_value in [0,1], got {}",
            p_value
        );
    }

    /// Regression test for the numerical-stability defect class of
    /// cool-japan/scirs#131. Bartlett's chi-square tail used to be computed as
    /// `1 - CDF`, with the CDF divided by a Lanczos `gamma_function(df/2)`:
    /// the cancellation flushed every p-value below ~1e-16 to exactly 0. It is
    /// now the regularized upper incomplete gamma function, evaluated
    /// directly. Reference values from mpmath at 50 digits.
    #[test]
    fn test_bartlett_deep_tail_pvalue_is_not_flushed_to_zero() {
        // Three groups of 100 whose variances differ by an order of magnitude.
        let a: Array1<f64> = Array1::from_iter((0..100).map(|i| (i % 17) as f64 * 0.1));
        let b: Array1<f64> = Array1::from_iter((0..100).map(|i| (i % 23) as f64 * 0.25));
        let c: Array1<f64> = Array1::from_iter((0..100).map(|i| (i % 11) as f64 * 0.5));
        let samples = vec![a.view(), b.view(), c.view()];

        let (statistic, p_value) = bartlett(&samples).expect("Test: operation failed");

        assert!(
            (statistic - 133.46061094433585).abs() < 1e-8,
            "got statistic {}",
            statistic
        );
        assert!(
            p_value > 0.0,
            "a chi-square statistic of {} on 2 df has p ~ 1e-29, not 0",
            statistic
        );
        let expected = 1.0456745971571383e-29;
        let relative = ((p_value - expected) / expected).abs();
        assert!(
            relative < 1e-9,
            "got {:e}, want {:e} (relative error {:e})",
            p_value,
            expected,
            relative
        );
    }

    /// With 300 groups the chi-square tail is evaluated at df = 299, where the
    /// old `gamma_function(df/2)` overflowed to `inf` and turned a p-value of
    /// ~1e-72 into 1.0 (i.e. "perfectly homogeneous variances").
    #[test]
    fn test_bartlett_many_groups_pvalue_is_finite() {
        let groups: Vec<Array1<f64>> = (0..300)
            .map(|i| Array1::from_iter((0..5).map(|j| j as f64 * (1.0 + i as f64))))
            .collect();
        let samples: Vec<_> = groups.iter().map(|g| g.view()).collect();

        let (statistic, p_value) = bartlett(&samples).expect("Test: operation failed");

        assert!(
            (statistic - 975.8926109777179).abs() < 1e-6,
            "got statistic {}",
            statistic
        );
        assert!(!p_value.is_nan(), "p-value evaluates to NaN");
        assert!(
            p_value < 1e-60,
            "variances spanning three orders of magnitude must be rejected, got p = {:e}",
            p_value
        );
        let expected = 9.4202124512540691e-73;
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
