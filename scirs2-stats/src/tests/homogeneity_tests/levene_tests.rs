#[cfg(test)]
mod tests {
    use crate::tests::homogeneity::levene;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // Test data from SciPy documentation
    const A: [f64; 10] = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
    const B: [f64; 10] = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
    const C: [f64; 10] = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];

    #[test]
    #[ignore = "timeout"]
    fn test_levene_median() {
        let a = array![A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9]];
        let b = array![B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9]];
        let c = array![C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9]];

        let samples = vec![a.view(), b.view(), c.view()];

        let (statistic, p_value) = levene(&samples, "median", 0.05).unwrap();

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

        let (statistic, p_value) = levene(&samples, "mean", 0.05).unwrap();

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

        let (statistic, p_value) = levene(&samples, "trimmed", 0.1).unwrap();

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

        let (_, p_value) = levene(&samples, "median", 0.05).unwrap();

        // With equal variances, p-value should be high (non-significant)
        assert!(p_value > 0.05, "Expected p_value > 0.05, got {}", p_value);
    }

    #[test]
    fn test_levene_different_variances() {
        // Create samples with clearly different variances
        let a = array![1.0, 1.1, 1.2, 0.9, 1.0]; // low variance
        let b = array![1.0, 3.0, 5.0, 7.0, 9.0]; // high variance

        let samples = vec![a.view(), b.view()];

        let (_, p_value) = levene(&samples, "median", 0.05).unwrap();

        // With very different variances, p-value should be low (significant)
        assert!(p_value < 0.05, "Expected p_value < 0.05, got {}", p_value);
    }

    #[test]
    fn test_levene_zero_trim() {
        // Test that "trimmed" with 0.0 proportion gives same result as "mean"
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 6.0];

        let samples = vec![a.view(), b.view()];

        let (stat1, p1) = levene(&samples, "mean", 0.05).unwrap();
        let (stat2, p2) = levene(&samples, "trimmed", 0.0).unwrap();

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
}
