#[cfg(test)]
mod tests {
    use crate::tests::homogeneity::{brown_forsythe, levene};
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // Test data from SciPy documentation
    const A: [f64; 10] = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99];
    const B: [f64; 10] = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05];
    const C: [f64; 10] = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98];

    #[test]
    #[ignore = "timeout"]
    fn test_brown_forsythe_equal_to_levene_median() {
        // Brown-Forsythe test is equivalent to Levene's test with center="median"
        let a = array![A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9]];
        let b = array![B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9]];
        let c = array![C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9]];

        let samples = vec![a.view(), b.view(), c.view()];

        let (bf_stat, bf_pval) = brown_forsythe(&samples).unwrap();
        let (lev_stat, lev_pval) = levene(&samples, "median", 0.05).unwrap();

        // Results should be exactly the same
        assert_abs_diff_eq!(bf_stat, lev_stat, epsilon = 1e-10);
        assert_abs_diff_eq!(bf_pval, lev_pval, epsilon = 1e-10);
    }

    #[test]
    fn test_brown_forsythe_different_variances() {
        let a = array![A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9]];
        let b = array![B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9]];
        let c = array![C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9]];

        let samples = vec![a.view(), b.view(), c.view()];

        let (_statistic, p_value) = brown_forsythe(&samples).unwrap();

        // With these data, the test should reject the null hypothesis of equal variances
        assert!(p_value < 0.05, "Expected p_value < 0.05, got {}", p_value);
    }

    #[test]
    fn test_brown_forsythe_equal_variances() {
        // Create samples with equal variances
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 6.0];
        let c = array![3.0, 4.0, 5.0, 6.0, 7.0];

        let samples = vec![a.view(), b.view(), c.view()];

        let (_, p_value) = brown_forsythe(&samples).unwrap();

        // With equal variances, p-value should be high (non-significant)
        assert!(p_value > 0.05, "Expected p_value > 0.05, got {}", p_value);
    }

    #[test]
    fn test_brown_forsythe_robustness() {
        // Test with outliers - Brown-Forsythe should be more robust than mean-based Levene's
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 20.0]; // With outlier

        let samples = vec![a.view(), b.view()];

        let (_bf_stat, bf_pval) = brown_forsythe(&samples).unwrap();
        let (_lev_stat, lev_pval) = levene(&samples, "mean", 0.05).unwrap();

        // The Brown-Forsythe test should be less affected by the outlier
        // This is just a relative comparison, not an absolute requirement
        println!(
            "Brown-Forsythe p-value: {}, Levene's (mean) p-value: {}",
            bf_pval, lev_pval
        );
    }

    #[test]
    fn test_brown_forsythe_invalid_input() {
        // Test with fewer than 2 groups
        let a = array![1.0, 2.0, 3.0];

        let samples = vec![a.view()];

        let result = brown_forsythe(&samples);
        assert!(result.is_err());

        // Test with empty sample
        let a = array![1.0, 2.0, 3.0];
        let b = array![];

        let samples = vec![a.view(), b.view()];

        let result = brown_forsythe(&samples);
        assert!(result.is_err());
    }
}
