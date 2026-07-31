#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    use crate::tests::normality::ks_2samp;

    #[test]
    fn test_ks_2samp_same_distribution() {
        // Two samples from the same uniform distribution
        let x = array![0.1, 0.2, 0.3, 0.4, 0.5];
        let y = array![0.15, 0.25, 0.35, 0.45, 0.55];

        let (stat, p_value) =
            ks_2samp(&x.view(), &y.view(), "two-sided").expect("Test: operation failed");

        // The test statistic should be reasonable for these small samples
        assert!(stat <= 0.5);
        // And the p-value should be large (not rejecting the null hypothesis)
        assert!(p_value >= 0.01);
    }

    #[test]
    fn test_ks_2samp_different_distributions() {
        // Two samples from clearly different distributions
        let x = array![0.1, 0.2, 0.3, 0.4, 0.5];
        let y = array![5.1, 5.2, 5.3, 5.4, 5.5];

        let (stat, p_value) =
            ks_2samp(&x.view(), &y.view(), "two-sided").expect("Test: operation failed");

        // The test statistic should be 1.0 (maximum difference)
        assert_relative_eq!(stat, 1.0, epsilon = 1e-10);
        // With such a small sample, the p-value might not be that small,
        // but there should be clear evidence these are from different distributions
        assert!(p_value <= 0.2);
    }

    #[test]
    fn test_ks_2samp_one_sided_less() {
        // Use larger samples for clearer results
        let x = array![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let y = array![1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4];

        // Test if x is stochastically less than y (which is true): the null
        // hypothesis of the "less" test ("CDF of x >= CDF of y", i.e. x <= y)
        // holds here, so we expect a HIGH p-value (fail to reject a true null).
        let (_stat_less, p_value_less) =
            ks_2samp(&x.view(), &y.view(), "less").expect("Test: operation failed");

        // Test the opposite direction (which should be false): the null
        // hypothesis of the "greater" test ("x >= y") does NOT hold here, so
        // we expect a comparatively LOWER p-value (evidence against a false null).
        let (_stat_greater, p_value_greater) =
            ks_2samp(&x.view(), &y.view(), "greater").expect("Test: operation failed");

        // A one-sided test whose null hypothesis actually matches the data
        // ("less" here) should not be rejected as readily as one whose null
        // is contradicted by the data ("greater" here), so its p-value
        // should be the larger of the two.
        assert!(
            p_value_less > p_value_greater,
            "p_value_less={p_value_less} should exceed p_value_greater={p_value_greater}"
        );
    }

    #[test]
    fn test_ks_2samp_one_sided_greater() {
        // Use larger samples for clearer results
        let y = array![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let x = array![1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4];

        // Test if x is stochastically greater than y (which is true): the
        // null hypothesis of the "greater" test ("x >= y") holds here, so we
        // expect a HIGH p-value (fail to reject a true null).
        let (_stat, p_value_greater) =
            ks_2samp(&x.view(), &y.view(), "greater").expect("Test: operation failed");

        // Test the opposite direction (which should be false): the null
        // hypothesis of the "less" test ("x <= y") does NOT hold here, so we
        // expect a comparatively LOWER p-value (evidence against a false null).
        let (_stat, p_value_less) =
            ks_2samp(&x.view(), &y.view(), "less").expect("Test: operation failed");

        // This verifies that the alternative hypothesis is working correctly:
        // the one-sided test whose null actually matches the data ("greater")
        // should have the larger p-value of the two.
        assert!(
            p_value_greater > p_value_less,
            "p_value_greater={p_value_greater} should exceed p_value_less={p_value_less}"
        );
    }

    #[test]
    fn test_ks_2samp_empty_arrays() {
        // Empty arrays should return errors
        let empty = array![];
        let nonempty = array![1.0, 2.0, 3.0];

        assert!(ks_2samp(&empty.view(), &nonempty.view(), "two-sided").is_err());
        assert!(ks_2samp(&nonempty.view(), &empty.view(), "two-sided").is_err());
        assert!(ks_2samp(&empty.view(), &empty.view(), "two-sided").is_err());
    }

    #[test]
    fn test_ks_2samp_invalid_alternative() {
        // Invalid alternative hypothesis should return an error
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];

        assert!(ks_2samp(&x.view(), &y.view(), "invalid").is_err());
    }
}
