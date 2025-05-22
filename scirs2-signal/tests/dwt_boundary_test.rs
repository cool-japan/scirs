#[cfg(test)]
mod dwt_boundary_tests {
    use scirs2_signal::dwt::extend_signal;

    #[test]
    fn test_symmetric_extension() {
        // Test with a simple signal
        let signal = vec![1.0, 2.0, 3.0, 4.0];

        // Extend with filter length 4 (pad by 3)
        let extended = extend_signal(&signal, 4, "symmetric").unwrap();

        // For symmetric extension, signal is reflected at the boundaries
        // Expected pattern: [2, 1, *1, 2, 3, 4*, 4, 3]
        // where * marks the original signal

        // Check length
        assert_eq!(extended.len(), signal.len() + 2 * (4 - 1));

        // Check values
        let expected = vec![2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(extended, expected);
    }

    #[test]
    fn test_periodic_extension() {
        // Test with a simple signal
        let signal = vec![1.0, 2.0, 3.0, 4.0];

        // Extend with filter length 4 (pad by 3)
        let extended = extend_signal(&signal, 4, "periodic").unwrap();

        // For periodic extension, signal is repeated
        // Expected pattern: [2, 3, 4, *1, 2, 3, 4*, 1, 2, 3]
        // where * marks the original signal

        // Check length
        assert_eq!(extended.len(), signal.len() + 2 * (4 - 1));

        // Check values
        let expected = vec![2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0];
        assert_eq!(extended, expected);
    }

    #[test]
    fn test_zero_extension() {
        // Test with a simple signal
        let signal = vec![1.0, 2.0, 3.0, 4.0];

        // Extend with filter length 4 (pad by 3)
        let extended = extend_signal(&signal, 4, "zero").unwrap();

        // For zero extension, signal is padded with zeros
        // Expected pattern: [0, 0, 0, *1, 2, 3, 4*, 0, 0, 0]
        // where * marks the original signal

        // Check length
        assert_eq!(extended.len(), signal.len() + 2 * (4 - 1));

        // Check values
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0];
        assert_eq!(extended, expected);
    }

    #[test]
    fn test_constant_extension() {
        // Test with a simple signal
        let signal = vec![1.0, 2.0, 3.0, 4.0];

        // Extend with filter length 4 (pad by 3)
        let extended = extend_signal(&signal, 4, "constant").unwrap();

        // For constant extension, signal is padded with edge values
        // Expected pattern: [1, 1, 1, *1, 2, 3, 4*, 4, 4, 4]
        // where * marks the original signal

        // Check length
        assert_eq!(extended.len(), signal.len() + 2 * (4 - 1));

        // Check values
        let expected = vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        assert_eq!(extended, expected);
    }

    #[test]
    fn test_reflect_extension() {
        // Test with a simple signal
        let signal = vec![1.0, 2.0, 3.0, 4.0];

        // Extend with filter length 4 (pad by 3)
        let extended = extend_signal(&signal, 4, "reflect").unwrap();

        // For reflect extension, signal is reflected without repeating edge values
        // Expected pattern: [3, 2, 1, *1, 2, 3, 4*, 3, 2, 1]
        // where * marks the original signal

        // Check length
        assert_eq!(extended.len(), signal.len() + 2 * (4 - 1));

        // Check values
        let expected = vec![3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(extended, expected);
    }

    #[test]
    fn test_invalid_mode() {
        // Test with an invalid extension mode
        let signal = vec![1.0, 2.0, 3.0, 4.0];

        // Should return an error
        let result = extend_signal(&signal, 4, "invalid_mode");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_signal() {
        // Test with an empty signal
        let signal: Vec<f64> = vec![];

        // Should return an empty extended signal
        let extended = extend_signal(&signal, 4, "symmetric").unwrap();
        assert_eq!(extended.len(), 2 * (4 - 1));
    }
}
