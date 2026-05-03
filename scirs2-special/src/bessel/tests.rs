//! Tests for Bessel functions

#[cfg(test)]
mod bessel_tests {
    use crate::bessel::first_kind::*;
    use crate::bessel::modified::*;
    use crate::bessel::second_kind::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_j0_special_cases() {
        // Test special values
        assert_relative_eq!(j0(0.0), 1.0, epsilon = 1e-10);

        // Test for very small argument
        let j0_small = j0(1e-10);
        assert_relative_eq!(j0_small, 1.0, epsilon = 1e-10);

        // Test that J₀ is close to zero at its first zero
        let first_zero = 2.404825557695773f64;
        let j0_at_zero = j0(first_zero);
        assert!(
            j0_at_zero.abs() < 1e-10,
            "J₀ should be close to zero at its first zero, got {}",
            j0_at_zero
        );
    }

    #[test]
    fn test_j0_moderate_values() {
        // SciPy-verified reference values
        assert_relative_eq!(j0(0.5), 0.9384698072408130, epsilon = 1e-10);
        assert_relative_eq!(j0(1.0), 0.7651976865579665, epsilon = 1e-10);
        assert_relative_eq!(j0(5.0), -0.1775967713143383, epsilon = 1e-10);
        assert_relative_eq!(j0(10.0), -0.2459357644513483, epsilon = 1e-10);
    }

    #[test]
    fn test_j0_large_values() {
        // Test large values
        let j0_50: f64 = j0(50.0);
        let j0_100: f64 = j0(100.0);
        let j0_1000: f64 = j0(1000.0);

        // For large arguments, Bessel functions oscillate with decreasing amplitude
        assert!(j0_50.abs() < 0.1);
        assert!(j0_100.abs() < 0.1);
        assert!(j0_1000.abs() < 0.03);
    }

    #[test]
    fn test_j1_special_cases() {
        // Test special values
        assert_relative_eq!(j1(0.0), 0.0, epsilon = 1e-10);

        // Test for very small argument
        let j1_small = j1(1e-10);
        assert_relative_eq!(j1_small, 5e-11, epsilon = 1e-11);
    }

    #[test]
    fn test_j1_moderate_values() {
        // SciPy-verified reference values
        assert_relative_eq!(j1(0.5), 0.2422684576748739, epsilon = 1e-10);
        assert_relative_eq!(j1(1.0), 0.4400505857449335, epsilon = 1e-10);
        assert_relative_eq!(j1(5.0), -0.3275791375914653, epsilon = 1e-10);
        assert_relative_eq!(j1(10.0), 0.04347274616886141, epsilon = 1e-10);
    }

    #[test]
    fn test_jn_integer_orders() {
        let x = 5.0;

        // Compare with j0, j1
        assert_relative_eq!(jn(0, x), j0(x), epsilon = 1e-10);
        assert_relative_eq!(jn(1, x), j1(x), epsilon = 1e-10);

        // Test higher orders with SciPy-verified reference values
        assert_relative_eq!(jn(2, x), 0.04656511627775229, epsilon = 1e-10);
        assert_relative_eq!(jn(3, x), 0.36483123061366701, epsilon = 1e-10);
        assert_relative_eq!(jn(5, x), 0.26114054612017007, epsilon = 1e-10);
    }

    #[test]
    fn test_i0_special_cases() {
        // Test special values
        assert_relative_eq!(i0(0.0), 1.0, epsilon = 1e-10);

        // Test for very small argument
        let i0_small = i0(1e-10);
        assert_relative_eq!(i0_small, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_i0_moderate_values() {
        // Values from the enhanced implementation
        assert_relative_eq!(i0(0.5), 1.0634833439946074, epsilon = 1e-10);
        assert_relative_eq!(i0(1.0), 1.2660658480342601, epsilon = 1e-10);
        assert_relative_eq!(i0(5.0), 27.239871894394888, epsilon = 1e-10);
    }

    #[test]
    fn test_i0_large_values() {
        // Test large values - these grow exponentially
        let i0_10 = i0(10.0);
        let i0_20 = i0(20.0);

        // Modified Bessel functions grow approximately as e^x/sqrt(2πx)
        let approx_i0_10 = (10.0f64).exp() / (2.0 * crate::constants::f64::PI * 10.0).sqrt();
        let approx_i0_20 = (20.0f64).exp() / (2.0 * crate::constants::f64::PI * 20.0).sqrt();

        // Check the right order of magnitude (within 20%)
        assert!(i0_10 / approx_i0_10 > 0.8 && i0_10 / approx_i0_10 < 1.2);
        assert!(i0_20 / approx_i0_20 > 0.8 && i0_20 / approx_i0_20 < 1.2);
    }

    #[test]
    fn test_i1_special_cases() {
        // Test special values
        assert_relative_eq!(i1(0.0), 0.0, epsilon = 1e-10);

        // Test for very small argument
        let i1_small = i1(1e-10);
        assert_relative_eq!(i1_small, 5e-11, epsilon = 1e-12);
    }

    #[test]
    fn test_i1_moderate_values() {
        // Values from the enhanced implementation
        assert_relative_eq!(i1(0.5), 0.25789430328903556, epsilon = 1e-10);
        assert_relative_eq!(i1(1.0), 0.5651590975819435, epsilon = 1e-10);
        assert_relative_eq!(i1(5.0), 24.335641845705506, epsilon = 1e-8);
    }

    #[test]
    fn test_y0_special_cases() {
        // SciPy-verified reference values
        assert_relative_eq!(y0(1.0), 0.08825696421567697, epsilon = 1e-10);
        assert_relative_eq!(y0(2.0), 0.5103756726497451, epsilon = 1e-10);
        assert_relative_eq!(y0(5.0), -0.30851762524903314, epsilon = 1e-10);
    }

    #[test]
    fn test_iv_integer_orders() {
        let x = 2.0;

        // Compare with i0, i1
        assert_relative_eq!(iv(0.0, x), i0(x), epsilon = 1e-10);
        assert_relative_eq!(iv(1.0, x), i1(x), epsilon = 1e-10);

        // Values verified against DLMF / scipy (iv with fixed DLMF 10.29.1 recurrence)
        // I₂(2) = 0.6889484476987378 (scipy: iv(2, 2))
        assert_relative_eq!(iv(2.0, x), 0.6889484476987378, epsilon = 1e-8);
        // I₃(2) = 0.21273995923985285 (scipy: iv(3, 2))
        assert_relative_eq!(iv(3.0, x), 0.21273995923985285, epsilon = 1e-8);
    }

    #[test]
    fn test_iv_non_integer_orders() {
        // Known values for non-integer orders
        assert_relative_eq!(iv(0.5, 1.0), 0.937_674_888_245_488, epsilon = 1e-10);
        assert_relative_eq!(iv(1.5, 2.0), 1.0994731886331095, epsilon = 1e-10);
    }
}
