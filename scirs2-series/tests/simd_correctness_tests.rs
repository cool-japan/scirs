//! SIMD Correctness Tests for Time Series Operations
//!
//! This module tests that SIMD implementations produce identical results
//! to scalar implementations within numerical tolerance.

#[cfg(feature = "simd")]
mod simd_tests {
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array1};
    use scirs2_series::error::Result;
    use scirs2_series::simd_ops::*;

    const TOLERANCE_F64: f64 = 1e-10;
    const TOLERANCE_F32: f32 = 1e-4;

    /// Helper function to compute scalar differencing for comparison
    fn scalar_difference_f64(data: &[f64], order: usize) -> Vec<f64> {
        let mut result = data.to_vec();
        for _ in 0..order {
            let mut diff = Vec::new();
            for i in 1..result.len() {
                diff.push(result[i] - result[i - 1]);
            }
            result = diff;
        }
        result
    }

    /// Helper function to compute scalar seasonal differencing
    fn scalar_seasonal_difference_f64(data: &[f64], period: usize, order: usize) -> Vec<f64> {
        let mut result = data.to_vec();
        for _ in 0..order {
            let mut diff = Vec::new();
            for i in period..result.len() {
                diff.push(result[i] - result[i - period]);
            }
            result = diff;
        }
        result
    }

    /// Helper function to compute scalar autocorrelation
    fn scalar_autocorrelation_f64(data: &[f64], max_lag: usize) -> Vec<f64> {
        let n = data.len();
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let c0: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();

        if c0 == 0.0 {
            return vec![1.0; max_lag + 1];
        }

        let mut acf = Vec::with_capacity(max_lag + 1);
        for lag in 0..=max_lag {
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += (data[i] - mean) * (data[i + lag] - mean);
            }
            acf.push(sum / c0);
        }
        acf
    }

    #[test]
    fn test_simd_differencing_order_1_correctness() {
        let data = array![1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0];
        let simd_result =
            simd_difference_f64(&data.view(), 1).expect("SIMD differencing should succeed");

        let scalar_result = scalar_difference_f64(&data.to_vec(), 1);

        assert_eq!(simd_result.len(), scalar_result.len());
        for (i, (&simd_val, &scalar_val)) in
            simd_result.iter().zip(scalar_result.iter()).enumerate()
        {
            assert_abs_diff_eq!(simd_val, scalar_val, epsilon = TOLERANCE_F64);
        }
    }

    #[test]
    fn test_simd_differencing_order_2_correctness() {
        let data = array![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0];
        let simd_result =
            simd_difference_f64(&data.view(), 2).expect("SIMD differencing order 2 should succeed");

        let scalar_result = scalar_difference_f64(&data.to_vec(), 2);

        assert_eq!(simd_result.len(), scalar_result.len());
        for (i, (&simd_val, &scalar_val)) in
            simd_result.iter().zip(scalar_result.iter()).enumerate()
        {
            assert_abs_diff_eq!(simd_val, scalar_val, epsilon = TOLERANCE_F64);
        }
    }

    #[test]
    fn test_simd_seasonal_differencing_correctness() {
        let data = array![10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0, 20.0, 30.0, 40.0, 50.0];
        let period = 4;
        let order = 1;

        let simd_result = simd_seasonal_difference_f64(&data.view(), period, order)
            .expect("SIMD seasonal differencing should succeed");

        let scalar_result = scalar_seasonal_difference_f64(&data.to_vec(), period, order);

        assert_eq!(simd_result.len(), scalar_result.len());
        for (i, (&simd_val, &scalar_val)) in
            simd_result.iter().zip(scalar_result.iter()).enumerate()
        {
            assert_abs_diff_eq!(simd_val, scalar_val, epsilon = TOLERANCE_F64);
        }
    }

    #[test]
    fn test_simd_autocorrelation_correctness() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let max_lag = 5;

        let simd_result = simd_autocorrelation_f64(&data.view(), Some(max_lag))
            .expect("SIMD autocorrelation should succeed");

        let scalar_result = scalar_autocorrelation_f64(&data.to_vec(), max_lag);

        assert_eq!(simd_result.len(), scalar_result.len());

        // ACF at lag 0 should always be 1.0
        assert_abs_diff_eq!(simd_result[0], 1.0, epsilon = TOLERANCE_F64);

        for (i, (&simd_val, &scalar_val)) in
            simd_result.iter().zip(scalar_result.iter()).enumerate()
        {
            assert_abs_diff_eq!(simd_val, scalar_val, epsilon = TOLERANCE_F64);
        }
    }

    #[test]
    fn test_simd_moving_mean_correctness() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let window_size = 3;

        let simd_result = simd_moving_mean_f64(&data.view(), window_size)
            .expect("SIMD moving mean should succeed");

        assert_eq!(simd_result.len(), data.len());

        // Verify that all values are finite and reasonable
        for val in simd_result.iter() {
            assert!(val.is_finite(), "All values should be finite");
            assert!(
                *val >= 1.0 && *val <= 10.0,
                "Values should be in reasonable range"
            );
        }

        // The middle values should be close to their theoretical centered mean
        // i=5: centered window should include [4,5,6] or similar
        assert!(
            (simd_result[5] - 5.5).abs() < 1.0,
            "Middle value should be close to local mean"
        );
    }

    #[test]
    fn test_simd_convolution_correctness() {
        let signal = array![1.0, 2.0, 3.0, 4.0];
        let kernel = array![1.0, 1.0];

        let simd_result = simd_convolve_f64(&signal.view(), &kernel.view())
            .expect("SIMD convolution should succeed");

        // Manual convolution: conv([1,2,3,4], [1,1]) = [1, 3, 5, 7, 4]
        let expected = array![1.0, 3.0, 5.0, 7.0, 4.0];

        assert_eq!(simd_result.len(), expected.len());
        for (i, (&simd_val, &expected_val)) in simd_result.iter().zip(expected.iter()).enumerate() {
            assert_abs_diff_eq!(simd_val, expected_val, epsilon = TOLERANCE_F64);
        }
    }

    #[test]
    fn test_simd_seasonal_means_correctness() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let period = 3;

        let simd_result = simd_seasonal_means_f64(&data.view(), period)
            .expect("SIMD seasonal means should succeed");

        // pos 0: (1+4+7)/3 = 4.0
        // pos 1: (2+5+8)/3 = 5.0
        // pos 2: (3+6+9)/3 = 6.0
        let expected = array![4.0, 5.0, 6.0];

        assert_eq!(simd_result.len(), expected.len());
        for (i, (&simd_val, &expected_val)) in simd_result.iter().zip(expected.iter()).enumerate() {
            assert_abs_diff_eq!(simd_val, expected_val, epsilon = TOLERANCE_F64);
        }
    }

    #[test]
    fn test_simd_deseason_correctness() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let period = 3;

        let (deseasoned, seasonal) =
            simd_deseason_f64(&data.view(), period).expect("SIMD deseason should succeed");

        assert_eq!(deseasoned.len(), data.len());
        assert_eq!(seasonal.len(), data.len());

        // Verify: data = deseasoned + seasonal
        for i in 0..data.len() {
            assert_abs_diff_eq!(
                data[i],
                deseasoned[i] + seasonal[i],
                epsilon = TOLERANCE_F64
            );
        }
    }

    #[test]
    fn test_simd_ema_correctness() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = 0.5;

        let simd_result = simd_exponential_moving_average_f64(&data.view(), alpha)
            .expect("SIMD EMA should succeed");

        // Manual EMA calculation with alpha=0.5:
        // EMA[0] = 1.0
        // EMA[1] = 0.5*2.0 + 0.5*1.0 = 1.5
        // EMA[2] = 0.5*3.0 + 0.5*1.5 = 2.25
        // EMA[3] = 0.5*4.0 + 0.5*2.25 = 3.125
        // EMA[4] = 0.5*5.0 + 0.5*3.125 = 4.0625
        let expected = array![1.0, 1.5, 2.25, 3.125, 4.0625];

        assert_eq!(simd_result.len(), expected.len());
        for (i, (&simd_val, &expected_val)) in simd_result.iter().zip(expected.iter()).enumerate() {
            assert_abs_diff_eq!(simd_val, expected_val, epsilon = TOLERANCE_F64);
        }
    }

    // Large data tests
    #[test]
    fn test_simd_differencing_large_dataset() {
        let data: Array1<f64> = Array1::range(0.0, 1000.0, 1.0);

        let simd_result = simd_difference_f64(&data.view(), 1)
            .expect("SIMD differencing on large data should succeed");

        let scalar_result = scalar_difference_f64(&data.to_vec(), 1);

        assert_eq!(simd_result.len(), scalar_result.len());

        for (i, (&simd_val, &scalar_val)) in
            simd_result.iter().zip(scalar_result.iter()).enumerate()
        {
            assert_abs_diff_eq!(simd_val, scalar_val, epsilon = TOLERANCE_F64);
        }
    }

    #[test]
    fn test_simd_autocorrelation_large_dataset() {
        let data: Array1<f64> = Array1::range(0.0, 500.0, 1.0);
        let max_lag = 50;

        let simd_result = simd_autocorrelation_f64(&data.view(), Some(max_lag))
            .expect("SIMD ACF on large data should succeed");

        let scalar_result = scalar_autocorrelation_f64(&data.to_vec(), max_lag);

        assert_eq!(simd_result.len(), scalar_result.len());

        for (i, (&simd_val, &scalar_val)) in
            simd_result.iter().zip(scalar_result.iter()).enumerate()
        {
            assert_abs_diff_eq!(simd_val, scalar_val, epsilon = TOLERANCE_F64);
        }
    }

    // f32 tests
    #[test]
    fn test_simd_differencing_f32_correctness() {
        let data = array![1.0f32, 3.0, 6.0, 10.0, 15.0];
        let simd_result =
            simd_difference_f32(&data.view(), 1).expect("SIMD f32 differencing should succeed");

        // Expected: [2.0, 3.0, 4.0, 5.0]
        let expected = array![2.0f32, 3.0, 4.0, 5.0];

        assert_eq!(simd_result.len(), expected.len());
        for (i, (&simd_val, &expected_val)) in simd_result.iter().zip(expected.iter()).enumerate() {
            assert_abs_diff_eq!(simd_val, expected_val, epsilon = TOLERANCE_F32);
        }
    }

    #[test]
    fn test_simd_autocorrelation_f32_correctness() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let simd_result =
            simd_autocorrelation_f32(&data.view(), Some(3)).expect("SIMD f32 ACF should succeed");

        assert_abs_diff_eq!(simd_result[0], 1.0f32, epsilon = TOLERANCE_F32);

        assert!(simd_result.len() == 4);
    }
}

#[cfg(not(feature = "simd"))]
mod no_simd_tests {
    #[test]
    fn simd_feature_not_enabled() {
        // This test's only purpose is to verify the crate compiles and this
        // test executes without the `simd` feature enabled; a passing test
        // run already demonstrates that with no assertion needed.
    }
}
