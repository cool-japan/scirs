//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use scirs2_core::ndarray::{Array1, Array2};
use statrs::statistics::Statistics;

use super::functions::invert_matrix;
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;
    #[test]
    fn test_invert_matrix_identity() {
        let a = array![[4.0_f64, 3.0], [6.0, 3.0]];
        let inv = invert_matrix(&a).expect("matrix should be invertible");
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += a[[i, k]] * inv[[k, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(acc, expected, epsilon = 1e-10);
            }
        }
    }
    #[test]
    fn test_invert_matrix_singular_errors() {
        let a = array![[1.0_f64, 2.0], [2.0, 4.0]];
        assert!(invert_matrix(&a).is_err());
    }
    #[test]
    fn test_is_stationary_returns_real_statistic() {
        let stationary = array![
            0.5_f64, -0.3, 0.2, -0.4, 0.1, -0.2, 0.3, -0.1, 0.25, -0.35, 0.15, -0.25, 0.05, -0.15,
            0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2, -0.25, 0.15, -0.2
        ];
        let (stat, p) = is_stationary(&stationary, Some(1)).expect("ADF should succeed");
        assert!(stat.is_finite());
        assert!((0.0..=1.0).contains(&p));
        let trending = array![
            1.0_f64, 2.3, 3.1, 4.6, 5.2, 6.9, 7.4, 8.8, 9.3, 10.7, 11.2, 12.9, 13.4, 14.1, 15.8,
            16.2, 17.9, 18.3, 19.7, 20.4, 21.1, 22.8, 23.2, 24.9
        ];
        let (stat_trend, _p_trend) = is_stationary(&trending, Some(1)).expect("ADF should succeed");
        assert!((stat - stat_trend).abs() > 1e-9);
    }
    #[test]
    fn test_is_stationary_too_short() {
        let ts = array![1.0_f64, 2.0];
        assert!(is_stationary(&ts, None).is_err());
    }
    #[test]
    fn test_is_stationary_linear_ramp_no_panic() {
        let ramp = Array1::from_vec((1..=20).map(|i| i as f64).collect());
        let (stat, p) = is_stationary(&ramp, None).expect("linear ramp must not error");
        assert!(stat.is_finite());
        assert!((0.0..=1.0).contains(&p));
    }
    #[test]
    fn test_is_stationary_constant_series_no_panic() {
        let constant = Array1::from_elem(20, 5.0_f64);
        let (stat, p) = is_stationary(&constant, None).expect("constant series must not error");
        assert!(stat.is_finite());
        assert!((0.0..=1.0).contains(&p));
    }
    #[test]
    fn test_detrend_constant() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let detrended = detrend(&x.view(), 0, "constant", None).expect("Operation failed");
        assert_relative_eq!(detrended.clone().mean(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[0], -2.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[2], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[4], 2.0, epsilon = 1e-10);
    }
    #[test]
    fn test_detrend_linear() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let detrended = detrend(&x.view(), 0, "linear", None).expect("Operation failed");
        for i in 1..detrended.len() {
            assert_relative_eq!(detrended[i] - detrended[i - 1], 0.0, epsilon = 1e-10);
        }
    }
    #[test]
    fn test_detrend_linear_with_breakpoints() {
        let x = array![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
        let breakpoints = vec![4];
        let detrended =
            detrend(&x.view(), 0, "linear", Some(&breakpoints)).expect("Operation failed");
        assert_relative_eq!(detrended[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[3], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[4], 0.0, epsilon = 1e-10);
        assert_relative_eq!(detrended[7], 0.0, epsilon = 1e-10);
    }
    #[test]
    fn test_detrend_2d() {
        let x = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .expect("Operation failed");
        let detrended = detrend_2d(&x.view(), 0, "constant", None).expect("Operation failed");
        for col in detrended.columns() {
            assert_relative_eq!(col.mean(), 0.0, epsilon = 1e-10);
        }
    }
    #[test]
    fn test_resample_upsample() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let resampled = resample(&x.view(), 8, 0, None).expect("Operation failed");
        assert_eq!(resampled.len(), 8);
        assert_relative_eq!(resampled[0], x[0], epsilon = 0.1);
        assert_relative_eq!(resampled[resampled.len() - 1], 2.5_f64, epsilon = 0.1);
        assert_relative_eq!(resampled[2], 2.0_f64, epsilon = 0.2);
    }
    #[test]
    fn test_resample_downsample() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let resampled = resample(&x.view(), 4, 0, None).expect("Operation failed");
        assert_eq!(resampled.len(), 4);
    }
    #[test]
    fn test_decimate() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let decimated = decimate(&x.view(), 2, Some(4), Some("iir"), 0).expect("Operation failed");
        assert_eq!(decimated.len(), 4);
    }
    #[test]
    fn test_invalid_detrend_type() {
        let x = array![1.0, 2.0, 3.0];
        let result = detrend(&x.view(), 0, "invalid", None);
        assert!(result.is_err());
    }
    #[test]
    fn test_invalid_axis() {
        let x = array![1.0, 2.0, 3.0];
        let result = detrend(&x.view(), 1, "constant", None);
        assert!(result.is_err());
    }
}
