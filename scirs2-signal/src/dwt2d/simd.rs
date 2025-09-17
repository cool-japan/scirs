//! SIMD-optimized functions for 2D DWT operations
//!
//! This module provides SIMD-accelerated implementations for common wavelet coefficient
//! operations such as thresholding and energy calculation. These optimizations can
//! provide significant performance improvements on supported hardware.

use super::types::ThresholdMethod;

/// Apply thresholding to a slice of coefficients using SIMD instructions where available.
///
/// This function automatically selects the best available implementation based on
/// the target architecture and falls back to scalar operations when SIMD is not available.
///
/// # Arguments
///
/// * `coeffs` - Mutable slice of coefficients to threshold
/// * `threshold` - The threshold value to apply
/// * `method` - The thresholding method (Hard, Soft, or Garrote)
///
/// # Performance
///
/// On x86_64 systems with AVX2 support, this can be 4-8x faster than scalar implementation.
/// On other architectures, it falls back to an optimized scalar version.
pub fn simd_threshold_coefficients(coeffs: &mut [f64], threshold: f64, method: ThresholdMethod) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        simd_threshold_avx2(coeffs, threshold, method);
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        // Fallback to scalar implementation
        for coeff in coeffs.iter_mut() {
            *coeff = crate::dwt2d::thresholding::apply_threshold(*coeff, threshold, method);
        }
    }
}

/// AVX2-optimized thresholding implementation for x86_64 systems.
///
/// This function processes 4 coefficients at a time using 256-bit AVX2 instructions.
/// It handles the bulk of the data with SIMD and processes remaining elements with scalar code.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn simd_threshold_avx2(coeffs: &mut [f64], threshold: f64, method: ThresholdMethod) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        let len = coeffs.len();
        let simd_len = len / 4 * 4; // Process 4 elements at a time

        unsafe {
            let threshold_vec = _mm256_set1_pd(threshold);
            let neg_threshold_vec = _mm256_set1_pd(-threshold);
            let zero_vec = _mm256_setzero_pd();
            let ones_vec = _mm256_set1_pd(1.0);

            // Process 4 coefficients at a time
            for i in (0..simd_len).step_by(4) {
                let data_ptr = coeffs.as_mut_ptr().add(i);
                let data_vec = _mm256_loadu_pd(data_ptr);
                let abs_data_vec = _mm256_andnot_pd(_mm256_set1_pd(-0.0), data_vec);

                let result_vec = match method {
                    ThresholdMethod::Hard => {
                        // Hard thresholding: zero if |x| <= threshold, keep x otherwise
                        let mask = _mm256_cmp_pd(abs_data_vec, threshold_vec, _CMP_GT_OQ);
                        _mm256_and_pd(data_vec, mask)
                    }
                    ThresholdMethod::Soft => {
                        // Soft thresholding: sign(x) * max(|x| - threshold, 0)
                        let sign_vec = _mm256_and_pd(data_vec, _mm256_set1_pd(-0.0));
                        let shrunk_vec = _mm256_sub_pd(abs_data_vec, threshold_vec);
                        let positive_shrunk = _mm256_max_pd(shrunk_vec, zero_vec);
                        _mm256_or_pd(sign_vec, positive_shrunk)
                    }
                    ThresholdMethod::Garrote => {
                        // Garrote thresholding: x * (1 - threshold^2 / x^2) if |x| > threshold
                        let mask = _mm256_cmp_pd(abs_data_vec, threshold_vec, _CMP_GT_OQ);
                        let threshold_sq = _mm256_mul_pd(threshold_vec, threshold_vec);
                        let data_sq = _mm256_mul_pd(data_vec, data_vec);
                        let ratio = _mm256_div_pd(threshold_sq, data_sq);
                        let factor = _mm256_sub_pd(ones_vec, ratio);
                        let result = _mm256_mul_pd(data_vec, factor);
                        _mm256_and_pd(result, mask)
                    }
                };

                _mm256_storeu_pd(data_ptr, result_vec);
            }
        }

        // Process remaining elements with scalar code
        for coeff in &mut coeffs[simd_len..] {
            *coeff = crate::dwt2d::thresholding::apply_threshold(*coeff, threshold, method);
        }
    }
}

/// Fallback implementation for non-AVX2 x86_64 systems.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
fn simd_threshold_avx2(_coeffs: &mut [f64], _threshold: f64, _method: ThresholdMethod) {
    // This should never be called due to the feature gates, but we need it for compilation
    unreachable!("AVX2 function called on non-AVX2 system");
}

/// Calculate the energy (sum of squares) of a slice of data using SIMD instructions.
///
/// This function computes the sum of squared values efficiently using vector instructions
/// when available, falling back to scalar computation otherwise.
///
/// # Arguments
///
/// * `data` - Slice of floating-point values
///
/// # Returns
///
/// * The sum of squares of all values in the slice
///
/// # Performance
///
/// On AVX2-capable systems, this can be significantly faster than scalar implementation,
/// especially for large arrays.
pub fn simd_calculate_energy(data: &[f64]) -> f64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        simd_energy_avx2(data)
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        // Fallback to scalar implementation
        data.iter().map(|&x| x * x).sum()
    }
}

/// AVX2-optimized energy calculation for x86_64 systems.
///
/// This function processes 4 values at a time, accumulating their squares
/// using 256-bit vector operations.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn simd_energy_avx2(data: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        let len = data.len();
        let simd_len = len / 4 * 4;
        let mut energy = 0.0;

        unsafe {
            let mut acc_vec = _mm256_setzero_pd();

            // Process 4 elements at a time
            for i in (0..simd_len).step_by(4) {
                let data_ptr = data.as_ptr().add(i);
                let data_vec = _mm256_loadu_pd(data_ptr);
                let squared_vec = _mm256_mul_pd(data_vec, data_vec);
                acc_vec = _mm256_add_pd(acc_vec, squared_vec);
            }

            // Horizontal sum of the accumulator vector
            let low_high = _mm256_extractf128_pd(acc_vec, 1);
            let low = _mm256_castpd256_pd128(acc_vec);
            let sum_vec = _mm_add_pd(low, low_high);
            let sum_arr: [f64; 2] = std::mem::transmute(sum_vec);
            energy += sum_arr[0] + sum_arr[1];
        }

        // Process remaining elements with scalar code
        for &value in &data[simd_len..] {
            energy += value * value;
        }

        energy
    }
}

/// Fallback implementation for non-AVX2 x86_64 systems.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
fn simd_energy_avx2(_data: &[f64]) -> f64 {
    // This should never be called due to the feature gates, but we need it for compilation
    unreachable!("AVX2 function called on non-AVX2 system");
}

/// Platform capability detection for SIMD optimization.
///
/// This structure contains information about the available SIMD capabilities
/// on the current platform, allowing for runtime selection of optimal algorithms.
#[derive(Debug, Clone, Copy)]
pub struct PlatformCapabilities {
    /// Whether any SIMD instructions are available
    pub simd_available: bool,
    /// Whether AVX2 instructions are available (x86_64)
    pub avx2_available: bool,
    /// Whether AVX-512 instructions are available (x86_64)
    pub avx512_available: bool,
}

impl PlatformCapabilities {
    /// Detect the available SIMD capabilities on the current platform.
    ///
    /// This function performs runtime detection of CPU features and returns
    /// a structure describing the available optimizations.
    ///
    /// # Returns
    ///
    /// * A `PlatformCapabilities` struct with detected features
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_signal::dwt2d::simd::PlatformCapabilities;
    ///
    /// let caps = PlatformCapabilities::detect();
    /// if caps.avx2_available {
    ///     println!("AVX2 optimizations available");
    /// }
    /// ```
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                simd_available: is_x86_feature_detected!("sse2"),
                avx2_available: is_x86_feature_detected!("avx2"),
                avx512_available: is_x86_feature_detected!("avx512f"),
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                simd_available: false,
                avx2_available: false,
                avx512_available: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_threshold_hard() {
        let mut coeffs = vec![1.0, -2.0, 0.5, -0.3, 3.0, -4.0];
        let original = coeffs.clone();

        simd_threshold_coefficients(&mut coeffs, 1.0, ThresholdMethod::Hard);

        // Values with |x| <= threshold should be zero
        assert_eq!(coeffs[2], 0.0); // 0.5 -> 0.0
        assert_eq!(coeffs[3], 0.0); // -0.3 -> 0.0

        // Values with |x| > threshold should be unchanged
        assert_eq!(coeffs[0], original[0]); // 1.0 -> 1.0 (boundary case)
        assert_eq!(coeffs[1], original[1]); // -2.0 -> -2.0
        assert_eq!(coeffs[4], original[4]); // 3.0 -> 3.0
        assert_eq!(coeffs[5], original[5]); // -4.0 -> -4.0
    }

    #[test]
    fn test_simd_threshold_soft() {
        let mut coeffs = vec![2.0, -3.0, 0.5, -0.8];

        simd_threshold_coefficients(&mut coeffs, 1.0, ThresholdMethod::Soft);

        // Soft thresholding: sign(x) * max(|x| - threshold, 0)
        assert_eq!(coeffs[0], 1.0);  // 2.0 -> 1.0
        assert_eq!(coeffs[1], -2.0); // -3.0 -> -2.0
        assert_eq!(coeffs[2], 0.0);  // 0.5 -> 0.0
        assert_eq!(coeffs[3], 0.0);  // -0.8 -> 0.0
    }

    #[test]
    fn test_simd_calculate_energy() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let energy = simd_calculate_energy(&data);

        // 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 1 + 4 + 9 + 16 + 25 = 55
        assert!((energy - 55.0).abs() < 1e-10);
    }

    #[test]
    fn test_platform_capabilities() {
        let caps = PlatformCapabilities::detect();

        // Just ensure the detection runs without panicking
        // The actual capabilities depend on the test machine
        println!("SIMD available: {}", caps.simd_available);
        println!("AVX2 available: {}", caps.avx2_available);
        println!("AVX512 available: {}", caps.avx512_available);
    }
}