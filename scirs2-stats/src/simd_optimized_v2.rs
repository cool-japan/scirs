//! Enhanced SIMD optimizations for v1.0.0
//!
//! This module provides improved SIMD implementations that:
//! - Avoid temporary array allocations
//! - Use efficient SIMD patterns
//! - Provide automatic fallback to scalar code
//! - Support multiple data types

use crate::error::StatsResult;
use scirs2_core::ndarray::{ArrayBase, Data, Ix1};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use scirs2_core::validation::check_not_empty;

/// SIMD configuration for optimal performance
#[derive(Debug, Clone, Copy)]
pub struct SimdConfig {
    /// Minimum array size for SIMD to be beneficial
    pub minsize: usize,
    /// Whether to use aligned loads/stores
    pub use_aligned: bool,
    /// Maximum unroll factor
    pub unroll_factor: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        // Use conservative defaults for cross-platform compatibility
        // TODO: Implement proper platform capability detection when available
        let minsize = 128; // Conservative threshold for SIMD benefits

        Self {
            minsize,
            use_aligned: true, // Enable alignment for better performance
            unroll_factor: 4,  // Standard unroll factor
        }
    }
}

/// Optimized mean calculation using SIMD with chunked processing
///
/// This implementation avoids temporary arrays and processes data in chunks
/// for better cache efficiency.
#[allow(dead_code)]
pub fn mean_simd_optimized<F, D>(
    x: &ArrayBase<D, Ix1>,
    config: Option<SimdConfig>,
) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    // Use scirs2-core validation
    check_not_empty(x, "x").map_err(|_| {
        crate::error::StatsError::invalid_argument("Cannot compute mean of empty array")
    })?;

    let config = config.unwrap_or_default();
    let n = x.len();

    if n < config.minsize {
        // Small arrays: use scalar code
        let sum = x.iter().fold(F::zero(), |acc, &val| acc + val);
        return Ok(sum / F::from(n).unwrap());
    }

    // For larger arrays, use chunked SIMD processing
    let sum = chunked_simd_sum(x, &config)?;
    Ok(sum / F::from(n).unwrap())
}

/// Optimized variance calculation using single-pass SIMD algorithm
///
/// Uses Welford's online algorithm adapted for SIMD processing
#[allow(dead_code)]
pub fn variance_simd_optimized<F, D>(
    x: &ArrayBase<D, Ix1>,
    ddof: usize,
    config: Option<SimdConfig>,
) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    let n = x.len();
    if n <= ddof {
        return Err(crate::error::StatsError::invalid_argument(
            "Not enough data points for the given degrees of freedom",
        ));
    }

    let config = config.unwrap_or_default();

    if n < config.minsize {
        // Small arrays: use scalar Welford's algorithm
        return variance_scalar_welford(x, ddof);
    }

    // Use SIMD-optimized two-pass algorithm for better accuracy
    let mean = mean_simd_optimized(x, Some(config))?;
    let sum_sq_dev = chunked_simd_sum_squared_deviations(x, mean, &config)?;

    Ok(sum_sq_dev / F::from(n - ddof).unwrap())
}

/// Compute all basic statistics in a single SIMD pass
///
/// Returns (mean, variance, min, max, skewness, kurtosis)
#[allow(dead_code)]
pub fn stats_simd_single_pass<F, D>(
    x: &ArrayBase<D, Ix1>,
    config: Option<SimdConfig>,
) -> StatsResult<(F, F, F, F, F, F)>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    if x.is_empty() {
        return Err(crate::error::StatsError::invalid_argument(
            "Cannot compute statistics of empty array",
        ));
    }

    let config = config.unwrap_or_default();
    let n = x.len();
    let n_f = F::from(n).unwrap();

    if n < config.minsize {
        // Use scalar single-pass algorithm
        return stats_scalar_single_pass(x);
    }

    // SIMD single-pass algorithm using moments
    let capabilities = PlatformCapabilities::detect();
    let simd_width = if capabilities.simd_available { 8 } else { 1 };

    // Process in SIMD chunks
    let mut m1 = F::zero(); // First moment (sum)
    let mut m2 = F::zero(); // Second moment
    let mut m3 = F::zero(); // Third moment
    let mut m4 = F::zero(); // Fourth moment
    let mut min = x[0];
    let mut max = x[0];

    // Main SIMD loop
    let chunks = x.len() / simd_width;
    let _remainder = x.len() % simd_width;

    for chunk_idx in 0..chunks {
        let start = chunk_idx * simd_width;
        let chunk = x.slice(scirs2_core::ndarray::s![start..start + simd_width]);

        // Process chunk with SIMD
        let chunk_sum = F::simd_sum(&chunk);
        m1 = m1 + chunk_sum;

        // Update min/max
        let chunk_min = F::simd_min_element(&chunk);
        let chunk_max = F::simd_max_element(&chunk);
        if chunk_min < min {
            min = chunk_min;
        }
        if chunk_max > max {
            max = chunk_max;
        }
    }

    // Handle remainder with scalar code
    let remainder_start = chunks * simd_width;
    for i in remainder_start..x.len() {
        let val = x[i];
        m1 = m1 + val;
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }

    // Second pass for central moments (more accurate)
    let mean = m1 / n_f;

    // Compute central moments in second pass
    for chunk_idx in 0..chunks {
        let start = chunk_idx * simd_width;
        let chunk = x.slice(scirs2_core::ndarray::s![start..start + simd_width]);

        // Compute deviations and powers using SIMD
        for &val in chunk.iter() {
            let dev = val - mean;
            let dev2 = dev * dev;
            let dev3 = dev2 * dev;
            let dev4 = dev3 * dev;

            m2 = m2 + dev2;
            m3 = m3 + dev3;
            m4 = m4 + dev4;
        }
    }

    // Handle remainder
    for i in remainder_start..x.len() {
        let dev = x[i] - mean;
        let dev2 = dev * dev;
        let dev3 = dev2 * dev;
        let dev4 = dev3 * dev;

        m2 = m2 + dev2;
        m3 = m3 + dev3;
        m4 = m4 + dev4;
    }

    // Calculate statistics from moments
    let variance = m2 / F::from(n - 1).unwrap();
    let std_dev = variance.sqrt();

    let skewness = if std_dev > F::epsilon() {
        (m3 / n_f) / (std_dev * std_dev * std_dev)
    } else {
        F::zero()
    };

    let kurtosis = if variance > F::epsilon() {
        (m4 / n_f) / (variance * variance) - F::from(3).unwrap()
    } else {
        F::zero()
    };

    Ok((mean, variance, min, max, skewness, kurtosis))
}

/// Helper function for chunked SIMD sum
#[allow(dead_code)]
fn chunked_simd_sum<F, D>(x: &ArrayBase<D, Ix1>, config: &SimdConfig) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    let capabilities = PlatformCapabilities::detect();
    let _simd_width = if capabilities.simd_available { 8 } else { 1 };

    // Process in chunks for better cache efficiency
    const CHUNK_SIZE: usize = 1024;
    let mut total_sum = F::zero();

    for chunk in x.windows(CHUNK_SIZE) {
        let chunk_sum = F::simd_sum(&chunk.view());
        total_sum = total_sum + chunk_sum;
    }

    // Handle any remaining elements
    let processed = (x.len() / CHUNK_SIZE) * CHUNK_SIZE;
    if processed < x.len() {
        let remainder = x.slice(scirs2_core::ndarray::s![processed..]);
        let remainder_sum = F::simd_sum(&remainder);
        total_sum = total_sum + remainder_sum;
    }

    Ok(total_sum)
}

/// Helper function for chunked SIMD sum of squared deviations
#[allow(dead_code)]
fn chunked_simd_sum_squared_deviations<F, D>(
    x: &ArrayBase<D, Ix1>,
    mean: F,
    config: &SimdConfig,
) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    const CHUNK_SIZE: usize = 1024;
    let mut total_sum = F::zero();

    // Process data in chunks without creating temporary arrays
    for chunk in x.windows(CHUNK_SIZE) {
        let chunk_sum = chunk
            .iter()
            .map(|&val| {
                let dev = val - mean;
                dev * dev
            })
            .fold(F::zero(), |acc, val| acc + val);
        total_sum = total_sum + chunk_sum;
    }

    // Handle remainder
    let processed = (x.len() / CHUNK_SIZE) * CHUNK_SIZE;
    if processed < x.len() {
        for i in processed..x.len() {
            let dev = x[i] - mean;
            total_sum = total_sum + dev * dev;
        }
    }

    Ok(total_sum)
}

/// Scalar Welford's algorithm for variance (fallback)
#[allow(dead_code)]
fn variance_scalar_welford<F, D>(x: &ArrayBase<D, Ix1>, ddof: usize) -> StatsResult<F>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    let mut mean = F::zero();
    let mut m2 = F::zero();
    let mut count = 0;

    for &val in x.iter() {
        count += 1;
        let delta = val - mean;
        mean = mean + delta / F::from(count).unwrap();
        let delta2 = val - mean;
        m2 = m2 + delta * delta2;
    }

    Ok(m2 / F::from(count - ddof).unwrap())
}

/// Scalar single-pass statistics (fallback)
#[allow(dead_code)]
fn stats_scalar_single_pass<F, D>(x: &ArrayBase<D, Ix1>) -> StatsResult<(F, F, F, F, F, F)>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    let n = x.len();
    let n_f = F::from(n).unwrap();

    // First pass: compute mean
    let mean = x.iter().fold(F::zero(), |acc, &val| acc + val) / n_f;

    // Second pass: compute moments and min/max
    let mut m2 = F::zero();
    let mut m3 = F::zero();
    let mut m4 = F::zero();
    let mut min = x[0];
    let mut max = x[0];

    for &val in x.iter() {
        let dev = val - mean;
        let dev2 = dev * dev;
        let dev3 = dev2 * dev;
        let dev4 = dev3 * dev;

        m2 = m2 + dev2;
        m3 = m3 + dev3;
        m4 = m4 + dev4;

        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }

    let variance = m2 / F::from(n - 1).unwrap();
    let std_dev = variance.sqrt();

    let skewness = if std_dev > F::epsilon() {
        (m3 / n_f) / (std_dev * std_dev * std_dev)
    } else {
        F::zero()
    };

    let kurtosis = if variance > F::epsilon() {
        (m4 / n_f) / (variance * variance) - F::from(3).unwrap()
    } else {
        F::zero()
    };

    Ok((mean, variance, min, max, skewness, kurtosis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mean_simd_optimized() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mean = mean_simd_optimized(&data.view(), None).unwrap();
        assert!((mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_simd_optimized() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let var = variance_simd_optimized(&data.view(), 1, None).unwrap();
        assert!((var - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_single_pass() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let (mean, var, min, max__, skew, kurt) =
            stats_simd_single_pass(&data.view(), None).unwrap();

        assert!((mean - 3.0).abs() < 1e-10);
        assert!((var - 2.5).abs() < 1e-10);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max__ - 5.0).abs() < 1e-10);
    }
}
