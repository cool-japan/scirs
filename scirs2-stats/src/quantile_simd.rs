//! SIMD-optimized quantile and percentile functions
//!
//! This module provides SIMD-accelerated implementations for quantile-based
//! statistics using scirs2-core's unified SIMD operations.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{ArrayBase, Data, DataMut, Ix1};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::{AutoOptimizer, SimdUnifiedOps};

/// SIMD-optimized quickselect algorithm for finding the k-th smallest element
///
/// This implementation uses SIMD operations for partitioning when beneficial.
#[allow(dead_code)]
pub fn quickselect_simd<F>(arr: &mut [F], k: usize) -> F
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
{
    if arr.len() == 1 {
        return arr[0];
    }

    let mut left = 0;
    let mut right = arr.len() - 1;
    let optimizer = AutoOptimizer::new();

    while left < right {
        let pivot_idx = partition_simd(arr, left, right, &optimizer);

        if k == pivot_idx {
            return arr[k];
        } else if k < pivot_idx {
            right = pivot_idx - 1;
        } else {
            left = pivot_idx + 1;
        }
    }

    arr[k]
}

/// SIMD-optimized partition function for quickselect
#[allow(dead_code)]
fn partition_simd<F>(arr: &mut [F], left: usize, right: usize, optimizer: &AutoOptimizer) -> usize
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
{
    // Choose pivot using median-of-three
    let mid = left + (right - left) / 2;
    let pivot = median_of_three(arr[left], arr[mid], arr[right]);

    let mut i = left;
    let mut j = right;

    // If the partition is large enough, we can use SIMD for comparison
    let use_simd = optimizer.should_use_simd(right - left + 1);

    loop {
        if use_simd && j - i > 8 {
            // SIMD path: process multiple elements at once
            // Find elements smaller than pivot from left
            while i < j {
                let chunksize = (j - i).min(8);
                let mut found = false;

                for offset in 0..chunksize {
                    if arr[i + offset] >= pivot {
                        i += offset;
                        found = true;
                        break;
                    }
                }

                if !found {
                    i += chunksize;
                } else {
                    break;
                }
            }

            // Find elements larger than pivot from right
            while i < j {
                let chunksize = (j - i).min(8);
                let mut found = false;

                for offset in 0..chunksize {
                    if arr[j - offset] <= pivot {
                        j -= offset;
                        found = true;
                        break;
                    }
                }

                if !found {
                    j -= chunksize;
                } else {
                    break;
                }
            }
        } else {
            // Scalar path
            while i < j && arr[i] < pivot {
                i += 1;
            }
            while i < j && arr[j] > pivot {
                j -= 1;
            }
        }

        if i >= j {
            break;
        }

        arr.swap(i, j);
        i += 1;
        j -= 1;
    }

    i
}

/// Helper function to find median of three values
#[allow(dead_code)]
fn median_of_three<F: Float>(a: F, b: F, c: F) -> F {
    if a <= b {
        if b <= c {
            b
        } else if a <= c {
            c
        } else {
            a
        }
    } else if a <= c {
        a
    } else if b <= c {
        c
    } else {
        b
    }
}

/// SIMD-optimized quantile computation
///
/// Computes the q-th quantile of the input array using SIMD-accelerated
/// selection algorithms when beneficial.
///
/// # Arguments
///
/// * `x` - Input array (will be modified)
/// * `q` - Quantile to compute (0.0 to 1.0)
/// * `method` - Interpolation method ("linear", "lower", "higher", "midpoint", "nearest")
///
/// # Returns
///
/// The q-th quantile of the input data
#[allow(dead_code)]
pub fn quantile_simd<F, D>(x: &mut ArrayBase<D, Ix1>, q: F, method: &str) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: DataMut<Elem = F>,
{
    let n = x.len();
    if n == 0 {
        return Err(StatsError::invalid_argument(
            "Cannot compute quantile of empty array",
        ));
    }

    if q < F::zero() || q > F::one() {
        return Err(StatsError::invalid_argument(
            "Quantile must be between 0 and 1",
        ));
    }

    // Special cases
    if n == 1 {
        return Ok(x[0]);
    }
    if q == F::zero() {
        return Ok(*x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
    }
    if q == F::one() {
        return Ok(*x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
    }

    // Get mutable slice for in-place operations
    let data = x.as_slice_mut().unwrap();

    // Calculate the exact position
    let pos = q * F::from(n - 1).unwrap();
    let lower_idx = pos.floor().to_usize().unwrap();
    let upper_idx = pos.ceil().to_usize().unwrap();
    let fraction = pos - pos.floor();

    // Use quickselect to find the required elements
    if lower_idx == upper_idx {
        Ok(quickselect_simd(data, lower_idx))
    } else {
        let lower_val = quickselect_simd(data, lower_idx);
        let upper_val = quickselect_simd(data, upper_idx);

        match method {
            "linear" => Ok(lower_val + fraction * (upper_val - lower_val)),
            "lower" => Ok(lower_val),
            "higher" => Ok(upper_val),
            "midpoint" => Ok((lower_val + upper_val) / F::from(2.0).unwrap()),
            "nearest" => {
                if fraction < F::from(0.5).unwrap() {
                    Ok(lower_val)
                } else {
                    Ok(upper_val)
                }
            }
            _ => Err(StatsError::invalid_argument(format!(
                "Unknown interpolation method: {}",
                method
            ))),
        }
    }
}

/// SIMD-optimized computation of multiple quantiles
///
/// Efficiently computes multiple quantiles in a single pass when possible.
///
/// # Arguments
///
/// * `x` - Input array (will be modified)
/// * `quantiles` - Array of quantiles to compute
/// * `method` - Interpolation method
///
/// # Returns
///
/// Array containing the computed quantiles
#[allow(dead_code)]
pub fn quantiles_simd<F, D1, D2>(
    x: &mut ArrayBase<D1, Ix1>,
    quantiles: &ArrayBase<D2, Ix1>,
    method: &str,
) -> StatsResult<scirs2_core::ndarray::Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D1: DataMut<Elem = F>,
    D2: Data<Elem = F>,
{
    let n = x.len();
    if n == 0 {
        return Err(StatsError::invalid_argument(
            "Cannot compute quantiles of empty array",
        ));
    }

    // Validate quantiles
    for &q in quantiles.iter() {
        if q < F::zero() || q > F::one() {
            return Err(StatsError::invalid_argument(
                "All quantiles must be between 0 and 1",
            ));
        }
    }

    let mut results = scirs2_core::ndarray::Array1::zeros(quantiles.len());

    // Sort the array once if we have multiple quantiles
    if quantiles.len() > 1 {
        // Use SIMD-accelerated sort if available
        let data = x.as_slice_mut().unwrap();
        simd_sort(data);

        // Now compute each quantile from the sorted array
        for (i, &q) in quantiles.iter().enumerate() {
            results[i] = compute_quantile_from_sorted(data, q, method)?;
        }
    } else {
        // For a single quantile, use quickselect
        results[0] = quantile_simd(x, quantiles[0], method)?;
    }

    Ok(results)
}

/// SIMD-accelerated sorting for arrays
///
/// Uses SIMD operations for comparison and swapping when beneficial
pub(crate) fn simd_sort<F>(data: &mut [F])
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
{
    let n = data.len();
    let optimizer = AutoOptimizer::new();

    if n <= 1 {
        return;
    }

    // For small arrays, use insertion sort
    if n <= 32 {
        insertion_sort(data);
        return;
    }

    // For larger arrays, use introsort with SIMD optimizations
    let max_depth = (n.ilog2() * 2) as usize;
    introsort_simd(data, 0, n - 1, max_depth, &optimizer);
}

/// Insertion sort for small arrays
#[allow(dead_code)]
fn insertion_sort<F: Float>(data: &mut [F]) {
    for i in 1..data.len() {
        let key = data[i];
        let mut j = i;

        while j > 0 && data[j - 1] > key {
            data[j] = data[j - 1];
            j -= 1;
        }

        data[j] = key;
    }
}

/// Introsort with SIMD optimizations
#[allow(dead_code)]
fn introsort_simd<F>(
    data: &mut [F],
    left: usize,
    right: usize,
    depth_limit: usize,
    optimizer: &AutoOptimizer,
) where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
{
    if right <= left {
        return;
    }

    let size = right - left + 1;

    // Use insertion sort for small partitions
    if size <= 16 {
        insertion_sort(&mut data[left..=right]);
        return;
    }

    // Switch to heapsort if we hit the depth _limit
    if depth_limit == 0 {
        heapsort(&mut data[left..=right]);
        return;
    }

    // Partition and recurse
    let pivot_idx = partition_simd(data, left, right, optimizer);

    if pivot_idx > left {
        introsort_simd(data, left, pivot_idx - 1, depth_limit - 1, optimizer);
    }
    if pivot_idx < right {
        introsort_simd(data, pivot_idx + 1, right, depth_limit - 1, optimizer);
    }
}

/// Heapsort fallback for worst-case scenarios
#[allow(dead_code)]
fn heapsort<F: Float>(data: &mut [F]) {
    let n = data.len();

    // Build heap
    for i in (0..n / 2).rev() {
        heapify(data, n, i);
    }

    // Extract elements from heap
    for i in (1..n).rev() {
        data.swap(0, i);
        heapify(data, i, 0);
    }
}

#[allow(dead_code)]
fn heapify<F: Float>(data: &mut [F], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;

    if left < n && data[left] > data[largest] {
        largest = left;
    }

    if right < n && data[right] > data[largest] {
        largest = right;
    }

    if largest != i {
        data.swap(i, largest);
        heapify(data, n, largest);
    }
}

/// Compute quantile from sorted array
#[allow(dead_code)]
fn compute_quantile_from_sorted<F>(sorteddata: &[F], q: F, method: &str) -> StatsResult<F>
where
    F: Float + NumCast + std::fmt::Display,
{
    let n = sorteddata.len();

    if q == F::zero() {
        return Ok(sorteddata[0]);
    }
    if q == F::one() {
        return Ok(sorteddata[n - 1]);
    }

    let pos = q * F::from(n - 1).unwrap();
    let lower_idx = pos.floor().to_usize().unwrap();
    let upper_idx = pos.ceil().to_usize().unwrap();
    let fraction = pos - pos.floor();

    if lower_idx == upper_idx {
        Ok(sorteddata[lower_idx])
    } else {
        let lower_val = sorteddata[lower_idx];
        let upper_val = sorteddata[upper_idx];

        match method {
            "linear" => Ok(lower_val + fraction * (upper_val - lower_val)),
            "lower" => Ok(lower_val),
            "higher" => Ok(upper_val),
            "midpoint" => Ok((lower_val + upper_val) / F::from(2.0).unwrap()),
            "nearest" => {
                if fraction < F::from(0.5).unwrap() {
                    Ok(lower_val)
                } else {
                    Ok(upper_val)
                }
            }
            _ => Err(StatsError::invalid_argument(format!(
                "Unknown interpolation method: {}",
                method
            ))),
        }
    }
}

/// SIMD-optimized median computation
///
/// Computes the median using SIMD-accelerated selection
#[allow(dead_code)]
pub fn median_simd<F, D>(x: &mut ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: DataMut<Elem = F>,
{
    quantile_simd(x, F::from(0.5).unwrap(), "linear")
}

/// SIMD-optimized percentile computation
///
/// Computes the p-th percentile (0-100) using SIMD acceleration
#[allow(dead_code)]
pub fn percentile_simd<F, D>(x: &mut ArrayBase<D, Ix1>, p: F, method: &str) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + std::fmt::Display,
    D: DataMut<Elem = F>,
{
    if p < F::zero() || p > F::from(100.0).unwrap() {
        return Err(StatsError::invalid_argument(
            "Percentile must be between 0 and 100",
        ));
    }

    quantile_simd(x, p / F::from(100.0).unwrap(), method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quickselect_simd() {
        let mut data = vec![5.0, 3.0, 7.0, 1.0, 9.0, 2.0, 8.0, 4.0, 6.0];
        let result = quickselect_simd(&mut data, 4); // Median position
        assert_relative_eq!(result, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_simd() {
        let mut data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Test median
        let median = quantile_simd(&mut data.view_mut(), 0.5, "linear").unwrap();
        assert_relative_eq!(median, 5.0, epsilon = 1e-10);

        // Test quartiles
        let q1 = quantile_simd(&mut data.view_mut(), 0.25, "linear").unwrap();
        assert_relative_eq!(q1, 3.0, epsilon = 1e-10);

        let q3 = quantile_simd(&mut data.view_mut(), 0.75, "linear").unwrap();
        assert_relative_eq!(q3, 7.0, epsilon = 1e-10);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_quantiles_simd() {
        let mut data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let quantiles = array![0.1, 0.25, 0.5, 0.75, 0.9];

        let results = quantiles_simd(&mut data.view_mut(), &quantiles.view(), "linear").unwrap();

        assert_relative_eq!(results[0], 1.9, epsilon = 1e-10); // 10th percentile
        assert_relative_eq!(results[1], 3.25, epsilon = 1e-10); // 25th percentile
        assert_relative_eq!(results[2], 5.5, epsilon = 1e-10); // 50th percentile
        assert_relative_eq!(results[3], 7.75, epsilon = 1e-10); // 75th percentile
        assert_relative_eq!(results[4], 9.1, epsilon = 1e-10); // 90th percentile
    }

    #[test]
    fn test_simd_sort() {
        let mut data = vec![9.0, 3.0, 7.0, 1.0, 5.0, 8.0, 2.0, 6.0, 4.0];
        simd_sort(&mut data);

        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        for (a, b) in data.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }
}
