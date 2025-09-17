//! Parallel statistical filtering operations
//!
//! This module provides parallel implementations of statistical filters
//! including median, rank-order, and bilateral filters for noise reduction
//! and edge-preserving smoothing.

use crate::error::{SignalError, SignalResult};
use scirs2_core::parallel_ops::*;

/// Helper for parallel operations (temporary replacement)
fn par_iter_with_setup<I, IT, S, F, R, RF, E>(
    items: I,
    _setup: S,
    map_fn: F,
    reduce_fn: RF,
) -> Result<Vec<R>, E>
where
    I: IntoIterator<Item = IT>,
    IT: Copy,
    S: Fn() -> (),
    F: Fn((), IT) -> Result<R, E>,
    RF: Fn(&mut Vec<R>, Result<R, E>) -> Result<(), E>,
    E: std::fmt::Debug,
{
    let mut results = Vec::new();
    for item in items {
        let result = map_fn((), item);
        reduce_fn(&mut results, result)?;
    }
    Ok(results)
}

/// Parallel median filtering for noise reduction
///
/// Applies a median filter in parallel chunks for efficient noise reduction
/// while preserving edges. Particularly effective for impulse noise.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `kernel_size` - Size of median filter kernel (must be odd)
/// * `chunk_size` - Size of chunks for parallel processing
///
/// # Returns
///
/// * Median filtered signal
pub fn parallel_median_filter(
    signal: &[f64],
    kernel_size: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if kernel_size % 2 == 0 {
        return Err(SignalError::ValueError(
            "Kernel size must be odd".to_string(),
        ));
    }

    let n = signal.len();
    let half_kernel = kernel_size / 2;
    let chunk = chunk_size.unwrap_or(1024.min((n / num_cpus::get()).max(n / 4).max(1)));
    let overlap = half_kernel;

    // Process signal in overlapping chunks (safe arithmetic to prevent overflow)
    let effective_chunk = chunk.saturating_sub(overlap).max(1); // Ensure at least size 1
    let n_chunks = (n + effective_chunk - 1) / effective_chunk;
    let results = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, i| {
            let start = i * effective_chunk;
            let end = (start + chunk).min(n);
            let chunk_start = start.saturating_sub(overlap);
            let chunk_end = (end + overlap).min(n);

            // Extract chunk with padding
            let chunk_data = &signal[chunk_start..chunk_end];
            let mut chunk_result = Vec::with_capacity(end - start);

            // Apply median filter to chunk
            for j in 0..(end - start) {
                let global_idx = start + j;
                let local_idx = global_idx - chunk_start;

                // Extract neighborhood for median computation
                let neighborhood_start = local_idx.saturating_sub(half_kernel);
                let neighborhood_end = (local_idx + half_kernel + 1).min(chunk_data.len());

                let mut neighborhood: Vec<f64> =
                    chunk_data[neighborhood_start..neighborhood_end].to_vec();
                neighborhood.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let median = neighborhood[neighborhood.len() / 2];
                chunk_result.push(median);
            }

            Ok(chunk_result)
        },
        |results, result: SignalResult<Vec<f64>>| {
            results.push(result?);
            Ok(())
        },
    )?;

    // Concatenate results
    let mut output = Vec::with_capacity(n);
    for chunk_result in results {
        output.extend(chunk_result);
    }
    output.truncate(n);

    Ok(output)
}

/// Parallel rank-order filtering
///
/// Applies rank-order filtering in parallel, where the output is the
/// k-th order statistic within a sliding window.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the sliding window
/// * `rank` - Rank to extract (0 = minimum, window_size-1 = maximum)
/// * `chunk_size` - Size of chunks for parallel processing
///
/// # Returns
///
/// * Rank-order filtered signal
pub fn parallel_rank_order_filter(
    signal: &[f64],
    window_size: usize,
    rank: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if rank >= window_size {
        return Err(SignalError::ValueError(
            "Rank must be less than window size".to_string(),
        ));
    }

    let n = signal.len();
    let half_window = window_size / 2;
    let chunk = chunk_size.unwrap_or(1024.min((n / num_cpus::get()).max(n / 4).max(1)));
    let overlap = half_window;

    // Process signal in overlapping chunks (safe arithmetic to prevent overflow)
    let effective_chunk = chunk.saturating_sub(overlap).max(1); // Ensure at least size 1
    let n_chunks = (n + effective_chunk - 1) / effective_chunk;

    let results = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, i| {
            let start = i * effective_chunk;
            let end = (start + chunk).min(n);
            let chunk_start = start.saturating_sub(overlap);
            let chunk_end = (end + overlap).min(n);

            // Extract chunk with padding
            let chunk_data = &signal[chunk_start..chunk_end];
            let mut chunk_result = Vec::with_capacity(end - start);

            // Apply rank-order filter to chunk
            for j in 0..(end - start) {
                let global_idx = start + j;
                let local_idx = global_idx - chunk_start;

                // Extract window
                let window_start = local_idx.saturating_sub(half_window);
                let window_end = (local_idx + half_window + 1).min(chunk_data.len());

                let mut window: Vec<f64> = chunk_data[window_start..window_end].to_vec();
                window.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let rank_val = if rank < window.len() {
                    window[rank]
                } else {
                    window[window.len() - 1]
                };
                chunk_result.push(rank_val);
            }

            Ok(chunk_result)
        },
        |results, result: SignalResult<Vec<f64>>| {
            results.push(result?);
            Ok(())
        },
    )?;

    // Concatenate results
    let mut output = Vec::with_capacity(n);
    for chunk_result in results {
        output.extend(chunk_result);
    }
    output.truncate(n);

    Ok(output)
}

/// Parallel bilateral filtering for edge-preserving smoothing
///
/// Applies bilateral filtering in parallel, which preserves edges while
/// smoothing noise by considering both spatial and intensity differences.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the spatial window
/// * `sigma_spatial` - Standard deviation for spatial Gaussian kernel
/// * `sigma_intensity` - Standard deviation for intensity Gaussian kernel
/// * `chunk_size` - Size of chunks for parallel processing
///
/// # Returns
///
/// * Bilateral filtered signal
pub fn parallel_bilateral_filter(
    signal: &[f64],
    window_size: usize,
    sigma_spatial: f64,
    sigma_intensity: f64,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    let half_window = window_size / 2;
    let chunk = chunk_size.unwrap_or(512.min(n / num_cpus::get())); // Smaller chunks due to computational intensity
    let overlap = half_window;

    // Precompute spatial kernel
    let spatial_kernel: Vec<f64> = (0..window_size)
        .map(|i| {
            let dist = (i as f64 - half_window as f64).abs();
            (-0.5 * (dist / sigma_spatial).powi(2)).exp()
        })
        .collect();

    // Process signal in overlapping chunks (safe arithmetic to prevent overflow)
    let effective_chunk_size = if chunk > overlap { chunk - overlap } else { n }; // Use full signal if overlap too large
    let n_chunks = if effective_chunk_size >= n {
        1
    } else {
        (n + effective_chunk_size - 1) / effective_chunk_size
    };

    let results = par_iter_with_setup(
        0..n_chunks,
        || (),
        |_, i| {
            let start = i * effective_chunk_size;
            let end = (start + effective_chunk_size).min(n);
            let chunk_start = start.saturating_sub(overlap);
            let chunk_end = (end + overlap).min(n);

            // Extract chunk with padding
            let chunk_data = &signal[chunk_start..chunk_end];
            let mut chunk_result = Vec::with_capacity(end - start);

            // Apply bilateral filter to chunk
            for j in 0..(end - start) {
                let global_idx = start + j;
                let local_idx = global_idx - chunk_start;
                let center_val = chunk_data[local_idx];

                // Extract window
                let window_start = local_idx.saturating_sub(half_window);
                let window_end = (local_idx + half_window + 1).min(chunk_data.len());

                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for (k, &val) in chunk_data[window_start..window_end].iter().enumerate() {
                    // Safe integer arithmetic to prevent overflow
                    let k_pos = k + window_start;
                    let local_pos = local_idx.saturating_sub(half_window);
                    let spatial_idx = if k_pos >= local_pos {
                        k_pos - local_pos
                    } else {
                        continue; // Skip invalid indices
                    };
                    if spatial_idx < spatial_kernel.len() {
                        let spatial_weight = spatial_kernel[spatial_idx];
                        let intensity_diff = (val - center_val).abs();
                        let intensity_weight =
                            (-0.5 * (intensity_diff / sigma_intensity).powi(2)).exp();
                        let total_weight = spatial_weight * intensity_weight;

                        weighted_sum += val * total_weight;
                        weight_sum += total_weight;
                    }
                }

                let filtered_val = if weight_sum > 1e-10 {
                    weighted_sum / weight_sum
                } else {
                    center_val
                };

                chunk_result.push(filtered_val);
            }

            Ok(chunk_result)
        },
        |results, result: SignalResult<Vec<f64>>| {
            results.push(result?);
            Ok(())
        },
    )?;

    // Concatenate results
    let mut output = Vec::with_capacity(n);
    for chunk_result in results {
        output.extend(chunk_result);
    }
    output.truncate(n);

    Ok(output)
}

/// Parallel percentile filtering
///
/// Applies percentile filtering in parallel, extracting the specified
/// percentile value within a sliding window.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the sliding window
/// * `percentile` - Percentile to extract (0.0 to 100.0)
/// * `chunk_size` - Size of chunks for parallel processing
///
/// # Returns
///
/// * Percentile filtered signal
pub fn parallel_percentile_filter(
    signal: &[f64],
    window_size: usize,
    percentile: f64,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if percentile < 0.0 || percentile > 100.0 {
        return Err(SignalError::ValueError(
            "Percentile must be between 0.0 and 100.0".to_string(),
        ));
    }

    let n = signal.len();
    let half_window = window_size / 2;
    let chunk = chunk_size.unwrap_or(1024.min((n / num_cpus::get()).max(n / 4).max(1)));
    let overlap = half_window;

    // Process signal in overlapping chunks
    let effective_chunk = chunk.saturating_sub(overlap).max(1);
    let n_chunks = (n + effective_chunk - 1) / effective_chunk;

    let results = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, i| {
            let start = i * effective_chunk;
            let end = (start + chunk).min(n);
            let chunk_start = start.saturating_sub(overlap);
            let chunk_end = (end + overlap).min(n);

            // Extract chunk with padding
            let chunk_data = &signal[chunk_start..chunk_end];
            let mut chunk_result = Vec::with_capacity(end - start);

            // Apply percentile filter to chunk
            for j in 0..(end - start) {
                let global_idx = start + j;
                let local_idx = global_idx - chunk_start;

                // Extract window
                let window_start = local_idx.saturating_sub(half_window);
                let window_end = (local_idx + half_window + 1).min(chunk_data.len());

                let mut window: Vec<f64> = chunk_data[window_start..window_end].to_vec();
                window.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // Calculate percentile index
                let percentile_index =
                    ((percentile / 100.0) * (window.len() - 1) as f64).round() as usize;
                let percentile_val = window[percentile_index.min(window.len() - 1)];
                chunk_result.push(percentile_val);
            }

            Ok(chunk_result)
        },
        |results, result: SignalResult<Vec<f64>>| {
            results.push(result?);
            Ok(())
        },
    )?;

    // Concatenate results
    let mut output = Vec::with_capacity(n);
    for chunk_result in results {
        output.extend(chunk_result);
    }
    output.truncate(n);

    Ok(output)
}

/// Parallel trimmed mean filtering
///
/// Applies trimmed mean filtering in parallel, computing the mean
/// after removing the highest and lowest values from each window.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the sliding window
/// * `trim_fraction` - Fraction of values to trim from each end (0.0 to 0.5)
/// * `chunk_size` - Size of chunks for parallel processing
///
/// # Returns
///
/// * Trimmed mean filtered signal
pub fn parallel_trimmed_mean_filter(
    signal: &[f64],
    window_size: usize,
    trim_fraction: f64,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if trim_fraction < 0.0 || trim_fraction > 0.5 {
        return Err(SignalError::ValueError(
            "Trim fraction must be between 0.0 and 0.5".to_string(),
        ));
    }

    let n = signal.len();
    let half_window = window_size / 2;
    let chunk = chunk_size.unwrap_or(1024.min((n / num_cpus::get()).max(n / 4).max(1)));
    let overlap = half_window;

    // Process signal in overlapping chunks
    let effective_chunk = chunk.saturating_sub(overlap).max(1);
    let n_chunks = (n + effective_chunk - 1) / effective_chunk;

    let results = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, i| {
            let start = i * effective_chunk;
            let end = (start + chunk).min(n);
            let chunk_start = start.saturating_sub(overlap);
            let chunk_end = (end + overlap).min(n);

            // Extract chunk with padding
            let chunk_data = &signal[chunk_start..chunk_end];
            let mut chunk_result = Vec::with_capacity(end - start);

            // Apply trimmed mean filter to chunk
            for j in 0..(end - start) {
                let global_idx = start + j;
                let local_idx = global_idx - chunk_start;

                // Extract window
                let window_start = local_idx.saturating_sub(half_window);
                let window_end = (local_idx + half_window + 1).min(chunk_data.len());

                let mut window: Vec<f64> = chunk_data[window_start..window_end].to_vec();
                window.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // Calculate trim indices
                let trim_count = (trim_fraction * window.len() as f64).floor() as usize;
                let start_idx = trim_count;
                let end_idx = window.len().saturating_sub(trim_count);

                // Compute trimmed mean
                let trimmed_mean = if start_idx < end_idx {
                    let sum: f64 = window[start_idx..end_idx].iter().sum();
                    sum / (end_idx - start_idx) as f64
                } else {
                    // If too much trimming, use median
                    window[window.len() / 2]
                };

                chunk_result.push(trimmed_mean);
            }

            Ok(chunk_result)
        },
        |results, result: SignalResult<Vec<f64>>| {
            results.push(result?);
            Ok(())
        },
    )?;

    // Concatenate results
    let mut output = Vec::with_capacity(n);
    for chunk_result in results {
        output.extend(chunk_result);
    }
    output.truncate(n);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_parallel_median_filter() {
        let signal = vec![1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0]; // Contains impulse noise
        let result = parallel_median_filter(&signal, 3, None).unwrap();

        assert_eq!(result.len(), signal.len());
        // Median filter should reduce the impulse noise
        assert!(result[3] < 50.0); // Should be much less than the original 100.0
    }

    #[test]
    fn test_parallel_rank_order_filter() {
        let signal: Vec<f64> = (0..50)
            .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
            .collect();

        // Test minimum filter (rank = 0)
        let min_result = parallel_rank_order_filter(&signal, 5, 0, None).unwrap();
        assert_eq!(min_result.len(), signal.len());

        // Test maximum filter (rank = window_size - 1)
        let max_result = parallel_rank_order_filter(&signal, 5, 4, None).unwrap();
        assert_eq!(max_result.len(), signal.len());
    }

    #[test]
    fn test_parallel_bilateral_filter() {
        let signal: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 10.0).sin() + 0.1 * (i as f64 * 0.5).sin()) // Signal with noise
            .collect();

        let result = parallel_bilateral_filter(&signal, 5, 1.0, 0.1, None).unwrap();
        assert_eq!(result.len(), signal.len());
    }

    #[test]
    fn test_parallel_percentile_filter() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Test 50th percentile (median)
        let result = parallel_percentile_filter(&signal, 5, 50.0, None).unwrap();
        assert_eq!(result.len(), signal.len());

        // Test 90th percentile
        let result = parallel_percentile_filter(&signal, 5, 90.0, None).unwrap();
        assert_eq!(result.len(), signal.len());

        // Test error condition
        let result = parallel_percentile_filter(&signal, 5, 150.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_trimmed_mean_filter() {
        let signal = vec![1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 200.0, 8.0, 9.0, 10.0]; // With outliers

        let result = parallel_trimmed_mean_filter(&signal, 5, 0.2, None).unwrap();
        assert_eq!(result.len(), signal.len());

        // Test error condition
        let result = parallel_trimmed_mean_filter(&signal, 5, 0.7, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_statistical_filter_error_conditions() {
        let signal = vec![1.0, 2.0, 3.0];

        // Even kernel size for median filter
        let result = parallel_median_filter(&signal, 4, None);
        assert!(result.is_err());

        // Invalid rank for rank-order filter
        let result = parallel_rank_order_filter(&signal, 3, 3, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_cases() {
        // Empty signal
        let empty_signal: Vec<f64> = vec![];
        let result = parallel_median_filter(&empty_signal, 3, None);
        if let Ok(filtered) = result {
            assert_eq!(filtered.len(), 0);
        }

        // Single element signal
        let single_signal = vec![5.0];
        let result = parallel_median_filter(&single_signal, 1, None).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 5.0);
    }
}
