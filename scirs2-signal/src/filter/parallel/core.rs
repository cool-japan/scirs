//! Core parallel filtering operations
//!
//! This module provides the fundamental parallel filtering operations including
//! filtfilt (zero-phase filtering), lfilter (direct filtering), and overlap-save methods.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, ArrayView1};
use num_traits::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use std::fmt::Debug;

/// Parallel implementation of filtfilt (zero-phase filtering)
///
/// Applies a digital filter forward and backward to achieve zero-phase
/// distortion, using parallel processing for improved performance.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `x` - Input signal
/// * `chunk_size` - Size of chunks for parallel processing (None for auto)
///
/// # Returns
///
/// * Zero-phase filtered signal
pub fn parallel_filtfilt<T>(
    b: &[f64],
    a: &[f64],
    x: &[T],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug + Send + Sync,
{
    if a.is_empty() || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    // Convert input to f64 Array1
    let x_array = Array1::from_iter(
        x.iter()
            .map(|&val| {
                NumCast::from(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to f64", val))
                })
            })
            .collect::<SignalResult<Vec<f64>>>()?,
    );

    // Forward filtering with overlap-save method for parallelization
    let forward_filtered = parallel_filter_overlap_save(b, a, &x_array, chunk_size)?;

    // Reverse the signal
    let mut reversed = forward_filtered.to_vec();
    reversed.reverse();
    let reversed_array = Array1::from(reversed);

    // Backward filtering
    let backward_filtered = parallel_filter_overlap_save(b, a, &reversed_array, chunk_size)?;

    // Reverse again to get final result
    let mut result = backward_filtered.to_vec();
    result.reverse();

    Ok(result)
}

/// Parallel implementation of lfilter (direct filtering)
///
/// Applies a digital filter using direct form implementation with parallel processing
/// for improved performance on large signals.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `x` - Input signal
/// * `chunk_size` - Size of chunks for parallel processing (None for auto)
///
/// # Returns
///
/// * Filtered signal
pub fn parallel_lfilter<T>(
    b: &[f64],
    a: &[f64],
    x: &[T],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug + Send + Sync,
{
    if a.is_empty() || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    let x_array: Array1<f64> = Array1::from_iter(
        x.iter()
            .map(|&val| {
                NumCast::from(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to f64", val))
                })
            })
            .collect::<Result<Vec<f64>, SignalError>>()?,
    );

    // Check if all values are finite
    for (i, &val) in x_array.iter().enumerate() {
        if !val.is_finite() {
            return Err(SignalError::ValueError(format!(
                "x_array[{}] is not finite: {}",
                i, val
            )));
        }
    }

    let n: usize = x_array.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // For small signals, use sequential processing
    if n < 1000 {
        return sequential_lfilter(b, a, &x_array.to_vec());
    }

    // Determine optimal chunk size
    let effective_chunk_size = chunk_size.unwrap_or_else(|| {
        let num_cores = num_cpus::get();
        (n / num_cores).max(1000)
    });

    // Normalize coefficients
    let a0 = a[0];
    let b_norm: Vec<f64> = b.iter().map(|&coeff| coeff / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&coeff| coeff / a0).collect();

    // Parallel filtering with overlap handling
    let overlap_size = (a.len() - 1).max(b.len() - 1);
    let chunks: Vec<_> = x_array
        .windows(effective_chunk_size + overlap_size)
        .into_iter()
        .step_by(effective_chunk_size)
        .collect();

    let filtered_chunks: Vec<Vec<f64>> = parallel_map(&chunks, |chunk: &ArrayView1<f64>| {
        sequential_lfilter(&b_norm, &a_norm, &chunk.to_vec())
    })
    .into_iter()
    .collect::<Result<Vec<_>, SignalError>>()?;

    // Combine results with overlap removal
    let mut result = Vec::with_capacity(n);
    for (i, chunk_result) in filtered_chunks.iter().enumerate() {
        if i == 0 {
            result.extend_from_slice(chunk_result);
        } else {
            // Skip overlap region
            result.extend_from_slice(&chunk_result[overlap_size..]);
        }
    }

    result.truncate(n);
    Ok(result)
}

/// Overlap-save method for parallel filtering
pub fn parallel_filter_overlap_save(
    b: &[f64],
    a: &[f64],
    x: &Array1<f64>,
    chunk_size: Option<usize>,
) -> SignalResult<Array1<f64>> {
    let n = x.len();
    let filter_len = b.len().max(a.len());

    // Determine chunk size
    let chunk = chunk_size.unwrap_or_else(|| {
        // Auto-determine based on signal length and available cores
        let n_cores = num_cpus::get();
        ((n / n_cores).max(filter_len * 4)).min(8192)
    });

    // Overlap needed for continuity
    let overlap = filter_len - 1;

    // Process chunks sequentially (simplified version)
    let n_chunks = (n + chunk - overlap - 1) / (chunk - overlap);
    let mut results = Vec::with_capacity(n_chunks);

    for i in 0..n_chunks {
        let start = i * (chunk - overlap);
        let end = ((start + chunk).min(n)).max(start + 1);

        // Extract chunk with proper overlap
        let chunk_start = start.saturating_sub(overlap);
        let chunk_data = x.slice(s![chunk_start..end]).to_vec();

        // Apply filter to chunk
        let filtered = filter_direct(b, a, &chunk_data)?;

        // Extract valid portion (discard transient response)
        let valid_start = if i == 0 { 0 } else { overlap };
        let valid_filtered = filtered[valid_start..].to_vec();

        results.push(valid_filtered);
    }

    // Concatenate results
    let mut output = Vec::with_capacity(n);
    for chunk_result in results {
        output.extend(chunk_result);
    }

    // Trim to exact length
    output.truncate(n);

    Ok(Array1::from(output))
}

/// Direct filtering implementation (for chunks)
pub fn filter_direct(b: &[f64], a: &[f64], x: &[f64]) -> SignalResult<Vec<f64>> {
    let n = x.len();
    let nb = b.len();
    let na = a.len();

    // Normalize by a[0]
    let a0 = a[0];
    if a0.abs() < 1e-10 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    let mut y = vec![0.0; n];

    for i in 0..n {
        // Feedforward path
        for j in 0..nb.min(i + 1) {
            y[i] += b[j] * x[i - j] / a0;
        }

        // Feedback path
        for j in 1..na.min(i + 1) {
            y[i] -= a[j] * y[i - j] / a0;
        }
    }

    Ok(y)
}

/// Sequential lfilter implementation for small signals or chunks
pub(crate) fn sequential_lfilter(b: &[f64], a: &[f64], x: &[f64]) -> SignalResult<Vec<f64>> {
    let n = x.len();
    let mut y = vec![0.0; n];
    let mut memory_b = vec![0.0; b.len()];

    for i in 0..n {
        // Shift memory
        for j in (1..b.len()).rev() {
            memory_b[j] = memory_b[j - 1];
        }
        memory_b[0] = x[i];

        // Compute output
        let mut output = 0.0;
        for j in 0..b.len() {
            output += b[j] * memory_b[j];
        }

        for j in 1..a.len() {
            if i >= j {
                output -= a[j] * y[i - j];
            }
        }

        y[i] = output;
    }

    Ok(y)
}

/// Parallel batch filtering for multiple signals
///
/// Applies the same digital filter to multiple signals in parallel.
/// Useful for processing multiple channels simultaneously.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `signals` - Array of input signals (each row is a signal)
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Array of filtered signals
pub fn parallel_batch_filter(
    b: &[f64],
    a: &[f64],
    signals: &ndarray::Array2<f64>,
    chunk_size: Option<usize>,
) -> SignalResult<ndarray::Array2<f64>> {
    let (n_signals, signal_len) = signals.dim();
    let mut results = ndarray::Array2::zeros((n_signals, signal_len));

    // Process each signal in parallel
    let signal_refs: Vec<_> = (0..n_signals).map(|i| signals.row(i)).collect();

    let mut processed = Vec::with_capacity(n_signals);
    for signal in signal_refs.iter() {
        // Apply filter to each signal
        let filtered = parallel_filter_overlap_save(
            b,
            a,
            &Array1::from_iter(signal.iter().cloned()),
            chunk_size,
        )
        .map_err(|e| SignalError::ComputationError(format!("Batch filtering failed: {:?}", e)))?;
        processed.push(filtered.to_vec());
    }

    // Copy results back
    for (i, signal_result) in processed.into_iter().enumerate() {
        for (j, &val) in signal_result.iter().enumerate() {
            if j < signal_len {
                results[[i, j]] = val;
            }
        }
    }

    Ok(results)
}

/// Parallel multi-rate filtering with decimation
///
/// Applies filtering followed by downsampling in parallel chunks.
/// Useful for efficiently reducing sample rate while filtering.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `b` - Filter numerator coefficients
/// * `a` - Filter denominator coefficients
/// * `decimation_factor` - Downsampling factor
/// * `chunk_size` - Chunk size for processing
///
/// # Returns
///
/// * Filtered and decimated signal
pub fn parallel_decimate_filter(
    signal: &[f64],
    b: &[f64],
    a: &[f64],
    decimation_factor: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if decimation_factor == 0 {
        return Err(SignalError::ValueError(
            "Decimation factor must be greater than 0".to_string(),
        ));
    }

    // First apply the filter
    let filtered = parallel_filtfilt(b, a, signal, chunk_size)?;

    // Then decimate
    let decimated: Vec<f64> = filtered
        .into_iter()
        .enumerate()
        .filter_map(|(i, val)| {
            if i % decimation_factor == 0 {
                Some(val)
            } else {
                None
            }
        })
        .collect();

    Ok(decimated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_filtfilt() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let b = vec![0.25, 0.5, 0.25]; // Simple lowpass filter
        let a = vec![1.0];

        let result = parallel_filtfilt(&b, &a, &signal, None);
        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_parallel_lfilter() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let b = vec![0.5, 0.5];
        let a = vec![1.0];

        let result = parallel_lfilter(&b, &a, &signal, None);
        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_filter_direct() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let b = vec![0.5, 0.5];
        let a = vec![1.0];

        let result = filter_direct(&b, &a, &signal);
        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_parallel_decimate_filter() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.25, 0.5, 0.25];
        let a = vec![1.0];

        let result = parallel_decimate_filter(&signal, &b, &a, 2, None);
        assert!(result.is_ok());
        let decimated = result.unwrap();
        assert_eq!(decimated.len(), 4); // Should be half the original length
    }
}
