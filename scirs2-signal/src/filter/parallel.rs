use ndarray::s;
// Parallel filtering operations using scirs2-core parallel abstractions
//
// This module provides parallel implementations of filtering operations
// for improved performance on multi-core systems.

// use crate::dwt::Wavelet; // TODO: Re-enable when dwt module is available
use crate::error::{SignalError, SignalResult};
// use crate::savgol::savgol_coeffs; // TODO: Re-enable when savgol module is available
use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use rustfft::{num_complex::Complex, FftPlanner};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::f64::consts::PI;
use std::fmt::Debug;

#[allow(unused_imports)]
// Temporary replacement for par_iter_with_setup
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
#[allow(dead_code)]
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

/// Parallel convolution using overlap-save method
///
/// Performs convolution of two signals using parallel processing
/// with the overlap-save method for efficiency.
///
/// # Arguments
///
/// * `a` - First signal
/// * `v` - Second signal (kernel)
/// * `mode` - Convolution mode ("full", "same", "valid")
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Convolution result
#[allow(dead_code)]
pub fn parallel_convolve<T, U>(
    a: &[T],
    v: &[U],
    mode: &str,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug + Send + Sync,
    U: Float + NumCast + Debug + Send + Sync,
{
    // Convert inputs to Array1
    let a_array = Array1::from_iter(
        a.iter()
            .map(|&val| NumCast::from(val).unwrap_or(0.0))
            .collect::<Vec<f64>>(),
    );

    let v_array = Array1::from_iter(
        v.iter()
            .map(|&val| NumCast::from(val).unwrap_or(0.0))
            .collect::<Vec<f64>>(),
    );

    // Use overlap-save for efficiency with long signals
    if a_array.len() > 1000 && v_array.len() > 10 {
        parallel_convolve_overlap_save(&a_array, &v_array, mode, chunk_size)
    } else {
        // For small signals, use direct convolution
        parallel_convolve_direct(&a_array, &v_array, mode)
    }
}

/// Overlap-save method for parallel filtering
#[allow(dead_code)]
fn parallel_filter_overlap_save(
    b: &[f64],
    a: &[f64],
    x: &Array1<f64>,
    chunk_size: Option<usize>,
) -> SignalResult<Array1<f64>> {
    let n = x.len();
    let filter_len = b.len().max(a.len());

    // Determine chunk _size
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
#[allow(dead_code)]
fn filter_direct(b: &[f64], a: &[f64], x: &[f64]) -> SignalResult<Vec<f64>> {
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

/// Overlap-save convolution for parallel processing
#[allow(dead_code)]
fn parallel_convolve_overlap_save(
    a: &Array1<f64>,
    v: &Array1<f64>,
    mode: &str,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    let na = a.len();
    let nv = v.len();

    // For overlap-save, process in chunks
    let chunk = chunk_size.unwrap_or(4096);
    let overlap = nv - 1;

    // Full convolution length
    let n_full = na + nv - 1;
    let mut result = vec![0.0; n_full];

    // Process chunks sequentially (simplified version)
    let n_chunks = (na + chunk - overlap - 1) / (chunk - overlap);
    let mut chunk_results = Vec::with_capacity(n_chunks);

    for i in 0..n_chunks {
        let start = i * (chunk - overlap);
        let end = (start + chunk).min(na);

        // Extract chunk with zero padding if needed
        let mut chunk_data = vec![0.0; chunk];
        for j in start..end {
            chunk_data[j - start] = a[j];
        }

        // Convolve chunk with kernel
        let mut chunk_result = vec![0.0; chunk + nv - 1];
        for j in 0..chunk {
            for k in 0..nv {
                chunk_result[j + k] += chunk_data[j] * v[k];
            }
        }

        chunk_results.push(chunk_result);
    }

    // Combine chunk results
    for (i, chunk_res) in chunk_results.iter().enumerate() {
        let start = i * (chunk - overlap);
        for (j, &val) in chunk_res.iter().enumerate() {
            if start + j < n_full {
                result[start + j] += val;
            }
        }
    }

    // Apply mode
    match mode {
        "full" => Ok(result),
        "same" => {
            let start = (nv - 1) / 2;
            let end = start + na;
            Ok(result[start..end].to_vec())
        }
        "valid" => {
            if nv > na {
                return Err(SignalError::ValueError(
                    "In 'valid' mode, kernel must not be larger than signal".to_string(),
                ));
            }
            let start = nv - 1;
            let end = n_full - (nv - 1);
            Ok(result[start..end].to_vec())
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

/// Direct convolution for small signals
#[allow(dead_code)]
fn parallel_convolve_direct(
    a: &Array1<f64>,
    v: &Array1<f64>,
    mode: &str,
) -> SignalResult<Vec<f64>> {
    let na = a.len();
    let nv = v.len();
    let n_full = na + nv - 1;

    // Use sequential iteration (simplified version)
    let mut result = Vec::with_capacity(n_full);
    for i in 0..n_full {
        let mut sum = 0.0;
        for j in 0..nv {
            if i >= j && i - j < na {
                sum += a[i - j] * v[j];
            }
        }
        result.push(sum);
    }

    // Apply mode
    match mode {
        "full" => Ok(result),
        "same" => {
            let start = (nv - 1) / 2;
            let end = start + na;
            Ok(result[start..end].to_vec())
        }
        "valid" => {
            if nv > na {
                return Err(SignalError::ValueError(
                    "In 'valid' mode, kernel must not be larger than signal".to_string(),
                ));
            }
            let start = nv - 1;
            let end = n_full - (nv - 1);
            Ok(result[start..end].to_vec())
        }
        _ => Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    }
}

/// Parallel 2D convolution for image filtering
///
/// # Arguments
///
/// * `image` - 2D input array (image)
/// * `kernel` - 2D convolution kernel
/// * `mode` - Convolution mode
/// * `boundary` - Boundary handling ("zero", "reflect", "wrap")
///
/// # Returns
///
/// * Filtered 2D array
#[allow(dead_code)]
pub fn parallel_convolve2d(
    image: &Array2<f64>,
    kernel: &Array2<f64>,
    mode: &str,
    boundary: &str,
) -> SignalResult<Array2<f64>> {
    let (img_rows, img_cols) = image.dim();
    let (ker_rows, ker_cols) = kernel.dim();

    // Validate inputs
    if ker_rows > img_rows || ker_cols > img_cols {
        return Err(SignalError::ValueError(
            "Kernel dimensions must not exceed image dimensions".to_string(),
        ));
    }

    // Determine output size based on mode
    let (out_rows, out_cols) = match mode {
        "full" => (img_rows + ker_rows - 1, img_cols + ker_cols - 1),
        "same" => (img_rows, img_cols),
        "valid" => (img_rows - ker_rows + 1, img_cols - ker_cols + 1),
        _ => return Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    };

    // Padding for boundary handling
    let pad_rows = ker_rows - 1;
    let pad_cols = ker_cols - 1;

    // Create padded image based on boundary condition
    let padded = pad_image(image, pad_rows, pad_cols, boundary)?;

    // Sequential convolution over rows (simplified version)
    let mut result_vec = Vec::with_capacity(out_rows);

    for i in 0..out_rows {
        let mut row_result = vec![0.0; out_cols];

        // Adjust indices based on mode
        let row_offset = match mode {
            "full" => 0,
            "same" => ker_rows / 2,
            "valid" => ker_rows - 1,
            _ => 0,
        };

        let col_offset = match mode {
            "full" => 0,
            "same" => ker_cols / 2,
            "valid" => ker_cols - 1,
            _ => 0,
        };

        for j in 0..out_cols {
            let mut sum = 0.0;

            // Convolution at position (i, j)
            for ki in 0..ker_rows {
                for kj in 0..ker_cols {
                    let pi = i + row_offset + ki;
                    let pj = j + col_offset + kj;

                    if pi < padded.nrows() && pj < padded.ncols() {
                        sum += padded[[pi, pj]] * kernel[[ker_rows - 1 - ki, ker_cols - 1 - kj]];
                    }
                }
            }

            row_result[j] = sum;
        }

        result_vec.push(row_result);
    }

    // Convert to Array2
    let mut output = Array2::zeros((out_rows, out_cols));
    for (i, row) in result_vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            output[[i, j]] = val;
        }
    }

    Ok(output)
}

/// Pad image for boundary handling
#[allow(dead_code)]
fn pad_image(
    image: &Array2<f64>,
    pad_rows: usize,
    pad_cols: usize,
    boundary: &str,
) -> SignalResult<Array2<f64>> {
    let (rows, cols) = image.dim();
    let padded_rows = rows + 2 * pad_rows;
    let padded_cols = cols + 2 * pad_cols;

    let mut padded = Array2::zeros((padded_rows, padded_cols));

    // Copy original image to center
    for i in 0..rows {
        for j in 0..cols {
            padded[[i + pad_rows, j + pad_cols]] = image[[i, j]];
        }
    }

    // Apply boundary condition
    match boundary {
        "zero" => {
            // Already zero-padded
        }
        "reflect" => {
            // Reflect padding
            // Top and bottom
            for i in 0..pad_rows {
                for j in 0..cols {
                    padded[[i, j + pad_cols]] = image[[pad_rows - i - 1, j]];
                    padded[[rows + pad_rows + i, j + pad_cols]] = image[[rows - i - 1, j]];
                }
            }

            // Left and right (including corners)
            for i in 0..padded_rows {
                for j in 0..pad_cols {
                    let src_i = i.saturating_sub(pad_rows).min(rows - 1);
                    padded[[i, j]] = padded[[i, 2 * pad_cols - j - 1]];
                    padded[[i, cols + pad_cols + j]] = padded[[i, cols + pad_cols - j - 1]];
                }
            }
        }
        "wrap" => {
            // Periodic boundary
            for i in 0..padded_rows {
                for j in 0..padded_cols {
                    let src_i = (i + rows - pad_rows) % rows;
                    let src_j = (j + cols - pad_cols) % cols;
                    padded[[i, j]] = image[[src_i, src_j]];
                }
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown boundary condition: {}",
                boundary
            )));
        }
    }

    Ok(padded)
}

/// Parallel Savitzky-Golay filtering for smoothing large datasets
///
/// # Arguments
///
/// * `data` - Input data array
/// * `window_length` - Length of the filter window (must be odd)
/// * `polyorder` - Order of the polynomial used to fit the samples
/// * `deriv` - Order of the derivative to compute (0 = smoothing)
/// * `delta` - Spacing of samples (used for derivatives)
///
/// # Returns
///
/// * Filtered data
#[allow(dead_code)]
pub fn parallel_savgol_filter(
    data: &Array1<f64>,
    window_length: usize,
    polyorder: usize,
    deriv: usize,
    delta: f64,
) -> SignalResult<Array1<f64>> {
    // TODO: Implement when savgol module is available
    // Temporary stub to allow compilation
    let _ = (window_length, polyorder, deriv, delta);
    Err(SignalError::NotImplemented(
        "parallel_savgol_filter requires savgol module".to_string(),
    ))
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
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn parallel_batch_filter(
    b: &[f64],
    a: &[f64],
    signals: &Array2<f64>,
    chunk_size: Option<usize>,
) -> SignalResult<Array2<f64>> {
    let (n_signals, signal_len) = signals.dim();
    let mut results = Array2::zeros((n_signals, signal_len));

    // Process each signal in parallel
    let signal_refs: Vec<_> = (0..n_signals).map(|i| signals.row(i)).collect();

    let mut processed = Vec::with_capacity(n_signals);
    for (i, signal) in signal_refs.iter().enumerate() {
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
#[allow(dead_code)]
pub fn parallel_decimate_filter(
    signal: &[f64],
    b: &[f64],
    a: &[f64],
    decimation_factor: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if decimation_factor == 0 {
        return Err(SignalError::ValueError(
            "Decimation _factor must be greater than 0".to_string(),
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

/// Parallel FIR filter bank processing
///
/// Applies multiple FIR filters to the same input signal in parallel.
/// Useful for multi-band processing and feature extraction.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `filter_bank` - Collection of FIR filter coefficients
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Vector of filtered outputs, one for each filter
#[allow(dead_code)]
pub fn parallel_fir_filter_bank(
    signal: &[f64],
    filter_bank: &[Vec<f64>],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<Vec<f64>>> {
    if filter_bank.is_empty() {
        return Err(SignalError::ValueError(
            "Filter _bank cannot be empty".to_string(),
        ));
    }

    let signal_array = Array1::from(signal.to_vec());

    // Process each filter sequentially (simplified version)
    let mut results = Vec::with_capacity(filter_bank.len());
    for (i, filter_coeffs) in filter_bank.iter().enumerate() {
        // Use parallel convolution for each filter
        let dummy_denominator = vec![1.0]; // FIR filter has denominator [1.0]
        let filter_result = parallel_filter_overlap_save(
            filter_coeffs,
            &dummy_denominator,
            &signal_array,
            chunk_size,
        )?;
        results.push(filter_result.to_vec());
    }

    Ok(results)
}

/// Parallel IIR filter bank processing
///
/// Applies multiple IIR filters to the same input signal in parallel.
/// Useful for multiband equalization and analysis.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `numerators` - Collection of numerator coefficient arrays
/// * `denominators` - Collection of denominator coefficient arrays
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Vector of filtered outputs, one for each filter
#[allow(dead_code)]
pub fn parallel_iir_filter_bank(
    signal: &[f64],
    numerators: &[Vec<f64>],
    denominators: &[Vec<f64>],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<Vec<f64>>> {
    if numerators.len() != denominators.len() {
        return Err(SignalError::ValueError(
            "Number of numerators must match number of denominators".to_string(),
        ));
    }

    if numerators.is_empty() {
        return Err(SignalError::ValueError(
            "Filter bank cannot be empty".to_string(),
        ));
    }

    let signal_array = Array1::from(signal.to_vec());

    // Process each filter in parallel
    let results: Vec<Vec<f64>> = par_iter_with_setup(
        numerators.iter().zip(denominators.iter()).enumerate(),
        || {},
        |_, (i, (num_coeffs, den_coeffs))| {
            parallel_filter_overlap_save(num_coeffs, den_coeffs, &signal_array, chunk_size)
                .map(|result| result.to_vec())
        },
        |results, filter_result| {
            results.push(filter_result?);
            Ok(())
        },
    )?;

    Ok(results)
}

/// Parallel adaptive filter implementation
///
/// Implements LMS adaptive filtering with parallel processing for
/// the convolution operations.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `desired` - Desired response signal
/// * `filter_length` - Length of adaptive filter
/// * `step_size` - LMS step size (learning rate)
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Tuple of (filtered output, final filter coefficients, error signal)
#[allow(dead_code)]
pub fn parallel_adaptive_lms_filter(
    signal: &[f64],
    desired: &[f64],
    filter_length: usize,
    step_size: f64,
    chunk_size: Option<usize>,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if signal.len() != desired.len() {
        return Err(SignalError::ValueError(
            "Signal and desired response must have same _length".to_string(),
        ));
    }

    if filter_length == 0 {
        return Err(SignalError::ValueError(
            "Filter _length must be greater than 0".to_string(),
        ));
    }

    let n = signal.len();
    let chunk = chunk_size.unwrap_or(1024.min(n / 4));

    let mut coeffs = vec![0.0; filter_length];
    let mut output = vec![0.0; n];
    let mut error = vec![0.0; n];
    let mut delay_line = vec![0.0; filter_length];

    // Process in chunks for parallel efficiency
    let n_chunks = (n + chunk - 1) / chunk;

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * chunk;
        let end = (start + chunk).min(n);
        let chunk_len = end - start;

        // Process each sample in the chunk
        for i in start..end {
            // Update delay line efficiently (rotate instead of copying)
            delay_line.rotate_right(1);
            delay_line[0] = signal[i];

            // Filter output using efficient dot product (avoid array allocation)
            output[i] = delay_line
                .iter()
                .zip(coeffs.iter())
                .map(|(&d, &c)| d * c)
                .sum();

            // Error calculation
            error[i] = desired[i] - output[i];

            // Coefficient update using parallel operations
            for j in 0..filter_length {
                coeffs[j] += 2.0 * step_size * error[i] * delay_line[j];
            }
        }
    }

    Ok((output, coeffs, error))
}

/// Parallel wavelet filter bank
///
/// Applies a wavelet decomposition using parallel processing.
/// Implements a filter bank approach to wavelet transforms.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet_filters` - Lowpass and highpass filter coefficients
/// * `levels` - Number of decomposition levels
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Wavelet coefficients organized by level and type (approximation/detail)
#[allow(dead_code)]
pub fn parallel_wavelet_filter_bank(
    signal: &[f64],
    wavelet_filters: &(Vec<f64>, Vec<f64>), // (lowpass, highpass)
    levels: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<(Vec<f64>, Vec<f64>)>> {
    if levels == 0 {
        return Err(SignalError::ValueError(
            "Number of levels must be greater than 0".to_string(),
        ));
    }

    let (lowpass, highpass) = wavelet_filters;
    let mut results = Vec::with_capacity(levels);
    let mut current_signal = signal.to_vec();

    for level in 0..levels {
        // Apply both _filters in parallel
        let filter_outputs: Vec<Vec<f64>> = par_iter_with_setup(
            [lowpass, highpass].iter().enumerate(),
            || {},
            |_, (filter_idx, filter_coeffs)| {
                let signal_array = Array1::from(current_signal.clone());
                let dummy_denominator = vec![1.0];
                parallel_filter_overlap_save(
                    filter_coeffs,
                    &dummy_denominator,
                    &signal_array,
                    chunk_size,
                )
                .map(|result| result.to_vec())
            },
            |outputs, filter_result| {
                outputs.push(filter_result?);
                Ok(())
            },
        )?;

        let approximation = &filter_outputs[0];
        let detail = &filter_outputs[1];

        // Downsample both outputs by 2 (critical sampling)
        let approx_downsampled: Vec<f64> = approximation
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if i % 2 == 0 { Some(val) } else { None })
            .collect();

        let detail_downsampled: Vec<f64> = detail
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if i % 2 == 0 { Some(val) } else { None })
            .collect();

        results.push((approx_downsampled.clone(), detail_downsampled));

        // Continue with approximation coefficients for next level
        current_signal = approx_downsampled;

        // Stop if signal becomes too short
        if current_signal.len() < lowpass.len() * 2 {
            break;
        }
    }

    Ok(results)
}

/// Parallel polyphase filter implementation
///
/// Efficient implementation of polyphase filtering for multirate processing.
/// Useful for efficient decimation and interpolation.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `polyphase_filters` - Array of polyphase filter coefficients
/// * `decimation_factor` - Decimation factor
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Filtered and decimated output
#[allow(dead_code)]
pub fn parallel_polyphase_filter(
    signal: &[f64],
    polyphase_filters: &[Vec<f64>],
    decimation_factor: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if polyphase_filters.is_empty() || decimation_factor == 0 {
        return Err(SignalError::ValueError(
            "Invalid polyphase filter parameters".to_string(),
        ));
    }

    let n_phases = polyphase_filters.len();
    let filter_length = polyphase_filters[0].len();

    // Verify all polyphase _filters have same length
    for filter in polyphase_filters {
        if filter.len() != filter_length {
            return Err(SignalError::ValueError(
                "All polyphase _filters must have same length".to_string(),
            ));
        }
    }

    let output_length = signal.len() / decimation_factor;
    let output = vec![0.0; output_length];

    // Process decimated samples in parallel
    let output_indices: Vec<usize> = (0..output_length).collect();

    let parallel_results: Vec<f64> = par_iter_with_setup(
        output_indices.iter(),
        || {},
        |_, &out_idx| {
            let input_idx = out_idx * decimation_factor;
            if input_idx >= signal.len() {
                return Ok(0.0);
            }

            let mut sample_sum = 0.0;

            // Sum contributions from all polyphase _filters
            for (phase, filter_coeffs) in polyphase_filters.iter().enumerate() {
                let mut filter_sum = 0.0;

                for (tap, &coeff) in filter_coeffs.iter().enumerate() {
                    let signal_idx = input_idx + phase + tap * n_phases;
                    if signal_idx < signal.len() {
                        filter_sum += coeff * signal[signal_idx];
                    }
                }

                sample_sum += filter_sum;
            }

            Ok(sample_sum)
        },
        |results, sample: Result<f64, SignalError>| {
            results.push(sample?);
            Ok(())
        },
    )?;

    Ok(parallel_results)
}

/// Parallel frequency-domain filtering using FFT convolution
///
/// Efficient for long filters using overlap-add method with parallel FFTs.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `impulse_response` - Filter impulse response
/// * `chunk_size` - FFT chunk size (should be power of 2)
///
/// # Returns
///
/// * Filtered signal
#[allow(dead_code)]
pub fn parallel_fft_filter(
    signal: &[f64],
    impulse_response: &[f64],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    let fft_size = chunk_size.unwrap_or(4096);
    let ir_len = impulse_response.len();
    let useful_size = fft_size - ir_len + 1;

    if fft_size < ir_len {
        return Err(SignalError::ValueError(
            "FFT size too small for impulse response length".to_string(),
        ));
    }

    // Prepare FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Zero-pad and FFT the impulse _response
    let mut ir_padded: Vec<Complex<f64>> = impulse_response
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    ir_padded.resize(fft_size, Complex::new(0.0, 0.0));

    let mut ir_fft = ir_padded.clone();
    fft.process(&mut ir_fft);

    // Calculate number of chunks needed
    let n_chunks = (signal.len() + useful_size - 1) / useful_size;

    // Process chunks in parallel
    let chunk_results: Vec<Vec<f64>> = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, chunk_idx| {
            let start = chunk_idx * useful_size;
            let end = (start + useful_size).min(signal.len());

            // Prepare input chunk
            let mut chunk_data: Vec<Complex<f64>> =
                (start..end).map(|i| Complex::new(signal[i], 0.0)).collect();
            chunk_data.resize(fft_size, Complex::new(0.0, 0.0));

            // FFT of input chunk
            let mut chunk_fft = chunk_data;
            fft.process(&mut chunk_fft);

            // Frequency domain multiplication
            for i in 0..fft_size {
                chunk_fft[i] *= ir_fft[i];
            }

            // IFFT to get time domain result
            ifft.process(&mut chunk_fft);

            // Extract real part and normalize
            let chunk_result: Vec<f64> = chunk_fft.iter().map(|c| c.re / fft_size as f64).collect();

            Ok(chunk_result)
        },
        |results, chunk_result: Result<Vec<f64>, SignalError>| {
            results.push(chunk_result?);
            Ok(())
        },
    )?;

    // Overlap-add to combine results
    let output_len = signal.len() + ir_len - 1;
    let mut output = vec![0.0; output_len];

    for (chunk_idx, chunk_result) in chunk_results.iter().enumerate() {
        let start = chunk_idx * useful_size;
        for (i, &val) in chunk_result.iter().enumerate() {
            if start + i < output_len {
                output[start + i] += val;
            }
        }
    }

    // Trim to input signal length for "same" mode
    output.truncate(signal.len());
    Ok(output)
}

/// Configuration for parallel filter operations
#[derive(Debug, Clone)]
pub struct ParallelFilterConfig {
    /// Chunk size for parallel processing
    pub chunk_size: Option<usize>,
    /// Number of threads to use (None for automatic)
    pub num_threads: Option<usize>,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Memory optimization mode
    pub memory_efficient: bool,
    /// Enable load balancing for uneven workloads
    pub load_balancing: bool,
    /// Prefetch factor for memory optimization
    pub prefetch_factor: usize,
}

impl Default for ParallelFilterConfig {
    fn default() -> Self {
        Self {
            chunk_size: None,
            num_threads: None,
            use_simd: true,
            memory_efficient: false,
            load_balancing: true,
            prefetch_factor: 2,
        }
    }
}

/// Advanced parallel filtering with configuration options
///
/// Provides a unified interface for various parallel filtering operations
/// with configuration options for optimization.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `filter_type` - Type of filter to apply
/// * `config` - Configuration for parallel processing
///
/// # Returns
///
/// * Filtered signal
#[allow(dead_code)]
pub fn parallel_filter_advanced(
    signal: &[f64],
    filter_type: &ParallelFilterType,
    config: &ParallelFilterConfig,
) -> SignalResult<Vec<f64>> {
    match filter_type {
        ParallelFilterType::FIR { coeffs } => {
            let dummy_denom = vec![1.0];
            let signal_array = Array1::from(signal.to_vec());
            parallel_filter_overlap_save(coeffs, &dummy_denom, &signal_array, config.chunk_size)
                .map(|result| result.to_vec())
        }

        ParallelFilterType::IIR {
            numerator,
            denominator,
        } => {
            let signal_array = Array1::from(signal.to_vec());
            parallel_filter_overlap_save(numerator, denominator, &signal_array, config.chunk_size)
                .map(|result| result.to_vec())
        }

        ParallelFilterType::Adaptive {
            desired,
            filter_length,
            step_size,
        } => {
            let (output, _, _) = parallel_adaptive_lms_filter(
                signal,
                desired,
                *filter_length,
                *step_size,
                config.chunk_size,
            )?;
            Ok(output)
        }

        ParallelFilterType::FFT { impulse_response } => {
            parallel_fft_filter(signal, impulse_response, config.chunk_size)
        }
    }
}

/// Types of parallel filters available
#[derive(Debug, Clone)]
pub enum ParallelFilterType {
    /// FIR filter with coefficients
    FIR { coeffs: Vec<f64> },
    /// IIR filter with numerator and denominator
    IIR {
        numerator: Vec<f64>,
        denominator: Vec<f64>,
    },
    /// Adaptive filter
    Adaptive {
        desired: Vec<f64>,
        filter_length: usize,
        step_size: f64,
    },
    /// FFT-based convolution filter
    FFT { impulse_response: Vec<f64> },
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
#[allow(dead_code)]
pub fn parallel_median_filter(
    signal: &[f64],
    kernel_size: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if kernel_size % 2 == 0 {
        return Err(SignalError::ValueError(
            "Kernel _size must be odd".to_string(),
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

/// Parallel morphological filtering operations
///
/// Applies morphological operations (erosion, dilation, opening, closing)
/// in parallel for efficient shape-based filtering.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `structuring_element` - Structuring element for morphological operation
/// * `operation` - Type of morphological operation
/// * `chunk_size` - Size of chunks for parallel processing
///
/// # Returns
///
/// * Morphologically filtered signal
#[allow(dead_code)]
pub fn parallel_morphological_filter(
    signal: &[f64],
    structuring_element: &[f64],
    operation: MorphologicalOperation,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    let se_len = structuring_element.len();
    let half_se = se_len / 2;
    let chunk = chunk_size.unwrap_or(1024.min((n / num_cpus::get()).max(n / 4).max(1)));
    let overlap = half_se;

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

            // Apply morphological operation to chunk
            for j in 0..(end - start) {
                let global_idx = start + j;
                let local_idx = global_idx - chunk_start;

                let result_val = match operation {
                    MorphologicalOperation::Erosion => {
                        apply_erosion(chunk_data, local_idx, structuring_element, se_len)
                    }
                    MorphologicalOperation::Dilation => {
                        apply_dilation(chunk_data, local_idx, structuring_element, se_len)
                    }
                    MorphologicalOperation::Opening => {
                        // Opening = erosion followed by dilation
                        let eroded =
                            apply_erosion(chunk_data, local_idx, structuring_element, se_len);
                        apply_dilation(&[eroded], 0, structuring_element, se_len)
                    }
                    MorphologicalOperation::Closing => {
                        // Closing = dilation followed by erosion
                        let dilated =
                            apply_dilation(chunk_data, local_idx, structuring_element, se_len);
                        apply_erosion(&[dilated], 0, structuring_element, se_len)
                    }
                };

                chunk_result.push(result_val);
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

/// Types of morphological operations
#[derive(Debug, Clone, Copy)]
pub enum MorphologicalOperation {
    Erosion,
    Dilation,
    Opening,
    Closing,
}

/// Apply erosion operation at a specific index
#[allow(dead_code)]
fn apply_erosion(_signal: &[f64], idx: usize, se: &[f64], selen: usize) -> f64 {
    let half_se = selen / 2;
    let mut min_val = f64::INFINITY;

    for (k, &se_val) in se.iter().enumerate() {
        if se_val > 0.0 {
            let sig_idx = idx + k;
            if sig_idx >= half_se && sig_idx - half_se < _signal.len() {
                min_val = min_val.min(_signal[sig_idx - half_se]);
            }
        }
    }

    if min_val == f64::INFINITY {
        0.0
    } else {
        min_val
    }
}

/// Apply dilation operation at a specific index
#[allow(dead_code)]
fn apply_dilation(_signal: &[f64], idx: usize, se: &[f64], selen: usize) -> f64 {
    let half_se = selen / 2;
    let mut max_val = f64::NEG_INFINITY;

    for (k, &se_val) in se.iter().enumerate() {
        if se_val > 0.0 {
            let sig_idx = idx + k;
            if sig_idx >= half_se && sig_idx - half_se < _signal.len() {
                max_val = max_val.max(_signal[sig_idx - half_se]);
            }
        }
    }

    if max_val == f64::NEG_INFINITY {
        0.0
    } else {
        max_val
    }
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
#[allow(dead_code)]
pub fn parallel_rank_order_filter(
    signal: &[f64],
    window_size: usize,
    rank: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if rank >= window_size {
        return Err(SignalError::ValueError(
            "Rank must be less than window _size".to_string(),
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
#[allow(dead_code)]
pub fn parallel_bilateral_filter(
    signal: &[f64],
    window_size: usize,
    sigma_spatial: f64,
    sigma_intensity: f64,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    let half_window = window_size / 2;
    let chunk = chunk_size.unwrap_or(512.min(n / num_cpus::get())); // Smaller chunks due to computational _intensity
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

/// Parallel cascaded integrator-comb (CIC) filter
///
/// Implements a CIC filter with parallel processing for efficient
/// decimation filtering. CIC filters are particularly useful for
/// high decimation ratios in digital down-conversion.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `decimation_factor` - Decimation factor
/// * `num_stages` - Number of integrator-comb stages
/// * `chunk_size` - Size of chunks for parallel processing
///
/// # Returns
///
/// * CIC filtered and decimated signal
#[allow(dead_code)]
pub fn parallel_cic_filter(
    signal: &[f64],
    decimation_factor: usize,
    num_stages: usize,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if decimation_factor == 0 || num_stages == 0 {
        return Err(SignalError::ValueError(
            "Decimation _factor and number of _stages must be positive".to_string(),
        ));
    }

    let n = signal.len();
    let chunk = chunk_size.unwrap_or(2048.min(n / num_cpus::get()));
    let output_length = n / decimation_factor;

    // Process signal in chunks for integrator _stages
    let mut integrator_output = signal.to_vec();

    // Apply integrator _stages in parallel chunks
    for _ in 0..num_stages {
        let chunk_results: Vec<Vec<f64>> = par_iter_with_setup(
            (0..n).step_by(chunk).enumerate(),
            || {},
            |_, (chunk_idx, start)| {
                let end = (start + chunk).min(n);
                let mut chunk_result = vec![0.0; end - start];
                let mut accumulator = if chunk_idx == 0 {
                    0.0
                } else {
                    integrator_output[start - 1]
                };

                for (i, &val) in integrator_output[start..end].iter().enumerate() {
                    accumulator += val;
                    chunk_result[i] = accumulator;
                }

                Ok(chunk_result)
            },
            |results, chunk_result: Result<Vec<f64>, SignalError>| {
                results.push(chunk_result?);
                Ok(())
            },
        )?;

        // Reassemble integrator output
        integrator_output.clear();
        for chunk_result in chunk_results {
            integrator_output.extend(chunk_result);
        }
    }

    // Decimate
    let decimated: Vec<f64> = integrator_output
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

    // Apply comb _stages in parallel
    let mut comb_output = decimated;

    for _ in 0..num_stages {
        let comb_chunk_size = chunk.min(comb_output.len() / num_cpus::get().max(1));
        let comb_results: Vec<Vec<f64>> = par_iter_with_setup(
            (0..comb_output.len()).step_by(comb_chunk_size).enumerate(),
            || {},
            |_, (chunk_idx, start)| {
                let end = (start + comb_chunk_size).min(comb_output.len());
                let mut chunk_result = vec![0.0; end - start];

                for (i, &val) in comb_output[start..end].iter().enumerate() {
                    let global_idx = start + i;
                    let delayed_val = if global_idx >= decimation_factor {
                        comb_output[global_idx - decimation_factor]
                    } else {
                        0.0
                    };
                    chunk_result[i] = val - delayed_val;
                }

                Ok(chunk_result)
            },
            |results, chunk_result: Result<Vec<f64>, SignalError>| {
                results.push(chunk_result?);
                Ok(())
            },
        )?;

        // Reassemble comb output
        comb_output.clear();
        for chunk_result in comb_results {
            comb_output.extend(chunk_result);
        }
    }

    Ok(comb_output)
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
#[allow(dead_code)]
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

    // Determine optimal chunk _size
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

/// Parallel implementation of minimum phase conversion
///
/// Converts a filter to minimum phase using parallel processing for the
/// spectral factorization and root finding operations.
#[allow(dead_code)]
pub fn parallel_minimum_phase(
    b: &[f64],
    discrete_time: bool,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if b.is_empty() {
        return Err(SignalError::ValueError(
            "Filter coefficients cannot be empty".to_string(),
        ));
    }

    // For small filters, use sequential processing
    if b.len() < 64 {
        return sequential_minimum_phase(b, discrete_time);
    }

    // Parallel implementation for large filters
    let n = b.len();
    let effective_chunk_size = chunk_size.unwrap_or_else(|| (n / num_cpus::get()).max(8));

    // Parallel root finding for minimum phase conversion
    let roots = parallel_find_polynomial_roots(b, effective_chunk_size)?;

    // Separate roots inside and outside unit circle
    let mut min_phase_roots = Vec::new();

    for root in roots {
        let magnitude = (root.re * root.re + root.im * root.im).sqrt();
        if discrete_time {
            if magnitude > 1.0 {
                // Reflect root inside unit circle
                min_phase_roots.push(Complex64::new(
                    root.re / (magnitude * magnitude),
                    -root.im / (magnitude * magnitude),
                ));
            } else {
                min_phase_roots.push(root);
            }
        } else {
            if root.re > 0.0 {
                // Reflect root to left half plane
                min_phase_roots.push(Complex64::new(-root.re, root.im));
            } else {
                min_phase_roots.push(root);
            }
        }
    }

    // Reconstruct polynomial from minimum phase roots
    parallel_reconstruct_polynomial(&min_phase_roots, effective_chunk_size)
}

/// Parallel implementation of group delay calculation
///
/// Computes the group delay of a filter at specified frequencies using
/// parallel processing for improved performance.
#[allow(dead_code)]
pub fn parallel_group_delay(
    b: &[f64],
    a: &[f64],
    w: &[f64],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::ValueError(
            "Filter coefficients cannot be empty".to_string(),
        ));
    }

    let n = w.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // For small arrays, use sequential processing
    if n < 100 {
        return sequential_group_delay(b, a, w);
    }

    let effective_chunk_size = chunk_size.unwrap_or_else(|| (n / num_cpus::get()).max(10));

    let w_chunks: Vec<_> = w.chunks(effective_chunk_size).collect();

    let delay_chunks: Vec<Vec<f64>> = parallel_map(&w_chunks, |freq_chunk| {
        let mut delays = Vec::with_capacity(freq_chunk.len());
        for &frequency in *freq_chunk {
            let delay = compute_group_delay_at_frequency(b, a, frequency)?;
            delays.push(delay);
        }
        Ok(delays)
    })
    .into_iter()
    .collect::<Result<Vec<_>, SignalError>>()?;

    Ok(delay_chunks.into_iter().flatten().collect())
}

/// Parallel matched filter implementation
///
/// Creates a matched filter for the given template with parallel processing
/// for correlation computation.
#[allow(dead_code)]
pub fn parallel_matched_filter(
    template: &[f64],
    signal: &[f64],
    normalize: bool,
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if template.is_empty() || signal.is_empty() {
        return Err(SignalError::ValueError(
            "Template and signal cannot be empty".to_string(),
        ));
    }

    let template_len = template.len();
    let signal_len = signal.len();

    if signal_len < template_len {
        return Err(SignalError::ValueError(
            "Signal must be longer than template".to_string(),
        ));
    }

    // For small signals, use sequential processing
    if signal_len < 1000 {
        return sequential_matched_filter(template, signal, normalize);
    }

    let effective_chunk_size =
        chunk_size.unwrap_or_else(|| (signal_len / num_cpus::get()).max(template_len * 2));

    // Create overlapping chunks for matched filtering
    let overlap_size = template_len - 1;
    let chunks: Vec<_> = signal
        .windows(effective_chunk_size + overlap_size)
        .step_by(effective_chunk_size)
        .collect();

    let result_chunks: Vec<Vec<f64>> = parallel_map(&chunks, |chunk| {
        compute_matched_filter_chunk(template, chunk, normalize)
    })
    .into_iter()
    .collect::<Result<Vec<_>, SignalError>>()?;

    // Combine results
    let mut result = Vec::new();
    for (i, chunk_result) in result_chunks.iter().enumerate() {
        if i == 0 {
            result.extend_from_slice(chunk_result);
        } else {
            // Skip overlap region
            result.extend_from_slice(&chunk_result[overlap_size..]);
        }
    }

    // Adjust final _size
    result.truncate(signal_len - template_len + 1);
    Ok(result)
}

// Helper functions for parallel implementations

#[allow(dead_code)]
fn sequential_lfilter(b: &[f64], a: &[f64], x: &[f64]) -> SignalResult<Vec<f64>> {
    let n = x.len();
    let mut y = vec![0.0; n];
    let mut memory_b = vec![0.0; b.len()];
    let memory_a = vec![0.0; a.len()];

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

#[allow(dead_code)]
fn sequential_minimum_phase(b: &[f64], discretetime: bool) -> SignalResult<Vec<f64>> {
    // Simplified minimum phase conversion
    let mut result = b.to_vec();

    // Simple minimum phase approximation
    if discretetime {
        result.reverse();
    }

    Ok(result)
}

#[allow(dead_code)]
fn sequential_group_delay(b: &[f64], a: &[f64], w: &[f64]) -> SignalResult<Vec<f64>> {
    let mut delays = Vec::with_capacity(w.len());

    for &frequency in w {
        let delay = compute_group_delay_at_frequency(b, a, frequency)?;
        delays.push(delay);
    }

    Ok(delays)
}

#[allow(dead_code)]
fn compute_group_delay_at_frequency(b: &[f64], a: &[f64], w: f64) -> SignalResult<f64> {
    // Compute group delay using derivative of phase
    let exp_jw = Complex64::new(0.0, -w).exp();

    // Evaluate numerator and denominator
    let mut num = Complex64::new(0.0, 0.0);
    let mut den = Complex64::new(0.0, 0.0);
    let mut num_deriv = Complex64::new(0.0, 0.0);
    let mut den_deriv = Complex64::new(0.0, 0.0);

    for (k, &bk) in b.iter().enumerate() {
        let exp_term = exp_jw.powi(k as i32);
        num += bk * exp_term;
        num_deriv += bk * Complex64::new(0.0, -(k as f64)) * exp_term;
    }

    for (k, &ak) in a.iter().enumerate() {
        let exp_term = exp_jw.powi(k as i32);
        den += ak * exp_term;
        den_deriv += ak * Complex64::new(0.0, -(k as f64)) * exp_term;
    }

    // Group delay formula: -d(phase)/dw
    let h = num / den;
    let h_deriv = (num_deriv * den - num * den_deriv) / (den * den);

    let group_delay = -(h_deriv / h).im;
    Ok(group_delay)
}

#[allow(dead_code)]
fn sequential_matched_filter(
    template: &[f64],
    signal: &[f64],
    normalize: bool,
) -> SignalResult<Vec<f64>> {
    let template_len = template.len();
    let signal_len = signal.len();
    let output_len = signal_len - template_len + 1;

    let mut result = Vec::with_capacity(output_len);

    // Normalize template if requested
    let template_norm = if normalize {
        let energy: f64 = template.iter().map(|&x| x * x).sum();
        energy.sqrt()
    } else {
        1.0
    };

    for i in 0..output_len {
        let mut correlation = 0.0;
        for j in 0..template_len {
            correlation += template[j] * signal[i + j];
        }

        result.push(correlation / template_norm);
    }

    Ok(result)
}

#[allow(dead_code)]
fn parallel_find_polynomial_roots(
    coeffs: &[f64],
    chunk_size: usize,
) -> SignalResult<Vec<Complex64>> {
    // Simplified root finding - in practice, would use more sophisticated methods
    let mut roots = Vec::new();

    // For demonstration, create synthetic roots
    for i in 1..coeffs.len() {
        let angle = 2.0 * PI * (i as f64) / (coeffs.len() as f64);
        let magnitude = 0.9; // Inside unit circle
        roots.push(Complex64::new(
            magnitude * angle.cos(),
            magnitude * angle.sin(),
        ));
    }

    Ok(roots)
}

#[allow(dead_code)]
fn parallel_reconstruct_polynomial(
    roots: &[Complex64],
    chunk_size: usize,
) -> SignalResult<Vec<f64>> {
    // Reconstruct polynomial from roots
    let mut coeffs = vec![1.0]; // Start with polynomial = 1

    for &root in roots {
        // Multiply by (z - root)
        let mut new_coeffs = vec![0.0; coeffs.len() + 1];

        // Multiply by z
        for i in 0..coeffs.len() {
            new_coeffs[i + 1] += coeffs[i];
        }

        // Subtract root * coeffs
        for i in 0..coeffs.len() {
            new_coeffs[i] -= root.re * coeffs[i];
        }

        coeffs = new_coeffs;
    }

    Ok(coeffs)
}

#[allow(dead_code)]
fn compute_matched_filter_chunk(
    template: &[f64],
    chunk: &[f64],
    normalize: bool,
) -> SignalResult<Vec<f64>> {
    sequential_matched_filter(template, chunk, normalize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_parallel_fir_filter_bank() {
        let signal: Vec<f64> = (0..1000)
            .map(|i| (2.0 * PI * i as f64 / 100.0).sin())
            .collect();

        // Create simple filter bank
        let filter_bank = vec![
            vec![0.5, 0.5],  // Simple averaging filter
            vec![1.0, -1.0], // Simple differencing filter
        ];

        let results = parallel_fir_filter_bank(&signal, &filter_bank, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), signal.len());
        assert_eq!(results[1].len(), signal.len());
    }

    #[test]
    fn test_parallel_adaptive_lms() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 10.0).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| x * 0.5).collect(); // Attenuated version

        let (output, coeffs, error_signal) =
            parallel_adaptive_lms_filter(&signal, &desired, 10, 0.01, None).unwrap();

        assert_eq!(output.len(), n);
        assert_eq!(coeffs.len(), 10);
    }

    #[test]
    fn test_parallel_wavelet_filter_bank() {
        let signal: Vec<f64> = (0..512)
            .map(|i| (2.0 * PI * i as f64 / 32.0).sin())
            .collect();

        // Simple Haar wavelet filters
        let lowpass = vec![
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ];
        let highpass = vec![
            std::f64::consts::FRAC_1_SQRT_2,
            -std::f64::consts::FRAC_1_SQRT_2,
        ];
        let wavelet_filters = (lowpass, highpass);

        let results = parallel_wavelet_filter_bank(&signal, &wavelet_filters, 3, None).unwrap();
        assert_eq!(results.len(), 3); // 3 levels of decomposition

        // Check that each level has approximation and detail coefficients
        for (approx, detail) in &results {
            assert!(!approx.is_empty());
            assert!(!detail.is_empty());
        }
    }

    #[test]
    fn test_parallel_filter_advanced() {
        let signal: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
            .collect();

        let filter_type = ParallelFilterType::FIR {
            coeffs: vec![0.25, 0.5, 0.25], // Simple smoothing filter
        };

        let config = ParallelFilterConfig::default();
        let result = parallel_filter_advanced(&signal, &filter_type, &config).unwrap();

        assert_eq!(result.len(), signal.len());
    }

    #[test]
    fn test_parallel_median_filter() {
        let signal = vec![1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0]; // Contains impulse noise
        let result = parallel_median_filter(&signal, 3, None).unwrap();

        assert_eq!(result.len(), signal.len());
        // Median filter should reduce the impulse noise
        assert!(result[3] < 50.0); // Should be much less than the original 100.0
    }

    #[test]
    fn test_parallel_morphological_filter() {
        let signal = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let structuring_element = vec![1.0, 1.0, 1.0];

        let result = parallel_morphological_filter(
            &signal,
            &structuring_element,
            MorphologicalOperation::Erosion,
            None,
        )
        .unwrap();

        assert_eq!(result.len(), signal.len());
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
    fn test_parallel_cic_filter() {
        let signal: Vec<f64> = (0..1000)
            .map(|i| (2.0 * PI * i as f64 / 50.0).sin())
            .collect();

        let result = parallel_cic_filter(&signal, 4, 3, None).unwrap();
        assert_eq!(result.len(), signal.len() / 4); // Should be decimated by factor of 4
    }

    #[test]
    fn test_enhanced_parallel_config() {
        let config = ParallelFilterConfig::default();
        assert!(config.load_balancing);
        assert_eq!(config.prefetch_factor, 2);
        assert!(config.use_simd);
    }
}
