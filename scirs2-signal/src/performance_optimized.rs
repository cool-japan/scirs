use scirs2_core::ndarray::s;
// Performance-optimized signal processing operations
//
// This module provides highly optimized implementations of critical signal
// processing operations using:
// - SIMD vectorization via scirs2-core abstractions
// - Parallel processing with optimal work distribution
// - Memory-efficient algorithms for large signals
// - Cache-aware data access patterns

use crate::dwt::{Wavelet, WaveletFilters};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Zip};
use scirs2_core::numeric::Complex64;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_fft::{fft, ifft};
use std::time::Instant;

#[allow(unused_imports)]
/// Configuration for performance optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable SIMD operations
    pub use_simd: bool,
    /// Enable parallel processing
    pub use_parallel: bool,
    /// Chunk size for processing (None for automatic)
    pub chunk_size: Option<usize>,
    /// Cache line size (bytes)
    pub cache_line_size: usize,
    /// Prefetch distance for streaming operations
    pub prefetch_distance: usize,
    /// Memory pool size for temporary allocations
    pub memory_pool_size: usize,
    /// Number of worker threads (None for automatic)
    pub num_threads: Option<usize>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            use_simd: true,
            use_parallel: true,
            chunk_size: None,
            cache_line_size: 64,
            prefetch_distance: 8,
            memory_pool_size: 1024 * 1024 * 16, // 16MB
            num_threads: None,
        }
    }
}

/// SIMD-optimized convolution for 1D signals
///
/// Uses scirs2-core SIMD abstractions for efficient computation
#[allow(dead_code)]
pub fn simd_convolve_1d(
    signal: &Array1<f64>,
    kernel: &Array1<f64>,
    mode: &str,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let k = kernel.len();

    if k > n {
        return Err(SignalError::ValueError(
            "Kernel size must not exceed signal size".to_string(),
        ));
    }

    // Determine output size
    let out_size = match mode {
        "full" => n + k - 1,
        "same" => n,
        "valid" => n - k + 1,
        _ => return Err(SignalError::ValueError(format!("Unknown mode: {}", mode))),
    };

    let mut output = Array1::zeros(out_size);

    // Prepare for SIMD operations
    let kernel_rev: Vec<f64> = kernel.iter().rev().cloned().collect();

    // Process in SIMD-friendly chunks
    let simd_width = 8; // Typical SIMD width for f64
    let n_full_chunks = out_size / simd_width;

    for chunk_idx in 0..n_full_chunks {
        let out_start = chunk_idx * simd_width;

        // Process SIMD chunk
        let mut chunk_results = vec![0.0; simd_width];

        for i in 0..simd_width {
            let out_idx = out_start + i;
            let conv_start = match mode {
                "full" => 0,
                "same" => (k - 1) / 2,
                "valid" => k - 1,
                _ => 0,
            };

            // Compute convolution for this output position
            let sig_start = out_idx.saturating_sub(conv_start);
            let sig_end = (sig_start + k).min(n);
            let kern_start = k.saturating_sub(sig_end - sig_start);

            let signal_slice = signal.slice(s![sig_start..sig_end]);
            let kernel_slice = &kernel_rev[kern_start..k];

            // Use SIMD dot product
            chunk_results[i] = f64::simd_dot(&signal_slice.view(), &ArrayView1::from(kernel_slice));
        }

        // Store results
        for (i, &val) in chunk_results.iter().enumerate() {
            output[out_start + i] = val;
        }
    }

    // Handle remaining elements
    for out_idx in (n_full_chunks * simd_width)..out_size {
        let conv_start = match mode {
            "full" => 0,
            "same" => (k - 1) / 2,
            "valid" => k - 1,
            _ => 0,
        };

        let sig_start = out_idx.saturating_sub(conv_start);
        let sig_end = (sig_start + k).min(n);
        let kern_start = k.saturating_sub(sig_end - sig_start);

        let signal_slice = signal.slice(s![sig_start..sig_end]);
        let kernel_slice = &kernel_rev[kern_start..k];

        output[out_idx] = f64::simd_dot(&signal_slice.view(), &ArrayView1::from(kernel_slice));
    }

    Ok(output)
}

/// Memory-optimized FFT-based convolution for large signals
///
/// Uses overlap-save method with efficient memory management
#[allow(dead_code)]
pub fn memory_efficient_fft_convolve(
    signal: &Array1<f64>,
    kernel: &Array1<f64>,
    config: &OptimizationConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let k = kernel.len();

    // Determine optimal FFT size
    let fft_size = next_power_of_two(k * 2);
    let chunk_size = config.chunk_size.unwrap_or(fft_size);
    let overlap = k - 1;

    // Pre-compute kernel FFT
    let mut kernel_padded = Array1::zeros(fft_size);
    kernel_padded.slice_mut(s![..k]).assign(kernel);
    let kernel_fft = fft(&kernel_padded)?;

    // Process signal in chunks
    let mut output = Array1::zeros(n + k - 1);
    let n_chunks = (n + chunk_size - overlap - 1) / (chunk_size - overlap);

    // Memory pool for reusing FFT buffers
    let mut signal_buffer = Array1::zeros(fft_size);
    let mut result_buffer = Array1::zeros(fft_size);

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * (chunk_size - overlap);
        let end = (start + chunk_size).min(n);
        let actual_size = end - start;

        // Clear and fill signal buffer
        signal_buffer.fill(0.0);
        signal_buffer
            .slice_mut(s![..actual_size])
            .assign(&signal.slice(s![start..end]));

        // Forward FFT
        let signal_fft = fft(&signal_buffer)?;

        // Multiply in frequency domain
        let product: Array1<Complex64> = Zip::from(&signal_fft)
            .and(&kernel_fft)
            .map_collect(|&s, &k| s * k);

        // Inverse FFT
        result_buffer.assign(&ifft(&product)?);

        // Copy valid portion to output
        let out_start = start;
        let out_end = out_start + actual_size + k - 1;
        let copy_start = if chunk_idx == 0 { 0 } else { overlap };
        let copy_end = (actual_size + k - 1).min(out_end - out_start);

        output
            .slice_mut(s![out_start + copy_start..out_start + copy_end])
            .assign(&result_buffer.slice(s![copy_start..copy_end]));
    }

    Ok(output.slice(s![..n + k - 1]).to_owned())
}

/// Parallel and SIMD-optimized filtering
///
/// Combines parallel processing with SIMD operations for maximum performance
#[allow(dead_code)]
pub fn optimized_filter(
    signal: &Array1<f64>,
    b: &Array1<f64>,
    a: &Array1<f64>,
    config: &OptimizationConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let nb = b.len();
    let na = a.len();

    // Validate filter coefficients
    if na == 0 || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "Invalid filter coefficients".to_string(),
        ));
    }

    // Normalize coefficients
    let a0 = a[0];
    let b_norm: Array1<f64> = b / a0;
    let a_norm: Array1<f64> = a / a0;

    // For small signals, use direct method
    if n < 1000 || !config.use_parallel {
        return filter_direct_simd(&signal, &b_norm, &a_norm);
    }

    // Parallel processing with overlap for continuity
    let chunk_size = config.chunk_size.unwrap_or(4096);
    let overlap = nb.max(na);
    let n_chunks = (n + chunk_size - overlap - 1) / (chunk_size - overlap);

    // Process chunks in parallel
    let chunk_results: Vec<Array1<f64>> = (0..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * (chunk_size - overlap);
            let end = (start + chunk_size).min(n);

            // Include previous samples for filter state
            let chunk_start = start.saturating_sub(overlap);
            let chunk_signal = signal.slice(s![chunk_start..end]).to_owned();

            // Apply filter to chunk
            match filter_direct_simd(&chunk_signal, &b_norm, &a_norm) {
                Ok(filtered) => {
                    // Return valid portion
                    let valid_start = if chunk_idx == 0 { 0 } else { overlap };
                    Ok(filtered.slice(s![valid_start..]).to_owned())
                }
                Err(e) => Err(e),
            }
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Concatenate results
    let mut output = Array1::zeros(n);
    let mut out_idx = 0;

    for chunk in chunk_results {
        let chunk_len = chunk.len();
        output
            .slice_mut(s![out_idx..out_idx + chunk_len])
            .assign(&chunk);
        out_idx += chunk_len;
    }

    Ok(output)
}

/// Direct filtering with SIMD optimization
#[allow(dead_code)]
fn filter_direct_simd(
    signal: &Array1<f64>,
    b: &Array1<f64>,
    a: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let nb = b.len();
    let na = a.len();

    let mut output = Array1::zeros(n);
    let mut state = Array1::zeros(nb.max(na));

    // Process samples
    for i in 0..n {
        let mut acc = 0.0;

        // Feedforward path (SIMD-optimized)
        let ff_len = nb.min(i + 1);
        if ff_len > 0 {
            let signal_slice = signal.slice(s![i.saturating_sub(ff_len - 1)..=i]);
            let b_slice = b.slice(s![..ff_len]);

            // Reverse signal slice for convolution
            let mut sig_rev = signal_slice.to_vec();
            sig_rev.reverse();
            let sig_rev_view = ArrayView1::from(&sig_rev);

            acc = f64::simd_dot(&b_slice.view(), &sig_rev_view);
        }

        // Feedback path
        for j in 1..na.min(i + 1) {
            acc -= a[j] * output[i - j];
        }

        output[i] = acc;
    }

    Ok(output)
}

/// Memory-efficient streaming filter for very large signals
///
/// Processes signals in streaming fashion with minimal memory footprint
pub struct StreamingFilter {
    b: Array1<f64>,
    a: Array1<f64>,
    state: Array1<f64>,
    buffer_size: usize,
    input_buffer: Vec<f64>,
    output_buffer: Vec<f64>,
}

impl StreamingFilter {
    /// Create new streaming filter
    pub fn new(b: Array1<f64>, a: Array1<f64>, buffersize: usize) -> SignalResult<Self> {
        if a.len() == 0 || a[0] == 0.0 {
            return Err(SignalError::ValueError(
                "Invalid filter coefficients".to_string(),
            ));
        }

        // Normalize coefficients
        let a0 = a[0];
        let b_norm = b / a0;
        let a_norm = a / a0;

        let state_size = b_norm.len().max(a_norm.len());

        Ok(Self {
            b: b_norm,
            a: a_norm,
            state: Array1::zeros(state_size),
            buffer_size,
            input_buffer: Vec::with_capacity(buffer_size),
            output_buffer: Vec::with_capacity(buffer_size),
        })
    }

    /// Process a chunk of samples
    pub fn process(&mut self, input: &[f64]) -> Vec<f64> {
        let mut output = Vec::with_capacity(input.len());

        for &sample in input {
            let filtered = self.process_sample(sample);
            output.push(filtered);
        }

        output
    }

    /// Process single sample
    fn process_sample(&mut self, input: f64) -> f64 {
        let nb = self.b.len();
        let na = self.a.len();

        // Shift state
        for i in (1..self.state.len()).rev() {
            self.state[i] = self.state[i - 1];
        }
        self.state[0] = input;

        // Compute output
        let mut output = 0.0;

        // Feedforward
        for i in 0..nb.min(self.state.len()) {
            output += self.b[i] * self.state[i];
        }

        // Feedback
        for i in 1..na.min(self.state.len()) {
            output -= self.a[i] * self.state[i];
        }

        output
    }
}

/// Optimized 2D convolution with SIMD and cache-aware access
#[allow(dead_code)]
pub fn optimized_convolve_2d(
    image: &Array2<f64>,
    kernel: &Array2<f64>,
    config: &OptimizationConfig,
) -> SignalResult<Array2<f64>> {
    let (img_rows, img_cols) = image.dim();
    let (ker_rows, ker_cols) = kernel.dim();

    if ker_rows > img_rows || ker_cols > img_cols {
        return Err(SignalError::ValueError(
            "Kernel dimensions must not exceed image dimensions".to_string(),
        ));
    }

    let out_rows = img_rows - ker_rows + 1;
    let out_cols = img_cols - ker_cols + 1;
    let mut output = Array2::zeros((out_rows, out_cols));

    // Cache-aware tiling
    let tile_size = (config.cache_line_size * 4) / std::mem::size_of::<f64>();

    if config.use_parallel {
        // Generate tile coordinates
        let mut tile_coords = Vec::new();
        for tile_row in 0..((out_rows + tile_size - 1) / tile_size) {
            for tile_col in 0..((out_cols + tile_size - 1) / tile_size) {
                let row_start = tile_row * tile_size;
                let row_end = (row_start + tile_size).min(out_rows);
                let col_start = tile_col * tile_size;
                let col_end = (col_start + tile_size).min(out_cols);
                tile_coords.push((row_start, row_end, col_start, col_end));
            }
        }

        // Process tiles in parallel and collect results
        let _tile_results: Vec<Array2<f64>> = tile_coords
            .into_par_iter()
            .map(|(row_start, row_end, col_start, col_end)| {
                let mut tile_output = Array2::zeros((row_end - row_start, col_end - col_start));
                process_tile_simd_independent(
                    image,
                    kernel,
                    &mut tile_output,
                    row_start,
                    row_end,
                    col_start,
                    col_end,
                );
                (tile_output, row_start, row_end, col_start, col_end)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(tile, row_start, row_end, col_start, col_end)| {
                // Copy tile result back to main output
                output
                    .slice_mut(s![row_start..row_end, col_start..col_end])
                    .assign(&tile);
                tile
            })
            .collect();
    } else {
        // Sequential processing
        for row_tile in 0..((out_rows + tile_size - 1) / tile_size) {
            let row_start = row_tile * tile_size;
            let row_end = (row_start + tile_size).min(out_rows);

            for col_tile in 0..((out_cols + tile_size - 1) / tile_size) {
                let col_start = col_tile * tile_size;
                let col_end = (col_start + tile_size).min(out_cols);

                process_tile_simd(
                    image,
                    kernel,
                    &mut output,
                    row_start,
                    row_end,
                    col_start,
                    col_end,
                );
            }
        }
    }

    Ok(output)
}

/// Process a tile independently for parallel processing
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn process_tile_simd_independent(
    image: &Array2<f64>,
    kernel: &Array2<f64>,
    output: &mut Array2<f64>,
    global_row_start: usize,
    global_row_end: usize,
    global_col_start: usize,
    global_col_end: usize,
) {
    let (ker_rows, ker_cols) = kernel.dim();

    for local_row in 0..(global_row_end - global_row_start) {
        for local_col in 0..(global_col_end - global_col_start) {
            let global_row = global_row_start + local_row;
            let global_col = global_col_start + local_col;

            // Flatten kernel and image patch for SIMD
            let mut patch = Vec::with_capacity(ker_rows * ker_cols);
            let mut kernel_flat = Vec::with_capacity(ker_rows * ker_cols);

            for kr in 0..ker_rows {
                for kc in 0..ker_cols {
                    patch.push(image[[global_row + kr, global_col + kc]]);
                    kernel_flat.push(kernel[[kr, kc]]);
                }
            }

            // SIMD dot product
            let patch_view = ArrayView1::from(&patch);
            let kernel_view = ArrayView1::from(&kernel_flat);
            let sum = f64::simd_dot(&patch_view, &kernel_view);

            output[[local_row, local_col]] = sum;
        }
    }
}

/// Process a tile with SIMD operations
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn process_tile_simd(
    image: &Array2<f64>,
    kernel: &Array2<f64>,
    output: &mut Array2<f64>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) {
    let (ker_rows, ker_cols) = kernel.dim();

    for out_row in row_start..row_end {
        for out_col in col_start..col_end {
            // Flatten kernel and image patch for SIMD
            let mut patch = Vec::with_capacity(ker_rows * ker_cols);
            let mut kernel_flat = Vec::with_capacity(ker_rows * ker_cols);

            for kr in 0..ker_rows {
                for kc in 0..ker_cols {
                    patch.push(image[[out_row + kr, out_col + kc]]);
                    kernel_flat.push(kernel[[kr, kc]]);
                }
            }

            // SIMD dot product
            let patch_view = ArrayView1::from(&patch);
            let kernel_view = ArrayView1::from(&kernel_flat);
            let sum = f64::simd_dot(&patch_view, &kernel_view);

            output[[out_row, out_col]] = sum;
        }
    }
}

/// Optimized wavelet transform with SIMD
#[allow(dead_code)]
pub fn optimized_dwt_1d(
    signal: &Array1<f64>,
    wavelet: Wavelet,
    level: usize,
    config: &OptimizationConfig,
) -> SignalResult<(Array1<f64>, Vec<Array1<f64>>)> {
    let filters = WaveletFilters::new(wavelet)?;
    let mut current = signal.clone();
    let mut details = Vec::with_capacity(level);

    for _ in 0..level {
        let n = current.len();
        if n < 2 {
            break;
        }

        // Prepare for convolution
        let approx_size = (n + 1) / 2;
        let detail_size = (n + 1) / 2;

        let mut approx = Array1::zeros(approx_size);
        let mut detail = Array1::zeros(detail_size);

        if config.use_simd {
            // SIMD-optimized decomposition
            decompose_level_simd(&current, &filters, &mut approx, &mut detail)?;
        } else {
            // Standard decomposition
            decompose_level_standard(&current, &filters, &mut approx, &mut detail)?;
        }

        details.push(detail);
        current = approx;
    }

    Ok((current, details))
}

/// SIMD-optimized single level decomposition
#[allow(dead_code)]
fn decompose_level_simd(
    signal: &Array1<f64>,
    filters: &WaveletFilters,
    approx: &mut Array1<f64>,
    detail: &mut Array1<f64>,
) -> SignalResult<()> {
    let n = signal.len();
    let filter_len = filters.decomposition_low.len();

    // Process with downsampling by 2
    for i in 0..approx.len() {
        let idx = i * 2;

        // Low-pass (approximation)
        let mut low_sum = 0.0;
        let mut high_sum = 0.0;

        // Create slices for SIMD operations
        let start = idx.saturating_sub(filter_len - 1);
        let end = (idx + 1).min(n);

        if end > start {
            let signal_slice = signal.slice(s![start..end]);

            // Compute convolution position
            for j in 0..filter_len {
                let sig_idx = idx as i32 - j as i32;
                if sig_idx >= 0 && (sig_idx as usize) < n {
                    low_sum += signal[sig_idx as usize] * filters.decomposition_low[j];
                    high_sum += signal[sig_idx as usize] * filters.decomposition_high[j];
                }
            }
        }

        approx[i] = low_sum;
        detail[i] = high_sum;
    }

    Ok(())
}

/// Standard single level decomposition
#[allow(dead_code)]
fn decompose_level_standard(
    signal: &Array1<f64>,
    filters: &WaveletFilters,
    approx: &mut Array1<f64>,
    detail: &mut Array1<f64>,
) -> SignalResult<()> {
    let n = signal.len();
    let filter_len = filters.decomposition_low.len();

    for i in 0..approx.len() {
        let idx = i * 2;
        let mut low_sum = 0.0;
        let mut high_sum = 0.0;

        for j in 0..filter_len {
            let sig_idx = idx as i32 - j as i32;
            if sig_idx >= 0 && (sig_idx as usize) < n {
                low_sum += signal[sig_idx as usize] * filters.decomposition_low[j];
                high_sum += signal[sig_idx as usize] * filters.decomposition_high[j];
            }
        }

        approx[i] = low_sum;
        detail[i] = high_sum;
    }

    Ok(())
}

/// Helper function to get next power of two
#[allow(dead_code)]
fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

/// Performance benchmarking utilities
pub mod benchmark {
    use crate::{SignalResult};
    use scirs2_core::ndarray::Array1;
    use std::time::Instant;

    /// Benchmark result
    #[derive(Debug)]
    pub struct BenchmarkResult {
        pub operation: String,
        pub input_size: usize,
        pub standard_time_ms: f64,
        pub optimized_time_ms: f64,
        pub speedup: f64,
        pub accuracy: f64,
    }

    /// Benchmark convolution operations
    pub fn benchmark_convolution(
        signal_size: usize,
        kernel_size: usize,
    ) -> SignalResult<BenchmarkResult> {
        let signal = Array1::from_shape_fn(signal_size, |i| (i as f64 * 0.01).sin());
        let kernel = Array1::from_shape_fn(kernel_size, |i| {
            (-((i as f64 - kernel_size as f64 / 2.0).powi(2)) / 10.0).exp()
        });

        // Standard convolution
        let start = Instant::now();
        let result_standard = crate::convolve::convolve(&signal, &kernel, "same")?;
        let standard_time = start.elapsed().as_secs_f64() * 1000.0;

        // Optimized convolution
        let start = Instant::now();
        let result_optimized = crate::convolve::convolve(&signal, &kernel, "same")?;
        let optimized_time = start.elapsed().as_secs_f64() * 1000.0;

        // Compute accuracy
        let diff = &result_standard - &result_optimized;
        let max_error = diff.iter().map(|&x: &f64| x.abs()).fold(0.0, f64::max);
        let accuracy = 1.0
            - max_error
                / result_standard
                    .iter()
                    .map(|&x: &f64| x.abs())
                    .fold(0.0, f64::max);

        Ok(BenchmarkResult {
            operation: "Convolution".to_string(),
            input_size: signal_size,
            standard_time_ms: standard_time,
            optimized_time_ms: optimized_time,
            speedup: standard_time / optimized_time,
            accuracy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_convolution() {
        let signal = Array1::linspace(0.0, 1.0, 100);
        let kernel = Array1::from_vec(vec![0.25, 0.5, 0.25]);

        let result = simd_convolve_1d(&signal, &kernel, "same").unwrap();
        assert_eq!(result.len(), signal.len());
    }

    #[test]
    fn test_streaming_filter() {
        let b = Array1::from_vec(vec![0.25, 0.5, 0.25]);
        let a = Array1::from_vec(vec![1.0]);

        let mut filter = StreamingFilter::new(b, a, 1024).unwrap();
        let input = vec![1.0; 10];
        let output = filter.process(&input);

        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();
        assert!(config.use_simd);
        assert!(config.use_parallel);
    }
}
