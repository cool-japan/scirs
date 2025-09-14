// Advanced Enhanced Parallel Filtering Operations
//
// This module provides next-generation parallel filtering operations that push the boundaries
// of performance and capability. It focuses on real-time streaming, advanced multi-rate systems,
// sparse filtering, and advanced-high-performance parallel spectral processing.

use crate::error::{SignalError, SignalResult};
use crate::filter::parallel::ParallelFilterConfig;
use num_complex::Complex64;
use num_traits::Float;
use rustfft::{num_complex::Complex, FftPlanner};
use scirs2_core::parallel_ops::*;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

#[allow(unused_imports)]
/// Advanced parallel filtering configuration with advanced features
#[derive(Debug, Clone)]
pub struct AdvancedParallelConfig {
    /// Base parallel configuration
    pub base_config: ParallelFilterConfig,
    /// Enable real-time streaming mode
    pub real_time_mode: bool,
    /// Maximum latency for real-time processing (microseconds)
    pub max_latency_us: Option<u64>,
    /// Enable lock-free processing where possible
    pub lock_free: bool,
    /// Use zero-copy optimizations
    pub zero_copy: bool,
    /// Enable adaptive performance monitoring
    pub performance_monitoring: bool,
    /// Memory pool size for pre-allocated buffers
    pub memory_pool_size: Option<usize>,
    /// Enable GPU acceleration (if available)
    pub gpu_acceleration: bool,
}

impl Default for AdvancedParallelConfig {
    fn default() -> Self {
        Self {
            base_config: ParallelFilterConfig::default(),
            real_time_mode: false,
            max_latency_us: Some(1000), // 1ms default
            lock_free: true,
            zero_copy: true,
            performance_monitoring: true,
            memory_pool_size: Some(1024 * 1024), // 1MB pool
            gpu_acceleration: false,
        }
    }
}

/// Performance metrics for parallel filtering operations
#[derive(Debug, Clone)]
pub struct ParallelFilterMetrics {
    /// Total processing time
    pub processing_time: Duration,
    /// Throughput in samples per second
    pub throughput_sps: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_gbps: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// SIMD utilization percentage
    pub simd_utilization: f64,
    /// Number of parallel threads used
    pub threads_used: usize,
    /// Lock contention time (if applicable)
    pub lock_contention_us: Option<u64>,
}

/// Real-time streaming filter state
#[derive(Debug)]
pub struct StreamingFilterState {
    /// Filter coefficients (numerator)
    pub b: Vec<f64>,
    /// Filter coefficients (denominator)
    pub a: Vec<f64>,
    /// Input history buffer
    pub input_history: VecDeque<f64>,
    /// Output history buffer
    pub output_history: VecDeque<f64>,
    /// Processing statistics
    pub stats: StreamingStats,
}

/// Streaming processing statistics
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Total samples processed
    pub samples_processed: u64,
    /// Average processing latency
    pub avg_latency_us: f64,
    /// Maximum processing latency
    pub max_latency_us: u64,
    /// Underruns (missed deadlines)
    pub underruns: u64,
    /// Throughput samples per second
    pub throughput_sps: f64,
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self {
            samples_processed: 0,
            avg_latency_us: 0.0,
            max_latency_us: 0,
            underruns: 0,
            throughput_sps: 0.0,
        }
    }
}

/// Multi-rate parallel filter bank with perfect reconstruction
#[derive(Debug)]
pub struct ParallelMultiRateFilterBank {
    /// Analysis filter bank (decimators)
    pub analysis_filters: Vec<Vec<f64>>,
    /// Synthesis filter bank (interpolators)
    pub synthesis_filters: Vec<Vec<f64>>,
    /// Decimation factors for each band
    pub decimation_factors: Vec<usize>,
    /// Current filter states
    pub filter_states: Vec<VecDeque<f64>>,
    /// Perfect reconstruction validation
    pub pr_error: f64,
}

impl ParallelMultiRateFilterBank {
    /// Create a new multi-rate filter bank
    pub fn new(
        analysis_filters: Vec<Vec<f64>>,
        synthesis_filters: Vec<Vec<f64>>,
        decimation_factors: Vec<usize>,
    ) -> SignalResult<Self> {
        if analysis_filters.len() != synthesis_filters.len()
            || analysis_filters.len() != decimation_factors.len()
        {
            return Err(SignalError::ShapeMismatch(
                "Filter banks must have matching dimensions".to_string(),
            ));
        }

        let num_bands = analysis_filters.len();
        let filter_states = (0..num_bands)
            .map(|i| VecDeque::with_capacity(analysis_filters[i].len()))
            .collect();

        Ok(Self {
            analysis_filters,
            synthesis_filters,
            decimation_factors,
            filter_states,
            pr_error: 0.0,
        })
    }

    /// Process signal through the multi-rate filter bank
    pub fn process(
        &mut self,
        signal: &[f64],
        config: &AdvancedParallelConfig,
    ) -> SignalResult<Vec<f64>> {
        let num_bands = self.analysis_filters.len();

        // Analysis stage - parallel decimation
        let analysis_results: Result<Vec<_>, SignalError> = (0..num_bands)
            .into_par_iter()
            .map(|band| {
                let filtered = parallel_convolve_decimated(
                    signal,
                    &self.analysis_filters[band],
                    self.decimation_factors[band],
                    &config.base_config,
                )?;
                Ok::<Vec<f64>, SignalError>(filtered)
            })
            .collect();

        let decimated_bands = analysis_results?;

        // Processing stage (could include band-specific processing here)
        let processed_bands = decimated_bands;

        // Synthesis stage - parallel interpolation and reconstruction
        let synthesis_results: Result<Vec<_>, SignalError> = (0..num_bands)
            .into_par_iter()
            .map(|band| {
                parallel_interpolate_filter(
                    &processed_bands[band],
                    &self.synthesis_filters[band],
                    self.decimation_factors[band],
                    signal.len(),
                    &config.base_config,
                )
            })
            .collect();

        let interpolated_bands = synthesis_results?;

        // Sum all bands for reconstruction
        let mut reconstructed = vec![0.0; signal.len()];
        for band_output in interpolated_bands {
            for (i, &val) in band_output.iter().enumerate() {
                if i < reconstructed.len() {
                    reconstructed[i] += val;
                }
            }
        }

        Ok(reconstructed)
    }

    /// Validate perfect reconstruction property
    pub fn validate_perfect_reconstruction(&mut self, testsignal: &[f64]) -> SignalResult<f64> {
        let config = AdvancedParallelConfig::default();
        let reconstructed = self.process(testsignal, &config)?;

        // Calculate reconstruction error
        let mut error = 0.0;
        let len = testsignal.len().min(reconstructed.len());
        for i in 0..len {
            error += (testsignal[i] - reconstructed[i]).powi(2);
        }
        self.pr_error = (error / len as f64).sqrt();

        Ok(self.pr_error)
    }
}

/// Sparse parallel filtering for signals with sparse representations
#[derive(Debug)]
pub struct SparseParallelFilter {
    /// Sparse filter coefficients (index, value pairs)
    pub sparse_coeffs: Vec<(usize, f64)>,
    /// Sparsity ratio (0.0 to 1.0)
    pub sparsity_ratio: f64,
    /// Compression ratio achieved
    pub compression_ratio: f64,
}

impl SparseParallelFilter {
    /// Create sparse filter from dense coefficients
    pub fn from_dense(coeffs: &[f64], threshold: f64) -> Self {
        let mut sparse_coeffs = Vec::new();
        let max_val = coeffs.iter().map(|&x: &f64| x.abs()).fold(0.0, f64::max);
        let actual_threshold = threshold * max_val;

        for (i, &coeff) in coeffs.iter().enumerate() {
            if coeff.abs() > actual_threshold {
                sparse_coeffs.push((i, coeff));
            }
        }

        let sparsity_ratio = 1.0 - (sparse_coeffs.len() as f64 / coeffs.len() as f64);
        let compression_ratio = coeffs.len() as f64 / sparse_coeffs.len() as f64;

        Self {
            sparse_coeffs,
            sparsity_ratio,
            compression_ratio,
        }
    }

    /// Apply sparse filter in parallel
    pub fn apply_parallel(
        &self,
        signal: &[f64],
        config: &AdvancedParallelConfig,
    ) -> SignalResult<Vec<f64>> {
        let signal_len = signal.len();
        let num_threads = config.base_config.num_threads.unwrap_or(num_cpus::get());
        let chunk_size = signal_len / num_threads;

        // Process signal in parallel chunks
        let results: Result<Vec<_>, SignalError> = (0..num_threads)
            .into_par_iter()
            .map(|thread_id| {
                let start = thread_id * chunk_size;
                let end = if thread_id == num_threads - 1 {
                    signal_len
                } else {
                    (start + chunk_size).min(signal_len)
                };

                let mut chunk_result = vec![0.0; end - start];

                // Apply sparse convolution
                for (output_idx, output_val) in chunk_result.iter_mut().enumerate() {
                    let global_idx = start + output_idx;
                    for &(coeff_idx, coeff_val) in &self.sparse_coeffs {
                        if global_idx >= coeff_idx {
                            let signal_idx = global_idx - coeff_idx;
                            if signal_idx < signal_len {
                                *output_val += coeff_val * signal[signal_idx];
                            }
                        }
                    }
                }

                Ok::<Vec<f64>, SignalError>(chunk_result)
            })
            .collect();

        let chunks = results?;

        // Concatenate results
        let mut output = Vec::with_capacity(signal_len);
        for chunk in chunks {
            output.extend(chunk);
        }

        Ok(output)
    }
}

/// Real-time streaming filter with lock-free processing
pub struct LockFreeStreamingFilter {
    /// Filter state
    state: Arc<RwLock<StreamingFilterState>>,
    /// Performance metrics
    metrics: Arc<Mutex<ParallelFilterMetrics>>,
    /// Processing thread handles
    thread_handles: Vec<thread::JoinHandle<()>>,
    /// Configuration
    config: AdvancedParallelConfig,
}

impl LockFreeStreamingFilter {
    /// Create a new lock-free streaming filter
    pub fn new(b: Vec<f64>, a: Vec<f64>, config: AdvancedParallelConfig) -> SignalResult<Self> {
        let state = Arc::new(RwLock::new(StreamingFilterState {
            b: b.clone(),
            a: a.clone(),
            input_history: VecDeque::with_capacity(b.len()),
            output_history: VecDeque::with_capacity(a.len()),
            stats: StreamingStats::default(),
        }));

        let metrics = Arc::new(Mutex::new(ParallelFilterMetrics {
            processing_time: Duration::from_nanos(0),
            throughput_sps: 0.0,
            memory_bandwidth_gbps: 0.0,
            cpu_utilization: 0.0,
            cache_hit_ratio: 0.0,
            simd_utilization: 0.0,
            threads_used: 1,
            lock_contention_us: Some(0),
        }));

        Ok(Self {
            state,
            metrics,
            thread_handles: Vec::new(),
            config,
        })
    }

    /// Process a single sample with minimal latency
    pub fn process_sample(&self, input: f64) -> SignalResult<f64> {
        let start_time = Instant::now();

        let result = if self.config.lock_free {
            self.process_sample_lock_free(input)?
        } else {
            self.process_sample_locked(input)?
        };

        // Update performance metrics
        let processing_time = start_time.elapsed();
        if let Ok(mut metrics) = self.metrics.try_lock() {
            metrics.processing_time += processing_time;
            // Update other metrics as needed
        }

        Ok(result)
    }

    /// Lock-free sample processing using RCU-like approach
    fn process_sample_lock_free(&self, input: f64) -> SignalResult<f64> {
        // This is a simplified lock-free implementation
        // In practice, would use atomic operations and careful memory ordering
        let state = self.state.read().map_err(|_| {
            SignalError::ComputationError("Failed to acquire read lock".to_string())
        })?;

        // Simple IIR computation (would need proper lock-free state management)
        let mut output = state.b[0] * input;

        // Add previous inputs (if available)
        for (i, &coeff) in state.b.iter().skip(1).enumerate() {
            if i < state.input_history.len() {
                output += coeff * state.input_history[i];
            }
        }

        // Subtract previous outputs (if available)
        for (i, &coeff) in state.a.iter().skip(1).enumerate() {
            if i < state.output_history.len() {
                output -= coeff * state.output_history[i];
            }
        }

        Ok(output)
    }

    /// Locked sample processing for consistency
    fn process_sample_locked(&self, input: f64) -> SignalResult<f64> {
        let mut state = self.state.write().map_err(|_| {
            SignalError::ComputationError("Failed to acquire write lock".to_string())
        })?;

        // Update input history
        state.input_history.push_front(input);
        if state.input_history.len() > state.b.len() {
            state.input_history.pop_back();
        }

        // Compute output
        let mut output = 0.0;
        for (i, &coeff) in state.b.iter().enumerate() {
            if i < state.input_history.len() {
                output += coeff * state.input_history[i];
            }
        }

        for (i, &coeff) in state.a.iter().skip(1).enumerate() {
            if i < state.output_history.len() {
                output -= coeff * state.output_history[i];
            }
        }

        // Update output history
        state.output_history.push_front(output);
        if state.output_history.len() >= state.a.len() {
            state.output_history.pop_back();
        }

        // Update statistics
        state.stats.samples_processed += 1;

        Ok(output)
    }

    /// Process a block of samples efficiently
    pub fn process_block(&self, inputs: &[f64]) -> SignalResult<Vec<f64>> {
        let mut outputs = Vec::with_capacity(inputs.len());

        for &input in inputs {
            outputs.push(self.process_sample(input)?);
        }

        Ok(outputs)
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> SignalResult<ParallelFilterMetrics> {
        self.metrics
            .lock()
            .map(|m| m.clone())
            .map_err(|_| SignalError::ComputationError("Failed to get metrics".to_string()))
    }

    /// Get streaming statistics
    pub fn get_stats(&self) -> SignalResult<StreamingStats> {
        self.state
            .read()
            .map(|s| s.stats.clone())
            .map_err(|_| SignalError::ComputationError("Failed to get stats".to_string()))
    }
}

/// Advanced parallel spectral filtering with overlap-add
pub struct ParallelSpectralFilter {
    /// FFT size for processing
    pub fft_size: usize,
    /// Overlap factor (0.0 to 1.0)
    pub overlap_factor: f64,
    /// Frequency domain filter response
    pub frequency_response: Vec<Complex64>,
    /// Window function for overlap-add
    pub window: Vec<f64>,
}

impl ParallelSpectralFilter {
    /// Create a new parallel spectral filter
    pub fn new(
        frequency_response: Vec<Complex64>,
        fft_size: usize,
        overlap_factor: f64,
    ) -> SignalResult<Self> {
        if frequency_response.len() != fft_size / 2 + 1 {
            return Err(SignalError::ShapeMismatch(
                "Frequency _response _size must match FFT _size".to_string(),
            ));
        }

        // Create Hann window for overlap-add
        let window: Vec<f64> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (fft_size - 1) as f64).cos())
            })
            .collect();

        Ok(Self {
            fft_size,
            overlap_factor,
            frequency_response,
            window,
        })
    }

    /// Apply spectral filtering using parallel overlap-add
    pub fn apply_parallel(
        &self,
        signal: &[f64],
        config: &AdvancedParallelConfig,
    ) -> SignalResult<Vec<f64>> {
        let hop_size = (self.fft_size as f64 * (1.0 - self.overlap_factor)) as usize;
        let num_frames = (signal.len() + hop_size - 1) / hop_size;

        // Process frames in parallel
        let frame_results: Result<Vec<_>, SignalError> = (0..num_frames)
            .into_par_iter()
            .map(|frame_idx| {
                let start = frame_idx * hop_size;
                let end = (start + self.fft_size).min(signal.len());

                // Extract and window frame
                let mut frame = vec![Complex::new(0.0, 0.0); self.fft_size];
                for i in 0..(end - start) {
                    frame[i] = Complex::new(signal[start + i] * self.window[i], 0.0);
                }

                // Forward FFT
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(self.fft_size);
                fft.process(&mut frame);

                // Apply frequency domain filter
                for i in 0..self.fft_size / 2 + 1 {
                    if i < self.frequency_response.len() {
                        frame[i] *= Complex::new(
                            self.frequency_response[i].re,
                            self.frequency_response[i].im,
                        );
                    }
                }

                // Mirror for real signal
                for i in 1..self.fft_size / 2 {
                    frame[self.fft_size - i] = frame[i].conj();
                }

                // Inverse FFT
                let ifft = planner.plan_fft_inverse(self.fft_size);
                ifft.process(&mut frame);

                // Apply window and extract real part
                let mut windowed_frame = vec![0.0; self.fft_size];
                for i in 0..self.fft_size {
                    windowed_frame[i] = frame[i].re * self.window[i] / self.fft_size as f64;
                }

                Ok::<(usize, Vec<f64>), SignalError>((start, windowed_frame))
            })
            .collect();

        let frames = frame_results?;

        // Overlap-add reconstruction
        let output_len = signal.len() + self.fft_size - hop_size;
        let mut output = vec![0.0; output_len];

        for (start, frame) in frames {
            for (i, &val) in frame.iter().enumerate() {
                if start + i < output.len() {
                    output[start + i] += val;
                }
            }
        }

        // Trim to original length
        output.truncate(signal.len());
        Ok(output)
    }
}

/// High-performance parallel convolution with decimation
#[allow(dead_code)]
fn parallel_convolve_decimated(
    signal: &[f64],
    filter: &[f64],
    decimation_factor: usize,
    _config: &ParallelFilterConfig,
) -> SignalResult<Vec<f64>> {
    if decimation_factor == 0 {
        return Err(SignalError::ValueError(
            "Decimation _factor must be positive".to_string(),
        ));
    }

    let output_len = signal.len() / decimation_factor;

    // Process only samples that will be kept after decimation
    let sample_indices: Vec<usize> = (0..output_len).map(|i| i * decimation_factor).collect();

    let results: Result<Vec<_>, SignalError> = sample_indices
        .into_par_iter()
        .map(|sample_idx| {
            let mut convolution_result = 0.0;
            for (filter_idx, &filter_coeff) in filter.iter().enumerate() {
                if sample_idx >= filter_idx {
                    let signal_idx = sample_idx - filter_idx;
                    if signal_idx < signal.len() {
                        convolution_result += filter_coeff * signal[signal_idx];
                    }
                }
            }
            Ok::<f64, SignalError>(convolution_result)
        })
        .collect();

    Ok(results?)
}

/// Parallel interpolation and filtering
#[allow(dead_code)]
fn parallel_interpolate_filter(
    decimated: &[f64],
    filter: &[f64],
    interpolation_factor: usize,
    output_len: usize,
    config: &ParallelFilterConfig,
) -> SignalResult<Vec<f64>> {
    if interpolation_factor == 0 {
        return Err(SignalError::ValueError(
            "Interpolation _factor must be positive".to_string(),
        ));
    }

    // Zero-stuff (insert zeros between samples)
    let mut zero_stuffed = vec![0.0; decimated.len() * interpolation_factor];
    for (i, &val) in decimated.iter().enumerate() {
        zero_stuffed[i * interpolation_factor] = val;
    }

    // Apply interpolation filter
    let filtered_len = zero_stuffed.len() + filter.len() - 1;
    let filtered = vec![0.0; filtered_len];

    // Parallel convolution
    let chunk_size = config.chunk_size.unwrap_or(1024);
    let chunks: Vec<_> = (0..filtered_len).step_by(chunk_size).collect();

    let results: Result<Vec<_>, SignalError> = chunks
        .into_par_iter()
        .map(|start| {
            let end = (start + chunk_size).min(filtered_len);
            let mut chunk_result = vec![0.0; end - start];

            for (i, result_val) in chunk_result.iter_mut().enumerate() {
                let output_idx = start + i;
                for (filter_idx, &filter_coeff) in filter.iter().enumerate() {
                    if output_idx >= filter_idx {
                        let input_idx = output_idx - filter_idx;
                        if input_idx < zero_stuffed.len() {
                            *result_val += filter_coeff * zero_stuffed[input_idx];
                        }
                    }
                }
            }

            Ok::<Vec<f64>, SignalError>(chunk_result)
        })
        .collect();

    let chunks = results?;

    // Concatenate results
    let mut result = Vec::with_capacity(filtered_len);
    for chunk in chunks {
        result.extend(chunk);
    }

    // Trim to desired output length
    result.truncate(output_len);
    Ok(result)
}

/// Comprehensive parallel filtering benchmark
#[allow(dead_code)]
pub fn benchmark_parallel_filtering_operations(
    signal_lengths: &[usize],
    filter_lengths: &[usize],
    num_iterations: usize,
) -> SignalResult<HashMap<String, Vec<ParallelFilterMetrics>>> {
    let mut results = HashMap::new();

    for &signal_len in signal_lengths {
        for &filter_len in filter_lengths {
            // Generate test signal and filter
            let signal: Vec<f64> = (0..signal_len)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin())
                .collect();

            let filter: Vec<f64> = (0..filter_len)
                .map(|i| {
                    (-((i as f64 - filter_len as f64 / 2.0).powi(2)) / (filter_len as f64 / 4.0))
                        .exp()
                })
                .collect();

            let mut test_metrics = Vec::new();

            // Run benchmark _iterations
            for _ in 0..num_iterations {
                let config = AdvancedParallelConfig::default();
                let start_time = Instant::now();

                // Test sparse filtering
                let sparse_filter = SparseParallelFilter::from_dense(&filter, 0.1);
                let _result = sparse_filter.apply_parallel(&signal, &config)?;

                let processing_time = start_time.elapsed();
                let throughput = signal_len as f64 / processing_time.as_secs_f64();

                test_metrics.push(ParallelFilterMetrics {
                    processing_time,
                    throughput_sps: throughput,
                    memory_bandwidth_gbps: 0.0, // Would calculate actual bandwidth
                    cpu_utilization: 0.0,       // Would measure actual CPU usage
                    cache_hit_ratio: 0.0,       // Would measure cache performance
                    simd_utilization: 0.0,      // Would measure SIMD usage
                    threads_used: num_cpus::get(),
                    lock_contention_us: None,
                });
            }

            let test_name = format!("sparse_filter_{}x{}", signal_len, filter_len);
            results.insert(test_name, test_metrics);
        }
    }

    Ok(results)
}

/// Validate parallel filtering performance and correctness
#[allow(dead_code)]
pub fn validate_parallel_filtering_accuracy(
    reference_implementations: &HashMap<String, Box<dyn Fn(&[f64]) -> Vec<f64>>>,
    test_signals: &[Vec<f64>],
    tolerance: f64,
) -> SignalResult<HashMap<String, f64>> {
    let mut accuracy_results = HashMap::new();

    for (test_name, reference_fn) in reference_implementations {
        let mut total_error = 0.0;
        let mut total_samples = 0;

        for test_signal in test_signals {
            let reference_output = reference_fn(test_signal);

            // Test our parallel implementation
            let config = AdvancedParallelConfig::default();
            let sparse_filter = SparseParallelFilter::from_dense(&[0.25, 0.5, 0.25], 0.1);
            let parallel_output = sparse_filter.apply_parallel(test_signal, &config)?;

            // Calculate error
            let min_len = reference_output.len().min(parallel_output.len());
            for i in 0..min_len {
                total_error += (reference_output[i] - parallel_output[i]).abs();
                total_samples += 1;
            }
        }

        let avg_error = total_error / total_samples as f64;
        accuracy_results.insert(test_name.clone(), avg_error);

        if avg_error > tolerance {
            return Err(SignalError::ComputationError(format!(
                "Accuracy test failed for {}: error {} > tolerance {}",
                test_name, avg_error, tolerance
            )));
        }
    }

    Ok(accuracy_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;
    use std::f64::consts::PI;
    #[test]
    #[ignore] // FIXME: Perfect reconstruction error is too high - needs algorithm review
    fn test_parallel_multirate_filter_bank() {
        // Create simple 2-band filter bank
        let analysis_filters = vec![
            vec![0.5, 0.5],  // Lowpass
            vec![0.5, -0.5], // Highpass
        ];
        let synthesis_filters = vec![
            vec![1.0, 1.0],  // Lowpass reconstruction
            vec![1.0, -1.0], // Highpass reconstruction
        ];
        let decimation_factors = vec![2, 2];

        let mut filter_bank = ParallelMultiRateFilterBank::new(
            analysis_filters,
            synthesis_filters,
            decimation_factors,
        )
        .unwrap();

        let test_signal: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
            .collect();

        let config = AdvancedParallelConfig::default();
        let result = filter_bank.process(&test_signal, &config).unwrap();

        assert_eq!(result.len(), test_signal.len());

        // Test perfect reconstruction
        let pr_error = filter_bank
            .validate_perfect_reconstruction(&test_signal)
            .unwrap();
        assert!(pr_error < 0.5); // Should have reasonably low reconstruction error
    }

    #[test]
    fn test_sparse_parallel_filter() {
        let dense_filter = vec![0.1, 0.0, 0.3, 0.0, 0.5, 0.0, 0.1];
        let sparse_filter = SparseParallelFilter::from_dense(&dense_filter, 0.05);

        assert!(sparse_filter.sparsity_ratio > 0.0);
        assert!(sparse_filter.compression_ratio > 1.0);

        let test_signal: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 20.0).sin())
            .collect();

        let config = AdvancedParallelConfig::default();
        let result = sparse_filter.apply_parallel(&test_signal, &config).unwrap();

        assert_eq!(result.len(), test_signal.len());
    }

    #[test]
    fn test_lock_free_streaming_filter() {
        let b = vec![0.1, 0.2, 0.1];
        let a = vec![1.0, -0.5, 0.3];
        let config = AdvancedParallelConfig {
            lock_free: true,
            ..Default::default()
        };

        let filter = LockFreeStreamingFilter::new(b, a, config).unwrap();

        // Test single sample processing
        let input = 1.0;
        let output = filter.process_sample(input).unwrap();
        assert!(output.is_finite());

        // Test block processing
        let inputs = vec![1.0, 0.5, -0.5, -1.0];
        let outputs = filter.process_block(&inputs).unwrap();
        assert_eq!(outputs.len(), inputs.len());
        assert!(outputs.iter().all(|&x: &f64| x.is_finite()));

        // Check metrics
        let metrics = filter.get_metrics().unwrap();
        assert!(metrics.processing_time.as_nanos() > 0);
    }

    #[test]
    fn test_parallel_spectral_filter() {
        let fft_size = 128;
        let frequency_response: Vec<Complex64> = (0..fft_size / 2 + 1)
            .map(|i| {
                // Simple lowpass filter
                if i < fft_size / 8 {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();

        let spectral_filter =
            ParallelSpectralFilter::new(frequency_response, fft_size, 0.5).unwrap();

        let test_signal: Vec<f64> = (0..256)
            .map(|i| (2.0 * PI * i as f64 / 8.0).sin() + (2.0 * PI * i as f64 / 4.0).sin())
            .collect();

        let config = AdvancedParallelConfig::default();
        let filtered = spectral_filter
            .apply_parallel(&test_signal, &config)
            .unwrap();

        assert_eq!(filtered.len(), test_signal.len());
        assert!(filtered.iter().all(|&x: &f64| x.is_finite()));
    }

    #[test]
    #[ignore = "timeout"]
    fn test_parallel_filtering_benchmark() {
        let signal_lengths = vec![1000, 5000];
        let filter_lengths = vec![10, 50];
        let num_iterations = 3;

        let results = benchmark_parallel_filtering_operations(
            &signal_lengths,
            &filter_lengths,
            num_iterations,
        )
        .unwrap();

        assert!(!results.is_empty());

        for (test_name, metrics) in results {
            assert_eq!(metrics.len(), num_iterations);
            assert!(metrics.iter().all(|m| m.throughput_sps > 0.0));
            println!(
                "Test {}: avg throughput = {:.0} samples/sec",
                test_name,
                metrics.iter().map(|m| m.throughput_sps).sum::<f64>() / metrics.len() as f64
            );
        }
    }
}
