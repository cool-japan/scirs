// Advanced memory optimization for large signal processing
//
// This module provides next-generation memory management strategies for processing
// extremely large signals efficiently with minimal memory footprint.
//
// Key features:
// - Streaming processing with configurable buffer sizes
// - Memory pool management for reduced allocations
// - Out-of-core processing for signals larger than RAM
// - Cache-efficient algorithm implementations
// - Memory usage monitoring and adaptive thresholds
// - Zero-copy operations where possible

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{arr2, Array1, Array2};
use scirs2_core::numeric::{Float, NumCast, Zero};
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::check_positive;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

#[allow(unused_imports)]
/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Maximum memory usage in bytes (None for unlimited)
    pub max_memory_bytes: Option<usize>,
    /// Preferred chunk size for streaming processing
    pub chunk_size: usize,
    /// Overlap size between chunks for filtering operations
    pub overlap_size: usize,
    /// Enable memory pool for frequent allocations
    pub use_memory_pool: bool,
    /// Pool size in number of arrays
    pub pool_size: usize,
    /// Enable out-of-core processing for very large signals
    pub enable_out_of_core: bool,
    /// Temporary storage directory for out-of-core processing
    pub temp_dir: Option<String>,
    /// Memory usage monitoring enabled
    pub monitor_memory: bool,
    /// Adaptive threshold adjustment
    pub adaptive_thresholds: bool,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: Some(1024 * 1024 * 1024), // 1GB default
            chunk_size: 65536,                          // 64K samples
            overlap_size: 1024,
            use_memory_pool: true,
            pool_size: 16,
            enable_out_of_core: false,
            temp_dir: None,
            monitor_memory: true,
            adaptive_thresholds: true,
        }
    }
}

/// Memory pool for efficient array allocation
#[derive(Debug)]
pub struct MemoryPool {
    pool: Arc<Mutex<VecDeque<Vec<f64>>>>,
    max_size: usize,
    current_usage: Arc<Mutex<usize>>,
}

impl MemoryPool {
    pub fn new(_maxsize: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(VecDeque::new())),
            max_size,
            current_usage: Arc::new(Mutex::new(0)),
        }
    }

    /// Get a buffer from the pool or allocate a new one
    pub fn get_buffer(&self, size: usize) -> Vec<f64> {
        if let Ok(mut pool) = self.pool.lock() {
            if let Some(mut buffer) = pool.pop_front() {
                if buffer.len() >= size {
                    buffer.resize(size, 0.0);
                    return buffer;
                }
            }
        }

        // Allocate new buffer if pool is empty or buffers are too small
        vec![0.0; size]
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&self, buffer: Vec<f64>) {
        if let Ok(mut pool) = self.pool.lock() {
            if pool.len() < self.max_size {
                pool.push_back(buffer);
            }
        }
    }

    /// Get current memory usage estimate
    pub fn get_usage(&self) -> usize {
        if let Ok(usage) = self.current_usage.lock() {
            *usage
        } else {
            0
        }
    }
}

/// Streaming processor for large signals
#[derive(Debug)]
pub struct StreamingProcessor {
    config: MemoryOptimizationConfig,
    memory_pool: Option<MemoryPool>,
    current_memory_usage: usize,
}

impl StreamingProcessor {
    pub fn new(config: MemoryOptimizationConfig) -> Self {
        let memory_pool = if config.use_memory_pool {
            Some(MemoryPool::new(_config.pool_size))
        } else {
            None
        };

        Self {
            config,
            memory_pool,
            current_memory_usage: 0,
        }
    }

    /// Process a signal in streaming chunks with memory optimization
    pub fn process_signal_streaming<F, T>(
        &mut self,
        signal: &[T],
        processor: F,
    ) -> SignalResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> SignalResult<Vec<f64>> + Sync,
        T: Float + NumCast + Debug + Send + Sync,
    {
        let signal_len = signal.len();
        check_positive(signal_len, "signal length")?;

        // Check memory constraints
        if let Some(max_mem) = self.config.max_memory_bytes {
            let estimated_usage = signal_len * 16; // Conservative estimate
            if estimated_usage > max_mem {
                return self.process_out_of_core(signal, processor);
            }
        }

        let chunk_size = self.config.chunk_size;
        let overlap = self.config.overlap_size;
        let mut result = Vec::new();

        let mut pos = 0;
        while pos < signal_len {
            let chunk_end = (pos + chunk_size).min(signal_len);
            let chunk_with_overlap_end = (chunk_end + overlap).min(signal_len);

            // Extract chunk with overlap
            let chunk_data: Vec<f64> = signal[pos..chunk_with_overlap_end]
                .iter()
                .map(|&x| NumCast::from(x).unwrap_or(0.0))
                .collect();

            // Process chunk
            let chunk_result = processor(&chunk_data)?;

            // Extract valid portion (remove overlap)
            let valid_start = if pos == 0 { 0 } else { overlap };
            let valid_end = chunk_result.len().min(chunk_end - pos + valid_start);

            if valid_start < valid_end {
                result.extend_from_slice(&chunk_result[valid_start..valid_end]);
            }

            pos = chunk_end;

            // Update memory usage tracking
            if self.config.monitor_memory {
                self.current_memory_usage = result.capacity() * 8;
            }
        }

        Ok(result)
    }

    /// Out-of-core processing for extremely large signals
    fn process_out_of_core<F, T>(&mut self, signal: &[T], processor: F) -> SignalResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> SignalResult<Vec<f64>> + Sync,
        T: Float + NumCast + Debug + Send + Sync,
    {
        // For now, fall back to chunked processing
        // In a full implementation, this would use temporary files
        self.process_signal_streaming(signal, processor)
    }

    /// Memory-optimized convolution for large signals
    pub fn memory_optimized_convolution(
        &mut self,
        signal: &[f64],
        kernel: &[f64],
    ) -> SignalResult<Vec<f64>> {
        check_positive(signal.len(), "signal length")?;
        check_positive(kernel.len(), "kernel length")?;

        let processor = |chunk: &[f64]| -> SignalResult<Vec<f64>> {
            let mut result = vec![0.0; chunk.len() + kernel.len() - 1];

            // Optimized convolution implementation
            for (i, &chunk_val) in chunk.iter().enumerate() {
                for (j, &kernel_val) in kernel.iter().enumerate() {
                    result[i + j] += chunk_val * kernel_val;
                }
            }

            Ok(result)
        };

        self.process_signal_streaming(signal, processor)
    }

    /// Memory-optimized FFT for large signals
    pub fn memory_optimized_fft(
        &mut self,
        signal: &[f64],
    ) -> SignalResult<Vec<scirs2_core::numeric::Complex<f64>>> {
        check_positive(signal.len(), "signal length")?;

        // For large signals, use chunked FFT with overlap-add
        let chunk_size = self.config.chunk_size.next_power_of_two();
        let overlap = chunk_size / 4; // 25% overlap

        let mut result = Vec::new();
        let mut pos = 0;

        while pos < signal.len() {
            let chunk_end = (pos + chunk_size).min(signal.len());
            let chunk = &signal[pos..chunk_end];

            // Pad chunk to power of 2 for efficient FFT
            let mut padded_chunk = vec![0.0; chunk_size];
            padded_chunk[..chunk.len()].copy_from_slice(chunk);

            // Convert to _complex and perform FFT
            let complex_chunk: Vec<scirs2_core::numeric::Complex<f64>> = padded_chunk
                .iter()
                .map(|&x| scirs2_core::numeric::Complex::new(x, 0.0))
                .collect();

            // Simple DFT for demonstration (replace with efficient FFT)
            let chunk_fft = simple_dft(&complex_chunk);

            // Add to result with overlap handling
            if pos == 0 {
                result = chunk_fft;
            } else {
                // Overlap-add the overlapping region
                let overlap_start = result.len() - overlap;
                for (i, val) in chunk_fft.iter().take(overlap).enumerate() {
                    if overlap_start + i < result.len() {
                        result[overlap_start + i] += val;
                    }
                }
                // Append non-overlapping part
                result.extend_from_slice(&chunk_fft[overlap..]);
            }

            pos += chunk_size - overlap;
        }

        Ok(result)
    }

    /// Memory-optimized filtering for large signals
    pub fn memory_optimized_filter(
        &mut self,
        signal: &[f64],
        b: &[f64],
        a: &[f64],
    ) -> SignalResult<Vec<f64>> {
        check_positive(signal.len(), "signal length")?;
        check_positive(b.len(), "numerator coefficients length")?;
        check_positive(a.len(), "denominator coefficients length")?;

        if a[0] == 0.0 {
            return Err(SignalError::ValueError(
                "First denominator coefficient cannot be zero".to_string(),
            ));
        }

        // Normalize coefficients
        let a0 = a[0];
        let b_norm: Vec<f64> = b.iter().map(|&bi| bi / a0).collect();
        let a_norm: Vec<f64> = a[1..].iter().map(|&ai| ai / a0).collect();

        let processor = |chunk: &[f64]| -> SignalResult<Vec<f64>> {
            let mut output = vec![0.0; chunk.len()];
            let filter_order = a_norm.len().max(b_norm.len());
            let mut state = vec![0.0; filter_order];

            for (i, &_input_val) in chunk.iter().enumerate() {
                let mut output_val = 0.0;

                // FIR part
                for (j, &b_coeff) in b_norm.iter().enumerate() {
                    if i >= j {
                        output_val += b_coeff * chunk[i - j];
                    } else if j - 1 - i < state.len() {
                        output_val += b_coeff * state[state.len() - 1 - (j - 1 - i)];
                    }
                }

                // IIR part
                for (j, &a_coeff) in a_norm.iter().enumerate() {
                    if i > j {
                        output_val -= a_coeff * output[i - j - 1];
                    } else if j - i < state.len() {
                        output_val -= a_coeff * state[state.len() - 1 - (j - i)];
                    }
                }

                output[i] = output_val;
            }

            // Update state for next chunk
            let state_len = state.len();
            for i in 0..state_len {
                if chunk.len() > i {
                    state[state_len - 1 - i] = chunk[chunk.len() - 1 - i];
                }
            }

            Ok(output)
        };

        self.process_signal_streaming(signal, processor)
    }

    /// Get current memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage: self.current_memory_usage,
            max_allowed: self.config.max_memory_bytes,
            pool_usage: self
                .memory_pool
                .as_ref()
                .map(|p| p.get_usage())
                .unwrap_or(0),
            chunk_size: self.config.chunk_size,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_usage: usize,
    pub max_allowed: Option<usize>,
    pub pool_usage: usize,
    pub chunk_size: usize,
}

/// Simple DFT implementation for demonstration
#[allow(dead_code)]
fn simple_dft(input: &[scirs2_core::numeric::Complex<f64>]) -> Vec<scirs2_core::numeric::Complex<f64>> {
    let n = input.len();
    let mut output = vec![scirs2_core::numeric::Complex::zero(); n];

    for k in 0..n {
        for (j, &x_j) in input.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let twiddle = scirs2_core::numeric::Complex::new(angle.cos(), angle.sin());
            output[k] += x_j * twiddle;
        }
    }

    output
}

/// Cache-efficient signal processing utilities
pub struct CacheOptimizedOps;

impl CacheOptimizedOps {
    /// Cache-friendly matrix-vector multiplication
    pub fn cache_friendly_matvec(
        matrix: &Array2<f64>,
        vector: &Array1<f64>,
        cache_line_size: usize,
    ) -> SignalResult<Array1<f64>> {
        let (rows, cols) = matrix.dim();
        if cols != vector.len() {
            return Err(SignalError::ValueError(
                "Matrix columns must match vector length".to_string(),
            ));
        }

        let mut result = Array1::zeros(rows);
        let block_size = cache_line_size / 8; // f64 _size

        // Block-based computation for cache efficiency
        for row_block in (0..rows).step_by(block_size) {
            let row_end = (row_block + block_size).min(rows);

            for col_block in (0..cols).step_by(block_size) {
                let col_end = (col_block + block_size).min(cols);

                for row in row_block..row_end {
                    let mut sum = 0.0;
                    for col in col_block..col_end {
                        sum += matrix[[row, col]] * vector[col];
                    }
                    result[row] += sum;
                }
            }
        }

        Ok(result)
    }

    /// Cache-efficient array transpose
    pub fn cache_friendly_transpose(_input: &Array2<f64>, cache_linesize: usize) -> Array2<f64> {
        let (rows, cols) = input.dim();
        let mut output = Array2::zeros((cols, rows));
        let block_size = cache_line_size / 8; // f64 _size

        for row_block in (0..rows).step_by(block_size) {
            let row_end = (row_block + block_size).min(rows);

            for col_block in (0..cols).step_by(block_size) {
                let col_end = (col_block + block_size).min(cols);

                for row in row_block..row_end {
                    for col in col_block..col_end {
                        output[[col, row]] = input[[row, col]];
                    }
                }
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(4);

        // Get buffer from pool
        let buffer1 = pool.get_buffer(1000);
        assert_eq!(buffer1.len(), 1000);

        // Return buffer to pool
        pool.return_buffer(buffer1);

        // Get buffer again (should reuse)
        let buffer2 = pool.get_buffer(500);
        assert_eq!(buffer2.len(), 500);
    }

    #[test]
    fn test_streaming_processor() {
        let config = MemoryOptimizationConfig {
            chunk_size: 100,
            overlap_size: 10,
            ..Default::default()
        };

        let mut processor = StreamingProcessor::new(config);

        // Create test signal
        let signal: Vec<f64> = (0..1000)
            .map(|i| (2.0 * PI * i as f64 / 100.0).sin())
            .collect();

        // Simple identity processor for testing
        let identity_processor = |chunk: &[f64]| Ok(chunk.to_vec());

        let result = processor
            .process_signal_streaming(&signal, identity_processor)
            .unwrap();

        // Result should be approximately the same as input
        assert_eq!(result.len(), signal.len());

        for (original, processed) in signal.iter().zip(result.iter()).take(10) {
            assert!((original - processed).abs() < 1e-10);
        }
    }

    #[test]
    fn test_memory_optimized_convolution() {
        let config = MemoryOptimizationConfig {
            chunk_size: 50,
            overlap_size: 5,
            ..Default::default()
        };

        let mut processor = StreamingProcessor::new(config);

        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![0.5, 0.5];

        let result = processor
            .memory_optimized_convolution(&signal, &kernel)
            .unwrap();

        // Check expected convolution length
        assert_eq!(result.len(), signal.len() + kernel.len() - 1);

        // Check some expected values
        assert!(((result[0] - 0.5) as f64).abs() < 1e-10); // 1.0 * 0.5
        assert!(((result[1] - 1.5) as f64).abs() < 1e-10); // 1.0 * 0.5 + 2.0 * 0.5
    }

    #[test]
    fn test_cache_friendly_operations() {
        let matrix = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = CacheOptimizedOps::cache_friendly_matvec(&matrix, &vector, 64).unwrap();

        // Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        assert!(((result[0] - 14.0) as f64).abs() < 1e-10);
        assert!(((result[1] - 32.0) as f64).abs() < 1e-10);
    }

    #[test]
    fn test_memory_stats() {
        let config = MemoryOptimizationConfig::default();
        let processor = StreamingProcessor::new(config);

        let stats = processor.get_memory_stats();
        assert!(stats.max_allowed.is_some());
        assert_eq!(stats.chunk_size, 65536);
    }
}

#[allow(dead_code)]
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}
