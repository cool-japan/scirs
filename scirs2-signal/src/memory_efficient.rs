// Memory-efficient signal processing algorithms
//
// This module provides memory-optimized implementations for processing
// large signals that may not fit entirely in memory, using streaming
// algorithms, chunked processing, and advanced memory management.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::{check_finite, check_positive};
use std::collections::{HashMap, VecDeque};

#[allow(unused_imports)]
/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Chunk size for streaming algorithms
    pub chunk_size: usize,
    /// Overlap between chunks (for windowed operations)
    pub overlap: usize,
    /// Use memory mapping for large files
    pub use_memory_mapping: bool,
    /// Enable garbage collection hints
    pub enable_gc_hints: bool,
    /// Cache size for intermediate results
    pub cache_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB default
            chunk_size: 8192,
            overlap: 512,
            use_memory_mapping: false,
            enable_gc_hints: true,
            cache_size: 64 * 1024, // 64KB cache
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Number of allocations
    pub allocations: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
}

/// Streaming signal processor for memory-efficient processing
pub struct StreamingProcessor {
    config: MemoryConfig,
    buffer: VecDeque<f64>,
    output_buffer: Vec<f64>,
    stats: MemoryStats,
}

impl StreamingProcessor {
    /// Create a new streaming processor
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            buffer: VecDeque::with_capacity(_config.chunk_size),
            output_buffer: Vec::with_capacity(_config.chunk_size),
            config,
            stats: MemoryStats {
                current_usage: 0,
                peak_usage: 0,
                allocations: 0,
                cache_hits: 0,
                cache_misses: 0,
            },
        }
    }

    /// Process a chunk of data
    pub fn process_chunk<F>(&mut self, chunk: &[f64], processor: F) -> SignalResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> SignalResult<Vec<f64>>,
    {
        // Add to buffer
        self.buffer.extend(chunk.iter().copied());
        self.stats.current_usage += chunk.len() * std::mem::size_of::<f64>();
        self.stats.peak_usage = self.stats.peak_usage.max(self.stats.current_usage);

        // Check memory limit
        if self.stats.current_usage > self.config.max_memory_bytes {
            // Force processing of buffered data
            self.flush_buffer(&processor)?;
        }

        // Process if buffer is full
        if self.buffer.len() >= self.config.chunk_size {
            let data: Vec<f64> = self.buffer.drain(..self.config.chunk_size).collect();
            let result = processor(&data)?;
            self.output_buffer.extend(result);

            // Update memory usage
            self.stats.current_usage = self.buffer.len() * std::mem::size_of::<f64>();
        }

        Ok(self.output_buffer.clone())
    }

    /// Flush remaining data in buffer
    pub fn flush_buffer<F>(&mut self, processor: &F) -> SignalResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> SignalResult<Vec<f64>>,
    {
        if !self.buffer.is_empty() {
            let data: Vec<f64> = self.buffer.drain(..).collect();
            let result = processor(&data)?;
            self.output_buffer.extend(result);
            self.stats.current_usage = 0;
        }
        Ok(self.output_buffer.clone())
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.output_buffer.clear();
        self.stats.current_usage = 0;
    }
}

/// Memory-efficient FFT processing with chunked operations
///
/// # Arguments
///
/// * `signal` - Input signal (can be very large)
/// * `chunk_size` - Size of each FFT chunk
/// * `overlap` - Overlap between chunks
/// * `config` - Memory configuration
///
/// # Returns
///
/// * Streaming FFT results
#[allow(dead_code)]
pub fn memory_efficient_fft<I>(
    signal_iter: I,
    chunk_size: usize,
    overlap: usize,
    config: &MemoryConfig,
) -> SignalResult<StreamingFFTResult>
where
    I: Iterator<Item = f64>,
{
    check_positive(chunk_size, "chunk_size")?;

    if overlap >= chunk_size {
        return Err(SignalError::ValueError(
            "Overlap must be less than chunk _size".to_string(),
        ));
    }

    let processor = StreamingProcessor::new(config.clone());
    let mut fft_results = Vec::new();
    let mut buffer = Vec::with_capacity(chunk_size);

    // Process signal in chunks
    for sample in signal_iter {
        buffer.push(sample);

        if buffer.len() >= chunk_size {
            // Compute FFT for this chunk
            let fft_chunk = compute_fft_chunk(&buffer, chunk_size)?;
            fft_results.push(fft_chunk);

            // Slide window with overlap
            let step = chunk_size - overlap;
            buffer.drain(..step);
        }
    }

    // Process remaining data
    if !buffer.is_empty() {
        let fft_chunk = compute_fft_chunk(&buffer, buffer.len())?;
        fft_results.push(fft_chunk);
    }

    Ok(StreamingFFTResult {
        chunks: fft_results,
        chunk_size,
        overlap,
        memory_stats: processor.get_stats().clone(),
    })
}

/// Result of streaming FFT processing
#[derive(Debug, Clone)]
pub struct StreamingFFTResult {
    /// FFT chunks
    pub chunks: Vec<Vec<f64>>,
    /// Size of each chunk
    pub chunk_size: usize,
    /// Overlap used
    pub overlap: usize,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

/// Compute FFT for a single chunk
#[allow(dead_code)]
fn compute_fft_chunk(_data: &[f64], _chunksize: usize) -> SignalResult<Vec<f64>> {
    // Convert to ndarray and compute FFT
    let signal = Array1::from(_data.to_vec());

    // Use a simple DFT implementation for this example
    let n = signal.len();
    let mut result = vec![0.0; n];

    for k in 0..n {
        let mut real = 0.0;
        let mut imag = 0.0;

        for j in 0..n {
            let angle = -2.0 * PI * (k * j) as f64 / n as f64;
            real += signal[j] * angle.cos();
            imag += signal[j] * angle.sin();
        }

        result[k] = (real * real + imag * imag).sqrt();
    }

    Ok(result)
}

/// Memory-efficient filtering with overlap-save method
///
/// # Arguments
///
/// * `signal_iter` - Iterator over input signal
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `config` - Memory configuration
///
/// # Returns
///
/// * Filtered signal chunks
#[allow(dead_code)]
pub fn memory_efficient_filter<I>(
    mut signal_iter: I,
    b: &[f64],
    a: &[f64],
    config: &MemoryConfig,
) -> SignalResult<StreamingFilterResult>
where
    I: Iterator<Item = f64>,
{
    if a.is_empty() || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "Invalid filter coefficients".to_string(),
        ));
    }

    let filter_order = a.len().max(b.len());
    let chunk_size = config.chunk_size;
    let overlap = filter_order * 2; // Ensure sufficient overlap for filter

    let processor = StreamingProcessor::new(config.clone());
    let mut filtered_chunks = Vec::new();
    let mut buffer = VecDeque::with_capacity(chunk_size + overlap);
    let mut filter_state = vec![0.0; filter_order];

    // Process signal in chunks with overlap
    loop {
        // Fill buffer with new data
        let mut chunk_filled = false;
        while buffer.len() < chunk_size {
            if let Some(sample) = signal_iter.next() {
                buffer.push_back(sample);
            } else {
                chunk_filled = true;
                break;
            }
        }

        if buffer.is_empty() {
            break;
        }

        // Extract chunk for processing
        let process_size = if chunk_filled {
            buffer.len()
        } else {
            chunk_size
        };
        let chunk_data: Vec<f64> = buffer._iter().take(process_size).copied().collect();

        // Apply filter to chunk
        let filtered_chunk = apply_filter_with_state(&chunk_data, b, a, &mut filter_state)?;
        filtered_chunks.push(filtered_chunk);

        // Remove processed data (keep overlap)
        if !chunk_filled {
            let remove_count = process_size - overlap;
            for _ in 0..remove_count {
                buffer.pop_front();
            }
        } else {
            buffer.clear();
        }
    }

    Ok(StreamingFilterResult {
        chunks: filtered_chunks,
        chunk_size,
        overlap,
        memory_stats: processor.get_stats().clone(),
    })
}

/// Result of streaming filter processing
#[derive(Debug, Clone)]
pub struct StreamingFilterResult {
    /// Filtered chunks
    pub chunks: Vec<Vec<f64>>,
    /// Size of each chunk
    pub chunk_size: usize,
    /// Overlap used
    pub overlap: usize,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

/// Apply filter with state preservation
#[allow(dead_code)]
fn apply_filter_with_state(
    input: &[f64],
    b: &[f64],
    a: &[f64],
    state: &mut [f64],
) -> SignalResult<Vec<f64>> {
    check_finite(input, "input value")?;
    check_finite(b, "b value")?;
    check_finite(a, "a value")?;

    let n = input.len();
    let mut output = vec![0.0; n];

    // Simple IIR filter implementation with state
    for i in 0..n {
        let mut y = 0.0;

        // FIR part
        for j in 0..b.len() {
            if i >= j {
                y += b[j] * input[i - j];
            } else if j - i - 1 < state.len() {
                y += b[j] * state[state.len() - (j - i)];
            }
        }

        // IIR part
        for j in 1..a.len() {
            if i >= j {
                y -= a[j] * output[i - j];
            } else if j - i - 1 < state.len() {
                y -= a[j] * state[state.len() - (j - i)];
            }
        }

        y /= a[0];
        output[i] = y;
    }

    // Update state with latest values
    if n >= state.len() {
        state.copy_from_slice(&output[n - state.len()..]);
    } else {
        // Shift old state and add new values
        state.copy_within(n.., 0);
        state[state.len() - n..].copy_from_slice(&output);
    }

    Ok(output)
}

/// Memory-efficient spectrogram computation
///
/// # Arguments
///
/// * `signal_iter` - Iterator over input signal
/// * `window_size` - Size of analysis window
/// * `hop_size` - Hop size between windows
/// * `config` - Memory configuration
///
/// # Returns
///
/// * Streaming spectrogram result
#[allow(dead_code)]
pub fn memory_efficient_spectrogram<I>(
    signal_iter: I,
    window_size: usize,
    hop_size: usize,
    config: &MemoryConfig,
) -> SignalResult<StreamingSpectrogramResult>
where
    I: Iterator<Item = f64>,
{
    check_positive(window_size, "window_size")?;
    check_positive(hop_size, "hop_size")?;

    let processor = StreamingProcessor::new(config.clone());
    let mut spectrogram_frames = Vec::new();
    let mut buffer = VecDeque::with_capacity(window_size * 2);

    // Apply window function (Hann window)
    let window: Vec<f64> = (0..window_size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (window_size - 1) as f64).cos()))
        .collect();

    // Process signal frame by frame
    for sample in signal_iter {
        buffer.push_back(sample);

        // Check if we have enough data for a frame
        while buffer.len() >= window_size {
            // Extract frame
            let frame_data: Vec<f64> = buffer._iter().take(window_size).copied().collect();

            // Apply window and compute spectrum
            let windowed: Vec<f64> = frame_data
                ._iter()
                .zip(window._iter())
                .map(|(x, w)| x * w)
                .collect();

            // Compute FFT magnitude
            let spectrum = compute_fft_chunk(&windowed, window_size)?;
            spectrogram_frames.push(spectrum);

            // Advance by hop _size
            for _ in 0..hop_size.min(buffer.len()) {
                buffer.pop_front();
            }
        }
    }

    Ok(StreamingSpectrogramResult {
        frames: spectrogram_frames,
        window_size,
        hop_size,
        frequencies: (0..window_size / 2 + 1)
            .map(|i| i as f64 / window_size as f64)
            .collect(),
        memory_stats: processor.get_stats().clone(),
    })
}

/// Result of streaming spectrogram computation
#[derive(Debug, Clone)]
pub struct StreamingSpectrogramResult {
    /// Spectrogram frames
    pub frames: Vec<Vec<f64>>,
    /// Window size used
    pub window_size: usize,
    /// Hop size used
    pub hop_size: usize,
    /// Frequency bins
    pub frequencies: Vec<f64>,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

/// Memory-efficient correlation computation using sliding window
///
/// # Arguments
///
/// * `signal1_iter` - Iterator over first signal
/// * `signal2_iter` - Iterator over second signal
/// * `max_lag` - Maximum lag to compute
/// * `config` - Memory configuration
///
/// # Returns
///
/// * Cross-correlation result
#[allow(dead_code)]
pub fn memory_efficient_correlation<I1, I2>(
    signal1_iter: I1,
    signal2_iter: I2,
    max_lag: usize,
    config: &MemoryConfig,
) -> SignalResult<StreamingCorrelationResult>
where
    I1: Iterator<Item = f64>,
    I2: Iterator<Item = f64>,
{
    check_positive(max_lag, "max_lag")?;

    let buffer_size = max_lag * 2;
    let mut buffer1 = VecDeque::with_capacity(buffer_size);
    let mut buffer2 = VecDeque::with_capacity(buffer_size);
    let mut correlation_values = vec![0.0; 2 * max_lag + 1];
    let mut sample_count = 0;

    let signal1_vec: Vec<f64> = signal1_iter.collect();
    let signal2_vec: Vec<f64> = signal2_iter.collect();

    if signal1_vec.len() != signal2_vec.len() {
        return Err(SignalError::ValueError(
            "Signals must have the same length".to_string(),
        ));
    }

    let n = signal1_vec.len();

    // Compute cross-correlation using FFT-based method for efficiency
    // For memory efficiency, we'll use a chunked approach
    let chunk_size = config.chunk_size.min(n);

    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n);
        let chunk1 = &signal1_vec[chunk_start..chunk_end];
        let chunk2 = &signal2_vec[chunk_start..chunk_end];

        // Compute correlation for this chunk
        for _lag in 0..=max_lag {
            let mut correlation = 0.0;
            let valid_length = chunk1.len().saturating_sub(_lag);

            for i in 0..valid_length {
                correlation += chunk1[i] * chunk2[i + _lag];
            }

            correlation_values[max_lag + _lag] += correlation;

            // Negative lags
            if _lag > 0 && _lag < chunk2.len() {
                let mut neg_correlation = 0.0;
                let valid_length = chunk2.len().saturating_sub(_lag);

                for i in 0..valid_length {
                    neg_correlation += chunk2[i] * chunk1[i + _lag];
                }

                correlation_values[max_lag - _lag] += neg_correlation;
            }
        }

        sample_count += chunk1.len();
    }

    // Normalize correlation values
    for val in &mut correlation_values {
        *val /= sample_count as f64;
    }

    let lags: Vec<i32> = (-(max_lag as i32)..=(max_lag as i32)).collect();

    Ok(StreamingCorrelationResult {
        correlation: correlation_values,
        lags,
        max_lag,
        memory_stats: MemoryStats {
            current_usage: 0,
            peak_usage: sample_count * std::mem::size_of::<f64>() * 2,
            allocations: 1,
            cache_hits: 0,
            cache_misses: 0,
        },
    })
}

/// Result of streaming correlation computation
#[derive(Debug, Clone)]
pub struct StreamingCorrelationResult {
    /// Cross-correlation values
    pub correlation: Vec<f64>,
    /// Lag values
    pub lags: Vec<i32>,
    /// Maximum lag computed
    pub max_lag: usize,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

/// Cache for intermediate results to reduce recomputation
pub struct MemoryCache<K, V> {
    cache: std::collections::HashMap<K, V>,
    max_size: usize,
    hits: usize,
    misses: usize,
}

impl<K, V> MemoryCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new cache with specified maximum size
    pub fn new(_maxsize: usize) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Get value from cache
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.cache.get(key) {
            self.hits += 1;
            Some(value.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert value into cache
    pub fn insert(&mut self, key: K, value: V) {
        if self.cache.len() >= self.max_size {
            // Simple LRU: remove first entry
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, value);
    }

    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize, f64) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        (self.hits, self.misses, hit_rate)
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_processor() {
        let config = MemoryConfig {
            chunk_size: 4,
            ..Default::default()
        };
        let mut processor = StreamingProcessor::new(config);

        let process_fn =
            |data: &[f64]| -> SignalResult<Vec<f64>> { Ok(data.iter().map(|x| x * 2.0).collect()) };

        let chunk1 = [1.0, 2.0];
        let chunk2 = [3.0, 4.0];
        let chunk3 = [5.0, 6.0];

        processor.process_chunk(&chunk1, &process_fn).unwrap();
        processor.process_chunk(&chunk2, &process_fn).unwrap();
        let result = processor.process_chunk(&chunk3, &process_fn).unwrap();

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]); // First chunk processed
    }

    #[test]
    fn test_memory_efficient_filter() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.5, 0.5];
        let a = vec![1.0];
        let config = MemoryConfig {
            chunk_size: 4,
            ..Default::default()
        };

        let result = memory_efficient_filter(signal.into_iter(), &b, &a, &config).unwrap();

        assert!(!result.chunks.is_empty());
        assert!(result.chunks.iter().all(|chunk| !chunk.is_empty()));
    }

    #[test]
    fn test_memory_efficient_spectrogram() {
        let signal: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
            .collect();

        let config = MemoryConfig::default();
        let result = memory_efficient_spectrogram(signal.into_iter(), 16, 8, &config).unwrap();

        assert!(!result.frames.is_empty());
        assert_eq!(result.window_size, 16);
        assert_eq!(result.hop_size, 8);
        assert_eq!(result.frequencies.len(), 9); // window_size/2 + 1
    }

    #[test]
    fn test_memory_cache() {
        let mut cache = MemoryCache::new(3);

        // Test misses
        assert!(cache.get(&"key1").is_none());
        assert!(cache.get(&"key2").is_none());

        // Test inserts and hits
        cache.insert("key1".to_string(), 100);
        cache.insert("key2".to_string(), 200);

        assert_eq!(cache.get(&"key1".to_string()), Some(100));
        assert_eq!(cache.get(&"key2".to_string()), Some(200));

        // Test LRU eviction
        cache.insert("key3".to_string(), 300);
        cache.insert("key4".to_string(), 400); // Should evict key1

        assert!(cache.get(&"key1".to_string()).is_none());
        assert_eq!(cache.get(&"key4".to_string()), Some(400));

        let (hits, misses, hit_rate) = cache.stats();
        assert!(hits > 0);
        assert!(misses > 0);
        assert!(hit_rate > 0.0 && hit_rate <= 1.0);
    }

    #[test]
    fn test_memory_efficient_correlation() {
        let signal1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let signal2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let config = MemoryConfig::default();

        let result =
            memory_efficient_correlation(signal1.into_iter(), signal2.into_iter(), 2, &config)
                .unwrap();

        assert_eq!(result.correlation.len(), 5); // 2*max_lag + 1
        assert_eq!(result.lags.len(), 5);
        assert_eq!(result.max_lag, 2);
    }
}
