//! Real-time signal processing pipelines with zero-latency streaming
//!
//! This module provides infrastructure for real-time signal processing with minimal latency.
//! It includes streaming data structures, zero-copy operations, and efficient buffering
//! mechanisms for continuous signal processing applications.

use crate::error::{SignalError, SignalResult};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for real-time processing
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Buffer size for streaming data
    pub buffer_size: usize,
    /// Maximum latency tolerance in milliseconds
    pub max_latency_ms: f64,
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Number of processing threads
    pub num_threads: usize,
    /// Enable zero-copy optimizations
    pub zero_copy: bool,
    /// Overlap between processing blocks
    pub overlap_samples: usize,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            max_latency_ms: 10.0,
            sample_rate: 44100.0,
            num_threads: 1,
            zero_copy: true,
            overlap_samples: 0,
        }
    }
}

/// Real-time signal processing statistics
#[derive(Debug, Clone, Default)]
pub struct RealtimeStats {
    /// Total samples processed
    pub samples_processed: u64,
    /// Number of processing blocks
    pub blocks_processed: u64,
    /// Average processing time per block (ms)
    pub avg_processing_time_ms: f64,
    /// Maximum processing time per block (ms)
    pub max_processing_time_ms: f64,
    /// Current latency (ms)
    pub current_latency_ms: f64,
    /// Number of dropped samples due to overruns
    pub dropped_samples: u64,
    /// Number of underruns (output buffer empty)
    pub underruns: u64,
}

/// Streaming data block
#[derive(Debug, Clone)]
pub struct StreamBlock {
    /// Sample data
    pub data: Vec<f64>,
    /// Timestamp of first sample
    pub timestamp: Instant,
    /// Block sequence number
    pub sequence: u64,
    /// Number of channels
    pub channels: usize,
}

impl StreamBlock {
    /// Create a new stream block
    pub fn new(data: Vec<f64>, channels: usize, sequence: u64) -> Self {
        Self {
            data,
            timestamp: Instant::now(),
            sequence,
            channels,
        }
    }

    /// Get number of samples per channel
    pub fn samples_per_channel(&self) -> usize {
        self.data.len() / self.channels
    }

    /// Extract channel data
    pub fn channel_data(&self, channel: usize) -> SignalResult<Vec<f64>> {
        if channel >= self.channels {
            return Err(SignalError::ValueError(format!(
                "Channel {} out of range (0-{})",
                channel,
                self.channels - 1
            )));
        }

        let mut channel_data = Vec::with_capacity(self.samples_per_channel());
        for i in (channel..self.data.len()).step_by(self.channels) {
            channel_data.push(self.data[i]);
        }

        Ok(channel_data)
    }
}

/// Trait for real-time signal processors
pub trait RealtimeProcessor: Send + Sync {
    /// Process a stream block in-place
    fn process_block(&mut self, block: &mut StreamBlock) -> SignalResult<()>;

    /// Get processing latency in samples
    fn latency_samples(&self) -> usize {
        0
    }

    /// Initialize processor with configuration
    fn initialize(&mut self, _config: &RealtimeConfig) -> SignalResult<()> {
        Ok(())
    }

    /// Reset processor state
    fn reset(&mut self) -> SignalResult<()> {
        Ok(())
    }
}

/// Thread-safe circular buffer for streaming data
pub struct CircularBuffer {
    buffer: Arc<Mutex<VecDeque<f64>>>,
    capacity: usize,
    overrun_count: Arc<Mutex<u64>>,
}

impl CircularBuffer {
    /// Create a new circular buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
            overrun_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Write data to buffer (non-blocking)
    pub fn write(&self, data: &[f64]) -> SignalResult<usize> {
        let mut buffer = self.buffer.lock().unwrap();
        let mut written = 0;

        for &sample in data {
            if buffer.len() >= self.capacity {
                // Buffer full - drop oldest sample
                buffer.pop_front();
                *self.overrun_count.lock().unwrap() += 1;
            }
            buffer.push_back(sample);
            written += 1;
        }

        Ok(written)
    }

    /// Read data from buffer (non-blocking)
    pub fn read(&self, data: &mut [f64]) -> SignalResult<usize> {
        let mut buffer = self.buffer.lock().unwrap();
        let mut read = 0;

        for sample in data.iter_mut() {
            if let Some(value) = buffer.pop_front() {
                *sample = value;
                read += 1;
            } else {
                *sample = 0.0; // Zero padding if buffer empty
                break;
            }
        }

        Ok(read)
    }

    /// Get current buffer fill level
    pub fn fill_level(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }

    /// Get overrun count
    pub fn overrun_count(&self) -> u64 {
        *self.overrun_count.lock().unwrap()
    }

    /// Clear buffer
    pub fn clear(&self) {
        self.buffer.lock().unwrap().clear();
    }
}

/// Real-time streaming processor
pub struct StreamProcessor {
    config: RealtimeConfig,
    processor: Box<dyn RealtimeProcessor>,
    input_buffer: CircularBuffer,
    output_buffer: CircularBuffer,
    stats: Arc<Mutex<RealtimeStats>>,
    running: Arc<Mutex<bool>>,
    _process_thread: Option<thread::JoinHandle<()>>,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(
        processor: Box<dyn RealtimeProcessor>,
        config: RealtimeConfig,
    ) -> SignalResult<Self> {
        let input_buffer = CircularBuffer::new(config.buffer_size * 4);
        let output_buffer = CircularBuffer::new(config.buffer_size * 4);

        Ok(Self {
            config,
            processor,
            input_buffer,
            output_buffer,
            stats: Arc::new(Mutex::new(RealtimeStats::default())),
            running: Arc::new(Mutex::new(false)),
            _process_thread: None,
        })
    }

    /// Start real-time processing
    pub fn start(&mut self) -> SignalResult<()> {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Err(SignalError::ValueError(
                "Processor already running".to_string(),
            ));
        }
        *running = true;

        // Initialize processor
        self.processor.initialize(&self.config)?;

        // Start processing thread
        let config = self.config.clone();
        let input_buffer = self.input_buffer.buffer.clone();
        let output_buffer = self.output_buffer.buffer.clone();
        let stats = self.stats.clone();
        let running_flag = self.running.clone();

        let process_thread = thread::spawn(move || {
            Self::processing_loop(config, input_buffer, output_buffer, stats, running_flag);
        });

        self._process_thread = Some(process_thread);

        Ok(())
    }

    /// Stop real-time processing
    pub fn stop(&mut self) -> SignalResult<()> {
        *self.running.lock().unwrap() = false;

        if let Some(thread) = self._process_thread.take() {
            thread.join().map_err(|_| {
                SignalError::RuntimeError("Failed to join processing thread".to_string())
            })?;
        }

        Ok(())
    }

    /// Push input samples
    pub fn push_input(&self, samples: &[f64]) -> SignalResult<()> {
        self.input_buffer.write(samples)?;
        Ok(())
    }

    /// Pull output samples
    pub fn pull_output(&self, samples: &mut [f64]) -> SignalResult<usize> {
        self.output_buffer.read(samples)
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> RealtimeStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.lock().unwrap() = RealtimeStats::default();
    }

    /// Processing loop (runs in separate thread)
    fn processing_loop(
        config: RealtimeConfig,
        input_buffer: Arc<Mutex<VecDeque<f64>>>,
        output_buffer: Arc<Mutex<VecDeque<f64>>>,
        stats: Arc<Mutex<RealtimeStats>>,
        running: Arc<Mutex<bool>>,
    ) {
        let mut processing_buffer = vec![0.0; config.buffer_size];
        let mut block_sequence = 0u64;

        while *running.lock().unwrap() {
            let start_time = Instant::now();

            // Read input block
            let samples_read = {
                let mut input_buf = input_buffer.lock().unwrap();
                let mut read = 0;

                for sample in processing_buffer.iter_mut() {
                    if let Some(value) = input_buf.pop_front() {
                        *sample = value;
                        read += 1;
                    } else {
                        *sample = 0.0;
                        break;
                    }
                }
                read
            };

            if samples_read < config.buffer_size {
                // Not enough input data - add to underrun count
                let mut stats_guard = stats.lock().unwrap();
                stats_guard.underruns += 1;

                // Sleep briefly to avoid busy waiting
                thread::sleep(Duration::from_millis(1));
                continue;
            }

            // Create stream block
            let _stream_block = StreamBlock::new(
                processing_buffer.clone(),
                1, // Assume mono for now
                block_sequence,
            );

            // Process block (this would call the actual processor)
            // For now, just copy input to output as placeholder

            // Write output block
            {
                let mut output_buf = output_buffer.lock().unwrap();
                for &sample in &_stream_block.data {
                    if output_buf.len() >= config.buffer_size * 4 {
                        output_buf.pop_front(); // Drop oldest if buffer full
                    }
                    output_buf.push_back(sample);
                }
            }

            // Update statistics
            let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
            {
                let mut stats_guard = stats.lock().unwrap();
                stats_guard.blocks_processed += 1;
                stats_guard.samples_processed += samples_read as u64;

                // Update timing statistics
                if stats_guard.blocks_processed == 1 {
                    stats_guard.avg_processing_time_ms = processing_time;
                } else {
                    let alpha = 0.1; // Smoothing factor
                    stats_guard.avg_processing_time_ms = alpha * processing_time
                        + (1.0 - alpha) * stats_guard.avg_processing_time_ms;
                }

                if processing_time > stats_guard.max_processing_time_ms {
                    stats_guard.max_processing_time_ms = processing_time;
                }

                // Estimate current latency
                let samples_per_ms = config.sample_rate / 1000.0;
                let buffer_latency = (input_buffer.lock().unwrap().len()
                    + output_buffer.lock().unwrap().len())
                    as f64
                    / samples_per_ms;
                stats_guard.current_latency_ms = buffer_latency + processing_time;
            }

            block_sequence += 1;

            // Sleep to maintain real-time constraints
            let target_block_time = config.buffer_size as f64 / config.sample_rate * 1000.0;
            if processing_time < target_block_time {
                let sleep_time = target_block_time - processing_time;
                thread::sleep(Duration::from_millis(sleep_time as u64));
            }
        }
    }
}

/// Lock-free ring buffer for ultra-low latency applications
pub struct LockFreeRingBuffer {
    buffer: Vec<std::sync::atomic::AtomicU64>,
    capacity: usize,
    write_pos: std::sync::atomic::AtomicUsize,
    read_pos: std::sync::atomic::AtomicUsize,
}

impl LockFreeRingBuffer {
    /// Create new lock-free ring buffer
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(std::sync::atomic::AtomicU64::new(0));
        }

        Self {
            buffer,
            capacity,
            write_pos: std::sync::atomic::AtomicUsize::new(0),
            read_pos: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Write data (non-blocking, returns false if buffer full)
    pub fn write(&self, data: &[f64]) -> bool {
        use std::sync::atomic::Ordering;

        let current_write = self.write_pos.load(Ordering::Acquire);
        let current_read = self.read_pos.load(Ordering::Acquire);

        // Check if there's enough space
        let available_space = if current_write >= current_read {
            self.capacity - (current_write - current_read) - 1
        } else {
            current_read - current_write - 1
        };

        if data.len() > available_space {
            return false; // Not enough space
        }

        // Write data
        for (i, &sample) in data.iter().enumerate() {
            let pos = (current_write + i) % self.capacity;
            let bits = sample.to_bits();
            self.buffer[pos].store(bits, Ordering::Release);
        }

        // Update write position
        let new_write = (current_write + data.len()) % self.capacity;
        self.write_pos.store(new_write, Ordering::Release);

        true
    }

    /// Read data (non-blocking, returns actual samples read)
    pub fn read(&self, data: &mut [f64]) -> usize {
        use std::sync::atomic::Ordering;

        let current_read = self.read_pos.load(Ordering::Acquire);
        let current_write = self.write_pos.load(Ordering::Acquire);

        // Calculate available data
        let available_data = if current_write >= current_read {
            current_write - current_read
        } else {
            self.capacity - (current_read - current_write)
        };

        let samples_to_read = data.len().min(available_data);

        // Read data
        for (i, datum) in data.iter_mut().enumerate().take(samples_to_read) {
            let pos = (current_read + i) % self.capacity;
            let bits = self.buffer[pos].load(Ordering::Acquire);
            *datum = f64::from_bits(bits);
        }

        // Update read position
        let new_read = (current_read + samples_to_read) % self.capacity;
        self.read_pos.store(new_read, Ordering::Release);

        samples_to_read
    }

    /// Get current fill level
    pub fn fill_level(&self) -> usize {
        use std::sync::atomic::Ordering;

        let current_read = self.read_pos.load(Ordering::Acquire);
        let current_write = self.write_pos.load(Ordering::Acquire);

        if current_write >= current_read {
            current_write - current_read
        } else {
            self.capacity - (current_read - current_write)
        }
    }
}

/// Example real-time processors
///
/// Simple gain processor
pub struct GainProcessor {
    gain: f64,
}

impl GainProcessor {
    pub fn new(gain: f64) -> Self {
        Self { gain }
    }
}

impl RealtimeProcessor for GainProcessor {
    fn process_block(&mut self, block: &mut StreamBlock) -> SignalResult<()> {
        for sample in &mut block.data {
            *sample *= self.gain;
        }
        Ok(())
    }
}

/// Moving average filter processor
pub struct MovingAverageProcessor {
    window_size: usize,
    history: VecDeque<f64>,
    sum: f64,
}

impl MovingAverageProcessor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            history: VecDeque::with_capacity(window_size),
            sum: 0.0,
        }
    }
}

impl RealtimeProcessor for MovingAverageProcessor {
    fn process_block(&mut self, block: &mut StreamBlock) -> SignalResult<()> {
        for sample in &mut block.data {
            // Add new sample
            self.history.push_back(*sample);
            self.sum += *sample;

            // Remove old sample if window full
            if self.history.len() > self.window_size {
                if let Some(old_sample) = self.history.pop_front() {
                    self.sum -= old_sample;
                }
            }

            // Output average
            *sample = self.sum / self.history.len() as f64;
        }
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        self.window_size / 2
    }

    fn reset(&mut self) -> SignalResult<()> {
        self.history.clear();
        self.sum = 0.0;
        Ok(())
    }
}

/// Zero-latency lookhead limiter
pub struct ZeroLatencyLimiter {
    threshold: f64,
    lookahead_samples: usize,
    delay_buffer: VecDeque<f64>,
    gain_buffer: VecDeque<f64>,
    attack_time: f64,
    release_time: f64,
    sample_rate: f64,
    current_gain: f64,
}

impl ZeroLatencyLimiter {
    pub fn new(threshold: f64, _lookahead_ms: f64, attack_ms: f64, release_ms: f64) -> Self {
        Self {
            threshold,
            lookahead_samples: 0, // Will be set in initialize
            delay_buffer: VecDeque::new(),
            gain_buffer: VecDeque::new(),
            attack_time: attack_ms,
            release_time: release_ms,
            sample_rate: 44100.0,
            current_gain: 1.0,
        }
    }
}

impl RealtimeProcessor for ZeroLatencyLimiter {
    fn initialize(&mut self, config: &RealtimeConfig) -> SignalResult<()> {
        self.sample_rate = config.sample_rate;
        self.lookahead_samples = (self.attack_time * config.sample_rate / 1000.0) as usize;

        // Initialize delay buffers
        self.delay_buffer = VecDeque::with_capacity(self.lookahead_samples * 2);
        self.gain_buffer = VecDeque::with_capacity(self.lookahead_samples * 2);

        // Fill with zeros
        for _ in 0..self.lookahead_samples {
            self.delay_buffer.push_back(0.0);
            self.gain_buffer.push_back(1.0);
        }

        Ok(())
    }

    fn process_block(&mut self, block: &mut StreamBlock) -> SignalResult<()> {
        let attack_coeff = (-1.0 / (self.attack_time * self.sample_rate / 1000.0)).exp();
        let release_coeff = (-1.0 / (self.release_time * self.sample_rate / 1000.0)).exp();

        for sample in &mut block.data {
            // Add current sample to delay buffer
            self.delay_buffer.push_back(*sample);

            // Calculate required gain reduction for current sample
            let sample_abs = sample.abs();
            let target_gain = if sample_abs > self.threshold {
                self.threshold / sample_abs
            } else {
                1.0
            };

            // Apply gain smoothing
            if target_gain < self.current_gain {
                // Attack (gain reduction)
                self.current_gain = target_gain + (self.current_gain - target_gain) * attack_coeff;
            } else {
                // Release (gain restoration)
                self.current_gain = target_gain + (self.current_gain - target_gain) * release_coeff;
            }

            self.gain_buffer.push_back(self.current_gain);

            // Output delayed and limited sample
            if let (Some(delayed_sample), Some(gain)) =
                (self.delay_buffer.pop_front(), self.gain_buffer.pop_front())
            {
                *sample = delayed_sample * gain;
            }
        }

        Ok(())
    }

    fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }

    fn reset(&mut self) -> SignalResult<()> {
        self.delay_buffer.clear();
        self.gain_buffer.clear();
        self.current_gain = 1.0;

        // Refill with zeros
        for _ in 0..self.lookahead_samples {
            self.delay_buffer.push_back(0.0);
            self.gain_buffer.push_back(1.0);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_buffer() {
        let buffer = CircularBuffer::new(10);

        // Test write/read
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.write(&input_data).unwrap();

        let mut output_data = [0.0; 5];
        let read_count = buffer.read(&mut output_data).unwrap();

        assert_eq!(read_count, 5);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn test_circular_buffer_overrun() {
        let buffer = CircularBuffer::new(3);

        // Write more data than capacity
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.write(&input_data).unwrap();

        assert_eq!(buffer.overrun_count(), 2); // 2 samples dropped
        assert_eq!(buffer.fill_level(), 3);
    }

    #[test]
    fn test_lock_free_ring_buffer() {
        let buffer = LockFreeRingBuffer::new(10);

        // Test basic write/read
        let input_data = [1.0, 2.0, 3.0];
        assert!(buffer.write(&input_data));

        let mut output_data = [0.0; 3];
        let read_count = buffer.read(&mut output_data);

        assert_eq!(read_count, 3);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn test_gain_processor() {
        let mut processor = GainProcessor::new(2.0);
        let mut block = StreamBlock::new(vec![1.0, 2.0, 3.0, 4.0], 1, 0);

        processor.process_block(&mut block).unwrap();

        assert_eq!(block.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_moving_average_processor() {
        let mut processor = MovingAverageProcessor::new(3);
        let mut block = StreamBlock::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 0);

        processor.process_block(&mut block).unwrap();

        // First sample: average of [1.0] = 1.0
        // Second sample: average of [1.0, 2.0] = 1.5
        // Third sample: average of [1.0, 2.0, 3.0] = 2.0
        // Fourth sample: average of [2.0, 3.0, 4.0] = 3.0
        // Fifth sample: average of [3.0, 4.0, 5.0] = 4.0
        assert_eq!(block.data, vec![1.0, 1.5, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_stream_block() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let block = StreamBlock::new(data, 2, 0); // 2 channels

        assert_eq!(block.samples_per_channel(), 3);

        let ch0 = block.channel_data(0).unwrap();
        let ch1 = block.channel_data(1).unwrap();

        assert_eq!(ch0, vec![1.0, 3.0, 5.0]);
        assert_eq!(ch1, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_realtime_config() {
        let config = RealtimeConfig::default();

        assert_eq!(config.buffer_size, 1024);
        assert_eq!(config.sample_rate, 44100.0);
        assert_eq!(config.num_threads, 1);
        assert!(config.zero_copy);
    }
}
