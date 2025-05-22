//! Streaming data processors for continuous data flows
//!
//! This module provides utilities for processing continuous data streams efficiently:
//!
//! - Stream processing with minimal memory overhead
//! - Pipeline-based data processing for complex transformations
//! - Backpressure handling for rate mismatches
//! - Buffer management for smooth data flow
//! - Fault tolerance with resume capabilities

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use crate::memory_efficient::chunked::{ChunkedArray, ChunkingStrategy};
use crate::memory_efficient::prefetch::{AccessPattern, PrefetchConfig};
use crate::parallel;
use ndarray::{Array, ArrayBase, Dimension, Ix1, Ix2, IxDyn, RawData};
use std::collections::{BTreeMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Stream processing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// Process data as it comes, with no buffering
    Immediate,
    /// Buffer data up to a certain size before processing
    Buffered,
    /// Adaptive processing based on system load and data rate
    Adaptive,
    /// Use a sliding window of data for processing
    SlidingWindow,
}

/// Input source for a data stream
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamSource {
    /// File input (memory mapped)
    File,
    /// Network socket input
    Network,
    /// Real-time sensor data
    Sensor,
    /// Generated data (simulation, etc.)
    Generated,
    /// Another stream processor
    Stream,
}

/// Stream processor state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// Stream is initialized but not started
    Initialized,
    /// Stream is currently running
    Running,
    /// Stream is paused (can be resumed)
    Paused,
    /// Stream has completed
    Completed,
    /// Stream has encountered an error
    Error,
}

/// Stream processor configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Processing mode
    pub mode: StreamMode,
    /// Buffer size in elements
    pub buffer_size: usize,
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Minimum batch size for processing
    pub min_batch_size: usize,
    /// Chunk size for chunked processing
    pub chunk_size: usize,
    /// Whether to use parallel processing
    pub parallel: bool,
    /// Number of worker threads for parallel processing
    pub workers: Option<usize>,
    /// Maximum processing rate (items per second, 0 for unlimited)
    pub rate_limit: usize,
    /// Timeout for waiting for data (milliseconds, 0 for none)
    pub timeout_ms: u64,
    /// Whether to enable prefetching
    pub enable_prefetch: bool,
    /// Prefetch configuration
    pub prefetch_config: Option<PrefetchConfig>,
    /// Whether to enable backpressure handling
    pub enable_backpressure: bool,
    /// Window size for sliding window mode
    pub window_size: usize,
    /// Window stride for sliding window mode
    pub window_stride: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            mode: StreamMode::Buffered,
            buffer_size: 1024 * 1024, // 1M elements
            max_batch_size: 65536,    // 64K elements
            min_batch_size: 1024,     // 1K elements
            chunk_size: 1024,         // 1K elements
            parallel: true,
            workers: None,
            rate_limit: 0,
            timeout_ms: 1000,
            enable_prefetch: true,
            prefetch_config: None,
            enable_backpressure: true,
            window_size: 1024,
            window_stride: 256,
        }
    }
}

/// Builder for stream processor configuration
#[derive(Debug, Clone)]
pub struct StreamConfigBuilder {
    config: StreamConfig,
}

impl StreamConfigBuilder {
    /// Create a new stream configuration builder with default values
    pub fn new() -> Self {
        Self {
            config: StreamConfig::default(),
        }
    }

    /// Set the processing mode
    pub fn mode(mut self, mode: StreamMode) -> Self {
        self.config.mode = mode;
        self
    }

    /// Set the buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Set the maximum batch size
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }

    /// Set the minimum batch size
    pub fn min_batch_size(mut self, size: usize) -> Self {
        self.config.min_batch_size = size;
        self
    }

    /// Set the chunk size
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Enable or disable parallel processing
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.parallel = enable;
        self
    }

    /// Set the number of worker threads
    pub fn workers(mut self, workers: Option<usize>) -> Self {
        self.config.workers = workers;
        self
    }

    /// Set the rate limit
    pub fn rate_limit(mut self, limit: usize) -> Self {
        self.config.rate_limit = limit;
        self
    }

    /// Set the timeout
    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        self.config.timeout_ms = timeout;
        self
    }

    /// Enable or disable prefetching
    pub fn enable_prefetch(mut self, enable: bool) -> Self {
        self.config.enable_prefetch = enable;
        self
    }

    /// Set the prefetch configuration
    pub fn prefetch_config(mut self, config: Option<PrefetchConfig>) -> Self {
        self.config.prefetch_config = config;
        self
    }

    /// Enable or disable backpressure handling
    pub fn enable_backpressure(mut self, enable: bool) -> Self {
        self.config.enable_backpressure = enable;
        self
    }

    /// Set the window size for sliding window mode
    pub fn window_size(mut self, size: usize) -> Self {
        self.config.window_size = size;
        self
    }

    /// Set the window stride for sliding window mode
    pub fn window_stride(mut self, stride: usize) -> Self {
        self.config.window_stride = stride;
        self
    }

    /// Build the configuration
    pub fn build(self) -> StreamConfig {
        self.config
    }
}

/// Stream processor statistics
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Number of items processed
    pub processed_items: usize,
    /// Number of batches processed
    pub processed_batches: usize,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average processing time per batch (milliseconds)
    pub avg_batch_time_ms: f64,
    /// Average throughput (items per second)
    pub avg_throughput: f64,
    /// Stream uptime in seconds
    pub uptime_seconds: f64,
    /// Number of times backpressure was applied
    pub backpressure_count: usize,
    /// Buffer high water mark (maximum fill level)
    pub buffer_high_water_mark: usize,
    /// Error count
    pub error_count: usize,
    /// Last error message
    pub last_error: Option<String>,
}

impl Default for StreamStats {
    fn default() -> Self {
        Self {
            processed_items: 0,
            processed_batches: 0,
            avg_batch_size: 0.0,
            avg_batch_time_ms: 0.0,
            avg_throughput: 0.0,
            uptime_seconds: 0.0,
            backpressure_count: 0,
            buffer_high_water_mark: 0,
            error_count: 0,
            last_error: None,
        }
    }
}

/// Stream input buffer for data queuing
#[derive(Debug)]
struct StreamBuffer<T: Clone + Send + 'static> {
    /// Buffer data queue
    data: VecDeque<T>,
    /// Maximum buffer size
    max_size: usize,
    /// Mutex for buffer access
    mutex: Mutex<()>,
    /// Condition variable for buffer synchronization
    condvar: Condvar,
    /// Whether the stream is closed
    closed: bool,
}

impl<T: Clone + Send + 'static> StreamBuffer<T> {
    /// Create a new stream buffer
    fn new(max_size: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(max_size),
            max_size,
            mutex: Mutex::new(()),
            condvar: Condvar::new(),
            closed: false,
        }
    }

    /// Add an item to the buffer
    fn push(&mut self, item: T) -> Result<(), CoreError> {
        let mut guard = self.mutex.lock().unwrap();

        // Check if the buffer is closed
        if self.closed {
            return Err(CoreError::StreamError(
                ErrorContext::new("Stream is closed".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Wait until there's space in the buffer
        while self.data.len() >= self.max_size {
            guard = self.condvar.wait(guard).unwrap();

            // Check if the buffer was closed while waiting
            if self.closed {
                return Err(CoreError::StreamError(
                    ErrorContext::new("Stream is closed".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        }

        // Add the item to the buffer
        self.data.push_back(item);

        // Notify any waiting consumers
        self.condvar.notify_one();

        Ok(())
    }

    /// Add multiple items to the buffer
    fn push_batch(&mut self, items: Vec<T>) -> Result<(), CoreError> {
        let mut guard = self.mutex.lock().unwrap();

        // Check if the buffer is closed
        if self.closed {
            return Err(CoreError::StreamError(
                ErrorContext::new("Stream is closed".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Wait until there's space in the buffer
        while self.data.len() + items.len() > self.max_size {
            guard = self.condvar.wait(guard).unwrap();

            // Check if the buffer was closed while waiting
            if self.closed {
                return Err(CoreError::StreamError(
                    ErrorContext::new("Stream is closed".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        }

        // Add the items to the buffer
        self.data.extend(items);

        // Notify any waiting consumers
        self.condvar.notify_one();

        Ok(())
    }

    /// Get a batch of items from the buffer
    fn pop_batch(&mut self, max_batch_size: usize, timeout_ms: u64) -> Result<Vec<T>, CoreError> {
        let mut guard = self.mutex.lock().unwrap();

        // Wait until there are items in the buffer
        if self.data.is_empty() && !self.closed {
            if timeout_ms > 0 {
                let timeout = Duration::from_millis(timeout_ms);
                let result = self.condvar.wait_timeout(guard, timeout);

                match result {
                    Ok((g, timeout_result)) => {
                        guard = g;

                        // Check if the timeout occurred
                        if timeout_result.timed_out() && self.data.is_empty() {
                            return Err(CoreError::TimeoutError(
                                ErrorContext::new("Timeout waiting for data".to_string())
                                    .with_location(ErrorLocation::new(file!(), line!())),
                            ));
                        }
                    }
                    Err(_) => {
                        return Err(CoreError::StreamError(
                            ErrorContext::new("Error waiting for data".to_string())
                                .with_location(ErrorLocation::new(file!(), line!())),
                        ));
                    }
                }
            } else {
                // No timeout, wait indefinitely
                guard = self.condvar.wait(guard).unwrap();
            }
        }

        // Check if the buffer is closed and empty
        if self.data.is_empty() && self.closed {
            return Err(CoreError::EndOfStream(
                ErrorContext::new("End of stream".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Get the items (up to max_batch_size)
        let batch_size = std::cmp::min(max_batch_size, self.data.len());
        let mut batch = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if let Some(item) = self.data.pop_front() {
                batch.push(item);
            } else {
                break;
            }
        }

        // Notify any waiting producers
        self.condvar.notify_one();

        Ok(batch)
    }

    /// Get the number of items in the buffer
    fn len(&self) -> usize {
        let _guard = self.mutex.lock().unwrap();
        self.data.len()
    }

    /// Check if the buffer is empty
    fn is_empty(&self) -> bool {
        let _guard = self.mutex.lock().unwrap();
        self.data.is_empty()
    }

    /// Close the buffer
    fn close(&mut self) {
        let _guard = self.mutex.lock().unwrap();
        self.closed = true;
        self.condvar.notify_all();
    }

    /// Check if the buffer is closed
    fn is_closed(&self) -> bool {
        let _guard = self.mutex.lock().unwrap();
        self.closed
    }

    /// Clear the buffer
    fn clear(&mut self) {
        let _guard = self.mutex.lock().unwrap();
        self.data.clear();
        self.condvar.notify_all();
    }
}

/// Stream processor for continuous data flows
pub struct StreamProcessor<T: Clone + Send + 'static, U: Clone + Send + 'static> {
    /// Configuration for the stream processor
    config: StreamConfig,
    /// Input buffer
    input_buffer: Arc<Mutex<StreamBuffer<T>>>,
    /// Processing function
    process_fn: Arc<dyn Fn(Vec<T>) -> Result<Vec<U>, CoreError> + Send + Sync>,
    /// Output buffer
    output_buffer: Arc<Mutex<StreamBuffer<U>>>,
    /// Current state of the stream processor
    state: Arc<RwLock<StreamState>>,
    /// Statistics for the stream processor
    stats: Arc<RwLock<StreamStats>>,
    /// Worker thread handle
    worker_thread: Option<JoinHandle<()>>,
    /// Start time of the stream processor
    start_time: Arc<RwLock<Option<Instant>>>,
}

impl<T: Clone + Send + 'static, U: Clone + Send + 'static> StreamProcessor<T, U> {
    /// Create a new stream processor
    pub fn new<F>(config: StreamConfig, process_fn: F) -> Self
    where
        F: Fn(Vec<T>) -> Result<Vec<U>, CoreError> + Send + Sync + 'static,
    {
        let input_buffer = Arc::new(Mutex::new(StreamBuffer::new(config.buffer_size)));
        let output_buffer = Arc::new(Mutex::new(StreamBuffer::new(config.buffer_size)));

        Self {
            config,
            input_buffer,
            process_fn: Arc::new(process_fn),
            output_buffer,
            state: Arc::new(RwLock::new(StreamState::Initialized)),
            stats: Arc::new(RwLock::new(StreamStats::default())),
            worker_thread: None,
            start_time: Arc::new(RwLock::new(None)),
        }
    }

    /// Start the stream processor
    pub fn start(&mut self) -> Result<(), CoreError> {
        let mut state = self.state.write().unwrap();

        // Check if the stream is already running
        if *state == StreamState::Running {
            return Err(CoreError::StreamError(
                ErrorContext::new("Stream already running".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Update state
        *state = StreamState::Running;

        // Set start time
        let mut start_time = self.start_time.write().unwrap();
        *start_time = Some(Instant::now());

        // Create worker thread
        let input_buffer = self.input_buffer.clone();
        let output_buffer = self.output_buffer.clone();
        let process_fn = self.process_fn.clone();
        let config = self.config.clone();
        let state = self.state.clone();
        let stats = self.stats.clone();
        let start_time_clone = self.start_time.clone();

        let worker = thread::spawn(move || {
            Self::worker_loop(
                input_buffer,
                output_buffer,
                process_fn,
                config,
                state,
                stats,
                start_time_clone,
            );
        });

        self.worker_thread = Some(worker);

        Ok(())
    }

    /// Worker loop for processing data
    fn worker_loop(
        input_buffer: Arc<Mutex<StreamBuffer<T>>>,
        output_buffer: Arc<Mutex<StreamBuffer<U>>>,
        process_fn: Arc<dyn Fn(Vec<T>) -> Result<Vec<U>, CoreError> + Send + Sync>,
        config: StreamConfig,
        state: Arc<RwLock<StreamState>>,
        stats: Arc<RwLock<StreamStats>>,
        start_time: Arc<RwLock<Option<Instant>>>,
    ) {
        // Setup rate limiting if needed
        let rate_limit = config.rate_limit;
        let mut last_batch_time = Instant::now();
        let mut batch_window = VecDeque::new();

        // Processing loop
        loop {
            // Check if we should continue
            {
                let current_state = state.read().unwrap();
                if *current_state != StreamState::Running {
                    break;
                }
            }

            // Rate limiting
            if rate_limit > 0 {
                // Calculate the minimum time per batch
                let min_time_per_batch =
                    Duration::from_secs_f64(config.min_batch_size as f64 / rate_limit as f64);

                // Wait if necessary
                let elapsed = last_batch_time.elapsed();
                if elapsed < min_time_per_batch {
                    thread::sleep(min_time_per_batch - elapsed);
                }
            }

            // Determine batch size based on the mode
            let batch_size = match config.mode {
                StreamMode::Immediate => 1,
                StreamMode::Buffered => config.max_batch_size,
                StreamMode::Adaptive => {
                    // Simple adaptive batch sizing based on processing time
                    let mut stats_guard = stats.write().unwrap();
                    let avg_time = stats_guard.avg_batch_time_ms;

                    if avg_time < 10.0 {
                        // Processing is fast, use larger batches
                        config.max_batch_size
                    } else if avg_time < 50.0 {
                        // Medium processing time, use medium batches
                        (config.max_batch_size + config.min_batch_size) / 2
                    } else {
                        // Slow processing, use smaller batches
                        config.min_batch_size
                    }
                }
                StreamMode::SlidingWindow => config.window_size,
            };

            // Get a batch of data from the input buffer
            let input_batch = match input_buffer
                .lock()
                .unwrap()
                .pop_batch(batch_size, config.timeout_ms)
            {
                Ok(batch) => batch,
                Err(err) => {
                    match err {
                        CoreError::EndOfStream(_) => {
                            // End of stream, update state and exit
                            let mut current_state = state.write().unwrap();
                            *current_state = StreamState::Completed;
                            break;
                        }
                        CoreError::TimeoutError(_) => {
                            // Timeout, continue
                            continue;
                        }
                        _ => {
                            // Other error, update stats and continue
                            let mut stats_guard = stats.write().unwrap();
                            stats_guard.error_count += 1;
                            stats_guard.last_error = Some(format!("{}", err));
                            continue;
                        }
                    }
                }
            };

            // Check if the batch is empty
            if input_batch.is_empty() {
                continue;
            }

            // For sliding window mode, manage the window
            let process_input = if config.mode == StreamMode::SlidingWindow {
                if batch_window.len() < config.window_size {
                    // Still filling the initial window
                    batch_window.extend(input_batch);

                    if batch_window.len() < config.window_size {
                        // Not enough data for a full window yet
                        continue;
                    }

                    // We now have a full window
                    batch_window.make_contiguous().to_vec()
                } else {
                    // Slide the window
                    let stride = std::cmp::min(config.window_stride, input_batch.len());

                    // Remove old elements
                    for _ in 0..stride {
                        batch_window.pop_front();
                    }

                    // Add new elements
                    batch_window.extend(input_batch);

                    // Return the window for processing
                    batch_window.make_contiguous().to_vec()
                }
            } else {
                // For non-window modes, just use the batch directly
                input_batch
            };

            // Process the batch
            let process_result = {
                let batch_start_time = Instant::now();
                let result = process_fn(process_input.clone());

                // Update processing statistics
                let mut stats_guard = stats.write().unwrap();
                stats_guard.processed_batches += 1;
                stats_guard.processed_items += process_input.len();

                // Update average batch size
                let total_items = stats_guard.processed_items;
                let total_batches = stats_guard.processed_batches;
                stats_guard.avg_batch_size = total_items as f64 / total_batches as f64;

                // Update processing time
                let batch_time = batch_start_time.elapsed().as_millis() as f64;
                stats_guard.avg_batch_time_ms =
                    (stats_guard.avg_batch_time_ms * (total_batches - 1) as f64 + batch_time)
                        / total_batches as f64;

                // Update throughput
                if let Some(start) = *start_time.read().unwrap() {
                    let uptime_seconds = start.elapsed().as_secs_f64();
                    stats_guard.uptime_seconds = uptime_seconds;
                    stats_guard.avg_throughput = total_items as f64 / uptime_seconds;
                }

                // Update buffer statistics
                let buffer_len = input_buffer.lock().unwrap().len();
                if buffer_len > stats_guard.buffer_high_water_mark {
                    stats_guard.buffer_high_water_mark = buffer_len;
                }

                result
            };

            // Handle the processing result
            match process_result {
                Ok(output_batch) => {
                    // Send the output to the output buffer
                    if !output_batch.is_empty() {
                        match output_buffer.lock().unwrap().push_batch(output_batch) {
                            Ok(_) => {}
                            Err(err) => {
                                // Error sending output, update stats
                                let mut stats_guard = stats.write().unwrap();
                                stats_guard.error_count += 1;
                                stats_guard.last_error = Some(format!("{}", err));
                            }
                        }
                    }
                }
                Err(err) => {
                    // Processing error, update stats
                    let mut stats_guard = stats.write().unwrap();
                    stats_guard.error_count += 1;
                    stats_guard.last_error = Some(format!("{}", err));
                }
            }

            // Update rate limiting info
            last_batch_time = Instant::now();
        }
    }

    /// Stop the stream processor
    pub fn stop(&mut self) -> Result<(), CoreError> {
        let mut state = self.state.write().unwrap();

        // Check if the stream is running
        if *state != StreamState::Running {
            return Err(CoreError::StreamError(
                ErrorContext::new("Stream not running".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Update state
        *state = StreamState::Paused;

        // Close the input buffer
        self.input_buffer.lock().unwrap().close();

        // Wait for the worker thread to finish
        if let Some(worker) = self.worker_thread.take() {
            match worker.join() {
                Ok(_) => {}
                Err(_) => {
                    return Err(CoreError::StreamError(
                        ErrorContext::new("Error joining worker thread".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Push data to the stream processor
    pub fn push(&self, data: T) -> Result<(), CoreError> {
        // Check if the stream is running
        let state = self.state.read().unwrap();
        if *state != StreamState::Running {
            return Err(CoreError::StreamError(
                ErrorContext::new("Stream not running".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Push data to the input buffer
        self.input_buffer.lock().unwrap().push(data)
    }

    /// Push a batch of data to the stream processor
    pub fn push_batch(&self, data: Vec<T>) -> Result<(), CoreError> {
        // Check if the stream is running
        let state = self.state.read().unwrap();
        if *state != StreamState::Running {
            return Err(CoreError::StreamError(
                ErrorContext::new("Stream not running".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Push data to the input buffer
        self.input_buffer.lock().unwrap().push_batch(data)
    }

    /// Pop processed data from the stream processor
    pub fn pop(&self) -> Result<U, CoreError> {
        // Check if the stream is running or completed
        let state = self.state.read().unwrap();
        if *state != StreamState::Running && *state != StreamState::Completed {
            return Err(CoreError::StreamError(
                ErrorContext::new("Stream not running or completed".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Pop data from the output buffer
        let result = self
            .output_buffer
            .lock()
            .unwrap()
            .pop_batch(1, self.config.timeout_ms)?;

        if result.is_empty() {
            Err(CoreError::TimeoutError(
                ErrorContext::new("Timeout waiting for data".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ))
        } else {
            Ok(result[0].clone())
        }
    }

    /// Pop a batch of processed data from the stream processor
    pub fn pop_batch(&self, max_size: usize) -> Result<Vec<U>, CoreError> {
        // Check if the stream is running or completed
        let state = self.state.read().unwrap();
        if *state != StreamState::Running && *state != StreamState::Completed {
            return Err(CoreError::StreamError(
                ErrorContext::new("Stream not running or completed".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Pop data from the output buffer
        self.output_buffer
            .lock()
            .unwrap()
            .pop_batch(max_size, self.config.timeout_ms)
    }

    /// Get the current state of the stream processor
    pub fn state(&self) -> StreamState {
        *self.state.read().unwrap()
    }

    /// Get the statistics for the stream processor
    pub fn stats(&self) -> StreamStats {
        self.stats.read().unwrap().clone()
    }

    /// Check if the stream is empty
    pub fn is_empty(&self) -> bool {
        self.input_buffer.lock().unwrap().is_empty()
            && self.output_buffer.lock().unwrap().is_empty()
    }

    /// Clear the stream buffers
    pub fn clear(&self) -> Result<(), CoreError> {
        // Check if the stream is not running
        let state = self.state.read().unwrap();
        if *state == StreamState::Running {
            return Err(CoreError::StreamError(
                ErrorContext::new("Cannot clear running stream".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Clear the buffers
        self.input_buffer.lock().unwrap().clear();
        self.output_buffer.lock().unwrap().clear();

        Ok(())
    }
}

impl<T: Clone + Send + 'static, U: Clone + Send + 'static> Drop for StreamProcessor<T, U> {
    fn drop(&mut self) {
        // Stop the stream if it's running
        if *self.state.read().unwrap() == StreamState::Running {
            let _ = self.stop();
        }
    }
}

/// A stage in a stream processing pipeline
#[derive(Debug)]
pub struct PipelineStage<I: Clone + Send + 'static, O: Clone + Send + 'static> {
    /// Name of the stage
    pub name: String,
    /// Stream processor for this stage
    processor: Arc<Mutex<StreamProcessor<I, O>>>,
    /// Whether this stage is parallel
    pub parallel: bool,
    /// Number of parallel instances
    pub parallelism: usize,
}

impl<I: Clone + Send + 'static, O: Clone + Send + 'static> PipelineStage<I, O> {
    /// Create a new pipeline stage
    pub fn new<F>(
        name: String,
        config: StreamConfig,
        process_fn: F,
        parallel: bool,
        parallelism: usize,
    ) -> Self
    where
        F: Fn(Vec<I>) -> Result<Vec<O>, CoreError> + Send + Sync + Clone + 'static,
    {
        let processor = StreamProcessor::new(config, process_fn);

        Self {
            name,
            processor: Arc::new(Mutex::new(processor)),
            parallel,
            parallelism,
        }
    }

    /// Get the processor for this stage
    pub fn processor(&self) -> Arc<Mutex<StreamProcessor<I, O>>> {
        self.processor.clone()
    }

    /// Start the stage
    pub fn start(&self) -> Result<(), CoreError> {
        self.processor.lock().unwrap().start()
    }

    /// Stop the stage
    pub fn stop(&self) -> Result<(), CoreError> {
        self.processor.lock().unwrap().stop()
    }

    /// Get the state of the stage
    pub fn state(&self) -> StreamState {
        self.processor.lock().unwrap().state()
    }

    /// Get the statistics for the stage
    pub fn stats(&self) -> StreamStats {
        self.processor.lock().unwrap().stats()
    }
}

/// Stream processing pipeline
pub struct Pipeline {
    /// Name of the pipeline
    pub name: String,
    /// Stages in the pipeline
    stages: Vec<Box<dyn AnyStage>>,
    /// Connections between stages
    connections: Vec<(usize, usize)>, // (from_stage, to_stage)
    /// Worker threads for the pipeline
    workers: Vec<JoinHandle<()>>,
    /// Pipeline state
    state: Arc<RwLock<StreamState>>,
    /// Pipeline statistics
    stats: Arc<RwLock<PipelineStats>>,
    /// Error context for the pipeline
    error_context: Arc<RwLock<Option<ErrorContext>>>,
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Statistics for each stage
    pub stage_stats: BTreeMap<String, StreamStats>,
    /// Total items processed
    pub total_items: usize,
    /// Pipeline uptime in seconds
    pub uptime_seconds: f64,
    /// Overall throughput (items per second)
    pub overall_throughput: f64,
    /// Bottleneck stage (slowest stage)
    pub bottleneck_stage: Option<String>,
    /// Bottleneck throughput (items per second)
    pub bottleneck_throughput: f64,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self {
            stage_stats: BTreeMap::new(),
            total_items: 0,
            uptime_seconds: 0.0,
            overall_throughput: 0.0,
            bottleneck_stage: None,
            bottleneck_throughput: f64::MAX,
        }
    }
}

/// Trait for pipeline stages of any type
pub trait AnyStage: Send + Sync {
    /// Get the name of the stage
    fn name(&self) -> &str;
    /// Start the stage
    fn start(&self) -> Result<(), CoreError>;
    /// Stop the stage
    fn stop(&self) -> Result<(), CoreError>;
    /// Get the state of the stage
    fn state(&self) -> StreamState;
    /// Get the statistics for the stage
    fn stats(&self) -> StreamStats;
    /// Check if the stage is empty
    fn is_empty(&self) -> bool;
    /// Push raw data to the stage
    fn push_raw(&self, data: Box<dyn std::any::Any + Send>) -> Result<(), CoreError>;
    /// Pop raw data from the stage
    fn pop_raw(&self) -> Result<Box<dyn std::any::Any + Send>, CoreError>;
}

/// Pipeline builder
pub struct PipelineBuilder {
    /// Name of the pipeline
    name: String,
    /// Stages in the pipeline
    stages: Vec<Box<dyn AnyStage>>,
    /// Connections between stages
    connections: Vec<(usize, usize)>, // (from_stage, to_stage)
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new(name: String) -> Self {
        Self {
            name,
            stages: Vec::new(),
            connections: Vec::new(),
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage<I, O, F>(
        &mut self,
        name: String,
        config: StreamConfig,
        process_fn: F,
        parallel: bool,
        parallelism: usize,
    ) -> usize
    where
        I: Clone + Send + 'static,
        O: Clone + Send + 'static,
        F: Fn(Vec<I>) -> Result<Vec<O>, CoreError> + Send + Sync + Clone + 'static,
    {
        let stage = PipelineStage::new(name, config, process_fn, parallel, parallelism);
        let stage_index = self.stages.len();
        self.stages.push(Box::new(StageWrapper::new(stage)));
        stage_index
    }

    /// Connect two stages in the pipeline
    pub fn connect(&mut self, from_stage: usize, to_stage: usize) -> &mut Self {
        if from_stage < self.stages.len() && to_stage < self.stages.len() {
            self.connections.push((from_stage, to_stage));
        }
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Pipeline {
        Pipeline {
            name: self.name,
            stages: self.stages,
            connections: self.connections,
            workers: Vec::new(),
            state: Arc::new(RwLock::new(StreamState::Initialized)),
            stats: Arc::new(RwLock::new(PipelineStats::default())),
            error_context: Arc::new(RwLock::new(None)),
        }
    }
}

impl Pipeline {
    /// Start the pipeline
    pub fn start(&mut self) -> Result<(), CoreError> {
        let mut state = self.state.write().unwrap();

        // Check if the pipeline is already running
        if *state == StreamState::Running {
            return Err(CoreError::StreamError(
                ErrorContext::new("Pipeline already running".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Start all stages
        for stage in &self.stages {
            stage.start()?;
        }

        // Create worker threads for each connection
        for (from_stage, to_stage) in &self.connections {
            let from_stage = &self.stages[*from_stage];
            let to_stage = &self.stages[*to_stage];

            let from_stage_clone = from_stage.clone_box();
            let to_stage_clone = to_stage.clone_box();
            let state_clone = self.state.clone();
            let error_context_clone = self.error_context.clone();

            let worker = thread::spawn(move || {
                Self::connection_worker(
                    from_stage_clone,
                    to_stage_clone,
                    state_clone,
                    error_context_clone,
                );
            });

            self.workers.push(worker);
        }

        // Update state
        *state = StreamState::Running;

        Ok(())
    }

    /// Worker function for processing data between stages
    fn connection_worker(
        from_stage: Box<dyn AnyStage>,
        to_stage: Box<dyn AnyStage>,
        state: Arc<RwLock<StreamState>>,
        error_context: Arc<RwLock<Option<ErrorContext>>>,
    ) {
        let mut consecutive_errors = 0;
        let error_threshold = 10; // Maximum number of consecutive errors before giving up

        // Processing loop
        loop {
            // Check if we should continue
            {
                let current_state = state.read().unwrap();
                if *current_state != StreamState::Running {
                    break;
                }
            }

            // Try to get data from the source stage
            match from_stage.pop_raw() {
                Ok(data) => {
                    // Reset error counter
                    consecutive_errors = 0;

                    // Try to push data to the destination stage
                    if let Err(err) = to_stage.push_raw(data) {
                        // Handle error
                        consecutive_errors += 1;

                        // Update error context
                        let mut error_context_guard = error_context.write().unwrap();
                        *error_context_guard = Some(
                            ErrorContext::new(format!(
                                "Error pushing data from {} to {}: {}",
                                from_stage.name(),
                                to_stage.name(),
                                err
                            ))
                            .with_location(ErrorLocation::new(file!(), line!())),
                        );

                        // Check if we should give up
                        if consecutive_errors >= error_threshold {
                            let mut current_state = state.write().unwrap();
                            *current_state = StreamState::Error;
                            break;
                        }

                        // Sleep before retrying
                        thread::sleep(Duration::from_millis(100));
                    }
                }
                Err(err) => {
                    match err {
                        CoreError::EndOfStream(_) => {
                            // End of stream, exit gracefully
                            break;
                        }
                        CoreError::TimeoutError(_) => {
                            // Timeout, continue
                            continue;
                        }
                        _ => {
                            // Other error, increment counter
                            consecutive_errors += 1;

                            // Update error context
                            let mut error_context_guard = error_context.write().unwrap();
                            *error_context_guard = Some(
                                ErrorContext::new(format!(
                                    "Error popping data from {}: {}",
                                    from_stage.name(),
                                    err
                                ))
                                .with_location(ErrorLocation::new(file!(), line!())),
                            );

                            // Check if we should give up
                            if consecutive_errors >= error_threshold {
                                let mut current_state = state.write().unwrap();
                                *current_state = StreamState::Error;
                                break;
                            }

                            // Sleep before retrying
                            thread::sleep(Duration::from_millis(100));
                        }
                    }
                }
            }
        }
    }

    /// Stop the pipeline
    pub fn stop(&mut self) -> Result<(), CoreError> {
        let mut state = self.state.write().unwrap();

        // Check if the pipeline is running
        if *state != StreamState::Running {
            return Err(CoreError::StreamError(
                ErrorContext::new("Pipeline not running".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Update state
        *state = StreamState::Paused;

        // Stop all stages
        for stage in &self.stages {
            stage.stop()?;
        }

        // Wait for worker threads to finish
        for worker in self.workers.drain(..) {
            match worker.join() {
                Ok(_) => {}
                Err(_) => {
                    return Err(CoreError::StreamError(
                        ErrorContext::new("Error joining worker thread".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Get the current state of the pipeline
    pub fn state(&self) -> StreamState {
        *self.state.read().unwrap()
    }

    /// Get the statistics for the pipeline
    pub fn stats(&self) -> PipelineStats {
        let mut stats = PipelineStats::default();

        // Collect stats from all stages
        for stage in &self.stages {
            let stage_stats = stage.stats();
            let stage_name = stage.name().to_string();

            stats
                .stage_stats
                .insert(stage_name.clone(), stage_stats.clone());

            // Update bottleneck stats
            let stage_throughput = stage_stats.avg_throughput;
            if stage_throughput > 0.0 && stage_throughput < stats.bottleneck_throughput {
                stats.bottleneck_throughput = stage_throughput;
                stats.bottleneck_stage = Some(stage_name);
            }

            // Update total items processed (use the final stage's count)
            if !self
                .connections
                .iter()
                .any(|(_, to)| *to == self.stages.len() - 1)
            {
                stats.total_items = stage_stats.processed_items;
            }
        }

        // Calculate overall statistics
        let mut max_uptime = 0.0;
        for (_, stage_stats) in &stats.stage_stats {
            if stage_stats.uptime_seconds > max_uptime {
                max_uptime = stage_stats.uptime_seconds;
            }
        }

        stats.uptime_seconds = max_uptime;

        if max_uptime > 0.0 {
            stats.overall_throughput = stats.total_items as f64 / max_uptime;
        }

        stats
    }

    /// Get the last error from the pipeline
    pub fn last_error(&self) -> Option<ErrorContext> {
        self.error_context.read().unwrap().clone()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.stages.iter().all(|stage| stage.is_empty())
    }

    /// Get a stage by index
    pub fn stage(&self, index: usize) -> Option<&dyn AnyStage> {
        self.stages.get(index).map(|s| s.as_ref())
    }

    /// Get the number of stages in the pipeline
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // Stop the pipeline if it's running
        if *self.state.read().unwrap() == StreamState::Running {
            let _ = self.stop();
        }
    }
}

/// Wrapper for pipeline stages to implement AnyStage
struct StageWrapper<I: Clone + Send + 'static, O: Clone + Send + 'static> {
    stage: PipelineStage<I, O>,
}

impl<I: Clone + Send + 'static, O: Clone + Send + 'static> StageWrapper<I, O> {
    /// Create a new stage wrapper
    fn new(stage: PipelineStage<I, O>) -> Self {
        Self { stage }
    }
}

impl<I: Clone + Send + 'static, O: Clone + Send + 'static> AnyStage for StageWrapper<I, O> {
    fn name(&self) -> &str {
        &self.stage.name
    }

    fn start(&self) -> Result<(), CoreError> {
        self.stage.start()
    }

    fn stop(&self) -> Result<(), CoreError> {
        self.stage.stop()
    }

    fn state(&self) -> StreamState {
        self.stage.state()
    }

    fn stats(&self) -> StreamStats {
        self.stage.stats()
    }

    fn is_empty(&self) -> bool {
        self.stage.processor.lock().unwrap().is_empty()
    }

    fn push_raw(&self, data: Box<dyn std::any::Any + Send>) -> Result<(), CoreError> {
        let input = match data.downcast::<Vec<I>>() {
            Ok(input) => *input,
            Err(data) => {
                // Try to downcast to a single item
                match data.downcast::<I>() {
                    Ok(item) => vec![*item],
                    Err(_) => {
                        return Err(CoreError::StreamError(
                            ErrorContext::new(format!(
                                "Type mismatch when pushing data to stage {}",
                                self.name()
                            ))
                            .with_location(ErrorLocation::new(file!(), line!())),
                        ));
                    }
                }
            }
        };

        self.stage.processor.lock().unwrap().push_batch(input)
    }

    fn pop_raw(&self) -> Result<Box<dyn std::any::Any + Send>, CoreError> {
        let output = self.stage.processor.lock().unwrap().pop_batch(100)?;
        Ok(Box::new(output))
    }
}

impl dyn AnyStage {
    /// Clone the stage into a new Box
    fn clone_box(&self) -> Box<dyn AnyStage> {
        self.clone_box_impl()
    }

    /// Implementation for clone_box
    fn clone_box_impl(&self) -> Box<dyn AnyStage>;
}

impl<I: Clone + Send + 'static, O: Clone + Send + 'static> Clone for StageWrapper<I, O> {
    fn clone(&self) -> Self {
        let stage = PipelineStage {
            name: self.stage.name.clone(),
            processor: self.stage.processor(),
            parallel: self.stage.parallel,
            parallelism: self.stage.parallelism,
        };

        Self { stage }
    }
}

impl<T: Clone + AnyStage> AnyStage for T {
    fn clone_box_impl(&self) -> Box<dyn AnyStage> {
        Box::new(self.clone())
    }
}

/// Extensions to the StreamProcessor to enable ndarray processing
impl<A, D> StreamProcessor<ArrayBase<Vec<A>, D>, ArrayBase<Vec<A>, D>>
where
    A: Clone + Send + Default + 'static,
    D: Dimension + Clone + Send + 'static,
{
    /// Create a new array stream processor
    pub fn new_array<F>(config: StreamConfig, process_fn: F) -> Self
    where
        F: Fn(Vec<ArrayBase<Vec<A>, D>>) -> Result<Vec<ArrayBase<Vec<A>, D>>, CoreError>
            + Send
            + Sync
            + 'static,
    {
        Self::new(config, process_fn)
    }

    /// Process arrays chunk-wise
    pub fn chunk_wise<F>(config: StreamConfig, chunk_size: usize, process_fn: F) -> Self
    where
        F: Fn(&ArrayBase<Vec<A>, D>) -> Result<ArrayBase<Vec<A>, D>, CoreError>
            + Send
            + Sync
            + Clone
            + 'static,
    {
        let chunking_strategy = ChunkingStrategy::Fixed(chunk_size);

        let process_fn_clone = process_fn.clone();
        let chunks_fn = move |arrays: Vec<ArrayBase<Vec<A>, D>>| -> Result<Vec<ArrayBase<Vec<A>, D>>, CoreError> {
            let mut results = Vec::with_capacity(arrays.len());

            for array in arrays {
                // Create chunked array
                let chunked = ChunkedArray::new(array, chunking_strategy);

                // Process each chunk and combine results
                let mut chunk_results = Vec::new();
                for chunk in chunked.chunks() {
                    let result = process_fn_clone(&chunk)?;
                    chunk_results.push(result);
                }

                // Combine chunk results
                let combined = ChunkedArray::combine_chunks(chunk_results)?;
                results.push(combined);
            }

            Ok(results)
        };

        Self::new(config, chunks_fn)
    }

    /// Process arrays in parallel
    #[cfg(feature = "parallel")]
    pub fn parallel<F>(config: StreamConfig, process_fn: F) -> Self
    where
        F: Fn(&ArrayBase<Vec<A>, D>) -> Result<ArrayBase<Vec<A>, D>, CoreError>
            + Send
            + Sync
            + Clone
            + 'static,
        A: Send + Sync,
    {
        let workers = config
            .workers
            .unwrap_or_else(|| parallel::get_num_workers());

        let process_fn_clone = process_fn.clone();
        let parallel_fn = move |arrays: Vec<ArrayBase<Vec<A>, D>>| -> Result<Vec<ArrayBase<Vec<A>, D>>, CoreError> {
            // Process arrays in parallel
            parallel::with_workers(workers, || {
                let results: Result<Vec<_>, _> = arrays
                    .par_iter()
                    .map(|array| process_fn_clone(array))
                    .collect();

                results
            })
        };

        Self::new(config, parallel_fn)
    }
}

/// Create a new stream processor with default configuration
pub fn create_stream_processor<T, U, F>(process_fn: F) -> StreamProcessor<T, U>
where
    T: Clone + Send + 'static,
    U: Clone + Send + 'static,
    F: Fn(Vec<T>) -> Result<Vec<U>, CoreError> + Send + Sync + 'static,
{
    StreamProcessor::new(StreamConfig::default(), process_fn)
}

/// Create a new pipeline
pub fn create_pipeline(name: &str) -> PipelineBuilder {
    PipelineBuilder::new(name.to_string())
}

/// Extension trait for error handling in stream processing
pub trait StreamError {
    /// Convert to a stream error
    fn to_stream_error(self, message: &str) -> CoreError;
}

impl<T> StreamError for std::result::Result<T, CoreError> {
    fn to_stream_error(self, message: &str) -> CoreError {
        match self {
            Ok(_) => CoreError::StreamError(
                ErrorContext::new(format!("Unexpected success: {}", message))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            Err(e) => CoreError::StreamError(
                ErrorContext::new(format!("{}: {}", message, e))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
        }
    }
}

/// Extension to CoreError for stream errors
impl CoreError {
    /// Create a new end of stream error
    pub fn end_of_stream(message: &str) -> Self {
        CoreError::EndOfStream(
            ErrorContext::new(message.to_string())
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    }

    /// Create a new stream error
    pub fn stream_error(message: &str) -> Self {
        CoreError::StreamError(
            ErrorContext::new(message.to_string())
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    }

    /// Create a new timeout error
    pub fn timeout_error(message: &str) -> Self {
        CoreError::TimeoutError(
            ErrorContext::new(message.to_string())
                .with_location(ErrorLocation::new(file!(), line!())),
        )
    }
}
