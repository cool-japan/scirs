//! Out-of-core processing for massive time series datasets
//!
//! This module provides functionality for processing time series data that doesn't fit in memory,
//! including chunked processing, streaming analysis, and memory-efficient algorithms.
//!
//! # Features
//!
//! - Chunked time series processing with configurable chunk sizes
//! - Memory-mapped file I/O for efficient disk access
//! - Streaming statistics computation (mean, variance, quantiles)
//! - Out-of-core decomposition and forecasting
//! - Progress tracking and memory usage monitoring
//! - Parallel processing of chunks
//!
//! # Examples
//!
//! ```rust,ignore
//! use scirs2_series::out_of_core::{ChunkedProcessor, ProcessingConfig};
//! use std::fs::File;
//! use std::io::Write;
//! use tempfile::tempdir;
//!
//! // Create a temporary CSV file for testing
//! let dir = tempdir().expect("Operation failed");
//! let file_path = dir.path().join("test_data.csv");
//! let mut file = File::create(&file_path).expect("Operation failed");
//! writeln!(file, "value").expect("Operation failed");
//! for i in 0..100 {
//!     writeln!(file, "{}", i as f64 * 0.1).expect("Operation failed");
//! }
//!
//! // Process the dataset in chunks
//! let config = ProcessingConfig::new()
//!     .with_chunk_size(10)
//!     .with_overlap(2)
//!     .with_parallel_processing(false);
//!
//! let mut processor = ChunkedProcessor::new(config);
//! let results = processor.process_csv_file(&file_path, 0, true)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{Result, TimeSeriesError};
use memmap2::{Mmap, MmapOptions};
use scirs2_core::ndarray::Array1;
use scirs2_core::validation::check_positive;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Configuration for out-of-core processing
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    /// Chunk size (number of data points per chunk)
    pub chunk_size: usize,
    /// Overlap between chunks (for continuity in analysis). Effective overlap is
    /// capped at `chunk_size - 1` so processing always advances by at least one
    /// point per chunk, regardless of how `chunk_size`/`overlap` are combined.
    pub overlap: usize,
    /// Enable parallel processing of chunks
    pub parallel_processing: bool,
    /// Maximum memory usage (bytes)
    pub max_memory_usage: usize,
    /// Number of worker threads
    pub num_threads: usize,
    /// Buffer size for file I/O
    pub buffer_size: usize,
    /// Enable progress reporting
    pub report_progress: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 100_000,
            overlap: 1_000,
            parallel_processing: true,
            max_memory_usage: 1_073_741_824, // 1 GB
            num_threads: num_cpus::get(),
            buffer_size: 8192,
            report_progress: true,
        }
    }
}

impl ProcessingConfig {
    /// Create new processing configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set overlap between chunks
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Enable or disable parallel processing
    pub fn with_parallel_processing(mut self, enabled: bool) -> Self {
        self.parallel_processing = enabled;
        self
    }

    /// Set maximum memory usage
    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory_usage = bytes;
        self
    }

    /// Set number of worker threads
    pub fn with_threads(mut self, numthreads: usize) -> Self {
        self.num_threads = numthreads;
        self
    }
}

/// Streaming statistics accumulator
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Number of observations
    pub count: u64,
    /// Running mean
    pub mean: f64,
    /// Running sum of squared deviations
    pub m2: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Running sum
    pub sum: f64,
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
        }
    }
}

impl StreamingStats {
    /// Create new streaming statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with a new value using Welford's online algorithm
    pub fn update(&mut self, value: f64) {
        if value.is_finite() {
            self.count += 1;
            self.sum += value;
            self.min = self.min.min(value);
            self.max = self.max.max(value);

            let delta = value - self.mean;
            self.mean += delta / self.count as f64;
            let delta2 = value - self.mean;
            self.m2 += delta * delta2;
        }
    }

    /// Merge with another StreamingStats
    pub fn merge(&mut self, other: &StreamingStats) {
        if other.count == 0 {
            return;
        }

        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = (self.count as f64 * self.mean + other.count as f64 * other.mean)
            / combined_count as f64;

        let combined_m2 = self.m2
            + other.m2
            + delta * delta * (self.count as f64 * other.count as f64) / combined_count as f64;

        self.count = combined_count;
        self.mean = combined_mean;
        self.m2 = combined_m2;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.sum += other.sum;
    }

    /// Get variance
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get sample variance
    pub fn sample_variance(&self) -> f64 {
        if self.count < 1 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }
}

/// Progress information for long-running operations
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Current chunk number
    pub chunk_number: usize,
    /// Total number of chunks
    pub total_chunks: usize,
    /// Current data points processed
    pub points_processed: u64,
    /// Total data points
    pub total_points: u64,
    /// Elapsed time in seconds
    pub elapsed_time: f64,
    /// Estimated remaining time in seconds
    pub estimated_remaining: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

impl ProgressInfo {
    /// Calculate completion percentage
    pub fn completion_percentage(&self) -> f64 {
        if self.total_points == 0 {
            0.0
        } else {
            (self.points_processed as f64 / self.total_points as f64) * 100.0
        }
    }

    /// Calculate processing rate (points per second)
    pub fn processing_rate(&self) -> f64 {
        if self.elapsed_time == 0.0 {
            0.0
        } else {
            self.points_processed as f64 / self.elapsed_time
        }
    }
}

/// Memory-mapped time series reader
pub struct MmapTimeSeriesReader {
    /// Memory-mapped file
    mmap: Mmap,
    /// File size in bytes
    file_size: usize,
    /// Number of data points
    num_points: usize,
    /// Data type size in bytes
    element_size: usize,
}

impl MmapTimeSeriesReader {
    /// Create new memory-mapped reader for binary f64 data
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to open file: {e}")))?;

        let file_size = file
            .metadata()
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to get file metadata: {e}")))?
            .len() as usize;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| TimeSeriesError::IOError(format!("Failed to memory map file: {e}")))?
        };

        let element_size = std::mem::size_of::<f64>();
        let num_points = file_size / element_size;

        Ok(Self {
            mmap,
            file_size,
            num_points,
            element_size,
        })
    }

    /// Get number of data points in the file
    pub fn len(&self) -> usize {
        self.num_points
    }

    /// Check if the file is empty
    pub fn is_empty(&self) -> bool {
        self.num_points == 0
    }

    /// Read a chunk of data starting at the given index
    pub fn read_chunk(&self, start_idx: usize, chunksize: usize) -> Result<Array1<f64>> {
        if start_idx >= self.num_points {
            return Err(TimeSeriesError::InvalidInput(
                "Start index out of bounds".to_string(),
            ));
        }

        let end_idx = (start_idx + chunksize).min(self.num_points);
        let actual_size = end_idx - start_idx;

        let mut chunk = Array1::zeros(actual_size);

        let byte_start = start_idx * self.element_size;
        let byte_size = actual_size * self.element_size;

        if byte_start + byte_size > self.file_size {
            return Err(TimeSeriesError::InvalidInput(
                "Chunk extends beyond file".to_string(),
            ));
        }

        let data_slice = &self.mmap[byte_start..byte_start + byte_size];
        let f64_slice =
            unsafe { std::slice::from_raw_parts(data_slice.as_ptr() as *const f64, actual_size) };

        chunk.assign(&Array1::from_vec(f64_slice.to_vec()));
        Ok(chunk)
    }

    /// Read a range of data with optional overlap
    pub fn read_range(&self, start_idx: usize, endidx: usize) -> Result<Array1<f64>> {
        let end_idx = endidx.min(self.num_points);

        if start_idx >= end_idx {
            return Ok(Array1::zeros(0));
        }

        self.read_chunk(start_idx, end_idx - start_idx)
    }
}

/// CSV time series reader for out-of-core processing
pub struct CsvTimeSeriesReader {
    /// File path
    file_path: PathBuf,
    /// Total number of lines (estimated)
    total_lines: Option<usize>,
    /// Column index to read (0-based)
    column_index: usize,
    /// Whether file has header
    has_header: bool,
}

impl CsvTimeSeriesReader {
    /// Create new CSV reader
    pub fn new<P: AsRef<Path>>(path: P, column_index: usize, hasheader: bool) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();

        if !file_path.exists() {
            return Err(TimeSeriesError::IOError("File does not exist".to_string()));
        }

        Ok(Self {
            file_path,
            total_lines: None,
            column_index,
            has_header: hasheader,
        })
    }

    /// Estimate total number of data lines in the file
    pub fn estimate_total_lines(&mut self) -> Result<usize> {
        if let Some(total) = self.total_lines {
            return Ok(total);
        }

        let file = File::open(&self.file_path)
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to open file: {e}")))?;

        let reader = BufReader::new(file);
        let mut count = 0;

        for line in reader.lines() {
            line.map_err(|e| TimeSeriesError::IOError(format!("Failed to read line: {e}")))?;
            count += 1;
        }

        // Subtract header if present
        if self.has_header && count > 0 {
            count -= 1;
        }

        self.total_lines = Some(count);
        Ok(count)
    }

    /// Read a chunk of data starting at the given line
    pub fn read_chunk(&self, start_line: usize, chunksize: usize) -> Result<Array1<f64>> {
        let file = File::open(&self.file_path)
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to open file: {e}")))?;

        let reader = BufReader::new(file);
        let mut data = Vec::new();
        let mut current_line = 0;
        let mut data_line = 0;

        for _line in reader.lines() {
            let _line =
                _line.map_err(|e| TimeSeriesError::IOError(format!("Failed to read line: {e}")))?;

            // Skip header
            if current_line == 0 && self.has_header {
                current_line += 1;
                continue;
            }

            // Check if we've reached the start of our chunk
            if data_line >= start_line {
                let fields: Vec<&str> = _line.split(',').collect();

                if self.column_index >= fields.len() {
                    return Err(TimeSeriesError::InvalidInput(format!(
                        "Column index {} out of bounds for _line with {} fields",
                        self.column_index,
                        fields.len()
                    )));
                }

                let value: f64 = fields[self.column_index].trim().parse().map_err(|e| {
                    TimeSeriesError::InvalidInput(format!("Failed to parse value: {e}"))
                })?;

                data.push(value);

                // Stop if we've read enough data
                if data.len() >= chunksize {
                    break;
                }
            }

            data_line += 1;
            current_line += 1;
        }

        Ok(Array1::from_vec(data))
    }
}

/// Chunked time series processor
pub struct ChunkedProcessor {
    /// Processing configuration
    config: ProcessingConfig,
    /// Current streaming statistics
    stats: StreamingStats,
    /// Progress information
    progress: ProgressInfo,
    /// Start time for timing
    start_time: Instant,
}

impl ChunkedProcessor {
    /// Create new chunked processor
    pub fn new(config: ProcessingConfig) -> Self {
        Self {
            config,
            stats: StreamingStats::new(),
            progress: ProgressInfo {
                chunk_number: 0,
                total_chunks: 0,
                points_processed: 0,
                total_points: 0,
                elapsed_time: 0.0,
                estimated_remaining: 0.0,
                memory_usage: 0,
            },
            start_time: Instant::now(),
        }
    }

    /// Process a binary file containing f64 values
    pub fn process_binary_file<P: AsRef<Path>>(&mut self, path: P) -> Result<StreamingStats> {
        let reader = MmapTimeSeriesReader::new(path)?;
        let total_points = reader.len();
        self.process_with_reader(
            Box::new(move |start, size| reader.read_chunk(start, size)),
            total_points,
        )
    }

    /// Process a CSV file
    pub fn process_csv_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        column_index: usize,
        has_header: bool,
    ) -> Result<StreamingStats> {
        let mut reader = CsvTimeSeriesReader::new(path, column_index, has_header)?;
        let total_lines = reader.estimate_total_lines()?;

        self.process_with_reader(
            Box::new(move |start, size| reader.read_chunk(start, size)),
            total_lines,
        )
    }

    /// Process data using a generic reader function
    fn process_with_reader<F>(&mut self, reader: F, totalpoints: usize) -> Result<StreamingStats>
    where
        F: Fn(usize, usize) -> Result<Array1<f64>> + Send + Sync + 'static,
    {
        self.start_time = Instant::now();
        self.progress.total_points = totalpoints as u64;
        // Estimate assuming no overlap; when overlap forces a smaller stride
        // than chunk_size, the actual chunk count (and hence chunk_number in
        // progress reports) legitimately exceeds this — it's a display
        // estimate only, points_processed/total_points is the accurate ratio.
        self.progress.total_chunks = totalpoints.div_ceil(self.config.chunk_size);

        let reader = Arc::new(reader);

        if self.config.parallel_processing {
            self.process_parallel(reader, totalpoints)
        } else {
            self.process_sequential(reader, totalpoints)
        }
    }

    /// Process chunks sequentially
    fn process_sequential<F>(
        &mut self,
        reader: Arc<F>,
        total_points: usize,
    ) -> Result<StreamingStats>
    where
        F: Fn(usize, usize) -> Result<Array1<f64>> + Send + Sync + 'static,
    {
        let mut start_idx = 0;
        // Leading elements of the next chunk that duplicate the tail of the
        // current one (overlap re-reads the same source range); they must be
        // skipped when folding into `stats` or every overlapping point would
        // be counted once per chunk that reads it.
        let mut skip = 0;

        while start_idx < total_points {
            let chunk_size = (self.config.chunk_size).min(total_points - start_idx);
            let chunk = reader(start_idx, chunk_size)?;

            // Update statistics, excluding the overlap already counted above
            for &value in chunk.iter().skip(skip) {
                self.stats.update(value);
            }

            // Update progress
            self.progress.chunk_number += 1;
            self.progress.points_processed += chunk.len().saturating_sub(skip) as u64;
            self.update_progress();

            if self.config.report_progress {
                self.report_progress();
            }

            let stride = (chunk_size - self.config.overlap.min(chunk_size)).max(1);
            skip = chunk_size.saturating_sub(stride);
            start_idx += stride;
        }

        Ok(self.stats.clone())
    }

    /// Process chunks in parallel
    fn process_parallel<F>(&mut self, reader: Arc<F>, totalpoints: usize) -> Result<StreamingStats>
    where
        F: Fn(usize, usize) -> Result<Array1<f64>> + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let num_threads = self.config.num_threads;

        // Calculate chunk boundaries
        let chunk_size = self.config.chunk_size;
        let overlap = self.config.overlap;
        let mut chunk_starts = Vec::new();
        // Per-chunk count of leading elements that duplicate the tail of the
        // previous chunk (see process_sequential for why these must not be
        // folded into stats twice); 0 for the first chunk.
        let mut chunk_skips = Vec::new();
        let mut start_idx = 0;
        let mut skip = 0;

        while start_idx < totalpoints {
            chunk_starts.push(start_idx);
            chunk_skips.push(skip);
            let current_chunk_size = chunk_size.min(totalpoints - start_idx);
            let stride = (current_chunk_size - overlap.min(current_chunk_size)).max(1);
            skip = current_chunk_size.saturating_sub(stride);
            start_idx += stride;
        }

        // Spawn worker threads
        let chunk_starts = Arc::new(chunk_starts);
        let chunk_skips = Arc::new(chunk_skips);
        let total_chunks = chunk_starts.len();

        for thread_id in 0..num_threads {
            let tx = tx.clone();
            let reader = Arc::clone(&reader);
            let chunk_starts = Arc::clone(&chunk_starts);
            let chunk_skips = Arc::clone(&chunk_skips);
            let chunk_size = self.config.chunk_size;
            let total_points = totalpoints;

            thread::spawn(move || {
                for (chunk_idx, &start_idx) in chunk_starts.iter().enumerate() {
                    if chunk_idx % num_threads != thread_id {
                        continue;
                    }

                    let current_chunk_size = chunk_size.min(total_points - start_idx);
                    let skip = chunk_skips[chunk_idx];

                    match reader(start_idx, current_chunk_size) {
                        Ok(chunk) => {
                            let mut local_stats = StreamingStats::new();
                            for &value in chunk.iter().skip(skip) {
                                local_stats.update(value);
                            }

                            if tx.send((chunk_idx, Ok(local_stats))).is_err() {
                                break; // Receiver dropped
                            }
                        }
                        Err(e) => {
                            if tx.send((chunk_idx, Err(e))).is_err() {
                                break; // Receiver dropped
                            }
                        }
                    }
                }
            });
        }

        drop(tx); // Close the channel

        // Collect results
        let mut completed_chunks = 0;
        while let Ok((_chunk_idx, result)) = rx.recv() {
            match result {
                Ok(chunk_stats) => {
                    self.stats.merge(&chunk_stats);
                    completed_chunks += 1;

                    self.progress.chunk_number = completed_chunks;
                    self.progress.points_processed =
                        (completed_chunks as f64 / total_chunks as f64 * totalpoints as f64) as u64;
                    self.update_progress();

                    if self.config.report_progress {
                        self.report_progress();
                    }
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        Ok(self.stats.clone())
    }

    /// Update progress timing information
    fn update_progress(&mut self) {
        self.progress.elapsed_time = self.start_time.elapsed().as_secs_f64();

        if self.progress.points_processed > 0 {
            let completion_ratio =
                self.progress.points_processed as f64 / self.progress.total_points as f64;
            let total_estimated_time = self.progress.elapsed_time / completion_ratio;
            self.progress.estimated_remaining = total_estimated_time - self.progress.elapsed_time;
        }

        // Estimate memory usage (rough approximation)
        self.progress.memory_usage =
            self.config.chunk_size * std::mem::size_of::<f64>() * self.config.num_threads;
    }

    /// Report progress to stdout
    fn report_progress(&self) {
        let completion = self.progress.completion_percentage();
        let rate = self.progress.processing_rate();
        let memory_mb = self.progress.memory_usage as f64 / 1_048_576.0;

        println!(
            "Progress: {:.1}% | Chunks: {}/{} | Rate: {:.0} pts/sec | Memory: {:.1} MB | ETA: {:.0}s",
            completion,
            self.progress.chunk_number,
            self.progress.total_chunks,
            rate,
            memory_mb,
            self.progress.estimated_remaining
        );
    }

    /// Get current progress information
    pub fn get_progress(&self) -> &ProgressInfo {
        &self.progress
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &StreamingStats {
        &self.stats
    }
}

/// Out-of-core moving average calculator
pub struct OutOfCoreMovingAverage {
    /// Window size
    window_size: usize,
    /// Circular buffer for values
    buffer: VecDeque<f64>,
    /// Current sum
    current_sum: f64,
}

impl OutOfCoreMovingAverage {
    /// Create new moving average calculator
    pub fn new(_windowsize: usize) -> Result<Self> {
        check_positive(_windowsize, "_windowsize")?;

        Ok(Self {
            window_size: _windowsize,
            buffer: VecDeque::with_capacity(_windowsize),
            current_sum: 0.0,
        })
    }

    /// Update with new value and return current moving average
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }

        self.buffer.push_back(value);
        self.current_sum += value;

        if self.buffer.len() > self.window_size {
            if let Some(old_value) = self.buffer.pop_front() {
                self.current_sum -= old_value;
            }
        }

        if self.buffer.len() == self.window_size {
            Some(self.current_sum / self.window_size as f64)
        } else {
            None
        }
    }

    /// Get current moving average without updating
    pub fn current_average(&self) -> Option<f64> {
        if self.buffer.len() == self.window_size {
            Some(self.current_sum / self.window_size as f64)
        } else {
            None
        }
    }

    /// Reset the calculator
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.current_sum = 0.0;
    }
}

/// Out-of-core quantile estimator using P² algorithm
pub struct OutOfCoreQuantileEstimator {
    /// Target quantile (0.0 to 1.0)
    quantile: f64,
    /// Marker positions
    positions: [f64; 5],
    /// Marker heights
    heights: [f64; 5],
    /// Number of observations
    count: usize,
    /// Initial values buffer
    initial_values: Vec<f64>,
}

impl OutOfCoreQuantileEstimator {
    /// Create new quantile estimator
    pub fn new(quantile: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&quantile) {
            return Err(TimeSeriesError::InvalidInput(
                "Quantile must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            quantile,
            positions: [
                1.0,
                1.0 + 2.0 * quantile,
                1.0 + 4.0 * quantile,
                3.0 + 2.0 * quantile,
                5.0,
            ],
            heights: [0.0; 5],
            count: 0,
            initial_values: Vec::new(),
        })
    }

    /// Update estimator with new value
    pub fn update(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        self.count += 1;

        // Collect first 5 values
        if self.count <= 5 {
            self.initial_values.push(value);

            if self.count == 5 {
                // Initialize markers
                self.initial_values
                    .sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));
                for i in 0..5 {
                    self.heights[i] = self.initial_values[i];
                }
            }
            return;
        }

        // Find cell k such that heights[k] <= value < heights[k+1]
        // Also update extreme markers if value is outside current range.
        let mut k = 0;
        if value < self.heights[0] {
            // New minimum: update marker 0 and keep k = 0
            self.heights[0] = value;
            k = 0;
        } else if value >= self.heights[4] {
            // New maximum: update marker 4 and keep k = 3
            self.heights[4] = value;
            k = 3;
        } else {
            for i in 0..4 {
                if value >= self.heights[i] && value < self.heights[i + 1] {
                    k = i;
                    break;
                }
            }
        }

        // Increment positions of markers k+1 through 4
        for i in (k + 1)..5 {
            self.positions[i] += 1.0;
        }

        // Desired positions after n observations (Jain & Chlamtac 1985):
        //   n₁'(n) = 1
        //   n₂'(n) = 1 + (n-1)*p/2        ← interpolates 0..p/2 of the range
        //   n₃'(n) = 1 + (n-1)*p           ← the p-th quantile position
        //   n₄'(n) = 1 + (n-1)*(1+p)/2    ← interpolates p..(1+p)/2
        //   n₅'(n) = n
        let n = self.count as f64;
        let p = self.quantile;
        let desired_positions = [
            1.0,                               // n₁ = 1 (minimum)
            1.0 + 0.5 * p * (n - 1.0),         // n₂
            1.0 + p * (n - 1.0),               // n₃ (target quantile)
            1.0 + 0.5 * (1.0 + p) * (n - 1.0), // n₄
            n,                                 // n₅ = n (maximum)
        ];

        // Adjust heights of markers 1-3 if necessary
        #[allow(clippy::needless_range_loop)]
        for i in 1..4 {
            let d = desired_positions[i] - self.positions[i];

            if (d >= 1.0 && self.positions[i + 1] - self.positions[i] > 1.0)
                || (d <= -1.0 && self.positions[i - 1] - self.positions[i] < -1.0)
            {
                let d_sign = if d > 0.0 { 1.0 } else { -1.0 };

                // Try parabolic formula
                let new_height = self.heights[i]
                    + d_sign / (self.positions[i + 1] - self.positions[i - 1])
                        * ((self.positions[i] - self.positions[i - 1] + d_sign)
                            * (self.heights[i + 1] - self.heights[i])
                            / (self.positions[i + 1] - self.positions[i])
                            + (self.positions[i + 1] - self.positions[i] - d_sign)
                                * (self.heights[i] - self.heights[i - 1])
                                / (self.positions[i] - self.positions[i - 1]));

                // Check if parabolic formula gives valid result
                if self.heights[i - 1] < new_height && new_height < self.heights[i + 1] {
                    self.heights[i] = new_height;
                } else {
                    // Use linear formula
                    self.heights[i] += d_sign
                        * (self.heights[(i as i32 + d_sign as i32) as usize] - self.heights[i])
                        / (self.positions[(i as i32 + d_sign as i32) as usize] - self.positions[i]);
                }

                self.positions[i] += d_sign;
            }
        }
    }

    /// Get current quantile estimate
    pub fn quantile_estimate(&self) -> Option<f64> {
        if self.count < 5 {
            None
        } else {
            Some(self.heights[2]) // Middle marker
        }
    }

    /// Get debug state information (heights and positions)
    #[allow(dead_code)]
    pub fn debug_state(&self) -> (Vec<f64>, Vec<f64>) {
        (self.heights.to_vec(), self.positions.to_vec())
    }
}

/// Utility functions for out-of-core processing
pub mod utils {
    use super::*;

    /// Estimate file size and number of data points for planning
    pub fn estimate_processing_requirements<P: AsRef<Path>>(
        file_path: P,
        data_type_size: usize,
    ) -> Result<(usize, usize, f64)> {
        let metadata = std::fs::metadata(file_path)
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to get file metadata: {e}")))?;

        let file_size_bytes = metadata.len() as usize;
        let estimated_points = file_size_bytes / data_type_size;
        let estimated_memory_gb = file_size_bytes as f64 / 1_073_741_824.0;

        Ok((file_size_bytes, estimated_points, estimated_memory_gb))
    }

    /// Suggest optimal chunk size based on available memory
    pub fn suggest_chunk_size(
        total_points: usize,
        available_memory_bytes: usize,
        safety_factor: f64,
    ) -> usize {
        let element_size = std::mem::size_of::<f64>();
        let max_chunk_points =
            (available_memory_bytes as f64 * safety_factor) as usize / element_size;

        // Ensure chunk size is reasonable (between 1K and 10M points)
        max_chunk_points.clamp(1_000, 10_000_000).min(total_points)
    }

    /// Convert CSV file to binary format for faster processing
    pub fn csv_to_binary<P1: AsRef<Path>, P2: AsRef<Path>>(
        csv_path: P1,
        binary_path: P2,
        column_index: usize,
        has_header: bool,
    ) -> Result<usize> {
        let input_file = File::open(csv_path)
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to open CSV file: {e}")))?;

        let output_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(binary_path)
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to create binary file: {e}")))?;

        let reader = BufReader::new(input_file);
        let mut writer = BufWriter::new(output_file);
        let mut count = 0;
        let mut line_number = 0;

        for line in reader.lines() {
            let line =
                line.map_err(|e| TimeSeriesError::IOError(format!("Failed to read line: {e}")))?;

            // Skip _header
            if line_number == 0 && has_header {
                line_number += 1;
                continue;
            }

            let fields: Vec<&str> = line.split(',').collect();

            if column_index >= fields.len() {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "Column _index {} out of bounds for line with {} fields",
                    column_index,
                    fields.len()
                )));
            }

            let value: f64 = fields[column_index].trim().parse().map_err(|e| {
                TimeSeriesError::InvalidInput(format!("Failed to parse value: {e}"))
            })?;

            // Write as binary f64
            let bytes = value.to_le_bytes();
            writer.write_all(&bytes).map_err(|e| {
                TimeSeriesError::IOError(format!("Failed to write binary data: {e}"))
            })?;

            count += 1;
            line_number += 1;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_streaming_stats() {
        let mut stats = StreamingStats::new();

        // Add some test data
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        for &value in &data {
            stats.update(value);
        }

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.variance() - 2.5).abs() < 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_streaming_stats_merge() {
        let mut stats1 = StreamingStats::new();
        let mut stats2 = StreamingStats::new();

        // Add data to first stats
        for i in 1..=5 {
            stats1.update(i as f64);
        }

        // Add data to second stats
        for i in 6..=10 {
            stats2.update(i as f64);
        }

        // Merge
        stats1.merge(&stats2);

        assert_eq!(stats1.count, 10);
        assert!((stats1.mean - 5.5).abs() < 1e-10);
        assert_eq!(stats1.min, 1.0);
        assert_eq!(stats1.max, 10.0);
    }

    #[test]
    fn test_out_of_core_moving_average() {
        let mut ma = OutOfCoreMovingAverage::new(3).expect("Operation failed");

        assert!(ma.update(1.0).is_none()); // Not enough data yet
        assert!(ma.update(2.0).is_none()); // Not enough data yet

        let avg = ma.update(3.0).expect("Operation failed"); // Now we have 3 values
        assert!((avg - 2.0).abs() < 1e-10);

        let avg = ma.update(4.0).expect("Operation failed"); // Window slides
        assert!((avg - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_csv_processing() {
        // Create a temporary CSV file
        let mut temp_file = NamedTempFile::new().expect("Operation failed");
        writeln!(temp_file, "time,value,other").expect("Operation failed");
        writeln!(temp_file, "1,10.5,x").expect("Operation failed");
        writeln!(temp_file, "2,20.3,y").expect("Operation failed");
        writeln!(temp_file, "3,15.7,z").expect("Operation failed");
        temp_file.flush().expect("Operation failed");

        let config = ProcessingConfig::new()
            .with_chunk_size(2)
            .with_parallel_processing(false);

        let mut processor = ChunkedProcessor::new(config);
        let stats = processor
            .process_csv_file(temp_file.path(), 1, true)
            .expect("Operation failed");

        assert_eq!(stats.count, 3);
        assert!((stats.mean - 15.5).abs() < 1e-10);
        assert_eq!(stats.min, 10.5);
        assert_eq!(stats.max, 20.3);
    }

    #[test]
    fn test_csv_processing_parallel_with_overlap_does_not_double_count() {
        // Same shape as test_csv_processing (chunk_size 2, default overlap 1000
        // forces the minimum 1-point stride), but exercises process_parallel:
        // per-chunk stats are merged from worker threads, so the overlap-skip
        // bookkeeping must be threaded through chunk_starts/chunk_skips too,
        // not just the sequential path.
        let mut temp_file = NamedTempFile::new().expect("Operation failed");
        writeln!(temp_file, "time,value,other").expect("Operation failed");
        for i in 1..=20 {
            writeln!(temp_file, "{i},{},z", i as f64 * 1.5).expect("Operation failed");
        }
        temp_file.flush().expect("Operation failed");

        let config = ProcessingConfig::new()
            .with_chunk_size(3)
            .with_parallel_processing(true)
            .with_threads(4);

        let mut processor = ChunkedProcessor::new(config);
        let stats = processor
            .process_csv_file(temp_file.path(), 1, true)
            .expect("Operation failed");

        assert_eq!(stats.count, 20);
        let expected_mean: f64 = (1..=20).map(|i| i as f64 * 1.5).sum::<f64>() / 20.0;
        assert!((stats.mean - expected_mean).abs() < 1e-9);
        assert_eq!(stats.min, 1.5);
        assert_eq!(stats.max, 30.0);
    }

    #[test]
    fn test_quantile_estimator() {
        let mut estimator = OutOfCoreQuantileEstimator::new(0.5).expect("Operation failed"); // Median

        // Add enough data to initialize
        for i in 1..=100 {
            estimator.update(i as f64);
        }

        let median = estimator.quantile_estimate().expect("Operation failed");

        // The true median of 1..=100 is 50.5. After fixing the P² desired-position
        // formula, the estimate should converge within ~10 units.
        assert!(
            (median - 50.5).abs() < 15.0,
            "Median estimate {} should be near 50.5 (true median of 1..=100)",
            median
        );
        // Must be within the observed range
        assert!(
            median >= 1.0 && median <= 100.0,
            "Median estimate {} must be in [1, 100]",
            median
        );
    }

    // ---- NEW TESTS FOR P² ALGORITHM FIX ----

    /// Feed 1000 uniform [0,1) values and check that the 0.5 quantile converges
    /// to the true median (0.5) within a tight tolerance.
    #[test]
    fn test_p2_median_uniform_convergence() {
        let mut estimator =
            OutOfCoreQuantileEstimator::new(0.5).expect("Failed to create estimator");

        // Use a deterministic LCG sequence for reproducibility
        let mut state: u64 = 12345;
        for _ in 0..1000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (state >> 33) as f64 / (1u64 << 31) as f64;
            estimator.update(x);
        }

        let est = estimator
            .quantile_estimate()
            .expect("Should have estimate after 1000 values");

        // P² converges well for large n; allow ±5% of the range
        assert!(
            (est - 0.5).abs() < 0.05,
            "P² median estimate {} should be near 0.5 for uniform data",
            est
        );
    }

    /// Test 0.1 and 0.9 quantile estimates on uniform [0,1) data.
    #[test]
    fn test_p2_tail_quantiles_uniform() {
        // 0.1-quantile
        let mut est10 = OutOfCoreQuantileEstimator::new(0.1).expect("Failed to create estimator");
        // 0.9-quantile
        let mut est90 = OutOfCoreQuantileEstimator::new(0.9).expect("Failed to create estimator");

        let mut state: u64 = 98765;
        for _ in 0..2000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (state >> 33) as f64 / (1u64 << 31) as f64;
            est10.update(x);
            est90.update(x);
        }

        let q10 = est10
            .quantile_estimate()
            .expect("Should have 0.1-quantile estimate");
        let q90 = est90
            .quantile_estimate()
            .expect("Should have 0.9-quantile estimate");

        // True values: 0.1 and 0.9 for Uniform(0,1); allow ±0.05 tolerance
        assert!(
            (q10 - 0.1).abs() < 0.05,
            "0.1-quantile estimate {} should be near 0.1",
            q10
        );
        assert!(
            (q90 - 0.9).abs() < 0.05,
            "0.9-quantile estimate {} should be near 0.9",
            q90
        );
    }

    /// After many updates the five marker heights must remain strictly ordered.
    #[test]
    fn test_p2_markers_ordered() {
        let mut estimator =
            OutOfCoreQuantileEstimator::new(0.5).expect("Failed to create estimator");

        let mut state: u64 = 42;
        for _ in 0..500 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (state >> 33) as f64 / (1u64 << 31) as f64;
            estimator.update(x);
        }

        let (heights, _positions) = estimator.debug_state();

        // Markers must be weakly ordered: h[0] <= h[1] <= ... <= h[4]
        for i in 0..4 {
            assert!(
                heights[i] <= heights[i + 1],
                "Marker heights must be ordered: heights[{}]={} > heights[{}]={}",
                i,
                heights[i],
                i + 1,
                heights[i + 1]
            );
        }
    }

    #[test]
    fn test_processing_config() {
        let config = ProcessingConfig::new()
            .with_chunk_size(5000)
            .with_overlap(500)
            .with_max_memory(2_000_000_000);

        assert_eq!(config.chunk_size, 5000);
        assert_eq!(config.overlap, 500);
        assert_eq!(config.max_memory_usage, 2_000_000_000);
        assert!(config.parallel_processing);
    }
}
