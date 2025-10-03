//! Streaming support for large datasets
//!
//! This module provides comprehensive streaming capabilities for handling datasets
//! that are too large to fit in memory, enabling processing of massive datasets
//! in a memory-efficient manner.

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

/// Configuration for streaming operations
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Size of each chunk in samples
    pub chunk_size: usize,
    /// Number of chunks to buffer in memory
    pub buffer_size: usize,
    /// Number of worker threads for parallel processing
    pub num_workers: usize,
    /// Memory limit in MB for the entire streaming operation
    pub memory_limit_mb: Option<usize>,
    /// Whether to enable compression for buffered chunks
    pub enable_compression: bool,
    /// Whether to prefetch chunks in background
    pub enable_prefetch: bool,
    /// Maximum number of chunks to process (None = all)
    pub max_chunks: Option<usize>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10_000,
            buffer_size: 3,
            num_workers: num_cpus::get(),
            memory_limit_mb: None,
            enable_compression: false,
            enable_prefetch: true,
            max_chunks: None,
        }
    }
}

/// A chunk of data from a streaming dataset
#[derive(Debug, Clone)]
pub struct DataChunk {
    /// Feature data for this chunk
    pub data: Array2<f64>,
    /// Target values for this chunk (if available)
    pub target: Option<Array1<f64>>,
    /// Chunk index in the stream
    pub chunk_index: usize,
    /// Global sample indices for this chunk
    pub sample_indices: Vec<usize>,
    /// Whether this is the last chunk in the stream
    pub is_last: bool,
}

impl DataChunk {
    /// Number of samples in this chunk
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Number of features in this chunk
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Convert chunk to a Dataset
    pub fn to_dataset(&self) -> Dataset {
        Dataset {
            data: self.data.clone(),
            target: self.target.clone(),
            targetnames: None,
            featurenames: None,
            feature_descriptions: None,
            description: None,
            metadata: Default::default(),
        }
    }
}

/// Iterator over streaming dataset chunks
pub struct StreamingIterator {
    config: StreamConfig,
    chunk_buffer: Arc<Mutex<VecDeque<DataChunk>>>,
    current_chunk: usize,
    total_chunks: Option<usize>,
    finished: bool,
    producer_handle: Option<thread::JoinHandle<Result<()>>>,
}

impl StreamingIterator {
    /// Create a new streaming iterator from a CSV file
    pub fn from_csv<P: AsRef<Path>>(path: P, config: StreamConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let chunk_buffer = Arc::new(Mutex::new(VecDeque::new()));
        let buffer_clone = Arc::clone(&chunk_buffer);
        let config_clone = config.clone();

        // Start producer thread
        let producer_handle =
            thread::spawn(move || Self::csv_producer(path, config_clone, buffer_clone));

        Ok(Self {
            config,
            chunk_buffer,
            current_chunk: 0,
            total_chunks: None,
            finished: false,
            producer_handle: Some(producer_handle),
        })
    }

    /// Create a new streaming iterator from a binary file
    pub fn from_binary<P: AsRef<Path>>(
        path: P,
        n_features: usize,
        config: StreamConfig,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let chunk_buffer = Arc::new(Mutex::new(VecDeque::new()));
        let buffer_clone = Arc::clone(&chunk_buffer);
        let config_clone = config.clone();

        let producer_handle = thread::spawn(move || {
            Self::binary_producer(path, n_features, config_clone, buffer_clone)
        });

        Ok(Self {
            config,
            chunk_buffer,
            current_chunk: 0,
            total_chunks: None,
            finished: false,
            producer_handle: Some(producer_handle),
        })
    }

    /// Create a streaming iterator from a data generator function
    pub fn from_generator<F>(
        generator: F,
        total_samples: usize,
        n_features: usize,
        config: StreamConfig,
    ) -> Result<Self>
    where
        F: Fn(usize, usize, usize) -> Result<(Array2<f64>, Option<Array1<f64>>)> + Send + 'static,
    {
        let chunk_buffer = Arc::new(Mutex::new(VecDeque::new()));
        let buffer_clone = Arc::clone(&chunk_buffer);
        let config_clone = config.clone();

        let producer_handle = thread::spawn(move || {
            Self::generator_producer(
                generator,
                total_samples,
                n_features,
                config_clone,
                buffer_clone,
            )
        });

        let total_chunks = total_samples.div_ceil(config.chunk_size);

        Ok(Self {
            config,
            chunk_buffer,
            current_chunk: 0,
            total_chunks: Some(total_chunks),
            finished: false,
            producer_handle: Some(producer_handle),
        })
    }

    /// Get the next chunk from the stream
    pub fn next_chunk(&mut self) -> Result<Option<DataChunk>> {
        if self.finished {
            return Ok(None);
        }

        // Check if we've reached the maximum number of chunks
        if let Some(max_chunks) = self.config.max_chunks {
            if self.current_chunk >= max_chunks {
                self.finished = true;
                return Ok(None);
            }
        }

        // Wait for a chunk to be available
        loop {
            {
                let mut buffer = self.chunk_buffer.lock().unwrap();
                if let Some(chunk) = buffer.pop_front() {
                    self.current_chunk += 1;

                    if chunk.is_last {
                        self.finished = true;
                    }

                    return Ok(Some(chunk));
                }
            }

            // Check if producer is finished
            if let Some(handle) = &self.producer_handle {
                if handle.is_finished() {
                    // Join the producer thread
                    let handle = self.producer_handle.take().unwrap();
                    handle.join().unwrap()?;

                    // Try one more time to get remaining chunks
                    let mut buffer = self.chunk_buffer.lock().unwrap();
                    if let Some(chunk) = buffer.pop_front() {
                        self.current_chunk += 1;
                        if chunk.is_last {
                            self.finished = true;
                        }
                        return Ok(Some(chunk));
                    } else {
                        self.finished = true;
                        return Ok(None);
                    }
                }
            }

            // Sleep briefly to avoid busy waiting
            thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Get streaming statistics
    pub fn stats(&self) -> StreamStats {
        let buffer = self.chunk_buffer.lock().unwrap();
        StreamStats {
            current_chunk: self.current_chunk,
            total_chunks: self.total_chunks,
            buffer_size: buffer.len(),
            buffer_capacity: self.config.buffer_size,
            finished: self.finished,
        }
    }

    // Producer for CSV files
    fn csv_producer(
        path: std::path::PathBuf,
        config: StreamConfig,
        buffer: Arc<Mutex<VecDeque<DataChunk>>>,
    ) -> Result<()> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Skip header if present
        let _header = lines.next();

        let mut chunk_data = Vec::new();
        let mut chunk_index = 0;
        let mut global_sample_index = 0;

        for line in lines {
            let line = line?;
            let values: Vec<f64> = line
                .split(',')
                .map(|s| s.trim().parse().unwrap_or(0.0))
                .collect();

            if !values.is_empty() {
                chunk_data.push((values, global_sample_index));
                global_sample_index += 1;

                if chunk_data.len() >= config.chunk_size {
                    let chunk = Self::create_chunk_from_data(&chunk_data, chunk_index, false)?;

                    // Wait for buffer space
                    loop {
                        let mut buffer_guard = buffer.lock().unwrap();
                        if buffer_guard.len() < config.buffer_size {
                            buffer_guard.push_back(chunk);
                            break;
                        }
                        drop(buffer_guard);
                        thread::sleep(std::time::Duration::from_millis(10));
                    }

                    chunk_data.clear();
                    chunk_index += 1;

                    if let Some(max_chunks) = config.max_chunks {
                        if chunk_index >= max_chunks {
                            break;
                        }
                    }
                }
            }
        }

        // Handle remaining data
        if !chunk_data.is_empty() {
            let chunk = Self::create_chunk_from_data(&chunk_data, chunk_index, true)?;
            let mut buffer_guard = buffer.lock().unwrap();
            buffer_guard.push_back(chunk);
        }

        Ok(())
    }

    // Producer for binary files
    fn binary_producer(
        path: std::path::PathBuf,
        n_features: usize,
        config: StreamConfig,
        buffer: Arc<Mutex<VecDeque<DataChunk>>>,
    ) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(&path)?;
        let mut chunk_index = 0;
        let mut global_sample_index = 0;

        let values_per_chunk = config.chunk_size * n_features;
        let bytes_per_chunk = values_per_chunk * std::mem::size_of::<f64>();

        loop {
            let mut buffer_data = vec![0u8; bytes_per_chunk];
            let bytes_read = file.read(&mut buffer_data)?;

            if bytes_read == 0 {
                break; // End of file
            }

            let values_read = bytes_read / std::mem::size_of::<f64>();
            let samples_read = values_read / n_features;

            if samples_read == 0 {
                break;
            }

            // Convert bytes to f64 values
            let float_data: Vec<f64> = buffer_data[..bytes_read]
                .chunks_exact(std::mem::size_of::<f64>())
                .map(|chunk| {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(chunk);
                    f64::from_le_bytes(bytes)
                })
                .collect();

            // Create data matrix
            let data = Array2::from_shape_vec((samples_read, n_features), float_data)
                .map_err(|e| DatasetsError::Other(format!("Shape error: {e}")))?;
            let sample_indices: Vec<usize> =
                (global_sample_index..global_sample_index + samples_read).collect();

            let chunk = DataChunk {
                data,
                target: None,
                chunk_index,
                sample_indices,
                is_last: bytes_read < bytes_per_chunk,
            };

            // Wait for buffer space
            loop {
                let mut buffer_guard = buffer.lock().unwrap();
                if buffer_guard.len() < config.buffer_size {
                    buffer_guard.push_back(chunk);
                    break;
                }
                drop(buffer_guard);
                thread::sleep(std::time::Duration::from_millis(10));
            }

            global_sample_index += samples_read;
            chunk_index += 1;

            if let Some(max_chunks) = config.max_chunks {
                if chunk_index >= max_chunks {
                    break;
                }
            }

            if bytes_read < bytes_per_chunk {
                break; // Last chunk
            }
        }

        Ok(())
    }

    // Producer for data generators
    fn generator_producer<F>(
        generator: F,
        total_samples: usize,
        n_features: usize,
        config: StreamConfig,
        buffer: Arc<Mutex<VecDeque<DataChunk>>>,
    ) -> Result<()>
    where
        F: Fn(usize, usize, usize) -> Result<(Array2<f64>, Option<Array1<f64>>)>,
    {
        let mut chunk_index = 0;
        let mut processed_samples = 0;

        while processed_samples < total_samples {
            let remaining_samples = total_samples - processed_samples;
            let chunk_samples = config.chunk_size.min(remaining_samples);

            // Generate chunk data
            let (data, target) = generator(chunk_samples, n_features, processed_samples)?;

            let sample_indices: Vec<usize> =
                (processed_samples..processed_samples + chunk_samples).collect();
            let is_last = processed_samples + chunk_samples >= total_samples;

            let chunk = DataChunk {
                data,
                target,
                chunk_index,
                sample_indices,
                is_last,
            };

            // Wait for buffer space
            loop {
                let mut buffer_guard = buffer.lock().unwrap();
                if buffer_guard.len() < config.buffer_size {
                    buffer_guard.push_back(chunk);
                    break;
                }
                drop(buffer_guard);
                thread::sleep(std::time::Duration::from_millis(10));
            }

            processed_samples += chunk_samples;
            chunk_index += 1;

            if let Some(max_chunks) = config.max_chunks {
                if chunk_index >= max_chunks {
                    break;
                }
            }
        }

        Ok(())
    }

    // Helper to create chunk from CSV data
    fn create_chunk_from_data(
        data: &[(Vec<f64>, usize)],
        chunk_index: usize,
        is_last: bool,
    ) -> Result<DataChunk> {
        if data.is_empty() {
            return Err(DatasetsError::InvalidFormat("Empty chunk data".to_string()));
        }

        let n_samples = data.len();
        let n_features = data[0].0.len() - 1; // Assume _last column is target

        let mut chunk_data = Array2::zeros((n_samples, n_features));
        let mut chunk_target = Array1::zeros(n_samples);
        let mut sample_indices = Vec::with_capacity(n_samples);

        for (i, (values, global_idx)) in data.iter().enumerate() {
            for j in 0..n_features {
                chunk_data[[i, j]] = values[j];
            }
            chunk_target[i] = values[n_features];
            sample_indices.push(*global_idx);
        }

        Ok(DataChunk {
            data: chunk_data,
            target: Some(chunk_target),
            chunk_index,
            sample_indices,
            is_last,
        })
    }
}

/// Statistics about streaming operation
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Current chunk being processed
    pub current_chunk: usize,
    /// Total number of chunks (if known)
    pub total_chunks: Option<usize>,
    /// Number of chunks currently buffered
    pub buffer_size: usize,
    /// Maximum buffer capacity
    pub buffer_capacity: usize,
    /// Whether streaming is finished
    pub finished: bool,
}

impl StreamStats {
    /// Get progress as a percentage (if total is known)
    pub fn progress_percent(&self) -> Option<f64> {
        self.total_chunks
            .map(|total| (self.current_chunk as f64 / total as f64) * 100.0)
    }

    /// Get buffer utilization as a percentage
    pub fn buffer_utilization(&self) -> f64 {
        (self.buffer_size as f64 / self.buffer_capacity as f64) * 100.0
    }
}

/// Parallel streaming processor for applying operations to chunks
pub struct StreamProcessor<T> {
    config: StreamConfig,
    phantom: std::marker::PhantomData<T>,
}

impl<T> StreamProcessor<T>
where
    T: Send + Sync + 'static,
{
    /// Create a new stream processor
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            phantom: std::marker::PhantomData,
        }
    }

    /// Process chunks in parallel using a custom function
    pub fn process_parallel<F, R>(
        &self,
        mut iterator: StreamingIterator,
        processor: F,
    ) -> Result<Vec<R>>
    where
        F: Fn(DataChunk) -> Result<R> + Send + Sync + Clone + 'static,
        R: Send + 'static,
    {
        use std::sync::mpsc;

        // Create channels for work distribution and result collection
        let (work_tx, work_rx) = mpsc::channel();
        let work_rx = Arc::new(Mutex::new(work_rx));

        let (result_tx, result_rx) = mpsc::channel();
        let mut worker_handles = Vec::new();

        // Start worker threads
        for worker_id in 0..self.config.num_workers {
            let work_rx_clone = Arc::clone(&work_rx);
            let result_tx_clone = result_tx.clone();
            let processor_clone = processor.clone();

            let handle = thread::spawn(move || {
                loop {
                    // Receive work chunk
                    let chunk = {
                        let rx = work_rx_clone.lock().unwrap();
                        rx.recv().ok()
                    };

                    match chunk {
                        Some(Some((chunk_id, chunk))) => {
                            // Process the chunk
                            match processor_clone(chunk) {
                                Ok(result) => {
                                    // Send result back with chunk ID to maintain order
                                    if result_tx_clone.send((chunk_id, Ok(result))).is_err() {
                                        eprintln!("Worker {worker_id} failed to send result");
                                        break;
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Worker {worker_id} processing error: {e}");
                                    // Send error result
                                    if result_tx_clone.send((chunk_id, Err(e))).is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                        Some(None) => break, // End signal
                        None => break,       // Channel closed
                    }
                }
            });

            worker_handles.push(handle);
        }

        // Send chunks to workers with chunk IDs to maintain order
        let mut chunk_count = 0;
        while let Some(chunk) = iterator.next_chunk()? {
            work_tx
                .send(Some((chunk_count, chunk)))
                .map_err(|e| DatasetsError::Other(format!("Work send error: {e}")))?;
            chunk_count += 1;
        }

        // Send end signals to all workers
        for _ in 0..self.config.num_workers {
            work_tx
                .send(None)
                .map_err(|e| DatasetsError::Other(format!("End signal send error: {e}")))?;
        }

        // Drop the work sender to signal no more work
        drop(work_tx);

        // Collect results in order
        let mut results: Vec<Option<R>> = (0..chunk_count).map(|_| None).collect();
        let mut received_count = 0;

        // Collect all results
        while received_count < chunk_count {
            match result_rx.recv() {
                Ok((chunk_id, result)) => {
                    match result {
                        Ok(value) => {
                            if chunk_id < results.len() {
                                results[chunk_id] = Some(value);
                                received_count += 1;
                            }
                        }
                        Err(e) => {
                            // Return error if any worker fails
                            return Err(e);
                        }
                    }
                }
                Err(_) => {
                    return Err(DatasetsError::Other(
                        "Failed to receive results from workers".to_string(),
                    ));
                }
            }
        }

        // Wait for all workers to finish
        for handle in worker_handles {
            if let Err(e) = handle.join() {
                eprintln!("Worker thread panicked: {e:?}");
            }
        }

        // Convert results to final vector (all should be Some at this point)
        let final_results: Vec<R> =
            results
                .into_iter()
                .collect::<Option<Vec<R>>>()
                .ok_or_else(|| {
                    DatasetsError::Other("Missing results from parallel processing".to_string())
                })?;

        Ok(final_results)
    }
}

/// Memory-efficient data transformer for streaming
pub struct StreamTransformer {
    #[allow(clippy::type_complexity)]
    transformations: Vec<Box<dyn Fn(&mut DataChunk) -> Result<()> + Send + Sync>>,
}

impl StreamTransformer {
    /// Create a new stream transformer
    pub fn new() -> Self {
        Self {
            transformations: Vec::new(),
        }
    }

    /// Add a transformation function
    pub fn add_transform<F>(mut self, transform: F) -> Self
    where
        F: Fn(&mut DataChunk) -> Result<()> + Send + Sync + 'static,
    {
        self.transformations.push(Box::new(transform));
        self
    }

    /// Apply all transformations to a chunk
    pub fn transform_chunk(&self, chunk: &mut DataChunk) -> Result<()> {
        for transform in &self.transformations {
            transform(chunk)?;
        }
        Ok(())
    }

    /// Add standard scaling transformation
    pub fn add_standard_scaling(self) -> Self {
        self.add_transform(|chunk| {
            // Simplified standard scaling (would need proper implementation)
            let mean = chunk.data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
            let std = chunk.data.std_axis(scirs2_core::ndarray::Axis(0), 0.0);

            for mut row in chunk.data.axis_iter_mut(scirs2_core::ndarray::Axis(0)) {
                for (i, val) in row.iter_mut().enumerate() {
                    if std[i] > 0.0 {
                        *val = (*val - mean[i]) / std[i];
                    }
                }
            }
            Ok(())
        })
    }

    /// Add missing value imputation
    pub fn add_missing_value_imputation(self) -> Self {
        self.add_transform(|chunk| {
            // Replace NaN values with column mean
            let means = chunk.data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();

            for mut row in chunk.data.axis_iter_mut(scirs2_core::ndarray::Axis(0)) {
                for (i, val) in row.iter_mut().enumerate() {
                    if val.is_nan() {
                        *val = means[i];
                    }
                }
            }
            Ok(())
        })
    }
}

impl Default for StreamTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common streaming operations
///
/// Stream a large CSV file
#[allow(dead_code)]
pub fn stream_csv<P: AsRef<Path>>(path: P, config: StreamConfig) -> Result<StreamingIterator> {
    StreamingIterator::from_csv(path, config)
}

/// Stream synthetic classification data
#[allow(dead_code)]
pub fn stream_classification(
    total_samples: usize,
    n_features: usize,
    n_classes: usize,
    config: StreamConfig,
) -> Result<StreamingIterator> {
    use crate::generators::make_classification;

    let generator = move |chunk_size: usize, _features: usize, start_idx: usize| {
        let dataset = make_classification(
            chunk_size,
            _features,
            n_classes,
            2,
            _features / 2,
            Some(42 + start_idx as u64),
        )?;
        Ok((dataset.data, dataset.target))
    };

    StreamingIterator::from_generator(generator, total_samples, n_features, config)
}

/// Stream synthetic regression data
#[allow(dead_code)]
pub fn stream_regression(
    total_samples: usize,
    n_features: usize,
    config: StreamConfig,
) -> Result<StreamingIterator> {
    use crate::generators::make_regression;

    let generator = move |chunk_size: usize, _features: usize, start_idx: usize| {
        let dataset = make_regression(
            chunk_size,
            _features,
            _features / 2,
            0.1,
            Some(42 + start_idx as u64),
        )?;
        Ok((dataset.data, dataset.target))
    };

    StreamingIterator::from_generator(generator, total_samples, n_features, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config() {
        let config = StreamConfig::default();
        assert_eq!(config.chunk_size, 10_000);
        assert_eq!(config.buffer_size, 3);
        assert!(config.num_workers > 0);
    }

    #[test]
    fn test_data_chunk() {
        let data = Array2::zeros((100, 5));
        let target = Array1::zeros(100);
        let chunk = DataChunk {
            data,
            target: Some(target),
            chunk_index: 0,
            sample_indices: (0..100).collect(),
            is_last: false,
        };

        assert_eq!(chunk.n_samples(), 100);
        assert_eq!(chunk.n_features(), 5);
        assert!(!chunk.is_last);
    }

    #[test]
    fn test_stream_stats() {
        let stats = StreamStats {
            current_chunk: 5,
            total_chunks: Some(10),
            buffer_size: 2,
            buffer_capacity: 3,
            finished: false,
        };

        assert_eq!(stats.progress_percent(), Some(50.0));
        assert!((stats.buffer_utilization() - 66.66666666666667).abs() < 1e-10);
    }

    #[test]
    fn test_stream_classification() {
        let config = StreamConfig {
            chunk_size: 100,
            buffer_size: 2,
            max_chunks: Some(3),
            ..Default::default()
        };

        let stream = stream_classification(1000, 10, 3, config).unwrap();
        assert!(stream.total_chunks.is_some());
    }

    #[test]
    fn test_stream_transformer() {
        let transformer = StreamTransformer::new()
            .add_standard_scaling()
            .add_missing_value_imputation();

        assert_eq!(transformer.transformations.len(), 2);
    }
}
